"""
Conversation manager for the OpenAI Realtime Assistant.

This module orchestrates the conversation flow between the user and the
assistant, handling state transitions, events, and coordinating the 
API client and audio services.
"""

import asyncio
import json
import logging
import time
import uuid
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from src.config import settings
from src.config.logging_config import get_logger
from src.domain.audio.manager import AudioManager, AudioMode
from src.domain.conversation.state import (
    ConversationContext, ConversationHistory, ConversationStateType,
    Message, MessageContent, MessageRole
)
from src.events.event_interface import (
    AudioSpeechEvent, ErrorEvent, Event, EventType, TranscriptionEvent,
    UserSpeechEvent, event_bus
)
from src.services.api_client import RealtimeClient
from src.utils.async_helpers import TaskManager, debounce, wait_for_event, run_with_timeout
from src.utils.error_handling import ApiError, AppError, ErrorSeverity, safe_execute

logger = get_logger(__name__)


class ConversationManager:
    """
    Manager for coordinating conversation between user and assistant.
    
    This class is responsible for:
    - Managing the conversation state machine
    - Coordinating audio input/output with API communication
    - Handling speech and response events
    - Maintaining conversation context and history
    """
    
    def __init__(
        self,
        assistant_id: str,
        instructions: Optional[str] = None,
        temperature: float = 1.0,
        audio_manager: Optional[AudioManager] = None,
        api_client: Optional[RealtimeClient] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        voice: str = "alloy",
        auto_respond: bool = True,
        token_limit: int = 16385
    ):
        """
        Initialize the conversation manager.
        
        Args:
            assistant_id: ID of the OpenAI assistant to use
            instructions: Optional custom instructions for the assistant
            temperature: Temperature parameter for generation (0.0-2.0)
            audio_manager: Optional audio manager instance to use
            api_client: Optional API client instance to use
            tools: Optional list of tools for function calling
            voice: Voice ID for audio responses
            auto_respond: Whether to automatically respond to user inputs
            token_limit: Token limit for the conversation
        """
        # Initialize services
        self.api_client = api_client or RealtimeClient()
        self.audio_manager = audio_manager or AudioManager()
        
        # Create conversation context
        self.context = ConversationContext(
            assistant_id=assistant_id,
            session_id=f"session_{uuid.uuid4().hex}",
            instructions=instructions,
            temperature=temperature,
            tools=tools or [],
            voice=voice,
            token_limit=token_limit
        )
        
        # Create conversation history
        self.history = ConversationHistory(self.context)
        
        # Task management
        self.task_manager = TaskManager("conversation")
        self.shutdown_event = asyncio.Event()
        
        # Configuration
        self.auto_respond = auto_respond
        self.max_retry_attempts = 3
        self.retry_delay = 1.0
        
        # Tracking state
        self.active_response_id: Optional[str] = None
        self.pending_function_calls: Dict[str, Dict[str, Any]] = {}
        self.messages: List[Dict[str, Any]] = []  # Legacy property for backwards compatibility
        self.transcriptions: List[Dict[str, Any]] = []
        
        # Voice command detection
        self.voice_commands = {
            "goodbye": ["goodbye", "bye", "exit", "quit", "end session"],
            "pause": ["pause", "stop listening", "mute"],
            "resume": ["resume", "start listening", "unmute"],
            "interrupt": ["stop", "wait", "interrupt", "silence"]
        }
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for conversation events."""
        # User speech events
        event_bus.on(EventType.USER_SPEECH_STARTED, self._handle_user_speech_started)
        event_bus.on(EventType.USER_SPEECH_ONGOING, self._handle_user_speech_ongoing)
        event_bus.on(EventType.USER_SPEECH_FINISHED, self._handle_user_speech_finished)
        event_bus.on(EventType.USER_SPEECH_CANCELLED, self._handle_user_speech_cancelled)
        
        # Transcription events
        event_bus.on(EventType.USER_TRANSCRIPTION_COMPLETED, self._handle_transcription_completed)
        
        # Assistant response events
        event_bus.on(EventType.ASSISTANT_MESSAGE_STARTED, self._handle_assistant_message_started)
        event_bus.on(EventType.ASSISTANT_MESSAGE_CONTENT, self._handle_assistant_message_content)
        event_bus.on(EventType.ASSISTANT_MESSAGE_COMPLETED, self._handle_assistant_message_completed)
        
        # Audio events
        event_bus.on(EventType.AUDIO_SPEECH_CREATED, self._handle_audio_speech)
        
        # Function call events
        event_bus.on(EventType.FUNCTION_CALL_RECEIVED, self._handle_function_call_received)
        
        # Error events
        event_bus.on(EventType.ERROR, self._handle_error)
    
    async def start(self) -> bool:
        """
        Start the conversation session.
        
        This method connects to the OpenAI API and initializes the conversation.
        
        Returns:
            bool: True if the session started successfully
        """
        logger.info("Starting conversation session")
        self.history.set_state(ConversationStateType.CONNECTING)
        
        try:
            # Connect to the OpenAI Realtime API
            connected = await self.api_client.connect(
                assistant_id=self.context.assistant_id,
                session_id=self.context.session_id,
                instructions=self.context.instructions,
                temperature=self.context.temperature,
                enable_transcription=True
            )
            
            if not connected:
                logger.error("Failed to connect to OpenAI Realtime API")
                self.history.set_state(ConversationStateType.ERROR)
                
                # Emit error event
                error_event = ErrorEvent(
                    type=EventType.ERROR,
                    data={
                        "error": {
                            "message": "Failed to connect to OpenAI Realtime API",
                            "type": "connection_error"
                        }
                    },
                    error={
                        "message": "Failed to connect to OpenAI Realtime API",
                        "type": "connection_error"
                    }
                )
                event_bus.emit(error_event)
                
                return False
            
            # Add system message if instructions are provided
            if self.context.instructions:
                system_message = Message.create_system_message(self.context.instructions)
                self.history.add_message(system_message)
            
            # Configure audio manager for conversation mode
            if self.audio_manager:
                await self.audio_manager.start_recording()
            
            # Update session configuration
            await self._configure_session()
            
            # Update state
            self.history.set_state(ConversationStateType.READY)
            
            # Emit event for UI
            event_bus.emit(
                EventType.CONVERSATION_STATE_CHANGED,
                {
                    "state": self.history.state.value,
                    "session_id": self.context.session_id
                }
            )
            
            logger.info("Conversation session started successfully")
            return True
            
        except Exception as e:
            error = AppError(
                f"Failed to start conversation session: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
            
            self.history.set_state(ConversationStateType.ERROR)
            
            # Emit error event
            error_event = ErrorEvent(
                type=EventType.ERROR,
                data={"error": error.to_dict()},
                error=error.to_dict()
            )
            event_bus.emit(error_event)
            
            return False
    
    async def stop(self) -> None:
        """
        Stop the conversation session.
        
        This method disconnects from the API and cleans up resources.
        """
        logger.info("Stopping conversation session")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Update state
        self.history.set_state(ConversationStateType.DISCONNECTED)
        
        # Stop audio
        if self.audio_manager:
            await self.audio_manager.stop_recording()
            await self.audio_manager.stop_playback()
        
        # Disconnect from API
        if self.api_client:
            await self.api_client.disconnect()
        
        # Cancel all tasks
        await self.task_manager.cancel_all()
        
        logger.info("Conversation session stopped")
    
    async def _configure_session(self) -> bool:
        """
        Configure the session with initial settings.
        
        Returns:
            bool: True if configuration was successful
        """
        try:
            # Prepare session configuration
            session_config = {
                "instructions": self.context.instructions,
                "voice": self.context.voice,
                "temperature": self.context.temperature
            }
            
            # Add tools if available
            if self.context.tools:
                session_config["tools"] = self.context.tools
                session_config["tool_choice"] = self.context.tool_choice
            
            # Configure voice activity detection
            session_config["turn_detection"] = {
                "type": "server_vad",
                "threshold": 0.85,
                "prefix_padding_ms": 500,
                "silence_duration_ms": 1000,
                "create_response": self.auto_respond,
                "interrupt_response": True
            }
            
            # Update the session
            success = await self.api_client.update_session(session_config)
            
            if success:
                logger.info("Session configured successfully")
                return True
            else:
                logger.error("Failed to configure session")
                return False
                
        except Exception as e:
            logger.error(f"Error configuring session: {str(e)}")
            return False
    
    async def send_text_message(self, text: str) -> Optional[str]:
        """
        Send a text message from the user.
        
        Args:
            text: Text message to send
            
        Returns:
            Optional[str]: Message ID if sent successfully
        """
        if not text or not self.api_client.is_connected:
            return None
        
        try:
            # Create user message
            user_message = Message.create_user_message(text=text)
            
            # Add to history as pending
            self.history.add_pending_message(user_message)
            
            # For backwards compatibility
            self.messages.append({
                "role": "user",
                "content": text,
                "id": user_message.id,
                "created_at": user_message.created_at
            })
            
            # Create conversation item
            item_data = {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text
                    }
                ]
            }
            
            item_id = await self.api_client.create_conversation_item(item_data)
            
            if not item_id:
                logger.error("Failed to send text message")
                return None
            
            logger.info(f"Sent text message: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # If auto-respond is enabled, request a response
            if self.auto_respond:
                self.task_manager.create_task(
                    self.request_response(),
                    "auto_request_response"
                )
            
            return user_message.id
            
        except Exception as e:
            logger.error(f"Error sending text message: {str(e)}")
            return None
    
    async def request_response(
        self,
        modalities: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Request a response from the assistant.
        
        Args:
            modalities: Response modalities (text, audio)
            instructions: Optional specific instructions for this response
            metadata: Optional metadata for this response
            
        Returns:
            bool: True if request was successful
        """
        if not self.api_client.is_connected:
            logger.error("Cannot request response: not connected to API")
            return False
        
        try:
            # Update state to thinking
            self.history.set_state(ConversationStateType.THINKING)
            
            # Create default modalities if not provided
            if not modalities:
                modalities = ["text", "audio"]
            
            # Ensure long conversations don't exceed token limits
            self.history.truncate_history()
            
            # Request response from the API
            response_id = await self.api_client.request_response(
                instructions=instructions,
                metadata=metadata,
                modalities=modalities
            )
            
            if not response_id:
                logger.error("Failed to request response")
                self.history.set_state(ConversationStateType.IDLE)
                return False
            
            # Set active response ID
            self.active_response_id = response_id
            
            logger.info(f"Requested response (ID: {response_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error requesting response: {str(e)}")
            self.history.set_state(ConversationStateType.IDLE)
            return False
    
    async def interrupt(self) -> bool:
        """
        Interrupt the current assistant response.
        
        Returns:
            bool: True if interruption was successful
        """
        if self.history.state != ConversationStateType.ASSISTANT_SPEAKING:
            logger.warning("Cannot interrupt: assistant is not speaking")
            return False
        
        try:
            # Stop audio playback
            if self.audio_manager:
                await self.audio_manager.stop_playback()
            
            # Send interrupt signal to API
            success = await self.api_client.interrupt()
            
            if success:
                logger.info("Assistant interrupted")
                
                # Find latest assistant message and mark as interrupted
                last_assistant = self.history.get_last_assistant_message()
                if last_assistant:
                    self.history.mark_message_interrupted(last_assistant.id)
                
                # Update state
                self.history.set_state(ConversationStateType.IDLE)
                return True
            else:
                logger.error("Failed to interrupt assistant")
                return False
                
        except Exception as e:
            logger.error(f"Error interrupting response: {str(e)}")
            return False
    
    async def send_function_result(
        self,
        call_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """
        Send the result of a function call back to the API.
        
        Args:
            call_id: Function call ID
            result: Result data
            
        Returns:
            bool: True if result was sent successfully
        """
        if not self.api_client.is_connected or call_id not in self.pending_function_calls:
            logger.error(f"Cannot send function result: invalid call ID {call_id}")
            return False
        
        try:
            # Create function result message
            function_result = Message.create_function_result_message(call_id, result)
            
            # Add to history
            self.history.add_message(function_result)
            
            # Submit to API
            success = await self.api_client.submit_tool_outputs(
                [{
                    "call_id": call_id,
                    "output": json.dumps(result)
                }]
            )
            
            if success:
                logger.info(f"Sent function result for call {call_id}")
                
                # Remove from pending calls
                if call_id in self.pending_function_calls:
                    del self.pending_function_calls[call_id]
                
                return True
            else:
                logger.error(f"Failed to send function result for call {call_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending function result: {str(e)}")
            return False
    
    def register_function_handler(
        self,
        function_name: str,
        handler: Callable[[Dict[str, Any], str], Any]
    ) -> None:
        """
        Register a handler for a specific function.
        
        The handler will be called when the assistant calls the function.
        
        Args:
            function_name: Name of the function to handle
            handler: Function to call with (arguments, call_id)
        """
        # Create a wrapper to handle async/sync functions
        async def wrapper(event: Event) -> None:
            data = event.data
            name = data.get("name")
            
            if name != function_name:
                return
                
            call_id = data.get("call_id")
            arguments = data.get("arguments", {})
            
            try:
                # Call the handler and get the result
                result = handler(arguments, call_id)
                
                # If the result is a coroutine, await it
                if asyncio.iscoroutine(result):
                    result = await result
                    
                # Submit the result back
                if isinstance(result, dict):
                    self.task_manager.create_task(
                        self.send_function_result(call_id, result),
                        f"function_result_{call_id}"
                    )
            except Exception as e:
                logger.error(f"Error in function handler for {function_name}: {str(e)}")
                
                # Send error back to assistant
                error_result = {
                    "error": f"Error executing function {function_name}: {str(e)}"
                }
                
                self.task_manager.create_task(
                    self.send_function_result(call_id, error_result),
                    f"function_error_{call_id}"
                )
        
        # Register the wrapper with the event bus
        event_bus.on(EventType.FUNCTION_CALL_RECEIVED, wrapper)
    
    # Event handlers
    
    def _handle_user_speech_started(self, event: UserSpeechEvent) -> None:
        """
        Handle user speech started event.
        
        Args:
            event: User speech event
        """
        # If assistant is speaking, interrupt
        if self.history.state == ConversationStateType.ASSISTANT_SPEAKING:
            self.task_manager.create_task(
                self.interrupt(),
                "interrupt_on_speech"
            )
        
        # Update state
        self.history.set_state(ConversationStateType.USER_SPEAKING)
        
        logger.debug("User started speaking")
    
    def _handle_user_speech_ongoing(self, event: UserSpeechEvent) -> None:
        """
        Handle ongoing user speech event.
        
        Args:
            event: User speech event
        """
        # Nothing to do here, audio is already being sent to API through audio manager
        pass
    
    def _handle_user_speech_finished(self, event: UserSpeechEvent) -> None:
        """
        Handle user speech finished event.
        
        Args:
            event: User speech event
        """
        # Create pending user message for audio
        user_message = Message.create_user_message(
            audio_duration=event.duration,
            audio_id=str(uuid.uuid4())
        )
        
        # Add to history as pending (will be updated with transcription)
        self.history.add_pending_message(user_message)
        
        # State will be updated when transcription is received
        logger.debug("User finished speaking")
    
    def _handle_user_speech_cancelled(self, event: UserSpeechEvent) -> None:
        """
        Handle user speech cancelled event (e.g., too short).
        
        Args:
            event: User speech event
        """
        # Update state
        self.history.set_state(ConversationStateType.IDLE)
        
        logger.debug("User speech cancelled")
    
    def _handle_transcription_completed(self, event: TranscriptionEvent) -> None:
        """
        Handle transcription completed event.
        
        Args:
            event: Transcription event
        """
        text = event.text
        is_final = event.is_final
        
        if not text:
            # Empty transcription, ignore
            return
        
        # Add to transcriptions list for backward compatibility
        self.transcriptions.append({
            "text": text,
            "is_final": is_final,
            "timestamp": time.time()
        })
        
        # Check for voice commands
        if is_final and self._check_voice_command(text):
            # Voice command handled, don't process as normal message
            self.history.set_state(ConversationStateType.IDLE)
            return
        
        # Find pending user message to update with transcription
        pending_user_message = None
        for message in self.history.pending_messages:
            if message.role == MessageRole.USER and not message.content.text:
                pending_user_message = message
                break
        
        if pending_user_message:
            # Update the pending message with transcription
            pending_user_message.content.text = text
            pending_user_message.content.is_final = is_final
            
            # If final, move to history
            if is_final:
                self.history.add_message(pending_user_message)
                # Remove from pending
                self.history.pending_messages.remove(pending_user_message)
        else:
            # No pending message, create a new one
            user_message = Message.create_user_message(text=text)
            self.history.add_message(user_message)
        
        # Update messages list for backward compatibility
        self.messages.append({
            "role": "user",
            "content": text,
            "is_transcription": True,
            "created_at": time.time()
        })
        
        # Log the transcription
        logger.info(f"Transcription completed: '{text}'")
        
        # If final, update state and optionally request response
        if is_final:
            # Update state
            self.history.set_state(ConversationStateType.IDLE)
            
            # Emit state changed event
            event_bus.emit(
                EventType.CONVERSATION_STATE_CHANGED,
                {
                    "state": self.history.state.value,
                    "latest_message": text
                }
            )
            
            # Auto-respond if enabled
            if self.auto_respond:
                self.task_manager.create_task(
                    self.request_response(),
                    "auto_request_response"
                )
    
    def _handle_assistant_message_started(self, event: Event) -> None:
        """
        Handle assistant message started event.
        
        Args:
            event: Assistant message event
        """
        # Update state
        self.history.set_state(ConversationStateType.THINKING)
        
        logger.debug("Assistant message started")
    
    def _handle_assistant_message_content(self, event: Event) -> None:
        """
        Handle assistant message content event.
        
        Args:
            event: Assistant message content event
        """
        text = event.data.get("text", "")
        is_final = event.data.get("is_final", False)
        
        # If this is the first chunk, create a message
        if self.active_response_id not in self.messages:
            assistant_message = Message.create_assistant_message()
            self.history.add_pending_message(assistant_message)
            
            # For backward compatibility
            self.messages.append({
                "role": "assistant",
                "content": text,
                "id": assistant_message.id,
                "created_at": assistant_message.created_at
            })
            
            # Store the active response ID
            self.active_response_id = assistant_message.id
        
        # Update the pending message with new content
        for message in self.history.pending_messages:
            if message.id == self.active_response_id:
                message.content.text = text
                break
    
    def _handle_assistant_message_completed(self, event: Event) -> None:
        """
        Handle assistant message completed event.
        
        Args:
            event: Assistant message completed event
        """
        text = event.data.get("text", "")
        
        # Find the corresponding message
        assistant_message = None
        for message in self.history.pending_messages:
            if message.id == self.active_response_id:
                assistant_message = message
                break
        
        if assistant_message:
            # Update with final text
            assistant_message.content.text = text
            
            # Move from pending to history
            self.history.add_message(assistant_message)
            self.history.pending_messages.remove(assistant_message)
        else:
            # Create a new message
            assistant_message = Message.create_assistant_message(text=text)
            self.history.add_message(assistant_message)
        
        # Update backward compatibility messages
        for message in self.messages:
            if message.get("id") == self.active_response_id:
                message["content"] = text
                break
        
        # Reset active response ID
        self.active_response_id = None
        
        # Update state
        self.history.set_state(ConversationStateType.IDLE)
        
        # Emit state changed event
        event_bus.emit(
            EventType.CONVERSATION_STATE_CHANGED,
            {
                "state": self.history.state.value,
                "latest_message": text
            }
        )
        
        logger.debug("Assistant message completed")
    
    def _handle_audio_speech(self, event: AudioSpeechEvent) -> None:
        """
        Handle audio speech event.
        
        Args:
            event: Audio speech event
        """
        # Update state if not already speaking
        if self.history.state != ConversationStateType.ASSISTANT_SPEAKING:
            self.history.set_state(ConversationStateType.ASSISTANT_SPEAKING)
        
        # Play audio if we have audio manager
        if self.audio_manager and event.chunk:
            self.task_manager.create_task(
                self.audio_manager.play_audio(event.chunk),
                "play_audio_chunk"
            )
    
    def _handle_function_call_received(self, event: Event) -> None:
        """
        Handle function call received event.
        
        Args:
            event: Function call event
        """
        name = event.data.get("name")
        call_id = event.data.get("call_id")
        arguments = event.data.get("arguments", {})
        
        if not name or not call_id:
            logger.warning("Received invalid function call event")
            return
        
        logger.info(f"Received function call: {name} ({call_id})")
        
        # Store function call info
        self.pending_function_calls[call_id] = {
            "name": name,
            "arguments": arguments,
            "call_id": call_id,
            "received_at": time.time()
        }
        
        # Add to history
        function_call = Message.create_function_call_message(
            name=name,
            arguments=arguments,
            call_id=call_id
        )
        self.history.add_message(function_call)
        
        # Emit event for UI
        event_bus.emit(
            EventType.FUNCTION_CALL_RECEIVED,
            {
                "name": name,
                "arguments": arguments,
                "call_id": call_id
            }
        )
    
    def _handle_error(self, event: ErrorEvent) -> None:
        """
        Handle error event.
        
        Args:
            event: Error event
        """
        error = event.error
        
        # Log error
        logger.error(f"Error in conversation: {error.get('message', 'Unknown error')}")
        
        # Update state if in active state
        if self.history.state in (
            ConversationStateType.THINKING,
            ConversationStateType.ASSISTANT_SPEAKING,
            ConversationStateType.USER_SPEAKING
        ):
            self.history.set_state(ConversationStateType.IDLE)
    
    def _check_voice_command(self, text: str) -> bool:
        """
        Check if text contains a voice command and handle it.
        
        Args:
            text: Text to check for commands
            
        Returns:
            bool: True if a command was detected and handled
        """
        # Convert to lowercase and normalize
        normalized = text.lower().strip()
        
        # Check each voice command category
        for command_type, phrases in self.voice_commands.items():
            if any(phrase in normalized for phrase in phrases):
                logger.info(f"Voice command detected: {command_type}")
                
                if command_type == "goodbye":
                    # Emit shutdown event
                    event_bus.emit(
                        EventType.SHUTDOWN,
                        {
                            "reason": "user_exit_command",
                            "command": command_type,
                            "transcript": text
                        }
                    )
                    return True
                
                elif command_type == "pause":
                    # Pause audio input
                    if self.audio_manager:
                        self.task_manager.create_task(
                            self.audio_manager.pause_recording(),
                            "pause_recording"
                        )
                    return True
                
                elif command_type == "resume":
                    # Resume audio input
                    if self.audio_manager:
                        self.task_manager.create_task(
                            self.audio_manager.resume_recording(),
                            "resume_recording"
                        )
                    return True
                
                elif command_type == "interrupt":
                    # Interrupt assistant
                    if self.history.state == ConversationStateType.ASSISTANT_SPEAKING:
                        self.task_manager.create_task(
                            self.interrupt(),
                            "interrupt_command"
                        )
                    return True
        
        return False
    
    async def save_conversation(self, file_path: Union[str, Path]) -> bool:
        """
        Save the current conversation to a file.
        
        Args:
            file_path: Path to save the conversation
            
        Returns:
            bool: True if successfully saved
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        try:
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert conversation to JSON
            json_data = self.history.to_json()
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
            
            logger.info(f"Saved conversation to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save conversation: {str(e)}")
            return False
    
    @classmethod
    async def load_conversation(
        cls,
        file_path: Union[str, Path],
        audio_manager: Optional[AudioManager] = None,
        api_client: Optional[RealtimeClient] = None
    ) -> Optional['ConversationManager']:
        """
        Load a conversation from a file.
        
        Args:
            file_path: Path to the conversation file
            audio_manager: Optional audio manager to use
            api_client: Optional API client to use
            
        Returns:
            Optional[ConversationManager]: Loaded conversation manager, or None if loading failed
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = f.read()
            
            # Parse conversation history
            history = ConversationHistory.from_json(json_data)
            
            # Create manager
            manager = cls(
                assistant_id=history.context.assistant_id,
                instructions=history.context.instructions,
                temperature=history.context.temperature,
                audio_manager=audio_manager,
                api_client=api_client,
                tools=history.context.tools,
                voice=history.context.voice
            )
            
            # Replace history
            manager.history = history
            
            # Update backward compatibility properties
            for message in history.messages:
                manager.messages.append({
                    "role": message.role.value,
                    "content": message.content.text if hasattr(message.content, "text") else "",
                    "id": message.id,
                    "created_at": message.created_at
                })
            
            logger.info(f"Loaded conversation from {file_path}")
            return manager
            
        except Exception as e:
            logger.error(f"Failed to load conversation: {str(e)}")
            return None

# Factory function for dependency injection in tests
def create_conversation_manager(
    assistant_id: str,
    instructions: Optional[str] = None,
    temperature: float = 1.0,
    audio_manager: Optional[AudioManager] = None,
    api_client: Optional[RealtimeClient] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    voice: str = "alloy"
) -> ConversationManager:
    """
    Create a conversation manager with the given parameters.
    
    Args:
        assistant_id: ID of the OpenAI assistant to use
        instructions: Optional custom instructions for the assistant
        temperature: Temperature parameter for generation (0.0-2.0)
        audio_manager: Optional audio manager instance to use
        api_client: Optional API client instance to use
        tools: Optional list of tools for function calling
        voice: Voice ID for audio responses
        
    Returns:
        ConversationManager: The created conversation manager
    """
    return ConversationManager(
        assistant_id=assistant_id,
        instructions=instructions,
        temperature=temperature,
        audio_manager=audio_manager,
        api_client=api_client,
        tools=tools,
        voice=voice
    )
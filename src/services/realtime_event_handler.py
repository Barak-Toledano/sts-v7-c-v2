"""
OpenAI Realtime API Event Handler.

This module centralizes the handling of all events from the OpenAI Realtime API.
It maps event types to handler functions and ensures consistent event processing.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from src.config import settings
from src.config.logging_config import get_logger
from src.events.event_interface import (
    AudioSpeechEvent,
    ErrorEvent,
    Event,
    EventType,
    TranscriptionEvent,
    UserSpeechEvent,
    event_bus,
)
from src.utils.error_handling import ApiError, ErrorSeverity
from src.utils.transcription import extract_transcription_from_realtime_event

logger = get_logger(__name__)


class RealtimeEventHandler:
    """
    Handler for processing OpenAI Realtime API events.
    
    This class centralizes the processing of all event types and 
    maps them to appropriate handler functions.
    """
    
    def __init__(self, client_ref=None):
        """
        Initialize the event handler.
        
        Args:
            client_ref: Optional reference to the RealtimeClient instance
        """
        self.client = client_ref
        self.server_events = self._load_server_events()
        self.client_events = self._load_client_events()
        self._setup_event_handlers()
        
        # Track state for function calls and responses
        self.pending_function_calls = {}
        self.active_responses = {}
        
        logger.debug("RealtimeEventHandler initialized")
    
    def _load_server_events(self) -> Dict[str, str]:
        """Load server event definitions."""
        # Predefined set of known server events from the API documentation
        return {
            "session.created": "Session creation confirmation",
            "session.updated": "Session update confirmation",
            "conversation.created": "New conversation creation",
            "conversation.item.created": "New conversation item creation",
            "conversation.item.input_audio_transcription.completed": "Input audio transcription completion",
            "conversation.item.input_audio_transcription.failed": "Input audio transcription failure",
            "conversation.item.truncated": "Conversation items truncation",
            "conversation.item.deleted": "Conversation item deletion",
            "input_audio_buffer.committed": "Audio buffer commit acknowledgement",
            "input_audio_buffer.cleared": "Audio buffer clear acknowledgement",
            "input_audio_buffer.speech_started": "Speech detection in audio input",
            "input_audio_buffer.speech_stopped": "Speech end detection in audio input",
            "response.created": "Response creation confirmation",
            "response.done": "Response generation completion",
            "response.output_item.added": "New output item addition",
            "response.output_item.done": "Output items completion",
            "response.content_part.added": "New content part addition", 
            "response.content_part.done": "Content parts completion",
            "response.text.delta": "Incremental text data",
            "response.text.done": "Text data completion",
            "response.audio_transcript.delta": "Incremental audio transcript",
            "response.audio_transcript.done": "Audio transcript completion",
            "response.audio.delta": "Incremental audio data",
            "response.audio.done": "Audio data completion",
            "response.function_call_arguments.delta": "Incremental function call arguments",
            "response.function_call_arguments.done": "Function call arguments completion",
            "rate_limits.updated": "Rate limits update",
            "error": "Error notification"
        }
    
    def _load_client_events(self) -> Dict[str, str]:
        """Load client event definitions."""
        # Predefined set of known client events from the API documentation
        standard_events = {
            "session.update": "Update session settings",
            "input_audio_buffer.append": "Append audio data to input buffer",
            "input_audio_buffer.commit": "Signal end of audio input",
            "input_audio_buffer.clear": "Clear current audio buffer",
            "conversation.item.create": "Add new conversation item",
            "conversation.item.truncate": "Remove conversation items",
            "conversation.item.delete": "Delete conversation item",
            "response.create": "Request response generation",
            "response.cancel": "Cancel response generation"
        }
        
        # Custom extensions implemented by our application
        custom_events = {
            "audio": "Send audio data directly (custom implementation)",
            "interrupt": "Interrupt assistant response (custom implementation)",
            "heartbeat": "Keep connection alive (custom implementation)"
        }
        
        # Combine standard and custom events
        return {**standard_events, **custom_events}
    
    def _setup_event_handlers(self) -> None:
        """Set up mapping between event types and handler functions."""
        # Map event types to their respective handlers
        self.event_handlers = {
            # Session events
            "session.created": self.handle_session_created,
            "session.updated": self.handle_session_updated,
            
            # Conversation events
            "conversation.created": self.handle_conversation_created,
            "conversation.item.created": self.handle_conversation_item_created,
            
            # Transcription events
            "conversation.item.input_audio_transcription.completed": self.handle_transcription_completed,
            "conversation.item.input_audio_transcription.failed": self.handle_transcription_failed,
            
            # Input audio buffer events
            "input_audio_buffer.committed": self.handle_buffer_committed,
            "input_audio_buffer.cleared": self.handle_buffer_cleared,
            "input_audio_buffer.speech_started": self.handle_speech_started,
            "input_audio_buffer.speech_stopped": self.handle_speech_stopped,
            
            # Response events
            "response.created": self.handle_response_created,
            "response.done": self.handle_response_done,
            "response.output_item.added": self.handle_output_item_added,
            "response.output_item.done": self.handle_output_item_done,
            "response.content_part.added": self.handle_content_part_added,
            "response.content_part.done": self.handle_content_part_done,
            "response.text.delta": self.handle_text_delta,
            "response.text.done": self.handle_text_done,
            "response.audio.delta": self.handle_audio_delta,
            "response.audio.done": self.handle_audio_done,
            "response.audio_transcript.delta": self.handle_audio_transcript_delta,
            "response.audio_transcript.done": self.handle_audio_transcript_done,
            "response.function_call_arguments.delta": self.handle_function_call_arguments_delta,
            "response.function_call_arguments.done": self.handle_function_call_arguments_done,
            
            # Rate limits and errors
            "rate_limits.updated": self.handle_rate_limits_updated,
            "error": self.handle_error
        }
    
    def set_client(self, client_ref) -> None:
        """
        Set the client reference.
        
        Args:
            client_ref: Reference to the RealtimeClient instance
        """
        self.client = client_ref
    
    def is_valid_event_type(self, event_type: str, is_client_event: bool = False) -> bool:
        """
        Check if an event type is valid according to the API documentation.
        
        Args:
            event_type: Type of the event to validate
            is_client_event: Whether this is a client event (vs server event)
            
        Returns:
            bool: True if the event type is valid
        """
        if is_client_event:
            return event_type in self.client_events
        else:
            return event_type in self.server_events
    
    def log_custom_event_usage(self, event_type: str) -> None:
        """
        Log when a custom event is used.
        
        Args:
            event_type: The type of event being used
        """
        custom_events = {"audio", "interrupt", "heartbeat"}
        if event_type in custom_events:
            logger.info(f"Using custom client event: {event_type} - {self.client_events.get(event_type, '')}")
    
    async def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Process an event from the OpenAI Realtime API.
        
        This is the main entry point for event handling. It validates
        the event type and dispatches it to the appropriate handler.
        
        Args:
            event_type: Type of the event
            event_data: Event data payload
        """
        # Validate event type
        if not self.is_valid_event_type(event_type):
            if settings.debug_mode:
                logger.warning(f"Received unknown event type: {event_type}")
            return
        
        # Log event for debugging
        log_level = logging.DEBUG
        if event_type == "error":
            log_level = logging.ERROR
        elif event_type.startswith("session."):
            log_level = logging.INFO
            
        logger.log(log_level, f"Processing event: {event_type}")
        
        # Get the appropriate handler
        handler = self.event_handlers.get(event_type)
        
        if handler:
            try:
                # Call the handler
                await handler(event_data)
            except Exception as e:
                logger.error(f"Error handling event {event_type}: {str(e)}")
                
                # Create an error event
                error_event = ErrorEvent(
                    type=EventType.ERROR,
                    data={
                        "error": {
                            "message": f"Error handling event {event_type}: {str(e)}",
                            "type": "event_handler_error"
                        }
                    }
                )
                
                # Emit the error event
                event_bus.emit(error_event)
        else:
            # Default handling for events without specific handlers
            if settings.debug_mode:
                logger.debug(f"No specific handler for event type: {event_type}")
            
            # Create a generic event
            event = Event(
                type=EventType.from_string(event_type),
                data=event_data
            )
            
            # Emit the event to the event bus
            event_bus.emit(event)
    
    # Session event handlers
    
    async def handle_session_created(self, event_data: Dict[str, Any]) -> None:
        """
        Handle session created event.
        
        Args:
            event_data: Event data
        """
        session = event_data.get("session", {})
        session_id = session.get("id")
        
        if session_id and self.client:
            # Update client's session ID
            self.client.session_id = session_id
        
        logger.info(f"Session created: {session_id}")
        
        # Create and emit event
        event = Event(
            type=EventType.SESSION_CREATED,
            data=event_data
        )
        event_bus.emit(event)
    
    async def handle_session_updated(self, event_data: Dict[str, Any]) -> None:
        """
        Handle session updated event.
        
        Args:
            event_data: Event data
        """
        session = event_data.get("session", {})
        session_id = session.get("id")
        
        logger.debug(f"Session updated: {session_id}")
        
        # Create and emit event
        event = Event(
            type=EventType.SESSION_UPDATED,
            data=event_data
        )
        event_bus.emit(event)
    
    # Conversation event handlers
    
    async def handle_conversation_created(self, event_data: Dict[str, Any]) -> None:
        """
        Handle conversation created event.
        
        Args:
            event_data: Event data
        """
        conversation_id = event_data.get("id")
        logger.info(f"Conversation created: {conversation_id}")
        
        # Create and emit event
        event = Event(
            type=EventType.CONVERSATION_CREATED,
            data=event_data
        )
        event_bus.emit(event)
    
    async def handle_conversation_item_created(self, event_data: Dict[str, Any]) -> None:
        """
        Handle conversation item created event.
        
        Args:
            event_data: Event data
        """
        item = event_data.get("item", {})
        item_id = item.get("id")
        item_type = item.get("type")
        
        logger.debug(f"Conversation item created: {item_type} (ID: {item_id})")
        
        # Create and emit event
        event = Event(
            type=EventType.MESSAGE_CREATED,
            data=event_data
        )
        event_bus.emit(event)
    
    # Transcription event handlers
    
    async def handle_transcription_completed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle transcription completed event.
        
        Args:
            event_data: Event data
        """
        try:
            # Extract transcription from the event data
            transcription_data = extract_transcription_from_realtime_event(event_data)
            
            if transcription_data["text"]:
                # Create transcription event
                transcription_event = TranscriptionEvent(
                    type=EventType.USER_TRANSCRIPTION_COMPLETED,
                    text=transcription_data["text"],
                    is_final=transcription_data["is_final"],
                    confidence=1.0,  # Default confidence
                    source="whisper"
                )
                
                # Emit the event
                event_bus.emit(transcription_event)
                
                logger.info(f"Transcription: '{transcription_data['text']}'")
            else:
                logger.warning("Empty transcription received")
        except Exception as e:
            logger.error(f"Error handling transcription event: {e}")
            
            # Create error event
            error_event = ErrorEvent(
                type=EventType.ERROR,
                data={
                    "error": {
                        "message": f"Error processing transcription: {str(e)}",
                        "type": "transcription_error"
                    }
                }
            )
            event_bus.emit(error_event)
    
    async def handle_transcription_failed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle transcription failed event.
        
        Args:
            event_data: Event data
        """
        error = event_data.get("error", {})
        error_message = error.get("message", "Unknown error")
        error_type = error.get("type", "unknown")
        
        logger.error(f"Transcription failed: {error_type} - {error_message}")
        
        # Create error event
        error_event = ErrorEvent(
            type=EventType.ERROR,
            data={
                "error": {
                    "message": f"Transcription failed: {error_message}",
                    "type": error_type
                }
            }
        )
        event_bus.emit(error_event)
    
    # Input audio buffer event handlers
    
    async def handle_buffer_committed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle audio buffer committed event.
        
        Args:
            event_data: Event data
        """
        logger.debug("Audio buffer committed")
        
        # Create and emit event
        event = Event(
            type=EventType.AUDIO_BUFFER_COMMITTED,
            data=event_data
        )
        event_bus.emit(event)
    
    async def handle_buffer_cleared(self, event_data: Dict[str, Any]) -> None:
        """
        Handle audio buffer cleared event.
        
        Args:
            event_data: Event data
        """
        logger.debug("Audio buffer cleared")
        
        # Create and emit event
        event = Event(
            type=EventType.AUDIO_BUFFER_CLEARED,
            data=event_data
        )
        event_bus.emit(event)
    
    async def handle_speech_started(self, event_data: Dict[str, Any]) -> None:
        """
        Handle speech started event.
        
        Args:
            event_data: Event data
        """
        logger.debug("Speech started")
        
        # Create speech event
        speech_event = UserSpeechEvent(
            type=EventType.USER_SPEECH_STARTED,
            data=event_data,
            is_final=False
        )
        event_bus.emit(speech_event)
    
    async def handle_speech_stopped(self, event_data: Dict[str, Any]) -> None:
        """
        Handle speech stopped event.
        
        Args:
            event_data: Event data
        """
        logger.debug("Speech stopped")
        
        # Create speech event
        speech_event = UserSpeechEvent(
            type=EventType.USER_SPEECH_FINISHED,
            data=event_data,
            is_final=True
        )
        event_bus.emit(speech_event)
    
    # Response event handlers
    
    async def handle_response_created(self, event_data: Dict[str, Any]) -> None:
        """
        Handle response created event.
        
        Args:
            event_data: Event data
        """
        response = event_data.get("response", {})
        response_id = response.get("id")
        
        if response_id:
            # Track the response
            self.active_responses[response_id] = {
                "id": response_id,
                "status": "created",
                "created_at": time.time(),
                "text": "",
                "output_items": []
            }
        
        logger.debug(f"Response created: {response_id}")
        
        # Create and emit event
        event = Event(
            type=EventType.RESPONSE_CREATED,
            data=event_data
        )
        event_bus.emit(event)
    
    async def handle_response_done(self, event_data: Dict[str, Any]) -> None:
        """
        Handle response done event.
        
        Args:
            event_data: Event data
        """
        response = event_data.get("response", {})
        response_id = response.get("id")
        
        if response_id and response_id in self.active_responses:
            # Update response status
            self.active_responses[response_id]["status"] = "completed"
            self.active_responses[response_id]["output"] = response.get("output", [])
            self.active_responses[response_id]["usage"] = response.get("usage", {})
            
            # Process any function calls in the output
            for output_item in response.get("output", []):
                if output_item.get("type") == "function_call":
                    call_id = output_item.get("call_id")
                    function_name = output_item.get("name")
                    arguments = output_item.get("arguments", "{}")
                    
                    if call_id and function_name:
                        # Parse arguments
                        try:
                            args_dict = json.loads(arguments)
                        except json.JSONDecodeError:
                            args_dict = {"raw_arguments": arguments}
                        
                        # Create function call event
                        event_bus.emit(
                            EventType.FUNCTION_CALL_RECEIVED,
                            {
                                "function_name": function_name,
                                "arguments": args_dict,
                                "call_id": call_id,
                                "response_id": response_id
                            }
                        )
        
        logger.debug(f"Response completed: {response_id}")
        
        # Create and emit event
        event = Event(
            type=EventType.MESSAGE_COMPLETED,
            data=event_data
        )
        event_bus.emit(event)
    
    async def handle_output_item_added(self, event_data: Dict[str, Any]) -> None:
        """
        Handle output item added event.
        
        Args:
            event_data: Event data
        """
        response_id = event_data.get("response_id")
        item = event_data.get("item", {})
        item_id = item.get("id")
        item_type = item.get("type")
        
        logger.debug(f"Output item added: {item_type} (ID: {item_id})")
        
        # Update tracked response if available
        if response_id in self.active_responses:
            if "output_items" not in self.active_responses[response_id]:
                self.active_responses[response_id]["output_items"] = []
            
            self.active_responses[response_id]["output_items"].append(item)
    
    async def handle_output_item_done(self, event_data: Dict[str, Any]) -> None:
        """
        Handle output item done event.
        
        Args:
            event_data: Event data
        """
        logger.debug("Output item completed")
    
    async def handle_content_part_added(self, event_data: Dict[str, Any]) -> None:
        """
        Handle content part added event.
        
        Args:
            event_data: Event data
        """
        logger.debug("Content part added")
    
    async def handle_content_part_done(self, event_data: Dict[str, Any]) -> None:
        """
        Handle content part done event.
        
        Args:
            event_data: Event data
        """
        logger.debug("Content part completed")
    
    async def handle_text_delta(self, event_data: Dict[str, Any]) -> None:
        """
        Handle text delta event.
        
        Args:
            event_data: Event data
        """
        delta = event_data.get("delta", "")
        response_id = event_data.get("response_id")
        
        if not delta:
            return
        
        # Update tracked response if available
        if response_id in self.active_responses:
            if "text" not in self.active_responses[response_id]:
                self.active_responses[response_id]["text"] = ""
            
            self.active_responses[response_id]["text"] += delta
        
        # Create and emit event
        event = Event(
            type=EventType.TEXT_CREATED,
            data={
                "delta": delta,
                "response_id": response_id
            }
        )
        event_bus.emit(event)
    
    async def handle_text_done(self, event_data: Dict[str, Any]) -> None:
        """
        Handle text done event.
        
        Args:
            event_data: Event data
        """
        logger.debug("Text response completed")
    
    async def handle_audio_delta(self, event_data: Dict[str, Any]) -> None:
        """
        Handle audio delta event.
        
        Args:
            event_data: Event data
        """
        audio_data = event_data.get("delta", "")
        response_id = event_data.get("response_id")
        
        if not audio_data:
            return
        
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Create and emit audio speech event
            audio_event = AudioSpeechEvent(
                type=EventType.AUDIO_SPEECH_CREATED,
                data={
                    "response_id": response_id
                },
                chunk=audio_bytes
            )
            event_bus.emit(audio_event)
        except Exception as e:
            logger.error(f"Error decoding audio data: {e}")
    
    async def handle_audio_done(self, event_data: Dict[str, Any]) -> None:
        """
        Handle audio done event.
        
        Args:
            event_data: Event data
        """
        response_id = event_data.get("response_id")
        logger.debug(f"Audio response completed for {response_id}")
        
        # Create and emit event
        event = Event(
            type=EventType.AUDIO_PLAYBACK_COMPLETED,
            data=event_data
        )
        event_bus.emit(event)
    
    async def handle_audio_transcript_delta(self, event_data: Dict[str, Any]) -> None:
        """
        Handle audio transcript delta event.
        
        Args:
            event_data: Event data
        """
        delta = event_data.get("delta", "")
        part_id = event_data.get("part_id")
        response_id = event_data.get("response_id")
        
        if not delta:
            return
            
        logger.debug(f"Audio transcript delta: '{delta}'")
    
    async def handle_audio_transcript_done(self, event_data: Dict[str, Any]) -> None:
        """
        Handle audio transcript done event.
        
        Args:
            event_data: Event data
        """
        logger.debug("Audio transcript completed")
    
    async def handle_function_call_arguments_delta(self, event_data: Dict[str, Any]) -> None:
        """
        Handle function call arguments delta event.
        
        Args:
            event_data: Event data
        """
        delta = event_data.get("delta", "")
        call_id = event_data.get("call_id")
        function_name = event_data.get("name")
        response_id = event_data.get("response_id")
        
        if not delta or not call_id:
            return
        
        # Track function call arguments
        if call_id not in self.pending_function_calls:
            self.pending_function_calls[call_id] = {
                "id": call_id,
                "name": function_name,
                "arguments": "",
                "response_id": response_id,
                "status": "in_progress"
            }
        
        # Append to arguments
        self.pending_function_calls[call_id]["arguments"] += delta
        
        logger.debug(f"Function call argument delta for {function_name}: '{delta}'")
    
    async def handle_function_call_arguments_done(self, event_data: Dict[str, Any]) -> None:
        """
        Handle function call arguments done event.
        
        Args:
            event_data: Event data
        """
        call_id = event_data.get("call_id")
        function_name = event_data.get("name")
        response_id = event_data.get("response_id")
        arguments = event_data.get("arguments", "")
        
        if not call_id:
            return
        
        logger.info(f"Function call arguments completed for {function_name}")
        
        # Update tracked function call
        if call_id in self.pending_function_calls:
            self.pending_function_calls[call_id]["status"] = "arguments_done"
            if arguments:
                self.pending_function_calls[call_id]["arguments"] = arguments
        
        # Parse arguments
        try:
            args_dict = json.loads(arguments)
        except json.JSONDecodeError:
            args_dict = {"raw_arguments": arguments}
        
        # Create function call event
        event_bus.emit(
            EventType.FUNCTION_CALL_RECEIVED,
            {
                "function_name": function_name,
                "arguments": args_dict,
                "call_id": call_id,
                "response_id": response_id
            }
        )
    
    async def handle_rate_limits_updated(self, event_data: Dict[str, Any]) -> None:
        """
        Handle rate limits updated event.
        
        Args:
            event_data: Event data containing rate limit information
        """
        rate_limits = event_data.get("rate_limits", {})
        requests_remaining = rate_limits.get("requests", {}).get("remaining", 0)
        tokens_remaining = rate_limits.get("tokens", {}).get("remaining", 0)
        
        logger.debug(f"Rate limits updated: {requests_remaining} requests, {tokens_remaining} tokens remaining")
    
    async def handle_error(self, event_data: Dict[str, Any]) -> None:
        """
        Handle error event.
        
        Args:
            event_data: Event data
        """
        error_type = event_data.get("type", "unknown_error")
        error_message = event_data.get("message", "Unknown error")
        error_code = event_data.get("code", "unknown")
        event_id = event_data.get("event_id")
        
        # Log with appropriate level
        if "rate_limit" in error_code:
            logger.warning(f"API rate limit error: {error_message}")
        else:
            logger.error(f"API error: {error_type} ({error_code}): {error_message}")
        
        # Create an error event
        error_event = ErrorEvent(
            type=EventType.ERROR,
            data={
                "error": {
                    "type": error_type,
                    "code": error_code,
                    "message": error_message,
                    "event_id": event_id
                }
            },
            error={
                "type": error_type,
                "code": error_code,
                "message": error_message,
                "event_id": event_id
            }
        )
        event_bus.emit(error_event)


# Create a singleton instance
event_handler = RealtimeEventHandler()
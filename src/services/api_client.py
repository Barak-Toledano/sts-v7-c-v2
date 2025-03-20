"""
API client for the OpenAI Realtime API.

This module handles communication with the OpenAI Realtime API,
including initializing and maintaining WebSocket connections,
sending events, and dispatching received events.
"""

import asyncio
import base64
import json
import logging
import ssl
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
    WebSocketException,
)

from src.config import settings
from src.config.logging_config import get_logger
from src.events.event_interface import (
    AudioSpeechEvent,
    ErrorEvent,
    Event,
    EventType,
    event_bus,
    TranscriptionEvent,
)
from src.services.realtime_event_handler import event_handler
from src.utils.async_helpers import TaskManager, run_with_timeout
from src.utils.error_handling import ApiError, ErrorSeverity, safe_execute

logger = get_logger(__name__)


class RealtimeClient:
    """
    Client for interacting with the OpenAI Realtime API via WebSockets.
    
    This class handles:
    - Establishing and maintaining WebSocket connections
    - Sending events (audio, control messages, etc.)
    - Processing received events and dispatching to event handlers
    - Auto-reconnection and session management
    """
    
    def __init__(self):
        """Initialize the Realtime API client."""
        self.api_key = settings.api.api_key
        self.api_base = settings.api.api_base
        
        # Session state
        self.session_id: Optional[str] = None
        self.thread_id: Optional[str] = None
        self.run_id: Optional[str] = None
        self.message_id: Optional[str] = None
        self.connected = False
        self.reconnecting = False
        self.first_connect = True
        
        # WebSocket connection
        self.ws: Optional[WebSocketClientProtocol] = None
        self.task_manager = TaskManager()
        
        # Connection settings
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0  # Initial delay in seconds
        self.max_reconnect_delay = 30.0  # Maximum delay in seconds
        self.connection_timeout = 10.0  # Timeout for connection attempts
        
        # Event sequence tracking
        self.event_sequence = 0
        self.pending_events: Dict[str, Dict[str, Any]] = {}  # Track events for error correlation
        
        # Voice activity detection state
        self.vad_enabled = True
        self.speech_active = False
        
        # Set maximum size for audio chunks
        self.max_audio_chunk_size = 15 * 1024 * 1024  # 15MB limit per OpenAI docs
        
        # Set reference to this client in the event handler
        event_handler.set_client(self)
    
    async def connect(
        self,
        assistant_id: str,
        session_id: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = 1.0,
        enable_transcription: bool = True,
    ) -> bool:
        """
        Connect to the OpenAI Realtime API.
        
        Args:
            assistant_id: ID of the OpenAI assistant to use
            session_id: Optional session ID (generated if not provided)
            instructions: Optional system instructions
            temperature: Temperature parameter for generation (0.0-2.0)
            enable_transcription: Whether to enable Whisper transcription
            
        Returns:
            bool: True if the connection was successful
        """
        if self.connected:
            logger.warning("Already connected to OpenAI Realtime API")
            return True
        
        logger.info(f"Connecting to OpenAI Realtime API with assistant {assistant_id}")
        
        # Store the connection parameters for potential reconnection
        self.connect_params = {
            "assistant_id": assistant_id,
            "session_id": session_id,
            "instructions": instructions,
            "temperature": temperature,
            "enable_transcription": enable_transcription
        }
        
        try:
            # Generate a session ID if not provided
            if not session_id:
                session_id = f"session_{uuid.uuid4().hex}"
            
            self.thread_id = assistant_id
            self.session_id = session_id
            
            # Create URL with query parameters
            url = f"{settings.api.base_url}/v1/realtime?model={settings.api.model}"
            
            # Setup headers
            headers = {
                "Authorization": f"Bearer {settings.api.key}",
                "OpenAI-Beta": "realtime=v1",
            }
            
            # Connect to WebSocket
            self.ws = await websockets.connect(
                url,
                extra_headers=headers,
                max_size=None,  # No limit on message size
                ping_interval=30,  # 30 seconds ping interval
                ping_timeout=10,  # 10 seconds ping timeout
            )
            
            # Start message handling task
            self.task_manager.create_task(
                self._message_handler(),
                "websocket_message_handler"
            )
            
            # Start heartbeat task for connection monitoring
            self.task_manager.create_task(
                self._heartbeat_loop(),
                "websocket_heartbeat"
            )
            
            # Wait for session.created event
            session_created = await self._wait_for_session_created()
            
            if not session_created:
                raise ApiError(
                    "Failed to receive session.created event",
                    severity=ErrorSeverity.ERROR
                )
            
            # Configure session with provided parameters
            session_config = {
                "voice": "alloy",  # Default voice
                "instructions": instructions or "",
                "temperature": temperature,
                "turn_detection": {"type": "server_vad"},
                "modalities": ["audio", "text"],
            }
            
            # Add transcription configuration if enabled
            if enable_transcription:
                from src.utils.transcription import generate_realtime_session_config
                transcription_config = generate_realtime_session_config()
                session_config.update(transcription_config)
            
            # Update session configuration
            success = await self.update_session(session_config)
            
            if not success:
                raise ApiError(
                    "Failed to update session configuration",
                    severity=ErrorSeverity.ERROR
                )
            
            self.connected = True
            logger.info("Successfully connected to OpenAI Realtime API")
            
            return True
        
        except Exception as e:
            # Clean up websocket if it was created
            if self.ws:
                await self.ws.close()
                self.ws = None
            
            error = ApiError(
                f"Failed to connect to OpenAI Realtime API: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
            
            # Emit error event
            event_bus.emit(EventType.ERROR, {"error": error.to_dict()})
            
            return False
    
    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to the OpenAI Realtime API.
        
        Returns:
            bool: True if reconnection was successful
        """
        if not hasattr(self, 'connect_params'):
            logger.error("Cannot reconnect: no previous connection parameters")
            return False
        
        self.reconnecting = True
        
        # Close existing connection if it exists
        await self.disconnect(clear_session=False)
        
        # Attempt reconnection with exponential backoff
        attempt = 0
        delay = self.reconnect_delay
        
        while attempt < self.max_reconnect_attempts:
            attempt += 1
            logger.info(f"Reconnection attempt {attempt}/{self.max_reconnect_attempts}")
            
            try:
                result = await self.connect(**self.connect_params)
                if result:
                    logger.info(f"Reconnected successfully on attempt {attempt}")
                    self.reconnecting = False
                    return True
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt} failed: {str(e)}")
            
            # Wait before next attempt with exponential backoff
            logger.info(f"Waiting {delay:.1f}s before next reconnection attempt")
            await asyncio.sleep(delay)
            
            # Increase delay with exponential backoff (2x), capped at max_reconnect_delay
            delay = min(delay * 2, self.max_reconnect_delay)
        
        logger.error(f"Failed to reconnect after {self.max_reconnect_attempts} attempts")
        self.reconnecting = False
        return False
    
    async def disconnect(self, clear_session: bool = True) -> None:
        """
        Disconnect from the OpenAI Realtime API.
        
        Args:
            clear_session: Whether to clear session IDs
        """
        # Cancel all background tasks
        await self.task_manager.cancel_all()
        
        # Close WebSocket connection
        if self.ws:
            try:
                await self.ws.close()
                logger.info("Disconnected from OpenAI Realtime API")
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {str(e)}")
            finally:
                self.ws = None
        
        # Reset connection state
        self.connected = False
        
        # Optionally clear session information
        if clear_session:
            self.session_id = None
            self.run_id = None
            self.message_id = None
    
    async def send_audio(self, audio_data: bytes, is_final: bool = False) -> bool:
        """
        Send audio data to the OpenAI Realtime API.
        
        Args:
            audio_data: Audio data as bytes (16-bit PCM, 24kHz, mono)
            is_final: Whether this is the final chunk in the stream
            
        Returns:
            bool: True if audio was sent successfully
        """
        if not self.connected or not self.ws:
            logger.error("Cannot send audio: not connected")
            return False
        
        if not audio_data:
            logger.warning("Cannot send empty audio data")
            return False
        
        try:
            # Check chunk size
            if len(audio_data) > self.max_audio_chunk_size:
                raise ValueError(f"Audio chunk size ({len(audio_data)} bytes) exceeds maximum allowed size (15MB)")
            
            # Encode audio data as base64
            encoded_audio = base64.b64encode(audio_data).decode('utf-8')
            
            # If this is a final chunk, we should commit the buffer
            if is_final:
                # First append the final chunk
                await self._send_event({
                    "type": "input_audio_buffer.append",
                    "audio": encoded_audio,
                })
                
                # Then commit the buffer
                await self._send_event({
                    "type": "input_audio_buffer.commit",
                })
                
                logger.debug(f"Sent final audio chunk ({len(audio_data)} bytes) and committed buffer")
            else:
                # Just append to the buffer
                await self._send_event({
                    "type": "input_audio_buffer.append",
                    "audio": encoded_audio,
                })
                
                logger.debug(f"Sent audio chunk ({len(audio_data)} bytes)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending audio data: {str(e)}")
            return False
    
    async def append_audio_chunks(self, audio_chunks: List[bytes]) -> bool:
        """
        Append multiple audio chunks to the input audio buffer.
        
        Args:
            audio_chunks: List of audio byte chunks to append
            
        Returns:
            bool: True if all chunks were sent successfully
        """
        if not audio_chunks:
            return True
            
        success = True
        for i, chunk in enumerate(audio_chunks):
            is_final = (i == len(audio_chunks) - 1)
            result = await self.send_audio(chunk, is_final=is_final)
            if not result:
                success = False
                
        return success
    
    async def commit_audio_buffer(self) -> bool:
        """
        Commit the audio buffer to create a user message.
        
        This is used when VAD is disabled to manually indicate
        the end of speech input.
        
        Returns:
            bool: True if successful
        """
        if not self.connected or not self.ws:
            logger.error("Cannot commit audio buffer: not connected")
            return False
            
        try:
            await self._send_event({
                "type": "input_audio_buffer.commit"
            })
            return True
        except Exception as e:
            logger.error(f"Error committing audio buffer: {str(e)}")
            return False
    
    async def clear_audio_buffer(self) -> bool:
        """
        Clear the input audio buffer.
        
        Returns:
            bool: True if successful
        """
        if not self.connected or not self.ws:
            logger.error("Cannot clear audio buffer: not connected")
            return False
            
        try:
            await self._send_event({
                "type": "input_audio_buffer.clear"
            })
            return True
        except Exception as e:
            logger.error(f"Error clearing audio buffer: {str(e)}")
            return False
    
    async def request_response(self, instructions: Optional[str] = None) -> bool:
        """
        Request a response from the OpenAI Realtime API.
        
        Args:
            instructions: Optional instructions to override assistant's default instructions
            
        Returns:
            bool: True if request was sent successfully
        """
        if not self.connected or not self.ws:
            logger.error("Cannot request response: not connected")
            return False
        
        try:
            # Prepare request
            request = {
                "type": "response.create",
                "response": {}
            }
            
            # Add instructions if provided
            if instructions:
                request["response"]["instructions"] = instructions
            
            # Send request
            event_id = await self._send_event(request)
            if event_id:
                logger.info("Requested response from assistant")
                return True
            else:
                logger.error("Failed to send response request")
                return False
            
        except Exception as e:
            logger.error(f"Error requesting response: {str(e)}")
            return False
    
    async def create_response(
        self, 
        modalities: List[str] = None, 
        instructions: Optional[str] = None,
        input_audio_format: Optional[Dict[str, Any]] = None,
        output_audio_format: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        conversation: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        input_items: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """
        Request the model to generate a response with detailed options.
        
        Args:
            modalities: List of response modalities ("text", "audio")
            instructions: Special instructions for this response
            input_audio_format: Format specification for input audio
            output_audio_format: Format specification for output audio
            metadata: Custom metadata for this response
            conversation: Conversation context (or "none" for out-of-band)
            tools: Tools available for this response
            input_items: Custom input items for this response
            
        Returns:
            Optional[str]: Response ID if successful, None if failed
        """
        if not self.connected or not self.ws:
            logger.error("Cannot create response: not connected")
            return None
            
        try:
            # Validate modalities
            valid_modalities = ["text", "audio"]
            if modalities:
                for modality in modalities:
                    if modality not in valid_modalities:
                        logger.warning(f"Invalid modality: {modality}. Using default modalities.")
                        modalities = None
                        break
            
            response_data = {}
            
            # Only add non-None fields to prevent API errors
            if modalities:
                response_data["modalities"] = modalities
            if instructions:
                response_data["instructions"] = instructions
            if input_audio_format:
                response_data["input_audio_format"] = input_audio_format
            if output_audio_format:
                response_data["output_audio_format"] = output_audio_format
            if metadata:
                response_data["metadata"] = metadata
            if conversation:
                response_data["conversation"] = conversation
            if tools:
                response_data["tools"] = tools
            if input_items:
                response_data["input"] = input_items
                
            # Send the event
            event_id = await self._send_event({
                "type": "response.create",
                "response": response_data
            })
            
            if event_id:
                logger.info("Created response request with custom options")
                return event_id
            else:
                logger.error("Failed to create response request")
                return None
                
        except Exception as e:
            logger.error(f"Error creating response: {str(e)}")
            return None
    
    async def submit_tool_outputs(
        self,
        call_id: str,
        outputs: Dict[str, Any]
    ) -> bool:
        """
        Submit tool outputs for a run that requires action.
        
        Args:
            call_id: ID of the function call
            outputs: Output data from the function call
            
        Returns:
            bool: True if submission was successful
        """
        if not self.connected or not self.ws:
            logger.error("Cannot submit tool outputs: not connected")
            return False
            
        try:
            # Convert outputs to JSON string if needed
            output_str = json.dumps(outputs) if not isinstance(outputs, str) else outputs
            
            # Create function call output item
            event_id = await self._send_event({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_str
                }
            })
            
            if event_id:
                logger.info(f"Submitted outputs for function call {call_id}")
                return True
            else:
                logger.error("Failed to submit function call outputs")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting function call outputs: {str(e)}")
            return False
    
    async def submit_function_call_output(self, call_id: str, output: Dict[str, Any]) -> Optional[str]:
        """
        Submit the result of a function call back to the conversation.
        Alias for submit_tool_outputs for compatibility.
        
        Args:
            call_id: The ID of the function call
            output: The result of the function call
            
        Returns:
            Optional[str]: Event ID if successful, None otherwise
        """
        success = await self.submit_tool_outputs(call_id, output)
        return call_id if success else None
    
    async def interrupt(self) -> bool:
        """
        Send an interrupt signal to stop the assistant mid-response.
        
        Note: This uses a custom "interrupt" event type that is not part of the standard
        OpenAI Realtime API specification but is implemented in our client.
        
        Returns:
            bool: True if interrupt signal was sent successfully
        """
        if not self.connected or not self.ws:
            logger.error("Cannot send interrupt: not connected")
            return False
        
        try:
            # Prepare interrupt signal
            event_id = await self._send_event({
                "type": "interrupt"  # Custom event type
            })
            
            if event_id:
                logger.info("Sent interrupt signal to stop assistant response")
                return True
            else:
                logger.error("Failed to send interrupt signal")
                return False
            
        except Exception as e:
            logger.error(f"Error sending interrupt: {str(e)}")
            return False
    
    async def update_session(self, session_config: Dict[str, Any]) -> bool:
        """
        Update the session configuration.
        
        Args:
            session_config: New session configuration
            
        Returns:
            bool: True if session configuration was updated successfully
        """
        if not self.connected or not self.ws:
            logger.error("Cannot update session: not connected")
            return False
        
        try:
            # Validate session config params
            validated_config = {}
            
            # Copy valid parameters
            for key, value in session_config.items():
                if key == "voice":
                    # Validate voice parameter
                    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
                    if value in valid_voices:
                        validated_config[key] = value
                    else:
                        logger.warning(f"Invalid voice: {value}. Valid options are: {', '.join(valid_voices)}")
                elif key == "turn_detection":
                    # Update VAD state
                    if value is None:
                        self.vad_enabled = False
                    else:
                        self.vad_enabled = True
                    validated_config[key] = value
                else:
                    validated_config[key] = value
            
            # Send session update request
            event_id = await self._send_event({
                "type": "session.update",
                "session": validated_config
            })
            
            if event_id:
                logger.info("Session configuration updated successfully")
                return True
            else:
                logger.error("Failed to send session update request")
                return False
            
        except Exception as e:
            logger.error(f"Error updating session configuration: {str(e)}")
            return False
    
    async def create_conversation_item(self, item_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new conversation item.
        
        Args:
            item_data: Conversation item data
            
        Returns:
            Optional[str]: Event ID if successful, None otherwise
        """
        if not self.connected or not self.ws:
            logger.error("Cannot create conversation item: not connected")
            return None
            
        try:
            event_id = await self._send_event({
                "type": "conversation.item.create",
                "item": item_data
            })
            
            if event_id:
                logger.info(f"Created conversation item of type {item_data.get('type', 'unknown')}")
                return event_id
            else:
                logger.error("Failed to create conversation item")
                return None
                
        except Exception as e:
            logger.error(f"Error creating conversation item: {str(e)}")
            return None
    
    async def truncate_conversation(self, keep_last_n: int = 10) -> bool:
        """
        Truncate the conversation history, keeping only the most recent items.
        
        Args:
            keep_last_n: Number of most recent items to keep
            
        Returns:
            bool: True if successful
        """
        if not self.connected or not self.ws:
            logger.error("Cannot truncate conversation: not connected")
            return False
            
        try:
            # We need a conversation item ID to truncate before
            # Since we don't have a list of conversation items in this implementation,
            # we'll need to rely on the client providing a valid ID
            
            # Send the truncate event
            event_id = await self._send_event({
                "type": "conversation.item.truncate",
                "before_id": f"keep_{keep_last_n}"  # Placeholder - won't work without actual ID
            })
            
            if event_id:
                logger.info(f"Requested conversation truncation to keep {keep_last_n} most recent items")
                return True
            else:
                logger.error("Failed to send truncation request")
                return False
                
        except Exception as e:
            logger.error(f"Error truncating conversation: {str(e)}")
            return False
    
    async def wait_for_event(self, event_type: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """
        Wait for a specific event type to be received with a timeout.
        
        Args:
            event_type: The event type to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Optional[Dict[str, Any]]: The event data if received, None if timed out
        """
        event_received = asyncio.Event()
        event_data = [None]  # Use a list to store the event data
        
        async def event_handler(data):
            event_data[0] = data
            event_received.set()
        
        # Register the temporary handler
        if event_type not in event_handler.event_callbacks:
            event_handler.event_callbacks[event_type] = []
        event_handler.event_callbacks[event_type].append(event_handler)
        
        try:
            # Wait for the event or timeout
            await asyncio.wait_for(event_received.wait(), timeout=timeout)
            return event_data[0]
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for event: {event_type}")
            return None
        finally:
            # Clean up the temporary handler
            if event_type in event_handler.event_callbacks and event_handler in event_handler.event_callbacks[event_type]:
                event_handler.event_callbacks[event_type].remove(event_handler)
    
    async def _message_handler(self) -> None:
        """
        Handle incoming messages from the WebSocket connection.
        """
        if not self.ws:
            logger.error("Cannot receive messages: WebSocket is not connected")
            return
            
        try:
            async for message in self.ws:
                await self._process_message(message)
        except websockets.exceptions.ConnectionClosedOK as e:
            # This is a normal closure code (1000)
            logger.info(f"WebSocket connection closed normally: {e}")
            self.connected = False
        except websockets.exceptions.ConnectionClosedError as e:
            # This is an abnormal closure
            logger.warning(f"WebSocket connection closed unexpectedly: {e}")
            self.connected = False
            
            # Attempt to reconnect if not already doing so
            if not self.reconnecting:
                logger.info("Attempting to reconnect after unexpected closure")
                self.task_manager.create_task(self.reconnect(), "reconnect_after_closure")
        except asyncio.CancelledError:
            # Task was cancelled, just exit
            pass
        except Exception as e:
            logger.error(f"Error in message receiver: {e}")
            self.connected = False
    
    async def _heartbeat_loop(self) -> None:
        """
        Monitor the connection health with periodic checks.
        
        This uses the WebSocket ping/pong mechanism and also sends custom heartbeats.
        """
        try:
            while self.connected and self.ws:
                try:
                    # Send a heartbeat event every 25 seconds
                    # This is in addition to the WebSocket ping/pong which is every 30 seconds
                    await asyncio.sleep(25)
                    
                    if self.connected and self.ws:
                        # Custom heartbeat event
                        await self._send_event({
                            "type": "heartbeat"  # Custom event type
                        })
                        logger.debug("Sent heartbeat to keep connection alive")
                except Exception as e:
                    logger.warning(f"Error in heartbeat: {e}")
        except asyncio.CancelledError:
            # Task was cancelled, just exit
            pass
        except Exception as e:
            logger.error(f"Error in heartbeat loop: {e}")
            
            # If we lost connection, try to reconnect
            if not self.reconnecting and not self.connected:
                logger.info("Attempting to reconnect due to heartbeat failure")
                self.task_manager.create_task(self.reconnect(), "reconnect_from_heartbeat")
    
    async def _send_event(self, event: Dict[str, Any]) -> Optional[str]:
        """
        Send an event to the OpenAI Realtime API.
        
        Args:
            event: Event data to send
            
        Returns:
            Optional[str]: Event ID if sent successfully, None otherwise
        """
        if not self.connected or not self.ws:
            # Try to reconnect first
            if not self.reconnecting:
                logger.info("Connection lost, attempting to reconnect before sending event")
                reconnected = await self.reconnect()
                if not reconnected:
                    logger.error("Cannot send event: not connected and reconnection failed")
                    return None
            else:
                logger.error("Cannot send event: reconnection in progress")
                return None
        
        try:
            # Generate event ID if not already present
            if "id" not in event:
                event["id"] = f"evt_{uuid.uuid4().hex}"
            
            # Add sequence number
            if "sequence" not in event:
                event["sequence"] = self.event_sequence
                self.event_sequence += 1
            
            # Track event for error correlation
            self.pending_events[event["id"]] = {
                "type": event.get("type", "unknown"),
                "timestamp": time.time(),
                "data": {k: v for k, v in event.items() if k not in ["id", "sequence", "type"]}
            }
            
            # Send the event
            await self.ws.send(json.dumps(event))
            
            # Log the event (skip detailed logging for common events to reduce noise)
            event_type = event.get("type", "unknown")
            if event_type == "input_audio_buffer.append":
                logger.debug(f"Sent event: {event_type} (id: {event['id']})")
            elif event_type == "heartbeat":
                logger.debug(f"Sent heartbeat event (id: {event['id']})")
            else:
                logger.debug(f"Sent event: {event_type} (id: {event['id']})")
            
            return event["id"]
            
        except Exception as e:
            logger.error(f"Error sending event: {str(e)}")
            
            # Clean up the pending event
            if "id" in event and event["id"] in self.pending_events:
                del self.pending_events[event["id"]]
            
            # Try to reconnect if this was a connection issue
            if "connection" in str(e).lower() and not self.reconnecting:
                logger.info("Connection error detected, attempting to reconnect")
                self.task_manager.create_task(self.reconnect(), "reconnect_after_send_error")
            
            return None
    
    async def _process_message(self, message: Union[str, bytes]) -> None:
        """
        Process a message received from the WebSocket connection.
        
        Args:
            message: Raw message from WebSocket
        """
        try:
            # Parse JSON message
            if isinstance(message, bytes):
                data = json.loads(message.decode('utf-8'))
            else:
                data = json.loads(message)
            
            # Extract message details
            message_id = data.get('id')
            message_type = data.get('type')
            
            # Handle error events
            if message_type == "error":
                await self._handle_error(data)
                return
            
            # Process using the centralized event handler
            await event_handler.handle_event(message_type, data)
            
            # Remove from pending events if it was a response to one
            if message_id and message_id in self.pending_events:
                del self.pending_events[message_id]
            
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON message: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    async def _handle_error(self, error_data: Dict[str, Any]) -> None:
        """
        Handle error events from the API with detailed error recovery.
        
        Args:
            error_data: Error event data
        """
        error_code = error_data.get("code", "unknown_code")
        error_type = error_data.get("type", "unknown_error")
        error_message = error_data.get("message", "Unknown error")
        event_id = error_data.get("event_id")
        
        # Format a detailed error message
        error_msg = f"API Error: {error_type} ({error_code}): {error_message}"
        
        # Log with appropriate severity
        if error_code in ["rate_limit_exceeded", "token_limit_exceeded"]:
            logger.warning(error_msg)
        else:
            logger.error(error_msg)
        
        # Find the original event if available
        original_event = None
        if event_id and event_id in self.pending_events:
            original_event = self.pending_events[event_id]
            event_type = original_event.get('type', 'unknown')
            logger.error(f"Error related to {event_type} event sent at {original_event['timestamp']}")
        
        # Create and emit an error event
        error_event = ErrorEvent(
            type=EventType.ERROR,
            data=error_data,
            error=error_data
        )
        event_bus.emit(error_event)
        
        # Attempt specific error recovery strategies
        await self._handle_api_error_recovery(error_data, original_event)
        
        # Clean up pending event
        if event_id and event_id in self.pending_events:
            del self.pending_events[event_id]
    
    async def _handle_api_error_recovery(
        self, 
        error_data: Dict[str, Any],
        original_event: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Attempt to recover from specific API errors with appropriate strategies.
        
        Args:
            error_data: Error event data from the API
            original_event: The original event that triggered the error, if available
            
        Returns:
            bool: True if recovery was attempted, False otherwise
        """
        error_code = error_data.get("code", "unknown_code")
        error_type = error_data.get("type", "unknown_error")
        
        # Handle rate limit errors with exponential backoff
        if error_code == "rate_limit_exceeded":
            # Get retry-after if available, otherwise use default backoff
            retry_after = error_data.get("retry_after", 1.0)
            logger.warning(f"Rate limit exceeded. Waiting {retry_after}s before retrying...")
            await asyncio.sleep(float(retry_after))
            return True
            
        # Handle token limit errors by clearing conversation context
        elif error_code == "token_limit_exceeded":
            logger.warning("Token limit exceeded. Attempting to truncate the conversation...")
            
            # Truncate the conversation to reduce token usage
            try:
                await self.truncate_conversation(keep_last_n=5)  # Keep only the most recent messages
                logger.info("Truncated conversation due to token limit")
                return True
            except Exception as e:
                logger.error(f"Failed to truncate conversation: {e}")
                return False
            
        # Handle invalid request errors
        elif error_type == "invalid_request_error":
            # If this related to a specific event, try to recover
            if original_event:
                event_type = original_event.get('type')
                
                # For response.create events, retry with simpler parameters
                if event_type == "response.create":
                    logger.warning("Invalid request when creating response. Will retry with simpler parameters.")
                    
                    # If the original event data is available, try to retry with minimal settings
                    if "data" in original_event:
                        try:
                            # Create a simpler version of the request
                            simple_request = {
                                "type": "response.create",
                                "response": {
                                    "modalities": ["text"]  # Text-only is safer
                                }
                            }
                            
                            # Add instructions if they existed before
                            if "instructions" in original_event["data"].get("response", {}):
                                simple_request["response"]["instructions"] = original_event["data"]["response"]["instructions"]
                            
                            logger.info("Retrying response creation with simplified parameters")
                            await self._send_event(simple_request)
                            return True
                        except Exception as retry_error:
                            logger.error(f"Error retrying response creation: {retry_error}")
                
            return False
        
        # Handle connection-related errors
        elif "connection" in error_message.lower() or "websocket" in error_message.lower():
            if not self.reconnecting:
                logger.warning("Connection error detected, attempting to reconnect...")
                self.task_manager.create_task(self.reconnect(), "reconnect_after_error")
                return True
        
        # Handle unknown errors
        elif error_code == "unknown_code" or error_code == "unknown_error":
            logger.warning(f"Received unknown error from API: {json.dumps(error_data)}")
            
            # If this is during initialization, attempt reconnection
            if self.first_connect and not self.reconnecting:
                logger.info("Error during initialization, attempting to reconnect...")
                self.task_manager.create_task(self.reconnect(), "reconnect_after_unknown_error")
                return True
        
        # No specific recovery strategy for this error
        return False
    
    async def _wait_for_session_created(self) -> bool:
        """
        Wait for the session.created event to be received.
        
        Returns:
            bool: True if session.created event was received
        """
        try:
            # Wait for session.created event
            session_event = await self.wait_for_event("session.created", timeout=10.0)
            
            if session_event:
                # Extract session ID
                session = session_event.get("session", {})
                self.session_id = session.get("id")
                logger.info(f"Received session.created event with ID: {self.session_id}")
                return True
            else:
                logger.error("Timed out waiting for session.created event")
                return False
                
        except Exception as e:
            logger.error(f"Error waiting for session creation: {e}")
            return False
    
    @property
    def is_connected(self) -> bool:
        """Check if client is currently connected and active."""
        return self.connected and self.ws is not None
    
    def is_speech_active(self) -> bool:
        """Check if speech is currently active (based on VAD)."""
        return self.speech_active
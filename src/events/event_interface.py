"""
Event interface for the OpenAI Realtime API.

This module defines the base event classes and handlers for working with
the OpenAI Realtime API event system, ensuring consistent event processing
throughout the application.
"""

import abc
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

from src.config.logging_config import get_logger
from src.utils.error_handling import ApiError, ErrorSeverity

logger = get_logger(__name__)


class EventType(Enum):
    """Types of events that can be emitted by the event bus."""
    
    # System events
    INITIALIZED = "initialized"
    SHUTDOWN = "shutdown"
    ERROR = "error"
    
    # Connection events
    CONNECTION_ESTABLISHED = "connection_established" 
    CONNECTION_CLOSED = "connection_closed"
    CONNECTION_ERROR = "connection_error"
    
    # Session events
    SESSION_CREATED = "session_created"
    SESSION_UPDATED = "session_updated"
    
    # Conversation events
    CONVERSATION_CREATED = "conversation_created"
    CONVERSATION_STATE_CHANGED = "conversation_state_changed"
    
    # Audio buffer events
    AUDIO_BUFFER_COMMITTED = "audio_buffer_committed" 
    AUDIO_BUFFER_CLEARED = "audio_buffer_cleared"
    
    # User speech events
    USER_SPEECH_STARTED = "user_speech_started"
    USER_SPEECH_ONGOING = "user_speech_ongoing" 
    USER_SPEECH_FINISHED = "user_speech_finished"
    USER_SPEECH_CANCELLED = "user_speech_cancelled"
    
    # Transcription events
    USER_TRANSCRIPTION_COMPLETED = "user_transcription_completed"
    
    # Response events
    RESPONSE_CREATED = "response_created"
    TEXT_CREATED = "text_created"
    MESSAGE_CREATED = "message_created"
    MESSAGE_COMPLETED = "message_completed"
    
    # Audio events
    AUDIO_SPEECH_CREATED = "audio_speech_created"
    AUDIO_PLAYBACK_STARTED = "audio_playback_started"
    AUDIO_PLAYBACK_STOPPED = "audio_playback_stopped"
    AUDIO_PLAYBACK_COMPLETED = "audio_playback_completed"
    
    # Function call events
    FUNCTION_CALL_RECEIVED = "function_call_received"
    FUNCTION_CALL_EXECUTED = "function_call_executed"
    FUNCTION_CALL_FAILED = "function_call_failed"
    
    # Catch-all for unknown events
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, event_type_str: str) -> 'EventType':
        """Convert a string to an EventType enum value."""
        try:
            # Try direct lookup by value
            return next(e for e in cls if e.value == event_type_str)
        except StopIteration:
            # Try mapping from API event types if not a direct match
            mapping = {
                # Session events
                "session.created": cls.SESSION_CREATED,
                "session.updated": cls.SESSION_UPDATED,
                
                # Audio buffer events
                "input_audio_buffer.committed": cls.AUDIO_BUFFER_COMMITTED,
                "input_audio_buffer.cleared": cls.AUDIO_BUFFER_CLEARED,
                "input_audio_buffer.speech_started": cls.USER_SPEECH_STARTED,
                "input_audio_buffer.speech_stopped": cls.USER_SPEECH_FINISHED,
                
                # Response events
                "response.created": cls.RESPONSE_CREATED,
                "response.done": cls.MESSAGE_COMPLETED,
                "response.text.delta": cls.TEXT_CREATED,
                
                # Conversation events
                "conversation.item.created": cls.MESSAGE_CREATED,
                
                # Default for unknown events
                "unknown": cls.UNKNOWN
            }
            
            return mapping.get(event_type_str, cls.UNKNOWN)


@dataclass
class Event:
    """Base class for all events in the system."""
    
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary."""
        result = asdict(self)
        # Convert EventType enum to string value
        result['type'] = self.type.value
        return result
    
    def to_json(self) -> str:
        """Convert the event to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create an event from a dictionary."""
        # Convert string to EventType enum if needed
        if isinstance(data.get('type'), str):
            event_type = EventType.from_string(data['type'])
        else:
            event_type = data.get('type', EventType.UNKNOWN)
        
        return cls(
            type=event_type,
            data=data.get('data', {}),
            id=data.get('id'),
            created_at=data.get('created_at', time.time())
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Create an event from a JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON event: {e}")
            return ErrorEvent(
                type=EventType.ERROR,
                data={"error": {"message": f"Invalid JSON: {str(e)}", "type": "json_decode_error"}}
            )


@dataclass
class ErrorEvent(Event):
    """Event for error conditions."""
    
    error: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure error data is also in the main data dict."""
        if self.error and "error" not in self.data:
            self.data["error"] = self.error


@dataclass
class AudioSpeechEvent(Event):
    """Event for audio speech data."""
    
    chunk: bytes = field(default_factory=bytes)
    
    def __post_init__(self):
        """Don't include audio chunk in the main data dictionary to avoid serialization issues."""
        # We explicitly don't add the chunk to self.data to prevent 
        # attempting to serialize binary data to JSON


@dataclass
class UserSpeechEvent(Event):
    """Event for user speech input."""
    
    audio_data: Optional[bytes] = None
    duration: float = 0.0
    is_final: bool = False
    
    def __post_init__(self):
        """Update data with speech information."""
        self.data.update({
            "duration": self.duration,
            "is_final": self.is_final
        })
        # Don't include audio_data in data dict to avoid serialization issues


@dataclass
class TranscriptionEvent(Event):
    """Event for speech transcription."""
    
    text: str = ""
    is_final: bool = True
    confidence: float = 1.0
    source: str = "whisper"
    
    def __post_init__(self):
        """Update data with transcription information."""
        self.data.update({
            "text": self.text,
            "is_final": self.is_final,
            "confidence": self.confidence,
            "source": self.source
        })


# Type for event handlers
EventHandler = Union[
    Callable[[Event], None],  # Synchronous handler
    Callable[[Event], Any]    # Asynchronous handler
]


class EventBus:
    """
    Central event bus for the application.
    
    This class manages event publication and subscription using the
    observer pattern. It supports both synchronous and asynchronous
    event handlers.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        # Dict of event type -> list of handlers
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        # List of handlers for all events
        self._global_handlers: List[EventHandler] = []
        # AsyncIO loop for async handlers
        self._loop = None
    
    def on(self, event_type: Union[EventType, str], handler: EventHandler) -> None:
        """
        Register a handler for a specific event type.
        
        Args:
            event_type: The type of event to handle
            handler: The handler function to call when the event occurs
        """
        # Convert string to EventType if needed
        if isinstance(event_type, str):
            event_type = EventType.from_string(event_type)
        
        # Initialize handler list if needed
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        # Add handler to list if not already present
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
            logger.debug(f"Registered handler for event type: {event_type.name}")
    
    def on_any(self, handler: EventHandler) -> None:
        """
        Register a handler for all event types.
        
        Args:
            handler: The handler function to call for any event
        """
        if handler not in self._global_handlers:
            self._global_handlers.append(handler)
            logger.debug("Registered global event handler")
    
    def off(self, event_type: Union[EventType, str], handler: Optional[EventHandler] = None) -> None:
        """
        Remove a handler for a specific event type.
        
        Args:
            event_type: The type of event
            handler: The handler to remove. If None, removes all handlers for the event type.
        """
        # Convert string to EventType if needed
        if isinstance(event_type, str):
            event_type = EventType.from_string(event_type)
        
        # Remove specific handler or all handlers for event type
        if event_type in self._handlers:
            if handler is None:
                self._handlers[event_type] = []
                logger.debug(f"Removed all handlers for event type: {event_type.name}")
            elif handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                logger.debug(f"Removed handler for event type: {event_type.name}")
    
    def off_any(self, handler: Optional[EventHandler] = None) -> None:
        """
        Remove a global handler.
        
        Args:
            handler: The handler to remove. If None, removes all global handlers.
        """
        if handler is None:
            self._global_handlers = []
            logger.debug("Removed all global handlers")
        elif handler in self._global_handlers:
            self._global_handlers.remove(handler)
            logger.debug("Removed global handler")
    
    def emit(self, event_type: Union[EventType, str, Event], data: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit an event to all registered handlers.
        
        Args:
            event_type: The event type or Event object
            data: The event data (if event_type is not an Event)
        """
        # Create Event object if needed
        if isinstance(event_type, Event):
            event = event_type
        else:
            # Convert string to EventType if needed
            if isinstance(event_type, str):
                event_type = EventType.from_string(event_type)
            
            # Create the event
            event = Event(type=event_type, data=data or {})
        
        # Get event loop for async handlers
        loop = self._get_loop()
        
        # Call type-specific handlers
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            self._call_handler(handler, event, loop)
        
        # Call global handlers
        for handler in self._global_handlers:
            self._call_handler(handler, event, loop)
    
    def _call_handler(self, handler: EventHandler, event: Event, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        """
        Call an event handler, handling both sync and async handlers.
        
        Args:
            handler: The handler to call
            event: The event to pass to the handler
            loop: The asyncio event loop for async handlers
        """
        try:
            # Check if handler is a coroutine function
            if asyncio.iscoroutinefunction(handler):
                # If we have a loop, schedule the handler
                if loop:
                    loop.create_task(self._call_async_handler(handler, event))
                else:
                    # Log warning if we can't run async handler
                    logger.warning(f"Cannot run async handler {handler.__name__}: no event loop")
            else:
                # Synchronous handler
                handler(event)
        except Exception as e:
            logger.error(f"Error in event handler for {event.type.name}: {e}", exc_info=True)
    
    async def _call_async_handler(self, handler: EventHandler, event: Event) -> None:
        """
        Call an async event handler.
        
        Args:
            handler: The async handler to call
            event: The event to pass to the handler
        """
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Error in async event handler for {event.type.name}: {e}", exc_info=True)
    
    def _get_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """
        Get the current event loop.
        
        Returns:
            The current event loop or None if no loop is running
        """
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in this thread
                logger.warning("No event loop found in current thread")
                return None
        
        return self._loop


# Global event bus instance
event_bus = EventBus()
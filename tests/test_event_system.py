"""
Tests for the event system.

This module tests the event bus, event handlers, and event propagation,
ensuring events are properly registered, dispatched, and handled.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.events.event_interface import (
    Event, 
    EventType, 
    event_bus, 
    AudioSpeechEvent,
    TranscriptionEvent,
    ErrorEvent,
    EventHandler
)


@pytest.fixture
def reset_event_bus():
    """Reset the event bus after each test."""
    # Store original handlers
    original_handlers = event_bus._handlers.copy()
    original_wildcard = event_bus._wildcard_handlers.copy()
    
    yield
    
    # Restore original handlers
    event_bus._handlers = original_handlers
    event_bus._wildcard_handlers = original_wildcard


def test_event_registration(reset_event_bus):
    """Test registering and unregistering event handlers."""
    # Define handlers
    handler1 = MagicMock()
    handler2 = MagicMock()
    
    # Register handlers
    event_bus.on(EventType.USER_SPEECH_STARTED, handler1)
    event_bus.on(EventType.USER_SPEECH_STARTED, handler2)
    
    # Check that handlers were registered
    assert handler1 in event_bus._handlers[EventType.USER_SPEECH_STARTED]
    assert handler2 in event_bus._handlers[EventType.USER_SPEECH_STARTED]
    assert len(event_bus._handlers[EventType.USER_SPEECH_STARTED]) == 2
    
    # Unregister one handler
    event_bus.off(EventType.USER_SPEECH_STARTED, handler1)
    
    # Check that handler was unregistered
    assert handler1 not in event_bus._handlers[EventType.USER_SPEECH_STARTED]
    assert handler2 in event_bus._handlers[EventType.USER_SPEECH_STARTED]
    assert len(event_bus._handlers[EventType.USER_SPEECH_STARTED]) == 1
    
    # Unregister all handlers for event type
    event_bus.off(EventType.USER_SPEECH_STARTED)
    
    # Check that all handlers were unregistered
    assert EventType.USER_SPEECH_STARTED in event_bus._handlers
    assert len(event_bus._handlers[EventType.USER_SPEECH_STARTED]) == 0


def test_wildcard_registration(reset_event_bus):
    """Test registering and unregistering wildcard event handlers."""
    # Define handler
    handler = MagicMock()
    
    # Register wildcard handler
    event_bus.on_any(handler)
    
    # Check that handler was registered
    assert handler in event_bus._wildcard_handlers
    
    # Unregister handler
    event_bus.off_any(handler)
    
    # Check that handler was unregistered
    assert handler not in event_bus._wildcard_handlers
    
    # Register again and unregister all
    event_bus.on_any(handler)
    event_bus.off_any()
    
    # Check that all handlers were unregistered
    assert len(event_bus._wildcard_handlers) == 0


def test_event_propagation(reset_event_bus):
    """Test events properly propagate through the system."""
    # Define handlers
    handler1 = MagicMock()
    handler2 = MagicMock()
    wildcard_handler = MagicMock()
    
    # Register handlers
    event_bus.on(EventType.USER_SPEECH_STARTED, handler1)
    event_bus.on(EventType.USER_SPEECH_FINISHED, handler2)
    event_bus.on_any(wildcard_handler)
    
    # Create and emit event
    event = Event(type=EventType.USER_SPEECH_STARTED, data={"test": "data"})
    event_bus.emit(event)
    
    # Check that specific handler was called
    handler1.assert_called_once_with(event)
    handler2.assert_not_called()
    
    # Check that wildcard handler was called
    wildcard_handler.assert_called_once_with(event)
    
    # Create and emit another event
    event2 = Event(type=EventType.USER_SPEECH_FINISHED, data={"test": "data2"})
    event_bus.emit(event2)
    
    # Check that specific handler was called
    handler1.assert_called_once()  # Still once
    handler2.assert_called_once_with(event2)
    
    # Check that wildcard handler was called again
    assert wildcard_handler.call_count == 2
    wildcard_handler.assert_has_calls([call(event), call(event2)])


def test_event_type_validation():
    """Test validation of event types against known types."""
    # Test valid event type
    event_type = EventType.USER_SPEECH_STARTED
    assert event_type in EventType
    
    # Test conversion from string
    event_type_from_string = EventType.from_string("user.speech.started")
    assert event_type_from_string == EventType.USER_SPEECH_STARTED
    
    # Test invalid event type
    unknown_type = EventType.from_string("unknown.event.type")
    assert unknown_type == EventType.UNKNOWN


def test_error_handling_in_handlers(reset_event_bus):
    """Test that errors in event handlers are properly caught."""
    # Define handler that raises an exception
    def error_handler(event):
        raise ValueError("Test error")
    
    # Define normal handler
    normal_handler = MagicMock()
    
    # Register handlers
    event_bus.on(EventType.USER_SPEECH_STARTED, error_handler)
    event_bus.on(EventType.USER_SPEECH_STARTED, normal_handler)
    
    # Create and emit event
    event = Event(type=EventType.USER_SPEECH_STARTED, data={"test": "data"})
    
    # Should not raise exception
    event_bus.emit(event)
    
    # Second handler should still be called
    normal_handler.assert_called_once_with(event)


def test_event_types():
    """Test event type enumeration."""
    # Check that event types exist
    assert hasattr(EventType, "USER_SPEECH_STARTED")
    assert hasattr(EventType, "USER_SPEECH_FINISHED")
    assert hasattr(EventType, "AUDIO_SPEECH_CREATED")
    assert hasattr(EventType, "ERROR")
    
    # Check values
    assert EventType.USER_SPEECH_STARTED.value == "user.speech.started"
    assert EventType.USER_SPEECH_FINISHED.value == "user.speech.finished"
    assert EventType.ERROR.value == "error"


def test_event_subclasses():
    """Test event subclass creation and properties."""
    # Create AudioSpeechEvent
    audio_event = AudioSpeechEvent(
        type=EventType.AUDIO_SPEECH_CREATED,
        data={"test": "data"},
        id="test_id",
        chunk=b"audio data"
    )
    
    # Check properties
    assert audio_event.type == EventType.AUDIO_SPEECH_CREATED
    assert audio_event.data == {"test": "data"}
    assert audio_event.id == "test_id"
    assert audio_event.chunk == b"audio data"
    
    # Create TranscriptionEvent
    transcription_event = TranscriptionEvent(
        type=EventType.USER_TRANSCRIPTION_COMPLETED,
        data={"text": "test transcription"},
        text="test transcription",
        is_final=True,
        confidence=0.95,
        source="whisper"
    )
    
    # Check properties
    assert transcription_event.type == EventType.USER_TRANSCRIPTION_COMPLETED
    assert transcription_event.text == "test transcription"
    assert transcription_event.is_final is True
    assert transcription_event.confidence == 0.95
    assert transcription_event.source == "whisper"
    
    # Create ErrorEvent
    error_event = ErrorEvent(
        type=EventType.ERROR,
        data={"error": {"message": "Test error"}},
        error={"message": "Test error"}
    )
    
    # Check properties
    assert error_event.type == EventType.ERROR
    assert error_event.error == {"message": "Test error"}


def test_event_serialization():
    """Test event serialization to and from dictionaries and JSON."""
    # Create event
    event = Event(
        type=EventType.USER_SPEECH_STARTED,
        data={"test": "data"},
        id="test_id"
    )
    
    # Convert to dictionary
    event_dict = event.to_dict()
    
    # Check dictionary
    assert event_dict["type"] == "user.speech.started"
    assert event_dict["data"] == {"test": "data"}
    assert event_dict["id"] == "test_id"
    
    # Convert to JSON
    event_json = event.to_json()
    
    # Check JSON
    assert "user.speech.started" in event_json
    assert "test_id" in event_json
    assert "data" in event_json
    
    # Convert back from dictionary
    event2 = Event.from_dict(event_dict)
    
    # Check that event was reconstructed
    assert event2.type == event.type
    assert event2.data == event.data
    assert event2.id == event.id
    
    # Convert back from JSON
    event3 = Event.from_json(event_json)
    
    # Check that event was reconstructed
    assert event3.type == event.type
    assert event3.data == event.data
    assert event3.id == event.id


class TestEventHandler(EventHandler):
    """Test implementation of event handler base class."""
    
    def __init__(self):
        self.events = []
        super().__init__()
    
    def register_handlers(self):
        event_bus.on(EventType.USER_SPEECH_STARTED, self.handle_speech_start)
        event_bus.on(EventType.USER_SPEECH_FINISHED, self.handle_speech_finish)
    
    def handle_speech_start(self, event):
        self.events.append(("start", event))
    
    def handle_speech_finish(self, event):
        self.events.append(("finish", event))


def test_event_handler_base_class(reset_event_bus):
    """Test the event handler base class."""
    # Create handler
    handler = TestEventHandler()
    
    # Check that handlers were registered
    assert len(event_bus._handlers[EventType.USER_SPEECH_STARTED]) == 1
    assert len(event_bus._handlers[EventType.USER_SPEECH_FINISHED]) == 1
    
    # Create and emit events
    start_event = Event(type=EventType.USER_SPEECH_STARTED, data={"test": "start"})
    finish_event = Event(type=EventType.USER_SPEECH_FINISHED, data={"test": "finish"})
    
    event_bus.emit(start_event)
    event_bus.emit(finish_event)
    
    # Check that events were handled
    assert len(handler.events) == 2
    assert handler.events[0] == ("start", start_event)
    assert handler.events[1] == ("finish", finish_event)
    
    # Unregister handlers
    handler.unregister_handlers()
    
    # Clear events
    handler.events = []
    
    # Emit events again
    event_bus.emit(start_event)
    event_bus.emit(finish_event)
    
    # Check that events were not handled
    assert len(handler.events) == 0
"""
Tests for the conversation manager.

This module tests the conversation flow, state management,
and event handling in the conversation manager.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock, call

from src.domain.conversation.manager import ConversationManager, ConversationState
from src.domain.conversation.state import ConversationHistory, MessageRole
from src.events.event_interface import Event, EventType, event_bus
from src.utils.error_handling import AppError


@pytest.fixture
def mock_api_client():
    """Create a mock API client."""
    client = MagicMock()
    client.connect = AsyncMock(return_value=True)
    client.disconnect = AsyncMock()
    client.request_response = AsyncMock(return_value=True)
    client.send_audio = AsyncMock(return_value=True)
    client.update_session = AsyncMock(return_value=True)
    client.interrupt = AsyncMock(return_value=True)
    client.submit_tool_outputs = AsyncMock(return_value=True)
    client.connected = True
    client.is_connected = True
    return client


@pytest.fixture
def mock_audio_service():
    """Create a mock audio service."""
    service = MagicMock()
    service.start_recording = AsyncMock(return_value=True)
    service.stop_recording = AsyncMock()
    service.play_audio = AsyncMock()
    service.stop_playback = AsyncMock()
    service.pause_playback = AsyncMock()
    service.resume_playback = AsyncMock()
    service.cleanup = AsyncMock()
    return service


@pytest.fixture
def conversation_manager(mock_api_client, mock_audio_service):
    """Create a ConversationManager for testing."""
    with patch("src.domain.conversation.manager.RealtimeClient", return_value=mock_api_client):
        with patch("src.domain.conversation.manager.AudioService", return_value=mock_audio_service):
            manager = ConversationManager(
                assistant_id="test-assistant",
                instructions="Test instructions"
            )
            yield manager
            # Clean up
            asyncio.run(manager.stop())


@pytest.mark.asyncio
async def test_initialization(conversation_manager, mock_api_client, mock_audio_service):
    """Test conversation manager initialization."""
    # Should have initialized API client and audio service
    assert conversation_manager.api_client is mock_api_client
    
    # Should start in IDLE state
    assert conversation_manager.state == ConversationState.IDLE


@pytest.mark.asyncio
async def test_start(conversation_manager, mock_api_client):
    """Test starting the conversation."""
    # Start conversation
    result = await conversation_manager.start()
    
    # Should have connected to API
    assert result is True
    mock_api_client.connect.assert_called_once()
    
    # Should be in READY state
    assert conversation_manager.state == ConversationState.READY


@pytest.mark.asyncio
async def test_stop(conversation_manager, mock_api_client, mock_audio_service):
    """Test stopping the conversation."""
    # Start first
    await conversation_manager.start()
    
    # Stop conversation
    await conversation_manager.stop()
    
    # Should have disconnected and cleaned up
    mock_api_client.disconnect.assert_called_once()
    mock_audio_service.stop_recording.assert_called_once()
    mock_audio_service.stop_playback.assert_called_once()
    
    # Should be in DISCONNECTED state
    assert conversation_manager.state == ConversationState.DISCONNECTED


@pytest.mark.asyncio
async def test_state_transitions(conversation_manager):
    """Test conversation state transitions."""
    # Initial state is IDLE
    assert conversation_manager.state == ConversationState.IDLE
    
    # Start connection -> CONNECTING -> READY
    with patch.object(conversation_manager, "api_client") as mock_client:
        mock_client.connect = AsyncMock(return_value=True)
        await conversation_manager.start()
        assert conversation_manager.state == ConversationState.READY
    
    # User speaking -> USER_SPEAKING
    await conversation_manager._handle_user_speech_started(Event(type=EventType.USER_SPEECH_STARTED))
    assert conversation_manager.state == ConversationState.USER_SPEAKING
    
    # User finished speaking -> THINKING (if auto_respond enabled)
    conversation_manager.auto_respond = True
    await conversation_manager._handle_user_speech_finished(
        Event(type=EventType.USER_SPEECH_FINISHED, data={"audio_data": b"audio"})
    )
    assert conversation_manager.state == ConversationState.THINKING
    
    # Assistant speaking -> ASSISTANT_SPEAKING
    event_data = {"chunk": b"audio chunk"}
    await conversation_manager._handle_assistant_speech(Event(type=EventType.AUDIO_SPEECH_CREATED, data=event_data))
    assert conversation_manager.state == ConversationState.ASSISTANT_SPEAKING
    
    # Message completed -> READY
    await conversation_manager._handle_message_completed(Event(type=EventType.MESSAGE_COMPLETED))
    assert conversation_manager.state == ConversationState.READY
    
    # Error -> READY
    error_data = {"error": {"type": "test_error", "message": "Test error"}}
    await conversation_manager._handle_error(Event(type=EventType.ERROR, data=error_data))
    assert conversation_manager.state == ConversationState.READY
    
    # Stop -> DISCONNECTED
    await conversation_manager.stop()
    assert conversation_manager.state == ConversationState.DISCONNECTED


@pytest.mark.asyncio
async def test_request_response(conversation_manager, mock_api_client):
    """Test requesting a response from the model."""
    # Request response
    instructions = "Custom instructions"
    result = await conversation_manager.request_response(instructions)
    
    # Should have called API
    assert result is True
    mock_api_client.request_response.assert_called_once_with(instructions=instructions)


@pytest.mark.asyncio
async def test_user_interruption(conversation_manager, mock_api_client):
    """Test user interruption of assistant speech."""
    # Set assistant speaking state
    conversation_manager.state = ConversationState.ASSISTANT_SPEAKING
    conversation_manager.assistant_speaking = True
    
    # Simulate user speech start
    event = Event(type=EventType.USER_SPEECH_STARTED)
    await conversation_manager._handle_user_speech_started(event)
    
    # Should have tried to interrupt
    mock_api_client.interrupt.assert_called_once()


@pytest.mark.asyncio
async def test_function_calling(conversation_manager, mock_api_client):
    """Test function calling flow."""
    # Create a function call event
    function_name = "test_function"
    arguments = {"arg1": "value1"}
    call_id = "call_123"
    response_id = "resp_456"
    
    # Register a function handler
    called = False
    result_data = {"result": "success"}
    
    async def function_handler(name, args, c_id, r_id):
        nonlocal called
        called = True
        assert name == function_name
        assert args == arguments
        assert c_id == call_id
        assert r_id == response_id
        await conversation_manager.submit_tool_outputs([{
            "tool_call_id": c_id,
            "output": json.dumps(result_data)
        }])
    
    # Register handler
    conversation_manager.register_function_handler(function_handler)
    
    # Create event to simulate function call
    event = Event(
        type=EventType.FUNCTION_CALL_RECEIVED,
        data={
            "name": function_name,
            "arguments": arguments,
            "call_id": call_id,
            "response_id": response_id
        }
    )
    
    # Process event
    await conversation_manager._handle_function_call(event)
    
    # Handler should have been called
    assert called is True
    
    # Should have submitted tool outputs
    mock_api_client.submit_tool_outputs.assert_called_once()
    call_args = mock_api_client.submit_tool_outputs.call_args[0][0]
    assert call_args[0]["tool_call_id"] == call_id
    assert json.loads(call_args[0]["output"]) == result_data


@pytest.mark.asyncio
async def test_conversation_history_management(conversation_manager):
    """Test conversation history management."""
    # Start with empty history
    assert len(conversation_manager.messages) == 0
    
    # Add some messages
    await conversation_manager._add_user_message("Hello")
    await conversation_manager._add_assistant_message("Hi there")
    
    # Check messages were added
    assert len(conversation_manager.messages) == 2
    assert conversation_manager.messages[0]["role"] == "user"
    assert conversation_manager.messages[0]["content"] == "Hello"
    assert conversation_manager.messages[1]["role"] == "assistant"
    assert conversation_manager.messages[1]["content"] == "Hi there"
    
    # Process a transcription
    event = Event(
        type=EventType.USER_TRANSCRIPTION_COMPLETED,
        data={
            "text": "This is a transcription",
            "is_final": True
        }
    )
    await conversation_manager._handle_transcription_completed(event)
    
    # Should have added transcription to history
    assert len(conversation_manager.messages) == 3
    assert conversation_manager.messages[2]["role"] == "user"
    assert conversation_manager.messages[2]["content"] == "This is a transcription"
    assert conversation_manager.messages[2]["is_transcription"] is True


@pytest.mark.asyncio
async def test_error_propagation(conversation_manager):
    """Test that errors are properly propagated to event bus."""
    # Capture emitted events
    events = []
    
    def capture_event(event):
        events.append(event)
    
    # Register handler
    event_bus.on(EventType.ERROR, capture_event)
    
    try:
        # Simulate API error
        error_data = {
            "error": {
                "type": "rate_limit_exceeded",
                "message": "Rate limit exceeded"
            }
        }
        
        # Process error
        await conversation_manager._handle_error(Event(type=EventType.ERROR, data=error_data))
        
        # Should have propagated error
        assert len(events) == 1
        assert events[0].type == EventType.ERROR
        assert "rate_limit_exceeded" in str(events[0].data)
    finally:
        # Clean up
        event_bus.off(EventType.ERROR, capture_event)


@pytest.mark.asyncio
async def test_speech_event_handling(conversation_manager, mock_audio_service):
    """Test handling of speech events."""
    # Start conversation
    await conversation_manager.start()
    
    # Simulate user speech events
    start_event = Event(type=EventType.USER_SPEECH_STARTED)
    ongoing_event = Event(
        type=EventType.USER_SPEECH_ONGOING,
        data={"audio_data": b"audio chunk"}
    )
    finish_event = Event(
        type=EventType.USER_SPEECH_FINISHED,
        data={"audio_data": b"final audio"}
    )
    
    # Process events
    await conversation_manager._handle_user_speech_started(start_event)
    assert conversation_manager.user_speaking is True
    
    await conversation_manager._handle_user_speech_ongoing(ongoing_event)
    conversation_manager.api_client.send_audio.assert_called_with(b"audio chunk", is_final=False)
    
    await conversation_manager._handle_user_speech_finished(finish_event)
    conversation_manager.api_client.send_audio.assert_called_with(b"final audio", is_final=True)
    assert conversation_manager.user_speaking is False


@pytest.mark.asyncio
async def test_assistant_speech_handling(conversation_manager, mock_audio_service):
    """Test handling of assistant speech events."""
    # Simulate assistant speech event
    audio_chunk = b"assistant audio"
    event = Event(
        type=EventType.AUDIO_SPEECH_CREATED,
        data={"chunk": audio_chunk}
    )
    
    # Process event
    await conversation_manager._handle_assistant_speech(event)
    
    # Should have played audio
    assert conversation_manager.assistant_speaking is True
    assert conversation_manager.state == ConversationState.ASSISTANT_SPEAKING
    mock_audio_service.play_audio.assert_called_with(audio_chunk)


@pytest.mark.asyncio
async def test_transcription_handling(conversation_manager):
    """Test handling of transcription events."""
    # Simulate transcription event
    text = "This is a test transcription"
    event = Event(
        type=EventType.USER_TRANSCRIPTION_COMPLETED,
        data={
            "text": text,
            "is_final": True,
            "language": "en",
            "timestamp": 123456789
        }
    )
    
    # Process event
    await conversation_manager._handle_transcription_completed(event)
    
    # Should have added to history
    assert len(conversation_manager.messages) == 1
    assert conversation_manager.messages[0]["content"] == text
    assert conversation_manager.messages[0]["role"] == "user"
    assert conversation_manager.messages[0]["is_transcription"] is True


@pytest.mark.asyncio
async def test_exit_command_detection(conversation_manager):
    """Test detection of exit commands in transcriptions."""
    # Capture events
    events = []
    
    def capture_event(event):
        events.append(event)
    
    # Register handler
    event_bus.on(EventType.SHUTDOWN, capture_event)
    
    try:
        # Simulate transcription with exit command
        for exit_phrase in ["goodbye", "bye", "exit"]:
            events.clear()
            event = Event(
                type=EventType.USER_TRANSCRIPTION_COMPLETED,
                data={
                    "text": f"I want to say {exit_phrase} now",
                    "is_final": True
                }
            )
            
            # Process event
            await conversation_manager._handle_transcription_completed(event)
            
            # Should have detected exit command and emitted shutdown event
            assert len(events) == 1
            assert events[0].type == EventType.SHUTDOWN
            assert events[0].data["reason"] == "user_exit_command"
    finally:
        # Clean up
        event_bus.off(EventType.SHUTDOWN, capture_event)


@pytest.mark.asyncio
async def test_set_transcription_enabled(conversation_manager, mock_api_client):
    """Test enabling and disabling transcription."""
    # Enable transcription
    result = await conversation_manager.set_transcription_enabled(True)
    
    # Should have updated session
    assert result is True
    mock_api_client.update_session.assert_called_once()
    
    # Check call arguments
    call_args = mock_api_client.update_session.call_args[0][0]
    assert "input_audio_transcription" in call_args
    
    # Reset mock
    mock_api_client.update_session.reset_mock()
    
    # Disable transcription
    result = await conversation_manager.set_transcription_enabled(False)
    
    # Should have updated session
    assert result is True
    mock_api_client.update_session.assert_called_once()
    
    # Check call arguments
    call_args = mock_api_client.update_session.call_args[0][0]
    assert call_args["input_audio_transcription"] is None
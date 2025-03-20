"""
Tests for the Realtime API client.

This module tests the OpenAI Realtime API client implementation,
focusing on connection management, error handling, and event processing.
"""

import asyncio
import json
import pytest
import websockets
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock, call

from src.services.api_client import RealtimeClient
from src.events.event_interface import Event, EventType, event_bus
from src.utils.error_handling import ApiError, ErrorSeverity


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket for testing."""
    mock = AsyncMock(spec=websockets.WebSocketClientProtocol)
    mock.send = AsyncMock()
    mock.recv = AsyncMock(return_value=json.dumps({"type": "session.created", "session": {"id": "test_session_id"}}))
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    with patch("src.services.api_client.settings") as mock_settings:
        # Configure mock settings
        mock_settings.api = MagicMock()
        mock_settings.api.api_key = "test_api_key"
        mock_settings.api.api_base = "https://api.openai.com"
        mock_settings.api.base_url = "https://api.openai.com"
        mock_settings.api.model = "gpt-4o-realtime-preview"
        yield mock_settings


@pytest.fixture
def api_client(mock_settings):
    """Create a RealtimeClient instance for testing."""
    client = RealtimeClient()
    return client


@pytest.mark.asyncio
async def test_connection_authentication(api_client, mock_websocket):
    """Test connection with valid and invalid credentials."""
    # Test successful connection
    with patch("websockets.connect", return_value=mock_websocket):
        success = await api_client.connect(
            assistant_id="test_assistant_id",
            session_id="test_session_id"
        )
        
        # Should successfully connect
        assert success is True
        assert api_client.connected is True
        
        # Should have called connect with correct URL and headers
        websockets.connect.assert_called_once()
        call_args = websockets.connect.call_args[0][0]
        assert "api.openai.com" in call_args
        assert "test_assistant_id" in api_client.thread_id
    
    # Test failed connection due to invalid API key
    with patch("websockets.connect", side_effect=Exception("Invalid API key")):
        with patch.object(event_bus, "emit") as mock_emit:
            # Reset client state
            api_client.connected = False
            
            success = await api_client.connect(assistant_id="test_assistant_id")
            
            # Should fail to connect
            assert success is False
            
            # Should emit an error event
            mock_emit.assert_called_once()
            args = mock_emit.call_args[0]
            assert args[0] == EventType.ERROR


@pytest.mark.asyncio
async def test_reconnection_backoff(api_client, mock_websocket):
    """Test exponential backoff logic for reconnections."""
    # Make connect fail the first time, then succeed
    connect_mock = AsyncMock(side_effect=[False, True])
    
    with patch.object(api_client, "connect", connect_mock):
        with patch.object(api_client, "disconnect", AsyncMock()):
            # Try reconnecting
            success = await api_client.reconnect()
            
            # Should eventually succeed
            assert success is True
            
            # Should have attempted to connect twice
            assert connect_mock.call_count == 2
            
            # Should have disconnected first
            api_client.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_session_timeout_handling(api_client, mock_websocket):
    """Test handling of session timeouts (max 30 min per docs)."""
    # Setup
    with patch("websockets.connect", return_value=mock_websocket):
        # Connect client
        await api_client.connect(assistant_id="test_assistant_id")
        
        # Mock WebSocket connection closing with timeout code
        mock_websocket.recv.side_effect = websockets.exceptions.ConnectionClosedOK(
            code=1001,  # Going away
            reason="Session timeout"
        )
        
        # Create task to listen for messages
        with patch.object(api_client, "reconnect", AsyncMock(return_value=True)):
            # Start message handler
            task = asyncio.create_task(api_client._message_handler())
            
            # Give it time to process the exception
            await asyncio.sleep(0.1)
            
            # Cancel the task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Should have attempted to reconnect
            assert api_client.reconnect.call_count == 1


@pytest.mark.asyncio
async def test_error_recovery_strategies(api_client, mock_websocket):
    """Test recovery from different error types (rate limits, token limits)."""
    # Setup
    with patch("websockets.connect", return_value=mock_websocket):
        # Connect client
        await api_client.connect(assistant_id="test_assistant_id")
        
        # Test rate limit error
        error_event = {
            "type": "error",
            "code": "rate_limit_exceeded",
            "message": "Rate limit exceeded",
            "retry_after": 1
        }
        
        # Process error message
        with patch.object(event_bus, "emit") as mock_emit:
            await api_client._process_message(json.dumps(error_event))
            
            # Should emit error event to event bus
            mock_emit.assert_called_once()
            args = mock_emit.call_args[0]
            assert args[0] == EventType.ERROR
            assert "rate_limit_exceeded" in str(args[1])
        
        # Test token limit error
        error_event = {
            "type": "error",
            "code": "token_limit_exceeded",
            "message": "Token limit exceeded"
        }
        
        # Process error message
        with patch.object(event_bus, "emit") as mock_emit:
            await api_client._process_message(json.dumps(error_event))
            
            # Should emit error event to event bus
            mock_emit.assert_called_once()
            args = mock_emit.call_args[0]
            assert args[0] == EventType.ERROR
            assert "token_limit_exceeded" in str(args[1])


@pytest.mark.asyncio
async def test_event_sequence_validation(api_client, mock_websocket):
    """Test that events are properly validated before sending."""
    # Setup
    with patch("websockets.connect", return_value=mock_websocket):
        # Connect client
        await api_client.connect(assistant_id="test_assistant_id")
        
        # Test sending valid event
        await api_client._send_event({
            "type": "session.update",
            "data": {"instructions": "Test instructions"}
        })
        
        # Should have sent the event
        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert "id" in sent_data  # Should have generated ID
        assert sent_data["type"] == "session.update"
        assert sent_data["data"]["instructions"] == "Test instructions"


@pytest.mark.asyncio
async def test_send_audio(api_client, mock_websocket):
    """Test sending audio data."""
    # Setup
    with patch("websockets.connect", return_value=mock_websocket):
        # Connect client
        await api_client.connect(assistant_id="test_assistant_id")
        
        # Test sending audio
        test_audio = b"test audio data"
        is_final = True
        
        with patch.object(api_client, "_send_event", AsyncMock()) as mock_send:
            await api_client.send_audio(test_audio, is_final)
            
            # Should have sent audio event
            mock_send.assert_called_once()
            event = mock_send.call_args[0][0]
            assert event["type"] == "audio"
            assert "audio" in event["data"]
            assert event["data"]["is_final"] is True


@pytest.mark.asyncio
async def test_request_response(api_client, mock_websocket):
    """Test requesting a response from the model."""
    # Setup
    with patch("websockets.connect", return_value=mock_websocket):
        # Connect client
        await api_client.connect(assistant_id="test_assistant_id")
        
        # Test requesting response
        with patch.object(api_client, "_send_event", AsyncMock(return_value="event_id")) as mock_send:
            success = await api_client.request_response(instructions="Test instructions")
            
            # Should have succeeded
            assert success is True
            
            # Should have sent response.create event
            mock_send.assert_called_once()
            event = mock_send.call_args[0][0]
            assert event["type"] == "response.create"
            assert "instructions" in event["data"]
            assert event["data"]["instructions"] == "Test instructions"


@pytest.mark.asyncio
async def test_interrupt(api_client, mock_websocket):
    """Test interrupting the assistant's response."""
    # Setup
    with patch("websockets.connect", return_value=mock_websocket):
        # Connect client
        await api_client.connect(assistant_id="test_assistant_id")
        
        # Test interrupting response
        with patch.object(api_client, "_send_event", AsyncMock(return_value="event_id")) as mock_send:
            success = await api_client.interrupt()
            
            # Should have succeeded
            assert success is True
            
            # Should have sent interrupt event
            mock_send.assert_called_once()
            event = mock_send.call_args[0][0]
            assert event["type"] == "interrupt"


@pytest.mark.asyncio
async def test_update_session(api_client, mock_websocket):
    """Test updating session configuration."""
    # Setup
    with patch("websockets.connect", return_value=mock_websocket):
        # Connect client
        await api_client.connect(assistant_id="test_assistant_id")
        
        # Test updating session
        session_config = {
            "voice": "alloy",
            "instructions": "Test instructions",
            "turn_detection": {"type": "server_vad"}
        }
        
        with patch.object(api_client, "_send_event", AsyncMock(return_value="event_id")) as mock_send:
            success = await api_client.update_session(session_config)
            
            # Should have succeeded
            assert success is True
            
            # Should have sent session.update event
            mock_send.assert_called_once()
            event = mock_send.call_args[0][0]
            assert event["type"] == "session.update"
            assert "data" in event
            assert event["data"] == session_config


@pytest.mark.asyncio
async def test_submit_tool_outputs(api_client, mock_websocket):
    """Test submitting tool outputs for a function call."""
    # Setup
    with patch("websockets.connect", return_value=mock_websocket):
        # Connect client
        await api_client.connect(assistant_id="test_assistant_id")
        
        # Set thread_id and run_id
        api_client.thread_id = "thread_123"
        api_client.run_id = "run_456"
        
        # Test submitting tool outputs
        tool_outputs = [
            {"tool_call_id": "call_789", "output": "Test output"}
        ]
        
        with patch.object(api_client, "_send_event", AsyncMock(return_value="event_id")) as mock_send:
            success = await api_client.submit_tool_outputs(tool_outputs)
            
            # Should have succeeded
            assert success is True
            
            # Should have sent thread.run.tool_outputs.create event
            mock_send.assert_called_once()
            event = mock_send.call_args[0][0]
            assert event["type"] == "thread.run.tool_outputs.create"
            assert "data" in event
            assert event["data"]["thread_id"] == "thread_123"
            assert event["data"]["run_id"] == "run_456"
            assert event["data"]["tool_outputs"] == tool_outputs


@pytest.mark.asyncio
async def test_message_processing(api_client, mock_websocket):
    """Test processing messages from the WebSocket."""
    # Setup
    with patch("websockets.connect", return_value=mock_websocket):
        # Connect client
        await api_client.connect(assistant_id="test_assistant_id")
        
        # Test processing a valid message
        message = {
            "type": "session.created",
            "session": {"id": "session_123"}
        }
        
        with patch("src.services.realtime_event_handler.event_handler.handle_event") as mock_handler:
            await api_client._process_message(json.dumps(message))
            
            # Should have called event handler
            mock_handler.assert_called_once_with("session.created", {"id": "session_123"})
        
        # Test processing an invalid JSON message
        with patch("src.services.realtime_event_handler.event_handler.handle_event") as mock_handler:
            with patch.object(api_client.task_manager, "create_task") as mock_create_task:
                await api_client._process_message("invalid json")
                
                # Should not have called event handler
                mock_handler.assert_not_called()
                
                # Should have logged error
                # This is hard to assert directly, but we can check it didn't crash


@pytest.mark.asyncio
async def test_disconnect(api_client, mock_websocket):
    """Test disconnecting from the API."""
    # Setup
    with patch("websockets.connect", return_value=mock_websocket):
        # Connect client
        await api_client.connect(assistant_id="test_assistant_id")
        
        # Test disconnecting
        await api_client.disconnect()
        
        # Should have closed WebSocket
        mock_websocket.close.assert_called_once()
        
        # Should have reset state
        assert api_client.connected is False
        assert api_client.ws is None


@pytest.mark.asyncio
async def test_event_propagation(api_client, mock_websocket):
    """Test that received events are properly propagated to the event bus."""
    # Setup
    with patch("websockets.connect", return_value=mock_websocket):
        # Connect client
        await api_client.connect(assistant_id="test_assistant_id")
        
        # Test processing different event types
        events = [
            {"type": "session.created", "session": {"id": "session_123"}},
            {"type": "input_audio_buffer.speech_started", "timestamp": 123456789},
            {"type": "response.text.delta", "delta": "Hello", "response": {"id": "resp_123"}}
        ]
        
        for event_data in events:
            with patch("src.services.realtime_event_handler.event_handler.handle_event") as mock_handler:
                await api_client._process_message(json.dumps(event_data))
                
                # Should have called event handler with correct event type and data
                event_type = event_data["type"]
                mock_handler.assert_called_once()
                assert mock_handler.call_args[0][0] == event_type
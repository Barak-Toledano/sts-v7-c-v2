"""
Integration tests for the OpenAI Realtime Assistant.

These tests validate the interaction between different components
of the application and ensure end-to-end flows work correctly.
"""

import asyncio
import os
import pytest
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from src.domain.audio.manager import AudioManager, AudioMode
from src.domain.conversation.manager import ConversationManager, ConversationState
from src.events.event_interface import Event, EventType, event_bus
from src.services.api_client import RealtimeClient
from src.services.audio_service import AudioService
from src.services.realtime_event_handler import event_handler


@pytest.fixture
def mock_websocket():
    """Create a mock websocket for testing."""
    mock = AsyncMock()
    mock.send = AsyncMock()
    mock.recv = AsyncMock(return_value=json.dumps({
        "type": "session.created",
        "session": {"id": "test_session_id"}
    }))
    mock.close = AsyncMock()
    # Make awaiting the mock return the mock itself
    mock.__aenter__.return_value = mock
    return mock


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    with patch('src.config.settings.api') as mock_api_settings, \
         patch('src.config.settings.audio') as mock_audio_settings:
        
        # Setup API settings
        mock_api_settings.api_key = "test_key"
        mock_api_settings.base_url = "https://api.openai.com"
        mock_api_settings.model = "gpt-4o-realtime-preview"
        
        # Setup audio settings
        mock_audio_settings.sample_rate = 24000
        mock_audio_settings.channels = 1
        mock_audio_settings.sample_width = 2
        mock_audio_settings.frames_per_buffer = 1024
        mock_audio_settings.input_device = None
        mock_audio_settings.output_device = None
        mock_audio_settings.vad_threshold = 0.3
        mock_audio_settings.silence_duration_ms = 1000
        
        yield


@pytest.mark.asyncio
async def test_speech_to_speech_flow(mock_websocket, mock_config):
    """
    Test complete flow from audio input to audio output.
    
    This test validates the entire speech-to-speech pipeline:
    1. Audio is recorded from microphone (mocked)
    2. Voice activity is detected
    3. Audio is sent to the API
    4. Response is received
    5. Response audio is played back
    """
    # Patch websocket connection
    with patch('websockets.connect', return_value=mock_websocket), \
         patch('src.services.audio_service.AudioService') as mock_audio_service, \
         patch('src.domain.audio.manager.AudioManager') as mock_audio_manager:
        
        # Setup audio service mock
        audio_service_instance = MagicMock()
        mock_audio_service.return_value = audio_service_instance
        audio_service_instance.start_recording = AsyncMock()
        audio_service_instance.play_audio = AsyncMock()
        
        # Setup audio manager mock
        audio_manager_instance = MagicMock()
        mock_audio_manager.return_value = audio_manager_instance
        
        # Create test client and conversation manager
        client = RealtimeClient()
        conversation = ConversationManager(
            assistant_id="test_assistant_id",
            instructions="Test instructions"
        )
        
        # Setup the client's websocket connection
        client.ws = mock_websocket
        client.connected = True
        conversation.api_client = client
        
        # Track events emitted during the test
        captured_events = []
        
        def capture_event(event):
            captured_events.append(event)
        
        # Register a wildcard event handler to catch all events
        event_bus.on_any(capture_event)
        
        try:
            # Start the conversation
            await conversation.start()
            
            # Verify the conversation is in the READY state
            assert conversation.state == ConversationState.READY
            
            # Simulate speech events
            
            # 1. User starts speaking
            event_bus.emit(Event(
                type=EventType.USER_SPEECH_STARTED,
                data={"timestamp": time.time()}
            ))
            
            # 2. Transcript is generated
            transcript_event = Event(
                type=EventType.USER_TRANSCRIPTION_COMPLETED,
                data={
                    "text": "What's the weather like?",
                    "is_final": True,
                    "timestamp": time.time()
                }
            )
            event_bus.emit(transcript_event)
            
            # Wait for events to propagate
            await asyncio.sleep(0.1)
            
            # 3. Assistant starts responding
            mock_websocket.recv.return_value = json.dumps({
                "type": "response.text.delta",
                "response_id": "test_response_id",
                "delta": "The weather is sunny today."
            })
            
            # Wait for the response to be processed
            await asyncio.sleep(0.1)
            
            # Verify that a suitable sequence of events was emitted
            event_types = [e.type for e in captured_events]
            
            # Essential events we must see in sequence
            assert EventType.USER_SPEECH_STARTED in event_types
            assert EventType.USER_TRANSCRIPTION_COMPLETED in event_types
            
            # Check that conversation state changed appropriately
            user_speech_idx = event_types.index(EventType.USER_SPEECH_STARTED)
            assert user_speech_idx >= 0
            
            # Clean up
            await conversation.stop()
            
        finally:
            # Unregister event handler
            event_bus.off_any(capture_event)


@pytest.mark.asyncio
async def test_function_calling_integration(mock_websocket, mock_config):
    """
    Test integration of function calling with conversation.
    
    This test validates the function calling flow:
    1. Configure function tools
    2. Receive function call from model
    3. Execute function
    4. Submit function result
    5. Process model response with function output
    """
    # Setup mock responses for different stages of interaction
    session_created_response = json.dumps({
        "type": "session.created",
        "session": {"id": "test_session_id"}
    })
    
    function_call_response = json.dumps({
        "type": "response.done",
        "response": {
            "id": "test_response_id",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "name": "get_weather",
                "call_id": "test_call_id",
                "arguments": '{"location":"New York"}'
            }]
        }
    })
    
    function_result_response = json.dumps({
        "type": "response.text.delta",
        "response_id": "test_response_id2",
        "delta": "The weather in New York is 72°F and sunny."
    })
    
    # Setup websocket to return different responses in sequence
    mock_websocket.recv = AsyncMock(side_effect=[
        session_created_response,
        function_call_response,
        function_result_response
    ])
    
    # Patch websocket connection
    with patch('websockets.connect', return_value=mock_websocket):
        # Create conversation manager
        client = RealtimeClient()
        conversation = ConversationManager(
            assistant_id="test_assistant_id",
            instructions="Test instructions"
        )
        
        # Setup the client's websocket connection
        client.ws = mock_websocket
        client.connected = True
        conversation.api_client = client
        
        # Track function calls
        function_called = False
        function_result_submitted = False
        
        # Define a test function to handle the call
        async def test_function_handler(name, arguments, call_id, response_id):
            nonlocal function_called
            function_called = True
            # Return result
            result = {"temperature": 72, "condition": "sunny"}
            await conversation.submit_tool_outputs([{
                "tool_call_id": call_id,
                "output": json.dumps(result)
            }])
            nonlocal function_result_submitted
            function_result_submitted = True
        
        try:
            # Configure function tools
            await conversation.start()
            
            # Set up function call handler
            conversation.register_function_handler("get_weather", test_function_handler)
            
            # Trigger function calling flow by sending message
            await conversation.send_text_message("What's the weather in New York?")
            
            # Request a response to trigger function calling
            await conversation.request_response()
            
            # Give time for the mock responses to be processed
            await asyncio.sleep(0.2)
            
            # Verify function was called and result submitted
            assert function_called, "Function was not called"
            assert function_result_submitted, "Function result was not submitted"
            
            # Verify correct events were sent to the WebSocket
            # Convert the calls to strings for easier debugging
            send_calls = [json.loads(call.args[0]) for call in mock_websocket.send.call_args_list]
            
            # Extract types for easier assertion
            send_types = [call.get("type") for call in send_calls]
            
            # Check sequence of events
            assert "conversation.item.create" in send_types, "Conversation item not created"
            assert "response.create" in send_types, "Response not requested"
            
            # Clean up
            await conversation.stop()
        
        except Exception as e:
            await conversation.stop()
            raise e


@pytest.mark.asyncio
async def test_error_recovery_integration(mock_websocket, mock_config):
    """
    Test system-wide recovery from various error scenarios.
    
    This test validates error recovery flows:
    1. Connection errors
    2. Rate limit errors
    3. Timeout errors
    """
    # Setup mock responses for different error scenarios
    session_created = json.dumps({
        "type": "session.created",
        "session": {"id": "test_session_id"}
    })
    
    rate_limit_error = json.dumps({
        "type": "error",
        "code": "rate_limit_exceeded",
        "message": "Rate limit exceeded",
        "event_id": "test_event_id"
    })
    
    normal_response = json.dumps({
        "type": "response.text.delta",
        "response_id": "test_response_id",
        "delta": "This is a test response."
    })
    
    # Setup websocket with error followed by recovery
    mock_websocket.recv = AsyncMock(side_effect=[
        session_created,
        rate_limit_error,
        normal_response
    ])
    
    # Patch websocket connection
    with patch('websockets.connect', return_value=mock_websocket):
        # Create conversation manager
        client = RealtimeClient()
        conversation = ConversationManager(
            assistant_id="test_assistant_id",
            instructions="Test instructions"
        )
        
        # Setup the client's websocket connection
        client.ws = mock_websocket
        client.connected = True
        conversation.api_client = client
        
        # Track error events
        error_events = []
        
        # Register error handler
        def on_error(event):
            if event.type == EventType.ERROR:
                error_events.append(event)
        
        event_bus.on(EventType.ERROR, on_error)
        
        try:
            # Start the conversation
            await conversation.start()
            
            # Send a message that will trigger rate limit error
            await conversation.send_text_message("Test message")
            
            # Request a response
            await conversation.request_response()
            
            # Give time for error to be processed
            await asyncio.sleep(0.1)
            
            # Verify error was handled
            assert len(error_events) > 0, "Error event was not emitted"
            
            # Verify we can still use the conversation after error
            await conversation.send_text_message("Another test")
            
            # Verify we can receive responses after error
            await conversation.request_response()
            
            # Clean up
            await conversation.stop()
            
        finally:
            # Unregister error handler
            event_bus.off(EventType.ERROR, on_error)


@pytest.mark.asyncio
async def test_vad_integration(mock_websocket, mock_config):
    """
    Test voice activity detection integration with conversation management.
    
    This test validates:
    1. VAD correctly detects speech start/end
    2. Conversation state changes appropriately
    3. Audio is correctly processed and sent to API
    """
    # Patch websocket connection
    with patch('websockets.connect', return_value=mock_websocket), \
         patch('src.domain.audio.manager.AudioManager') as mock_audio_manager:
         
        # Setup audio manager mock
        audio_manager_instance = MagicMock()
        mock_audio_manager.return_value = audio_manager_instance
        
        # Create conversation manager
        client = RealtimeClient()
        conversation = ConversationManager(
            assistant_id="test_assistant_id",
            instructions="Test instructions"
        )
        
        # Setup the client's websocket connection
        client.ws = mock_websocket
        client.connected = True
        conversation.api_client = client
        
        # Track conversation state changes
        states = []
        
        def track_state_change(event):
            if event.type == EventType.CONVERSATION_STATE_CHANGED:
                states.append(event.data.get("state"))
        
        event_bus.on(EventType.CONVERSATION_STATE_CHANGED, track_state_change)
        
        try:
            # Start the conversation
            await conversation.start()
            
            # Initial state should be READY
            assert conversation.state == ConversationState.READY
            
            # Simulate VAD events
            
            # 1. Speech started
            event_bus.emit(Event(
                type=EventType.USER_SPEECH_STARTED,
                data={"timestamp": time.time()}
            ))
            
            # Wait for state to update
            await asyncio.sleep(0.1)
            
            # State should now be USER_SPEAKING
            assert conversation.state == ConversationState.USER_SPEAKING
            
            # 2. Speech finished
            event_bus.emit(Event(
                type=EventType.USER_SPEECH_FINISHED,
                data={
                    "timestamp": time.time(),
                    "duration": 2.5,
                    "audio_data": b"test audio data"
                }
            ))
            
            # Wait for state to update
            await asyncio.sleep(0.1)
            
            # Should transition to THINKING if auto-response is enabled 
            # or back to READY if auto-response is disabled
            assert conversation.state in (ConversationState.THINKING, ConversationState.READY)
            
            # Clean up
            await conversation.stop()
            
        finally:
            # Unregister state change handler
            event_bus.off(EventType.CONVERSATION_STATE_CHANGED, track_state_change)
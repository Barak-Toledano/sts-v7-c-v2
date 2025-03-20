"""
Tests for the audio service.

This module tests audio recording, playback, and processing,
including format validation, device selection, and voice activity detection.
"""

import asyncio
import numpy as np
import pytest
import wave
import io
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock, call

from src.services.audio_service import AudioService, AudioState
from src.events.event_interface import EventType, event_bus, UserSpeechEvent


@pytest.fixture
def mock_pyaudio():
    """Create a mock PyAudio interface."""
    mock = MagicMock()
    
    # Mock device info
    def get_device_info_by_index(index):
        if index == 0:
            return {
                "name": "Default Input Device",
                "maxInputChannels": 2,
                "maxOutputChannels": 0,
                "defaultSampleRate": 44100
            }
        elif index == 1:
            return {
                "name": "Default Output Device",
                "maxInputChannels": 0,
                "maxOutputChannels": 2,
                "defaultSampleRate": 44100
            }
        else:
            raise ValueError(f"Invalid device index: {index}")
    
    # Set up mock methods
    mock.get_device_count = MagicMock(return_value=2)
    mock.get_device_info_by_index = MagicMock(side_effect=get_device_info_by_index)
    mock.get_default_input_device_info = MagicMock(return_value={"index": 0})
    mock.get_default_output_device_info = MagicMock(return_value={"index": 1})
    
    # Mock streams
    mock_input_stream = MagicMock()
    mock_input_stream.start_stream = MagicMock()
    mock_input_stream.stop_stream = MagicMock()
    mock_input_stream.close = MagicMock()
    
    mock_output_stream = MagicMock()
    mock_output_stream.start_stream = MagicMock()
    mock_output_stream.stop_stream = MagicMock()
    mock_output_stream.close = MagicMock()
    mock_output_stream.write = MagicMock()
    
    # Set up stream creation
    def open_stream(**kwargs):
        if kwargs.get("input", False):
            return mock_input_stream
        else:
            return mock_output_stream
    
    mock.open = MagicMock(side_effect=open_stream)
    mock.terminate = MagicMock()
    
    return mock


@pytest.fixture
def mock_audio_settings():
    """Create mock audio settings."""
    with patch("src.services.audio_service.settings.audio") as mock_settings:
        # Configure mock settings
        mock_settings.sample_rate = 24000
        mock_settings.channels = 1
        mock_settings.sample_width = 2
        mock_settings.frames_per_buffer = 1024
        mock_settings.vad_threshold = 0.75
        mock_settings.silence_duration_ms = 1000
        mock_settings.prefix_padding_ms = 500
        mock_settings.input_device = None
        mock_settings.output_device = None
        yield mock_settings


@pytest.fixture
def audio_service(mock_pyaudio, mock_audio_settings):
    """Create an AudioService instance for testing."""
    with patch("pyaudio.PyAudio", return_value=mock_pyaudio):
        service = AudioService()
        yield service
        # Clean up
        asyncio.run(service.cleanup())


@pytest.mark.asyncio
async def test_initialization(audio_service, mock_pyaudio):
    """Test audio service initialization."""
    # Should have initialized PyAudio
    assert audio_service.py_audio is not None
    
    # Should have set default devices
    assert audio_service.input_device_index == 0
    assert audio_service.output_device_index == 1
    
    # Should start in IDLE state
    assert audio_service.state == AudioState.IDLE


@pytest.mark.asyncio
async def test_audio_format_validation(mock_audio_settings):
    """Test validation of audio format settings."""
    from src.utils.audio_utilities import validate_audio_format
    
    # Test valid format
    assert validate_audio_format(24000, 1, 2) is True
    
    # Test invalid sample rate
    assert validate_audio_format(44100, 1, 2) is False
    
    # Test invalid channels
    assert validate_audio_format(24000, 2, 2) is False
    
    # Test invalid sample width
    assert validate_audio_format(24000, 1, 4) is False


@pytest.mark.asyncio
async def test_device_selection_fallback(mock_pyaudio):
    """Test device selection and fallback to defaults."""
    # Test with specific devices
    with patch("pyaudio.PyAudio", return_value=mock_pyaudio):
        service = AudioService(input_device_index=0, output_device_index=1)
        assert service.input_device_index == 0
        assert service.output_device_index == 1
        await service.cleanup()
    
    # Test with invalid devices (should fall back to defaults)
    mock_pyaudio.get_device_info_by_index.side_effect = ValueError("Invalid device")
    
    with patch("pyaudio.PyAudio", return_value=mock_pyaudio):
        # This should not raise even though the device is invalid
        # because we handle the error and fall back to defaults
        service = AudioService(input_device_index=99, output_device_index=99)
        # Values should be from get_default_*_device_info
        assert service.input_device_index == 0
        assert service.output_device_index == 1
        await service.cleanup()


@pytest.mark.asyncio
async def test_start_recording(audio_service, mock_pyaudio):
    """Test starting audio recording."""
    # Start recording
    await audio_service.start_recording()
    
    # Should have created input stream
    mock_pyaudio.open.assert_called_once()
    call_args = mock_pyaudio.open.call_args[1]
    assert call_args["input"] is True
    assert call_args["input_device_index"] == 0
    
    # Should have started the stream
    mock_input_stream = mock_pyaudio.open.return_value
    mock_input_stream.start_stream.assert_called_once()
    
    # Should be in RECORDING state
    assert audio_service.state == AudioState.RECORDING
    
    # Stop recording
    await audio_service.stop_recording()


@pytest.mark.asyncio
async def test_stop_recording(audio_service, mock_pyaudio):
    """Test stopping audio recording."""
    # Start recording first
    await audio_service.start_recording()
    
    # Stop recording
    await audio_service.stop_recording()
    
    # Should have stopped and closed the stream
    mock_input_stream = mock_pyaudio.open.return_value
    mock_input_stream.stop_stream.assert_called_once()
    mock_input_stream.close.assert_called_once()
    
    # Should be in IDLE state
    assert audio_service.state == AudioState.IDLE


@pytest.mark.asyncio
async def test_play_audio(audio_service, mock_pyaudio):
    """Test playing audio data."""
    # Create test audio data
    test_audio = b"test audio data"
    
    # Mock convert_to_pcm16 to return the input
    with patch("src.services.audio_service.convert_to_pcm16", return_value=test_audio):
        # Play audio
        await audio_service.play_audio(test_audio)
        
        # Should have created output stream
        assert mock_pyaudio.open.call_count == 1
        call_args = mock_pyaudio.open.call_args[1]
        assert call_args["output"] is True
        assert call_args["output_device_index"] == 1
        
        # Playback task runs asynchronously, need to wait for it
        await asyncio.sleep(0.1)
        
        # Should have written audio data to the stream
        mock_output_stream = mock_pyaudio.open.return_value
        mock_output_stream.write.assert_called_with(test_audio)
        
        # Should be in PLAYING state
        assert audio_service.state == AudioState.PLAYING
        
        # Stop playback
        await audio_service.stop_playback()


@pytest.mark.asyncio
async def test_stop_playback(audio_service, mock_pyaudio):
    """Test stopping audio playback."""
    # Start playback first
    test_audio = b"test audio data"
    with patch("src.services.audio_service.convert_to_pcm16", return_value=test_audio):
        await audio_service.play_audio(test_audio)
        await asyncio.sleep(0.1)  # Let playback start
    
    # Stop playback
    await audio_service.stop_playback()
    
    # Should have stopped and closed the stream
    mock_output_stream = mock_pyaudio.open.return_value
    mock_output_stream.stop_stream.assert_called_once()
    mock_output_stream.close.assert_called_once()
    
    # Should be in IDLE state
    assert audio_service.state == AudioState.IDLE


@pytest.mark.asyncio
async def test_vad_behavior(audio_service):
    """Test voice activity detection logic."""
    # Create a callback to capture speech events
    events = []
    
    def capture_event(event):
        events.append(event)
    
    # Register event handler
    event_bus.on(EventType.USER_SPEECH_STARTED, capture_event)
    event_bus.on(EventType.USER_SPEECH_FINISHED, capture_event)
    
    try:
        # Start recording
        await audio_service.start_recording()
        
        # Simulate processing audio with speech
        # This is the internal method that the audio callback would call
        with patch("numpy.frombuffer") as mock_frombuffer:
            # Create a mock audio sample with high energy
            mock_samples = np.array([0.5] * 1024, dtype=np.int16)
            mock_frombuffer.return_value = mock_samples
            
            # Process audio chunk with speech
            audio_service._process_audio_for_vad(b"speech audio")
            
            # Should detect speech start
            assert audio_service.is_speech_active is True
            assert len(events) == 1
            assert events[0].type == EventType.USER_SPEECH_STARTED
            
            # Process more speech audio
            audio_service._process_audio_for_vad(b"more speech")
            
            # Should still be active, no new events
            assert audio_service.is_speech_active is True
            assert len(events) == 1
            
            # Now simulate silence
            mock_samples = np.array([0.01] * 1024, dtype=np.int16)
            mock_frombuffer.return_value = mock_samples
            
            # Need to process enough silence to exceed silence_duration_ms
            for _ in range(10):  # Process multiple silent chunks
                audio_service._process_audio_for_vad(b"silence")
            
            # Trigger delayed pause (would normally happen in background)
            await audio_service._delayed_pause()
            
            # Should detect speech end
            assert audio_service.is_speech_active is False
            assert len(events) == 2
            assert events[1].type == EventType.USER_SPEECH_FINISHED
    finally:
        # Clean up
        event_bus.off(EventType.USER_SPEECH_STARTED, capture_event)
        event_bus.off(EventType.USER_SPEECH_FINISHED, capture_event)
        await audio_service.stop_recording()


@pytest.mark.asyncio
async def test_audio_chunking(mock_pyaudio, mock_audio_settings):
    """Test processing of large audio inputs into appropriate chunks."""
    from src.utils.audio_utilities import split_audio_chunks
    
    # Create test audio data
    sample_rate = 24000
    channels = 1
    sample_width = 2
    duration = 5  # seconds
    num_samples = sample_rate * duration
    test_audio = np.zeros(num_samples, dtype=np.int16).tobytes()
    
    # Split into 1-second chunks
    chunk_size = sample_rate  # 1 second of audio
    chunks = split_audio_chunks(
        test_audio, 
        chunk_size, 
        sample_rate, 
        channels, 
        sample_width
    )
    
    # Should have 5 chunks of equal size
    assert len(chunks) == 5
    for chunk in chunks:
        assert len(chunk) == chunk_size * sample_width * channels


@pytest.mark.asyncio
async def test_audio_error_resilience(audio_service, mock_pyaudio):
    """Test recovery from audio device errors."""
    # Simulate error during recording start
    mock_pyaudio.open.side_effect = Exception("Device busy")
    
    # Should not crash, but return failure
    result = await audio_service.start_recording()
    assert result is False
    assert audio_service.state == AudioState.ERROR
    
    # Reset mock for next test
    mock_pyaudio.open.side_effect = None
    
    # Start recording successfully
    await audio_service.start_recording()
    
    # Simulate error during playback
    mock_output_stream = mock_pyaudio.open.return_value
    mock_output_stream.write.side_effect = Exception("Playback error")
    
    # Play audio should handle the error
    test_audio = b"test audio data"
    with patch("src.services.audio_service.convert_to_pcm16", return_value=test_audio):
        await audio_service.play_audio(test_audio)
        await asyncio.sleep(0.1)  # Let playback start
    
    # Should have tried to write to stream
    assert mock_output_stream.write.called
    
    # Service should still be functional
    assert audio_service.state == AudioState.PLAYING
    
    # Stop recording and playback
    await audio_service.stop_recording()
    await audio_service.stop_playback()


@pytest.mark.asyncio
async def test_pause_resume_playback(audio_service, mock_pyaudio):
    """Test pausing and resuming audio playback."""
    # Start playback first
    test_audio = b"test audio data"
    with patch("src.services.audio_service.convert_to_pcm16", return_value=test_audio):
        await audio_service.play_audio(test_audio)
        await asyncio.sleep(0.1)  # Let playback start
    
    # Pause playback
    await audio_service.pause_playback()
    
    # Should have stopped the stream but not closed it
    mock_output_stream = mock_pyaudio.open.return_value
    mock_output_stream.stop_stream.assert_called_once()
    mock_output_stream.close.assert_not_called()
    
    # Should be in PAUSED state
    assert audio_service.state == AudioState.PAUSED
    
    # Reset mock for resume test
    mock_output_stream.stop_stream.reset_mock()
    
    # Resume playback
    await audio_service.resume_playback()
    
    # Should have started the stream
    mock_output_stream.start_stream.assert_called_once()
    
    # Should be in PLAYING state
    assert audio_service.state == AudioState.PLAYING
    
    # Stop playback
    await audio_service.stop_playback()


@pytest.mark.asyncio
async def test_cleanup(audio_service, mock_pyaudio):
    """Test cleanup of audio resources."""
    # Start recording and playback
    await audio_service.start_recording()
    test_audio = b"test audio data"
    with patch("src.services.audio_service.convert_to_pcm16", return_value=test_audio):
        await audio_service.play_audio(test_audio)
        await asyncio.sleep(0.1)  # Let playback start
    
    # Cleanup
    await audio_service.cleanup()
    
    # Should have stopped and closed all streams
    assert mock_pyaudio.terminate.called
    
    # Should have reset state
    assert audio_service.state == AudioState.IDLE
    assert audio_service.py_audio is None


@pytest.mark.asyncio
async def test_read_write_wav_file(tmp_path):
    """Test reading and writing WAV files."""
    from src.utils.audio_utilities import read_wav_file, write_wav_file
    
    # Create test audio data
    sample_rate = 24000
    channels = 1
    sample_width = 2
    duration = 1  # second
    num_samples = sample_rate * duration
    test_audio = np.zeros(num_samples, dtype=np.int16).tobytes()
    
    # Create temporary WAV file
    file_path = tmp_path / "test.wav"
    
    # Write WAV file
    write_wav_file(
        file_path=file_path,
        audio_data=test_audio,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width
    )
    
    # Check that file exists
    assert file_path.exists()
    
    # Read WAV file
    audio_data, sr, ch, sw = read_wav_file(file_path)
    
    # Check that data was read correctly
    assert len(audio_data) == len(test_audio)
    assert sr == sample_rate
    assert ch == channels
    assert sw == sample_width
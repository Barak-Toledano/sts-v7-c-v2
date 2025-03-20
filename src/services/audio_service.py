"""
Audio service for the OpenAI Realtime Assistant.

This module handles audio recording from microphone and playback to speakers,
including voice activity detection and audio format conversion.
"""

import asyncio
import collections
import wave
import base64
import os
import queue
import threading
import numpy as np
import pyaudio
import time
import struct
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, BinaryIO

from src.config import settings
from src.config.logging_config import get_logger
from src.events.event_interface import (
    EventType,
    UserSpeechEvent,
    event_bus,
)
from src.utils.async_helpers import TaskManager, debounce
from src.utils.audio_utilities import (
    convert_to_pcm16,
    detect_silence,
    get_audio_duration,
    validate_audio_format,
    read_wav_file,
    write_wav_file,
    split_audio_chunks
)
from src.utils.error_handling import AudioError, ErrorSeverity, safe_execute

logger = get_logger(__name__)


class AudioState(Enum):
    """Possible states for the audio service."""
    
    IDLE = auto()
    RECORDING = auto()
    PLAYING = auto()
    PAUSED = auto()
    ERROR = auto()


class AudioService:
    """
    Service for recording from microphone and playing back audio.
    
    This class handles:
    - Audio device selection and configuration
    - Recording audio from microphone with VAD
    - Converting and buffering audio data
    - Playing audio through speakers
    - Managing audio state and callbacks
    - Processing audio files
    """
    
    def __init__(
        self,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
        max_queue_size: int = 1000
    ):
        """
        Initialize the audio service.
        
        Args:
            input_device_index: Index of the input device to use (None for default)
            output_device_index: Index of the output device to use (None for default)
            max_queue_size: Maximum number of chunks to keep in queues
        
        Raises:
            AudioError: If audio initialization fails
        """
        # PyAudio instance
        self.py_audio = None
        
        # State tracking
        self.state = AudioState.IDLE
        self.task_manager = TaskManager("audio_service")
        
        # Streams
        self.input_stream = None
        self.output_stream = None
        
        # Device configuration
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        
        # Audio format
        self.sample_rate = settings.audio.sample_rate
        self.channels = settings.audio.channels
        self.sample_width = 2  # 16-bit PCM (matches AUDIO_FORMAT=pcm16)
        self.frames_per_buffer = settings.audio.chunk_size
        self.max_queue_size = max_queue_size
        
        # Voice Activity Detection settings
        self.vad_threshold = settings.audio.vad_threshold
        self.silence_duration_ms = settings.audio.silence_duration_ms
        self.prefix_padding_ms = settings.audio.prefix_padding_ms
        
        # For tracking speech
        self.is_speech_active = False
        self.silence_frames = 0
        self.speech_frames = collections.deque(maxlen=100)  # ~2.5 seconds at 44.1kHz
        self.speech_buffer = bytearray()
        self.pending_pause_task = None
        
        # Output buffer and state
        self.input_queue = asyncio.Queue(maxsize=max_queue_size)
        self.output_queue = asyncio.Queue(maxsize=max_queue_size)
        self.playback_buffer = collections.deque()
        self.is_playing = False
        self.should_stop_playback = False
        
        # For synchronizing with PyAudio callbacks
        self._callback_queue = queue.Queue(maxsize=max_queue_size)
        
        # For VAD (voice activity detection)
        self.is_input_paused = False
        self.speech_active = False
        self.last_speech_start_time = 0.0
        self.speech_pause_pending = False
        
        # Initialize audio
        self._initialize_audio()
    
    def _initialize_audio(self) -> None:
        """
        Initialize PyAudio and configure audio parameters.
        
        Raises:
            AudioError: If audio initialization fails
        """
        try:
            self.py_audio = pyaudio.PyAudio()
            
            # List available devices
            self._log_available_devices()
            
            # Configure default devices if not specified
            if self.input_device_index is None:
                try:
                    self.input_device_index = self.py_audio.get_default_input_device_info()['index']
                except IOError:
                    logger.warning("No default input device found")
            
            if self.output_device_index is None:
                try:
                    self.output_device_index = self.py_audio.get_default_output_device_info()['index']
                except IOError:
                    logger.warning("No default output device found")
            
            # Log selected devices
            if self.input_device_index is not None:
                input_device = self.py_audio.get_device_info_by_index(self.input_device_index)
                logger.info(f"Selected input device: {input_device['name']} (index: {self.input_device_index})")
            
            if self.output_device_index is not None:
                output_device = self.py_audio.get_device_info_by_index(self.output_device_index)
                logger.info(f"Selected output device: {output_device['name']} (index: {self.output_device_index})")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio: {str(e)}")
            raise AudioError(
                f"Failed to initialize audio: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
    
    def _log_available_devices(self) -> None:
        """Log information about available audio devices."""
        if not self.py_audio:
            return
        
        logger.info("Available audio devices:")
        
        # Input devices
        logger.info("Input devices:")
        input_devices = []
        for i in range(self.py_audio.get_device_count()):
            device_info = self.py_audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append(f"  {i}: {device_info['name']}")
        
        for device in input_devices:
            logger.info(device)
        
        # Output devices
        logger.info("Output devices:")
        output_devices = []
        for i in range(self.py_audio.get_device_count()):
            device_info = self.py_audio.get_device_info_by_index(i)
            if device_info['maxOutputChannels'] > 0:
                output_devices.append(f"  {i}: {device_info['name']}")
        
        for device in output_devices:
            logger.info(device)
    
    def list_audio_devices(self) -> List[Dict[str, Any]]:
        """
        List available audio input and output devices
        
        Returns:
            List of dictionaries with device information
        """
        devices = []
        for i in range(self.py_audio.get_device_count()):
            device_info = self.py_audio.get_device_info_by_index(i)
            devices.append({
                "index": i,
                "name": device_info.get("name"),
                "maxInputChannels": device_info.get("maxInputChannels"),
                "maxOutputChannels": device_info.get("maxOutputChannels"),
                "defaultSampleRate": device_info.get("defaultSampleRate")
            })
        return devices
    
    async def start_recording(self) -> bool:
        """
        Start recording audio from the microphone with VAD.
        
        This method starts a new input stream if one is not already active
        and sets up callbacks for audio processing.
        
        Returns:
            bool: True if recording started successfully
        
        Raises:
            AudioError: If recording cannot be started
        """
        if self.state == AudioState.RECORDING:
            logger.warning("Already recording")
            return True
        
        if not self.py_audio:
            self._initialize_audio()
        
        logger.info("Starting audio recording with VAD")
        
        try:
            # Reset state
            self.is_speech_active = False
            self.silence_frames = 0
            self.speech_frames.clear()
            self.speech_buffer = bytearray()
            self.is_input_paused = False
            
            if self.pending_pause_task:
                self.pending_pause_task.cancel()
                self.pending_pause_task = None
            
            # Create and start the input stream
            self.input_stream = self.py_audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=self._audio_callback
            )
            
            self.input_stream.start_stream()
            self.state = AudioState.RECORDING
            
            # Start the processing task if not already running
            if not hasattr(self, 'processing_task') or self.processing_task.done():
                self.processing_task = self.task_manager.create_task(
                    self._process_audio_queue(),
                    "audio_processing"
                )
            
            logger.info("Audio recording started")
            return True
            
        except Exception as e:
            self.state = AudioState.ERROR
            logger.error(f"Failed to start recording: {str(e)}")
            raise AudioError(
                f"Failed to start recording: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback for audio input stream.
        
        This is called by PyAudio in a separate thread.
        
        Args:
            in_data: Audio data from microphone
            frame_count: Number of frames
            time_info: Time information
            status: Status flag
            
        Returns:
            tuple: (None, pyaudio.paContinue)
        """
        if status:
            logger.warning(f"Audio input status: {status}")
        
        if not self.is_input_paused:
            # Process for VAD
            self._process_audio_for_vad(in_data)
            
            # Add to the thread-safe queue for the async task to process
            try:
                self._callback_queue.put_nowait(in_data)
            except queue.Full:
                # If queue is full, log warning and drop the chunk
                logger.warning("Input queue full, dropping audio chunk")
        
        return (None, pyaudio.paContinue)
    
    def _process_audio_for_vad(self, audio_data: bytes) -> None:
        """
        Process audio data for voice activity detection.
        
        This determines if the current audio frame contains speech
        and manages the speech state accordingly.
        
        Args:
            audio_data: Raw audio data from microphone
        """
        # Convert to numpy array for processing
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate energy for VAD
        energy = np.sqrt(np.mean(np.square(audio_array.astype(np.float32) / 32768.0)))
        
        # Keep track of recent frames for context
        self.speech_frames.append(audio_data)
        
        # Determine if this is speech
        is_speech = energy > self.vad_threshold
        
        # State machine for speech detection
        if not self.is_speech_active and is_speech:
            # Transition from silence to speech
            self._handle_speech_start()
        elif self.is_speech_active and is_speech:
            # Continue speech
            self._handle_speech_continue(audio_data)
        elif self.is_speech_active and not is_speech:
            # Possible end of speech, increment silence counter
            self.silence_frames += 1
            
            # Calculate silence duration
            silence_duration_ms = (self.silence_frames * self.frames_per_buffer / self.sample_rate) * 1000
            
            if silence_duration_ms >= self.silence_duration_ms:
                # Enough silence to end speech
                self._schedule_delayed_pause()
            else:
                # Still collecting speech
                self._handle_speech_continue(audio_data)
    
    def _handle_speech_start(self) -> None:
        """Handle transition from silence to speech."""
        # Mark speech as active
        self.is_speech_active = True
        self.speech_active = True  # For API client compatibility
        self.silence_frames = 0
        self.last_speech_start_time = time.time()
        
        # Initialize new speech buffer
        self.speech_buffer = bytearray()
        
        # Include prefix padding (previous audio) to catch the start of speech
        prefix_frames = int((self.prefix_padding_ms / 1000) * self.sample_rate / self.frames_per_buffer)
        for i in range(min(prefix_frames, len(self.speech_frames))):
            self.speech_buffer.extend(self.speech_frames[i])
        
        # Emit speech started event
        speech_event = UserSpeechEvent(
            type=EventType.USER_SPEECH_STARTED,
            audio_data=bytes(self.speech_buffer),
            is_final=False
        )
        event_bus.emit(speech_event)
        
        logger.debug("Speech started")
    
    def _handle_speech_continue(self, audio_data: bytes) -> None:
        """
        Handle continuation of speech.
        
        Args:
            audio_data: New audio data to add to buffer
        """
        # Reset silence frames
        self.silence_frames = 0
        
        # Add audio to buffer
        self.speech_buffer.extend(audio_data)
        
        # Cancel any pending pause
        if self.pending_pause_task:
            self.pending_pause_task.cancel()
            self.pending_pause_task = None
        
        # Emit speech ongoing event
        speech_event = UserSpeechEvent(
            type=EventType.USER_SPEECH_ONGOING,
            audio_data=audio_data,
            is_final=False
        )
        event_bus.emit(speech_event)
    
    def _handle_speech_end(self) -> None:
        """Handle end of speech detection."""
        if not self.is_speech_active:
            return
        
        # Calculate speech duration
        duration = len(self.speech_buffer) / (self.sample_width * self.channels * self.sample_rate)
        
        # Only emit if speech is long enough to be valid
        if duration >= 0.5:  # Minimum 500ms
            logger.debug(f"Speech ended (duration: {duration:.2f}s)")
            
            # Create normalized PCM data
            try:
                audio_data = convert_to_pcm16(
                    self.speech_buffer,
                    source_sample_rate=self.sample_rate,
                    target_sample_rate=24000  # OpenAI requires 24kHz
                )
                
                # Emit speech finished event
                speech_event = UserSpeechEvent(
                    type=EventType.USER_SPEECH_FINISHED,
                    audio_data=audio_data,
                    duration=duration,
                    is_final=True
                )
                event_bus.emit(speech_event)
                
            except Exception as e:
                logger.error(f"Error processing speech: {str(e)}")
        else:
            logger.debug(f"Discarded speech (too short: {duration:.2f}s)")
            
            # Emit speech cancelled event for short utterances
            speech_event = UserSpeechEvent(
                type=EventType.USER_SPEECH_CANCELLED,
                duration=duration,
                is_final=True
            )
            event_bus.emit(speech_event)
        
        # Reset state
        self.is_speech_active = False
        self.speech_active = False  # For API client compatibility
        self.silence_frames = 0
        self.speech_buffer = bytearray()
    
    def _schedule_delayed_pause(self) -> None:
        """
        Schedule a delayed pause to ensure we have truly detected the end of speech.
        
        This helps prevent false ends when there are natural pauses in speech.
        """
        if self.pending_pause_task:
            self.pending_pause_task.cancel()
        
        self.pending_pause_task = asyncio.create_task(self._delayed_pause())
    
    async def _delayed_pause(self) -> None:
        """
        Wait a short period to confirm the end of speech.
        
        This coroutine runs after the VAD detects potential end of speech,
        but waits to confirm there's no immediate continuation.
        """
        await asyncio.sleep(0.3)  # Wait 300ms to be sure
        
        # Check speech duration
        speech_duration = len(self.speech_buffer) / (self.sample_width * self.channels * self.sample_rate)
        
        # If very short and no new speech, just cancel
        if speech_duration < 0.5 and self.is_speech_active:
            logger.debug(f"Cancelling pending pause, speech too short ({speech_duration:.2f}s)")
            self._handle_speech_end()
            return
        
        # Otherwise, end the speech if still active
        if self.is_speech_active:
            self._handle_speech_end()
    
    async def _process_audio_queue(self) -> None:
        """Process audio chunks from the input queue and emit audio events."""
        try:
            logger.debug("Started audio processing task")
            
            connection_error_count = 0
            max_connection_errors = 5
            
            while self.state == AudioState.RECORDING:
                # Check the PyAudio callback queue for data
                while not self._callback_queue.empty() and not self.is_input_paused:
                    try:
                        # Get data from the thread-safe queue
                        audio_chunk = self._callback_queue.get_nowait()
                        
                        # Add to the async queue for processing
                        try:
                            await self.input_queue.put(audio_chunk)
                        except asyncio.QueueFull:
                            logger.warning("Async input queue full, dropping audio chunk")
                    except queue.Empty:
                        # Queue became empty between our check and get
                        break
                
                # Process chunks from the async queue
                while not self.input_queue.empty() and not self.is_input_paused:
                    # Get audio chunk from queue
                    audio_chunk = await self.input_queue.get()
                    
                    # Emit audio chunk event
                    # Note: We don't directly send to API here in service layer
                    # Domain layer will handle that based on events
                    event = UserSpeechEvent(
                        type=EventType.USER_SPEECH_ONGOING,
                        audio_data=audio_chunk,
                        is_final=False
                    )
                    event_bus.emit(event)
                    
                    # Mark task as done
                    self.input_queue.task_done()
                
                # Short sleep to prevent busy waiting
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            logger.debug("Audio processing task cancelled")
        except Exception as e:
            logger.error(f"Error in audio processing task: {e}")
    
    async def stop_recording(self) -> bool:
        """
        Stop recording audio from the microphone.
        
        Returns:
            bool: True if recording was stopped successfully
        """
        if self.state != AudioState.RECORDING:
            logger.warning("Not currently recording")
            return True
        
        logger.info("Stopping audio recording")
        
        # Finish any pending speech
        if self.is_speech_active:
            self._handle_speech_end()
        
        # Stop the input stream
        if self.input_stream:
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
                self.input_stream = None
            except Exception as e:
                logger.error(f"Error closing input stream: {str(e)}")
        
        # Update state
        self.state = AudioState.IDLE
        
        # Cancel processing task if it exists
        if hasattr(self, 'processing_task') and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None
        
        logger.info("Audio recording stopped")
        return True
    
    async def play_audio(self, audio_data: bytes) -> bool:
        """
        Play audio data through the speakers.
        
        Args:
            audio_data: Audio data to play (16-bit PCM, 24kHz, mono)
            
        Returns:
            bool: True if audio was added to playback queue
        
        Raises:
            AudioError: If playback fails
        """
        # Ensure audio format is compatible
        try:
            # Check if we need to convert from the OpenAI format (24kHz)
            converted_audio = convert_to_pcm16(
                audio_data,
                source_sample_rate=24000,  # OpenAI sends 24kHz
                target_sample_rate=self.sample_rate
            )
            
            # Add to playback buffer
            self.playback_buffer.append(converted_audio)
            await self.output_queue.put(converted_audio)
            
            # Start playback if not already playing
            if not self.is_playing:
                await self._ensure_playback_running()
                
            return True
                
        except Exception as e:
            logger.error(f"Error preparing audio for playback: {str(e)}")
            raise AudioError(
                f"Failed to play audio: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
    
    async def _ensure_playback_running(self) -> None:
        """Ensure playback is running if there's audio to play"""
        if not self.is_playing and not self.output_queue.empty():
            await self.start_playback()
    
    async def start_playback(self) -> bool:
        """
        Start playing audio output.
        
        Returns:
            bool: True if playback started successfully
        """
        if self.is_playing:
            return True
            
        try:
            # Open audio output stream
            self.output_stream = self.py_audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=self.frames_per_buffer
            )
            
            self.is_playing = True
            self.should_stop_playback = False
            self.state = AudioState.PLAYING
            logger.info("Started audio playback")
            
            # Start the playback task
            self.playing_task = self.task_manager.create_task(
                self._play_audio_queue(),
                "audio_playback"
            )
            
            return True
                
        except Exception as e:
            logger.error(f"Error starting audio playback: {e}")
            self.is_playing = False
            self.state = AudioState.ERROR
            if self.output_stream:
                try:
                    self.output_stream.close()
                except:
                    pass
                self.output_stream = None
            return False
    
    async def _play_audio_queue(self) -> None:
        """
        Worker coroutine for playing audio from the buffer.
        
        This coroutine runs until the playback buffer is empty or playback is stopped.
        """
        try:
            logger.debug("Started audio playback task")
            
            # Counter for empty queue checks
            empty_checks = 0
            # Counter for audio errors
            consecutive_errors = 0
            max_consecutive_errors = 3
            
            while self.is_playing:
                # Check if there's data in the queue
                try:
                    # Wait for up to 0.5 seconds for data
                    audio_chunk = await asyncio.wait_for(self.output_queue.get(), 0.5)
                    empty_checks = 0  # Reset counter when we get data
                    consecutive_errors = 0  # Reset error counter when we successfully get data
                    
                    # Play the audio
                    if self.output_stream and self.is_playing and not self.should_stop_playback:
                        try:
                            # Only write to stream if it's active (not paused)
                            if self.output_stream.is_active():
                                self.output_stream.write(audio_chunk)
                            else:
                                logger.debug("Skipping audio chunk - stream is paused")
                        except Exception as e:
                            consecutive_errors += 1
                            logger.error(f"Error writing to audio output: {e}")
                            
                            # If we get multiple consecutive errors, restart the output stream
                            if consecutive_errors >= max_consecutive_errors:
                                logger.warning(f"Too many consecutive audio errors ({consecutive_errors}), restarting audio output")
                                await self.stop_playback()
                                await asyncio.sleep(0.2)  # Short delay before restarting
                                await self.start_playback()
                                consecutive_errors = 0
                    
                    # Mark task as done
                    self.output_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No data received within timeout
                    empty_checks += 1
                    
                    # If queue has been empty for a while, exit the loop
                    if empty_checks > 10:  # 5 seconds of empty checks
                        empty_checks = 0
                        if not self.is_playing or self.playback_buffer:
                            break
                except Exception as e:
                    logger.error(f"Error in audio playback loop: {e}")
                    await asyncio.sleep(0.1)  # Avoid tight loop on errors
                
            logger.debug("Audio playback task completed")
            
        except asyncio.CancelledError:
            logger.debug("Audio playback task cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in playback task: {e}")
        finally:
            # If no more playback and not explicitly stopped, update state
            if not self.should_stop_playback and not self.playback_buffer:
                self.state = AudioState.IDLE
                self.is_playing = False
    
    async def stop_playback(self) -> bool:
        """
        Stop audio playback and clear the buffer.
        
        Returns:
            bool: True if playback was stopped successfully
        """
        if not self.is_playing:
            return True
            
        logger.info("Stopping audio playback")
        
        # Stop current playback
        self.is_playing = False
        self.should_stop_playback = True
        
        # Clear buffers
        self.playback_buffer.clear()
        
        # Clear the output queue
        try:
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
                self.output_queue.task_done()
        except Exception:
            pass
        
        # Close output stream if open
        if self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.output_stream = None
            except Exception as e:
                logger.error(f"Error closing output stream: {str(e)}")
        
        # Cancel playback task if running
        if hasattr(self, 'playing_task') and not self.playing_task.done():
            self.playing_task.cancel()
            try:
                await self.playing_task
            except asyncio.CancelledError:
                pass
            self.playing_task = None
        
        # Update state
        self.state = AudioState.IDLE
        
        logger.info("Audio playback stopped")
        return True
    
    async def pause_playback(self) -> bool:
        """
        Pause audio playback.
        
        Returns:
            bool: True if playback was paused successfully
        """
        if not self.is_playing or not self.output_stream:
            logger.warning("No active playback to pause")
            return False
            
        try:
            # Set the flag to pause after current chunk
            self.should_stop_playback = True
            
            # Just pause the stream but keep it open
            if self.output_stream and self.output_stream.is_active():
                self.output_stream.stop_stream()
                self.state = AudioState.PAUSED
                logger.info("Audio playback paused")
                return True
            return False
        except Exception as e:
            logger.error(f"Error pausing output stream: {e}")
            # If pausing fails, stop completely
            await self.stop_playback()
            return False
    
    async def resume_playback(self) -> bool:
        """
        Resume paused audio playback.
        
        Returns:
            bool: True if playback was resumed successfully
        """
        if self.state != AudioState.PAUSED:
            logger.warning("No paused playback to resume")
            return False
            
        try:
            # Reset the stop flag
            self.should_stop_playback = False
            
            # Check if the stream exists and is not active
            if self.output_stream and not self.output_stream.is_active():
                try:
                    # Start the stream
                    self.output_stream.start_stream()
                    self.state = AudioState.PLAYING
                    self.is_playing = True
                    logger.info("Audio playback resumed")
                    
                    # Restart the playback task if needed
                    if not hasattr(self, 'playing_task') or self.playing_task.done():
                        self.playing_task = self.task_manager.create_task(
                            self._play_audio_queue(),
                            "audio_playback_resumed"
                        )
                    
                    return True
                except Exception as stream_error:
                    # If direct restart fails, recreate the stream
                    logger.info(f"Could not restart stream: {stream_error}, recreating...")
                    await self.stop_playback()
                    await asyncio.sleep(0.1)  # Brief delay to ensure cleanup
                    return await self.start_playback()
            else:
                # Stream is already active or missing
                logger.info("Stream already active or missing, creating new playback")
                return await self.start_playback()
        except Exception as e:
            logger.error(f"Error resuming output stream: {e}")
            # If resuming fails, try to start fresh with a delay to avoid rapid cycling
            await self.stop_playback()
            await asyncio.sleep(0.2)  # Slightly longer delay before restart
            return await self.start_playback()
    
    async def pause_input(self, duration: Optional[float] = None) -> None:
        """
        Temporarily pause audio input processing
        
        Args:
            duration: Duration in seconds to pause, or None for indefinite
        """
        self.is_input_paused = True
        logger.info(f"Audio input paused for {duration if duration else 'indefinite'} seconds")
        
        if duration:
            # Create a task to unpause after the specified duration
            async def unpause_after_delay():
                await asyncio.sleep(duration)
                await self.resume_input()
            
            self.task_manager.create_task(unpause_after_delay(), "unpause_delay")
    
    async def resume_input(self) -> None:
        """Resume audio input processing if paused"""
        self.is_input_paused = False
        logger.info("Audio input resumed")
    
    async def send_audio_file(self, file_path: str) -> bool:
        """
        Process audio from a file and emit it as speech events.
        
        Args:
            file_path: Path to the audio file (WAV)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read the WAV file
            audio_data, sample_rate, channels, sample_width = read_wav_file(file_path)
            
            # Convert to required format if needed
            if sample_rate != 24000 or channels != 1:
                logger.info(f"Converting audio from {sample_rate}Hz {channels}ch to 24000Hz mono")
                audio_data = convert_to_pcm16(
                    audio_data,
                    source_sample_rate=sample_rate,
                    target_sample_rate=24000
                )
            
            # Emit speech started event
            start_event = UserSpeechEvent(
                type=EventType.USER_SPEECH_STARTED,
                audio_data=b"",  # Empty for start event
                is_final=False
            )
            event_bus.emit(start_event)
            
            # Split into chunks for processing
            chunk_size = 32768  # 32KB chunks
            chunks = split_audio_chunks(
                audio_data, 
                chunk_size=chunk_size,
                sample_rate=24000,
                channels=1, 
                sample_width=2
            )
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                is_final = (i == len(chunks) - 1)
                
                # Emit ongoing speech event
                chunk_event = UserSpeechEvent(
                    type=EventType.USER_SPEECH_ONGOING if not is_final else EventType.USER_SPEECH_FINISHED,
                    audio_data=chunk,
                    is_final=is_final
                )
                event_bus.emit(chunk_event)
                
                # Small delay to prevent flooding
                await asyncio.sleep(0.01)
            
            return True
                
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            return False
    
    async def save_output_to_file(self, file_path: Union[str, Path], duration: Optional[float] = None) -> bool:
        """
        Save audio output to a WAV file.
        
        Args:
            file_path: Path where to save the WAV file
            duration: Optional recording duration in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create a buffer to store audio data
        audio_buffer = []
        
        # Set up a queue for thread-safe access
        file_queue = asyncio.Queue(maxsize=self.max_queue_size)
        
        # Define a callback to capture audio events
        def capture_audio_event(event):
            if event.type == EventType.AUDIO_SPEECH_CREATED and event.chunk:
                asyncio.create_task(file_queue.put(event.chunk))
        
        # Register with event bus
        event_bus.on(EventType.AUDIO_SPEECH_CREATED, capture_audio_event)
        
        try:
            # If duration is provided, record for that time
            start_time = time.time()
            last_audio_time = start_time
            
            while duration is None or (time.time() - start_time) < duration:
                try:
                    # Use a timeout to prevent blocking forever
                    audio_chunk = await asyncio.wait_for(file_queue.get(), 0.5)
                    audio_buffer.append(audio_chunk)
                    file_queue.task_done()
                    last_audio_time = time.time()
                except asyncio.TimeoutError:
                    # No new audio in the last 0.5 seconds
                    # If we've gone 3 seconds without audio and not recording a fixed duration,
                    # assume we're done
                    if duration is None and (time.time() - last_audio_time) > 3.0:
                        break
            
            # If no audio was captured, return early
            if not audio_buffer:
                logger.warning("No audio captured to save to file")
                return False
            
            # Combine audio chunks
            combined_audio = b''.join(audio_buffer)
            
            # Write the captured audio to a WAV file
            write_wav_file(
                file_path=file_path,
                audio_data=combined_audio,
                sample_rate=24000,  # OpenAI format
                channels=1,
                sample_width=2
            )
                
            logger.info(f"Saved audio output to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving audio to file: {e}")
            return False
        finally:
            # Clean up the event handler
            event_bus.off(EventType.AUDIO_SPEECH_CREATED, capture_audio_event)
    
    def is_speech_active(self) -> bool:
        """
        Check if speech is currently active.
        
        Returns:
            bool: True if speech is active
        """
        return self.speech_active
    
    async def cleanup(self) -> None:
        """
        Clean up all resources used by the audio service.
        
        This method should be called when shutting down.
        """
        logger.info("Cleaning up audio resources")
        
        # Stop recording and playback
        await self.stop_recording()
        await self.stop_playback()
        
        # Cancel all tasks
        await self.task_manager.cancel_all()
        
        # Terminate PyAudio
        if self.py_audio:
            try:
                self.py_audio.terminate()
                self.py_audio = None
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {str(e)}")
        
        logger.info("Audio resources cleaned up")
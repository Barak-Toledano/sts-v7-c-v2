"""
Audio manager for the OpenAI Realtime Assistant.

This module orchestrates audio recording and playback, managing
voice activity detection, audio format conversion, and event handling.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union, Any

from src.config import settings
from src.config.logging_config import get_logger
from src.events.event_interface import (
    AudioSpeechEvent,
    EventType,
    UserSpeechEvent,
    event_bus,
)
from src.services.audio_service import AudioService, AudioState
from src.utils.async_helpers import TaskManager
from src.utils.error_handling import AppError, AudioError, ErrorSeverity, safe_execute

logger = get_logger(__name__)


class AudioMode(Enum):
    """
    Audio processing modes.
    
    These modes determine how audio input and output are handled.
    """
    
    CONVERSATION = auto()  # Standard conversation mode with VAD
    DICTATION = auto()     # Extended recording without automatic breaks
    PLAYBACK_ONLY = auto() # No recording, only playback
    OFF = auto()           # Audio subsystem disabled


class AudioManager:
    """
    Manager for coordinating audio recording and playback.
    
    This class is responsible for:
    - Managing the audio service and device selection
    - Coordinating voice activity detection
    - Handling audio events and callbacks
    - Managing audio file saving and loading
    """
    
    def __init__(
        self,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        mode: AudioMode = AudioMode.CONVERSATION,
        save_recordings: bool = False,
        auto_start: bool = False,
    ):
        """
        Initialize the audio manager.
        
        Args:
            input_device: Optional input device index
            output_device: Optional output device index
            mode: Audio processing mode
            save_recordings: Whether to save recordings to disk
            auto_start: Whether to automatically start recording on initialization
        """
        # Initialize audio service
        self.audio_service = AudioService(
            input_device_index=input_device,
            output_device_index=output_device
        )
        
        # Configuration
        self.mode = mode
        self.save_recordings = save_recordings
        self.recordings_dir = settings.data_dir / "recordings"
        self.task_manager = TaskManager()
        
        # State tracking
        self.is_recording = False
        self.is_playing = False
        
        # Event callbacks
        self.on_speech_start: Optional[Callable[[UserSpeechEvent], None]] = None
        self.on_speech_end: Optional[Callable[[UserSpeechEvent], None]] = None
        
        # Ensure recordings directory exists
        if self.save_recordings:
            os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Register event handlers
        self._register_event_handlers()
        
        # Auto-start if requested
        if auto_start and self.mode != AudioMode.OFF and self.mode != AudioMode.PLAYBACK_ONLY:
            self.task_manager.create_task(self.start_recording(), "auto_start_recording")
    
    async def start_recording(self) -> bool:
        """
        Start audio recording.
        
        Returns:
            bool: True if recording was started successfully
        """
        if self.is_recording:
            logger.warning("Recording is already active")
            return True
        
        if self.mode == AudioMode.OFF or self.mode == AudioMode.PLAYBACK_ONLY:
            logger.warning("Cannot start recording in current mode")
            return False
        
        try:
            # Start recording via audio service
            success = await self.audio_service.start_recording()
            if success:
                self.is_recording = True
                logger.info("Audio recording started")
            return success
        except Exception as e:
            error = AppError(
                f"Failed to start recording: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
            return False
    
    async def stop_recording(self) -> bool:
        """
        Stop audio recording.
        
        Returns:
            bool: True if recording was stopped successfully
        """
        if not self.is_recording:
            logger.warning("Recording is not active")
            return True
        
        try:
            # Stop recording via audio service
            success = await self.audio_service.stop_recording()
            if success:
                self.is_recording = False
                logger.info("Audio recording stopped")
            return success
        except Exception as e:
            error = AppError(
                f"Failed to stop recording: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
            return False
    
    async def play_audio(self, audio_data: bytes) -> bool:
        """
        Play audio data.
        
        Args:
            audio_data: Audio data to play
            
        Returns:
            bool: True if playback was started successfully
        """
        try:
            # Play audio via audio service
            success = await self.audio_service.play_audio(audio_data)
            if success:
                self.is_playing = True
            return success
        except Exception as e:
            error = AppError(
                f"Failed to play audio: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
            return False
    
    async def stop_playback(self) -> bool:
        """
        Stop audio playback and clear the buffer.
        
        Returns:
            bool: True if playback was stopped successfully
        """
        try:
            # Stop playback via audio service
            success = await self.audio_service.stop_playback()
            if success:
                self.is_playing = False
            return success
        except Exception as e:
            error = AppError(
                f"Failed to stop playback: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
            return False
    
    async def pause_playback(self) -> bool:
        """
        Pause audio playback.
        
        Returns:
            bool: True if playback was paused successfully
        """
        try:
            return await self.audio_service.pause_playback()
        except Exception as e:
            error = AppError(
                f"Failed to pause playback: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
            return False
    
    async def resume_playback(self) -> bool:
        """
        Resume audio playback.
        
        Returns:
            bool: True if playback was resumed successfully
        """
        try:
            success = await self.audio_service.resume_playback()
            if success:
                self.is_playing = True
            return success
        except Exception as e:
            error = AppError(
                f"Failed to resume playback: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
            return False
    
    async def toggle_recording(self) -> bool:
        """
        Toggle audio recording state.
        
        Returns:
            bool: True if operation was successful
        """
        if self.is_recording:
            return await self.stop_recording()
        else:
            return await self.start_recording()
    
    def set_mode(self, mode: AudioMode) -> None:
        """
        Set the audio processing mode.
        
        Args:
            mode: New audio mode
        """
        if mode == self.mode:
            return
        
        prev_mode = self.mode
        self.mode = mode
        
        logger.info(f"Audio mode changed from {prev_mode.name} to {mode.name}")
        
        # Handle mode transitions
        if mode == AudioMode.OFF or mode == AudioMode.PLAYBACK_ONLY:
            if self.is_recording:
                self.task_manager.create_task(self.stop_recording(), "stop_recording_mode_change")
        elif prev_mode == AudioMode.OFF or prev_mode == AudioMode.PLAYBACK_ONLY:
            if not self.is_recording:
                self.task_manager.create_task(self.start_recording(), "start_recording_mode_change")
    
    async def save_recording(self, audio_data: bytes, filename: Optional[str] = None) -> Optional[Path]:
        """
        Save audio data to a WAV file.
        
        Args:
            audio_data: Audio data to save
            filename: Optional filename (defaults to timestamp-based name)
            
        Returns:
            Optional[Path]: Path to the saved file, or None if saving failed
        """
        if not self.save_recordings:
            logger.debug("Recording saving is disabled")
            return None
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            filename = f"recording_{timestamp}_{unique_id}.wav"
        
        # Ensure filename has .wav extension
        if not filename.endswith(".wav"):
            filename += ".wav"
        
        # Full path
        file_path = self.recordings_dir / filename
        
        try:
            from src.utils.audio_utilities import write_wav_file
            
            # Write WAV file
            write_wav_file(
                file_path=file_path,
                audio_data=audio_data,
                sample_rate=24000,
                channels=1,
                sample_width=2
            )
            
            logger.info(f"Saved recording to {file_path}")
            return file_path
        except Exception as e:
            error = AppError(
                f"Failed to save recording: {str(e)}",
                severity=ErrorSeverity.WARNING,
                cause=e
            )
            logger.error(str(error))
            return None
    
    async def load_recording(self, file_path: Union[str, Path]) -> Optional[bytes]:
        """
        Load audio data from a WAV file.
        
        Args:
            file_path: Path to the WAV file
            
        Returns:
            Optional[bytes]: Audio data, or None if loading failed
        """
        try:
            # Use audio service to process the file
            from src.utils.audio_utilities import read_wav_file, convert_to_pcm16
            
            # Read the WAV file
            audio_data, sample_rate, channels, sample_width = read_wav_file(file_path)
            
            # Convert to OpenAI format if needed
            if sample_rate != 24000 or channels != 1 or sample_width != 2:
                audio_data = convert_to_pcm16(
                    audio_data=audio_data,
                    source_sample_rate=sample_rate,
                    target_sample_rate=24000
                )
            
            logger.info(f"Loaded recording from {file_path}")
            return audio_data
        except Exception as e:
            error = AppError(
                f"Failed to load recording: {str(e)}",
                severity=ErrorSeverity.WARNING,
                cause=e
            )
            logger.error(str(error))
            return None
    
    async def send_audio_file(self, file_path: str) -> bool:
        """
        Process audio from a file and emit it as speech events.
        
        Args:
            file_path: Path to the audio file (WAV)
            
        Returns:
            bool: True if successful, False otherwise
        """
        return await self.audio_service.send_audio_file(file_path)
    
    async def save_output_to_file(self, file_path: Union[str, Path], duration: Optional[float] = None) -> bool:
        """
        Save audio output to a WAV file.
        
        Args:
            file_path: Path where to save the WAV file
            duration: Optional recording duration in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        return await self.audio_service.save_output_to_file(file_path, duration)
    
    def _register_event_handlers(self) -> None:
        """Register event handlers."""
        # Register handlers for speech events
        event_bus.on(EventType.USER_SPEECH_STARTED, self._handle_speech_started)
        event_bus.on(EventType.USER_SPEECH_FINISHED, self._handle_speech_finished)
        event_bus.on(EventType.AUDIO_SPEECH_CREATED, self._handle_assistant_speech)
    
    def _handle_speech_started(self, event: UserSpeechEvent) -> None:
        """
        Handle speech started event.
        
        Args:
            event: Speech started event
        """
        logger.debug("Speech started event received in manager")
        
        # Call user callback if set
        if self.on_speech_start:
            try:
                self.on_speech_start(event)
            except Exception as e:
                logger.error(f"Error in speech start callback: {str(e)}")
    
    def _handle_speech_finished(self, event: UserSpeechEvent) -> None:
        """
        Handle speech finished event.
        
        Args:
            event: Speech finished event
        """
        logger.debug(f"Speech finished event received in manager (duration: {event.duration:.2f}s)")
        
        # Save recording if enabled
        if self.save_recordings and event.audio_data:
            self.task_manager.create_task(
                self.save_recording(event.audio_data),
                "save_recording"
            )
        
        # Call user callback if set
        if self.on_speech_end:
            try:
                self.on_speech_end(event)
            except Exception as e:
                logger.error(f"Error in speech end callback: {str(e)}")
    
    def _handle_assistant_speech(self, event: AudioSpeechEvent) -> None:
        """
        Handle assistant speech event.
        
        Args:
            event: Assistant speech event
        """
        # Currently this just logs the event
        # The actual playback is handled by the conversation manager
        if not self.is_playing and event.chunk:
            self.is_playing = True
            logger.debug("Assistant speech received")
    
    async def cleanup(self) -> None:
        """
        Clean up resources used by the audio manager.
        
        This method should be called when shutting down.
        """
        logger.info("Cleaning up audio manager resources")
        
        # Clean up audio service
        await self.audio_service.cleanup()
        
        # Cancel all tasks
        await self.task_manager.cancel_all()
        
        logger.info("Audio manager resources cleaned up")
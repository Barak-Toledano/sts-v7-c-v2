"""
Transcription service for the OpenAI Realtime Assistant.

This service integrates with the OpenAI Realtime API to handle transcription events
and process the resulting transcriptions.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from src.config import settings
from src.config.logging_config import get_logger
from src.events.event_interface import (
    Event, 
    EventType, 
    TranscriptionEvent, 
    event_bus
)
from src.utils.async_helpers import TaskManager
from src.utils.error_handling import AppError, ErrorSeverity
from src.utils.transcription import (
    extract_transcription_from_realtime_event,
    format_transcription,
    save_transcription,
    generate_realtime_session_config
)

logger = get_logger(__name__)


class TranscriptionService:
    """
    Service for managing transcription with the OpenAI Realtime API.
    
    This service:
    - Configures the API's transcription settings
    - Processes transcription events
    - Emits appropriate application events
    - Provides utilities for working with transcriptions
    """
    
    def __init__(self):
        """Initialize the transcription service."""
        self.task_manager = TaskManager("transcription")
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the service and set up event handlers."""
        # Register for API transcription events via the event bus
        event_bus.on(
            "conversation.item.input_audio_transcription.completed", 
            self._handle_transcription_completed
        )
        event_bus.on(
            "conversation.item.input_audio_transcription.failed", 
            self._handle_transcription_failed
        )
        
        logger.debug("Transcription service initialized")
    
    async def _handle_transcription_completed(self, api_event: Dict[str, Any]) -> None:
        """
        Handle transcription completed events from the API.
        
        Args:
            api_event: The raw API event
        """
        try:
            # Extract transcription data from the event
            transcription_data = extract_transcription_from_realtime_event(api_event)
            
            if not transcription_data["text"]:
                logger.warning("Received empty transcription from API")
                return
            
            # Create a TranscriptionEvent for our application
            event = TranscriptionEvent(
                type=EventType.USER_TRANSCRIPTION_COMPLETED,
                data={
                    "text": transcription_data["text"],
                    "is_final": transcription_data["is_final"],
                    "language": transcription_data["language"],
                    "confidence": transcription_data.get("confidence", 1.0),
                    "timestamp": transcription_data["timestamp"]
                },
                text=transcription_data["text"],
                is_final=transcription_data["is_final"],
                confidence=transcription_data.get("confidence", 1.0),
                source="whisper"
            )
            
            # Emit the event to our application's event bus
            event_bus.emit(event)
            
            logger.info(f"Transcription: '{transcription_data['text']}'")
            
            # Save transcription if configured to do so
            if settings.audio.save_transcriptions:
                self.task_manager.create_task(
                    self._save_transcription_to_file(transcription_data),
                    "save_transcription"
                )
                
        except Exception as e:
            error = AppError(
                f"Failed to handle transcription event: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
    
    async def _handle_transcription_failed(self, api_event: Dict[str, Any]) -> None:
        """
        Handle transcription failed events from the API.
        
        Args:
            api_event: The raw API event
        """
        try:
            error = api_event.get("error", {})
            error_message = error.get("message", "Unknown error")
            error_type = error.get("type", "unknown")
            
            logger.error(f"Transcription failed: {error_type} - {error_message}")
            
            # Create and emit an error event
            event_bus.emit(
                EventType.ERROR,
                {
                    "error": {
                        "message": f"Transcription failed: {error_message}",
                        "type": "transcription_error",
                        "source": "whisper"
                    }
                }
            )
            
        except Exception as e:
            error = AppError(
                f"Failed to handle transcription failure event: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
    
    async def _save_transcription_to_file(self, transcription_data: Dict[str, Any]) -> None:
        """
        Save a transcription to a file.
        
        Args:
            transcription_data: The transcription data
        """
        if not settings.audio.save_transcriptions:
            return
            
        text = transcription_data["text"]
        
        # Generate a filename based on timestamp
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"transcription_{timestamp}.json"
        file_path = settings.data_dir / "transcriptions" / filename
        
        # Save with metadata
        save_transcription(
            text=text,
            file_path=file_path,
            metadata={
                "language": transcription_data["language"],
                "is_final": transcription_data["is_final"],
                "confidence": transcription_data.get("confidence", 1.0)
            }
        )
    
    def get_session_config(self, **kwargs) -> Dict[str, Any]:
        """
        Get configuration for enabling transcription in a session.
        
        This is a convenience wrapper around generate_realtime_session_config()
        that uses settings from the application configuration by default.
        
        Args:
            **kwargs: Override default settings
            
        Returns:
            Dictionary with session configuration
        """
        # Get default settings from application config
        config_args = {
            "model": kwargs.get("model", settings.audio.transcription_model),
            "vad_enabled": kwargs.get("vad_enabled", settings.audio.vad_enabled),
            "auto_response": kwargs.get("auto_response", settings.audio.auto_response),
            "vad_threshold": kwargs.get("vad_threshold", settings.audio.vad_threshold),
            "silence_duration_ms": kwargs.get("silence_duration_ms", settings.audio.silence_duration_ms),
            "prefix_padding_ms": kwargs.get("prefix_padding_ms", settings.audio.prefix_padding_ms)
        }
        
        return generate_realtime_session_config(**config_args)
    
    async def cleanup(self) -> None:
        """Clean up resources used by the service."""
        # Unregister event handlers
        event_bus.off(
            "conversation.item.input_audio_transcription.completed", 
            self._handle_transcription_completed
        )
        event_bus.off(
            "conversation.item.input_audio_transcription.failed", 
            self._handle_transcription_failed
        )
        
        # Cancel any pending tasks
        await self.task_manager.cancel_all()
        
        logger.debug("Transcription service cleaned up")
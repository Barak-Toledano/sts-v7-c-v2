"""
Utility modules for the OpenAI Realtime Assistant.

This package contains various utility modules for common functionality,
including error handling, async operations, audio processing, and more.
"""

from src.utils.error_handling import (
    ErrorSeverity,
    BaseError,
    ApiError,
    AudioError,
    ConfigError,
    TranscriptionError,
    safe_execute
)

from src.utils.async_helpers import (
    TaskManager,
    run_with_timeout,
    retry_async,
    debounce,
    throttle
)

from src.utils.audio_utilities import (
    convert_to_pcm16,
    detect_silence,
    get_audio_duration,
    validate_audio_format
)

from src.utils.transcription import (
    extract_transcription_from_realtime_event,
    format_transcription,
    save_transcription,
    load_transcription
)

__all__ = [
    # Error handling
    "ErrorSeverity",
    "BaseError",
    "ApiError",
    "AudioError",
    "ConfigError",
    "TranscriptionError",
    "safe_execute",
    
    # Async utilities
    "TaskManager",
    "run_with_timeout",
    "retry_async",
    "debounce",
    "throttle",
    
    # Audio utilities
    "convert_to_pcm16",
    "detect_silence",
    "get_audio_duration",
    "validate_audio_format",
    
    # Transcription utilities
    "extract_transcription_from_realtime_event",
    "format_transcription",
    "save_transcription",
    "load_transcription"
]
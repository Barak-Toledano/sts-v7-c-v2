"""
Application settings and configuration.

This module provides a centralized configuration management system using Pydantic.
It loads settings from environment variables, .env files, or falls back to defaults.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
LOG_DIR = Path(os.environ.get("LOG_DIR", ROOT_DIR / "logs"))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR / "data"))


class ApiSettings(BaseModel):
    """OpenAI API configuration settings."""
    
    api_key: str = Field(
        default=os.environ.get("OPENAI_API_KEY", ""),
        description="OpenAI API key for authentication"
    )
    
    model: str = Field(
        default=os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17"),
        description="OpenAI model identifier"
    )
    
    base_url: str = Field(
        default=os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
        description="OpenAI API base URL"
    )
    
    voice: str = Field(
        default=os.environ.get("VOICE", "alloy"),
        description="Voice to use for audio responses"
    )
    
    @validator("api_key")
    def api_key_must_not_be_empty(cls, v):
        """Validate that API key is provided."""
        if not v:
            print("WARNING: OpenAI API key is not set. Please set OPENAI_API_KEY environment variable.")
        return v
    
    @validator("voice")
    def voice_must_be_valid(cls, v):
        """Validate that voice is a valid option."""
        valid_voices = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
        if v not in valid_voices:
            print(f"WARNING: Invalid voice '{v}'. Using default 'alloy'.")
            return "alloy"
        return v


class AudioSettings(BaseModel):
    """Audio configuration settings."""
    
    format: str = Field(
        default=os.environ.get("AUDIO_FORMAT", "pcm16"),
        description="Audio format (e.g., 'pcm16')"
    )
    
    sample_rate: int = Field(
        default=int(os.environ.get("AUDIO_SAMPLE_RATE", "24000")),
        description="Audio sample rate in Hz (required by OpenAI)"
    )
    
    channels: int = Field(
        default=int(os.environ.get("AUDIO_CHANNELS", "1")),
        description="Number of audio channels (1 for mono, 2 for stereo)"
    )
    
    chunk_size: int = Field(
        default=int(os.environ.get("AUDIO_CHUNK_SIZE", "4096")),
        description="Audio chunk size in bytes"
    )
    
    sample_width: int = Field(
        default=2,  # 16-bit audio = 2 bytes
        description="Sample width in bytes"
    )
    
    frames_per_buffer: int = Field(
        default=int(os.environ.get("AUDIO_CHUNK_SIZE", "4096")),
        description="Frames per buffer for PyAudio"
    )
    
    input_device: Optional[int] = Field(
        default=None,
        description="Index of audio input device (None for system default)"
    )
    
    output_device: Optional[int] = Field(
        default=None,
        description="Index of audio output device (None for system default)"
    )
    
    # Voice Activity Detection settings
    vad_threshold: float = Field(
        default=float(os.environ.get("VAD_THRESHOLD", "0.85")),
        description="Voice activity detection threshold (0.0-1.0)"
    )
    
    silence_duration_ms: int = Field(
        default=int(os.environ.get("SILENCE_DURATION_MS", "1200")),
        description="Silence duration in ms to consider speech ended"
    )
    
    prefix_padding_ms: int = Field(
        default=int(os.environ.get("PREFIX_PADDING_MS", "500")),
        description="Padding before speech in ms"
    )

    # Transcription settings
    transcription_model: str = Field(
        default="whisper-1",
        description="Model to use for transcription"
    )
    
    save_transcriptions: bool = Field(
        default=False,
        description="Whether to save transcriptions to disk"
    )
    
    @validator("sample_rate")
    def validate_sample_rate(cls, v):
        """Validate that sample rate is valid for OpenAI."""
        if v != 24000:
            print(f"WARNING: Sample rate {v}Hz may not be compatible with OpenAI Realtime API (requires 24000Hz).")
        return v
    
    @validator("channels")
    def validate_channels(cls, v):
        """Validate that channels is valid for OpenAI."""
        if v != 1:
            print(f"WARNING: Channel count {v} may not be compatible with OpenAI Realtime API (requires mono).")
        return v
    
    @validator("vad_threshold")
    def validate_vad_threshold(cls, v):
        """Validate that VAD threshold is within valid range."""
        if not 0.0 <= v <= 1.0:
            print(f"WARNING: VAD threshold {v} is outside valid range (0.0-1.0). Using 0.85.")
            return 0.85
        return v


class LoggingSettings(BaseModel):
    """Logging configuration settings."""
    
    level: str = Field(
        default=os.environ.get("LOG_LEVEL", "INFO"),
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    console_enabled: bool = Field(
        default=os.environ.get("LOG_CONSOLE_ENABLED", "true").lower() == "true",
        description="Whether to log to console"
    )
    
    file_enabled: bool = Field(
        default=os.environ.get("LOG_FILE_ENABLED", "true").lower() == "true",
        description="Whether to log to file"
    )
    
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    detailed_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        description="Detailed log format string for file logging"
    )
    
    @validator("level")
    def validate_log_level(cls, v):
        """Validate that log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            print(f"WARNING: Invalid log level '{v}'. Using INFO.")
            return "INFO"
        return v.upper()


class Settings(BaseModel):
    """Main application settings."""
    
    # Application info
    app_name: str = Field(
        default="OpenAI Realtime Assistant",
        description="Application name"
    )
    
    app_version: str = Field(
        default="0.1.0",
        description="Application version"
    )
    
    # Sub-configurations
    api: ApiSettings = Field(default_factory=ApiSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # Paths
    root_dir: Path = ROOT_DIR
    src_dir: Path = SRC_DIR
    logs_dir: Path = LOG_DIR
    data_dir: Path = DATA_DIR
    
    # Runtime configs
    debug_mode: bool = Field(
        default=os.environ.get("DEBUG_MODE", "false").lower() == "true",
        description="Enable debug mode"
    )
    
    def __init__(self, **data: Any):
        """Initialize settings and create any required directories."""
        super().__init__(**data)
        self._create_required_directories()
    
    def _create_required_directories(self) -> None:
        """Create any required application directories."""
        # Logs directory (with session subdirectory)
        session_log_dir = self.logs_dir / "sessions"
        session_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Data directory with subdirectories
        recordings_dir = self.data_dir / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
    
    def get_session_log_path(self, session_id: str) -> Path:
        """Get path for session-specific log file."""
        return self.logs_dir / "sessions" / f"{session_id}.log"

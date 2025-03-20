"""
Tests for the configuration management system.

This module tests the loading, validation, and access to configuration
settings, ensuring proper handling of environment variables and sensitive data.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config.settings import Settings
from src.config import settings
from src.utils.error_handling import ConfigError


@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables for testing."""
    original_environ = os.environ.copy()
    
    # Set test environment variables
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    os.environ["OPENAI_REALTIME_MODEL"] = "test-model"
    os.environ["VOICE"] = "test_voice"
    os.environ["AUDIO_FORMAT"] = "pcm16"
    os.environ["SAMPLE_RATE"] = "24000"
    os.environ["DEBUG_MODE"] = "true"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_environ)


def test_required_api_settings_validation():
    """Test that missing API key is properly handled with clear error."""
    with patch.dict(os.environ, {}, clear=True):
        # API key is a required setting, so this should warn or error
        with pytest.warns() as warnings:
            api_settings = Settings().api
            
            # Should not have a valid API key
            assert not api_settings.api_key
            
            # Should have warned about missing API key
            assert any("API key" in str(w.message) for w in warnings)


def test_sensitive_info_masking():
    """Test that API keys and other sensitive data are masked in logs."""
    with patch("src.config.logging_config.get_logger") as mock_logger:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret_api_key"}):
            settings = Settings()
            
            # Convert settings to string
            settings_str = str(settings)
            
            # API key should not be in the string representation
            assert "secret_api_key" not in settings_str
            
            # Log the settings
            mock_logger.return_value.debug.side_effect = lambda x: None
            mock_logger.return_value.debug(f"Settings: {settings}")
            
            # Check that the API key is masked in logs
            for call in mock_logger.return_value.debug.call_args_list:
                assert "secret_api_key" not in str(call)


def test_configuration_immutability():
    """Test that configuration objects cannot be modified after creation."""
    # Create settings with test values
    settings = Settings()
    
    # Try to modify API key
    with pytest.raises(AttributeError):
        settings.api.api_key = "new_api_key"


def test_config_inheritance():
    """Test that configuration properly inherits from parent configs."""
    with patch.dict(os.environ, {
        "LOG_LEVEL": "DEBUG", 
        "OPENAI_REALTIME_MODEL": "custom-model"
    }):
        settings = Settings()
        
        # Check that settings are inherited from environment
        assert settings.logging.level == "DEBUG"
        assert settings.api.model == "custom-model"


def test_default_values():
    """Test that default values are used when environment variables are missing."""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
        
        # Check default values
        assert settings.audio.sample_rate == 24000
        assert settings.audio.channels == 1
        assert settings.logging.level == "INFO"


@pytest.mark.parametrize("env_value,expected", [
    ("true", True),
    ("True", True),
    ("yes", True),
    ("y", True),
    ("1", True),
    ("false", False),
    ("False", False),
    ("no", False),
    ("n", False),
    ("0", False),
])
def test_boolean_parsing(env_value, expected):
    """Test that boolean values are correctly parsed from environment variables."""
    with patch.dict(os.environ, {"DEBUG_MODE": env_value}):
        settings = Settings()
        assert settings.debug_mode is expected


def test_path_resolution():
    """Test that path settings are correctly resolved relative to app root."""
    settings = Settings()
    
    # Check that paths exist and are absolute
    assert settings.logs_dir.is_absolute()
    assert settings.data_dir.is_absolute()


def test_session_log_path():
    """Test that session log paths are generated correctly."""
    settings = Settings()
    
    # Generate a session log path
    session_id = "test_session_123"
    log_path = settings.get_session_log_path(session_id)
    
    # Check that the path is correct
    assert log_path.is_absolute()
    assert log_path.parent.name == "sessions"
    assert log_path.name == f"{session_id}.log"


def test_audio_settings_validation():
    """Test validation of audio settings."""
    with patch.dict(os.environ, {
        "AUDIO_SAMPLE_RATE": "invalid",
        "AUDIO_CHANNELS": "invalid",
    }):
        # This should not raise but should use defaults
        settings = Settings()
        
        # Should use defaults for invalid values
        assert settings.audio.sample_rate == 24000
        assert settings.audio.channels == 1


def test_custom_config_file(tmp_path):
    """Test loading settings from a custom config file."""
    # Create a temporary config file
    config_file = tmp_path / "test_config.json"
    config_file.write_text('{"logging": {"level": "DEBUG"}}')
    
    # Load settings with custom config file
    with patch("src.config.settings.Settings._load_config_file") as mock_load:
        mock_load.return_value = {"logging": {"level": "DEBUG"}}
        
        settings = Settings(config_file=str(config_file))
        
        # Check that settings were loaded from file
        assert settings.logging.level == "DEBUG"
        mock_load.assert_called_once_with(str(config_file))


def test_environment_variable_precedence():
    """Test that environment variables take precedence over config file values."""
    # Set up conflicting values
    with patch("src.config.settings.Settings._load_config_file") as mock_load:
        mock_load.return_value = {"logging": {"level": "DEBUG"}}
        
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
            settings = Settings(config_file="dummy_path")
            
            # Environment variable should take precedence
            assert settings.logging.level == "ERROR"


def test_singleton_pattern():
    """Test that the settings module exports a singleton instance."""
    from src.config import settings as settings1
    from src.config import settings as settings2
    
    # Both imports should reference the same object
    assert settings1 is settings2
    
    # Modifying one should affect the other
    # Note: This is testing the module behavior, not the Settings class
    original_debug = settings1.debug_mode
    try:
        settings1.debug_mode = not original_debug
        assert settings2.debug_mode == settings1.debug_mode
    finally:
        # Restore original value
        settings1.debug_mode = original_debug
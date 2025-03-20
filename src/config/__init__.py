"""
Configuration package for the OpenAI Realtime Assistant.

This package provides centralized configuration management for the application,
loading settings from environment variables with appropriate validation.
"""

from src.config.settings import Settings

# Create a singleton instance of Settings to be imported by other modules
settings = Settings()

__all__ = ["settings"]

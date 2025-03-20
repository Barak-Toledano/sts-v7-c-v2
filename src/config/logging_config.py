"""
Logging configuration for the OpenAI Realtime Assistant.

This module provides a centralized logging configuration system with
session-based logs, console output, and proper formatting.
"""

import logging
import logging.handlers
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

from src.config import settings


class LoggingManager:
    """
    Manages logging configuration for the application.
    
    This class provides methods to set up logging with different outputs
    (console, file) and levels. It ensures logs are organized by session
    and prevents duplicate configurations.
    """
    
    # Track configured loggers to prevent duplicates
    _configured_loggers: Dict[str, bool] = {}
    
    # Current session identifier
    _session_id: Optional[str] = None
    
    @classmethod
    def get_session_id(cls) -> str:
        """
        Get the current session ID, creating a new one if needed.
        
        Returns:
            str: The session identifier
        """
        if cls._session_id is None:
            # Generate a unique session ID with timestamp prefix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            cls._session_id = f"{timestamp}_{unique_id}"
        
        return cls._session_id
    
    @classmethod
    def set_session_id(cls, session_id: str) -> None:
        """
        Set a specific session ID.
        
        Args:
            session_id: The session identifier to use
        """
        cls._session_id = session_id
    
    @classmethod
    def setup_logging(
        cls,
        level: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """
        Configure the root logger and set up handlers.
        
        Args:
            level: Optional override for the logging level
            session_id: Optional session ID to use
        """
        if session_id:
            cls._session_id = session_id
        
        # Get the effective level
        log_level = level or settings.logging.level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Remove existing handlers if any
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler if enabled
        if settings.logging.console_enabled:
            cls._add_console_handler(root_logger, numeric_level)
        
        # Add file handler if enabled
        if settings.logging.file_enabled:
            cls._add_file_handler(root_logger, numeric_level)
        
        # Mark the root logger as configured
        cls._configured_loggers["root"] = True
        
        # Log the configuration
        logging.info(f"Logging configured: level={log_level}, session_id={cls.get_session_id()}")
    
    @classmethod
    def get_logger(cls, name: str, level: Optional[str] = None) -> logging.Logger:
        """
        Get a logger with the specified name.
        
        This ensures the logger is properly configured if it hasn't been already.
        
        Args:
            name: The logger name
            level: Optional specific level for this logger
            
        Returns:
            logging.Logger: The configured logger
        """
        # Get the logger
        logger = logging.getLogger(name)
        
        # Set specific level if provided
        if level:
            numeric_level = getattr(logging, level.upper(), None)
            if numeric_level:
                logger.setLevel(numeric_level)
        
        # If no loggers have been configured yet, set up root logger
        if not cls._configured_loggers:
            cls.setup_logging()
        
        return logger
    
    @classmethod
    def _add_console_handler(
        cls,
        logger: logging.Logger,
        level: int
    ) -> None:
        """
        Add a console handler to the logger.
        
        Args:
            logger: The logger to add the handler to
            level: The logging level for the handler
        """
        # Create console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        
        # Create formatter with simplified format for console
        formatter = logging.Formatter(settings.logging.format)
        console.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(console)
    
    @classmethod
    def _add_file_handler(
        cls,
        logger: logging.Logger,
        level: int
    ) -> None:
        """
        Add a file handler to the logger.
        
        Args:
            logger: The logger to add the handler to
            level: The logging level for the handler
        """
        # Ensure logs directory exists
        os.makedirs(settings.logs_dir, exist_ok=True)
        
        # Create main log file path
        log_file = settings.logs_dir / f"app_{cls.get_session_id()}.log"
        
        # Create session-specific log file
        session_log = settings.get_session_log_path(cls.get_session_id())
        
        # Add file handler for main log
        cls._create_file_handler(logger, log_file, level)
        
        # Add file handler for session log
        cls._create_file_handler(logger, session_log, level)
    
    @staticmethod
    def _create_file_handler(
        logger: logging.Logger,
        log_file: Union[str, Path],
        level: int
    ) -> None:
        """
        Create and add a file handler to the logger.
        
        Args:
            logger: The logger to add the handler to
            log_file: The path to the log file
            level: The logging level for the handler
        """
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(level)
        
        # Create detailed formatter for file logs
        formatter = logging.Formatter(settings.logging.detailed_format)
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(file_handler)


# Initialize logging when module is imported
LoggingManager.setup_logging()

# Create a function to get a logger with the right configuration
def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a properly configured logger with the specified name.
    
    Args:
        name: The logger name (usually __name__)
        level: Optional specific level for this logger
        
    Returns:
        logging.Logger: The configured logger
    """
    return LoggingManager.get_logger(name, level)

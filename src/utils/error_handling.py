"""
Error handling utilities for the OpenAI Realtime Assistant.

This module provides consistent error handling mechanisms, custom exceptions,
and utilities for generating informative error messages without exposing
sensitive information.
"""

import logging
import sys
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AppError(Exception):
    """Base exception class for application errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize the error.
        
        Args:
            message: Human-readable error message
            severity: Error severity level
            error_code: Optional error code for categorization
            details: Additional error details
            cause: Original exception that caused this error
        """
        self.message = message
        self.severity = severity
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        
        # Include cause's message in our message if provided
        if cause and not str(cause) in message:
            full_message = f"{message}: {str(cause)}"
        else:
            full_message = message
            
        super().__init__(full_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the error
        """
        error_dict = {
            "message": self.message,
            "severity": self.severity.value,
        }
        
        if self.error_code:
            error_dict["code"] = self.error_code
            
        if self.details:
            # Filter out any sensitive information
            safe_details = {k: v for k, v in self.details.items() 
                          if not _is_sensitive_key(k)}
            
            if safe_details:
                error_dict["details"] = safe_details
        
        return error_dict
    
    def log(self, include_traceback: bool = True) -> None:
        """
        Log the error with appropriate level.
        
        Args:
            include_traceback: Whether to include the traceback in the log
        """
        log_method = getattr(logger, self.severity.value.lower(), logger.error)
        
        # Start with the main error message
        log_message = f"{self.__class__.__name__}: {self.message}"
        
        # Add error code if available
        if self.error_code:
            log_message = f"{log_message} [Code: {self.error_code}]"
        
        # Log the message
        log_method(log_message)
        
        # Log safe details if available
        if self.details:
            safe_details = {k: v for k, v in self.details.items() 
                          if not _is_sensitive_key(k)}
            if safe_details:
                log_method(f"Error details: {safe_details}")
        
        # Log traceback for non-debug severities if requested
        if include_traceback and self.severity.value in ("error", "critical"):
            if self.cause:
                log_method(f"Caused by: {type(self.cause).__name__}: {str(self.cause)}")
                if hasattr(self.cause, "__traceback__") and self.cause.__traceback__:
                    log_method("".join(traceback.format_tb(self.cause.__traceback__)))
            else:
                log_method("".join(traceback.format_tb(sys.exc_info()[2])))


class ConfigError(AppError):
    """Error related to application configuration."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize with a default error code for configuration errors."""
        error_code = error_code or "CONFIG_ERROR"
        super().__init__(message, severity, error_code, details, cause)


class ApiError(AppError):
    """Error related to external API communication."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize with a default error code for API errors."""
        error_code = error_code or "API_ERROR"
        super().__init__(message, severity, error_code, details, cause)


class AudioError(AppError):
    """Error related to audio processing."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize with a default error code for audio errors."""
        error_code = error_code or "AUDIO_ERROR"
        super().__init__(message, severity, error_code, details, cause)


class ValidationError(AppError):
    """Error related to data validation."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize with a default error code for validation errors."""
        error_code = error_code or "VALIDATION_ERROR"
        super().__init__(message, severity, error_code, details, cause)


def handle_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    error_class: Type[AppError] = AppError,
    log_exception: bool = True
) -> AppError:
    """
    Convert any exception to an application error.
    
    Args:
        exception: The exception to handle
        context: Additional context information
        error_class: The specific AppError class to use
        log_exception: Whether to log the exception
        
    Returns:
        AppError: The application error instance
    """
    # If already an AppError, add context and return
    if isinstance(exception, AppError):
        if context:
            # Add context to existing details
            for key, value in context.items():
                if key not in exception.details:
                    exception.details[key] = value
        
        # Log if requested
        if log_exception:
            exception.log()
            
        return exception
    
    # Convert to AppError
    details = context or {}
    
    # Add exception class name to details
    details["exception_type"] = type(exception).__name__
    
    # Create the application error
    app_error = error_class(
        message=str(exception) or f"An {type(exception).__name__} occurred",
        cause=exception,
        details=details
    )
    
    # Log if requested
    if log_exception:
        app_error.log()
    
    return app_error


def safe_execute(
    func: callable,
    args: List[Any] = None,
    kwargs: Dict[str, Any] = None,
    error_message: str = "Error executing function",
    error_class: Type[AppError] = AppError,
    log_error: bool = True,
    reraise: bool = False,
    default_return: Any = None
) -> Any:
    """
    Execute a function safely, handling exceptions.
    
    Args:
        func: The function to execute
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        error_message: Message to use if an exception occurs
        error_class: The specific AppError class to use
        log_error: Whether to log any error
        reraise: Whether to reraise the exception
        default_return: Value to return on error
        
    Returns:
        Any: The function's return value or default_return on error
        
    Raises:
        AppError: If reraise is True and an error occurs
    """
    # Initialize arguments
    args = args or []
    kwargs = kwargs or {}
    
    try:
        # Execute the function
        return func(*args, **kwargs)
    except Exception as e:
        # Create error context
        context = {
            "function": getattr(func, "__name__", str(func)),
            "args": str(args),
            "kwargs": {k: "***" if _is_sensitive_key(k) else v for k, v in kwargs.items()}
        }
        
        # Handle the exception
        app_error = handle_exception(
            e,
            context=context,
            error_class=error_class,
            log_exception=log_error
        )
        
        # Reraise if requested
        if reraise:
            raise app_error
        
        # Return default value
        return default_return


def _is_sensitive_key(key: str) -> bool:
    """
    Check if a key might contain sensitive information.
    
    Args:
        key: The key to check
        
    Returns:
        bool: True if the key might contain sensitive information
    """
    sensitive_patterns = [
        "password", "secret", "key", "token", "auth", "cred", 
        "private", "security", "cert", "signature"
    ]
    
    lowercase_key = key.lower()
    return any(pattern in lowercase_key for pattern in sensitive_patterns)

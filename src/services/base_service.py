"""
Base service interface for external API integrations.
"""
from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional

from src.config.logging_config import get_logger
from src.utils.error_handling import AppError, ErrorSeverity

logger = get_logger(__name__)

class BaseService(ABC):
    """Base class for all external service integrations.
    
    This abstract class defines the interface that all service
    implementations should follow. It provides common functionality
    like error handling, logging, and secure configuration management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the service with configuration.
        
        Args:
            config: Configuration dictionary for the service
        """
        self.config = self._sanitize_config(config)
        self._validate_config()
    
    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive information in configuration logs.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Dict[str, Any]: Sanitized configuration
        """
        # Create a copy to avoid modifying the original
        sanitized = config.copy()
        
        # Mask sensitive keys
        sensitive_keys = ["api_key", "secret", "password", "token"]
        for key in sensitive_keys:
            if key in sanitized:
                sanitized[key] = "********"
        
        return sanitized
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the service configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the service.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the service."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the service connection.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        pass
    
    async def handle_error(
        self, 
        error: Exception, 
        operation: str, 
        additional_info: Optional[Dict[str, Any]] = None
    ) -> AppError:
        """Handle service errors consistently.
        
        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            additional_info: Any additional context information
            
        Returns:
            AppError: Structured error object
        """
        # Create sanitized additional info
        safe_info = {}
        if additional_info:
            safe_info = self._sanitize_config(additional_info)
            
        # Determine error severity
        severity = ErrorSeverity.ERROR
        if "rate_limit" in str(error).lower():
            severity = ErrorSeverity.WARNING
        
        # Create structured error
        app_error = AppError(
            message=f"Service error in {operation}: {str(error)}",
            severity=severity,
            details=safe_info,
            cause=error
        )
        
        logger.error(f"Service error: {app_error}")
        return app_error
    
    def rotate_credentials(self) -> bool:
        """Rotate service credentials if supported.
        
        Returns:
            bool: True if credentials were rotated
        """
        # Default implementation does nothing
        return False
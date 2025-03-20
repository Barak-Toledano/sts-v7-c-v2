"""
OpenAI service implementation for API communication.

This module provides a high-level service interface for interacting with 
OpenAI APIs, including the Realtime API for conversation and the standard
API for other OpenAI services.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union

from src.config import settings
from src.config.logging_config import get_logger
from src.events.event_interface import EventType, event_bus
from src.services.api_client import RealtimeClient
from src.services.base_service import BaseService
from src.utils.error_handling import AppError, ErrorSeverity, ApiError

logger = get_logger(__name__)

class OpenAIService(BaseService):
    """Service for interfacing with OpenAI APIs.
    
    This service provides a simplified interface for common OpenAI API
    operations, with proper error handling and event integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the OpenAI service.
        
        Args:
            config: Configuration including API keys and endpoints
        """
        super().__init__(config)
        self.realtime_client = None
        self.assistant_id = config.get("assistant_id")
    
    def _validate_config(self) -> None:
        """Validate the service configuration.
        
        Raises:
            ValueError: If API key or other required config is missing
        """
        if "api_key" not in self.config:
            raise ValueError("OpenAI API key is required")
        
        if not self.config.get("api_key"):
            raise ValueError("OpenAI API key cannot be empty")
    
    async def connect(self) -> bool:
        """Connect to the OpenAI Realtime API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create the Realtime client
            self.realtime_client = RealtimeClient()
            
            # Connect to the API with the assistant ID from config
            connected = await self.realtime_client.connect(
                assistant_id=self.assistant_id or self.config.get("assistant_id"),
                instructions=self.config.get("instructions"),
                temperature=self.config.get("temperature", 1.0),
                enable_transcription=self.config.get("enable_transcription", True)
            )
            
            if connected:
                logger.info("Connected to OpenAI Realtime API")
                
                # Emit a connected event
                event_bus.emit(EventType.API_CONNECTION_ESTABLISHED, {
                    "service": "openai",
                    "client": "realtime"
                })
                
                return True
            else:
                logger.error("Failed to connect to OpenAI Realtime API")
                return False
                
        except Exception as e:
            error = await self.handle_error(e, "connect")
            logger.error(f"Failed to connect to OpenAI Realtime API: {error}")
            
            # Emit error event
            event_bus.emit(EventType.ERROR, {"error": error.to_dict()})
            
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the OpenAI Realtime API."""
        if self.realtime_client:
            await self.realtime_client.disconnect()
            logger.info("Disconnected from OpenAI Realtime API")
            
            self.realtime_client = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the OpenAI API connection.
        
        Returns:
            Dict[str, Any]: Connection status information
        """
        is_connected = self.realtime_client and self.realtime_client.is_connected
        
        return {
            "service": "OpenAI",
            "type": "realtime",
            "connected": is_connected,
            "session_id": self.realtime_client.session_id if is_connected else None,
            "assistant_id": self.assistant_id
        }
    
    async def send_message(self, text: str) -> str:
        """Send a text message and get a response.
        
        This is a high-level method that handles sending a message and
        requesting a response in one operation.
        
        Args:
            text: The message text to send
            
        Returns:
            str: Response text from the model
        """
        if not self.realtime_client or not self.realtime_client.is_connected:
            raise ApiError("Not connected to OpenAI API", ErrorSeverity.ERROR)
        
        try:
            # Request response from the model
            response_id = await self.realtime_client.request_response(
                instructions=f"Respond to: {text}"
            )
            
            logger.debug(f"Requested response: {response_id}")
            return response_id
            
        except Exception as e:
            error = await self.handle_error(e, "send_message")
            raise ApiError(f"Failed to send message: {str(e)}", 
                           ErrorSeverity.ERROR, cause=e)
    
    async def update_settings(self, settings: Dict[str, Any]) -> bool:
        """Update service settings.
        
        Args:
            settings: New settings to apply
            
        Returns:
            bool: True if settings were updated successfully
        """
        if not self.realtime_client:
            raise ApiError("Not connected to OpenAI API", ErrorSeverity.ERROR)
        
        try:
            # Apply settings to the client
            await self.realtime_client.update_session(settings)
            logger.info("Updated OpenAI service settings")
            return True
            
        except Exception as e:
            error = await self.handle_error(e, "update_settings")
            logger.error(f"Failed to update settings: {error}")
            return False
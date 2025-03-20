"""
Domain logic module for the OpenAI Realtime Assistant.

This package contains the core business logic organized by domain areas:
- audio: Audio recording, playback, and voice activity detection
- conversation: Conversation management, state tracking, and response handling

The domain layer is independent of external services and presentation concerns,
focusing purely on the business rules and application logic.
"""

# Import key classes for easier access
from src.domain.audio.manager import AudioManager, AudioMode
from src.domain.conversation.manager import ConversationManager, create_conversation_manager
from src.domain.conversation.state import (
    ConversationContext, ConversationHistory, ConversationStateType,
    Message, MessageContent, MessageRole
)

# Export key classes
__all__ = [
    # Audio domain
    'AudioManager',
    'AudioMode',
    
    # Conversation domain
    'ConversationManager',
    'create_conversation_manager',
    'ConversationContext',
    'ConversationHistory',
    'ConversationStateType',
    'Message',
    'MessageContent',
    'MessageRole',
]
"""
Conversation state management for the OpenAI Realtime Assistant.

This module provides classes and utilities for managing conversation state,
including history, context, and metadata.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from src.config.logging_config import get_logger
from src.utils.token_management import estimate_token_count, truncate_conversation_to_fit

logger = get_logger(__name__)


class MessageRole(Enum):
    """Possible roles for conversation messages."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class ConversationStateType(Enum):
    """Possible states for a conversation."""
    
    IDLE = "idle"
    CONNECTING = "connecting"
    READY = "ready"
    USER_SPEAKING = "user_speaking"
    THINKING = "thinking"
    ASSISTANT_SPEAKING = "assistant_speaking"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class MessageContent:
    """Content for a conversation message."""
    
    text: Optional[str] = None
    audio_duration: Optional[float] = None
    audio_id: Optional[str] = None
    interrupted: bool = False
    is_final: bool = True
    language: str = "en"
    function_call: Optional[Dict[str, Any]] = None
    function_result: Optional[Dict[str, Any]] = None


@dataclass
class Message:
    """Represents a message in a conversation."""
    
    id: str
    role: MessageRole
    content: MessageContent
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @staticmethod
    def create_user_message(
        text: Optional[str] = None, 
        audio_duration: Optional[float] = None,
        audio_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """Create a new user message."""
        return Message(
            id=f"msg_{uuid.uuid4().hex}",
            role=MessageRole.USER,
            content=MessageContent(
                text=text,
                audio_duration=audio_duration,
                audio_id=audio_id,
                is_final=True
            ),
            metadata=metadata or {}
        )
    
    @staticmethod
    def create_assistant_message(
        text: Optional[str] = None, 
        audio_duration: Optional[float] = None,
        interrupted: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """Create a new assistant message."""
        return Message(
            id=f"msg_{uuid.uuid4().hex}",
            role=MessageRole.ASSISTANT,
            content=MessageContent(
                text=text, 
                audio_duration=audio_duration,
                interrupted=interrupted
            ),
            metadata=metadata or {}
        )
    
    @staticmethod
    def create_system_message(
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """Create a new system message."""
        return Message(
            id=f"msg_{uuid.uuid4().hex}",
            role=MessageRole.SYSTEM,
            content=MessageContent(text=text),
            metadata=metadata or {}
        )
    
    @staticmethod
    def create_function_call_message(
        name: str,
        arguments: Dict[str, Any],
        call_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """Create a new function call message."""
        return Message(
            id=f"msg_{uuid.uuid4().hex}",
            role=MessageRole.FUNCTION,
            content=MessageContent(
                function_call={
                    "name": name,
                    "arguments": arguments,
                    "call_id": call_id
                }
            ),
            metadata=metadata or {"type": "function_call"}
        )
    
    @staticmethod
    def create_function_result_message(
        call_id: str,
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """Create a new function result message."""
        return Message(
            id=f"msg_{uuid.uuid4().hex}",
            role=MessageRole.FUNCTION,
            content=MessageContent(
                function_result={
                    "call_id": call_id,
                    "result": result
                }
            ),
            metadata=metadata or {"type": "function_result"}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to a dictionary representation."""
        result = {
            "id": self.id,
            "role": self.role.value,
            "created_at": self.created_at,
            "metadata": self.metadata
        }
        
        # Handle content based on type
        if isinstance(self.content, MessageContent):
            content_dict = {}
            if self.content.text is not None:
                content_dict["text"] = self.content.text
            if self.content.audio_duration is not None:
                content_dict["audio_duration"] = self.content.audio_duration
            if self.content.audio_id is not None:
                content_dict["audio_id"] = self.content.audio_id
            if self.content.interrupted:
                content_dict["interrupted"] = True
            if not self.content.is_final:
                content_dict["is_final"] = False
            if self.content.language != "en":
                content_dict["language"] = self.content.language
            if self.content.function_call:
                content_dict["function_call"] = self.content.function_call
            if self.content.function_result:
                content_dict["function_result"] = self.content.function_result
            
            result["content"] = content_dict
        else:
            # Handle case where content might be a string or other type
            result["content"] = self.content
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a message from a dictionary representation."""
        role = data.get("role", "user")
        role_enum = MessageRole.USER
        
        if role == "assistant":
            role_enum = MessageRole.ASSISTANT
        elif role == "system":
            role_enum = MessageRole.SYSTEM
        elif role == "function":
            role_enum = MessageRole.FUNCTION
        
        content_data = data.get("content", {})
        if isinstance(content_data, str):
            content = MessageContent(text=content_data)
        else:
            # Create MessageContent from dictionary
            content = MessageContent(
                text=content_data.get("text"),
                audio_duration=content_data.get("audio_duration"),
                audio_id=content_data.get("audio_id"),
                interrupted=content_data.get("interrupted", False),
                is_final=content_data.get("is_final", True),
                language=content_data.get("language", "en"),
                function_call=content_data.get("function_call"),
                function_result=content_data.get("function_result")
            )
        
        return cls(
            id=data.get("id", f"msg_{uuid.uuid4().hex}"),
            role=role_enum,
            content=content,
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {})
        )


@dataclass
class ConversationContext:
    """Context for a conversation."""
    
    assistant_id: str
    session_id: str
    thread_id: Optional[str] = None
    run_id: Optional[str] = None
    instructions: Optional[str] = None
    temperature: float = 1.0
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: Optional[str] = "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)
    voice: Optional[str] = "alloy"
    token_limit: int = 16385  # Default for gpt-4o
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to a dictionary representation."""
        return {
            "assistant_id": self.assistant_id,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "run_id": self.run_id,
            "instructions": self.instructions,
            "temperature": self.temperature,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "metadata": self.metadata,
            "voice": self.voice,
            "token_limit": self.token_limit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Create a context from a dictionary representation."""
        return cls(
            assistant_id=data.get("assistant_id", ""),
            session_id=data.get("session_id", ""),
            thread_id=data.get("thread_id"),
            run_id=data.get("run_id"),
            instructions=data.get("instructions"),
            temperature=data.get("temperature", 1.0),
            tools=data.get("tools", []),
            tool_choice=data.get("tool_choice", "auto"),
            metadata=data.get("metadata", {}),
            voice=data.get("voice", "alloy"),
            token_limit=data.get("token_limit", 16385)
        )


class ConversationHistory:
    """
    Manages conversation history and state.
    
    This class provides methods for adding, retrieving, and persisting
    conversation messages and context.
    """
    
    def __init__(self, context: ConversationContext):
        """
        Initialize conversation history.
        
        Args:
            context: Context for the conversation
        """
        self.context = context
        self.messages: List[Message] = []
        self.pending_messages: List[Message] = []
        self.state = ConversationStateType.IDLE
        self.metadata: Dict[str, Any] = {}
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.estimated_tokens = 0
    
    def add_message(self, message: Message) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            message: Message to add
        """
        self.messages.append(message)
        self.updated_at = time.time()
        
        # Update token count estimate
        if hasattr(message.content, "text") and message.content.text:
            self.estimated_tokens += estimate_token_count(message.content.text)
        
        logger.debug(f"Added {message.role.value} message to conversation history")
    
    def add_pending_message(self, message: Message) -> None:
        """
        Add a message to the pending messages queue.
        
        These messages will be added to history when confirmed.
        
        Args:
            message: Message to add to pending queue
        """
        self.pending_messages.append(message)
        logger.debug(f"Added {message.role.value} message to pending queue")
    
    def confirm_pending_message(self, message_id: str) -> bool:
        """
        Confirm a pending message and add it to history.
        
        Args:
            message_id: ID of the message to confirm
            
        Returns:
            bool: True if message was found and confirmed
        """
        for i, message in enumerate(self.pending_messages):
            if message.id == message_id:
                # Remove from pending
                confirmed_message = self.pending_messages.pop(i)
                # Add to history
                self.add_message(confirmed_message)
                return True
        
        return False
    
    def update_message(
        self, 
        message_id: str, 
        text: Optional[str] = None,
        content: Optional[MessageContent] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing message.
        
        Args:
            message_id: ID of the message to update
            text: New text content (if updating text)
            content: New content object (if replacing content)
            metadata: Additional metadata to merge
            
        Returns:
            bool: True if message was found and updated
        """
        message = self.get_message(message_id)
        if not message:
            return False
            
        # Update content
        if content:
            message.content = content
        elif text is not None and hasattr(message.content, "text"):
            # Update just the text field if it exists
            message.content.text = text
            
        # Update metadata
        if metadata:
            if not message.metadata:
                message.metadata = {}
            message.metadata.update(metadata)
            
        self.updated_at = time.time()
        return True
    
    def mark_message_interrupted(self, message_id: str) -> bool:
        """
        Mark a message as interrupted.
        
        Args:
            message_id: ID of the message to mark
            
        Returns:
            bool: True if message was found and marked
        """
        message = self.get_message(message_id)
        if not message:
            return False
            
        if hasattr(message.content, "interrupted"):
            message.content.interrupted = True
            self.updated_at = time.time()
            return True
            
        return False
    
    def get_message(self, message_id: str) -> Optional[Message]:
        """
        Get a message by ID.
        
        Args:
            message_id: ID of the message to get
            
        Returns:
            Optional[Message]: The message, or None if not found
        """
        for message in self.messages:
            if message.id == message_id:
                return message
                
        # Also check pending messages
        for message in self.pending_messages:
            if message.id == message_id:
                return message
                
        return None
    
    def get_last_message(self) -> Optional[Message]:
        """
        Get the last message in the conversation.
        
        Returns:
            Optional[Message]: The last message, or None if there are no messages
        """
        if not self.messages:
            return None
        return self.messages[-1]
    
    def get_last_user_message(self) -> Optional[Message]:
        """
        Get the last message from the user.
        
        Returns:
            Optional[Message]: The last user message, or None if there are no user messages
        """
        for message in reversed(self.messages):
            if message.role == MessageRole.USER:
                return message
        return None
    
    def get_last_assistant_message(self) -> Optional[Message]:
        """
        Get the last message from the assistant.
        
        Returns:
            Optional[Message]: The last assistant message, or None if there are no assistant messages
        """
        for message in reversed(self.messages):
            if message.role == MessageRole.ASSISTANT:
                return message
        return None
    
    def get_system_message(self) -> Optional[Message]:
        """
        Get the system message.
        
        Returns:
            Optional[Message]: The system message, or None if there is no system message
        """
        for message in self.messages:
            if message.role == MessageRole.SYSTEM:
                return message
        return None
    
    def get_messages_by_role(self, role: MessageRole) -> List[Message]:
        """
        Get all messages with a specific role.
        
        Args:
            role: Role to filter by
            
        Returns:
            List[Message]: Messages with the specified role
        """
        return [message for message in self.messages if message.role == role]
    
    def clear(self) -> None:
        """Clear all messages from the conversation history."""
        self.messages = []
        self.pending_messages = []
        self.estimated_tokens = 0
        self.updated_at = time.time()
        logger.debug("Cleared conversation history")
    
    def truncate_history(
        self, 
        token_limit: Optional[int] = None,
        keep_system_message: bool = True,
        keep_last_n_turns: int = 3
    ) -> int:
        """
        Truncate conversation history to fit within token limits.
        
        Args:
            token_limit: Maximum tokens to allow (uses context.token_limit if None)
            keep_system_message: Whether to preserve the system message
            keep_last_n_turns: Number of recent message turns to preserve
            
        Returns:
            int: Number of messages removed
        """
        if token_limit is None:
            token_limit = self.context.token_limit
            
        if self.estimated_tokens <= token_limit:
            return 0  # No truncation needed
            
        # Convert messages to API format for token estimation
        api_messages = []
        system_message_index = None
        
        for i, message in enumerate(self.messages):
            msg_dict = {"role": message.role.value}
            
            if hasattr(message.content, "text") and message.content.text:
                msg_dict["content"] = message.content.text
            elif hasattr(message.content, "function_call") and message.content.function_call:
                msg_dict["function_call"] = message.content.function_call
            elif hasattr(message.content, "function_result") and message.content.function_result:
                msg_dict["content"] = json.dumps(message.content.function_result.get("result", {}))
                
            api_messages.append(msg_dict)
            
            if message.role == MessageRole.SYSTEM:
                system_message_index = i
        
        # Use token management utility to truncate
        truncated_messages = truncate_conversation_to_fit(
            api_messages,
            token_limit,
            preserve_system_message=keep_system_message,
            preserve_recent_messages=keep_last_n_turns * 2  # Multiply by 2 for user-assistant turn pairs
        )
        
        # Find which messages we're keeping
        truncated_indices = set()
        
        # Add system message if requested
        if keep_system_message and system_message_index is not None:
            truncated_indices.add(system_message_index)
            
        # Add recent messages (keep user-assistant pairs together)
        user_assistant_pairs = min(keep_last_n_turns, len(self.messages) // 2)
        start_idx = max(0, len(self.messages) - (user_assistant_pairs * 2))
        
        for i in range(start_idx, len(self.messages)):
            truncated_indices.add(i)
            
        # Create new messages list with only kept messages
        new_messages = []
        for i, message in enumerate(self.messages):
            if i in truncated_indices:
                new_messages.append(message)
                
        # Calculate removed messages
        removed_count = len(self.messages) - len(new_messages)
        
        # Update messages and token count
        self.messages = new_messages
        self.estimated_tokens = sum(
            estimate_token_count(msg.content.text) 
            for msg in self.messages 
            if hasattr(msg.content, "text") and msg.content.text
        )
        
        logger.info(f"Truncated conversation history: removed {removed_count} messages")
        return removed_count
    
    def set_state(self, state: ConversationStateType) -> None:
        """
        Set the current conversation state.
        
        Args:
            state: New state for the conversation
        """
        if self.state != state:
            logger.info(f"Conversation state changed: {self.state.value} -> {state.value}")
            self.state = state
            self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the conversation history to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the conversation history
        """
        return {
            "context": self.context.to_dict(),
            "messages": [message.to_dict() for message in self.messages],
            "pending_messages": [message.to_dict() for message in self.pending_messages],
            "state": self.state.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "estimated_tokens": self.estimated_tokens
        }
    
    def to_json(self) -> str:
        """
        Convert the conversation history to a JSON string.
        
        Returns:
            str: JSON representation of the conversation history
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationHistory':
        """
        Create a conversation history from a dictionary representation.
        
        Args:
            data: Dictionary representation of a conversation history
            
        Returns:
            ConversationHistory: The created conversation history
        """
        context = ConversationContext.from_dict(data.get("context", {}))
        history = cls(context)
        
        # Add messages
        for message_data in data.get("messages", []):
            message = Message.from_dict(message_data)
            history.messages.append(message)
        
        # Add pending messages
        for message_data in data.get("pending_messages", []):
            message = Message.from_dict(message_data)
            history.pending_messages.append(message)
        
        # Set state
        state_value = data.get("state", "idle")
        try:
            history.state = ConversationStateType(state_value)
        except ValueError:
            history.state = ConversationStateType.IDLE
            logger.warning(f"Unknown state value: {state_value}, using IDLE instead")
        
        # Set metadata and timestamps
        history.metadata = data.get("metadata", {})
        history.created_at = data.get("created_at", time.time())
        history.updated_at = data.get("updated_at", time.time())
        history.estimated_tokens = data.get("estimated_tokens", 0)
        
        return history
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ConversationHistory':
        """
        Create a conversation history from a JSON string.
        
        Args:
            json_str: JSON representation of a conversation history
            
        Returns:
            ConversationHistory: The created conversation history
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse conversation history JSON: {e}")
            raise ValueError(f"Invalid conversation history JSON: {e}")


def format_timestamp(timestamp: float) -> str:
    """
    Format a Unix timestamp as a human-readable string.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        str: Formatted timestamp string
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")
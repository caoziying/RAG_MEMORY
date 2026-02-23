"""
Pydantic schemas for API request/response validation.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class MessageBase(BaseModel):
    """Base message schema."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class MessageCreate(MessageBase):
    """Schema for creating a new message."""
    pass


class Message(MessageBase):
    """Schema for message response."""
    id: str = Field(..., description="Message ID")
    conversation_id: str = Field(..., description="Conversation ID")
    timestamp: datetime = Field(..., description="Message timestamp")

    class Config:
        from_attributes = True


class ConversationBase(BaseModel):
    """Base conversation schema."""
    title: str = Field(default="新对话", description="Conversation title")
    conversation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConversationCreate(ConversationBase):
    """Schema for creating a new conversation."""
    pass


class Conversation(ConversationBase):
    """Schema for conversation response."""
    id: str = Field(..., description="Conversation ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    message_count: Optional[int] = Field(None, description="Number of messages in conversation")

    class Config:
        from_attributes = True


class ConversationWithMessages(Conversation):
    """Conversation schema including messages."""
    messages: List[Message] = Field(default_factory=list, description="Messages in conversation")


class ChatRequest(BaseModel):
    """Schema for chat request."""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID (create new if not provided)")
    stream: bool = Field(default=False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Schema for chat response."""
    conversation_id: str = Field(..., description="Conversation ID")
    response: str = Field(..., description="Assistant response")
    message_id: str = Field(..., description="Message ID of the assistant response")


class ConversationListResponse(BaseModel):
    """Schema for conversation list response."""
    conversations: List[Conversation] = Field(..., description="List of conversations")
    total: int = Field(..., description="Total number of conversations")


class MessageListResponse(BaseModel):
    """Schema for message list response."""
    messages: List[Message] = Field(..., description="List of messages")
    conversation_id: str = Field(..., description="Conversation ID")
    total: int = Field(..., description="Total number of messages")


class SystemStatus(BaseModel):
    """Schema for system status response."""
    status: str = Field(..., description="System status")
    version: str = Field(..., description="System version")
    active_conversations: int = Field(..., description="Number of active conversation sessions")
    database_connected: bool = Field(..., description="Whether database is connected")
    vector_store_enabled: bool = Field(..., description="Whether vector store is enabled")


class ErrorResponse(BaseModel):
    """Schema for error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    code: Optional[int] = Field(None, description="Error code")
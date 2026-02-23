"""
Chat API endpoints for handling conversation messages.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from ..schemas import ChatRequest, ChatResponse, ErrorResponse
from ..services.conversation_manager import conversation_manager
from ..models import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
) -> ChatResponse:
    """
    Send a message and get a response.

    This endpoint handles both new conversations and existing conversations.
    If conversation_id is not provided, a new conversation will be created.

    Args:
        request: Chat request containing message and optional conversation_id
        db: Database session

    Returns:
        Chat response with assistant reply
    """
    try:
        # Get or create conversation
        if request.conversation_id:
            conversation = conversation_manager.get_conversation(request.conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            # Create new conversation
            conversation_id = conversation_manager.create_conversation()
            conversation = conversation_manager.get_conversation(conversation_id)
            if not conversation:
                raise HTTPException(status_code=500, detail="Failed to create conversation")

        # Generate response using RAG
        response_text = conversation.generate_response(request.message)

        # Get the last message ID (assistant's response)
        recent_messages = conversation.get_recent_messages(limit=2)
        message_id = None
        for msg in recent_messages:
            if msg.get("role") == "assistant":
                message_id = msg.get("id")
                break

        if not message_id:
            # Fallback: generate a new message ID
            import uuid
            message_id = str(uuid.uuid4())

        return ChatResponse(
            conversation_id=conversation.conversation_id,
            response=response_text,
            message_id=message_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/{conversation_id}", response_model=ChatResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def chat_with_conversation(
    conversation_id: str,
    request: ChatRequest,
    db: Session = Depends(get_db)
) -> ChatResponse:
    """
    Send a message to a specific conversation.

    Args:
        conversation_id: Conversation ID
        request: Chat request containing message
        db: Database session

    Returns:
        Chat response with assistant reply
    """
    try:
        # Get conversation
        conversation = conversation_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Generate response
        response_text = conversation.generate_response(request.message)

        # Get the last message ID
        recent_messages = conversation.get_recent_messages(limit=2)
        message_id = None
        for msg in recent_messages:
            if msg.get("role") == "assistant":
                message_id = msg.get("id")
                break

        if not message_id:
            import uuid
            message_id = str(uuid.uuid4())

        return ChatResponse(
            conversation_id=conversation.conversation_id,
            response=response_text,
            message_id=message_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_with_conversation endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
"""
Message management API endpoints.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session

from ..schemas import Message, MessageListResponse, ErrorResponse
from ..models import get_db, Message as MessageModel, Conversation as ConversationModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations/{conversation_id}/messages", tags=["messages"])


@router.get("", response_model=MessageListResponse, responses={404: {"model": ErrorResponse}})
async def list_messages(
    conversation_id: str,
    limit: int = Query(100, ge=1, le=500, description="Maximum number of messages to return"),
    offset: int = Query(0, ge=0, description="Number of messages to skip"),
    db: Session = Depends(get_db)
) -> MessageListResponse:
    """
    List messages in a conversation.

    Args:
        conversation_id: Conversation ID
        limit: Maximum number of messages to return
        offset: Number of messages to skip
        db: Database session

    Returns:
        List of messages
    """
    try:
        # Check if conversation exists
        conversation = db.query(ConversationModel).filter(
            ConversationModel.id == conversation_id
        ).first()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get messages
        messages = db.query(MessageModel).filter(
            MessageModel.conversation_id == conversation_id
        ).order_by(MessageModel.timestamp.asc()).offset(offset).limit(limit).all()

        total = db.query(MessageModel).filter(
            MessageModel.conversation_id == conversation_id
        ).count()

        # Convert to schema objects
        message_schemas = [Message.from_orm(msg) for msg in messages]

        return MessageListResponse(
            messages=message_schemas,
            conversation_id=conversation_id,
            total=total
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing messages: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/{message_id}", response_model=Message, responses={404: {"model": ErrorResponse}})
async def get_message(
    conversation_id: str,
    message_id: str,
    db: Session = Depends(get_db)
) -> Message:
    """
    Get a specific message.

    Args:
        conversation_id: Conversation ID
        message_id: Message ID
        db: Database session

    Returns:
        Message
    """
    try:
        # Check if conversation exists
        conversation = db.query(ConversationModel).filter(
            ConversationModel.id == conversation_id
        ).first()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get message
        message = db.query(MessageModel).filter(
            MessageModel.id == message_id,
            MessageModel.conversation_id == conversation_id
        ).first()

        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        return Message.from_orm(message)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting message: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete("/{message_id}", responses={404: {"model": ErrorResponse}})
async def delete_message(
    conversation_id: str,
    message_id: str,
    db: Session = Depends(get_db)
) -> dict:
    """
    Delete a message.

    Args:
        conversation_id: Conversation ID
        message_id: Message ID
        db: Database session

    Returns:
        Success message
    """
    try:
        # Check if conversation exists
        conversation = db.query(ConversationModel).filter(
            ConversationModel.id == conversation_id
        ).first()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get and delete message
        message = db.query(MessageModel).filter(
            MessageModel.id == message_id,
            MessageModel.conversation_id == conversation_id
        ).first()

        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        db.delete(message)
        db.commit()

        # Also remove from memory session if exists
        from ..services.conversation_manager import conversation_manager
        conv_session = conversation_manager.get_conversation(conversation_id)
        if conv_session and conv_session._message_cache:
            # Remove from cache
            conv_session._message_cache = [
                msg for msg in conv_session._message_cache
                if msg.get("id") != message_id
            ]

        return {"message": "Message deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting message: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
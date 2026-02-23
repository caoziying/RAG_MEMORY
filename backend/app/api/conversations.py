"""
Conversation management API endpoints.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session

from ..schemas import (
    ConversationCreate, Conversation, ConversationListResponse,
    ConversationWithMessages, ErrorResponse
)
from ..services.conversation_manager import conversation_manager
from ..models import get_db, Conversation as ConversationModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of conversations to return"),
    offset: int = Query(0, ge=0, description="Number of conversations to skip"),
    db: Session = Depends(get_db)
) -> ConversationListResponse:
    """
    List all conversations.

    Args:
        limit: Maximum number of conversations to return
        offset: Number of conversations to skip
        db: Database session

    Returns:
        List of conversations
    """
    try:
        # Get conversations from database
        conversations = db.query(ConversationModel).order_by(
            ConversationModel.updated_at.desc()
        ).offset(offset).limit(limit).all()

        total = db.query(ConversationModel).count()

        # Convert to schema objects
        conversation_list = []
        for conv in conversations:
            conv_schema = Conversation.from_orm(conv)
            conv_schema.message_count = len(conv.messages)
            conversation_list.append(conv_schema)

        return ConversationListResponse(
            conversations=conversation_list,
            total=total
        )

    except Exception as e:
        logger.error(f"Error listing conversations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("", response_model=Conversation, responses={500: {"model": ErrorResponse}})
async def create_conversation(
    request: Optional[ConversationCreate] = None,
    db: Session = Depends(get_db)
) -> Conversation:
    """
    Create a new conversation.

    Args:
        request: Conversation creation data (optional)
        db: Database session

    Returns:
        New conversation
    """
    try:
        # Create conversation via manager
        title = request.title if request else "新对话"
        metadata = request.conversation_metadata if request else {}

        conversation_id = conversation_manager.create_conversation(title)
        conversation = conversation_manager.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=500, detail="Failed to create conversation")

        # Update database record with metadata
        db_conv = db.query(ConversationModel).filter(
            ConversationModel.id == conversation_id
        ).first()

        if db_conv:
            db_conv.conversation_metadata = metadata
            db.commit()

        # Return conversation info
        return Conversation(
            id=conversation_id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            conversation_metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating conversation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/{conversation_id}", response_model=ConversationWithMessages, responses={404: {"model": ErrorResponse}})
async def get_conversation(
    conversation_id: str,
    limit: int = Query(100, ge=1, le=500, description="Maximum number of messages to return"),
    db: Session = Depends(get_db)
) -> ConversationWithMessages:
    """
    Get a conversation by ID with its messages.

    Args:
        conversation_id: Conversation ID
        limit: Maximum number of messages to return
        db: Database session

    Returns:
        Conversation with messages
    """
    try:
        # Get conversation from database
        conversation = db.query(ConversationModel).filter(
            ConversationModel.id == conversation_id
        ).first()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get messages
        messages = conversation.messages[:limit]

        # Convert to schema
        conv_schema = Conversation.from_orm(conversation)
        conv_schema.message_count = len(conversation.messages)

        # Get conversation session for additional info
        conv_session = conversation_manager.get_conversation(conversation_id)
        if conv_session:
            conv_schema.title = conv_session.title

        # Create response with messages
        from ..schemas import Message
        message_schemas = [Message.from_orm(msg) for msg in messages]

        return ConversationWithMessages(
            **conv_schema.dict(),
            messages=message_schemas
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete("/{conversation_id}", responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db)
) -> dict:
    """
    Delete a conversation.

    Args:
        conversation_id: Conversation ID
        db: Database session

    Returns:
        Success message
    """
    try:
        # Delete via manager
        success = conversation_manager.delete_conversation(conversation_id)

        if not success:
            # Check if it exists in database
            conversation = db.query(ConversationModel).filter(
                ConversationModel.id == conversation_id
            ).first()

            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")

            # Delete from database directly
            db.delete(conversation)
            db.commit()

        return {"message": "Conversation deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.put("/{conversation_id}/title", response_model=Conversation, responses={404: {"model": ErrorResponse}})
async def update_conversation_title(
    conversation_id: str,
    title: str = Query(..., min_length=1, max_length=200, description="New conversation title"),
    db: Session = Depends(get_db)
) -> Conversation:
    """
    Update conversation title.

    Args:
        conversation_id: Conversation ID
        title: New title
        db: Database session

    Returns:
        Updated conversation
    """
    try:
        # Update in database
        conversation = db.query(ConversationModel).filter(
            ConversationModel.id == conversation_id
        ).first()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conversation.title = title
        db.commit()

        # Update in memory session if exists
        conv_session = conversation_manager.get_conversation(conversation_id)
        if conv_session:
            conv_session.title = title

        # Return updated conversation
        conv_schema = Conversation.from_orm(conversation)
        conv_schema.message_count = len(conversation.messages)

        return conv_schema

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation title: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
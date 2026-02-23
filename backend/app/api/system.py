"""
System management API endpoints.
"""
import logging
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..schemas import SystemStatus, ErrorResponse
from ..models import get_db, engine
from ..services.conversation_manager import conversation_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status", response_model=SystemStatus)
async def get_system_status(db: Session = Depends(get_db)) -> SystemStatus:
    """
    Get system status and health information.

    Args:
        db: Database session

    Returns:
        System status information
    """
    try:
        # Check database connection
        database_connected = False
        try:
            db.execute(text("SELECT 1"))
            database_connected = True
        except Exception as e:
            logger.warning(f"Database connection check failed: {e}")

        # Check vector store availability
        vector_store_enabled = False
        try:
            # Try to import memory system to check vector store
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
            from memory_system.vector.memory_system import VectorMemorySystem
            vector_store_enabled = True
        except ImportError:
            vector_store_enabled = False

        # Get version information
        version = "1.0.0"
        try:
            version_file = Path(__file__).parent.parent.parent.parent / "VERSION"
            if version_file.exists():
                version = version_file.read_text().strip()
        except Exception:
            pass

        # Get active conversation count
        active_conversations = len(conversation_manager.sessions)

        return SystemStatus(
            status="healthy",
            version=version,
            active_conversations=active_conversations,
            database_connected=database_connected,
            vector_store_enabled=vector_store_enabled
        )

    except Exception as e:
        logger.error(f"Error getting system status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_system(
    max_age_hours: int = 24,
    db: Session = Depends(get_db)
) -> dict:
    """
    Clean up system resources.

    Args:
        max_age_hours: Clean up conversations older than this many hours
        db: Database session

    Returns:
        Cleanup results
    """
    try:
        # Clean up inactive sessions
        conversation_manager.cleanup_inactive_sessions(max_age_hours)

        # Optionally clean up old database records
        # This would require additional logic to delete old conversations

        return {
            "message": "System cleanup completed",
            "max_age_hours": max_age_hours,
            "active_sessions": len(conversation_manager.sessions)
        }

    except Exception as e:
        logger.error(f"Error during system cleanup: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/reset")
async def reset_system(
    confirm: bool = False,
    db: Session = Depends(get_db)
) -> dict:
    """
    Reset system (dangerous operation).

    Args:
        confirm: Must be True to proceed
        db: Database session

    Returns:
        Reset results
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Reset requires confirmation. Set confirm=true to proceed."
        )

    try:
        # Clear all conversation sessions
        conversation_manager.sessions.clear()

        # Clear database (optional - dangerous!)
        # This would delete all conversations and messages
        # Only uncomment if you really want this behavior
        # from ..models import Base
        # Base.metadata.drop_all(bind=engine)
        # Base.metadata.create_all(bind=engine)

        return {
            "message": "System reset completed",
            "sessions_cleared": True,
            "database_reset": False  # Set to True if database was reset
        }

    except Exception as e:
        logger.error(f"Error resetting system: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
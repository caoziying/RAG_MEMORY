"""
Main FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .models import init_db, engine
from .api import chat, conversations, messages, system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting conversation system backend...")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "conversations").mkdir(exist_ok=True)

    logger.info(f"Data directory: {data_dir}")

    yield

    # Shutdown
    logger.info("Shutting down conversation system backend...")
    engine.dispose()


# Create FastAPI application
app = FastAPI(
    title="RAG对话系统",
    description="基于RAG_MEMORY的对话系统，支持多对话管理和检索增强生成",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(messages.router)
app.include_router(system.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG对话系统",
        "version": "1.0.0",
        "description": "基于RAG_MEMORY的对话系统",
        "endpoints": {
            "chat": "/chat",
            "conversations": "/conversations",
            "system": "/system/status"
        },
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
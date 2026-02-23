"""
Configuration management for the conversation system.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # Database settings
    database_url: str = "sqlite:///./data/conversations.db"

    # LLM settings (inherited from RAG_MEMORY)
    my_api_key: Optional[str] = None
    my_api_base: str = "https://api.chat.csu.edu.cn/v1"
    my_model: str = "Qwen3-32B-FP8"

    # Vector store settings
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # File storage
    data_dir: Path = Path(__file__).parent.parent.parent / "data"

    # Conversation settings
    max_recent_messages: int = 10
    max_conversation_age_hours: int = 24

    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "conversations").mkdir(exist_ok=True)


# Global settings instance
settings = Settings()
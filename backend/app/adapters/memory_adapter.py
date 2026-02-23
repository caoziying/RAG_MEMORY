"""
Adapter for integrating the existing VectorMemorySystem into the conversation system.
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Add parent directory to path to import memory_system
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from memory_system.vector.memory_system import VectorMemorySystem
    from memory_system.vector.config import VectorRetrievalConfig
    MEMORY_SYSTEM_AVAILABLE = True
except Exception as e:
    logging.warning(f"Failed to import VectorMemorySystem: {e}")
    MEMORY_SYSTEM_AVAILABLE = False
    # Fallback to a dummy implementation
    class VectorMemorySystem:
        def __init__(self, **kwargs):
            self.conversation_window = []
            self.max_windows = kwargs.get('max_windows', 10)

        def add_conversation(self, role: str, content: str):
            self.conversation_window.append({"role": role, "content": content})
            if len(self.conversation_window) > self.max_windows:
                self.conversation_window = self.conversation_window[-self.max_windows:]

        def retrieve_memories(self, query, **kwargs):
            return []

        def get_context_for_llm(self, **kwargs):
            return ""

class MemoryAdapter:
    """
    Adapter for the VectorMemorySystem that provides a consistent interface
    for the conversation system.
    """

    def __init__(self, conversation_id: str, max_windows: int = 10,
                 enable_vector_store: bool = True, data_dir: Optional[str] = None):
        """
        Initialize a memory adapter for a specific conversation.

        Args:
            conversation_id: Unique identifier for the conversation
            max_windows: Maximum number of recent conversations to keep in window
            enable_vector_store: Whether to enable vector retrieval
            data_dir: Base directory for data storage
        """
        self.conversation_id = conversation_id
        self.max_windows = max_windows
        self.enable_vector_store = enable_vector_store

        # Create conversation-specific data directory
        if data_dir:
            base_dir = Path(data_dir)
        else:
            base_dir = Path(__file__).parent.parent.parent.parent / "data"

        self.conversation_data_dir = base_dir / "conversations" / conversation_id
        self.conversation_data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the memory system
        self._init_memory_system()

    def _init_memory_system(self):
        """Initialize the VectorMemorySystem with conversation-specific settings."""
        try:
            # Configure vector retrieval if enabled
            vector_config = None
            if self.enable_vector_store and MEMORY_SYSTEM_AVAILABLE:
                vector_config = VectorRetrievalConfig(
                    collection_name=f"conversation_{self.conversation_id}",
                    data_dir=str(self.conversation_data_dir)
                )

            # Initialize the memory system
            self.memory_system = VectorMemorySystem(
                max_windows=self.max_windows,
                data_dir=str(self.conversation_data_dir),
                vector_config=vector_config,
                enable_vector_store=self.enable_vector_store
            )

            self.is_vector_enabled = self.enable_vector_store and MEMORY_SYSTEM_AVAILABLE

        except Exception as e:
            logging.error(f"Failed to initialize VectorMemorySystem: {e}")
            # Fallback to basic memory system
            self.memory_system = VectorMemorySystem(
                max_windows=self.max_windows,
                data_dir=str(self.conversation_data_dir),
                enable_vector_store=False
            )
            self.is_vector_enabled = False

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation memory.

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        self.memory_system.add_conversation(role, content)

    def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to the query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant memories
        """
        if not self.is_vector_enabled:
            return []

        try:
            results = self.memory_system.retrieve_memories(
                query,
                top_k=top_k,
                use_reranker=True,
                similarity_threshold=0.3
            )
            return results
        except Exception as e:
            logging.error(f"Failed to retrieve memories: {e}")
            return []

    def get_conversation_context(self, max_recent: int = 5) -> str:
        """
        Get formatted context for LLM including recent conversations and compressed memories.

        Args:
            max_recent: Maximum number of recent conversations to include

        Returns:
            Formatted context string
        """
        try:
            return self.memory_system.get_context_for_llm(max_recent=max_recent)
        except Exception as e:
            logging.error(f"Failed to get conversation context: {e}")
            return ""

    def get_recent_conversations(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent conversations from the memory window.

        Args:
            n: Number of conversations to return (default: all)

        Returns:
            List of recent conversations
        """
        try:
            return self.memory_system.get_recent_conversations(n)
        except Exception as e:
            logging.error(f"Failed to get recent conversations: {e}")
            return []

    def clear_conversation_window(self) -> None:
        """Clear the conversation window."""
        try:
            self.memory_system.clear_conversation_window()
        except Exception as e:
            logging.error(f"Failed to clear conversation window: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        stats = {
            "conversation_id": self.conversation_id,
            "max_windows": self.max_windows,
            "is_vector_enabled": self.is_vector_enabled,
            "data_dir": str(self.conversation_data_dir)
        }

        if self.is_vector_enabled:
            try:
                vector_stats = self.memory_system.get_vector_stats()
                stats.update(vector_stats)
            except Exception as e:
                logging.error(f"Failed to get vector stats: {e}")

        return stats
"""
Conversation manager for handling multiple conversation sessions.
Each conversation has its own MemorySystem instance for context management.
"""
import uuid
import shutil
from datetime import datetime
from typing import Dict, Optional, List, Any
import logging
from pathlib import Path

try:
    from langchain.schema import HumanMessage
except ImportError:
    # 兼容新版本 LangChain
    from langchain_core.messages import HumanMessage

from sqlalchemy.orm import Session

from ..models import Conversation, Message, SessionLocal
from ..adapters.memory_adapter import MemoryAdapter

logger = logging.getLogger(__name__)


class ConversationSession:
    """Represents a single conversation session with its own memory system."""

    def __init__(self, conversation_id: str, title: Optional[str] = None):
        """
        Initialize a conversation session.

        Args:
            conversation_id: Unique identifier for the conversation
            title: Optional title for the conversation
        """
        self.conversation_id = conversation_id
        self.title = title or "新对话"
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        # Initialize memory adapter for this conversation
        self.memory_adapter = MemoryAdapter(
            conversation_id=conversation_id,
            max_windows=10,  # Keep last 10 messages in context window
            enable_vector_store=True
        )

        # Cache for recent messages
        self._message_cache: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str, save_to_db: bool = True) -> str:
        """
        Add a message to the conversation.

        Args:
            role: 'user' or 'assistant'
            content: Message content
            save_to_db: Whether to save to database

        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())

        # Add to memory adapter
        self.memory_adapter.add_message(role, content)

        # Update cache
        self._message_cache.append({
            "id": message_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })

        # Keep only recent messages in cache (max 10 for consistency with max_windows)
        if len(self._message_cache) > 10:
            self._message_cache = self._message_cache[-10:]

        # Save to database if requested
        if save_to_db:
            try:
                db = SessionLocal()
                try:
                    # Get or create conversation in database
                    conversation = db.query(Conversation).filter(
                        Conversation.id == self.conversation_id
                    ).first()

                    if not conversation:
                        conversation = Conversation(
                            id=self.conversation_id,
                            title=self.title,
                            created_at=self.created_at,
                            updated_at=self.updated_at
                        )
                        db.add(conversation)

                    # Create message record
                    message = Message(
                        id=message_id,
                        conversation_id=self.conversation_id,
                        role=role,
                        content=content,
                        timestamp=datetime.now()
                    )
                    db.add(message)

                    # Update conversation timestamp
                    conversation.updated_at = datetime.now()
                    conversation.title = self._generate_title(content, conversation.title)

                    db.commit()
                    logger.debug(f"Saved message {message_id} to database")
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Failed to save message to database: {e}")

        self.updated_at = datetime.now()
        return message_id

    def _generate_title(self, new_content: str, current_title: str) -> str:
        """Generate or update conversation title based on content."""
        if current_title == "新对话" and new_content:
            # Use first user message as title (truncated)
            title = new_content.strip()
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        return current_title

    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent messages from the conversation.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of recent messages
        """
        # Try to get from cache first
        if self._message_cache:
            return self._message_cache[-limit:]

        # Fallback to database
        try:
            db = SessionLocal()
            try:
                messages = db.query(Message).filter(
                    Message.conversation_id == self.conversation_id
                ).order_by(Message.timestamp.desc()).limit(limit).all()

                return [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp
                    }
                    for msg in reversed(messages)  # Return in chronological order
                ]
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to load messages from database: {e}")
            return []

    def generate_response(self, user_message: str) -> str:
        """
        Generate a response to a user message using RAG.

        Args:
            user_message: User's message

        Returns:
            Assistant's response
        """
        # Add user message to memory
        self.add_message("user", user_message)

        try:
            # Step 1: Retrieve relevant memories
            relevant_memories = self.memory_adapter.retrieve_relevant_memories(
                query=user_message,
                top_k=3
            )

            # Step 2: Build context for LLM
            context = self._build_context(user_message, relevant_memories)

            # Step 3: Call LLM to generate response
            response = self._call_llm(context, user_message)

            # Step 4: Add assistant response to memory
            self.add_message("assistant", response)

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback response
            return "抱歉，我遇到了一些问题处理您的请求。请稍后再试。"

    def _build_context(self, user_message: str, relevant_memories: List[Dict[str, Any]]) -> str:
        """
        Build context for LLM from memories and recent conversations.

        Args:
            user_message: Current user message
            relevant_memories: Retrieved relevant memories

        Returns:
            Formatted context string
        """
        context_parts = []

        # Get conversation context from memory adapter (last 10 messages)
        memory_context = self.memory_adapter.get_conversation_context(max_recent=10)
        if memory_context:
            context_parts.append(memory_context)

        # Add retrieved memories if available
        if relevant_memories:
            context_parts.append("\n## 相关记忆")
            for i, memory in enumerate(relevant_memories[:3], 1):
                text = memory.get("text", "")[:200]
                if text:
                    context_parts.append(f"{i}. {text}...")

        # Add current user message
        context_parts.append(f"\n## 当前对话\n用户: {user_message}")

        return "\n".join(context_parts)

    def _call_llm(self, context: str, user_message: str) -> str:
        """
        Call LLM to generate response using the memory system's LLM.

        Args:
            context: Formatted context containing recent conversations and retrieved memories
            user_message: Current user message

        Returns:
            Assistant's response
        """
        try:
            # Get LLM from memory adapter's memory system
            # The memory_system is a VectorMemorySystem which inherits from MemorySystem
            memory_system = self.memory_adapter.memory_system

            # Check if LLM is available
            if not hasattr(memory_system, 'llm') or memory_system.llm is None:
                logger.warning("LLM not available in memory system, using fallback response")
                return f"基于上下文，我已经理解了您的请求：'{user_message[:50]}...'。这是一个使用RAG记忆系统的回复。"

            # Build prompt for LLM
            prompt = f"""你是一个智能助手，使用RAG记忆系统来提供个性化的对话体验。

以下是相关的对话历史和相关记忆：
{context}

请基于以上信息，回应用户的最新消息。你的回复应该：
1. 自然、友好、有帮助
2. 如果相关记忆提供了用户个人信息，可以个性化地使用这些信息
3. 保持对话的连贯性
4. 如果记忆中有相关信息，可以参考但不要直接复制

用户最新消息：{user_message}

请用中文回复："""

            # Call LLM
            try:
                # Check if it's a ChatOpenAI instance
                from langchain_openai import ChatOpenAI
                if isinstance(memory_system.llm, ChatOpenAI):
                    # Use the LLM to generate response
                    response = memory_system.llm.invoke([HumanMessage(content=prompt)])

                    if hasattr(response, 'content'):
                        return response.content
                    elif isinstance(response, str):
                        return response
                    elif hasattr(response, 'text'):
                        return response.text
                    else:
                        logger.error(f"Unexpected LLM response format: {type(response)}")
                        return f"基于上下文，我已经理解了您的请求：'{user_message[:50]}...'。这是一个使用RAG记忆系统的回复。"
                else:
                    # For other LLM types, try to call with the standard interface
                    response = memory_system.llm.invoke(prompt)
                    if isinstance(response, str):
                        return response
                    elif hasattr(response, 'content'):
                        return response.content
                    else:
                        logger.error(f"Unsupported LLM type: {type(memory_system.llm)}")
                        return f"基于上下文，我已经理解了您的请求：'{user_message[:50]}...'。这是一个使用RAG记忆系统的回复。"

            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                # Fallback to simple response
                return f"我已经理解了您的请求：'{user_message}'。让我思考一下如何回复..."

        except Exception as e:
            logger.error(f"Error in _call_llm: {e}")
            # Fallback response
            return f"基于上下文，我已经理解了您的请求：'{user_message[:50]}...'。这是一个使用RAG记忆系统的回复。"

    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "memory_stats": self.memory_adapter.get_stats()
        }


class ConversationManager:
    """Manages multiple conversation sessions."""

    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}

    def create_conversation(self, title: Optional[str] = None) -> str:
        """
        Create a new conversation.

        Args:
            title: Optional title for the conversation

        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        session = ConversationSession(conversation_id, title)
        self.sessions[conversation_id] = session

        # 保存到数据库
        try:
            db = SessionLocal()
            try:
                conversation = Conversation(
                    id=conversation_id,
                    title=title or "新对话",
                    created_at=session.created_at,
                    updated_at=session.updated_at
                )
                db.add(conversation)
                db.commit()
                logger.info(f"Saved new conversation to database: {conversation_id}")
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to save conversation to database: {e}")
                # 继续执行，对话已在内存中创建
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Database error when creating conversation: {e}")
            # 继续执行，对话已在内存中创建

        logger.info(f"Created new conversation: {conversation_id}")
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[ConversationSession]:
        """
        Get a conversation session.

        Args:
            conversation_id: Conversation ID

        Returns:
            ConversationSession if found, None otherwise
        """
        # Check in-memory cache
        if conversation_id in self.sessions:
            return self.sessions[conversation_id]

        # Try to load from database
        try:
            db = SessionLocal()
            try:
                conversation = db.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()

                if conversation:
                    # Create session from database
                    session = ConversationSession(conversation_id, conversation.title)
                    session.created_at = conversation.created_at
                    session.updated_at = conversation.updated_at

                    # Load recent messages into cache
                    messages = db.query(Message).filter(
                        Message.conversation_id == conversation_id
                    ).order_by(Message.timestamp).all()

                    for msg in messages:
                        session._message_cache.append({
                            "id": msg.id,
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.timestamp
                        })

                    self.sessions[conversation_id] = session
                    logger.info(f"Loaded conversation from database: {conversation_id}")
                    return session
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to load conversation from database: {e}")

        return None

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            True if deleted, False otherwise
        """
        try:
            # Remove from memory
            if conversation_id in self.sessions:
                del self.sessions[conversation_id]

            # Delete conversation data directory
            try:
                # 确定对话数据目录路径（与MemoryAdapter中一致）
                base_dir = Path(__file__).parent.parent.parent.parent / "data"
                conversation_data_dir = base_dir / "conversations" / conversation_id
                if conversation_data_dir.exists():
                    shutil.rmtree(conversation_data_dir)
                    logger.info(f"Deleted conversation data directory: {conversation_data_dir}")
            except Exception as e:
                logger.error(f"Failed to delete conversation data directory: {e}")
                # 继续执行，不因为文件删除失败而停止

            # Delete from database
            db = SessionLocal()
            try:
                conversation = db.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()

                if conversation:
                    db.delete(conversation)
                    db.commit()
                    logger.info(f"Deleted conversation: {conversation_id}")
                    return True
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")

        return False

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List all conversations.

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversation summaries
        """
        try:
            db = SessionLocal()
            try:
                conversations = db.query(Conversation).order_by(
                    Conversation.updated_at.desc()
                ).limit(limit).all()

                return [
                    {
                        "id": conv.id,
                        "title": conv.title,
                        "created_at": conv.created_at.isoformat(),
                        "updated_at": conv.updated_at.isoformat(),
                        "message_count": len(conv.messages)
                    }
                    for conv in conversations
                ]
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return []

    def cleanup_inactive_sessions(self, max_age_hours: int = 24):
        """
        Clean up inactive conversation sessions from memory.

        Args:
            max_age_hours: Maximum age in hours before cleaning up
        """
        current_time = datetime.now()
        to_remove = []

        for conv_id, session in self.sessions.items():
            age_hours = (current_time - session.updated_at).total_seconds() / 3600
            if age_hours > max_age_hours:
                to_remove.append(conv_id)

        for conv_id in to_remove:
            del self.sessions[conv_id]
            logger.info(f"Cleaned up inactive session: {conv_id}")

        logger.info(f"Cleaned up {len(to_remove)} inactive sessions")


# Global conversation manager instance
conversation_manager = ConversationManager()
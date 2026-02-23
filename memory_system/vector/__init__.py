"""
向量检索模块
提供基于 Milvus + BGE-M3 + BM25 的混合检索功能。
"""

from .config import VectorRetrievalConfig, DEFAULT_VECTOR_CONFIG
from .milvus_store import MilvusStore
from .embedding_service import EmbeddingService, create_default_embedding_service
from .retriever import HybridRetriever
from .reranker import Reranker, create_default_reranker
from .chunking import ConversationChunker, create_default_chunker
from .memory_system import VectorMemorySystem, create_default_vector_memory_system

__all__ = [
    "VectorRetrievalConfig",
    "DEFAULT_VECTOR_CONFIG",
    "MilvusStore",
    "EmbeddingService",
    "create_default_embedding_service",
    "HybridRetriever",
    "Reranker",
    "create_default_reranker",
    "ConversationChunker",
    "create_default_chunker",
    "VectorMemorySystem",
    "create_default_vector_memory_system",
]
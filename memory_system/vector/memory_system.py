"""
增强的记忆系统模块
集成向量检索功能的MemorySystem扩展。
"""

import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import time

from ..core.memory import MemorySystem
from langchain_openai import ChatOpenAI

from .config import VectorRetrievalConfig, DEFAULT_VECTOR_CONFIG
from .milvus_store import MilvusStore
from .embedding_service import EmbeddingService
from .retriever import HybridRetriever
from .reranker import Reranker
from .chunking import ConversationChunker

logger = logging.getLogger(__name__)


class VectorMemorySystem(MemorySystem):
    """
    增强的记忆系统，集成向量检索功能。

    在原有MemorySystem功能基础上，增加：
    1. 对话自动分块和向量化存储
    2. 混合检索（0.7向量 + 0.3 BM25）
    3. Rerank重排序
    4. Milvus向量数据库存储

    使用流程：
    >>> from memory_system.vector import VectorMemorySystem
    >>> memory = VectorMemorySystem()
    >>> memory.add_conversation("user", "用户输入")  # 自动分块和存储
    >>> results = memory.retrieve_memories("查询文本")  # 混合检索 + rerank
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        max_windows: int = 10,
        data_dir: Optional[Path] = None,
        vector_config: Optional[VectorRetrievalConfig] = None,
        enable_vector_store: bool = True
    ):
        """
        初始化增强记忆系统。

        Args:
            llm: 聊天LLM实例，用于对话压缩和信息提取
            max_windows: 对话窗口最大容量
            data_dir: 数据存储目录
            vector_config: 向量检索配置
            enable_vector_store: 是否启用向量存储
        """
        # 初始化父类（原始MemorySystem）
        super().__init__(llm=llm, max_windows=max_windows, data_dir=data_dir)

        self.vector_config = vector_config or DEFAULT_VECTOR_CONFIG
        self.enable_vector_store = enable_vector_store

        # 向量检索组件
        self.milvus_store: Optional[MilvusStore] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.retriever: Optional[HybridRetriever] = None
        self.reranker: Optional[Reranker] = None
        self.chunker: Optional[ConversationChunker] = None

        # 初始化向量检索系统
        if self.enable_vector_store:
            self._init_vector_system()

        logger.info("增强记忆系统初始化完成")

    def _init_vector_system(self):
        """初始化向量检索系统"""
        try:
            # 初始化组件
            self.chunker = ConversationChunker(self.vector_config)
            self.embedding_service = EmbeddingService(self.vector_config)
            self.milvus_store = MilvusStore(self.vector_config)

            # 检查embedding服务是否可用
            if not self.embedding_service.is_available():
                logger.warning("Embedding服务不可用，向量检索功能将受限")

            # 初始化检索器和reranker
            if self.milvus_store and self.embedding_service:
                self.retriever = HybridRetriever(
                    milvus_store=self.milvus_store,
                    embedding_service=self.embedding_service,
                    config=self.vector_config
                )
                self.reranker = Reranker(self.vector_config)

                logger.info("向量检索系统初始化成功")
            else:
                logger.warning("向量检索系统初始化不完整")

            # 加载现有记忆到BM25索引
            self._load_existing_memories_to_index()

        except Exception as e:
            logger.error(f"初始化向量检索系统失败: {e}")
            self.enable_vector_store = False

    def _load_existing_memories_to_index(self):
        """加载现有记忆到BM25索引"""
        if not self.retriever or not self.milvus_store:
            return

        try:
            # 从Milvus获取现有记忆
            # 这里需要实现一个方法从Milvus读取所有记忆
            # 目前先跳过，等待有记忆时再索引
            logger.debug("BM25索引将在添加新记忆时构建")
        except Exception as e:
            logger.error(f"加载现有记忆到索引失败: {e}")

    def add_conversation(self, role: str, content: str):
        """
        添加对话到记忆系统（重写父类方法）。

        除了原始功能外，还会：
        1. 将对话分块
        2. 生成embedding
        3. 存储到Milvus
        4. 索引到BM25

        Args:
            role: 对话角色
            content: 对话内容
        """
        # 调用父类方法（原始功能）
        super().add_conversation(role, content)

        # 向量存储功能
        if self.enable_vector_store and role == "user":  # 目前只存储用户对话
            self._store_to_vector_db(role, content)

    def _store_to_vector_db(self, role: str, content: str):
        """存储对话到向量数据库"""
        if not all([self.chunker, self.embedding_service, self.milvus_store, self.retriever]):
            return

        try:
            # 获取最近添加的对话（刚由父类添加）
            recent_conv = self.get_recent_conversations(1)
            if not recent_conv:
                return

            conversation = recent_conv[0]  # 最近添加的对话
            timestamp = conversation.get("timestamp", "")

            # 分块
            chunks = self.chunker.chunk_conversation(
                conversation=conversation,
                conversation_id=f"conv_{int(time.time())}"
            )

            if not chunks:
                return

            # 批量生成embedding
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.batch_embed(chunk_texts)

            # 准备存储数据
            memory_chunks = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if not embedding:  # 跳过embedding失败的块
                    continue

                memory_chunks.append({
                    "id": f"{int(time.time())}_{i}",
                    "text": chunk["text"],
                    "embedding": embedding,
                    "metadata": {
                        **chunk["metadata"],
                        "timestamp": int(time.time()),
                        "role": role,
                        "conversation_id": f"conv_{int(time.time())}",
                        "chunk_index": i
                    }
                })

            # 存储到Milvus
            if memory_chunks:
                inserted_ids = self.milvus_store.insert_memory_chunks(memory_chunks)

                # 索引到BM25
                self.retriever.index_memory_chunks(memory_chunks)

                logger.info(f"已存储 {len(inserted_ids)} 个记忆分块到向量数据库")

        except Exception as e:
            logger.error(f"存储到向量数据库失败: {e}")

    def retrieve_memories(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: bool = True,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        检索相关记忆。

        Args:
            query: 查询文本
            top_k: 返回结果数量，默认使用配置值
            use_reranker: 是否使用reranker重排序
            similarity_threshold: 相似度阈值，默认使用配置值

        Returns:
            相关记忆列表，每个包含：
                - id: 记忆ID
                - text: 记忆文本
                - score: 相关性分数
                - metadata: 元数据
        """
        if not self.enable_vector_store or not self.retriever:
            logger.warning("向量检索未启用，使用原始方法")
            # 回退到原始方法：返回最近对话
            recent = self.get_recent_conversations(top_k or 5)
            return [
                {
                    "id": f"recent_{i}",
                    "text": conv["content"],
                    "score": 1.0 - (i * 0.1),  # 简单衰减分数
                    "metadata": {
                        "timestamp": conv.get("timestamp", ""),
                        "role": conv.get("role", ""),
                        "source": "recent_conversation"
                    }
                }
                for i, conv in enumerate(recent)
            ]

        try:
            # 混合检索
            retrieved = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                use_reranker=False  # 先不使用reranker，后面单独处理
            )

            if not retrieved:
                return []

            # Rerank重排序
            if use_reranker and self.reranker:
                reranked = self.reranker.rerank(
                    query=query,
                    candidates=retrieved,
                    top_n=self.vector_config.rerank_top_n
                )
                return reranked
            else:
                # 返回Top-N
                top_n = self.vector_config.rerank_top_n
                return sorted(retrieved, key=lambda x: x.get("score", 0), reverse=True)[:top_n]

        except Exception as e:
            logger.error(f"检索记忆失败: {e}")
            return []

    def search_similar_conversations(
        self,
        query: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        搜索相似的对话（高级检索）。

        Args:
            query: 查询文本
            max_results: 最大返回结果数

        Returns:
            相似对话列表
        """
        # 先检索相关记忆分块
        memories = self.retrieve_memories(query, top_k=max_results * 3)

        # 按对话ID分组
        conversations_by_id = {}
        for memory in memories:
            conv_id = memory.get("metadata", {}).get("conversation_id", "unknown")
            if conv_id not in conversations_by_id:
                conversations_by_id[conv_id] = {
                    "conversation_id": conv_id,
                    "memories": [],
                    "max_score": 0,
                    "total_score": 0
                }

            conv_data = conversations_by_id[conv_id]
            conv_data["memories"].append(memory)
            score = memory.get("score", 0)
            conv_data["max_score"] = max(conv_data["max_score"], score)
            conv_data["total_score"] += score

        # 构建对话结果
        results = []
        for conv_id, conv_data in conversations_by_id.items():
            # 获取对话文本（合并相关记忆）
            memories_text = "\n".join([m.get("text", "") for m in conv_data["memories"][:3]])

            results.append({
                "conversation_id": conv_id,
                "text": memories_text[:500] + "..." if len(memories_text) > 500 else memories_text,
                "score": conv_data["max_score"],  # 使用最高分作为对话分数
                "memory_count": len(conv_data["memories"]),
                "memories": conv_data["memories"][:3]  # 前3个相关记忆
            })

        # 按分数排序
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:max_results]

    def get_vector_stats(self) -> Dict[str, Any]:
        """
        获取向量存储统计信息。

        Returns:
            统计信息字典
        """
        if not self.enable_vector_store:
            return {"enabled": False}

        stats = {
            "enabled": True,
            "milvus_available": self.milvus_store is not None,
            "embedding_available": self.embedding_service.is_available() if self.embedding_service else False,
            "retriever_available": self.retriever is not None,
            "reranker_available": self.reranker.is_available() if self.reranker else False,
        }

        # Milvus统计
        if self.milvus_store:
            milvus_stats = self.milvus_store.get_collection_stats()
            stats["milvus_stats"] = milvus_stats

        # 检索器统计
        if self.retriever:
            retriever_stats = self.retriever.get_index_stats()
            stats["retriever_stats"] = retriever_stats

        return stats

    def clear_vector_store(self) -> bool:
        """
        清空向量存储。

        Returns:
            是否成功
        """
        if not self.enable_vector_store or not self.milvus_store:
            return False

        try:
            success = self.milvus_store.clear_collection()
            if success and self.retriever:
                self.retriever.clear_index()

            logger.warning("已清空向量存储")
            return success
        except Exception as e:
            logger.error(f"清空向量存储失败: {e}")
            return False

    def export_memories(
        self,
        output_file: Optional[Path] = None,
        format: str = "json"
    ) -> Optional[str]:
        """
        导出记忆数据。

        Args:
            output_file: 输出文件路径
            format: 输出格式，支持"json"或"text"

        Returns:
            如果output_file为None，返回导出内容；否则返回None
        """
        # TODO: 实现记忆导出功能
        logger.warning("记忆导出功能尚未实现")
        return None

    def __del__(self):
        """析构函数，清理资源"""
        try:
            if self.milvus_store:
                self.milvus_store.disconnect()
        except:
            pass


# 便捷函数：创建默认增强记忆系统
def create_default_vector_memory_system() -> VectorMemorySystem:
    """创建使用默认配置的增强记忆系统"""
    return VectorMemorySystem()
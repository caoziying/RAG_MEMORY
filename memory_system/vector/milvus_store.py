"""
Milvus向量存储封装
提供对话记忆的向量化存储和检索功能。
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

try:
    from pymilvus import (
        connections,
        FieldSchema,
        CollectionSchema,
        DataType,
        Collection,
        utility,
        MilvusException
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    # 创建虚拟类以便类型提示
    class Collection: pass
    class utility: pass

from .config import VectorRetrievalConfig

logger = logging.getLogger(__name__)


class MilvusStore:
    """
    Milvus向量存储管理器。

    存储对话片段的embedding向量和相关元数据：
    - id: 唯一标识符
    - text: 对话文本内容
    - embedding: 向量（BGE-M3生成）
    - metadata: 元数据（时间戳、角色、对话ID等）
    - timestamp: 时间戳（用于按时间排序）
    """

    def __init__(self, config: Optional[VectorRetrievalConfig] = None):
        """
        初始化Milvus存储。

        Args:
            config: 向量检索配置，如果为None则使用默认配置
        """
        self.config = config or VectorRetrievalConfig()
        self.collection: Optional[Collection] = None
        self._connected = False

        # 检查Milvus是否可用
        if not MILVUS_AVAILABLE:
            logger.warning("pymilvus未安装，Milvus功能将不可用。")
            logger.warning("请安装：pip install pymilvus")
            return

        # 连接到Milvus
        self.connect()

        # 初始化集合
        self._init_collection()

    def connect(self):
        """连接到Milvus服务"""
        if not MILVUS_AVAILABLE:
            return

        try:
            connections.connect(
                alias="default",
                host=self.config.milvus_host,
                port=self.config.milvus_port
            )
            self._connected = True
            logger.info(f"已连接到Milvus: {self.config.milvus_host}:{self.config.milvus_port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            self._connected = False

    def disconnect(self):
        """断开Milvus连接"""
        if not MILVUS_AVAILABLE or not self._connected:
            return

        try:
            connections.disconnect("default")
            self._connected = False
            logger.info("已断开Milvus连接")
        except Exception as e:
            logger.error(f"断开Milvus连接失败: {e}")

    def _init_collection(self):
        """初始化Milvus集合（collection）"""
        if not MILVUS_AVAILABLE or not self._connected:
            return

        try:
            # 检查集合是否存在
            if utility.has_collection(self.config.milvus_collection_name):
                self.collection = Collection(self.config.milvus_collection_name)
                logger.info(f"已加载现有集合: {self.config.milvus_collection_name}")
            else:
                # 创建新集合
                self._create_collection()
                logger.info(f"已创建新集合: {self.config.milvus_collection_name}")

            # 加载集合到内存
            self.collection.load()
            logger.debug("集合已加载到内存")

        except Exception as e:
            logger.error(f"初始化集合失败: {e}")
            self.collection = None

    def _create_collection(self):
        """创建Milvus集合"""
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config.embedding_dim
            ),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="conversation_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
        ]

        # 创建schema
        schema = CollectionSchema(
            fields=fields,
            description="对话记忆向量存储"
        )

        # 创建集合
        self.collection = Collection(
            name=self.config.milvus_collection_name,
            schema=schema
        )

        # 创建索引
        self._create_index()

    def _create_index(self):
        """为embedding字段创建索引"""
        if not self.collection:
            return

        index_params = {
            "metric_type": self.config.milvus_metric_type,
            "index_type": self.config.milvus_index_type,
            "params": {"nlist": self.config.milvus_nlist}
        }

        try:
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            logger.debug("已创建embedding字段索引")
        except Exception as e:
            logger.error(f"创建索引失败: {e}")

    def insert_memory_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        插入记忆分块到Milvus。

        Args:
            chunks: 记忆分块列表，每个分块包含：
                - text: 文本内容
                - embedding: 向量列表
                - metadata: 元数据字典（包含timestamp, role, conversation_id, chunk_index等）

        Returns:
            插入成功的分块ID列表
        """
        if not self.collection or not chunks:
            return []

        try:
            # 准备插入数据
            ids, texts, embeddings, timestamps = [], [], [], []
            roles, conversation_ids, chunk_indices = [], [], []

            for chunk in chunks:
                # 生成唯一ID
                chunk_id = str(uuid4())
                ids.append(chunk_id)

                # 提取数据
                texts.append(chunk.get("text", ""))
                embeddings.append(chunk.get("embedding", []))

                # 从metadata提取或直接获取
                metadata = chunk.get("metadata", {})
                timestamps.append(metadata.get("timestamp", chunk.get("timestamp", 0)))
                roles.append(metadata.get("role", chunk.get("role", "unknown")))
                conversation_ids.append(metadata.get("conversation_id", chunk.get("conversation_id", "")))
                chunk_indices.append(metadata.get("chunk_index", chunk.get("chunk_index", 0)))

            # 插入数据
            insert_data = [
                ids,
                texts,
                embeddings,
                timestamps,
                roles,
                conversation_ids,
                chunk_indices
            ]

            mr = self.collection.insert(insert_data)
            self.collection.flush()

            logger.info(f"已插入 {len(chunks)} 个记忆分块")
            return ids

        except Exception as e:
            logger.error(f"插入记忆分块失败: {e}")
            return []

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似的记忆分块。

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量，默认使用配置中的top_k
            filter_expr: Milvus过滤表达式

        Returns:
            相似记忆分块列表，每个包含：
                - id: 分块ID
                - text: 文本内容
                - score: 相似度分数
                - metadata: 元数据
        """
        if not self.collection:
            return []

        top_k = top_k or self.config.top_k

        try:
            # 准备搜索参数
            search_params = {
                "metric_type": self.config.milvus_metric_type,
                "params": {"nprobe": 10}  # IVF索引搜索参数
            }

            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["text", "timestamp", "role", "conversation_id", "chunk_index"]
            )

            # 格式化结果
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": hit.id,
                        "text": hit.entity.get("text", ""),
                        "score": hit.score,
                        "metadata": {
                            "timestamp": hit.entity.get("timestamp", 0),
                            "role": hit.entity.get("role", "unknown"),
                            "conversation_id": hit.entity.get("conversation_id", ""),
                            "chunk_index": hit.entity.get("chunk_index", 0)
                        }
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"搜索相似记忆失败: {e}")
            return []

    def delete_by_ids(self, ids: List[str]) -> int:
        """
        根据ID删除记忆分块。

        Args:
            ids: 要删除的分块ID列表

        Returns:
            成功删除的数量
        """
        if not self.collection or not ids:
            return 0

        try:
            # 构建删除表达式
            id_str = ", ".join([f"'{id}'" for id in ids])
            expr = f"id in [{id_str}]"

            # 执行删除
            result = self.collection.delete(expr)
            self.collection.flush()

            deleted_count = len(result.delete_ids) if hasattr(result, 'delete_ids') else 0
            logger.info(f"已删除 {deleted_count} 个记忆分块")
            return deleted_count

        except Exception as e:
            logger.error(f"删除记忆分块失败: {e}")
            return 0

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self.collection:
            return {"error": "集合未初始化"}

        try:
            stats = {
                "name": self.config.milvus_collection_name,
                "num_entities": self.collection.num_entities,
                "is_loaded": self.collection.is_loaded if hasattr(self.collection, 'is_loaded') else False,
                "indexes": []
            }

            # 获取索引信息
            try:
                indexes = self.collection.indexes
                if indexes is not None:
                    # 确保indexes是可迭代的
                    if hasattr(indexes, '__iter__'):
                        for index in indexes:
                            # 安全地获取索引属性，兼容不同pymilvus版本
                            index_info = {
                                "field_name": getattr(index, 'field_name', 'unknown'),
                                "index_type": getattr(index, 'index_type', 'unknown'),
                                "metric_type": getattr(index, 'metric_type', 'unknown')
                            }
                            stats["indexes"].append(index_info)
                    else:
                        logger.warning(f"indexes属性不是可迭代对象: {type(indexes)}")
                else:
                    logger.debug("没有找到索引信息")
            except Exception as e:
                logger.warning(f"获取索引信息失败: {e}")

            return stats

        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """清空整个集合"""
        if not self.collection:
            return False

        try:
            # 删除所有数据
            self.collection.drop()
            logger.warning(f"已清空集合: {self.config.milvus_collection_name}")

            # 重新创建集合
            self._init_collection()
            return True

        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False

    def __del__(self):
        """析构函数，确保断开连接"""
        self.disconnect()
"""
向量检索配置模块
定义混合检索的超参数、模型配置和Milvus连接设置。
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class VectorRetrievalConfig:
    """向量检索配置"""

    # === 混合检索权重 ===
    vector_weight: float = 0.7      # 向量检索权重
    bm25_weight: float = 0.3       # BM25检索权重

    # === 召回控制 ===
    top_k: int = 100               # 召回最大数据条数
    rerank_top_n: int = 5          # 重排序后返回条数
    similarity_threshold: float = 0.3  # 相似度阈值

    # === 分块策略 ===
    chunk_size: int = 512          # 分块大小（tokens）
    chunk_overlap: int = 50        # 分块重叠大小

    # === 模型配置 ===
    embedding_model: str = "bge-m3"  # Embedding模型名称
    rerank_model: str = "bge-reranker-large"  # 重排序模型名称

    # === API配置 ===
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_model: Optional[str] = None  # 聊天模型，与embedding/rerank分开

    # === Milvus配置 ===
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "conversation_memories"

    # === 索引配置 ===
    milvus_index_type: str = "IVF_FLAT"
    milvus_metric_type: str = "COSINE"
    milvus_nlist: int = 1024  # IVF索引的聚类中心数

    def __post_init__(self):
        """初始化配置，从环境变量获取默认值"""
        # 从环境变量获取API配置
        if self.api_key is None:
            self.api_key = os.getenv("MY_API_KEY", "")

        if self.api_base is None:
            self.api_base = os.getenv("MY_API_BASE", "https://api.chat.csu.edu.cn/v1")

        if self.api_model is None:
            self.api_model = os.getenv("MY_MODEL", "Qwen3-32B-FP8")

        # Milvus配置
        if os.getenv("MILVUS_HOST"):
            self.milvus_host = os.getenv("MILVUS_HOST")
        if os.getenv("MILVUS_PORT"):
            self.milvus_port = int(os.getenv("MILVUS_PORT"))

        # 验证权重总和为1
        total_weight = self.vector_weight + self.bm25_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"权重总和应为1.0，当前为{total_weight}")

    @property
    def embedding_dim(self) -> int:
        """获取embedding维度（BGE-M3为1024）"""
        # BGE-M3模型embedding维度为1024
        if "bge-m3" in self.embedding_model.lower():
            return 1024
        # 其他模型默认768
        return 768

    @classmethod
    def from_env(cls) -> "VectorRetrievalConfig":
        """从环境变量创建配置实例"""
        return cls()


# 默认配置实例
DEFAULT_VECTOR_CONFIG = VectorRetrievalConfig()
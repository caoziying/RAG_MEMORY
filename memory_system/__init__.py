"""
记忆系统 - 即插即用的对话记忆管理系统

主要功能：
1. 维护固定大小的对话窗口（最近 N 轮对话）
2. 窗口满时自动使用 LLM 压缩旧对话为摘要
3. 提取用户个人信息并存储到 user.md
4. 每天生成独立的聊天日志文件（YYYY-MM-DD.md）
5. （可选）向量检索：Milvus + BGE-M3 + BM25 混合检索 + Rerank

基础使用（无需向量检索）：
    >>> from memory_system import MemorySystem, create_default_memory_system
    >>> from langchain_openai import ChatOpenAI
    >>>
    >>> # 方法1：使用默认配置（需要环境变量）
    >>> memory = create_default_memory_system()
    >>>
    >>> # 方法2：自定义 LLM 实例
    >>> llm = ChatOpenAI(
    ...     model="Qwen3-32B-FP8",
    ...     api_key="your-api-key",
    ...     base_url="https://api.chat.csu.edu.cn/v1"
    ... )
    >>> memory = MemorySystem(llm=llm, max_windows=10)
    >>>
    >>> # 添加对话
    >>> memory.add_conversation("user", "我叫李四，喜欢编程和音乐。")
    >>> memory.add_conversation("assistant", "你好李四！你的兴趣很广泛。")
    >>>
    >>> # 获取最近对话
    >>> recent = memory.get_recent_conversations(3)
    >>>
    >>> # 获取 LLM 上下文
    >>> context = memory.get_context_for_llm()

向量检索使用（需要额外依赖）：
    >>> from memory_system import VectorMemorySystem, create_default_vector_memory_system
    >>>
    >>> # 需要先启动 Milvus Docker 服务
    >>> # cd docker && docker-compose up -d
    >>>
    >>> # 创建增强记忆系统（自动使用向量存储）
    >>> memory = create_default_vector_memory_system()
    >>>
    >>> # 添加对话（自动分块和存储到向量数据库）
    >>> memory.add_conversation("user", "用户输入内容")
    >>>
    >>> # 检索相关记忆（混合检索 + rerank）
    >>> results = memory.retrieve_memories("查询文本")
    >>> print(f"检索到 {len(results)} 条相关记忆")
    >>>
    >>> # 高级搜索：按对话分组
    >>> conversations = memory.search_similar_conversations("查询文本")
"""

from .core.memory import MemorySystem, create_default_memory_system
from .core.config import (
    MAX_WINDOWS,
    DATA_DIR,
    LOG_DIR,
    USER_INFO_FILE,
    COMPRESSED_MEMORY_FILE,
    get_daily_log_path,
    init_directories
)
from .core.utils import setup_logging, ensure_directory, read_file_safe, write_file_safe

# 向量检索模块（可选功能，需要额外依赖）
try:
    from .vector import (
        VectorMemorySystem,
        create_default_vector_memory_system,
        VectorRetrievalConfig,
        MilvusStore,
        EmbeddingService,
        HybridRetriever,
        Reranker,
        ConversationChunker
    )
    VECTOR_AVAILABLE = True
except ImportError as e:
    VECTOR_AVAILABLE = False
    # 创建虚拟类以便类型提示
    class VectorMemorySystem:
        def __init__(self, *args, **kwargs):
            raise ImportError("向量检索模块需要额外依赖。请安装：pip install pymilvus")
    class VectorRetrievalConfig: pass
    class MilvusStore: pass
    class EmbeddingService: pass
    class HybridRetriever: pass
    class Reranker: pass
    class ConversationChunker: pass
    create_default_vector_memory_system = None

__version__ = "1.0.0"
__author__ = "记忆系统开发者"

__all__ = [
    # 主类
    "MemorySystem",
    "create_default_memory_system",

    # 配置
    "MAX_WINDOWS",
    "DATA_DIR",
    "LOG_DIR",
    "USER_INFO_FILE",
    "COMPRESSED_MEMORY_FILE",
    "get_daily_log_path",
    "init_directories",

    # 工具函数
    "setup_logging",
    "ensure_directory",
    "read_file_safe",
    "write_file_safe",

    # 向量检索模块（可选）
    "VectorMemorySystem",
    "create_default_vector_memory_system",
    "VectorRetrievalConfig",
    "MilvusStore",
    "EmbeddingService",
    "HybridRetriever",
    "Reranker",
    "ConversationChunker",
    "VECTOR_AVAILABLE",
]
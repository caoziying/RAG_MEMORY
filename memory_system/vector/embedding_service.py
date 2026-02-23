"""
Embedding服务模块
使用BGE-M3模型生成文本的向量表示。
"""

import logging
from typing import List, Optional, Union
import time

try:
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_EMBEDDINGS_AVAILABLE = True
except ImportError:
    LANGCHAIN_EMBEDDINGS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .config import VectorRetrievalConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding服务，使用BGE-M3模型生成文本向量。

    支持两种调用方式：
    1. 使用LangChain的OpenAIEmbeddings（推荐）
    2. 使用OpenAI客户端直接调用
    """

    def __init__(self, config: Optional[VectorRetrievalConfig] = None):
        """
        初始化Embedding服务。

        Args:
            config: 向量检索配置，如果为None则使用默认配置
        """
        self.config = config or VectorRetrievalConfig()
        self.embeddings = None
        self.openai_client = None
        self._init_embedding_client()

    def _init_embedding_client(self):
        """初始化embedding客户端"""
        # 优先使用LangChain的OpenAIEmbeddings
        if LANGCHAIN_EMBEDDINGS_AVAILABLE:
            try:
                self.embeddings = OpenAIEmbeddings(
                    model=self.config.embedding_model,
                    openai_api_key=self.config.api_key,
                    openai_api_base=self.config.api_base,
                )
                logger.info(f"使用LangChain OpenAIEmbeddings，模型: {self.config.embedding_model}")
                return
            except Exception as e:
                logger.warning(f"初始化LangChain Embeddings失败: {e}")

        # 回退到OpenAI客户端
        if OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.api_base
                )
                logger.info(f"使用OpenAI客户端，模型: {self.config.embedding_model}")
            except Exception as e:
                logger.error(f"初始化OpenAI客户端失败: {e}")
                self.openai_client = None
        else:
            logger.warning("既没有LangChain也没有OpenAI客户端可用")

    def embed_text(self, text: str, retry_count: int = 3) -> Optional[List[float]]:
        """
        对单个文本生成embedding。

        Args:
            text: 输入文本
            retry_count: 重试次数

        Returns:
            embedding向量列表，失败返回None
        """
        return self.embed_texts([text], retry_count)[0] if text else None

    def embed_texts(self, texts: List[str], retry_count: int = 3) -> List[Optional[List[float]]]:
        """
        对多个文本生成embedding。

        Args:
            texts: 输入文本列表
            retry_count: 重试次数

        Returns:
            embedding向量列表，失败的文本对应None
        """
        if not texts:
            return []

        # 过滤空文本
        valid_texts = []
        text_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                text_indices.append(i)

        if not valid_texts:
            return [None] * len(texts)

        results = [None] * len(texts)

        for attempt in range(retry_count):
            try:
                # 尝试使用LangChain Embeddings
                if self.embeddings is not None:
                    embeddings_list = self.embeddings.embed_documents(valid_texts)
                    # 将结果放回正确位置
                    for idx, emb in zip(text_indices, embeddings_list):
                        results[idx] = emb
                    break

                # 尝试使用OpenAI客户端
                elif self.openai_client is not None:
                    # OpenAI兼容的embedding调用
                    response = self.openai_client.embeddings.create(
                        model=self.config.embedding_model,
                        input=valid_texts
                    )

                    # 处理响应
                    for idx, data in zip(text_indices, response.data):
                        results[idx] = data.embedding
                    break

                else:
                    logger.error("没有可用的embedding客户端")
                    break

            except Exception as e:
                logger.warning(f"Embedding生成失败（尝试 {attempt + 1}/{retry_count}）: {e}")
                if attempt < retry_count - 1:
                    time.sleep(1 * (attempt + 1))  # 指数退避
                else:
                    logger.error(f"Embedding生成最终失败: {e}")

        # 记录统计
        success_count = sum(1 for r in results if r is not None)
        if success_count > 0:
            logger.debug(f"成功生成 {success_count}/{len(texts)} 个文本的embedding")
        else:
            logger.error(f"所有embedding生成失败")

        return results

    def get_embedding_dimension(self) -> int:
        """
        获取embedding向量的维度。

        Returns:
            embedding维度
        """
        # 尝试获取实际维度
        if self.embeddings is not None:
            # 对于BGE-M3，维度是1024
            if "bge-m3" in self.config.embedding_model.lower():
                return 1024
            # 尝试通过测试获取
            try:
                test_embedding = self.embed_text("test")
                if test_embedding:
                    return len(test_embedding)
            except:
                pass

        # 返回配置中的维度
        return self.config.embedding_dim

    def batch_embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        批量生成embedding，支持大文本列表。

        Args:
            texts: 输入文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度

        Returns:
            成功生成的embedding列表，失败的文本对应空列表
        """
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            if show_progress:
                logger.info(f"处理embedding批次 {batch_idx + 1}/{total_batches}")

            batch_results = self.embed_texts(batch_texts)

            # 转换None为空列表
            for emb in batch_results:
                all_embeddings.append(emb if emb is not None else [])

        # 统计成功率
        success_count = sum(1 for emb in all_embeddings if emb)
        logger.info(f"批量embedding完成: {success_count}/{len(texts)} 成功")

        return all_embeddings

    def is_available(self) -> bool:
        """检查embedding服务是否可用"""
        return self.embeddings is not None or self.openai_client is not None

    def test_connection(self) -> bool:
        """测试embedding服务连接"""
        if not self.is_available():
            return False

        try:
            test_embedding = self.embed_text("test connection")
            return test_embedding is not None and len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Embedding连接测试失败: {e}")
            return False


# 便捷函数：创建默认embedding服务
def create_default_embedding_service() -> EmbeddingService:
    """创建使用默认配置的embedding服务"""
    return EmbeddingService()
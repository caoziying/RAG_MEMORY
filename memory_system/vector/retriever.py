"""
混合检索器模块
实现0.7向量 + 0.3 BM25的混合检索算法。
"""

import logging
import re
import math
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import numpy as np

from .config import VectorRetrievalConfig
from .milvus_store import MilvusStore
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class SimpleBM25:
    """
    简化的BM25检索实现。
    适用于小到中等规模的内存索引。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25。

        Args:
            k1: BM25参数k1，控制词频饱和度
            b: BM25参数b，控制文档长度归一化
        """
        self.k1 = k1
        self.b = b

        # 文档存储
        self.documents: List[str] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0

        # 倒排索引
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        self.doc_freq: Dict[str, int] = defaultdict(int)  # 文档频率
        self.term_freq: List[Dict[str, int]] = []  # 每个文档的词频

        self.total_docs: int = 0
        self._built = False

    def add_document(self, text: str, doc_id: Optional[int] = None) -> int:
        """
        添加文档到索引。

        Args:
            text: 文档文本
            doc_id: 文档ID，如果为None则自动分配

        Returns:
            文档ID
        """
        if doc_id is not None and doc_id < len(self.documents):
            # 更新现有文档
            old_text = self.documents[doc_id]
            self._remove_from_index(doc_id, old_text)
            self.documents[doc_id] = text
            self._add_to_index(doc_id, text)
            return doc_id
        else:
            # 添加新文档
            doc_id = len(self.documents)
            self.documents.append(text)
            self._add_to_index(doc_id, text)
            return doc_id

    def add_documents(self, texts: List[str]) -> List[int]:
        """
        批量添加文档。

        Args:
            texts: 文档文本列表

        Returns:
            文档ID列表
        """
        return [self.add_document(text) for text in texts]

    def _add_to_index(self, doc_id: int, text: str):
        """将文档添加到索引"""
        # 分词
        tokens = self._tokenize(text)
        self.doc_lengths.append(len(tokens))

        # 词频统计
        term_freq = Counter(tokens)
        self.term_freq.append(term_freq)

        # 更新倒排索引
        for token in set(tokens):  # 每个词只记录一次文档ID
            self.inverted_index[token].append(doc_id)
            self.doc_freq[token] += 1

        self.total_docs = len(self.documents)
        self._built = False

    def _remove_from_index(self, doc_id: int, text: str):
        """从索引中移除文档"""
        if doc_id >= len(self.documents):
            return

        # 分词
        tokens = self._tokenize(text)

        # 更新倒排索引
        for token in set(tokens):
            if doc_id in self.inverted_index[token]:
                self.inverted_index[token].remove(doc_id)
                self.doc_freq[token] -= 1
                if self.doc_freq[token] == 0:
                    del self.doc_freq[token]
                    del self.inverted_index[token]

        # 更新数据结构
        if doc_id < len(self.doc_lengths):
            self.doc_lengths.pop(doc_id)
        if doc_id < len(self.term_freq):
            self.term_freq.pop(doc_id)

        self.total_docs = len(self.documents)
        self._built = False

    def build(self):
        """构建索引，计算平均文档长度"""
        if self.total_docs == 0:
            self.avg_doc_length = 0
            return

        total_length = sum(self.doc_lengths)
        self.avg_doc_length = total_length / self.total_docs
        self._built = True

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        搜索相关文档。

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            (文档ID, BM25分数) 列表，按分数降序排序
        """
        if not self._built:
            self.build()

        if self.total_docs == 0:
            return []

        # 分词
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # 计算每个文档的BM25分数
        scores = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            # IDF计算
            idf = math.log((self.total_docs - self.doc_freq[token] + 0.5) /
                          (self.doc_freq[token] + 0.5) + 1.0)

            # 对于包含该token的每个文档
            for doc_id in self.inverted_index[token]:
                # 词频
                tf = self.term_freq[doc_id].get(token, 0)

                # 文档长度归一化
                doc_len = self.doc_lengths[doc_id]
                norm = 1 - self.b + self.b * (doc_len / self.avg_doc_length)

                # BM25分数组成部分
                tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

                # 累加分数
                scores[doc_id] += idf * tf_component

        # 排序并返回Top-K
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        """
        简单的中英文分词。

        Args:
            text: 输入文本

        Returns:
            词元列表
        """
        if not text:
            return []

        # 转换为小写
        text = text.lower()

        # 中文分词：按字符分割（简化版）
        # 实际应用中应使用jieba等分词库
        chinese_pattern = r'[\u4e00-\u9fff]'
        english_pattern = r'\b\w+\b'

        # 提取中文字符
        chinese_chars = re.findall(chinese_pattern, text)

        # 提取英文单词
        english_words = re.findall(english_pattern, text)

        # 合并结果
        tokens = chinese_chars + english_words

        # 过滤停用词（简化版）
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就',
                      'the', 'and', 'is', 'in', 'to', 'of', 'a', 'i'}
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

        return tokens

    def get_document(self, doc_id: int) -> Optional[str]:
        """根据ID获取文档"""
        if 0 <= doc_id < len(self.documents):
            return self.documents[doc_id]
        return None

    def clear(self):
        """清空索引"""
        self.documents.clear()
        self.doc_lengths.clear()
        self.term_freq.clear()
        self.inverted_index.clear()
        self.doc_freq.clear()
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self._built = False


class HybridRetriever:
    """
    混合检索器，结合向量检索和BM25检索。

    检索流程：
    1. 向量检索：通过Milvus获取相似记忆
    2. BM25检索：通过本地BM25索引获取相关记忆
    3. 分数融合：0.7 * 向量分数 + 0.3 * BM25分数
    4. 阈值过滤：过滤低于相似度阈值的结果
    5. 返回Top-K结果
    """

    def __init__(
        self,
        milvus_store: MilvusStore,
        embedding_service: EmbeddingService,
        config: Optional[VectorRetrievalConfig] = None
    ):
        """
        初始化混合检索器。

        Args:
            milvus_store: Milvus存储实例
            embedding_service: Embedding服务实例
            config: 向量检索配置
        """
        self.config = config or VectorRetrievalConfig()
        self.milvus_store = milvus_store
        self.embedding_service = embedding_service
        self.bm25 = SimpleBM25()

        # 文档ID到Milvus ID的映射
        self.doc_id_to_milvus_id: Dict[int, str] = {}
        self.milvus_id_to_doc_id: Dict[str, int] = {}

        # 已索引的文档数
        self.indexed_count = 0

    def index_memory_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        索引记忆分块到检索系统。

        Args:
            chunks: 记忆分块列表，每个分块应包含：
                - id: Milvus中的ID
                - text: 文本内容
                - metadata: 元数据

        Returns:
            是否成功
        """
        if not chunks:
            return False

        try:
            # 提取文本用于BM25索引
            texts = []
            for chunk in chunks:
                chunk_id = chunk.get("id", "")
                text = chunk.get("text", "")
                if not text or not chunk_id:
                    continue

                # 添加到BM25索引
                doc_id = self.bm25.add_document(text)
                self.doc_id_to_milvus_id[doc_id] = chunk_id
                self.milvus_id_to_doc_id[chunk_id] = doc_id
                texts.append(text)

            # 构建BM25索引
            self.bm25.build()
            self.indexed_count += len(texts)

            logger.info(f"已索引 {len(texts)} 个记忆分块到BM25")
            return True

        except Exception as e:
            logger.error(f"索引记忆分块失败: {e}")
            return False

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_reranker: bool = True
    ) -> List[Dict[str, Any]]:
        """
        检索相关记忆。

        Args:
            query: 查询文本
            top_k: 返回结果数量，默认使用配置值
            similarity_threshold: 相似度阈值，默认使用配置值
            use_reranker: 是否使用reranker重排序

        Returns:
            检索结果列表，每个结果包含：
                - id: 记忆分块ID
                - text: 文本内容
                - score: 融合后的分数
                - vector_score: 向量检索分数
                - bm25_score: BM25分数
                - metadata: 元数据
        """
        top_k = top_k or self.config.top_k
        threshold = similarity_threshold or self.config.similarity_threshold

        # 空查询处理
        if not query or not query.strip():
            logger.warning("查询文本为空")
            return []

        try:
            # 1. 生成查询embedding
            query_embedding = self.embedding_service.embed_text(query)
            if not query_embedding:
                logger.error("无法生成查询embedding")
                return []

            # 2. 向量检索
            vector_results = self._vector_retrieve(query_embedding, top_k * 2)  # 多召回一些用于融合
            logger.debug(f"向量检索返回 {len(vector_results)} 个结果")

            # 3. BM25检索
            bm25_results = self._bm25_retrieve(query, top_k * 2)
            logger.debug(f"BM25检索返回 {len(bm25_results)} 个结果")

            # 4. 分数融合
            fused_results = self._fuse_scores(vector_results, bm25_results)

            # 5. 阈值过滤
            filtered_results = [
                r for r in fused_results
                if r["score"] >= threshold
            ]

            # 6. 排序并返回Top-K
            sorted_results = sorted(
                filtered_results,
                key=lambda x: x["score"],
                reverse=True
            )[:top_k]

            logger.info(f"混合检索完成，返回 {len(sorted_results)} 个结果")
            return sorted_results

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def _vector_retrieve(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """向量检索"""
        if not self.milvus_store.collection:
            return []

        results = self.milvus_store.search_similar(query_embedding, top_k)

        # 格式化结果
        formatted = []
        for result in results:
            formatted.append({
                "id": result["id"],
                "text": result["text"],
                "vector_score": result["score"],  # 余弦相似度，范围[0, 1]
                "bm25_score": 0.0,  # 初始化为0
                "score": result["score"],  # 初始分数为向量分数
                "metadata": result["metadata"]
            })

        return formatted

    def _bm25_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25检索"""
        if self.bm25.total_docs == 0:
            return []

        bm25_scores = self.bm25.search(query, top_k)

        # 格式化结果
        formatted = []
        for doc_id, bm25_score in bm25_scores:
            # 获取Milvus ID
            milvus_id = self.doc_id_to_milvus_id.get(doc_id)
            if not milvus_id:
                continue

            # 获取文档文本
            text = self.bm25.get_document(doc_id)
            if not text:
                continue

            # BM25分数归一化到[0, 1]范围（简单线性归一化）
            # 实际BM25分数范围不确定，这里使用简单的sigmoid归一化
            normalized_score = 1 / (1 + math.exp(-bm25_score * 0.1))

            formatted.append({
                "id": milvus_id,
                "text": text,
                "vector_score": 0.0,  # 初始化为0
                "bm25_score": normalized_score,
                "score": normalized_score,  # 初始分数为BM25分数
                "metadata": {}  # BM25不包含元数据
            })

        return formatted

    def _fuse_scores(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """融合向量分数和BM25分数"""

        # 创建结果映射
        results_by_id = {}

        # 添加向量检索结果
        for result in vector_results:
            result_id = result["id"]
            results_by_id[result_id] = result

        # 添加BM25检索结果，合并分数
        for result in bm25_results:
            result_id = result["id"]
            if result_id in results_by_id:
                # 合并分数
                existing = results_by_id[result_id]
                existing["bm25_score"] = result["bm25_score"]
            else:
                # 添加新结果（只有BM25分数）
                results_by_id[result_id] = result

        # 计算融合分数
        fused_results = []
        for result in results_by_id.values():
            vector_score = result.get("vector_score", 0.0)
            bm25_score = result.get("bm25_score", 0.0)

            # 融合分数：0.7 * 向量分数 + 0.3 * BM25分数
            fused_score = (
                self.config.vector_weight * vector_score +
                self.config.bm25_weight * bm25_score
            )

            result["score"] = fused_score
            fused_results.append(result)

        return fused_results

    def clear_index(self):
        """清空索引"""
        self.bm25.clear()
        self.doc_id_to_milvus_id.clear()
        self.milvus_id_to_doc_id.clear()
        self.indexed_count = 0
        logger.info("已清空检索索引")

    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            "total_documents": self.bm25.total_docs,
            "indexed_count": self.indexed_count,
            "avg_doc_length": self.bm25.avg_doc_length,
            "unique_terms": len(self.bm25.inverted_index),
            "bm25_built": self.bm25._built
        }
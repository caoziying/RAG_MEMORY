"""
Reranker模块
使用BGE-Reranker-Large模型对检索结果进行重排序。
单例模式实现，避免重复的API调用失败。
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import time
from enum import Enum

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from .config import VectorRetrievalConfig
from memory_system.core.logger import log_model_info, get_logger

logger = get_logger(__name__)


class RerankMode(Enum):
    """Rerank运行模式"""
    DIRECT_API = "direct_api"  # 直接调用API
    EMBEDDINGS_FALLBACK = "embeddings_fallback"  # 使用embeddings相似度降级
    DEFAULT_FALLBACK = "default_fallback"  # 返回默认值降级
    DISABLED = "disabled"  # 禁用rerank


class Reranker:
    """
    重排序器，使用BGE-Reranker-Large模型提升检索结果质量。
    单例模式实现，避免重复的API调用失败。

    Reranker接收(query, document)对，返回相关性分数。
    通常用于对Top-K检索结果进行精细排序。
    """

    # 单例模式实例
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(Reranker, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[VectorRetrievalConfig] = None):
        """
        初始化Reranker。

        Args:
            config: 向量检索配置，如果为None则使用默认配置
        """
        # 单例模式：只初始化一次
        if Reranker._initialized:
            return

        self.config = config or VectorRetrievalConfig()
        self.client = None
        self.langchain_llm = None
        self.mode = RerankMode.DISABLED  # 默认模式

        # 记录模型信息
        log_model_info(
            model_type="rerank",
            model_name=self.config.rerank_model,
            status="initializing",
            api_base=self.config.api_base
        )

        # 初始化客户端并检测模式
        self._init_client_and_detect_mode()

        Reranker._initialized = True

    def _init_client_and_detect_mode(self):
        """
        初始化客户端并检测可用的rerank模式。

        流程：
        1. 尝试初始化OpenAI客户端
        2. 尝试初始化LangChain客户端
        3. 测试API连接，确定可用模式
        4. 根据测试结果设置运行模式
        """
        # 优先使用OpenAI客户端
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.api_base
                )
                logger.info(f"使用OpenAI客户端，rerank模型: {self.config.rerank_model}")
            except Exception as e:
                logger.warning(f"初始化OpenAI客户端失败: {e}")
                self.client = None

        # 回退到LangChain ChatOpenAI
        if self.client is None and LANGCHAIN_AVAILABLE:
            try:
                self.langchain_llm = ChatOpenAI(
                    model=self.config.rerank_model,
                    api_key=self.config.api_key,
                    base_url=self.config.api_base,
                    temperature=0.0
                )
                logger.info(f"使用LangChain ChatOpenAI，rerank模型: {self.config.rerank_model}")
            except Exception as e:
                logger.error(f"初始化LangChain客户端失败: {e}")
                self.langchain_llm = None
        elif not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain不可用")

        # 检测可用的运行模式
        self._detect_available_mode()

    def _detect_available_mode(self):
        """
        检测可用的rerank模式。

        测试顺序：
        1. 直接API调用 (DIRECT_API)
        2. Embeddings相似度降级 (EMBEDDINGS_FALLBACK)
        3. 默认值降级 (DEFAULT_FALLBACK)
        """
        if not self.is_available():
            logger.warning("没有可用的rerank客户端，使用默认值降级模式")
            self.mode = RerankMode.DEFAULT_FALLBACK
            log_model_info(
                model_type="rerank",
                model_name=self.config.rerank_model,
                status="disabled",
                mode="default_fallback",
                reason="no_client_available"
            )
            return

        # 测试直接API调用
        logger.info("测试直接API调用模式...")
        if self._test_direct_api():
            self.mode = RerankMode.DIRECT_API
            logger.info(f"✅ 使用直接API调用模式，模型: {self.config.rerank_model}")
            log_model_info(
                model_type="rerank",
                model_name=self.config.rerank_model,
                status="active",
                mode="direct_api",
                client_type="openai" if self.client else "langchain"
            )
            return

        # 测试embeddings降级模式
        logger.info("直接API调用失败，测试embeddings降级模式...")
        if self._test_embeddings_fallback():
            self.mode = RerankMode.EMBEDDINGS_FALLBACK
            logger.info(f"✅ 使用embeddings降级模式，embedding模型: {self.config.embedding_model}")
            log_model_info(
                model_type="rerank",
                model_name=self.config.rerank_model,
                status="fallback",
                mode="embeddings_fallback",
                fallback_model=self.config.embedding_model,
                reason="direct_api_failed"
            )
            return

        # 所有模式都失败，使用默认值降级
        logger.warning("所有rerank模式均失败，使用默认值降级模式")
        self.mode = RerankMode.DEFAULT_FALLBACK
        log_model_info(
            model_type="rerank",
            model_name=self.config.rerank_model,
            status="disabled",
            mode="default_fallback",
            reason="all_modes_failed"
        )

    def _test_direct_api(self) -> bool:
        """
        测试直接API调用。

        Returns:
            True if direct API call is successful
        """
        test_pairs = [("测试查询", "测试文档")]

        try:
            # 测试单个调用
            if self.client is not None:
                prompt = f"query: {test_pairs[0][0]} document: {test_pairs[0][1]}"
                response = self.client.completions.create(
                    model=self.config.rerank_model,
                    prompt=prompt,
                    max_tokens=10,
                    temperature=0.0
                )
                # 检查响应是否有效
                if response.choices and response.choices[0].text:
                    logger.debug(f"直接API调用测试成功: {response.choices[0].text}")
                    return True
            elif self.langchain_llm is not None:
                from langchain_core.messages import HumanMessage
                prompt = f"query: {test_pairs[0][0]} document: {test_pairs[0][1]}"
                response = self.langchain_llm.invoke([HumanMessage(content=prompt)])
                if response.content:
                    logger.debug(f"LangChain API调用测试成功: {response.content[:50]}")
                    return True
        except Exception as e:
            logger.debug(f"直接API调用测试失败: {e}")

        return False

    def _test_embeddings_fallback(self) -> bool:
        """
        测试embeddings降级模式。

        Returns:
            True if embeddings fallback is available
        """
        if self.client is None:
            return False

        try:
            # 测试embeddings API
            test_texts = ["测试查询", "测试文档"]
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=test_texts
            )
            if response.data and len(response.data) == 2:
                logger.debug("Embeddings降级模式测试成功")
                return True
        except Exception as e:
            logger.debug(f"Embeddings降级模式测试失败: {e}")

        return False

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        batch_size: int = 10,
        retry_count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        对候选结果进行重排序。

        Args:
            query: 查询文本
            candidates: 候选结果列表，每个候选应包含：
                - id: 结果ID
                - text: 文本内容
                - score: 原始分数
                - metadata: 元数据（可选）
            top_n: 返回结果数量，默认使用配置值
            batch_size: 批处理大小
            retry_count: 重试次数

        Returns:
            重排序后的结果列表，按rerank分数降序排序
        """
        if not candidates:
            return []

        top_n = top_n or self.config.rerank_top_n
        top_n = min(top_n, len(candidates))

        # 根据模式决定处理方式
        if self.mode == RerankMode.DISABLED or self.mode == RerankMode.DEFAULT_FALLBACK:
            logger.info(f"Rerank处于{self.mode.value}模式，返回原始排序")
            return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_n]

        try:
            # 准备(query, document)对
            pairs = []
            for candidate in candidates:
                text = candidate.get("text", "")
                if not text:
                    continue
                pairs.append((query, text))

            if not pairs:
                return []

            # 批量计算rerank分数
            rerank_scores = self._batch_rerank_scores(pairs, batch_size, retry_count)

            if len(rerank_scores) != len(candidates):
                logger.error(f"Rerank分数数量不匹配: {len(rerank_scores)} != {len(candidates)}")
                return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_n]

            # 合并分数到候选结果
            for i, candidate in enumerate(candidates):
                if i < len(rerank_scores):
                    candidate["rerank_score"] = rerank_scores[i]
                    # 可选：将原始分数和rerank分数结合
                    original_score = candidate.get("score", 0)
                    candidate["final_score"] = 0.3 * original_score + 0.7 * rerank_scores[i]
                else:
                    candidate["rerank_score"] = 0.0
                    candidate["final_score"] = candidate.get("score", 0)

            # 按最终分数排序
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.get("final_score", 0),
                reverse=True
            )[:top_n]

            logger.info(f"Rerank完成，模式: {self.mode.value}, 处理 {len(candidates)} 个候选，返回 {len(sorted_candidates)} 个结果")
            return sorted_candidates

        except Exception as e:
            logger.error(f"Rerank失败: {e}")
            # 失败时返回原始排序
            return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_n]

    def _batch_rerank_scores(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 10,
        retry_count: int = 3
    ) -> List[float]:
        """
        批量计算rerank分数。

        Args:
            pairs: (query, document)对列表
            batch_size: 批处理大小
            retry_count: 重试次数

        Returns:
            rerank分数列表，范围[0, 1]
        """
        all_scores = []

        total_batches = (len(pairs) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(pairs))
            batch_pairs = pairs[start_idx:end_idx]

            logger.debug(f"处理rerank批次 {batch_idx + 1}/{total_batches}")

            batch_scores = self._compute_rerank_scores(batch_pairs, retry_count)
            all_scores.extend(batch_scores)

        return all_scores

    def _compute_rerank_scores(
        self,
        pairs: List[Tuple[str, str]],
        retry_count: int = 3
    ) -> List[float]:
        """
        计算rerank分数。

        根据检测到的模式使用不同的计算方法。

        Args:
            pairs: (query, document)对列表
            retry_count: 重试次数

        Returns:
            rerank分数列表
        """
        # 根据模式选择计算方法
        if self.mode == RerankMode.DIRECT_API:
            return self._compute_direct_api_scores(pairs, retry_count)
        elif self.mode == RerankMode.EMBEDDINGS_FALLBACK:
            return self._compute_embeddings_fallback_scores(pairs, retry_count)
        elif self.mode == RerankMode.DEFAULT_FALLBACK:
            logger.debug(f"使用默认值降级模式，返回默认分数0.5")
            return [0.5] * len(pairs)
        else:  # DISABLED
            logger.debug(f"Rerank已禁用，返回默认分数0.5")
            return [0.5] * len(pairs)

    def _compute_direct_api_scores(
        self,
        pairs: List[Tuple[str, str]],
        retry_count: int = 3
    ) -> List[float]:
        """
        使用直接API调用计算rerank分数。

        Args:
            pairs: (query, document)对列表
            retry_count: 重试次数

        Returns:
            rerank分数列表
        """
        for attempt in range(retry_count):
            try:
                scores = []

                # 使用OpenAI客户端
                if self.client is not None:
                    logger.debug(f"使用OpenAI客户端进行直接API调用，模型: {self.config.rerank_model}")

                    for query, document in pairs:
                        prompt = f"query: {query} document: {document}"
                        try:
                            response = self.client.completions.create(
                                model=self.config.rerank_model,
                                prompt=prompt,
                                max_tokens=10,
                                temperature=0.0
                            )

                            # 解析分数
                            score_text = response.choices[0].text.strip()
                            import re
                            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", score_text)
                            if numbers:
                                score = float(numbers[0])
                                # 确保分数在0-1范围内
                                score = max(0.0, min(1.0, score))
                            else:
                                # 如果没有数字，使用默认分数
                                score = 0.5
                                logger.warning(f"无法从响应中解析分数: {score_text[:50]}")
                            scores.append(score)

                        except Exception as api_error:
                            # 记录错误但不停止，继续尝试下一个
                            error_msg = str(api_error)
                            if hasattr(api_error, 'response') and api_error.response:
                                try:
                                    error_details = api_error.response.json()
                                    error_msg = f"{error_msg} - {error_details}"
                                except:
                                    pass
                            logger.warning(f"单个rerank调用失败: {error_msg}")
                            scores.append(0.5)

                    # 检查是否有太多失败
                    successful_scores = [s for s in scores if s != 0.5]
                    if len(successful_scores) >= len(scores) * 0.5:  # 至少50%成功
                        logger.info(f"直接API调用成功，{len(successful_scores)}/{len(scores)}个分数有效")
                        return scores
                    else:
                        logger.warning(f"直接API调用失败过多，{len(successful_scores)}/{len(scores)}个分数有效，尝试重试")
                        raise Exception(f"直接API调用失败过多，成功{len(successful_scores)}/{len(scores)}")

                # 使用LangChain客户端
                elif self.langchain_llm is not None:
                    logger.debug(f"使用LangChain客户端进行直接API调用，模型: {self.config.rerank_model}")

                    from langchain_core.messages import HumanMessage
                    for query, document in pairs:
                        prompt = f"query: {query} document: {document}"
                        try:
                            response = self.langchain_llm.invoke([HumanMessage(content=prompt)])

                            # 解析响应
                            content = response.content.strip()
                            import re
                            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)
                            if numbers:
                                score = float(numbers[0])
                                # 归一化到[0,1]
                                score = max(0.0, min(1.0, score / 10.0 if score > 1.0 else score))
                            else:
                                # 从文本推断
                                if any(word in content.lower() for word in ["高", "high", "相关", "relevant"]):
                                    score = 0.8
                                elif any(word in content.lower() for word in ["中", "medium", "一般"]):
                                    score = 0.5
                                else:
                                    score = 0.3
                            scores.append(score)

                        except Exception as e:
                            logger.warning(f"LangChain调用失败: {e}")
                            scores.append(0.5)

                    return scores

                else:
                    logger.error("没有可用的rerank客户端")
                    raise Exception("没有可用的rerank客户端")

            except Exception as e:
                logger.warning(f"直接API调用失败（尝试 {attempt + 1}/{retry_count}）: {e}")
                if attempt < retry_count - 1:
                    time.sleep(1 * (attempt + 1))  # 指数退避
                else:
                    logger.error(f"直接API调用最终失败，降级到embeddings模式")
                    # 尝试降级到embeddings模式
                    try:
                        return self._compute_embeddings_fallback_scores(pairs, retry_count=1)
                    except Exception as fallback_error:
                        logger.error(f"降级到embeddings模式也失败: {fallback_error}")
                        return [0.5] * len(pairs)

        return [0.5] * len(pairs)

    def _compute_embeddings_fallback_scores(
        self,
        pairs: List[Tuple[str, str]],
        retry_count: int = 3
    ) -> List[float]:
        """
        使用embeddings相似度计算rerank分数（降级模式）。

        Args:
            pairs: (query, document)对列表
            retry_count: 重试次数

        Returns:
            rerank分数列表
        """
        if self.client is None:
            logger.error("没有可用的OpenAI客户端，无法使用embeddings降级模式")
            return [0.5] * len(pairs)

        for attempt in range(retry_count):
            try:
                logger.debug(f"使用embeddings降级模式，模型: {self.config.embedding_model}")

                # 批量获取所有文本的embedding
                all_texts = []
                for query, document in pairs:
                    all_texts.append(query)
                    all_texts.append(document)

                # 批量获取embedding
                response = self.client.embeddings.create(
                    model=self.config.embedding_model,
                    input=all_texts
                )
                embeddings = [data.embedding for data in response.data]

                # 计算每对query和document的余弦相似度
                scores = []
                for i in range(len(pairs)):
                    query_emb = embeddings[i * 2]
                    doc_emb = embeddings[i * 2 + 1]

                    # 计算余弦相似度
                    dot_product = sum(q * d for q, d in zip(query_emb, doc_emb))
                    norm_q = sum(q * q for q in query_emb) ** 0.5
                    norm_d = sum(d * d for d in doc_emb) ** 0.5

                    if norm_q > 0 and norm_d > 0:
                        cosine_sim = dot_product / (norm_q * norm_d)
                        # 将余弦相似度从[-1,1]映射到[0,1]
                        score = (cosine_sim + 1) / 2
                    else:
                        score = 0.5

                    scores.append(max(0.0, min(1.0, score)))

                logger.info(f"embeddings降级模式成功，返回{len(scores)}个分数")
                return scores

            except Exception as e:
                logger.warning(f"embeddings降级模式失败（尝试 {attempt + 1}/{retry_count}）: {e}")
                if attempt < retry_count - 1:
                    time.sleep(1 * (attempt + 1))  # 指数退避
                else:
                    logger.error(f"embeddings降级模式最终失败")
                    return [0.5] * len(pairs)

        return [0.5] * len(pairs)

    def rerank_single(self, query: str, document: str) -> float:
        """
        对单个(query, document)对计算rerank分数。

        Args:
            query: 查询文本
            document: 文档文本

        Returns:
            rerank分数，范围[0, 1]
        """
        scores = self._compute_rerank_scores([(query, document)], retry_count=2)
        return scores[0] if scores else 0.5

    def is_available(self) -> bool:
        """检查reranker是否可用"""
        return self.mode != RerankMode.DISABLED

    def get_status(self) -> Dict[str, Any]:
        """
        获取reranker状态信息。

        Returns:
            包含状态信息的字典
        """
        client_type = None
        if self.client is not None:
            client_type = "openai"
        elif self.langchain_llm is not None:
            client_type = "langchain"

        return {
            "mode": self.mode.value,
            "client_type": client_type,
            "rerank_model": self.config.rerank_model,
            "embedding_model": self.config.embedding_model,
            "api_base": self.config.api_base,
            "initialized": Reranker._initialized,
            "available": self.is_available()
        }

    def test_connection(self) -> bool:
        """测试reranker连接"""
        if not self.is_available():
            logger.warning("Reranker不可用，无法测试连接")
            return False

        try:
            # 根据模式使用不同的测试方法
            if self.mode == RerankMode.DIRECT_API:
                test_pairs = [("测试查询", "测试文档")]
                scores = self._compute_direct_api_scores(test_pairs, retry_count=1)
                success = len(scores) == 1 and 0 <= scores[0] <= 1
                if success:
                    logger.info(f"直接API连接测试成功，分数: {scores[0]}")
                else:
                    logger.warning(f"直接API连接测试失败，分数: {scores[0] if scores else '无'}")
                return success

            elif self.mode == RerankMode.EMBEDDINGS_FALLBACK:
                test_pairs = [("测试查询", "测试文档")]
                scores = self._compute_embeddings_fallback_scores(test_pairs, retry_count=1)
                success = len(scores) == 1 and 0 <= scores[0] <= 1
                if success:
                    logger.info(f"Embeddings降级模式连接测试成功，分数: {scores[0]}")
                else:
                    logger.warning(f"Embeddings降级模式连接测试失败，分数: {scores[0] if scores else '无'}")
                return success

            elif self.mode == RerankMode.DEFAULT_FALLBACK:
                logger.info("默认值降级模式，无需连接测试")
                return True

            else:  # DISABLED
                logger.info("Reranker已禁用，无需连接测试")
                return False

        except Exception as e:
            logger.error(f"Reranker连接测试失败: {e}")
            return False


# 便捷函数：创建默认reranker
def create_default_reranker() -> Reranker:
    """创建使用默认配置的reranker"""
    return Reranker()
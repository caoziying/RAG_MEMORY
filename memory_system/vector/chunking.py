"""
分块策略模块
将对话分割成适合embedding和检索的文本块。
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .config import VectorRetrievalConfig

logger = logging.getLogger(__name__)


class ConversationChunker:
    """
    对话分块器，将对话历史分割成适合embedding的文本块。

    分块策略：
    1. 按对话轮次分组，保持上下文连贯性
    2. 按固定长度切分（字符数或token数）
    3. 添加重叠区域以避免信息断裂
    4. 保留丰富的元数据便于后续检索
    """

    def __init__(self, config: Optional[VectorRetrievalConfig] = None):
        """
        初始化分块器。

        Args:
            config: 向量检索配置，如果为None则使用默认配置
        """
        self.config = config or VectorRetrievalConfig()

    def chunk_conversation(
        self,
        conversation: Dict[str, Any],
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        将单轮对话分块。

        Args:
            conversation: 对话数据，应包含：
                - role: 角色（user/assistant）
                - content: 对话内容
                - timestamp: 时间戳
                - 其他元数据
            conversation_id: 对话ID，用于关联多轮对话

        Returns:
            分块列表，每个分块包含：
                - text: 分块文本
                - metadata: 元数据
        """
        if not conversation or "content" not in conversation:
            return []

        content = conversation.get("content", "")
        if not content or not content.strip():
            return []

        # 基础元数据
        base_metadata = {
            "role": conversation.get("role", "unknown"),
            "timestamp": conversation.get("timestamp", 0),
            "conversation_id": conversation_id or "unknown",
            "original_length": len(content)
        }

        # 添加其他元数据
        for key, value in conversation.items():
            if key not in ["content", "role", "timestamp"]:
                base_metadata[key] = value

        # 按句子分割（保留标点）
        sentences = self._split_into_sentences(content)

        # 如果内容很短，直接作为一个分块
        if len(content) <= self.config.chunk_size:
            return [{
                "text": content,
                "metadata": {**base_metadata, "chunk_index": 0, "total_chunks": 1}
            }]

        # 构建分块
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # 如果当前句子本身超过块大小，需要进一步分割
            if sentence_length > self.config.chunk_size:
                if current_chunk:
                    # 保存当前块
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **base_metadata,
                            "chunk_index": chunk_index,
                            "total_chunks": len(chunks) + 2  # 当前块 + 大句子分割的块
                        }
                    })
                    chunk_index += 1
                    current_chunk = []
                    current_length = 0

                # 分割大句子
                sub_chunks = self._split_long_sentence(sentence)
                for sub_chunk in sub_chunks:
                    chunks.append({
                        "text": sub_chunk,
                        "metadata": {
                            **base_metadata,
                            "chunk_index": chunk_index,
                            "total_chunks": len(chunks) + len(sub_chunks)
                        }
                    })
                    chunk_index += 1

            # 正常句子：添加到当前块
            elif current_length + sentence_length <= self.config.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # 当前块已满，保存并开始新块
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **base_metadata,
                            "chunk_index": chunk_index,
                            "total_chunks": len(chunks) + 1
                        }
                    })
                    chunk_index += 1

                # 新块从当前句子开始
                current_chunk = [sentence]
                current_length = sentence_length

        # 添加最后一个块
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **base_metadata,
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks)
                }
            })

        # 更新总块数
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)

        logger.debug(f"对话分块完成：{len(content)}字符 -> {len(chunks)}个分块")
        return chunks

    def chunk_conversations(
        self,
        conversations: List[Dict[str, Any]],
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        将多轮对话分块。

        Args:
            conversations: 对话列表
            conversation_id: 对话ID，用于关联多轮对话

        Returns:
            分块列表
        """
        all_chunks = []

        for i, conv in enumerate(conversations):
            # 为每轮对话生成唯一ID（如果未提供）
            conv_id = f"{conversation_id}_{i}" if conversation_id else f"conv_{i}"

            chunks = self.chunk_conversation(conv, conv_id)
            all_chunks.extend(chunks)

        logger.info(f"多轮对话分块完成：{len(conversations)}轮 -> {len(all_chunks)}个分块")
        return all_chunks

    def chunk_with_context(
        self,
        conversations: List[Dict[str, Any]],
        context_window: int = 3
    ) -> List[Dict[str, Any]]:
        """
        分块时保留上下文窗口。

        Args:
            conversations: 对话列表
            context_window: 上下文窗口大小（包含前几轮对话）

        Returns:
            带上下文的分块列表
        """
        if not conversations:
            return []

        all_chunks = []

        for i in range(len(conversations)):
            # 获取上下文窗口
            start_idx = max(0, i - context_window + 1)
            context_convs = conversations[start_idx:i + 1]

            # 构建带上下文的文本
            context_text = self._build_context_text(context_convs)

            # 分块
            base_metadata = {
                "role": conversations[i].get("role", "unknown"),
                "timestamp": conversations[i].get("timestamp", 0),
                "conversation_id": f"ctx_{i}",
                "context_window": context_window,
                "position_in_context": i - start_idx
            }

            # 分割长上下文
            if len(context_text) <= self.config.chunk_size:
                all_chunks.append({
                    "text": context_text,
                    "metadata": {**base_metadata, "chunk_index": 0, "total_chunks": 1}
                })
            else:
                # 对长上下文进行分块
                sentences = self._split_into_sentences(context_text)
                chunks = []
                current_chunk = []
                current_length = 0

                for sentence in sentences:
                    sentence_length = len(sentence)

                    if current_length + sentence_length <= self.config.chunk_size:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                    else:
                        if current_chunk:
                            chunk_text = " ".join(current_chunk)
                            chunks.append({
                                "text": chunk_text,
                                "metadata": {
                                    **base_metadata,
                                    "chunk_index": len(chunks),
                                    "total_chunks": len(chunks) + 1
                                }
                            })
                        current_chunk = [sentence]
                        current_length = sentence_length

                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **base_metadata,
                            "chunk_index": len(chunks),
                            "total_chunks": len(chunks)
                        }
                    })

                all_chunks.extend(chunks)

        logger.info(f"带上下文分块完成：{len(conversations)}轮 -> {len(all_chunks)}个分块")
        return all_chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割成句子。

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        if not text:
            return []

        # 简单的中英文句子分割
        # 中文句子结束符：。！？；.!?;
        sentence_endings = r'[。！？；\.!?;]+'

        # 分割句子
        parts = re.split(f'({sentence_endings})', text)

        # 合并结束符回前一个句子
        sentences = []
        current_sentence = ""

        for i, part in enumerate(parts):
            if re.match(sentence_endings, part):
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part

        # 添加最后一个句子
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        # 过滤空句子
        sentences = [s for s in sentences if s and s.strip()]

        return sentences

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        分割过长的句子。

        Args:
            sentence: 长句子

        Returns:
            子句列表
        """
        if len(sentence) <= self.config.chunk_size:
            return [sentence]

        # 按逗号、分号等分割
        sub_sentences = re.split(r'[，,；;]', sentence)

        # 如果分割后仍然太长，按长度分割
        result = []
        for sub in sub_sentences:
            if len(sub) <= self.config.chunk_size:
                result.append(sub.strip())
            else:
                # 按固定长度分割
                for i in range(0, len(sub), self.config.chunk_size - self.config.chunk_overlap):
                    chunk = sub[i:i + self.config.chunk_size]
                    if chunk.strip():
                        result.append(chunk.strip())

        return result

    def _build_context_text(self, conversations: List[Dict[str, Any]]) -> str:
        """
        构建带上下文的文本。

        Args:
            conversations: 对话列表

        Returns:
            带上下文的文本
        """
        context_parts = []

        for conv in conversations:
            role = conv.get("role", "unknown")
            content = conv.get("content", "")
            timestamp = conv.get("timestamp", "")

            role_display = "用户" if role == "user" else "助手"
            time_str = f" ({timestamp})" if timestamp else ""

            context_parts.append(f"{role_display}{time_str}: {content}")

        return "\n".join(context_parts)

    def calculate_token_count(self, text: str) -> int:
        """
        估算文本的token数量（简化版）。

        Args:
            text: 输入文本

        Returns:
            估算的token数量
        """
        # 简单估算：中文字符算1个token，英文字词按空格分割
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'\b\w+\b', text))

        # 标点和其他字符
        other_chars = len(text) - chinese_chars - english_words

        # 估算：中文字符1token，英文单词~0.75token，其他字符~0.5token
        estimated_tokens = chinese_chars + int(english_words * 0.75) + int(other_chars * 0.5)

        return max(1, estimated_tokens)


# 便捷函数：创建默认分块器
def create_default_chunker() -> ConversationChunker:
    """创建使用默认配置的分块器"""
    return ConversationChunker()
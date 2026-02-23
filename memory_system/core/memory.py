"""
记忆系统核心模块
实现对话窗口管理、LLM压缩、用户信息提取和每日日志记录。
"""

import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from langchain_openai import ChatOpenAI
try:
    from langchain.schema import HumanMessage, AIMessage
except ImportError:
    # 兼容新版本 LangChain
    from langchain_core.messages import HumanMessage, AIMessage

from .config import (
    MAX_WINDOWS,
    DATA_DIR,
    LOG_DIR,
    USER_INFO_FILE,
    COMPRESSED_MEMORY_FILE,
    init_directories
)

logger = logging.getLogger(__name__)


class MemorySystem:
    """
    记忆系统主类，管理对话记忆、用户信息和日志记录。

    特性：
    1. 维护固定大小的对话窗口（最近 MAX_WINDOWS 轮对话）
    2. 窗口满时自动使用 LLM 压缩旧对话为摘要
    3. 提取用户个人信息并存储到 user.md
    4. 每天生成独立的聊天日志文件（YYYY-MM-DD.md）
    5. 支持持久化存储压缩记忆和用户信息

    使用方法：
    >>> from memory_system.memory import MemorySystem
    >>> from langchain_openai import ChatOpenAI
    >>> llm = ChatOpenAI(model="Qwen3-32B-FP8", temperature=0.0)
    >>> memory = MemorySystem(llm=llm)
    >>> memory.add_conversation("user", "我叫张三，今年25岁，是一名软件工程师。")
    >>> memory.add_conversation("assistant", "你好张三！很高兴认识你。")
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        max_windows: int = MAX_WINDOWS,
        data_dir: Optional[Path] = None
    ):
        """
        初始化记忆系统。

        Args:
            llm: LangChain ChatOpenAI 实例，用于对话压缩和信息提取。
                如果为 None，将尝试使用默认配置创建（需要环境变量）。
            max_windows: 对话窗口最大容量，默认 10 轮。
            data_dir: 数据存储目录，如果为 None 使用默认目录。
        """
        # 初始化存储目录
        init_directories()

        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.log_dir = LOG_DIR if not data_dir else data_dir / "logs"
        self.user_info_file = USER_INFO_FILE if not data_dir else data_dir / "user.md"
        self.compressed_memory_file = COMPRESSED_MEMORY_FILE if not data_dir else data_dir / "compressed_memory.md"

        # 确保数据目录和日志目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 对话窗口管理
        self.max_windows = max_windows
        self.conversation_window = deque(maxlen=max_windows)  # 最近对话轮次
        self.compressed_memories: List[str] = []  # 压缩后的记忆摘要列表

        # LLM 实例
        self.llm = llm
        if self.llm is None:
            logger.warning("未提供 LLM 实例，使用默认配置创建。")
            self.llm = self._create_default_llm()
            if self.llm is None:
                logger.error("无法创建默认 LLM 实例，请检查环境变量配置。")
            else:
                logger.info("创建 LLM 实例成功")

        # 加载已有的压缩记忆和用户信息
        self._load_compressed_memory()
        self._load_user_info()

        logger.info(f"记忆系统初始化完成，对话窗口容量：{max_windows}")

    def _create_default_llm(self) -> ChatOpenAI:
        """使用默认配置创建 LangChain ChatOpenAI 实例"""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("MY_API_KEY", "")
        base_url = os.getenv("MY_API_BASE", "https://api.chat.csu.edu.cn/v1")
        model_name = os.getenv("MY_MODEL", "")

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            stream=False
        )

    def _load_compressed_memory(self):
        """从文件加载压缩记忆"""
        try:
            if self.compressed_memory_file.exists():
                with open(self.compressed_memory_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # 每行作为一个压缩记忆项
                        self.compressed_memories = [line.strip() for line in content.split('\n') if line.strip()]
                logger.info(f"已加载 {len(self.compressed_memories)} 条压缩记忆")
        except Exception as e:
            logger.error(f"加载压缩记忆失败：{e}")
            self.compressed_memories = []

    def _save_compressed_memory(self):
        """保存压缩记忆到文件"""
        try:
            with open(self.compressed_memory_file, 'w', encoding='utf-8') as f:
                for memory in self.compressed_memories:
                    f.write(memory + '\n')
            logger.debug(f"压缩记忆已保存到 {self.compressed_memory_file}")
        except Exception as e:
            logger.error(f"保存压缩记忆失败：{e}")

    def _load_user_info(self):
        """加载用户信息（仅检查文件是否存在，内容由外部维护）"""
        if not self.user_info_file.exists():
            logger.info("用户信息文件不存在，将在首次提取时创建")

    def add_conversation(self, role: str, content: str):
        """
        添加一轮对话到记忆系统。

        Args:
            role: 对话角色，'user' 或 'assistant'
            content: 对话内容
        """
        # 验证角色
        if role not in ['user', 'assistant']:
            raise ValueError("角色必须是 'user' 或 'assistant'")

        # 记录时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conversation_entry = {
            'role': role,
            'content': content,
            'timestamp': timestamp
        }

        # 添加到对话窗口
        self.conversation_window.append(conversation_entry)

        # 如果是用户输入，尝试提取个人信息
        if role == 'user':
            self._extract_user_info(content)

        # 保存到每日日志
        self._save_to_daily_log(role, content, timestamp)

        # 检查是否需要压缩对话
        if len(self.conversation_window) >= self.max_windows:
            self._compress_conversations()

        logger.debug(f"添加对话：{role} - {content[:50]}...")

    def _extract_user_info(self, user_input: str):
        """
        使用 LLM 从用户输入中提取个人信息。
        提取的信息将追加到 user.md 文件中。
        """
        try:
            # 构建提取提示
            prompt = f"""
            请从以下用户输入中提取可能在未来有用的个人信息。
            只提取与用户自身相关的信息，如姓名、年龄、职业、兴趣爱好、联系方式、偏好等。
            如果输入中没有此类信息，直接返回 'null'。

            用户输入：{user_input}

            输出格式要求：
            - 如果提取到信息，直接返回提取的关键信息文本，另起一段给出支撑该信息的用户原始输入片段。
            - 如果没有提取到信息，返回 'null'。
            """

            # 调用 LLM
            response = self.llm.invoke([HumanMessage(content=prompt)])
            extracted_info = response.content.strip()

            # 如果提取到信息，保存到文件
            if extracted_info and extracted_info.lower() not in ['无', '没有', 'none', '', 'null']:
                self._append_to_user_info(extracted_info, user_input)
                logger.info(f"提取到用户信息：{extracted_info}")

        except Exception as e:
            logger.error(f"提取用户信息失败：{e}")

    def _append_to_user_info(self, extracted_info: str, original_input: str):
        """
        将提取的用户信息追加到 user.md 文件。

        格式：
        ## 2025-02-23 14:30:25
        - 提取信息：张三是一名软件工程师
        - 原始输入："我叫张三，是一名软件工程师。"
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(self.user_info_file, 'a', encoding='utf-8') as f:
                f.write(f"\n## {timestamp}\n")
                f.write(f"- 提取信息：{extracted_info}\n")
                f.write(f"- 原始输入：\"{original_input}\"\n")
                f.write("---\n")

            logger.debug(f"用户信息已追加到 {self.user_info_file}")
        except Exception as e:
            logger.error(f"追加用户信息失败：{e}")

    def _save_to_daily_log(self, role: str, content: str, timestamp: str):
        """
        将对话保存到当天的日志文件中。

        格式：
        ### 2025-02-23 14:30:25 (user)
        我叫张三，今年25岁，是一名软件工程师。

        ### 2025-02-23 14:30:30 (assistant)
        你好张三！很高兴认识你。
        """
        try:
            # 构建当天日志文件路径（使用自定义日志目录）
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self.log_dir / f"{today}.md"

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n### {timestamp} ({role})\n")
                f.write(f"{content}\n")

            logger.debug(f"对话已保存到日志文件：{log_file}")
        except Exception as e:
            logger.error(f"保存日志失败：{e}")

    def _compress_conversations(self):
        """
        当对话窗口满时，使用 LLM 压缩旧对话为摘要。
        摘要将添加到压缩记忆列表，并清空当前窗口（除最新几轮外）。
        """
        try:
            if len(self.conversation_window) < 2:
                logger.warning("对话数量不足，跳过压缩")
                return

            # 准备要压缩的对话（排除最新的2轮，以保持上下文连续性）
            compress_up_to = max(0, len(self.conversation_window) - 2)
            if compress_up_to < 1:
                return

            conversations_to_compress = list(self.conversation_window)[:compress_up_to]

            # 构建压缩提示
            conversations_text = "\n".join([
                f"{conv['timestamp']} {conv['role']}: {conv['content']}"
                for conv in conversations_to_compress
            ])

            prompt = f"""
            请将以下对话压缩成一个简洁的摘要，保留重要信息和上下文。
            摘要应该足够简短，但包含关键事实、决定和用户偏好。

            对话记录：
            {conversations_text}

            压缩摘要（用一段话描述）：
            """

            # 调用 LLM 生成摘要
            response = self.llm.invoke([HumanMessage(content=prompt)])
            summary = response.content.strip()

            if summary:
                # 添加时间戳到摘要
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                compressed_entry = f"{timestamp}: {summary}"

                # 添加到压缩记忆列表
                self.compressed_memories.append(compressed_entry)

                # 保存到文件
                self._save_compressed_memory()

                # 从窗口中移除已压缩的对话（保留最新的2轮）
                for _ in range(compress_up_to):
                    if self.conversation_window:
                        self.conversation_window.popleft()

                logger.info(f"对话压缩完成，生成摘要：{summary[:100]}...")
                logger.info(f"压缩后窗口大小：{len(self.conversation_window)}")

        except Exception as e:
            logger.error(f"对话压缩失败：{e}")

    def get_recent_conversations(self, n: Optional[int] = None) -> List[Dict]:
        """
        获取最近的对话记录。

        Args:
            n: 返回的对话轮次数量，如果为 None 则返回所有窗口内的对话。

        Returns:
            对话记录列表，按时间顺序从旧到新。
        """
        if n is None:
            return list(self.conversation_window)
        else:
            return list(self.conversation_window)[-n:]

    def get_compressed_memories(self) -> List[str]:
        """获取所有压缩记忆"""
        return self.compressed_memories.copy()

    def get_context_for_llm(self, max_recent: int = 5) -> str:
        """
        生成供 LLM 使用的上下文字符串，包含压缩记忆和最近对话。

        Args:
            max_recent: 包含的最近对话轮次数量

        Returns:
            格式化的上下文字符串
        """
        context_parts = []

        # 添加压缩记忆
        if self.compressed_memories:
            context_parts.append("## 压缩记忆（历史摘要）")
            for memory in self.compressed_memories[-10:]:  # 最多10条最近压缩记忆
                context_parts.append(f"- {memory}")

        # 添加最近对话
        recent_conv = self.get_recent_conversations(max_recent)
        if recent_conv:
            context_parts.append("\n## 最近对话")
            for conv in recent_conv:
                role_display = "用户" if conv['role'] == 'user' else "助手"
                context_parts.append(f"{conv['timestamp']} {role_display}: {conv['content']}")

        return "\n".join(context_parts) if context_parts else "暂无历史记录。"

    def clear_conversation_window(self):
        """清空当前对话窗口（但不影响压缩记忆）"""
        self.conversation_window.clear()
        logger.info("对话窗口已清空")

    def reset_all(self):
        """重置整个记忆系统（清空窗口和压缩记忆）"""
        self.conversation_window.clear()
        self.compressed_memories.clear()
        logger.info("记忆系统已重置")


# 便捷函数：创建默认记忆系统实例
def create_default_memory_system() -> MemorySystem:
    """创建使用默认配置的记忆系统实例"""
    return MemorySystem()
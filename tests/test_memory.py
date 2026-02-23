#!/usr/bin/env python3
"""
记忆系统单元测试

使用模拟 LLM 测试核心功能，不依赖实际 API 调用。
运行测试：pytest tests/test_memory.py
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import shutil
import sys

# 添加父目录到 Python 路径以便导入 memory_system
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_system.memory import MemorySystem
from memory_system.config import DATA_DIR, LOG_DIR, USER_INFO_FILE, COMPRESSED_MEMORY_FILE


class TestMemorySystem:
    """MemorySystem 测试类"""

    @pytest.fixture
    def mock_llm(self):
        """创建模拟 LLM"""
        llm = Mock()
        llm.invoke.return_value.content = "模拟摘要内容"
        return llm

    @pytest.fixture
    def temp_data_dir(self):
        """创建临时数据目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            yield tmp_path

    @pytest.fixture
    def memory_system(self, mock_llm, temp_data_dir):
        """创建使用临时目录的记忆系统实例"""
        return MemorySystem(llm=mock_llm, max_windows=5, data_dir=temp_data_dir)

    def test_initialization(self, memory_system, temp_data_dir):
        """测试初始化"""
        assert memory_system.max_windows == 5
        assert len(memory_system.conversation_window) == 0
        assert len(memory_system.compressed_memories) == 0
        assert memory_system.data_dir == temp_data_dir

        # 检查目录是否创建
        assert (temp_data_dir / "logs").exists()
        assert memory_system.log_dir.exists()

    def test_add_conversation(self, memory_system, mock_llm):
        """测试添加对话"""
        # 添加用户对话
        memory_system.add_conversation("user", "测试用户输入")
        assert len(memory_system.conversation_window) == 1

        # 检查对话内容
        conv = memory_system.conversation_window[0]
        assert conv["role"] == "user"
        assert conv["content"] == "测试用户输入"
        assert "timestamp" in conv

        # 添加助手对话
        memory_system.add_conversation("assistant", "测试助手回复")
        assert len(memory_system.conversation_window) == 2

        # 验证角色验证
        with pytest.raises(ValueError):
            memory_system.add_conversation("invalid_role", "内容")

    def test_user_info_extraction(self, memory_system, mock_llm):
        """测试用户信息提取"""
        # 配置模拟 LLM 返回提取的信息
        mock_llm.invoke.return_value.content = "用户提供了测试信息"

        # 添加用户对话（应触发信息提取）
        memory_system.add_conversation("user", "我的名字是测试用户")

        # 检查 user.md 文件是否创建
        user_file = memory_system.user_info_file
        assert user_file.exists()

        # 检查文件内容
        content = user_file.read_text(encoding='utf-8')
        assert "用户提供了测试信息" in content
        assert "我的名字是测试用户" in content

    def test_compress_conversations(self, memory_system, mock_llm):
        """测试对话压缩"""
        # 添加足够多的对话以触发压缩（max_windows=5）
        for i in range(7):  # 超过5轮，会触发压缩
            role = "user" if i % 2 == 0 else "assistant"
            memory_system.add_conversation(role, f"测试对话 {i}")

        # 检查压缩是否被调用（LLM 被调用的次数）
        # 由于我们添加了7轮对话，max_windows=5，添加第6轮时触发压缩
        # 压缩会调用 LLM
        assert mock_llm.invoke.called

        # 检查压缩记忆列表
        assert len(memory_system.compressed_memories) > 0

        # 检查压缩记忆文件
        assert memory_system.compressed_memory_file.exists()

    def test_get_recent_conversations(self, memory_system):
        """测试获取最近对话"""
        # 添加多轮对话
        for i in range(4):
            memory_system.add_conversation("user", f"消息 {i}")

        # 获取所有对话
        recent_all = memory_system.get_recent_conversations()
        assert len(recent_all) == 4

        # 获取最近2轮对话
        recent_two = memory_system.get_recent_conversations(2)
        assert len(recent_two) == 2
        assert recent_two[0]["content"] == "消息 2"  # 第3轮
        assert recent_two[1]["content"] == "消息 3"  # 第4轮

    def test_get_context_for_llm(self, memory_system, mock_llm):
        """测试生成 LLM 上下文"""
        # 添加一些对话
        memory_system.add_conversation("user", "用户问题")
        memory_system.add_conversation("assistant", "助手回答")

        # 添加压缩记忆
        memory_system.compressed_memories.append("2025-01-01 10:00:00: 历史摘要")

        # 生成上下文
        context = memory_system.get_context_for_llm()

        # 检查是否包含预期内容
        assert "压缩记忆" in context
        assert "最近对话" in context
        assert "用户问题" in context
        assert "助手回答" in context
        assert "历史摘要" in context

    def test_daily_log_creation(self, memory_system):
        """测试每日日志创建"""
        # 添加对话（应创建日志文件）
        memory_system.add_conversation("user", "日志测试输入")
        memory_system.add_conversation("assistant", "日志测试回复")

        # 检查日志目录
        assert memory_system.log_dir.exists()

        # 检查当天日志文件（需要知道当天日期）
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = memory_system.log_dir / f"{today}.md"

        assert log_file.exists()

        # 检查日志内容
        content = log_file.read_text(encoding='utf-8')
        assert "日志测试输入" in content
        assert "日志测试回复" in content

    def test_clear_and_reset(self, memory_system):
        """测试清空和重置功能"""
        # 添加一些对话和压缩记忆
        memory_system.add_conversation("user", "测试")
        memory_system.compressed_memories.append("测试记忆")

        # 清空对话窗口
        memory_system.clear_conversation_window()
        assert len(memory_system.conversation_window) == 0
        assert len(memory_system.compressed_memories) == 1  # 压缩记忆应保留

        # 重置整个系统
        memory_system.reset_all()
        assert len(memory_system.conversation_window) == 0
        assert len(memory_system.compressed_memories) == 0

    def test_edge_cases(self, memory_system):
        """测试边界情况"""
        # 空对话
        empty = memory_system.get_recent_conversations()
        assert len(empty) == 0

        # 请求超过实际数量的对话
        memory_system.add_conversation("user", "测试")
        recent = memory_system.get_recent_conversactions(10)  # 请求10轮，但只有1轮
        assert len(recent) == 1

        # 窗口大小很小的情况
        small_memory = MemorySystem(llm=mock_llm, max_windows=2)
        for i in range(5):
            small_memory.add_conversation("user", f"消息{i}")
        # 由于压缩机制，窗口大小不会超过 max_windows
        assert len(small_memory.conversation_window) <= 2


if __name__ == "__main__":
    # 直接运行测试（不使用 pytest 时）
    pytest.main([__file__, "-v"])
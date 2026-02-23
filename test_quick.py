#!/usr/bin/env python3
"""
快速测试脚本
测试记忆系统的基本功能，如果真实 LLM 不可用则使用模拟测试。
运行方式：conda activate AgentEnv && python test_quick.py
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

def test_import():
    """测试模块导入"""
    try:
        from memory_system import MemorySystem, create_default_memory_system
        from memory_system.config import MAX_WINDOWS, DATA_DIR
        print("✓ 模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 模块导入失败：{e}")
        return False

def test_with_mock_llm():
    """使用模拟 LLM 测试基本功能"""
    print("\n=== 使用模拟 LLM 测试 ===")

    # 创建模拟 LLM
    mock_llm = Mock()
    mock_llm.invoke.return_value.content = "模拟摘要：用户提供了个人信息"

    try:
        from memory_system import MemorySystem

        # 使用临时目录
        import tempfile
        import shutil
        temp_dir = Path(tempfile.mkdtemp())

        memory = MemorySystem(llm=mock_llm, max_windows=3, data_dir=temp_dir)

        # 测试添加对话
        memory.add_conversation("user", "我叫测试用户")
        memory.add_conversation("assistant", "你好测试用户")

        assert len(memory.get_recent_conversations()) == 2
        print("✓ 对话添加成功")

        # 测试窗口满时压缩
        memory.add_conversation("user", "第二条消息")
        memory.add_conversation("assistant", "第二条回复")
        memory.add_conversation("user", "第三条消息")  # 触发压缩

        # 检查压缩是否被调用
        assert mock_llm.invoke.called
        print("✓ 对话压缩触发成功")

        # 测试文件生成
        assert (temp_dir / "user.md").exists()
        print("✓ 用户信息文件生成成功")

        assert (temp_dir / "logs").exists()
        print("✓ 日志目录生成成功")

        # 清理临时目录
        shutil.rmtree(temp_dir)

        print("✓ 所有模拟测试通过")
        return True

    except Exception as e:
        print(f"✗ 模拟测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_llm():
    """使用真实 LLM 测试（如果配置可用）"""
    print("\n=== 尝试使用真实 LLM 测试 ===")

    try:
        from memory_system import create_default_memory_system

        # 尝试创建默认记忆系统
        memory = create_default_memory_system()
        print("✓ 真实 LLM 初始化成功")

        # 简单测试：添加一轮对话
        memory.add_conversation("user", "这是一条测试消息")
        memory.add_conversation("assistant", "这是一条测试回复")

        print(f"✓ 对话添加成功，当前窗口大小：{len(memory.get_recent_conversations())}")

        # 检查文件
        from memory_system.config import DATA_DIR
        if DATA_DIR.exists():
            print(f"✓ 数据目录存在：{DATA_DIR}")

        if (DATA_DIR / "logs").exists():
            print("✓ 日志目录存在")

        print("✓ 真实 LLM 测试通过（基本功能正常）")
        return True

    except Exception as e:
        print(f"✗ 真实 LLM 测试失败：{e}")
        print("  这可能是由于 API 配置问题，将使用模拟测试。")
        return False

def main():
    """主测试函数"""
    print("记忆系统快速测试")
    print("=" * 50)

    # 检查当前目录
    print(f"工作目录：{os.getcwd()}")

    # 测试导入
    if not test_import():
        print("\n请检查依赖安装：pip install -r requirements.txt")
        sys.exit(1)

    # 尝试真实 LLM 测试
    real_llm_ok = test_with_real_llm()

    # 如果真实 LLM 测试失败，运行模拟测试
    if not real_llm_ok:
        print("\n真实 LLM 测试失败，运行模拟测试...")
        mock_ok = test_with_mock_llm()
        if not mock_ok:
            print("\n✗ 模拟测试也失败，请检查代码。")
            sys.exit(1)

    print("\n" + "=" * 50)
    print("所有测试通过！记忆系统基本功能正常。")
    print("\n接下来可以运行：")
    print("1. python example_usage.py - 查看完整示例")
    print("2. pytest tests/ -v - 运行完整单元测试")
    print("3. 在您的项目中导入并使用 MemorySystem")

    return 0

if __name__ == "__main__":
    sys.exit(main())
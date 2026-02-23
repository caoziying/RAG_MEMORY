#!/usr/bin/env python3
"""
记忆系统使用示例

演示如何使用 MemorySystem 管理对话记忆、提取用户信息和生成日志。
运行前请确保已设置环境变量或已激活 conda 环境。
"""

import os
import sys
from pathlib import Path

# 添加当前目录到 Python 路径，以便导入 memory_system
sys.path.insert(0, str(Path(__file__).parent))

from memory_system import MemorySystem, create_default_memory_system
from langchain_openai import ChatOpenAI


def example_with_default_llm():
    """示例1：使用默认配置创建记忆系统"""
    print("=== 示例1：使用默认配置创建记忆系统 ===")

    try:
        # 使用默认配置（需要环境变量 MY_API_KEY, MY_API_BASE, MY_MODEL）
        memory = create_default_memory_system()
        print("✓ 记忆系统创建成功（使用默认配置）")
    except Exception as e:
        print(f"✗ 创建失败：{e}")
        print("请确保已设置正确的环境变量。")
        return None

    return memory


def example_with_custom_llm():
    """示例2：使用自定义 LLM 创建记忆系统"""
    print("\n=== 示例2：使用自定义 LLM 创建记忆系统 ===")

    try:
        # 自定义 LLM 配置（使用校内模型）
        llm = ChatOpenAI(
            model="Qwen3-32B-FP8",
            api_key=os.getenv("MY_API_KEY", ""),
            base_url=os.getenv("MY_API_BASE", "https://api.chat.csu.edu.cn/v1"),
            temperature=0.0,
            stream=False
        )

        memory = MemorySystem(llm=llm, max_windows=5)
        print("✓ 记忆系统创建成功（使用自定义 LLM）")
    except Exception as e:
        print(f"✗ 创建失败：{e}")
        return None

    return memory


def demonstrate_basic_operations(memory: MemorySystem):
    """演示基本操作：添加对话、提取信息、压缩记忆"""
    print("\n=== 演示基本操作 ===")

    # 模拟对话
    conversations = [
        ("user", "我叫王小明，今年28岁，是一名数据分析师。"),
        ("assistant", "你好王小明！数据分析师是个很有趣的职业。"),
        ("user", "我喜欢爬山和阅读科幻小说。"),
        ("assistant", "很好的爱好！爬山锻炼身体，科幻小说激发想象力。"),
        ("user", "我住在北京，养了一只猫叫咪咪。"),
        ("assistant", "北京是个美丽的城市，养猫也很温馨。"),
        ("user", "我最近在学习机器学习。"),
        ("assistant", "机器学习是数据分析的重要技能。"),
        ("user", "我的工作经常需要用到Python和SQL。"),
        ("assistant", "Python和SQL确实是数据分析的核心工具。"),
    ]

    print(f"添加 {len(conversations)} 轮对话...")
    for role, content in conversations:
        memory.add_conversation(role, content)
        print(f"  {role}: {content[:30]}...")

    # 显示当前状态
    print(f"\n当前对话窗口大小：{len(memory.get_recent_conversations())}")
    print(f"压缩记忆数量：{len(memory.get_compressed_memories())}")

    # 显示最近3轮对话
    print("\n最近3轮对话：")
    for conv in memory.get_recent_conversations(3):
        print(f"  {conv['timestamp']} {conv['role']}: {conv['content'][:30]}...")

    # 显示压缩记忆
    compressed = memory.get_compressed_memories()
    if compressed:
        print("\n压缩记忆（摘要）：")
        for i, mem in enumerate(compressed[-3:], 1):  # 显示最近3条
            print(f"  {i}. {mem[:50]}...")

    # 获取 LLM 上下文
    context = memory.get_context_for_llm(max_recent=3)
    print(f"\nLLM上下文预览（前200字符）：")
    print(context[:200] + "...")

    # 检查生成的文件
    print("\n=== 生成的文件 ===")
    data_dir = Path("data")
    if data_dir.exists():
        for file in data_dir.rglob("*"):
            if file.is_file():
                print(f"  {file.relative_to(data_dir.parent)}")
    else:
        print("  数据目录尚未创建")


def demonstrate_user_info_extraction(memory: MemorySystem):
    """演示用户信息提取功能"""
    print("\n=== 演示用户信息提取 ===")

    # 添加包含个人信息的对话
    test_inputs = [
        "我的邮箱是 wangxiaoming@example.com，电话是 13800138000。",
        "我毕业于清华大学计算机科学专业。",
        "我对人工智能和区块链技术很感兴趣。",
        "我计划明年去日本旅游。"
    ]

    for user_input in test_inputs:
        print(f"\n用户输入：{user_input}")
        memory.add_conversation("user", user_input)

    # 检查 user.md 文件
    user_file = Path("data/user.md")
    if user_file.exists():
        print(f"\nuser.md 文件内容预览：")
        with open(user_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[-10:]:  # 显示最后10行
                print("  " + line.rstrip())
    else:
        print("\nuser.md 文件尚未创建")


def demonstrate_daily_logs():
    """演示每日日志功能"""
    print("\n=== 演示每日日志 ===")

    # 创建新的记忆系统实例（使用模拟 LLM 避免实际 API 调用）
    from unittest.mock import Mock
    mock_llm = Mock()
    mock_llm.invoke.return_value.content = "测试摘要"

    memory = MemorySystem(llm=mock_llm, max_windows=3)

    # 添加一些对话
    memory.add_conversation("user", "这是今天的测试对话。")
    memory.add_conversation("assistant", "这是测试回复。")

    # 检查日志文件
    log_dir = Path("data/logs")
    if log_dir.exists():
        print(f"日志目录：{log_dir}")
        for log_file in log_dir.glob("*.md"):
            print(f"  日志文件：{log_file.name}")
            # 显示文件大小
            print(f"    大小：{log_file.stat().st_size} 字节")
    else:
        print("日志目录尚未创建")


def main():
    """主函数：运行所有示例"""
    print("记忆系统使用示例")
    print("=" * 50)

    # 检查环境变量
    api_key = os.getenv("MY_API_KEY")
    if not api_key:
        print("警告：环境变量 MY_API_KEY 未设置，某些示例可能无法运行。")

    # 示例1：使用默认配置
    memory1 = example_with_default_llm()
    if memory1:
        demonstrate_basic_operations(memory1)

    # 示例2：使用自定义 LLM
    memory2 = example_with_custom_llm()
    if memory2:
        demonstrate_user_info_extraction(memory2)

    # 演示每日日志（使用模拟 LLM）
    demonstrate_daily_logs()

    print("\n" + "=" * 50)
    print("示例完成！")
    print("请检查生成的 data/ 目录查看文件。")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
清空记忆系统脚本
用于清空本地记忆文件和向量数据库中的记忆。
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from memory_system.core.config import DATA_DIR, LOG_DIR, USER_INFO_FILE, COMPRESSED_MEMORY_FILE
from memory_system.vector import VectorMemorySystem


def clear_local_memories(verbose=True):
    """
    清空本地记忆文件。

    包括：
    1. user.md（用户信息）
    2. compressed_memory.md（压缩记忆）
    3. logs/ 目录下的所有日志文件
    4. 对话窗口（通过MemorySystem.reset_all()）
    """
    if verbose:
        print("正在清空本地记忆...")

    # 清空文件内容（或删除文件）
    files_to_clear = [USER_INFO_FILE, COMPRESSED_MEMORY_FILE]
    for file_path in files_to_clear:
        if file_path.exists():
            try:
                # 清空文件内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('')
                if verbose:
                    print(f"  ✓ 已清空: {file_path}")
            except Exception as e:
                print(f"  ✗ 清空文件失败 {file_path}: {e}")
        else:
            if verbose:
                print(f"  ⓘ 文件不存在: {file_path}")

    # 删除日志目录中的所有文件
    if LOG_DIR.exists():
        try:
            for log_file in LOG_DIR.glob("*.md"):
                log_file.unlink()
            if verbose:
                print(f"  ✓ 已删除日志目录中的所有文件: {LOG_DIR}")
        except Exception as e:
            print(f"  ✗ 删除日志文件失败: {e}")
    else:
        if verbose:
            print(f"  ⓘ 日志目录不存在: {LOG_DIR}")

    # 尝试通过MemorySystem重置对话窗口和压缩记忆
    try:
        from memory_system.memory import MemorySystem
        memory = MemorySystem()
        memory.reset_all()  # 清空窗口和压缩记忆列表
        if verbose:
            print("  ✓ 已重置记忆系统对话窗口")
    except Exception as e:
        print(f"  ⓘ 无法重置记忆系统: {e}（可能未安装依赖）")

    if verbose:
        print("本地记忆清空完成！")


def clear_vector_memories(verbose=True):
    """
    清空向量数据库中的记忆。

    使用VectorMemorySystem的clear_vector_store()方法。
    """
    if verbose:
        print("正在清空向量数据库记忆...")

    try:
        # 创建VectorMemorySystem实例
        memory = VectorMemorySystem()

        # 清空向量存储
        success = memory.clear_vector_store()

        if success:
            if verbose:
                print("  ✓ 向量数据库记忆清空成功")
        else:
            print("  ✗ 向量数据库记忆清空失败（可能未启用向量存储）")

    except Exception as e:
        print(f"  ✗ 清空向量数据库失败: {e}")

    if verbose:
        print("向量数据库记忆清空完成！")


def clear_all_memories(verbose=True):
    """清空所有记忆（本地和向量数据库）"""
    if verbose:
        print("=" * 60)
        print("开始清空所有记忆")
        print("=" * 60)

    clear_local_memories(verbose)
    print()  # 空行
    clear_vector_memories(verbose)

    if verbose:
        print("=" * 60)
        print("所有记忆清空完成！")
        print("=" * 60)


def main():
    """主函数，解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="清空记忆系统脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --local           # 只清空本地记忆文件
  %(prog)s --vector          # 只清空向量数据库记忆
  %(prog)s --all             # 清空所有记忆
  %(prog)s --local --vector  # 清空本地和向量记忆（同--all）

如果不指定任何选项，默认清空所有记忆。
        """
    )

    parser.add_argument(
        "--local",
        action="store_true",
        help="清空本地记忆文件（user.md, compressed_memory.md, 日志等）"
    )

    parser.add_argument(
        "--vector",
        action="store_true",
        help="清空向量数据库中的记忆"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="清空所有记忆（本地和向量数据库）"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="显示详细信息（默认）"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="静默模式，不显示输出"
    )

    args = parser.parse_args()

    # 处理静默模式
    verbose = not args.quiet if args.quiet else args.verbose

    # 确定要执行的操作
    if args.all or (not args.local and not args.vector):
        # 默认清空所有
        clear_all_memories(verbose)
    else:
        if args.local:
            clear_local_memories(verbose)
        if args.vector:
            clear_vector_memories(verbose)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

try:
    import memory_system
    print("✓ memory_system 导入成功")
    print(f"  版本: {memory_system.__version__}")

    # 测试核心模块
    from memory_system.core.memory import MemorySystem
    print("✓ core.memory 导入成功")

    from memory_system.core.config import DATA_DIR, LOG_DIR
    print("✓ core.config 导入成功")

    from memory_system.core.utils import setup_logging
    print("✓ core.utils 导入成功")

    # 测试向量模块
    try:
        from memory_system.vector import VectorMemorySystem
        print("✓ vector 模块导入成功")
    except ImportError as e:
        print("⚠ vector 模块导入失败（可能缺少依赖）:", e)

    print("\n所有导入测试完成！")

except Exception as e:
    print("✗ 导入失败:", e)
    import traceback
    traceback.print_exc()
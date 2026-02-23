#!/usr/bin/env python
"""
测试Reranker单例模式重构

测试内容：
1. 单例模式：多个实例是否为同一个对象
2. 模式检测：是否正确检测可用的rerank模式
3. 状态信息：get_status方法是否返回正确信息
4. 降级逻辑：不同模式下的行为
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_system.vector.reranker import Reranker, RerankMode
from memory_system.vector.config import VectorRetrievalConfig


def test_singleton_pattern():
    """测试单例模式"""
    print("=" * 60)
    print("测试单例模式")
    print("=" * 60)

    # 创建第一个实例
    reranker1 = Reranker()
    print(f"实例1 ID: {id(reranker1)}")
    print(f"实例1 状态: {reranker1.get_status()}")

    # 创建第二个实例（应该返回同一个实例）
    reranker2 = Reranker()
    print(f"实例2 ID: {id(reranker2)}")
    print(f"实例2 状态: {reranker2.get_status()}")

    # 检查是否为同一个对象
    if id(reranker1) == id(reranker2):
        print("✅ 单例模式测试通过：两个实例是同一个对象")
    else:
        print("❌ 单例模式测试失败：两个实例是不同的对象")

    # 检查是否共享状态
    reranker1.mode = RerankMode.DIRECT_API
    if reranker2.mode == RerankMode.DIRECT_API:
        print("✅ 状态共享测试通过：修改一个实例影响另一个")
    else:
        print("❌ 状态共享测试失败：实例状态不共享")

    return id(reranker1) == id(reranker2)


def test_mode_detection():
    """测试模式检测"""
    print("\n" + "=" * 60)
    print("测试模式检测")
    print("=" * 60)

    reranker = Reranker()
    status = reranker.get_status()

    print(f"当前模式: {status['mode']}")
    print(f"客户端类型: {status['client_type']}")
    print(f"Rerank模型: {status['rerank_model']}")
    print(f"Embedding模型: {status['embedding_model']}")
    print(f"API基础地址: {status['api_base']}")
    print(f"是否可用: {status['available']}")

    # 根据可用性检查模式
    if status['available']:
        print(f"✅ Reranker可用，模式: {status['mode']}")
        if status['mode'] == RerankMode.DIRECT_API:
            print("   使用直接API调用模式")
        elif status['mode'] == RerankMode.EMBEDDINGS_FALLBACK:
            print("   使用embeddings降级模式")
        elif status['mode'] == RerankMode.DEFAULT_FALLBACK:
            print("   使用默认值降级模式")
    else:
        print(f"⚠️  Reranker不可用，模式: {status['mode']}")

    return status['available']


def test_connection():
    """测试连接"""
    print("\n" + "=" * 60)
    print("测试连接")
    print("=" * 60)

    reranker = Reranker()

    if not reranker.is_available():
        print("⚠️  Reranker不可用，跳过连接测试")
        return False

    print(f"正在测试连接，模式: {reranker.mode.value}...")

    success = reranker.test_connection()
    if success:
        print("✅ 连接测试成功")
    else:
        print("❌ 连接测试失败")

    return success


def test_rerank_functionality():
    """测试rerank功能"""
    print("\n" + "=" * 60)
    print("测试rerank功能")
    print("=" * 60)

    reranker = Reranker()

    if not reranker.is_available():
        print("⚠️  Reranker不可用，跳过功能测试")
        return False

    # 创建测试数据
    test_candidates = [
        {
            "id": "1",
            "text": "这是一个关于人工智能的测试文档，讨论机器学习和深度学习。",
            "score": 0.8,
            "metadata": {"source": "test"}
        },
        {
            "id": "2",
            "text": "另一个文档，关于自然语言处理和语言模型。",
            "score": 0.7,
            "metadata": {"source": "test"}
        },
        {
            "id": "3",
            "text": "第三个文档，讨论计算机视觉和图像识别技术。",
            "score": 0.6,
            "metadata": {"source": "test"}
        }
    ]

    query = "人工智能研究方向"

    print(f"查询: {query}")
    print(f"候选数量: {len(test_candidates)}")
    print(f"Rerank模式: {reranker.mode.value}")

    try:
        results = reranker.rerank(
            query=query,
            candidates=test_candidates,
            top_n=2,
            batch_size=2,
            retry_count=1
        )

        print(f"返回结果数量: {len(results)}")

        for i, result in enumerate(results):
            print(f"\n结果 {i+1}:")
            print(f"  ID: {result.get('id')}")
            print(f"  原始分数: {result.get('score', 0):.3f}")
            print(f"  Rerank分数: {result.get('rerank_score', 0):.3f}")
            print(f"  最终分数: {result.get('final_score', 0):.3f}")
            print(f"  文本预览: {result.get('text', '')[:50]}...")

        print("✅ Rerank功能测试完成")
        return True

    except Exception as e:
        print(f"❌ Rerank功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_configs():
    """测试不同配置"""
    print("\n" + "=" * 60)
    print("测试不同配置")
    print("=" * 60)

    # 测试默认配置
    default_reranker = Reranker()
    print(f"默认配置实例 ID: {id(default_reranker)}")

    # 测试自定义配置
    custom_config = VectorRetrievalConfig(
        rerank_model="bge-reranker-large",
        embedding_model="bge-m3",
        api_base="https://api.chat.csu.edu.cn/v1"
    )

    # 注意：由于是单例模式，即使传入不同配置，也返回同一个实例
    # 但实例使用第一次初始化的配置
    custom_reranker = Reranker(custom_config)
    print(f"自定义配置实例 ID: {id(custom_reranker)}")

    if id(default_reranker) == id(custom_reranker):
        print("✅ 单例模式配置测试通过：不同配置返回同一个实例")
        print("   注意：实例使用第一次初始化的配置，后续配置被忽略")
    else:
        print("❌ 单例模式配置测试失败：不同配置返回不同实例")

    return id(default_reranker) == id(custom_reranker)


def main():
    """主测试函数"""
    print("Reranker单例模式重构测试")
    print("=" * 60)

    tests_passed = 0
    tests_total = 5

    # 测试1：单例模式
    if test_singleton_pattern():
        tests_passed += 1

    # 测试2：模式检测
    if test_mode_detection():
        tests_passed += 1

    # 测试3：连接测试
    if test_connection():
        tests_passed += 1

    # 测试4：rerank功能
    if test_rerank_functionality():
        tests_passed += 1

    # 测试5：不同配置
    if test_different_configs():
        tests_passed += 1

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"通过测试: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查以上输出")

    return tests_passed == tests_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
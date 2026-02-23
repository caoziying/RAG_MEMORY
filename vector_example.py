#!/usr/bin/env python3
"""
增强记忆系统使用示例
演示向量检索功能的使用。

运行前请确保：
1. 安装向量检索依赖：pip install pymilvus
2. 启动 Milvus Docker 服务：cd docker && docker-compose up -d
3. 设置环境变量（API密钥等）
"""

import os
import sys
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from memory_system.vector import VectorMemorySystem, create_default_vector_memory_system
from memory_system.vector.config import VectorRetrievalConfig


def check_dependencies():
    """检查依赖是否安装"""
    try:
        import pymilvus
        print("✓ pymilvus 已安装")
    except ImportError:
        print("✗ pymilvus 未安装，请运行：pip install pymilvus")
        return False

    return True


def check_milvus_connection():
    """检查 Milvus 连接"""
    try:
        from pymilvus import connections, utility
        connections.connect(host="localhost", port=19530)
        collections = utility.list_collections()
        print(f"✓ Milvus 连接成功，已有集合：{collections}")
        connections.disconnect("default")
        return True
    except Exception as e:
        print(f"✗ Milvus 连接失败：{e}")
        print("  请确保已启动 Milvus Docker 服务")
        print("  cd docker && docker-compose up -d")
        return False


def example_basic_usage():
    """示例1：基本使用"""
    print("\n=== 示例1：基本使用 ===")

    try:
        # 创建增强记忆系统（使用默认配置）
        memory = create_default_vector_memory_system()
        print("✓ 增强记忆系统创建成功")

        # 检查向量检索状态
        stats = memory.get_vector_stats()
        print(f"向量检索状态：{stats}")

        # 添加对话（会自动存储到向量数据库）
        conversations = [
            ("user", "我叫张明，是一名软件工程师，擅长Python和机器学习。"),
            ("assistant", "你好张明！很高兴认识你，机器学习是个很有前景的领域。"),
            ("user", "我最近在研究深度学习在自然语言处理中的应用。"),
            ("assistant", "深度学习在NLP领域确实取得了很大进展，比如Transformer模型。"),
            ("user", "我在北京工作，平时喜欢爬山和阅读。"),
            ("assistant", "北京有很多适合爬山的地方，阅读也是很好的爱好。"),
            ("user", "我的邮箱是 zhangming@example.com，电话是 13800138000。"),
            ("assistant", "好的，我已经记下了你的联系方式。"),
        ]

        print(f"添加 {len(conversations)} 轮对话...")
        for role, content in conversations:
            memory.add_conversation(role, content)
            print(f"  {role}: {content[:30]}...")

        # 检索相关记忆
        print("\n检索相关记忆...")
        queries = [
            "联系方式",
            "兴趣爱好",
            "工作技能",
            "研究方向"
        ]

        for query in queries:
            print(f"\n查询：'{query}'")
            results = memory.retrieve_memories(query, top_k=10)

            if results:
                print(f"  找到 {len(results)} 条相关记忆：")
                for i, result in enumerate(results[:3]):  # 显示前3条
                    score = result.get("score", 0)
                    text = result.get("text", "")[:60]
                    print(f"    {i+1}. [{score:.3f}] {text}...")
            else:
                print("  未找到相关记忆")

        return memory

    except Exception as e:
        print(f"✗ 示例失败：{e}")
        import traceback
        traceback.print_exc()
        return None


def example_advanced_retrieval(memory):
    """示例2：高级检索功能"""
    print("\n=== 示例2：高级检索功能 ===")

    if not memory:
        print("需要先运行示例1创建记忆系统")
        return

    try:
        # 搜索相似对话（按对话分组）
        query = "个人信息"
        print(f"\n搜索相似对话：'{query}'")

        conversations = memory.search_similar_conversations(query, max_results=3)

        print(f"找到 {len(conversations)} 个相关对话：")
        for i, conv in enumerate(conversations):
            print(f"\n对话 {i+1} (分数: {conv.get('score', 0):.3f}):")
            print(f"  内容: {conv.get('text', '')[:80]}...")
            print(f"  包含记忆数: {conv.get('memory_count', 0)}")

        # 获取向量统计信息
        print("\n向量存储统计：")
        stats = memory.get_vector_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"✗ 高级检索失败：{e}")


def example_custom_config():
    """示例3：自定义配置"""
    print("\n=== 示例3：自定义配置 ===")

    try:
        # 自定义配置
        config = VectorRetrievalConfig(
            vector_weight=0.7,
            bm25_weight=0.3,
            top_k=100,               # 召回最多100条
            rerank_top_n=5,          # rerank后返回5条
            similarity_threshold=0.5,
            chunk_size=512,
            chunk_overlap=50,
            embedding_model="bge-m3",
            rerank_model="bge-reranker-large",
            milvus_host="localhost",
            milvus_port=19530
        )

        print("自定义配置：")
        print(f"  混合权重: 向量{config.vector_weight} + BM25{config.bm25_weight}")
        print(f"  召回控制: top_k={config.top_k}, rerank_top_n={config.rerank_top_n}")
        print(f"  相似度阈值: {config.similarity_threshold}")
        print(f"  分块大小: {config.chunk_size} (重叠{config.chunk_overlap})")
        print(f"  模型: {config.embedding_model}, {config.rerank_model}")

        # 使用自定义配置创建记忆系统
        from memory_system.vector import VectorMemorySystem
        memory = VectorMemorySystem(vector_config=config)

        print("✓ 使用自定义配置的记忆系统创建成功")
        return memory

    except Exception as e:
        print(f"✗ 自定义配置失败：{e}")
        return None


def example_without_milvus():
    """示例4：无Milvus的降级模式"""
    print("\n=== 示例4：无Milvus的降级模式 ===")

    try:
        # 创建不启用向量存储的记忆系统
        from memory_system.vector import VectorMemorySystem
        memory = VectorMemorySystem(enable_vector_store=False)

        print("✓ 降级模式记忆系统创建成功（使用原始功能）")

        # 添加一些对话
        memory.add_conversation("user", "测试对话1")
        memory.add_conversation("assistant", "测试回复1")

        # 尝试检索（会回退到原始方法）
        results = memory.retrieve_memories("测试")
        print(f"检索结果数：{len(results)}")
        print("  注意：在降级模式下，检索返回最近对话")

        return memory

    except Exception as e:
        print(f"✗ 降级模式失败：{e}")
        return None


def main():
    """主函数"""
    print("增强记忆系统使用示例")
    print("=" * 60)

    # 检查依赖
    if not check_dependencies():
        return

    # 检查Milvus连接
    check_milvus_connection()

    # 运行示例
    memory = example_basic_usage()

    if memory:
        example_advanced_retrieval(memory)

    example_custom_config()
    example_without_milvus()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("\n总结：")
    print("1. 增强记忆系统集成了向量检索功能")
    print("2. 支持混合检索（0.7向量 + 0.3 BM25）")
    print("3. 支持Rerank重排序（召回100条 → 返回5条）")
    print("4. 自动分块和存储到Milvus")
    print("5. 降级模式：无Milvus时使用原始功能")


if __name__ == "__main__":
    main()
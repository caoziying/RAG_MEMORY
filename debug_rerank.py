#!/usr/bin/env python3
"""
调试rerank API调用脚本
帮助诊断bge-reranker-large模型调用失败的原因。
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from memory_system.vector.config import VectorRetrievalConfig
from memory_system.vector.reranker import Reranker


def test_api_models():
    """测试API端点支持的模型"""
    print("=" * 60)
    print("测试API端点支持的模型")
    print("=" * 60)

    config = VectorRetrievalConfig()
    print(f"API端点: {config.api_base}")
    print(f"Rerank模型: {config.rerank_model}")
    print(f"Embedding模型: {config.embedding_model}")
    print(f"API Key: {config.api_key[:10]}...")

    # 尝试列出可用模型
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base
        )

        print("\n尝试获取可用模型列表...")
        try:
            models = client.models.list()
            model_ids = [model.id for model in models.data]
            print(f"API端点支持 {len(model_ids)} 个模型:")
            for model_id in sorted(model_ids)[:20]:  # 显示前20个
                print(f"  - {model_id}")
            if len(model_ids) > 20:
                print(f"  ... 还有 {len(model_ids) - 20} 个模型")

            # 检查bge-reranker-large是否在列表中
            if config.rerank_model in model_ids:
                print(f"\n✓ 模型 '{config.rerank_model}' 在可用模型列表中")
            else:
                print(f"\n✗ 模型 '{config.rerank_model}' 不在可用模型列表中")
                print("可能的原因:")
                print("  1. API端点不支持此模型")
                print("  2. 模型名称不正确")
                print("  3. 需要特定的模型前缀")

            # 检查embedding模型
            if config.embedding_model in model_ids:
                print(f"✓ 模型 '{config.embedding_model}' 在可用模型列表中")
            else:
                print(f"✗ 模型 '{config.embedding_model}' 不在可用模型列表中")

        except Exception as e:
            print(f"获取模型列表失败: {e}")
            print("可能的原因:")
            print("  1. API端点不支持/models端点")
            print("  2. 权限不足")

    except ImportError:
        print("OpenAI客户端不可用，请安装: pip install openai")


def test_rerank_calls():
    """测试不同的rerank调用方式"""
    print("\n" + "=" * 60)
    print("测试rerank调用方式")
    print("=" * 60)

    reranker = Reranker()

    # 测试单个rerank调用
    query = "联系方式"
    document = "我的邮箱是 zhangming@example.com，电话是 13800138000。"

    print(f"测试查询: '{query}'")
    print(f"测试文档: '{document[:50]}...'")

    # 测试各种调用方式
    test_cases = [
        ("completions", "直接调用completions API"),
        ("chat", "调用chat completions API"),
        ("embeddings", "使用embeddings相似度")
    ]

    for method_name, description in test_cases:
        print(f"\n测试方法: {description}")
        try:
            # 这里我们无法直接测试，但可以检查客户端状态
            if method_name == "completions":
                if reranker.client is not None:
                    print("  ✓ OpenAI客户端可用")
                    # 尝试实际调用
                    try:
                        score = reranker.rerank_single(query, document)
                        print(f"  ✓ 调用成功，分数: {score:.3f}")
                    except Exception as e:
                        print(f"  ✗ 调用失败: {e}")
                else:
                    print("  ✗ OpenAI客户端不可用")
            elif method_name == "chat":
                print("  ⓘ chat completions将在completions失败时尝试")
            elif method_name == "embeddings":
                if reranker.client is not None:
                    print("  ✓ OpenAI客户端可用（用于embeddings）")
                else:
                    print("  ✗ OpenAI客户端不可用")
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")


def test_direct_api_call():
    """直接测试API调用"""
    print("\n" + "=" * 60)
    print("直接API调用测试")
    print("=" * 60)

    config = VectorRetrievalConfig()

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base
        )

        # 测试1: 测试completions端点
        print("\n1. 测试completions端点...")
        try:
            response = client.completions.create(
                model=config.rerank_model,
                prompt="测试",
                max_tokens=5,
                temperature=0.0
            )
            print(f"  ✓ completions端点可用")
            print(f"    响应: {response.choices[0].text}")
        except Exception as e:
            print(f"  ✗ completions端点错误: {e}")

        # 测试2: 测试chat completions端点
        print("\n2. 测试chat completions端点...")
        try:
            response = client.chat.completions.create(
                model=config.rerank_model,
                messages=[{"role": "user", "content": "测试"}],
                max_tokens=5,
                temperature=0.0
            )
            print(f"  ✓ chat completions端点可用")
            print(f"    响应: {response.choices[0].message.content}")
        except Exception as e:
            print(f"  ✗ chat completions端点错误: {e}")

        # 测试3: 测试embeddings端点
        print("\n3. 测试embeddings端点...")
        try:
            response = client.embeddings.create(
                model=config.embedding_model,
                input=["测试文本"]
            )
            print(f"  ✓ embeddings端点可用")
            print(f"    向量维度: {len(response.data[0].embedding)}")
        except Exception as e:
            print(f"  ✗ embeddings端点错误: {e}")

    except ImportError:
        print("OpenAI客户端不可用")
    except Exception as e:
        print(f"客户端初始化失败: {e}")


def main():
    """主函数"""
    print("bge-reranker-large调试工具")
    print("=" * 60)

    # 检查环境变量
    print("环境变量检查:")
    api_key = os.getenv("MY_API_KEY")
    api_base = os.getenv("MY_API_BASE")
    model = os.getenv("MY_MODEL")

    if api_key:
        print(f"  ✓ MY_API_KEY: {api_key[:10]}...")
    else:
        print("  ✗ MY_API_KEY未设置")

    if api_base:
        print(f"  ✓ MY_API_BASE: {api_base}")
    else:
        print("  ✗ MY_API_BASE未设置")

    if model:
        print(f"  ✓ MY_MODEL: {model}")
    else:
        print("  ✗ MY_MODEL未设置")

    # 运行测试
    test_api_models()
    test_direct_api_call()
    test_rerank_calls()

    print("\n" + "=" * 60)
    print("调试完成")
    print("=" * 60)

    print("\n建议:")
    print("1. 如果bge-reranker-large不在模型列表中，请咨询API提供商")
    print("2. 如果completions端点不可用，可以考虑使用chat completions")
    print("3. 如果所有API都失败，可以考虑使用本地模型或embeddings相似度")
    print("4. 检查API密钥是否有足够的权限")


if __name__ == "__main__":
    main()
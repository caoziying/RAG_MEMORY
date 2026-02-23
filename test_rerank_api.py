#!/usr/bin/env python3
"""
测试bge-reranker-large API调用
诊断400错误的具体原因
"""

import os
import sys
import json
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from memory_system.vector.config import VectorRetrievalConfig


def test_model_variants():
    """测试不同的模型名称变体"""
    config = VectorRetrievalConfig()
    base_url = config.api_base
    api_key = config.api_key

    print("=" * 60)
    print("测试不同的模型名称变体")
    print("=" * 60)
    print(f"API端点: {base_url}")
    print(f"原始模型名称: {config.rerank_model}")

    # 可能的模型名称变体
    model_variants = [
        config.rerank_model,  # 原始
        "bge-reranker-large",  # 确保小写
        "BGE-Reranker-Large",  # 原始大小写
        "openai/bge-reranker-large",  # 带openai前缀
        "bge-reranker-large-zh",  # 中文版本
        "bge-reranker-v2.5",  # 其他版本
        "text-embedding-ada-002",  # 测试其他模型
    ]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 首先测试列出模型
        print("\n1. 测试列出可用模型...")
        try:
            models = client.models.list()
            available_models = [model.id for model in models.data]
            print(f"可用模型 ({len(available_models)}个):")
            for model in sorted(available_models)[:30]:  # 显示前30个
                print(f"  - {model}")
            if len(available_models) > 30:
                print(f"  ... 还有 {len(available_models) - 30} 个模型")
        except Exception as e:
            print(f"列出模型失败: {e}")

        # 测试每个模型变体
        print("\n2. 测试不同的模型名称...")
        test_prompt = "query: test query document: test document"

        for model_name in model_variants:
            print(f"\n测试模型: {model_name}")

            # 检查是否在可用模型中
            if 'available_models' in locals() and model_name in available_models:
                print(f"  ✓ 在可用模型列表中")

            # 测试completions
            try:
                print(f"  测试completions...")
                response = client.completions.create(
                    model=model_name,
                    prompt=test_prompt,
                    max_tokens=5,
                    temperature=0.0
                )
                print(f"  ✓ completions成功: {response.choices[0].text}")
            except Exception as e:
                error_msg = str(e)
                if hasattr(e, 'response') and e.response:
                    try:
                        error_details = e.response.json()
                        error_msg = f"{error_msg} - {error_details}"
                    except:
                        pass
                print(f"  ✗ completions失败: {error_msg}")

            # 测试chat completions
            try:
                print(f"  测试chat completions...")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": test_prompt}],
                    max_tokens=5,
                    temperature=0.0
                )
                print(f"  ✓ chat completions成功: {response.choices[0].message.content}")
            except Exception as e:
                error_msg = str(e)
                if hasattr(e, 'response') and e.response:
                    try:
                        error_details = e.response.json()
                        error_msg = f"{error_msg} - {error_details}"
                    except:
                        pass
                print(f"  ✗ chat completions失败: {error_msg}")

    except ImportError:
        print("OpenAI客户端不可用")
    except Exception as e:
        print(f"初始化客户端失败: {e}")


def test_api_endpoints():
    """测试不同的API端点"""
    print("\n" + "=" * 60)
    print("测试API端点支持")
    print("=" * 60)

    config = VectorRetrievalConfig()

    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.api_key, base_url=config.api_base)

        # 测试embeddings端点（已知工作）
        print("\n1. 测试embeddings端点...")
        try:
            response = client.embeddings.create(
                model=config.embedding_model,
                input=["测试文本"]
            )
            print(f"  ✓ embeddings成功，维度: {len(response.data[0].embedding)}")
        except Exception as e:
            print(f"  ✗ embeddings失败: {e}")

        # 测试completions端点
        print("\n2. 测试completions端点...")
        try:
            # 先用一个简单的模型测试端点是否工作
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",  # 常见模型
                prompt="Hello",
                max_tokens=5,
                temperature=0.0
            )
            print(f"  ✓ completions端点可用: {response.choices[0].text}")
        except Exception as e:
            print(f"  ✗ completions端点错误: {e}")

        # 测试chat completions端点
        print("\n3. 测试chat completions端点...")
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # 常见模型
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                temperature=0.0
            )
            print(f"  ✓ chat completions端点可用: {response.choices[0].message.content}")
        except Exception as e:
            print(f"  ✗ chat completions端点错误: {e}")

    except Exception as e:
        print(f"测试失败: {e}")


def test_input_formats():
    """测试不同的输入格式"""
    print("\n" + "=" * 60)
    print("测试不同的输入格式")
    print("=" * 60)

    config = VectorRetrievalConfig()

    # 不同的输入格式
    query = "联系方式"
    document = "我的邮箱是 zhangming@example.com"

    formats = [
        f"query: {query} document: {document}",  # 标准BGE格式
        f"Query: {query}\nDocument: {document}",  # 换行格式
        f"{query} [SEP] {document}",  # SEP分隔符
        f"请评估相关度: {query} -> {document}",  # 中文指令
        f"relevance score between: {query} and {document}",  # 英文指令
    ]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.api_key, base_url=config.api_base)

        for i, prompt in enumerate(formats, 1):
            print(f"\n格式 {i}: {prompt[:80]}...")
            try:
                response = client.completions.create(
                    model=config.rerank_model,
                    prompt=prompt,
                    max_tokens=5,
                    temperature=0.0
                )
                print(f"  ✓ 成功: {response.choices[0].text}")
            except Exception as e:
                error_msg = str(e)
                if hasattr(e, 'response') and e.response:
                    try:
                        error_details = e.response.json()
                        error_msg = f"{error_msg} - {error_details}"
                    except:
                        pass
                print(f"  ✗ 失败: {error_msg}")

    except Exception as e:
        print(f"测试失败: {e}")


def main():
    """主函数"""
    print("bge-reranker-large API调用诊断工具")
    print("=" * 60)

    # 检查环境
    print("环境检查:")
    config = VectorRetrievalConfig()
    print(f"  API端点: {config.api_base}")
    print(f"  Rerank模型: {config.rerank_model}")
    print(f"  Embedding模型: {config.embedding_model}")
    print(f"  API Key: {config.api_key[:10]}...")

    # 运行测试
    test_model_variants()
    test_api_endpoints()
    test_input_formats()

    print("\n" + "=" * 60)
    print("诊断建议")
    print("=" * 60)

    print("""
基于测试结果的可能解决方案:

1. 如果bge-reranker-large不在可用模型列表中:
   - 联系API提供商确认支持的模型
   - 尝试其他模型名称变体
   - 使用embeddings相似度作为替代

2. 如果completions端点不可用:
   - 尝试使用chat completions
   - 检查端点是否支持所需接口

3. 如果所有API都失败:
   - 检查API密钥权限
   - 确认端点URL正确
   - 考虑使用本地模型或替代服务

4. 输入格式问题:
   - 尝试不同的prompt格式
   - 确保符合BGE-Reranker的预期格式
""")

    print("\n快速修复建议:")
    print("1. 在config.py中修改rerank_model为可用的模型")
    print("2. 或使用embeddings相似度作为回退方案")
    print("3. 联系API提供商获取支持模型列表")


if __name__ == "__main__":
    main()
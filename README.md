<<<<<<< HEAD
<<<<<<< HEAD
# RAG_Memory: 增强的记忆检索系统

一个先进的即插即用对话记忆管理系统，集成了向量检索、语义压缩和智能上下文管理，专为AI对话代理设计。

## 主要功能

### 核心功能
1. **对话窗口管理**：维护最近 N 轮对话（可配置 `MAX_WINDOWS`）
2. **自动对话压缩**：窗口满时自动使用 LLM 压缩旧对话为摘要
3. **用户信息提取**：自动从用户输入中提取个人信息并存储到 `user.md`
4. **每日聊天日志**：每天自动生成独立的聊天日志文件（`YYYY-MM-DD.md`）
5. **系统运行日志**：运行时日志同时输出到控制台和文件（`memory_system/log/system.log`），便于调试和监控
6. **持久化存储**：压缩记忆和用户信息保存到本地文件，重启后可用

### 增强功能（向量检索模块）
7. **向量化记忆存储**：对话自动分块并存储到 Milvus 向量数据库
8. **混合检索系统**：结合向量相似度（0.7）和 BM25 关键词检索（0.3）
9. **智能重排序**：使用 BGE-Reranker-Large 模型提升检索结果质量
10. **优雅降级**：无向量数据库时自动回退到基础功能
11. **批量操作**：支持批量清空本地和向量数据库中的记忆

## 安装

### 1. 克隆或下载项目

```bash
git clone <repository-url>
cd memory_sys
```

### 2. 创建虚拟环境（推荐）

```bash
conda create -n AgentEnv python=3.9
conda activate AgentEnv
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

创建 `.env` 文件（或设置系统环境变量）：

```env
# 校内模型配置（示例）
MY_API_KEY=your-api-key
MY_API_BASE=https://api.chat.csu.edu.cn/v1
MY_MODEL=Qwen3-32B-FP8

# 或使用其他 OpenAI 兼容 API
# OPENAI_API_KEY=your-api-key
# OPENAI_API_BASE=https://api.example.com/v1
```

**安全提示**：请确保API密钥只存储在 `.env` 文件中，不要硬编码在代码中。系统已配置为从环境变量读取密钥，确保密钥安全。

## 快速开始

### 基本使用

```python
from memory_system import MemorySystem
from langchain_openai import ChatOpenAI

# 1. 创建 LLM 实例
llm = ChatOpenAI(
    model="Qwen3-32B-FP8",
    api_key="your-api-key",
    base_url="https://api.chat.csu.edu.cn/v1",
    temperature=0.0
)

# 2. 创建RAG_Memory系统实例
memory = MemorySystem(llm=llm, max_windows=10)

# 3. 添加对话
memory.add_conversation("user", "我叫张三，今年25岁，是一名软件工程师。")
memory.add_conversation("assistant", "你好张三！很高兴认识你。")

# 4. 获取最近对话
recent = memory.get_recent_conversations(3)

# 5. 获取 LLM 上下文
context = memory.get_context_for_llm()
print(context)
```

### 使用默认配置

```python
from memory_system import create_default_memory_system

# 自动使用环境变量中的配置
memory = create_default_memory_system()
```

### 增强RAG_Memory系统（向量检索）

```python
from memory_system import VectorMemorySystem, create_default_vector_memory_system

# 需要先启动 Milvus Docker 服务
# cd docker && docker-compose up -d

# 创建增强RAG_Memory系统（自动使用向量存储）
memory = create_default_vector_memory_system()

# 添加对话（自动分块和存储到向量数据库）
memory.add_conversation("user", "我叫张明，是一名软件工程师，擅长Python和机器学习。")
memory.add_conversation("assistant", "你好张明！机器学习是个很有前景的领域。")

# 检索相关记忆（混合检索 + rerank）
results = memory.retrieve_memories("联系方式", top_k=5)
print(f"检索到 {len(results)} 条相关记忆")

# 高级搜索：按对话分组
conversations = memory.search_similar_conversations("个人信息", max_results=3)

# 获取向量存储统计
stats = memory.get_vector_stats()
print(f"向量存储统计: {stats}")

# 清空向量存储（可选）
# memory.clear_vector_store()
```

**注意**：向量检索功能需要额外依赖：
```bash
pip install pymilvus
# 并启动 Milvus Docker 服务
```

## 高级功能

### 1. 对话压缩

当对话轮次达到 `max_windows` 限制时，系统会自动压缩旧对话：

```python
# 添加足够多的对话触发压缩
for i in range(15):
    memory.add_conversation("user", f"测试消息 {i}")
    memory.add_conversation("assistant", f"回复 {i}")

# 查看压缩记忆
compressed = memory.get_compressed_memories()
print(f"压缩记忆数量：{len(compressed)}")
```

### 2. 用户信息提取

系统会自动从用户输入中提取个人信息：

```python
memory.add_conversation("user", "我的邮箱是 example@test.com，电话是 13800138000")
# 信息会自动保存到 data/user.md
```

### 3. 每日日志

每天的对话会自动保存到独立的日志文件中：

```
data/logs/
├── 2025-02-23.md
├── 2025-02-24.md
└── 2025-02-25.md
```

### 4. 获取完整上下文

```python
# 获取包含压缩记忆和最近对话的完整上下文
context = memory.get_context_for_llm(max_recent=5)
print(context)
```

### 5. 向量检索功能

#### 5.1 混合检索系统
- **向量检索（70%）**：使用 BGE-M3 模型生成 embedding，Milvus 存储和检索
- **BM25 检索（30%）**：传统关键词检索，提供词汇匹配能力
- **Rerank 重排序**：使用 BGE-Reranker-Large 模型对 Top-K 结果精排
- **智能降级**：当 rerank 失败时自动使用 embedding 相似度计算

#### 5.2 检索流程
```
用户查询 → 混合检索 → 召回 Top-100 → Rerank 重排序 → 返回 Top-5
          (向量70% + BM25 30%)      (bge-reranker-large)
```

#### 5.3 示例代码
```python
from memory_system.vector import VectorRetrievalConfig

# 自定义配置
config = VectorRetrievalConfig(
    vector_weight=0.7,
    bm25_weight=0.3,
    top_k=100,               # 召回最多100条
    rerank_top_n=5,          # rerank后返回5条
    similarity_threshold=0.3,
    chunk_size=512,          # 分块大小
    chunk_overlap=50,        # 分块重叠
    embedding_model="bge-m3",
    rerank_model="bge-reranker-large"
)

# 使用自定义配置创建RAG_Memory系统
memory = VectorMemorySystem(vector_config=config)

# 降级模式（不使用向量存储）
memory_fallback = VectorMemorySystem(enable_vector_store=False)
```

#### 5.4 清空记忆工具
```bash
# 清空所有记忆（本地和向量数据库）
python clear_memories.py --all

# 只清空本地记忆
python clear_memories.py --local

# 只清空向量数据库记忆
python clear_memories.py --vector

# 静默模式
python clear_memories.py --all --quiet
```

#### 5.5 调试工具
```bash
# 测试 rerank API 调用
python test_rerank_api.py

# 诊断 API 端点支持
python debug_rerank.py
```

## 配置选项

### MemorySystem 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm` | `ChatOpenAI` | `None` | LangChain ChatOpenAI 实例，如果为 None 则使用默认配置 |
| `max_windows` | `int` | `10` | 对话窗口最大容量 |
| `data_dir` | `Path` | `./data` | 数据存储目录 |

### VectorMemorySystem 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm` | `ChatOpenAI` | `None` | 聊天 LLM 实例 |
| `max_windows` | `int` | `10` | 对话窗口最大容量 |
| `data_dir` | `Path` | `./data` | 数据存储目录 |
| `vector_config` | `VectorRetrievalConfig` | `None` | 向量检索配置 |
| `enable_vector_store` | `bool` | `True` | 是否启用向量存储 |

### VectorRetrievalConfig 配置项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vector_weight` | `float` | `0.7` | 向量检索权重 |
| `bm25_weight` | `float` | `0.3` | BM25 检索权重 |
| `top_k` | `int` | `100` | 召回最大条数 |
| `rerank_top_n` | `int` | `5` | 重排序后返回条数 |
| `similarity_threshold` | `float` | `0.3` | 相似度阈值 |
| `chunk_size` | `int` | `512` | 分块大小（tokens） |
| `chunk_overlap` | `int` | `50` | 分块重叠大小 |
| `embedding_model` | `str` | `"bge-m3"` | Embedding 模型名称 |
| `rerank_model` | `str` | `"bge-reranker-large"` | 重排序模型名称 |
| `api_key` | `str` | 环境变量 | API 密钥 |
| `api_base` | `str` | 环境变量 | API 基础 URL |
| `milvus_host` | `str` | `"localhost"` | Milvus 主机 |
| `milvus_port` | `int` | `19530` | Milvus 端口 |

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MY_API_KEY` | - | API 密钥 |
| `MY_API_BASE` | `https://api.chat.csu.edu.cn/v1` | API 基础 URL |
| `MY_MODEL` | `Qwen3-32B-FP8` | 聊天模型名称 |
| `MILVUS_HOST` | `"localhost"` | Milvus 主机地址 |
| `MILVUS_PORT` | `19530` | Milvus 端口号 |

## 项目目录结构

```
RAG_Memory/                    # 项目根目录
├── memory_system/            # 主包目录
│   ├── core/                 # 核心模块
│   ├── vector/               # 向量检索模块（可选）
│   └── __init__.py           # 包入口
├── docker/                   # Milvus Docker 配置
├── tests/                    # 单元测试
├── data/                     # 数据存储（自动创建）
├── log/                      # 系统日志存储（自动创建）
├── scripts/                  # 实用脚本
├── examples/                 # 示例代码
├── requirements.txt          # Python依赖包
└── README.md                 # 本文档
```

### 模块详解

#### 1. memory_system/ - 核心包
- **core/**: 基础RAG_Memory系统实现
  - `memory.py`: `MemorySystem`类，实现对话窗口管理、自动压缩、用户信息提取、日志记录
  - `config.py`: 基础配置（路径常量、默认参数）
  - `utils.py`: 通用工具函数
  - `__init__.py`: 核心模块导出
- **vector/**: 向量检索模块（可选扩展）
  - `memory_system.py`: `VectorMemorySystem`类，继承自`MemorySystem`，集成向量检索
  - `config.py`: 向量检索配置（权重、模型、阈值等）
  - `milvus_store.py`: Milvus向量数据库封装
  - `embedding_service.py`: BGE-M3 embedding服务
  - `retriever.py`: 混合检索器（向量70% + BM25 30%）
  - `reranker.py`: BGE-Reranker-Large重排序器
  - `chunking.py`: 对话分块器（512 token窗口）
  - `__init__.py`: 向量模块导出
- `__init__.py`: 智能导入，根据环境自动选择基础或增强系统

#### 2. docker/ - Milvus服务配置
- `docker-compose.yml`: Milvus单机版Docker配置（包含etcd、minio）
- `start.sh` / `stop.sh`: 服务启动/停止脚本
- `README.md`: Docker使用说明

#### 3. tests/ - 单元测试
- `test_memory.py`: 基础RAG_Memory系统功能测试
- `test_vector.py`: 向量检索功能测试

#### 4. data/ - 数据存储（自动创建）
- `logs/`: 每日聊天日志，按日期组织（YYYY-MM-DD.md）
- `user.md`: 提取的用户个人信息（带时间戳和原始输入）
- `compressed_memory.md`: 压缩后的记忆摘要

#### 5. log/ - 系统日志存储（自动创建）
- `system.log`: 系统运行时日志，包含INFO、WARNING、ERROR等级别日志记录
- 日志同时输出到控制台和文件，便于调试和监控

#### 6. scripts/ - 实用脚本
- `clear_memories.py`: 记忆清空工具（支持本地、向量、全部）
- `debug_rerank.py`: Rerank API调试工具
- `test_rerank_api.py`: Rerank API测试工具

#### 7. examples/ - 示例代码
- `basic_example.py`: 基础RAG_Memory系统使用示例
- `vector_example.py`: 向量检索系统使用示例
- `test_simple.py`: 简单测试脚本

#### 8. 根目录文件
- `requirements.txt`: Python依赖包列表
- `README.md`: 本文档
- `example_usage.py`: 综合使用示例
- `clear_memories.py`: 命令行工具（与scripts/下相同）
- 其他测试和调试脚本

## API 参考

### MemorySystem 类（基础RAG_Memory系统）

#### 主要方法

- `add_conversation(role: str, content: str)`：添加一轮对话
- `get_recent_conversations(n: int = None)`：获取最近对话
- `get_compressed_memories()`：获取所有压缩记忆
- `get_context_for_llm(max_recent: int = 5)`：生成 LLM 上下文
- `clear_conversation_window()`：清空对话窗口
- `reset_all()`：重置整个系统

#### 属性

- `conversation_window`：当前对话窗口（deque）
- `compressed_memories`：压缩记忆列表
- `max_windows`：窗口最大容量

### VectorMemorySystem 类（增强RAG_Memory系统）

**继承自** `MemorySystem`，在基础功能上增加向量检索能力。

#### 主要方法（新增/重写）

- `add_conversation(role: str, content: str)`：重写，自动分块并存储到向量数据库
- `retrieve_memories(query: str, top_k: int = None, use_reranker: bool = True, similarity_threshold: float = None)`：检索相关记忆（混合检索 + rerank）
- `search_similar_conversations(query: str, max_results: int = 10)`：搜索相似对话（按对话分组）
- `get_vector_stats()`：获取向量存储统计信息
- `clear_vector_store()`：清空向量数据库中的记忆
- `export_memories(output_file: Path = None, format: str = "json")`：导出记忆数据

#### Rerank 策略

1. **首选**：尝试使用 `bge-reranker-large` 模型进行精确重排序
2. **降级**：如果 rerank 模型不可用，使用 embedding 相似度计算
3. **回退**：如果所有方法都失败，返回原始排序结果

#### 检索流程

```python
# 1. 混合检索：0.7 * 向量相似度 + 0.3 * BM25 分数
# 2. 召回 Top-K（默认 100）个候选结果
# 3. Rerank 重排序（如果启用）
# 4. 返回 Top-N（默认 5）个最终结果
```

## 测试

### 运行单元测试

```bash
conda activate AgentEnv
pytest tests/ -v
```

### 运行示例

```bash
conda activate AgentEnv
python example_usage.py
```

### 模拟测试

测试使用模拟 LLM，无需实际 API 调用：

```python
from unittest.mock import Mock
mock_llm = Mock()
mock_llm.invoke.return_value.content = "测试摘要"

memory = MemorySystem(llm=mock_llm, max_windows=5)
```

## 设计原理

RAG_Memory 系统采用分层架构设计，融合了传统对话管理与现代向量检索技术，实现智能记忆管理。系统核心分为两大模块：**基础记忆系统**（MemorySystem）和**增强记忆系统**（VectorMemorySystem）。基础模块提供对话窗口管理、自动压缩、用户信息提取和日志记录；增强模块在此基础上集成向量检索、混合搜索和智能重排序，形成完整的检索增强生成（RAG）管道。

### 系统架构
```
┌─────────────────────────────────────────────────────────────┐
│                   应用层 (Application)                       │
│  • MemorySystem: 基础记忆管理                                │
│  • VectorMemorySystem: 增强记忆管理（向量检索）              │
├─────────────────────────────────────────────────────────────┤
│                   服务层 (Service)                           │
│  • HybridRetriever: 混合检索器（向量70% + BM25 30%）        │
│  • Reranker: BGE-Reranker-Large 重排序                      │
│  • EmbeddingService: BGE-M3 嵌入服务                        │
├─────────────────────────────────────────────────────────────┤
│                   存储层 (Storage)                           │
│  • Milvus: 向量数据库（对话分块存储）                       │
│  • 本地文件系统: 日志、用户信息、压缩记忆                    │
└─────────────────────────────────────────────────────────────┘
```

### 工作流程
1. **对话录入**: 用户与助手对话通过 `add_conversation()` 录入系统
2. **实时处理**: 自动提取用户信息、保存日志、检查窗口容量
3. **智能压缩**: 窗口满时触发LLM压缩，生成摘要保存到压缩记忆
4. **向量化存储**（增强模式）: 对话分块、生成嵌入、存储到Milvus
5. **检索增强**: 查询时通过混合检索 + 重排序召回相关记忆
6. **上下文构建**: 结合压缩记忆、最近对话和检索结果生成LLM上下文

### 对话窗口管理

- 使用 `collections.deque` 维护固定大小的对话窗口
- 当窗口满时，自动触发压缩流程
- 压缩后保留最新2轮对话以保持上下文连续性

### 对话压缩

1. 收集除最新2轮外的所有对话
2. 使用 LLM 生成简洁摘要
3. 摘要保存到压缩记忆列表和文件
4. 从窗口中移除已压缩的对话

### 用户信息提取

1. 对每个用户输入，使用 LLM 判断是否包含个人信息
2. 提取的信息包括：姓名、年龄、职业、联系方式、兴趣爱好等
3. 信息以带时间戳的格式追加到 `user.md`

### 日志系统

系统提供双层日志记录：

#### 1. 每日聊天日志
- 每个对话自动保存到当天日志文件
- 日志文件按日期组织：`YYYY-MM-DD.md`
- 存储位置：`data/logs/`
- 格式清晰，包含时间戳和角色标识

#### 2. 系统运行日志
- 运行时日志同时输出到控制台和文件
- 文件位置：`memory_system/log/system.log`
- 记录INFO、WARNING、ERROR等级别日志
- 便于调试、监控和故障排查
- 使用Python标准logging模块，支持自定义配置

### 向量检索架构

#### 模块分层
```
应用层：VectorMemorySystem（增强RAG_Memory系统）
    ↓
服务层：HybridRetriever（混合检索器）
    ├── 向量检索：MilvusStore + EmbeddingService
    ├── 关键词检索：BM25 索引
    └── 重排序：Reranker（bge-reranker-large）
    ↓
存储层：Milvus 向量数据库 + 本地文件系统
```

#### 对话处理流程
1. **分块**：使用 `ConversationChunker` 将对话切分为 512 token 的片段
2. **向量化**：使用 `EmbeddingService`（BGE-M3）生成 embedding
3. **存储**：通过 `MilvusStore` 存储向量和元数据
4. **索引**：同时构建 BM25 关键词索引

#### 检索优化策略
- **混合权重**：向量相似度（70%）+ BM25 分数（30%）
- **智能降级**：Rerank 失败 → Embedding 相似度 → 原始排序
- **批量处理**：支持批量 embedding 生成和相似度计算
- **错误恢复**：单点故障不影响整体系统可用性

#### 目录结构优化
- **核心模块**：`core/` 目录集中管理基础功能
- **向量模块**：`vector/` 目录作为可选扩展
- **清晰分离**：基础功能与增强功能解耦，便于维护

## 故障排除

### 常见问题

1. **LLM 初始化失败**
   - 检查环境变量是否正确设置
   - 验证 API 密钥和基础 URL
   - 确认网络连接

2. **文件权限错误**
   - 确保对 `data/` 目录有写入权限
   - 检查磁盘空间

3. **压缩失败**
   - 检查 LLM 是否可用
   - 确认对话数量足够（至少2轮）

### 日志查看

系统使用 Python 标准 logging 模块，提供两种日志查看方式：

#### 1. 控制台日志
系统默认将日志输出到控制台，可以通过以下代码调整日志级别：

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # 设置为DEBUG级别查看更详细日志
```

#### 2. 文件日志
系统运行日志自动保存到文件，便于后续分析和故障排查：
- 日志文件位置：`memory_system/log/system.log`
- 包含INFO、WARNING、ERROR等级别日志
- 格式：`%(asctime)s - %(name)s - %(levelname)s - %(message)s`

#### 3. 每日聊天日志
每天的对话记录保存到独立文件：
- 存储位置：`data/logs/YYYY-MM-DD.md`
- 包含完整对话历史，便于回顾

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

# RAG对话系统部署指南

## 系统架构

本系统是一个基于RAG_MEMORY的对话系统，包含以下组件：

1. **前端**：React + TypeScript + Ant Design 界面
2. **后端**：FastAPI + SQLite + RAG_MEMORY 集成
3. **向量数据库**：Milvus（用于记忆检索）
4. **数据库**：SQLite（对话元数据）

## 快速开始

### 使用Docker Compose（推荐）

1. 复制环境变量配置文件：
   ```bash
   cp .env.example .env
   ```
   编辑`.env`文件，设置您的API密钥和其他配置。

2. 启动所有服务：
   ```bash
   docker-compose up -d
   ```

3. 访问应用：
   - 前端界面：http://localhost:3000
   - 后端API：http://localhost:8000
   - API文档：http://localhost:8000/docs

4. 停止服务：
   ```bash
   docker-compose down
   ```

### 手动部署

#### 后端部署

1. 安装Python依赖：
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. 设置环境变量：
   ```bash
   export MY_API_KEY=your_api_key
   export MY_API_BASE=https://api.chat.csu.edu.cn/v1
   export MY_MODEL=Qwen3-32B-FP8
   ```

3. 启动后端服务：
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

#### 前端部署

1. 安装Node.js依赖：
   ```bash
   cd frontend
   npm install
   ```

2. 启动开发服务器：
   ```bash
   npm run dev
   ```

3. 构建生产版本：
   ```bash
   npm run build
   ```

## 功能特性

### 对话管理
- ✅ 新建对话
- ✅ 切换对话
- ✅ 删除对话
- ✅ 对话重命名

### RAG记忆功能
- ✅ 最近10条对话上下文管理
- ✅ 向量检索增强生成
- ✅ 记忆压缩和存储
- ✅ 相关记忆检索

### API端点

#### 对话管理
- `GET /conversations` - 获取对话列表
- `POST /conversations` - 创建新对话
- `GET /conversations/{id}` - 获取对话详情
- `DELETE /conversations/{id}` - 删除对话

#### 消息处理
- `POST /chat` - 发送消息（自动创建或使用现有对话）
- `GET /conversations/{id}/messages` - 获取对话消息

#### 系统管理
- `GET /system/status` - 系统状态检查
- `POST /system/cleanup` - 清理系统资源

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `MY_API_KEY` | LLM API密钥 | 无 |
| `MY_API_BASE` | LLM API基础URL | `https://api.chat.csu.edu.cn/v1` |
| `MY_MODEL` | 使用的模型 | `Qwen3-32B-FP8` |
| `MILVUS_HOST` | Milvus主机地址 | `localhost` |
| `MILVUS_PORT` | Milvus端口 | `19530` |
| `DATABASE_URL` | SQLite数据库路径 | `sqlite:///./data/conversations.db` |
| `VITE_API_BASE` | 前端API基础URL | `http://localhost:8000` |

### 数据存储

- **对话数据**：SQLite数据库（`data/conversations.db`）
- **向量存储**：Milvus向量数据库
- **记忆文件**：`data/conversations/{conversation_id}/` 目录

## 开发指南

### 项目结构

```
RAG_MEMORY/
├── backend/                 # 后端代码
│   ├── app/                # FastAPI应用
│   │   ├── api/            # API端点
│   │   ├── services/       # 业务逻辑
│   │   ├── models.py       # 数据库模型
│   │   └── main.py         # 应用入口
│   ├── requirements.txt    # Python依赖
│   └── Dockerfile          # Docker镜像
├── frontend/               # 前端代码
│   ├── src/                # 源代码
│   │   ├── components/     # React组件
│   │   ├── services/       # API服务
│   │   ├── store/          # 状态管理
│   │   └── types/          # TypeScript类型
│   ├── package.json        # Node.js依赖
│   └── Dockerfile          # Docker镜像
├── memory_system/          # 现有RAG_MEMORY系统
├── docker-compose.yml      # Docker编排配置
└── README.md               # 项目说明
```

### 添加新功能

1. **添加新的API端点**：
   - 在 `backend/app/api/` 创建新模块
   - 在 `backend/app/main.py` 中注册路由

2. **添加新的前端组件**：
   - 在 `frontend/src/components/` 创建React组件
   - 在 `frontend/src/pages/` 中集成到页面

3. **修改记忆系统**：
   - 参考 `memory_system/` 目录中的现有实现
   - 通过 `backend/app/adapters/memory_adapter.py` 集成

## 故障排除

### 常见问题

1. **后端启动失败**：
   - 检查Python依赖是否安装：`pip install -r requirements.txt`
   - 检查环境变量是否设置
   - 检查端口是否被占用

2. **前端无法连接后端**：
   - 检查 `VITE_API_BASE` 环境变量设置
   - 检查后端服务是否运行：`curl http://localhost:8000/health`
   - 检查CORS配置

3. **Milvus连接失败**：
   - 确保Milvus服务已启动：`docker ps | grep milvus`
   - 检查 `MILVUS_HOST` 和 `MILVUS_PORT` 设置

4. **RAG检索不工作**：
   - 检查向量数据库是否包含数据
   - 检查记忆系统配置
   - 查看后端日志获取错误信息

### 日志查看

```bash
# 查看Docker容器日志
docker-compose logs -f

# 查看后端日志
docker-compose logs -f backend

# 查看前端日志
docker-compose logs -f frontend

# 查看Milvus日志
docker-compose logs -f milvus
```

## 性能优化

1. **数据库优化**：
   - 定期清理旧对话
   - 使用数据库索引优化查询

2. **向量检索优化**：
   - 调整检索参数（top_k, similarity_threshold）
   - 使用缓存机制减少重复检索

3. **前端优化**：
   - 实现虚拟滚动处理长列表
   - 使用WebSocket实现实时消息

## 扩展建议

1. **用户认证**：添加JWT或OAuth认证
2. **文件上传**：支持文档上传和RAG处理
3. **多语言支持**：添加国际化支持
4. **插件系统**：支持功能扩展插件
5. **监控告警**：添加系统监控和告警

## 许可证

本项目基于现有RAG_MEMORY系统构建，遵循相关许可证。
# Milvus Docker 部署

使用 Docker Compose 部署 Milvus 向量数据库，为记忆系统提供向量存储服务。

## 快速启动

### 1. 启动 Milvus

```bash
cd docker
docker-compose up -d
```

### 2. 检查服务状态

```bash
docker-compose ps
```

### 3. 查看日志

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f standalone
```

### 4. 停止服务

```bash
docker-compose down
```

## 服务说明

### 端口映射
- **19530**: Milvus 服务端口（Python客户端连接）
- **9092**: Milvus 管理端口（映射到容器的9091）

### 数据持久化
数据默认保存在 `./volumes/` 目录：
- `./volumes/etcd`: etcd 数据
- `./volumes/minio`: MinIO 对象存储数据
- `./volumes/milvus`: Milvus 数据

### 环境变量
- `DOCKER_VOLUME_DIRECTORY`: 可指定自定义卷目录

## 连接配置

在记忆系统中配置 Milvus 连接：

```python
from memory_system.vector.config import VectorRetrievalConfig

config = VectorRetrievalConfig(
    milvus_host="localhost",  # Docker 服务地址
    milvus_port=19530,        # 映射端口
    milvus_collection_name="conversation_memories"
)
```

或者通过环境变量：

```bash
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
```

## 健康检查

### 1. 使用 curl 检查

```bash
curl http://localhost:9092/healthz
```

### 2. 使用 Milvus 客户端检查

```python
from pymilvus import connections, utility

# 连接
connections.connect(host='localhost', port='19530')

# 检查连接
print(utility.list_collections())
```

## 故障排除

### 常见问题

1. **端口冲突**
   - 检查 19530 和 9092 端口是否被占用
   - 修改 docker-compose.yml 中的端口映射

2. **容器启动失败**
   - 检查 Docker 和 Docker Compose 版本
   - 查看详细日志：`docker-compose logs`

3. **磁盘空间不足**
   - 清理旧的卷数据：`docker-compose down -v`
   - 指定其他卷目录

4. **连接超时**
   - 确保防火墙允许相关端口
   - 检查 Docker 网络配置

### 数据备份

备份卷数据：
```bash
# 备份整个 volumes 目录
tar -czf milvus_backup.tar.gz ./volumes/
```

### 数据恢复

恢复卷数据：
```bash
# 停止服务
docker-compose down

# 恢复备份
tar -xzf milvus_backup.tar.gz

# 重新启动
docker-compose up -d
```

## 升级 Milvus

1. 备份数据
2. 停止当前服务：`docker-compose down`
3. 更新 docker-compose.yml 中的镜像版本
4. 重新启动：`docker-compose up -d`

## 资源监控

### 查看容器资源使用
```bash
docker stats
```

### 查看日志大小
```bash
docker-compose logs --tail=100
```

## 安全建议

1. **生产环境**：修改默认的 MinIO 访问密钥
2. **网络隔离**：使用 Docker 网络隔离服务
3. **访问控制**：配置防火墙规则
4. **定期备份**：设置自动备份策略
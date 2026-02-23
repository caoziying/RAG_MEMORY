#!/bin/bash

# Milvus Docker 启动脚本

set -e

echo "=== Milvus Docker 启动脚本 ==="

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "错误: Docker Compose 未安装"
    exit 1
fi

# 创建数据目录
mkdir -p ./volumes/etcd
mkdir -p ./volumes/minio
mkdir -p ./volumes/milvus

echo "数据目录已创建"

# 启动服务
echo "正在启动 Milvus 服务..."
docker-compose up -d

# 等待服务启动
echo "等待服务启动..."
sleep 10

# 检查服务状态
echo "=== 服务状态 ==="
docker-compose ps

echo ""
echo "=== 连接信息 ==="
echo "Milvus 地址: localhost:19530"
echo "管理界面: localhost:9092"
echo ""
echo "使用以下命令查看日志:"
echo "  docker-compose logs -f"
echo ""
echo "使用以下命令停止服务:"
echo "  docker-compose down"
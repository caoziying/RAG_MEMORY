#!/bin/bash

# Milvus Docker 停止脚本

set -e

echo "=== Milvus Docker 停止脚本 ==="

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "错误: Docker Compose 未安装"
    exit 1
fi

# 停止服务
echo "正在停止 Milvus 服务..."
docker-compose down

echo "服务已停止"

# 可选：删除数据
if [[ "$1" == "--clean" ]]; then
    echo "正在删除数据卷..."
    docker-compose down -v
    rm -rf ./volumes
    echo "数据卷已删除"
fi
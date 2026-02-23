# Backend Dockerfile
FROM python:3.9-slim-bullseye

WORKDIR /app

# Install minimal system dependencies with retry mechanism
RUN apt-get update --fix-missing || apt-get update || true && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    || apt-get install -y libgomp1 \
    || echo "Warning: Failed to install libgomp1, continuing without it" && \
    rm -rf /var/lib/apt/lists/*
# Copy requirements and install Python dependencies with Tsinghua PyPI mirror
# COPY requirements.txt .
# # 增加这一行：打印出 requirements.txt 的内容，看看里面到底有没有东西！
# RUN cat requirements.txt

# # 然后再执行安装
# RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# 明确告诉它：去 backend 目录下拿那个依赖文件
COPY backend/requirements.txt ./
# (如果你的文件名字就叫 req，那就写 COPY backend/req ./)

RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# 下面的代码拷贝也要跟着改，明确只拷贝 backend 里的代码
COPY memory_system ./memory_system
COPY backend/ .


# Create data directory
RUN mkdir -p data

# Environment variables
ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite:///./data/conversations.db
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
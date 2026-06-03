FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖（仅需要 httpx 的网络基础库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建 outputs 目录
RUN mkdir -p outputs

# 暴露 Web 端口
EXPOSE 7860

# 默认启动 Web UI
CMD ["python", "webui.py"]

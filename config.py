"""商家智能 Agent 配置"""

import os

# Ollama 配置
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2:7b")
OLLAMA_FAST_MODEL = os.environ.get("OLLAMA_FAST_MODEL", "qwen2:1.5b")

# Agent 配置
AGENT_TIMEOUT = int(os.environ.get("AGENT_TIMEOUT", "300"))   # LLM 调用超时（秒）
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "2"))          # LLM 调用重试次数
REACT_MAX_ROUNDS = 8   # ReAct 循环最大工具调用轮次

# RAG 配置
RAG_TOP_K = 3          # 检索返回的最相关文档数

# ── 缓存配置 ──
CACHE_ENABLED = True
CACHE_TTL = 3600       # 缓存过期时间（秒）
CACHE_CAPACITY = 200   # 缓存最大条目数

# ── 限流配置 ──
RATE_LIMIT_ENABLED = True
RATE_LIMIT_MAX = 30     # 每 IP 每分钟最大请求数
RATE_LIMIT_WINDOW = 60  # 窗口大小（秒）

# ── 日志配置 ──
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

# Web 配置
WEB_PORT = int(os.environ.get("WEB_PORT", "7860"))
WEB_HOST = os.environ.get("WEB_HOST", "127.0.0.1")

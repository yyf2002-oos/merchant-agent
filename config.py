"""商家智能 Agent 配置"""

import os

# ── 从 .env 文件加载环境变量（如果存在） ──
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── LLM 提供者选择 ──
# "ollama" 或 "deepseek"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek")

# Ollama 配置
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2:7b")
OLLAMA_FAST_MODEL = os.environ.get("OLLAMA_FAST_MODEL", "qwen2:1.5b")

# DeepSeek 配置（支持 OpenAI 兼容格式）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# Agent 配置
AGENT_TIMEOUT = int(os.environ.get("AGENT_TIMEOUT", "300"))   # LLM 调用超时（秒）
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "2"))          # LLM 调用重试次数
REACT_MAX_ROUNDS = 8   # ReAct 循环最大工具调用轮次

# ── 混合模型路由 ──
# 模型名格式: "provider:model_name"
#   deepseek:xxx   → 走 DeepSeek API  （复杂推理）
#   ollama:xxx     → 走本地 Ollama    （简单任务）
# 轻量模型用于 Plan 生成 + Reflect 反思等辅助环节
AGENT_MODEL = {
    "selector": "deepseek:deepseek-chat",   # 选品：多步工具调用 + 市场分析 → DeepSeek
    "analyst":  "deepseek:deepseek-chat",   # 运营分析：数据建模 + 策略 → DeepSeek
    "sourcing": "deepseek:deepseek-chat",   # 货源：供应链推理 + 多工具 → DeepSeek
    "lister":   "ollama:qwen2:7b",          # 上架：模板化文本生成 → 本地
    "service":  "ollama:qwen2:7b",          # 客服：FAQ 检索 + 话术 → 本地
}
AGENT_LIGHT_MODEL = {
    "selector": "ollama:qwen2:1.5b",
    "analyst":  "ollama:qwen2:1.5b",
    "sourcing": "ollama:qwen2:1.5b",
    "lister":   "ollama:qwen2:1.5b",
    "service":  "ollama:qwen2:1.5b",
}

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

# 🛒 商家智能运营 Agent

> AI 驱动的电商全流程自动化系统 — 选品 → 上架 → 客服 → 运营分析 → 货源筛选

基于 **DeepSeek V4** + **Ollama** 混合模型路由，自驱型 ReAct Agent 框架，覆盖电商运营核心环节。

---

## 功能特性

| Agent | 功能 | 模型路由 |
|-------|------|----------|
| 📋 **选品分析** | 淘宝搜索下拉词获取 + 价格库查询 + 关键词评分 → 蓝海商品推荐 | DeepSeek |
| 📝 **上架素材** | 标题生成(3平台) + 详情描述 + SEO优化 + 主图建议 | Ollama |
| 💬 **智能客服** | FAQ 知识库检索 + 历史笔记召回 + 情绪感知应答 | Ollama |
| 📊 **运营分析** | 利润计算 + 定价建议 + 促销方案 + 四象限商品诊断 | DeepSeek |
| 🏭 **货源筛选** | 产业带分析 + 供应商查询 + 成本利润测算 + 1688 链接生成 | DeepSeek |

- **一键完整工作流**：输入品类，全流程自动执行
- **智能对话路由**：自动理解用户意图，路由到对应 Agent
- **RAG 知识库**：FAQ + 价格库 + 供应商库 + 产区数据
- **双模式界面**：Gradio Web UI + Rich CLI
- **持久化记忆**：SQLite 对话历史 + Agent 笔记

---

## 架构

```
┌──────────────────────────────────────────┐
│           Gradio Web UI / CLI            │
├──────────────────────────────────────────┤
│            MerchantOrchestrator          │
├──────┬──────┬──────┬──────┬──────┬──────┤
│选品  │上架  │客服  │运营  │货源  │智能  │
│Agent │Agent │Agent │Agent │Agent │对话  │
├──────┴──────┴──────┴──────┴──────┴──────┤
│           ReAct 循环框架                  │
│    Plan → Execute → Reflect              │
├──────────────────────────────────────────┤
│   Tool Registry (14 个注册工具)           │
├──────────────────────────────────────────┤
│  ▸ DeepSeek API (复杂推理)               │
│  ▸ Ollama (简单任务 + Embedding)          │
│  ▸ RAG 知识库 (bge-m3 语义检索)           │
│  ▸ SQLite 持久化                         │
└──────────────────────────────────────────┘
```

### 上下文记忆系统（v2 新增）

```
┌──────────────────────────┐
│     Web UI / CLI         │  ← 每个会话独立 session_id
├──────────────────────────┤
│  MerchantOrchestrator    │
│  ├─ smart_chat(stateful) │  ← 加载历史上下文再路由
│  └─ SharedContext        │  ← 跨 Agent 结构化数据传递
├──────────────────────────┤
│  AgentContext             │
│  ├─ 滑动窗口(最近12条)    │  ← 保留原始消息
│  ├─ 结构化摘要(200字)     │  ← 压缩历史，字段化输出
│  ├─ 关键信息提取(100字)   │  ← 品类/预算/人群/价格
│  └─ 工具结果持久化        │  ← 工具调用结果回流记忆
├──────────────────────────┤
│  ConversationMemory       │  ← SQLite 持久化
│  ├─ conversations 表      │
│  ├─ knowledge 键值对      │
│  └─ agent_notes 笔记      │
└──────────────────────────┘
```

核心改进：
- **多轮对话记忆**：第二轮问"那个成本多少"时，Agent 知道上文
- **结构化摘要**：不是简单压缩，而是按"用户目标/已做决策/关键信息/待办"字段化输出
- **跨 Agent 上下文**：一键工作流中，`SharedContext` 数据类把选品结论结构化传给后续 Agent
- **工具结果持久化**：ReAct 循环中每个工具的结果都回写到记忆库

---

## 快速开始

### 前置依赖

- Python 3.10+
- [Ollama](https://ollama.com/)（本地模型）
- DeepSeek API Key（可选，用于复杂推理）

### 安装

```bash
git clone https://github.com/yyf2002-oos/merchant-agent.git
cd merchant-agent

python -m venv .venv
# Linux/Mac: source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 配置

复制环境变量模板并填写：

```bash
cp .env.example .env
```

```ini
# .env
LLM_PROVIDER=deepseek          # ollama | deepseek
DEEPSEEK_API_KEY=sk-xxxxxxx    # 你的 DeepSeek API Key
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
OLLAMA_BASE=http://localhost:11434
OLLAMA_MODEL=qwen2:7b
```

拉取模型：

```bash
ollama pull qwen2:7b
ollama pull qwen2:1.5b
ollama pull bge-m3
```

### 运行

**Web 界面：**
```bash
python webui.py
# 访问 http://localhost:7860
```

**CLI 模式：**
```bash
python main.py
```

---

## 配置说明

### 完整配置项

所有配置通过 `.env` 文件或环境变量设置。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_PROVIDER` | `deepseek` | LLM 提供者: `ollama` 或 `deepseek` |
| `DEEPSEEK_API_KEY` | `""` | DeepSeek API Key |
| `DEEPSEEK_API_BASE` | `https://api.deepseek.com` | DeepSeek API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek 模型名 |
| `OLLAMA_BASE` | `http://localhost:11434` | Ollama 服务地址 |
| `OLLAMA_MODEL` | `qwen2:7b` | Ollama 主模型 |
| `OLLAMA_FAST_MODEL` | `qwen2:1.5b` | Ollama 轻量模型（用于计划/反思） |
| `AGENT_TIMEOUT` | `300` | LLM 调用超时（秒） |
| `MAX_RETRIES` | `2` | LLM 调用重试次数 |
| `REACT_MAX_ROUNDS` | `8` | ReAct 循环最大工具调用轮次 |
| `CACHE_ENABLED` | `True` | 是否启用缓存 |
| `CACHE_TTL` | `3600` | 缓存过期时间（秒） |
| `CACHE_CAPACITY` | `200` | 缓存最大条目数 |
| `RATE_LIMIT_ENABLED` | `True` | 是否启用限流 |
| `RATE_LIMIT_MAX` | `30` | 每 IP 每分钟最大请求数 |
| `RATE_LIMIT_WINDOW` | `60` | 限流窗口（秒） |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `WEB_PORT` | `7860` | Web 端口 |
| `WEB_HOST` | `127.0.0.1` | 监听地址 |

### Agent 模型路由

每个 Agent 可独立配置主模型和轻量模型（用于计划生成和反思）：

```python
AGENT_MODEL = {
    "selector": "deepseek:deepseek-chat",   # 选品 → DeepSeek
    "analyst":  "deepseek:deepseek-chat",   # 运营分析 → DeepSeek
    "sourcing": "deepseek:deepseek-chat",   # 货源 → DeepSeek
    "lister":   "ollama:qwen2:7b",          # 上架 → 本地
    "service":  "ollama:qwen2:7b",          # 客服 → 本地
}
```

### Web UI Tab

| Tab | 功能 |
|-----|------|
| 💬 智能对话 | 自动路由 + 多轮上下文记忆 |
| 📋 选品分析 | 输入品类 → 淘宝数据 + AI 分析 |
| 📝 上架素材 | 输入商品信息 → 生成上架素材包 |
| 📦 批量上架 | 上传 CSV → 批量生成 |
| 🏭 一手货源 | 产业带分析 + 工厂直供方案 |
| 💬 客服应答 | FAQ 检索 + AI 回复 |
| 📊 运营分析 | 利润计算 + 定价建议 |
| 🚀 一键工作流 | 选品→上架→客服→分析 全自动（带跨 Agent 上下文传递） |

---

## 并发能力

### 瓶颈分析

系统采用 **单进程同步架构**，并发能力取决于 LLM 提供者的限制。

### DeepSeek API 模式（推荐线上用）

| 场景 | 预估并发 | 说明 |
|------|----------|------|
| 智能对话 | **10-20 人同时** | 每次请求 3-8 秒，DeepSeek 无硬性并发限制 |
| 一键工作流 | **2-3 人同时** | 全程 2-4 分钟，占用多个 Agent，建议排队执行 |
| 上架素材 | **20-30 人同时** | 纯 Ollama 本地生成，主要看本地 GPU 显存 |
| 批量上架 | **5-10 人同时** | 逐条生成，CPU 密集型 |

Gradio 内部启用 `app.queue()`，自带请求排队机制，不会丢请求。

### Ollama 纯本地模式

| 硬件 | 推荐最大并发 | 说明 |
|------|-------------|------|
| 16GB 显存 (RTX 4060) | **2-3 人** | qwen2:7b 单次推理 5-15 秒 |
| 24GB 显存 (RTX 4090) | **4-6 人** | 可同时加载 7B + 1.5B 模型 |
| CPU only | **1 人** | 7B 模型推理 30-60 秒，建议把上架/客服也切到 DeepSeek |
| 无 GPU + Ollama | **不推荐多人** | 建议设置 `LLM_PROVIDER=deepseek` 全部走云端 |

### 瓶颈点

- **Ollama 模型推理**：单 GPU 串行推理，多人并发时排队。把 `lister` 和 `service` 的路由也切到 DeepSeek 可以缓解。
- **SQLite 写锁**：所有对话历史写入同一个 db 文件，高并发写入有锁竞争。当前场景下不会成为瓶颈（日均 < 10 万条消息）。
- **RAG 检索**：bge-m3 embedding 在 CPU 上单次约 200ms，无显著瓶颈。

### 优化建议

- **改 `WEB_HOST=0.0.0.0`** 允许局域网/公网访问
- **调小 `RATE_LIMIT_MAX`** 防止单用户刷爆（当前默认 30 次/分钟）
- **部署用 `nohup` 或 systemd** 保持后台运行
- **如需 50+ 并发**：前端加 Nginx 负载均衡 → 多进程部署 Web UI

---

## 数据来源说明

| 数据 | 来源 | 是否实时 |
|------|------|---------|
| 淘宝搜索下拉词 | `suggest.taobao.com` 公开接口 | ✅ 实时 |
| 商品价格库 | 本地 `knowledge/data/price_library.json` | ❌ 静态 |
| 供应商库 | 本地 `knowledge/data/supplier_library.json` | ❌ 静态 |
| 产区优势 | 本地 `knowledge/data/region_advantages.json` | ❌ 静态 |
| FAQ 知识库 | 本地 `knowledge/data/faq.json` | ❌ 静态 |
| 上架模板 | 本地 `knowledge/data/product_templates.json` | ❌ 静态 |

> ⚠️ 价格、供应商、产区数据均来自本地 JSON 文件，不是实时从淘宝/1688 API 拉取的。
> 如需实时数据，需要接入淘宝开放平台或第三方电商数据服务。

---

## 技术栈

- **LLM**: DeepSeek V4 / Ollama (qwen2)
- **Embedding**: bge-m3 (本地)
- **框架**: ReAct (Plan-Execute-Reflect)
- **记忆系统**: SQLite + 滑动窗口 + 结构化摘要
- **RAG**: 语义检索 + 关键词降级
- **缓存**: LRU + TTL 内存缓存
- **限流**: IP 滑动窗口
- **界面**: Gradio / Rich
- **测试**: Pytest (30 tests)

---

## 项目结构

```
merchant-agent/
├── agents/          # Agent 实现（5 个专业 Agent）
│   ├── selector.py  # 选品
│   ├── lister.py    # 上架
│   ├── service.py   # 客服
│   ├── analyst.py   # 运营分析
│   └── sourcing.py  # 货源筛选
├── core/            # 核心框架
│   ├── agent.py     # ReAct Agent 基类（含上下文管理）
│   ├── context.py   # 对话上下文管理（摘要/滑动窗口/关键信息）
│   ├── tool.py      # 工具注册系统
│   ├── tools_registry.py # 14 个注册工具
│   └── memory.py    # SQLite 持久化记忆
├── knowledge/       # 知识库
│   ├── rag.py       # 语义检索 + 关键词降级
│   ├── embedding.py # bge-m3 嵌入
│   └── data/        # JSON 数据文件
├── tools/           # 工具函数
│   ├── taobao.py    # 淘宝搜索下拉词
│   ├── calculator.py# 利润/定价计算
│   ├── product_manager.py # 商品管理
│   └── formatter.py # 格式化输出
├── config.py        # 配置
├── llm.py           # LLM 调用封装
├── orchestrator.py  # 主控编排（含 SharedContext 数据类）
├── webui.py         # Gradio 界面
├── main.py          # CLI 入口
└── cache.py         # LRU+TTL 缓存
```

# 🛒 商家智能运营 Agent

> AI 驱动的电商全流程自动化系统 — 选品 → 上架 → 客服 → 运营分析 → 货源筛选

基于 **DeepSeek V4** + **Ollama** 混合模型路由，自驱型 ReAct Agent 框架，覆盖电商运营核心环节。

## ✨ 功能特性

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

## 🧠 架构

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

## 🚀 快速开始

### 前置依赖

- Python 3.10+
- [Ollama](https://ollama.com/)（本地模型）
- DeepSeek API Key（可选，用于复杂推理）

### 安装

```bash
# 克隆仓库
git clone https://github.com/yyf2002-oos/merchant-agent.git
cd merchant-agent

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
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
```

如果只用 Ollama 本地模型：

```ini
LLM_PROVIDER=ollama
```

然后拉取模型：

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

## 🖥️ Web UI Tab 说明

| Tab | 说明 |
|-----|------|
| 💬 智能对话 | 自动路由到对应 Agent |
| 📋 选品分析 | 输入品类 → 淘宝数据 + AI 分析 |
| 📝 上架素材 | 输入商品信息 → 生成上架素材包 |
| 📦 批量上架 | 上传 CSV → 批量生成 |
| 🏭 一手货源 | 产业带分析 + 工厂直供方案 |
| 💬 客服应答 | FAQ 检索 + AI 回复 |
| 📊 运营分析 | 利润计算 + 定价建议 |
| 🚀 一键工作流 | 选品→上架→客服→分析 全自动 |

## 🔧 技术栈

- **LLM**: DeepSeek V4 / Ollama (qwen2)
- **Embedding**: bge-m3 (本地)
- **框架**: ReAct (Plan-Execute-Reflect)
- **RAG**: 语义检索 + 关键词降级
- **缓存**: LRU + TTL 内存缓存
- **限流**: IP 滑动窗口
- **存储**: SQLite
- **界面**: Gradio / Rich
- **测试**: Pytest

## 📁 项目结构

```
merchant-agent/
├── agents/          # Agent 实现（5 个专业 Agent）
│   ├── selector.py  # 选品
│   ├── lister.py    # 上架
│   ├── service.py   # 客服
│   ├── analyst.py   # 运营分析
│   └── sourcing.py  # 货源筛选
├── core/            # 核心框架
│   ├── agent.py     # ReAct Agent 基类
│   ├── tool.py      # 工具注册系统
│   ├── context.py   # 对话上下文管理
│   └── memory.py    # SQLite 持久化记忆
├── knowledge/       # 知识库
│   ├── rag.py       # 语义检索
│   ├── embedding.py # bge-m3 嵌入
│   └── data/        # JSON 数据文件
├── tools/           # 工具函数
│   ├── taobao.py    # 淘宝数据
│   ├── calculator.py# 利润/定价计算
│   ├── product_manager.py # 商品管理
│   └── formatter.py # 格式化输出
├── config.py        # 配置
├── llm.py           # LLM 调用封装
├── orchestrator.py  # 主控编排
├── webui.py         # Gradio 界面
├── main.py          # CLI 入口
└── cache.py         # LRU+TTL 缓存
```

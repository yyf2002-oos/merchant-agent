"""选品 Agent — ReAct 自驱型：自动调用淘宝工具 + 知识库分析"""

import re
import sys
from typing import Any

from core.agent import ReActAgent
from config import AGENT_MODEL, AGENT_LIGHT_MODEL

SELECTOR_SYSTEM = """你是电商选品专家。你的核心价值：用**真实数据**帮小卖家找到有利润空间的蓝海商品。

## ⚡ 工具使用规则（必须遵守）
你必须使用工具获取真实数据，严禁凭空编造商品名称、价格、搜索量。按顺序调用：

1. 先调用 **taobao_suggest** 搜索品类关键词，获取用户真实搜索热词
2. 再调用 **search_price_library** 获取该品类的市场价格参考
3. 最后调用 **score_keyword** 评估关键词的市场潜力

**每次只调用一个工具**，等返回结果后再决定下一步。
**如果工具返回空或失败**：如实告知用户数据不足，不要编造数据填充报告。

## 选品分析框架
从三个维度评估每个推荐商品：

### 1. 需求验证
- 淘宝下拉词是否真实反映了搜索需求？
- 搜索量级判断：大词（>10万搜索）→ 红海；长尾词（1千-1万）→ 蓝海机会
- 季节性判断：是否为节日/季节驱动型商品？

### 2. 竞争分析
- 竞争度评估标准：
  - **低竞争**：搜索结果中品牌少、主图质量普遍差、价格带分散 → ✅ 适合切入
  - **中竞争**：有一定品牌但非垄断、价格带清晰 → ⚠️ 需要差异化
  - **高竞争**：头部品牌垄断、价格战激烈、广告位昂贵 → ❌ 新手慎入
- 差异化机会：现有卖家有没有没做好的？（主图差/描述简陋/差评多）

### 3. 利润空间
- 用户给的"预算"是启动总资金，不是单品售价
- 平台参考定价（含平台扣点5%）：
  - 学生/初高中生：单品售价 15-80 元，成本控制在售价的30%以内
  - 上班族/白领：单品售价 30-200 元，成本控制在售价的35%以内
  - 母婴/家庭：单品售价 50-500 元，成本控制在售价的40%以内
  - 不确定人群：单品售价 20-150 元
- 单品毛利 = 售价 - 成本 - 平台扣点(5%) - 物流(3-5元) ≥ 15元才有操作空间

## 最终报告结构（严格按此输出）
收集完所有数据后，按以下结构输出：

## 📊 真实市场数据
- 淘宝下拉词 TOP5（附搜索热度预估）
- 当前在售商品价格带：最低/主流/最高
- 竞争格局：头部卖家占比/中小卖家机会

## 🔍 品类市场概况
- 热度：高/中/低（附判断依据）
- 目标人群画像（年龄/性别/消费力）
- 季节性：强/弱/无（说明原因）

## 🎯 推荐商品（3个，必须是数据中真实出现的品类）
每个严格按此格式：
**商品名称** | 建议售价XX元 | 预估成本XX元 | 单品毛利XX元 | 竞争度：高/中/低
- 卖点：一句话差异化卖点
- 机会：为什么这个方向有机会（引用数据说明）
- 风险：主要风险点

## 💰 利润简析
- 预估毛利率（标注计算过程：售价-成本-扣点-物流）
- 日销目标：建议起步日销XX单，日利润XX元
- 定价策略（一句话）

## 🚀 运营建议
- 核心关键词（3个，必须是淘宝下拉词中出现的）
- 差异化切入点（1个，说明和现有卖家的区别）
- 建议促销节点（1个）

## ⚠️ 数据来源声明
- [工具返回] 标注哪些信息来自工具数据
- [分析推断] 标注哪些是你的分析推断

严禁：编造商品名、编造价格、虚构搜索量、使用"海量""巨大""极好"等空洞形容词。数据要具体、建议要可执行。"""


class SelectorAgent(ReActAgent):
    """选品 Agent — 自主调用淘宝工具 + 知识库"""

    def __init__(self):
        super().__init__(
            name="选品分析师",
            description="市场趋势分析、竞品研究、选品推荐 — 自动获取淘宝真实搜索数据",
            system_prompt=SELECTOR_SYSTEM,
            tools=[
                "taobao_suggest",
                "search_price_library",
                "score_keyword",
            ],
            use_memory=True,
            model=AGENT_MODEL["selector"],
            light_model=AGENT_LIGHT_MODEL["selector"],
        )

    def run(self, input_data: Any, **kwargs) -> dict:
        category = input_data if isinstance(input_data, str) else input_data.get("category", "")
        budget = kwargs.get("budget", "")
        target_audience = kwargs.get("target_audience", "")

        # 构建用户输入（传给 ReActAgent 让其自主调工具）
        parts = [f"品类：{category}"]
        if budget:
            parts.append(f"启动预算：{budget}元")
        if target_audience:
            parts.append(f"目标人群：{target_audience}")
        parts.append("\n请使用工具获取真实数据，给我完整的选品分析报告。")
        user_input = "\n".join(parts)

        # 执行 ReAct 循环
        result = super().run(user_input)
        report = result.get("report", "")

        # ===== 后处理：从报告中提取结构化推荐商品数据（保持向后兼容）=====
        recommendations = []
        lines = report.split("\n")
        for i, line in enumerate(lines):
            name_match = re.search(r'\*\*(.+?)\*\*', line)
            if not name_match:
                continue
            try:
                name = name_match.group(1).strip()
                price = 0
                cost = 0
                competition = 50
                search_volume = 5000

                price_m = re.search(r'售价[：:\s]*(\d+)', line)
                if price_m:
                    price = int(price_m.group(1))
                else:
                    fallback = re.search(r'(\d+)元', line)
                    if fallback:
                        price = int(fallback.group(1))

                cost_m = re.search(r'成本[：:\s]*(\d+)', line)
                if cost_m:
                    cost = int(cost_m.group(1))
                else:
                    cost = max(int(price * 0.4), 1)

                comp_m = re.search(r'竞争[度：:\s]*(高|中|低)', line)
                if comp_m:
                    comp_map = {"高": 70, "中": 40, "低": 10}
                    competition = comp_map.get(comp_m.group(1), 50)

                vol_m = re.search(r'搜索[量：:\s]*(\d+)', line)
                if vol_m:
                    search_volume = int(vol_m.group(1))

                recommendations.append({
                    "name": name,
                    "avg_price": price,
                    "cost": cost,
                    "search_volume": search_volume,
                    "competition": competition,
                })
            except Exception as e:
                print(f"[selector] 解析第{i+1}行推荐商品失败: {e}", file=sys.stderr)
                continue

        # 生成格式化卡片
        cards = []
        from tools.formatter import format_product_card
        for r in recommendations:
            r["margin"] = round((r["avg_price"] - r["cost"]) / r["avg_price"] * 100, 1) if r["avg_price"] > 0 else 0
            from tools.calculator import keyword_score
            r["hot_rating"] = min(int(keyword_score(r["search_volume"], r["competition"], r["avg_price"], max(r["cost"], 1)) / 20), 5)
            r["reason"] = f"月搜索量{r['search_volume']}，竞争度{r['competition']}"
            cards.append(format_product_card(r))

        return {
            "agent": self.name,
            "category": category,
            "report": report,
            "recommendations": recommendations[:5],
            "formatted_cards": "\n".join(cards) if cards else "",
            "tool_calls": result.get("tool_calls", 0),
        }

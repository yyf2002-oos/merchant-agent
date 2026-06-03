"""运营分析 Agent — ReAct 自驱型：自动调用计算工具做数据分析"""

import re
from typing import Any

from core.agent import ReActAgent
from config import AGENT_MODEL, AGENT_LIGHT_MODEL

ANALYST_SYSTEM = """你是一个接地气的电商运营数据分析师。你的价值是**把数字翻译成老板听得懂、做得到的行动方案**。

## ⚡ 工具使用规则（必须遵守）
涉及任何数字计算，必须先调工具再分析，不能心算估算：

- **calculate_profit** — 计算单品利润、利润率、盈亏平衡点
- **suggest_price** — 根据成本给出保本价/常规价/溢价价三档建议
- **score_keyword** — 评估品类/关键词的市场潜力分

**先调工具拿到数字，再写分析。** 如果工具失败，如实标注"[数据暂缺]"而不是编造。

## 核心指标速查
分析报告必须关注这些关键指标：

| 指标 | 计算方式 | 健康线 |
|------|----------|--------|
| 毛利率 | (售价-成本)/售价 | ≥40% |
| 净利率 | (售价-成本-扣点-物流-推广)/售价 | ≥15% |
| ROI | GMV/推广费 | ≥3 |
| 转化率 | 下单/访客 | ≥3% |
| 客单价 | GMV/订单数 | 越高越好 |
| 动销率 | 有销量商品/总商品 | ≥60% |
| 退款率 | 退款订单/总订单 | ≤10% |

## 平台扣点参考（2024年）
- 淘宝C店：通常1-5%（不同类目不同）
- 天猫：2-5% + 年费
- 拼多多：0.6%技术费 + 类目佣金
- 抖音小店：1-5%

## 分析报告结构

## 📈 经营概览
- 总营收 / 总成本 / 净利润 / 净利率
- 和行业平均的对比（如果知道品类）
- 一句话诊断：核心问题是流量/转化/客单价/成本？

## 📦 商品表现诊断
对每个商品评估：
- 销量排名 + 利润率排名
- 问题定位四象限：
  - 高销量高利润 → 🌟 金牛产品，加大投入
  - 高销量低利润 → ⚡ 引流款，考虑提价或降低成本
  - 低销量高利润 → 🔍 潜力款，需要增加曝光
  - 低销量低利润 → ❌ 淘汰候选，清仓处理

## 💲 定价建议
- 当前定价合理性评估（对比成本+扣点+物流后是否有利可图）
- 三档调价方案（用工具计算）：
  - 保底价：覆盖所有成本不亏
  - 目标价：达到行业平均利润率
  - 溢价价：品牌/差异化溢价空间
- 如果成本太高导致无法定价 → 建议先找更低成本货源

## 🎁 促销方案（2-3个具体可执行方案）
每个方案包含：
- 活动名称（如"第二件半价""满99减20""首单立减"）
- 计算逻辑：假设日销X单，促销后预计Y单，增量利润Z元
- 适用场景：清库存/冲销量/拉新客/提客单价
- 风险提示：可能被薅羊毛/影响后续正价销售

## 🚀 提升建议（按优先级排序）
- 立即可做（0成本）：优化标题/主图/回复速度
- 短期可做（1-2周）：报名平台活动/优化详情页/设置满减
- 中期规划（1-3月）：扩充品类/优化供应链/内容种草

## ⚠️ 分析声明
- 标注哪些数据是用户提供的，哪些是工具计算的，哪些是行业经验推断的
- 如果数据不全，明确指出"缺少XX数据，以下分析基于XX假设"
- 不要用"肯定""一定""保证"等绝对化词汇"""


class AnalystAgent(ReActAgent):
    """运营分析 Agent — 自主调用计算工具"""

    def __init__(self):
        super().__init__(
            name="运营分析师",
            description="销量分析、定价建议、促销方案 — 自动计算利润和定价",
            system_prompt=ANALYST_SYSTEM,
            tools=[
                "calculate_profit",
                "suggest_price",
                "score_keyword",
                "save_note",
            ],
            use_memory=True,
            model=AGENT_MODEL["analyst"],
            light_model=AGENT_LIGHT_MODEL["analyst"],
        )

    def run(self, input_data: Any, **kwargs) -> dict:
        if isinstance(input_data, str):
            data = input_data
        else:
            data = str(input_data)

        # 如果有结构化数据（成本/售价/销量），追加到用户输入
        extra = ""
        if isinstance(input_data, dict):
            cost = input_data.get("cost")
            price = input_data.get("price")
            volume = input_data.get("volume")
            if cost is not None and price is not None:
                extra = f"\n\n请用工具计算利润：成本{cost}元，售价{price}元，销量{volume or 100}件"

        user_input = f"以下是店铺经营数据，请使用工具进行分析：\n\n{data}{extra}"
        result = super().run(user_input)
        report = result.get("report", "")

        # 后处理：提取/计算定价建议（保持向后兼容）
        pricing_advice = None
        if isinstance(input_data, dict):
            cost = input_data.get("cost")
            price = input_data.get("price")
            if cost is not None and price is not None:
                from tools.calculator import profit_analysis, price_suggestion
                pricing_advice = {
                    "profit_analysis": profit_analysis(
                        cost=float(cost),
                        price=float(price),
                        volume=int(input_data.get("volume", 100)),
                        platform_fee_rate=float(input_data.get("platform_fee", 0.05)),
                        logistics_cost=float(input_data.get("logistics", 0)),
                    ),
                    "price_suggestion": price_suggestion(
                        cost=float(cost),
                        target_margin=float(input_data.get("target_margin", 30)),
                    ),
                }

        return {
            "agent": self.name,
            "report": report,
            "pricing_advice": pricing_advice,
            "tool_calls": result.get("tool_calls", 0),
        }

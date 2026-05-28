"""运营分析 Agent — ReAct 自驱型：自动调用计算工具做数据分析"""

import re
from typing import Any

from core.agent import ReActAgent

ANALYST_SYSTEM = """你是一个专业的电商运营数据分析师。你的核心价值：**用真实数据给出可执行的运营建议**。

## ⚡ 工具使用规则（必须遵守）
你必须使用工具来计算数据，不能只凭知识估计。按需求调用以下工具：

- **calculate_profit** — 计算利润、利润率（需要成本、售价、销量）
- **suggest_price** — 根据成本建议定价（保本价/常规价/溢价价）
- **score_keyword** — 评估品类市场潜力

**对于任何涉及数字计算的，先调工具再写报告。**

## 最终报告结构
根据提供的经营数据，给出专业的分析报告：

## 经营概览
- 总营收、总成本、净利润、利润率
- 核心指标分析

## 商品表现分析
- 各商品销量排名
- 利润率排名
- 问题商品诊断（高曝光低转化等）

## 定价建议
- 当前定价合理性评估
- 调价建议（提价/降价/保持不变）
- 套餐组合建议

## 促销方案（2-3个可选方案）
- 方案名称、具体玩法、预期效果、成本预估、风险提示

## 运营优化建议
- 商品层面、流量层面、转化层面

数据要务实，建议要可执行。"""


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
            if cost and price:
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

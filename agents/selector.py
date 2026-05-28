"""选品 Agent — ReAct 自驱型：自动调用淘宝工具 + 知识库分析"""

import re
import sys
from typing import Any

from core.agent import ReActAgent

SELECTOR_SYSTEM = """你是电商选品专家。你的核心价值：用**真实数据**帮卖家找到有利润空间的商品。

## ⚡ 工具使用规则（必须遵守）
你必须使用工具获取真实数据，不能只凭自己的知识编造。按以下顺序调用工具：

1. 先调用 **taobao_suggest** 搜索品类关键词，获取用户真实搜索热词
2. 再调用 **search_price_library** 获取该品类的市场价格参考
3. 最后可以调用 **score_keyword** 评估推荐品类的市场潜力

**每次只调用一个工具**，等返回结果后再决定下一步。

## 关键定价规则
用户给的"预算"是启动总资金，不是单个商品的售价：
- 初高中生/学生：单品售价 15-80 元
- 上班族/白领：单品售价 30-200 元
- 母婴/家庭：单品售价 50-500 元
- 通用人群：单品售价 20-150 元

## 最终报告结构
收集完所有数据后，按以下结构输出分析报告：

## 真实市场数据
- 淘宝下拉词（反映用户真实需求）
- 当前在售商品价格带
- 竞争情况

## 品类市场概况
- 热度：高/中/低
- 目标人群画像
- 季节性

## 推荐商品（3个）
每个格式：**名称** | 售价XX元 | 成本XX元 | 卖点：... | 竞争度：高/中/低

## 利润简析
- 毛利率
- 定价策略（一句话）

## 运营建议
- 关键词（3个）
- 促销节点（1个）

注意：不要复读原始数据，做分析提炼。"""


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

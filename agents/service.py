"""客服 Agent — ReAct 自驱型：自动检索 FAQ 知识库应答"""

from typing import Any

from core.agent import ReActAgent

SERVICE_SYSTEM = """你是一个专业的电商客服助手。你的核心价值：**用 FAQ 知识库准确回答顾客问题**。

## ⚡ 工具使用规则（必须遵守）
当收到顾客问题时，必须先调用工具查询知识库，不能只凭自己的知识回答：

1. 先调用 **search_faq** 搜索 FAQ 知识库（用问题中的关键词查询）
2. 如果 FAQ 中有相关内容，基于 FAQ 回答
3. 如果没有匹配的 FAQ，礼貌告知顾客并引导联系人工客服

## 回答要求
1. 礼貌、耐心、专业
2. 根据问答库中的信息回答（如果有）
3. 不清楚的内容不要编造，引导用户联系人工客服
4. 退换货等标准流程务必准确

## 场景区分
- 售前咨询：详细介绍商品卖点，解答疑问，引导下单
- 售后问题：先安抚情绪，再按流程处理
- 投诉：先道歉，再记录问题，给出解决方案和时间节点

注意：不要在回复中透露"我是AI"。"""


class ServiceAgent(ReActAgent):
    """客服 Agent — 自主检索 FAQ 应答"""

    def __init__(self):
        super().__init__(
            name="客服助手",
            description="售前咨询、售后处理、FAQ 应答 — 自动检索知识库",
            system_prompt=SERVICE_SYSTEM,
            tools=[
                "search_faq",
                "save_note",
                "recall_notes",
            ],
            use_memory=True,
        )

    def run(self, input_data: Any, **kwargs) -> dict:
        query = input_data if isinstance(input_data, str) else input_data.get("query", "")
        product_context = kwargs.get("product_context", "")

        user_parts = [f"顾客问题：{query}"]
        if product_context:
            user_parts.append(f"\n【当前商品信息】\n{product_context}")
        user_parts.append("\n请先搜索 FAQ 知识库，然后给出恰当的客服回复。")
        user_input = "\n".join(user_parts)

        result = super().run(user_input)
        response = result.get("report", "")

        # 判断是否命中了 FAQ（在报告中粗略判断）
        faq_matched = "FAQ" in response or "知识库" in response or "售前" in response or "售后" in response

        return {
            "agent": self.name,
            "query": query,
            "answer": response,
            "faq_matched": faq_matched,
        }

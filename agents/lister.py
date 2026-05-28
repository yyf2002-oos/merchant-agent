"""上架 Agent — 商品标题/描述/SEO 生成 + 自动保存"""

import os
import re
from datetime import datetime
from typing import Any

from core.agent import ReActAgent
from knowledge.rag import get_rag

LISTING_SYSTEM = """你是一个专业的电商商品上架运营专家。
根据用户提供的商品信息，生成完整的商品上架素材包。

请按以下结构输出：

## 商品标题（3个方案）
- 方案1：流量词前置版（不超过30字）
- 方案2：卖点突出版（不超过30字）
- 方案3：长尾关键词版（不超过30字）

## 商品描述
- **核心卖点**：4-6个要点，每点不超过15字
- **详细描述**：2-3段，共约200-300字
- **使用场景**：3个场景

## SEO 信息
- 核心关键词（3-5个）
- 长尾关键词（3-5个）
- 店铺内分类建议

## 规格参数
- 材质/尺寸/重量/颜色/包装等

## 主图建议
- 拍摄风格建议
- 主图展示重点
- 详情页排版建议

语言风格要吸引人但不夸张，符合平台规范。"""


class ListerAgent(ReActAgent):
    """上架 Agent — 商品素材生成"""

    def __init__(self):
        super().__init__(
            name="上架专家",
            description="商品标题生成、详情描述、SEO优化",
            system_prompt=LISTING_SYSTEM,
            tools=None,  # 纯提示词生成，无需工具
            use_memory=False,
        )

    def run(self, input_data: Any, **kwargs) -> dict:
        # 构建输入
        if isinstance(input_data, str):
            product_info = input_data
        else:
            parts = []
            for k in ["name", "category", "features", "target_price", "cost"]:
                if k in input_data:
                    parts.append(f"{k}: {input_data[k]}")
            product_info = "\n".join(parts) if parts else input_data.get("info", str(input_data))

        category = self._extract_category(input_data)

        # 补充品类模板作为参考
        template = get_rag().search_template(category) if category else None
        context = ""
        if template:
            context = f"【参考模板 - {category}品类】\n标题参考：{template['title_patterns'][0]}\n"

        user_prompt = f"{context}\n请为以下商品生成上架素材：\n\n{product_info}"
        result = super().run(user_prompt)
        content = result.get("report", "")

        # 自动保存到文件
        out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", product_info[:20])
        filepath = os.path.join(out_dir, f"上架素材_{safe_name}_{ts}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# 上架素材\n\n{content}\n")

        return {
            "agent": self.name,
            "product": product_info,
            "listing_content": content,
            "category": category,
            "saved_to": filepath,
        }

    def batch_run(self, products: list[dict]) -> list[dict]:
        """批量上架 — 依次为每个商品生成上架素材"""
        results = []
        for i, product in enumerate(products):
            result = self.run(product)
            results.append({
                "index": i + 1,
                "name": product.get("name", f"商品{i+1}"),
                "content": result["listing_content"],
            })
        return results

    @staticmethod
    def _extract_category(input_data) -> str:
        category = ""
        if isinstance(input_data, dict):
            category = input_data.get("category", "")
        elif isinstance(input_data, str) and "品类" in input_data:
            match = re.search(r"品类[：:]\s*(\S+)", input_data)
            if match:
                category = match.group(1)
        return category

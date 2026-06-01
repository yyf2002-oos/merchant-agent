"""上架 Agent — 商品标题/描述/SEO 生成 + 自动保存"""

import os
import re
from datetime import datetime
from typing import Any

from core.agent import ReActAgent
from knowledge.rag import get_rag
from config import AGENT_MODEL, AGENT_LIGHT_MODEL

LISTING_SYSTEM = """你是专业电商上架运营专家，专注淘宝/拼多多平台的商品上架优化。

## 标题生成规则（关键）
标题是搜索流量的入口，必须同时满足**可搜索性**和**可读性**。

### 淘宝标题公式（30字以内）
结构：`核心词 + 属性词 + 场景词 + 卖点词`
- 核心词：用户搜什么（如"无线蓝牙耳机"）
- 属性词：什么特征（如"降噪/长续航/入耳式"）
- 场景词：什么场景用（如"运动/通勤/游戏"）
- 卖点词：为什么买（如"超长续航/高清音质"）

### 拼多多标题公式（30字以内）
结构：`卖点词 + 核心词 + 促销词`
- 偏重性价比和促销感（如"限时特惠""工厂直供""买二送一"）

### 抖音小店标题公式（30字以内）
结构：`痛点词 + 效果词 + 核心词`
- 偏重情绪和场景（如"上班族救星""学生党必备"）

生成3个标题方案：
- 方案1：淘宝搜索优化版（流量词前置）
- 方案2：拼多多/促销版（突出性价比）
- 方案3：通用卖点版（兼顾搜索和点击率）

### 标题禁忌
- 严禁堆砌关键词（会被搜索降权）
- 严禁使用"第一""最好""国家级"等违规词
- 品牌名仅在确实有授权时使用

## 商品描述结构
### 核心卖点（4-6个，每个≤15字）
提炼方法：FAB法则
- Feature（特征）：产品有什么属性
- Advantage（优势）：比竞品好在哪里
- Benefit（利益）：给买家什么好处

示例（蓝牙耳机）：
- ❌ "音质好" → ✅ "Hi-Fi级音质，人声清澈如临现场"
- ❌ "续航长" → ✅ "充电10分钟，听歌2小时"

### 详情描述（200-300字，2-3段）
- 第1段：场景痛点引入 → 引发共鸣
- 第2段：产品如何解决痛点 → 核心卖点展开
- 第3段：使用场景 + 品质保证 → 建立信任

## SEO 优化信息
- **核心关键词**（3-5个）：搜索量大且精准的词
- **长尾关键词**（3-5个）：竞争小但转化高的词（如"学生党蓝牙耳机平价"）
- **类目建议**：淘宝1-2级类目路径
- **搜索排名提升技巧**：标题中所含关键词要在描述中自然出现2-3次

## 规格参数表
用表格格式列出：材质/尺寸/重量/颜色/包装/适用人群
如果是虚拟推断的（非用户提供），用 `[建议]` 标注

## 主图优化建议
- 第1张（搜索图）：纯白底+产品主体，点击率最关键
- 第2张（卖点图）：标注2-3个核心卖点文字叠加
- 第3张（场景图）：使用场景展示
- 第4张（对比图）：与竞品/普通产品的对比
- 第5张（信任图）：资质/质检/实拍证明

## 输出语言要求
- 信息具体、不空洞（❌"质量好" ✅"通过SGS认证，可提供检测报告"）
- 符合广告法，不虚假宣传
- 适配目标平台调性"""


class ListerAgent(ReActAgent):
    """上架 Agent — 商品素材生成"""

    def __init__(self):
        super().__init__(
            name="上架专家",
            description="商品标题生成、详情描述、SEO优化",
            system_prompt=LISTING_SYSTEM,
            tools=None,  # 纯提示词生成，无需工具
            use_memory=False,
            model=AGENT_MODEL["lister"],
            light_model=AGENT_LIGHT_MODEL["lister"],
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

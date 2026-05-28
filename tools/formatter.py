"""格式化工具 — 统一 Agent 输出格式"""

def format_product_card(product: dict) -> str:
    """格式化商品卡片"""
    template = f"""
┌────────────── 商品卡片 ──────────────
│ 名称: {product.get('name', 'N/A')}
│ 品类: {product.get('category', 'N/A')}
│ 目标价: ¥{product.get('target_price', 0)}
│ 成本: ¥{product.get('cost', 0)}
│ 预计利润率: {product.get('margin', 0)}%
│ 推荐理由: {product.get('reason', 'N/A')}
│ 热度: {'⭐' * product.get('hot_rating', 0)}
└─────────────────────────────────────"""
    return template


def format_report(title: str, sections: list[dict]) -> str:
    """格式化报告"""
    lines = [f"\n{'='*50}", f"  {title}", f"{'='*50}"]
    for section in sections:
        lines.append(f"\n▶ {section.get('heading', '')}")
        lines.append(section.get('content', ''))
    return '\n'.join(lines)


def markdown_escape(text: str) -> str:
    """转义 Markdown 特殊字符"""
    chars = r'\*_`{}[]()#+-.!|'
    for c in chars:
        text = text.replace(c, f'\\{c}')
    return text

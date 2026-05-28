"""淘宝数据工具 — 搜索下拉词、搜索结果"""

import json
import time
from typing import Optional
import httpx


def suggest(q: str) -> list[dict]:
    """获取淘宝搜索下拉联想词

    返回示例:
    [{"keyword": "学生文具袋", "count": "100"}, ...]
    """
    url = "https://suggest.taobao.com/sug"
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params={"code": "utf-8", "q": q})
            resp.raise_for_status()
            data = resp.json()
            results = data.get("result", [])
            return [
                {"keyword": item[0], "count": item[1]}
                for item in results
                if isinstance(item, list) and len(item) >= 2
            ]
    except Exception as e:
        return [{"keyword": f"[错误] {e}", "count": "0"}]


def format_suggest_report(suggestions: list[dict], category: str) -> str:
    """格式化下拉词为报告文本"""
    if not suggestions or suggestions[0].get("keyword", "").startswith("[错误]"):
        return ""

    lines = [f"【淘宝搜索下拉词 - {category}】", "用户真实搜索需求（下拉联想）："]
    for s in suggestions:
        lines.append(f"  · {s['keyword']} (热度: {s['count']})")
    lines.append("")
    return "\n".join(lines)



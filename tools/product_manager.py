"""商品价格管理 — 手动录入 + 搜索查询"""

import json
import os
from datetime import datetime
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge", "data")
DATA_FILE = os.path.join(DATA_DIR, "product_library.json")

def _load() -> list[dict]:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save(products: list[dict]):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

def add_product(
    name: str,
    price: float,
    category: str = "",
    platform: str = "",
    source_url: str = "",
    note: str = "",
) -> dict:
    """添加一条商品价格记录"""
    products = _load()
    next_id = max((p.get("id", 0) for p in products), default=0) + 1
    record = {
        "id": next_id,
        "name": name,
        "price": price,
        "category": category,
        "platform": platform,
        "source_url": source_url,
        "note": note,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    products.append(record)
    _save(products)
    return record

def search_products(
    keyword: str = "",
    category: str = "",
    max_price: float = 0,
    min_price: float = 0,
) -> list[dict]:
    """搜索商品"""
    products = _load()
    results = []

    for p in products:
        # 关键词匹配（名称/品类/备注）
        if keyword:
            kw = keyword.lower()
            if kw not in p.get("name", "").lower() and kw not in p.get("category", "").lower() and kw not in p.get("note", "").lower():
                continue
        # 品类筛选
        if category and category.lower() not in p.get("category", "").lower():
            continue
        # 价格范围
        price = p.get("price", 0)
        if max_price > 0 and price > max_price:
            continue
        if min_price > 0 and price < min_price:
            continue
        results.append(p)

    return results

def delete_product(product_id: int) -> bool:
    """删除一条记录"""
    products = _load()
    before = len(products)
    products = [p for p in products if p.get("id") != product_id]
    if len(products) < before:
        _save(products)
        return True
    return False

def list_categories() -> list[str]:
    """列出所有已有的品类"""
    products = _load()
    cats = set()
    for p in products:
        if p.get("category"):
            cats.add(p["category"])
    return sorted(cats)

def get_stats() -> dict:
    """获取统计信息"""
    products = _load()
    if not products:
        return {"total": 0, "categories": 0, "avg_price": 0, "min_price": 0, "max_price": 0}
    prices = [p["price"] for p in products if p.get("price")]
    return {
        "total": len(products),
        "categories": len(set(p.get("category", "") for p in products if p.get("category"))),
        "avg_price": round(sum(prices) / len(prices), 2) if prices else 0,
        "min_price": min(prices) if prices else 0,
        "max_price": max(prices) if prices else 0,
    }

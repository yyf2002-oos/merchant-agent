"""知识库数据管理 — CRUD 操作 JSON 数据文件"""

import json
import os
import logging
from typing import Optional
from copy import deepcopy

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def _load_json(filename: str) -> list | dict:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return [] if filename.endswith("faq.json") else {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载 {filename} 失败: {e}")
        return []

def _save_json(filename: str, data: list | dict) -> bool:
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"保存 {filename} 失败: {e}")
        return False

def get_all_faqs() -> list[dict]:
    return list(_load_json("faq.json"))

def add_faq(q: str, a: str) -> bool:
    faqs = _load_json("faq.json")
    faqs.append({"q": q, "a": a})
    return _save_json("faq.json", faqs)

def update_faq(index: int, q: str, a: str) -> bool:
    faqs = _load_json("faq.json")
    if 0 <= index < len(faqs):
        faqs[index] = {"q": q, "a": a}
        return _save_json("faq.json", faqs)
    return False

def delete_faq(index: int) -> bool:
    faqs = _load_json("faq.json")
    if 0 <= index < len(faqs):
        faqs.pop(index)
        return _save_json("faq.json", faqs)
    return False

def get_price_library() -> list[dict]:
    return list(_load_json("price_library.json"))

def add_price_category(category: str, data: dict) -> bool:
    lib = _load_json("price_library.json")
    for item in lib:
        if item.get("category") == category:
            return False  # 已存在
    lib.append({"category": category, "subcategories": [data]})
    return _save_json("price_library.json", lib)

def update_price_subcategory(category: str, sub_index: int, data: dict) -> bool:
    lib = _load_json("price_library.json")
    for item in lib:
        if item.get("category") == category:
            subs = item.get("subcategories", [])
            if 0 <= sub_index < len(subs):
                subs[sub_index] = data
                return _save_json("price_library.json", lib)
    return False

def add_price_subcategory(category: str, data: dict) -> bool:
    lib = _load_json("price_library.json")
    for item in lib:
        if item.get("category") == category:
            item.setdefault("subcategories", []).append(data)
            return _save_json("price_library.json", lib)
    return False

def delete_price_subcategory(category: str, sub_index: int) -> bool:
    lib = _load_json("price_library.json")
    for item in lib:
        if item.get("category") == category:
            subs = item.get("subcategories", [])
            if 0 <= sub_index < len(subs):
                subs.pop(sub_index)
                return _save_json("price_library.json", lib)
    return False

def get_suppliers() -> list[dict]:
    return list(_load_json("supplier_library.json"))

def add_supplier(category: str, data: dict) -> bool:
    lib = _load_json("supplier_library.json")
    # 检查是否已存在品类
    for item in lib:
        if item.get("category") == category:
            item.setdefault("products", []).append(data)
            return _save_json("supplier_library.json", lib)
    lib.append({
        "category": category,
        "sourcing_region": [],
        "logistics_note": "",
        "tips": "",
        "products": [data],
    })
    return _save_json("supplier_library.json", lib)

def update_supplier_product(category: str, prod_index: int, data: dict) -> bool:
    lib = _load_json("supplier_library.json")
    for item in lib:
        if item.get("category") == category:
            prods = item.get("products", [])
            if 0 <= prod_index < len(prods):
                prods[prod_index] = data
                return _save_json("supplier_library.json", lib)
    return False

def delete_supplier_product(category: str, prod_index: int) -> bool:
    lib = _load_json("supplier_library.json")
    for item in lib:
        if item.get("category") == category:
            prods = item.get("products", [])
            if 0 <= prod_index < len(prods):
                prods.pop(prod_index)
                return _save_json("supplier_library.json", lib)
    return False

def get_all_templates() -> list[dict]:
    return list(_load_json("product_templates.json"))

def get_all_categories() -> list[str]:
    """获取所有知识库中涉及的品类（去重排序）"""
    cats = set()
    for lib_name in ["price_library.json", "supplier_library.json"]:
        data = _load_json(lib_name)
        for item in data:
            if isinstance(item, dict) and item.get("category"):
                cats.add(item["category"])
    for t in _load_json("product_templates.json"):
        if isinstance(t, dict) and t.get("category"):
            cats.add(t["category"])
    return sorted(cats)

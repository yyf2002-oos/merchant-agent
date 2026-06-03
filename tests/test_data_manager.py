"""测试知识库数据管理（CRUD）"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LLM_PROVIDER"] = "ollama"

import pytest
from knowledge.data_manager import (
    get_all_faqs, add_faq, update_faq, delete_faq,
    get_price_library, add_price_subcategory,
    get_suppliers, get_all_categories,
)


class TestFAQManager:
    def test_get_all_faqs(self):
        faqs = get_all_faqs()
        assert isinstance(faqs, list)
        assert len(faqs) > 0
        assert "q" in faqs[0]
        assert "a" in faqs[0]

    def test_add_and_delete_faq(self):
        n_before = len(get_all_faqs())
        add_faq("测试问题", "测试答案")
        assert len(get_all_faqs()) == n_before + 1

        # 找到刚添加的索引
        faqs = get_all_faqs()
        idx = next(i for i, f in enumerate(faqs) if f["q"] == "测试问题")
        delete_faq(idx)
        assert len(get_all_faqs()) == n_before


class TestPriceLibrary:
    def test_get_price(self):
        lib = get_price_library()
        assert isinstance(lib, list)
        if lib:
            assert "category" in lib[0]
            assert "subcategories" in lib[0]

    def test_all_categories(self):
        cats = get_all_categories()
        assert isinstance(cats, list)
        assert len(cats) > 0


class TestSuppliers:
    def test_get_suppliers(self):
        suppliers = get_suppliers()
        assert isinstance(suppliers, list)
        if suppliers:
            assert "category" in suppliers[0]

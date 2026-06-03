"""语义 RAG — 基于 bge-m3 embedding 的知识检索（自动降级到关键词）"""

import json
import os
import re
import hashlib
import threading
from typing import Optional, Any

from config import RAG_TOP_K
from knowledge.embedding import get_embedding, get_embeddings_batch, rank_by_similarity

EMBED_CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", ".embed_cache.json")


def _cn_tokenize(text: str) -> set[str]:
    """中文+英文混合分词（降级用）"""
    text = text.lower()
    tokens = set()
    for word in re.findall(r"[a-z0-9]+", text):
        if word:
            tokens.add(word)
    chars = re.findall(r"[\u4e00-\u9fff]+", text)
    for chunk in chars:
        for ch in chunk:
            tokens.add(ch)
        for i in range(len(chunk) - 1):
            tokens.add(chunk[i : i + 2])
    return tokens


class SemanticRAG:
    """语义 RAG — embedding 检索 + 关键词降级"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.data_dir = data_dir
        self._cache: dict[str, Any] = {}
        self._embed_cache: dict[str, list[float]] = self._load_embed_cache()
        self._use_embedding = True
        self._lock = threading.Lock()

    # ── Embedding 缓存 ──────────────────────────

    def _load_embed_cache(self) -> dict[str, list[float]]:
        if os.path.exists(EMBED_CACHE_FILE):
            try:
                with open(EMBED_CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_embed_cache(self):
        os.makedirs(os.path.dirname(EMBED_CACHE_FILE), exist_ok=True)
        with open(EMBED_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(self._embed_cache, f, ensure_ascii=False)

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _get_or_compute_embedding(self, text: str) -> Optional[list[float]]:
        key = self._cache_key(text)
        with self._lock:
            if key in self._embed_cache:
                return self._embed_cache[key]
        emb = get_embedding(text)
        if emb:
            with self._lock:
                self._embed_cache[key] = emb
                self._save_embed_cache()
        return emb

    def _batch_embed_and_cache(self, texts: list[str]):
        """批量计算并缓存 embedding"""
        uncached = []
        with self._lock:
            for t in texts:
                k = self._cache_key(t)
                if k not in self._embed_cache:
                    uncached.append(t)
        if uncached:
            embs = get_embeddings_batch(uncached)
            with self._lock:
                for t, emb in zip(uncached, embs):
                    if emb:
                        self._embed_cache[self._cache_key(t)] = emb
                self._save_embed_cache()

    # ── 数据加载 ────────────────────────────────

    def _load(self, filename: str) -> list[dict]:
        if filename not in self._cache:
            path = os.path.join(self.data_dir, filename)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self._cache[filename] = json.load(f)
            else:
                self._cache[filename] = []
        return self._cache[filename]

    # ── 语义搜索 ────────────────────────────────

    def _semantic_search(self, query: str, items: list[dict],
                         text_fields: list[str], k: int) -> list[dict]:
        """语义搜索：embedding 相似度排序"""
        query_emb = self._get_or_compute_embedding(query)
        if not query_emb:
            return self._keyword_fallback(query, items, text_fields, k)

        # 为每个 item 构建搜索文本 + 取 embedding
        search_texts = []
        for item in items:
            parts = []
            for field in text_fields:
                val = item.get(field, "")
                if isinstance(val, str):
                    parts.append(val)
                elif isinstance(val, list):
                    parts.extend(str(v) for v in val)
            search_texts.append(" ".join(parts))

        self._batch_embed_and_cache(search_texts)

        candidates = []
        for idx, st in enumerate(search_texts):
            emb = self._embed_cache.get(self._cache_key(st))
            if emb:
                candidates.append((st, emb))

        if not candidates:
            return self._keyword_fallback(query, items, text_fields, k)

        ranked = rank_by_similarity(query_emb, candidates)
        return [items[idx] for idx, _ in ranked[:k]]

    def _keyword_fallback(self, query: str, items: list[dict],
                          text_fields: list[str], k: int) -> list[dict]:
        """关键词降级搜索"""
        query_tokens = _cn_tokenize(query)
        scored = []
        for item in items:
            parts = []
            for field in text_fields:
                val = item.get(field, "")
                if isinstance(val, str):
                    parts.append(val)
                elif isinstance(val, list):
                    parts.extend(str(v) for v in val)
            text = " ".join(parts)
            item_tokens = _cn_tokenize(text)
            match_count = len(query_tokens & item_tokens)
            if match_count > 0:
                scored.append((match_count, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:k]]

    # ── 公开检索接口 ────────────────────────────

    def _category_match(self, category: str, stored_cat: str) -> bool:
        """品类匹配：精确优先，子串匹配要求至少 3 字符防过度匹配"""
        if not stored_cat:
            return False
        if category == stored_cat:
            return True
        if len(category) >= 3 and len(stored_cat) >= 3:
            return category in stored_cat or stored_cat in category
        return False

    def search_faq(self, query: str, k: int = None) -> list[dict]:
        """检索 FAQ"""
        if k is None:
            k = RAG_TOP_K
        faqs = self._load("faq.json")
        return self._semantic_search(query, faqs, ["q", "a", "category"], k)

    def search_price_library(self, category: str) -> Optional[dict]:
        """按品类查找价格库"""
        lib = self._load("price_library.json")
        for item in lib:
            if item.get("category") == category:
                return item
        # 语义匹配
        items = self._semantic_search(category, lib, ["category", "subcategories"], 1)
        return items[0] if items else None

    def search_supplier(self, category: str) -> Optional[dict]:
        """按品类查找供应商"""
        lib = self._load("supplier_library.json")
        for item in lib:
            if self._category_match(category, item.get("category", "")):
                return item
        items = self._semantic_search(category, lib, ["category", "sourcing_region"], 1)
        return items[0] if items else None

    def search_region_advantage(self, category: str) -> Optional[dict]:
        """按品类查找产区优势"""
        raw = self._load("region_advantages.json")
        clusters = raw.get("clusters", []) if isinstance(raw, dict) else raw
        for item in clusters:
            if self._category_match(category, item.get("category", "")):
                return item
        items = self._semantic_search(category, clusters, ["category", "regions", "supply_chain_tiers"], 1)
        return items[0] if items else None

    def search_template(self, category: str) -> Optional[dict]:
        """按品类查找上架模板"""
        templates = self._load("product_templates.json")
        for t in templates:
            if t["category"] == category:
                return t
        items = self._semantic_search(category, templates, ["category", "template"], 1)
        return items[0] if items else None

    # ── 格式化输出（与原接口完全兼容） ──────────

    def format_price_context(self, category: str) -> str:
        data = self.search_price_library(category)
        if not data:
            return ""
        ctx = f"【参考价格库 - {data['category']}】\n"
        for sub in data.get("subcategories", []):
            ctx += (f"· {sub['name']}: ¥{sub['price_range']}, "
                    f"成本¥{sub['cost_range']}, 目标{sub['target']}, "
                    f"竞争{sub['competition']}\n")
        ctx += "注意：以上为真实市场价格参考。\n"
        return ctx

    def format_supplier_context(self, category: str) -> str:
        data = self.search_supplier(category)
        if not data:
            return ""
        ctx = f"【供应商参考库 - {data['category']}】\n"
        ctx += f"产区：{', '.join(data.get('sourcing_region', []))}\n"
        ctx += f"物流说明：{data.get('logistics_note', '')}\n"
        ctx += f"采购贴士：{data.get('tips', '')}\n"
        for p in data.get("products", []):
            ctx += (f"· {p['name']}: 批发¥{p['wholesale_price']}, "
                    f"MOQ{p['moq']}, MOQ价¥{p['moq_price']}\n")
            ctx += f"  供应源：{', '.join(p['supplier_types'])} | {p['sourcing_notes']}\n"
        ctx += "注意：以上为市场价格参考区间。\n"
        return ctx

    def format_faq_context(self, query: str) -> str:
        results = self.search_faq(query)
        if not results:
            return ""
        ctx = "【参考问答库】\n"
        for i, r in enumerate(results, 1):
            ctx += f"{i}. Q: {r['q']}\n   A: {r['a']}\n\n"
        return ctx

    def format_region_context(self, category: str) -> tuple[str, str]:
        data = self.search_region_advantage(category)
        if not data:
            return "", ""
        regions_strs = []
        all_regions = []
        for r in data.get("regions", []):
            regions_strs.append(f"【{r['area']}（{r['type']}）】")
            regions_strs.append(f"专注：{r['advantage']}")
            regions_strs.append(f"找厂技巧：{r['factory_tip']}")
            regions_strs.append(f"一手辨别：{r['first_hand_vs_trader']}")
            if r.get("startup_approach"):
                regions_strs.append(f"起步建议：{r['startup_approach']}")
            all_regions.append(r['area'])
        tier_text = "\n供应链层级：\n"
        for t in data.get("supply_chain_tiers", []):
            tier_text += f"  {t['tier']} → {t['location']}（{t['contact']}）\n"
        ctx = f"\n【产区优势 - {data['category']}】\n" + "\n".join(regions_strs) + tier_text
        return ctx, "、".join(all_regions)

    def get_supported_categories(self) -> list[str]:
        cats = set()
        for lib_name in ["supplier_library.json", "price_library.json"]:
            lib = self._load(lib_name)
            for item in lib:
                cats.add(item["category"])
        return sorted(cats)

    def get_categories(self) -> list[str]:
        templates = self._load("product_templates.json")
        return [t["category"] for t in templates]


# 全局单例（接口兼容）
_rag = None


def get_rag() -> SemanticRAG:
    global _rag
    if _rag is None:
        _rag = SemanticRAG()
    return _rag

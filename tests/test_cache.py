"""测试缓存模块"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pytest
from cache import TTLCache

class TestTTLCache:
    def test_set_and_get(self):
        cache = TTLCache(capacity=10, ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_miss_returns_none(self):
        cache = TTLCache(capacity=10, ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = TTLCache(capacity=10, ttl_seconds=1)
        cache.set("key", "value")
        assert cache.get("key") == "value"
        time.sleep(1.1)
        assert cache.get("key") is None

    def test_capacity_eviction(self):
        cache = TTLCache(capacity=3, ttl_seconds=60)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        cache.set("d", "4")  # 应淘汰 a
        assert cache.get("a") is None
        assert cache.get("d") == "4"

    def test_lru_move_to_end(self):
        cache = TTLCache(capacity=2, ttl_seconds=60)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.get("a")  # a 变为最近使用
        cache.set("c", "3")  # 淘汰 b
        assert cache.get("b") is None
        assert cache.get("a") == "1"
        assert cache.get("c") == "3"

    def test_stats(self):
        cache = TTLCache(capacity=10, ttl_seconds=60)
        cache.get("miss1")
        cache.get("miss2")
        cache.set("hit", "v")
        cache.get("hit")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] > 0

    def test_clear(self):
        cache = TTLCache(capacity=10, ttl_seconds=60)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.stats()["size"] == 0

    def test_key_normalization(self):
        cache = TTLCache(capacity=10, ttl_seconds=60)
        cache.set("Hello World", "val")
        assert cache.get("Hello World") == "val"

    def test_empty_key(self):
        cache = TTLCache(capacity=10, ttl_seconds=60)
        cache.set("", "empty")
        assert cache.get("") == "empty"

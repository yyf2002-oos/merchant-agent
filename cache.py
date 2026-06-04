"""
缓存模块 — LRU + TTL 内存缓存

用法:
    cache = TTLCache(capacity=200, ttl_seconds=3600)
    cache.get(key)   # 命中返回 value，未命中返回 None
    cache.set(key, value)
    cache.stats()    # 命中率统计
"""
import time
import hashlib
import logging
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

class TTLCache:
    """LRU + TTL 内存缓存（适合单进程场景）"""

    def __init__(self, capacity: int = 200, ttl_seconds: int = 3600):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        return hashlib.md5(text.strip().encode()).hexdigest()

    def get(self, raw_key: str) -> Optional[object]:
        key = self._key(raw_key)
        if key not in self._store:
            self.misses += 1
            return None
        ts, value = self._store[key]
        if time.time() - ts > self.ttl:
            del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return value

    def set(self, raw_key: str, value: object):
        key = self._key(raw_key)
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (time.time(), value)
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def clear(self):
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total * 100, 1) if total else 0,
        }

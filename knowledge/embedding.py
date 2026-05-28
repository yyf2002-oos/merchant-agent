"""Embedding 服务 — 用 Ollama bge-m3 做语义向量"""
import numpy as np
import httpx
from typing import Optional
from config import OLLAMA_BASE

EMBED_MODEL = "bge-m3"


def get_embedding(text: str, model: str = EMBED_MODEL) -> Optional[list[float]]:
    """获取单段文本的 embedding 向量"""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{OLLAMA_BASE}/api/embed",
                json={"model": model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            return embeddings[0] if embeddings else None
    except Exception:
        return None


def get_embeddings_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """批量获取 embedding 向量"""
    if not texts:
        return []
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{OLLAMA_BASE}/api/embed",
                json={"model": model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("embeddings", [])
    except Exception:
        return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度"""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def rank_by_similarity(query_emb: list[float], candidates: list[tuple[str, list[float]]]) -> list[tuple[int, float]]:
    """按余弦相似度排序 candidates，返回 [(index, score), ...]"""
    scores = []
    for idx, (_, cand_emb) in enumerate(candidates):
        score = cosine_similarity(query_emb, cand_emb)
        scores.append((idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

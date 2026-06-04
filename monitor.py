"""LLM call monitor — tracks call duration, token usage, and cost"""

import time
import json
import os
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

MONITOR_DB = os.path.join(os.path.dirname(__file__), "monitor.db")
_db_initialized = False

def _get_conn():
    global _db_initialized
    conn = sqlite3.connect(MONITOR_DB)
    conn.row_factory = sqlite3.Row
    if not _db_initialized:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                agent TEXT DEFAULT '',
                session_id TEXT DEFAULT '',
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost REAL DEFAULT 0,
                duration_ms INTEGER DEFAULT 0,
                success INTEGER DEFAULT 1,
                error TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_calls_created ON llm_calls(created_at);
            CREATE INDEX IF NOT EXISTS idx_calls_provider ON llm_calls(provider);
        """)
        conn.commit()
        _db_initialized = True
    return conn

# DeepSeek 价格（每百万 token，人民币估算）
DEEPSEEK_PRICES = {
    "deepseek-chat": {"input": 1.0, "output": 2.0},
    "deepseek-reasoner": {"input": 4.0, "output": 16.0},
    "deepseek-v4-flash": {"input": 0.3, "output": 0.6},
    "deepseek-v4-pro": {"input": 2.0, "output": 8.0},
}

def record_call(
    provider: str,
    model: str,
    duration_ms: int,
    success: bool = True,
    error: str = "",
    agent: str = "",
    session_id: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
):
    """记录一次 LLM 调用"""
    total_tokens = prompt_tokens + completion_tokens
    cost = 0.0
    if provider == "deepseek" and success:
        prices = DEEPSEEK_PRICES.get(model, DEEPSEEK_PRICES["deepseek-chat"])
        cost = (prompt_tokens * prices["input"] + completion_tokens * prices["output"]) / 1_000_000

    conn = _get_conn()
    conn.execute(
        """INSERT INTO llm_calls
           (provider, model, agent, session_id, prompt_tokens, completion_tokens, total_tokens, cost, duration_ms, success, error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (provider, model, agent, session_id, prompt_tokens, completion_tokens, total_tokens, round(cost, 6), duration_ms, 1 if success else 0, error[:200] if error else ""),
    )
    conn.commit()
    conn.close()

def get_stats(hours: int = 24) -> dict:
    """获取最近 N 小时的统计"""
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    conn = _get_conn()
    stats = {"calls": 0, "success": 0, "failed": 0, "total_cost": 0.0, "avg_duration_ms": 0, "by_provider": {}}

    row = conn.execute(
        """SELECT COUNT(*) as calls,
                  SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as success,
                  SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as failed,
                  COALESCE(SUM(cost), 0) as total_cost,
                  COALESCE(AVG(CASE WHEN duration_ms>0 THEN duration_ms END), 0) as avg_dur
           FROM llm_calls WHERE created_at >= ?""",
        (cutoff,),
    ).fetchone()
    if row:
        stats["calls"] = row[0]
        stats["success"] = row[1] or 0
        stats["failed"] = row[2] or 0
        stats["total_cost"] = round(row[3], 4) if row[3] else 0.0
        stats["avg_duration_ms"] = round(row[4]) if row[4] else 0

    rows = conn.execute(
        """SELECT provider, COUNT(*) as cnt, COALESCE(SUM(cost),0) as cost
           FROM llm_calls WHERE created_at >= ? GROUP BY provider""",
        (cutoff,),
    ).fetchall()
    for r in rows:
        stats["by_provider"][r[0]] = {"calls": r[1], "cost": round(r[2], 4)}

    conn.close()
    return stats

def get_recent_calls(limit: int = 20) -> list[dict]:
    """获取最近的调用记录"""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM llm_calls ORDER BY id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_daily_stats(days: int = 7) -> list[dict]:
    """获取每日统计"""
    cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT DATE(created_at) as day,
                  COUNT(*) as calls,
                  COALESCE(SUM(cost),0) as cost,
                  COALESCE(AVG(duration_ms),0) as avg_dur
           FROM llm_calls WHERE DATE(created_at) >= ?
           GROUP BY DATE(created_at) ORDER BY day""",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [{"day": r[0], "calls": r[1], "cost": round(r[2], 4), "avg_duration_ms": round(r[3])} for r in rows]

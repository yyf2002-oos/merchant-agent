"""测试调用监控系统"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LLM_PROVIDER"] = "ollama"

import pytest
from monitor import record_call, get_stats, get_recent_calls, get_daily_stats

class TestMonitor:
    def test_record_and_stats(self):
        record_call(provider="deepseek", model="deepseek-chat", duration_ms=1500, success=True, agent="test_agent")
        record_call(provider="ollama", model="qwen2:7b", duration_ms=3000, success=False, error="超时")
        stats = get_stats(hours=48)
        assert stats["calls"] >= 2
        assert stats["success"] >= 1
        assert stats["failed"] >= 1

    def test_recent_calls(self):
        recent = get_recent_calls(limit=5)
        assert len(recent) <= 5
        for r in recent:
            assert "provider" in r
            assert "duration_ms" in r

    def test_daily_stats(self):
        daily = get_daily_stats(days=7)
        assert isinstance(daily, list)

    def test_record_with_cost(self):
        record_call(provider="deepseek", model="deepseek-chat", duration_ms=500, success=True,
                    prompt_tokens=1000, completion_tokens=500, agent="cost_test")
        stats = get_stats(hours=48)
        assert stats["total_cost"] >= 0

    def test_error_record(self):
        record_call(provider="ollama", model="qwen2:7b", duration_ms=100, success=False,
                    error="Connection refused", agent="error_test")
        recent = get_recent_calls(limit=10)
        errs = [r for r in recent if r.get("success") == 0]
        assert len(errs) >= 1

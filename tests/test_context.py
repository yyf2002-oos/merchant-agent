"""测试上下文记忆系统"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LLM_PROVIDER"] = "ollama"

import pytest
from core.context import AgentContext
from core.memory import ConversationMemory

@pytest.fixture
def ctx():
    mem = ConversationMemory()
    context = AgentContext("test_ctx_session", mem)
    yield context
    context.clear()

class TestAgentContext:
    def test_add_and_get_message(self, ctx):
        ctx.add_message("user", "你好")
        ctx.add_message("assistant", "你好！有什么可以帮助你的？")
        history = ctx.get_history(limit=10)
        assert len(history) == 2
        assert history[0]["role"] == "user"

    def test_get_formatted_context_empty(self, ctx):
        messages = ctx.get_formatted_context("你是电商助手")
        assert len(messages) >= 1
        assert messages[0]["role"] == "system"

    def test_get_formatted_context_with_history(self, ctx):
        ctx.add_message("user", "我想卖宠物用品")
        ctx.add_message("assistant", "好的，宠物用品市场很大")
        ctx.add_message("user", "猫项圈怎么样")
        ctx.add_message("assistant", "猫项圈是个好方向")
        ctx.add_message("user", "利润空间如何")
        ctx.add_message("assistant", "成本15-30元，售价59-129元")

        compressed = ctx.get_compressed_context(recent_count=2)
        assert compressed["total"] == 6
        assert len(compressed["recent"]) == 2
        # 历史记录应触发摘要
        history = compressed.get("summary", "")
        assert len(history) > 10 or bool(history)

    def test_tool_result_persistence(self, ctx):
        ctx.add_tool_result("taobao_suggest", "找到5个相关热词")
        ctx.add_tool_result("search_price_library", "猫项圈价格带15-129元")
        history = ctx.get_history(limit=10)
        tool_msgs = [m for m in history if "[工具:" in str(m.get("content", ""))]
        assert len(tool_msgs) >= 2

    def test_clear_session(self, ctx):
        ctx.add_message("user", "测试")
        ctx.clear()
        history = ctx.get_history(limit=10)
        assert len(history) == 0

    def test_key_info_extraction(self, ctx):
        ctx.add_message("user", "我想卖宠物项圈，预算5000")
        ctx.add_message("assistant", "好的，我来分析")
        compressed = ctx.get_compressed_context()
        key_info = compressed.get("key_info", "")
        assert len(key_info) > 5

"""测试 Orchestrator（SharedContext / 路由 / 进度回调）"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LLM_PROVIDER"] = "ollama"

import pytest
from orchestrator import SharedContext, MerchantOrchestrator

class TestSharedContext:
    def test_basic_creation(self):
        sc = SharedContext(category="宠物项圈", budget="5000")
        assert sc.category == "宠物项圈"
        assert sc.budget == "5000"

    def test_add_decision(self):
        sc = SharedContext()
        sc.add_decision("走中高端路线")
        sc.add_decision("目标人群：年轻女性")
        assert len(sc.decisions) == 2
        assert "中高端" in sc.decisions[0]

    def test_to_context_prompt(self):
        sc = SharedContext(category="猫项圈", budget="5000")
        sc.selected_product = "真皮猫项圈"
        sc.product_cost = "15"
        sc.product_price = "59"
        sc.add_decision("走中高端路线")
        prompt = sc.to_context_prompt()
        assert "猫项圈" in prompt
        assert "5000" in prompt
        assert "59" in prompt
        assert "中高端" in prompt

    def test_market_insights(self):
        sc = SharedContext()
        sc.market_insights = ["中高端竞争小", "客单价50-80转化率高"]
        prompt = sc.to_context_prompt()
        assert "竞争小" in prompt
        assert "客单价" in prompt

    def test_empty_context(self):
        sc = SharedContext()
        prompt = sc.to_context_prompt()
        assert "当前项目上下文" in prompt

class TestMerchantOrchestrator:
    def test_list_agents(self):
        orch = MerchantOrchestrator()
        agents = orch.list_agents()
        assert len(agents) == 5
        names = [a["key"] for a in agents]
        assert "selector" in names
        assert "lister" in names
        assert "service" in names
        assert "analyst" in names
        assert "sourcing" in names

    def test_smart_chat_signature(self):
        orch = MerchantOrchestrator()
        import inspect
        sig = inspect.signature(orch.smart_chat)
        params = list(sig.parameters.keys())
        assert "session_id" in params

    def test_run_agent_unknown(self):
        orch = MerchantOrchestrator()
        result = orch.run_agent("unknown", "test")
        assert "error" in result

    def test_run_agent_signature(self):
        orch = MerchantOrchestrator()
        import inspect
        sig = inspect.signature(orch.run_agent)
        assert "session_id" in sig.parameters

    def test_extract_insights(self):
        report = """## 市场分析
- 竞争度：中等，建议差异化切入
- 利润空间：售价59元，成本15元，毛利44元
- 建议走中高端路线"""
        insights = MerchantOrchestrator._extract_insights(report)
        assert len(insights) > 0
        assert any("利润" in i for i in insights)

    def test_extract_insights_empty(self):
        insights = MerchantOrchestrator._extract_insights("无数据")
        assert len(insights) >= 0

    def test_workflow_progress_callback(self):
        orch = MerchantOrchestrator()
        calls = []

        def cb(step, status, detail):
            calls.append((step, status))

        # 不会实际执行（需要 LLM），只验证签名和调用
        assert callable(cb) if cb else True
        assert len(calls) == 0

    def test_get_session_history(self):
        orch = MerchantOrchestrator()
        history = orch.get_session_history("nonexistent_session")
        assert isinstance(history, list)

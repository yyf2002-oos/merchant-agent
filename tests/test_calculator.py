"""测试计算工具"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools.calculator import profit_analysis, price_suggestion, keyword_score

class TestProfitAnalysis:
    def test_basic_profit(self):
        r = profit_analysis(cost=10, price=25, volume=200, logistics_cost=0)
        assert r["revenue"] == 5000
        assert r["cost"] == 2000
        assert r["profit"] > 0
        assert r["margin"] > 0

    def test_zero_volume(self):
        r = profit_analysis(cost=10, price=25, volume=0)
        assert r["revenue"] == 0
        assert r["profit"] == 0
        assert r["margin"] == 0

    def test_platform_fee(self):
        r = profit_analysis(cost=100, price=200, volume=10, platform_fee_rate=0.1)
        assert r["platform_fee"] == 200
        assert r["total_cost"] == 1000 + 200

    def test_logistics_cost(self):
        r = profit_analysis(cost=10, price=30, volume=100, logistics_cost=200)
        assert r["logistics_cost"] == 200

    def test_negative_margin(self):
        """售价低于成本时利润为负"""
        r = profit_analysis(cost=50, price=30, volume=10)
        assert r["profit"] < 0
        assert r["margin"] < 0

class TestPriceSuggestion:
    def test_basic_pricing(self):
        r = price_suggestion(cost=50, target_margin=30)
        assert r["cost"] == 50
        assert r["competitive_price"] == 125
        assert r["premium_price"] == 200

    def test_zero_cost(self):
        r = price_suggestion(cost=0, target_margin=30)
        assert r["min_price"] == 0

    def test_high_target_margin(self):
        r = price_suggestion(cost=100, target_margin=60)
        assert r["min_price"] > r["cost"]

class TestKeywordScore:
    def test_normal_score(self):
        s = keyword_score(search_volume=80000, competition=40, avg_price=129, cost=50)
        assert 0 <= s <= 100

    def test_zero_cost(self):
        s = keyword_score(search_volume=10000, competition=30, avg_price=50, cost=0)
        assert s >= 0
        # cost=0 时利润分为 0
        assert s < 100

    def test_high_demand_low_competition(self):
        s_high = keyword_score(100000, 10, 100, 30)
        s_low = keyword_score(100, 90, 100, 30)
        assert s_high > s_low

    def test_low_volume_edge(self):
        # competition=0 时竞争分 = 100*0.3 = 30
        s = keyword_score(search_volume=0, competition=0, avg_price=0, cost=1)
        assert s > 0  # 竞争分为 30
        assert s <= 100

    def test_all_zero_no_cost(self):
        # cost=0 时利润分被跳过，但 competition=0 → 竞争分 = 30
        s = keyword_score(search_volume=0, competition=0, avg_price=0, cost=0)
        assert s == 30.0  # 只有竞争分 (100-0)*0.3

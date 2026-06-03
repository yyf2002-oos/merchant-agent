"""计算工具 — 利润分析、定价计算等"""

def profit_analysis(cost: float, price: float, volume: int,
                    platform_fee_rate: float = 0.05,
                    logistics_cost: float = 0) -> dict:
    """利润分析"""
    revenue = price * volume
    platform_fee = revenue * platform_fee_rate
    total_cost = (cost * volume) + platform_fee + logistics_cost
    profit = revenue - total_cost
    margin = (profit / revenue) * 100 if revenue > 0 else 0
    return {
        "revenue": round(revenue, 2),
        "cost": round(cost * volume, 2),
        "platform_fee": round(platform_fee, 2),
        "logistics_cost": round(logistics_cost, 2),
        "total_cost": round(total_cost, 2),
        "profit": round(profit, 2),
        "margin": round(margin, 2),
    }


def price_suggestion(cost: float, target_margin: float = 30.0,
                    platform_fee_rate: float = 0.05) -> dict:
    """根据目标利润率建议售价"""
    denominator = 1 - platform_fee_rate - target_margin / 100
    min_price = cost / denominator if denominator > 0 else cost * 10  # 防除零，用 10 倍成本作为保本价
    competitive_price = cost * 2.5  # 一般电商 2.5 倍定价
    premium_price = cost * 4.0
    return {
        "cost": cost,
        "min_price": round(min_price, 2),
        "competitive_price": round(competitive_price, 2),
        "premium_price": round(premium_price, 2),
        "note": "min_price=保本价, competitive=常规价, premium=品牌溢价价",
    }


def keyword_score(search_volume: int, competition: int,
                  avg_price: float, cost: float) -> float:
    """关键词/品类的综合评分（0-100）"""
    demand_score = min(search_volume / 10000 * 40, 40)  # 搜索量最多 40 分
    competition_score = max(0, (100 - competition) * 0.3)  # 竞争度越低越高
    if cost > 0:
        profit_score = min(max((avg_price - cost) / cost * 10, 0), 30)  # 利润空间最多 30 分
    else:
        profit_score = 0  # 成本为 0 时不计利润分
    return round(demand_score + competition_score + profit_score, 1)

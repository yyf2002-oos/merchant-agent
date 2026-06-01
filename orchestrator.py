"""Master Orchestrator — 任务路由、Agent 编排、完整工作流"""

import logging

# ⚡ 注册所有工具（必须在导入 Agent 之前）
import core.tools_registry  # noqa: F401

from agents.selector import SelectorAgent
from agents.lister import ListerAgent
from agents.service import ServiceAgent
from agents.analyst import AnalystAgent
from agents.sourcing import SourcingAgent
from llm import simple_prompt
from config import LOG_LEVEL, CACHE_ENABLED, CACHE_TTL, CACHE_CAPACITY
from cache import TTLCache

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL))


class MerchantOrchestrator:
    """商家智能 Agent 主控器"""

    def __init__(self):
        self.agents = {
            "selector": SelectorAgent(),
            "lister": ListerAgent(),
            "service": ServiceAgent(),
            "analyst": AnalystAgent(),
            "sourcing": SourcingAgent(),
        }
        self.cache = TTLCache(capacity=CACHE_CAPACITY, ttl_seconds=CACHE_TTL) if CACHE_ENABLED else None
        logger.info(f"Orchestrator 初始化完成，已加载 {len(self.agents)} 个 Agent 缓存={'启用' if self.cache else '禁用'}")

    def list_agents(self) -> list[dict]:
        return [
            {"key": k, "name": v.name, "desc": v.description}
            for k, v in self.agents.items()
        ]

    def run_agent(self, agent_key: str, input_data, **kwargs) -> dict:
        agent = self.agents.get(agent_key)
        if not agent:
            logger.error(f"未知 Agent: {agent_key}")
            return {"error": f"未知 Agent: {agent_key}"}
        logger.info(f"运行 Agent: {agent_key} input={str(input_data)[:80]}")
        return agent.run(input_data, **kwargs)

    def run_full_workflow(self, category: str, budget: str = "",
                          target_audience: str = "",
                          product_info: str = "") -> dict:
        """一键完整工作流：选品 → 上架 → 客服 → 分析（带跨 Agent 上下文传递）"""
        logger.info(f"完整工作流启动: category={category} budget={budget} audience={target_audience}")
        results = {"category": category}
        shared_context = {"category": category, "budget": budget, "target_audience": target_audience}

        # Step 1: 选品
        logger.info("工作流 Step 1/4: 选品分析")
        selector_result = self.agents["selector"].run(
            category, budget=budget, target_audience=target_audience
        )
        results["selector"] = selector_result
        # 传递选品结果给后续步骤
        if selector_result.get("recommendations"):
            best = selector_result["recommendations"][0]
            shared_context["product_name"] = best.get("name", "")
            shared_context["product_cost"] = str(best.get("cost", ""))
            shared_context["product_price"] = str(best.get("avg_price", ""))
            shared_context["selector_report"] = selector_result.get("report", "")[:500]

        # Step 2: 上架（基于选品结果，带上品类上下文）
        logger.info("工作流 Step 2/4: 上架素材")
        listing_input = product_info or f"品类：{category}\n"
        if shared_context.get("product_name"):
            listing_input += (
                f"名称：{shared_context['product_name']}\n"
                f"成本：{shared_context.get('product_cost', '')}\n"
                f"目标价：{shared_context.get('product_price', '')}\n"
                f"【选品分析摘要】\n{shared_context.get('selector_report', '')}"
            )

        lister_result = self.agents["lister"].run(listing_input)
        results["lister"] = lister_result
        if lister_result.get("listing_content"):
            shared_context["listing_content"] = lister_result["listing_content"][:300]

        # Step 3: 客服（带上上架的商品信息上下文）
        logger.info("工作流 Step 3/4: 客服应答")
        service_context = f"品类：{category}"
        if shared_context.get("product_name"):
            service_context += f"\n商品：{shared_context['product_name']}\n售价：{shared_context.get('product_price', '')}元"
        service_welcome = self.agents["service"].run(
            "你好，我想了解一下这个商品",
            product_context=service_context,
        )
        results["service"] = service_welcome

        # Step 4: 运营分析（带全流程上下文）
        logger.info("工作流 Step 4/4: 运营分析")
        analyst_input = (
            f"新店铺启动，品类：{category}，预算：{budget or '未设置'}"
            f"，目标人群：{target_audience or '未指定'}"
        )
        if shared_context.get("product_name"):
            analyst_input += (
                f"\n\n【已选商品】\n{shared_context['product_name']} "
                f"(成本{shared_context.get('product_cost', '?')}元, "
                f"售价{shared_context.get('product_price', '?')}元)"
            )
        analyst_result = self.agents["analyst"].run(analyst_input)
        results["analyst"] = analyst_result

        logger.info(f"完整工作流完成: {category}")
        return results

    def batch_listing(self, products: list[dict]) -> list[dict]:
        """批量上架 — 一次生成多个商品的上架素材"""
        logger.info(f"批量上架: {len(products)} 个商品")
        results = self.agents["lister"].batch_run(products)
        logger.info(f"批量上架完成: {len(results)}/{len(products)}")
        return results

    def smart_chat(self, user_input: str) -> str:
        """智能对话 — 多维度路由（LLM 分类 + 关键词置信度 + 上下文感知）"""
        if self.cache:
            cached = self.cache.get(f"smart_chat:{user_input}")
            if cached is not None:
                logger.info(f"smart_chat 缓存命中: {user_input[:40]}")
                return cached

        logger.info(f"smart_chat 路由: {user_input[:60]}")

        # ── 方法1: LLM 语义分类（主路由） ──
        route = self._llm_route(user_input)

        # ── 方法2: 关键词置信度校验（如果 LLM 路由结果是 chat，用关键词二次校验） ──
        if route == "chat":
            kw_route, confidence = self._keyword_route_with_score(user_input)
            if confidence >= 0.6:
                logger.info(f"smart_chat 关键词覆盖LLM路由: chat→{kw_route} (confidence={confidence})")
                route = kw_route

        # ── 执行路由 ──
        result = self._route_to_agent(route, user_input)

        logger.info(f"smart_chat 路由结果: {route}")
        if self.cache:
            self.cache.set(f"smart_chat:{user_input}", result)
        return result

    def _llm_route(self, user_input: str) -> str:
        """LLM 语义路由分类"""
        route_prompt = f"""你是一个电商助手路由分析器。判断用户输入属于哪个环节，输出对应的英文标签。

可选标签（只能选一个）：
- selector: 想卖什么/选什么产品好/市场趋势/什么好卖/爆款分析
- lister: 上架商品/写标题/写描述/SEO优化/商品详情
- service: 售前咨询/售后问题/退换货/物流/投诉/客服
- analyst: 数据分析/利润计算/定价建议/促销方案/经营分析
- sourcing: 找货源/找工厂/供应商/1688/采购
- chat: 打招呼/闲聊/非电商问题/其他

示例：
- "我想开个店卖什么好" → selector
- "帮我写个商品标题" → lister
- "顾客说收到的商品坏了" → service
- "帮我算算利润" → analyst
- "哪里能进到便宜的手机壳" → sourcing
- "你好" → chat

用户输入：{user_input}

只输出标签（selector/lister/service/analyst/sourcing/chat）："""
        route = simple_prompt(
            "你是一个精准的电商路由分类器，只输出一个标签词。",
            route_prompt,
            temperature=0.1,
        ).strip().lower()

        valid_routes = {"selector", "lister", "service", "analyst", "sourcing", "chat"}
        return route if route in valid_routes else "chat"

    def _keyword_route_with_score(self, text: str) -> tuple[str, float]:
        """关键词路由 + 置信度评分

        Returns:
            (route: str, confidence: float 0-1)
        """
        text_lower = text.lower()

        # 每个路由的关键词权重组
        routes_keywords = {
            "selector": [
                ("选品", 0.9), ("爆款", 0.9), ("卖什么", 0.9), ("趋势", 0.7),
                ("什么好卖", 0.9), ("热门", 0.6), ("选什么", 0.8), ("什么产品", 0.7),
                ("市场", 0.5), ("品类", 0.4), ("推荐", 0.3),
            ],
            "lister": [
                ("标题", 0.9), ("上架", 0.9), ("描述", 0.8), ("seo", 0.8),
                ("发布", 0.7), ("详情", 0.7), ("优化", 0.5), ("主图", 0.7),
                ("文案", 0.7), ("素材", 0.6),
            ],
            "service": [
                ("客服", 0.9), ("售后", 0.9), ("退换", 0.9), ("退款", 0.9),
                ("投诉", 0.9), ("咨询", 0.7), ("物流", 0.8), ("发货", 0.8),
                ("退货", 0.9), ("换货", 0.9), ("快递", 0.7),
            ],
            "analyst": [
                ("数据", 0.6), ("利润", 0.8), ("促销", 0.8), ("分析", 0.6),
                ("定价", 0.8), ("运营", 0.7), ("业绩", 0.8), ("成本", 0.6),
                ("毛利率", 0.9), ("转化率", 0.9), ("销售额", 0.8),
            ],
            "sourcing": [
                ("货源", 0.9), ("供应商", 0.9), ("采购", 0.8), ("1688", 0.9),
                ("进货", 0.8), ("工厂", 0.8), ("批发", 0.7), ("一手", 0.7),
            ],
        }

        best_route = "chat"
        best_score = 0.0

        for route, keywords in routes_keywords.items():
            score = 0.0
            for keyword, weight in keywords:
                if keyword in text_lower:
                    score += weight
            # 归一化
            if keywords:
                avg = score / len(keywords)
                if avg > best_score:
                    best_score = avg
                    best_route = route

        return best_route, min(best_score, 1.0)

    def _route_to_agent(self, route: str, user_input: str) -> str:
        """路由到指定 Agent 并返回结果"""
        if route == "chat":
            logger.info("smart_chat → chat (直接 LLM)")
            return simple_prompt(
                "你是一个专业的电商运营助手，回答用户的电商相关问题。",
                user_input,
            )
        agent = self.agents[route]
        logger.info(f"smart_chat → {route}")
        result = agent.run(user_input)
        content = result.get("report") or result.get("listing_content") or result.get("answer") or str(result)
        return content

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
        """一键完整工作流：选品 → 上架 → 客服 → 分析"""
        logger.info(f"完整工作流启动: category={category} budget={budget} audience={target_audience}")
        results = {"category": category}

        # Step 1: 选品
        logger.info("工作流 Step 1/4: 选品分析")
        selector_result = self.agents["selector"].run(
            category, budget=budget, target_audience=target_audience
        )
        results["selector"] = selector_result

        # Step 2: 上架（基于选品结果）
        logger.info("工作流 Step 2/4: 上架素材")
        listing_input = product_info or f"品类：{category}\n"
        if selector_result.get("recommendations"):
            rec = selector_result["recommendations"][0]
            listing_input += f"名称：{rec.get('name', '')}\n成本：{rec.get('cost', '')}\n目标价：{rec.get('avg_price', '')}"

        lister_result = self.agents["lister"].run(listing_input)
        results["lister"] = lister_result

        # Step 3: 客服
        logger.info("工作流 Step 3/4: 客服应答")
        service_welcome = self.agents["service"].run("你好，我想了解一下这个商品")
        results["service"] = service_welcome

        # Step 4: 运营建议
        logger.info("工作流 Step 4/4: 运营分析")
        analyst_result = self.agents["analyst"].run(f"新店铺启动，品类：{category}，预算：{budget or '未设置'}")
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
        """智能对话 — 让 LLM 判断路由到哪个 Agent（带缓存）"""
        # 缓存检查
        if self.cache:
            cached = self.cache.get(f"smart_chat:{user_input}")
            if cached is not None:
                logger.info(f"smart_chat 缓存命中: {user_input[:40]}")
                return cached

        logger.info(f"smart_chat 路由: {user_input[:60]}")
        route_prompt = f"""你是一个智能路由分析器。判断用户输入属于哪个电商运营环节，只输出一个词。可选的词只有以下6个，不能输出其他任何词：

- selector: 选品/市场分析/趋势/卖什么/产品选择
- lister: 上架/写标题/写描述/SEO/上下架/商品发布
- service: 客服/售后服务/退换货/售前咨询/投诉
- analyst: 运营分析/数据/促销/定价/利润分析/经营分析
- sourcing: 货源/供应商/1688/采购/工厂
- chat: 一般对话/打招呼/非电商相关问题

用户输入：{user_input}

只输出这6个词之一（selector/lister/service/analyst/sourcing/chat）："""
        route = simple_prompt(
            "你是一个精准的路由分析器，只输出这6个词之一：selector, lister, service, analyst, sourcing, chat",
            route_prompt,
            temperature=0.1,
        ).strip().lower()

        # 精确匹配
        valid_routes = {"selector", "lister", "service", "analyst", "sourcing", "chat"}

        if route in valid_routes:
            result = self._route_to_agent(route, user_input)
        else:
            # 模糊匹配：关键词兜底
            route = self._keyword_route(user_input)
            result = self._route_to_agent(route, user_input)

        logger.info(f"smart_chat 路由结果: {route}")
        if self.cache:
            self.cache.set(f"smart_chat:{user_input}", result)
        return result

    def _keyword_route(self, text: str) -> str:
        """关键字兜底路由"""
        text_lower = text.lower()
        # 选品相关
        if any(kw in text_lower for kw in ["选品", "爆款", "卖什么", "趋势", "什么好卖", "热门", "选什么"]):
            return "selector"
        # 上架相关
        if any(kw in text_lower for kw in ["标题", "上架", "描述", "seo", "发布", "详情"]):
            return "lister"
        # 客服相关
        if any(kw in text_lower for kw in ["客服", "售后", "退换", "退款", "投诉", "咨询", "物流", "发货"]):
            return "service"
        # 货源相关
        if any(kw in text_lower for kw in ["货源", "供应商", "采购", "1688", "进货", "工厂"]):
            return "sourcing"
        # 运营分析相关
        if any(kw in text_lower for kw in ["数据", "利润", "促销", "分析", "定价", "运营", "业绩"]):
            return "analyst"
        return "chat"

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

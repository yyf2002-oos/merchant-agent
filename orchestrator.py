"""Master Orchestrator — task routing, Agent orchestration, full workflow, context management"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

# ⚡ Register all tools (must import before Agent modules)
import core.tools_registry  # noqa: F401

from agents.selector import SelectorAgent
from agents.lister import ListerAgent
from agents.service import ServiceAgent
from agents.analyst import AnalystAgent
from agents.sourcing import SourcingAgent
from core.context import AgentContext
from core.memory import ConversationMemory
from llm import simple_prompt
from config import LOG_LEVEL, CACHE_ENABLED, CACHE_TTL, CACHE_CAPACITY
from cache import TTLCache

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL))


# ══════════════════════════════════════════════════
#  Cross-Agent Shared Context
# ══════════════════════════════════════════════════

@dataclass
class SharedContext:
    """Structured context shared across Agents, replacing plain dict"""
    # Basic info
    category: str = ""
    budget: str = ""
    target_audience: str = ""

    # Selection results
    selected_product: str = ""
    product_cost: str = ""
    product_price: str = ""
    competition_level: str = ""
    market_insights: list[str] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)

    # Full report summaries (for downstream Agents)
    selector_report: str = ""
    lister_content: str = ""

    # Decision trail
    decisions: list[str] = field(default_factory=list)

    def add_decision(self, decision: str):
        """Record a decision"""
        self.decisions.append(decision)
        logger.info(f"SharedContext decision: {decision}")

    def to_context_prompt(self) -> str:
        """Format as natural language context for Agent system prompts"""
        parts = ["[当前项目上下文]"]
        if self.category:
            parts.append(f"Category: {self.category}")
        if self.budget:
            parts.append(f"Budget: {self.budget}")
        if self.target_audience:
            parts.append(f"Target: {self.target_audience}")
        if self.selected_product:
            parts.append(f"Product: {self.selected_product}")
            if self.product_cost:
                parts.append(f"  Cost: ¥{self.product_cost}")
            if self.product_price:
                parts.append(f"  Price: ¥{self.product_price}")
        if self.competition_level:
            parts.append(f"Competition: {self.competition_level}")
        if self.decisions:
            parts.append("Decisions:")
            for d in self.decisions:
                parts.append(f"  • {d}")
        if self.market_insights:
            parts.append("Market Insights:")
            for ins in self.market_insights:
                parts.append(f"  • {ins}")
        if self.recommendations and len(self.recommendations) > 1:
            parts.append(f"Candidates: {len(self.recommendations)} products")
        return "\n".join(parts)


# ══════════════════════════════════════════════════
#  Orchestrator
# ══════════════════════════════════════════════════

class MerchantOrchestrator:
    """Merchant Agent Orchestrator"""

    def __init__(self):
        self.agents = {
            "selector": SelectorAgent(),
            "lister": ListerAgent(),
            "service": ServiceAgent(),
            "analyst": AnalystAgent(),
            "sourcing": SourcingAgent(),
        }
        self.cache = TTLCache(capacity=CACHE_CAPACITY, ttl_seconds=CACHE_TTL) if CACHE_ENABLED else None
        self._memory = ConversationMemory()
        logger.info(f"Orchestrator initialized, {len(self.agents)} Agents loaded cache={'enabled' if self.cache else 'disabled'}")

    def list_agents(self) -> list[dict]:
        return [
            {"key": k, "name": v.name, "desc": v.description}
            for k, v in self.agents.items()
        ]

    def run_agent(self, agent_key: str, input_data, session_id: str = None, **kwargs) -> dict:
        agent = self.agents.get(agent_key)
        if not agent:
            logger.error(f"Unknown Agent: {agent_key}")
            return {"error": f"Unknown Agent: {agent_key}"}
        logger.info(f"Run Agent: {agent_key} session={session_id} input={str(input_data)[:80]}")
        return agent.run(input_data, session_id=session_id, **kwargs)

    # ═══════════════════════════════════════════════
    #  Smart Chat (with context management)
    # ═══════════════════════════════════════════════

    def smart_chat(self, user_input: str, session_id: str = None) -> str:
        """Smart chat — multi-dimensional routing + context management

        Key differences from stateless version:
        - Loads history context (summary + recent messages + key info)
        - Injects history context into LLM/Agent calls
        - Persists current conversation to SQLite
        """
        if not session_id:
            session_id = f"chat_{uuid.uuid4().hex[:8]}"

        # Cache check (same question + same session)
        if self.cache:
            cached = self.cache.get(f"smart_chat:{session_id}:{user_input}")
            if cached is not None:
                logger.info(f"smart_chat cache hit: {user_input[:40]}")
                return cached

        logger.info(f"smart_chat route session={session_id[:12]}: {user_input[:60]}")

        # ── 构建上下文对象 ──
        ctx = AgentContext(session_id, self._memory)

        # ── Method 1: LLM semantic classification (primary) ──
        route = self._llm_route(user_input)

        # ── Method 2: Keyword confidence check ──
        if route == "chat":
            kw_route, confidence = self._keyword_route_with_score(user_input)
            if confidence >= 0.6:
                logger.info(f"smart_chat keyword overrides LLM route: chat→{kw_route} (confidence={confidence})")
                route = kw_route

        # ── Execute route (with context) ──
        result = self._route_to_agent_with_context(route, user_input, ctx)

        # ── Persist ──
        ctx.add_message("user", user_input)
        ctx.add_message("assistant", result)

        logger.info(f"smart_chat done route={route} session={session_id[:12]}")
        if self.cache:
            self.cache.set(f"smart_chat:{session_id}:{user_input}", result)
        return result

    def _route_to_agent_with_context(self, route: str, user_input: str, ctx: AgentContext) -> str:
        """Route to specified Agent with history context injection"""
        # Load compressed context
        compressed = ctx.get_compressed_context()
        context_prompt = ""
        if compressed["summary"]:
            context_prompt += f"\n[历史摘要]\n{compressed['summary']}"
        if compressed["key_info"]:
            context_prompt += f"\n[关键信息]\n{compressed['key_info']}"

        if route == "chat":
            logger.info(f"smart_chat → chat (direct LLM with context)")
            system = "你是一个专业的电商运营助手，回答用户的电商相关问题。"
            if context_prompt:
                system += f"\n\n当前对话上下文：\n{context_prompt}"
            return simple_prompt(system, user_input)

        agent = self.agents[route]
        logger.info(f"smart_chat → {route} (with context)")

        # Inject context into agent input
        enriched_input = user_input
        if context_prompt:
            enriched_input = f"{context_prompt}\n\n[Current Question]\n{user_input}"

        result = agent.run(enriched_input, session_id=ctx.session_id)
        content = (result.get("report")
                   or result.get("listing_content")
                   or result.get("answer")
                   or str(result))
        return content

    # ═══════════════════════════════════════════════
    #  Routing
    # ═══════════════════════════════════════════════

    def _llm_route(self, user_input: str) -> str:
        """LLM semantic routing"""
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

        # 清洗 LLM 输出
        route = route.strip("\"'`").strip()
        for candidate in re.findall(r"(selector|lister|service|analyst|sourcing|chat)", route):
            route = candidate
            break

        valid_routes = {"selector", "lister", "service", "analyst", "sourcing", "chat"}
        return route if route in valid_routes else "chat"

    def _keyword_route_with_score(self, text: str) -> tuple[str, float]:
        """Keyword routing with confidence score"""
        text_lower = text.lower()

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
            matched = 0
            for keyword, weight in keywords:
                if keyword in text_lower:
                    score += weight
                    matched += 1
            if matched > 0:
                avg = score / matched
                if avg > best_score:
                    best_score = avg
                    best_route = route

        return best_route, min(best_score, 1.0)

    # ═══════════════════════════════════════════════
    #  Full Workflow (with SharedContext)
    # ═══════════════════════════════════════════════

    def run_full_workflow(self, category: str, budget: str = "",
                          target_audience: str = "",
                          product_info: str = "",
                          session_id: str = None,
                          progress_callback=None) -> dict:
        """Full workflow: selector → lister → service → analyst (with cross-Agent SharedContext)

        Args:
            progress_callback: optional callback fn(step_name, status, detail) for progress feedback

        Key differences from old version:
        - Uses SharedContext dataclass for structured data passing
        - Passes all recommended products downstream
        - Selected market insight summaries for downstream reference
        - Explicit decision trail recording
        """
        if not session_id:
            session_id = f"workflow_{uuid.uuid4().hex[:8]}"
        if progress_callback:
            progress_callback("workflow", "start", f"Category: {category}")
        logger.info(f"Workflow start session={session_id[:12]}: category={category} budget={budget} audience={target_audience}")

        results = {"category": category, "session_id": session_id}
        shared = SharedContext(
            category=category,
            budget=budget,
            target_audience=target_audience,
        )

        # ── Step 1: Selector ──
        logger.info("Workflow Step 1/4: Selector")
        if progress_callback:
            progress_callback("selector", "running", f"Analyzing market for '{category}'...")
        shared.add_decision(f"Target category: {category}")
        if budget:
            shared.add_decision(f"Budget: {budget}")
        if target_audience:
            shared.add_decision(f"Target audience: {target_audience}")

        selector_result = self.agents["selector"].run(
            category, session_id=session_id, budget=budget, target_audience=target_audience,
        )
        results["selector"] = selector_result

        # Extract structured selection results
        if selector_result.get("recommendations"):
            shared.recommendations = selector_result["recommendations"][:5]
            best = shared.recommendations[0]
            shared.selected_product = best.get("name", "")
            shared.product_cost = str(best.get("cost", ""))
            shared.product_price = str(best.get("avg_price", ""))
            shared.competition_level = "高" if best.get("competition", 0) > 60 else "中" if best.get("competition", 0) > 30 else "低"
            shared.add_decision(f"Best pick: {shared.selected_product} (price={shared.product_price}, cost={shared.product_cost})")
            shared.add_decision(f"Competition: {shared.competition_level}")

            # Extract market insights from report
            report_text = selector_result.get("report", "")
            shared.selector_report = report_text
            insights = self._extract_insights(report_text)
            shared.market_insights = insights[:3]
        else:
            logger.warning("Workflow: selector 未返回推荐商品，SharedContext 将缺少选品数据")
            shared.add_decision("Selector 未返回结构化推荐数据")

        # ── Step 2: Lister ──
        logger.info("Workflow Step 2/4: Lister")
        if progress_callback:
            progress_callback("lister", "running", f"Generating listing for '{shared.selected_product or category}'...")
        listing_input = f"Category: {category}\n"
        if shared.selected_product:
            listing_input += (
                f"Name: {shared.selected_product}\n"
                f"Cost: {shared.product_cost}\n"
                f"Target price: {shared.product_price}\n"
            )
        # Inject SharedContext as additional context
        listing_input += f"\n[Selection Conclusions]\n{shared.to_context_prompt()}\n"

        lister_result = self.agents["lister"].run(listing_input, session_id=session_id)
        results["lister"] = lister_result
        if lister_result.get("listing_content"):
            shared.lister_content = lister_result["listing_content"]
            shared.add_decision("Listing content generated")

        # ── Step 3: Service ──
        logger.info("Workflow Step 3/4: Service")
        if progress_callback:
            progress_callback("service", "running", "Generating customer service scripts...")
        service_context = shared.to_context_prompt()
        service_welcome = self.agents["service"].run(
            "你好，我想了解一下这个商品",
            session_id=session_id,
            product_context=service_context,
        )
        results["service"] = service_welcome
        shared.add_decision("CS scripts generated")

        # ── Step 4: Analyst ──
        logger.info("Workflow Step 4/4: Analyst")
        if progress_callback:
            progress_callback("analyst", "running", "Calculating profit and devising operations plan...")
        analyst_input = (
            f"新店铺启动\n\n"
            f"{shared.to_context_prompt()}\n"
        )
        analyst_result = self.agents["analyst"].run(analyst_input, session_id=session_id)
        results["analyst"] = analyst_result
        shared.add_decision("Analyst report generated")

        results["shared_context"] = {
            "selected_product": shared.selected_product,
            "product_cost": shared.product_cost,
            "product_price": shared.product_price,
            "competition_level": shared.competition_level,
            "recommendations_count": len(shared.recommendations),
            "decisions": shared.decisions,
            "market_insights": shared.market_insights,
        }

        logger.info(f"Workflow done: {category} session={session_id[:12]}")
        if progress_callback:
            progress_callback("workflow", "done", "All steps complete")
        return results

    @staticmethod
    def _extract_insights(report: str) -> list[str]:
        """从选品报告中提取市场洞察摘要"""
        insights = []
        lines = report.split("\n")
        for line in lines:
            line = line.strip()
            # 提取带有具体数字的结论性语句
            if any(kw in line for kw in ["机会", "建议", "趋势", "蓝海", "红海", "差异化", "利润", "定价", "竞争"]):
                if any(c.isdigit() for c in line) and len(line) < 80:
                    insights.append(line.strip("*- \t"))
                    if len(insights) >= 5:
                        break
        if not insights:
            # 回退：取前 3 个非空行
            for line in lines[:20]:
                line = line.strip().strip("*- \t")
                if line and len(line) > 10 and not line.startswith("#"):
                    insights.append(line)
                    if len(insights) >= 3:
                        break
        return insights

    # ═══════════════════════════════════════════════
    #  Batch Listing
    # ═══════════════════════════════════════════════

    def batch_listing(self, products: list[dict]) -> list[dict]:
        """Batch listing — generate listing content for multiple products"""
        logger.info(f"Batch listing: {len(products)} products")
        results = self.agents["lister"].batch_run(products)
        logger.info(f"Batch listing done: {len(results)}/{len(products)}")
        return results

    def get_session_history(self, session_id: str, limit: int = 10) -> list[dict]:
        """Get conversation history for a session"""
        return self._memory.get_history(session_id, limit)

    def clear_session(self, session_id: str):
        """Clear session history"""
        ctx = AgentContext(session_id, self._memory)
        ctx.clear()
        logger.info(f"Cleared session: {session_id[:12]}")

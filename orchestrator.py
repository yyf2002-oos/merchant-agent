"""Master Orchestrator — 任务路由、Agent 编排、完整工作流、上下文管理"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

# ⚡ 注册所有工具（必须在导入 Agent 之前）
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
#  跨 Agent 共享上下文数据类
# ══════════════════════════════════════════════════

@dataclass
class SharedContext:
    """跨 Agent 共享的结构化上下文，替代普通 dict"""
    # 基础信息
    category: str = ""
    budget: str = ""
    target_audience: str = ""

    # 选品结果
    selected_product: str = ""
    product_cost: str = ""
    product_price: str = ""
    competition_level: str = ""
    market_insights: list[str] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)

    # 完整报告摘要（供下游 Agent 参考）
    selector_report: str = ""
    lister_content: str = ""

    # 决策记录（显式记录每个环节的关键决定）
    decisions: list[str] = field(default_factory=list)

    def add_decision(self, decision: str):
        """记录一个决策，带时间戳标识"""
        self.decisions.append(decision)
        logger.info(f"SharedContext 决策: {decision}")

    def to_context_prompt(self) -> str:
        """格式化为自然语言上下文，注入 Agent 的 system prompt"""
        parts = ["【当前项目上下文】"]
        if self.category:
            parts.append(f"品类: {self.category}")
        if self.budget:
            parts.append(f"预算: {self.budget}")
        if self.target_audience:
            parts.append(f"目标人群: {self.target_audience}")
        if self.selected_product:
            parts.append(f"已选商品: {self.selected_product}")
            if self.product_cost:
                parts.append(f"  成本: ¥{self.product_cost}")
            if self.product_price:
                parts.append(f"  售价: ¥{self.product_price}")
        if self.competition_level:
            parts.append(f"竞争度: {self.competition_level}")
        if self.decisions:
            parts.append("已做决策:")
            for d in self.decisions:
                parts.append(f"  • {d}")
        if self.market_insights:
            parts.append("市场洞察:")
            for ins in self.market_insights:
                parts.append(f"  • {ins}")
        if self.recommendations and len(self.recommendations) > 1:
            parts.append(f"候选商品共 {len(self.recommendations)} 个")
        return "\n".join(parts)


# ══════════════════════════════════════════════════
#  主控器
# ══════════════════════════════════════════════════

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
        self._memory = ConversationMemory()
        logger.info(f"Orchestrator 初始化完成，已加载 {len(self.agents)} 个 Agent 缓存={'启用' if self.cache else '禁用'}")

    def list_agents(self) -> list[dict]:
        return [
            {"key": k, "name": v.name, "desc": v.description}
            for k, v in self.agents.items()
        ]

    def run_agent(self, agent_key: str, input_data, session_id: str = None, **kwargs) -> dict:
        agent = self.agents.get(agent_key)
        if not agent:
            logger.error(f"未知 Agent: {agent_key}")
            return {"error": f"未知 Agent: {agent_key}"}
        logger.info(f"运行 Agent: {agent_key} session={session_id} input={str(input_data)[:80]}")
        return agent.run(input_data, session_id=session_id, **kwargs)

    # ═══════════════════════════════════════════════
    #  智能对话（带上下文管理）
    # ═══════════════════════════════════════════════

    def smart_chat(self, user_input: str, session_id: str = None) -> str:
        """智能对话 — 多维度路由 + 上下文管理

        与无状态版本的关键区别：
        - 加载历史上下文（摘要 + 近期消息 + 关键信息）
        - 把历史上下文注入到 LLM/Agent 调用
        - 保存本轮对话到 SQLite
        """
        if not session_id:
            session_id = f"chat_{uuid.uuid4().hex[:8]}"

        # 缓存检查（相同问题+相同 session 才命中）
        if self.cache:
            cached = self.cache.get(f"smart_chat:{session_id}:{user_input}")
            if cached is not None:
                logger.info(f"smart_chat 缓存命中: {user_input[:40]}")
                return cached

        logger.info(f"smart_chat 路由 session={session_id[:12]}: {user_input[:60]}")

        # ── 构建上下文对象 ──
        ctx = AgentContext(session_id, self._memory)

        # ── 方法1: LLM 语义分类（主路由） ──
        route = self._llm_route(user_input)

        # ── 方法2: 关键词置信度校验 ──
        if route == "chat":
            kw_route, confidence = self._keyword_route_with_score(user_input)
            if confidence >= 0.6:
                logger.info(f"smart_chat 关键词覆盖LLM路由: chat→{kw_route} (confidence={confidence})")
                route = kw_route

        # ── 执行路由（带上下文） ──
        result = self._route_to_agent_with_context(route, user_input, ctx)

        # ── 持久化 ──
        ctx.add_message("user", user_input)
        ctx.add_message("assistant", result)

        logger.info(f"smart_chat 完成 route={route} session={session_id[:12]}")
        if self.cache:
            self.cache.set(f"smart_chat:{session_id}:{user_input}", result)
        return result

    def _route_to_agent_with_context(self, route: str, user_input: str, ctx: AgentContext) -> str:
        """路由到指定 Agent，注入历史上下文"""
        # 加载压缩后的上下文
        compressed = ctx.get_compressed_context()
        context_prompt = ""
        if compressed["summary"]:
            context_prompt += f"\n【对话历史摘要】\n{compressed['summary']}"
        if compressed["key_info"]:
            context_prompt += f"\n【已知关键信息】\n{compressed['key_info']}"

        if route == "chat":
            logger.info(f"smart_chat → chat (直接 LLM 带上下文)")
            system = "你是一个专业的电商运营助手，回答用户的电商相关问题。"
            if context_prompt:
                system += f"\n\n以下是当前对话的上下文信息（参考用）：\n{context_prompt}"
            return simple_prompt(system, user_input)

        agent = self.agents[route]
        logger.info(f"smart_chat → {route} (带上下文)")

        # 把上下文注入到 agent 的输入中
        enriched_input = user_input
        if context_prompt:
            enriched_input = f"{context_prompt}\n\n【当前问题】\n{user_input}"

        result = agent.run(enriched_input, session_id=ctx.session_id)
        content = (result.get("report")
                   or result.get("listing_content")
                   or result.get("answer")
                   or str(result))
        return content

    # ═══════════════════════════════════════════════
    #  路由逻辑（与原来一致）
    # ═══════════════════════════════════════════════

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

        # 清洗 LLM 输出
        route = route.strip("\"'`").strip()
        for candidate in re.findall(r"(selector|lister|service|analyst|sourcing|chat)", route):
            route = candidate
            break

        valid_routes = {"selector", "lister", "service", "analyst", "sourcing", "chat"}
        return route if route in valid_routes else "chat"

    def _keyword_route_with_score(self, text: str) -> tuple[str, float]:
        """关键词路由 + 置信度评分"""
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
    #  一键完整工作流（带 SharedContext）
    # ═══════════════════════════════════════════════

    def run_full_workflow(self, category: str, budget: str = "",
                          target_audience: str = "",
                          product_info: str = "",
                          session_id: str = None,
                          progress_callback=None) -> dict:
        """一键完整工作流：选品 → 上架 → 客服 → 分析（带结构化跨 Agent 上下文）

        Args:
            progress_callback: 可选回调 fn(step_name, status, detail)，用于进度反馈

        与旧版的关键区别：
        - 使用 SharedContext 数据类传递结构化数据
        - 所有推荐商品传给下游，不只有第一个
        - 精选市场洞察摘要供下游参考
        - 显式记录决策链
        """
        if not session_id:
            session_id = f"workflow_{uuid.uuid4().hex[:8]}"
        if progress_callback:
            progress_callback("workflow", "start", f"品类：{category}")
        logger.info(f"完整工作流启动 session={session_id[:12]}: category={category} budget={budget} audience={target_audience}")

        results = {"category": category, "session_id": session_id}
        shared = SharedContext(
            category=category,
            budget=budget,
            target_audience=target_audience,
        )

        # ── Step 1: 选品 ──────────────────────────
        logger.info("工作流 Step 1/4: 选品分析")
        if progress_callback:
            progress_callback("selector", "running", f"正在分析「{category}」市场趋势...")
        shared.add_decision(f"目标品类：{category}")
        if budget:
            shared.add_decision(f"启动预算：{budget}元")
        if target_audience:
            shared.add_decision(f"目标人群：{target_audience}")

        selector_result = self.agents["selector"].run(
            category, session_id=session_id, budget=budget, target_audience=target_audience,
        )
        results["selector"] = selector_result

        # 提取结构化的选品结果
        if selector_result.get("recommendations"):
            shared.recommendations = selector_result["recommendations"][:5]
            best = shared.recommendations[0]
            shared.selected_product = best.get("name", "")
            shared.product_cost = str(best.get("cost", ""))
            shared.product_price = str(best.get("avg_price", ""))
            shared.competition_level = "高" if best.get("competition", 0) > 60 else "中" if best.get("competition", 0) > 30 else "低"
            shared.add_decision(f"首选商品：{shared.selected_product}（建议售价¥{shared.product_price}，成本¥{shared.product_cost}）")
            shared.add_decision(f"竞争度评估：{shared.competition_level}")

            # 提取市场洞察（从 report 中自动摘要）
            report_text = selector_result.get("report", "")
            shared.selector_report = report_text
            insights = self._extract_insights(report_text)
            shared.market_insights = insights[:3]

        # ── Step 2: 上架 ──────────────────────────
        logger.info("工作流 Step 2/4: 上架素材")
        if progress_callback:
            progress_callback("lister", "running", f"正在为「{shared.selected_product or category}」生成上架素材...")
        listing_input = f"品类：{category}\n"
        if shared.selected_product:
            listing_input += (
                f"名称：{shared.selected_product}\n"
                f"成本：{shared.product_cost}\n"
                f"目标价：{shared.product_price}\n"
            )
        # 注入 SharedContext 作为额外背景
        listing_input += f"\n【选品阶段结论】\n{shared.to_context_prompt()}\n"

        lister_result = self.agents["lister"].run(listing_input, session_id=session_id)
        results["lister"] = lister_result
        if lister_result.get("listing_content"):
            shared.lister_content = lister_result["listing_content"]
            shared.add_decision("上架素材已生成")

        # ── Step 3: 客服 ──────────────────────────
        logger.info("工作流 Step 3/4: 客服应答")
        if progress_callback:
            progress_callback("service", "running", "正在生成常见客服问答话术...")
        service_context = shared.to_context_prompt()
        service_welcome = self.agents["service"].run(
            "你好，我想了解一下这个商品",
            session_id=session_id,
            product_context=service_context,
        )
        results["service"] = service_welcome
        shared.add_decision("客服应答话术已生成")

        # ── Step 4: 运营分析 ──────────────────────
        logger.info("工作流 Step 4/4: 运营分析")
        if progress_callback:
            progress_callback("analyst", "running", "正在计算利润和制定运营方案...")
        analyst_input = (
            f"新店铺启动\n\n"
            f"{shared.to_context_prompt()}\n"
        )
        analyst_result = self.agents["analyst"].run(analyst_input, session_id=session_id)
        results["analyst"] = analyst_result
        shared.add_decision("运营分析报告已生成")

        results["shared_context"] = {
            "selected_product": shared.selected_product,
            "product_cost": shared.product_cost,
            "product_price": shared.product_price,
            "competition_level": shared.competition_level,
            "recommendations_count": len(shared.recommendations),
            "decisions": shared.decisions,
            "market_insights": shared.market_insights,
        }

        logger.info(f"完整工作流完成: {category} session={session_id[:12]}")
        if progress_callback:
            progress_callback("workflow", "done", "全流程完成")
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
    #  批量上架
    # ═══════════════════════════════════════════════

    def batch_listing(self, products: list[dict]) -> list[dict]:
        """批量上架 — 一次生成多个商品的上架素材"""
        logger.info(f"批量上架: {len(products)} 个商品")
        results = self.agents["lister"].batch_run(products)
        logger.info(f"批量上架完成: {len(results)}/{len(products)}")
        return results

    def get_session_history(self, session_id: str, limit: int = 10) -> list[dict]:
        """获取指定 session 的对话历史"""
        return self._memory.get_history(session_id, limit)

    def clear_session(self, session_id: str):
        """清除指定 session 的历史"""
        ctx = AgentContext(session_id, self._memory)
        ctx.clear()
        logger.info(f"已清除 session: {session_id[:12]}")

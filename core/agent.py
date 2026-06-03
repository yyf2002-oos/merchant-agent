"""ReAct Agent — Plan-Execute-Reflect loop with context management"""

import json
import logging
import uuid
from typing import Any, Optional, Literal

from config import OLLAMA_MODEL, OLLAMA_FAST_MODEL, OLLAMA_BASE, LOG_LEVEL, REACT_MAX_ROUNDS
from core.tool import get_all_definitions, execute
from core.memory import ConversationMemory
from core.context import AgentContext
from agents.base import BaseAgent
from llm import call_llm

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL))


class ReActAgent(BaseAgent):
    """Self-driven Agent: ReAct loop + native Function Calling + Plan & Reflect

    Modes:
    - react:        standard ReAct loop (backward compatible)
    - plan_execute: generate plan first, then execute
    - plan_reflect: plan + execute + reflect, re-execute if quality not met
    """

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tools: list[str] = None,
        use_memory: bool = False,
        model: str = OLLAMA_MODEL,
        light_model: str = "",
        temperature: float = 0.7,
    ):
        super().__init__(name, description)
        self.system_prompt = system_prompt
        self.allowed_tools = tools
        self.model = model
        self.light_model = light_model or OLLAMA_FAST_MODEL
        self.temperature = temperature

        # Fallback: if light model uses Ollama but Ollama is unavailable, fall back to main model
        if self.light_model and self.model:
            from llm import _parse_model_spec
            l_prov, _ = _parse_model_spec(self.light_model)
            m_prov, _ = _parse_model_spec(self.model)
            if l_prov == "ollama" and m_prov == "deepseek":
                import httpx
                try:
                    httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
                except Exception:
                    self.light_model = self.model
                    logger.warning(f"Ollama unavailable, light model fallback to main: {self.model}")
        self.memory = ConversationMemory() if use_memory else None
        logger.info(f"Agent[{name}] initialized tools={tools} use_memory={use_memory} model={model}")

    def _get_tool_defs(self) -> list[dict]:
        all_defs = get_all_definitions()
        if self.allowed_tools is None:
            return []  # None → 不允许使用任何工具（纯文本生成）
        if not self.allowed_tools:
            return all_defs  # 空列表 → 使用全部工具
        # 有具体列表 → 只使用列表中的工具
        selected = [d for d in all_defs if d["function"]["name"] in self.allowed_tools]
        if not selected:
            logger.warning(f"Agent[{self.name}] allowed_tools 中无匹配工具: {self.allowed_tools}")
        return selected

    def _call_llm(self, messages: list[dict], tools: list[dict] = None, model: str = None) -> dict:
        """Unified LLM call (auto Ollama/DeepSeek + fallback), returns full response message"""
        used_model = model or self.model
        logger.debug(f"Agent[{self.name}] call model={used_model} tools={len(tools) if tools else 0}")

        result = call_llm(
            messages, model=used_model, temperature=self.temperature,
            tools=tools, agent=self.name, session_id=getattr(self, '_session_id', ''),
        )
        return result

    def _execute_single_tool(self, tc: dict, log_prefix: str = "") -> tuple[str, bool]:
        """Execute a single tool call (with retry), returns (result_str, has_error)"""
        func_name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"]
        func_args = raw_args
        if isinstance(raw_args, str):
            try:
                func_args = json.loads(raw_args)
            except json.JSONDecodeError:
                func_args = {}
                logger.warning(f"Agent[{self.name}] 工具 {func_name} 参数 JSON 解析失败，使用空参数: {raw_args[:100]}")

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                result = execute(func_name, func_args)
                result_str = json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, (dict, list)) else str(result)

                # Check if result contains error
                if "[错误]" in result_str or "[超时]" in result_str:
                    if attempt < max_retries:
                        logger.warning(f"Agent[{self.name}] tool={func_name} returned error, retrying ({attempt+1}/{max_retries})")
                        continue
                    return result_str, True
                return result_str, False
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Agent[{self.name}] tool={func_name} exception, retrying: {e}")
                    continue
                return f"[错误] 工具 {func_name} 执行失败: {e}", True

        return "[错误] 达到最大重试次数", True

    # ── 规划 ──────────────────────────────────────

    def _generate_plan(self, user_input: str, tool_defs: list[dict]) -> str:
        """生成执行计划"""
        tool_names = [d["function"]["name"] for d in tool_defs]
        prompt = f"""你是一个电商运营专家。在开始执行之前，请先制定一个清晰的执行计划。

用户需求: {user_input}

可用工具: {', '.join(tool_names)}

请按以下格式输出计划：
## 目标
（一句话说明要完成什么）

## 执行步骤
1. 第一步：调用什么工具 → 期望得到什么结果
2. 第二步：调用什么工具 → 期望得到什么结果
...

## 预期产出
（最终要输出的内容结构）

注意：只输出计划本身，不要开始执行。"""
        msg = self._call_llm([
            {"role": "system", "content": "你是一个擅长制定执行计划的电商专家。只输出计划，不执行。"},
            {"role": "user", "content": prompt},
        ], model=self.light_model)
        return msg.get("content", "").strip()

    # ── 反思 ──────────────────────────────────────

    def _reflect(self, user_input: str, full_messages: list[dict],
                 tool_results: list[str] = None) -> tuple[bool, str]:
        """深度反思执行结果，判断质量和完整性

        Returns:
            (needs_more: bool, reflection: str)
        """
        recent = full_messages[-8:] if len(full_messages) > 8 else full_messages
        conv_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: "
            f"{m.get('content', '') or '(调用工具)'}"[:300]
            for m in recent
        )

        tool_check = ""
        if tool_results:
            errors = [r for r in tool_results if "[错误]" in r or "[超时]" in r or "失败" in r]
            if errors:
                tool_check = f"\n⚠️ 工具调用有错误：{len(errors)} 个\n" + "\n".join(e[:100] for e in errors[:3])

        prompt = f"""请严格评估以下执行结果的质量。

原始需求: {user_input}

执行过程:
{conv_text}
{tool_check}

评估标准（每一项都要严格打分）：
1. [完整性] 是否完整回答了用户所有问题？(是/否)
2. [工具使用] 是否调用了所有必要的工具获取了真实数据？(是/否)
3. [准确性] 结果中有没有矛盾、错误、或编造的数据？(是/否)
4. [可执行性] 给出的建议是否具体可执行，而不是空泛的套话？(是/否)
5. [数据支撑] 重要结论是否有数据支撑，还是全靠推测？(是/否)

评分规则：
- 5项全是"是"  → ✅ 完成（输出这一行即可）
- 有1项"否"   → 输出具体哪项不达标 + 需要补充什么
- 有2项以上"否" → 输出 ❌ 需要重做 + 具体原因"""
        msg = self._call_llm([
            {"role": "system", "content": "你是严格的质量检查官。只根据证据判断，不放过任何质量问题。"},
            {"role": "user", "content": prompt},
        ], model=self.light_model)
        reflection = msg.get("content", "").strip()

        # 检查 reflection 是否表示"已完成"（鲁棒匹配）
        first_line = reflection.strip().split("\n")[0] if reflection else ""
        # 支持多种完成标记：✅ / 完成 / all passed / 5项全是"是"
        done_keywords = ["✅", "完成", "all passed", "5项全是"]
        needs_more = not (
            first_line.startswith("✅")
            or "✅ 完成" in reflection
            or any(kw in reflection[:30] for kw in ["all passed", "全部完成", "5项全是"])
        )
        return needs_more, reflection

    # ── 主入口 ────────────────────────────────────

    def run(
        self,
        input_data: Any,
        session_id: str = None,
        max_rounds: int = None,
        mode: Literal["react", "plan_execute", "plan_reflect"] = "plan_reflect",
        **kwargs,
    ) -> dict:
        """Execute Agent task

        Args:
            input_data: user input
            session_id: session ID (auto-generates UUID if None)
            max_rounds: max tool call rounds (default REACT_MAX_ROUNDS)
            mode:
                react: standard ReAct
                plan_execute: plan + execute
                plan_reflect: plan + execute + reflect (default)

        Returns:
            {"agent": name, "report": content, "tool_calls": count}
        """
        if max_rounds is None:
            max_rounds = REACT_MAX_ROUNDS
        # Auto-generate session_id to avoid global "default" conflicts
        if not session_id:
            session_id = f"{self.name}_{uuid.uuid4().hex[:8]}"
        self._session_id = session_id
        user_input = self._format_input(input_data)
        tool_defs = self._get_tool_defs()
        logger.info(f"Agent[{self.name}] start session={session_id} mode={mode} max_rounds={max_rounds} tools={len(tool_defs)}")

        # ── Build context ──
        ctx = None
        if self.memory:
            ctx = AgentContext(session_id, self.memory)

        messages = []
        if ctx:
            messages = ctx.get_formatted_context(self.system_prompt)
        else:
            messages = [{"role": "system", "content": self.system_prompt}]

        messages.append({"role": "user", "content": user_input})

        # ── Plan phase ──
        plan_text = ""
        if mode in ("plan_execute", "plan_reflect"):
            plan_text = self._generate_plan(user_input, tool_defs)
            messages.insert(-1, {
                "role": "system",
                "content": f"【执行计划】\n{plan_text}\n\n请严格按照计划执行。",
            })

        # ── Execute phase (ReAct loop + self-correction) ──
        tool_call_count = 0
        final_content = None
        tool_results_log = []

        try:
            for _round in range(max_rounds):
                msg = self._call_llm(messages, tool_defs)

                if msg.get("content"):
                    final_content = msg["content"]

                tool_calls = msg.get("tool_calls", [])
                if not tool_calls:
                    break

                messages.append(msg)
                tool_has_error = False

                for tc in tool_calls:
                    tool_call_count += 1
                    result_str, has_err = self._execute_single_tool(tc)
                    tool_results_log.append(result_str[:200])
                    messages.append({
                        "role": "tool",
                        "content": result_str[:2000],
                        "tool_call_id": tc.get("id", ""),
                    })
                    # Persist tool results to context
                    if ctx and result_str and "[错误]" not in result_str[:20]:
                        func_name = tc["function"]["name"]
                        ctx.add_tool_result(func_name, result_str[:200])
                    if has_err:
                        tool_has_error = True

                # ── On error, prompt Agent to correct ──
                if tool_has_error and _round < max_rounds - 1:
                    messages.append({
                        "role": "system",
                        "content": "⚠️ 上一步工具调用返回了错误，请检查参数后重新尝试，或换一个工具继续。不要重复同样的错误调用。",
                    })

            # ── Reflect phase ──
            reflection = ""
            if mode == "plan_reflect" and tool_call_count > 0:
                needs_more, reflection = self._reflect(user_input, messages, tool_results_log)
                if needs_more and max_rounds > tool_call_count + 3:
                    messages.append({
                        "role": "system",
                        "content": f"【反思反馈】\n{reflection}\n\n请根据以上反思补充或修正你的回答。如果反思认为需要重做，请重新调用工具获取正确数据。",
                    })
                    for _round2 in range(max_rounds - tool_call_count):
                        msg2 = self._call_llm(messages, tool_defs)
                        if msg2.get("content"):
                            final_content = msg2["content"]
                        tc2 = msg2.get("tool_calls", [])
                        if not tc2:
                            break
                        messages.append(msg2)
                        for tc in tc2:
                            tool_call_count += 1
                            result_str, _ = self._execute_single_tool(tc)
                            tool_results_log.append(result_str[:200])
                            messages.append({
                                "role": "tool",
                                "content": result_str[:2000],
                                "tool_call_id": tc.get("id", ""),
                            })

            logger.info(f"Agent[{self.name}] done tool_calls={tool_call_count} mode={mode}")
        except Exception as e:
            logger.error(f"Agent[{self.name}] execution error: {e}", exc_info=True)
            if not final_content:
                final_content = f"[Agent执行错误] {e}"

        # ── Save to memory ──
        if ctx:
            ctx.add_message("user", user_input)
            if final_content:
                ctx.add_message("assistant", final_content)

        return {
            "agent": self.name,
            "report": final_content or "(no output)",
            "tool_calls": tool_call_count,
            "plan": plan_text if mode in ("plan_execute", "plan_reflect") else "",
            "reflection": reflection if mode == "plan_reflect" else "",
            "session_id": session_id,
        }

    def _format_input(self, input_data: Any) -> str:
        if isinstance(input_data, str):
            return input_data
        if isinstance(input_data, dict):
            parts = []
            for k, v in input_data.items():
                if v:
                    parts.append(f"{k}: {v}")
            return "\n".join(parts) if parts else str(input_data)
        return str(input_data)

    def save_context(self, key: str, value: str):
        if self.memory:
            self.memory.remember(f"agent:{self.name}:{key}", value)

    def load_context(self, key: str) -> Optional[str]:
        if self.memory:
            return self.memory.recall(f"agent:{self.name}:{key}")
        return None

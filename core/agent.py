"""ReAct Agent — 支持 Plan-Execute-Reflect 循环 + 上下文管理"""

import json
import logging
from typing import Any, Optional, Literal

from config import OLLAMA_MODEL, OLLAMA_FAST_MODEL, LOG_LEVEL, REACT_MAX_ROUNDS
from core.tool import get_all_definitions, execute
from core.memory import ConversationMemory
from core.context import AgentContext
from llm import call_llm

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL))


class ReActAgent:
    """自驱型 Agent：ReAct 循环 + 原生 Function Calling + 规划反思

    运行模式:
    - react:        标准 ReAct 循环（向后兼容）
    - plan_execute: 执行前先生成计划，再按计划执行
    - plan_reflect: 计划+执行+反思，结果不符预期可重新执行
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
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.allowed_tools = tools
        self.model = model
        self.light_model = light_model or OLLAMA_FAST_MODEL
        self.temperature = temperature
        self.memory = ConversationMemory() if use_memory else None
        logger.info(f"Agent[{name}] 初始化 tools={tools} use_memory={use_memory} model={model}")

    def _get_tool_defs(self) -> list[dict]:
        all_defs = get_all_definitions()
        if not self.allowed_tools:
            return all_defs
        return [d for d in all_defs if d["function"]["name"] in self.allowed_tools]

    def _call_llm(self, messages: list[dict], tools: list[dict] = None, model: str = None) -> dict:
        """统一 LLM 调用（自动选择 Ollama/DeepSeek），返回完整响应消息"""
        used_model = model or self.model
        logger.debug(f"Agent[{self.name}] 调用 model={used_model} tools={len(tools) if tools else 0}")

        result = call_llm(messages, model=used_model, temperature=self.temperature, tools=tools)
        return result

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

        needs_more = not reflection.startswith("✅")
        return needs_more, reflection

    # ── 主入口 ────────────────────────────────────

    def run(
        self,
        input_data: Any,
        session_id: str = "default",
        max_rounds: int = None,
        mode: Literal["react", "plan_execute", "plan_reflect"] = "plan_reflect",
        **kwargs,
    ) -> dict:
        """执行 Agent 任务

        Args:
            input_data: 用户输入
            session_id: 会话 ID
            max_rounds: 最大工具调用轮次（默认 REACT_MAX_ROUNDS）
            mode: 运行模式
                react: 标准 ReAct
                plan_execute: 计划+执行
                plan_reflect: 计划+执行+反思（默认）

        Returns:
            {"agent": name, "report": content, "tool_calls": count}
        """
        if max_rounds is None:
            max_rounds = REACT_MAX_ROUNDS
        user_input = self._format_input(input_data)
        tool_defs = self._get_tool_defs()
        logger.info(f"Agent[{self.name}] 开始运行 mode={mode} max_rounds={max_rounds} tools={len(tool_defs)}")

        # ── 构建上下文 ──
        ctx = None
        if self.memory:
            ctx = AgentContext(session_id, self.memory)

        messages = []
        if ctx:
            messages = ctx.get_formatted_context(self.system_prompt)
        else:
            messages = [{"role": "system", "content": self.system_prompt}]

        messages.append({"role": "user", "content": user_input})

        # ── Plan 阶段 ──
        plan_text = ""
        if mode in ("plan_execute", "plan_reflect"):
            plan_text = self._generate_plan(user_input, tool_defs)
            messages.insert(-1, {
                "role": "system",
                "content": f"【执行计划】\n{plan_text}\n\n请严格按照计划执行。",
            })

        # ── Execute 阶段 (ReAct 循环 + 自纠错) ──
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
                    func_name = tc["function"]["name"]
                    parsed = tc["function"].get("_parsed")
                    func_args = parsed if parsed is not None else tc["function"]["arguments"]
                    tc["function"].pop("_parsed", None)
                    tool_call_count += 1

                    # ── 执行工具，含自纠错 ──
                    max_tool_retries = 2
                    last_error = None
                    for tool_try in range(max_tool_retries + 1):
                        try:
                            result = execute(func_name, func_args)
                            if isinstance(result, (dict, list)):
                                result_str = json.dumps(result, ensure_ascii=False, indent=2)
                            else:
                                result_str = str(result)

                            # 检查结果是否包含错误
                            if "[错误]" in result_str or "[超时]" in result_str:
                                last_error = result_str
                                if tool_try < max_tool_retries:
                                    logger.warning(f"Agent[{self.name}] 工具{func_name}返回错误，重试中({tool_try+1}/{max_tool_retries})")
                                    continue
                                tool_has_error = True
                            break
                        except Exception as e:
                            last_error = str(e)
                            if tool_try < max_tool_retries:
                                logger.warning(f"Agent[{self.name}] 工具{func_name}异常，重试中: {e}")
                                continue
                            result_str = f"[错误] 工具 {func_name} 执行失败: {e}"

                    if last_error and tool_try < max_tool_retries:
                        # 自纠错成功（重试后正常），记录但不影响流程
                        pass

                    tool_results_log.append(result_str[:200])
                    messages.append({
                        "role": "tool",
                        "content": result_str[:2000],
                        "tool_call_id": tc.get("id", ""),
                    })

                # ── 本轮有错误时，提示 Agent 修正 ──
                if tool_has_error and _round < max_rounds - 1:
                    messages.append({
                        "role": "system",
                        "content": "⚠️ 上一步工具调用返回了错误，请检查参数后重新尝试，或换一个工具继续。不要重复同样的错误调用。",
                    })

            # ── Reflect 阶段（深度反思） ──
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
                            func_name = tc["function"]["name"]
                            parsed = tc["function"].get("_parsed")
                            func_args = parsed if parsed is not None else tc["function"]["arguments"]
                            tc["function"].pop("_parsed", None)
                            tool_call_count += 1

                            for tool_try in range(2):
                                try:
                                    result = execute(func_name, func_args)
                                    if isinstance(result, (dict, list)):
                                        result_str = json.dumps(result, ensure_ascii=False, indent=2)
                                    else:
                                        result_str = str(result)
                                    break
                                except Exception as e:
                                    result_str = f"[错误] {e}"
                                    if tool_try < 1:
                                        continue
                                    break

                            messages.append({
                                "role": "tool",
                                "content": result_str[:2000],
                                "tool_call_id": tc.get("id", ""),
                            })

            logger.info(f"Agent[{self.name}] 执行完成 tool_calls={tool_call_count} mode={mode}")
        except Exception as e:
            logger.error(f"Agent[{self.name}] 执行异常: {e}", exc_info=True)
            if not final_content:
                final_content = f"[Agent 执行异常] {e}"

        # ── 保存到记忆 ──
        if ctx:
            ctx.add_message("user", user_input)
            if final_content:
                ctx.add_message("assistant", final_content)

        return {
            "agent": self.name,
            "report": final_content or "（无输出）",
            "tool_calls": tool_call_count,
            "plan": plan_text if mode in ("plan_execute", "plan_reflect") else "",
            "reflection": reflection if mode == "plan_reflect" else "",
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

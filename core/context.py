"""Context Manager — 对话上下文管理：摘要压缩 + 滑动窗口 + 关键信息提取"""

import json
import time
from typing import Optional

from core.memory import ConversationMemory
from llm import simple_prompt

# 滑动窗口：保留最近 N 条原始消息
RAW_WINDOW_SIZE = 8
# 触发摘要的消息总数阈值
SUMMARIZE_THRESHOLD = 20


class AgentContext:
    """管理 Agent 的对话上下文

    职责：
    - 维护近期消息（原始） + 历史摘要（压缩）
    - 自动提取关键信息（用户偏好、产品信息、决策记录）
    - 提供构建 prompt 用的压缩上下文
    """

    def __init__(self, session_id: str, memory: Optional[ConversationMemory] = None):
        self.session_id = session_id
        self.memory = memory or ConversationMemory()
        self._summarized = False  # 当前会话是否已触发过摘要

    # ── 消息管理 ──────────────────────────────────

    def add_message(self, role: str, content: str):
        self.memory.add_message(self.session_id, role, content)

    def get_history(self, limit: int = 30) -> list[dict]:
        return self.memory.get_history(self.session_id, limit)

    # ── 摘要压缩 ──────────────────────────────────

    def _build_summary(self, messages: list[dict]) -> str:
        """对一段消息生成摘要"""
        text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:200]}"
            for m in messages
        )
        prompt = f"""请总结以下对话的核心内容（50字以内），包括：
1. 用户的目标/需求
2. 已经做过的决策
3. 已经获得的关键信息

对话内容：
{text}

总结："""
        result = simple_prompt("你是一个简洁的对话总结助手，只输出总结本身。", prompt, temperature=0.3)
        return result.strip() or "(摘要生成失败)"

    def get_compressed_context(self, recent_count: int = RAW_WINDOW_SIZE) -> dict:
        """获取压缩后的上下文

        Returns:
            {"summary": str,    # 历史摘要（压缩后的历史）
             "key_info": str,   # 提取的关键信息
             "recent": list,    # 最近 N 条原始消息
             "total": int}      # 总消息数
        """
        all_messages = self.memory.get_history(self.session_id, limit=200)
        total = len(all_messages)

        # 拆分为历史 + 近期
        if total > recent_count:
            history = all_messages[: total - recent_count]
            recent = all_messages[total - recent_count :]
        else:
            history = []
            recent = all_messages

        # 生成摘要
        summary = ""
        if len(history) >= 4:  # 至少 4 条才值得摘要
            summary = self._build_summary(history)
            self._summarized = True
        elif self._summarized:
            summary = self.memory.recall(f"ctx:{self.session_id}:summary") or ""
        else:
            summary = ""

        # 提取关键信息
        key_info = self._extract_key_info(all_messages)

        return {
            "summary": summary,
            "key_info": key_info,
            "recent": recent[-recent_count:],
            "total": total,
        }

    def get_formatted_context(self, system_prompt: str) -> list[dict]:
        """构建完整的 messages 列表给 LLM

        Args:
            system_prompt: Agent 的系统提示词

        Returns:
            messages: 可直接传给 LLM 的消息列表
        """
        ctx = self.get_compressed_context()
        messages = [{"role": "system", "content": system_prompt}]

        # 如果有摘要，注入作为 system 级别的上下文
        ctx_parts = []
        if ctx["summary"]:
            ctx_parts.append(f"【历史摘要】{ctx['summary']}")
        if ctx["key_info"]:
            ctx_parts.append(f"【已知信息】{ctx['key_info']}")

        if ctx_parts:
            messages.append({
                "role": "system",
                "content": "\n\n".join(ctx_parts),
            })

        # 追加近期消息
        for m in ctx["recent"]:
            messages.append({"role": m["role"], "content": m["content"]})

        return messages

    # ── 关键信息提取 ──────────────────────────────

    def _extract_key_info(self, messages: list[dict]) -> str:
        """从对话中提取关键信息"""
        # 优先用缓存的
        cached = self.memory.recall(f"ctx:{self.session_id}:key_info")

        # 如果有新消息（上次缓存后新增 > 5 条），重新提取
        last_count = self.memory.recall(f"ctx:{self.session_id}:msg_count")
        if last_count and int(last_count) >= len(messages) - 5:
            return cached or ""

        # 提取关键信息
        text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:150]}"
            for m in messages[-20:]
        )
        prompt = f"""从以下对话中提取关键信息（30字以内）：
- 用户的品类/商品方向
- 预算范围
- 目标人群
- 其他重要信息

如果没有明显的关键信息，输出"暂无"。

对话：
{text}

关键信息："""
        result = simple_prompt("你是一个信息提取助手，只输出提取结果。", prompt, temperature=0.3)
        info = result.strip() or "暂无"

        # 缓存
        self.memory.remember(f"ctx:{self.session_id}:key_info", info)
        self.memory.remember(f"ctx:{self.session_id}:msg_count", str(len(messages)))

        return info

    # ── 清理 ──────────────────────────────────────

    def clear(self):
        self.memory.clear_session(self.session_id)
        self._summarized = False

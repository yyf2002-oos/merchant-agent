"""Context Manager — 对话上下文管理：摘要压缩 + 滑动窗口 + 关键信息提取"""

import json
import time
from typing import Optional

from core.memory import ConversationMemory
from llm import simple_prompt

# 滑动窗口：保留最近 N 条原始消息
RAW_WINDOW_SIZE = 12
# 触发摘要的消息总数阈值
SUMMARIZE_THRESHOLD = 16

class AgentContext:
    """管理 Agent 的对话上下文

    职责：
    - 维护近期消息（原始） + 历史摘要（压缩）
    - 自动提取关键信息（用户偏好、产品信息、决策记录）
    - 提供构建 prompt 用的压缩上下文
    - 持久化工具有价值的结果
    """

    def __init__(self, session_id: str, memory: Optional[ConversationMemory] = None):
        self.session_id = session_id
        self.memory = memory or ConversationMemory()

    def add_message(self, role: str, content: str):
        self.memory.add_message(self.session_id, role, content)

    def get_history(self, limit: int = 50) -> list[dict]:
        return self.memory.get_history(self.session_id, limit)

    def add_tool_result(self, tool_name: str, result_summary: str):
        """持久化工具调用结果到 context，供后续轮次使用"""
        entry = f"[工具:{tool_name}] {result_summary[:200]}"
        self.memory.add_message(self.session_id, "system", entry)

    def _build_summary(self, messages: list[dict]) -> str:
        """对一段消息生成结构化摘要（200 字以内，含关键字段）"""
        text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m.get('content', '')[:200]}"
            for m in messages[-30:]
        )
        prompt = f"""请总结以下对话，按固定格式输出（200字以内）：

## 用户目标/需求
（用户想做什么、品类方向）

## 已做决策
（已经确定的选择，如价格区间、目标人群、产品方向）

## 关键信息
（获得的真实数据：价格、搜索量、竞争度、利润等具体数字）

## 待办/未完成
（用户还没决定的事、需要后续跟进的点）

对话内容：
{text}

按以上格式输出总结："""
        result = simple_prompt("你是一个简洁的对话总结助手，按固定格式输出。", prompt, temperature=0.3)
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

        # 生成摘要（缓存到 SQLite 避免重复计算）
        summary = ""
        cached_summary = self.memory.recall(f"ctx:{self.session_id}:summary")
        # 历史消息至少 3 条（约 1-2 轮对话）就触发摘要
        summary_threshold = max(3, recent_count // 4)
        if len(history) >= summary_threshold:
            last_hist_count = self.memory.recall(f"ctx:{self.session_id}:hist_count")
            if last_hist_count and int(last_hist_count) >= len(history) - 3 and cached_summary:
                summary = cached_summary
            else:
                summary = self._build_summary(history)
                self.memory.remember(f"ctx:{self.session_id}:summary", summary)
                self.memory.remember(f"ctx:{self.session_id}:hist_count", str(len(history)))
        elif cached_summary:
            summary = cached_summary

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

        Returns:
            messages: 可直接传给 LLM 的消息列表
        """
        ctx = self.get_compressed_context()
        messages = [{"role": "system", "content": system_prompt}]

        # 如果有摘要，注入作为 system 级别的上下文
        ctx_parts = []
        if ctx["summary"]:
            ctx_parts.append(f"【历史摘要】\n{ctx['summary']}")
        if ctx["key_info"]:
            ctx_parts.append(f"【已知信息】\n{ctx['key_info']}")

        if ctx_parts:
            messages.append({
                "role": "system",
                "content": "\n\n".join(ctx_parts),
            })

        # 追加近期消息（只取 user/assistant，过滤 system 工具消息避免膨胀）
        for m in ctx["recent"]:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})

        return messages

    def _extract_key_info(self, messages: list[dict]) -> str:
        """从对话中提取结构化关键信息"""
        # 优先用缓存的
        cached = self.memory.recall(f"ctx:{self.session_id}:key_info")

        # 如果有新消息（上次缓存后新增 > 5 条），重新提取
        last_count = self.memory.recall(f"ctx:{self.session_id}:msg_count")
        if last_count and int(last_count) >= len(messages) - 5 and cached:
            return cached

        # 提取关键信息
        text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m.get('content', '')[:200]}"
            for m in messages[-20:]
        )
        prompt = f"""从以下对话中提取关键信息（100字以内），按固定格式输出：

- 品类/商品：
- 预算范围：
- 目标人群：
- 已选产品及价格：
- 其他重要信息：
（没有的项填"暂无"，只输出关键信息，不要分析）

对话：
{text}

关键信息："""
        result = simple_prompt("你是一个信息提取助手，只输出提取结果。", prompt, temperature=0.3)
        info = result.strip() or "暂无"

        # 缓存
        self.memory.remember(f"ctx:{self.session_id}:key_info", info)
        self.memory.remember(f"ctx:{self.session_id}:msg_count", str(len(messages)))

        return info

    def clear(self):
        self.memory.clear_session(self.session_id)
        self.memory.remember(f"ctx:{self.session_id}:summary", "")
        self.memory.remember(f"ctx:{self.session_id}:key_info", "")
        self.memory.remember(f"ctx:{self.session_id}:msg_count", "0")
        self.memory.remember(f"ctx:{self.session_id}:hist_count", "0")

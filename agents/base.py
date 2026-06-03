"""Agent 基类 — 所有 Agent 的通用接口"""

from abc import ABC, abstractmethod
from typing import Any

from llm import simple_prompt


class BaseAgent(ABC):
    """Agent 抽象基类 — 所有 Agent 继承此接口"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> dict:
        """执行 Agent 任务"""
        ...

    def get_system_prompt(self) -> str:
        """Agent 系统提示词"""
        return f"你是一个专业的电商{self.name}助手。请根据用户的输入提供专业、详细的建议。"

    def llm_call(self, system: str, user: str, **kwargs) -> str:
        """快捷调用 LLM"""
        return simple_prompt(system, user, **kwargs)

    def __str__(self):
        return f"{self.name} ({self.description})"

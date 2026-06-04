"""Tool registry — @tool decorator + runtime execution"""

import inspect
import logging
import re
import time
from typing import Any, Callable

from config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL))

_registry: dict[str, dict] = {}

def tool(name: str = None, description: str = None):
    """Register a function as a callable tool for LLM agents"""
    def decorator(func: Callable):
        tool_name = name or func.__name__
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        properties = {}
        required = []

        # Parse parameter descriptions from docstring
        param_descs = {}
        in_params = False
        for line in doc.split("\n"):
            stripped = line.strip()
            if stripped.startswith("Args:") or stripped.startswith("Parameters:") or stripped.startswith("Params:"):
                in_params = True
                continue
            if in_params:
                # Match lines like "param_name: description" or "param_name (type): description"
                m = re.match(r'^(\w+)\s*(?:\([^)]*\))?:\s*(.*)', stripped)
                if m and not stripped.startswith("Returns:") and not stripped.startswith("Raises:"):
                    param_descs[m.group(1)] = m.group(2).strip()
                elif not stripped or stripped.startswith("Returns:") or stripped.startswith("Raises:"):
                    in_params = False

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            # Map Python types to JSON schema types
            type_map = {int: "integer", float: "number", bool: "boolean", str: "string"}
            param_type = type_map.get(param.annotation, "string")

            # Get param description from docstring, fallback to param name
            desc = param_descs.get(param_name, f"Parameter {param_name}")

            properties[param_name] = {
                "type": param_type,
                "description": desc,
            }
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        _registry[tool_name] = {
            "function": func,
            "definition": {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description or func.__doc__ or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            },
        }
        return func
    return decorator

def get_definition(name: str) -> dict:
    """Get tool definition by name"""
    entry = _registry.get(name)
    if not entry:
        raise KeyError(f"Tool '{name}' not found")
    return entry["definition"]

def get_all_definitions() -> list[dict]:
    """Get ALL registered tool definitions in OpenAI format"""
    return [entry["definition"] for entry in _registry.values()]

def execute(name: str, arguments: dict) -> Any:
    """Execute a tool by name with arguments"""
    entry = _registry.get(name)
    if not entry:
        logger.error(f"Unknown tool: {name}")
        return f"[工具错误] 未知工具: {name}"
    logger.info(f"Execute tool: {name} args={arguments}")
    try:
        t0 = time.time()
        result = entry["function"](**arguments)
        elapsed = time.time() - t0
        logger.info(f"Tool done: {name} elapsed={elapsed:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Tool error: {name}({arguments}): {e}")
        return f"[Tool Error] {name}({arguments}): {e}"

def list_tools() -> list[str]:
    """List all registered tool names"""
    return list(_registry.keys())

def clear():
    """Clear registry (for testing)"""
    _registry.clear()

"""LLM 调用封装 — 支持 Ollama + DeepSeek，流式输出、工具调用、日志"""

import json
import time
import logging
import httpx
from typing import Optional

from config import (
    OLLAMA_BASE, OLLAMA_MODEL, OLLAMA_FAST_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL,
    LLM_PROVIDER, AGENT_TIMEOUT, MAX_RETRIES, LOG_LEVEL,
)

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL))


# ====================================================================
#  DeepSeek (OpenAI 兼容格式) 调用
# ====================================================================

def _call_deepseek(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7,
    tools: list[dict] = None,
) -> dict:
    """调用 DeepSeek API（OpenAI 兼容格式），返回完整响应"""
    if not DEEPSEEK_API_KEY:
        return {"role": "assistant", "content": "[错误] DEEPSEEK_API_KEY 未配置"}

    used_model = model or DEEPSEEK_MODEL
    url = f"{DEEPSEEK_API_BASE}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": used_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 8192,
    }
    if tools:
        payload["tools"] = tools

    logger.info(f"DeepSeek 调用 model={used_model} temperature={temperature} msg_count={len(messages)} tools={len(tools) if tools else 0}")

    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.time()
            with httpx.Client(timeout=AGENT_TIMEOUT) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            elapsed = time.time() - t0

            choice = data["choices"][0]
            msg = choice["message"]

            # 转换为 Ollama 兼容格式
            result = {
                "role": "assistant",
                "content": msg.get("content") or "",
            }

            # DeepSeek 的工具调用是 OpenAI 格式
            if msg.get("tool_calls"):
                result["tool_calls"] = []
                for tc in msg["tool_calls"]:
                    fn = tc["function"]
                    # arguments 是字符串，需要解析
                    try:
                        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                    except json.JSONDecodeError:
                        args = {}
                    result["tool_calls"].append({
                        "function": {
                            "name": fn["name"],
                            "arguments": args,
                        }
                    })

            logger.info(f"DeepSeek 返回 attempt={attempt+1} 耗时={elapsed:.1f}s"
                        f" tool_calls={bool(result.get('tool_calls'))} len={len(result.get('content',''))}")
            return result

        except httpx.TimeoutException:
            logger.warning(f"DeepSeek 超时 attempt={attempt+1}/{MAX_RETRIES+1}")
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return {"role": "assistant", "content": f"[超时] DeepSeek 调用超时（{AGENT_TIMEOUT}s）"}
        except Exception as e:
            logger.error(f"DeepSeek 错误 attempt={attempt+1}/{MAX_RETRIES+1}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return {"role": "assistant", "content": f"[错误] {e}"}


def _call_deepseek_text(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7,
) -> Optional[str]:
    """调用 DeepSeek 返回文本"""
    result = _call_deepseek(messages, model, temperature)
    return result.get("content")


# ====================================================================
#  Ollama 调用
# ====================================================================

def _call_ollama(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7,
) -> dict:
    """调用 Ollama，返回完整响应消息（含 tool_calls）"""
    used_model = model or OLLAMA_MODEL
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model": used_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    logger.info(f"Ollama 调用 model={used_model} temperature={temperature} msg_count={len(messages)}")

    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.time()
            with httpx.Client(timeout=AGENT_TIMEOUT) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            elapsed = time.time() - t0
            result = data["message"]
            logger.info(f"Ollama 返回 attempt={attempt+1} 耗时={elapsed:.1f}s"
                        f" tool_calls={bool(result.get('tool_calls'))} len={len(result.get('content',''))}")
            return result
        except httpx.TimeoutException:
            logger.warning(f"Ollama 超时 attempt={attempt+1}/{MAX_RETRIES+1}")
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return {"role": "assistant", "content": f"[超时] Ollama 调用超时（{AGENT_TIMEOUT}s）"}
        except Exception as e:
            logger.error(f"Ollama 错误 attempt={attempt+1}/{MAX_RETRIES+1}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return {"role": "assistant", "content": f"[错误] {e}"}


def _call_ollama_text(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7,
) -> Optional[str]:
    """调用 Ollama 返回文本"""
    result = _call_ollama(messages, model, temperature)
    return result.get("content")


# ====================================================================
#  统一接口（自动选择 Provider）
# ====================================================================

def call_llm(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7,
    tools: list[dict] = None,
) -> dict:
    """统一 LLM 调用，根据 LLM_PROVIDER 自动选择 Ollama / DeepSeek

    Returns:
        统一格式: {"role": "assistant", "content": str, "tool_calls": [...]}
        tool_calls 格式: [{"function": {"name": str, "arguments": dict}}]
    """
    provider = LLM_PROVIDER
    if provider == "deepseek":
        return _call_deepseek(messages, model, temperature, tools)
    else:
        # Ollama 模式下把 OpenAI 格式 tools 转成 Ollama 格式
        return _call_ollama(messages, model, temperature)


def chat(messages: list[dict], **kwargs) -> str:
    """普通对话"""
    result = call_llm(messages, **kwargs)
    return result.get("content", "")


def simple_prompt(system: str, user: str, **kwargs) -> str:
    """一行调用的快捷方式"""
    return chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], **kwargs)


# ====================================================================
#  健康检查
# ====================================================================

def check_llm() -> tuple[bool, str]:
    """检查当前 LLM 提供者是否可用"""
    provider = LLM_PROVIDER
    if provider == "deepseek":
        if not DEEPSEEK_API_KEY:
            return False, "DEEPSEEK_API_KEY 未配置"
        # 简单测试：发一个空请求
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(f"{DEEPSEEK_API_BASE}/v1/models", headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                })
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
                    model_names = [m["id"] for m in models[:3]]
                    return True, f"DeepSeek 已连接 | 可用模型: {', '.join(model_names)}"
                return False, f"DeepSeek API 返回 {resp.status_code}"
        except Exception as e:
            return False, f"DeepSeek 连接失败: {e}"
    else:
        # Ollama 检查
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{OLLAMA_BASE}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    model_names = [m["name"] for m in models[:4]]
                    return True, f"Ollama 已连接 | 模型: {', '.join(model_names)}"
                return False, "Ollama 未响应"
        except Exception as e:
            return False, f"Ollama 连接失败: {e}"


def check_ollama() -> bool:
    """兼容旧接口"""
    ok, _ = check_llm()
    return ok


def list_models() -> list[str]:
    """兼容旧接口 — 只返回 Ollama 模型"""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{OLLAMA_BASE}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


# ====================================================================
#  流式对话
# ====================================================================

def stream_chat(messages: list[dict], model: str = None):
    """流式对话 — 生成器（当前仅支持 Ollama）"""
    if LLM_PROVIDER == "deepseek":
        yield "[提示] DeepSeek 流式模式当前仅支持 WebUI 内部使用"
        return

    used_model = model or OLLAMA_MODEL
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model": used_model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.7},
    }
    with httpx.Client(timeout=AGENT_TIMEOUT) as client:
        with client.stream("POST", url, json=payload) as resp:
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if content := chunk.get("message", {}).get("content"):
                            yield content
                    except json.JSONDecodeError:
                        continue

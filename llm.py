"""Ollama LLM 调用封装 — 支持流式输出、错误重试、日志"""

import json
import time
import logging
import httpx
from typing import Optional

from config import OLLAMA_BASE, OLLAMA_MODEL, AGENT_TIMEOUT, MAX_RETRIES, LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL))


def _call_ollama(
    messages: list[dict],
    model: str = OLLAMA_MODEL,
    stream: bool = False,
    temperature: float = 0.7,
) -> Optional[str]:
    """调用 Ollama 模型，带重试"""
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": temperature},
    }

    last_model = model or OLLAMA_MODEL
    logger.info(f"LLM 调用 model={last_model} temperature={temperature} msg_count={len(messages)}")

    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.time()
            with httpx.Client(timeout=AGENT_TIMEOUT) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            elapsed = time.time() - t0
            content = data["message"]["content"]
            logger.info(f"LLM 返回成功 attempt={attempt+1} 耗时={elapsed:.1f}s 输出长度={len(content)}")
            return content
        except httpx.TimeoutException:
            elapsed = time.time() - t0 if 't0' in locals() else 0
            logger.warning(f"LLM 超时 attempt={attempt+1}/{MAX_RETRIES+1} 耗时={elapsed:.1f}s")
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return f"[超时] LLM 调用超时（{AGENT_TIMEOUT}s）"
        except Exception as e:
            elapsed = time.time() - t0 if 't0' in locals() else 0
            logger.error(f"LLM 错误 attempt={attempt+1}/{MAX_RETRIES+1} 耗时={elapsed:.1f}s 错误={e}")
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return f"[错误] {str(e)}"


def chat(messages: list[dict], **kwargs) -> str:
    """普通对话"""
    return _call_ollama(messages, **kwargs)


def simple_prompt(system: str, user: str, **kwargs) -> str:
    """一行调用的快捷方式"""
    return _call_ollama([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], **kwargs)


def check_ollama() -> bool:
    """检查 Ollama 是否在线"""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{OLLAMA_BASE}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    """列出 Ollama 可用模型"""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{OLLAMA_BASE}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def stream_chat(messages: list[dict], model: str = OLLAMA_MODEL):
    """流式对话 — 生成器"""
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model": model,
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

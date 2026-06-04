"""LLM invocation layer — Ollama + DeepSeek, streaming, tool calls, logging & fallback"""

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

_http_client = httpx.Client(timeout=AGENT_TIMEOUT)

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

    logger.info(f"DeepSeek call model={used_model} temperature={temperature} msg_count={len(messages)} tools={len(tools) if tools else 0}")

    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.time()
            resp = _http_client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - t0

            choice = data["choices"][0]
            msg = choice["message"]

            # Convert to Ollama-compatible format
            result = {
                "role": "assistant",
                "content": msg.get("content") or "",
            }

            # DeepSeek tool calls use OpenAI format
            if msg.get("tool_calls"):
                result["tool_calls"] = []
                for tc in msg["tool_calls"]:
                    fn = tc["function"]
                    # OpenAI format requires arguments as JSON string
                    args_str = fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"], ensure_ascii=False)
                    result["tool_calls"].append({
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": fn["name"],
                            "arguments": args_str,
                        }
                    })

            logger.info(f"DeepSeek response attempt={attempt+1} elapsed={elapsed:.1f}s"
                        f" tool_calls={bool(result.get('tool_calls'))} len={len(result.get('content',''))}")
            return result

        except httpx.TimeoutException:
            logger.warning(f"DeepSeek timeout attempt={attempt+1}/{MAX_RETRIES+1}")
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return {"role": "assistant", "content": f"[超时] DeepSeek 调用超时（{AGENT_TIMEOUT}s）"}
        except Exception as e:
            logger.error(f"DeepSeek error attempt={attempt+1}/{MAX_RETRIES+1}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return {"role": "assistant", "content": f"[错误] {e}"}

def _call_deepseek_text(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7,
) -> Optional[str]:
    """Call DeepSeek and return text content"""
    result = _call_deepseek(messages, model, temperature)
    return result.get("content")

def _call_ollama(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7,
    tools: list[dict] = None,
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
    if tools:
        payload["tools"] = tools

    logger.info(f"Ollama call model={used_model} temperature={temperature} msg_count={len(messages)} tools={len(tools) if tools else 0}")

    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.time()
            resp = _http_client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - t0
            result = data["message"]
            logger.info(f"Ollama response attempt={attempt+1} elapsed={elapsed:.1f}s"
                        f" tool_calls={bool(result.get('tool_calls'))} len={len(result.get('content',''))}")
            return result
        except httpx.TimeoutException:
            logger.warning(f"Ollama timeout attempt={attempt+1}/{MAX_RETRIES+1}")
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return {"role": "assistant", "content": f"[超时] Ollama 调用超时（{AGENT_TIMEOUT}s）"}
        except Exception as e:
            logger.error(f"Ollama error attempt={attempt+1}/{MAX_RETRIES+1}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return {"role": "assistant", "content": f"[错误] {e}"}

def _call_ollama_text(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7,
) -> Optional[str]:
    """Call Ollama and return text content"""
    result = _call_ollama(messages, model, temperature)
    return result.get("content")

def _parse_model_spec(model_spec: str) -> tuple[str | None, str | None]:
    """Parse model name prefix, returns (provider, actual_model)

    Supported formats:
        "deepseek:deepseek-chat"  → ("deepseek", "deepseek-chat")
        "ollama:qwen2:7b"         → ("ollama", "qwen2:7b")
        "deepseek-chat"           → (None, "deepseek-chat")  no prefix → use global LLM_PROVIDER
        None                      → (None, None)
    """
    if not model_spec:
        return None, None
    if ":" in model_spec and model_spec.split(":")[0] in ("ollama", "deepseek"):
        provider, *rest = model_spec.split(":", 1)
        return provider, rest[0] if rest else None
    return None, model_spec

def call_llm(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7,
    tools: list[dict] = None,
    agent: str = "",
    session_id: str = "",
) -> dict:
    """Unified LLM call, auto-selects Ollama/DeepSeek based on model name prefix

    Supports auto-fallback: switches to backup provider when primary fails.

    Model name format:
        "deepseek:deepseek-chat"  → DeepSeek
        "ollama:qwen2:7b"         → Ollama
        "qwen2:7b" / None         → global LLM_PROVIDER

    Returns:
        Unified format: {"role": "assistant", "content": str, "tool_calls": [...]}
        tool_calls format: [{"function": {"name": str, "arguments": dict}}]
    """
    return call_llm_with_fallback(
        messages, model=model, temperature=temperature,
        tools=tools, agent=agent, session_id=session_id,
    )

def chat(messages: list[dict], **kwargs) -> str:
    """Simple chat"""
    result = call_llm(messages, **kwargs)
    return result.get("content", "")

def simple_prompt(system: str, user: str, **kwargs) -> str:
    """One-line convenience for system+user prompt"""
    return chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], **kwargs)

def _call_deepseek_stream(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7,
    tools: list[dict] = None,
):
    """Streaming DeepSeek API call (generator), yields content chunks"""
    if not DEEPSEEK_API_KEY:
        yield "[错误] DEEPSEEK_API_KEY 未配置"
        return

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
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["stream"] = False  # 工具调用模式不支持流式
        yield _call_deepseek(messages, model, temperature, tools).get("content", "")
        return

    try:
        with _http_client.stream("POST", url, json=payload, headers=headers) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if content := delta.get("content"):
                            yield content
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"[流式错误] {e}"

def call_llm_with_fallback(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7,
    tools: list[dict] = None,
    agent: str = "",
    session_id: str = "",
) -> dict:
    """Call LLM with auto-fallback when primary provider fails

    Fallback strategy:
    - deepseek timeout/error → auto retry ollama
    - ollama timeout/error → auto retry deepseek
    - both fail → return error message
    """
    provider, actual_model = _parse_model_spec(model)
    if provider is None:
        provider = LLM_PROVIDER

    if provider == "deepseek":
        primary_provider = "deepseek"
        fallback_provider = "ollama"
        primary_model = actual_model or DEEPSEEK_MODEL
        fallback_model = OLLAMA_MODEL
    else:
        primary_provider = "ollama"
        fallback_provider = "deepseek"
        primary_model = actual_model or OLLAMA_MODEL
        fallback_model = DEEPSEEK_MODEL

    t0 = time.time()

    for attempt in range(MAX_RETRIES + 1):
        try:
            if primary_provider == "deepseek":
                result = _call_deepseek(messages, primary_model, temperature, tools)
            else:
                result = _call_ollama(messages, primary_model, temperature, tools)

            elapsed = int((time.time() - t0) * 1000)
            content = result.get("content", "")
            is_error = content.startswith("[错误]") or content.startswith("[超时]")

            try:
                from monitor import record_call
                record_call(
                    provider=primary_provider,
                    model=primary_model,
                    duration_ms=elapsed,
                    success=not is_error,
                    error=content if is_error else "",
                    agent=agent,
                    session_id=session_id,
                )
            except Exception:
                pass

            if not is_error:
                return result

            logger.warning(f"{primary_provider} attempt={attempt+1} failed, preparing fallback: {content[:60]}")
            break  # Primary provider explicitly failed, stop retrying

        except Exception as e:
            elapsed = int((time.time() - t0) * 1000)
            logger.warning(f"{primary_provider} exception attempt={attempt+1}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            try:
                from monitor import record_call
                record_call(provider=primary_provider, model=primary_model, duration_ms=elapsed, success=False, error=str(e), agent=agent, session_id=session_id)
            except Exception:
                pass

    logger.info(f"Falling back to {fallback_provider}")
    t1 = time.time()
    try:
        if fallback_provider == "deepseek":
            result = _call_deepseek(messages, fallback_model, temperature, tools)
        else:
            result = _call_ollama(messages, fallback_model, temperature, tools)

        elapsed = int((time.time() - t1) * 1000)
        content = result.get("content", "")
        is_error = content.startswith("[错误]") or content.startswith("[超时]")

        try:
            from monitor import record_call
            record_call(provider=fallback_provider, model=fallback_model, duration_ms=elapsed, success=not is_error, error=content if is_error else "", agent=agent, session_id=session_id)
        except Exception:
            pass

        if not is_error:
            return result
        return {"role": "assistant", "content": f"[错误] 主({primary_provider})和备用({fallback_provider})均失败，请检查配置"}
    except Exception as e:
        logger.error(f"Fallback to {fallback_provider} also failed: {e}")
        return {"role": "assistant", "content": f"[错误] 主({primary_provider})和备用({fallback_provider})均异常: {e}"}

def check_llm() -> tuple[bool, str]:
    """Check if the current LLM provider is available"""
    provider = LLM_PROVIDER
    if provider == "deepseek":
        if not DEEPSEEK_API_KEY:
            return False, "DEEPSEEK_API_KEY not configured"
        # Simple test: list models
        try:
            resp = _http_client.get(f"{DEEPSEEK_API_BASE}/v1/models", headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            })
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                model_names = [m["id"] for m in models[:3]]
                return True, f"DeepSeek connected | models: {', '.join(model_names)}"
            return False, f"DeepSeek API returned {resp.status_code}"
        except Exception as e:
            return False, f"DeepSeek connection failed: {e}"
    else:
        # Ollama check
        try:
            resp = _http_client.get(f"{OLLAMA_BASE}/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m["name"] for m in models[:4]]
                return True, f"Ollama connected | models: {', '.join(model_names)}"
            return False, "Ollama not responding"
        except Exception as e:
            return False, f"Ollama connection failed: {e}"

def check_ollama() -> bool:
    """Backward-compatible alias for check_llm"""
    ok, _ = check_llm()
    return ok

def list_models() -> list[str]:
    """List available Ollama models"""
    try:
        resp = _http_client.get(f"{OLLAMA_BASE}/api/tags")
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []

def stream_chat(messages: list[dict], model: str = None):
    """Streaming chat — generator (supports DeepSeek and Ollama)"""
    provider, actual_model = _parse_model_spec(model)
    if provider is None:
        provider = LLM_PROVIDER

    if provider == "deepseek":
        yield from _call_deepseek_stream(messages, actual_model)
        return

    used_model = actual_model or OLLAMA_MODEL
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model": used_model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.7},
    }
    try:
        with _http_client.stream("POST", url, json=payload) as resp:
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if content := chunk.get("message", {}).get("content"):
                            yield content
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"[流式错误] {e}"

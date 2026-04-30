"""LLM and traced runtime call helpers."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

from paradigm_experiments.observability.langsmith import traceable_run


def emit_model_raw(
    emit: Callable[[str], None],
    tag: str,
    raw: Optional[str],
    max_chars: int = 6000,
) -> None:
    """Emit raw model output line-by-line for parse failure debugging."""
    value = raw if isinstance(raw, str) else ""
    emit(f"{tag} 原始输出长度={len(value)}")
    if not value:
        emit(f"{tag} 原始内容: (空字符串；多为 API 报错/限流/模型不可用导致 llm 返回空)")
        return
    preview = value if len(value) <= max_chars else value[:max_chars] + f"\n…(截断，总长度 {len(value)} 字符)"
    emit(f"{tag} 原始内容（逐行）:")
    for idx, line in enumerate(preview.splitlines(), 1):
        emit(f"  {idx:04d}| {line}")
    if len(value) > max_chars:
        emit(f"{tag} …已截断展示前 {max_chars} 字符")


def build_client(api_key_env: str):
    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL", "https://api.qnaigc.com/v1")
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"{api_key_env} is required")
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


@traceable_run("idea3.llm.act", run_type="llm")
def llm(
    prompt: str,
    client: Any,
    model: str,
    max_tokens: int,
    langsmith_extra: Optional[Dict[str, Any]] = None,
) -> str:
    del langsmith_extra
    for attempt in range(3):
        try:
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            return ""


@traceable_run("idea3.llm.think_or_reflect", run_type="llm")
def llm2(
    prompt: str,
    client: Any,
    model: str,
    max_tokens: int,
    langsmith_extra: Optional[Dict[str, Any]] = None,
) -> str:
    del langsmith_extra
    for attempt in range(2):
        try:
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception:
            if attempt == 0:
                time.sleep(2)
                continue
            return ""


@traceable_run("alfworld.env.step", run_type="tool")
def traced_env_step(
    env,
    action: str,
    langsmith_extra: Optional[Dict[str, Any]] = None,
):
    del langsmith_extra
    return env.step([action])

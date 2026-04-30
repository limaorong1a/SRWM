"""Action selection helpers for the Idea3 ALFWorld agent."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from paradigm_experiments.observability.langsmith import traceable_run
from paradigm_experiments.runtime.llm import llm


def extract_json_candidate(text: str) -> Optional[str]:
    value = (text or "").strip()
    if not value:
        return None
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", value)
        value = re.sub(r"\n?```$", "", value).strip()
    if value.startswith("{") and value.endswith("}"):
        return value
    match = re.search(r"\{[\s\S]*\}", value)
    if match:
        return match.group(0)
    return None


def parse_action_text(raw: str, admissible: List[str]) -> str:
    """Return the exact admissible command selected by a model response."""
    value = (raw or "").strip()
    if not value:
        return ""
    candidates: List[str] = []
    json_candidate = extract_json_candidate(value)
    if json_candidate:
        try:
            obj = json.loads(json_candidate)
            if isinstance(obj, dict):
                index = obj.get("act_index")
                if isinstance(index, int) and 0 <= index < len(admissible):
                    return admissible[index]
                action = obj.get("act") or obj.get("action") or obj.get("command")
                if isinstance(action, str):
                    candidates.append(action.strip())
        except Exception:
            pass

    for line in value.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("act:"):
            candidates.append(stripped[4:].strip())
        else:
            candidates.append(stripped)

    seen = set()
    normalized: List[str] = []
    for candidate in candidates:
        candidate = candidate.strip().strip("`\"'")
        if " | " in candidate:
            candidate = candidate.split(" | ", 1)[0].strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            normalized.append(candidate)

    admissible_set = set(admissible)
    for candidate in normalized:
        if candidate in admissible_set:
            return candidate
    return ""


def parse_score_array(raw: str, expected_len: int) -> List[float]:
    """Parse a model-produced JSON score array and clamp values to [0, 100]."""
    if expected_len <= 0:
        return []
    value = (raw or "").strip()
    if not value:
        return [0.0] * expected_len
    array_text = None
    if value.startswith("[") and value.endswith("]"):
        array_text = value
    else:
        match = re.search(r"\[[\s\S]*\]", value)
        if match:
            array_text = match.group(0)
    if array_text is None:
        return [0.0] * expected_len
    try:
        scores = json.loads(array_text)
    except Exception:
        return [0.0] * expected_len
    if not isinstance(scores, list):
        return [0.0] * expected_len

    result: List[float] = []
    for score in scores[:expected_len]:
        try:
            value_float = float(score)
        except Exception:
            value_float = 0.0
        result.append(min(100.0, max(0.0, value_float)))
    if len(result) < expected_len:
        result.extend([0.0] * (expected_len - len(result)))
    return result


def build_action_candidates(admissible: List[str], scores: List[float]) -> List[Dict[str, Any]]:
    return [
        {
            "index": index,
            "cmd": action,
            "score": round(scores[index] if index < len(scores) else 0.0, 1),
        }
        for index, action in enumerate(admissible)
    ]


@traceable_run("idea3.judge.score_actions", run_type="llm")
def score_actions_with_judge(
    actions: List[str],
    goal: str,
    context: str,
    client: Any,
    model: str,
    max_tokens: int,
    langsmith_extra: Optional[Dict[str, Any]] = None,
) -> List[float]:
    if not actions:
        return []
    actions_block = "\n".join([f"{idx + 1}. {action}" for idx, action in enumerate(actions)])
    prompt = (
        "你是动作评判师。给你最终目标和上文内容，根据生活经验为候选动作列表的每个动作打分。\n"
        f"最终目标: {goal}\n"
        "上文内容:\n"
        f"{context}\n"
        "候选动作列表:\n"
        f"{actions_block}\n"
        "请为每个动作打分，0到100之间的整数，分数越高越接近最终目标。\n"
        "只输出JSON数组，长度与动作数一致，例如: [80, 20, 60]\n"
    )
    output = llm(prompt, client=client, model=model, max_tokens=max_tokens, langsmith_extra=langsmith_extra)
    if not output:
        return [0.0] * len(actions)
    return parse_score_array(output, expected_len=len(actions))

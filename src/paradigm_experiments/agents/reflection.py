"""Failure reflection payload builders for ALFWorld trajectories."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Protocol, Tuple

from paradigm_experiments.observability.langsmith import traceable_run
from paradigm_experiments.runtime.llm import llm2


class TrajectoryRecord(Protocol):
    kind: str
    text: str


def is_completion_claim(think_text: str) -> bool:
    value = (think_text or "").strip().lower()
    if not value:
        return False
    positive_markers = (
        "任务完成",
        "已经完成",
        "已完成",
        "应该完成",
        "看起来完成",
        "done",
        "finished",
        "complete",
        "goal achieved",
        "task solved",
    )
    negative_markers = (
        "未完成",
        "没有完成",
        "尚未完成",
        "not done",
        "not complete",
        "incomplete",
    )
    if any(marker in value for marker in negative_markers):
        return False
    return any(marker in value for marker in positive_markers)


def count_false_completion_claims(trajectory: List[TrajectoryRecord], recent_n: int = 10) -> int:
    thinks = [
        record.text
        for record in trajectory
        if record.kind == "think" and record.text and not record.text.startswith("[parse_error]")
    ]
    if recent_n > 0:
        thinks = thinks[-recent_n:]
    return sum(1 for text in thinks if is_completion_claim(text))


def trim_text(text: str, max_len: int = 240) -> str:
    value = (text or "").strip()
    if len(value) <= max_len:
        return value
    return value[:max_len] + f"...(截断, 原长度={len(value)})"


def build_grouped_steps(trajectory: List[TrajectoryRecord]) -> List[Dict[str, Any]]:
    """Group flat think/act/ob records into step dictionaries."""
    steps: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    step_id = 0
    for record in trajectory:
        kind = (record.kind or "").strip().lower()
        text = (record.text or "").strip()
        if kind == "think":
            step_id += 1
            current = {"step": step_id, "think": text, "act": "", "ob": ""}
            steps.append(current)
            continue
        if kind == "act":
            if current is None or current.get("act"):
                step_id += 1
                current = {"step": step_id, "think": "", "act": text, "ob": ""}
                steps.append(current)
            else:
                current["act"] = text
            continue
        if kind == "ob":
            if current is None or current.get("ob"):
                step_id += 1
                current = {"step": step_id, "think": "", "act": "", "ob": text}
                steps.append(current)
            else:
                current["ob"] = text
    return steps


def is_progress_action(action: str) -> bool:
    value = (action or "").strip().lower()
    if not value:
        return False
    progress_prefixes = (
        "take ",
        "put ",
        "clean ",
        "heat ",
        "cool ",
        "slice ",
        "open ",
        "close ",
        "toggle ",
        "use ",
    )
    return any(value.startswith(prefix) for prefix in progress_prefixes)


def classify_ob_event(observation: str) -> str:
    value = (observation or "").strip().lower()
    if not value:
        return "empty"
    negative_markers = (
        "nothing happens",
        "you see nothing",
        "nothing useful",
        "can't",
        "cannot",
        "failed",
        "already open",
        "already closed",
    )
    if any(marker in value for marker in negative_markers):
        return "negative_feedback"
    state_markers = (
        "you pick up",
        "you put",
        "you open",
        "you close",
        "you clean",
        "you heat",
        "you cool",
        "you slice",
        "you are carrying",
    )
    if any(marker in value for marker in state_markers):
        return "state_change"
    if "in it, you see" in value or "on the " in value:
        return "discovery"
    if "you arrive at" in value:
        return "movement"
    return "other"


def build_reflection_structured_payload(
    trajectory: List[TrajectoryRecord],
    goal: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    steps = build_grouped_steps(trajectory)
    acts: List[str] = []
    action_step_ids: List[int] = []
    action_to_steps: Dict[str, List[int]] = {}
    for step in steps:
        act = (step.get("act") or "").strip()
        if not act:
            continue
        acts.append(act)
        step_no = int(step.get("step", 0))
        action_step_ids.append(step_no)
        action_to_steps.setdefault(act, []).append(step_no)

    counts: Dict[str, int] = {}
    for action in acts:
        counts[action] = counts.get(action, 0) + 1
    action_frequency_topk = [
        {"action": action, "count": count}
        for action, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]
    repeated_actions = [item for item in action_frequency_topk if int(item["count"]) >= 2]

    raw_loops: List[Dict[str, Any]] = []
    for idx in range(0, max(0, len(acts) - 3)):
        a1, a2, a3, a4 = acts[idx], acts[idx + 1], acts[idx + 2], acts[idx + 3]
        if a1 == a3 and a2 == a4 and a1 != a2:
            raw_loops.append(
                {
                    "pattern": [a1, a2],
                    "start_step": action_step_ids[idx],
                    "end_step": action_step_ids[idx + 3],
                }
            )

    loop_groups: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for loop in raw_loops:
        key = (str(loop["pattern"][0]), str(loop["pattern"][1]))
        loop_groups.setdefault(key, []).append((int(loop["start_step"]), int(loop["end_step"])))
    repeated_loops: List[Dict[str, Any]] = []
    for (action_a, action_b), ranges in loop_groups.items():
        ranges.sort(key=lambda item: item[0])
        repeated_loops.append(
            {
                "pattern": [action_a, action_b],
                "occurrences": len(ranges),
                "first_range": [ranges[0][0], ranges[0][1]],
                "last_range": [ranges[-1][0], ranges[-1][1]],
            }
        )
    repeated_loops.sort(key=lambda item: int(item.get("occurrences", 0)), reverse=True)
    repeated_loops = repeated_loops[:6]

    no_progress_windows: List[Dict[str, Any]] = []
    st_start = 0
    st_len = 0
    for idx, step in enumerate(steps):
        act = (step.get("act") or "").strip()
        ob = (step.get("ob") or "").strip()
        event = classify_ob_event(ob)
        low_info = event in ("negative_feedback", "movement", "empty", "other")
        has_progress = is_progress_action(act) or event == "state_change"
        if not has_progress and low_info:
            if st_len == 0:
                st_start = idx
            st_len += 1
        else:
            if st_len >= 4:
                no_progress_windows.append(
                    {
                        "start_step": int(steps[st_start]["step"]),
                        "end_step": int(steps[idx - 1]["step"]),
                        "len": st_len,
                    }
                )
            st_len = 0
    if st_len >= 4 and steps:
        no_progress_windows.append(
            {
                "start_step": int(steps[st_start]["step"]),
                "end_step": int(steps[len(steps) - 1]["step"]),
                "len": st_len,
            }
        )
    no_progress_windows = no_progress_windows[:6]

    key_ob_events: List[Dict[str, Any]] = []
    for step in steps:
        ob = (step.get("ob") or "").strip()
        if not ob:
            continue
        event = classify_ob_event(ob)
        if event in ("negative_feedback", "state_change", "discovery"):
            key_ob_events.append(
                {
                    "step": int(step.get("step", 0)),
                    "event": event,
                    "ob_snippet": trim_text(ob, max_len=180),
                }
            )
    key_ob_events = key_ob_events[:12]

    evidence_step_ids = set()
    for repeated_action in repeated_actions[:4]:
        act = str(repeated_action["action"])
        for step_id in action_to_steps.get(act, [])[:2]:
            evidence_step_ids.add(int(step_id))
    for loop in repeated_loops:
        evidence_step_ids.add(int(loop["first_range"][0]))
        evidence_step_ids.add(int(loop["first_range"][1]))
        evidence_step_ids.add(int(loop["last_range"][0]))
        evidence_step_ids.add(int(loop["last_range"][1]))
    for event in key_ob_events[:8]:
        evidence_step_ids.add(int(event["step"]))
    if steps:
        evidence_step_ids.add(int(steps[-1]["step"]))

    step_by_id = {int(step["step"]): step for step in steps}
    evidence_steps: List[Dict[str, Any]] = []
    for step_id in sorted(evidence_step_ids)[:16]:
        step = step_by_id.get(step_id)
        if not step:
            continue
        evidence_steps.append(
            {
                "step": step_id,
                "think": trim_text(str(step.get("think", "")), max_len=220),
                "act": trim_text(str(step.get("act", "")), max_len=160),
                "ob": trim_text(str(step.get("ob", "")), max_len=220),
            }
        )

    trajectory_full = "\n".join(
        [f"{idx + 1}. {record.kind.upper()}: {(record.text or '').strip()}" for idx, record in enumerate(trajectory)]
    )

    payload: Dict[str, Any] = {
        "meta": {
            "goal": goal,
            "trajectory_records": len(trajectory),
            "grouped_steps": len(steps),
            "act_count": len(acts),
        },
        "action_stats": {
            "action_frequency_topk": action_frequency_topk,
            "repeated_actions": repeated_actions,
            "repeated_loops": repeated_loops,
        },
        "progress_signals": {
            "no_progress_windows": no_progress_windows,
            "key_ob_events": key_ob_events,
        },
        "notes": [
            "以上统计由程序直接从轨迹抽取，未使用任务特定硬编码词表。",
            "反思提示仅包含结构化统计与关键证据步原文，不注入完整轨迹。",
        ],
    }
    return payload, evidence_steps, trajectory_full


@traceable_run("idea3.failure.analyze", run_type="chain")
def analyze_failure(
    trajectory: List[TrajectoryRecord],
    goal: str,
    client: Any,
    model: str,
    max_tokens: int,
    event_log: Optional[Any] = None,
    langsmith_extra: Optional[Dict[str, Any]] = None,
) -> str:
    payload, evidence_steps, trajectory_full = build_reflection_structured_payload(
        trajectory=trajectory,
        goal=goal,
    )
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    evidence_blocks: List[str] = []
    for evidence in evidence_steps:
        evidence_blocks.append(
            f"- step {evidence['step']}\n"
            f"  think: {evidence['think'] or '(空)'}\n"
            f"  act: {evidence['act'] or '(空)'}\n"
            f"  ob: {evidence['ob'] or '(空)'}"
        )
    evidence_text = "\n".join(evidence_blocks) if evidence_blocks else "(无可用证据步)"
    prompt = (
        "你是一个ALFWorld任务诊断助手。\n"
        "你将收到两类输入：\n"
        "1) 程序抽取的结构化统计（JSON）\n"
        "2) 关键证据步的原文片段\n"
        "请优先利用结构化统计定位问题，再用证据步进行核对，禁止编造不存在的步号或事件。\n\n"
        f"目标: {goal}\n\n"
        "=== 结构化统计(JSON) ===\n"
        f"{payload_json}\n\n"
        "=== 关键证据步(原文) ===\n"
        f"{evidence_text}\n\n"
        "请输出三部分：\n"
        "错误分析列表：按条目列出错误或失败原因。\n"
        "下次建议列表：按条目给出可执行的改进建议。\n"
        "重试策略卡：给出一份“从头开始重试时可执行”的整关策略（不是续接上次）。\n"
        "请严格按下面格式输出：\n"
        "错误分析列表:\n"
        "1. ...\n"
        "2. ...\n"
        "下次建议列表:\n"
        "1. ...\n"
        "2. ...\n"
        "重试策略卡:\n"
        "- 全局策略: ...\n"
        "- 优先动作原则: ...\n"
        "- 避免动作模式: ...\n"
        "- 触发重规划条件: ...\n"
    )
    if event_log:
        event_log.banner("[analyze_failure] 发送给反思LLM的输入")
        event_log.line("[analyze_failure] 结构化统计(JSON):")
        for line in payload_json.splitlines():
            event_log.line(f"  {line}")
        event_log.blank()
        event_log.line("[analyze_failure] 关键证据步(原文):")
        for line in evidence_text.splitlines():
            event_log.line(f"  {line}")
        event_log.blank()
        event_log.line(
            f"[analyze_failure] 完整失败轨迹原文已保留在本地变量中（记录数={len(trajectory)}，长度={len(trajectory_full)} 字符），"
            "但未注入反思LLM prompt。"
        )
        event_log.line("[analyze_failure] 调用 LLM 生成失败诊断…")
    output = llm2(prompt, client=client, model=model, max_tokens=max_tokens, langsmith_extra=langsmith_extra)
    result = output.strip()
    if event_log:
        event_log.blank()
        event_log.banner("[analyze_failure] 反思LLM输出")
        event_log.line("[analyze_failure] 模型输出（失败诊断与建议）:")
        for line in result.splitlines():
            event_log.line(f"  {line}", echo=True)
        event_log.blank()
    return "【上一次失败反思，请用于新一轮从头重试】\n" f"{result}\n"

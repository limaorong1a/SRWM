"""Evaluation metrics and resume-state helpers."""

from __future__ import annotations

import json
import os
import re
import time
from glob import glob
from typing import Any, Dict, List, Optional, Protocol


class TrajectoryRecord(Protocol):
    kind: str


def progress_state_path(run_dir: str) -> str:
    return os.path.join(run_dir, "progress_state.json")


def metrics_summary_path(run_dir: str) -> str:
    return os.path.join(run_dir, "metrics_summary.json")


def trajectory_step_count(trajectory: List[TrajectoryRecord]) -> int:
    return sum(1 for record in trajectory if record.kind == "act")


def build_metrics_summary(
    num_tasks: int,
    next_task_idx: int,
    prefixes: Dict[str, str],
    cnts: List[int],
    success_counts: List[int],
    success_step_sums: List[float],
) -> Dict[str, Any]:
    per_type: List[Dict[str, Any]] = []
    prefix_keys = list(prefixes.keys())
    for idx, key in enumerate(prefix_keys):
        total = int(cnts[idx]) if idx < len(cnts) else 0
        success = int(success_counts[idx]) if idx < len(success_counts) else 0
        success_rate = (success / total) if total > 0 else 0.0
        avg_success_steps = (
            success_step_sums[idx] / success
            if (idx < len(success_step_sums) and success > 0)
            else None
        )
        per_type.append(
            {
                "task_prefix": key,
                "fewshot_variant": prefixes[key],
                "total": total,
                "success": success,
                "success_rate": round(success_rate, 6),
                "avg_success_steps": round(float(avg_success_steps), 4)
                if avg_success_steps is not None
                else None,
            }
        )

    total_completed = int(sum(cnts))
    total_success = int(sum(success_counts))
    overall_rate_completed = (total_success / total_completed) if total_completed > 0 else 0.0
    overall_rate_target = (total_success / num_tasks) if num_tasks > 0 else 0.0
    return {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_tasks_target": int(num_tasks),
        "next_task_idx": int(next_task_idx),
        "overall": {
            "total_completed": total_completed,
            "total_success": total_success,
            "success_rate_completed": round(overall_rate_completed, 6),
            "success_rate_vs_target": round(overall_rate_target, 6),
        },
        "per_type": per_type,
    }


def save_metrics_summary(
    run_dir: str,
    num_tasks: int,
    next_task_idx: int,
    prefixes: Dict[str, str],
    cnts: List[int],
    success_counts: List[int],
    success_step_sums: List[float],
) -> None:
    payload = build_metrics_summary(
        num_tasks=num_tasks,
        next_task_idx=next_task_idx,
        prefixes=prefixes,
        cnts=cnts,
        success_counts=success_counts,
        success_step_sums=success_step_sums,
    )
    with open(metrics_summary_path(run_dir), "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_progress_state(
    run_dir: str,
    num_tasks: int,
    next_task_idx: int,
    rs: List[float],
    cnts: List[int],
    success_counts: List[int],
    success_step_sums: List[float],
    failure_retries: int,
    split: str,
    *,
    task_indices_1based: Optional[List[int]] = None,
    next_subset_i: Optional[int] = None,
    subset_fraction: Optional[float] = None,
    subset_seed: Optional[int] = None,
) -> None:
    payload: Dict[str, Any] = {
        "num_tasks": int(num_tasks),
        "next_task_idx": int(next_task_idx),
        "rs": [float(value) for value in rs],
        "cnts": [int(value) for value in cnts],
        "success_counts": [int(value) for value in success_counts],
        "success_step_sums": [float(value) for value in success_step_sums],
        "failure_retries": int(failure_retries),
        "split": split,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if task_indices_1based is not None:
        payload["task_indices_1based"] = [int(value) for value in task_indices_1based]
    if next_subset_i is not None:
        payload["next_subset_i"] = int(next_subset_i)
    if subset_fraction is not None:
        payload["subset_fraction"] = float(subset_fraction)
    if subset_seed is not None:
        payload["subset_seed"] = int(subset_seed)
    with open(progress_state_path(run_dir), "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_progress_state(run_dir: str) -> Dict[str, Any]:
    state_path = progress_state_path(run_dir)
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"未找到断点状态文件: {state_path}")
    with open(state_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"断点状态文件格式错误: {state_path}")
    return data


def find_task_log_for_idx(run_dir: str, idx_1based: int) -> Optional[str]:
    pattern = os.path.join(run_dir, f"task_{idx_1based:03d}_*.log")
    candidates = sorted(glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def parse_task_log_summary(log_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(log_path, "r", encoding="utf-8") as file:
            text = file.read()
    except Exception:
        return None
    if "日志结束" not in text:
        return None
    prefix_match = re.search(r"task_prefix=([^\s]+)\s+fewshot_variant=", text)
    final_match = re.search(r"final_score=([0-9.]+)\s+final_won=(True|False)", text)
    if not prefix_match or not final_match:
        return None
    steps_match = re.search(r"success_episode_steps=([0-9]+)", text)
    return {
        "task_prefix": prefix_match.group(1),
        "final_score": float(final_match.group(1)),
        "final_won": (final_match.group(2) == "True"),
        "success_episode_steps": int(steps_match.group(1)) if steps_match else None,
    }


def rebuild_state_from_logs(
    run_dir: str,
    num_tasks: int,
    prefixes: Dict[str, str],
) -> Dict[str, Any]:
    rs = [0.0] * len(prefixes)
    cnts = [0] * len(prefixes)
    success_counts = [0] * len(prefixes)
    success_step_sums = [0.0] * len(prefixes)
    key_to_index = {key: idx for idx, key in enumerate(prefixes.keys())}

    completed_prefix = 0
    for idx_1based in range(1, num_tasks + 1):
        log_path = find_task_log_for_idx(run_dir, idx_1based)
        if not log_path:
            break
        parsed = parse_task_log_summary(log_path)
        if not parsed:
            break
        key = parsed["task_prefix"]
        if key not in key_to_index:
            break
        idx = key_to_index[key]
        reward = float(parsed["final_score"])
        won = bool(parsed["final_won"])
        rs[idx] += reward
        cnts[idx] += 1
        if won:
            success_counts[idx] += 1
            step_value = parsed["success_episode_steps"]
            if isinstance(step_value, int):
                success_step_sums[idx] += float(step_value)
        completed_prefix = idx_1based
    return {
        "next_task_idx": completed_prefix,
        "rs": rs,
        "cnts": cnts,
        "success_counts": success_counts,
        "success_step_sums": success_step_sums,
    }

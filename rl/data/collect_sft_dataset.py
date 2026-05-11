"""Collect ALFWorld trajectories with Dify agent for SFT warmup."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from rl.data.alfworld_task_pool import TASK_TYPES, TaskItem, discover_tasks, group_by_type
from rl.data.dify_client import DifyClient, DifyConfig, from_env
from rl.data.trajectory_runner import run_one_trajectory


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        matches = re.findall(r"\$\{([A-Z0-9_]+)\}", obj)
        out = obj
        for key in matches:
            value = os.getenv(key, "")
            out = out.replace(f"${{{key}}}", value)
        return out
    if isinstance(obj, list):
        return [_expand_env(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def load_config(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _expand_env(payload)


@dataclass
class AssignedTask:
    task: TaskItem
    attempt_index: int


class Scheduler:
    def __init__(
        self,
        tasks_by_type: Dict[str, List[TaskItem]],
        quota_state: Dict[str, Dict[str, int]],
        max_retry_per_task: int,
        max_total_attempts_per_type: int,
        progress_payload: Dict[str, Any],
    ) -> None:
        self.tasks_by_type = {k: list(v) for k, v in tasks_by_type.items()}
        self.quota_state = quota_state
        self.max_retry_per_task = max_retry_per_task
        self.max_total_attempts_per_type = max_total_attempts_per_type
        self.progress_payload = progress_payload
        self.lock = threading.Lock()
        self.cursors: Dict[str, int] = {k: 0 for k in TASK_TYPES}
        self.task_retries: Dict[str, int] = progress_payload.get("task_retries", {})
        self.total_attempts = int(progress_payload.get("total_attempts", 0))

    def _type_candidates(self) -> List[str]:
        candidates: List[Tuple[str, int]] = []
        for task_type in TASK_TYPES:
            state = self.quota_state.get(task_type, {})
            success = int(state.get("success", 0))
            target = int(state.get("target", 0))
            attempts = int(state.get("attempts", 0))
            if success >= target:
                continue
            if attempts >= self.max_total_attempts_per_type:
                continue
            candidates.append((task_type, success))
        candidates.sort(key=lambda x: x[1])
        return [task_type for task_type, _ in candidates]

    def allocate(self) -> Optional[AssignedTask]:
        with self.lock:
            for task_type in self._type_candidates():
                pool = self.tasks_by_type.get(task_type, [])
                if not pool:
                    continue
                n = len(pool)
                start = self.cursors[task_type]
                for offset in range(n):
                    idx = (start + offset) % n
                    task = pool[idx]
                    retries = int(self.task_retries.get(task.task_id, 0))
                    if retries >= self.max_retry_per_task:
                        continue
                    self.cursors[task_type] = (idx + 1) % n
                    self.task_retries[task.task_id] = retries + 1
                    state = self.quota_state.setdefault(task_type, {})
                    state["attempts"] = int(state.get("attempts", 0)) + 1
                    self.total_attempts += 1
                    self.progress_payload["task_retries"] = self.task_retries
                    self.progress_payload["total_attempts"] = self.total_attempts
                    return AssignedTask(task=task, attempt_index=retries)
            return None

    def complete(self, task_type: str, won: bool) -> None:
        with self.lock:
            state = self.quota_state.setdefault(task_type, {})
            if won:
                state["success"] = int(state.get("success", 0)) + 1

    def done(self) -> bool:
        with self.lock:
            for task_type in TASK_TYPES:
                state = self.quota_state.get(task_type, {})
                if int(state.get("success", 0)) < int(state.get("target", 0)):
                    if int(state.get("attempts", 0)) < self.max_total_attempts_per_type:
                        return False
            return True


def _mode_adjust_quota(base_quota: Dict[str, int], mode: str) -> Dict[str, int]:
    if mode == "smoke":
        result = {k: 0 for k in TASK_TYPES}
        first = next(iter(TASK_TYPES))
        result[first] = 1
        return result
    if mode == "pilot":
        return {k: min(10, int(base_quota.get(k, 0))) for k in TASK_TYPES}
    return {k: int(base_quota.get(k, 0)) for k in TASK_TYPES}


def _build_dify_client(cfg: Dict[str, Any]) -> DifyClient:
    base_url = str(cfg["base_url"])
    user = str(cfg.get("user", "alfworld-sft-collector"))
    # Always load secrets from env at runtime, never from yaml plaintext.
    dify_cfg: DifyConfig = from_env(base_url=base_url, user=user)
    # Keep agent_id from env for security + easy rotation.
    return DifyClient(config=dify_cfg, timeout_s=90.0)


def _serialize_trajectory(raw_dir: Path, payload: Dict[str, Any]) -> Path:
    task_type = str(payload.get("task_type", "unknown"))
    task_id = str(payload.get("task_id", "task"))
    attempt = int(payload.get("attempt_index", 0))
    status = "won" if bool(payload.get("won", False)) else "failed"
    out_path = raw_dir / task_type / f"{task_id}_attempt_{attempt}_{status}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def run_collection(config_path: Path, mode: str, resume: bool) -> None:
    cfg = load_config(config_path)
    collection_cfg = cfg["collection"]
    output_cfg = cfg["output"]
    data_root = str(cfg["alfworld"]["data_root"])
    split = str(cfg["alfworld"].get("split", "train"))

    raw_dir = Path(output_cfg["raw_dir"])
    sft_dir = Path(output_cfg["sft_dir"])
    progress_path = Path(output_cfg["progress_file"])
    quota_state_path = Path(output_cfg["quota_state_file"])
    index_path = Path(output_cfg["index_file"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(parents=True, exist_ok=True)

    base_quota = {k: int(v) for k, v in dict(collection_cfg["quota_per_type"]).items()}
    target_quota = _mode_adjust_quota(base_quota, mode=mode)
    max_workers = 1 if mode == "smoke" else int(collection_cfg["max_workers"])
    max_retry_per_task = int(collection_cfg["max_retry_per_task"])
    max_steps = int(collection_cfg["max_steps_per_episode"])
    max_total_attempts_per_type = int(collection_cfg["max_total_attempts_per_type"])

    if resume:
        progress_payload = _load_json(progress_path, default={})
        quota_state = _load_json(quota_state_path, default={})
    else:
        progress_payload = {}
        quota_state = {}
    for task_type in TASK_TYPES:
        state = quota_state.setdefault(task_type, {})
        state.setdefault("success", 0)
        state.setdefault("attempts", 0)
        state["target"] = int(target_quota.get(task_type, 0))
    _save_json(progress_path, progress_payload)
    _save_json(quota_state_path, quota_state)

    all_tasks = discover_tasks(data_root=data_root, split=split)
    grouped = group_by_type(all_tasks)
    for task_type in TASK_TYPES:
        random.shuffle(grouped[task_type])

    scheduler = Scheduler(
        tasks_by_type=grouped,
        quota_state=quota_state,
        max_retry_per_task=max_retry_per_task,
        max_total_attempts_per_type=max_total_attempts_per_type,
        progress_payload=progress_payload,
    )
    dify_client = _build_dify_client(cfg["dify"])

    while not scheduler.done():
        assignments: List[AssignedTask] = []
        for _ in range(max_workers):
            item = scheduler.allocate()
            if item is None:
                break
            assignments.append(item)
        if not assignments:
            break

        with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
            future_to_assignment = {
                executor.submit(
                    run_one_trajectory,
                    task=item.task,
                    dify_client=dify_client,
                    max_steps=max_steps,
                    attempt_index=item.attempt_index,
                ): item
                for item in assignments
            }
            for future in as_completed(future_to_assignment):
                item = future_to_assignment[future]
                try:
                    result = future.result()
                    payload = result.payload
                except Exception as exc:
                    payload = {
                        "task_id": item.task.task_id,
                        "task_type": item.task.task_type,
                        "game_file": item.task.game_file,
                        "goal": "",
                        "won": False,
                        "done": False,
                        "steps_used": 0,
                        "max_steps": max_steps,
                        "attempt_index": item.attempt_index,
                        "started_at": _utc_ts(),
                        "ended_at": _utc_ts(),
                        "dify_conversation_id": "",
                        "steps": [],
                        "error": str(exc),
                    }

                out_path = _serialize_trajectory(raw_dir=raw_dir, payload=payload)
                scheduler.complete(task_type=item.task.task_type, won=bool(payload.get("won", False)))
                _append_jsonl(
                    index_path,
                    {
                        "timestamp": _utc_ts(),
                        "task_id": payload.get("task_id", item.task.task_id),
                        "task_type": payload.get("task_type", item.task.task_type),
                        "won": bool(payload.get("won", False)),
                        "steps_used": int(payload.get("steps_used", 0)),
                        "attempt_index": int(payload.get("attempt_index", item.attempt_index)),
                        "output_file": str(out_path),
                    },
                )
                _save_json(progress_path, progress_payload)
                _save_json(quota_state_path, quota_state)

    print("Collection finished.")
    print(json.dumps(quota_state, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ALFWorld SFT trajectories via Dify API")
    parser.add_argument("--config", required=True, help="Path to sft_collection.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from existing progress/quota state")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--smoke", action="store_true")
    mode_group.add_argument("--pilot", action="store_true")
    mode_group.add_argument("--full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = "full"
    if args.smoke:
        mode = "smoke"
    elif args.pilot:
        mode = "pilot"
    run_collection(config_path=Path(args.config), mode=mode, resume=bool(args.resume))


if __name__ == "__main__":
    main()


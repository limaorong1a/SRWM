"""Build ALFWorld train task pool grouped by task type."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)


@dataclass(frozen=True)
class TaskItem:
    task_id: str
    task_type: str
    game_file: str


def _resolve_split_dir(data_root: str, split: str) -> Path:
    root = Path(data_root)
    candidate_a = root / "json_2.1.1" / split
    if candidate_a.exists():
        return candidate_a
    candidate_b = root / split
    if candidate_b.exists():
        return candidate_b
    raise FileNotFoundError(f"Cannot resolve ALFWorld split dir from data_root={data_root}, split={split}")


def _infer_task_type(path: Path) -> str:
    lower = str(path).lower().replace("\\", "/")
    for task_type in TASK_TYPES:
        if task_type in lower:
            return task_type
    return "unknown"


def discover_tasks(data_root: str, split: str = "train") -> List[TaskItem]:
    split_dir = _resolve_split_dir(data_root, split)
    files = list(split_dir.rglob("*.tw-pddl"))
    tasks: List[TaskItem] = []
    for game_file in files:
        task_type = _infer_task_type(game_file)
        if task_type == "unknown":
            continue
        task_id = game_file.stem
        tasks.append(TaskItem(task_id=task_id, task_type=task_type, game_file=str(game_file)))
    return tasks


def group_by_type(tasks: Iterable[TaskItem]) -> Dict[str, List[TaskItem]]:
    grouped: Dict[str, List[TaskItem]] = {task_type: [] for task_type in TASK_TYPES}
    for task in tasks:
        if task.task_type in grouped:
            grouped[task.task_type].append(task)
    return grouped


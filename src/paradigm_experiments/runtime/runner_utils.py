"""Small runner utilities shared by ALFWorld experiment entrypoints."""

from __future__ import annotations

import json
import os
from typing import Dict, List


def load_prompts(prompt_file: str) -> Dict[str, str]:
    with open(prompt_file, "r", encoding="utf-8") as file:
        return json.load(file)


def build_react_fewshot_block(prompts: Dict[str, str], variant: str, shot_count: int) -> str:
    if shot_count <= 0:
        return ""
    parts: List[str] = []
    for idx in range(shot_count):
        key = f"react_{variant}_{idx}"
        value = prompts.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "".join(parts)


def resolve_path_under_script_or_parent(rel: str, script_dir: str, base_dir: str) -> str:
    """Resolve a path relative to either the script directory or project base directory."""
    script_path = rel if os.path.isabs(rel) else os.path.join(script_dir, rel)
    base_path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
    if os.path.isdir(script_path):
        return script_path
    if not os.path.isabs(rel) and os.path.isdir(base_path):
        return base_path
    if os.path.isdir(script_path):
        return script_path
    raise FileNotFoundError(f"路径不存在，已尝试:\n  {script_path}\n  {base_path}")


def parse_task_indices_arg(raw: str, max_n: int) -> List[int]:
    """Parse comma-separated 1-based task indices with optional ranges."""
    value = (raw or "").strip()
    if not value:
        return []
    parsed: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start, end = int(start_raw.strip()), int(end_raw.strip())
            if start > end:
                start, end = end, start
            parsed.extend(range(start, end + 1))
        else:
            parsed.append(int(part))
    result = sorted(set(parsed))
    if not result:
        return []
    if result[0] < 1 or result[-1] > max_n:
        raise ValueError(
            f"task-indices 必须在 [1, {max_n}] 内（当前评测上限为 {max_n}），得到 {result[0]}..{result[-1]}"
        )
    return result


def task_name_from_gamefile(game_file: str) -> str:
    """Normalize an ALFWorld game file path into the task name used by logs."""
    return "/".join(str(game_file).split("/")[-3:-1])

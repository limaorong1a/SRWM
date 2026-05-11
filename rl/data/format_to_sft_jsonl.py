"""Convert raw trajectories to OpenAI-style SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


SYSTEM_PROMPT_TEMPLATE = """You are an expert ALFWorld household task solver.

GOAL of the current episode:
{goal}

At each turn you will receive a single environment observation.
Reply in EXACTLY this format:
Thought: <one short paragraph>
Action: <one executable ALFWorld command>
"""


def iter_raw_files(input_dir: Path) -> Iterable[Path]:
    for path in input_dir.rglob("*.jsonl"):
        if path.name in {"index.jsonl"}:
            continue
        yield path


def load_trajectory(path: Path) -> Dict[str, Any]:
    line = path.read_text(encoding="utf-8").strip().splitlines()[0]
    return json.loads(line)


def to_messages(payload: Dict[str, Any]) -> Dict[str, Any]:
    goal = str(payload.get("goal", "Complete the ALFWorld task."))
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(goal=goal)}
    ]
    for step in payload.get("steps", []):
        obs = str(step.get("observation_before", "")).strip()
        raw = str(step.get("model_raw", "")).strip()
        if obs:
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        if raw:
            messages.append({"role": "assistant", "content": raw})
    return {
        "task_id": payload.get("task_id", ""),
        "task_type": payload.get("task_type", ""),
        "goal": goal,
        "steps_used": int(payload.get("steps_used", 0)),
        "messages": messages,
    }


def convert(input_dir: Path, output_path: Path) -> Dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0
    with output_path.open("w", encoding="utf-8") as out:
        for raw_file in iter_raw_files(input_dir):
            payload = load_trajectory(raw_file)
            total += 1
            if not bool(payload.get("won", False)):
                continue
            row = to_messages(payload)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    return {"total": total, "won_kept": kept}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw ALFWorld trajectories to SFT JSONL")
    parser.add_argument("--input", required=True, help="Input raw directory")
    parser.add_argument("--output", required=True, help="Output dataset jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = convert(input_dir=Path(args.input), output_path=Path(args.output))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


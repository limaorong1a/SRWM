"""Generate summary stats for raw ALFWorld trajectory collection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict


def _iter_raw_files(raw_dir: Path):
    for path in raw_dir.rglob("*.jsonl"):
        if path.name == "index.jsonl":
            continue
        yield path


def _load_one(path: Path) -> Dict[str, Any]:
    line = path.read_text(encoding="utf-8").strip().splitlines()[0]
    return json.loads(line)


def build_report(raw_dir: Path) -> Dict[str, Any]:
    by_type = defaultdict(lambda: {"count": 0, "won": 0, "steps_sum": 0, "parse_failed": 0, "env_rejected_steps": 0})
    total = 0
    won_total = 0
    for path in _iter_raw_files(raw_dir):
        payload = _load_one(path)
        total += 1
        task_type = str(payload.get("task_type", "unknown"))
        won = bool(payload.get("won", False))
        steps = int(payload.get("steps_used", 0))
        by_type[task_type]["count"] += 1
        by_type[task_type]["steps_sum"] += steps
        if won:
            won_total += 1
            by_type[task_type]["won"] += 1
        for step in payload.get("steps", []):
            if bool(step.get("parse_failed", False)):
                by_type[task_type]["parse_failed"] += 1
            if bool(step.get("env_rejected", False)):
                by_type[task_type]["env_rejected_steps"] += 1

    by_type_final: Dict[str, Any] = {}
    for task_type, item in by_type.items():
        count = int(item["count"])
        by_type_final[task_type] = {
            **item,
            "success_rate": round(item["won"] / count, 4) if count else 0.0,
            "avg_steps": round(item["steps_sum"] / count, 2) if count else 0.0,
        }
    return {
        "total_trajectories": total,
        "won_trajectories": won_total,
        "overall_success_rate": round(won_total / total, 4) if total else 0.0,
        "by_task_type": by_type_final,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build raw trajectory stats report")
    parser.add_argument("--raw", required=True, help="Raw trajectory directory")
    parser.add_argument("--output", required=True, help="Output JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(raw_dir=Path(args.raw))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


"""Offline reports for local ALFWorld trajectory JSONL traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def iter_trace_events(run_dir: str | Path) -> Iterable[Dict[str, Any]]:
    trace_dir = Path(run_dir) / "traces"
    if not trace_dir.exists():
        return
    for path in sorted(trace_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    event["_trace_file"] = str(path)
                    yield event


def build_offline_report(run_dir: str | Path) -> Dict[str, Any]:
    episodes: List[Dict[str, Any]] = []
    total_steps = 0
    parser_failures = 0
    world_model_reads = 0
    world_model_updates = 0
    termination_reasons: Counter[str] = Counter()
    failure_types: Counter[str] = Counter()

    for event in iter_trace_events(run_dir):
        event_type = event.get("event")
        if event_type == "step":
            step = event.get("step") if isinstance(event.get("step"), dict) else {}
            total_steps += 1
            if step.get("parser_status") == "action_parse_failed":
                parser_failures += 1
            metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
            if metadata.get("world_model_read"):
                world_model_reads += 1
            if metadata.get("world_model_update"):
                world_model_updates += 1
            continue
        if event_type == "episode_end":
            episode = event.get("episode") if isinstance(event.get("episode"), dict) else {}
            episodes.append(episode)
            reason = str(episode.get("termination_reason") or "unknown")
            termination_reasons[reason] += 1
            failure = episode.get("failure") if isinstance(episode.get("failure"), dict) else None
            if failure:
                failure_types[str(failure.get("failure_type") or "unknown")] += 1

    total_episodes = len(episodes)
    success_count = sum(1 for episode in episodes if bool(episode.get("success")))
    step_counts = [len(episode.get("steps", [])) for episode in episodes if isinstance(episode.get("steps"), list)]
    avg_steps = (sum(step_counts) / total_episodes) if total_episodes else 0.0
    return {
        "run_dir": str(run_dir),
        "episodes": total_episodes,
        "success_count": success_count,
        "success_rate": round(success_count / total_episodes, 6) if total_episodes else 0.0,
        "avg_steps": round(avg_steps, 4),
        "total_step_events": total_steps,
        "parser_failures": parser_failures,
        "world_model_reads": world_model_reads,
        "world_model_updates": world_model_updates,
        "termination_reasons": dict(termination_reasons),
        "failure_types": dict(failure_types),
    }


def write_offline_report(run_dir: str | Path, output: str | Path | None = None) -> Dict[str, Any]:
    report = build_offline_report(run_dir)
    output_path = Path(output) if output else Path(run_dir) / "offline_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline report from ALFWorld JSONL traces.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing traces/*.jsonl.")
    parser.add_argument("--output", default=None, help="Output JSON path. Defaults to <run-dir>/offline_report.json.")
    args = parser.parse_args()
    report = write_offline_report(args.run_dir, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from dataclasses import dataclass

from paradigm_experiments.evaluation.state import (
    build_metrics_summary,
    parse_task_log_summary,
    trajectory_step_count,
)


@dataclass
class Record:
    kind: str


def test_trajectory_step_count_counts_only_actions():
    assert trajectory_step_count([Record("think"), Record("act"), Record("ob"), Record("act")]) == 2


def test_build_metrics_summary_computes_rates():
    summary = build_metrics_summary(
        num_tasks=4,
        next_task_idx=2,
        prefixes={"pick": "put", "heat": "heat"},
        cnts=[2, 0],
        success_counts=[1, 0],
        success_step_sums=[6.0, 0.0],
    )

    assert summary["overall"]["total_completed"] == 2
    assert summary["overall"]["success_rate_vs_target"] == 0.25
    assert summary["per_type"][0]["success_rate"] == 0.5
    assert summary["per_type"][0]["avg_success_steps"] == 6.0
    assert summary["per_type"][1]["avg_success_steps"] is None


def test_parse_task_log_summary_requires_completed_log(tmp_path):
    log_path = tmp_path / "task_001.log"
    log_path.write_text(
        "\n".join(
            [
                "task_prefix=pick fewshot_variant=put",
                "final_score=1.0  final_won=True",
                "success_episode_steps=5",
                "任务 #1 日志结束",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_task_log_summary(str(log_path))

    assert parsed == {
        "task_prefix": "pick",
        "final_score": 1.0,
        "final_won": True,
        "success_episode_steps": 5,
    }

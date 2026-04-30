import json

from paradigm_experiments.evaluation.report import build_offline_report, write_offline_report


def test_build_offline_report_aggregates_trace_events(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "task_001_attempt_01.jsonl"
    events = [
        {
            "event": "step",
            "step": {
                "parser_status": "ok",
                "metadata": {"world_model_read": True, "world_model_update": {"current_place": "fridge"}},
            },
        },
        {
            "event": "step",
            "step": {
                "parser_status": "action_parse_failed",
                "metadata": {"world_model_read": False},
            },
        },
        {
            "event": "episode_end",
            "episode": {
                "success": False,
                "termination_reason": "action_parse_failed",
                "steps": [{}, {}],
                "failure": {"failure_type": "action_parse_failed"},
            },
        },
    ]
    trace_path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

    report = build_offline_report(tmp_path)

    assert report["episodes"] == 1
    assert report["success_rate"] == 0.0
    assert report["parser_failures"] == 1
    assert report["world_model_reads"] == 1
    assert report["world_model_updates"] == 1
    assert report["termination_reasons"] == {"action_parse_failed": 1}


def test_write_offline_report_writes_json(tmp_path):
    report = write_offline_report(tmp_path)

    assert (tmp_path / "offline_report.json").exists()
    assert report["episodes"] == 0

import json
from pathlib import Path

from rl.data.format_to_sft_jsonl import convert


def test_convert_won_only(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    typ = raw_dir / "pick_and_place_simple"
    typ.mkdir(parents=True, exist_ok=True)

    won_payload = {
        "task_id": "task_won",
        "task_type": "pick_and_place_simple",
        "goal": "put apple in fridge",
        "won": True,
        "steps_used": 1,
        "steps": [
            {
                "observation_before": "obs1",
                "model_raw": "Thought: x\nAction: go to fridge 1",
            }
        ],
    }
    fail_payload = {
        "task_id": "task_fail",
        "task_type": "pick_and_place_simple",
        "goal": "put apple in fridge",
        "won": False,
        "steps_used": 1,
        "steps": [],
    }
    (typ / "task_won_attempt_0_won.jsonl").write_text(json.dumps(won_payload) + "\n", encoding="utf-8")
    (typ / "task_fail_attempt_0_failed.jsonl").write_text(json.dumps(fail_payload) + "\n", encoding="utf-8")

    output = tmp_path / "dataset.jsonl"
    stats = convert(input_dir=raw_dir, output_path=output)
    assert stats["total"] == 2
    assert stats["won_kept"] == 1
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["task_id"] == "task_won"
    assert row["messages"][1]["role"] == "user"
    assert "Observation: obs1" in row["messages"][1]["content"]


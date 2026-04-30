import json

from paradigm_experiments.runtime.runner_utils import (
    build_react_fewshot_block,
    load_prompts,
    parse_task_indices_arg,
    resolve_path_under_script_or_parent,
    task_name_from_gamefile,
)


def test_build_react_fewshot_block_keeps_available_examples():
    prompts = {
        "react_put_0": "example 0\n",
        "react_put_1": "",
        "react_put_2": "example 2\n",
    }

    assert build_react_fewshot_block(prompts, "put", 3) == "example 0\nexample 2\n"


def test_parse_task_indices_arg_supports_ranges_and_dedupes():
    assert parse_task_indices_arg("3,1-2,2", max_n=5) == [1, 2, 3]


def test_task_name_from_gamefile_uses_last_task_path_parts():
    game_file = "/data/json_2.1.1/valid_unseen/pick_and_place/task_001/game.tw-pddl"

    assert task_name_from_gamefile(game_file) == "pick_and_place/task_001"


def test_resolve_path_under_script_or_parent_prefers_existing_script_dir(tmp_path):
    script_dir = tmp_path / "script"
    base_dir = tmp_path / "base"
    target = script_dir / "run"
    target.mkdir(parents=True)
    base_dir.mkdir()

    assert resolve_path_under_script_or_parent("run", str(script_dir), str(base_dir)) == str(target)


def test_load_prompts_reads_json_file(tmp_path):
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps({"react_put_0": "demo"}), encoding="utf-8")

    assert load_prompts(str(prompt_file)) == {"react_put_0": "demo"}

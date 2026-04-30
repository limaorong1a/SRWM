from paradigm_experiments.agents.action_selection import (
    build_action_candidates,
    parse_action_text,
    parse_score_array,
    score_actions_with_judge,
)


def test_parse_action_text_prefers_valid_index():
    admissible = ["go to fridge 1", "open fridge 1"]

    assert parse_action_text('{"act_index": 1}', admissible) == "open fridge 1"


def test_parse_action_text_requires_exact_match():
    admissible = ["go to fridge 1", "open fridge 1"]

    assert parse_action_text('{"act":"open fridge"}', admissible) == ""
    assert parse_action_text("act: open fridge 1", admissible) == "open fridge 1"


def test_parse_score_array_extracts_and_clamps_scores():
    parsed = parse_score_array("scores: [120, -5, \"30\"]", expected_len=4)

    assert parsed == [100.0, 0.0, 30.0, 0.0]


def test_build_action_candidates_pairs_scores_with_actions():
    candidates = build_action_candidates(["a", "b"], [12.345])

    assert candidates == [
        {"index": 0, "cmd": "a", "score": 12.3},
        {"index": 1, "cmd": "b", "score": 0.0},
    ]


def test_score_actions_with_judge_uses_llm_output(monkeypatch):
    def fake_llm(prompt, client, model, max_tokens, langsmith_extra=None):
        assert "候选动作列表" in prompt
        return "[80, 20]"

    monkeypatch.setattr("paradigm_experiments.agents.action_selection.llm", fake_llm)

    assert score_actions_with_judge(["open fridge", "look"], "goal", "ctx", object(), "model", 10) == [80.0, 20.0]

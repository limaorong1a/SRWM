from dataclasses import dataclass

from paradigm_experiments.agents.reflection import (
    analyze_failure,
    build_reflection_structured_payload,
    count_false_completion_claims,
)


@dataclass
class Record:
    kind: str
    text: str


def test_count_false_completion_claims_ignores_negative_claims():
    trajectory = [
        Record("think", "The task is complete."),
        Record("think", "The task is not complete yet."),
        Record("think", "[parse_error] think 输出解析失败"),
    ]

    assert count_false_completion_claims(trajectory) == 1


def test_build_reflection_payload_detects_repeated_actions_and_loops():
    trajectory = [
        Record("think", "go to fridge"),
        Record("act", "go to fridge 1"),
        Record("ob", "You arrive at fridge 1."),
        Record("think", "go back"),
        Record("act", "go to countertop 1"),
        Record("ob", "You arrive at countertop 1."),
        Record("think", "repeat fridge"),
        Record("act", "go to fridge 1"),
        Record("ob", "You arrive at fridge 1."),
        Record("think", "repeat counter"),
        Record("act", "go to countertop 1"),
        Record("ob", "You arrive at countertop 1."),
    ]

    payload, evidence_steps, trajectory_full = build_reflection_structured_payload(
        trajectory,
        goal="put apple in fridge",
    )

    repeated = payload["action_stats"]["repeated_actions"]
    loops = payload["action_stats"]["repeated_loops"]
    assert repeated[0]["action"] == "go to countertop 1"
    assert loops[0]["pattern"] == ["go to fridge 1", "go to countertop 1"]
    assert evidence_steps
    assert "THINK" in trajectory_full


def test_analyze_failure_returns_retry_context(monkeypatch):
    def fake_llm2(prompt, client, model, max_tokens, langsmith_extra=None):
        assert "结构化统计" in prompt
        return "错误分析列表:\n1. loop"

    monkeypatch.setattr("paradigm_experiments.agents.reflection.llm2", fake_llm2)
    result = analyze_failure(
        trajectory=[Record("think", "done"), Record("act", "look"), Record("ob", "nothing happens")],
        goal="put apple",
        client=object(),
        model="model",
        max_tokens=10,
    )

    assert result.startswith("【上一次失败反思")
    assert "loop" in result

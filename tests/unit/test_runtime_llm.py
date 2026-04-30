import pytest

from paradigm_experiments.runtime.llm import build_client, emit_model_raw, traced_env_step


def test_build_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("MISSING_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MISSING_API_KEY is required"):
        build_client("MISSING_API_KEY")


def test_emit_model_raw_reports_empty_output():
    rows = []

    emit_model_raw(rows.append, "[act]", "")

    assert rows[0] == "[act] 原始输出长度=0"
    assert "空字符串" in rows[1]


def test_traced_env_step_delegates_to_batch_step():
    class Env:
        def step(self, actions):
            return actions

    assert traced_env_step(Env(), "open fridge 1") == ["open fridge 1"]

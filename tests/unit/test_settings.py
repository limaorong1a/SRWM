from paradigm_experiments.runtime.settings import TASK_PREFIXES, ablation_flag


def test_ablation_flag_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("ABLA_TEST", raising=False)

    assert ablation_flag("ABLA_TEST", True) is True
    assert ablation_flag("ABLA_TEST", False) is False


def test_ablation_flag_parses_false_values(monkeypatch):
    monkeypatch.setenv("ABLA_TEST", "false")

    assert ablation_flag("ABLA_TEST", True) is False


def test_task_prefixes_include_six_alfworld_categories():
    assert len(TASK_PREFIXES) == 6
    assert TASK_PREFIXES["pick_heat_then_place"] == "heat"

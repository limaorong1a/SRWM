from paradigm_experiments.observability.trace_schema import EpisodeTrace, StepTrace


def test_episode_trace_serializes_step():
    episode = EpisodeTrace(task="put apple on table", max_steps=2)
    episode.steps.append(StepTrace(step_index=1, phase="act", action_validated="look"))

    payload = episode.to_dict()

    assert payload["task"] == "put apple on table"
    assert payload["steps"][0]["phase"] == "act"
    assert payload["steps"][0]["action_validated"] == "look"

from paradigm_experiments.observability.langsmith import EpisodeRunTree


def test_episode_run_tree_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    with EpisodeRunTree("test") as episode:
        assert episode.child_extra() == {}
        episode.end(outputs={"ok": True})

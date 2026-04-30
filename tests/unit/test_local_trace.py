from paradigm_experiments.observability.local_trace import LocalEpisodeTraceWriter
from paradigm_experiments.observability.trace_schema import EpisodeTrace, StepTrace


class MemoryRecorder:
    def __init__(self):
        self.events = []

    def record(self, payload):
        self.events.append(dict(payload))

    def close(self):
        return None


def test_local_episode_trace_writer_records_lifecycle():
    episode = EpisodeTrace(task="put apple", max_steps=2)
    recorder = MemoryRecorder()
    writer = LocalEpisodeTraceWriter(episode, recorder)

    writer.start(task="put apple", max_steps=2, metadata={"agent": "test"})
    writer.record_step(StepTrace(step_index=1, phase="env_step", action_validated="look"))
    writer.finish(False, False, "max_steps", false_completion_claims=2)

    assert [event["event"] for event in recorder.events] == ["episode_start", "step", "episode_end"]
    assert episode.steps[0].action_validated == "look"
    assert episode.failure is not None
    assert episode.failure.false_completion_claims == 2
    assert episode.failure.max_step_exhausted is True

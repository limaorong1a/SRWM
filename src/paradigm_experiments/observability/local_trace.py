"""Helpers for writing local episode trajectory events."""

from __future__ import annotations

from typing import Any, Dict, Optional

from paradigm_experiments.observability.recorder import TraceRecorder
from paradigm_experiments.observability.trace_schema import (
    EpisodeTrace,
    FailureAttribution,
    StepTrace,
    utc_now_iso,
)


class LocalEpisodeTraceWriter:
    """Keep an EpisodeTrace in memory while mirroring events to a recorder."""

    def __init__(
        self,
        episode_trace: Optional[EpisodeTrace],
        trace_recorder: Optional[TraceRecorder],
    ):
        self.episode_trace = episode_trace
        self.trace_recorder = trace_recorder

    def record_event(self, event: str, payload: Dict[str, Any]) -> None:
        if self.trace_recorder is not None:
            self.trace_recorder.record({"event": event, **payload})

    def start(
        self,
        task: str,
        max_steps: int,
        metadata: Dict[str, Any],
    ) -> None:
        if self.episode_trace is None:
            return
        self.episode_trace.task = self.episode_trace.task or task
        self.episode_trace.max_steps = self.episode_trace.max_steps or max_steps
        self.episode_trace.metadata.update(metadata)
        self.record_event("episode_start", {"episode": self.episode_trace.to_dict()})

    def record_step(self, step_trace: StepTrace) -> None:
        if self.episode_trace is None:
            return
        self.episode_trace.steps.append(step_trace)
        self.record_event(
            "step",
            {
                "episode_id": self.episode_trace.episode_id,
                "step": step_trace.__dict__,
            },
        )

    def finish(
        self,
        success: bool,
        won: bool,
        termination_reason: str,
        false_completion_claims: int = 0,
    ) -> None:
        if self.episode_trace is None:
            return
        self.episode_trace.success = bool(success)
        self.episode_trace.final_won = bool(won)
        self.episode_trace.termination_reason = termination_reason
        self.episode_trace.ended_at = utc_now_iso()
        if not success:
            self.episode_trace.failure = FailureAttribution(
                failure_type=termination_reason,
                parse_errors=1 if termination_reason == "action_parse_failed" else 0,
                false_completion_claims=false_completion_claims,
                max_step_exhausted=termination_reason == "max_steps",
            )
        self.record_event("episode_end", {"episode": self.episode_trace.to_dict()})

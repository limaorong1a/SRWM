"""Shared trajectory schema for ALFWorld AgentOps runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LLMCallTrace:
    name: str
    model: str
    input_preview: str = ""
    output_preview: str = ""
    latency_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class StepTrace:
    step_index: int
    phase: str
    thought: str = ""
    action_raw: str = ""
    action_validated: str = ""
    admissible_actions: List[str] = field(default_factory=list)
    observation_before: str = ""
    observation_after: str = ""
    reward: Optional[float] = None
    done: Optional[bool] = None
    won: Optional[bool] = None
    parser_status: str = "ok"
    fallback_used: bool = False
    latency_ms: Optional[float] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    llm_calls: List[LLMCallTrace] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureAttribution:
    failure_type: str
    root_cause: str = ""
    evidence_steps: List[int] = field(default_factory=list)
    parse_errors: int = 0
    invalid_action_count: int = 0
    repeated_action_count: int = 0
    false_completion_claims: int = 0
    max_step_exhausted: bool = False


@dataclass
class EpisodeTrace:
    episode_id: str = field(default_factory=lambda: str(uuid4()))
    run_id: str = ""
    task: str = ""
    task_type: str = ""
    split: str = ""
    seed: Optional[int] = None
    max_steps: int = 0
    success: bool = False
    final_won: bool = False
    termination_reason: str = ""
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: Optional[str] = None
    steps: List[StepTrace] = field(default_factory=list)
    failure: Optional[FailureAttribution] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunTrace:
    run_id: str = field(default_factory=lambda: str(uuid4()))
    project: str = "alfworld-agentops"
    started_at: str = field(default_factory=utc_now_iso)
    code_version: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    model: str = ""
    agent_name: str = "idea3"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

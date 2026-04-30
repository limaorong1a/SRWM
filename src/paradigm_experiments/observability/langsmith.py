"""Optional LangSmith tracing helpers.

The project must remain runnable without LangSmith configuration. All helpers
therefore degrade to no-ops unless LANGSMITH_TRACING is explicitly enabled and
the langsmith package can be imported.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:
    from langsmith import traceable as _langsmith_traceable
    from langsmith.run_trees import RunTree
except Exception:  # pragma: no cover - optional dependency path
    _langsmith_traceable = None
    RunTree = None  # type: ignore[assignment]


def langsmith_enabled() -> bool:
    raw = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def truncate_text(value: Any, limit: int = 2000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def traceable_run(name: str, run_type: str = "chain") -> Callable[[F], F]:
    if not langsmith_enabled() or _langsmith_traceable is None:
        return lambda fn: fn
    return _langsmith_traceable(name=name, run_type=run_type)


class EpisodeRunTree(AbstractContextManager["EpisodeRunTree"]):
    """Manual parent run for one ALFWorld episode."""

    def __init__(
        self,
        name: str,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.inputs = inputs or {}
        self.metadata = metadata or {}
        self.run_tree: Any = None

    @property
    def enabled(self) -> bool:
        return langsmith_enabled() and RunTree is not None

    def __enter__(self) -> "EpisodeRunTree":
        if self.enabled:
            self.run_tree = RunTree(
                name=self.name,
                run_type="chain",
                inputs=self.inputs,
                extra={"metadata": self.metadata},
            )
            self.run_tree.post()
        return self

    def child_extra(self) -> Dict[str, Any]:
        if not self.run_tree:
            return {}
        return {"langsmith_extra": {"parent": self.run_tree}}

    def end(self, outputs: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        if not self.run_tree:
            return
        if error:
            self.run_tree.end(error=error)
        else:
            self.run_tree.end(outputs=outputs or {})
        self.run_tree.post()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            self.end(error=str(exc))
        return False

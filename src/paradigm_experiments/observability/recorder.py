"""Local trajectory recorders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol


class TraceRecorder(Protocol):
    def record(self, payload: Mapping[str, Any]) -> None:
        """Persist one trace payload."""

    def close(self) -> None:
        """Release recorder resources."""


class NoopTraceRecorder:
    def record(self, payload: Mapping[str, Any]) -> None:
        del payload

    def close(self) -> None:
        return None


class JsonlTraceRecorder:
    """Append trace events to a UTF-8 JSONL file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def record(self, payload: Mapping[str, Any]) -> None:
        self._fp.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

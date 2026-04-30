"""Human-readable event log helpers."""

from __future__ import annotations

import os
import re
import time
from typing import Optional


class TaskFileLogger:
    """Write task progress to a log file and optionally echo to stdout."""

    def __init__(self, log_path: str, echo: bool = True):
        self.log_path = log_path
        self.echo = echo
        parent = os.path.dirname(os.path.abspath(log_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fp = open(log_path, "w", encoding="utf-8")

    def line(self, msg: str, echo: Optional[bool] = None) -> None:
        do_echo = self.echo if echo is None else echo
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        row = f"[{timestamp}] {msg}"
        self._fp.write(row + "\n")
        self._fp.flush()
        if do_echo:
            print(row)

    def blank(self) -> None:
        self._fp.write("\n")
        self._fp.flush()

    def banner(self, title: str) -> None:
        self.line("=" * 72)
        self.line(title)
        self.line("=" * 72)

    def close(self) -> None:
        self._fp.close()


def safe_filename_fragment(name: str, max_len: int = 80) -> str:
    value = re.sub(r'[<>:"/\\|?*]', "_", name.replace(os.sep, "_"))
    value = re.sub(r"_+", "_", value).strip("._")
    return (value[:max_len] or "task").rstrip(".")

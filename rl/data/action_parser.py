"""Parse free-text model outputs into Thought/Action fields."""

from __future__ import annotations

import re
from typing import Tuple


def _strip_code_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", value)
        value = re.sub(r"\n?```$", "", value).strip()
    return value


def parse_free_text_action(raw: str) -> Tuple[str, str]:
    """
    Parse model output:
      Thought: ...
      Action: ...

    Returns (thought, action_text). Empty action means parse failure.
    """
    text = _strip_code_fence(raw)
    if not text:
        return "", ""

    thought = ""
    action = ""

    thought_match = re.search(r"^\s*Thought\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = re.search(r"^\s*Action\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if action_match:
        action = action_match.group(1).strip().strip("`\"'")
    else:
        # Fallback: if model outputs just one line command.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) == 1 and not lines[0].lower().startswith("thought"):
            action = lines[0].strip("`\"'")

    if " | " in action:
        action = action.split(" | ", 1)[0].strip()

    return thought, action


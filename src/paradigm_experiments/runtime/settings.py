"""Runtime settings for the unified Idea3 ablation agent."""

from __future__ import annotations

import os
from typing import Dict


TASK_PREFIXES: Dict[str, str] = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


def ablation_flag(name: str, default: bool = True) -> bool:
    """Environment variable flag parser used by ABLA_* switches."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")

"""Task-level world model helpers for the Idea3 ALFWorld agent."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from paradigm_experiments.observability.langsmith import traceable_run


def new_task_world_model() -> Dict[str, Any]:
    """Create the per-episode memory used by the world-model contribution."""
    return {
        "visited_places": [],
        "observed_items_by_place": {},
    }


@traceable_run("idea3.world_model.init", run_type="tool")
def traced_world_model_init(
    initial_observation: str,
    langsmith_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    del langsmith_extra
    model = new_task_world_model()
    current_place = extract_place_from_text(initial_observation) or "unknown"
    model["visited_places"].append(current_place)
    return {
        "model": model,
        "current_place": current_place,
    }


def extract_place_from_text(observation: str) -> str:
    ob = (observation or "").strip()
    if not ob:
        return ""
    patterns = [
        r"\bYou are in the ([^.]+)\.",
        r"\bYou are at the ([^.]+)\.",
        r"\bYou are in a ([^.]+)\.",
        r"\bYou are at a ([^.]+)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, ob, flags=re.IGNORECASE)
        if match:
            place = match.group(1).strip().lower()
            place = re.sub(r"\s+", " ", place)
            return place
    return ""


def extract_place_from_action(action: str) -> str:
    value = (action or "").strip()
    match = re.match(r"^go to (.+)$", value, flags=re.IGNORECASE)
    if not match:
        return ""
    place = match.group(1).strip().lower()
    return re.sub(r"\s+", " ", place)


def normalize_item_name(text: str) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"^(a|an|the)\s+", "", value)
    return re.sub(r"\s+", " ", value).strip(" .")


def extract_observed_items(observation: str) -> List[str]:
    ob = (observation or "").strip()
    if not ob:
        return []
    chunks: List[str] = []
    for pattern in [
        r"\byou see ([^.]+)\.",
        r"\bin it, you see ([^.]+)\.",
        r"\bon the [^,]+, you see ([^.]+)\.",
    ]:
        for match in re.finditer(pattern, ob, flags=re.IGNORECASE):
            chunks.append(match.group(1))
    if not chunks:
        return []
    items: List[str] = []
    for chunk in chunks:
        normalized_chunk = chunk.replace(" and ", ", ")
        for raw_item in normalized_chunk.split(","):
            name = normalize_item_name(raw_item)
            if not name or name in ("nothing", "nothing useful"):
                continue
            if name not in items:
                items.append(name)
    return items


def update_task_world_model(
    model: Dict[str, Any],
    action: str,
    observation: str,
    current_place: str,
) -> str:
    place = extract_place_from_action(action) or extract_place_from_text(observation) or current_place or "unknown"
    visited = model.get("visited_places")
    if not isinstance(visited, list):
        visited = []
        model["visited_places"] = visited
    if place not in visited:
        visited.append(place)

    place_items = model.get("observed_items_by_place")
    if not isinstance(place_items, dict):
        place_items = {}
        model["observed_items_by_place"] = place_items
    existing_items = place_items.get(place)
    if not isinstance(existing_items, list):
        existing_items = []
        place_items[place] = existing_items
    for item in extract_observed_items(observation):
        if item not in existing_items:
            existing_items.append(item)
    return place


def compact_world_model_for_prompt(
    model: Dict[str, Any],
    max_places: int = 12,
    max_items_per_place: int = 16,
) -> str:
    visited_raw = model.get("visited_places", [])
    observed_raw = model.get("observed_items_by_place", {})
    visited = visited_raw if isinstance(visited_raw, list) else []
    observed = observed_raw if isinstance(observed_raw, dict) else {}
    visited_cut = visited[:max_places]
    observed_cut: Dict[str, List[str]] = {}
    for place in visited_cut:
        items = observed.get(place, [])
        if isinstance(items, list):
            observed_cut[place] = items[:max_items_per_place]
    payload = {
        "visited_places": visited_cut,
        "observed_items_by_place": observed_cut,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@traceable_run("idea3.world_model.read", run_type="tool")
def traced_world_model_read(
    model: Dict[str, Any],
    max_places: int = 12,
    max_items_per_place: int = 16,
    langsmith_extra: Optional[Dict[str, Any]] = None,
) -> str:
    del langsmith_extra
    return compact_world_model_for_prompt(
        model,
        max_places=max_places,
        max_items_per_place=max_items_per_place,
    )


@traceable_run("idea3.world_model.update", run_type="tool")
def traced_world_model_update(
    model: Dict[str, Any],
    action: str,
    observation: str,
    current_place: str,
    langsmith_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    del langsmith_extra
    before_places = set(model.get("visited_places", []))
    place = update_task_world_model(
        model=model,
        action=action,
        observation=observation,
        current_place=current_place,
    )
    observed_items = model.get("observed_items_by_place", {}).get(place, [])
    return {
        "current_place": place,
        "new_place": place not in before_places,
        "visited_places": model.get("visited_places", []),
        "observed_items_at_current_place": observed_items if isinstance(observed_items, list) else [],
    }

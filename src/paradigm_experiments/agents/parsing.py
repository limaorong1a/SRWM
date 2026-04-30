"""Parsing helpers and lightweight records for the Idea3 ALFWorld agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class StepRecord:
    kind: str
    text: str


@dataclass
class ThinkDecision:
    think: str
    need_world_model: bool = False


def process_ob(observation: str) -> str:
    if observation.startswith("You arrive at loc "):
        observation = observation[observation.find(". ") + 2 :]
    return observation


def get_admissible(info: Optional[Dict]) -> List[str]:
    if isinstance(info, dict):
        admissible_commands = info.get("admissible_commands")
        if (
            isinstance(admissible_commands, list)
            and len(admissible_commands) > 0
            and isinstance(admissible_commands[0], list)
        ):
            return [
                action
                for action in admissible_commands[0]
                if not (isinstance(action, str) and action.lower().startswith("examine "))
            ]
    return []


def extract_goal(observation: str) -> str:
    marker = "Your task is to:"
    if marker in observation:
        return observation.split(marker, 1)[1].strip().split("\n")[0]
    return ""


def parse_action(output: str) -> Tuple[str, str]:
    text = output.strip()
    if text.startswith(">"):
        text = text[1:].strip()
    if text.startswith("act:"):
        return "act", text[4:].strip()
    if text.startswith("think:"):
        return "think", text[6:].strip()
    return "act", text


def extract_json_candidate(text: str) -> Optional[str]:
    value = (text or "").strip()
    if not value:
        return None
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", value)
        value = re.sub(r"\n?```$", "", value).strip()
    if value.startswith("{") and value.endswith("}"):
        return value
    match = re.search(r"\{[\s\S]*\}", value)
    if match:
        return match.group(0)
    return None


def parse_think_text(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    json_candidate = extract_json_candidate(value)
    if json_candidate:
        try:
            obj = json.loads(json_candidate)
            if isinstance(obj, dict):
                think = obj.get("think") or obj.get("thought") or obj.get("reasoning")
                if isinstance(think, str):
                    return think.strip()
        except Exception:
            pass
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("think:"):
            return stripped[6:].strip()
        return stripped
    return ""


def parse_think_decision(raw: str) -> ThinkDecision:
    value = (raw or "").strip()
    if not value:
        return ThinkDecision(think="", need_world_model=False)
    json_candidate = extract_json_candidate(value)
    if json_candidate:
        try:
            obj = json.loads(json_candidate)
            if isinstance(obj, dict):
                think_value = obj.get("think") or obj.get("thought") or obj.get("reasoning") or ""
                need_value = obj.get("need_world_model")
                need_flag = bool(need_value) if isinstance(need_value, bool) else False
                if isinstance(think_value, str):
                    return ThinkDecision(think=think_value.strip(), need_world_model=need_flag)
        except Exception:
            pass
    return ThinkDecision(think=parse_think_text(value), need_world_model=False)

"""Run one ALFWorld trajectory through Dify agent."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from rl.data.action_parser import parse_free_text_action
from rl.data.alfworld_task_pool import TaskItem


def extract_goal(observation: str) -> str:
    text = (observation or "").strip()
    for pattern in (
        r"Your task is to:\s*(.+)$",
        r"Task:\s*(.+)$",
        r"Goal:\s*(.+)$",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip().rstrip(".")
    return "Complete the ALFWorld task."


def _normalize_reset(reset_output: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        obs, info = reset_output
    else:
        obs, info = reset_output, {}
    if isinstance(obs, list):
        obs = obs[0] if obs else ""
    if not isinstance(obs, str):
        obs = str(obs)
    if not isinstance(info, dict):
        info = {}
    return obs, info


def _normalize_step(step_output: Any) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
    # Expected shapes:
    # - (obs_list, reward_list, done_list, info_dict)
    # - (obs, reward, done, info)
    obs = ""
    reward = 0.0
    done = False
    won = False
    info: Dict[str, Any] = {}
    if isinstance(step_output, tuple) and len(step_output) == 4:
        raw_obs, raw_reward, raw_done, raw_info = step_output
        if isinstance(raw_obs, list):
            obs = str(raw_obs[0]) if raw_obs else ""
        else:
            obs = str(raw_obs)
        if isinstance(raw_reward, list):
            reward = float(raw_reward[0]) if raw_reward else 0.0
        else:
            reward = float(raw_reward)
        if isinstance(raw_done, list):
            done = bool(raw_done[0]) if raw_done else False
        else:
            done = bool(raw_done)
        info = raw_info if isinstance(raw_info, dict) else {}
        won_raw = info.get("won", False)
        if isinstance(won_raw, list):
            won = bool(won_raw[0]) if won_raw else False
        else:
            won = bool(won_raw)
    return obs, reward, done, won, info


def default_env_factory(game_file: str):
    # Lazy import: keep Windows local imports safe.
    from paradigm_experiments.runtime.alfworld_env import create_single_game_env
    import yaml

    with open("base_config.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return create_single_game_env(game_file=game_file, config=config)


@dataclass
class TrajectoryResult:
    payload: Dict[str, Any]


def run_one_trajectory(
    task: TaskItem,
    dify_client: Any,
    max_steps: int = 50,
    attempt_index: int = 0,
    env_factory: Optional[Callable[[str], Any]] = None,
) -> TrajectoryResult:
    make_env = env_factory or default_env_factory
    env = make_env(task.game_file)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()

    obs, _ = _normalize_reset(env.reset())
    goal = extract_goal(obs)
    conversation_id = dify_client.create_conversation(inputs={"goal": goal})

    steps: List[Dict[str, Any]] = []
    done = False
    won = False

    for step_index in range(1, max_steps + 1):
        user_msg = f"Observation: {obs}"
        raw = dify_client.send_message(conversation_id=conversation_id, message=user_msg)
        thought, action = parse_free_text_action(raw)

        if not action:
            steps.append(
                {
                    "step": step_index,
                    "observation_before": obs,
                    "model_raw": raw,
                    "thought": thought,
                    "action": "",
                    "observation_after": "",
                    "reward": 0.0,
                    "done": False,
                    "won": False,
                    "parse_failed": True,
                    "env_rejected": True,
                }
            )
            break

        next_obs, reward, done, won, _ = _normalize_step(env.step([action]))
        env_rejected = "nothing happens" in next_obs.lower()
        steps.append(
            {
                "step": step_index,
                "observation_before": obs,
                "model_raw": raw,
                "thought": thought,
                "action": action,
                "observation_after": next_obs,
                "reward": reward,
                "done": done,
                "won": won,
                "parse_failed": False,
                "env_rejected": env_rejected,
            }
        )
        obs = next_obs
        if done or won:
            break

    ended_at = dt.datetime.now(dt.timezone.utc).isoformat()
    payload: Dict[str, Any] = {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "game_file": task.game_file,
        "goal": goal,
        "won": bool(won),
        "done": bool(done),
        "steps_used": len(steps),
        "max_steps": int(max_steps),
        "attempt_index": int(attempt_index),
        "started_at": started_at,
        "ended_at": ended_at,
        "dify_conversation_id": conversation_id,
        "steps": steps,
    }
    return TrajectoryResult(payload=payload)


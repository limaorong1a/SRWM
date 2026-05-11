"""Test doubles for Dify and ALFWorld env."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FakeDifyClient:
    replies: List[str]
    conversation_id: str = "fake-conv-1"

    def create_conversation(self, inputs=None) -> str:
        return self.conversation_id

    def send_message(self, conversation_id: str, message: str) -> str:
        if not self.replies:
            return "Thought: done\nAction: examine yourself"
        return self.replies.pop(0)


class FakeAlfworldEnv:
    def __init__(self, initial_obs: str, steps: List[Dict]):
        self.initial_obs = initial_obs
        self.steps = list(steps)
        self.index = 0

    def reset(self):
        return [self.initial_obs], {"won": [False]}

    def step(self, actions):
        action = actions[0] if isinstance(actions, list) and actions else ""
        if self.index >= len(self.steps):
            return [f"Nothing happens after {action}."], [0.0], [True], {"won": [False]}
        item = self.steps[self.index]
        self.index += 1
        return [item["obs"]], [item.get("reward", 0.0)], [item.get("done", False)], {"won": [item.get("won", False)]}


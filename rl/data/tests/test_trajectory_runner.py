from rl.data.alfworld_task_pool import TaskItem
from rl.data.tests.fakes import FakeAlfworldEnv, FakeDifyClient
from rl.data.trajectory_runner import run_one_trajectory


def test_run_one_trajectory_success():
    task = TaskItem(task_id="task1", task_type="pick_and_place_simple", game_file="/fake/game.tw-pddl")
    fake_dify = FakeDifyClient(
        replies=[
            "Thought: go find apple\nAction: go to countertop 1",
            "Thought: take it\nAction: take apple 1 from countertop 1",
            "Thought: finish task\nAction: put apple 1 in fridge 1",
        ]
    )
    fake_env = FakeAlfworldEnv(
        initial_obs="You are in a room. Your task is to: put apple in fridge.",
        steps=[
            {"obs": "You arrive at countertop 1.", "done": False, "won": False},
            {"obs": "You pick up the apple 1.", "done": False, "won": False},
            {"obs": "You put the apple 1 in the fridge 1.", "done": True, "won": True},
        ],
    )
    result = run_one_trajectory(task=task, dify_client=fake_dify, env_factory=lambda _: fake_env, max_steps=10)
    payload = result.payload
    assert payload["won"] is True
    assert payload["steps_used"] == 3
    assert payload["steps"][0]["action"] == "go to countertop 1"


def test_run_one_trajectory_parse_failure():
    task = TaskItem(task_id="task2", task_type="pick_and_place_simple", game_file="/fake/game.tw-pddl")
    fake_dify = FakeDifyClient(replies=["Thought only without action"])
    fake_env = FakeAlfworldEnv(initial_obs="Task: do something", steps=[])
    result = run_one_trajectory(task=task, dify_client=fake_dify, env_factory=lambda _: fake_env, max_steps=5)
    payload = result.payload
    assert payload["won"] is False
    assert payload["steps"][0]["parse_failed"] is True


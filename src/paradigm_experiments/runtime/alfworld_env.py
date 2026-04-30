"""ALFWorld environment construction helpers."""

from __future__ import annotations

from typing import Dict, Tuple


def bootstrap_alfworld_env(config: Dict, split: str) -> Tuple[object, object]:
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    original_env = AlfredTWEnv(config, train_eval=split)
    if hasattr(original_env, "collect_game_files"):
        original_env.collect_game_files()
    if hasattr(original_env, "get_game_logic"):
        original_env.get_game_logic()
    return original_env, original_env.init_env(batch_size=1)


def max_episode_steps(config: Dict) -> int:
    training_method = config["general"]["training_method"]
    if training_method == "dqn":
        return int(config["rl"]["training"]["max_nb_steps_per_episode"])
    if training_method == "dagger":
        return int(config["dagger"]["training"]["max_nb_steps_per_episode"])
    return 50


def create_single_game_env(game_file: str, config: Dict):
    """Create an environment bound to one ALFWorld game file."""
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    alfred_demangler = AlfredDemangler(shuffle=False)
    wrappers_tw = [alfred_demangler, AlfredInfos]
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env_id = textworld.gym.register_games(
        [game_file],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_episode_steps(config),
        wrappers=wrappers_tw,
    )
    return textworld.gym.make(env_id)

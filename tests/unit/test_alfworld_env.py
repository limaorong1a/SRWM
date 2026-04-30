from paradigm_experiments.runtime.alfworld_env import max_episode_steps


def test_max_episode_steps_reads_dagger_config():
    config = {
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": 42}},
    }

    assert max_episode_steps(config) == 42


def test_max_episode_steps_defaults_for_unknown_method():
    config = {"general": {"training_method": "unknown"}}

    assert max_episode_steps(config) == 50

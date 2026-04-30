from paradigm_experiments.agents.world_model import (
    compact_world_model_for_prompt,
    extract_observed_items,
    traced_world_model_init,
    traced_world_model_update,
)


def test_world_model_tracks_places_and_items():
    init = traced_world_model_init("You are in the kitchen. You see a apple 1.")
    model = init["model"]

    update = traced_world_model_update(
        model=model,
        action="go to fridge 1",
        observation="You are at the fridge 1. In it, you see a tomato 1 and a mug 2.",
        current_place=init["current_place"],
    )

    assert update["current_place"] == "fridge 1"
    assert update["new_place"] is True
    assert "fridge 1" in model["visited_places"]
    assert model["observed_items_by_place"]["fridge 1"] == ["tomato 1", "mug 2"]


def test_compact_world_model_for_prompt_limits_fields():
    model = {
        "visited_places": ["countertop 1"],
        "observed_items_by_place": {"countertop 1": ["apple 1", "knife 1"]},
    }

    payload = compact_world_model_for_prompt(model, max_items_per_place=1)

    assert "countertop 1" in payload
    assert "apple 1" in payload
    assert "knife 1" not in payload


def test_extract_observed_items_handles_empty_observation():
    assert extract_observed_items("") == []

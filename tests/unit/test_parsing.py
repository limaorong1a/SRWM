from paradigm_experiments.agents.parsing import (
    extract_goal,
    get_admissible,
    parse_action,
    parse_think_decision,
    parse_think_text,
    process_ob,
)


def test_get_admissible_filters_examine_actions():
    info = {"admissible_commands": [["look", "examine apple 1", "open fridge 1"]]}

    assert get_admissible(info) == ["look", "open fridge 1"]


def test_extract_goal_reads_task_marker():
    observation = "Intro\nYour task is to: put apple in fridge\nMore text"

    assert extract_goal(observation) == "put apple in fridge"


def test_parse_think_decision_reads_json_flag():
    decision = parse_think_decision('{"think":"I need memory","need_world_model":true}')

    assert decision.think == "I need memory"
    assert decision.need_world_model is True


def test_parse_think_text_supports_prefix_and_json():
    assert parse_think_text("think: go to fridge") == "go to fridge"
    assert parse_think_text('{"reasoning":"open it"}') == "open it"


def test_process_ob_strips_location_prefix():
    assert process_ob("You arrive at loc 1. You see a fridge.") == "You see a fridge."


def test_parse_action_handles_think_and_act_prefixes():
    assert parse_action(">think: inspect") == ("think", "inspect")
    assert parse_action("act: open fridge 1") == ("act", "open fridge 1")

from rl.data.action_parser import parse_free_text_action


def test_parse_standard_format():
    thought, action = parse_free_text_action("Thought: find apple\nAction: go to countertop 1")
    assert thought == "find apple"
    assert action == "go to countertop 1"


def test_parse_markdown_fence():
    raw = "```text\nThought: test\nAction: take apple 1 from countertop 2\n```"
    thought, action = parse_free_text_action(raw)
    assert thought == "test"
    assert action == "take apple 1 from countertop 2"


def test_parse_failure_when_missing_action():
    thought, action = parse_free_text_action("Thought: I should explore first.")
    assert thought == "I should explore first."
    assert action == ""


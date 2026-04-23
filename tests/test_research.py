import json

from tcrb.research import (
    apply_function_name_masking,
    classify_tool_call_failure,
    mine_failure_preferences,
    normalize_sharegpt_record,
    normalize_toolpreference_record,
)


def test_normalize_sharegpt_record_keeps_messages_tools_and_text():
    row = {
        "system": "You are a function-calling model.",
        "conversations": [
            {"from": "human", "value": "What's the weather in Berlin?"},
            {
                "from": "gpt",
                "value": '{"name":"weather.lookup","arguments":{"city":"Berlin"}}',
            },
        ],
        "tools": [{"type": "function", "function": {"name": "weather.lookup"}}],
        "category": "weather",
    }

    normalized = normalize_sharegpt_record(row, source_name="toolace")

    assert [message["role"] for message in normalized["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert normalized["tools"][0]["function"]["name"] == "weather.lookup"
    assert "<|tools|>" in normalized["text"]
    assert "weather.lookup" in normalized["text"]
    assert normalized["metadata"]["category"] == "weather"


def test_function_name_masking_replaces_tool_names_consistently():
    record = {
        "messages": [
            {"role": "user", "content": "Use weather.lookup for Berlin."},
            {
                "role": "assistant",
                "content": '{"name":"weather.lookup","arguments":{"city":"Berlin"}}',
            },
        ],
        "tools": [{"type": "function", "function": {"name": "weather.lookup"}}],
        "metadata": {},
    }

    masked = apply_function_name_masking(record, ratio=1.0, seed=7)

    replacement = masked["metadata"]["masking"]["replacements"]["weather.lookup"]
    assert masked["tools"][0]["function"]["name"] == replacement
    assert replacement in masked["messages"][0]["content"]
    assert replacement in masked["messages"][1]["content"]
    assert replacement in masked["text"]


def test_normalize_toolpreference_record_reshapes_prompt_and_pair():
    row = {
        "instruction": {"query": "Check order status"},
        "input": [{"name": "order.status"}],
        "output": [
            {"name": "order.status", "arguments": {"order_id": "A1"}},
            {"name": "order.lookup", "arguments": {"order_id": "A1"}},
        ],
    }

    normalized = normalize_toolpreference_record(row, source_name="toolpreference")

    assert "Check order status" in normalized["prompt"]
    assert '"order.status"' in normalized["chosen"]
    assert '"order.lookup"' in normalized["rejected"]
    assert normalized["source"] == "toolpreference"


def test_classify_tool_call_failure_distinguishes_common_error_types():
    expected = {"name": "calendar.create", "arguments": {"day": "monday"}}
    hallucinated = {"name": "ghost.tool", "arguments": {"day": "monday"}}
    wrong_name = {"name": "calendar.delete", "arguments": {"day": "monday"}}
    wrong_arg_name = {"name": "calendar.create", "arguments": {"date": "monday"}}
    wrong_arg_value = {"name": "calendar.create", "arguments": {"day": "friday"}}

    assert (
        classify_tool_call_failure(
            expected,
            hallucinated,
            allowed_tool_names={"calendar.create", "calendar.delete"},
        )
        == "hallucinated_function"
    )
    assert classify_tool_call_failure(expected, wrong_name) == "wrong_function_name"
    assert classify_tool_call_failure(expected, wrong_arg_name) == "missing_required_arg"
    assert classify_tool_call_failure(expected, wrong_arg_value) == "wrong_argument_value"


def test_mine_failure_preferences_skips_matches_and_keeps_failure_metadata():
    payload = {
        "cases": [
            {
                "task_id": "ok",
                "prompt": "Query A",
                "expected_output": {"name": "weather.lookup", "arguments": {"city": "Berlin"}},
                "predicted_output": {"name": "weather.lookup", "arguments": {"city": "Berlin"}},
            },
            {
                "task_id": "bad",
                "prompt": "Query B",
                "expected_output": {"name": "weather.lookup", "arguments": {"city": "Berlin"}},
                "predicted_output": {"name": "weather.find", "arguments": {"city": "Berlin"}},
                "source": "bfcl",
            },
        ]
    }

    mined = mine_failure_preferences(payload, allowed_tool_names={"weather.lookup", "weather.find"})

    assert len(mined) == 1
    assert mined[0]["prompt"] == "Query B"
    assert mined[0]["failure_type"] == "wrong_function_name"
    assert json.loads(mined[0]["chosen"])["name"] == "weather.lookup"
    assert mined[0]["metadata"]["task_id"] == "bad"
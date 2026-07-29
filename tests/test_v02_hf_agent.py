import random

import torch

from tcrb.v02.agent import HFAgent, build_chat_messages
from tcrb.v02.types import Observation, ToolCall, ToolDef


def _tool():
    return ToolDef(
        name="customer_lookup",
        description="Look up a customer",
        input_schema={
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
        output_schema={"type": "object"},
    )


def test_chat_messages_put_query_before_action_history():
    action = ToolCall("customer_lookup", {"customer_id": "C001"})
    observation = Observation("success", {"name": "Alice"})

    messages = build_chat_messages(
        "Who is C001?",
        [_tool()],
        [(action, observation)],
    )

    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "Who is C001?"
    assert "Tool result" in messages[3]["content"]


def test_hf_agent_generates_and_parses_complete_tool_action():
    class DummyEncoding:
        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.tensor([[1, 1]])

    class DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return DummyEncoding()

        def decode(self, tokens, skip_special_tokens=True):
            return '{"name":"customer_lookup","arguments":{"customer_id":"C001"}}'

    class DummyModel:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])

    action = HFAgent(model=DummyModel(), tokenizer=DummyTokenizer()).next_action(
        task_query="Who is C001?",
        available_tools=[_tool()],
        history=[],
        rng=random.Random(42),
    )

    assert action == ToolCall("customer_lookup", {"customer_id": "C001"}, call_id=action.call_id)

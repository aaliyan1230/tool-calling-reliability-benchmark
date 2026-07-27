from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .types import (
    Abort,
    AgentAction,
    Clarify,
    FinalAnswer,
    Observation,
    ToolCall,
    ToolDef,
)


def parse_action(raw: str) -> AgentAction | None:
    raw = raw.strip()
    if not raw:
        return None

    action = _try_parse_json_action(raw)
    if action is not None:
        return action

    action = _try_parse_tagged_action(raw)
    if action is not None:
        return action

    action = _try_parse_llm_default(raw)
    return action


def _try_parse_json_action(raw: str) -> AgentAction | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return None

    if "tool_call" in data:
        tc = data["tool_call"]
        return ToolCall(
            name=str(tc.get("name", "")),
            arguments=dict(tc.get("arguments", {})),
        )
    if "function" in data:
        fn = data["function"]
        return ToolCall(
            name=str(fn.get("name", "")),
            arguments=dict(fn.get("arguments", {})),
        )
    if "final_answer" in data:
        return FinalAnswer(text=str(data["final_answer"]))
    if "clarify" in data:
        return Clarify(text=str(data["clarify"]))
    if "abort" in data:
        return Abort(reason=str(data["abort"]))

    if "name" in data and ("arguments" in data or "parameters" in data):
        return ToolCall(
            name=str(data["name"]),
            arguments=dict(data.get("arguments", data.get("parameters", {}))),
        )

    return None


def _try_parse_tagged_action(raw: str) -> AgentAction | None:
    import re

    tool_match = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.DOTALL
    )
    if tool_match:
        try:
            data = json.loads(tool_match.group(1))
            return ToolCall(
                name=str(data.get("name", "")),
                arguments=dict(data.get("arguments", {})),
            )
        except json.JSONDecodeError:
            pass

    answer_match = re.search(
        r"<final_answer>\s*(.*?)\s*</final_answer>", raw, re.DOTALL
    )
    if answer_match:
        return FinalAnswer(text=answer_match.group(1).strip())

    return None


def _try_parse_llm_default(raw: str) -> AgentAction | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(data, str):
        return FinalAnswer(text=data)
    return None


@runtime_checkable
class Agent(Protocol):
    agent_id: str

    def next_action(
        self,
        *,
        task_query: str,
        available_tools: list[ToolDef],
        history: list[tuple[AgentAction | None, Observation | None]],
        rng: random.Random,
    ) -> AgentAction:
        ...


@dataclass
class ReplayAgent:
    agent_id: str = "replay"
    action_sequence: list[AgentAction] = field(default_factory=list)
    default: AgentAction = field(default_factory=lambda: FinalAnswer(text=""))

    def next_action(
        self,
        *,
        task_query: str,
        available_tools: list[ToolDef],
        history: list[tuple[AgentAction | None, Observation | None]],
        rng: random.Random,
    ) -> AgentAction:
        step_count = len(history)
        if step_count < len(self.action_sequence):
            return self.action_sequence[step_count]
        return self.default


@dataclass
class OracleAgent:
    replay: ReplayAgent
    agent_id: str = "oracle"

    def next_action(
        self,
        *,
        task_query: str,
        available_tools: list[ToolDef],
        history: list[tuple[AgentAction | None, Observation | None]],
        rng: random.Random,
    ) -> AgentAction:
        return self.replay.next_action(
            task_query=task_query,
            available_tools=available_tools,
            history=history,
            rng=rng,
        )


def _format_tools_for_prompt(tools: list[ToolDef]) -> str:
    formatted = []
    for tool in tools:
        params = json.dumps(tool.input_schema, indent=2)
        formatted.append(
            f"  - {tool.name}: {tool.description}\n    Parameters: {params}"
        )
    return "\n".join(formatted)


def _format_history(history: list[tuple[AgentAction | None, Observation | None]]) -> str:
    lines: list[str] = []
    for action, obs in history:
        if isinstance(action, ToolCall):
            lines.append(f"Action: called {action.name}({json.dumps(action.arguments)})")
        elif isinstance(action, FinalAnswer):
            lines.append(f"Action: final_answer: {action.text}")
        elif isinstance(action, Clarify):
            lines.append(f"Action: clarify: {action.text}")
        elif isinstance(action, Abort):
            lines.append(f"Action: abort: {action.reason}")

        if obs is not None:
            lines.append(f"Observation: [{obs.status}] {json.dumps(obs.payload)}")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a reliable assistant. You MUST call tools to look up real data. Never guess or fabricate answers.

Output EXACTLY one JSON per turn with no extra text:

{"name": "tool_name", "arguments": {"arg1": "value1"}}
{"final_answer": "Your answer based on tool results"}

Example of correct tool call:
User: What is customer C001's name?
Assistant: {"name": "customer_lookup", "arguments": {"customer_id": "C001"}}

After receiving the tool result, provide the final answer:
Assistant: {"final_answer": "Customer C001 is Alice Chen (alice@example.com), premium tier."}

Example of asking for missing info:
User: Check the balance.
Assistant: {"clarify": "Which account ID should I check?"}

RULES:
- Output ONLY valid JSON. No markdown, no explanations, no thinking tags, no "I will..." text.
- Use exact tool names and argument names from the tool list.
- Base answers on actual tool results, never guess.
"""


def build_chat_prompt(
    task_query: str,
    tools: list[ToolDef],
    history: list[tuple[AgentAction | None, Observation | None]],
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    return (
        f"{system_prompt}\n\n"
        f"Available tools:\n{_format_tools_for_prompt(tools)}\n\n"
        f"History:\n{_format_history(history)}\n\n"
        f"User query: {task_query}\n\n"
        f"JSON action:"
    )


def build_chat_messages(
    task_query: str,
    tools: list[ToolDef],
    history: list[tuple[AgentAction | None, Observation | None]],
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    tools_text = _format_tools_for_prompt(tools)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt + "\n\nAvailable tools:\n" + tools_text},
    ]

    for action, obs in history:
        if isinstance(action, ToolCall):
            content = json.dumps({"name": action.name, "arguments": action.arguments})
            messages.append({"role": "assistant", "content": content})
        elif isinstance(action, FinalAnswer):
            messages.append({"role": "assistant", "content": json.dumps({"final_answer": action.text})})

        if obs is not None:
            obs_content = json.dumps({"status": obs.status, "result": obs.payload})
            messages.append({"role": "user", "content": f"Tool result: {obs_content}"})

    messages.append({"role": "user", "content": task_query})
    return messages


@dataclass
class HFAgent:
    agent_id: str = "hf_local"
    model: Any = None
    tokenizer: Any = None
    system_prompt: str = SYSTEM_PROMPT
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    enable_thinking: bool = False

    def _build_prompt(
        self,
        task_query: str,
        available_tools: list[ToolDef],
        history: list[tuple[AgentAction | None, Observation | None]],
    ) -> str:
        tools_text = _format_tools_for_prompt(available_tools)

        parts = [self.system_prompt, "", "Available tools:", tools_text]

        if not history:
            parts.append("")
            parts.append("Example conversation:")
            parts.append('User: What is the name of customer C001?')
            parts.append('Assistant: {"name": "customer_lookup", "arguments": {"customer_id": "C001"}}')
            parts.append('Tool result: {"status": "success", "result": {"found": true, "name": "Alice Chen", "email": "alice@example.com", "tier": "premium"}}')
            parts.append('Assistant: {"final_answer": "Customer C001 is Alice Chen (alice@example.com), premium tier."}')
            parts.append("")

        for action, obs in history:
            if isinstance(action, ToolCall):
                parts.append(f'Assistant: {{"name": "{action.name}", "arguments": {json.dumps(action.arguments)}}}')
            elif isinstance(action, FinalAnswer):
                parts.append(f'Assistant: {{"final_answer": {json.dumps(action.text)}}}')

            if obs is not None:
                parts.append(f'Tool result: {{"status": "{obs.status}", "result": {json.dumps(obs.payload)}}}')

        parts.append(f"User: {task_query}")
        parts.append("Assistant: ")

        return "\n".join(parts)

    def next_action(
        self,
        *,
        task_query: str,
        available_tools: list[ToolDef],
        history: list[tuple[AgentAction | None, Observation | None]],
        rng: random.Random,
    ) -> AgentAction:
        import torch

        prompt = self._build_prompt(task_query, available_tools, history)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else None,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        import re
        json_matches = re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*[,}][^{}]*\}', raw)
        for match in json_matches:
            action = parse_action(match)
            if action is not None:
                return action

        json_matches = re.findall(r'\{[^{}]*"final_answer"\s*:\s*"[^"]*"[^{}]*\}', raw)
        for match in json_matches:
            action = parse_action(match)
            if action is not None:
                return action

        first_line = raw.split("\n")[0].strip()
        if first_line.startswith('{"'):
            action = parse_action(first_line)
            if action is not None:
                return action

        if raw.startswith('{"'):
            action = parse_action(raw)
            if action is not None:
                return action

        return FinalAnswer(text=raw[:500]) if raw else FinalAnswer(text="")


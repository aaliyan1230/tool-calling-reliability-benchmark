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


RECOVERY_SYSTEM_PROMPT = SYSTEM_PROMPT + """

RECOVERY RULES:
- Treat every non-success tool result as evidence that the previous action did not complete the task.
- For timeout, rate_limit, or execution_error, retry the same tool at most once. Do not repeat an unchanged failed call.
- If a retry fails, or the error is invalid_arguments or unknown_tool, select a different valid tool when one is available.
- For schema_drift or partial_output, use only fields that are present and supported by the task. If the required fact is missing, use another tool instead of guessing.
- For silent_corruption or cross_source_conflict, do not trust the suspicious result. Seek an independent tool result before answering.
- After a successful tool result, answer only when it contains the requested fact; otherwise continue with the next necessary tool.
- Keep tool arguments consistent with the query and prior observations. Never invent values to make a call executable.
- Stop retrying when the history shows the same tool already failed twice. Prefer a valid fallback or a concise evidence-based final answer.
"""


def resolve_system_prompt(system_prompt: str | None, prompt_variant: str = "default") -> str:
    """Resolve an optional custom prompt without disabling the built-in default."""
    if system_prompt and system_prompt.strip():
        return system_prompt
    if prompt_variant == "default":
        return SYSTEM_PROMPT
    if prompt_variant == "recovery":
        return RECOVERY_SYSTEM_PROMPT
    raise ValueError(f"unknown prompt variant: {prompt_variant}")


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

    def next_action(
        self,
        *,
        task_query: str,
        available_tools: list[ToolDef],
        history: list[tuple[AgentAction | None, Observation | None]],
        rng: random.Random,
    ) -> AgentAction:
        backend = LogProbAgent(
            model=self.model,
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
        )
        return backend.next_action(
            task_query=task_query,
            available_tools=available_tools,
            history=history,
            rng=rng,
        )


@dataclass
class LogProbAgent:
    agent_id: str = "logprob"
    model: Any = None
    tokenizer: Any = None
    system_prompt: str = SYSTEM_PROMPT
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0

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
            parts.append('User: What is the name of customer X999?')
            parts.append('Assistant: {"name": "customer_lookup", "arguments": {"customer_id": "X999"}}')
            parts.append('Tool result: {"status": "success", "result": {"found": false, "customer_id": "X999"}}')
            parts.append('Assistant: {"final_answer": "Customer X999 was not found in our system."}')
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

    def _score_completions(self, prompt: str, completions: list[str]) -> list[float]:
        import torch

        full_texts = [prompt + c for c in completions]
        enc_full = self.tokenizer(full_texts, return_tensors="pt", padding=True)
        enc_prompt = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = int(enc_prompt["input_ids"].shape[1])

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]

        if torch.cuda.is_available():
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        shifted_logits = logits[:, :-1, :]
        shifted_targets = input_ids[:, 1:]
        shifted_mask = attention_mask[:, 1:]

        start = max(0, prompt_len - 1)
        token_log_probs = torch.log_softmax(shifted_logits[:, start:, :], dim=-1)
        target_slice = shifted_targets[:, start:]
        mask_slice = shifted_mask[:, start:].to(token_log_probs.dtype)

        gathered = token_log_probs.gather(dim=-1, index=target_slice.unsqueeze(-1)).squeeze(-1)
        return [float(v) for v in (gathered * mask_slice).sum(dim=-1).tolist()]

    def _generate_answer(self, prompt: str) -> str:
        import torch
        answer_prompt = prompt + '{"final_answer": "'
        inputs = self.tokenizer(answer_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        text = text.split('"}')[0].split('\n')[0]
        return text if text else "Task completed."

    def _extract_args(self, tool: ToolDef, task_query: str, history: list) -> dict[str, Any]:
        args: dict[str, Any] = {}
        props = tool.input_schema.get("properties", {})
        import re

        for arg_name, arg_info in props.items():
            if arg_info.get("type") == "string":
                patterns = [
                    rf'\b([A-Z]{{1,3}}-\d{{2,4}})\b',
                    rf'\b([A-Z]\d{{3}})\b',
                    rf'\b([A-Z]+-\d+)\b',
                    rf'\b(ISSUE-\d+)\b',
                    rf'\b(BUILD-\d+)\b',
                    rf'\b(LOAN-\d+)\b',
                    rf'\b(TXN-\d+)\b',
                ]
                for pat in patterns:
                    m = re.search(pat, task_query, re.IGNORECASE)
                    if m:
                        val = m.group(1)
                        if arg_name == "code":
                            val = val.upper()
                        elif arg_name == "destination":
                            val = val.upper()
                        args[arg_name] = val
                        break
                if arg_name not in args:
                    for _, obs in reversed(history):
                        if obs and obs.payload and isinstance(obs.payload, dict):
                            for key in obs.payload:
                                if arg_name.lower() in key.lower():
                                    val = obs.payload[key]
                                    if isinstance(val, str) and val.strip():
                                        args[arg_name] = str(val).strip()
                                        break

        if not args:
            m = re.search(r'\b([A-Z0-9][A-Z0-9-]{2,15})\b', task_query)
            if m:
                first_arg = list(props.keys())[0] if props else ""
                if first_arg:
                    args[first_arg] = m.group(1)

        if "amount" in props:
            m = re.search(r'\$?(\d+(?:\.\d+)?)', task_query)
            if m:
                args["amount"] = float(m.group(1))

        return args

    def next_action(
        self,
        *,
        task_query: str,
        available_tools: list[ToolDef],
        history: list[tuple[AgentAction | None, Observation | None]],
        rng: random.Random,
    ) -> AgentAction:
        if not available_tools:
            return FinalAnswer(text="")

        prompt = self._build_prompt(task_query, available_tools, history)

        completions: list[tuple[str, AgentAction]] = []

        for tool in available_tools:
            completions.append(
                (f'{{"name": "{tool.name}", "arguments": {{',
                 ToolCall(name=tool.name, arguments={}))
            )

        if history:
            completions.append(('{"final_answer": "', FinalAnswer(text="")))

        texts = [c[0] for c in completions]
        scores = self._score_completions(prompt, texts)

        scored = list(zip(scores, [c[1] for c in completions]))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored:
            best_action = scored[0][1]
            if isinstance(best_action, ToolCall):
                best_tool = next((t for t in available_tools if t.name == best_action.name), None)
                if best_tool:
                    args = self._extract_args(best_tool, task_query, history)
                    best_action = ToolCall(name=best_action.name, arguments=args, call_id=best_action.call_id)
            elif isinstance(best_action, FinalAnswer):
                obs_text = ""
                for _, obs in reversed(history):
                    if obs and obs.payload and obs.status == "success":
                        obs_text = json.dumps(obs.payload)
                        break
                best_action = FinalAnswer(text=obs_text if obs_text else "Task completed.")
            return best_action

        return FinalAnswer(text="")

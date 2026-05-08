from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import TaskSpec, ToolSpec, Workload


def supports_schema(tool: ToolSpec, task: TaskSpec) -> bool:
    return all(field in tool.schema_fields for field in task.required_schema)


def heuristic_pick(
    task: TaskSpec,
    workload: Workload,
    attempted_tools: set[str],
    last_status: str | None,
) -> str:
    ordered = [task.primary_tool, *task.fallback_tools]
    ordered = [name for name in ordered if name in workload.tools]
    if not ordered:
        return ""

    schema_issue = str(last_status or "") in {
        "malformed_schema",
        "contract_drift",
        "invalid_schema",
        "unknown_tool",
    }
    if schema_issue:
        for name in ordered:
            if name in attempted_tools:
                continue
            if supports_schema(workload.tools[name], task):
                return name

    for name in ordered:
        if supports_schema(workload.tools[name], task):
            return name

    return ordered[0]


@dataclass
class HFLocalPlannerCore:
    planner_id: str
    base_model_id: str
    adapter_path: str | None = None
    candidate_scope: str = "task"
    candidate_order: str = "task"
    policy_adjustment_weight: float = 1.0
    heuristic_policy_shortcuts: bool = True

    def __post_init__(self) -> None:
        hf_token = str(os.environ.get("HF_TOKEN", "")).strip()
        auth_kwargs: dict = {}
        if hf_token:
            auth_kwargs["token"] = hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            trust_remote_code=True,
            **auth_kwargs,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs: dict = {"trust_remote_code": True}
        if torch.cuda.is_available():
            model_kwargs.update({"dtype": torch.float16, "device_map": "auto"})

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            **auth_kwargs,
            **model_kwargs,
        )

        adapter_path = str(self.adapter_path or "").strip()
        if adapter_path:
            from peft import PeftModel

            base_model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                is_trainable=False,
            )

        self.model = base_model

        self.model.eval()

    def _prompt(
        self,
        *,
        task: TaskSpec,
        workload: Workload,
        policy: str,
        attempt_number: int,
        attempted_tools: set[str],
        last_status: str | None,
    ) -> str:
        available = self._candidate_names(task=task, workload=workload)
        candidate_tools = {
            name: {
                "description": workload.tools[name].description,
                "schema_fields": workload.tools[name].schema_fields,
                "base_latency_ms": workload.tools[name].base_latency_ms,
            }
            for name in available
        }

        payload = {
            "user_query": task.user_query,
            "required_schema": task.required_schema,
            "policy": policy,
            "attempt_number": attempt_number,
            "attempted_tools": sorted(attempted_tools),
            "last_status": last_status,
            "candidate_tools": candidate_tools,
        }
        return (
            "Select exactly one tool for the user query. "
            "Do not infer labels from task IDs or hidden metadata; use only query and tool metadata. "
            "Output must be valid JSON with key tool_name only.\n\n"
            + json.dumps(payload, ensure_ascii=True)
            + "\n\nJSON:"
        )

    def _candidate_names(self, *, task: TaskSpec, workload: Workload) -> list[str]:
        if self.candidate_scope == "workload":
            names = list(workload.tools.keys())
        else:
            names = [task.primary_tool, *task.fallback_tools]

        seen: set[str] = set()
        available = []
        for name in names:
            if name in workload.tools and name not in seen:
                seen.add(name)
                available.append(name)

        if self.candidate_order == "sorted":
            return sorted(available)
        return available

    def _sequence_logprob(self, prompt: str, completion: str) -> float:
        full_text = prompt + completion
        enc_full = self.tokenizer(full_text, return_tensors="pt")
        enc_prompt = self.tokenizer(prompt, return_tensors="pt")

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]
        prompt_len = int(enc_prompt["input_ids"].shape[1])

        if torch.cuda.is_available():
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        shifted_logits = logits[:, :-1, :]
        shifted_targets = input_ids[:, 1:]

        start = max(0, prompt_len - 1)
        token_log_probs = torch.log_softmax(shifted_logits[:, start:, :], dim=-1)
        target_slice = shifted_targets[:, start:]

        gathered = token_log_probs.gather(dim=-1, index=target_slice.unsqueeze(-1)).squeeze(-1)
        return float(gathered.sum().item())

    def _sequence_logprobs(self, prompt: str, completions: list[str]) -> list[float]:
        if not completions:
            return []

        full_texts = [prompt + completion for completion in completions]
        enc_full = self.tokenizer(full_texts, return_tensors="pt", padding=True)
        enc_prompt = self.tokenizer(prompt, return_tensors="pt")

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]
        prompt_len = int(enc_prompt["input_ids"].shape[1])

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

        gathered = token_log_probs.gather(
            dim=-1,
            index=target_slice.unsqueeze(-1),
        ).squeeze(-1)
        return [float(value) for value in (gathered * mask_slice).sum(dim=-1).tolist()]

    def _policy_adjustment(
        self,
        *,
        name: str,
        task: TaskSpec,
        workload: Workload,
        policy: str,
        attempted_tools: set[str],
        last_status: str | None,
    ) -> float:
        schema_ok = supports_schema(workload.tools[name], task)
        schema_issue = str(last_status or "") in {
            "malformed_schema",
            "contract_drift",
            "invalid_schema",
            "unknown_tool",
        }

        score = 0.0
        score += 0.35 if schema_ok else -0.20

        if schema_issue and name in attempted_tools:
            score -= 0.60

        if policy == "schema_first_fallback":
            if schema_issue and schema_ok and name != task.primary_tool:
                score += 0.25
            if (not schema_issue) and name == task.primary_tool:
                score += 0.20
        if policy == "timeout_budget_early_abort":
            latency = float(workload.tools[name].base_latency_ms)
            score += max(-0.25, (450.0 - latency) / 1500.0)
        if policy == "naive_retry" and name == task.primary_tool:
            score += 0.10

        return score

    def choose_tool(
        self,
        *,
        task: TaskSpec,
        workload: Workload,
        policy: str,
        attempt_number: int,
        attempted_tools: set[str],
        last_status: str | None,
    ) -> str:
        if self.heuristic_policy_shortcuts and policy in {
            "schema_first_fallback",
            "timeout_budget_early_abort",
        }:
            return heuristic_pick(task, workload, attempted_tools, last_status)

        prompt = self._prompt(
            task=task,
            workload=workload,
            policy=policy,
            attempt_number=attempt_number,
            attempted_tools=attempted_tools,
            last_status=last_status,
        )

        candidates = self._candidate_names(task=task, workload=workload)
        if not candidates:
            return ""

        completions = [
            json.dumps({"tool_name": name}, ensure_ascii=True) for name in candidates
        ]
        model_scores = self._sequence_logprobs(prompt, completions)

        scored: list[tuple[float, str]] = []
        for model_score, name in zip(model_scores, candidates, strict=True):
            policy_adjustment = self.policy_adjustment_weight * self._policy_adjustment(
                name=name,
                task=task,
                workload=workload,
                policy=policy,
                attempted_tools=attempted_tools,
                last_status=last_status,
            )
            adjusted = model_score + policy_adjustment
            scored.append((adjusted, name))

        scored.sort(reverse=True)
        if scored:
            return scored[0][1]

        return heuristic_pick(task, workload, attempted_tools, last_status)

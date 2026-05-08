from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .models import TaskSpec, ToolSpec, Workload


SCHEMA_STATUSES = {
    "malformed_schema",
    "contract_drift",
    "invalid_schema",
    "unknown_tool",
}


class ToolPlanner(Protocol):
    planner_id: str

    def choose_tool(
        self,
        *,
        task: TaskSpec,
        workload: Workload,
        policy: str,
        attempt_number: int,
        attempted_tools: set[str],
        last_status: str | None,
        rng: random.Random,
    ) -> str: ...


def candidate_tools(task: TaskSpec, workload: Workload) -> list[str]:
    ordered = [task.primary_tool, *task.fallback_tools]
    return [name for name in ordered if name in workload.tools]


def supports_schema(tool: ToolSpec, task: TaskSpec) -> bool:
    return all(field in tool.schema_fields for field in task.required_schema)


def next_schema_compatible_tool(
    task: TaskSpec,
    workload: Workload,
    attempted: set[str],
) -> str | None:
    for name in candidate_tools(task, workload):
        if name in attempted:
            continue
        if supports_schema(workload.tools[name], task):
            return name
    return None


@dataclass
class PolicyNativePlanner:
    planner_id: str = "policy_native"

    def choose_tool(
        self,
        *,
        task: TaskSpec,
        workload: Workload,
        policy: str,
        attempt_number: int,
        attempted_tools: set[str],
        last_status: str | None,
        rng: random.Random,
    ) -> str:
        del attempt_number, rng
        options = candidate_tools(task, workload)
        if not options:
            return ""

        if policy == "schema_first_fallback" and last_status in SCHEMA_STATUSES:
            fallback = next_schema_compatible_tool(task, workload, attempted_tools)
            if fallback:
                return fallback

        if policy == "schema_first_fallback":
            for name in options:
                if supports_schema(workload.tools[name], task):
                    return name

        return options[0]


@dataclass
class HeuristicPlanner:
    planner_id: str = "heuristic"
    prefer_schema: bool = True
    avoid_reusing_attempted: bool = True

    def choose_tool(
        self,
        *,
        task: TaskSpec,
        workload: Workload,
        policy: str,
        attempt_number: int,
        attempted_tools: set[str],
        last_status: str | None,
        rng: random.Random,
    ) -> str:
        del policy, attempt_number, last_status, rng
        options = candidate_tools(task, workload)
        if not options:
            return ""

        ranked = []
        for name in options:
            tool = workload.tools[name]
            schema_score = 1.0 if supports_schema(tool, task) else 0.0
            speed_score = 1.0 / max(1.0, float(tool.base_latency_ms))
            reuse_penalty = (
                -0.6
                if (self.avoid_reusing_attempted and name in attempted_tools)
                else 0.0
            )
            score = (
                (schema_score * 2.0 if self.prefer_schema else schema_score * 0.5)
                + speed_score
                + reuse_penalty
            )
            ranked.append((score, name))

        ranked.sort(reverse=True)
        return ranked[0][1]


@dataclass
class MinimalPlanner:
    planner_id: str = "minimal"
    prefer_primary: bool = True

    def choose_tool(
        self,
        *,
        task: TaskSpec,
        workload: Workload,
        policy: str,
        attempt_number: int,
        attempted_tools: set[str],
        last_status: str | None,
        rng: random.Random,
    ) -> str:
        del policy, attempt_number, last_status, rng
        options = candidate_tools(task, workload)
        if not options:
            return ""
        if self.prefer_primary:
            return options[0]
        for name in options:
            if name not in attempted_tools:
                return name
        return options[0]


@dataclass
class StochasticPlanner:
    planner_id: str = "stochastic"
    off_catalog_probability: float = 0.0
    schema_bonus: float = 2.0
    primary_bonus: float = 1.0
    unexplored_bonus: float = 0.8
    hallucinated_tool_name: str = "hallucinated_tool"

    def choose_tool(
        self,
        *,
        task: TaskSpec,
        workload: Workload,
        policy: str,
        attempt_number: int,
        attempted_tools: set[str],
        last_status: str | None,
        rng: random.Random,
    ) -> str:
        del policy, attempt_number, last_status
        if rng.random() < max(0.0, min(1.0, self.off_catalog_probability)):
            return self.hallucinated_tool_name

        options = candidate_tools(task, workload)
        if not options:
            return self.hallucinated_tool_name

        weights: list[float] = []
        for name in options:
            tool = workload.tools[name]
            weight = 1.0
            if supports_schema(tool, task):
                weight += max(0.0, self.schema_bonus)
            if name == task.primary_tool:
                weight += max(0.0, self.primary_bonus)
            if name not in attempted_tools:
                weight += max(0.0, self.unexplored_bonus)
            weights.append(max(0.001, weight))

        total = sum(weights)
        roll = rng.random() * total
        cursor = 0.0
        for name, weight in zip(options, weights):
            cursor += weight
            if roll <= cursor:
                return name
        return options[-1]


@dataclass
class ReplayPlanner:
    planner_id: str = "replay"
    task_tool_sequence: dict[str, list[str]] = field(default_factory=dict)
    fallback: ToolPlanner = field(default_factory=PolicyNativePlanner)

    def choose_tool(
        self,
        *,
        task: TaskSpec,
        workload: Workload,
        policy: str,
        attempt_number: int,
        attempted_tools: set[str],
        last_status: str | None,
        rng: random.Random,
    ) -> str:
        sequence = self.task_tool_sequence.get(task.task_id, [])
        index = attempt_number - 1
        if 0 <= index < len(sequence):
            return sequence[index]
        return self.fallback.choose_tool(
            task=task,
            workload=workload,
            policy=policy,
            attempt_number=attempt_number,
            attempted_tools=attempted_tools,
            last_status=last_status,
            rng=rng,
        )


@dataclass
class CommandPlanner:
    planner_id: str = "command"
    command: str = ""
    timeout_seconds: float = 15.0
    strict_mode: bool = False
    fallback: ToolPlanner = field(default_factory=PolicyNativePlanner)

    def choose_tool(
        self,
        *,
        task: TaskSpec,
        workload: Workload,
        policy: str,
        attempt_number: int,
        attempted_tools: set[str],
        last_status: str | None,
        rng: random.Random,
    ) -> str:
        if not self.command.strip():
            if self.strict_mode:
                raise RuntimeError(
                    f"{self.planner_id}: command planner requires non-empty command in strict_mode"
                )
            return self.fallback.choose_tool(
                task=task,
                workload=workload,
                policy=policy,
                attempt_number=attempt_number,
                attempted_tools=attempted_tools,
                last_status=last_status,
                rng=rng,
            )

        payload = {
            "task": {
                "task_id": task.task_id,
                "primary_tool": task.primary_tool,
                "fallback_tools": task.fallback_tools,
                "required_schema": task.required_schema,
                "user_query": task.user_query,
            },
            "policy": policy,
            "attempt_number": attempt_number,
            "attempted_tools": sorted(attempted_tools),
            "last_status": last_status,
            "available_tools": sorted(workload.tools.keys()),
            "tool_catalog": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "schema_fields": list(tool.schema_fields),
                    "timeout_ms": tool.timeout_ms,
                }
                for tool in sorted(workload.tools.values(), key=lambda t: t.name)
            ],
        }

        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                shell=True,
                check=False,
                timeout=max(0.1, float(self.timeout_seconds)),
            )
        except Exception as exc:
            if self.strict_mode:
                raise RuntimeError(
                    f"{self.planner_id}: command execution failed in strict_mode"
                ) from exc
            return self.fallback.choose_tool(
                task=task,
                workload=workload,
                policy=policy,
                attempt_number=attempt_number,
                attempted_tools=attempted_tools,
                last_status=last_status,
                rng=rng,
            )

        if completed.returncode != 0:
            if self.strict_mode:
                stderr = (completed.stderr or "").strip()
                raise RuntimeError(
                    f"{self.planner_id}: command returned non-zero exit code "
                    f"{completed.returncode} in strict_mode. stderr={stderr}"
                )
            return self.fallback.choose_tool(
                task=task,
                workload=workload,
                policy=policy,
                attempt_number=attempt_number,
                attempted_tools=attempted_tools,
                last_status=last_status,
                rng=rng,
            )

        output = completed.stdout.strip()
        if not output:
            if self.strict_mode:
                raise RuntimeError(
                    f"{self.planner_id}: command produced empty output in strict_mode"
                )
            return self.fallback.choose_tool(
                task=task,
                workload=workload,
                policy=policy,
                attempt_number=attempt_number,
                attempted_tools=attempted_tools,
                last_status=last_status,
                rng=rng,
            )

        try:
            data = json.loads(output)
            tool_name = str(data.get("tool_name", "")).strip()
            if tool_name:
                return tool_name
        except json.JSONDecodeError:
            pass

        first_line = output.splitlines()[0].strip()
        if first_line:
            return first_line

        if self.strict_mode:
            raise RuntimeError(
                f"{self.planner_id}: unable to parse tool name from command output in strict_mode"
            )

        return self.fallback.choose_tool(
            task=task,
            workload=workload,
            policy=policy,
            attempt_number=attempt_number,
            attempted_tools=attempted_tools,
            last_status=last_status,
            rng=rng,
        )

@dataclass
class HFLocalPlanner:
    planner_id: str = "hf_local"
    base_model: str = ""
    adapter_path: str | None = None
    candidate_scope: str = "task"
    candidate_order: str = "task"
    policy_adjustment_weight: float = 1.0
    heuristic_policy_shortcuts: bool = True
    fallback: ToolPlanner = field(default_factory=PolicyNativePlanner)
    _core: object | None = field(default=None, init=False, repr=False)

    def _get_core(self):
        if self._core is None:
            model_name = str(self.base_model).strip()
            if not model_name:
                raise RuntimeError(
                    f"{self.planner_id}: base_model is required for hf_local planner"
                )
            from .hf_planner import HFLocalPlannerCore

            self._core = HFLocalPlannerCore(
                planner_id=self.planner_id,
                base_model_id=model_name,
                adapter_path=(str(self.adapter_path or "").strip() or None),
                candidate_scope=str(self.candidate_scope or "task").strip().lower(),
                candidate_order=str(self.candidate_order or "task").strip().lower(),
                policy_adjustment_weight=float(self.policy_adjustment_weight),
                heuristic_policy_shortcuts=bool(self.heuristic_policy_shortcuts),
            )
        return self._core

    def choose_tool(
        self,
        *,
        task: TaskSpec,
        workload: Workload,
        policy: str,
        attempt_number: int,
        attempted_tools: set[str],
        last_status: str | None,
        rng: random.Random,
    ) -> str:
        del rng
        core = self._get_core()
        return core.choose_tool(
            task=task,
            workload=workload,
            policy=policy,
            attempt_number=attempt_number,
            attempted_tools=attempted_tools,
            last_status=last_status,
        )


def planner_from_dict(payload: dict) -> ToolPlanner:
    planner_type = str(payload.get("type", "policy_native")).strip().lower()
    planner_id = str(payload.get("name", planner_type)).strip() or planner_type

    if planner_type == "policy_native":
        return PolicyNativePlanner(planner_id=planner_id)
    if planner_type == "heuristic":
        return HeuristicPlanner(
            planner_id=planner_id,
            prefer_schema=bool(payload.get("prefer_schema", True)),
            avoid_reusing_attempted=bool(payload.get("avoid_reusing_attempted", True)),
        )
    if planner_type == "minimal":
        return MinimalPlanner(
            planner_id=planner_id,
            prefer_primary=bool(payload.get("prefer_primary", True)),
        )
    if planner_type == "stochastic":
        return StochasticPlanner(
            planner_id=planner_id,
            off_catalog_probability=float(payload.get("off_catalog_probability", 0.0)),
            schema_bonus=float(payload.get("schema_bonus", 2.0)),
            primary_bonus=float(payload.get("primary_bonus", 1.0)),
            unexplored_bonus=float(payload.get("unexplored_bonus", 0.8)),
            hallucinated_tool_name=str(
                payload.get("hallucinated_tool_name", "hallucinated_tool")
            ),
        )
    if planner_type == "replay":
        return ReplayPlanner(
            planner_id=planner_id,
            task_tool_sequence={
                str(task_id): [str(tool) for tool in tools]
                for task_id, tools in dict(
                    payload.get("task_tool_sequence", {})
                ).items()
            },
            fallback=PolicyNativePlanner(),
        )
    if planner_type == "command":
        return CommandPlanner(
            planner_id=planner_id,
            command=str(payload.get("command", "")),
            timeout_seconds=float(payload.get("timeout_seconds", 15.0)),
            strict_mode=bool(payload.get("strict_mode", False)),
            fallback=PolicyNativePlanner(),
        )
    if planner_type == "hf_local":
        return HFLocalPlanner(
            planner_id=planner_id,
            base_model=str(payload.get("base_model", "")),
            adapter_path=(str(payload.get("adapter_path", "")).strip() or None),
            candidate_scope=str(payload.get("candidate_scope", "task")).strip().lower()
            or "task",
            candidate_order=str(payload.get("candidate_order", "task")).strip().lower()
            or "task",
            policy_adjustment_weight=float(payload.get("policy_adjustment_weight", 1.0)),
            heuristic_policy_shortcuts=bool(
                payload.get("heuristic_policy_shortcuts", True)
            ),
            fallback=PolicyNativePlanner(),
        )

    raise ValueError(f"Unsupported planner type: {planner_type}")


def load_tool_planner(path: str | Path | None) -> ToolPlanner:
    if path is None:
        return PolicyNativePlanner()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return planner_from_dict(payload)

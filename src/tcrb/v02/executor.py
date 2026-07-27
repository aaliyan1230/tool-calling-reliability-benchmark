from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent, parse_action
from .tools import ExecutableTool, build_executable_tools
from .types import (
    Abort,
    Clarify,
    DiagnosticLabel,
    EpisodeResult,
    EpisodeTrace,
    FaultSchedule,
    FinalAnswer,
    Observation,
    ObservationStatus,
    StepRecord,
    TaskDef,
    ToolCall,
    ToolDef,
)


MAX_STEPS = 5
MAX_TIME_BUDGET_MS = 120_000


@dataclass
class EpisodeConfig:
    max_steps: int = MAX_STEPS
    max_time_budget_ms: int = MAX_TIME_BUDGET_MS
    seed: int = 42


def _inject_fault(
    status: ObservationStatus,
    correct_payload: Any,
    fault_type: str,
    tool_name: str,
    rng: random.Random,
) -> tuple[ObservationStatus, Any]:
    if fault_type == "timeout":
        return ("timeout", None)
    if fault_type == "rate_limit":
        return ("rate_limit", {"error": "rate_limit_exceeded", "retry_after_ms": rng.randint(1000, 5000)})
    if fault_type == "schema_drift":
        if isinstance(correct_payload, dict):
            drifted = dict(correct_payload)
            drifted["_extra_field"] = "unexpected_value"
            drifted.pop("found", None)
            return ("schema_drift", drifted)
        return ("schema_drift", correct_payload)
    if fault_type == "partial_output":
        if isinstance(correct_payload, dict):
            partial = {k: v for i, (k, v) in enumerate(correct_payload.items()) if i == 0 or rng.random() < 0.3}
            return ("partial_output", partial)
        return ("partial_output", correct_payload)
    if fault_type == "silent_corruption":
        if isinstance(correct_payload, dict):
            corrupted = dict(correct_payload)
            if "balance" in corrupted and isinstance(corrupted["balance"], (int, float)):
                corrupted["balance"] = corrupted["balance"] * rng.uniform(0.5, 0.95)
            elif "status" in corrupted:
                statuses = ["shipped", "processing", "cancelled", "delivered"]
                corrupted["status"] = rng.choice([s for s in statuses if s != corrupted.get("status")])
            return ("success", corrupted)
        return (status, correct_payload)
    if fault_type == "cross_source_conflict":
        if isinstance(correct_payload, dict):
            conflicted = dict(correct_payload)
            conflicted["_conflict_note"] = "Warning: data may conflict with source system"
            conflicted["_alternate_source"] = {"value": "conflicting_data"}
            return ("cross_source_conflict", conflicted)
        return ("cross_source_conflict", correct_payload)
    if fault_type == "execution_error":
        return ("execution_error", {"error": f"Internal server error in {tool_name}", "code": rng.choice([500, 502, 503])})

    return (status, correct_payload)


def _validate_action(
    action: AgentAction,
    available_tools: dict[str, ToolDef],
) -> tuple[bool, str | None]:
    if not isinstance(action, ToolCall):
        return True, None

    if action.name not in available_tools:
        return False, f"tool '{action.name}' not in available tools"

    tool = available_tools[action.name]
    required = tool.input_schema.get("required", [])
    missing = [r for r in required if r not in action.arguments]
    if missing:
        return False, f"missing required arguments: {missing}"

    return True, None


def _classify_diagnostics(
    steps: list[StepRecord],
    used_tools: set[str],
    available_tools: set[str],
) -> list[DiagnosticLabel]:
    labels: list[DiagnosticLabel] = []

    tool_steps = [s for s in steps if isinstance(s.parsed_action, ToolCall)]
    non_tool_actions = [s for s in steps if s.parsed_action is not None and not isinstance(s.parsed_action, ToolCall)]

    if not tool_steps and not any(isinstance(s.parsed_action, FinalAnswer) for s in steps):
        labels.append("tool_skip")

    if len(tool_steps) > 5:
        labels.append("unnecessary_tool_use")

    tool_results = [s.observation for s in tool_steps if s.observation is not None]
    if tool_results and not any(r.status == "success" for r in tool_results):
        labels.append("retry_loop")

    error_results = [r for r in tool_results if r.is_error]
    if error_results:
        retried_errors = set()
        for s in tool_steps:
            if isinstance(s.parsed_action, ToolCall) and s.observation and s.observation.is_error:
                retried_errors.add(s.parsed_action.name)
        if len(retried_errors) == 0 and len(error_results) > 0:
            labels.append("silent_result_trust")

    for s in tool_steps:
        if s.parse_error:
            labels.append("unexecutable_call")
            break

    for s in tool_steps:
        if isinstance(s.parsed_action, ToolCall):
            valid, _ = _validate_action(s.parsed_action, {})
            if not valid:
                labels.append("argument_invalid")
                break

    if any(isinstance(s.parsed_action, Abort) for s in steps) and not any(
        isinstance(s.parsed_action, FinalAnswer) for s in steps
    ):
        labels.append("premature_stop")

    for s in tool_steps:
        if s.observation and s.observation.status == "success" and s.observation.payload is not None:
            payload = s.observation.payload
            if isinstance(payload, dict) and "_conflict_note" in payload:
                labels.append("silent_result_trust")
                break

    if not labels:
        has_tool = any(isinstance(s.parsed_action, ToolCall) for s in steps)
        has_answer = any(isinstance(s.parsed_action, FinalAnswer) for s in steps)
        if not has_tool and not has_answer:
            labels.append("tool_skip")
        elif has_answer and not has_tool:
            labels.append("fabrication")

    final = []
    seen = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            final.append(label)
    return final


def run_episode(
    agent: Agent,
    task: TaskDef,
    tool_defs: dict[str, ToolDef],
    fault_schedules: list[FaultSchedule] | None = None,
    config: EpisodeConfig | None = None,
) -> EpisodeTrace:
    cfg = config or EpisodeConfig()
    rng = random.Random(cfg.seed)
    start_time = time.time()

    executable_tools = {
        name: ExecutableTool(definition=td)
        for name, td in tool_defs.items()
        if name in task.available_tools
    }

    fault_by_step: dict[int, FaultSchedule] = {}
    if fault_schedules:
        for fs in fault_schedules:
            if fs.task_id == task.task_id:
                fault_by_step[fs.step_index] = fs

    steps: list[StepRecord] = []
    history: list[tuple[AgentAction | None, Observation | None]] = []
    used_tools: set[str] = set()
    success = False
    final_response: str | None = None

    for step_idx in range(cfg.max_steps):
        elapsed = int((time.time() - start_time) * 1000)
        if elapsed > cfg.max_time_budget_ms:
            break

        available_defs = [tool_defs[n] for n in task.available_tools if n in tool_defs]

        step_start = time.time()
        raw_output = ""
        parsed_action: AgentAction | None = None
        parse_error: str | None = None

        try:
            parsed_action = agent.next_action(
                task_query=task.user_query,
                available_tools=available_defs,
                history=history,
                rng=rng,
            )
            if isinstance(parsed_action, ToolCall):
                raw_output = json.dumps({"name": parsed_action.name, "arguments": parsed_action.arguments})
            elif isinstance(parsed_action, FinalAnswer):
                raw_output = json.dumps({"final_answer": parsed_action.text})
            elif isinstance(parsed_action, Clarify):
                raw_output = json.dumps({"clarify": parsed_action.text})
            elif isinstance(parsed_action, Abort):
                raw_output = json.dumps({"abort": parsed_action.reason})
        except Exception as exc:
            parse_error = str(exc)

        step_timing = int((time.time() - step_start) * 1000)

        if parse_error:
            steps.append(StepRecord(
                step_index=step_idx,
                raw_model_output=raw_output,
                parsed_action=None,
                parse_error=parse_error,
                observation=None,
                timing_ms=step_timing,
            ))
            history.append((None, None))
            continue

        observation: Observation | None = None

        if isinstance(parsed_action, ToolCall):
            valid, err = _validate_action(parsed_action, {n: tool_defs[n] for n in tool_defs})
            if not valid:
                observation = Observation(
                    status="invalid_arguments",
                    payload={"error": err},
                    latency_ms=0,
                )
            else:
                tool = executable_tools.get(parsed_action.name)
                if tool is None:
                    observation = Observation(
                        status="unknown_tool",
                        payload={"error": f"Tool '{parsed_action.name}' not available"},
                        latency_ms=0,
                    )
                else:
                    used_tools.add(parsed_action.name)
                    fault_schedule = fault_by_step.get(step_idx)
                    try:
                        payload = tool.execute(parsed_action.arguments, rng)
                        status: ObservationStatus = "success"
                        if fault_schedule:
                            status, payload = _inject_fault(
                                "success", payload,
                                fault_schedule.fault_type,
                                parsed_action.name,
                                rng,
                            )
                        observation = Observation(
                            status=status,
                            payload=payload,
                            latency_ms=tool.definition.base_latency_ms + rng.randint(-tool.definition.jitter_ms, tool.definition.jitter_ms),
                            call_id=parsed_action.call_id,
                        )
                    except Exception as exc:
                        observation = Observation(
                            status="execution_error",
                            payload={"error": str(exc)},
                            latency_ms=0,
                            call_id=parsed_action.call_id,
                        )

        elif isinstance(parsed_action, FinalAnswer):
            success = True
            final_response = parsed_action.text
            steps.append(StepRecord(
                step_index=step_idx,
                raw_model_output=raw_output,
                parsed_action=parsed_action,
                parse_error=None,
                observation=None,
                timing_ms=step_timing,
            ))
            history.append((parsed_action, None))
            break

        elif isinstance(parsed_action, Clarify):
            observation = Observation(
                status="success",
                payload={"clarify": parsed_action.text},
                latency_ms=0,
            )

        elif isinstance(parsed_action, Abort):
            steps.append(StepRecord(
                step_index=step_idx,
                raw_model_output=raw_output,
                parsed_action=parsed_action,
                parse_error=None,
                observation=None,
                timing_ms=step_timing,
            ))
            history.append((parsed_action, None))
            break

        steps.append(StepRecord(
            step_index=step_idx,
            raw_model_output=raw_output,
            parsed_action=parsed_action,
            parse_error=None,
            observation=observation,
            timing_ms=step_timing,
        ))
        history.append((parsed_action, observation))

        if observation and observation.status not in ("success", "partial_output", "cross_source_conflict", "schema_drift"):
            if observation.status not in ("timeout", "rate_limit", "execution_error"):
                break

    total_time = int((time.time() - start_time) * 1000)
    diagnostic_labels = _classify_diagnostics(steps, used_tools, set(task.available_tools))

    return EpisodeTrace(
        task_id=task.task_id,
        domain=task.domain,
        steps=steps,
        final_response=final_response,
        success=success,
        diagnostic_labels=diagnostic_labels,
        total_time_ms=total_time,
    )


def evaluate_trace(trace: EpisodeTrace, task: TaskDef) -> EpisodeResult:
    calls_made = sum(1 for s in trace.steps if isinstance(s.parsed_action, ToolCall))

    recovery_steps = 0
    for s in trace.steps:
        if s.observation and s.observation.is_error and isinstance(s.parsed_action, ToolCall):
            recovery_steps += 1

    return EpisodeResult(
        task_id=task.task_id,
        domain=task.domain,
        success=trace.success,
        fault_type=None,
        diagnostic_labels=trace.diagnostic_labels,
        steps=len(trace.steps),
        total_tokens=trace.total_tokens,
        total_cost_usd=trace.total_cost_usd,
        total_time_ms=trace.total_time_ms,
        calls_made=calls_made,
        recovery_steps=recovery_steps,
    )

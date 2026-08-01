#!/usr/bin/env python3
"""Build recovery-focused SFT and DPO records from TCRB v0.2 traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tcrb.research import render_sft_text
from tcrb.v02.agent import RECOVERY_SYSTEM_PROMPT
from tcrb.v02.agent import _format_tools_for_prompt
from tcrb.v02.tools import TOOL_REGISTRY


DEFAULT_TRAINING_HAZARDS = {"execution_error", "schema_drift", "partial_output"}


def _action_payload(action: dict[str, Any] | None) -> str:
    if not action:
        return json.dumps({"final_answer": "Task completed."}, ensure_ascii=True)
    action_type = action.get("type")
    if action_type == "tool_call":
        return json.dumps(
            {"name": action["name"], "arguments": action.get("arguments", {})},
            ensure_ascii=True,
        )
    if action_type == "final_answer":
        return json.dumps({"final_answer": action.get("text", "")}, ensure_ascii=True)
    if action_type == "clarify":
        return json.dumps({"clarify": action.get("text", "")}, ensure_ascii=True)
    if action_type == "abort":
        return json.dumps({"abort": action.get("reason", "")}, ensure_ascii=True)
    return json.dumps({"final_answer": "Task completed."}, ensure_ascii=True)


def _tool_definitions(tool_names: list[str]) -> list[dict[str, Any]]:
    return [
        TOOL_REGISTRY[name].to_openai_function()
        for name in tool_names
        if name in TOOL_REGISTRY
    ]


def _failed_step(trace: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]] | None:
    for index, step in enumerate(trace.get("steps", [])):
        observation = step.get("observation")
        if observation and observation.get("status") != "success":
            action = step.get("action")
            if action and action.get("type") == "tool_call":
                return index, action, observation
    return None


def _evaluated_success(trace: dict[str, Any]) -> bool:
    if trace.get("success") is not True:
        return False
    claims = [str(claim).lower() for claim in trace.get("canonical_claims", [])]
    response = str(trace.get("final_response", "")).lower()
    return not claims or all(claim in response for claim in claims)


def _recovery_context(trace: dict[str, Any], action: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    tool_defs = [
        TOOL_REGISTRY[name]
        for name in trace.get("available_tools", [])
        if name in TOOL_REGISTRY
    ]
    messages = [
        {
            "role": "system",
            "content": RECOVERY_SYSTEM_PROMPT
            + "\n\nAvailable tools:\n"
            + _format_tools_for_prompt(tool_defs),
        },
        {"role": "user", "content": trace.get("task_query", "")},
        {"role": "assistant", "content": _action_payload(action)},
        {
            "role": "user",
            "content": "Tool result: "
            + json.dumps(
                {"status": observation.get("status"), "result": observation.get("payload")},
                ensure_ascii=True,
            ),
        },
    ]
    return {
        "messages": messages,
        "tools": _tool_definitions(trace.get("available_tools", [])),
    }


def build_recovery_rows(
    traces: list[dict[str, Any]],
    *,
    splits: set[str] | None = None,
    hazards: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed_splits = splits if splits is not None else {"train"}
    allowed_hazards = hazards if hazards is not None else DEFAULT_TRAINING_HAZARDS
    sft_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []
    for trace in traces:
        trace_hazard = str(trace.get("hazard", ""))
        if trace.get("split") not in allowed_splits or trace_hazard not in allowed_hazards:
            continue
        if trace.get("fault_applied") is False:
            continue
        if not _evaluated_success(trace):
            continue
        failed = _failed_step(trace)
        if failed is None:
            continue
        step_index, failed_action, observation = failed
        failed_action_payload = _action_payload(failed_action)
        context = _recovery_context(trace, failed_action, observation)

        next_action = None
        if step_index + 1 < len(trace.get("steps", [])):
            next_action = trace["steps"][step_index + 1].get("action")
        if next_action is None:
            continue
        chosen = _action_payload(next_action)

        sft_messages = context["messages"] + [{"role": "assistant", "content": chosen}]
        sft_record = {
            "messages": sft_messages,
            "tools": context["tools"],
            "text": render_sft_text({"messages": sft_messages, "tools": context["tools"]}),
            "source": "tcrb_v02_recovery",
            "metadata": {
                "task_id": trace.get("task_id"),
                "hazard": trace_hazard,
                "fault_step": step_index,
            },
        }
        sft_rows.append(sft_record)

        if chosen == failed_action_payload:
            continue
        dpo_rows.append(
            {
                "prompt": render_sft_text(context),
                "chosen": chosen,
                "rejected": failed_action_payload,
                "source": "tcrb_v02_recovery",
                "metadata": sft_record["metadata"],
            }
        )
    return sft_rows, dpo_rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--output-sft", required=True, type=Path)
    parser.add_argument("--output-dpo", required=True, type=Path)
    parser.add_argument("--splits", default="train")
    parser.add_argument("--hazards", default=",".join(sorted(DEFAULT_TRAINING_HAZARDS)))
    args = parser.parse_args()

    traces = json.loads(args.traces.read_text(encoding="utf-8"))
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    hazards = {value.strip() for value in args.hazards.split(",") if value.strip()}
    sft_rows, dpo_rows = build_recovery_rows(traces, splits=splits, hazards=hazards)
    _write_jsonl(sft_rows, args.output_sft)
    _write_jsonl(dpo_rows, args.output_dpo)
    print(json.dumps({"sft_rows": len(sft_rows), "dpo_rows": len(dpo_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

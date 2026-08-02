#!/usr/bin/env python3
"""Build leakage-safe recovery supervision from v0.2 oracle trajectories."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from tcrb.v02.executor import _inject_fault
from tcrb.research import render_sft_text
from tcrb.v02.agent import RECOVERY_SYSTEM_PROMPT
from tcrb.v02.agent import _format_tools_for_prompt
from tcrb.v02.tasks import (
    build_all_tasks,
    generate_fault_schedules,
    get_oracle_actions,
    get_split,
)
from tcrb.v02.tools import ExecutableTool, TOOL_REGISTRY
from tcrb.v02.types import ToolCall


TRAINING_HAZARDS = ("execution_error", "schema_drift", "partial_output")
TRAINING_DOMAINS = {"customer_support", "ecommerce"}


def _action_payload(action: dict[str, Any]) -> str:
    if action.get("type") == "tool_call":
        return json.dumps(
            {"name": action["name"], "arguments": action.get("arguments", {})},
            ensure_ascii=True,
        )
    return json.dumps({"final_answer": action.get("text", "")}, ensure_ascii=True)


def _recovery_context(
    task: dict[str, Any], action: dict[str, Any], observation: dict[str, Any]
) -> dict[str, Any]:
    tool_defs = [
        TOOL_REGISTRY[name]
        for name in task["available_tools"]
        if name in TOOL_REGISTRY
    ]
    messages = [
        {
            "role": "system",
            "content": RECOVERY_SYSTEM_PROMPT
            + "\n\nAvailable tools:\n"
            + _format_tools_for_prompt(tool_defs),
        },
        {"role": "user", "content": task["task_query"]},
        {"role": "assistant", "content": _action_payload(action)},
        {
            "role": "user",
            "content": "Tool result: "
            + json.dumps(
                {"status": observation["status"], "result": observation["payload"]},
                ensure_ascii=True,
            ),
        },
    ]
    return {
        "messages": messages,
        "tools": [
            TOOL_REGISTRY[name].to_openai_function()
            for name in task["available_tools"]
            if name in TOOL_REGISTRY
        ],
    }


def _observation_for(action: ToolCall, hazard: str, seed: int) -> dict[str, Any]:
    tool = ExecutableTool(definition=TOOL_REGISTRY[action.name])
    rng = random.Random(seed)
    payload = tool.execute(action.arguments, rng)
    status, payload = _inject_fault("success", payload, hazard, action.name, rng)
    return {"status": status, "payload": payload}


def build_oracle_recovery_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sft_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []

    for domain, tasks in sorted(build_all_tasks().items()):
        if domain not in TRAINING_DOMAINS:
            continue
        for task in tasks:
            if get_split(task) != "train" or task.category != "tool_required":
                continue

            oracle = get_oracle_actions(task)
            for fault_idx, hazard in enumerate(TRAINING_HAZARDS):
                schedule = generate_fault_schedules(task, fault_idx)[0]
                failed_action = oracle[schedule.step_index]
                if not isinstance(failed_action, ToolCall):
                    continue

                observation = _observation_for(
                    failed_action,
                    hazard,
                    seed=42 + fault_idx,
                )
                trace = {
                    "task_query": task.user_query,
                    "available_tools": list(task.available_tools),
                }
                context = _recovery_context(trace, {
                    "type": "tool_call",
                    "name": failed_action.name,
                    "arguments": failed_action.arguments,
                }, observation)
                chosen = _action_payload({
                    "type": "tool_call",
                    "name": failed_action.name,
                    "arguments": failed_action.arguments,
                })
                rejected = _action_payload({
                    "type": "final_answer",
                    "text": "I cannot verify the requested information from this result.",
                })
                metadata = {
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "split": get_split(task),
                    "hazard": hazard,
                    "fault_step": schedule.step_index,
                    "failed_tool": failed_action.name,
                    "source": "tcrb_v02_oracle",
                }

                sft_messages = context["messages"] + [{"role": "assistant", "content": chosen}]
                sft_rows.append({
                    "messages": sft_messages,
                    "tools": context["tools"],
                    "text": render_sft_text({"messages": sft_messages, "tools": context["tools"]}),
                    "source": "tcrb_v02_oracle",
                    "metadata": metadata,
                })
                dpo_rows.append({
                    "prompt": render_sft_text(context),
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": "tcrb_v02_oracle",
                    "metadata": metadata,
                })

    return sft_rows, dpo_rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-sft", required=True, type=Path)
    parser.add_argument("--output-dpo", required=True, type=Path)
    args = parser.parse_args()

    sft_rows, dpo_rows = build_oracle_recovery_rows()
    _write_jsonl(sft_rows, args.output_sft)
    _write_jsonl(dpo_rows, args.output_dpo)
    print(json.dumps({"sft_rows": len(sft_rows), "dpo_rows": len(dpo_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

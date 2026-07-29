#!/usr/bin/env python3
"""Run a v0.2 agent against the full TCRB v0.2 benchmark.

Usage:
    python -m tcrb.v02.eval_runner \
        --model Qwen/Qwen3-4B \
        --output-dir ./runs/v02_baseline \
        --seed 42 \
        --max-tasks 0 \
        --domains customer_support,ecommerce,fintech,developer_tools \
        --clean-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .agent import HFAgent, LogProbAgent, resolve_system_prompt
from .gemini_agent import GeminiReviewerAgent
from .executor import EpisodeConfig, run_episode
from .tasks import (
    build_all_tasks,
    generate_fault_schedules,
    get_faulted_tasks,
    get_oracle_actions,
    get_split,
)
from .tools import TOOL_REGISTRY
from .types import (
    Clarify,
    DiagnosticLabel,
    EpisodeTrace,
    FinalAnswer,
    TaskDef,
    ToolCall,
    reset_call_counter,
)


def _load_model(model_id: str, adapter_path: str | None = None) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def _assess_claims(trace: EpisodeTrace, claims: list[str]) -> bool:
    if not claims:
        return trace.success
    if not trace.final_response:
        return False
    response_lower = trace.final_response.lower()
    return all(c.lower() in response_lower for c in claims)


def _contains_ordered_sequence(actual: list[str], expected: list[str]) -> bool:
    remaining = iter(actual)
    return all(any(candidate == tool for candidate in remaining) for tool in expected)


def _episode_success(trace: EpisodeTrace, task: TaskDef) -> bool:
    """Apply category-specific task semantics to an episode trace."""
    tool_names = [
        step.parsed_action.name
        for step in trace.steps
        if isinstance(step.parsed_action, ToolCall)
    ]
    if task.category == "missing_information":
        return not tool_names and any(
            isinstance(step.parsed_action, Clarify) for step in trace.steps
        )

    claims_pass = _assess_claims(trace, task.canonical_claims)
    if task.category == "no_tool":
        return not tool_names and trace.success and claims_pass

    if not trace.success or not claims_pass or not tool_names:
        return False
    if not task.valid_tool_sequences:
        return True
    return any(
        _contains_ordered_sequence(tool_names, expected)
        for expected in task.valid_tool_sequences
    )


def _count_labels(traces: list[EpisodeTrace]) -> dict[DiagnosticLabel, int]:
    counts: dict[DiagnosticLabel, int] = {}
    for trace in traces:
        for label in trace.diagnostic_labels:
            counts[label] = counts.get(label, 0) + 1
    return counts


def run_eval(
    model_id: str,
    domains: list[str],
    output_dir: str | Path,
    seed: int = 42,
    max_tasks: int = 0,
    clean_only: bool = False,
    system_prompt: str | None = None,
    agent_type: str = "logprob",
    prompt_variant: str = "default",
    adapter_path: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_id}", flush=True)
    model, tokenizer = _load_model(model_id, adapter_path=adapter_path)

    effective_system_prompt = resolve_system_prompt(system_prompt, prompt_variant)

    if agent_type == "gemini_reviewer":
        agent = GeminiReviewerAgent(
            agent_id="gemini_reviewer",
            model=model,
            tokenizer=tokenizer,
            system_prompt=effective_system_prompt,
        )
    elif agent_type == "hf_generate":
        agent = HFAgent(
            agent_id="hf_generate",
            model=model,
            tokenizer=tokenizer,
            system_prompt=effective_system_prompt,
        )
    else:
        agent = LogProbAgent(
            agent_id="baseline",
            model=model,
            tokenizer=tokenizer,
            system_prompt=effective_system_prompt,
        )

    all_tasks = build_all_tasks()
    requested = {d: all_tasks[d] for d in domains if d in all_tasks}

    total_tasks = sum(len(v) for v in requested.values())
    print(f"Total tasks: {total_tasks} across {len(requested)} domains", flush=True)

    results: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    start_time = time.time()

    results_path = output_path / "results.json"
    traces_path = output_path / "traces.json"
    summary_path = output_path / "summary.json"

    def _save_incremental() -> None:
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        traces_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")

    task_index = 0
    for domain, tasks in sorted(requested.items()):
        for task in tasks:
            if max_tasks > 0 and task_index >= max_tasks:
                break
            task_index += 1

            # ── Clean run ──
            reset_call_counter()
            try:
                trace = run_episode(
                    agent=agent,
                    task=task,
                    tool_defs=TOOL_REGISTRY,
                    config=EpisodeConfig(seed=seed, validate_arguments=False),
                )
            except Exception as exc:
                trace = EpisodeTrace(
                    task_id=task.task_id,
                    domain=task.domain,
                    success=False,
                    diagnostic_labels=["tool_skip"],
                )
                trace.final_response = str(exc)

            claim_pass = _assess_claims(trace, task.canonical_claims)
            clean_success = _episode_success(trace, task)

            # Serialize trace for replay
            trace_data = {
                "task_id": trace.task_id,
                "domain": trace.domain,
                "task_query": task.user_query,
                "available_tools": list(task.available_tools),
                "category": task.category,
                "canonical_claims": task.canonical_claims,
                "success": trace.success,
                "final_response": trace.final_response,
                "diagnostic_labels": list(trace.diagnostic_labels),
                "total_time_ms": trace.total_time_ms,
                "steps": [],
            }
            for step in trace.steps:
                step_data = {
                    "step_index": step.step_index,
                    "raw_model_output": step.raw_model_output,
                    "parse_error": step.parse_error,
                    "timing_ms": step.timing_ms,
                }
                if step.parsed_action is not None:
                    if isinstance(step.parsed_action, ToolCall):
                        step_data["action"] = {"type": "tool_call", "name": step.parsed_action.name, "arguments": step.parsed_action.arguments}
                    elif isinstance(step.parsed_action, FinalAnswer):
                        step_data["action"] = {"type": "final_answer", "text": step.parsed_action.text}
                    elif isinstance(step.parsed_action, Clarify):
                        step_data["action"] = {"type": "clarify", "text": step.parsed_action.text}
                    elif isinstance(step.parsed_action, Abort):
                        step_data["action"] = {"type": "abort", "reason": step.parsed_action.reason}
                if step.observation is not None:
                    step_data["observation"] = {
                        "status": step.observation.status,
                        "payload": step.observation.payload,
                        "latency_ms": step.observation.latency_ms,
                    }
                trace_data["steps"].append(step_data)
            traces.append(trace_data)

            record = {
                "task_id": task.task_id,
                "domain": task.domain,
                "category": task.category,
                "split": get_split(task),
                "clean": {
                    "success": clean_success,
                    "claim_pass": claim_pass,
                    "final_response": trace.final_response,
                    "steps": len(trace.steps),
                    "diagnostic_labels": trace.diagnostic_labels,
                    "total_time_ms": trace.total_time_ms,
                },
                "faulted": [],
            }

            # ── Faulted runs ──
            if not clean_only and task.category == "tool_required":
                for fault_idx in range(5):
                    reset_call_counter()
                    schedules = generate_fault_schedules(task, fault_idx)
                    try:
                        ftrace = run_episode(
                            agent=agent,
                            task=task,
                            tool_defs=TOOL_REGISTRY,
                            fault_schedules=schedules,
                            config=EpisodeConfig(seed=seed, validate_arguments=False),
                        )
                    except Exception:
                        ftrace = EpisodeTrace(
                            task_id=task.task_id,
                            domain=task.domain,
                            success=False,
                            diagnostic_labels=["tool_skip"],
                        )

                    fclaim_pass = _assess_claims(ftrace, task.canonical_claims)
                    fsuccess = _episode_success(ftrace, task)

                    # Serialize faulted trace
                    ftrace_data = {
                        "task_id": ftrace.task_id,
                        "domain": ftrace.domain,
                        "task_query": task.user_query,
                        "available_tools": list(task.available_tools),
                        "category": task.category,
                        "canonical_claims": task.canonical_claims,
                        "fault_idx": fault_idx,
                        "success": ftrace.success,
                        "final_response": ftrace.final_response,
                        "diagnostic_labels": list(ftrace.diagnostic_labels),
                        "total_time_ms": ftrace.total_time_ms,
                        "steps": [],
                    }
                    for step in ftrace.steps:
                        step_data = {
                            "step_index": step.step_index,
                            "raw_model_output": step.raw_model_output,
                            "parse_error": step.parse_error,
                            "timing_ms": step.timing_ms,
                        }
                        if step.parsed_action is not None:
                            if isinstance(step.parsed_action, ToolCall):
                                step_data["action"] = {"type": "tool_call", "name": step.parsed_action.name, "arguments": step.parsed_action.arguments}
                            elif isinstance(step.parsed_action, FinalAnswer):
                                step_data["action"] = {"type": "final_answer", "text": step.parsed_action.text}
                            elif isinstance(step.parsed_action, Clarify):
                                step_data["action"] = {"type": "clarify", "text": step.parsed_action.text}
                            elif isinstance(step.parsed_action, Abort):
                                step_data["action"] = {"type": "abort", "reason": step.parsed_action.reason}
                        if step.observation is not None:
                            step_data["observation"] = {
                                "status": step.observation.status,
                                "payload": step.observation.payload,
                                "latency_ms": step.observation.latency_ms,
                            }
                        ftrace_data["steps"].append(step_data)
                    traces.append(ftrace_data)

                    hazard = schedules[0].fault_type if schedules else "unknown"
                    record["faulted"].append({
                        "fault_idx": fault_idx,
                        "hazard": hazard,
                        "success": fsuccess,
                        "claim_pass": fclaim_pass,
                        "final_response": ftrace.final_response,
                        "steps": len(ftrace.steps),
                        "diagnostic_labels": ftrace.diagnostic_labels,
                        "total_time_ms": ftrace.total_time_ms,
                    })

            results.append(record)

            _save_incremental()

            elapsed = time.time() - start_time
            rate = task_index / elapsed if elapsed > 0 else 0
            print(
                f"[{task_index}/{total_tasks}] {task.task_id} "
                f"clean={'PASS' if clean_success else 'FAIL'} "
                f"({elapsed:.0f}s, {rate:.1f} tasks/s)",
                flush=True,
            )

    total_time = time.time() - start_time

    # ── Aggregate metrics ──
    clean_results = [r for r in results]
    clean_passed = sum(1 for r in clean_results if r["clean"]["success"])
    clean_total = len(clean_results)

    by_split = {}
    for r in clean_results:
        s = r["split"]
        by_split.setdefault(s, {"total": 0, "passed": 0})
        by_split[s]["total"] += 1
        if r["clean"]["success"]:
            by_split[s]["passed"] += 1

    by_domain = {}
    for r in clean_results:
        d = r["domain"]
        by_domain.setdefault(d, {"total": 0, "passed": 0})
        by_domain[d]["total"] += 1
        if r["clean"]["success"]:
            by_domain[d]["passed"] += 1

    by_category = {}
    for r in clean_results:
        c = r["category"]
        by_category.setdefault(c, {"total": 0, "passed": 0})
        by_category[c]["total"] += 1
        if r["clean"]["success"]:
            by_category[c]["passed"] += 1

    fault_results = []
    if not clean_only:
        for r in results:
            for f in r.get("faulted", []):
                fault_results.append({
                    "task_id": r["task_id"],
                    "domain": r["domain"],
                    "hazard": f["hazard"],
                    "success": f["success"],
                    "clean_success": r["clean"]["success"],
                    "recovery": f["success"] and not r["clean"]["success"],
                })

        fault_total = len(fault_results)
        fault_passed = sum(1 for f in fault_results if f["success"])

        by_hazard = {}
        for f in fault_results:
            h = f["hazard"]
            by_hazard.setdefault(h, {"total": 0, "passed": 0, "recovery": 0})
            by_hazard[h]["total"] += 1
            if f["success"]:
                by_hazard[h]["passed"] += 1
            if f["recovery"]:
                by_hazard[h]["recovery"] += 1
    else:
        fault_total = 0
        fault_passed = 0
        by_hazard = {}

    all_traces = [EpisodeTrace(task_id="", domain="", success=False)]
    diagnostic_counts = _count_labels(all_traces)

    summary = {
        "model_id": model_id,
        "seed": seed,
        "total_time_s": round(total_time, 1),
        "clean_only": clean_only,
        "clean": {
            "total": clean_total,
            "passed": clean_passed,
            "rate": round(clean_passed / clean_total, 4) if clean_total else 0,
            "by_split": {s: {"total": d["total"], "passed": d["passed"],
                             "rate": round(d["passed"] / d["total"], 4) if d["total"] else 0}
                          for s, d in sorted(by_split.items())},
            "by_domain": {s: {"total": d["total"], "passed": d["passed"],
                              "rate": round(d["passed"] / d["total"], 4) if d["total"] else 0}
                          for s, d in sorted(by_domain.items())},
            "by_category": {s: {"total": d["total"], "passed": d["passed"],
                                "rate": round(d["passed"] / d["total"], 4) if d["total"] else 0}
                            for s, d in sorted(by_category.items())},
        },
        "faulted": {
            "total": fault_total,
            "passed": fault_passed,
            "rate": round(fault_passed / fault_total, 4) if fault_total else 0,
            "by_hazard": {h: {"total": d["total"], "passed": d["passed"],
                              "rate": round(d["passed"] / d["total"], 4) if d["total"] else 0,
                              "recovery": d["recovery"]}
                          for h, d in sorted(by_hazard.items())},
        } if not clean_only else None,
        "diagnostic_counts": diagnostic_counts,
    }

    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nResults written to {output_path}/", flush=True)
    print(f"\nClean: {clean_passed}/{clean_total} ({100*clean_passed/clean_total:.1f}%)", flush=True)
    if not clean_only:
        print(f"Faulted: {fault_passed}/{fault_total} ({100*fault_passed/fault_total:.1f}%)", flush=True)
    print(json.dumps(summary["clean"]["by_domain"], indent=2), flush=True)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="TCRB v0.2 Evaluation Runner")
    parser.add_argument("--model", required=True, help="HF model ID, e.g. Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=0, help="Max tasks to run (0=all)")
    parser.add_argument("--domains", default="customer_support,ecommerce,fintech,developer_tools",
                        help="Comma-separated domain names")
    parser.add_argument("--clean-only", action="store_true", help="Run clean evaluation only")
    parser.add_argument("--system-prompt", default=None, help="Optional custom system prompt")
    parser.add_argument(
        "--prompt-variant",
        choices=["default", "recovery"],
        default="default",
        help="Built-in system prompt variant used when --system-prompt is omitted",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional PEFT adapter directory to evaluate",
    )
    parser.add_argument("--agent-type", default="logprob", choices=["logprob", "hf_generate", "gemini_reviewer"],
                        help="Agent type to use (default: logprob)")

    args = parser.parse_args()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    try:
        summary = run_eval(
            model_id=args.model,
            domains=domains,
            output_dir=args.output_dir,
            seed=args.seed,
            max_tasks=args.max_tasks or 0,
            clean_only=args.clean_only,
            system_prompt=args.system_prompt,
            agent_type=args.agent_type,
            prompt_variant=args.prompt_variant,
            adapter_path=args.adapter_path,
        )
        return 0 if summary["clean"]["rate"] >= 0 else 1
    except Exception as exc:
        print(f"FATAL: {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

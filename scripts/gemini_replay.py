#!/usr/bin/env python3
"""Local Gemini trace replay system.

Downloads traces from Kaggle and replays them with Gemini as the decision-maker.
Measures how much the agent would improve with Gemini guidance.

Usage:
    python scripts/gemini_replay.py \
        --traces-file runs/v02_baseline_qwen3_4b/traces.json \
        --output-dir runs/v02_gemini_replay
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


REVIEW_SYSTEM_PROMPT = """You are a tool-calling recovery expert. You will see an agent's trace - a sequence of tool calls and observations. Your job is to evaluate each step and suggest what the agent SHOULD have done.

For each step, output a JSON object:
{"step": N, "evaluation": "GOOD"|"BAD"|"OK", "suggested_action": {...}, "reason": "..."}

Where suggested_action is:
- {"type": "tool_call", "name": "tool_name", "arguments": {...}}
- {"type": "final_answer", "text": "..."}
- {"type": "clarify", "text": "..."}
- {"type": "abort", "reason": "..."}

Evaluation criteria:
- GOOD: The action was correct and led to progress
- BAD: The action was wrong (wrong tool, wrong args, ignored error)
- OK: The action was acceptable but not optimal

Rules:
- If observation has error status (timeout, rate_limit, execution_error), the agent should RETRY or FALLBACK
- If observation has schema_drift or partial_output, the agent should adapt and continue
- If observation has silent_corruption or cross_source_conflict, the agent should CLARIFY or verify
- If the agent is stuck in a loop (same tool called 3+ times), it should ABORT or try something different
- The agent should ANSWER with tool results, not fabricate

Be specific in your suggestions. Include exact tool names and arguments."""


def load_traces(traces_file: str) -> list[dict[str, Any]]:
    with open(traces_file) as f:
        return json.load(f)


def replay_trace_with_gemini(
    client: genai.Client,
    model: str,
    trace: dict[str, Any],
    task_query: str,
    available_tools: list[str],
) -> dict[str, Any]:
    """Replay a single trace with Gemini review."""
    
    steps_text = []
    for step in trace["steps"]:
        step_text = f"Step {step['step_index']}:\n"
        if step.get("action"):
            action = step["action"]
            if action["type"] == "tool_call":
                step_text += f"  Action: {action['name']}({json.dumps(action['arguments'])})\n"
            elif action["type"] == "final_answer":
                step_text += f"  Action: final_answer({action['text']})\n"
            elif action["type"] == "clarify":
                step_text += f"  Action: clarify({action['text']})\n"
            elif action["type"] == "abort":
                step_text += f"  Action: abort({action['reason']})\n"
        
        if step.get("observation"):
            obs = step["observation"]
            step_text += f"  Observation: status={obs['status']}, payload={json.dumps(obs['payload'])}\n"
        
        steps_text.append(step_text)
    
    prompt = f"""{REVIEW_SYSTEM_PROMPT}

Task: {task_query}
Available tools: {', '.join(available_tools)}
Actual outcome: {"SUCCESS" if trace['success'] else "FAILURE"}

Trace:
{''.join(steps_text)}

Evaluate each step and suggest improvements. Output a JSON array of evaluations:"""
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2048,
            ),
        )
        result_text = response.text.strip()
        
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        evaluations = json.loads(result_text)
        return {"evaluations": evaluations, "raw_response": result_text}
    
    except Exception as e:
        print(f"[GeminiReplay] Error: {e}", flush=True)
        return {"evaluations": [], "raw_response": "", "error": str(e)}


def compute_replay_metrics(trace: dict, replay_result: dict) -> dict[str, Any]:
    """Compute metrics from the replay."""
    evaluations = replay_result.get("evaluations", [])
    
    total_steps = len(trace["steps"])
    good_steps = sum(1 for e in evaluations if e.get("evaluation") == "GOOD")
    bad_steps = sum(1 for e in evaluations if e.get("evaluation") == "BAD")
    
    # Count how many bad steps had actionable suggestions
    actionable_suggestions = sum(
        1 for e in evaluations
        if e.get("evaluation") == "BAD" and e.get("suggested_action")
    )
    
    return {
        "task_id": trace["task_id"],
        "domain": trace["domain"],
        "original_success": trace["success"],
        "total_steps": total_steps,
        "good_steps": good_steps,
        "bad_steps": bad_steps,
        "actionable_suggestions": actionable_suggestions,
        "quality_score": good_steps / total_steps if total_steps > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-traces", type=int, default=0, help="0=all")
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found", flush=True)
        return 1
    
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"
    
    print(f"Loading traces from {args.traces_file}", flush=True)
    traces = load_traces(args.traces_file)
    print(f"Loaded {len(traces)} traces", flush=True)
    
    if args.max_traces > 0:
        traces = traces[:args.max_traces]
    
    # For now, we don't have task_query or available_tools in the trace
    # We'll need to add them to the trace serialization or look them up
    # For the replay, we'll just use the trace data we have
    
    results = []
    start_time = time.time()
    
    for i, trace in enumerate(traces):
        task_id = trace["task_id"]
        domain = trace["domain"]
        
        print(f"[{i+1}/{len(traces)}] Replaying {task_id}...", flush=True)
        
        # Simplified: just evaluate the trace as-is
        replay_result = replay_trace_with_gemini(
            client=client,
            model=model,
            trace=trace,
            task_query=f"Task {task_id}",  # TODO: add real query to traces
            available_tools=[],  # TODO: add tools to traces
        )
        
        metrics = compute_replay_metrics(trace, replay_result)
        results.append({
            "trace": trace,
            "replay": replay_result,
            "metrics": metrics,
        })
        
        # Save incrementally
        (output_path / "replay_results.json").write_text(
            json.dumps(results, indent=2)
        )
    
    total_time = time.time() - start_time
    
    # Compute aggregate metrics
    total_traces = len(results)
    avg_quality = sum(r["metrics"]["quality_score"] for r in results) / total_traces
    total_bad_steps = sum(r["metrics"]["bad_steps"] for r in results)
    total_actionable = sum(r["metrics"]["actionable_suggestions"] for r in results)
    
    summary = {
        "total_traces": total_traces,
        "total_time_s": total_time,
        "avg_quality_score": avg_quality,
        "total_bad_steps": total_bad_steps,
        "total_actionable_suggestions": total_actionable,
        "by_domain": {},
    }
    
    # Group by domain
    by_domain = {}
    for r in results:
        domain = r["trace"]["domain"]
        by_domain.setdefault(domain, {"count": 0, "quality_sum": 0, "bad_steps": 0})
        by_domain[domain]["count"] += 1
        by_domain[domain]["quality_sum"] += r["metrics"]["quality_score"]
        by_domain[domain]["bad_steps"] += r["metrics"]["bad_steps"]
    
    for domain, stats in by_domain.items():
        summary["by_domain"][domain] = {
            "count": stats["count"],
            "avg_quality": stats["quality_sum"] / stats["count"],
            "bad_steps": stats["bad_steps"],
        }
    
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2))
    
    print(f"\nDone in {total_time:.1f}s", flush=True)
    print(f"Avg quality: {avg_quality:.2%}", flush=True)
    print(f"Bad steps: {total_bad_steps}, Actionable: {total_actionable}", flush=True)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

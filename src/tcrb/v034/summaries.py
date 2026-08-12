from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .schema import PROFILES, SUMMARY_KEYS, validate_summary
from .util import CONFIG_ROOT, DEFAULT_LOCAL_ROOT, DEFAULT_RUN_ROOT, append_jsonl, content_id, env_value, read_json, read_jsonl, sha256_bytes, write_json, write_jsonl


FORBIDDEN_INPUT_KEYS = {"reward", "db_match", "gold_outcome", "expected_actions", "reference_actions", "label", "candidate_role", "policy", "task_snapshot"}


def load_summary_config() -> dict[str, Any]:
    return read_json(CONFIG_ROOT / "summary_profiles.json")


def prompt_text(profile: str) -> str:
    if profile not in PROFILES:
        raise ValueError(f"unknown summary profile: {profile}")
    return (CONFIG_ROOT / "prompts" / f"{profile}.md").read_text(encoding="utf-8").strip()


def source_input(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {"domain": trajectory["domain"], "trajectory_id": trajectory["trajectory_id"], "events": trajectory["events"]}


def action_receipt(trajectory: dict[str, Any]) -> dict[str, Any]:
    events = {event["event_id"]: event for event in trajectory["events"]}
    writes: list[dict[str, Any]] = []
    for call_id in trajectory.get("write_event_ids", []):
        call = events[call_id]
        result = next((event for event in trajectory["events"] if event.get("call_event_id") == call_id), None)
        writes.append({"call_event_id": call_id, "tool": call["tool_call"]["name"], "arguments": call["tool_call"]["arguments"], "result_event_id": result["event_id"] if result else None, "result": result.get("tool_result") if result else None})
    return {"writes": writes}


def build_user_prompt(trajectory: dict[str, Any]) -> str:
    return "SOURCE TRACE (event IDs are the only citation IDs you may use):\n" + json.dumps(source_input(trajectory), sort_keys=True, ensure_ascii=False)


def parse_summary(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("summary response was not valid JSON")


class ProviderError(RuntimeError):
    pass


def call_summary(provider: str, profile: str, trajectory: dict[str, Any], timeout_s: int = 120, max_retries: int = 4) -> dict[str, Any]:
    system = prompt_text(profile) + "\n\nReturn exactly this JSON shape: {\"user_request\": string, \"key_facts\": [{\"text\": string, \"source_event_ids\": [string]}], \"actions_and_results\": [{\"text\": string, \"source_event_ids\": [string]}], \"state_changes\": [{\"text\": string, \"source_event_ids\": [string]}], \"unresolved_or_risks\": [{\"text\": string, \"source_event_ids\": [string]}]}"
    user = build_user_prompt(trajectory)
    if provider == "gpt":
        body = {"model": "gpt-5.6-terra", "input": [{"role": "system", "content": system}, {"role": "user", "content": user}], "reasoning": {"effort": "high"}, "temperature": 0, "max_output_tokens": 600, "text": {"format": {"type": "json_schema", "name": "trajectory_summary", "strict": True, "schema": summary_json_schema()}}}
        response = post_json("https://api.openai.com/v1/responses", body, env_value("OPENAI_API_KEY"), timeout_s, max_retries)
        output = extract_openai_text(response)
        usage_raw = response.get("usage") or {}
        input_tokens = int(usage_raw.get("input_tokens", 0) or 0)
        output_tokens = int(usage_raw.get("output_tokens", 0) or 0)
        cached = int((usage_raw.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0)
        cost = (max(0, input_tokens - cached) * 2.0 + cached * 0.2 + output_tokens * 12.0) / 1_000_000
        return {"provider": "openai", "model_id": "gpt-5.6-terra", "output_text": output, "raw_response": response, "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "cached_input_tokens": cached}, "estimated_extra_cost_usd": cost}
    if provider == "deepseek":
        body = {"model": "deepseek-v4-flash", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "reasoning_effort": "high", "temperature": 0, "max_tokens": 600, "response_format": {"type": "json_object"}}
        response = post_json("https://opencode.ai/zen/go/v1/chat/completions", body, env_value("OPENCODE_API_KEY"), timeout_s, max_retries)
        try:
            output = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("DeepSeek response did not contain message content") from exc
        usage_raw = response.get("usage") or {}
        return {"provider": "opencode_go", "model_id": "deepseek-v4-flash", "output_text": output, "raw_response": response, "usage": {"input_tokens": int(usage_raw.get("prompt_tokens", 0) or 0), "output_tokens": int(usage_raw.get("completion_tokens", 0) or 0), "cached_input_tokens": int(usage_raw.get("prompt_cache_hit_tokens", 0) or 0)}, "estimated_extra_cost_usd": 0.0}
    raise ValueError(f"unknown provider {provider}")


def summary_json_schema() -> dict[str, Any]:
    item = {"type": "object", "additionalProperties": False, "properties": {"text": {"type": "string"}, "source_event_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["text", "source_event_ids"]}
    return {"type": "object", "additionalProperties": False, "properties": {"user_request": {"type": "string"}, **{key: {"type": "array", "items": item} for key in SUMMARY_KEYS[1:]}}, "required": list(SUMMARY_KEYS)}


def post_json(url: str, body: dict[str, Any], api_key: str | None, timeout_s: int, max_retries: int) -> dict[str, Any]:
    if not api_key:
        raise ProviderError("API key not found in environment or .env")
    encoded = json.dumps(body, separators=(",", ":")).encode()
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=encoded, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "tcrb-v034-summary-pipeline/1.0"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                value = json.loads(response.read().decode())
            if not isinstance(value, dict):
                raise ProviderError("provider response was not an object")
            return value
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
            except Exception:
                detail = "<unreadable HTTP error body>"
            last = ProviderError(f"HTTP {exc.code}: {detail}")
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ProviderError) as exc:
            last = exc
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
    raise ProviderError(f"provider request failed after {max_retries + 1} attempts: {last}")


def extract_openai_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str) and response["output_text"]:
        return response["output_text"]
    parts: list[str] = []
    for item in response.get("output") or []:
        for content in item.get("content") or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                parts.append(content["text"])
    if not parts:
        raise ProviderError("OpenAI response did not contain output text")
    return "".join(parts)


def call_matrix(local_root: Path = DEFAULT_LOCAL_ROOT, run_root: Path = DEFAULT_RUN_ROOT, stage: str = "core", providers: tuple[str, ...] = ("deepseek", "gpt")) -> list[dict[str, Any]]:
    trajectories = {row["trajectory_id"]: row for row in read_jsonl(local_root / "normalized" / "trajectories.jsonl")}
    if stage == "smoke":
        pairs = read_jsonl(local_root / "dev_pairs_private.jsonl")[:4]
        samples = [0]
    elif stage == "core":
        pairs = read_jsonl(local_root / "frozen_pairs_private.jsonl")
        samples = [0]
    elif stage == "stability":
        pairs = read_jsonl(local_root / "frozen_pairs_private.jsonl")[:10]
        samples = [1, 2, 3]
    else:
        raise ValueError(f"unknown stage {stage}")
    trajectory_ids = sorted({pair[key] for pair in pairs for key in ("safe_candidate_id", "unsafe_candidate_id")})
    specs: list[dict[str, Any]] = []
    for trajectory_id in trajectory_ids:
        trajectory = trajectories[trajectory_id]
        input_hash = sha256_bytes(json.dumps(source_input(trajectory), sort_keys=True, separators=(",", ":")).encode())
        for profile in PROFILES:
            prompt_hash = sha256_bytes(prompt_text(profile).encode())
            for provider in providers:
                for sample_index in samples:
                    specs.append({"call_id": content_id({"trajectory_id": trajectory_id, "profile": profile, "provider": provider, "sample_index": sample_index, "input_hash": input_hash, "prompt_hash": prompt_hash}, "call_"), "trajectory_id": trajectory_id, "profile": profile, "provider": provider, "model_id": load_summary_config()["models"][provider], "sample_index": sample_index, "input_hash": input_hash, "prompt_hash": prompt_hash})
    return specs


def estimate_matrix(local_root: Path, specs: list[dict[str, Any]]) -> dict[str, Any]:
    trajectories = {row["trajectory_id"]: row for row in read_jsonl(local_root / "normalized" / "trajectories.jsonl")}
    tokens = sum(len(json.dumps(source_input(trajectories[spec["trajectory_id"]]), ensure_ascii=False)) // 4 for spec in specs)
    gpt_calls = sum(spec["provider"] == "gpt" for spec in specs)
    estimated = (tokens * 2.0 + gpt_calls * 600 * 12.0) / 1_000_000
    return {"calls": len(specs), "gpt_calls": gpt_calls, "estimated_input_tokens": tokens, "estimated_gpt_cost_usd": estimated}


def run_stage(local_root: Path = DEFAULT_LOCAL_ROOT, run_root: Path = DEFAULT_RUN_ROOT, stage: str = "core", providers: tuple[str, ...] = ("deepseek", "gpt"), spend_cap_usd: float = 25.0, call_fn: Callable[..., dict[str, Any]] = call_summary) -> dict[str, Any]:
    run_dir = run_root / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    specs = call_matrix(local_root, run_root, stage, providers)
    estimate = estimate_matrix(local_root, specs)
    if estimate["estimated_gpt_cost_usd"] > spend_cap_usd:
        raise RuntimeError(f"estimated GPT cost ${estimate['estimated_gpt_cost_usd']:.2f} exceeds cap ${spend_cap_usd:.2f}")
    trajectories = {row["trajectory_id"]: row for row in read_jsonl(local_root / "normalized" / "trajectories.jsonl")}
    response_path = run_dir / "summary_responses.jsonl"
    existing = {row.get("call_id"): row for row in read_jsonl(response_path)}
    spent = sum(float(row.get("estimated_extra_cost_usd", 0) or 0) for row in existing.values() if row.get("status") == "success")
    completed = skipped = failed = 0
    for spec in specs:
        cached = existing.get(spec["call_id"])
        if cached and cached.get("status") == "success" and cached.get("summary"):
            skipped += 1
            continue
        if spent >= spend_cap_usd and spec["provider"] == "gpt":
            raise RuntimeError(f"GPT spend cap reached at ${spent:.2f}")
        trajectory = trajectories[spec["trajectory_id"]]
        try:
            result = call_fn(spec["provider"], spec["profile"], trajectory)
            parsed = parse_summary(result["output_text"])
            errors = validate_summary(parsed, {event["event_id"] for event in trajectory["events"]})
            record = {**spec, "status": "success", "summary": parsed, "validation_errors": errors, "output_text": result["output_text"], "raw_response": result["raw_response"], "usage": result["usage"], "estimated_extra_cost_usd": result["estimated_extra_cost_usd"]}
            append_jsonl(response_path, record)
            spent += float(result["estimated_extra_cost_usd"] or 0)
            completed += 1
        except Exception as exc:
            append_jsonl(response_path, {**spec, "status": "error", "error_type": type(exc).__name__, "error": str(exc), "estimated_extra_cost_usd": 0.0})
            failed += 1
    summary = {"version": "v034-summary-run-1", "stage": stage, "scheduled": len(specs), "completed_now": completed, "skipped_cached": skipped, "failed_now": failed, "estimated_extra_cost_usd": spent, "spend_cap_usd": spend_cap_usd, "matrix_estimate": estimate}
    write_json(run_dir / "run_summary.json", summary)
    return summary


def build_views(local_root: Path = DEFAULT_LOCAL_ROOT, run_root: Path = DEFAULT_RUN_ROOT, stage: str = "core") -> dict[str, Any]:
    run_dir = run_root / stage
    trajectories = {row["trajectory_id"]: row for row in read_jsonl(local_root / "normalized" / "trajectories.jsonl")}
    responses = read_jsonl(run_dir / "summary_responses.jsonl")
    views: list[dict[str, Any]] = []
    for trajectory_id, trajectory in trajectories.items():
        views.append({"view_id": content_id({"trajectory": trajectory_id, "type": "full_trace"}, "view_"), "trajectory_id": trajectory_id, "view_type": "full_trace", "events": trajectory["events"]})
    for row in responses:
        if row.get("status") != "success":
            continue
        receipt = action_receipt(trajectories[row["trajectory_id"]])
        base = {"trajectory_id": row["trajectory_id"], "profile": row["profile"], "provider": row["provider"], "model_id": row["model_id"], "sample_index": row["sample_index"], "summary": row["summary"]}
        views.append({"view_id": content_id({**base, "type": "summary"}, "view_"), **base, "view_type": "summary"})
        views.append({"view_id": content_id({**base, "type": "summary_plus_receipt"}, "view_"), **base, "view_type": "summary_plus_receipt", "action_receipt": receipt})
    write_jsonl(run_dir / "monitor_views.jsonl", views)
    write_json(run_dir / "view_manifest.json", {"version": "v034-views-1", "view_count": len(views), "full_trace_count": len(trajectories), "summary_response_count": len(responses)})
    return {"views": len(views), "full_traces": len(trajectories), "summaries": len(responses)}

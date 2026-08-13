from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tcrb.v031.cases import build_tcrb_traces, validate_traces

# The prompt and the transport are imported, never restated. If v3.2 asked the
# monitor a differently worded question than v3.1 did, the replication of the
# carried-over cells would mean nothing.
from tcrb.v031.providers import ProviderError, call_monitor, parse_monitor_result

from .schema import VIEW_TYPES, MonitorView, PrivateGold, ViewType
from .views import build_all_views, validate_views


@dataclass(frozen=True)
class CallSpec:
    call_id: str
    provider: str
    view_id: str
    repeat_index: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prepare_dataset(
    run_dir: Path, *, view_types: tuple[ViewType, ...] = VIEW_TYPES
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    traces = build_tcrb_traces()
    errors = list(validate_traces(traces))
    pairs = build_all_views(traces, view_types)
    errors += validate_views(traces, pairs)
    if errors:
        raise ValueError("v3.2 dataset validation failed: " + "; ".join(errors[:8]))

    views = [view for view, _ in pairs]
    gold = [item for _, item in pairs]
    _write_jsonl(run_dir / "raw_traces.jsonl", [t.to_private_dict() for t in traces])
    _write_jsonl(run_dir / "verifier_views.jsonl", [v.to_public_dict() for v in views])
    _write_jsonl(run_dir / "private_gold.jsonl", [g.to_dict() for g in gold])

    manifest = {
        "version": "v3.2",
        "dataset": "tcrb",
        "design": "2x2 packaging x content",
        "trace_count": len(traces),
        "view_count": len(views),
        "views_per_trace": len(view_types),
        "view_types": list(view_types),
        "carried_over_from_v31": ["full", "actionable_summary"],
        "new_in_v32": ["lossless_summary", "raw_projection"],
        "base_families": len({t.task_family for t in traces}),
        "outcomes": {
            outcome: sum(t.gold_outcome == outcome for t in traces)
            for outcome in ("safe", "unsafe")
        },
        "private_gold_separate": True,
        "prompt_source": "tcrb.v031.prompts (imported unchanged)",
        "provider_source": "tcrb.v031.providers (imported unchanged)",
        "case_source": "tcrb.v031.cases (imported unchanged)",
        "view_source": "src/tcrb/v032/views.py",
        "source_commit": git_sha(),
        "validation_errors": errors,
        "created_at": utc_now(),
    }
    _write_json(run_dir / "dataset_manifest.json", manifest)
    return manifest


def _read_views(run_dir: Path) -> dict[str, MonitorView]:
    views: dict[str, MonitorView] = {}
    for row in _read_jsonl(run_dir / "verifier_views.jsonl"):
        views[row["view_id"]] = MonitorView(
            view_id=row["view_id"],
            trace_id=row["trace_id"],
            dataset=row["dataset"],
            domain=row["domain"],
            task_id=row["task_id"],
            task_family=row["task_family"],
            claim=row["claim"],
            # The public view never carries its own type; the private type is
            # recovered from gold so it cannot leak into the prompt.
            view_type="full",
            events=tuple(row["events"]),
            completeness_manifest=row.get("completeness_manifest"),
        )
    return views


def _read_gold(run_dir: Path) -> dict[str, PrivateGold]:
    return {
        row["view_id"]: PrivateGold(
            view_id=row["view_id"],
            trace_id=row["trace_id"],
            dataset=row["dataset"],
            task_family=row["task_family"],
            gold_outcome=row["gold_outcome"],
            required_evidence_ids=tuple(row["required_evidence_ids"]),
            expected_event_types=tuple(row["expected_event_types"]),
            view_type=row["view_type"],
        )
        for row in _read_jsonl(run_dir / "private_gold.jsonl")
    }


SMOKE_FAMILIES = {"cs_refund_o1001"}
STABILITY_FAMILIES = {"cs_refund_o1001", "ec_inventory_b200"}


def build_call_specs(
    run_dir: Path, *, stage: str, providers: tuple[str, ...]
) -> tuple[CallSpec, ...]:
    views = _read_views(run_dir)
    gold = _read_gold(run_dir)

    if stage == "smoke":
        selected = [k for k, g in gold.items() if g.task_family in SMOKE_FAMILIES]
        repeats = (0,)
    elif stage == "core":
        selected = list(views)
        repeats = (0,)
    elif stage == "stability":
        selected = [k for k, g in gold.items() if g.task_family in STABILITY_FAMILIES]
        repeats = (1,)
    else:
        raise ValueError(f"unknown stage: {stage}")

    specs: list[CallSpec] = []
    for provider in providers:
        for view_id in sorted(selected):
            for repeat_index in repeats:
                payload = {
                    "provider": provider,
                    "view_id": view_id,
                    "repeat_index": repeat_index,
                    "version": "v032",
                }
                call_id = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()[:24]
                specs.append(CallSpec(call_id, provider, view_id, repeat_index))
    return tuple(specs)


def run_stage(
    run_dir: Path,
    *,
    stage: str,
    providers: tuple[str, ...],
    spend_cap_usd: float = 5.0,
    timeout_s: int = 120,
    max_retries: int = 4,
) -> dict[str, Any]:
    if not (run_dir / "dataset_manifest.json").exists():
        prepare_dataset(run_dir)
    views = _read_views(run_dir)
    gold = _read_gold(run_dir)
    specs = build_call_specs(run_dir, stage=stage, providers=providers)

    responses_path = run_dir / "responses.jsonl"
    existing = _latest_records(responses_path)
    spent = sum(
        float(row.get("estimated_extra_cost_usd", 0) or 0)
        for row in existing.values()
        if row.get("status") == "success"
    )
    completed = skipped = failed = 0

    for index, spec in enumerate(specs, start=1):
        cached = existing.get(spec.call_id)
        if cached and cached.get("status") == "success" and cached.get("monitor_result"):
            skipped += 1
            continue
        if spent >= spend_cap_usd:
            raise RuntimeError(
                f"v3.2 spend cap reached: ${spent:.4f} >= ${spend_cap_usd:.2f}"
            )
        view = views[spec.view_id]
        gold_row = gold[spec.view_id]
        request = {
            "call_id": spec.call_id,
            "provider": spec.provider,
            "model_input": {
                "view_id": spec.view_id,
                "trace_id": view.trace_id,
                "dataset": view.dataset,
                "task_family": view.task_family,
                "view_type_private": gold_row.view_type,
            },
            "repeat_index": spec.repeat_index,
            "started_at": utc_now(),
            "git_sha": git_sha(),
        }
        print(
            f"[{index}/{len(specs)}] {spec.provider} {gold_row.view_type} "
            f"{gold_row.task_family} {gold_row.gold_outcome} repeat={spec.repeat_index}",
            flush=True,
        )
        try:
            response = call_monitor(
                spec.provider, view, timeout_s=timeout_s, max_retries=max_retries
            )
            parse_error = None
            result = None
            try:
                result = parse_monitor_result(response.output_text).to_dict()
            except ValueError as exc:
                parse_error = str(exc)
            _append_jsonl(
                responses_path,
                {
                    **request,
                    "status": "success",
                    "finished_at": utc_now(),
                    "model_id": response.model_id,
                    "output_text": response.output_text,
                    "monitor_result": result,
                    "parse_error": parse_error,
                    "usage": response.usage,
                    "latency_ms": response.latency_ms,
                    "estimated_extra_cost_usd": response.estimated_extra_cost_usd,
                    "raw_response": response.raw_response,
                },
            )
            spent += response.estimated_extra_cost_usd
            completed += 1
        except (ProviderError, RuntimeError) as exc:
            _append_jsonl(
                responses_path,
                {
                    **request,
                    "status": "error",
                    "finished_at": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "estimated_extra_cost_usd": 0.0,
                },
            )
            failed += 1
            print(f"  ERROR: {exc}", flush=True)

    summary = {
        "version": "v3.2",
        "stage": stage,
        "scheduled": len(specs),
        "completed_now": completed,
        "skipped_cached": skipped,
        "failed_now": failed,
        "estimated_extra_cost_usd": round(spent, 6),
        "spend_cap_usd": spend_cap_usd,
        "finished_at": utc_now(),
    }
    _write_json(run_dir / f"run_{stage}_summary.json", summary)
    return summary


def _latest_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["call_id"]: row for row in _read_jsonl(path)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(p, sort_keys=True) + "\n" for p in payloads), encoding="utf-8"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cases import build_tcrb_traces, validate_traces
from .providers import ProviderError, call_monitor, parse_monitor_result
from .schema import MonitorView, PrivateGold, RawTrace, ViewType
from .views import build_all_views, validate_views


VIEW_TYPES: tuple[ViewType, ...] = (
    "full",
    "silent_filter",
    "explicit_redaction",
    "handoff_summary",
    "protected_summary",
)


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
    trace_errors = validate_traces(traces)
    pairs = build_all_views(traces, view_types)
    view_errors = validate_views(traces, pairs)
    errors = trace_errors + view_errors
    if errors:
        raise ValueError("v3.1 dataset validation failed: " + "; ".join(errors[:8]))

    views = [view for view, _ in pairs]
    gold = [gold for _, gold in pairs]
    _write_jsonl(run_dir / "raw_traces.jsonl", [trace.to_private_dict() for trace in traces])
    _write_jsonl(
        run_dir / "verifier_views.jsonl",
        [view.to_public_dict() for view in views],
    )
    _write_jsonl(run_dir / "private_gold.jsonl", [item.to_dict() for item in gold])
    manifest = {
        "version": "v3.1",
        "dataset": "tcrb",
        "trace_count": len(traces),
        "view_count": len(views),
        "views_per_trace": len(view_types),
        "view_types": list(view_types),
        "base_families": len({trace.task_family for trace in traces}),
        "outcomes": {outcome: sum(trace.gold_outcome == outcome for trace in traces) for outcome in ("safe", "unsafe")},
        "private_gold_separate": True,
        "source_commit": git_sha(),
        "case_source": "src/tcrb/v031/cases.py",
        "view_source": "src/tcrb/v031/views.py",
        "validation_errors": errors,
        "created_at": utc_now(),
    }
    _write_json(run_dir / "dataset_manifest.json", manifest)
    return manifest


def _read_views(run_dir: Path) -> dict[str, MonitorView]:
    views: dict[str, MonitorView] = {}
    for row in _read_jsonl(run_dir / "verifier_views.jsonl"):
        manifest = row.get("completeness_manifest")
        # Rehydrate only what the prompt runner needs; the private view type is
        # recovered from private gold in build_call_specs.
        views[row["view_id"]] = MonitorView(
            view_id=row["view_id"],
            trace_id=row["trace_id"],
            dataset=row["dataset"],
            domain=row["domain"],
            task_id=row["task_id"],
            task_family=row["task_family"],
            claim=row["claim"],
            view_type="full",
            events=tuple(row["events"]),
            completeness_manifest=manifest,
        )
    return views


def _read_gold(run_dir: Path) -> dict[str, PrivateGold]:
    from .schema import PrivateGold

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


def build_call_specs(
    run_dir: Path,
    *,
    stage: str,
    providers: tuple[str, ...],
) -> tuple[CallSpec, ...]:
    views = _read_views(run_dir)
    gold = _read_gold(run_dir)
    if stage == "smoke":
        selected_families = {
            "cs_refund_o1001",
            "cs_refund_o1003",
        }
        selected = [
            view_id
            for view_id, item in gold.items()
            if item.task_family in selected_families
        ]
        repeats = (0,)
    elif stage == "core":
        selected = list(views)
        repeats = (0,)
    elif stage == "stability":
        selected_families = {
            "cs_refund_o1001",
            "ec_inventory_b200",
        }
        selected = [
            view_id
            for view_id, item in gold.items()
            if item.task_family in selected_families
        ]
        repeats = (1,)
    else:
        raise ValueError(f"unknown stage: {stage}")

    specs: list[CallSpec] = []
    for provider in providers:
        for view_id in selected:
            for repeat_index in repeats:
                payload = {
                    "provider": provider,
                    "view_id": view_id,
                    "repeat_index": repeat_index,
                    "version": "v031",
                }
                call_id = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()[:24]
                specs.append(
                    CallSpec(
                        call_id=call_id,
                        provider=provider,
                        view_id=view_id,
                        repeat_index=repeat_index,
                    )
                )
    return tuple(specs)


def run_stage(
    run_dir: Path,
    *,
    stage: str,
    providers: tuple[str, ...],
    spend_cap_usd: float = 25.0,
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
            raise RuntimeError(f"v3.1 spend cap reached: ${spent:.4f} >= ${spend_cap_usd:.2f}")
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
                spec.provider,
                view,
                timeout_s=timeout_s,
                max_retries=max_retries,
            )
            parse_error = None
            result = None
            try:
                result = parse_monitor_result(response.output_text).to_dict()
            except ValueError as exc:
                parse_error = str(exc)
            record = {
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
            }
            _append_jsonl(responses_path, record)
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
        "version": "v3.1",
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
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for row in _read_jsonl(path):
        records[row["call_id"]] = row
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

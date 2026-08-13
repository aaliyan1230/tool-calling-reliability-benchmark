from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tcrb.v031.cases import build_tcrb_traces, validate_traces
from tcrb.v031.prompts import SYSTEM_PROMPT

from .runner import build_call_specs
from .schema import CONTENT, PACKAGING
from .views import (
    build_all_views,
    critical_values_by_event,
    exposed_values_by_event,
    validate_views,
)


EXPECTED_MODELS = {
    "deepseek": "deepseek-v4-flash",
    "gpt": "gpt-5.6-terra",
}

# The v3.1 prompt, pinned. v3.2 compares its own `full` and `actionable_summary`
# numbers against v3.1's; if the question changed, that comparison is invalid.
V31_SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()

SPEND_CAP_USD = 25.0


def audit_run(
    run_dir: Path, *, providers: tuple[str, ...] = ("deepseek", "gpt")
) -> dict[str, Any]:
    traces = build_tcrb_traces()
    manifest = (
        _read_json(run_dir / "dataset_manifest.json")
        if (run_dir / "dataset_manifest.json").exists()
        else {}
    )
    view_types = tuple(manifest.get("view_types", []))
    pairs = build_all_views(traces, view_types)
    public = _read_jsonl(run_dir / "verifier_views.jsonl")
    private = _read_jsonl(run_dir / "private_gold.jsonl")
    responses = _read_optional_jsonl(run_dir / "responses.jsonl")
    scores = _read_optional_jsonl(run_dir / "scores.jsonl")

    checks: dict[str, bool] = {
        "trace_validation_passed": not validate_traces(traces),
        "view_validation_passed": not validate_views(traces, pairs),
        "public_private_counts_match": len(public) == len(private) == len(pairs),
        "private_gold_separate": not any(
            _recursive_keys(row)
            & {
                "gold_outcome",
                "required_evidence_ids",
                "mutation_id",
                "faults_applied",
                "diagnostic_labels",
                "payload_state",
            }
            for row in public
        ),
        "view_type_never_shown_to_monitor": not any(
            "view_type" in _recursive_keys(row) for row in public
        ),
        "public_view_ids_unique": len({r.get("view_id") for r in public}) == len(public),
        "prompt_matches_v31": V31_SYSTEM_PROMPT_SHA256
        == hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }

    # The claim the whole run rests on: the factor labels describe the views.
    content_ok = True
    packaging_ok = True
    by_id = {trace.trace_id: trace for trace in traces}
    for view, _ in pairs:
        raw = by_id[view.trace_id]
        needed = critical_values_by_event(raw)
        exposed = exposed_values_by_event(view)
        for event_id, values in needed.items():
            seen = exposed.get(event_id, set())
            if CONTENT[view.view_type] == "values_present":
                content_ok &= not (values - seen)
            else:
                content_ok &= not seen
        has_wrapper = any(e.get("source") == "handoff_summarizer" for e in view.events)
        packaging_ok &= has_wrapper == (PACKAGING[view.view_type] == "summary")
    checks["content_labels_are_truthful"] = content_ok
    checks["packaging_labels_are_truthful"] = packaging_ok

    # The two carried-over views must be identical to what v3.1 showed, or the
    # replication is not a replication.
    checks["carried_over_views_match_v31"] = _carried_over_matches_v31(traces)

    expected_ids = {
        spec.call_id
        for stage in ("smoke", "core", "stability")
        for spec in build_call_specs(run_dir, stage=stage, providers=providers)
    }
    latest = {row.get("call_id"): row for row in responses}
    current = [latest[item] for item in expected_ids if item in latest]
    checks.update(
        {
            "locked_call_ids_present": len(current) == len(expected_ids),
            "locked_calls_successful": len(current) == len(expected_ids)
            and all(r.get("status") == "success" for r in current),
            "locked_calls_parsed": len(current) == len(expected_ids)
            and all(isinstance(r.get("monitor_result"), dict) for r in current),
            "model_ids_match": all(
                r.get("model_id") == EXPECTED_MODELS.get(r.get("provider"))
                for r in current
                if r.get("status") == "success"
            ),
            "scores_cover_current_calls": bool(scores)
            and {r.get("call_id") for r in scores} == set(latest),
            "cost_within_cap": _cost(responses) <= SPEND_CAP_USD,
        }
    )

    audit = {
        "version": "v3.2",
        "passed": all(checks.values()),
        "checks": checks,
        "providers": list(providers),
        "expected_locked_calls": len(expected_ids),
        "current_locked_responses": len(current),
        "response_rows": len(responses),
        "score_rows": len(scores),
        "estimated_cost_usd": round(_cost(responses), 6),
        "spend_cap_usd": SPEND_CAP_USD,
        "system_prompt_sha256": V31_SYSTEM_PROMPT_SHA256,
        "artifact_sha256": {
            name: _sha256(run_dir / name)
            for name in (
                "dataset_manifest.json",
                "raw_traces.jsonl",
                "verifier_views.jsonl",
                "private_gold.jsonl",
                "responses.jsonl",
                "scores.jsonl",
                "summary.json",
            )
            if (run_dir / name).exists()
        },
    }
    _write_json(run_dir / "audit.json", audit)
    (run_dir / "audit.md").write_text(_markdown(audit), encoding="utf-8")
    return audit


def _carried_over_matches_v31(traces) -> bool:
    """`full` and `actionable_summary` must be byte-identical to v3.1's."""
    from tcrb.v031.views import build_view as v31_build_view

    from .views import build_view as v32_build_view

    for trace in traces:
        for view_type in ("full", "actionable_summary"):
            old, _ = v31_build_view(trace, view_type)
            new, _ = v32_build_view(trace, view_type)
            if json.dumps(list(old.events), sort_keys=True) != json.dumps(
                list(new.events), sort_keys=True
            ):
                return False
    return True


def _cost(rows: list[dict[str, Any]]) -> float:
    return sum(float(r.get("estimated_extra_cost_usd", 0) or 0) for r in rows)


def _markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# v3.2 integrity audit",
        "",
        f"Overall: **{'PASS' if audit['passed'] else 'FAIL'}**",
        "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'FAIL'} — {key.replace('_', ' ')}"
        for key, value in audit["checks"].items()
    )
    lines.extend(
        [
            "",
            f"Locked responses: {audit['current_locked_responses']}/{audit['expected_locked_calls']}",
            f"Response rows retained: {audit['response_rows']}",
            f"Score rows: {audit['score_rows']}",
            f"Estimated cost: ${audit['estimated_cost_usd']:.4f} (cap ${audit['spend_cap_usd']:.2f})",
            "",
            "## Artifact hashes",
            "",
        ]
    )
    lines.extend(
        f"- `{name}`: `{digest}`" for name, digest in audit["artifact_sha256"].items()
    )
    return "\n".join(lines) + "\n"


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_recursive_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_recursive_keys(child))
        return keys
    return set()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path) if path.exists() else []


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

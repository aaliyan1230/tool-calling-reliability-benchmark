from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .cases import build_tcrb_traces, validate_traces
from .runner import build_call_specs
from .views import build_all_views, validate_views


EXPECTED_MODELS = {
    "deepseek": "deepseek-v4-flash",
    "gpt": "gpt-5.6-terra",
}


def audit_run(run_dir: Path, *, providers: tuple[str, ...] = ("deepseek", "gpt")) -> dict[str, Any]:
    traces = build_tcrb_traces()
    manifest = _read_json(run_dir / "dataset_manifest.json") if (run_dir / "dataset_manifest.json").exists() else {}
    configured_views = tuple(manifest.get("view_types", []))
    pairs = build_all_views(traces, configured_views) if configured_views else build_all_views(traces)
    public = _read_jsonl(run_dir / "verifier_views.jsonl")
    private = _read_jsonl(run_dir / "private_gold.jsonl")
    responses = _read_jsonl(run_dir / "responses.jsonl") if (run_dir / "responses.jsonl").exists() else []
    scores = _read_jsonl(run_dir / "scores.jsonl") if (run_dir / "scores.jsonl").exists() else []
    checks: dict[str, bool] = {
        "trace_validation_passed": not validate_traces(traces),
        "view_validation_passed": not validate_views(traces, pairs),
        "public_private_counts_match": len(public) == len(private) == len(pairs),
        "private_gold_separate": not any(_recursive_keys(row) & {
            "gold_outcome", "required_evidence_ids", "mutation_id", "faults_applied",
            "diagnostic_labels", "payload_state",
        } for row in public),
        "public_view_ids_unique": len({row.get("view_id") for row in public}) == len(public),
        "private_view_ids_unique": len({row.get("view_id") for row in private}) == len(private),
    }
    expected_ids = {
        spec.call_id
        for stage in ("smoke", "core", "stability")
        for spec in build_call_specs(run_dir, stage=stage, providers=providers)
    } if (run_dir / "verifier_views.jsonl").exists() else set()
    latest = {row.get("call_id"): row for row in responses}
    current = [latest[item] for item in expected_ids if item in latest]
    checks.update(
        {
            "locked_call_ids_present": len(current) == len(expected_ids),
            "locked_calls_successful": len(current) == len(expected_ids)
            and all(row.get("status") == "success" for row in current),
            "locked_calls_parsed": len(current) == len(expected_ids)
            and all(isinstance(row.get("monitor_result"), dict) for row in current),
            "model_ids_match": all(
                row.get("model_id") == EXPECTED_MODELS.get(row.get("provider"))
                for row in current
                if row.get("status") == "success"
            ),
            "scores_cover_current_calls": bool(scores)
            and {row.get("call_id") for row in scores} == set(latest),
            "cost_within_cap": _cost(responses) <= 25.0,
        }
    )
    audit = {
        "version": "v3.1",
        "passed": all(checks.values()),
        "checks": checks,
        "providers": list(providers),
        "expected_locked_calls": len(expected_ids),
        "current_locked_responses": len(current),
        "response_rows": len(responses),
        "score_rows": len(scores),
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


def _cost(rows: list[dict[str, Any]]) -> float:
    return sum(float(row.get("estimated_extra_cost_usd", 0) or 0) for row in rows)


def _markdown(audit: dict[str, Any]) -> str:
    lines = ["# v3.1 integrity audit", "", f"Overall: **{'PASS' if audit['passed'] else 'FAIL'}**", ""]
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
            "",
            "## Artifact hashes",
            "",
        ]
    )
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in audit["artifact_sha256"].items())
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

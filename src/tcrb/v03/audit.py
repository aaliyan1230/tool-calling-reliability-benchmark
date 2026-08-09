from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .cases import build_case_variants, validate_case_variants
from .runner import build_call_specs


PRIVATE_KEYS = {"expected_flag", "payload_state", "corruption_reason", "private_gold"}
EXPECTED_MODELS = {"deepseek": "deepseek-v4-flash", "gpt": "gpt-5.6-terra"}


def audit_run(run_dir: Path) -> dict[str, Any]:
    """Verify that saved results match the current locked experiment."""
    public_rows = _read_jsonl(run_dir / "verifier_views.jsonl")
    private_rows = _read_jsonl(run_dir / "private_gold.jsonl")
    raw_responses = _read_jsonl(run_dir / "responses.jsonl")
    score_rows = _read_jsonl(run_dir / "scores.jsonl")
    summary = _read_json(run_dir / "summary.json")
    providers = sorted({row["provider"] for row in score_rows})

    expected_specs = []
    for stage in ("core", "stability"):
        expected_specs.extend(build_call_specs(stage=stage, providers=providers))
    expected_ids = {spec.call_id for spec in expected_specs}
    latest_by_id = {row["call_id"]: row for row in raw_responses}
    current_responses = [latest_by_id[call_id] for call_id in expected_ids if call_id in latest_by_id]
    score_ids = {row["call_id"] for row in score_rows}
    public_case_ids = {row["case_id"] for row in public_rows}
    private_case_ids = {row["case_id"] for row in private_rows}

    model_ids_correct = all(
        row.get("model_id") == EXPECTED_MODELS.get(row["provider"])
        for row in current_responses
        if row.get("status") == "success"
    )
    checks = {
        "dataset_validation_passed": validate_case_variants(build_case_variants()) == [],
        "public_has_64_unique_variants": len(public_rows) == 64 and len(public_case_ids) == 64,
        "private_has_64_matching_variants": len(private_rows) == 64
        and private_case_ids == public_case_ids,
        "private_keys_absent_from_public_views": not any(
            PRIVATE_KEYS & _recursive_keys(row) for row in public_rows
        ),
        "all_locked_call_ids_present": len(current_responses) == len(expected_ids),
        "all_locked_calls_succeeded": len(current_responses) == len(expected_ids)
        and all(row.get("status") == "success" for row in current_responses),
        "all_locked_calls_have_parsed_results": len(current_responses) == len(expected_ids)
        and all(isinstance(row.get("monitor_result"), dict) for row in current_responses),
        "served_model_ids_match_plan": model_ids_correct,
        "scores_cover_exact_locked_calls": score_ids == expected_ids,
        "summary_has_no_invalid_results": summary.get("responses_by_status", {}).get(
            "invalid_or_error"
        )
        == 0,
        "estimated_paid_cost_within_15_usd": float(
            summary.get("estimated_extra_cost_usd", 0)
        )
        <= 15.0,
    }
    artifact_names = (
        "dataset_manifest.json",
        "verifier_views.jsonl",
        "private_gold.jsonl",
        "responses.jsonl",
        "scores.jsonl",
        "summary.json",
    )
    artifact_hashes = {
        name: _sha256(run_dir / name)
        for name in artifact_names
        if (run_dir / name).exists()
    }
    audit = {
        "passed": all(checks.values()),
        "checks": checks,
        "providers": providers,
        "expected_locked_calls": len(expected_ids),
        "current_locked_responses": len(current_responses),
        "scored_rows": len(score_rows),
        "archived_superseded_attempts": len(raw_responses) - len(current_responses),
        "artifact_sha256": artifact_hashes,
    }
    (run_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
    (run_dir / "trajectory-walkthrough.md").write_text(
        _trajectory_walkthrough(public_rows, private_rows, score_rows), encoding="utf-8"
    )
    return audit


def _trajectory_walkthrough(
    public_rows: list[dict[str, Any]],
    private_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
) -> str:
    views = {row["case_id"]: row for row in public_rows}
    gold = {row["case_id"]: row for row in private_rows}
    primary = [
        row
        for row in score_rows
        if row.get("repeat_index") == 0 and not row.get("exploratory") and row.get("valid")
    ]
    providers = sorted({row["provider"] for row in primary})
    lines = [
        "# Concrete recorded trajectories",
        "",
        "The trace data below is deterministic and simulated. The model decisions and reasons are recorded API outputs, not reconstructed examples.",
        "",
    ]
    for provider in providers:
        baseline = [row for row in primary if row["provider"] == provider and row["prompt_variant"] == "baseline"]
        evidence = {
            row["case_id"]: row
            for row in primary
            if row["provider"] == provider and row["prompt_variant"] == "evidence_first"
        }
        changed = [
            row
            for row in baseline
            if row["case_id"] in evidence
            and row["payload_state"] == "correct"
            and row["self_label"] == "warning"
            and row["flagged"]
            and not evidence[row["case_id"]]["flagged"]
        ]
        selected = changed[0] if changed else baseline[0]
        paired = evidence.get(selected["case_id"])
        if paired is None:
            continue
        view = views[selected["case_id"]]
        private = gold[selected["case_id"]]
        lines.extend(
            [
                f"## {EXPECTED_MODELS.get(provider, provider)} — `{selected['case_id']}`",
                "",
                "### What the monitor saw",
                "",
                "```json",
                json.dumps(view, indent=2, sort_keys=True),
                "```",
                "",
                "### Hidden answer used only for scoring",
                "",
                f"The payload is **{private['payload_state']}**, so the expected action is "
                f"**{'flag' if private['expected_flag'] else 'allow'}**.",
                "",
                "### Recorded baseline decision",
                "",
                _decision_sentence(selected),
                "",
                "### Recorded evidence-first decision",
                "",
                _decision_sentence(paired),
                "",
                "### Simple reading",
                "",
                (
                    "The external records did not change. Only the instruction to rank those records above the tool’s self-label changed, and the monitor stopped raising this false alarm."
                    if selected["flagged"] and not paired["flagged"]
                    else "This model did not provide a clean false-alarm correction example in the saved core run; the paired decisions are shown without claiming an improvement."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _decision_sentence(row: dict[str, Any]) -> str:
    ids = ", ".join(f"`{item}`" for item in row.get("evidence_ids", [])) or "none"
    return (
        f"**{row['decision']}** (confidence {float(row['confidence']):.2f}); cited {ids}. "
        f"Reason: “{row['reason']}”"
    )


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Pilot integrity audit",
        "",
        f"Overall: **{'PASS' if audit['passed'] else 'FAIL'}**",
        "",
    ]
    for name, passed in audit["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}")
    lines.extend(
        [
            "",
            f"Locked calls checked: {audit['current_locked_responses']}/{audit['expected_locked_calls']}.",
            f"Archived superseded attempts retained: {audit['archived_superseded_attempts']}.",
            "",
            "## Artifact hashes",
            "",
        ]
    )
    for name, digest in audit["artifact_sha256"].items():
        lines.append(f"- `{name}`: `{digest}`")
    lines.append("")
    return "\n".join(lines)


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_recursive_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_recursive_keys(item))
        return keys
    return set()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

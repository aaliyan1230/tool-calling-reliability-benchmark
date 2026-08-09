from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .cases import BaseCase, build_base_cases, build_case_variants, validate_case_variants
from .prompts import prompt_bundle
from .providers import ProviderError, call_monitor, parse_monitor_result
from .schema import PilotVariant, PromptVariant


SMOKE_CASE_IDS = {"cs_refund_o1001", "dev_build401_production_commit"}
STABILITY_CASE_IDS = {
    "cs_refund_o1001",
    "ec_payment_o1001",
    "fi_transaction_txn03",
    "dev_build401_production_commit",
}


@dataclass(frozen=True)
class CallSpec:
    call_id: str
    provider: str
    prompt_variant: PromptVariant
    variant: PilotVariant
    repeat_index: int
    exploratory: bool


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prepare_dataset(run_dir: Path, *, include_stress: bool = False) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    variants = build_case_variants()
    errors = validate_case_variants(variants)
    if errors:
        raise ValueError("dataset validation failed:\n" + "\n".join(errors))

    _write_jsonl(
        run_dir / "verifier_views.jsonl",
        (variant.verifier_view.to_dict() for variant in variants),
    )
    _write_jsonl(
        run_dir / "private_gold.jsonl",
        (variant.private_gold.to_dict() for variant in variants),
    )
    if include_stress:
        stress = build_case_variants(include_distractors=True)
        _write_jsonl(
            run_dir / "stress_verifier_views.jsonl",
            (variant.verifier_view.to_dict() for variant in stress),
        )
    manifest = {
        "created_at": utc_now(),
        "git_sha": git_sha(),
        "base_cases": 16,
        "variants": 64,
        "domains": {
            "customer_support": 4,
            "ecommerce": 4,
            "fintech": 4,
            "developer_tools": 4,
        },
        "validation_errors": errors,
        "private_gold_separate": True,
    }
    _write_json(run_dir / "dataset_manifest.json", manifest)
    return manifest


def build_call_specs(
    *,
    stage: str,
    providers: Iterable[str],
) -> tuple[CallSpec, ...]:
    base_cases = build_base_cases()
    exploratory = stage == "stress"
    if stage == "smoke":
        selected = tuple(case for case in base_cases if case.case_id in SMOKE_CASE_IDS)
        repeats = (0,)
    elif stage == "core":
        selected = base_cases
        repeats = (0,)
    elif stage == "stability":
        selected = tuple(case for case in base_cases if case.case_id in STABILITY_CASE_IDS)
        repeats = (1, 2)
    elif stage == "stress":
        selected = tuple(case for case in base_cases if case.case_id in STABILITY_CASE_IDS)
        repeats = (0,)
    else:
        raise ValueError("stage must be smoke, core, stability, or stress")

    variants = build_case_variants(selected, include_distractors=exploratory)
    specs: list[CallSpec] = []
    for provider in providers:
        if provider not in {"deepseek", "gpt"}:
            raise ValueError(f"unknown provider: {provider}")
        for prompt_variant in ("baseline", "evidence_first"):
            for variant in variants:
                for repeat_index in repeats:
                    bundle = prompt_bundle(variant.verifier_view, prompt_variant)
                    payload = {
                        "provider": provider,
                        "prompt_variant": prompt_variant,
                        "prompt_sha256": hashlib.sha256(
                            json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "view": variant.verifier_view.to_dict(),
                        "repeat_index": repeat_index,
                        "exploratory": exploratory,
                    }
                    call_id = hashlib.sha256(
                        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()[:24]
                    specs.append(
                        CallSpec(
                            call_id=call_id,
                            provider=provider,
                            prompt_variant=prompt_variant,
                            variant=variant,
                            repeat_index=repeat_index,
                            exploratory=exploratory,
                        )
                    )
    return tuple(specs)


def run_stage(
    run_dir: Path,
    *,
    stage: str,
    providers: tuple[str, ...],
    spend_cap_usd: float = 15.0,
    timeout_s: int = 120,
    max_retries: int = 4,
) -> dict[str, Any]:
    if not (run_dir / "dataset_manifest.json").exists():
        prepare_dataset(run_dir, include_stress=stage == "stress")
    specs = build_call_specs(stage=stage, providers=providers)
    responses_path = run_dir / "responses.jsonl"
    existing = _latest_records(responses_path)
    spent = sum(
        float(record.get("estimated_extra_cost_usd", 0) or 0)
        for record in existing.values()
        if record.get("status") == "success"
    )
    completed = 0
    skipped = 0
    failed = 0
    for index, spec in enumerate(specs, start=1):
        cached = existing.get(spec.call_id)
        if (
            cached
            and cached.get("status") == "success"
            and isinstance(cached.get("monitor_result"), dict)
        ):
            skipped += 1
            continue
        if spec.provider == "gpt" and spent >= spend_cap_usd:
            raise RuntimeError(
                f"GPT spend cap reached: ${spent:.4f} >= ${spend_cap_usd:.2f}"
            )

        bundle = prompt_bundle(spec.variant.verifier_view, spec.prompt_variant)
        request_record = {
            "call_id": spec.call_id,
            "case_id": spec.variant.verifier_view.case_id,
            "base_case_id": spec.variant.private_gold.base_case_id,
            "domain": spec.variant.verifier_view.domain,
            "provider": spec.provider,
            "prompt_variant": spec.prompt_variant,
            "repeat_index": spec.repeat_index,
            "exploratory": spec.exploratory,
            "prompt_sha256": hashlib.sha256(
                json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "started_at": utc_now(),
            "git_sha": git_sha(),
        }
        print(
            f"[{index}/{len(specs)}] {spec.provider} {spec.prompt_variant} "
            f"{spec.variant.verifier_view.case_id} repeat={spec.repeat_index}",
            flush=True,
        )
        try:
            response = call_monitor(
                spec.provider,
                spec.variant.verifier_view,
                spec.prompt_variant,
                timeout_s=timeout_s,
                max_retries=max_retries,
            )
            parse_error = None
            monitor_result = None
            try:
                monitor_result = parse_monitor_result(response.output_text).to_dict()
            except ValueError as exc:
                parse_error = str(exc)
            record = {
                **request_record,
                "finished_at": utc_now(),
                "status": "success",
                "served_provider": response.provider,
                "model_id": response.model_id,
                "output_text": response.output_text,
                "monitor_result": monitor_result,
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
            failed += 1
            record = {
                **request_record,
                "finished_at": utc_now(),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "estimated_extra_cost_usd": 0.0,
            }
            _append_jsonl(responses_path, record)
            print(f"  ERROR: {exc}", flush=True)

    summary = {
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
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records[payload["call_id"]] = payload
    return records


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_jsonl(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

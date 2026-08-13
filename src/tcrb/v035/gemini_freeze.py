"""Freeze Gemini's matched narrow/broad comparison with hash checks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from tcrb.v034.util import REPO_ROOT, read_json, read_jsonl, sha256_bytes, sha256_file, write_json
from tcrb.v035.model_expansion_freeze import COHORTS, EXPECTED
from tcrb.v035.prewrite import OUTPUT, _append_log
from tcrb.v035.prewrite_monitor import monitor_input
from tcrb.v035.registries import model_registry_sha256, policy_bundle_sha256, policy_bundle_version


MODEL = "gemini-3.6-flash"
POLICIES = ("narrow", "broad")
SNAPSHOT = OUTPUT / "frozen_gemini"


def artifact_names() -> list[str]:
    names: list[str] = []
    for policy in POLICIES:
        policy_suffix = "" if policy == "narrow" else "_broad"
        for _, stem in COHORTS:
            stem_with_model = stem + policy_suffix + "_" + MODEL
            names.extend(stem_with_model + ext for ext in (".jsonl", "_analysis.json", "_manifest.json", "_inputs.jsonl"))
    return names


def _validate_source_artifacts(names: list[str]) -> None:
    for name in names:
        if not (OUTPUT / name).exists():
            raise FileNotFoundError(OUTPUT / name)
    for policy in POLICIES:
        policy_suffix = "" if policy == "narrow" else "_broad"
        for cohort, stem in COHORTS:
            stem_with_model = stem + policy_suffix + "_" + MODEL
            analysis = read_json(OUTPUT / (stem_with_model + "_analysis.json"))
            expected = EXPECTED[cohort]
            if not analysis.get("passed") or analysis.get("n") != expected or analysis.get("expected_n") != expected:
                raise ValueError(f"incomplete analysis for {policy}/{cohort}: {analysis}")
            rows = read_jsonl(OUTPUT / (stem_with_model + ".jsonl"))
            request_ids = [row.get("request_id") for row in rows]
            if len(request_ids) != expected or len(set(request_ids)) != expected:
                raise ValueError(f"unexpected result rows for {policy}/{cohort}: {len(request_ids)}")


def freeze() -> dict[str, Any]:
    if SNAPSHOT.exists():
        raise FileExistsError(f"freeze already exists: {SNAPSHOT}; remove it only with explicit approval")
    names = artifact_names()
    _validate_source_artifacts(names)
    SNAPSHOT.mkdir(parents=True)
    artifacts: dict[str, str] = {}
    for name in names:
        destination = SNAPSHOT / name
        shutil.copy2(OUTPUT / name, destination)
        artifacts[name] = sha256_file(destination)
    smoke_manifest = OUTPUT / "smoke" / f"{MODEL}_broad_manifest.json"
    references = {
        "configs/v035/monitor_models.json": sha256_file(REPO_ROOT / "configs/v035/monitor_models.json"),
        "configs/v035/policy_bundles.json": sha256_file(REPO_ROOT / "configs/v035/policy_bundles.json"),
        "src/tcrb/v035/prewrite_monitor.py": sha256_file(REPO_ROOT / "src/tcrb/v035/prewrite_monitor.py"),
        "src/tcrb/v035/registries.py": sha256_file(REPO_ROOT / "src/tcrb/v035/registries.py"),
        "src/tcrb/v035/gemini_freeze.py": sha256_file(Path(__file__)),
        "outputs/v035/prewrite/manifest.json": sha256_file(OUTPUT / "manifest.json"),
    }
    first_trace = read_jsonl(OUTPUT / "traces_private.jsonl")[0]
    narrow_hash = sha256_bytes(json.dumps(monitor_input(first_trace, "runtime", "narrow")["policy_rules"], sort_keys=True, separators=(",", ":")).encode())
    manifest = {
        "freeze_version": "gemini_comparison_v1",
        "dataset_id": read_json(OUTPUT / "manifest.json")["dataset_id"],
        "model": MODEL,
        "policies": list(POLICIES),
        "view": "runtime",
        "prompt_version": "prewrite_monitor_v1",
        "reasoning_effort": "none",
        "thinking_level": "minimal",
        "temperature": 0,
        "temperature_applied": False,
        "output_cap_tokens": 400,
        "completed_unique_calls": 80,
        "cohorts": {"control": 12, "development": 16, "holdout_v2": 12},
        "artifact_sha256": artifacts,
        "smoke_manifest_sha256": sha256_file(smoke_manifest),
        "reference_sha256": references,
        "model_registry_sha256": model_registry_sha256(),
        "policy_bundle_sha256": {"narrow": narrow_hash, "broad": policy_bundle_sha256("broad")},
        "policy_bundle_version": {policy: policy_bundle_version(policy) if policy == "broad" else "trace_policy_rules" for policy in POLICIES},
        "snapshot_dir": str(SNAPSHOT.relative_to(REPO_ROOT)),
        "immutable": True,
    }
    write_json(SNAPSHOT / "freeze_manifest.json", manifest)
    _append_log(f"Frozen Gemini comparison: {manifest['freeze_version']}, {len(artifacts)} artifacts, 80 completed unique calls, snapshot={SNAPSHOT.relative_to(REPO_ROOT)}.")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze"])
    parser.parse_args(argv)
    print(json.dumps(freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

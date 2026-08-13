"""Freeze the expanded-model narrow-policy comparison with hash checks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from tcrb.v034.util import REPO_ROOT, read_json, read_jsonl, sha256_bytes, sha256_file, write_json
from tcrb.v035.model_expansion_freeze import COHORTS, EXPECTED, MODELS
from tcrb.v035.prewrite import OUTPUT, _append_log
from tcrb.v035.prewrite_monitor import monitor_input
from tcrb.v035.registries import model_registry_sha256


SNAPSHOT = OUTPUT / "frozen_model_expansion_narrow"


def artifact_names() -> list[str]:
    names: list[str] = []
    for model in MODELS:
        suffix = "_" + model
        for _, stem in COHORTS:
            names.extend(stem + suffix + ext for ext in (".jsonl", "_analysis.json", "_manifest.json", "_inputs.jsonl"))
    return names


def _validate_source_artifacts(names: list[str]) -> None:
    for name in names:
        source = OUTPUT / name
        if not source.exists():
            raise FileNotFoundError(source)
    for model in MODELS:
        for cohort, stem in COHORTS:
            stem_with_model = stem + "_" + model
            analysis = read_json(OUTPUT / (stem_with_model + "_analysis.json"))
            expected = EXPECTED[cohort]
            if not analysis.get("passed") or analysis.get("n") != expected or analysis.get("expected_n") != expected:
                raise ValueError(f"incomplete analysis for {model}/{cohort}: {analysis}")
            rows = read_jsonl(OUTPUT / (stem_with_model + ".jsonl"))
            request_ids = [row.get("request_id") for row in rows]
            if len(request_ids) != expected or len(set(request_ids)) != expected:
                raise ValueError(f"unexpected result rows for {model}/{cohort}: {len(request_ids)}")


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
    references = {
        "configs/v035/policy_bundles.json": sha256_file(REPO_ROOT / "configs/v035/policy_bundles.json"),
        "configs/v035/monitor_models.json": sha256_file(REPO_ROOT / "configs/v035/monitor_models.json"),
        "src/tcrb/v035/prewrite_monitor.py": sha256_file(REPO_ROOT / "src/tcrb/v035/prewrite_monitor.py"),
        "src/tcrb/v035/registries.py": sha256_file(REPO_ROOT / "src/tcrb/v035/registries.py"),
        "src/tcrb/v035/model_expansion_narrow_freeze.py": sha256_file(Path(__file__)),
        "outputs/v035/prewrite/manifest.json": sha256_file(OUTPUT / "manifest.json"),
    }
    manifest = {
        "freeze_version": "model_expansion_narrow_v1",
        "dataset_id": read_json(OUTPUT / "manifest.json")["dataset_id"],
        "policy_mode": "narrow",
        "policy_bundle_version": "trace_policy_rules",
        "policy_bundle_sha256": sha256_bytes(json.dumps(monitor_input(read_jsonl(OUTPUT / "traces_private.jsonl")[0], "runtime", "narrow")["policy_rules"], sort_keys=True, separators=(",", ":")).encode()),
        "models": list(MODELS),
        "view": "runtime",
        "prompt_version": "prewrite_monitor_v1",
        "reasoning_effort": "none",
        "temperature": 0,
        "output_cap_tokens": 400,
        "completed_unique_calls": 120,
        "cohorts": {"control": 12, "development": 16, "holdout_v2": 12},
        "artifact_sha256": artifacts,
        "reference_sha256": references,
        "model_registry_sha256": model_registry_sha256(),
        "snapshot_dir": str(SNAPSHOT.relative_to(REPO_ROOT)),
        "immutable": True,
    }
    write_json(SNAPSHOT / "freeze_manifest.json", manifest)
    _append_log(f"Frozen expanded-model narrow comparison: {manifest['freeze_version']}, {len(artifacts)} artifacts, 120 completed unique calls, snapshot={SNAPSHOT.relative_to(REPO_ROOT)}.")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze"])
    parser.parse_args(argv)
    print(json.dumps(freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

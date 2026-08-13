"""Freeze the broader-policy comparison as a hash-checked snapshot."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from tcrb.v034.util import REPO_ROOT, read_json, sha256_file, write_json
from tcrb.v035.prewrite import OUTPUT, _append_log
from tcrb.v035.registries import model_registry_sha256, policy_bundle_sha256, policy_bundle_version


SNAPSHOT = OUTPUT / "frozen_broad_policy"
MODELS = ("deepseek-v4-flash", "gpt-5.6-luna")
COHORT_FILES = (("control", "monitor_runtime_control"), ("development", "monitor_runtime_main_development"), ("holdout_v2", "monitor_runtime_main_holdout_v2"))


def _artifact_names() -> list[str]:
    names: list[str] = []
    for model in MODELS:
        suffix = "" if model == "deepseek-v4-flash" else "_" + model
        for _, stem in COHORT_FILES:
            if stem == "monitor_runtime_control":
                stem_with_policy = stem + "_broad" + suffix
            else:
                cohort = "development" if "development" in stem else "holdout_v2"
                stem_with_policy = stem + "_broad" + suffix
            names.extend([stem_with_policy + ext for ext in (".jsonl", "_analysis.json", "_manifest.json", "_inputs.jsonl")])
    return names


def freeze() -> dict[str, Any]:
    if SNAPSHOT.exists():
        raise FileExistsError(f"freeze already exists: {SNAPSHOT}")
    SNAPSHOT.mkdir(parents=True)
    artifacts: dict[str, str] = {}
    for name in _artifact_names():
        source = OUTPUT / name
        if not source.exists():
            raise FileNotFoundError(source)
        destination = SNAPSHOT / name
        shutil.copy2(source, destination)
        artifacts[name] = sha256_file(destination)
    references = {
        "configs/v035/policy_bundles.json": sha256_file(REPO_ROOT / "configs/v035/policy_bundles.json"),
        "configs/v035/monitor_models.json": sha256_file(REPO_ROOT / "configs/v035/monitor_models.json"),
        "src/tcrb/v035/prewrite_monitor.py": sha256_file(REPO_ROOT / "src/tcrb/v035/prewrite_monitor.py"),
        "src/tcrb/v035/registries.py": sha256_file(REPO_ROOT / "src/tcrb/v035/registries.py"),
        "outputs/v035/prewrite/manifest.json": sha256_file(OUTPUT / "manifest.json"),
    }
    manifest = {
        "freeze_version": "broad_policy_comparison_v1",
        "dataset_id": read_json(OUTPUT / "manifest.json")["dataset_id"],
        "policy_mode": "broad",
        "policy_bundle_version": policy_bundle_version("broad"),
        "policy_bundle_sha256": policy_bundle_sha256("broad"),
        "models": list(MODELS),
        "view": "runtime",
        "prompt_version": "prewrite_monitor_v1",
        "reasoning_effort": "none",
        "temperature": 0,
        "output_cap_tokens": 400,
        "completed_unique_calls": 80,
        "cohorts": {"control": 12, "development": 16, "holdout_v2": 12},
        "artifact_sha256": artifacts,
        "reference_sha256": references,
        "model_registry_sha256": model_registry_sha256(),
        "snapshot_dir": str(SNAPSHOT.relative_to(REPO_ROOT)),
        "immutable": True,
    }
    write_json(SNAPSHOT / "freeze_manifest.json", manifest)
    _append_log(f"Frozen broad-policy comparison: {manifest['freeze_version']}, {len(artifacts)} artifacts, 80 completed unique calls, snapshot={SNAPSHOT.relative_to(REPO_ROOT)}.")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze"])
    parser.parse_args(argv)
    print(json.dumps(freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

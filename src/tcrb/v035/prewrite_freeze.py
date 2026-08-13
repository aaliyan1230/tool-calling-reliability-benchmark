"""Create an immutable, hash-addressed snapshot of the monitor comparison."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from tcrb.v034.util import REPO_ROOT, read_json, sha256_file, write_json
from tcrb.v035.prewrite import OUTPUT, _append_log


FREEZE_VERSION = "prewrite_monitor_comparison_v1"
SNAPSHOT = OUTPUT / "frozen_comparison"

RESULT_FILES = (
    "monitor_runtime_control.jsonl",
    "monitor_runtime_control_analysis.json",
    "monitor_runtime_control_manifest.json",
    "monitor_runtime_control_inputs.jsonl",
    "monitor_runtime_main_development.jsonl",
    "monitor_runtime_main_development_analysis.json",
    "monitor_runtime_main_development_manifest.json",
    "monitor_runtime_main_development_inputs.jsonl",
    "monitor_runtime_main_holdout_v2.jsonl",
    "monitor_runtime_main_holdout_v2_analysis.json",
    "monitor_runtime_main_holdout_v2_manifest.json",
    "monitor_runtime_main_holdout_v2_inputs.jsonl",
    "monitor_runtime_control_gpt-5.6-luna.jsonl",
    "monitor_runtime_control_gpt-5.6-luna_analysis.json",
    "monitor_runtime_control_gpt-5.6-luna_manifest.json",
    "monitor_runtime_control_gpt-5.6-luna_inputs.jsonl",
    "monitor_runtime_main_development_gpt-5.6-luna.jsonl",
    "monitor_runtime_main_development_gpt-5.6-luna_analysis.json",
    "monitor_runtime_main_development_gpt-5.6-luna_manifest.json",
    "monitor_runtime_main_development_gpt-5.6-luna_inputs.jsonl",
    "monitor_runtime_main_holdout_v2_gpt-5.6-luna.jsonl",
    "monitor_runtime_main_holdout_v2_gpt-5.6-luna_analysis.json",
    "monitor_runtime_main_holdout_v2_gpt-5.6-luna_manifest.json",
    "monitor_runtime_main_holdout_v2_gpt-5.6-luna_inputs.jsonl",
)

REFERENCE_FILES = (
    OUTPUT / "manifest.json",
    REPO_ROOT / "src" / "tcrb" / "v035" / "prewrite_monitor.py",
    REPO_ROOT / "docs" / "v3" / "v3.5" / "PREWRITE_RESULTS.md",
    REPO_ROOT / "docs" / "v3" / "v3.5" / "LUNA_RESULTS.md",
)


def freeze() -> dict[str, Any]:
    if SNAPSHOT.exists():
        raise FileExistsError(f"freeze already exists: {SNAPSHOT}; remove it only with explicit approval")
    SNAPSHOT.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for name in RESULT_FILES:
        source = OUTPUT / name
        if not source.exists():
            raise FileNotFoundError(source)
        destination = SNAPSHOT / name
        shutil.copy2(source, destination)
        hashes[name] = sha256_file(destination)
    references = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in REFERENCE_FILES}
    manifest = {
        "freeze_version": FREEZE_VERSION,
        "dataset_id": read_json(OUTPUT / "manifest.json")["dataset_id"],
        "comparison": {
            "models": ["deepseek-v4-flash", "gpt-5.6-luna"],
            "view": "runtime",
            "prompt_version": "prewrite_monitor_v1",
            "reasoning": "none",
            "temperature": 0,
            "output_cap_tokens": 400,
            "subsets": {"control": 12, "development": 16, "holdout_v2": 12},
            "calls": 40,
        },
        "snapshot_dir": str(SNAPSHOT.relative_to(REPO_ROOT)),
        "artifact_sha256": hashes,
        "reference_sha256": references,
        "immutable": True,
    }
    write_json(SNAPSHOT / "freeze_manifest.json", manifest)
    _append_log(f"Frozen monitor comparison: {FREEZE_VERSION}, 24 artifacts, dataset={manifest['dataset_id']}, snapshot={SNAPSHOT.relative_to(REPO_ROOT)}.")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze"])
    args = parser.parse_args(argv)
    print(__import__("json").dumps(freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

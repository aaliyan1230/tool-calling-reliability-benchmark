from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .reporting import build_report


TEXT_ARTIFACTS = (
    "dataset_manifest.json",
    "verifier_views.jsonl",
    "private_gold.jsonl",
    "responses.jsonl",
    "scores.jsonl",
    "summary.json",
    "audit.json",
    "audit.md",
    "trajectory-walkthrough.md",
    "run_smoke_summary.json",
    "run_core_summary.json",
    "run_stability_summary.json",
)
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"OPENAI_API_KEY\s*="),
    re.compile(r"OPENCODE_API_KEY\s*="),
)


def export_results(run_dir: Path, destination: Path) -> dict[str, Any]:
    """Export the audited run and presentation files into a shareable folder."""
    audit_path = run_dir / "audit.json"
    if not audit_path.exists():
        raise FileNotFoundError("run audit before exporting results")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise ValueError("refusing to export a run whose integrity audit failed")

    report_result = build_report(run_dir, output_dir=destination / "figures")
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in TEXT_ARTIFACTS:
        source = run_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        _assert_no_secret(source)
        target = destination / name
        shutil.copyfile(source, target)
        copied.append(target)

    readme_path = destination / "README.md"
    readme_path.write_text(_results_readme(audit), encoding="utf-8")
    copied.append(readme_path)
    return {
        "destination": str(destination),
        "audit_passed": True,
        "copied_data_files": len(copied),
        "report_artifacts": report_result["artifacts"],
    }


def _assert_no_secret(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"possible secret found; refusing to export {path.name}")


def _results_readme(audit: dict[str, Any]) -> str:
    return f"""# TCRB v0.3 evidence-provenance pilot results

This folder is the complete shareable result package for the fixed two-model pilot.

## What ran

- 16 deterministic simulated base cases across four domains
- 4 matched variants per case: correct/corrupt payload × verified/warning self-label
- 2 prompts: baseline and evidence-first
- 2 models: DeepSeek V4 Flash and GPT-5.6 Terra
- 256 core calls plus 128 repeat calls
- Estimated paid GPT cost: $0.398920

The raw trace data is simulated. The model responses in `responses.jsonl` are recorded API outputs. `scores.jsonl` contains the 384 current locked logical results. The raw file also retains {audit['archived_superseded_attempts']} superseded errors/retries for auditability.

## Start here

1. `figures/pilot-brief.md` — one-page result
2. `figures/figure-1-label-evidence-interaction.png` — direct 2×2 behavior
3. `figures/figure-2-prompt-improvement.png` — paired changes with uncertainty
4. `figures/figure-3-repeatability.png` — three-run stability check
5. `trajectory-walkthrough.md` — exact trace and recorded decisions
6. `audit.md` — integrity checks and artifact hashes

## Reproduce analysis

```bash
PYTHONPATH=src python3 -m tcrb.v03 --run-dir docs/v3/results analyze
PYTHONPATH=src python3 -m tcrb.v03 --run-dir docs/v3/results report --output-dir docs/v3/results/figures
PYTHONPATH=src python3 -m tcrb.v03 --run-dir docs/v3/results audit
```

The pre-registered outreach gate did not pass. Treat this as a small, model-specific pilot and a promising test of evidence-first prompting, not proof of a broad failure mode.
"""

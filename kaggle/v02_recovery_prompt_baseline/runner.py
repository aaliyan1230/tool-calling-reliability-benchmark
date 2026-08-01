#!/usr/bin/env python3
"""Run the native v0.2 baseline with the recovery prompt and no adapter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


REPO_URL = "https://github.com/aaliyan1230/tool-calling-reliability-benchmark.git"
MODEL_ID = "Qwen/Qwen3-4B"
OUTPUT_DIR = Path("/kaggle/working/v02_recovery_prompt_baseline")


def load_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    for path in (
        Path("/kaggle/input/tcrb-repo-snapshot/hf_token.txt"),
        Path("/kaggle/input/datasets/aaliyanshaikh/tcrb-repo-snapshot/hf_token.txt"),
    ):
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value.splitlines()[0].strip()
    return None


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = OUTPUT_DIR / "status.json"
    try:
        token = load_hf_token()
        if token:
            os.environ["HF_TOKEN"] = token

        repo_dir = Path("/tmp/tcrb_repo")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", "feat/tcrb-v0.2", REPO_URL, str(repo_dir)],
            check=True,
            text=True,
            capture_output=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "transformers>=4.51",
                "torch>=2.0",
                "accelerate>=0.30",
                "google-genai>=1.0",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        sys.path.insert(0, str(repo_dir / "src"))

        from tcrb.v02.eval_runner import run_eval

        summary = run_eval(
            model_id=MODEL_ID,
            domains=["customer_support", "ecommerce", "fintech", "developer_tools"],
            output_dir=OUTPUT_DIR,
            seed=42,
            max_tasks=0,
            clean_only=False,
            agent_type="hf_generate",
            prompt_variant="recovery",
            adapter_path=None,
        )
        status = {
            "status": "success",
            "model_id": MODEL_ID,
            "agent_type": "hf_generate",
            "prompt_variant": "recovery",
            "clean_rate": summary["clean"]["rate"],
            "faulted_rate": summary["faulted"]["rate"],
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2), flush=True)
        return 0
    except Exception as exc:
        status = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

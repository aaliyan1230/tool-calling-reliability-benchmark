#!/usr/bin/env python3
"""Train and evaluate the TCRB v0.2 recovery SFT adapter on Kaggle."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


REPO_URL = "https://github.com/aaliyan1230/tool-calling-reliability-benchmark.git"
MODEL_ID = "Qwen/Qwen3-4B"
DATASET_PATH = Path("/kaggle/input/tcrb-v02-recovery-sft-data/recovery_sft.jsonl")
OUTPUT_DIR = Path("/kaggle/working/v02_recovery_sft")
ADAPTER_DIR = OUTPUT_DIR / "adapter"
EVAL_DIR = OUTPUT_DIR / "evaluation"


def load_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    for path in (
        Path("/kaggle/input/tcrb-repo-snapshot/hf_token.txt"),
        Path("/kaggle/input/tcrb-repo-snapshot/token.txt"),
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
        else:
            print("[runner] WARNING: HF_TOKEN not found", flush=True)

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
                "datasets>=2.18",
                "peft>=0.14",
                "trl>=0.15",
                "bitsandbytes>=0.43",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        sys.path.insert(0, str(repo_dir / "src"))

        from tcrb.research import ResearchRecipe, run_sft_training

        recipe = ResearchRecipe(
            stage="sft",
            base_model=MODEL_ID,
            output_dir=str(ADAPTER_DIR),
            learning_rate=1e-4,
            num_train_epochs=2.0,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            max_seq_length=2048,
            packing=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            load_in_4bit=True,
            fp16=True,
            bf16=False,
        )
        print(f"[runner] Training from {DATASET_PATH}", flush=True)
        run_sft_training(recipe, dataset_path=DATASET_PATH)
        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from tcrb.v02.eval_runner import run_eval

        summary = run_eval(
            model_id=MODEL_ID,
            domains=["customer_support", "ecommerce", "fintech", "developer_tools"],
            output_dir=EVAL_DIR,
            seed=42,
            max_tasks=0,
            clean_only=False,
            prompt_variant="recovery",
            adapter_path=str(ADAPTER_DIR),
        )
        status = {
            "status": "success",
            "model_id": MODEL_ID,
            "adapter_path": str(ADAPTER_DIR),
            "clean_rate": summary["clean"]["rate"],
            "faulted_rate": summary["faulted"]["rate"],
            "evaluation_dir": str(EVAL_DIR),
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

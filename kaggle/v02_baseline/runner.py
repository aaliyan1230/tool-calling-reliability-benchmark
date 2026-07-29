#!/usr/bin/env python3
"""TCRB v0.2 baseline evaluation runner for Kaggle.

Environment variables:
  TCRB_MODEL         - HF model ID (default: Qwen/Qwen3-4B)
  TCRB_MAX_TASKS     - Max tasks to run, 0=all (default: 4 for smoke test)
  TCRB_DOMAINS       - Comma-separated domains (default: all)
  TCRB_OUTPUT_DIR    - Output directory (default: /kaggle/working/v02_baseline)
  TCRB_SEED          - Random seed (default: 42)
  TCRB_CLEAN_ONLY    - Run only clean eval (default: 1)
  TCRB_PROMPT_VARIANT - Built-in prompt variant: default or recovery (default: default)
  TCRB_ADAPTER_PATH  - Optional PEFT adapter directory to evaluate
  TCRB_BRANCH        - Git branch to clone (default: feat/tcrb-v0.2)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


REPO_URL = "https://github.com/aaliyan1230/tool-calling-reliability-benchmark.git"


def env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(env(name, str(default)))
    except ValueError:
        return default


def env_flag(name: str, default: bool = False) -> bool:
    raw = env(name, str(default)).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def load_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token
    except Exception:
        pass

    for slug in ["tcrb-repo-snapshot"]:
        for filename in ["hf_token.txt", "token.txt"]:
            path = Path(f"/kaggle/input/{slug}/{filename}")
            if path.exists():
                content = path.read_text().strip()
                if content:
                    return content.splitlines()[0].strip()

    return None


def load_gemini_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    for slug in ["tcrb-repo-snapshot"]:
        for filename in ["gemini_key.txt"]:
            path = Path(f"/kaggle/input/{slug}/{filename}")
            if path.exists():
                content = path.read_text().strip()
                if content:
                    return content.splitlines()[0].strip()

    return None


def main() -> int:
    output_dir = Path(env("TCRB_OUTPUT_DIR", "/kaggle/working/v02_baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"

    try:
        token = load_hf_token()
        if token:
            os.environ["HF_TOKEN"] = token
            print("[runner] HF_TOKEN loaded", flush=True)
        else:
            print("[runner] WARNING: HF_TOKEN not found - gated models may fail", flush=True)

        gemini_key = load_gemini_key()
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            print("[runner] GEMINI_API_KEY loaded", flush=True)
        else:
            print("[runner] WARNING: GEMINI_API_KEY not found - Gemini reviewer will not work", flush=True)

        branch = env("TCRB_BRANCH", "feat/tcrb-v0.2")
        repo_dir = Path("/tmp/tcrb_repo")

        print(f"[runner] Cloning {REPO_URL} branch={branch}", flush=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, REPO_URL, str(repo_dir)],
            check=True, text=True, capture_output=True,
        )

        print("[runner] Installing dependencies", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "transformers>=4.45", "torch>=2.0", "accelerate>=0.20", "google-genai>=1.0"],
            check=True, text=True, capture_output=True,
        )

        src_dir = str(repo_dir / "src")
        if "PYTHONPATH" in os.environ:
            os.environ["PYTHONPATH"] = f"{src_dir}:{os.environ['PYTHONPATH']}"
        else:
            os.environ["PYTHONPATH"] = src_dir
        sys.path.insert(0, src_dir)

        for entry in sys.path:
            print(f"[runner] sys.path: {entry}", flush=True)
        print(f"[runner] PYTHONPATH: {os.environ.get('PYTHONPATH')}", flush=True)

        print("[runner] Importing v0.2 modules", flush=True)
        import importlib
        import pkgutil
        for name in ["tcrb", "tcrb.v02"]:
            try:
                mod = importlib.import_module(name)
                fpath = getattr(mod, "__file__", "unknown")
                print(f"[runner]   {name} -> {fpath}", flush=True)
            except ModuleNotFoundError:
                print(f"[runner]   {name} -> NOT FOUND", flush=True)
                for finder in sys.meta_path:
                    print(f"[runner]     meta_path finder: {type(finder).__name__}", flush=True)
                pass

        from tcrb.v02.eval_runner import run_eval

        model_id = env("TCRB_MODEL", "Qwen/Qwen3-4B")
        max_tasks = env_int("TCRB_MAX_TASKS", 4)
        domains_str = env("TCRB_DOMAINS", "customer_support,ecommerce,fintech,developer_tools")
        domains = [d.strip() for d in domains_str.split(",") if d.strip()]
        seed = env_int("TCRB_SEED", 42)
        clean_only = env_flag("TCRB_CLEAN_ONLY", False)
        agent_type = env("TCRB_AGENT_TYPE", "hf_generate")
        prompt_variant = env("TCRB_PROMPT_VARIANT", "default")
        adapter_path = env("TCRB_ADAPTER_PATH", "") or None

        print(f"[runner] Model: {model_id}", flush=True)
        print(f"[runner] Max tasks: {max_tasks}", flush=True)
        print(f"[runner] Domains: {domains}", flush=True)
        print(f"[runner] Clean only: {clean_only}", flush=True)
        print(f"[runner] Agent type: {agent_type}", flush=True)
        print(f"[runner] Prompt variant: {prompt_variant}", flush=True)
        print(f"[runner] Adapter path: {adapter_path or 'none'}", flush=True)

        summary = run_eval(
            model_id=model_id,
            domains=domains,
            output_dir=str(output_dir),
            seed=seed,
            max_tasks=max_tasks,
            clean_only=clean_only,
            agent_type=agent_type,
            prompt_variant=prompt_variant,
            adapter_path=adapter_path,
        )

        status = {
            "status": "success",
            "model_id": model_id,
            "clean_rate": summary["clean"]["rate"],
            "prompt_variant": prompt_variant,
            "adapter_path": adapter_path,
            "duration_s": summary["total_time_s"],
            "output_dir": str(output_dir),
        }
        status_path.write_text(json.dumps(status, indent=2))
        print(json.dumps(status, indent=2), flush=True)

        for f in sorted(output_dir.rglob("*")):
            if f.is_file():
                print(f"  Output: {f.relative_to(output_dir)} ({f.stat().st_size} bytes)", flush=True)

        return 0

    except Exception as exc:
        status = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        status_path.write_text(json.dumps(status, indent=2))
        print(json.dumps(status, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

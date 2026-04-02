from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    started = time.time()
    print("[northstar] Running:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(cwd), text=True, check=False)
    elapsed = time.time() - started
    print(f"[northstar] Stage finished in {elapsed:.1f}s", flush=True)
    if completed.returncode != 0:
        raise SystemExit(f"[northstar] Command failed with exit code {completed.returncode}")


def load_env_file(repo_root: Path) -> bool:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return False

    loaded_any = False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if key not in os.environ:
            os.environ[key] = value
            loaded_any = True
    return loaded_any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HF-only north-star pipeline: multi-seed base/ft, delta, and transfer matrix"
    )
    parser.add_argument(
        "--workload",
        default="workloads/sample_tasks.json",
        help="Target workload path",
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.json",
        help="Benchmark config path",
    )
    parser.add_argument(
        "--seeds",
        default="11,22,33",
        help="Comma-separated seeds for multi-seed runs",
    )
    parser.add_argument(
        "--base-planner-config",
        default="configs/planners/hf_qwen2_5_3b_base.json",
        help="Base planner config",
    )
    parser.add_argument(
        "--ft-planner-config",
        default="configs/planners/hf_qwen2_5_3b_ft.json",
        help="Finetuned planner config",
    )
    parser.add_argument(
        "--matrix-manifest",
        default="workloads/enriched/manifest.json",
        help="Manifest path for transfer matrix",
    )
    parser.add_argument(
        "--matrix-target-toolset",
        default="customer_support",
        help="Target toolset id for transfer matrix",
    )
    parser.add_argument(
        "--matrix-toolsets",
        default="customer_support,ecommerce_ops,fintech_risk",
        help="Comma-separated toolsets for transfer matrix",
    )
    parser.add_argument(
        "--matrix-max-tasks",
        type=int,
        default=18,
        help="Max tasks per toolset for transfer matrix",
    )
    parser.add_argument(
        "--label-prefix",
        default="northstar-hf",
        help="Run label prefix",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()

    loaded = load_env_file(repo_root)
    if loaded:
        print("[northstar] Loaded environment from .env", flush=True)
    print("[northstar] HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")), flush=True)

    base_ms_label = f"{args.label_prefix}-base-ms"
    ft_ms_label = f"{args.label_prefix}-ft-ms"
    delta_json = f"runs/{args.label_prefix}-delta/delta-ms.json"
    delta_md = f"runs/{args.label_prefix}-delta/delta-ms.md"
    matrix_label = f"{args.label_prefix}-matrix"

    run_cmd(
        [
            "uv",
            "run",
            "tcrb",
            "multi-seed",
            "--config",
            args.config,
            "--workload",
            args.workload,
            "--seeds",
            args.seeds,
            "--planner-config",
            args.base_planner_config,
            "--label",
            base_ms_label,
        ],
        cwd=repo_root,
    )

    run_cmd(
        [
            "uv",
            "run",
            "tcrb",
            "multi-seed",
            "--config",
            args.config,
            "--workload",
            args.workload,
            "--seeds",
            args.seeds,
            "--planner-config",
            args.ft_planner_config,
            "--label",
            ft_ms_label,
        ],
        cwd=repo_root,
    )

    run_cmd(
        [
            "uv",
            "run",
            "tcrb",
            "eval-delta",
            "--base-run",
            f"runs/{base_ms_label}/multi_seed.json",
            "--finetuned-run",
            f"runs/{ft_ms_label}/multi_seed.json",
            "--output-json",
            delta_json,
            "--output-report",
            delta_md,
        ],
        cwd=repo_root,
    )

    run_cmd(
        [
            "uv",
            "run",
            "python",
            "scripts/run_transfer_matrix.py",
            "--manifest",
            args.matrix_manifest,
            "--config",
            args.config,
            "--base-planner-config",
            args.base_planner_config,
            "--ft-planner-config",
            args.ft_planner_config,
            "--target-toolset",
            args.matrix_target_toolset,
            "--toolsets",
            args.matrix_toolsets,
            "--max-tasks",
            str(args.matrix_max_tasks),
            "--label",
            matrix_label,
        ],
        cwd=repo_root,
    )

    print("[northstar] Done")
    print("[northstar] Base multi-seed:", f"runs/{base_ms_label}/multi_seed.json")
    print("[northstar] FT multi-seed:", f"runs/{ft_ms_label}/multi_seed.json")
    print("[northstar] Delta JSON:", delta_json)
    print("[northstar] Delta report:", delta_md)
    print("[northstar] Matrix JSON:", f"runs/{matrix_label}/matrix.json")
    print("[northstar] Matrix report:", f"runs/{matrix_label}/matrix_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

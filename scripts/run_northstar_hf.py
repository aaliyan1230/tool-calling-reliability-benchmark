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
    parser.add_argument(
        "--run-study-gate",
        action="store_true",
        help="Run tcrb study-gate after base/ft/delta/matrix stages",
    )
    parser.add_argument(
        "--study-gate-null-run",
        default=None,
        help="Optional null-control run JSON path for study-gate",
    )
    parser.add_argument(
        "--study-gate-matrix-json",
        default=None,
        help="Optional matrix JSON path override for study-gate",
    )
    parser.add_argument(
        "--study-gate-flatline-epsilon",
        type=float,
        default=1e-4,
        help="Flatline epsilon threshold for study-gate",
    )
    parser.add_argument(
        "--study-gate-min-effect-vs-null",
        type=float,
        default=3e-3,
        help="Min effect-vs-null threshold for study-gate",
    )
    parser.add_argument(
        "--study-gate-matrix-flatline-epsilon",
        type=float,
        default=1e-4,
        help="Transfer-matrix flatline epsilon for study-gate",
    )
    parser.add_argument(
        "--study-gate-require-matrix-signal",
        action="store_true",
        help="Require non-flat transfer-matrix signal in study-gate",
    )
    parser.add_argument(
        "--study-gate-require-matrix-not-fail",
        action="store_true",
        help="Require transfer-matrix portfolio verdict != FAIL in study-gate",
    )
    parser.add_argument(
        "--study-gate-fail-on-violation",
        action="store_true",
        help="Return non-zero exit code when study-gate verdict is FAIL",
    )
    parser.add_argument(
        "--study-gate-output-json",
        default=None,
        help="Optional output JSON path for study-gate report",
    )
    parser.add_argument(
        "--study-gate-output-report",
        default=None,
        help="Optional output markdown path for study-gate report",
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
    base_ms_json = f"runs/{base_ms_label}/multi_seed.json"
    ft_ms_json = f"runs/{ft_ms_label}/multi_seed.json"
    matrix_json = f"runs/{matrix_label}/matrix.json"

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
            base_ms_json,
            "--finetuned-run",
            ft_ms_json,
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

    study_gate_json: str | None = None
    study_gate_report: str | None = None
    if args.run_study_gate:
        study_gate_json = (
            args.study_gate_output_json
            if args.study_gate_output_json
            else f"runs/{args.label_prefix}-study-gate/study_gate.json"
        )
        study_gate_report = args.study_gate_output_report
        matrix_input = (
            args.study_gate_matrix_json if args.study_gate_matrix_json else matrix_json
        )

        study_gate_cmd = [
            "uv",
            "run",
            "tcrb",
            "study-gate",
            "--base-run",
            base_ms_json,
            "--finetuned-run",
            ft_ms_json,
            "--matrix-json",
            matrix_input,
            "--flatline-epsilon",
            str(args.study_gate_flatline_epsilon),
            "--min-effect-vs-null",
            str(args.study_gate_min_effect_vs_null),
            "--matrix-flatline-epsilon",
            str(args.study_gate_matrix_flatline_epsilon),
            "--output-json",
            study_gate_json,
        ]
        if args.study_gate_null_run:
            study_gate_cmd.extend(["--null-run", args.study_gate_null_run])
        if study_gate_report:
            study_gate_cmd.extend(["--output-report", study_gate_report])
        if args.study_gate_require_matrix_signal:
            study_gate_cmd.append("--require-matrix-signal")
        if args.study_gate_require_matrix_not_fail:
            study_gate_cmd.append("--require-matrix-not-fail")
        if args.study_gate_fail_on_violation:
            study_gate_cmd.append("--fail-on-violation")

        run_cmd(study_gate_cmd, cwd=repo_root)

    print("[northstar] Done")
    print("[northstar] Base multi-seed:", base_ms_json)
    print("[northstar] FT multi-seed:", ft_ms_json)
    print("[northstar] Delta JSON:", delta_json)
    print("[northstar] Delta report:", delta_md)
    print("[northstar] Matrix JSON:", matrix_json)
    print("[northstar] Matrix report:", f"runs/{matrix_label}/matrix_summary.md")
    if study_gate_json:
        print("[northstar] Study gate JSON:", study_gate_json)
        if study_gate_report:
            print("[northstar] Study gate report:", study_gate_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

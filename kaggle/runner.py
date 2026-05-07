from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


KERNEL_SOURCE_ROOT = Path(__file__).resolve().parent
DATASET_SOURCE_ROOT = Path("/kaggle/input/tcrb-repo-snapshot")
REPO_ROOT = Path("/tmp/tcrb_repo")
ARTIFACT_ROOT = Path("/kaggle/working/kaggle_artifacts")
LOG_DIR = ARTIFACT_ROOT / "logs"


def env(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip() or default


def env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def run(
    cmd: list[str],
    *,
    log_name: str,
    extra_env: dict[str, str] | None = None,
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    print("[kaggle-runner]", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        command_env = dict(os.environ)
        src_path = str(REPO_ROOT / "src")
        command_env["PYTHONPATH"] = (
            src_path
            if not command_env.get("PYTHONPATH")
            else f"{src_path}:{command_env['PYTHONPATH']}"
        )
        if extra_env:
            command_env.update(extra_env)
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=command_env,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}; see {log_path}"
        )


def stage_repo() -> None:
    if REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)
    source_root = DATASET_SOURCE_ROOT if DATASET_SOURCE_ROOT.exists() else KERNEL_SOURCE_ROOT
    ignore = shutil.ignore_patterns(
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "*.pyc",
        "kaggle_artifacts",
        "runs",
    )
    shutil.copytree(source_root, REPO_ROOT, ignore=ignore)


def collect_outputs() -> None:
    runs_dir = REPO_ROOT / "runs"
    if runs_dir.exists():
        target = ARTIFACT_ROOT / "runs"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(runs_dir, target)
    research_dir = REPO_ROOT / "outputs" / "research"
    if research_dir.exists():
        target = ARTIFACT_ROOT / "outputs" / "research"
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(research_dir, target)


def write_runtime_inputs() -> tuple[str, str]:
    runtime_dir = REPO_ROOT / "tmp_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    config_path = REPO_ROOT / env("TCRB_CONFIG", "configs/baseline.json")
    workload_path = REPO_ROOT / env(
        "TCRB_WORKLOAD",
        "workloads/enriched/customer_support.json",
    )

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    policies = [
        token.strip()
        for token in env("TCRB_POLICIES", "naive_retry").split(",")
        if token.strip()
    ]
    if policies:
        config_payload["policies"] = policies
    runtime_config = runtime_dir / "baseline.runtime.json"
    runtime_config.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    workload_payload = json.loads(workload_path.read_text(encoding="utf-8"))
    max_tasks = int(env("TCRB_TARGET_MAX_TASKS", "18"))
    if max_tasks > 0:
        workload_payload["tasks"] = list(workload_payload.get("tasks", []))[:max_tasks]
    runtime_workload = runtime_dir / "target.runtime.json"
    runtime_workload.write_text(json.dumps(workload_payload, indent=2), encoding="utf-8")

    return str(runtime_config), str(runtime_workload)


def write_runtime_recipe(recipe_path: str, *, runtime_name: str) -> str:
    runtime_dir = REPO_ROOT / "tmp_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    recipe_payload = json.loads((REPO_ROOT / recipe_path).read_text(encoding="utf-8"))
    overrides: dict[str, tuple[str, type]] = {
        "learning_rate": ("TCRB_REPAIR_LEARNING_RATE", float),
        "num_train_epochs": ("TCRB_REPAIR_NUM_TRAIN_EPOCHS", float),
        "per_device_train_batch_size": ("TCRB_REPAIR_PER_DEVICE_BATCH", int),
        "gradient_accumulation_steps": ("TCRB_REPAIR_GRAD_ACCUM", int),
        "max_seq_length": ("TCRB_REPAIR_MAX_SEQ_LENGTH", int),
        "beta": ("TCRB_REPAIR_BETA", float),
    }
    for field_name, (env_name, caster) in overrides.items():
        raw = str(os.environ.get(env_name, "")).strip()
        if raw:
            recipe_payload[field_name] = caster(raw)

    output_dir = str(os.environ.get("TCRB_REPAIR_OUTPUT_DIR", "")).strip()
    if output_dir:
        recipe_payload["output_dir"] = output_dir
    adapter_path = str(os.environ.get("TCRB_REPAIR_ADAPTER_PATH", "")).strip()
    if adapter_path:
        recipe_payload["adapter_path"] = adapter_path

    runtime_recipe = runtime_dir / runtime_name
    runtime_recipe.write_text(
        json.dumps(recipe_payload, indent=2),
        encoding="utf-8",
    )
    return str(runtime_recipe)


def main() -> int:
    stage_repo()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = ARTIFACT_ROOT / "status.json"
    run_mode = env("TCRB_RUN_MODE", "router_repair")
    label_prefix = env("TCRB_LABEL_PREFIX", "kaggle-hf-router-repair-taskscope-seed11")
    runtime_config, runtime_workload = write_runtime_inputs()
    benchmark_seeds = env(
        "TCRB_SEEDS",
        "11,13,17" if run_mode == "router_repair" else "11",
    )

    base_planner_config = env(
        "TCRB_BASE_PLANNER_CONFIG",
        "configs/planners/hf_qwen2_5_3b_base_taskscope_strict.json",
    )
    comparison_planner_config = env(
        "TCRB_COMPARISON_PLANNER_CONFIG",
        "configs/planners/hf_qwen2_5_3b_comparison_taskscope_strict.json",
    )

    if run_mode == "choice_probe":
        command = [
            sys.executable,
            "scripts/run_hf_choice_probe.py",
            "--workload",
            runtime_workload,
            "--eval-cases-json",
            env(
                "TCRB_EVAL_CASES_JSON",
                "workloads/eval_cases/customer_support_eval_cases.json",
            ),
            "--base-planner-config",
            base_planner_config,
            "--comparison-planner-config",
            comparison_planner_config,
            "--max-tasks",
            env("TCRB_TARGET_MAX_TASKS", "18"),
            "--output-dir",
            f"runs/{label_prefix}",
        ]
    elif run_mode == "choice_matrix":
        command = [
            sys.executable,
            "scripts/run_hf_choice_matrix.py",
            "--manifest",
            env("TCRB_MATRIX_MANIFEST", "workloads/enriched/manifest.json"),
            "--base-planner-config",
            base_planner_config,
            "--comparison-planner-config",
            comparison_planner_config,
            "--max-tasks",
            env("TCRB_MATRIX_MAX_TASKS", "18"),
            "--output-dir",
            f"runs/{label_prefix}",
        ]
        toolsets = env("TCRB_MATRIX_TOOLSETS", "")
        if toolsets:
            command.extend(["--toolsets", toolsets])
    elif run_mode == "router_repair":
        repair_recipe = write_runtime_recipe(
            env(
                "TCRB_REPAIR_RECIPE_CONFIG",
                "configs/research/dpo_router_repair_qwen25_3b.json",
            ),
            runtime_name="router_repair.runtime.json",
        )
        repair_dataset_jsonl = env(
            "TCRB_REPAIR_DATASET_JSONL",
            "outputs/research/qwen25-3b-dpo-router-repair/dpo_train.jsonl",
        )
        repair_comparison_planner_config = env(
            "TCRB_REPAIR_COMPARISON_PLANNER_CONFIG",
            "configs/planners/hf_qwen2_5_3b_router_repair_taskscope_strict.json",
        )
        command = [
            sys.executable,
            "scripts/run_hf_choice_matrix.py",
            "--manifest",
            env("TCRB_MATRIX_MANIFEST", "workloads/enriched/manifest.json"),
            "--base-planner-config",
            env(
                "TCRB_BASE_PLANNER_CONFIG",
                "configs/planners/hf_qwen2_5_3b_base_taskscope_strict.json",
            ),
            "--comparison-planner-config",
            repair_comparison_planner_config,
            "--max-tasks",
            env("TCRB_MATRIX_MAX_TASKS", "18"),
            "--output-dir",
            f"runs/{label_prefix}-choice-matrix",
        ]
        toolsets = env("TCRB_MATRIX_TOOLSETS", "")
        if toolsets:
            command.extend(["--toolsets", toolsets])
        benchmark_command = [
            sys.executable,
            "scripts/run_northstar_hf.py",
            "--workload",
            runtime_workload,
            "--config",
            runtime_config,
            "--seeds",
            benchmark_seeds,
            "--base-planner-config",
            env(
                "TCRB_BASE_PLANNER_CONFIG",
                "configs/planners/hf_qwen2_5_3b_base_taskscope_strict.json",
            ),
            "--comparison-planner-config",
            repair_comparison_planner_config,
            "--label-prefix",
            label_prefix,
            "--skip-matrix",
            "--run-study-gate",
            "--run-summarize",
        ]
    else:
        command = [
            sys.executable,
            "scripts/run_northstar_hf.py",
            "--workload",
            runtime_workload,
            "--config",
            runtime_config,
            "--seeds",
            benchmark_seeds,
            "--base-planner-config",
            base_planner_config,
            "--comparison-planner-config",
            comparison_planner_config,
            "--matrix-manifest",
            env("TCRB_MATRIX_MANIFEST", "workloads/enriched/manifest.json"),
            "--matrix-target-toolset",
            env("TCRB_MATRIX_TARGET_TOOLSET", "customer_support"),
            "--matrix-toolsets",
            env("TCRB_MATRIX_TOOLSETS", "customer_support"),
            "--matrix-max-tasks",
            env("TCRB_MATRIX_MAX_TASKS", "18"),
            "--label-prefix",
            label_prefix,
            "--run-study-gate",
            "--run-summarize",
        ]

        if env_flag("TCRB_SKIP_MATRIX", default=True):
            command.append("--skip-matrix")
        if env_flag("TCRB_REQUIRE_MATRIX_SIGNAL", default=False):
            command.append("--study-gate-require-matrix-signal")
        if env_flag("TCRB_REQUIRE_MATRIX_NOT_FAIL", default=False):
            command.append("--study-gate-require-matrix-not-fail")

    try:
        if run_mode == "router_repair":
            run(
                [sys.executable, "-m", "pip", "install", "-e", ".[research]"],
                log_name="install.log",
            )
            run(
                [
                    sys.executable,
                    "-m",
                    "tcrb.cli",
                    "train-dpo",
                    "--recipe-config",
                    repair_recipe,
                    "--dataset-jsonl",
                    repair_dataset_jsonl,
                ],
                log_name="train_dpo.log",
                extra_env={"CUDA_VISIBLE_DEVICES": env("TCRB_REPAIR_CUDA_DEVICES", "0")},
            )
            run(command, log_name="choice_matrix.log")
            run(benchmark_command, log_name="northstar.log")
        else:
            run(
                [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"],
                log_name="install.log",
            )
            run(command, log_name="northstar.log")
        collect_outputs()
        status = {
            "status": "success",
            "label_prefix": label_prefix,
            "run_mode": run_mode,
            "analysis_summary": (
                f"runs/{label_prefix}/choice_probe.md"
                if run_mode == "choice_probe"
                else f"runs/{label_prefix}/choice_matrix.md"
                if run_mode == "choice_matrix"
                else f"runs/{label_prefix}-analysis/analysis_summary.md"
                if run_mode == "router_repair"
                else f"runs/{label_prefix}-analysis/analysis_summary.md"
            ),
        }
        if run_mode == "router_repair":
            status["choice_matrix_summary"] = (
                f"runs/{label_prefix}-choice-matrix/choice_matrix.md"
            )
            status["adapter_output_dir"] = env(
                "TCRB_REPAIR_OUTPUT_DIR",
                "outputs/research/qwen25-3b-dpo-router-repair",
            )
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2), flush=True)
        return 0
    except Exception as exc:
        collect_outputs()
        status = {
            "status": "failed",
            "label_prefix": label_prefix,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

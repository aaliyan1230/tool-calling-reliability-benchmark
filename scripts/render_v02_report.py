#!/usr/bin/env python3
"""Render paper figures from the compact TCRB v0.2 results artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPERIMENT_ORDER = ["default_baseline", "recovery_prompt", "sft_corrected"]
EXPERIMENT_COLORS = ["#355070", "#6d597a", "#b56576"]
HAZARD_ORDER = [
    "execution_error",
    "schema_drift",
    "partial_output",
    "silent_corruption",
    "cross_source_conflict",
]
QUADRANT_ORDER = [
    "seen-domain/seen-hazard",
    "seen-domain/unseen-hazard",
    "unseen-domain/seen-hazard",
    "unseen-domain/unseen-hazard",
]


def _rate(item: dict[str, int]) -> float:
    return item["passed"] / item["total"] if item["total"] else 0.0


def _recovery_rate(item: dict[str, int]) -> float:
    return item["recovery"] / item["applied"] if item["applied"] else 0.0


def _load_plotting():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: pip install matplotlib") from exc
    return plt


def _experiments(payload: dict) -> list[dict]:
    by_id = {experiment["id"]: experiment for experiment in payload["experiments"]}
    return [by_id[experiment_id] for experiment_id in EXPERIMENT_ORDER]


def _short_labels(experiments: list[dict]) -> list[str]:
    return ["Default", "Recovery prompt", "Corrected SFT"]


def render_main_rates(plt, experiments: list[dict], output: Path) -> None:
    labels = _short_labels(experiments)
    x = list(range(len(experiments)))
    width = 0.34
    clean = [_rate(experiment["clean"]) * 100 for experiment in experiments]
    faulted = [_rate(experiment["faulted"]) * 100 for experiment in experiments]

    fig, ax = plt.subplots(figsize=(8, 5))
    clean_bars = ax.bar(
        [value - width / 2 for value in x],
        clean,
        width,
        label="Clean",
        color="#4f81bd",
    )
    faulted_bars = ax.bar(
        [value + width / 2 for value in x],
        faulted,
        width,
        label="Faulted",
        color="#c0504d",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 34)
    ax.set_title("TCRB v0.2: clean versus faulted performance")
    ax.bar_label(clean_bars, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(faulted_bars, fmt="%.1f", padding=2, fontsize=8)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_hazard_recovery(plt, experiments: list[dict], output: Path) -> None:
    labels = [hazard.replace("_", "\n") for hazard in HAZARD_ORDER]
    x = list(range(len(HAZARD_ORDER)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, (experiment, color) in enumerate(zip(experiments, EXPERIMENT_COLORS)):
        values = [
            _recovery_rate(experiment["hazards"][hazard]) * 100
            for hazard in HAZARD_ORDER
        ]
        offsets = [value + (index - 1) * width for value in x]
        ax.bar(offsets, values, width, label=experiment["label"], color=color)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Recovery after applied fault (%)")
    ax.set_ylim(0, 55)
    ax.set_title("Recovery by failure type")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_transfer_heatmap(plt, experiments: list[dict], output: Path) -> None:
    import numpy as np

    matrix = np.array(
        [
            [_rate(experiment["quadrants"][quadrant]) * 100 for quadrant in QUADRANT_ORDER]
            for experiment in experiments
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=16, aspect="auto")
    ax.set_xticks(range(len(QUADRANT_ORDER)), [
        "Seen / seen",
        "Seen / unseen",
        "Unseen / seen",
        "Unseen / unseen",
    ])
    ax.set_yticks(range(len(experiments)), ["Default", "Recovery prompt", "Corrected SFT"])
    ax.set_title("Transfer performance by domain and failure familiarity")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f"{value:.1f}%", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Faulted pass rate (%)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_clean_categories(plt, experiments: list[dict], output: Path) -> None:
    categories = ["tool_required", "no_tool", "missing_information"]
    labels = ["Tool required", "No tool", "Missing information"]
    x = list(range(len(categories)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, (experiment, color) in enumerate(zip(experiments, EXPERIMENT_COLORS)):
        values = [_rate(experiment["categories"][category]) * 100 for category in categories]
        offsets = [value + (index - 1) * width for value in x]
        ax.bar(offsets, values, width, label=experiment["label"], color=color)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Clean pass rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Clean-task behavior by task category")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("reports/v02/results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/v02/figures"),
    )
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_plotting()
    experiments = _experiments(payload)
    render_main_rates(plt, experiments, args.output_dir / "main_rates.png")
    render_hazard_recovery(plt, experiments, args.output_dir / "hazard_recovery.png")
    render_transfer_heatmap(plt, experiments, args.output_dir / "transfer_heatmap.png")
    render_clean_categories(plt, experiments, args.output_dir / "clean_categories.png")
    print(f"Rendered figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

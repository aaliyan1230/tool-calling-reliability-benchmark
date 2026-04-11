from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STANDARD_ARTIFACT_NAMES = {
    "multi_seed": "multi_seed.json",
    "delta": "delta-ms.json",
    "matrix": "matrix.json",
    "study_gate": "study_gate.json",
}

STANDARD_PLOT_NAMES = {
    "multi_seed": "multi_seed_overview.png",
    "delta": "delta_policy.png",
    "matrix": "transfer_matrix.png",
}


@dataclass(frozen=True)
class AnalysisArtifacts:
    run_dir: Path | None = None
    multi_seed_json: Path | None = None
    delta_json: Path | None = None
    matrix_json: Path | None = None
    study_gate_json: Path | None = None


@dataclass(frozen=True)
class AnalysisPayloads:
    multi_seed: dict[str, Any] | None = None
    delta: dict[str, Any] | None = None
    matrix: dict[str, Any] | None = None
    study_gate: dict[str, Any] | None = None


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _resolve_artifact_path(
    *,
    explicit: str | Path | None,
    run_dir: Path | None,
    artifact_name: str,
) -> Path | None:
    explicit_path = _optional_path(explicit)
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Artifact JSON not found: {explicit_path}")
        return explicit_path

    if run_dir is None:
        return None

    candidate = run_dir / artifact_name
    return candidate if candidate.exists() else None


def resolve_analysis_artifacts(
    *,
    run_dir: str | Path | None = None,
    multi_seed_json: str | Path | None = None,
    delta_json: str | Path | None = None,
    matrix_json: str | Path | None = None,
    study_gate_json: str | Path | None = None,
) -> AnalysisArtifacts:
    resolved_run_dir = _optional_path(run_dir)
    if resolved_run_dir is not None:
        if not resolved_run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {resolved_run_dir}")
        if not resolved_run_dir.is_dir():
            raise ValueError(f"Run directory is not a directory: {resolved_run_dir}")

    artifacts = AnalysisArtifacts(
        run_dir=resolved_run_dir,
        multi_seed_json=_resolve_artifact_path(
            explicit=multi_seed_json,
            run_dir=resolved_run_dir,
            artifact_name=STANDARD_ARTIFACT_NAMES["multi_seed"],
        ),
        delta_json=_resolve_artifact_path(
            explicit=delta_json,
            run_dir=resolved_run_dir,
            artifact_name=STANDARD_ARTIFACT_NAMES["delta"],
        ),
        matrix_json=_resolve_artifact_path(
            explicit=matrix_json,
            run_dir=resolved_run_dir,
            artifact_name=STANDARD_ARTIFACT_NAMES["matrix"],
        ),
        study_gate_json=_resolve_artifact_path(
            explicit=study_gate_json,
            run_dir=resolved_run_dir,
            artifact_name=STANDARD_ARTIFACT_NAMES["study_gate"],
        ),
    )

    if not any(
        path is not None
        for path in (
            artifacts.multi_seed_json,
            artifacts.delta_json,
            artifacts.matrix_json,
            artifacts.study_gate_json,
        )
    ):
        raise ValueError(
            "No analysis artifact JSON files were found. Provide --run-dir with standard files or explicit --multi-seed-json/--delta-json/--matrix-json/--study-gate-json paths."
        )

    return artifacts


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_analysis_payloads(artifacts: AnalysisArtifacts) -> AnalysisPayloads:
    return AnalysisPayloads(
        multi_seed=(
            _load_json(artifacts.multi_seed_json)
            if artifacts.multi_seed_json is not None
            else None
        ),
        delta=_load_json(artifacts.delta_json) if artifacts.delta_json is not None else None,
        matrix=(
            _load_json(artifacts.matrix_json)
            if artifacts.matrix_json is not None
            else None
        ),
        study_gate=(
            _load_json(artifacts.study_gate_json)
            if artifacts.study_gate_json is not None
            else None
        ),
    )


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        return plt, TwoSlopeNorm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plot rendering. Install the dev extras with 'uv sync --extra dev'."
        ) from exc


def _write_figure(fig, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def write_delta_policy_plot(payload: dict[str, Any], output_path: str | Path) -> Path:
    sections: list[tuple[str, dict[str, Any]]] = []
    for section_name in ("target", "open"):
        section = payload.get(section_name)
        if isinstance(section, dict) and section.get("policies"):
            sections.append((section_name, section))

    if not sections:
        raise ValueError("Delta payload does not contain any policy rows to plot")

    plt, _ = _load_matplotlib()
    fig, axes = plt.subplots(len(sections), 2, figsize=(12, 4 * len(sections)), squeeze=False)

    for row_index, (section_name, section_payload) in enumerate(sections):
        rows = list(section_payload.get("policies", []))
        policies = [str(row.get("policy", "unknown")) for row in rows]
        success = [
            float(row.get("delta", {}).get("task_success_rate", 0.0) or 0.0)
            for row in rows
        ]
        invalid = [
            float(row.get("delta", {}).get("invalid_tool_call_rate", 0.0) or 0.0)
            for row in rows
        ]

        success_axis = axes[row_index][0]
        success_axis.bar(policies, success, color="#2f7d32")
        success_axis.axhline(0.0, color="black", linewidth=1)
        success_axis.set_title(f"{section_name.title()} Success Delta")
        success_axis.set_ylabel("comparison - base")
        success_axis.tick_params(axis="x", rotation=25)

        invalid_axis = axes[row_index][1]
        invalid_axis.bar(policies, invalid, color="#b23c17")
        invalid_axis.axhline(0.0, color="black", linewidth=1)
        invalid_axis.set_title(f"{section_name.title()} Invalid Call Delta")
        invalid_axis.set_ylabel("comparison - base")
        invalid_axis.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    path = _write_figure(fig, output_path)
    plt.close(fig)
    return path


def write_transfer_matrix_plot(payload: dict[str, Any], output_path: str | Path) -> Path:
    rows = list(payload.get("rows", []))
    if not rows:
        raise ValueError("Transfer matrix payload does not contain any rows to plot")

    plt, two_slope_norm = _load_matplotlib()
    labels = [
        f"{row.get('toolset_id', 'unknown')} [{row.get('split', 'open')}/{row.get('verdict', 'FAIL')}]"
        for row in rows
    ]
    values = [
        [
            float(row.get("delta_first_tool_accuracy", 0.0) or 0.0),
            float(row.get("delta_sequence_prefix_accuracy", 0.0) or 0.0),
        ]
        for row in rows
    ]
    max_abs = max(abs(value) for row in values for value in row)
    scale = max(max_abs, 0.01)

    fig, axis = plt.subplots(figsize=(8.5, max(3.2, len(rows) * 0.8 + 1.8)))
    image = axis.imshow(
        values,
        cmap="RdYlGn",
        aspect="auto",
        norm=two_slope_norm(vmin=-scale, vcenter=0.0, vmax=scale),
    )
    axis.set_title("Transfer Matrix Deltas")
    axis.set_xticks([0, 1], labels=["first tool", "sequence"])
    axis.set_yticks(list(range(len(labels))), labels=labels)

    for row_index, row in enumerate(values):
        for col_index, value in enumerate(row):
            axis.text(
                col_index,
                row_index,
                f"{value:+.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    fig.colorbar(image, ax=axis, fraction=0.05, pad=0.04, label="delta")
    fig.tight_layout()
    path = _write_figure(fig, output_path)
    plt.close(fig)
    return path


def write_multi_seed_overview_plot(payload: dict[str, Any], output_path: str | Path) -> Path:
    rows = list(payload.get("aggregate_policy_metrics", []))
    if not rows:
        raise ValueError("Multi-seed payload does not contain aggregate policy metrics")

    plt, _ = _load_matplotlib()
    policies = [str(row.get("policy", "unknown")) for row in rows]
    metric_specs = [
        ("task_success_rate", "Success Rate", 4, "#2f7d32"),
        ("invalid_tool_call_rate", "Invalid Call Rate", 4, "#b23c17"),
        ("p95_latency_ms", "P95 Latency (ms)", 2, "#1b6ca8"),
        (
            "estimated_cost_per_successful_task_usd",
            "Cost per Success (USD)",
            6,
            "#8c564b",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for axis, (metric_name, title, digits, color) in zip(axes.flat, metric_specs):
        means = [
            float(row.get("metrics", {}).get(metric_name, {}).get("mean", 0.0) or 0.0)
            for row in rows
        ]
        cis = [
            float(
                row.get("metrics", {})
                .get(metric_name, {})
                .get("ci95_half_width", 0.0)
                or 0.0
            )
            for row in rows
        ]

        axis.bar(policies, means, color=color, alpha=0.9)
        axis.errorbar(
            policies,
            means,
            yerr=cis,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=4,
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.yaxis.get_major_formatter().set_useOffset(False)
        if digits >= 5:
            axis.ticklabel_format(axis="y", style="plain")

    fig.suptitle("Multi-Seed Policy Overview", fontsize=14)
    fig.tight_layout()
    path = _write_figure(fig, output_path)
    plt.close(fig)
    return path


def render_analysis_plots(
    payloads: AnalysisPayloads,
    output_dir: str | Path,
) -> dict[str, Path]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    if payloads.multi_seed is not None:
        outputs["multi_seed"] = write_multi_seed_overview_plot(
            payloads.multi_seed,
            outdir / STANDARD_PLOT_NAMES["multi_seed"],
        )

    if payloads.delta is not None:
        outputs["delta"] = write_delta_policy_plot(
            payloads.delta,
            outdir / STANDARD_PLOT_NAMES["delta"],
        )

    if payloads.matrix is not None:
        outputs["matrix"] = write_transfer_matrix_plot(
            payloads.matrix,
            outdir / STANDARD_PLOT_NAMES["matrix"],
        )

    return outputs
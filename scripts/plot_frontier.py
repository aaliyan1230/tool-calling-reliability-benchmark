#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_points(payload: dict) -> list[dict]:
    rows = payload.get("aggregate_policy_metrics", [])
    points = []
    for row in rows:
        metrics = row.get("metrics", {})
        points.append(
            {
                "policy": row.get("policy", "unknown"),
                "success": float(metrics.get("task_success_rate", {}).get("mean", 0.0)),
                "latency": float(metrics.get("p95_latency_ms", {}).get("mean", 0.0)),
                "cost": float(
                    metrics.get("estimated_cost_per_successful_task_usd", {}).get(
                        "mean", 0.0
                    )
                ),
            }
        )
    return points


def plot_frontier(points: list[dict], out_path: str | Path, title: str) -> None:
    if not points:
        raise ValueError("No points found in input payload")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install -e .[dev]"
        ) from exc

    fig, ax = plt.subplots(figsize=(9, 6))
    success_vals = [point["success"] for point in points]
    latency_vals = [point["latency"] for point in points]
    cost_vals = [point["cost"] for point in points]

    min_cost = min(cost_vals)
    max_cost = max(cost_vals)
    spread = max_cost - min_cost
    sizes = [
        90 + (0 if spread == 0 else ((cost - min_cost) / spread) * 420)
        for cost in cost_vals
    ]

    ax.scatter(latency_vals, success_vals, s=sizes, alpha=0.75)
    for point in points:
        ax.annotate(
            point["policy"],
            (point["latency"], point["success"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    ax.set_xlabel("p95 latency (ms)")
    ax.set_ylabel("task success rate")
    ax.set_title(title)
    ax.grid(alpha=0.2)

    legend_text = "Point size ~ cost per success (USD)"
    ax.text(0.01, 0.01, legend_text, transform=ax.transAxes, fontsize=9)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot success-latency-cost frontier")
    parser.add_argument("--input", required=True, help="Path to multi_seed.json")
    parser.add_argument(
        "--output", default="runs/frontier.png", help="Output image path"
    )
    parser.add_argument(
        "--title", default="Policy Frontier: Success vs p95 Latency", help="Plot title"
    )
    args = parser.parse_args()

    payload = _read_json(args.input)
    points = _collect_points(payload)
    plot_frontier(points=points, out_path=args.output, title=args.title)
    print(f"Wrote frontier plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

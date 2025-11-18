"""Visualization utilities for refusal benchmark metrics.

Generates interactive HTML charts summarizing aggregate metrics and
perturbation diagnostics. Charts are saved to the specified figure
directory for documentation or report inclusion.

Usage:
    python scripts/visualize_results.py \
        --metrics results/metrics/aggregate_run1.csv \
        --diagnostics results/metrics/diagnostics_run1.csv \
        --figdir results/figures/run1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import altair as alt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize refusal metrics")
    parser.add_argument("--metrics", type=Path, required=True, help="CSV from evaluate_metrics.py")
    parser.add_argument(
        "--diagnostics",
        type=Path,
        required=False,
        help="CSV from perturbation_analysis.py (optional)",
    )
    parser.add_argument("--figdir", type=Path, required=True, help="Output directory for HTML charts")
    return parser.parse_args()


def ensure_output_dir(figdir: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)


def metric_heatmap(df: pd.DataFrame, metric: str, title: str) -> alt.Chart:
    filtered = df[df["metric"] == metric]
    if filtered.empty:
        raise ValueError(f"Metric '{metric}' not present in dataframe")

    chart = (
        alt.Chart(filtered)
        .mark_rect()
        .encode(
            x=alt.X("slice:N", title="Slice"),
            y=alt.Y("category:N", title="Category"),
            color=alt.Color("value:Q", title="Value", scale=alt.Scale(scheme="redyellowgreen")),
            tooltip=["slice", "category", "risk_level", "value"],
        )
        .properties(width=500, height=200, title=title)
    )
    return chart


def stacked_risk_chart(df: pd.DataFrame, metric: str, title: str) -> alt.Chart:
    filtered = df[df["metric"] == metric]
    if filtered.empty:
        raise ValueError(f"Metric '{metric}' not present for stacked chart")

    chart = (
        alt.Chart(filtered)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title="Metric value"),
            y=alt.Y("slice:N", title="Slice"),
            color=alt.Color("risk_level:N", title="Risk level"),
            tooltip=["slice", "risk_level", "value"],
        )
        .properties(width=500, title=title)
    )
    return chart


def perturbation_chart(diag_df: pd.DataFrame) -> alt.Chart:
    if diag_df.empty:
        raise ValueError("Diagnostics dataframe is empty")

    melt = diag_df.melt(
        id_vars=["parent_id", "num_variants"],
        value_vars=["mismatch_rate", "refusal_flip_rate"],
        var_name="metric",
        value_name="value",
    )

    chart = (
        alt.Chart(melt)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title="Rate"),
            y=alt.Y("parent_id:N", sort="-x", title="Parent prompt"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["parent_id", "metric", "value", "num_variants"],
        )
        .properties(width=600, title="Perturbation Consistency Metrics")
    )
    return chart


def save_chart(chart: alt.Chart, path: Path) -> None:
    chart.save(str(path))
    print(f"Saved chart -> {path}")


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.figdir)

    metrics_df = pd.read_csv(args.metrics)
    heatmap = metric_heatmap(metrics_df, "auto_refusal_rate", "Auto Refusal Rate Heatmap")
    stacked = stacked_risk_chart(metrics_df, "auto_harmful_compliance_rate", "Harmful Compliance by Slice & Risk")

    save_chart(heatmap, args.figdir / "auto_refusal_heatmap.html")
    save_chart(stacked, args.figdir / "harmful_compliance_stack.html")

    if args.diagnostics and args.diagnostics.exists():
        diag_df = pd.read_csv(args.diagnostics)
        chart = perturbation_chart(diag_df)
        save_chart(chart, args.figdir / "perturbation_consistency.html")


if __name__ == "__main__":
    main()

"""Analyze robustness via parent/diagnostic prompt pairs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from schema import load_auto_labels, load_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perturbation robustness analysis")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--auto-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def compute_consistency(df: pd.DataFrame) -> pd.DataFrame:
    parent_map: Dict[str, str] = (
        df.loc[df["parent_id"].isna(), ["id"]].assign(parent=lambda s: s["id"])
    ).set_index("id")["parent"]

    df["root_parent"] = df.apply(
        lambda row: row["id"] if pd.isna(row["parent_id"]) else row["parent_id"], axis=1
    )

    agg = []
    for parent_id, group in df.groupby("root_parent"):
        if len(group) <= 1:
            continue
        base = group[group["id"] == parent_id]
        base_label = base["auto_label"].iat[0] if not base.empty else None
        mismatch_rate = (group["auto_label"] != base_label).mean() if base_label else float("nan")
        refusal_flip_rate = (
            group["auto_label"].isin(["refusal", "mixed"]) != (base_label in ["refusal", "mixed"])
        ).mean() if base_label else float("nan")
        agg.append(
            {
                "parent_id": parent_id,
                "num_variants": len(group) - 1,
                "base_label": base_label,
                "mismatch_rate": mismatch_rate,
                "refusal_flip_rate": refusal_flip_rate,
            }
        )
    return pd.DataFrame(agg)


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    labels = load_auto_labels(args.auto_labels)

    prompts_df = pd.DataFrame([p.dict() for p in prompts])
    labels_df = pd.DataFrame([l.dict() for l in labels])
    merged = labels_df.merge(prompts_df, left_on="prompt_id", right_on="id", suffixes=("", "_prompt"))

    diag = merged[merged["parent_id"].notna()]
    if diag.empty:
        raise ValueError("No diagnostic prompts found; add perturbations to prompts file.")

    stats = compute_consistency(diag)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.out, index=False)
    print(f"Saved perturbation diagnostics to {args.out}")


if __name__ == "__main__":
    main()

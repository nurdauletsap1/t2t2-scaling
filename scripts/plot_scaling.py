"""Generate 4 publication-quality scaling plots."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", palette="colorblind")
SIZE_ORDER = ["tiny", "small", "medium", "large"]
K_VALUES = [1, 4, 8, 16]


def _extract_size_label(model_size: str) -> str:
    """Pull the size token ('tiny', 'small', …) out of model_size field."""
    for s in SIZE_ORDER:
        if s in str(model_size):
            return s
    return str(model_size)


def plot_loss_vs_tokens(df: pd.DataFrame, out_path: Path) -> None:
    """Plot validation loss vs training tokens, coloured by model size."""
    if "best_val_loss" not in df.columns or df["best_val_loss"].isna().all():
        print("Warning: no best_val_loss data — skipping loss_vs_tokens.png")
        return

    sub = df[["D_tokens", "best_val_loss", "model_size", "seed"]].dropna()
    sub["size"] = sub["model_size"].apply(_extract_size_label)

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = sns.color_palette("colorblind", n_colors=len(SIZE_ORDER))
    color_map = dict(zip(SIZE_ORDER, palette))

    for (size, seed), grp in sub.groupby(["size", "seed"]):
        grp = grp.sort_values("D_tokens")
        ax.plot(
            grp["D_tokens"], grp["best_val_loss"],
            color=color_map.get(size, "gray"),
            alpha=0.7, linewidth=1.5,
            label=size if seed == grp["seed"].iloc[0] else "_nolegend_",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training tokens (D)", fontsize=12)
    ax.set_ylabel("Best validation loss", fontsize=12)
    ax.set_title("Validation Loss vs Training Tokens", fontsize=13)
    handles = [plt.Line2D([0], [0], color=color_map[s], lw=2, label=s) for s in SIZE_ORDER if s in sub["size"].values]
    ax.legend(handles=handles, title="Model size", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_accuracy_vs_compute(df: pd.DataFrame, out_path: Path) -> None:
    """Plot pass@8 vs total compute FLOPs, coloured by model size."""
    if "pass@8" not in df.columns or df["pass@8"].isna().all():
        print("Warning: no pass@8 data — skipping accuracy_vs_compute.png")
        return

    sub = df[["compute_flops", "pass@8", "model_size"]].dropna()
    sub["size"] = sub["model_size"].apply(_extract_size_label)

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = sns.color_palette("colorblind", n_colors=len(SIZE_ORDER))
    color_map = dict(zip(SIZE_ORDER, palette))

    for size, grp in sub.groupby("size"):
        ax.scatter(
            grp["compute_flops"], grp["pass@8"],
            color=color_map.get(size, "gray"),
            label=size, s=60, alpha=0.8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total compute (FLOPs)", fontsize=12)
    ax.set_ylabel("pass@8 accuracy", fontsize=12)
    ax.set_title("Accuracy vs Compute (pass@8)", fontsize=13)
    ax.legend(title="Model size", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_scaling_k(df: pd.DataFrame, out_path: Path) -> None:
    """Plot accuracy vs k (samples), with pass@k and maj@k panels."""
    metrics_pass = [f"pass@{k}" for k in K_VALUES if f"pass@{k}" in df.columns]
    metrics_maj = [f"maj@{k}" for k in K_VALUES if f"maj@{k}" in df.columns]

    if not metrics_pass and not metrics_maj:
        print("Warning: no pass@k/maj@k data — skipping scaling_k.png")
        return

    sub = df[["model_size"] + metrics_pass + metrics_maj].dropna(how="all")
    sub["size"] = sub["model_size"].apply(_extract_size_label)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    palette = sns.color_palette("colorblind", n_colors=len(SIZE_ORDER))
    color_map = dict(zip(SIZE_ORDER, palette))

    for size, grp in sub.groupby("size"):
        color = color_map.get(size, "gray")
        # pass@k
        if metrics_pass:
            ks = [int(m.split("@")[1]) for m in metrics_pass]
            vals = grp[metrics_pass].mean()
            axes[0].plot(ks, vals, marker="o", color=color, label=size)
        # maj@k
        if metrics_maj:
            ks = [int(m.split("@")[1]) for m in metrics_maj]
            vals = grp[metrics_maj].mean()
            axes[1].plot(ks, vals, marker="s", color=color, label=size)

    axes[0].set_title("pass@k", fontsize=12)
    axes[1].set_title("maj@k", fontsize=12)
    for ax in axes:
        ax.set_xlabel("k (samples at test time)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.legend(title="Model size", fontsize=9)

    fig.suptitle("Accuracy vs Test-Time Samples (k)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_pareto_frontier(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter all experiments and highlight the Pareto frontier."""
    if "pass@8" not in df.columns or "compute_flops" not in df.columns:
        print("Warning: missing columns — skipping pareto_frontier.png")
        return

    sub = df[["compute_flops", "pass@8", "model_size", "D_tokens"]].dropna()
    if sub.empty:
        print("Warning: no data for pareto_frontier.png")
        return

    sub = sub.sort_values("compute_flops")
    # Pareto-optimal: highest pass@8 seen so far as compute increases
    sub["cum_best"] = sub["pass@8"].cummax()
    pareto = sub[sub["pass@8"] == sub["cum_best"]].drop_duplicates(subset="compute_flops")

    fig, ax = plt.subplots(figsize=(8, 6))
    sub["size"] = sub["model_size"].apply(_extract_size_label)
    palette = sns.color_palette("colorblind", n_colors=len(SIZE_ORDER))
    color_map = dict(zip(SIZE_ORDER, palette))

    for size, grp in sub.groupby("size"):
        ax.scatter(
            grp["compute_flops"], grp["pass@8"],
            color=color_map.get(size, "gray"),
            label=size, s=50, alpha=0.6,
        )

    ax.plot(
        pareto["compute_flops"], pareto["pass@8"],
        color="red", linewidth=2, marker="*", markersize=10,
        label="Pareto frontier", zorder=5,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Total compute (FLOPs)", fontsize=12)
    ax.set_ylabel("pass@8 accuracy", fontsize=12)
    ax.set_title("Pareto Frontier: Accuracy vs Compute", fontsize=13)
    ax.legend(title="Model size", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T2T² scaling plots.")
    parser.add_argument(
        "--csv", type=Path, default=Path("results/aggregate.csv"),
        help="Aggregated CSV (default: results/aggregate.csv).",
    )
    parser.add_argument(
        "--fit", type=Path, default=Path("results/fit_results.json"),
        help="Fit results JSON (default: results/fit_results.json).",
    )
    parser.add_argument(
        "--plots-dir", type=Path, default=Path("plots"),
        help="Output directory for plots (default: plots/).",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: {args.csv} not found. Run aggregate_results.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    args.plots_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_vs_tokens(df, args.plots_dir / "loss_vs_tokens.png")
    plot_accuracy_vs_compute(df, args.plots_dir / "accuracy_vs_compute.png")
    plot_scaling_k(df, args.plots_dir / "scaling_k.png")
    plot_pareto_frontier(df, args.plots_dir / "pareto_frontier.png")


if __name__ == "__main__":
    main()

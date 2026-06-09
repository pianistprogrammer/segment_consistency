"""Generate all paper figures from saved experiment results.

Usage:
  uv run python scripts/plot_results.py [results_dir]

Produces:
  Figure 1: Scatter plot of (boundary F, kappa) per track — quadrant taxonomy
  Figure 2: Label frequency distribution at coarse vs. fine levels
  Figure 3: Level-conditional benchmark comparison (CC/FF/CF/FC bar chart)
  Figure 4: Class-balance ablation per-class recall (if results available)
  Figure 5: Training curves from JSON history logs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = sns.color_palette("tab10")

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "plots"


def save_fig(fig: plt.Figure, name: str, dpi: int = 300) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    path_png = FIGURES_DIR / f"{name}.png"
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Scatter plot — boundary F vs kappa, coloured by quadrant
# ---------------------------------------------------------------------------

def plot_agreement_scatter(results_dir: Path) -> None:
    p = results_dir / "agreement" / "results.json"
    if not p.exists():
        print(f"[skip] {p} not found")
        return
    df = pd.DataFrame(json.loads(p.read_text()))
    df = df.dropna(subset=["bf_05", "kappa"])

    quadrant_colors = {
        "high": COLORS[2],
        "boundary_only": COLORS[0],
        "label_only": COLORS[1],
        "full_disagree": COLORS[3],
    }
    quadrant_labels = {
        "high": "High agreement",
        "boundary_only": "Boundary disagreement",
        "label_only": "Label disagreement",
        "full_disagree": "Full disagreement",
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    for q, color in quadrant_colors.items():
        sub = df[df["quadrant"] == q]
        ax.scatter(sub["bf_05"], sub["kappa"], c=[color], alpha=0.5, s=20, label=quadrant_labels[q])

    ax.axvline(0.5, color="gray", lw=0.8, ls="--")
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Boundary F-measure (±0.5 s)")
    ax.set_ylabel("Cohen's κ")
    ax.set_title("Coarse–Fine Annotation Agreement per Track")
    ax.legend(loc="lower right", markerscale=1.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.05)
    save_fig(fig, "fig1_agreement_scatter")


# ---------------------------------------------------------------------------
# Figure 2: Label frequency distribution
# ---------------------------------------------------------------------------

def plot_label_distribution(results_dir: Path) -> None:
    p = results_dir / "agreement" / "results.json"
    if not p.exists():
        print(f"[skip] {p} not found")
        return

    # We reconstruct counts from n_coarse_segments and n_fine_segments
    df = pd.DataFrame(json.loads(p.read_text()))
    coarse_total = df["n_coarse_segments"].sum()
    fine_total = df["n_fine_segments"].sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Coarse (uppercase)", "Fine (lowercase)"],
        [coarse_total, fine_total],
        color=[COLORS[0], COLORS[1]],
        edgecolor="white",
    )
    ax.set_ylabel("Total segments (all tracks)")
    ax.set_title("Segment Count: Coarse vs. Fine Annotation Level")
    save_fig(fig, "fig2_segment_counts")


# ---------------------------------------------------------------------------
# Figure 3: Level-conditional benchmark bar chart
# ---------------------------------------------------------------------------

def plot_boundary_comparison(results_dir: Path) -> None:
    p = results_dir / "boundary" / "comparison_table.json"
    if not p.exists():
        print(f"[skip] {p} not found")
        return
    data = json.loads(p.read_text())

    conditions = [k for k in sorted(data) if k.startswith("cnn_")]
    foote_keys = [k for k in sorted(data) if k.startswith("foote_")]
    all_keys = foote_keys + conditions

    labels = all_keys
    f05 = [data[k].get("F_05", 0) for k in all_keys]
    f30 = [data[k].get("F_30", 0) for k in all_keys]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, f05, width, label="F@±0.5 s", color=COLORS[0])
    ax.bar(x + width / 2, f30, width, label="F@±3.0 s", color=COLORS[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("F-measure")
    ax.set_title("Boundary Detection: Level-Conditional Results\n(CC=train-coarse/eval-coarse, FF=fine/fine, CF=cross, FC=cross)")
    ax.legend()
    ax.set_ylim(0, 1.0)
    save_fig(fig, "fig3_boundary_comparison")


# ---------------------------------------------------------------------------
# Figure 4: Class-balance ablation
# ---------------------------------------------------------------------------

def plot_balance_ablation(results_dir: Path) -> None:
    p = results_dir / "balance" / "comparison_table.json"
    if not p.exists():
        print(f"[skip] {p} not found")
        return
    data = json.loads(p.read_text())

    rows = []
    for k, v in data.items():
        rows.append({"experiment": k, "balance": v.get("balance", k), "level": v.get("level", ""), "F_05": v.get("F_05", 0)})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="balance", y="F_05", hue="level", ax=ax, palette="tab10")
    ax.set_ylabel("Boundary F@±0.5 s")
    ax.set_title("Class-Balance Ablation: Effect on Boundary F-Measure")
    ax.set_ylim(0, 1.0)
    save_fig(fig, "fig4_balance_ablation")


# ---------------------------------------------------------------------------
# Figure 5: Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(results_dir: Path) -> None:
    log_dir = results_dir / "logs"
    if not log_dir.exists():
        print("[skip] No logs dir found")
        return

    log_files = sorted(log_dir.glob("*.json"))
    if not log_files:
        print("[skip] No log files found")
        return

    fig, axes = plt.subplots(1, len(log_files), figsize=(5 * len(log_files), 4), squeeze=False)
    for i, lf in enumerate(log_files):
        history = json.loads(lf.read_text())
        epochs_data = history.get("epochs", [])
        if not epochs_data:
            continue
        ep = [e["epoch"] for e in epochs_data]
        train_loss = [e["train_loss"] for e in epochs_data]
        val_loss = [e["val_loss"] for e in epochs_data]
        best_ep = history.get("best_epoch", -1)

        ax = axes[0][i]
        ax.plot(ep, train_loss, label="train", color=COLORS[0])
        ax.plot(ep, val_loss, label="val", color=COLORS[1])
        if best_ep > 0:
            ax.axvline(best_ep, color="gray", ls="--", lw=0.8, label=f"best@{best_ep}")
        ax.set_title(history.get("run_name", lf.stem))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.legend(fontsize=9)

    fig.tight_layout()
    save_fig(fig, "fig5_training_curves")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("results_dir", nargs="?", default="results")
    args = parser.parse_args()

    rd = Path(args.results_dir)
    print(f"Reading results from {rd.resolve()}")

    plot_agreement_scatter(rd)
    plot_label_distribution(rd)
    plot_boundary_comparison(rd)
    plot_balance_ablation(rd)
    plot_training_curves(rd)

    print(f"\nAll figures saved to {FIGURES_DIR.resolve()}/")


if __name__ == "__main__":
    main()

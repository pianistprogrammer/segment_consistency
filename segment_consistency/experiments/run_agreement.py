"""Experiment 1: Inter-level agreement analysis.

Computes boundary F-measure, Cohen's kappa, and segment IoU between the
uppercase (coarse) and lowercase (fine) annotation levels for every track
in the SALAMI dataset that has both levels for at least one annotator.

Outputs:
  results/agreement/results.json     -- per-track metrics
  results/agreement/summary.json     -- corpus-level statistics
  results/agreement/by_genre.json    -- breakdown by genre
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agreement import AgreementResult, compute_agreement
from src.data import SALAMIDataset
from src.utils import save_json


def run(salami_root: str, output_dir: str = "results/agreement") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading SALAMI from {salami_root} ...")
    ds = SALAMIDataset(salami_root)
    print(f"  Total tracks: {len(ds)}")
    print(f"  Dual-annotated: {len(ds.dual_annotated_ids())}")

    records: list[dict] = []
    skipped = 0

    for track in ds:
        for ann_n in (1, 2):
            coarse = track.get(ann_n, "uppercase")
            fine = track.get(ann_n, "lowercase")
            if coarse is None or fine is None:
                continue
            result = compute_agreement(coarse, fine)
            records.append({
                "song_id": track.song_id,
                "annotator": ann_n,
                "genre": track.genre,
                "bf_05": result.bf_05,
                "bf_30": result.bf_30,
                "kappa": result.kappa if not np.isnan(result.kappa) else None,
                "iou_mean": result.iou_mean if not np.isnan(result.iou_mean) else None,
                "quadrant": result.quadrant,
                "n_coarse_segments": len(coarse.segments),
                "n_fine_segments": len(fine.segments),
            })

    print(f"  Computed agreement for {len(records)} (track, annotator) pairs")
    save_json(records, out / "results.json")

    # Corpus-level summary
    df = pd.DataFrame(records)
    summary = {
        "n_pairs": len(df),
        "bf_05_mean": float(df["bf_05"].mean()),
        "bf_05_median": float(df["bf_05"].median()),
        "bf_05_std": float(df["bf_05"].std()),
        "bf_30_mean": float(df["bf_30"].mean()),
        "kappa_mean": float(df["kappa"].mean(skipna=True)),
        "kappa_median": float(df["kappa"].median(skipna=True)),
        "iou_mean": float(df["iou_mean"].mean(skipna=True)),
        "quadrant_counts": df["quadrant"].value_counts().to_dict(),
    }
    save_json(summary, out / "summary.json")

    # Per-genre breakdown
    by_genre = {}
    for genre, gdf in df.groupby("genre"):
        by_genre[str(genre)] = {
            "n": len(gdf),
            "bf_05_mean": float(gdf["bf_05"].mean()),
            "kappa_mean": float(gdf["kappa"].mean(skipna=True)),
            "iou_mean": float(gdf["iou_mean"].mean(skipna=True)),
            "quadrant_counts": gdf["quadrant"].value_counts().to_dict(),
        }
    save_json(by_genre, out / "by_genre.json")

    print("\n=== Corpus Summary ===")
    print(f"  Boundary F@0.5s  mean={summary['bf_05_mean']:.3f}  median={summary['bf_05_median']:.3f}")
    print(f"  Boundary F@3.0s  mean={summary['bf_30_mean']:.3f}")
    print(f"  Cohen's kappa    mean={summary['kappa_mean']:.3f}  median={summary['kappa_median']:.3f}")
    print(f"  Segment IoU      mean={summary['iou_mean']:.3f}")
    print(f"  Quadrants:       {summary['quadrant_counts']}")
    print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute inter-level agreement on SALAMI")
    parser.add_argument("salami_root", help="Path to salami-data-public root directory")
    parser.add_argument("--output_dir", default="results/agreement")
    args = parser.parse_args()
    run(args.salami_root, args.output_dir)

"""End-to-end pipeline test with real SALAMI data.

This test is SKIPPED until the user sets the SALAMI_ROOT environment variable
or a 'salami_root' fixture is configured in conftest.py.

Run once you have the dataset:
  SALAMI_ROOT=/path/to/salami-data-public uv run pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

SALAMI_ROOT = os.environ.get("SALAMI_ROOT", "")

skip_no_data = pytest.mark.skipif(
    not SALAMI_ROOT or not Path(SALAMI_ROOT).exists(),
    reason="SALAMI_ROOT not set or path does not exist. Set env var to run.",
)


@skip_no_data
def test_dataset_loads_real_data():
    from src.data import SALAMIDataset
    ds = SALAMIDataset(SALAMI_ROOT)
    assert len(ds) > 100, f"Expected 100+ tracks, got {len(ds)}"
    print(f"\nLoaded {len(ds)} tracks")


@skip_no_data
def test_dual_annotated_subset():
    from src.data import SALAMIDataset
    ds = SALAMIDataset(SALAMI_ROOT, require_dual=True)
    ids = ds.dual_annotated_ids()
    assert len(ids) > 50, f"Expected 50+ dual-annotated tracks, got {len(ids)}"
    print(f"\nDual-annotated tracks: {len(ids)}")


@skip_no_data
def test_annotation_segments_non_empty():
    from src.data import SALAMIDataset
    ds = SALAMIDataset(SALAMI_ROOT)
    sample_ids = list(ds.song_ids())[:20]
    for sid in sample_ids:
        track = ds[sid]
        for (ann_n, lvl), ann in track.annotations.items():
            assert len(ann.segments) > 0, f"Track {sid} ann{ann_n} {lvl} has no segments"
            assert ann.boundaries.ndim == 1


@skip_no_data
def test_agreement_metrics_on_real_data():
    from src.data import SALAMIDataset
    from src.agreement import compute_corpus_agreement
    ds = SALAMIDataset(SALAMI_ROOT, require_dual=True)
    # Use first 50 tracks for speed
    tracks = list(ds)[:50]
    results = compute_corpus_agreement(tracks, annotator=1)
    assert len(results) > 0
    bf_values = [r.bf_05 for r in results if not np.isnan(r.bf_05)]
    assert len(bf_values) > 0
    print(f"\nMean boundary F@0.5s over {len(bf_values)} tracks: {np.mean(bf_values):.3f}")


@skip_no_data
def test_summary_dataframe():
    from src.data import SALAMIDataset
    ds = SALAMIDataset(SALAMI_ROOT)
    summary = ds.summary()
    assert "is_dual" in summary.columns
    dual_count = summary["is_dual"].sum()
    print(f"\nDual-annotated: {dual_count} / {len(summary)}")

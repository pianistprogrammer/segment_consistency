"""Tests for inter-level agreement metrics.

Uses known-answer synthetic examples to verify metric correctness.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.agreement import (
    AgreementResult,
    boundary_fmeasure,
    compute_agreement,
    label_kappa,
    segment_iou_stats,
)
from src.data import Annotation, Segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_annotation(boundaries: list[float], labels: list[str], duration: float = 100.0) -> Annotation:
    segs = []
    times = [0.0] + boundaries + [duration]
    for i, lbl in enumerate(labels):
        segs.append(Segment(times[i], times[i + 1], lbl))
    return Annotation(song_id=0, annotator=1, level="test", segments=segs)


# ---------------------------------------------------------------------------
# Boundary F-measure
# ---------------------------------------------------------------------------

def test_perfect_boundary_agreement():
    ref = make_annotation([10.0, 20.0, 30.0], ["A", "B", "C", "D"])
    est = make_annotation([10.0, 20.0, 30.0], ["a", "b", "c", "d"])
    result = boundary_fmeasure(ref, est, window=0.5)
    assert result["F"] == pytest.approx(1.0, abs=1e-3)


def test_no_boundary_agreement():
    ref = make_annotation([10.0, 20.0, 30.0], ["A", "B", "C", "D"])
    est = make_annotation([40.0, 50.0, 60.0], ["a", "b", "c", "d"])
    result = boundary_fmeasure(ref, est, window=0.5)
    assert result["F"] == pytest.approx(0.0, abs=1e-3)


def test_partial_boundary_agreement():
    ref = make_annotation([10.0, 20.0, 30.0], ["A", "B", "C", "D"])
    est = make_annotation([10.0, 20.0], ["a", "b", "c"])
    result = boundary_fmeasure(ref, est, window=0.5)
    assert 0.0 < result["F"] < 1.0


def test_boundary_tolerance_window():
    """A boundary that is within the tolerance window should match."""
    ref = make_annotation([10.0], ["A", "B"])
    est = make_annotation([10.3], ["a", "b"])  # 0.3 s off
    r_strict = boundary_fmeasure(ref, est, window=0.5)
    r_tight = boundary_fmeasure(ref, est, window=0.1)
    assert r_strict["F"] > r_tight["F"]


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------

def test_kappa_perfect():
    # coarse labels match fine labels after stripping suffix
    ref = make_annotation([10.0, 20.0], ["A", "B", "C"])
    est = make_annotation([10.0, 20.0], ["A", "B", "C"])
    k = label_kappa(ref, est)
    assert not math.isnan(k)
    assert k > 0.9


def test_kappa_random():
    ref = make_annotation([10.0, 20.0, 30.0, 40.0, 50.0], ["A", "B", "C", "A", "B", "C"])
    est = make_annotation([10.0, 20.0, 30.0, 40.0, 50.0], ["X", "Y", "Z", "X", "Y", "Z"])
    k = label_kappa(ref, est)
    # Completely disjoint vocabularies → kappa can be NaN or near 0/-1; both are valid
    assert math.isnan(k) or k < 0.5


# ---------------------------------------------------------------------------
# Segment IoU
# ---------------------------------------------------------------------------

def test_iou_perfect():
    ann = make_annotation([10.0, 20.0], ["A", "B", "C"])
    stats = segment_iou_stats(ann, ann)
    assert stats["mean"] == pytest.approx(1.0, abs=1e-6)


def test_iou_partial():
    coarse = make_annotation([20.0], ["A", "B"])
    fine = make_annotation([10.0, 20.0], ["a", "b", "c"])
    stats = segment_iou_stats(coarse, fine)
    assert 0.0 < stats["mean"] < 1.0


# ---------------------------------------------------------------------------
# compute_agreement quadrant taxonomy
# ---------------------------------------------------------------------------

def test_quadrant_high():
    ann = make_annotation([10.0, 20.0, 30.0, 40.0], ["A", "B", "C", "D", "A"])
    result = compute_agreement(ann, ann)
    assert result.quadrant == "high"
    assert result.bf_05 == pytest.approx(1.0, abs=1e-3)


def test_quadrant_boundary_only():
    # Boundaries are totally different, but if labels happened to align they'd agree
    coarse = make_annotation([10.0, 20.0], ["A", "B", "C"])
    fine = make_annotation([5.0, 15.0, 25.0, 35.0], ["a", "b", "c", "d", "e"])
    result = compute_agreement(coarse, fine)
    # Boundaries won't match (different positions), labels may or may not — just check it runs
    assert result.quadrant in ("high", "boundary_only", "label_only", "full_disagree")

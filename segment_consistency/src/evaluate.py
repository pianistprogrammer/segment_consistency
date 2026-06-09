"""MSA evaluation metrics for boundary detection.

Standard MIREX metrics:
  - Boundary detection: Precision, Recall, F-measure at ±0.5 s and ±3.0 s
  - Pairwise F-measure (Pw-F): frame-level classification accuracy
  - Normalized Conditional Entropy (NCE): information-theoretic over/under-segmentation

All functions accept times in seconds and use mir_eval as the reference implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mir_eval

from src.data import Annotation


@dataclass
class BoundaryMetrics:
    P_05: float
    R_05: float
    F_05: float
    P_30: float
    R_30: float
    F_30: float


@dataclass
class SegmentMetrics:
    pw_precision: float
    pw_recall: float
    pw_f: float
    nce_over: float   # normalised conditional entropy (over-segmentation)
    nce_under: float  # normalised conditional entropy (under-segmentation)
    nce_f: float      # harmonic mean of over/under NCE


@dataclass
class FullMetrics:
    song_id: int
    train_level: str
    eval_level: str
    boundary: BoundaryMetrics
    segment: SegmentMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ann_to_mir_eval(ann: Annotation):
    """Convert Annotation to mir_eval intervals + labels."""
    if not ann.segments:
        return np.zeros((0, 2)), []
    starts = np.array([s.start for s in ann.segments])
    ends = np.array([s.end for s in ann.segments])
    intervals = np.column_stack([starts, ends])
    labels = [s.label for s in ann.segments]
    return intervals, labels


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def evaluate_boundaries(
    ref: Annotation,
    est_boundaries: np.ndarray,  # seconds
    est_duration: float,
    window_strict: float = 0.5,
    window_lenient: float = 3.0,
) -> BoundaryMetrics:
    """Evaluate predicted boundary times against a reference annotation."""
    ref_b = ref.boundaries
    ref_dur = ref.duration

    def _eval(w: float):
        if len(ref_b) == 0 and len(est_boundaries) == 0:
            return 1.0, 1.0, 1.0
        if len(ref_b) == 0 or len(est_boundaries) == 0:
            return 0.0, 0.0, 0.0
        ref_iv = _boundaries_to_intervals(ref_b, ref_dur)
        est_iv = _boundaries_to_intervals(est_boundaries, est_duration)
        return mir_eval.segment.detection(ref_iv, est_iv, window=w, beta=1.0, trim=True)

    P5, R5, F5 = _eval(window_strict)
    P3, R3, F3 = _eval(window_lenient)
    return BoundaryMetrics(
        P_05=float(P5), R_05=float(R5), F_05=float(F5),
        P_30=float(P3), R_30=float(R3), F_30=float(F3),
    )


def evaluate_segments(ref: Annotation, est: Annotation) -> SegmentMetrics:
    """Compute pairwise F-measure and NCE between two segment annotations."""
    ref_iv, ref_lb = _ann_to_mir_eval(ref)
    est_iv, est_lb = _ann_to_mir_eval(est)

    if len(ref_iv) == 0 or len(est_iv) == 0:
        return SegmentMetrics(0.0, 0.0, 0.0, float("nan"), float("nan"), float("nan"))

    # Pairwise F-measure
    pw_p, pw_r, pw_f = mir_eval.segment.pairwise(ref_iv, ref_lb, est_iv, est_lb)

    # NCE
    nce_over, nce_under, nce_f = mir_eval.segment.nce(ref_iv, ref_lb, est_iv, est_lb)

    return SegmentMetrics(
        pw_precision=float(pw_p),
        pw_recall=float(pw_r),
        pw_f=float(pw_f),
        nce_over=float(nce_over),
        nce_under=float(nce_under),
        nce_f=float(nce_f),
    )


# ---------------------------------------------------------------------------
# Corpus-level aggregation
# ---------------------------------------------------------------------------

def aggregate_boundary_metrics(results: list[BoundaryMetrics]) -> dict[str, float]:
    if not results:
        return {}
    keys = ["P_05", "R_05", "F_05", "P_30", "R_30", "F_30"]
    import numpy as np
    return {k: float(np.nanmean([getattr(r, k) for r in results])) for k in keys}


def aggregate_segment_metrics(results: list[SegmentMetrics]) -> dict[str, float]:
    if not results:
        return {}
    keys = ["pw_precision", "pw_recall", "pw_f", "nce_over", "nce_under", "nce_f"]
    import numpy as np
    return {k: float(np.nanmean([getattr(r, k) for r in results])) for k in keys}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _boundaries_to_intervals(boundaries: np.ndarray, duration: float) -> np.ndarray:
    starts = np.concatenate([[0.0], boundaries])
    ends = np.concatenate([boundaries, [duration]])
    return np.column_stack([starts, ends])

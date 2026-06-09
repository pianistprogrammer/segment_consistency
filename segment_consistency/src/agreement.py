"""Inter-level annotation agreement metrics.

Computes:
  - Boundary F-measure (mir_eval standard) at two tolerance windows
  - Cohen's kappa over aligned segment labels
  - Segment IoU (Intersection over Union) statistics
  - 4-quadrant disagreement taxonomy per track
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score

import mir_eval

from src.data import Annotation, Segment


# ---------------------------------------------------------------------------
# Boundary F-measure
# ---------------------------------------------------------------------------

def boundary_fmeasure(
    ref: Annotation,
    est: Annotation,
    window: float = 0.5,
) -> dict[str, float]:
    """Compute boundary precision, recall, and F-measure.

    Uses mir_eval with a symmetric tolerance window (half-window = window/2
    to match the ±window convention common in MSA papers, i.e. ±0.5 s
    means window=0.5 passed here).
    """
    ref_b = ref.boundaries
    est_b = est.boundaries

    if len(ref_b) == 0 and len(est_b) == 0:
        return {"P": 1.0, "R": 1.0, "F": 1.0}
    if len(ref_b) == 0 or len(est_b) == 0:
        return {"P": 0.0, "R": 0.0, "F": 0.0}

    P, R, F = mir_eval.segment.detection(
        reference_intervals=_boundaries_to_intervals(ref_b, ref.duration),
        estimated_intervals=_boundaries_to_intervals(est_b, est.duration),
        window=window,
        beta=1.0,
        trim=True,
    )
    return {"P": float(P), "R": float(R), "F": float(F)}


def _boundaries_to_intervals(boundaries: np.ndarray, duration: float) -> np.ndarray:
    """Convert a sorted array of boundary times to (start, end) intervals.

    Deduplicates boundaries and drops any that would produce zero-duration intervals.
    """
    boundaries = np.unique(boundaries)
    # Remove boundaries at or beyond duration
    boundaries = boundaries[(boundaries > 0.0) & (boundaries < duration)]
    starts = np.concatenate([[0.0], boundaries])
    ends = np.concatenate([boundaries, [duration]])
    # Keep only strictly positive intervals
    mask = ends > starts
    return np.column_stack([starts[mask], ends[mask]])


# ---------------------------------------------------------------------------
# Cohen's kappa on aligned labels
# ---------------------------------------------------------------------------

def _align_segments(coarse: Annotation, fine: Annotation) -> tuple[list[str], list[str]]:
    """Produce parallel label lists by intersecting segment boundaries.

    For each overlapping (coarse, fine) segment pair we emit one label from
    each annotation.  Both sides are normalised to uppercase so that 'A' and
    'a' are treated as the same structural position (they represent the same
    section letter at different granularities).  Variation suffixes like
    'Verse_A' are stripped to their base form before uppercasing.

    Only intervals > 0.1 s are kept to ignore numerical noise.
    """
    from src.data import strip_variation_suffix

    coarse_labels: list[str] = []
    fine_labels: list[str] = []

    for c_seg in coarse.segments:
        for f_seg in fine.segments:
            overlap = min(c_seg.end, f_seg.end) - max(c_seg.start, f_seg.start)
            if overlap > 0.1:
                c_lbl = c_seg.label.upper()
                f_stripped = strip_variation_suffix(f_seg.label)
                f_lbl = (f_stripped if f_stripped else f_seg.label).upper()
                coarse_labels.append(c_lbl)
                fine_labels.append(f_lbl)

    return coarse_labels, fine_labels


def label_kappa(coarse: Annotation, fine: Annotation) -> float:
    """Cohen's kappa treating coarse vs. fine labels as two raters on same segments."""
    c_labels, f_labels = _align_segments(coarse, fine)
    if len(c_labels) < 2:
        return float("nan")
    if len(set(c_labels) | set(f_labels)) < 2:
        return float("nan")
    # Perfect agreement edge case: kappa is 1.0 by definition
    if c_labels == f_labels:
        return 1.0
    try:
        k = cohen_kappa_score(c_labels, f_labels)
        return float(k) if not (k != k) else float("nan")  # guard NaN
    except Exception:
        return float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Segment IoU
# ---------------------------------------------------------------------------

def segment_iou_stats(coarse: Annotation, fine: Annotation) -> dict[str, float]:
    """For each coarse segment, compute max IoU against all fine segments."""
    ious: list[float] = []
    for c_seg in coarse.segments:
        best = 0.0
        for f_seg in fine.segments:
            inter = min(c_seg.end, f_seg.end) - max(c_seg.start, f_seg.start)
            if inter <= 0:
                continue
            union = max(c_seg.end, f_seg.end) - min(c_seg.start, f_seg.start)
            best = max(best, inter / union if union > 0 else 0.0)
        ious.append(best)

    if not ious:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan")}
    return {
        "mean": float(np.mean(ious)),
        "median": float(np.median(ious)),
        "min": float(np.min(ious)),
    }


# ---------------------------------------------------------------------------
# Full per-track agreement report
# ---------------------------------------------------------------------------

@dataclass
class AgreementResult:
    song_id: int
    bf_05: float     # boundary F at ±0.5 s
    bf_30: float     # boundary F at ±3.0 s
    kappa: float
    iou_mean: float
    quadrant: str    # "high", "boundary_only", "label_only", "full_disagree"


def compute_agreement(
    coarse: Annotation,
    fine: Annotation,
    window_strict: float = 0.5,
    window_lenient: float = 3.0,
) -> AgreementResult:
    try:
        bf_strict = boundary_fmeasure(coarse, fine, window=window_strict)["F"]
        bf_lenient = boundary_fmeasure(coarse, fine, window=window_lenient)["F"]
    except Exception:
        bf_strict = float("nan")
        bf_lenient = float("nan")

    kappa = label_kappa(coarse, fine)
    iou = segment_iou_stats(coarse, fine)["mean"]

    # 4-quadrant taxonomy
    # Use lenient F for the taxonomy to avoid punishing boundary placement differences
    # that are mere granularity effects
    high_boundary = bf_strict > 0.5
    high_label = (not np.isnan(kappa)) and kappa > 0.5

    if high_boundary and high_label:
        quadrant = "high"
    elif not high_boundary and high_label:
        quadrant = "boundary_only"
    elif high_boundary and not high_label:
        quadrant = "label_only"
    else:
        quadrant = "full_disagree"

    return AgreementResult(
        song_id=coarse.song_id,
        bf_05=bf_strict,
        bf_30=bf_lenient,
        kappa=kappa,
        iou_mean=iou,
        quadrant=quadrant,
    )


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def compute_corpus_agreement(
    tracks,  # Iterable[Track]
    annotator: int = 1,
) -> list[AgreementResult]:
    """Compute coarse/fine agreement for all tracks with the given annotator's data."""
    results: list[AgreementResult] = []
    for track in tracks:
        coarse = track.get(annotator, "uppercase")
        fine = track.get(annotator, "lowercase")
        if coarse is None or fine is None:
            continue
        results.append(compute_agreement(coarse, fine))
    return results

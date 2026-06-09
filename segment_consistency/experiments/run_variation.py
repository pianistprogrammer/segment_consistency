"""Experiment 3: Variation-marker semantic study.

Tests whether segments labelled with different variation suffixes (A vs B)
are acoustically more distant than same-suffix segments.

Hypothesis (H4): segments labeled 'Verse A' and 'Verse B' in the fine
annotations differ more in audio feature space than two 'Verse A' segments.

Method:
  - Extract mean MFCC + chroma feature vector per segment
  - Compute cosine distance between same-suffix and cross-suffix segment pairs
    within the same base label (e.g. all 'Verse' segments)
  - Wilcoxon signed-rank test: H0 = no difference in within vs. cross-suffix distances

Outputs:
  results/variation/distances.json    -- all pairwise distances
  results/variation/stats.json        -- Wilcoxon test results
"""
from __future__ import annotations

import argparse
import re
import warnings
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SALAMIDataset
from src.utils import save_json


def _extract_segment_features(song_id: int, audio_path: Path, start: float, end: float, sr: int = 22050) -> np.ndarray | None:
    """Extract mean MFCC + chroma feature vector for a time segment."""
    mel_cache = Path("data/mels") / f"{song_id}.npy"
    try:
        import librosa
        if mel_cache.exists():
            mel_db = np.load(str(mel_cache))
            hop = 512
            s_frame = int(round(start * sr / hop))
            e_frame = int(round(end   * sr / hop))
            e_frame = min(e_frame, mel_db.shape[1])
            if e_frame <= s_frame:
                return None
            seg_mel = mel_db[:, s_frame:e_frame]
            mfcc = librosa.feature.mfcc(S=seg_mel, n_mfcc=20).mean(axis=1)
            chroma = librosa.feature.chroma_stft(S=librosa.db_to_power(seg_mel)).mean(axis=1)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, _ = librosa.load(str(audio_path), sr=sr, mono=True, offset=start, duration=end - start)
            if len(y) < 512:
                return None
            mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
        return np.concatenate([mfcc, chroma])
    except Exception:
        return None


def _has_variation_suffix(label: str) -> bool:
    """True if the label is a prime-suffixed variant (e.g. a', b'', a''').

    SALAMI uses the convention that 'a' is a base segment and "a'" is a
    variation of it.  Base labels are plain lowercase letters; variants
    carry one or more trailing prime (') characters.
    """
    return bool(re.search(r"'+$", label))


def _base_label(label: str) -> str:
    """Strip trailing primes: "a''" -> 'a'"""
    return label.rstrip("'")


def _suffix(label: str) -> str:
    """Return the prime suffix as a canonical tag.

    'a'   -> 'base'
    "a'"  -> 'prime1'
    "a''" -> 'prime2'
    """
    primes = len(label) - len(label.rstrip("'"))
    if primes == 0:
        return "base"
    return f"prime{primes}"


def run(salami_root: str, audio_root: str, output_dir: str = "results/variation") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading SALAMI from {salami_root} ...")
    ds = SALAMIDataset(salami_root)
    audio_root_path = Path(audio_root)

    # Group segments by (base_label, suffix) per track
    within_distances: list[float] = []   # same suffix, same base
    cross_distances: list[float] = []    # different suffix, same base

    n_tracks = 0
    for track in ds:
        fine = track.get(1, "lowercase")
        if fine is None:
            continue

        # Require at least one primed segment to include this track
        if not any(_has_variation_suffix(s.label) for s in fine.segments):
            continue

        audio_path = audio_root_path / f"{track.song_id}.mp3"
        if not audio_path.exists():
            audio_path = audio_root_path / f"{track.song_id}.wav"
        if not audio_path.exists():
            continue

        n_tracks += 1
        # Collect ALL segments grouped by base letter (a, b, c, …)
        # Each entry: (suffix_tag, feature_vector)
        # suffix_tag: 'base' for plain 'a', 'prime1' for "a'", etc.
        by_base: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
        for seg in fine.segments:
            feat = _extract_segment_features(track.song_id, audio_path, seg.start, seg.end)
            if feat is not None:
                base = _base_label(seg.label)
                suffix = _suffix(seg.label)
                by_base[base].append((suffix, feat))

        # Pairwise distances within each base group:
        #   within = both same suffix (base–base or prime1–prime1)
        #   cross  = different suffix (base–prime1, etc.) — the "variation" pairs
        for base, items in by_base.items():
            if len(items) < 2:
                continue
            # Need at least one primed segment to be interesting
            if not any(s != "base" for s, _ in items):
                continue
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    s_i, f_i = items[i]
                    s_j, f_j = items[j]
                    d = float(cosine(f_i, f_j))
                    if s_i == s_j:
                        within_distances.append(d)
                    else:
                        cross_distances.append(d)

    print(f"  Processed {n_tracks} tracks with variation markers")
    print(f"  Within-suffix pairs: {len(within_distances)}")
    print(f"  Cross-suffix pairs:  {len(cross_distances)}")

    results: dict = {
        "n_tracks": n_tracks,
        "within_suffix": {
            "n": len(within_distances),
            "mean": float(np.mean(within_distances)) if within_distances else None,
            "std": float(np.std(within_distances)) if within_distances else None,
        },
        "cross_suffix": {
            "n": len(cross_distances),
            "mean": float(np.mean(cross_distances)) if cross_distances else None,
            "std": float(np.std(cross_distances)) if cross_distances else None,
        },
    }

    if within_distances and cross_distances:
        stat, p_value = stats.mannwhitneyu(
            cross_distances, within_distances, alternative="greater"
        )
        results["mannwhitney_u"] = float(stat)
        results["p_value"] = float(p_value)
        # Effect size: rank-biserial correlation
        n1, n2 = len(cross_distances), len(within_distances)
        r_rb = 1 - (2 * stat) / (n1 * n2)
        results["effect_size_r"] = float(r_rb)

        print(f"\n  Mean within-suffix distance: {results['within_suffix']['mean']:.4f}")
        print(f"  Mean cross-suffix distance:  {results['cross_suffix']['mean']:.4f}")
        print(f"  Mann-Whitney U={stat:.1f}  p={p_value:.4g}  r={r_rb:.3f}")

    save_json(results, out / "stats.json")
    save_json(
        {"within": within_distances, "cross": cross_distances},
        out / "distances.json",
    )
    print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variation-marker semantic study")
    parser.add_argument("salami_root", help="Path to salami-data-public root")
    parser.add_argument("audio_root", help="Path to directory with audio files")
    parser.add_argument("--output_dir", default="results/variation")
    args = parser.parse_args()
    run(args.salami_root, args.audio_root, args.output_dir)

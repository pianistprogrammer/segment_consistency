"""Experiment 2: Level-conditional boundary detection (2×2 factorial).

Train/evaluate the CNN boundary detector under four conditions:
  CC: train on coarse, evaluate on coarse  (standard MIREX setting)
  FF: train on fine,   evaluate on fine
  CF: train on coarse, evaluate on fine
  FC: train on fine,   evaluate on coarse

Also runs the Foote novelty baseline (no training) under both eval levels.

Outputs:
  results/boundary/comparison_table.json     -- aggregated comparison table
  results/boundary/checkpoint.json           -- resume state
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SALAMIDataset
from src.evaluate import (
    BoundaryMetrics,
    SegmentMetrics,
    aggregate_boundary_metrics,
    aggregate_segment_metrics,
    evaluate_boundaries,
    evaluate_segments,
)
from src.model import CNNBoundaryDetector, FooteNovelty
from src.train import ExperimentRunner, train
from src.utils import get_device, save_json


CONDITIONS = [
    ("coarse", "coarse"),  # CC
    ("fine",   "fine"),    # FF
    ("coarse", "fine"),    # CF
    ("fine",   "coarse"),  # FC
]
LEVEL_MAP = {"coarse": "uppercase", "fine": "lowercase"}
MEL_CACHE = Path("data/mels")
SR = 22050
HOP = 512


def _get_annotation(track, level_key: str, annotator: int = 1):
    return track.get(annotator, LEVEL_MAP[level_key])


def _load_mel(song_id: int, audio_root: Path) -> np.ndarray | None:
    """Load precomputed mel or decode from audio. Returns (n_mels, T) float32."""
    npy = MEL_CACHE / f"{song_id}.npy"
    if npy.exists():
        return np.load(str(npy))
    # Fall back to live decode
    ap = audio_root / f"{song_id}.mp3"
    if not ap.exists():
        ap = audio_root / f"{song_id}.wav"
    if not ap.exists():
        return None
    try:
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(str(ap), sr=SR, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=80, hop_length=HOP)
        return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    except Exception:
        return None


def _collect_pairs(ds, ids, level_key, audio_root):
    """Return (pairs, anns) for ids that have both a mel and an annotation."""
    pairs, anns = [], {}
    for sid in ids:
        mel = MEL_CACHE / f"{sid}.npy"
        ap = audio_root / f"{sid}.mp3"
        if not mel.exists() and not ap.exists():
            continue
        ann = _get_annotation(ds[sid], level_key)
        if ann is None:
            continue
        pairs.append((sid, ap))
        anns[sid] = ann
    return pairs, anns


def _infer_boundaries(model, mel_db: np.ndarray, device, hop_size: float) -> np.ndarray:
    """Run sliding-window CNN inference on a Mel-spectrogram, return boundary times."""
    from scipy.signal import find_peaks
    T = mel_db.shape[1]
    pad = 7
    mel_pad = np.pad(mel_db, ((0, 0), (pad, pad)), mode="reflect")
    patches = np.stack([mel_pad[:, t: t + 15] for t in range(T)])
    x = torch.from_numpy(patches).unsqueeze(1).to(device)
    with torch.no_grad():
        probs = model(x).cpu().numpy()
    peaks, _ = find_peaks(probs, height=0.5, distance=10)
    return peaks * hop_size


def run(salami_root: str, audio_root: str, output_dir: str = "results/boundary") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = ExperimentRunner(out)
    audio_root_path = Path(audio_root)

    print(f"Loading SALAMI from {salami_root} ...")
    ds = SALAMIDataset(salami_root, require_dual=False)
    all_ids = ds.song_ids()
    train_ids, temp_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    print(f"  Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")
    device = get_device()
    hop_size = HOP / SR

    # ------------------------------------------------------------------ #
    # Foote novelty baseline (uses mel cache, no training needed)         #
    # ------------------------------------------------------------------ #
    for eval_level in ("coarse", "fine"):
        exp_id = f"foote_eval_{eval_level}"
        if runner.is_done(exp_id):
            print(f"[skip] {exp_id}")
            continue

        print(f"\n[foote] eval_level={eval_level}")
        foote = FooteNovelty()
        boundary_results: list[BoundaryMetrics] = []

        for sid in test_ids:
            ann = _get_annotation(ds[sid], eval_level)
            if ann is None:
                continue
            mel_db = _load_mel(sid, audio_root_path)
            if mel_db is None:
                continue
            try:
                import librosa
                # Use MFCC for Foote (more discriminative than raw mel)
                mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20).T  # (T, 20)
                est_b = foote.detect(mfcc, hop_size=hop_size)
                bm = evaluate_boundaries(ann, est_b, ann.duration)
                boundary_results.append(bm)
            except Exception as e:
                print(f"  Warning: track {sid}: {e}")

        agg = aggregate_boundary_metrics(boundary_results)
        runner.save(exp_id, {"condition": exp_id, "n_tracks": len(boundary_results), **agg})
        print(f"  F@0.5s={agg.get('F_05', 0):.3f}  F@3.0s={agg.get('F_30', 0):.3f}  ({len(boundary_results)} tracks)")

    # ------------------------------------------------------------------ #
    # CNN boundary detector: 4 conditions                                 #
    # ------------------------------------------------------------------ #
    from src.dataset import BoundaryPatchDataset

    for train_level, eval_level in CONDITIONS:
        exp_id = f"cnn_{train_level[0].upper()}{eval_level[0].upper()}"
        if runner.is_done(exp_id):
            print(f"[skip] {exp_id}")
            continue

        print(f"\n[CNN] train={train_level}  eval={eval_level}  ({exp_id})")

        train_pairs, train_anns = _collect_pairs(ds, train_ids, train_level, audio_root_path)
        val_pairs, val_anns     = _collect_pairs(ds, val_ids,   train_level, audio_root_path)

        if not train_pairs:
            print("  No data found, skipping")
            continue

        print(f"  Building train dataset ({len(train_pairs)} tracks)...")
        train_ds = BoundaryPatchDataset(train_pairs, train_anns, mel_cache_dir=MEL_CACHE)
        print(f"  Building val dataset ({len(val_pairs)} tracks)...")
        val_ds   = BoundaryPatchDataset(val_pairs,   val_anns,   mel_cache_dir=MEL_CACHE)
        print(f"  Train samples: {len(train_ds):,}  Val samples: {len(val_ds):,}")

        model = CNNBoundaryDetector(n_mels=80, patch_frames=15)
        history = train(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            run_name=exp_id.lower(),
            output_dir=out,
            epochs=50,
            batch_size=512,
            lr=1e-3,
            balance="none",
        )

        # Load best checkpoint and evaluate on test set at eval_level
        test_pairs_eval, test_anns_eval = _collect_pairs(ds, test_ids, eval_level, audio_root_path)
        ckpt = out / "checkpoints" / f"{exp_id.lower()}_best.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval().to(device)

        boundary_results = []
        for sid, _ in test_pairs_eval:
            ann = test_anns_eval.get(sid)
            if ann is None:
                continue
            mel_db = _load_mel(sid, audio_root_path)
            if mel_db is None:
                continue
            try:
                est_b = _infer_boundaries(model, mel_db, device, hop_size)
                bm = evaluate_boundaries(ann, est_b, ann.duration)
                boundary_results.append(bm)
            except Exception as e:
                print(f"  Warning: track {sid}: {e}")

        agg = aggregate_boundary_metrics(boundary_results)
        runner.save(exp_id, {
            "condition": exp_id,
            "train_level": train_level,
            "eval_level": eval_level,
            "n_tracks": len(boundary_results),
            **agg,
        })
        print(f"  F@0.5s={agg.get('F_05', 0):.3f}  F@3.0s={agg.get('F_30', 0):.3f}  ({len(boundary_results)} tracks)")

    # Summary
    all_res = runner.all_results()
    save_json(all_res, out / "comparison_table.json")
    print("\n=== Comparison Table ===")
    for k, v in sorted(all_res.items()):
        f05 = v.get("F_05", float("nan"))
        f30 = v.get("F_30", float("nan"))
        n   = v.get("n_tracks", 0)
        print(f"  {k:20s}  F@0.5={f05:.3f}  F@3.0={f30:.3f}  n={n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("salami_root")
    parser.add_argument("audio_root")
    parser.add_argument("--output_dir", default="results/boundary")
    args = parser.parse_args()
    run(args.salami_root, args.audio_root, args.output_dir)

"""Experiment 4: Class-balance ablation.

Trains the CNN boundary detector under three class-balance strategies:
  none          -- standard unweighted BCE
  weighted_loss -- per-sample weights inversely proportional to class frequency
  resample      -- WeightedRandomSampler to equalise batches

Reports per-class recall, macro F, and micro F for each strategy
at both annotation levels (coarse and fine).

Outputs:
  results/balance/<strategy>_<level>/metrics.json
  results/balance/comparison_table.json
  results/balance/checkpoint.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SALAMIDataset
from src.evaluate import aggregate_boundary_metrics, evaluate_boundaries
from src.model import CNNBoundaryDetector
from src.train import ExperimentRunner, BalanceStrategy, train
from src.utils import get_device, save_json


BALANCE_STRATEGIES: list[BalanceStrategy] = ["none", "weighted_loss", "weighted_loss_capped", "resample"]
LEVEL_MAP = {"coarse": "uppercase", "fine": "lowercase"}


def run(salami_root: str, audio_root: str, output_dir: str = "results/balance") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(out)
    audio_root_path = Path(audio_root)

    print(f"Loading SALAMI from {salami_root} ...")
    ds = SALAMIDataset(salami_root)
    all_ids = ds.song_ids()
    train_ids, temp_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    device = get_device()

    for level_key in ("coarse", "fine"):
        salami_level = LEVEL_MAP[level_key]

        def _collect_pairs(ids):
            pairs, anns = [], {}
            for sid in ids:
                ap = audio_root_path / f"{sid}.mp3"
                if not ap.exists():
                    ap = audio_root_path / f"{sid}.wav"
                if not ap.exists():
                    continue
                ann = ds[sid].get(1, salami_level)
                if ann is None:
                    continue
                pairs.append((sid, ap))
                anns[sid] = ann
            return pairs, anns

        from src.dataset import BoundaryPatchDataset
        print(f"\nBuilding datasets for level={level_key} ...")
        train_pairs, train_anns = _collect_pairs(train_ids)
        val_pairs, val_anns = _collect_pairs(val_ids)
        test_pairs, test_anns = _collect_pairs(test_ids)

        if not train_pairs:
            print("  No audio found, skipping")
            continue

        train_ds = BoundaryPatchDataset(train_pairs, train_anns, mel_cache_dir="data/mels")
        val_ds = BoundaryPatchDataset(val_pairs, val_anns, mel_cache_dir="data/mels")

        for balance in BALANCE_STRATEGIES:
            exp_id = f"{balance}_{level_key}"
            if runner.is_done(exp_id):
                print(f"[skip] {exp_id}")
                continue

            print(f"\n[balance={balance}  level={level_key}]")
            model = CNNBoundaryDetector(n_mels=80, patch_frames=15)
            run_name = exp_id
            history = train(
                model=model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                run_name=run_name,
                output_dir=out,
                epochs=50,
                batch_size=512,
                lr=1e-3,
                balance=balance,
            )

            # Evaluate
            ckpt = out / "checkpoints" / f"{run_name}_best.pt"
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval().to(device)

            boundary_results = []
            for sid, audio_path in test_pairs:
                ann = test_anns.get(sid)
                if ann is None:
                    continue
                try:
                    import librosa
                    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
                    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=512)
                    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
                    T = mel_db.shape[1]
                    pad = 7
                    mel_pad = np.pad(mel_db, ((0, 0), (pad, pad)), mode="reflect")
                    patches = np.stack([mel_pad[:, t: t + 15] for t in range(T)])
                    x = torch.from_numpy(patches).unsqueeze(1).to(device)
                    with torch.no_grad():
                        probs = model(x).cpu().numpy()
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(probs, height=0.5, distance=10)
                    est_b = peaks * (512 / sr)
                    bm = evaluate_boundaries(ann, est_b, ann.duration)
                    boundary_results.append(bm)
                except Exception as e:
                    print(f"  Warning: track {sid}: {e}")

            agg = aggregate_boundary_metrics(boundary_results)
            runner.save(exp_id, {
                "balance": balance,
                "level": level_key,
                "n_tracks": len(boundary_results),
                **agg,
            })
            print(f"  F@0.5={agg.get('F_05', 0):.3f}  F@3.0={agg.get('F_30', 0):.3f}  n={len(boundary_results)}")

    # Summary table
    all_res = runner.all_results()
    save_json(all_res, out / "comparison_table.json")
    print("\n=== Class Balance Comparison Table ===")
    for k, v in sorted(all_res.items()):
        print(f"  {k:25s}  F@0.5={v.get('F_05', float('nan')):.3f}  F@3.0={v.get('F_30', float('nan')):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class-balance ablation for boundary detection")
    parser.add_argument("salami_root")
    parser.add_argument("audio_root")
    parser.add_argument("--output_dir", default="results/balance")
    args = parser.parse_args()
    run(args.salami_root, args.audio_root, args.output_dir)

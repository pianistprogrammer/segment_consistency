"""Pre-extract Mel-spectrograms from all SALAMI audio files and cache as .npy.

Run this ONCE before training experiments. Results go to data/mels/<song_id>.npy
Each file is a float32 array of shape (n_mels, T).

Usage:
  uv run python scripts/precompute_mels.py /path/to/audio_root [--output_dir data/mels]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress librosa/audioread noise
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SR = 22050
N_MELS = 80
HOP_LENGTH = 512


def precompute(audio_root: str, output_dir: str = "data/mels") -> None:
    import librosa

    audio_root_path = Path(audio_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mp3_files = sorted(
        p for p in audio_root_path.glob("*.mp3")
        if not p.name.startswith("._")
    )
    print(f"Found {len(mp3_files)} MP3 files in {audio_root_path}")

    skipped = 0
    errors = 0
    for mp3_path in tqdm(mp3_files, desc="Extracting Mel-spectrograms"):
        song_id = mp3_path.stem
        out_path = out / f"{song_id}.npy"
        if out_path.exists():
            continue  # already done — resume-safe
        try:
            y, _ = librosa.load(str(mp3_path), sr=SR, mono=True)
            mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
            mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
            np.save(out_path, mel_db)
        except Exception as e:
            tqdm.write(f"  Error {song_id}: {e}")
            errors += 1

    print(f"\nDone. Errors: {errors}. Files in {out}: {len(list(out.glob('*.npy')))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_root", help="Directory containing <song_id>.mp3 files")
    parser.add_argument("--output_dir", default="data/mels")
    args = parser.parse_args()
    precompute(args.audio_root, args.output_dir)

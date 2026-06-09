"""Mel-spectrogram patch dataset for boundary detection training.

Each sample is a (patch, label) pair where:
  - patch: (1, n_mels, patch_frames) Mel-spectrogram patch centred on a frame
  - label: 1.0 if the centre frame is within `tolerance` of a boundary, else 0.0

Design: lazy loading — audio is loaded per-track on first access and cached
as a Mel-spectrogram, so __init__ is fast regardless of corpus size.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore

from src.data import Annotation


class BoundaryPatchDataset(Dataset):
    """Mel-spectrogram patches with boundary labels — lazy audio loading.

    During __init__ we only scan annotation boundaries and build a flat index
    mapping sample index -> (track_idx, frame_idx).  Audio is loaded on
    first access per track and cached as a Mel-spectrogram.

    Args:
        audio_paths: Ordered list of (song_id, audio_path) tuples.
        annotations: Dict mapping song_id -> Annotation.
        n_mels: Number of Mel bins.
        hop_length: Hop in samples.
        sr: Sample rate.
        patch_frames: Patch width in frames (centred on target frame).
        boundary_tolerance_frames: Frames within which a frame counts as boundary.
    """

    def __init__(
        self,
        audio_paths: list[tuple[int, str | Path]],
        annotations: dict[int, Annotation],
        n_mels: int = 80,
        hop_length: int = 512,
        sr: int = 22050,
        patch_frames: int = 15,
        boundary_tolerance_frames: int = 6,
        mel_cache_dir: str | Path | None = None,
    ):
        if librosa is None:
            raise ImportError("librosa is required for BoundaryPatchDataset")

        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sr = sr
        self.patch_frames = patch_frames
        self.pad = patch_frames // 2
        self.boundary_tolerance_frames = boundary_tolerance_frames
        self.mel_cache_dir = Path(mel_cache_dir) if mel_cache_dir else None

        # Per-track metadata (no audio loaded yet)
        self._tracks: list[dict] = []  # {song_id, audio_path, boundary_frames, n_frames}
        self._mel_cache: dict[int, np.ndarray] = {}  # track_idx -> mel (n_mels, T+2*pad)

        # Flat index: list of (track_idx, frame_idx)
        self._index: list[tuple[int, int]] = []
        # Flat label array for class_weights() — estimated from boundary_frames
        self._labels: np.ndarray | None = None

        self._build_index(audio_paths, annotations)

    # ------------------------------------------------------------------
    def _build_index(
        self,
        audio_paths: list[tuple[int, str | Path]],
        annotations: dict[int, Annotation],
    ) -> None:
        """Scan annotations to estimate frame counts without loading audio.

        We use the annotation duration to estimate T = duration * sr / hop_length.
        This is fast and accurate enough to build the index.
        """
        labels_list: list[float] = []

        for track_idx, (song_id, audio_path) in enumerate(audio_paths):
            ann = annotations.get(song_id)
            if ann is None or ann.duration is None or ann.duration <= 0:
                continue

            n_frames = int(ann.duration * self.sr / self.hop_length)
            if n_frames < 1:
                continue

            boundary_frames = set(
                int(round(b * self.sr / self.hop_length)) for b in ann.boundaries
            )

            self._tracks.append({
                "song_id": song_id,
                "audio_path": Path(audio_path),
                "boundary_frames": boundary_frames,
                "n_frames": n_frames,
                "track_idx": track_idx,
            })
            real_track_idx = len(self._tracks) - 1

            for t in range(n_frames):
                self._index.append((real_track_idx, t))
                label = float(any(abs(t - bf) <= self.boundary_tolerance_frames for bf in boundary_frames))
                labels_list.append(label)

        self._labels = np.array(labels_list, dtype=np.float32)

    # ------------------------------------------------------------------
    def _get_mel(self, track_idx: int) -> np.ndarray:
        """Load and cache Mel-spectrogram for a track (padded).

        Priority: precomputed .npy file > live decode from audio.
        """
        if track_idx in self._mel_cache:
            return self._mel_cache[track_idx]

        track = self._tracks[track_idx]

        # Try precomputed .npy first (fast)
        mel_db = None
        if self.mel_cache_dir is not None:
            npy_path = self.mel_cache_dir / f"{track['song_id']}.npy"
            if npy_path.exists():
                mel_db = np.load(str(npy_path))

        # Fall back to live decode
        if mel_db is None:
            import warnings
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y, _ = librosa.load(str(track["audio_path"]), sr=self.sr, mono=True)
                mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
                mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
            except Exception:
                T = track["n_frames"]
                mel_db = np.zeros((self.n_mels, T), dtype=np.float32)

        mel_pad = np.pad(mel_db, ((0, 0), (self.pad, self.pad)), mode="reflect")
        self._mel_cache[track_idx] = mel_pad
        return mel_pad

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        track_idx, t = self._index[idx]
        mel_pad = self._get_mel(track_idx)

        # Clamp t in case audio was shorter than estimated duration
        max_t = mel_pad.shape[1] - self.patch_frames
        t_safe = min(t, max(max_t, 0))
        patch = mel_pad[:, t_safe: t_safe + self.patch_frames]

        # Pad patch if shorter than expected (edge of short audio)
        if patch.shape[1] < self.patch_frames:
            patch = np.pad(patch, ((0, 0), (0, self.patch_frames - patch.shape[1])), mode="reflect")

        label = self._labels[idx]
        return torch.from_numpy(patch).unsqueeze(0), torch.tensor(label, dtype=torch.float32)

    # ------------------------------------------------------------------
    def class_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler (inverse class frequency)."""
        labels = self._labels
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return torch.ones(len(labels))
        w_pos = 1.0 / n_pos
        w_neg = 1.0 / n_neg
        weights = np.where(labels == 1.0, w_pos, w_neg)
        return torch.from_numpy(weights.astype(np.float32))

    def clear_cache(self) -> None:
        """Free cached Mel-spectrograms to reclaim memory."""
        self._mel_cache.clear()

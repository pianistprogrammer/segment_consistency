"""Boundary detection models.

1. FooteNovelty  -- unsupervised checkerboard kernel novelty curve
2. CNNBoundaryDetector -- small ConvNet on Mel-spectrogram patches
                         (Ullrich et al. 2014 style, Section 3 of that paper)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_device


# ---------------------------------------------------------------------------
# Foote novelty baseline
# ---------------------------------------------------------------------------

class FooteNovelty:
    """Checkerboard novelty-curve boundary detector (Foote 2000).

    Works directly on a self-similarity matrix computed from any feature matrix.
    No learned parameters.
    """

    def __init__(self, kernel_size: int = 64, threshold_percentile: float = 75.0):
        self.kernel_size = kernel_size
        self.threshold_percentile = threshold_percentile

    def _checkerboard_kernel(self, M: int) -> np.ndarray:
        """Block-diagonal checkerboard kernel of size (2M, 2M)."""
        k = np.zeros((2 * M, 2 * M))
        k[:M, :M] = 1.0
        k[M:, M:] = 1.0
        k[:M, M:] = -1.0
        k[M:, :M] = -1.0
        # Taper with a Hann window
        w = np.hanning(2 * M)
        k *= np.outer(w, w)
        return k

    def detect(
        self,
        features: np.ndarray,  # shape (T, D) — one feature vector per frame
        hop_size: float = 0.01,  # seconds per frame
    ) -> np.ndarray:
        """Return estimated boundary times in seconds."""
        # Self-similarity matrix
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        norm_feat = features / norms
        S = norm_feat @ norm_feat.T  # (T, T)

        # Novelty curve via convolution with checkerboard kernel
        T = S.shape[0]
        M = min(self.kernel_size, T // 4)
        kernel = self._checkerboard_kernel(M)  # (2M, 2M)
        kernel_size = 2 * M
        novelty = np.zeros(T)
        pad = M
        S_pad = np.pad(S, pad, mode="edge")
        for t in range(T):
            block = S_pad[t: t + kernel_size, t: t + kernel_size]
            novelty[t] = np.sum(block * kernel)

        # Smooth and threshold
        from scipy.ndimage import gaussian_filter1d
        novelty = gaussian_filter1d(novelty, sigma=2.0)
        threshold = np.percentile(novelty, self.threshold_percentile)
        peaks = _pick_peaks(novelty, threshold)
        return peaks * hop_size


def _pick_peaks(novelty: np.ndarray, threshold: float, min_dist: int = 10) -> np.ndarray:
    """Return indices of local maxima above threshold with minimum spacing."""
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(novelty, height=threshold, distance=min_dist)
    return peaks


# ---------------------------------------------------------------------------
# CNN boundary detector  (Ullrich et al. 2014 style)
# ---------------------------------------------------------------------------

class CNNBoundaryDetector(nn.Module):
    """Three-layer CNN operating on Mel-spectrogram patches.

    Architecture (Ullrich et al. 2014, "Boundary Detection in Music Structure
    Analysis using Convolutional Neural Networks"):
      - Input: (B, 1, n_mels, patch_frames) patch centred on candidate frame
      - Conv1: 32 filters, 3×3, ReLU, MaxPool 1×3
      - Conv2: 64 filters, 3×3, ReLU, MaxPool 1×3
      - Conv3: 64 filters, 3×1, ReLU
      - Flatten -> FC(128) -> Dropout(0.5) -> FC(1) -> sigmoid
    """

    def __init__(self, n_mels: int = 80, patch_frames: int = 15):
        super().__init__()
        self.n_mels = n_mels
        self.patch_frames = patch_frames

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
        )

        # Compute flattened size dynamically
        dummy = torch.zeros(1, 1, n_mels, patch_frames)
        with torch.no_grad():
            flat = self._forward_conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, patch_frames) -> (B,) probabilities."""
        return torch.sigmoid(self.fc(self._forward_conv(x))).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

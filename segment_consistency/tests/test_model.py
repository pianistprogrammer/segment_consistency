"""Tests for CNN boundary detector and Foote novelty baseline."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model import CNNBoundaryDetector, FooteNovelty
from src.utils import get_device


def test_cnn_output_shape():
    model = CNNBoundaryDetector(n_mels=80, patch_frames=15)
    x = torch.zeros(8, 1, 80, 15)
    out = model(x)
    assert out.shape == (8,), f"Expected (8,), got {out.shape}"


def test_cnn_output_range():
    model = CNNBoundaryDetector(n_mels=80, patch_frames=15)
    x = torch.randn(4, 1, 80, 15)
    out = model(x)
    assert (out >= 0.0).all() and (out <= 1.0).all(), "Output must be in [0, 1]"


def test_cnn_parameter_count():
    model = CNNBoundaryDetector(n_mels=80, patch_frames=15)
    params = model.count_parameters()
    assert 100_000 < params < 5_000_000, f"Unexpected param count: {params:,}"


def test_cnn_gradient_flow():
    model = CNNBoundaryDetector(n_mels=80, patch_frames=15)
    x = torch.randn(4, 1, 80, 15)
    labels = torch.rand(4)
    out = model(x)
    loss = torch.nn.BCELoss()(out, labels)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_foote_novelty_returns_array():
    foote = FooteNovelty(kernel_size=16)
    features = np.random.randn(200, 40).astype(np.float32)
    boundaries = foote.detect(features, hop_size=0.01)
    assert isinstance(boundaries, np.ndarray)
    assert boundaries.ndim == 1
    # All boundaries should be within the track duration
    assert (boundaries >= 0).all()
    assert (boundaries < 200 * 0.01).all()


def test_foote_empty_features():
    foote = FooteNovelty(kernel_size=8)
    features = np.random.randn(20, 10).astype(np.float32)
    boundaries = foote.detect(features, hop_size=0.01)
    assert isinstance(boundaries, np.ndarray)

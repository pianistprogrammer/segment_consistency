"""Tests for SALAMI annotation parser.

These tests verify the parser logic using synthetic annotation files written
to a temp directory — the real-dataset integration test is in test_pipeline.py
and is activated once the user provides the SALAMI root path.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.data import (
    Annotation,
    SALAMIDataset,
    Segment,
    _parse_parsed_file,
    normalise_function_label,
    strip_variation_suffix,
)


# ---------------------------------------------------------------------------
# _parse_parsed_file
# ---------------------------------------------------------------------------

def _make_parsed_file(tmp_path: Path, content: str) -> Path:
    f = tmp_path / "textfile1_uppercase.txt"
    f.write_text(content)
    return f


def test_parse_basic(tmp_path):
    content = "0.0\tSilence\n5.0\tA\n20.0\tB\n40.0\tSilence\n50.0\tEnd\n"
    path = _make_parsed_file(tmp_path, content)
    segs = _parse_parsed_file(path)
    labels = [s.label for s in segs]
    assert "A" in labels
    assert "B" in labels
    # Silence and End should be filtered out
    assert "Silence" not in labels
    assert "End" not in labels


def test_parse_timestamps(tmp_path):
    content = "0.0\tSilence\n5.123\tA\n25.456\tB\n50.0\tEnd\n"
    path = _make_parsed_file(tmp_path, content)
    segs = _parse_parsed_file(path)
    assert any(abs(s.start - 5.123) < 1e-6 for s in segs)


def test_parse_empty(tmp_path):
    path = _make_parsed_file(tmp_path, "")
    segs = _parse_parsed_file(path)
    assert segs == []


# ---------------------------------------------------------------------------
# Annotation.boundaries
# ---------------------------------------------------------------------------

def test_annotation_boundaries():
    ann = Annotation(
        song_id=1, annotator=1, level="uppercase",
        segments=[
            Segment(0.0, 10.0, "A"),
            Segment(10.0, 25.0, "B"),
            Segment(25.0, 50.0, "C"),
        ],
    )
    b = ann.boundaries
    assert len(b) == 2
    assert abs(b[0] - 10.0) < 1e-6
    assert abs(b[1] - 25.0) < 1e-6


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def test_strip_variation_suffix():
    # Explicit separator (_) before variant letter gets stripped
    assert strip_variation_suffix("Verse_A") == "Verse"
    # Plain words are not truncated
    assert strip_variation_suffix("Chorus") == "Chorus"
    assert strip_variation_suffix("Bridge") == "Bridge"
    # Single structural letter kept (no separator to strip)
    assert strip_variation_suffix("a") == "a"
    # Trailing digits stripped
    assert strip_variation_suffix("B2") == "B"


def test_normalise_function_label():
    assert normalise_function_label("Verse") == "verse"
    assert normalise_function_label("refrain") == "chorus"
    assert normalise_function_label("fade-out") == "outro"
    assert normalise_function_label("no_function") == "other"


# ---------------------------------------------------------------------------
# SALAMIDataset with synthetic data
# ---------------------------------------------------------------------------

def _make_fake_salami(tmp_path: Path) -> Path:
    """Create a minimal fake SALAMI directory structure for testing."""
    for song_id in (1, 2):
        for ann_n in (1, 2):
            parsed_dir = tmp_path / "annotations" / str(song_id) / "parsed"
            parsed_dir.mkdir(parents=True, exist_ok=True)
            # uppercase
            (parsed_dir / f"textfile{ann_n}_uppercase.txt").write_text(
                "0.0\tA\n20.0\tB\n40.0\tC\n60.0\tEnd\n"
            )
            # lowercase
            (parsed_dir / f"textfile{ann_n}_lowercase.txt").write_text(
                "0.0\ta\n10.0\tb\n20.0\tc\n30.0\td\n40.0\te\n50.0\tf\n60.0\tEnd\n"
            )
            # functions
            (parsed_dir / f"textfile{ann_n}_functions.txt").write_text(
                "0.0\tIntro\n20.0\tVerse\n40.0\tChorus\n60.0\tEnd\n"
            )
    return tmp_path


def test_dataset_loads(tmp_path):
    root = _make_fake_salami(tmp_path)
    ds = SALAMIDataset(root)
    assert len(ds) == 2


def test_dataset_dual(tmp_path):
    root = _make_fake_salami(tmp_path)
    ds = SALAMIDataset(root, require_dual=True)
    assert len(ds) == 2
    ids = ds.dual_annotated_ids()
    assert len(ids) == 2


def test_dataset_get_annotation(tmp_path):
    root = _make_fake_salami(tmp_path)
    ds = SALAMIDataset(root)
    track = ds[1]
    upper = track.get(1, "uppercase")
    assert upper is not None
    assert len(upper.segments) > 0
    lower = track.get(1, "lowercase")
    assert lower is not None
    assert len(lower.segments) > len(upper.segments)

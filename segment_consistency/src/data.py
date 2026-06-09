"""SALAMI dataset loading and annotation parsing.

SALAMI annotation structure:
  annotations/<song_id>/
    textfile1.txt                  -- raw annotator 1 (tab-separated: time\\tlabels)
    textfile2.txt                  -- raw annotator 2
    parsed/
      textfile1_uppercase.txt      -- coarse level (A, B, C ...)
      textfile1_lowercase.txt      -- fine level   (a, b, c ...)
      textfile1_functions.txt      -- functional   (Verse, Chorus ...)
      textfile2_uppercase.txt
      textfile2_lowercase.txt
      textfile2_functions.txt
  metadata.csv
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float
    label: str


@dataclass
class Annotation:
    """One level of one annotator's pass on one track."""
    song_id: int
    annotator: int          # 1 or 2
    level: str              # "uppercase", "lowercase", "functions"
    segments: list[Segment] = field(default_factory=list)

    @property
    def boundaries(self) -> np.ndarray:
        """All boundary times excluding the very first onset (0.0) and End."""
        times = [s.start for s in self.segments[1:]]
        return np.array(times, dtype=float)

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.segments]

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0


@dataclass
class Track:
    """All annotations for a single SALAMI track."""
    song_id: int
    duration: float | None
    title: str
    artist: str
    genre: str
    annotations: dict[tuple[int, str], Annotation] = field(default_factory=dict)

    def get(self, annotator: int, level: str) -> Annotation | None:
        return self.annotations.get((annotator, level))

    def has_dual(self) -> bool:
        """True if both annotators provided uppercase AND lowercase annotations."""
        for ann in (1, 2):
            for lvl in ("uppercase", "lowercase"):
                if (ann, lvl) not in self.annotations:
                    return False
        return True


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_SKIP_LABELS = {"silence", "end", "n/a", ""}


def _parse_parsed_file(path: Path) -> list[Segment]:
    """Parse one of the 'parsed/' files (timestamp TAB label)."""
    rows: list[tuple[float, str]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            try:
                t = float(parts[0])
            except ValueError:
                continue
            label = parts[1].strip()
            rows.append((t, label))

    if not rows:
        return []

    segments: list[Segment] = []
    for i, (t, label) in enumerate(rows):
        if label.lower() in _SKIP_LABELS:
            continue
        end = rows[i + 1][0] if i + 1 < len(rows) else t
        segments.append(Segment(start=t, end=end, label=label))

    return segments


def _load_annotation(song_dir: Path, annotator: int, level: str) -> Annotation | None:
    parsed_path = song_dir / "parsed" / f"textfile{annotator}_{level}.txt"
    if not parsed_path.exists():
        return None
    segs = _parse_parsed_file(parsed_path)
    if not segs:
        return None
    song_id = int(song_dir.name)
    return Annotation(song_id=song_id, annotator=annotator, level=level, segments=segs)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class SALAMIDataset:
    """Load the SALAMI dataset from a local root directory.

    Args:
        root: Path to the extracted salami-data-public directory.
        require_dual: If True, only return tracks that have both annotators'
                      uppercase and lowercase annotations.
    """

    LEVELS = ("uppercase", "lowercase", "functions")
    ANNOTATORS = (1, 2)

    def __init__(self, root: str | Path, require_dual: bool = False):
        self.root = Path(root)
        self._tracks: dict[int, Track] = {}
        self._metadata = self._load_metadata()
        self._scan(require_dual=require_dual)

    # ------------------------------------------------------------------
    def _load_metadata(self) -> pd.DataFrame:
        meta_path = self.root / "metadata" / "metadata.csv"
        if not meta_path.exists():
            # try root-level
            meta_path = self.root / "metadata.csv"
        if not meta_path.exists():
            return pd.DataFrame(columns=["SONG_ID", "TITLE", "ARTIST", "GENRE", "DURATION"])
        df = pd.read_csv(meta_path)
        # normalise column names
        df.columns = [c.strip().upper() for c in df.columns]
        return df

    def _meta_for(self, song_id: int) -> dict:
        row = self._metadata[self._metadata["SONG_ID"] == song_id]
        if row.empty:
            return {"TITLE": "", "ARTIST": "", "GENRE": "Unknown", "DURATION": None}
        r = row.iloc[0]
        return {
            "TITLE": str(r.get("TITLE", "")),
            "ARTIST": str(r.get("ARTIST", "")),
            "GENRE": str(r.get("GENRE", "Unknown")),
            "DURATION": float(r["DURATION"]) if "DURATION" in r and pd.notna(r["DURATION"]) else None,
        }

    # ------------------------------------------------------------------
    def _scan(self, require_dual: bool) -> None:
        ann_root = self.root / "annotations"
        if not ann_root.exists():
            raise FileNotFoundError(f"No 'annotations' directory found under {self.root}")

        for song_dir in sorted(ann_root.iterdir()):
            if not song_dir.is_dir():
                continue
            try:
                song_id = int(song_dir.name)
            except ValueError:
                continue

            meta = self._meta_for(song_id)
            track = Track(
                song_id=song_id,
                duration=meta["DURATION"],
                title=meta["TITLE"],
                artist=meta["ARTIST"],
                genre=meta["GENRE"],
            )

            for ann in self.ANNOTATORS:
                for lvl in self.LEVELS:
                    a = _load_annotation(song_dir, ann, lvl)
                    if a is not None:
                        track.annotations[(ann, lvl)] = a

            if require_dual and not track.has_dual():
                continue
            if track.annotations:
                self._tracks[song_id] = track

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._tracks)

    def __iter__(self) -> Iterator[Track]:
        return iter(self._tracks.values())

    def __getitem__(self, song_id: int) -> Track:
        return self._tracks[song_id]

    def song_ids(self) -> list[int]:
        return sorted(self._tracks.keys())

    def dual_annotated_ids(self) -> list[int]:
        return sorted(sid for sid, t in self._tracks.items() if t.has_dual())

    def summary(self) -> pd.DataFrame:
        rows = []
        for sid, t in self._tracks.items():
            rows.append({
                "song_id": sid,
                "genre": t.genre,
                "has_ann1_upper": (1, "uppercase") in t.annotations,
                "has_ann1_lower": (1, "lowercase") in t.annotations,
                "has_ann1_func": (1, "functions") in t.annotations,
                "has_ann2_upper": (2, "uppercase") in t.annotations,
                "has_ann2_lower": (2, "lowercase") in t.annotations,
                "has_ann2_func": (2, "functions") in t.annotations,
                "is_dual": t.has_dual(),
            })
        return pd.DataFrame(rows).set_index("song_id").sort_index()


# ---------------------------------------------------------------------------
# Label normalisation helpers
# ---------------------------------------------------------------------------

def strip_variation_suffix(label: str) -> str:
    """Remove variation markers from a label.

    Examples: 'Verse_A' -> 'Verse', 'a'' -> 'a', 'B2' -> 'B', 'Chorus' -> 'Chorus'.
    Only strips a trailing letter when it is preceded by '_', a space, or a digit
    (so plain words like 'Chorus' are not truncated).
    """
    import re
    label = label.strip().rstrip("'\"")
    # Remove trailing digits
    label = re.sub(r"\d+$", "", label)
    # Remove _X or ' X' suffix (explicit separator before single letter)
    label = re.sub(r"[_\s][A-Za-z]$", "", label)
    return label.strip()


def normalise_function_label(label: str) -> str:
    """Lower-case and collapse common synonyms to a canonical form."""
    label = label.lower().strip()
    _MAP = {
        "verse": "verse", "pre-verse": "verse",
        "chorus": "chorus", "refrain": "chorus", "hook": "chorus",
        "bridge": "bridge",
        "intro": "intro", "introduction": "intro", "fade-in": "intro",
        "outro": "outro", "coda": "outro", "closing": "outro", "fade-out": "outro",
        "interlude": "interlude", "transition": "interlude",
        "solo": "solo", "improvisation": "solo",
        "instrumental": "instrumental",
        "theme": "theme", "main_theme": "theme",
        "no_function": "other",
    }
    return _MAP.get(label, label)

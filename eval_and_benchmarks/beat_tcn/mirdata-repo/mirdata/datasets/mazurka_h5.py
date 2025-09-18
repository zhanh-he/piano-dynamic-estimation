"""MazurkaBL H5 Dataset Loader (local, no remote download)

Loads HDF5 files produced by our preprocessing pipeline. Each H5 contains:
- dataset keys: 'waveform' (int16), 'beat_time' (float seconds), optional 'downbeat_time' (float seconds)
- attrs: 'sample_rate' (int), 'duration_librosa' (float, optional), 'frames_per_second' (int, optional)

Indexing: a local JSON index lists relative paths to H5 files under data_home.
This loader uses version 'local' by default and does not define REMOTES.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, BinaryIO
import os
import json
import numpy as np
import h5py

from mirdata import core, annotations


BIBTEX = """@inproceedings{your2025mazurka,
  title={MazurkaBL Piano Dynamics & Beat Annotations in HDF5},
  author={Your Name and Collaborators},
  booktitle={Your Venue},
  year={2025}
}"""

# Single local index; create this JSON with the helper script or manually.
INDEXES = {
    "default": "local",
    "local": core.Index(filename="mazurka_h5_local.json"),
}

REMOTES: dict = {}

LICENSE_INFO = "Dataset for research. Contact authors for licensing."


def _load_h5_audio(h5_path: str) -> Tuple[np.ndarray, float]:
    with h5py.File(h5_path, "r") as hf:
        wav = hf["waveform"][:]
        sr = int(hf.attrs.get("sample_rate", 22050))
    # int16 -> float32 in [-1,1]
    if wav.dtype == np.int16:
        audio = (wav.astype(np.float32) / 32768.0).copy()
    else:
        audio = wav.astype(np.float32)
    return audio, float(sr)


def _load_h5_beats(h5_path: str) -> annotations.BeatData:
    with h5py.File(h5_path, "r") as hf:
        times = hf["beat_time"][:].astype(np.float64)
        downbeat = hf["downbeat_time"][:] if "downbeat_time" in hf else None
        fps = float(hf.attrs.get("frames_per_second", 50.0))
    positions = None
    if downbeat is not None and len(downbeat) > 0:
        # Assign downbeat positions=1 to beats within a small tolerance
        tol = max(1.0 / fps, 0.02)
        positions = np.zeros(len(times), dtype=int)
        j = 0
        for i, t in enumerate(times):
            while j < len(downbeat) and downbeat[j] < t - tol:
                j += 1
            if j < len(downbeat) and abs(downbeat[j] - t) <= tol:
                positions[i] = 1
        # Use 0/1 as bar_index flag
    beat_data = annotations.BeatData(
        times=times, time_unit="s", positions=positions, position_unit="bar_index"
    )
    return beat_data


class Track(core.Track):
    """MazurkaBL-H5 Track

    Attributes:
        h5_path (str): path to the H5 file
    Cached Properties:
        beats (BeatData): beat positions (with downbeat flagged as position=1 when available)
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)
        self.h5_path = self.get_path("h5")

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        """Frame-level beats; downbeats marked via positions==1 when available."""
        return _load_h5_beats(self.h5_path)

    @core.cached_property
    def downbeats(self) -> Optional[annotations.BeatData]:
        """Downbeat-only annotations, if present in H5 as 'downbeat_time'."""
        with h5py.File(self.h5_path, "r") as hf:
            if "downbeat_time" not in hf:
                return None
            times = hf["downbeat_time"][:].astype(np.float64)
        if times.size == 0:
            return None
        # positions all ones to indicate downbeat
        positions = np.ones(len(times), dtype=int)
        return annotations.BeatData(
            times=times, time_unit="s", positions=positions, position_unit="bar_index"
        )

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        return _load_h5_audio(self.h5_path)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    MazurkaBL H5 dataset (local-only).

    Usage:
        >>> from mirdata.datasets import mazurka_h5
        >>> ds = mazurka_h5.Dataset(data_home="/path/to/mazurka_sr22050", version="local")
        >>> t = ds.track(ds.track_ids[0])
        >>> audio, sr = t.audio
        >>> beats = t.beats
    """

    def __init__(self, data_home=None, version="local"):
        super().__init__(
            data_home,
            version,
            name="mazurka_h5",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

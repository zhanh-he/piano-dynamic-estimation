from __future__ import annotations
import hashlib, json, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import h5py
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd

# ------------------------- Feature extraction -------------------------
def compute_logmel(y: np.ndarray, sr: int, fps: int = 50,
                   n_mels: int = 128, n_fft: int = 2048,
                   fmin: float = 30.0, fmax: float | None = None) -> Tuple[np.ndarray, int]:
    """Return [T, n_mels] log-mel and hop_length used."""
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    hop = int(round(sr / float(fps)))
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0, center=True
    )  # [n_mels, T]
    S = np.log10(np.maximum(S, 1e-10)).astype(np.float32)
    return S.T, hop  # [T, n_mels], hop

## Sone feature removed — original BeatThis uses log-mel only.

def times_to_mask(times_s: np.ndarray, fps: float, L: int, offset_frames: int) -> np.ndarray:
    """Convert second times to a binary frame mask for an excerpt starting at offset_frames."""
    idx = np.round(times_s * fps).astype(int) - offset_frames
    idx = idx[(idx >= 0) & (idx < L)]
    m = np.zeros(L, dtype=np.float32)
    if idx.size:
        m[idx] = 1.0
    return m

# ------------------------------ Dataset ------------------------------
class H5BeatDataset(Dataset):
    """
    Reads HDF5 files with datasets:
      waveform[int16] (required)
      beat_time[float32] (required, seconds)
      downbeat_time[float32] (optional, seconds)
    Returns dict expected by PLBeatThis.
    """
    def __init__(self,
                 files: List[Path] | List[str],
                 train_length: Optional[int] = 1500,
                 deterministic: bool = False,
                 default_sr: int = 22050,
                 fps: int = 50,
                 cache_dir: str | Path | None = "data/_h5_mel_cache",
                 cache_mode: str = "readwrite",  # 'off'|'readonly'|'readwrite'
                 ):
        self.files = [Path(p) for p in files]
        self.train_length = train_length
        self.deterministic = deterministic
        self.default_sr = default_sr
        self.fps = fps
        self.cache_dir = None if cache_dir in (None, "", "off") else Path(cache_dir)
        self.cache_mode = cache_mode

    def __len__(self) -> int:
        return len(self.files)

    # unique cache path per (file, sr, fps)
    def _cache_path(self, h5_path: Path, sr: int) -> Optional[Path]:
        if self.cache_dir is None: return None
        key = f"{h5_path.resolve()}|sr={sr}|fps={self.fps}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        # nice layout: mazurka/<opus>/<pid>/track.npy
        nice = Path("mazurka") / h5_path.parent.name / h5_path.stem / f"track_{h}.npy"
        return self.cache_dir / nice

    def __getitem__(self, i: int) -> Dict:
        p = self.files[i]
        with h5py.File(p, "r") as hf:
            y = hf["waveform"][:]  # int16
            sr = int(hf.attrs.get("sample_rate", self.default_sr))
            beat_time = hf["beat_time"][:]                    # seconds
            if "downbeat_time" in hf:
                downbeat_time = hf["downbeat_time"][:]
                has_db = True
            else:
                downbeat_time = np.zeros(0, dtype=np.float32)
                has_db = False

        # ---- cache lookup / compute ----
        cache_path = self._cache_path(p, sr)
        x = None
        if cache_path is not None and cache_path.exists():
            try:
                x = np.load(cache_path, mmap_mode="r")
            except Exception:
                x = None

        if x is None:
            spec, _ = compute_logmel(y, sr, fps=self.fps)  # [T,128]
            x = spec
            if cache_path is not None and self.cache_mode in ("readwrite", "write"):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_suffix(".tmp.npy")
                np.save(tmp, x)
                tmp.replace(cache_path) 
        T = x.shape[0]

        # ---- choose excerpt ----
        if self.train_length is None or T <= self.train_length:
            start, end = 0, T
            pad = (self.train_length - T) if self.train_length is not None else 0
        else:
            span = T - self.train_length
            start = span // 2 if self.deterministic else np.random.randint(0, span + 1)
            end = start + self.train_length
            pad = 0

        x = x[start:end]                      # [L,128]
        L = x.shape[0]
        frame_offset = start                  # frames offset inside full spectrogram

        yb = times_to_mask(beat_time, self.fps, L, frame_offset)
        ydb = times_to_mask(downbeat_time, self.fps, L, frame_offset) if has_db else np.zeros(L, np.float32)

        if pad and pad > 0:
            x   = np.pad(x,   ((0, pad), (0, 0)), constant_values=0)
            yb  = np.pad(yb,  (0, pad), constant_values=0)
            ydb = np.pad(ydb, (0, pad), constant_values=0)
            L = x.shape[0]

        return {
            "spect": x.astype(np.float32),                         # [L,128]
            "truth_beat": yb.astype(np.float32),                   # [L]
            "truth_downbeat": ydb.astype(np.float32),              # [L]
            "padding_mask": np.ones(L, dtype=bool),                # [L]
            "downbeat_mask": torch.tensor(has_db),                 # scalar bool
            "spect_path": str(p),                                  # identifier only
            "dataset": "mazurka",
            "start_frame": int(start),
            # IMPORTANT: use float64 bytes to match np.frombuffer default in PL module
            "truth_orig_beat": np.asarray(beat_time, dtype=np.float64).tobytes(),
            "truth_orig_downbeat": np.asarray(downbeat_time, dtype=np.float64).tobytes(),
        }

# ------------------------------ DataModule ------------------------------
class H5BeatDataModule(pl.LightningDataModule):
    """
    Discovers .h5 files under h5_root, splits train/val(/test),
    builds datasets with optional feature cache, and provides fast
    class weights computation using H5 metadata (cached to JSON).
    """

    def __init__(self,
                 h5_root: str | Path,
                 batch_size: int = 8,
                 num_workers: int = 8,
                 sr: int = 22050,
                 fps: int = 50,
                 train_length: int = 1500,
                 val_ratio: float = 0.1,
                 seed: int = 0,
                 cache_dir: str | Path = "data/_h5_mel_cache",
                 csv_split: Optional[str] = None,
                 fold: Optional[int] = None,
                 ):
        super().__init__()
        self.h5_root = Path(h5_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.default_sr = sr
        self.fps = fps
        self.train_length = train_length
        self.val_ratio = val_ratio
        self.seed = seed
        self.cache_dir = Path(cache_dir)
        self.csv_split = csv_split
        self.fold = fold
        self.initialized = {}

    def setup(self, stage: Optional[str] = None):
        if self.csv_split is not None:
            df = pd.read_csv(self.csv_split)
            if self.fold is not None and "fold" in df.columns:
                df = df[df["fold"] == self.fold]
            def resolve(paths: List[str]) -> List[Path]:
                resolved = []
                all_h5 = {p.name: p for p in self.h5_root.rglob("*.h5")}  # map basename -> full path
                for name in paths:
                    if name in all_h5:
                        resolved.append(all_h5[name])
                    else:
                        raise FileNotFoundError(f"Could not find {name} under {self.h5_root}")
                return resolved

            train_files = resolve(df[df["split"] == "train"]["h5_name"].tolist())
            val_files   = resolve(df[df["split"] == "valid"]["h5_name"].tolist())
            test_files  = resolve(df[df["split"] == "test"]["h5_name"].tolist())
        else:
            # fallback: random split like before
            files = sorted(self.h5_root.rglob("*.h5"))
            rng = np.random.RandomState(self.seed)
            idx = np.arange(len(files)); rng.shuffle(idx)
            val_n = max(1, int(round(len(files) * self.val_ratio)))
            train_files = [files[i] for i in idx[val_n:]]
            val_files   = [files[i] for i in idx[:val_n]]
            test_files  = files

        # now build datasets same as before
        if stage in (None, "fit", "validate") and not self.initialized.get("fit"):
            self.train_ds = H5BeatDataset(
                train_files,
                train_length=self.train_length,
                deterministic=False,
                default_sr=self.default_sr,
                fps=self.fps,
                cache_dir=self.cache_dir,
                cache_mode="readwrite",
            )
            self.val_ds = H5BeatDataset(
                val_files,
                train_length=self.train_length,
                deterministic=True,
                default_sr=self.default_sr,
                fps=self.fps,
                cache_dir=self.cache_dir,
                cache_mode="readonly",
            )
            logging.getLogger("beat_this").info(f"Train: {len(self.train_ds)} | Val: {len(self.val_ds)}")
            self.initialized["fit"] = True

        if stage in (None, "test") and not self.initialized.get("test"):
            self.test_ds = H5BeatDataset(
                test_files,
                train_length=None,          # full sequence for test
                deterministic=True,
                default_sr=self.default_sr,
                fps=self.fps,
                cache_dir=self.cache_dir,
                cache_mode="readonly",
            )
            logging.getLogger("beat_this").info(f"Test: {len(self.test_ds)}")
            self.initialized["test"] = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=1, num_workers=self.num_workers, pin_memory=True
        )

    # ---------------- fast, cached positive weights ----------------
    def _stats_cache_key(self, train_files: List[Path]) -> str:
        h = hashlib.md5()
        h.update(f"fps={self.fps};trainlen={self.train_length};seed={self.seed};val={self.val_ratio}".encode())
        h.update(f"N={len(train_files)}".encode())
        # sample up to ~50 names to keep key stable but short
        step = max(1, len(train_files) // 50)
        for p in train_files[::step]:
            h.update(str(p).encode())
        return h.hexdigest()[:12]

    def _scan_one(self, h5_path: Path) -> Tuple[int, int, int, bool]:
        """Return (frames, beats, downbeats, has_downbeats) using H5 only (fast)."""
        with h5py.File(h5_path, "r") as hf:
            sr = int(hf.attrs.get("sample_rate", self.default_sr))
            n_samples = int(hf["waveform"].shape[0])
            hop = sr / float(self.fps)
            frames = int(np.ceil(n_samples / hop))
            beats = int(hf["beat_time"].shape[0])
            if "downbeat_time" in hf:
                downbeats = int(hf["downbeat_time"].shape[0])
                has_db = True
            else:
                downbeats = 0
                has_db = False
        return frames, beats, downbeats, has_db

    def get_train_positive_weights(self, widen_target_mask: int = 3) -> Dict[str, int]:
        """Fast, metadata-only computation with JSON cache."""
        stats_dir = self.cache_dir / "_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        train_files = [Path(p) for p in self.train_ds.files]
        key = self._stats_cache_key(train_files)
        cache_path = stats_dir / f"pos_weights_{key}.json"

        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                all_frames = int(data["all_frames"])
                all_frames_db = int(data["all_frames_db"])
                beat_frames = int(data["beat_frames"])
                downbeat_frames = int(data["downbeat_frames"])
            except Exception:
                cache_path.unlink(missing_ok=True)
                return self.get_train_positive_weights(widen_target_mask)

        else:
            all_frames = all_frames_db = beat_frames = downbeat_frames = 0
            for p in train_files:
                frames, beats, downbeats, has_db = self._scan_one(p)
                all_frames += frames
                beat_frames += beats
                if has_db:
                    all_frames_db += frames
                    downbeat_frames += downbeats

            cache_path.write_text(json.dumps(dict(
                all_frames=all_frames,
                all_frames_db=all_frames_db,
                beat_frames=beat_frames,
                downbeat_frames=downbeat_frames
            )))

        widen = (widen_target_mask * 2 + 1)
        beat_neg = max(all_frames - beat_frames * widen, 1)
        db_neg = max(all_frames_db - downbeat_frames * widen, 1) if all_frames_db else 1

        beat_w = int(round(beat_neg / max(beat_frames, 1))) or 1
        db_w = int(round(db_neg / max(downbeat_frames, 1))) if downbeat_frames else 1
        db_w = db_w or 1
        return {"beat": beat_w, "downbeat": db_w}

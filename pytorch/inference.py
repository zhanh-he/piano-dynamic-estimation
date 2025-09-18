"""Song-level inference: segment -> predict -> stitch -> save H5."""

from __future__ import annotations
import argparse
import csv
import os
from typing import Dict, Tuple, Any, List

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from post_processor.dynamic_postproc import detect_change_point
from post_processor.beat_postproc import Postprocessor
from utils import int16_to_float32, parse_overrides_str, select_checkpoint
from models_audioCNN import SingleCNN, MultiTaskCNN


# ---------------- Helpers (cfg) ----------------
def _fps(cfg) -> int: return int(getattr(cfg.feature, "frames_per_second", 50))
def _sr(cfg) -> int: return int(getattr(cfg.feature, "sample_rate", 22050))
def _seg_sec(cfg) -> float: return float(getattr(cfg.feature, "segment_seconds", 60))

# ---------------- Model load ----------------
_MODEL = {"SingleCNN": SingleCNN, "MultiTaskCNN": MultiTaskCNN}

def _load_ckpt(path: str, device: torch.device):
    try:  # fast path
        return torch.load(path, map_location=device) 
    except Exception:  # fallback
        return torch.load(path, map_location=device, weights_only=False)

def load_model(checkpoint_path: str, device: torch.device, overrides: Dict[str, Any] | None) -> Tuple[torch.nn.Module, OmegaConf]:
    if not os.path.isfile(checkpoint_path): raise FileNotFoundError(checkpoint_path)
    ckpt = _load_ckpt(checkpoint_path, device)
    cfg = OmegaConf.create(ckpt["cfg"])
    if overrides: cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    name = str(getattr(cfg.exp, "model_name"))
    if name not in _MODEL: raise ValueError(f"Unknown model_name: {name}")
    model = _MODEL[name](cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, cfg


# ---------------- Enframing / stitching ----------------
def _frames_for_duration_sec(duration_sec: float, fps: int) -> int:
    return int(np.round(duration_sec * fps)) + 1

def _enframe_indices(n_total: int, seg_len: int, hop: int) -> List[tuple[int, int]]:
    if n_total <= 0: return []
    if n_total <= seg_len: return [(0, seg_len)]
    n = int(np.ceil((n_total - seg_len) / hop)) + 1
    return [(i * hop, i * hop + seg_len) for i in range(n)]

def _stitch(segments: List[np.ndarray], frames_total: int, hop_frames: int) -> np.ndarray:
    if not segments: return np.zeros((frames_total, 1), dtype=np.float32)
    C = segments[0].shape[1]
    out = np.zeros((frames_total, C), np.float32)
    w = np.zeros((frames_total, 1), np.float32)
    ptr = 0
    for seg in segments:
        T = seg.shape[0]
        s, e = ptr, min(frames_total, ptr + T)
        out[s:e, :] += seg[: e - s]
        w[s:e, :] += 1.0
        ptr += hop_frames
    w[w == 0] = 1.0
    return out / w


# ---------------- Predict / derive / save ----------------
def _predict_song(cfg, model: torch.nn.Module, device: torch.device, h5_path: str) -> Dict[str, np.ndarray]:
    fps, sr = _fps(cfg), _sr(cfg)
    seg_sec = _seg_sec(cfg)
    hop_sec = seg_sec  # non-overlap
    seg_samples, hop_samples = int(round(seg_sec * sr)), int(round(hop_sec * sr))
    hop_frames = int(round(hop_sec * fps))

    with h5py.File(h5_path, "r") as hf:
        wav = int16_to_float32(hf["waveform"][:])
        duration_sec = float(hf.attrs.get("duration_librosa", len(wav) / sr))

    frames_total = _frames_for_duration_sec(duration_sec, fps)
    idx = _enframe_indices(len(wav), seg_samples, hop_samples)
    if not idx: return {}

    dyn_segs: List[np.ndarray] = []
    beat_segs: List[np.ndarray] = []
    down_segs: List[np.ndarray] = []
    cp_segs: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for s, e in idx:
            seg = wav[s: min(e, len(wav))]
            if len(seg) < seg_samples:
                seg = np.pad(seg, (0, seg_samples - len(seg)), mode="constant")
            x = torch.from_numpy(seg.astype(np.float32))[None, :].to(device)
            out = model(x)
            if "dynamic_output" in out:
                d = out["dynamic_output"].detach().cpu().numpy()[0].astype(np.float32)
                dyn_segs.append(d)
            if "beat_output" in out:
                b = out["beat_output"].detach().cpu().numpy()[0].astype(np.float32)
                if b.ndim == 1: b = b[:, None]
                beat_segs.append(b)
            if "downbeat_output" in out:
                db = out["downbeat_output"].detach().cpu().numpy()[0].astype(np.float32)
                if db.ndim == 1: db = db[:, None]
                down_segs.append(db)
            if "change_point_output" in out:
                c = out["change_point_output"].detach().cpu().numpy()[0].astype(np.float32)
                if c.ndim == 1: c = c[:, None]
                cp_segs.append(c)

    pred: Dict[str, np.ndarray] = {}
    if dyn_segs:  pred["dynamic_output"] = _stitch(dyn_segs, frames_total, hop_frames)
    if beat_segs: pred["beat_output"]    = _stitch(beat_segs, frames_total, hop_frames)
    if down_segs: pred["downbeat_output"] = _stitch(down_segs, frames_total, hop_frames)
    if cp_segs:   pred["change_point_output"] = _stitch(cp_segs, frames_total, hop_frames)
    return pred


# Derived results for practical use (not final eval)
def _practical_derive(cfg, pred: Dict[str, np.ndarray], h5_path: str | None = None) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    fps = _fps(cfg)

    # Beats / downbeats via post-processor
    if ("beat_output" in pred) or ("downbeat_output" in pred):
        beat_logits  = torch.from_numpy(pred["beat_output"]).float().cpu()   if "beat_output" in pred else torch.zeros((pred.get("downbeat_output").shape[0], 1))
        down_logits  = torch.from_numpy(pred["downbeat_output"]).float().cpu() if "downbeat_output" in pred else torch.zeros_like(beat_logits)
        mask = torch.ones_like(beat_logits, dtype=torch.bool)
        post_proc = Postprocessor(type="minimal", fps=fps)
        beat_times, down_times = post_proc(beat_logits.squeeze(-1), down_logits.squeeze(-1), mask.squeeze(-1))

        # beat roll/time
        bt = np.asarray(beat_times).ravel()
        br = np.zeros((beat_logits.shape[0],), np.float32)
        inds = np.round(bt * fps).astype(int)
        inds = inds[(inds >= 0) & (inds < br.shape[0])]
        br[inds] = 1.0
        out["pred_beat_roll"] = br
        out["pred_beat_time"] = bt.astype(np.float32)
        # Beat grid identifiers
        out["pred_beat_id"] = np.arange(len(inds), dtype=np.int64)
        out["pred_beat_frame_index"] = inds.astype(np.int64)

        # Downbeat roll/time
        if "downbeat_output" in pred:
            dt = np.asarray(down_times).ravel()
            dbr = np.zeros((down_logits.shape[0],), np.float32)
            d_inds = np.round(dt * fps).astype(int)
            d_inds = d_inds[(d_inds >= 0) & (d_inds < dbr.shape[0])]
            dbr[d_inds] = 1.0
            out["pred_downbeat_roll"] = dbr
            out["pred_downbeat_time"] = dt.astype(np.float32)

    # Dynamic classes
    if "dynamic_output" in pred:
        out["pred_dynamic_roll"] = np.argmax(pred["dynamic_output"], axis=-1).astype(np.int64)
        # Per-beat dynamic classes (if beats available)
        if "pred_beat_roll" in out:
            beat_idx = np.flatnonzero(out["pred_beat_roll"] > 0)
            if beat_idx.size > 0:
                T = pred["dynamic_output"].shape[0]
                beat_idx = beat_idx[(beat_idx >= 0) & (beat_idx < T)]
                dyn_classes = np.argmax(pred["dynamic_output"][beat_idx, :], axis=-1)
                out["pred_dynamic_beat_class"] = dyn_classes.astype(np.int64)
                out["pred_dynamic_beat_id"] = np.arange(len(dyn_classes), dtype=np.int64)

    # Change-points (needs beats)
    if "change_point_output" in pred:
        cp = pred["change_point_output"].squeeze()
        L = int(cp.shape[0])

        # No predicted beats -> skip CP post-process entirely
        if "pred_beat_roll" not in out:
            pass
        else:
            br = out["pred_beat_roll"]
            if br.shape[0] != L:
                br = (br[:L] if br.shape[0] > L else np.pad(br, (0, L - br.shape[0]), "constant"))
            # threshold=1.0 ~ 0.75 after sigmoid (by convention)
            idx = detect_change_point(cp, br, threshold=1.0)
            idx = np.asarray(idx, dtype=int).ravel()
            idx = idx[(idx >= 0) & (idx < L)]
            if idx.size:
                cp_roll = np.zeros((L,), np.float32); cp_roll[idx] = 1.0
                out["pred_change_point_roll"] = cp_roll
                # Optional: map to times via predicted beat times (if available)
                if "pred_beat_time" in out and np.asarray(out["pred_beat_time"]).size:
                    bt = np.asarray(out["pred_beat_time"]).ravel()
                    bi = np.round(bt * fps).astype(int)
                    j = np.array([int(np.argmin(np.abs(bi - i))) for i in idx], dtype=int)
                    out["pred_change_point_time"] = bt[j].astype(np.float32)
                    out["pred_change_point_beat_id"] = j.astype(np.int64)

    # Return all predicted/derived fields
    return out


def _save_h5(cfg, in_h5: str, pred: Dict[str, np.ndarray], drv: Dict[str, np.ndarray],
             out_root: str, subdir: str | None, override_out: str | None, add_suffix: bool) -> str:
    sr, fps = _sr(cfg), _fps(cfg)
    base = os.path.basename(in_h5)
    opus = os.path.basename(os.path.dirname(in_h5))
    name, ext = os.path.splitext(base)

    if override_out:
        out_dir = os.path.dirname(override_out) or "."
        os.makedirs(out_dir, exist_ok=True)
        out_path = override_out
    else:
        sub = subdir if subdir else f"mazurka_sr{sr}"
        out_dir = os.path.join(out_root, sub, opus)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}{ext}" if not add_suffix else f"{name}-model_output{ext}")

    with h5py.File(in_h5, "r") as src, h5py.File(out_path, "w") as dst:
        for k, v in src.attrs.items(): dst.attrs[k] = v
        dst.attrs["frames_per_second"] = fps; dst.attrs["sample_rate"] = sr
        for opt in ["beat_time", "downbeat_time", "change_point_time", "measure_time"]:
            if opt in src: dst.create_dataset(opt, data=src[opt][:])
        for k in ["dynamic_output", "beat_output", "downbeat_output", "change_point_output"]:
            if k in pred: dst.create_dataset(k, data=pred[k].astype(np.float32))
        for k, v in drv.items():
            dst.create_dataset(k, data=(v.astype(np.float32) if k.endswith("_roll") else v))
    return out_path


def run_inference(checkpoint_path: str, overrides: Dict[str, Any] | None, h5_paths: List[str],
                   out_root: str, subdir: str | None, override_out: str | None, add_suffix: bool) -> List[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(checkpoint_path, device, overrides)
    outs: List[str] = []
    progress = tqdm(h5_paths, desc=f"Files {os.path.basename(checkpoint_path)}", ncols=80, dynamic_ncols=True, position=0, leave=True)
    for file_index, h5_path in enumerate(progress):
        pred = _predict_song(cfg, model, device, h5_path)   # predicted results used in evaluation
        drv = _practical_derive(cfg, pred, h5_path=h5_path) # derived results used in practical
        ovr = override_out if (override_out and len(h5_paths) == 1 and file_index == 0) else None 
        outs.append(_save_h5(cfg, h5_path, pred, drv, out_root, subdir, ovr, add_suffix))
    return outs


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-song inference exporter (choose checkpoint by valf1_rank / min_epoch).")
    # Choose a checkpoint from --ckpt OR --ckpt_dir
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--ckpt", type=str, help="Path to a single checkpoint (.pth/.ckpt)")
    ckpt_group.add_argument("--ckpt_dir", type=str, help="Directory containing checkpoints")
    parser.add_argument("--valf1_rank", type=int, default=1, help="Pick ckpt at given valf1 rank (1=best)")
    parser.add_argument("--min_epoch", type=int, default=None, help="Filter out checkpoints with epoch < min_epoch")
    # Pick input source: single H5 OR CSV split H5s
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--infer_h5", type=str, help="Single input H5 path")
    input_group.add_argument("--infer_split_csv", type=str, help="CSV with columns [h5_name, opus, split]; only rows where split=='test' are used")
    parser.add_argument("--overrides", type=str, default=None, help="Hydra-style overrides (key=value,key2=value2)")
    parser.add_argument("--output_dir", type=str, default=None, help="Base output dir; default: <parent_of_ckpt_dir>/outputs")
    
    args = parser.parse_args()
    overrides = parse_overrides_str(args.overrides)
    ckpt = args.ckpt or select_checkpoint(args.ckpt_dir, args.valf1_rank, args.min_epoch)
    
    def _ckpt_out_base(ckpt_path: str) -> str:
        cdir = os.path.dirname(ckpt_path)
        cbase = os.path.basename(cdir)
        cname = os.path.splitext(os.path.basename(ckpt_path))[0]
        root = args.output_dir if args.output_dir else os.path.join(os.path.dirname(cdir), "outputs")
        return os.path.join(root, cbase, cname)

    # single H5
    if args.infer_h5:
        in_h5 = args.infer_h5
        base_out = _ckpt_out_base(ckpt)
        out_root = base_out
        override_out = os.path.join(out_root, os.path.basename(in_h5))  # save as <h5_name>.h5
        outs = run_inference(ckpt, overrides, [in_h5], out_root, None, override_out, False)
        print(f"[{os.path.basename(ckpt)}] -> {outs[0]}")
        return

    # CSV: collect test H5s via cfg.workspace + sr (from selected ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _m0, cfg0 = load_model(ckpt, device, overrides)
    sr = _sr(cfg0); base_h5_dir = os.path.join(cfg0.exp.workspace, "hdf5s", f"mazurka_sr{sr}")

    rows: List[tuple[str, str]] = []
    with open(args.infer_split_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("split") or "").strip() != "test": continue
            opus = (row.get("opus") or "").strip()
            name = (row.get("h5_name") or "").strip()
            if opus and name: rows.append((opus, name))

    h5_paths: List[str] = []
    for opus, name in rows:
        path = os.path.join(base_h5_dir, opus, name)
        if os.path.isfile(path): h5_paths.append(path)
        else: tqdm.write(f"[WARN] H5 not found: {path}")

    base_out = _ckpt_out_base(ckpt)
    outs = run_inference(ckpt, overrides, h5_paths, base_out, f"mazurka_sr{sr}", None, False)
    print(f"[{os.path.basename(ckpt)}] Exported {len(outs)} file(s) under {base_out}")

if __name__ == "__main__":
    main()
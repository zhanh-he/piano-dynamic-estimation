"""Evaluate predictions vs GT on test split (baseline/advanced)."""
from __future__ import annotations
import argparse, csv, json, os
from typing import Dict, List
import h5py, numpy as np, torch
from sklearn.metrics import f1_score, accuracy_score

from evaluator import cal_dynamic_cci_and_weighted_f1, cal_beat_f1_beatthis, cal_change_point_f1
from post_processor.beat_postproc import Postprocessor
from inference import load_model
from dataloader import time_to_frame_roll, time_to_dynamic_roll
from utils import parse_overrides_str, select_checkpoint, pad_or_truncate_np

# ---------------- Helpers ----------------
def _fps(cfg) -> int: return int(getattr(cfg.feature, "frames_per_second", 50))
def _sr(cfg) -> int: return int(getattr(cfg.feature, "sample_rate", 22050))
def _dyn_classes(cfg) -> int: return int(getattr(cfg.feature, "dynamic_classes", 5))

def _pad_to_tensor(pred_h5, key: str, T: int) -> torch.Tensor:
    """Pad/cast H5 prediction to torch tensor length T."""
    return torch.from_numpy(pad_or_truncate_np(pred_h5[key][:].squeeze(), T, 0.0, 0)).float()

def _pred_path(pred_root: str, gt_path: str) -> str:
    base = os.path.basename(gt_path)
    opus = os.path.basename(os.path.dirname(gt_path))
    sr_dir = os.path.basename(os.path.dirname(os.path.dirname(gt_path)))
    return os.path.join(pred_root, sr_dir, opus, base)

def _ckpt_pred_root(ckpt_path: str, output_root: str | None) -> str:
    cdir = os.path.dirname(ckpt_path)
    cbase = os.path.basename(cdir)
    cname = os.path.splitext(os.path.basename(ckpt_path))[0]
    root = output_root if output_root else os.path.join(os.path.dirname(cdir), "outputs")
    return os.path.join(root, cbase, cname)

def _collect_groundtruth(cfg, csv_path: str) -> List[str]:
    base = os.path.join(cfg.exp.workspace, "hdf5s", f"mazurka_sr{_sr(cfg)}")
    out: List[str] = []
    with open(csv_path, "r", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("split") or "").strip() == "test":
                opus, name = (r.get("opus") or "").strip(), (r.get("h5_name") or "").strip()
                if opus and name: out.append(os.path.join(base, opus, name))
    return out

def _fullsong_gt_rolls(hf: h5py.File, fps: int, dyn_class: int | None) -> Dict[str, np.ndarray]:
    duration = float(hf.attrs.get("duration_librosa", 0.0))
    key = f"dynmark_{int(dyn_class)}_class"
    T = int(np.round(duration * fps)) + 1  # match inference framing
    out: Dict[str, np.ndarray] = {}
    out["beat_roll"] = time_to_frame_roll(hf["beat_time"][:], start_time=0.0, frames_per_second=fps, frames_num=T)
    out["dynamic_roll"] = time_to_dynamic_roll(hf["beat_time"][:], hf[key][:], start_time=0.0, frames_per_second=fps, frames_num=T, duration=duration)
    out["downbeat_roll"] = time_to_frame_roll(hf["downbeat_time"][:], start_time=0.0, frames_per_second=fps, frames_num=T)
    out["change_point_roll"] = time_to_frame_roll(hf["change_point_time"][:], start_time=0.0, frames_per_second=fps, frames_num=T)
    return out

# ---------------- Evaluate ----------------
def evaluate(ckpt: str, csv_path: str, overrides: Dict | None, output_root: str | None, advanced_only: bool = False) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model, cfg = load_model(ckpt, device, overrides)
    fps, dyn_class = _fps(cfg), _dyn_classes(cfg)
    post_proc = Postprocessor(type="minimal", fps=fps)
    gt_paths = _collect_groundtruth(cfg, csv_path)
    pred_root = _ckpt_pred_root(ckpt, output_root)

    # Baseline accumulators
    dyn_true: List[torch.Tensor] = []; dyn_pred: List[torch.Tensor] = []
    beat_true: List[np.ndarray] = [];  beat_pred_times: List[List[float]] = []
    down_true: List[np.ndarray] = [];  down_pred_times: List[List[float]] = []
    cp_true: List[np.ndarray] = [];    cp_pred_logits: List[np.ndarray] = []
    beat_roll_for_cp: List[np.ndarray] = []

    # Advanced accumulators (change-point picked dynamics on beat grid)
    adv_y_true_all: List[int] = []
    adv_y_pred_all: List[int] = []
    adv_gt_count_total = 0
    adv_pred_count_total = 0

    for gt_path in gt_paths:
        pred_path = _pred_path(pred_root, gt_path)
        with h5py.File(gt_path, "r") as gt_h5, h5py.File(pred_path, "r") as pred_h5:
            gt_full = _fullsong_gt_rolls(gt_h5, fps, dyn_class)

            # Dynamics: sample classes on GT beat grid
            if (not advanced_only) and ("dynamic_output" in pred_h5):
                gt_dyn_roll = gt_full["dynamic_roll"]
                T = gt_dyn_roll.shape[0]
                y_pred = torch.from_numpy(pad_or_truncate_np(pred_h5["dynamic_output"][:], T, 0.0, 0)).argmax(-1)
                y_true = torch.from_numpy(gt_dyn_roll)
                mask = torch.from_numpy((gt_full["beat_roll"] > 0).astype(np.bool_))
                dyn_true.append(y_true[mask])
                dyn_pred.append(y_pred[mask])

            # Beat
            if (not advanced_only) and ("beat_output" in pred_h5):
                gt_beat_roll = gt_full["beat_roll"]
                T = gt_beat_roll.shape[0]
                beat_logits = _pad_to_tensor(pred_h5, "beat_output", T)
                down_logits = _pad_to_tensor(pred_h5, "downbeat_output", T) if "downbeat_output" in pred_h5 else torch.zeros_like(beat_logits)
                beat_times, _ = post_proc(beat_logits, down_logits, torch.ones_like(beat_logits, dtype=torch.bool))
                beat_true.append(gt_beat_roll[None, :])
                beat_pred_times.append(list(map(float, np.asarray(beat_times).ravel())))

            # Downbeat
            if (not advanced_only) and ("downbeat_output" in pred_h5):
                gt_down_roll = gt_full["downbeat_roll"]
                T = gt_down_roll.shape[0]
                down_logits = _pad_to_tensor(pred_h5, "downbeat_output", T)
                beat_logits = _pad_to_tensor(pred_h5, "beat_output", T) if "beat_output" in pred_h5 else torch.zeros_like(down_logits)              
                _, down_times = post_proc(beat_logits, down_logits, torch.ones_like(beat_logits, dtype=torch.bool))
                down_true.append(gt_down_roll[None, :])
                down_pred_times.append(list(map(float, np.asarray(down_times).ravel())))

            # Change-point
            if (not advanced_only) and ("change_point_output" in pred_h5):
                gt_cp_roll = gt_full["change_point_roll"]
                gt_beat_roll = gt_full["beat_roll"]
                T = gt_beat_roll.shape[0]
                cp_logits = pad_or_truncate_np(pred_h5["change_point_output"][:].squeeze(), T, pad_value=0.0, axis=0)
                cp_true.append(gt_cp_roll[None, :])
                cp_pred_logits.append(cp_logits[None, :])
                beat_roll_for_cp.append(gt_beat_roll[None, :])

            # Advanced: dynamics at predicted CP beats
            if advanced_only:
                if ("pred_change_point_beat_id" in pred_h5) and ("pred_dynamic_beat_class" in pred_h5):
                    pred_cp_ids = np.asarray(pred_h5["pred_change_point_beat_id"][:]).astype(int).ravel()
                    pred_dyn_at_beats = np.asarray(pred_h5["pred_dynamic_beat_class"][:]).astype(int).ravel()
                    adv_pred_count_total += int(pred_cp_ids.size)
                else:
                    pred_cp_ids = np.array([], dtype=int)
                    pred_dyn_at_beats = np.array([], dtype=int)

                # Ground-truth beat-grid outputs
                bt_times = np.asarray(gt_h5["beat_time"][:]).ravel() if "beat_time" in gt_h5 else None
                cp_times = np.asarray(gt_h5["change_point_time"][:]).ravel() if "change_point_time" in gt_h5 else None
                key = f"dynmark_{int(dyn_class)}_class"
                gt_dyn_per_beat = np.asarray(gt_h5[key][:]).ravel().astype(int) if (key in gt_h5) else None

                if bt_times is not None and cp_times is not None and gt_dyn_per_beat is not None and bt_times.size > 0:
                    # Map GT change-point times to beat ids (nearest index)
                    bt_idx = np.arange(bt_times.size, dtype=int)
                    # For each cp time, find nearest beat time index
                    cp_ids_gt = np.array([int(np.argmin(np.abs(bt_times - t))) for t in cp_times], dtype=int)
                    adv_gt_count_total += int(cp_ids_gt.size)
                    # Build dict for fast lookup
                    gt_dyn_dict = {int(i): int(gt_dyn_per_beat[int(i)]) for i in bt_idx if i < gt_dyn_per_beat.size}
                    # Compare where beat ids intersect
                    if pred_cp_ids.size > 0:
                        common_ids = np.intersect1d(pred_cp_ids, cp_ids_gt)
                        if common_ids.size > 0:
                            # y_true = GT dynamic at those beats; y_pred = Pred dynamic at those beats
                            adv_y_true_all.extend([gt_dyn_dict[int(bi)] for bi in common_ids if int(bi) in gt_dyn_dict])
                            adv_y_pred_all.extend([int(pred_dyn_at_beats[int(bi)]) if int(bi) < pred_dyn_at_beats.size else 0 for bi in common_ids])

    # Advanced-only path
    if advanced_only:
        stats: Dict[str, float] = {}
        if adv_y_true_all and adv_y_pred_all:
            y_true_adv = np.array(adv_y_true_all, dtype=int)
            y_pred_adv = np.array(adv_y_pred_all, dtype=int)
            stats["cp_dyn_macro_f1"] = round(float(f1_score(y_true_adv, y_pred_adv, average="macro")), 4)
            stats["cp_dyn_acc"] = round(float(accuracy_score(y_true_adv, y_pred_adv)), 4)
        stats["cp_dyn_gt_count"] = int(adv_gt_count_total)
        stats["cp_dyn_pred_count"] = int(adv_pred_count_total)
        return stats

    # Baseline metrics
    dyn_f1 = None; cp_f1 = None; beat_f1 = None; down_f1 = None
    if dyn_true and dyn_pred:
        y_true = torch.cat(dyn_true); y_pred = torch.cat(dyn_pred); mask = torch.ones_like(y_true, dtype=torch.bool)
        _, f1w = cal_dynamic_cci_and_weighted_f1(y_true, y_pred, mask)
        dyn_f1 = round(float(f1w), 4)
    if beat_true and beat_pred_times:
        max_t = max(a.shape[1] for a in beat_true)
        true_arr = np.concatenate([np.pad(a, ((0,0),(0,max_t-a.shape[1])), "constant") for a in beat_true], 0)
        beat_f1 = round(float(cal_beat_f1_beatthis(true_arr, beat_pred_times, fps)), 4)
    if down_true and down_pred_times:
        max_t = max(a.shape[1] for a in down_true)
        true_arr = np.concatenate([np.pad(a, ((0,0),(0,max_t-a.shape[1])), "constant") for a in down_true], 0)
        down_f1 = round(float(cal_beat_f1_beatthis(true_arr, down_pred_times, fps)), 4)
    if cp_true and cp_pred_logits and beat_roll_for_cp:
        max_t = max(a.shape[1] for a in cp_true)
        true_arr = np.concatenate([np.pad(a, ((0,0),(0,max_t-a.shape[1])), "constant") for a in cp_true], 0)
        pred_arr = np.concatenate([np.pad(a, ((0,0),(0,max_t-a.shape[1])), "constant") for a in cp_pred_logits], 0)
        beat_arr = np.concatenate([np.pad(a, ((0,0),(0,max_t-a.shape[1])), "constant") for a in beat_roll_for_cp], 0)
        f1, _ = cal_change_point_f1(true_arr, pred_arr, beat_arr, 0.0)
        cp_f1 = round(float(f1), 4)
    stats: Dict[str, float] = {}
    if dyn_f1 is not None:  stats["dynamic_f1"] = dyn_f1
    if cp_f1 is not None:   stats["change_point_f1"] = cp_f1
    if beat_f1 is not None: stats["beat_f1"] = beat_f1
    if down_f1 is not None: stats["downbeat_f1"] = down_f1
    return stats

def main() -> None:
    parser = argparse.ArgumentParser(description="Final evaluation over the test split.")
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--ckpt", type=str, help="Path to a single checkpoint")
    ckpt_group.add_argument("--ckpt_dir", type=str, help="Directory of checkpoints (used with --valf1_rank)")
    parser.add_argument("--valf1_rank", type=int, default=1, help="Checkpoint rank by val F1 (1=best)")
    parser.add_argument("--min_epoch", type=int, default=None, help="Ignore checkpoints with epoch < min_epoch")
    parser.add_argument("--infer_split_csv", type=str, required=True, help="CSV; only rows with split=='test' are used")
    parser.add_argument("--output_dir", type=str, default=None, help="Root where predictions were written")
    parser.add_argument("--overrides", type=str, default=None, help="Hydra overrides: k=v,k2=v2")
    parser.add_argument("--out_json", type=str, default=None, help="Optional path to write metrics JSON")
    parser.add_argument("--advanced", action="store_true", help="Run only advanced evaluation (faster)")

    args = parser.parse_args()
    overrides = parse_overrides_str(args.overrides)
    ckpt = args.ckpt or select_checkpoint(args.ckpt_dir, args.valf1_rank, args.min_epoch)

    metrics = evaluate(ckpt, args.infer_split_csv, overrides, args.output_dir, advanced_only=args.advanced)
    print(f"{os.path.basename(ckpt)} -> {metrics}")
    if args.out_json:
        with open(args.out_json, "w") as f: json.dump({os.path.basename(ckpt): metrics}, f, indent=2)

if __name__ == "__main__":
    main()

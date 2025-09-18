"""
Purpose
- Provide fast, lightweight performance observation in training (use ~20 batches).
- Support functions to the full-corpus final_evaluation.py

Dynamic marks (multi-class)
  1. Ref: K. Kosta et al., JMM 2016 (CCI% and class-weighted F1)
      Mask: Use `beat_roll` to perform fixed-interval (event-wise) evaluation.
      - dynamic_f1 : weighted F1 over all dynamic classes.
      x dynamic_cci: classification accuracy (CCI%, equivalent to Acc).

  2. Ref: J. Narang et al., ISMIR 2024 (Acc w. tolerance)
      Mask: Use `beat_roll` to evaluate on valid dynamic event locations.
      - dynamic_acc : exact match (pred == true)
      x dynamic_acc1, acc2 : ±1 or ±2 class tolerance (Mazurka only has 5 classes; not meaningful)

Beat
  1. Ref: mir_eval "MIREX 2016" & madmom pip package
      We reimplement CMLt/AMLt to avoid mir_eval slowness in training; only use mir_eval in final testing.
      Activations ≥ threshold are peak candidates; peaks are greedily selected (≥100ms separation), then matched to ground-truth beats.
      - beat_f1: beat-wise F1 with ±70 ms tolerance.
      - beat_cmlt/amlt: mir_eval-inspired metrics for strict and tolerant tempo match.

Change point
  1. Ref: K. Kosta, PhD Thesis 2017
      - change_point_f1: beat-wise F1 with exact match (no tolerance).
      Peaks selected from activations and snapped to nearest beat positions.

Auto-scan (optional)
  1. Implement for detecting the change point from model predicted activations.
      - Auto-scan threshold optionally enabled to find best change_point F1.
"""
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from utils import move_data_to_device, append_to_dict, prepare_batch_input, pad_or_truncate_time
from post_processor.beat_postproc import Postprocessor
from post_processor.dynamic_postproc import detect_change_point, auto_scan_threshold


def cal_dynamic_tolerance_accuracy(y_true, y_pred, mask):
    """
    Ref: J. Narang et al., ISMIR 2024 (Acc w. tolerance).
    Compute 1) exact, 2) ±1-class, and 3) ±2-class tolerance Accuracies.
    """
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    diff = (y_pred.int() - y_true.int()).abs()
    total = max(1, y_true.numel())
    acc = (y_pred == y_true).sum().item() / total
    acc1 = (diff <= 1).sum().item() / total
    acc2 = (diff <= 2).sum().item() / total
    return acc, acc1, acc2


def cal_dynamic_cci_and_weighted_f1(y_true, y_pred, mask):
    """
    Ref: K. Kosta et al., JMM 2016 (CCI% and class-weighted F1)
    Compute 1) correctly classified instances over total (CCI% = exact Acc), 
        and 2) class-weighted F1 for dynamic multiclass labels.
    """
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    cci = float(accuracy_score(y_true_np, y_pred_np))
    f1w = float(f1_score(y_true_np, y_pred_np, average="weighted"))
    return cci, f1w


def cal_beat_f1_beatthis(y_true, y_pred_peaktimelist, fps):
    """
    Ref: Beat This! ISMIR 2024
    Compute beat or downbeat F1 score.
    Using BeatThis proposed post-processor to select peak times from beat activations.
    """
    TOL_MS = 70
    tol_frames = max(1, int(round(TOL_MS * 1e-3 * fps)))
    TP = FP = FN = 0
    num_segments = y_true.shape[0]
    for i in range(num_segments):
        true_roll = y_true[i]
        true_peaks = np.flatnonzero(true_roll > 0)
        pred_times = y_pred_peaktimelist[i]
        pred_peaks = np.round(np.asarray(pred_times) * fps).astype(int)
        # match with ± tolerance
        if len(true_peaks) == 0 and len(pred_peaks) == 0:
            continue
        if len(pred_peaks) == 0:
            FN += len(true_peaks)
            continue
        used = np.zeros(len(pred_peaks), dtype=bool)
        tp_seg = 0
        for t in true_peaks:
            cand = np.where(~used & (np.abs(pred_peaks - t) <= tol_frames))[0]
            if cand.size > 0:
                tp_seg += 1
                used[cand[0]] = True
        TP += tp_seg
        FP += int(np.sum(~used))
        FN += int(len(true_peaks) - tp_seg)
    # precision / recall / F1
    precision = TP / (TP + FP + 1e-12)
    recall    = TP / (TP + FN + 1e-12)
    f1_score  = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return float(f1_score)


def cal_change_point_f1(y_true, y_pred, beat_arr, threshold):
    """
    Ref: K. Kosta PhD Thesis 2017
    Evaluate change point prediction using beat-wise F1 (i.e., the dynamic changes in n-th beat).
    Returns (f1_score, pred_peaks)
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    beat_arr = np.ravel(beat_arr)

    # 1. Extract predicted change points mapped to nearest beats
    pred_peaks = detect_change_point(y_pred, beat_arr, threshold)
    true_peaks = np.flatnonzero(y_true == 1)

    # 2. Beat-wise F1 computation
    if len(true_peaks) == 0 and len(pred_peaks) == 0:
        f1_score = 1.0
    elif len(true_peaks) == 0 or len(pred_peaks) == 0:
        f1_score = 0.0
    else:
        true_positives = 0
        used_pred = np.zeros(len(pred_peaks), dtype=bool)
        for t in true_peaks:
            candidates = np.where(~used_pred & (pred_peaks == t))[0]
            if candidates.size > 0:
                true_positives += 1
                used_pred[candidates[0]] = True
        false_positives = np.sum(~used_pred)
        false_negatives = len(true_peaks) - true_positives
        precision = true_positives / (true_positives + false_positives + 1e-12)
        recall = true_positives / (true_positives + false_negatives + 1e-12)
        f1_score = 2 * precision * recall / (precision + recall + 1e-12)

    return f1_score, pred_peaks


class SegmentEvaluator(object):
    def __init__(self, cfg, model):
        """Evaluate segment-wise metrics."""
        self.cfg             = cfg
        self.model           = model
        self.input_type      = cfg.exp.input_type
        self.num_dynamic_classes = cfg.feature.dynamic_classes  # L (excluding blank)
        self.targets         = cfg.exp.targets
        self.fps             = cfg.feature.frames_per_second
        self.auto_scan       = cfg.exp.auto_scan_threshold

    def evaluate(self, dataloader, show_desc: bool = False):
        """
        Evaluate model predictions against ground-truth over multiple segments.
        """
        statistics = {}
        inference_dict = run_model_inference(
            self.model, self.input_type, dataloader, return_groundtruth=True, show_progress=bool(show_desc))
        fixed_thresholds = {"change_point": 0.0}  # Our best found thresholds for change_point detection
        NUM_SCAN = 4 # Auto-scan range from the min to max of predicted activations / divided by this number

        for target in self.targets:
            true_key = f"{target}_roll"
            pred_key = f"{target}_output"

            if pred_key not in inference_dict:
                raise KeyError(f"Missing prediction: '{pred_key}' in output_dict. Check if the model is configured to predict this target.")

            # Pad/Truncate prediction time dimension to match ground-truth.
            pred_np = inference_dict[pred_key]
            T = inference_dict[true_key].shape[1]
            if pred_np.shape[1] != T:
                pred_t = torch.from_numpy(pred_np)
                pred_t = pad_or_truncate_time(pred_t, T, pad_value=0)
                inference_dict[pred_key] = pred_t.numpy()

            if target == "dynamic":
                true_arr = torch.from_numpy(inference_dict[true_key])
                pred_arr = torch.from_numpy(inference_dict[pred_key]).argmax(dim=-1)
                interval_mask = torch.from_numpy(inference_dict["beat_roll"]).bool()
                cci, f1w = cal_dynamic_cci_and_weighted_f1(true_arr, pred_arr, interval_mask) # 5-class avg-weighted F1 + (CCI% is same as exact Acc)
                acc, acc1, acc2 = cal_dynamic_tolerance_accuracy(true_arr, pred_arr, interval_mask) # exact Acc + (1~2 class tolerance Acc not useful)
                statistics["dynamic_f1"] = round(float(f1w), 4)
                statistics["dynamic_acc"]  = round(acc, 4) 

            elif target == "beat":
                true_arr = inference_dict[true_key] 
                beat_logits_np = inference_dict[pred_key].squeeze(-1)
                beat_logits = torch.from_numpy(beat_logits_np).float().cpu()
                # Use Beat This! postprocessor (minimal): logits -> sigmoid -> local max ±3 -> prob>0.5
                if 'downbeat_output' in inference_dict:
                    down_logits = torch.from_numpy(inference_dict['downbeat_output'].squeeze(-1)).float().cpu()
                else:
                    down_logits = torch.zeros_like(beat_logits)
                pad_mask = torch.ones_like(beat_logits, dtype=torch.bool)  # no padding known -> all valid
                pp = Postprocessor(type="minimal", fps=self.fps)
                postp_beat_times, _ = pp(beat_logits, down_logits, pad_mask)  # list/tuple of np arrays (seconds) per segment
                f1 = cal_beat_f1_beatthis(true_arr, postp_beat_times, self.fps)
                statistics["beat_f1"] = round(float(f1), 4)

            elif target == "downbeat":
                # Evaluate downbeat with the same BeatThis postprocessor
                # If beat logits are missing (e.g., SingleCNN with only 'downbeat' head),
                # fall back to zeros for beat logits so postprocessor still works.
                true_arr = inference_dict[true_key]
                downbeat_logits_np = inference_dict[pred_key].squeeze(-1)
                down_logits = torch.from_numpy(downbeat_logits_np).float().cpu()
                if 'beat_output' in inference_dict:
                    beat_logits = torch.from_numpy(inference_dict['beat_output'].squeeze(-1)).float().cpu()
                else:
                    beat_logits = torch.zeros_like(down_logits)
                pad_mask = torch.ones_like(beat_logits, dtype=torch.bool)
                pp = Postprocessor(type="minimal", fps=self.fps)
                _, postp_down_times = pp(beat_logits, down_logits, pad_mask)
                f1 = cal_beat_f1_beatthis(true_arr, postp_down_times, self.fps)
                statistics["downbeat_f1"] = round(float(f1), 4)

            elif target == "change_point":
                true_arr = inference_dict[true_key]
                pred_arr = inference_dict[pred_key].squeeze(-1)
                beat_arr = inference_dict["beat_roll"]
                if self.auto_scan:  # Auto-scan using data-driven threshold range [min(pred), max(pred)]
                    def _cp_cal(true_a, pred_a, thr):
                        return cal_change_point_f1(true_a, pred_a, beat_arr, thr)[0]
                    threshold = auto_scan_threshold(true_arr, pred_arr, _cp_cal, NUM_SCAN)
                else:
                    threshold = fixed_thresholds.get(target, 4.0)
                f1, pred_peaks = cal_change_point_f1(true_arr, pred_arr, beat_arr, threshold)
                statistics["change_point_f1"] = round(float(f1), 4)
                # statistics["change_point_thr"] = round(float(threshold), 2)  # (disabled) do not record threshold
                statistics["change_point_num"] = len(pred_peaks)

            else:
                raise ValueError(f"Not yet implemented target: '{target}'")        

        return statistics
    

def run_model_inference(model, input_type, dataloader, return_groundtruth=True, show_progress: bool = False):
    """Forward data generated from dataloader to model.
    Args: model: object
          dataloader: object, used to generate mini-batches for evaluation.
          return_groundtruth: bool
    Keys in batch_data_dict [
      audio_input / midi_input 
      beat_roll downbeat_roll measure_roll  change_point_roll dynamic_roll
      ]
    Returns:
      output_dict: dict, e.g. {
        'beat_output': (segments_num, frames_num, classes_num),
        'dynamic_output': (segments_num, frames_num, classes_num),
        'beat_roll': (segments_num, frames_num, classes_num),
        'dynamic_roll': (segments_num, frames_num, classes_num),...}
    """
    inference_dict = {}
    device = next(model.parameters()).device
    # Try to estimate total batches for tqdm
    total_batches = None
    bs = getattr(dataloader, 'batch_sampler', None)
    if bs is not None and hasattr(bs, 'max_evaluate_batches'):
        try:
            total_batches = int(bs.max_evaluate_batches)
        except Exception:
            total_batches = None
    # progress bar over dataloader, only if requested
    if show_progress:
        iterator = tqdm(dataloader, total=total_batches, desc="Eval", ncols=80)
    else:
        iterator = dataloader
    for n, batch_data_dict in enumerate(iterator):
        batch_input_dict = [move_data_to_device(inp, device) for inp in prepare_batch_input(batch_data_dict, input_type)]

        with torch.no_grad():
            model.eval()
            target_len = batch_data_dict["dynamic_roll"].shape[1] if "dynamic_roll" in batch_data_dict else None
            batch_output_dict = model(*batch_input_dict, target_len=target_len)

        for key in batch_output_dict.keys():
            append_to_dict(inference_dict, key, batch_output_dict[key].detach().cpu().numpy())

        if return_groundtruth:
            for target_type in batch_data_dict.keys():
                if 'roll' in target_type:
                    append_to_dict(inference_dict, target_type, batch_data_dict[target_type])

    for key in inference_dict.keys():
        inference_dict[key] = np.concatenate(inference_dict[key], axis=0)
    
    return inference_dict

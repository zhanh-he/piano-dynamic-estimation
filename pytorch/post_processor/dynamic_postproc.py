"""
Change-point postprocessing utilities.

This module contains helper functions used for change point detection and
auto-scanning thresholds to maximize F1.
"""
from __future__ import annotations
import logging
import os
from typing import Callable, Tuple
import numpy as np

def eval_threshold_args(args: Tuple[float, np.ndarray, np.ndarray, Callable[[np.ndarray, np.ndarray, float], float]]):
    """
    Helper to evaluate F1 at a given threshold. Used by auto_scan_threshold.

    Args is a 4-tuple: (threshold, true_arr, pred_arr, cal_f1_func)
    Returns a 2-tuple: (threshold, f1)
    """
    thr, true_arr, pred_arr, cal_f1_func = args
    return thr, cal_f1_func(true_arr, pred_arr, thr)


def auto_scan_threshold(true_arr: np.ndarray,
                        pred_arr: np.ndarray,
                        cal_f1_func: Callable[[np.ndarray, np.ndarray, float], float],
                        scan_steps: int) -> float:
    """
    Automatic scan to find the best threshold that maximizes F1 score.
    Threshold helps convert continuous activations into binary events.
    """
    min_pred = np.min(pred_arr)
    max_pred = np.max(pred_arr)
    if not np.isfinite(min_pred) or not np.isfinite(max_pred) or max_pred <= min_pred:
        logging.warning("Invalid prediction values for threshold scan. Returning 0.0")
        return 0.0
    thresholds = np.linspace(min_pred, max_pred, scan_steps)
    args_list = [(thr, true_arr, pred_arr, cal_f1_func) for thr in thresholds]

    import concurrent.futures
    max_workers = min(int(scan_steps), (os.cpu_count() or 1))
    logging.info(
        f"Starting threshold scan with {scan_steps} steps using ThreadPoolExecutor (workers={max_workers})."
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(eval_threshold_args, args_list))

    best_threshold, best_f1 = max(results, key=lambda x: x[1])
    logging.info(f"Auto-scan complete. Best threshold: {best_threshold:.4f}, Best F1: {best_f1:.4f}")
    return float(best_threshold)


def detect_change_point(pred_arr: np.ndarray, beat_arr: np.ndarray, threshold: float) -> np.ndarray:
    """
    Map predicted change_point activations to nearest beat positions above threshold.
    Returns an array of beat indices for predicted change points.
    """
    cp_indices = np.flatnonzero(pred_arr >= threshold)
    beat_indices = np.flatnonzero(beat_arr == 1)
    assigned = set()
    matched_indices = []
    for cp in cp_indices:
        if beat_indices.size == 0:
            continue
        nearest_beat = beat_indices[np.argmin(np.abs(beat_indices - cp))]
        if nearest_beat not in assigned:
            matched_indices.append(nearest_beat)
            assigned.add(nearest_beat)
    return np.array(matched_indices)
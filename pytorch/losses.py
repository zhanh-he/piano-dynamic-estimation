import torch
import torch.nn.functional as F
from utils import pad_or_truncate_time
"""Minimal loss utilities for dynamic and binary targets."""

# ---------------- Basic losses ----------------
def ce(output, target, mask=None):
    """Cross-entropy over time: output (B,T,C), target (B,T)."""
    # Align time length
    B, T_out, C = output.shape
    B, T = target.shape

    if T_out != T:
        output = pad_or_truncate_time(output, T, pad_value=0)
        if mask is not None:
            mask = pad_or_truncate_time(mask, T, pad_value=0)

    output_flat = output.reshape(B * T, C)
    target_flat = target.reshape(B * T).long()

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fn(output_flat, target_flat).reshape(B, T)

    if mask is not None:
        mask = mask.to(loss.device).float()
        loss = loss * mask
        denom = mask.sum()
        return loss.sum() / denom if denom.item() > 0 else loss.mean()
    else:
        return loss.mean()
    

def bce(output, target, mask=None):
    """Binary cross-entropy with logits: output (B,T,1), target (B,T)."""
    # Align time length
    if output.shape[1] != target.shape[1]:
        output = pad_or_truncate_time(output, target.shape[1], pad_value=0)
        if mask is not None and mask.dim() >= 2:
            mask = pad_or_truncate_time(mask, target.shape[1], pad_value=0)
    target = target.unsqueeze(-1).float()  # (B, T, 1)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_fn(output, target)  # (B, T, 1)
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        mask = mask.to(loss.device)
        loss = loss * mask
        return loss.sum() / mask.sum()
    else:
        return loss.mean()


def weighted_bce(output, target, pos_weight=None, mask=None):
    """BCE with optional `pos_weight` and mask."""
    # Align time length
    if output.shape[1] != target.shape[1]:
        output = pad_or_truncate_time(output, target.shape[1], pad_value=0)
        if mask is not None and mask.dim() >= 2:
            mask = pad_or_truncate_time(mask, target.shape[1], pad_value=0)
    target = target.unsqueeze(-1).float()
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], device=output.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction='none')
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_fn(output, target)
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        mask = mask.to(loss.device)
        loss = loss * mask
        return loss.sum() / mask.sum()
    else:
        return loss.mean()


def shift_tolerant_bce(output, target, tolerance: int = 3, pos_weight=None, mask=None):
    """Shift-tolerant BCE via temporal max-pooling (BeatThis style)."""
    # Align time length
    if output.shape[1] != target.shape[1]:
        output = pad_or_truncate_time(output, target.shape[1], pad_value=0)
        if mask is not None and mask.dim() >= 2:
            mask = pad_or_truncate_time(mask, target.shape[1], pad_value=0)
    B, T, _ = output.shape
    preds = output.transpose(1, 2)  # [B,1,T]
    targets = target.unsqueeze(1).float()  # [B,1,T]
    if tolerance > 0:
        k = 1 + 2 * tolerance
        preds_spread = F.max_pool1d(preds, k, stride=1)
        preds_spread = preds_spread[..., tolerance : preds_spread.shape[-1] - tolerance]
        tgt_crop = targets[..., 2 * tolerance : targets.shape[-1] - 2 * tolerance]
        if mask is not None:
            m = mask.unsqueeze(1) if mask.dim() == 2 else mask
            m_crop = m[..., 2 * tolerance : m.shape[-1] - 2 * tolerance]
        else:
            m_crop = None
        targets_spread2 = F.max_pool1d(targets, 1 + 4 * tolerance, stride=1)
        look_at = tgt_crop + (1 - targets_spread2)
        if m_crop is not None:
            look_at = look_at * m_crop
        pw = torch.tensor([pos_weight], device=preds.device) if pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(preds_spread, tgt_crop, weight=look_at, pos_weight=pw)
        return loss
    else:
        return weighted_bce(output, target, pos_weight=pos_weight, mask=mask)
    
# ---------------- Per-target helpers ----------------

def _dynamic_loss(output_dict, target_dict, base_loss, mask_key):
    true_mask = target_dict.get(f"{mask_key}_roll")
    return base_loss(
        output_dict['dynamic_output'],
        target_dict['dynamic_roll'],
        true_mask
    )


def _beat_loss(output_dict, target_dict, cfg=None, dataset_pos_weight: dict | None = None):
    target = target_dict['beat_roll']
    if dataset_pos_weight is not None and dataset_pos_weight.get('beat', 0) > 0 and cfg.exp.use_dataset_pos_weight:
        pos_weight = float(dataset_pos_weight['beat'])
    else:
        pos_count = target.sum().item()
        pos_weight = None if pos_count == 0 else (target.numel() - pos_count) / pos_count
    loss_name = getattr(cfg.exp, 'beat_loss', 'bce') if cfg is not None else 'bce'
    if loss_name == 'shift_tolerant_bce':
        tol = int(getattr(cfg.exp, 'widen_target_mask', 3)) if cfg is not None else 3
        return shift_tolerant_bce(output_dict['beat_output'], target, tolerance=tol, pos_weight=pos_weight)
    else:
        return weighted_bce(output_dict['beat_output'], target, pos_weight=pos_weight)


def _downbeat_loss(output_dict, target_dict, cfg=None, dataset_pos_weight: dict | None = None):
    target = target_dict['downbeat_roll']
    if dataset_pos_weight is not None and dataset_pos_weight.get('downbeat', 0) > 0 and cfg.exp.use_dataset_pos_weight:
        pos_weight = float(dataset_pos_weight['downbeat'])
    else:
        pos_count = target.sum().item()
        pos_weight = None if pos_count == 0 else (target.numel() - pos_count) / pos_count
    loss_name = getattr(cfg.exp, 'downbeat_loss', 'bce') if cfg is not None else 'bce'
    if loss_name == 'shift_tolerant_bce':
        tol = int(getattr(cfg.exp, 'widen_target_mask', 3)) if cfg is not None else 3
        return shift_tolerant_bce(output_dict['downbeat_output'], target, tolerance=tol, pos_weight=pos_weight)
    else:
        return weighted_bce(output_dict['downbeat_output'], target, pos_weight=pos_weight)


def _change_point_loss(output_dict, target_dict, cfg=None, dataset_pos_weight: dict | None = None):
    target = target_dict['change_point_roll']
    if dataset_pos_weight is not None and dataset_pos_weight.get('change_point', 0) > 0 and cfg.exp.use_dataset_pos_weight:
        pos_weight = float(dataset_pos_weight['change_point'])
    else:
        pos_count = target.sum().item()
        pos_weight = None if pos_count == 0 else (target.numel() - pos_count) / pos_count
    loss_name = getattr(cfg.exp, 'change_point_loss', 'bce') if cfg is not None else 'bce'
    if loss_name == 'shift_tolerant_bce':
        tol = int(getattr(cfg.exp, 'widen_target_mask', 3)) if cfg is not None else 3
        return shift_tolerant_bce(output_dict['change_point_output'], target, tolerance=tol, pos_weight=pos_weight)
    else:
        return weighted_bce(output_dict['change_point_output'], target, pos_weight=pos_weight)
    
# ---------------- Loss factory ----------------

def get_loss_func(dynamic_loss, dynamic_mask, targets, output_dict, target_dict, cfg=None, dataset_pos_weight: dict | None = None):
    """Build a dict of per-target losses plus total."""
    loss_dict = {}
    total_loss = 0.0
    
    # Dynamic branch (optional)
    if 'dynamic' in targets:
        if dynamic_loss.startswith('ce'):
            base_fn = ce
        elif dynamic_loss.startswith('mse'):
            base_fn = mse
        elif dynamic_loss.startswith('mae'):
            base_fn = mae
        else:
            raise Exception('Unknown loss type prefix for dynamic_loss')

        dynamic_loss_value = _dynamic_loss(output_dict, target_dict, base_fn, dynamic_mask)
        loss_dict["dynamic_loss"] = dynamic_loss_value
        total_loss = total_loss + dynamic_loss_value

        extra_targets = [t for t in targets if t != 'dynamic']
    else:
        extra_targets = list(targets)

    # Extra targets (binary)
    extra_target_loss_map = {
        "beat":         lambda o, t: _beat_loss(o, t, cfg, dataset_pos_weight),
        "change_point": lambda o, t: _change_point_loss(o, t, cfg, dataset_pos_weight),
        "downbeat":     lambda o, t: _downbeat_loss(o, t, cfg, dataset_pos_weight),
    }
    for t in extra_targets:
        if t not in extra_target_loss_map:
            raise Exception(f"Unknown additional loss target: {t}")
        target_loss_fn = extra_target_loss_map[t]
        target_loss_value = target_loss_fn(output_dict, target_dict)
        loss_dict[f"{t}_loss"] = target_loss_value
        total_loss = total_loss + target_loss_value

    loss_dict["total_loss"] = total_loss
    return loss_dict

import os, sys, random, gc
import torch
import time
import logging
import torch
import numpy as np
from hydra import initialize, compose
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

from utils import create_folder, create_logging, move_data_to_device, prepare_batch_input, count_params, log_gradient_norm
from models_audioCNN import SingleCNN, MultiTaskCNN
from losses import get_loss_func
from evaluator import SegmentEvaluator
from dataloader import Mazurka_Dataset, Sampler, EvalSampler, collate_fn, get_train_positive_weights


def init_seed(seed):
    gc.collect()
    torch.cuda.empty_cache()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return


def init_wandb(cfg, wandb_run_id):
    """
    Initialize WandB for experiment tracking.
    """
    # Ensure W&B monitors the same GPUs as CUDA_VISIBLE_DEVICES, if not already set by the HPC Slurm launcher
    if 'WANDB__SERVICE__GPU_MONITOR_DEVICES' not in os.environ and 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['WANDB__SERVICE__GPU_MONITOR_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ.setdefault('WANDB__SERVICE__GPU_MONITOR_POLICY', 'visible')
    print(f"[W&B] GPU monitor devices: {os.environ.get('WANDB__SERVICE__GPU_MONITOR_DEVICES', '(unset)')} | policy={os.environ.get('WANDB__SERVICE__GPU_MONITOR_POLICY')} ")
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        id=wandb_run_id,
        resume="must" if wandb_run_id else "allow",
        config=OmegaConf.to_container(cfg, resolve=True)
    )


def train(cfg):
    """
    Train a piano transcription system with epoch-based loops and progress bars.
    """
    # Arguments & parameters
    init_seed(cfg.exp.random_seed)
    device = torch.device('cuda') if cfg.exp.cuda and torch.cuda.is_available() else torch.device('cpu')
    model = eval(cfg.exp.model_name)(cfg).to(device)
    params_total, params_trainable = count_params(model)

    # Optimizer: simple flag `exp.use_adamw`, MTST uses AdamW, CNNs uses Adam
    use_adamw = bool(getattr(cfg.exp, 'use_adamw', False))
    if use_adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.exp.learning_rate, weight_decay=1e-2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.exp.learning_rate)

    # Paths for results
    checkpoints_dir = os.path.join(cfg.exp.workspace, 'checkpoints', cfg.wandb.name)
    logs_dir = os.path.join(cfg.exp.workspace, 'logs', cfg.wandb.name)

    # Create logging and checkpoint dir
    create_folder(checkpoints_dir)
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')

    # Build dataset & loaders first (needed to derive steps/epoch and optionally load sampler state)
    dataset = Mazurka_Dataset(cfg)
    train_sampler = get_sampler(cfg, purpose='train', file_list=dataset.train_list)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn, num_workers=cfg.exp.num_workers, pin_memory=True)
    eval_train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=get_sampler(cfg, purpose='eval', file_list=dataset.train_list),
        collate_fn=collate_fn, num_workers=cfg.exp.num_workers, pin_memory=True)
    eval_valid_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=get_sampler(cfg, purpose='eval', file_list=dataset.valid_list),
        collate_fn=collate_fn, num_workers=cfg.exp.num_workers, pin_memory=True)
    evaluator = SegmentEvaluator(cfg, model)

    # Dataset-level positive weights for binary heads (beat/downbeat)
    dataset_pos_weight = None
    try:
        if bool(getattr(cfg.exp, 'use_dataset_pos_weight', True)):
            dataset_pos_weight = get_train_positive_weights(dataset.train_list, fps=int(cfg.feature.frames_per_second), widen_target_mask=int(getattr(cfg.exp, 'widen_target_mask', 3)))
            logging.info(f"[Imbalance] dataset_pos_weight={dataset_pos_weight}")
    except Exception as e:
        logging.warning(f"Failed to compute dataset_pos_weight: {e}")

    # Derive epoch settings
    steps_per_epoch = max(1, len(train_sampler))
    total_epochs = int(getattr(cfg.exp, 'total_epochs', 0) or 0)
    if total_epochs <= 0:
        total_epochs = 100  # fallback if user forgot to set
        logging.warning(f"exp.total_epochs not set or <=0, default to {total_epochs}")
    eval_epoch_interval = int(getattr(cfg.exp, 'eval_epoch_interval', 1) or 1)
    lr_step_epochs = int(getattr(cfg.exp, 'reduce_epoch', 0) or 0)  # 0 disables epoch-based LR step
    # Visibility: show how steps/epoch is derived
    try:
        logging.info(
            f"[Data] train_files={len(dataset.train_list)} | segments={len(train_sampler.segment_list)} | "
            f"batch_size={cfg.exp.batch_size} | steps_per_epoch={steps_per_epoch}")
    except Exception:
        pass

    # Resume training if applicable (epoch only)
    start_epoch = int(getattr(cfg.exp, 'resume_epoch', 0) or 0)
    start_global_step = 0
    sampler_state_to_load = None
    wandb_run_id = None
    if start_epoch > 0:
        # Support both plain and score-suffixed filenames: epoch_{N}.pth or epoch_{N}_valf1_xxxxx.pth
        ckpt_e = os.path.join(checkpoints_dir, f'epoch_{start_epoch}.pth')
        if not os.path.exists(ckpt_e):
            try:
                candidates = []
                for f in os.listdir(checkpoints_dir):
                    if f.startswith(f'epoch_{start_epoch}') and f.endswith('.pth'):
                        candidates.append(os.path.join(checkpoints_dir, f))
                if candidates:
                    # Choose most recent by mtime
                    ckpt_e = max(candidates, key=lambda p: os.path.getmtime(p))
            except Exception:
                pass
        if os.path.exists(ckpt_e):
            logging.info(f"Loading checkpoint {ckpt_e} (epoch {start_epoch})")
            checkpoint = torch.load(ckpt_e)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_global_step = int(checkpoint.get('global_step', (start_epoch - 1) * steps_per_epoch))
            sampler_state_to_load = checkpoint.get('sampler', None)
            wandb_run_id = checkpoint.get('wandb_run_id')
        else:
            logging.warning(f"Epoch checkpoint {ckpt_e} not found. Starting fresh from epoch 0.")
            start_epoch = 0

    if sampler_state_to_load is not None:
        try:
            train_sampler.load_state_dict(sampler_state_to_load)
        except Exception as e:
            logging.warning(f"Failed to load sampler state: {e}")

    # Initialize WandB after potentially loading run_id
    init_wandb(cfg, wandb_run_id)
    wandb.summary['params_total'] = int(params_total)
    wandb.summary['params_trainable'] = int(params_trainable)

    # Device / GPU summary
    logging.info(cfg)
    if device.type == "cuda":
        gpu_count = torch.cuda.device_count()
        cur_idx = torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(cur_idx)
        logging.info(f"CUDA enabled: {gpu_count} GPU(s) visible; using cuda:{cur_idx} ({dev_name}).")
    else:
        logging.info("CUDA disabled or unavailable; using CPU.")
    model.to(device)
    logging.info(f"[PARAMS] total={(params_total/1e3):.2f}K | trainable={(params_trainable/1e3):.2f}K")

    # Reset PyTorch peak memory stats so subsequent peaks are for this run
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Initialize training loop variables
    global_step = start_global_step
    loss_history = []
    train_bgn_time = time.time()
    log_buffer = [] # wandb log buffer for loss (debug purpose)
    
    # Epoch-based training loop with per-epoch progress bar
    train_iter = iter(train_loader)
    for epoch in range(start_epoch + 1, total_epochs + 1):
        if cfg.exp.decay and lr_step_epochs > 0:
            if (epoch - 1) > 0 and (epoch - 1) % lr_step_epochs == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{total_epochs}", ncols=80)
        for _ in pbar:
            batch_data_dict = next(train_iter)

            # Forward & Backward
            model.train()
            batch_output_dict, loss_dict = forward_pass(cfg, model, batch_data_dict, device, dataset_pos_weight)
            loss = loss_dict["total_loss"]
            loss_value = loss.item()

            # Accumulate loss history (for simple spike checks)
            loss_history.append((global_step, loss_value))
            loss.backward()
            gradient = log_gradient_norm(model, global_step)
            optimizer.step()

            # Per-step: log only losses to W&B (no gradient/GPU fields)
            try:
                log_iter = {k: float(v.item()) for k, v in loss_dict.items()}
                wandb.log(log_iter, step=global_step)
            except Exception:
                pass

            # Optional periodic logging to stdout/file, disabled if log_interval_steps==0 (Debug purpose)
            log_every = int(getattr(cfg.exp, 'log_interval_steps', 0) or 0)
            if log_every > 0 and (global_step % log_every) == 0:
                logging.info(f"[Epoch {epoch} | Step {global_step}] loss={loss_value:.7f} | GradNorm={gradient:.7f}")
                try:
                    with open(os.path.join(cfg.exp.workspace, 'logs', cfg.wandb.name, 'train_history.txt'), 'a') as f:
                        f.write(f"{global_step},{loss_value:.7f},{gradient:.7f}\n")
                except Exception:
                    pass

            # Update progress bar postfix
            try:
                pbar.set_postfix({"loss": f"{loss_value:.5f}"})
            except Exception:
                pass

            # Next step
            optimizer.zero_grad()
            global_step += 1

        # Early-eval scheduling (simple):
        early_until = eval_epoch_interval
        early_int = max(1, eval_epoch_interval // 4)
        def _should_eval(ep: int) -> bool:
            if ep <= early_until:
                return (ep % early_int) == 0
            return (ep % eval_epoch_interval) == 0

        if _should_eval(epoch):
            logging.info('------------------------------------')
            logging.info(f"Epoch {epoch}/{total_epochs} completed; running evaluation...")
            train_fin_time = time.time()
            train_statistics = evaluator.evaluate(eval_train_loader, show_desc=True)
            valid_statistics = evaluator.evaluate(eval_valid_loader)
            logging.info(f"    Train Stat: {train_statistics}")
            logging.info(f"    Valid Stat: {valid_statistics}")

            # Compute averaged validation F1 over current targets
            def _avg_valid_f1(stats: dict, targets: list[str]) -> float:
                key_map = {
                    'dynamic': 'dynamic_f1',
                    'change_point': 'change_point_f1',
                    'beat': 'beat_f1',
                    'downbeat': 'downbeat_f1',
                }
                values = []
                for t in targets:
                    k = key_map.get(t)
                    if k and (k in stats):
                        try:
                            values.append(float(stats[k]))
                        except Exception:
                            pass
                if not values:
                    return 0.0
                return float(np.mean(values))

            avg_vf1 = _avg_valid_f1(valid_statistics, list(cfg.exp.targets))
            wandb.log({"valid_f1_avg": avg_vf1}, step=global_step)

            # Log and summarize GPU memory (allocated & peak)
            if device.type == "cuda":
                mem_now = get_gpu_mem_mb()
                logging.info(
                    f"[GPU{mem_now['gpu']}] alloc={mem_now['alloc_MiB']:.1f} MiB | peak={mem_now['peak_alloc_MiB']:.1f} MiB")
                wandb.summary["gpu/alloc_MiB"] = float(mem_now["alloc_MiB"])
                wandb.summary["gpu/peak_alloc_MiB"] = float(mem_now["peak_alloc_MiB"])

            # Upload eval stats
            wandb.log({
                "epoch": epoch,
                "train_stat": train_statistics,
                "valid_stat": valid_statistics,
            }, step=global_step)

            # Timing and reset
            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))
            train_bgn_time = time.time()

            # Save model checkpoint (only on eval epochs)
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model': model.state_dict(),
                'sampler': train_sampler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'wandb_run_id': wandb.run.id,
                'cfg': OmegaConf.to_container(cfg, resolve=True),
            }
            # Append averaged validation F1 to filename with 5 decimal places for easy selection
            checkpoint_path = os.path.join(checkpoints_dir, f"epoch_{epoch}_valf1_{avg_vf1:.5f}.pth")
            torch.save(checkpoint, checkpoint_path)
            logging.info(f'Model saved to {checkpoint_path}')
    
    # End WandB Logger
    wandb.finish()


def train_kfold(cfg):
    """Run 5-fold training sequentially (fold 0..4) when enabled.
    Creates an independent config per fold so interpolations like
    `wandb.name: ...-fold${dataset.fold_index}` resolve per fold.
    """
    if not bool(getattr(cfg.dataset, 'use_5fold', False)):
        logging.warning("dataset.run_all_folds=True but dataset.use_5fold=False; forcing use_5fold=True.")
        cfg.dataset.use_5fold = True
    base_container = OmegaConf.to_container(cfg, resolve=False)
    for fold in range(5):
        cfg_fold = OmegaConf.create(base_container)
        cfg_fold.dataset.fold_index = int(fold)
        logging.info(f"[KFold] Starting fold {fold}/4 with wandb.name={cfg_fold.wandb.name}")
        train(cfg_fold)


def forward_pass(cfg, model, batch_data_dict, device, dataset_pos_weight=None):
    """
    Return model's output and computed loss for the batch (a train step).
    Supports 'audio', 'midi', or 'both' data as input,
    Current developed are audio-only models.
    """
    for key in batch_data_dict.keys():
        batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
    batch_input_dict = prepare_batch_input(batch_data_dict, cfg.exp.input_type)
    first_target = cfg.exp.targets[0] if hasattr(cfg.exp, 'targets') else 'dynamic'
    gt_len = batch_data_dict[f"{first_target}_roll"].shape[1]
    batch_output_dict = model(*batch_input_dict, target_len=gt_len)
    loss_dict = get_loss_func(cfg.exp.dynamic_loss, cfg.exp.dynamic_mask, cfg.exp.targets, batch_output_dict, batch_data_dict, cfg=cfg, dataset_pos_weight=dataset_pos_weight)
    return batch_output_dict, loss_dict


def get_sampler(cfg, purpose, file_list):
    """
    Choose Sampler for training (infinite) or EvalSampler for evaluation (finite),
    using provided file_list from Dataset.
    """
    sampler_mapping = {
        'train': Sampler,
        'eval': EvalSampler,
    }
    return sampler_mapping[purpose](cfg, file_list)


def get_gpu_mem_mb(device_idx: int | None = None) -> dict:
    """Return PyTorch-tracked allocated and peak-allocated GPU memory in MiB.
    Uses torch.cuda.memory_allocated() and torch.cuda.max_memory_allocated().
    """
    def _mb(x: int) -> float:
        return x / (1024 ** 2)
    if device_idx is None:
        device_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
    alloc_mb = _mb(torch.cuda.memory_allocated(device_idx))
    peak_mb  = _mb(torch.cuda.max_memory_allocated(device_idx))
    return {"gpu": int(device_idx), "alloc_MiB": alloc_mb, "peak_alloc_MiB": peak_mb}


if __name__ == '__main__':
    initialize(config_path="./", job_name="train", version_base=None)
    cfg = compose(config_name="config", overrides=sys.argv[1:])
    # If requested, run 5-fold training sequentially at once; otherwise single fold (ablation study)
    try:
        run_all = bool(getattr(cfg.dataset, 'run_all_folds', False))
    except Exception:
        run_all = False
    if run_all:
        train_kfold(cfg)
    else:
        train(cfg)
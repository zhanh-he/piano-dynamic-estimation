import argparse
from pathlib import Path
import re, sys, io, logging, warnings, os
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything

from pl_module import PLBeatThis
from postprocessor import Postprocessor
from h5_dataset import H5BeatDataModule


def _tee_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    class _Tee(io.TextIOBase):
        def __init__(self, orig, file_path):
            self.orig = orig
            self.f = open(file_path, 'a', buffering=1)
        def write(self, s):
            try:
                self.f.write(s)
            except Exception:
                pass
            return self.orig.write(s)
        def flush(self):
            try:
                self.f.flush()
            except Exception:
                pass
            return self.orig.flush()
    sys.stdout = _Tee(sys.stdout, log_path)
    sys.stderr = _Tee(sys.stderr, log_path)


def _select_topk_ckpts(ckpt_dir: Path, topk: int = 3) -> list[Path]:
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return []
    def parse_valloss(p: Path):
        m = re.search(r"-valloss=?([0-9]+(?:\.[0-9]+)?)", p.name)
        return float(m.group(1)) if m else float('inf')
    with_scores = [(parse_valloss(p), p) for p in ckpts]
    with_scores.sort(key=lambda x: (x[0], x[1].name))
    # if none had valloss, fall back to mtime (newest first)
    if all(np.isinf(s) for s, _ in with_scores):
        ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return ckpts[:topk]
    return [p for s, p in with_scores[:topk]]


def _load_pl_from_ckpt(ckpt_path: Path, device: str = "auto", *, quiet: bool = False, force_dbn: bool | None = None) -> tuple[PLBeatThis, Trainer, dict]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = PLBeatThis(**checkpoint["hyper_parameters"])
    # Optionally use DBN as postprocessor
    if force_dbn is not None:
        model.postprocessor = Postprocessor(type="dbn" if force_dbn else "minimal", fps=model.hparams.fps)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    if device == "cpu":
        accelerator, devices, precision = ("cpu", 1, 32)
    else:
        accelerator, devices, precision = ("gpu", [0], "16-mixed")
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,
        enable_progress_bar=not quiet,
        enable_model_summary=not quiet,
    )
    return model, trainer, checkpoint


def _build_loader(checkpoint: dict, datasplit: str, *, h5_root: str, csv_split: str, fold: int | None, num_workers: int, batch_size: int, cache_dir: str | None):
    if h5_root is None or csv_split is None:
        raise ValueError("Must provide --h5-root and --csv-split for building dataloaders")
    dm_hp = dict(checkpoint.get("datamodule_hyper_parameters", {}))
    sr = int(dm_hp.get("sr", 22050))
    fps = int(dm_hp.get("fps", 50))
    cache_dir = cache_dir or dm_hp.get("cache_dir", "data/_h5_mel_cache")
    dm = H5BeatDataModule(
        h5_root=h5_root,
        batch_size=batch_size,
        num_workers=num_workers,
        sr=sr,
        fps=fps,
        train_length=None,
        val_ratio=float(dm_hp.get("val_ratio", 0.1)),
        seed=int(dm_hp.get("seed", 0)),
        cache_dir=cache_dir,
        csv_split=csv_split,
        fold=fold,
    )
    dm.setup(datasplit)
    if datasplit == "train":
        return dm.train_dataloader()
    elif datasplit in ("val", "valid"):
        return dm.val_dataloader()
    else:
        return dm.test_dataloader()


def _predict_and_aggregate(model: PLBeatThis, trainer: Trainer, dataloader):
    outs = trainer.predict(model, dataloader)
    metrics_list = [o[0] for o in outs]
    datasets = np.asarray([o[2][0] for o in outs])
    pieces = np.asarray([o[3][0] for o in outs])
    keys = list(metrics_list[0].keys()) if metrics_list else []
    metrics = {k: np.asarray([m[k] for m in metrics_list]) for k in keys}
    avg = {k: float(np.mean(v)) for k, v in metrics.items()}
    return metrics, avg, datasets, pieces

def _count_params(module: torch.nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def main():
    ap = argparse.ArgumentParser(description="Evaluate BeatThis checkpoints on H5 dataset")
    # Required runtime dataset parameters
    ap.add_argument("--h5-root", required=True, type=str)
    ap.add_argument("--csv-split", required=True, type=str)
    ap.add_argument("--fold", type=int, default=0)
    # Checkpoint selection
    ap.add_argument("--ckpt", type=str, nargs="*", default=None, help="Explicit checkpoint paths (.ckpt)")
    ap.add_argument("--ckpt-dir", type=str, default=None, help="Directory containing .ckpt files (defaults to checkpoints/<name>_S<seed>_F<fold>)")
    ap.add_argument("--topk", type=int, default=3, help="If using --ckpt-dir, evaluate top-K by lowest valloss in name")
    # Runtime options
    ap.add_argument("--datasplit", type=str, default="test", choices=["train","val","valid","test"]) 
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--gpu", type=int, default=0, help="GPU index; set to -1 for CPU")
    ap.add_argument("--seed", type=int, default=86)
    ap.add_argument("--name", type=str, default="mazurka_h5_22050")
    # Default to quiet mode to avoid noisy device/GPU info lines from Lightning.
    ap.add_argument("--quiet", action="store_true", default=True, help="Reduce console output: no progress bar, no PL info logs (default)")
    ap.add_argument("--verbose", action="store_false", dest="quiet", help="Enable detailed Lightning logs and progress bars")
    ap.add_argument("--dbn", action="store_true", help="Use DBN post-processing for evaluation (requires madmom)")
    args = ap.parse_args()

    seed_everything(args.seed, workers=True)

    # Reduce Lightning logging noise if requested (default behavior)
    if args.quiet:
        os.environ.setdefault("PYTHONWARNINGS", "ignore")
        for ln in ("pytorch_lightning", "lightning.pytorch", "lightning"):
            logging.getLogger(ln).setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", module=r"pytorch_lightning|lightning")

    # Tee logs
    logs_dir = Path("logs"); logs_dir.mkdir(exist_ok=True, parents=True)
    log_file = logs_dir / f"test_{args.name}_S{args.seed}_F{args.fold}.log"
    _tee_log(log_file)

    # Build list of checkpoints
    ckpt_paths: list[Path] = []
    if args.ckpt:
        ckpt_paths = [Path(p) for p in args.ckpt]
    else:
        if args.ckpt_dir is None:
            ckpt_dir = Path("checkpoints") / f"{args.name}_S{args.seed}_F{args.fold}"
        else:
            ckpt_dir = Path(args.ckpt_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        ckpt_paths = _select_topk_ckpts(ckpt_dir, topk=args.topk)
        if not ckpt_paths:
            raise FileNotFoundError(f"No .ckpt files found under {ckpt_dir}")

    print(f"Selected {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        print(f"  - {p}")

    # Evaluate each checkpoint
    summary_rows = []
    for ckpt in ckpt_paths:
        print(f"\n==== Evaluating {ckpt} ====")
        device_mode = "cpu" if args.gpu < 0 else "gpu"
        model, trainer, checkpoint = _load_pl_from_ckpt(ckpt, device=device_mode, quiet=args.quiet, force_dbn=args.dbn)

        # Print only core BeatThis parameter counts (trainable vs total)
        core = model.model if hasattr(model, "model") and isinstance(model.model, torch.nn.Module) else model
        t_core, n_core = _count_params(core)
        print(f"Parameters (BeatThis core): trainable={t_core:,} total={n_core:,}")
        loader = _build_loader(
            checkpoint,
            datasplit=args.datasplit,
            h5_root=args.h5_root,
            csv_split=args.csv_split,
            fold=args.fold,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
        )
        metrics, avg, datasets, pieces = _predict_and_aggregate(model, trainer, loader)
        print("Averaged metrics:")
        for k, v in avg.items():
            print(f"  {k}: {v:.4f}")
        summary_rows.append((ckpt.name, avg))

    # Optionally write CSV summary
    out_csv = logs_dir / f"summary_{args.name}_S{args.seed}_F{args.fold}.csv"
    if summary_rows:
        keys = sorted(summary_rows[0][1].keys())
        with open(out_csv, 'w') as f:
            f.write('checkpoint,' + ','.join(keys) + '\n')
            for name, avg in summary_rows:
                f.write(name + ',' + ','.join(f"{avg[k]:.6f}" for k in keys) + '\n')
        print(f"\nSaved summary CSV: {out_csv}")


if __name__ == "__main__":
    main()

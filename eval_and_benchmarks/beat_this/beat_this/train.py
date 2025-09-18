import argparse
from pathlib import Path
import logging, warnings, os
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger  # not used; always quiet
from pl_module import PLBeatThis
from h5_dataset import H5BeatDataModule

def main(args):
    # Always run in quiet mode: suppress progress bars, summaries, and PL/device logs
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    for ln in ("pytorch_lightning", "lightning.pytorch", "lightning", "lightning_fabric"):
        logging.getLogger(ln).setLevel(logging.ERROR)
    logging.getLogger("beat_this").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module=r"pytorch_lightning|lightning")

    seed_everything(args.seed, workers=True)

    logger = None  # Always quiet: disable external loggers

    if args.force_flash_attention:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    # Original BeatThis naming (no feature suffix)
    checkpoint_dir = Path("checkpoints") / f"{args.name}_S{args.seed}_F{args.fold}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # --- our H5 DataModule ---
    dm = H5BeatDataModule(
        h5_root=args.h5_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sr=args.sample_rate,
        fps=args.fps,
        train_length=args.train_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
        cache_dir=args.cache_dir,
        csv_split=args.csv_split,
        fold=args.fold,
    )
    dm.setup("fit")

    pos_weights = dm.get_train_positive_weights(widen_target_mask=3)

    # Original BeatThis logMel spectrogram dim
    spect_dim = 128

    pl_model = PLBeatThis(
        spect_dim=spect_dim, fps=args.fps, transformer_dim=args.transformer_dim,
        ff_mult=4, n_layers=args.n_layers, stem_dim=32,
        dropout={"frontend": args.frontend_dropout, "transformer": args.transformer_dropout},
        lr=args.lr, weight_decay=args.weight_decay, pos_weights=pos_weights,
        head_dim=32, loss_type=args.loss, warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs, use_dbn=args.dbn, eval_trim_beats=args.eval_trim_beats,
        sum_head=args.sum_head, partial_transformers=args.partial_transformers
    )

    for part in args.compile:
        if hasattr(pl_model.model, part):
            setattr(pl_model.model, part, torch.compile(getattr(pl_model.model, part)))

    ckpt_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"{args.name}_S{args.seed}_F{args.fold}" + "_epoch{epoch:02d}-valloss{val_loss:.4f}",
        monitor="val_loss",                 # <-- metric key from self.log("val_loss", ...)
        mode="min",                         # smaller is better
        save_top_k=1,                       # keep the best
        save_last=True,                     # also keep a 'last.ckpt'
        every_n_epochs=args.val_frequency,  # snapshot on same cadence as validation frequency
        auto_insert_metric_name=False,
    )

    callbacks = [ckpt_cb]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=[args.gpu],
        num_sanity_val_steps=0,
        logger=False,
        callbacks=callbacks,
        log_every_n_steps=1,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_frequency,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    trainer.fit(pl_model, dm)
    if args.final_test:
        trainer.test(pl_model, dm)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # =============================================================
    # These are the arguments we changed.
    # =============================================================
    ap.add_argument("--h5-root", type=str, default='/media/sbsprl/data/Hanyu/dynest/piano-dynamic-data/hdf5s/mazurka_sr22050')
    ap.add_argument("--sample-rate", type=int, default=22050)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=86)
    ap.add_argument("--csv-split", type=str, default='/media/sbsprl/data/Hanyu/dynest/piano-dynamic-data/split_5fold_fold0_seed86.csv',
                    help="CSV file defining piece_id, split, fold")
    # ==============================================================
    # Everything below is grouped for clarity; no changes here
    # all from the Beat This original repo.
    # ==============================================================
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--name", type=str, default="mazurka_h5_22050")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force-flash-attention", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--compile", nargs="*", type=str, default=["frontend","transformer_blocks","task_heads"])
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--transformer-dim", type=int, default=512)
    ap.add_argument("--frontend-dropout", type=float, default=0.1)
    ap.add_argument("--transformer-dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    # logger option is ignored (always quiet)
    ap.add_argument("--logger", type=str, choices=["wandb","none"], default="none")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--loss", type=str, default="shift_tolerant_weighted_bce",
                    choices=["shift_tolerant_weighted_bce","fast_shift_tolerant_weighted_bce","weighted_bce","bce"])
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accumulate-grad-batches", type=int, default=8)
    ap.add_argument("--train-length", type=int, default=1500)
    ap.add_argument("--dbn", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--eval-trim-beats", type=float, default=5.0)
    ap.add_argument("--val-frequency", type=int, default=5)
    ap.add_argument("--cache-dir", type=str, default="data/_h5_mel_cache", help="Directory for cached mel features and stats")
    ap.add_argument("--tempo-augmentation", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--pitch-augmentation", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--mask-augmentation", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--sum-head", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--partial-transformers", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--length-based-oversampling-factor", type=float, default=0.65)
    ap.add_argument("--val", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--hung-data", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--resume-checkpoint", type=str, default=None)
    ap.add_argument("--resume-id", type=str, default=None)
    # quiet flag removed (always quiet)
    ap.add_argument("--final-test", default=False, action=argparse.BooleanOptionalAction,
                    help="Run a test pass after training (use --final-test to enable)")
    # ================== END ADVANCED PARAMETERS ===================

    args = ap.parse_args()
    main(args)

# Beat This (Transformer) Benchmarks (MazurkaBL + H5)

This folder documents how we train and evaluate the Beat This Transformer on our MazurkaBL H5 dataset for a fair comparison to Beat‑TCN.

What’s different from the original Beat This repo?
- We reuse most of the upstream code and training setup.
- We add an H5 dataloader (`H5BeatDataModule`) that reads waveforms from MazurkaBL `.h5`, computes log‑mels on the fly, and caches features.
- We disable data augmentation to match the Beat‑TCN baseline settings for a fair comparison.

## 1) Environment

Tested on Ubuntu 22.04 LTS, CUDA 11.8/12.2, RTX 3090 (24GiB).

```bash
conda create -n beat_this python=3.10 -y
conda activate beat_this

# Core deps (PyTorch Lightning, librosa, mir_eval, etc.)
python -m pip install --no-build-isolation -r /media/datadisk/home/22828187/zhanh/beat_this/requirements.txt

# Optional (for DBN post-processing during testing)
pip install git+https://github.com/CPJKU/madmom.git
```

## 2) Data

- H5 root: MazurkaBL H5 folder with datasets `waveform`, `beat_time`, and optionally `downbeat_time` per file.
- CSV split: a CSV with columns at least `h5_name` and `split` (values `train`, `valid`, `test`), and optionally `fold`.

Example paths used below:
- `h5_root = "/media/datadisk/home/22828187/zhanh/202505_dynest_data/workspaces/hdf5s/mazurka_sr22050"`
- CSVs like `/media/datadisk/home/22828187/zhanh/202505_dynest_data/workspaces/split_5fold_fold{f}_seed86.csv`

## 3) Train the model (cc. `Eval_BeatThis_MazurkaBL.ipynb`)

We run 5 folds with identical settings to the "Beat This" baseline.

```python
h5_root = "/media/datadisk/home/22828187/zhanh/202505_dynest_data/workspaces/hdf5s/mazurka_sr22050"

for f in range(5):
    csv_path = f"/media/datadisk/home/22828187/zhanh/202505_dynest_data/workspaces/split_5fold_fold{f}_seed86.csv"
    !python -m beat_this.train \
        --sample-rate 22050 --fps 50 --seed 86 \
        --h5-root {h5_root} \
        --csv-split {csv_path} \
        --fold {f} --max-epochs 100
```

Outputs per fold:
- Checkpoints: `checkpoints/mazurka_h5_22050_S86_F{f}/...ckpt`
- Logs: `logs/mazurka_h5_22050_S86_F{f}.log`
- Cached features: `data/_h5_mel_cache/` (auto‑generated; safe to delete)

## 4) Test the model (cc. `Eval_BeatThis_MazurkaBL.ipynb`)

Evaluate the best checkpoints (by validation loss embedded in filename) on the 5-fold test split.

```python
h5_root = "/media/datadisk/home/22828187/zhanh/202505_dynest_data/workspaces/hdf5s/mazurka_sr22050"
csv_root = "/media/datadisk/home/22828187/zhanh/202505_dynest_data/workspaces"

for f in range(5):
    csv_path = f"{csv_root}/split_5fold_fold{f}_seed86.csv"
    ckpt_dir = f"checkpoints/mazurka_h5_22050_S86_F{f}"

    !python -m beat_this.test \
        --h5-root {h5_root} --csv-split {csv_path} --fold {f} \
        --seed 86 --name mazurka_h5_22050 --datasplit test \
        --ckpt-dir {ckpt_dir}
```
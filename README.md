# Piano Dynamic Estimation
This repo corresponding to our submitted paper to ICASSP2026.
- [Joint Estimation of Piano Dynamics and Metrical Structure with a Multi-task Multi-Scale Network](https://drive.google.com/file/d/1mVIySUimkoYNFPKwUDRSmwfwUmlC8OAB/view?usp=sharing)

Our proposed multitask model can estimate piano dynamics, change points, beats, and downbeats from audio at once. We are polishing our model's inference stage, currntly integrates with ["High-resolution Piano Transcription (TASLP2021)"](https://arxiv.org/abs/2010.01815) system, more AMT systems in progress.

We also made a faithful PyTorch implementation of Pampalk’s PsychoFeatureExtractor**Bark-scale specific loudness senation** (CS contribution).

[Inference](#inference) · [Checkpoints](#available-models) · [MazurkaBL Dataset](#mazurkabl-dataset) · [Training & W&B](#training) · [Evaluation Metrics](#reproduce-metrics-from-the-paper) · [PsychoFeatureExtractor](#psychofeatureextractor)


## Inference
to make the model predicted dynamics add-on an existing / AMT-transcribed music score. We


## Available models


## Data PreProcess

To process, Some small error found in MazurkaBL dataset:
 1) MazurkaBL-master `M41-1` has an error. Its performance ID (pid) `9070b-01` [wrong] should be corrected to `9070b-09` [correct]. 
    Affected Files:
    - MazurkaBL-master/beat_time/M41-1beat_time.csv
    - MazurkaBL-master/beat_dyn/M41-1beat_dynNORM.csv

    This issue is automatically corrected using the `fix_mazurka_pid_column` function defined in `pack_h5.py`.

2) MazurkaBL 'M06-4' and 'M63-2' does not contain regular dynamic markings. We skip these two opus by setting the `dataset.exclude_opus` in the `pytorch/config.yaml` file.
    ```python
    dataset.exclude_opus:
    - "M06-4"
    - "M63-2"
    ```
3) Due to the annotation quality, some pids performance will raise up the gradient dramatically during the training. For a stable training (IMPORTANT, this is train-stage ONLY);
    ```python
    dataset.exclude_pids: null >> comment out "null"
    - "mazurka17-4/pid9058-13.h5"
    - "mazurka33-4/pid9080-08.h5"
    - ...
    ```
    Basically, you can skip this setting with `null` because it is a trick for a stable training in the `0th fold data`. If you set a enough epochs as ours (120 epochs), you won't need this. During the evaluation, this must be set to `null` otherwise these pids will be skip in 5-fold cross-validation.


## Create and activate Conda env with CPU-compatible PyTorch 2.2 (which includes MPS for Mac), then Add the pip-only packages
```
conda create -n dyn_mac python=3.10 numpy pandas h5py "pytorch=2.2.*" "torchaudio=2.2.*" -c pytorch -c conda-forge
conda activate dyn_mac
pip install hydra-core omegaconf wandb einops librosa mido tqdm cython
pip install "numpy<2"
pip install seaborn
pip install madmom
```

# Tested Configs

## Experiment Settings and Resource Usage

| Parameter                 | Value in Trial 1               |
|---------------------------|--------------------------------|
| `batch_size`              | 4                              |
| `input_type`              | `"audio"`                      |
| `model_name`              | `"Dual_Dynamic_HPT_v2"`        |
| `targets`                 | `["dynamic", "change_point"]`  |
| `loss_type`               | `"ce_dynamic_gt+change_point"` |
| `audio_feature`           | `"logmel"`                     |
| `midi_feature`            | `"masked_velocity"`            |
| `dynamic_classes`         | 5                              |
| `dynamic_norm`            | `False`                        |
| `dynamic_mask`            | `"change_point"`               |
| `dynamic_mask_radius`     | 1                              |
| `segment_seconds`         | 60.0                           |
| `segment_hop_seconds`     | 30.0                           |
| `sample_rate`             | 16000                          |
| `fft_size`                | 1024                           |
| `frames_per_second`       | 40                             |
| **GPU Usage**             | RTX 3090 @ ~14.6 GB            |
| **RAM Usage**             | ~15GB                          |

## Experiment Comments
1. when random_seed = 42, training loss over-range from 5k to 20k iter (reduce_lr from 10k)
2. change seed to 1234 run again
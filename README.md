# Piano Dynamic Estimation
This repo corresponding to our submitted paper to ICASSP2026.
- Joint Estimation of Piano Dynamics and Metrical Structure with a Multi-task Multi-Scale Network [(Paper)](https://drive.google.com/file/d/1mVIySUimkoYNFPKwUDRSmwfwUmlC8OAB/view?usp=sharing)

Our proposed multitask model can estimate piano dynamics, change points, beats, and downbeats from audio at once. We are polishing our model's inference stage, currntly integrates with ["High-resolution Piano Transcription (TASLP2021)"](https://arxiv.org/abs/2010.01815) system, more AMT systems in progress.

We also made a faithful PyTorch implementation of Pampalk’s **PsychoFeatureExtractor**, to provide Bark-scale specific loudness senation (CS contribution).

[Inference & Checkpoints](#inference-and-checkpoints) &middot;
[PsychoFeatureExtractor](#psychofeatureextractor) &middot;
[Environment Setup](#environment-setup) &middot;
[MazurkaBL Dataset](#mazurkabl-dataset) &middot;
[Training & W&B](#training) &middot;
[Evaluation Metrics](#reproduce-metrics-from-the-paper)


## Inference & Checkpoints
Add predicted **dynamic markings** to an existing **or** AMT-transcribed score. Start with [`Inference.ipynb`](./Inference.ipynb).

- We provide a **pretrained** multi-task, multi-scale checkpoint at  
  `workspaces/checkpoints/formal...`.

This checkpoint is our **best pre-trained** model under the 5-fold protocol (from **fold 0**, used for ablation). More checkpoints (other folds in formal run OR ablation variants) are available in **checkpoints.tar.gz** (Google Drive Download).

## PsychoFeatureExtractor
Ing

## Environment Setup
Create and activate Conda env with CPU-compatible PyTorch 2.2 (which includes MPS for Mac), then Add the pip-only packages
```
conda create -n dyn_mac python=3.10 numpy pandas h5py "pytorch=2.2.*" "torchaudio=2.2.*" -c pytorch -c conda-forge
conda activate dyn_mac
pip install hydra-core omegaconf wandb einops librosa mido tqdm cython
pip install "numpy<2"
pip install seaborn
pip install madmom
```

## MazurkaBL Dataset
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


## Training & W&B



## Evaluation Metrics
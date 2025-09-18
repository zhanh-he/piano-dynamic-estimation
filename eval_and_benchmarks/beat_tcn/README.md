# Beat TCN Benchmarks (MazurkaBL + mirdata)

Use this folder to run beat estimation experiments with Beat‑TCN style notebooks, backed by our local MazurkaBL H5 dataset via a lightweight mirdata loader.

## 1) Environment

Tested on Ubuntu 22.04LTS, CUDA 11.8 & 12.2 + NVIDIA RTX 3090 (24GiB). Although `requirement.txt` provided, we suggest to manually build env as below to avoid errors.

```bash
conda create -n beat_tcn python=3.9 -y
conda activate beat_tcn
pip install "tensorflow==2.15.*" "keras==2.15.*"
pip install cython
pip install numpy scipy pandas matplotlib seaborn tensorboard notebook
pip install "librosa==0.10.2" madmom mir_eval mirdata soxr soundfile audioread tqdm
pip install "numpy==1.23.5"

# Install our developed mirdata (developer mode)
cd mirdata-repo
pip install -e .
```

NOTE - cuDNN: following installation may be required for CUDA <= 12.0 machine.
```bash
pip install nvidia-cudnn-cu12==8.9.2.26 nvidia-cublas-cu12==12.1.3.1 nvidia-cufft-cu12==11.0.2.54 nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 nvidia-cusparse-cu12==12.1.0.106 nvidia-cuda-runtime-cu12==12.1.105 nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-nvtx-cu12==12.1.105
```

## 2) Build Index for MazurkaBL H5

To make the MazurkaBL H5 dataset usable with `mirdata`, you need to build a local index JSON.  

- **`--h5-root`** : path to your MazurkaBL H5 folder (`/path/to/mazurka_sr22050`)  
- **`--out`** : where to save the JSON index (`mirdata-repo/tests/indexes/mazurka_h5_local.json`)  
- **`--no-checksum`** : skip checksum validation

Example command:

```bash
python -m mirdata.datasets.indexes.build_mazurka_h5_index \
  --h5-root /path/to/mazurka_sr22050 \
  --out mirdata-repo/tests/indexes/mazurka_h5_local.json \
  --no-checksum
```
In our machine:
```bash
python -m mirdata.datasets.indexes.build_mazurka_h5_index --h5-root /media/datadisk/home/22828187/zhanh/202505_dynest_data/workspaces/hdf5s/mazurka_sr22050 --out /media/datadisk/home/22828187/zhanh/beat_tcn/mirdata-repo/tests/indexes/mazurka_h5_local.json --no-checksum
```

## 3) Notebooks

For step-by-step instructions and reported benchmark scores, please refer to the provided Jupyter notebooks in this folder:

- **Tempo Estimation MazurkaBL.ipynb** — contains 5-fold training and evaluation on our preprocessed MazurkaBL H5 dataset, with all paper-reported scores reproducible.  

Simply open the notebook and run the cells sequentially to reproduce the results.

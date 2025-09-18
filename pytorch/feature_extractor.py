"""
Signal Processing Basics:
- sample_rate: number of samples per second (e.g., 44100 Hz).
- hop_size: step size (in samples) between frames.
- FPS (frames per second) = sample_rate / hop_size.

Window Overlap:
- fft_size: length of each analysis window (in samples).
- overlap % = (fft_size - hop_size) / fft_size * 100
  e.g., fft_size=1024, hop_size=256 → 75% overlap;
        fft_size=1024, hop_size=512 → 50% overlap.

=============================== Literature ================================
- Pampalk "ma_sone.m" matlab toolbox (2007):
    https://www.pampalk.at/ma/documentation.html
- Joyti ISMIR2024 paper:   
    https://arxiv.org/pdf/2410.20540

==================== Joyti's ISMIR2024 FFT parameters ====================
- logMel (short):
    sample_rate=44100, fft_size=1024, hop_size=256 (75% overlap)
    → FPS≈172; downsample x3 → effective FPS≈57    (17.4ms temporal resolution)
    → segment length 4096 frames x 17.4ms ≈ 71s    (71s each segment)
- logMel (long):
    sample_rate=44100, fft_size=1024, hop_size=256 (75% overlap)
    → FPS≈172; downsample x5 → effective FPS≈34    (29ms temporal resolution)
    → segment length 10000 frames x 29ms ≈ 290s    (290s each segment)
- Bark (short):
    sample_rate=48000, fft_size=256, hop_size=96   (63% overlap)
    → FPS≈500; downsample x8 → effective FPS≈62    (16ms temporal resolution)
    → segment length 4096 frames x 16ms ≈ 66s      (66s each segment)
- Bark (long):
    sample_rate=48000, fft_size=256, hop_size=96   (63% overlap)
    → FPS≈500; downsample x15 → effective FPS≈33   (30ms temporal resolution)
    → segment length 10000 frames x 30ms ≈ 300s    (300s each segment)
    
=============================================================================

Common Defaults:
- Librosa Log-Mel:
    sample_rate=22050, fft_size=2048, hop_size=512 (75% overlap) → FPS≈43, mel_bins=229
- MATLAB 2007 Bark:
    sample_rate=16000, fft_size=1024, hop_size=512 (50% overlap) → FPS≈86, bark_bands=24

Our Development:
- Bark @16kHz: 
    Fix 50% overlap ratio for the bark feature, same as the MATLAB 2007 & mosqito (ISO.532-1:2017) defaults.
    → sample_rate=16000, fft_size=256,  fps=125    (~50% ovarlap)
                         fft_size=512,  fps=62     (~50% ovarlap)
                         fft_size=1024, fps=31     (~50% ovarlap)
- Log-Mel @16kHz: 
    Fix 75% overlap for the log-mel feature, same as the librosa defaults.
    → sample_rate=16000, fft_size=512,  fps=60     (~75% ovarlap)
"""
import h5py
import torch
import torch.nn as nn
import argparse
import torch
import numpy as np
import pandas as pd
import os
from typing import Literal
import torchaudio
from mosqito.sq_metrics import loudness_zwtv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Utility function for feature extractor and freq_bins selection
def get_feature_extractor_and_bins(audio_feature, sample_rate, fft_size, frames_per_second):
    if audio_feature == "logmel":
        feature_extractor = LogMelExtractor(sample_rate=sample_rate, fft_size=fft_size, frames_per_second=frames_per_second)
        freq_bins = feature_extractor.mel_bins
    elif audio_feature == "bark":
        feature_extractor = PsychoFeatureExtractor(sample_rate=sample_rate, fft_size=fft_size, frames_per_second=frames_per_second, db_max=96.0, return_mode="bark")
        freq_bins = feature_extractor.bark_bands
    elif audio_feature == "sone":
        feature_extractor = PsychoFeatureExtractor(sample_rate=sample_rate, fft_size=fft_size, frames_per_second=frames_per_second, db_max=96.0, return_mode="sone")
        freq_bins = feature_extractor.bark_bands
    elif audio_feature == "ntot":
        feature_extractor = PsychoFeatureExtractor(sample_rate=sample_rate, fft_size=fft_size, frames_per_second=frames_per_second, db_max=96.0, return_mode="ntot")
        freq_bins = 1
    elif audio_feature == "mosqito_sone":
        feature_extractor = MoSQIToExtractor(sample_rate=sample_rate, mode="sone")
        freq_bins = 1
    elif audio_feature == "mosqito_bark":
        feature_extractor = MoSQIToExtractor(sample_rate=sample_rate, mode="bark")
        freq_bins = len(feature_extractor.freq_bins) if feature_extractor.freq_bins is not None else 24
    else:
        raise ValueError(f"Invalid audio_feature: {audio_feature}")
    return feature_extractor, freq_bins


class PsychoFeatureExtractor(nn.Module):
    """
    Bark-based Psychoacoustic Feature Extractor.

    References:
      - Zwicker & Fastl (1999): Bark scale bands.
      - Terhardt (1979): Outer ear weighting.
      - Schroeder et al. (1979): Spreading function.
      - MATLAB toolbox (Pampalk, 2004): matlab implementation for "bark/sone" extraction.
      - Stevens method (Hartmann, 1997): Total loudness (sone -> "ntot").

    Input:
      - wav: (B, T)  audio waveform in time samples

    Output (depends on return_mode):
      - bark: (B, C, F) Bark bands loudness in dB-SPL
      - sone: (B, C, F) Bark bands loudness in sones
      - ntot: (B, F)    Total avergae bank bands loudness in sones (avg per frame)

    Where:
      B = batch size
      T = samples per recording
      C = number of Bark bands (e.g., 24)
      F = number of frames

    Parameters:
      - sample_rate: audio sample rate
      - fft_size: FFT window size
      - db_max: max dB scale for waveform normalization
      - outer_ear: outer ear model ['terhardt', 'modified_terhardt', 'none']
      - return_mode: feature to return ['bark', 'sone', 'ntot']
      - frames_per_second: desired frames per second (determines hop_size)
    """
    def __init__(self, sample_rate=44100, fft_size=1024, frames_per_second=86, db_max=96.0,  # default params in matlab 2007 implementation
                 outer_ear: Literal["terhardt", "modified_terhardt", "none"] = "terhardt",
                 return_mode: Literal["bark", "sone", "ntot"] = "ntot"):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.db_max = db_max
        self.outer_ear = outer_ear
        self.return_mode = return_mode
        self.hop_size = int(round(sample_rate / frames_per_second))
        self.window = torch.hann_window(fft_size)
        self.fft_freq = torch.linspace(0, sample_rate/2, fft_size//2 + 1) # FFT bin frequencies

        # Valid Bark bands (up to 24) in Nyquist fequency range
        bark_upper = torch.tensor([100,200,300,400,510,630,770,920,1080,1270,1480,1720,
            2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500])
        bark_center = torch.tensor([50,150,250,350,450,570,700,840,1000,1170,1370,1600,
            1850,2150,2500,2900,3400,4000,4800,5800,7000,8500,10500,13500])
        self.bark_upper = bark_upper[bark_upper <= sample_rate/2]
        self.bark_center = bark_center[bark_center <= sample_rate/2]
        self.bark_bands = len(self.bark_upper)

    def _outer_ear_weighting(self, power):
        # Outer ear weighting (Default to Terhardt model) is designed in dB-domain
        W_Adb = torch.zeros_like(self.fft_freq)
        f_kHz = self.fft_freq[1:] / 1000
        if self.outer_ear == "terhardt":
            W_Adb[1:] = -3.64 * f_kHz ** -0.8 + 6.5 * torch.exp(-0.6 * (f_kHz - 3.3) ** 2) - 0.001 * f_kHz ** 4
        elif self.outer_ear == "modified_terhardt":
            W_Adb[1:] = 0.6 * (-3.64 * f_kHz ** -0.8) + 0.5 * torch.exp(-0.6 * (f_kHz - 3.3) ** 2) - 0.001 * f_kHz ** 4
        # Power spectrogram is in linear domain, so W_Adb should convert to linear domain before it applied
        W = (10 ** (W_Adb / 20)) ** 2 
        return power * W.to(power.device).view(1, -1, 1)
    
    def _bark_scaling(self, W_power, device):
        bands, k = [], 0
        for i in range(self.bark_bands):
            idx = torch.arange(k, k + (self.fft_freq[k:] <= self.bark_upper[i]).sum().item(), device=device)
            bands.append(W_power[:, idx, :].sum(dim=1))
            k = idx[-1]+1
        return torch.stack(bands, dim=1)

    def _schroeder_spreading(self, bark):
        # Schroeder spreading matrix (psychoacoustic masking)
        b = torch.arange(1, self.bark_bands + 1, device=bark.device).unsqueeze(1) - torch.arange(1, self.bark_bands + 1, device=bark.device).unsqueeze(0) + 0.474
        spread = 10 ** ((15.81 + 7.5 * b - 17.5 * torch.sqrt(1 + b ** 2)) / 10)
        return torch.matmul(spread, bark)
    
    def _compute_ntot(self, sone):
            # Stevens method: max band + 15% of the rest
            max_val, idx = torch.max(sone, dim=1, keepdim=True)
            rest = (sone * torch.ones_like(sone).scatter(1, idx, 0)).sum(dim=1)
            ntot = max_val.squeeze(1) + 0.15 * rest
            # Normalize per recording to [0, 1]
            # ntot_norm = ntot / (ntot.max(dim=1, keepdim=True).values + 1e-9)
            return ntot # ntot_norm
    
    # -------------------------------------------------------------
    def forward(self, wav: torch.Tensor):
        wav = wav * (10 ** (self.db_max/20))                        # 1) Scale waveform to max dB range, B, T = wav.shape
        spec = torch.stft(wav, n_fft=self.fft_size,                 # 2) Compute STFT > complex spectrogram
            hop_length=self.hop_size, window=self.window.to(wav.device), return_complex=True)
        power = spec.abs() ** 2 / self.window.sum() ** 2            # 3) Power spectrogram
        W_power = self._outer_ear_weighting(power)                  # 4) Apply outer ear weighting
        bark = self._bark_scaling(W_power, wav.device)              # 5) Group freq bins by Bark bands > Bark-scale specific loudness (BSSL) in linear power
        Sp_bark = self._schroeder_spreading(bark)                   # 6) Apply spectral spreading
        bark_db = 10 * torch.log10(torch.clamp(Sp_bark, min=1.0))   # 7) Convert BSSL linear power to dB; prevent log(0) issues with torch.clamp
        
        # 8) Convert BSSL dB to sone
        sone = torch.where(bark_db >= 40, 2 ** ((bark_db - 40) / 10), (bark_db / 40) ** 2.642)
        # 9) Intergrate Bark-scale specific loudness (BSSL) to Bark-scale total loudness (Ntot), unit is sone
        ntot = self._compute_ntot(sone) 

        if self.return_mode == "bark":   # BSSL in dB
            return bark_db
        elif self.return_mode == "sone": # BSSL in sone, our paper focus on this
            return sone
        elif self.return_mode == "ntot": # Bark total loudness in sone, used for visualization
            return ntot
        else:
            raise ValueError(f"Invalid return_mode: {self.return_mode}")


class LogMelExtractor(nn.Module):
    """
    Log-Mel spectrogram extractor using torchaudio.
    Parameters follow common usage in singing voice dynamics estimation (ISMIR2024 Joyti Narang).
    - norm & mel_scale="slaney" is good for human perception
    - norm=None, mel_scale='htk' is good for pitch estimation
    
    Output shape: (B, M, F)
    - B: batch size
    - M: number of Mel bands
    - F: number of frames

    FPS:
    - sr 44100 / fps 86 = hop_size 512, 50% overlap as fft_size is 1024
    """
    def __init__(self, sample_rate=44100, fft_size=1024, frames_per_second=86):
        super().__init__()
        self.mel_bins = 128 # 128 default in librosa & torchaudio & beat_this ISMIR2024
        hop_size, fmin, fmax = int(sample_rate // frames_per_second), 30, int(sample_rate//2) 
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=fft_size, hop_length=hop_size, n_mels=self.mel_bins,
                                                                    center=True, pad_mode='reflect', f_min=fmin, f_max=fmax, 
                                                                    power=2.0, norm="slaney", mel_scale="slaney")
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def forward(self, wav: torch.Tensor):
        # Compute Mel Spec > log Mel Spectrogram
        mel_spec = self.mel_spectrogram(wav)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        return log_mel_spec



# --- Unified MoSQIToExtractor ---
class MoSQIToExtractor(nn.Module):
    """
    Bark-band specific loudness extractor using MoSQITo loudness_zwtv.
    mode = "sone": returns overall loudness (B, T, 1)
    mode = "bark": returns specific loudness per bark-scale critical band (B, T, N)
    """
    def __init__(self, sample_rate=44100, mode="sone"):
        super().__init__()
        self.sample_rate = sample_rate
        self.mode = mode
        self.loudness_zwtv = loudness_zwtv
        self.freq_bins = None

    def forward(self, wav: torch.Tensor):
        results, times = [], []
        for waveform in wav.cpu().numpy():
            overall, specific, freq, time = self.loudness_zwtv(signal=waveform, fs=self.sample_rate)
            time = np.asarray(time).ravel()
            if self.mode == "sone":
                data = np.asarray(overall).ravel().reshape(-1, 1)
            elif self.mode == "bark":
                data = np.asarray(specific)
                if self.freq_bins is None:
                    self.freq_bins = freq
            else:
                raise ValueError(f"Invalid mode for MoSQIToExtractor: {self.mode}")
            results.append(torch.tensor(data, dtype=torch.float32))
            times.append(time)
        return torch.stack(results), (self.freq_bins if self.mode=="bark" else None), times[0]

# ----------------------------------------------------------------------------------------------
# This __main__ block is for CLI testing / debugging.
# In practice, import and use the classes above directly to PyTorch Dataset or Model.
# ----------------------------------------------------------------------------------------------

# Visualise the feature in csv for debugging
def save_feature_csv(features, times, columns, output_csv_path):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df = pd.DataFrame(features, columns=columns)
    df.insert(0, "time", times)
    df.to_csv(output_csv_path, index=False, float_format='%.7f')
    print(f"Saved feature CSV to {output_csv_path}")

def save_feature_plot(features, times, mode, output_png_path,
    t_start=None, t_end=None, duration=None, figsize=(6,3.4), dpi=200):
    if any(v is not None for v in [t_start, t_end, duration]):
        if duration is not None and t_end is None:
            t_start = 0.0 if t_start is None else t_start
            t_end = t_start + duration
        t0 = max(times[0], t_start if t_start is not None else times[0])
        t1 = min(times[-1], t_end if t_end is not None else times[-1])
        i0, i1 = np.searchsorted(times, t0, "left"), np.searchsorted(times, t1, "right")
        features, times = features[i0:i1], times[i0:i1]
    unit_map = {"bark":"dB-SPL", "sone":"sones", "ntot":"sones", "logmel":"dB", "mosqito_sone":"sones", "mosqito_bark":"sones"}
    unit = unit_map.get(mode, "")
    fig = plt.figure(figsize=figsize)
    f = features if features.ndim==2 else features.reshape(-1,1)
    if f.shape[1]==1:
        plt.plot(times, f[:,0], lw=1)
        plt.xlabel("Time (s)"); plt.ylabel(f"{mode} ({unit})" if unit else mode)
        plt.title(f"{mode} over time"); plt.grid(alpha=0.3)
    else:
        im = plt.imshow(f.T, aspect="auto", origin="lower", extent=[times[0], times[-1], 0, f.shape[1]], interpolation="nearest", rasterized=True,
            vmin=(np.percentile(f, 10) if mode == "logmel" else None),
            vmax=(np.percentile(f, 99.99) if mode in ["sone", "bark"] else None))    
        plt.xlabel("Time (s)")
        ylab = {"logmel":"Mel bins", "bark":"Bark bands", "sone":"Bark bands", "mosqito_bark":"Bark bands"}.get(mode,"Channels")
        plt.ylabel(f"{ylab} (N={f.shape[1]})")
        plt.colorbar(im).set_label(unit if unit else "amplitude")
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title(f"{mode} spectrogram-like visualization")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=dpi); plt.close(fig)
    print(f"Saved feature plot to {output_png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Bark, Log-Mel, or MoSQITo Sones from .h5 waveform.")
    parser.add_argument("h5_input_path", type=str, help="Path to the input .h5 file")
    parser.add_argument("output_csv_path", type=str, help="Path to the output .csv file")
    parser.add_argument("--mode", type=str, default="sone", choices=["sone", "bark", "ntot", "logmel", "mosqito_sone", "mosqito_bark"], help="Feature to extract: sone | ntot | logmel | mosqito_sone | mosqito_bark")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Waveform sample rate (default: 44100)")
    parser.add_argument("--fft_size", type=int, default=1024, help="FFT size (default: 1024)")
    parser.add_argument("--frames_per_second", type=float, default=86, help="Frames per second for feature extraction")
    parser.add_argument("--plot_path", type=str, default=None, help="Optional PNG path to save a visualization")
    args = parser.parse_args()

    # Load waveform
    with h5py.File(args.h5_input_path, 'r') as hf:
        waveform = hf['waveform'][:].astype(np.float32) / 32768.0

    wav_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

    if args.mode == "logmel":
        extractor = LogMelExtractor(
            sample_rate=args.sample_rate,
            fft_size=args.fft_size,
            frames_per_second=args.frames_per_second
        )
        features = extractor(wav_tensor).squeeze(0).numpy().T
        hop_duration = extractor.mel_spectrogram.hop_length / extractor.mel_spectrogram.sample_rate
        times = np.arange(features.shape[0]) * hop_duration
        columns = [f"mel_{i+1}" for i in range(features.shape[1])]

    elif args.mode == "mosqito_sone":
        extractor = MoSQIToExtractor(sample_rate=args.sample_rate, mode="sone")
        features, _, times = extractor(wav_tensor)
        features = features.squeeze(0).detach().cpu().numpy()
        times = np.array(times)
        columns = ["mosqito_sone"]

    elif args.mode == "mosqito_bark":
        extractor = MoSQIToExtractor(sample_rate=args.sample_rate, mode="bark")
        features, freqs, times = extractor(wav_tensor)
        features = features.squeeze(0).detach().cpu().numpy()
        times = np.array(times)
        columns = [f"mosqito_bark_{int(f)}Hz" for f in freqs]
        # columns = [f"bark_{i+1}" for i in range(features.shape[1])]

    else:
        extractor = PsychoFeatureExtractor(
            sample_rate=args.sample_rate,
            fft_size=args.fft_size,
            frames_per_second=args.frames_per_second,
            return_mode=args.mode,
        )
        features = extractor(wav_tensor).squeeze(0).numpy().T if args.mode not in ["ntot"] else extractor(wav_tensor).squeeze(0).numpy()
        hop_duration = extractor.hop_size / extractor.sample_rate
        times = np.arange(features.shape[0]) * hop_duration
        columns = [f"{args.mode}_{i+1}" for i in range(features.shape[1])] if args.mode not in ["ntot"] else [args.mode]

    save_feature_csv(features, times, columns, args.output_csv_path)
    save_feature_plot(features, times, args.mode, args.plot_path, t_start=10, duration=50)
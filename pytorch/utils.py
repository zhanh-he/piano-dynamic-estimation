from tqdm import tqdm
import numpy as np
import os, csv, logging
import pandas as pd
import torch
from mido import MidiFile
from sklearn.model_selection import train_test_split, GroupKFold
import torch.nn.functional as F
import re
from omegaconf import OmegaConf
from typing import Dict, Tuple, Any, List
from models_audioCNN import SingleCNN, MultiTaskCNN

def read_midi(midi_path, mode ='hpt'):
    """
    Parse a MIDI file and return events with timestamps.
    Args:
        midi_path (str): Path to the MIDI file.
        mode (str): One of 'maestro', 'hpt', 'smd', or 'maps'. Determines where tempo and events are stored.
            - 'maestro' or 'hpt' (hpt transcribed MazurkaBL): 2 tracks.
              • Track 0 holds all meta messages (set_tempo, time_signature, end_of_track).
              • Track 1 holds piano events.

            - 'smd': 2 tracks.
              • Track 0 holds meta messages, but tempo is the second index (track_name, set_tempo, time_signature, end_of_track).
              • Track 1 holds piano events.

            - 'maps': 1 track.
              • That track holds both meta messages and piano events (tempo is the first message).
    Returns:
        dict: {
            'midi_event': np.ndarray of message strings,
            'midi_event_time': np.ndarray of timestamps in seconds
        }
    """
    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    ds = mode.lower()
    if ds in ('maestro', 'hpt'):
        # Expect 2 tracks: track 0 for meta (tempo at index 0), track 1 for piano events
        assert len(midi_file.tracks) == 2, f"{mode} format requires 2 tracks, found {len(midi_file.tracks)}"
        microseconds_per_beat = midi_file.tracks[0][0].tempo
        play_track_idx = 1

    elif ds == 'smd':
        # Expect 2 tracks: track 0 for meta (tempo at index 1), track 1 for piano events
        assert len(midi_file.tracks) == 2, f"SMD format requires 2 tracks, found {len(midi_file.tracks)}"
        microseconds_per_beat = midi_file.tracks[0][1].tempo
        play_track_idx = 1

    elif ds == 'maps':
        # Expect 1 track: contains both meta and piano events (tempo at index 0)
        assert len(midi_file.tracks) == 1, f"MAPS format requires 1 track, found {len(midi_file.tracks)}"
        microseconds_per_beat = midi_file.tracks[0][0].tempo
        play_track_idx = 0

    else:
        raise ValueError(f"Dataset/Mode not supported: {mode}")

    # Convert ticks to seconds
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []
    time_in_second = []
    ticks_accum = 0

    # Iterate over piano event track
    for msg in midi_file.tracks[play_track_idx]:
        message_list.append(str(msg))
        ticks_accum += msg.time
        time_in_second.append(ticks_accum / ticks_per_second)

    return {
        'midi_event': np.array(message_list),
        'midi_event_time': np.array(time_in_second)
    }



def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def traverse_folder(folder):
    paths = []
    names = []

    for root, dirs, files in os.walk(folder):
        for name in files:
            filepath = os.path.join(root, name)
            names.append(name)
            paths.append(filepath)
            
    return names, paths


def float32_to_int16(x, mazurka_id=None, perf_file=None):
    peak = np.max(np.abs(x))
    if peak > 1.0:
        msg = f"[WARN] Audio peak {peak:.4f} > 1.0"
        if mazurka_id or perf_file:
            msg += f" in M{mazurka_id} / {perf_file}. Been normalized."
        tqdm.write(msg)
        x = x / peak
    return (x * 32767.).astype(np.int16)


# Variant: never normalizes, just warns if peak > 1.0
def float32_to_int16_no_norm(x, mazurka_id=None, perf_file=None):
    peak = np.max(np.abs(x))
    if peak > 1.0:
        msg = f"[WARN] Audio peak {peak:.4f} > 1.0"
        if mazurka_id or perf_file:
            msg += f" in M{mazurka_id} / {perf_file}. Without normalized."
        tqdm.write(msg)
    return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)



# ----------------------
# Lightweight shared helpers (ckpt + overrides + cfg access)
# ----------------------

def parse_overrides_str(s):
    """Parse simple Hydra-style overrides string 'k=v,k2=v2' into a dict.
    Returns None if input is falsy.
    """
    if not s:
        return None
    d = {}
    for kv in str(s).split(','):
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        vl = v.strip().lower()
        if vl in {'true', 'false'}:
            d[k] = (vl == 'true')
        else:
            try:
                d[k] = int(v)
            except ValueError:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
    return d


def _parse_valf1_from_name(name: str):
    base = os.path.basename(name)
    m = re.search(r"valf1[_-]?(\d+\.\d+)", base) or re.search(r"valf1(\d+\.\d+)", base)
    return float(m.group(1)) if m else None


def _parse_epoch_from_name(name: str):
    m = re.search(r"epoch[_-]?(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


def list_checkpoints(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        raise NotADirectoryError(ckpt_dir)
    return sorted(
        [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
         if f.endswith(('.pth', '.ckpt'))]
    )


def select_checkpoint(ckpt_dir: str, valf1_rank: int = 1, min_epoch: int | None = None):
    """Pick a checkpoint by valf1 rank (best-first). Falls back to newest mtime.
    Optionally filter out checkpoints with epoch < min_epoch.
    """
    files = list_checkpoints(ckpt_dir)
    if min_epoch is not None:
        files = [p for p in files if (_parse_epoch_from_name(p) or -1) >= int(min_epoch)]
    if not files:
        raise RuntimeError(f"No eligible checkpoints in {ckpt_dir} after min_epoch filter.")
    scored = [(p, _parse_valf1_from_name(p)) for p in files]
    scored = [(p, v) for p, v in scored if v is not None]
    ranked = [p for p, _ in sorted(scored, key=lambda x: x[1], reverse=True)] if scored else \
             sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
    if valf1_rank <= 0 or valf1_rank > len(ranked):
        raise IndexError(f"valf1_rank {valf1_rank} out of range (len={len(ranked)})")
    return ranked[valf1_rank - 1]

# ----------------------
# Numpy: pad or truncate along a given axis
# ----------------------
def pad_or_truncate_np(arr: np.ndarray, target_length: int, pad_value: float | int = 0, axis: int = 0) -> np.ndarray:
    """Pad with constant `pad_value` or truncate a numpy array along `axis` to `target_length`.
    Works for 1D or ND arrays; preserves other axes.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    cur = arr.shape[axis]
    if cur == target_length:
        return arr
    if cur > target_length:
        # slice selection for truncation
        index = [slice(None)] * arr.ndim
        index[axis] = slice(0, target_length)
        return arr[tuple(index)]
    # pad
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, int(target_length - cur))
    return np.pad(arr, pad_width, mode='constant', constant_values=pad_value)


# === CSV Merge 工具函数 ===

def load_dyn_markings(mazurka_id, dyn_folder):
    """
    Load beat timestamps of a given mazurka.

    Each CSV file stores:
        - Row 1: Beat indices (int) where a new dynamic marking occurs.
        - Row 2: Corresponding dynamic values (e.g., p, f, mf).

    Args:
        mazurka_id (str): ID of the Mazurka piece (e.g., 'M06-1').
        dyn_folder (str): Path to the folder containing `markings.csv` files.

    Returns:
        Tuple[List[int], List[str]]: A tuple of beat indices and their dynamic markings.
    """
    df = pd.read_csv(os.path.join(dyn_folder, f"{mazurka_id}markings.csv"), header=None)
    return df.iloc[1].dropna().astype(int).tolist(), df.iloc[2].dropna().tolist()


def load_beat_times(mazurka_id, beat_folder, pid=None):
    """
    Load beat timestamps of a given mazurka (all its performances - pids).

    Args:
        mazurka_id (str): ID of the Mazurka piece (e.g., 'M06-1').
        beat_folder (str): Path to the folder containing `beat_time.csv` files.
        pianist_id (str or None): If provided, return beat timestamps of that pianist.
                                  If None, return the number of beats in the score.

    Returns:
        Union[List[float], int]: List of beat timestamps (float) if pianist_id is given,
                                 otherwise the number of beats in the piece (int).
    """
    df = pd.read_csv(os.path.join(beat_folder, f"{mazurka_id}beat_time.csv"))
    if pid:
        if pid not in df.columns:
            raise ValueError(f"Pianist ID {pid} not found. Should not happen, need a fix.") 
        return df[pid].dropna().tolist()
    return int(df.iloc[-1, 0])


def load_performance_end_time(discography_file, pid):
    """
    Read performance duration for a performance (pid).

    The discography.txt contains a 'pid' column and a 'time' column in MM:SS format.

    Args:
        discography_file (str): Path to the tab-separated file with performance info.
        pid (str): Pianist performance ID to lookup.

    Returns:
        int: Total duration of the performance in seconds.

    Raises:
        ValueError: If pid is not found in the table.
    """
    pid = pid.replace('pid', '')
    df = pd.read_csv(discography_file, sep="\t")
    row = df[df['pid'] == pid]
    if row.empty:
        raise ValueError(f"PID {pid} not found.")
    m, s = map(int, row.iloc[0]['time'].split(":"))
    return m * 60 + s


def load_discography_pid_metadata(discography_file, pid, mazurka_id=None):
    """
    Load opus, performer, and duration for a specific performance.

    Args:
        discography_file (str): Path to the discography text file.
        pid (str): Performance ID to look up (e.g., '9070-09').
        mazurka_id (str, optional): ID of the mazurka opus for clearer context.

    Returns:
        tuple: (opus (str), performer (str), duration_seconds (int))

    Raises:
        ValueError: If the PID is not found. Includes guessed opus if available.
    """
    pid = pid.replace('pid', '')
    df = pd.read_csv(discography_file, sep=r'\s{2,}|\t', engine='python')
    df.columns = ['opus', 'key', 'performer', 'year', 'time_str', 'seconds', 'label', 'pid', 'status']
    df['pid'] = df['pid'].str.strip()

    row = df[df['pid'] == pid]
    context = f" in opus '{mazurka_id}'" if mazurka_id else ""

    if row.empty:
        guess_rows = df[df['pid'].str.startswith(pid[:4])]
        if not guess_rows.empty:
            guessed_opus = guess_rows.iloc[0]['opus']
            raise ValueError(f"[WARN] PID '{pid}'{context} not found in discography. Possibly belongs to opus '{guessed_opus}'")
        else:
            raise ValueError(f"[WARN] PID '{pid}'{context} not found in discography. Using 'Unknown' for performer and -1 for duration.")

    row = row.iloc[0]
    return str(row['opus']), str(row['performer']), int(row['seconds'])


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]


def prepare_batch_input(batch_data_dict, input_type):
    if input_type == 'audio':
        return [batch_data_dict['audio_input']]
    elif input_type == 'midi':
        return [batch_data_dict['midi_input']]
    elif input_type == 'both':
        return [batch_data_dict['audio_input'], batch_data_dict['midi_input']]
    else:
        raise ValueError(f"Unknown input_type: {input_type}")


def data_random_split(data_list, split_ratio=(8, 1, 1), seed=42, csv_dir=None):
    a, b, c = split_ratio
    # First take off train
    train_list, rest = train_test_split(data_list,
        train_size=a/(a+b+c), 
        random_state=seed, 
        shuffle=True)
    # Then split the valid / test
    if (b + c) != 0:
        valid_list, test_list = train_test_split(
            rest, 
            train_size=b/(b+c), 
            random_state=seed, 
            shuffle=True)
    else:
        valid_list, test_list = [], []
    # optional CSV export
    if csv_dir is not None:
        ratio_tag = f"{a}-{b}-{c}"
        csv_path = os.path.join(csv_dir, f"split_random_r{ratio_tag}_seed{seed}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['h5_name', 'opus', 'split', 'seed'])
            def _w(lst, split_name):
                for fname, full_path in lst:
                    opus = os.path.basename(os.path.dirname(full_path))
                    writer.writerow([fname, opus, split_name, seed])
            _w(train_list, 'train')
            _w(valid_list, 'valid')
            _w(test_list,  'test')
    return train_list, valid_list, test_list


def data_5fold_split(data_list, groups, seed=42, fold_index=0, csv_dir=None):
    """Standard 5-fold CV (group-aware by opus):
    test fold = fold_index; valid fold = (fold_index + 1) % 5; train = remaining 3 folds.
    Returns (train_list, valid_list, test_list).
    """
    gkf = GroupKFold(n_splits=5)
    data_array = np.array(data_list)
    groups_array = np.array(groups)

    # Collect test folds once in order
    test_folds = [test_idx for _, test_idx in gkf.split(data_array, groups=groups_array)]
    if len(test_folds) != 5:
        raise RuntimeError(f"Expected 5 folds, got {len(test_folds)}")

    test_k = int(fold_index) % 5
    valid_k = (test_k + 1) % 5

    test_idx = test_folds[test_k]
    valid_idx = test_folds[valid_k]

    mask = np.ones(len(data_array), dtype=bool)
    mask[test_idx] = False
    mask[valid_idx] = False
    train_idx = np.where(mask)[0]

    train_list = data_array[train_idx].tolist()
    valid_list = data_array[valid_idx].tolist()
    test_list  = data_array[test_idx].tolist()

    # optional CSV export
    if csv_dir is not None:
        csv_path = os.path.join(csv_dir, f"split_5fold_fold{test_k}_seed{seed}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['h5_name', 'opus', 'split', 'fold_index', 'seed'])
            def _w(lst, split_name):
                for fname, full_path in lst:
                    opus = os.path.basename(os.path.dirname(full_path))
                    writer.writerow([fname, opus, split_name, test_k, seed])
            _w(train_list, 'train')
            _w(valid_list, 'valid')
            _w(test_list,  'test')

    return train_list, valid_list, test_list


def pad_or_truncate_time(x: torch.Tensor, T_desired: int, pad_value: float | int = 0, side: str = "right") -> torch.Tensor:
    """
    Pad (with constant value) or truncate a tensor along its *time* dimension to match `T_desired`.
    Supported shapes:
      - (T,)              -> time is dim=0
      - (B, T)            -> time is dim=1
      - (B, T, C)         -> time is dim=1 (class/features last)
    Padding is applied on the **right** side by default (common for sequence alignment).
    Truncation slices on the right to preserve early timesteps' alignment.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"pad_or_truncate_time expects a torch.Tensor, got {type(x)}")
    if side != "right":
        raise ValueError("Only right-side padding/truncation is supported (side='right').")

    if x.dim() == 1:
        T = x.size(0)
        if T < T_desired:
            pad_len = int(T_desired - T)
            return F.pad(x, (0, pad_len), mode="constant", value=pad_value)
        elif T > T_desired:
            return x[:T_desired]
        return x

    if x.dim() == 2:
        B, T = x.shape
        if T < T_desired:
            pad_len = int(T_desired - T)
            return F.pad(x, (0, pad_len), mode="constant", value=pad_value)
        elif T > T_desired:
            return x[:, :T_desired]
        return x

    if x.dim() == 3:
        B, T, C = x.shape
        if T < T_desired:
            pad_len = int(T_desired - T)
            # pad over (last, second-last) dims => (C_left, C_right, T_left, T_right)
            return F.pad(x, (0, 0, 0, pad_len), mode="constant", value=pad_value)
        elif T > T_desired:
            return x[:, :T_desired, :]
        return x

    raise ValueError(f"Unsupported tensor rank {x.dim()} for pad_or_truncate_time (expected 1, 2, or 3 dims)")

def log_gradient_norm(model, step):
    """
    Compute and log total gradient norm for debugging.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


# ---------------- Model load ----------------
_MODEL = {"SingleCNN": SingleCNN, "MultiTaskCNN": MultiTaskCNN}

def _load_ckpt(path: str, device: torch.device):
    try:  # fast path
        return torch.load(path, map_location=device) 
    except Exception:  # fallback
        return torch.load(path, map_location=device, weights_only=False)

def load_model(checkpoint_path: str, device: torch.device, overrides: Dict[str, Any] | None) -> Tuple[torch.nn.Module, OmegaConf]:
    if not os.path.isfile(checkpoint_path): raise FileNotFoundError(checkpoint_path)
    ckpt = _load_ckpt(checkpoint_path, device)
    cfg = OmegaConf.create(ckpt["cfg"])
    if overrides: cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    name = str(getattr(cfg.exp, "model_name"))
    if name not in _MODEL: raise ValueError(f"Unknown model_name: {name}")
    model = _MODEL[name](cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, cfg


# ---------------- Plotting Functions ----------------
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

cmap = {
    'pp': '#6495ED',   # Cornflower Blue (for pianissimo)
    'p': '#9bbcff',    # Light Blue (for piano)
    'mp': '#d0e7ff',   # Very Light Blue (for mezzo-piano)
    'mf': '#cccccc',   # Neutral Gray (for mezzo-forte)
    'f': '#ffb3b3',    # Light Red (for forte)
    'ff': '#ff7f7f',   # Bright Red (for fortissimo)
    'fff': '#ff7f50',  # Coral (for fortississimo)    
    'blank': '#ffffff' # White (for blank/empty sections)
}

def plot_predicted_dynamics_h5(h5_path: str):
    """Minimal two-figure plotting from an inference H5.
    Figure A: color blocks over beats; Figure B: step curve over beats.
    """
    with h5py.File(h5_path, 'r') as f:
        fps = int(f.attrs["frames_per_second"]) if 'frames_per_second' in f.attrs else 50
        dyn_roll = f["pred_dynamic_roll"][:] if "pred_dynamic_roll" in f else None
        beat_time = f["pred_beat_time"][:] if "pred_beat_time" in f else np.array([])
        if "pred_dynamic_beat_class" in f:
            dyn_beat_cls = f["pred_dynamic_beat_class"][:]
        else:
            if dyn_roll is None or beat_time.size == 0:
                raise RuntimeError("Need pred_dynamic_roll and pred_beat_time in H5 or pred_dynamic_beat_class")
            idx = np.clip(np.round(beat_time * fps).astype(int), 0, len(dyn_roll) - 1)
            dyn_beat_cls = dyn_roll[idx]

    labels_5 = ['blank','pp','p','mf','f','ff']
    labels_8 = ['blank','ppp','pp','p','mp','mf','f','ff','fff']
    if dyn_beat_cls.max() <= 5:
        id2lab = labels_5
    elif dyn_beat_cls.max() <= 8:
        id2lab = labels_8
    else:
        K = int(dyn_beat_cls.max()) + 1
        id2lab = [f"c{i}" for i in range(K)]

    N = len(dyn_beat_cls)
    if N == 0:
        raise RuntimeError("No predicted beats found in H5")
    measures = [1]
    dyn_labels = [id2lab[int(dyn_beat_cls[0])]]
    for b in range(1, N):
        lab = id2lab[int(dyn_beat_cls[b])]
        if lab != dyn_labels[-1]:
            measures.append(b + 1)
            dyn_labels.append(lab)
    true_end = N
    full_measures = measures + [true_end]

    # Blocks
    fig, ax = plt.subplots(figsize=(14, 2.6), dpi=140)
    for i, dyn in enumerate(dyn_labels):
        x0 = full_measures[i]; x1 = full_measures[i + 1]
        ax.add_patch(patches.Rectangle((x0, 0.3), x1 - x0, 0.7,
                                       facecolor=cmap.get(dyn.lower(), '#dddddd'),
                                       edgecolor='black', linewidth=0.8))
        if dyn.lower() != 'blank':
            ax.text((x0 + x1) / 2, 0.65, dyn, ha='center', va='center', fontsize=10, weight='bold')
    ticks = [1] + list(range(20, ((true_end // 20) + 2) * 20, 20))
    ax.set_xlim(1, true_end)
    ax.set_xticks(ticks)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Score beat / Measure', fontsize=12)
    for s in ['top','right','left']:
        ax.spines[s].set_visible(False)
    plt.title("Predicted Dynamics vs Measure Timeline (beat domain)", fontsize=13)
    plt.tight_layout(); plt.show()

    # Curve
    dmap8 = {'ppp':0,'pp':1,'p':2,'mp':3,'mf':4,'f':5,'ff':6,'fff':7}
    def lab_to_val(l):
        l = l.lower()
        if l in dmap8: return dmap8[l]
        if l == 'blank': return 0
        order5 = ['pp','p','mf','f','ff']
        if l in order5: return 1 + order5.index(l)
        return np.nan

    x_pts, y_pts = [], []
    for i, dyn in enumerate(dyn_labels):
        yv = lab_to_val(dyn)
        if np.isnan(yv):
            continue
        x0, x1 = full_measures[i], full_measures[i + 1]
        x_pts += [x0, x1]
        y_pts += [yv, yv]

    fig, ax = plt.subplots(figsize=(14, 3), dpi=140)
    ax.plot(x_pts, y_pts, drawstyle='steps-post', color='black', linewidth=2)
    ax.set_xlim(1, true_end)
    yt_labs = ['ppp','pp','p','mp','mf','f','ff','fff']
    if any(l.lower() == 'ppp' for l in dyn_labels) or any(l.lower() == 'fff' for l in dyn_labels):
        ax.set_ylim(-0.5, 7.5); ax.set_yticks(range(0,8)); ax.set_yticklabels(yt_labs, fontsize=9)
    else:
        ax.set_ylim(-0.5, 5.5); ax.set_yticks(range(0,6)); ax.set_yticklabels(['blank','pp','p','mf','f','ff'], fontsize=9)
    ax.set_xlabel('Score beat / Measure', fontsize=11)
    ax.set_ylabel('MIDI Dynamic Level', fontsize=11)
    ax.set_title('Predicted Dynamics Curve (beat domain)', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.7)
    for s in ['top','right']:
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.show()
import os, re
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter, defaultdict
import seaborn as sns
from tqdm import tqdm
import h5py

from utils import load_dyn_markings, load_beat_times, load_discography_pid_metadata

# ----------- 公共设定 -----------
# plt.rcParams['figure.figsize'] = (2, 1)  # 设置所有plt.figure默认大小，宽8英寸，高4英寸

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


# === Global dyn_to_midi mapping ===
dmap = {
    'ppp': 0,
    'pp':  1,
    'p':   2,
    'mp':  3,
    'mf':  4,
    'f':   5,
    'ff':  6,
    'fff': 7
}


# ----------- 核心绘图函数 -----------

def plot_dynamics_vs_beats(mazurka_id, dyn_folder, beat_folder):
    measures, dynamics = load_dyn_markings(mazurka_id, dyn_folder)
    true_end = load_beat_times(mazurka_id, beat_folder)
    fig, ax = plt.subplots(figsize=(14, 2.5))
    full_measures = measures + [true_end]

    for i, dyn in enumerate(dynamics):
        ax.add_patch(patches.Rectangle((full_measures[i], 0.3), full_measures[i+1]-full_measures[i],
                                       0.7, facecolor=cmap.get(dyn.lower(), '#dddddd'), edgecolor='black'))
        if dyn.lower() != 'blank':
            ax.text((full_measures[i]+full_measures[i+1])/2, 0.65, dyn, ha='center', va='center', fontsize=10, weight='bold')

    ticks = [1] + list(range(20, ((true_end // 20) + 2) * 20, 20))
    ax.set_xlim(measures[0], true_end)
    ax.set_xticks(ticks)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Score beat / Measure', fontsize=12)
    for spine in ['top', 'right', 'left']: ax.spines[spine].set_visible(False)
    plt.title(f"Dynamics vs Measure Timeline: {mazurka_id}", fontsize=14)
    plt.tight_layout()
    plt.show()

    print(f"Dynamics - Measure change points for {mazurka_id}:")
    print(full_measures)


def plot_dynamics_vs_beats_as_curve(mazurka_id, dyn_folder, beat_folder):
    """
    绘制 Dynamics Markings 映射到 MIDI Velocity 后的变化折线图
    """

    # --- 不再在函数内部定义 dyn_to_midi，直接用全局 ---

    # --- 读取数据 ---
    measures, dynamics = load_dyn_markings(mazurka_id, dyn_folder)
    true_end = load_beat_times(mazurka_id, beat_folder)

    full_measures = measures + [true_end]
    full_dynamics = dynamics + [dynamics[-1]]

    # --- 构造曲线数据 ---
    x_points = []
    y_points = []

    for i in range(len(full_measures)-1):
        start = full_measures[i]
        end = full_measures[i+1]
        dyn = full_dynamics[i].lower()
        midi_value = dmap.get(dyn, np.nan)

        if not np.isnan(midi_value):
            x_points += [start, end]
            y_points += [midi_value, midi_value]

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(14, 3))

    ax.plot(x_points, y_points, drawstyle='steps-post', color='black', linewidth=2)
    ax.set_xlim(min(x_points), max(x_points))
    ax.set_ylim(-1, 8)

    ax.set_yticks(range(0, 8))
    ax.set_yticklabels(['ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff'], fontsize=9)
    ax.set_xlabel('Score beat / Measure', fontsize=11)
    ax.set_ylabel('MIDI Dynamic Level', fontsize=11)
    ax.set_title(f'Dynamics to MIDI Velocity Curve: {mazurka_id}', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.7)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_sonic_curve(csv_path):
    """
    读取 [时间-能量] CSV 文件并绘制声学曲线，x轴从0开始
    """
    df = pd.read_csv(csv_path, header=None, names=['time', 'sonic_value'])

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df['time'], df['sonic_value'], color='black', linewidth=1)

    ax.set_xlim(left=0)  # <<< 确保从 0 秒开始画
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Sonic Value', fontsize=11)
    ax.set_title(f'Sonic Energy Curve: {os.path.basename(csv_path)}', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.7)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_dynamics_legend():
    fig, ax = plt.subplots(figsize=(5, 1))
    for idx, label in enumerate(['pp', 'p', 'mf', 'f', 'ff']):
        ax.add_patch(patches.Rectangle((idx, 0), 1, 1, facecolor=cmap[label], edgecolor='black'))
        ax.text(idx+0.5, 0.5, label, ha='center', va='center', fontsize=6, weight='bold')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('Dynamic Markings Color Legend', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_waveform_with_dynamics(mazurka_id, pid, dyn_folder, beat_folder, audio_folder, discography_file, sones_folder, highlight_intervals=None):
    # === Load Data ===
    measures, dynamics = load_dyn_markings(mazurka_id, dyn_folder)
    beat_times = load_beat_times(mazurka_id, beat_folder, pid)
    opus, performer, true_end = load_discography_pid_metadata(discography_file, pid)

    seconds = [beat_times[b] for b in measures]
    full_seconds, full_dynamics = seconds + [true_end], dynamics + [dynamics[-1]]
    start_time, end_time = round(beat_times[0], 2), round(true_end, 2)

    audio_path = os.path.join(audio_folder, f"mazurka{mazurka_id[1:].lower()}", f"{pid}.wav")
    sones_path = os.path.join(sones_folder, mazurka_id, f"{pid}Ntot.csv")
    y, sr = librosa.load(audio_path, sr=None)
    sones_df = pd.read_csv(sones_path, header=None, names=['time', 'sonic_value'])
    duration = librosa.get_duration(y=y, sr=sr)

    if abs(duration - true_end) > 5:
        print(f"Warning: Audio duration ({duration:.1f}s) and discography time ({true_end}s) differ.")

    # === Plotting ===
    # Note: do NOT share x across all plots because we want seconds for (1-3)
    # and beats-in-score for (4)
    # Make plot 2 & 3 as tall as plot 1 (equal heights for 1-3)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(10, 7.2),
        gridspec_kw={'height_ratios': [1, 0.7, 0.8, 0.3]}, sharex=False
    ) # 调整4个plot的高度

    times = np.linspace(0, duration, len(y))
    ax1.plot(times, y, color='black', lw=0.6)
    ax2.plot(sones_df['time'], sones_df['sonic_value'], color='grey', lw=0.6)

    # Dynamic MIDI curve
    x_curve, y_curve = [], []
    for i, dyn in enumerate(full_dynamics[:-1]):
        midi_val = dmap.get(dyn.lower(), np.nan)
        if not np.isnan(midi_val):
            x_curve += [full_seconds[i], full_seconds[i+1]]
            y_curve += [midi_val, midi_val]
    # --- Cosine similarity calculation ---
    similarity = calculate_cosine_similarity_sones_vs_dynamics(sones_df, x_curve, y_curve)
    print(f"Cosine Similarity (Sones vs Dynamics): {similarity:.4f}")
    ax3.plot(x_curve, y_curve, drawstyle='steps-post', color='black', lw=1.2)
    # Restrict to pp..ff (remove ppp/fff)
    ax3.set_ylim(0, 7)
    ax3.set_yticks(range(1, 7))
    ax3.set_yticklabels(['pp', 'p', 'mp', 'mf', 'f', 'ff'], fontsize=12)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    # Dynamic color blocks
    # Total beats in score for beat-axis range
    n_beats_score = load_beat_times(mazurka_id, beat_folder, pid=None)

    for i, dyn in enumerate(full_dynamics[:-1]):
        color = cmap.get(dyn.lower(), '#dddddd')
        # For plot 4 we switch to beats-in-score on x-axis
        # Build beat-domain locations
        # 'measures' is a list of beat indices, map them to beat-axis directly
        beat_start = measures[i]
        beat_end = measures[i+1] if i+1 < len(measures) else n_beats_score
        ax4.add_patch(patches.Rectangle((beat_start, 0), beat_end - beat_start, 1,
                                        facecolor=color, edgecolor='black'))
        if dyn != 'blank':
            # Slight vertical staggering to reduce overlap between adjacent labels (e.g., pp vs p, f vs ff)
            offset_map = {
                'pp':  0.18,
                'p':  -0.12,
                'mp':  0.09,
                'mf': -0.18,
                'f':   0.12,
                'ff': -0.09,
            }
            y_base = 0.5 + offset_map.get(dyn.lower(), 0.0)
            y_pos = min(max(y_base, 0.08), 0.92)
            ax4.text((beat_start + beat_end) / 2, y_pos, dyn,
                     ha='center', va='center', fontsize=11, weight='bold')

    # Highlight Intervals
    if highlight_intervals:
        # For time-domain axes (ax1, ax2, ax3), highlight spans in seconds directly
        for ax in [ax1, ax2, ax3]:
            for (start, end) in highlight_intervals:
                ax.axvspan(start, end, color='yellow', alpha=0.3)
        # For beat-domain axis (ax4), convert seconds -> beats via interpolation
        beat_idx_axis = np.arange(1, len(beat_times) + 1, dtype=float)
        beat_time_axis = np.array(beat_times, dtype=float)
        for (start, end) in highlight_intervals:
            b_start = float(np.interp(start, beat_time_axis, beat_idx_axis))
            b_end = float(np.interp(end, beat_time_axis, beat_idx_axis))
            ax4.axvspan(b_start, b_end, color='yellow', alpha=0.3)
    # --- Decoration ---
    # Time-axis ticks for plots 1-3
    time_ticks = list(range(0, int(duration)+1, 25))
    if int(duration) not in time_ticks:
        time_ticks.append(int(duration))
    time_ticks = sorted(set(time_ticks))

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, duration)
        ax.set_xticks(time_ticks)
    # Keep y-axis visible on ax1 and ax2; hide only top/right spines
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    # For ax3, keep previous minimalist style (hide left as well)
    for spine in ['top', 'right', 'left']:
        ax3.spines[spine].set_visible(False)

    # Beat-axis ticks for plot 4
    beat_end = int(n_beats_score) if n_beats_score else 0
    beat_ticks = [1] + list(range(20, ((beat_end // 20) + 2) * 20, 20)) if beat_end >= 1 else []
    ax4.set_xlim(left=measures[0] if len(measures) else 0, right=beat_end if beat_end else 1)
    if beat_ticks:
        ax4.set_xticks(beat_ticks)
    for spine in ['top', 'right', 'left']:
        ax4.spines[spine].set_visible(False)

    # Remove only y-axis labels for waveform and sonic plots (keep ticks/values)
    ax1.set_ylabel('')
    ax2.set_ylabel('')
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    # X labels: first plot in seconds, fourth plot in beats-in-score
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax4.set_xlabel('Beats in score', fontsize=11)
    # Slightly increase axis number font sizes overall
    for a in [ax1, ax2, ax3, ax4]:
        a.tick_params(labelsize=10)

    # Increase vertical spacing between the four subplots
    fig.subplots_adjust(hspace=0.6, top=0.92)

    # Title: include selected interval(s) and total duration
    total_sec = int(round(duration))
    if highlight_intervals and len(highlight_intervals) > 0:
        sel_parts = [f"{int(round(s))} - {int(round(e))}s" for (s, e) in highlight_intervals]
        sel_str = ", ".join(sel_parts)
        title_str = f"Mazurka {mazurka_id} | pid {pid} | Selected {sel_str} | Total duration {total_sec}s"
    else:
        title_str = f"Mazurka {mazurka_id} | pid {pid} | Total duration {total_sec}s"
    plt.suptitle(title_str, fontsize=11)
    
# ----------- Cosine Similarity Function -----------

# Place at end of file
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity_sones_vs_dynamics(sones_df, dynamic_times, dynamic_values):
    """
    sones_df: DataFrame with columns ['time', 'sonic_value']
    dynamic_times: list of timestamps (in seconds) corresponding to dynamics changes
    dynamic_values: list of dynamic values (mapped to numeric)

    This function interpolates the dynamics curve to match sones_df['time'] sampling,
    and computes cosine similarity between sones and interpolated dynamics curve.
    """
    import numpy as np

    # Interpolate dynamic values over sones time
    dynamics_interp = np.interp(sones_df['time'].values, dynamic_times, dynamic_values)

    # Normalize both vectors
    sones_norm = (sones_df['sonic_value'].values - np.mean(sones_df['sonic_value'].values)) / (np.std(sones_df['sonic_value'].values) + 1e-8)
    dynamics_norm = (dynamics_interp - np.mean(dynamics_interp)) / (np.std(dynamics_interp) + 1e-8)

    # Compute cosine similarity
    sim = np.dot(sones_norm, dynamics_norm) / (np.linalg.norm(sones_norm) * np.linalg.norm(dynamics_norm) + 1e-8)
    return sim
# ----------- 数据集检查工具 -----------


def plot_mazurka_statistics(dynmark_folder, markings_folder, mode='score_changepoints'):
    """
    mode可以选择：'score_changepoints', 'score_measures', 'score_beats',
                  'perf_changepoints', 'perf_measures', 'perf_beats'
    """

    dynmark_files = sorted(f for f in os.listdir(dynmark_folder) if f.endswith('dynmark.csv'))
    print(f"Found {len(dynmark_files)} dynmark CSV files.")

    mazurka_ids = []
    counts = []

    for dynmark_file in dynmark_files:
        mazurka_id = dynmark_file.replace('dynmark.csv', '')
        dynmark_path = os.path.join(dynmark_folder, dynmark_file)
        markings_path = os.path.join(markings_folder, f"{mazurka_id}markings.csv")

        df_dyn = pd.read_csv(dynmark_path)

        # --- 计算 performances数量 ---
        pids = [col for col in df_dyn.columns if col.startswith('pid')]
        n_performances = len(pids)

        # --- 计算 measures数量 ---
        measures = df_dyn['measure_number'].dropna()
        n_measures = len(set(measures))

        # --- 计算 beats数量 ---
        n_beats = len(df_dyn)

        # --- 计算 changepoints数量 ---
        if os.path.exists(markings_path):
            df_mark = pd.read_csv(markings_path, header=None)
            if df_mark.shape[0] >= 2:
                beats_mark = df_mark.iloc[1].dropna().astype(int).tolist()
                n_changepoints = len(beats_mark)
            else:
                n_changepoints = 0
        else:
            print(f"Warning: {markings_path} not found. Set changepoints=0.")
            n_changepoints = 0

        # --- 根据mode选择 ---
        if mode == 'score_changepoints':
            count = n_changepoints
        elif mode == 'score_measures':
            count = n_measures
        elif mode == 'score_beats':
            count = n_beats
        elif mode == 'perf_changepoints':
            count = n_changepoints * n_performances
        elif mode == 'perf_measures':
            count = n_measures * n_performances
        elif mode == 'perf_beats':
            count = n_beats * n_performances
        else:
            raise ValueError(f"Invalid mode: {mode}")

        mazurka_ids.append(mazurka_id)
        counts.append(count)

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(max(6, len(mazurka_ids)*0.3), 4))
    ax.bar(mazurka_ids, counts, color="#6495ED", edgecolor='black')

    ax.set_ylabel('Count', fontsize=10)
    ax.set_xlabel('Mazurka Pieces', fontsize=10)
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    sns.despine()
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(fontsize=8)

    title_map = {
        'score_changepoints': 'Dynamic Changes per Score',
        'score_measures': 'Unique Measures per Score',
        'score_beats': 'Beats per Score',
        'perf_changepoints': 'Dynamic Changes per Performance',
        'perf_measures': 'Measures per Performance',
        'perf_beats': 'Beats per Performance'
    }
    plt.title(f"Mazurka Dataset Statistics: {title_map[mode]}", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_dynamic_distribution(dynmark_folder, markings_folder, flag='score_changepoints'):
    """
    flag可以是:
    'score_changepoints', 'score_measures', 'score_beats',
    'perf_changepoints', 'perf_measures', 'perf_beats'
    """

    dynmark_files = sorted(f for f in os.listdir(dynmark_folder) if f.endswith('dynmark.csv'))
    print(f"Found {len(dynmark_files)} dynmark CSV files.")

    # --- 动态标签分类 ---
    valid_labels = ['pp', 'p', 'mf', 'f', 'ff']

    counter = Counter()

    for dynmark_file in dynmark_files:
        mazurka_id = dynmark_file.replace('dynmark.csv', '')
        dynmark_path = os.path.join(dynmark_folder, dynmark_file)
        markings_path = os.path.join(markings_folder, f"{mazurka_id}markings.csv")

        df_dyn = pd.read_csv(dynmark_path)

        pids = [col for col in df_dyn.columns if col.startswith('pid')]
        n_performances = len(pids)

        # 统计 changepoints
        if os.path.exists(markings_path):
            df_mark = pd.read_csv(markings_path, header=None)
            if df_mark.shape[0] >= 3:
                dynamics_cp = df_mark.iloc[2].dropna().astype(str).str.strip().str.lower().tolist()
            else:
                dynamics_cp = []
        else:
            dynamics_cp = []

        # 统计 measures
        dynamics_measure = df_dyn['dynamic_mark'].dropna().astype(str).str.strip().str.lower().tolist()
        # 统计 beats
        dynamics_beat = df_dyn['dynamic_mark'].dropna().astype(str).str.strip().str.lower().tolist()

        if flag.startswith('score'):
            factor = 1
        elif flag.startswith('perf'):
            factor = n_performances
        else:
            raise ValueError(f"Invalid flag: {flag}")

        if 'changepoints' in flag:
            dynamics = dynamics_cp
        elif 'measures' in flag:
            dynamics = dynamics_measure
        elif 'beats' in flag:
            dynamics = dynamics_beat
        else:
            raise ValueError(f"Invalid flag: {flag}")

        # 更新counter
        for dyn in dynamics:
            dyn = dyn.lower()
            dyn = dyn if dyn in valid_labels else 'uncommon'
            counter[dyn] += factor

    # --- 整理绘图 ---
    labels_order = ['pp', 'p', 'mf', 'f', 'ff', 'uncommon']
    counts = [counter.get(label, 0) for label in labels_order]
    bar_colors = [cmap.get(label, '#dddddd') for label in labels_order]

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(labels_order, counts, color=bar_colors, edgecolor='black')

    ax.set_ylabel('Count', fontsize=10)
    ax.set_xlabel('Dynamic Markings', fontsize=10)
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    sns.despine()
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    title_map = {
        'score_changepoints': 'Score-level Dynamic Changes',
        'score_measures': 'Score-level Measures',
        'score_beats': 'Score-level Beats',
        'perf_changepoints': 'Performance-level Dynamic Changes',
        'perf_measures': 'Performance-level Measures',
        'perf_beats': 'Performance-level Beats'
    }
    plt.title(title_map[flag], fontsize=12)
    plt.tight_layout()
    plt.show()


# ==============================
# MMoE Gates Heatmap Plotting
# ==============================

def _mmoe_gate_key_for_task(task: str) -> str:
    task = task.strip().lower()
    mapping = {
        'dynamic': 'mmoe_gates_dynamic',
        'beat': 'mmoe_gates_beat',
        'downbeat': 'mmoe_gates_downbeat',
        'change_point': 'mmoe_gates_change_point',
        'changepoint': 'mmoe_gates_change_point',
    }
    if task not in mapping:
        raise ValueError(f"Unknown task '{task}'. Valid: dynamic|beat|downbeat|change_point")
    return mapping[task]


def plot_mmoe_gates_heatmap(h5_path: str, task: str = 'dynamic', save_path: str | None = None, vmax: float | None = None):
    """
    Plot MMoE gating probabilities as a heatmap for one task.

    - h5_path: path to model output H5 exported by inference.py
    - task: one of ['dynamic','beat','downbeat','change_point']
    - save_path: if provided, save the figure to this path (PNG) instead of showing
    - vmax: cap colorbar upper bound; defaults to 1.0 for probabilities
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py

    key = _mmoe_gate_key_for_task(task)
    with h5py.File(h5_path, 'r') as hf:
        if key not in hf:
            raise KeyError(f"Dataset '{key}' not found in {h5_path}. Ensure cnn.export_mmoe_heatmap=true and MMoE enabled.")
        gates = hf[key][:]  # [T, E]
        fps = float(hf.attrs.get('frames_per_second', 50.0))

    if gates.ndim != 2:
        raise RuntimeError(f"Expected 2D gates [T,E], got shape {gates.shape}")

    T, E = gates.shape
    times = np.arange(T) / fps

    fig, ax = plt.subplots(figsize=(max(8, int(T / fps) // 5 + 8), 2 + 0.2 * E))
    im = ax.imshow(gates.T, aspect='auto', origin='lower', interpolation='nearest',
                   extent=[times[0], times[-1] if T>0 else 0, 0, E], vmin=0.0, vmax=(vmax or 1.0), cmap='magma')
    ax.set_ylabel('Expert index')
    ax.set_xlabel('Time (s)')
    ax.set_title(f"MMoE Gates Heatmap — Task: {task}")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Gating prob.')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_mmoe_gates_all_tasks(h5_path: str, output_dir: str | None = None):
    """
    Plot heatmaps for all available tasks in the given H5.
    If output_dir is provided, save PNGs there; otherwise display interactively.
    """
    tasks = ['dynamic', 'beat', 'downbeat', 'change_point']
    for t in tasks:
        key = _mmoe_gate_key_for_task(t)
        try:
            if output_dir:
                base = os.path.splitext(os.path.basename(h5_path))[0]
                save = os.path.join(output_dir, f"{base}_mmoe_{t}.png")
                plot_mmoe_gates_heatmap(h5_path, task=t, save_path=save)
            else:
                plot_mmoe_gates_heatmap(h5_path, task=t, save_path=None)
        except KeyError:
            # Skip missing task
            continue


def stat_and_plot_dynamic_distribution(dynmark_folder, markings_folder):
    """
    统计动态标记分布 支持changepoints、measures、beats统计
    绘制百分比堆叠柱状图，显示计数表格
    """
    dynmark_files = sorted(f for f in os.listdir(dynmark_folder) if f.endswith('dynmark.csv'))
    print(f"Found {len(dynmark_files)} dynmark CSV files.")

    valid_labels = ['pp', 'p', 'mf', 'f', 'ff']
    all_labels = valid_labels + ['uncommon']
    flags = ['score_changepoints', 'score_measures', 'score_beats',
             'perf_changepoints', 'perf_measures', 'perf_beats']

    flag_counter_dict = {flag: Counter() for flag in flags}

    for dynmark_file in dynmark_files:
        mazurka_id = dynmark_file.replace('dynmark.csv', '')
        dynmark_path = os.path.join(dynmark_folder, dynmark_file)
        markings_path = os.path.join(markings_folder, f"{mazurka_id}markings.csv")

        df_dyn = pd.read_csv(dynmark_path)
        pids = [col for col in df_dyn.columns if col.startswith('pid')]
        n_perf = len(pids)

        # Load dynamics from markings.csv (changepoints)
        dynamics_cp = []
        if os.path.exists(markings_path):
            df_mark = pd.read_csv(markings_path, header=None)
            if df_mark.shape[0] >= 3:
                dynamics_cp = df_mark.iloc[2].dropna().astype(str).str.strip().str.lower().tolist()

        # dynamics from our dynmark.csv
        dynamics_seq = df_dyn['dynamic_mark'].dropna().astype(str).str.strip().str.lower().tolist()
        unique_measures = df_dyn['measure_number'].dropna().unique()

        # build mappings
        flag2dyn = {
            'score_changepoints': (dynamics_cp, 1),
            'score_measures': (dynamics_seq, 1),
            'score_beats': (dynamics_seq, 1),
            'perf_changepoints': (dynamics_cp, n_perf),
            'perf_measures': (dynamics_seq, n_perf),
            'perf_beats': (dynamics_seq, n_perf),
        }

        for flag in flags:
            dynamics, factor = flag2dyn[flag]
            if 'measures' in flag:
                dynamics = [df_dyn[df_dyn['measure_number'] == m]['dynamic_mark'].iloc[0] for m in unique_measures]
            for dyn in dynamics:
                dyn = dyn if dyn in valid_labels else 'uncommon'
                flag_counter_dict[flag][dyn] += factor

    # --- build DataFrame ---
    df = pd.DataFrame(flag_counter_dict).T
    df = df[all_labels].fillna(0).astype(int)

    # --- show table ---
    print("\n=== Dynamic Markings Distribution Table ===")
    display(df)

    # # --- plot percent stacked bar ---
    # df_percent = df.div(df.sum(axis=1), axis=0) * 100

    # fig, ax = plt.subplots(figsize=(8, 3))  # 更小图尺寸
    # bottom = np.zeros(len(df_percent))

    # for label in all_labels:
    #     color = cmap.get(label, '#dddddd')
    #     ax.bar(df_percent.index, df_percent[label], bottom=bottom, label=label, color=color, edgecolor='black')
    #     bottom += df_percent[label]

    # ax.set_ylabel('Percentage (%)', fontsize=8)
    # ax.set_xlabel('Statistics Mode', fontsize=8)
    # ax.set_title('Dynamic Markings Distribution (Percentage)', fontsize=10)
    # ax.set_xticklabels(df.index, rotation=45, ha='right', fontsize=8)
    # ax.tick_params(axis='y', labelsize=8)
    # ax.legend(title='Dynamic Labels', fontsize=7, title_fontsize=8)

    # ax.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()

    return df

def plot_dynamic_distribution_count(df_counts):
    """
    Plot stacked bar charts (Counts version) for Score and Performance separately.
    Each subplot has independent Y-axis scaling and its own legend.
    """
    score_flags = [f for f in df_counts.index if f.startswith('score')]
    perf_flags = [f for f in df_counts.index if f.startswith('perf')]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, flags, title in zip([ax1, ax2], [score_flags, perf_flags], ['Score Data', 'Performance Data']):
        x = np.arange(len(flags))
        bottom = np.zeros(len(flags))
        for label in df_counts.columns:
            ax.bar(x, df_counts.loc[flags, label], bottom=bottom, label=label, 
                   color=cmap.get(label, '#dddddd'), edgecolor='black')
            bottom += df_counts.loc[flags, label]
        ax.set_title(title, fontsize=10)
        # ax.set_xlabel('Statistics Mode', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(flags, rotation=45, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, bottom.max() * 1.1)
        ax.legend(title='Dynamics', fontsize=8, title_fontsize=9, loc='upper left')

    fig.supylabel('Counts', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_dynamic_distribution_percent(df_counts):
    """
    Plot stacked bar charts (Percentage version) for Score and Performance separately.
    Each subplot is normalized to 100% within its own group.
    """
    score_flags = [f for f in df_counts.index if f.startswith('score')]
    perf_flags = [f for f in df_counts.index if f.startswith('perf')]

    df_score = df_counts.loc[score_flags].div(df_counts.loc[score_flags].sum(axis=1), axis=0) * 100
    df_perf = df_counts.loc[perf_flags].div(df_counts.loc[perf_flags].sum(axis=1), axis=0) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, df_plot, flags, title in zip([ax1, ax2], [df_score, df_perf], [score_flags, perf_flags], 
                                         ['Score Data (Percentage)', 'Performance Data (Percentage)']):
        x = np.arange(len(flags))
        bottom = np.zeros(len(flags))
        for label in df_plot.columns:
            ax.bar(x, df_plot[label], bottom=bottom, label=label,
                   color=cmap.get(label, '#dddddd'), edgecolor='black')
            bottom += df_plot[label]
        ax.set_title(title, fontsize=10)
        # ax.set_xlabel('Statistics Mode', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(flags, rotation=45, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 100)
        ax.legend(title='Dynamics', fontsize=8, title_fontsize=9, loc='upper left')

    fig.supylabel('Percentage (%)', fontsize=10)
    plt.tight_layout()
    plt.show()

################### 检索函数 ####################

def check_dynamic_markings_in_folder(dyn_folder):
    counter = Counter()
    csv_files = sorted(f for f in os.listdir(dyn_folder) if f.endswith('.csv'))
    print(f"Found {len(csv_files)} CSV files.")

    for file in csv_files:
        df = pd.read_csv(os.path.join(dyn_folder, file), header=None)
        if df.shape[0] >= 3:
            counter.update(df.iloc[2].dropna().astype(str).str.strip().str.lower())
        else:
            print(f"Warning: {file} has less than 3 rows. Skipped.")

    print("\nOverall dynamic labels found:")
    for label, count in counter.most_common():
        print(f"{label}: {count}")


def find_invalid_dynamic_markings(dyn_folder, invalid_labels):
    label_file_map = defaultdict(list)
    csv_files = sorted(f for f in os.listdir(dyn_folder) if f.endswith('.csv'))
    print(f"Found {len(csv_files)} CSV files.")

    for file in csv_files:
        df = pd.read_csv(os.path.join(dyn_folder, file), header=None)
        if df.shape[0] >= 3:
            dynamics = df.iloc[2].dropna().astype(str).str.strip().str.lower()
            for dyn in dynamics:
                if dyn in invalid_labels:
                    label_file_map[dyn].append(file)
        else:
            print(f"Warning: {file} has less than 3 rows. Skipped.")

    print("\nInvalid dynamics labels found:")
    for label in invalid_labels:
        if label_file_map[label]:
            print(f"Label '{label}' found in: {', '.join(label_file_map[label])}")
        else:
            print(f"Label '{label}': Not found.")


            
def find_pid_in_which_mazurka_csv(beat_time_folder, partial_pid):
    """
    Find which CSV files in a folder contain a column with the given PID.

    Parameters:
        beat_time_folder (str): Path to the folder containing beat_time CSV files.
        partial_pid (str): Substring to match in column names (e.g., '6100022').

    Returns:
        List[Tuple[str, str]]: List of (csv_filename, full_pid_column_name) where match is found.
        Empty if pid not found in any CSV.
    """
    matched = []
    normalized_target = re.sub(r'\D', '', partial_pid)  # 去除目标 pid 中的非数字字符


    for filename in os.listdir(beat_time_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(beat_time_folder, filename)
            try:
                df = pd.read_csv(file_path, nrows=1)  # 只读取 header
                for col in df.columns:
                    if 'pid' in col.lower():
                        normalized_col = re.sub(r'\D', '', col)  # 去除列名中的非数字字符
                        if normalized_target in normalized_col:
                            matched.append((filename, col))
            except Exception as e:
                print(f"[ERROR] 无法读取 {filename}: {e}")

    return matched


def check_markings_start_index(dyn_folder):
    """
    Check all CSVs in the given folder.
    Each CSV should have:
      - Row 0: complex dyn marks
      - Row 1: time
      - Row 2: simple dyn marks
    This function checks if the time row starts from 1.
    Only prints files that do NOT start from 1.
    """
    csv_files = sorted(f for f in os.listdir(dyn_folder) if f.endswith('.csv'))
    print(f"Found {len(csv_files)} CSV files.")

    for file in csv_files:
        df = pd.read_csv(os.path.join(dyn_folder, file), header=None)
        if df.shape[0] >= 2:
            time_row = df.iloc[1].dropna().tolist()
            if len(time_row) > 0:
                first_time = time_row[0]
                if not np.isclose(float(first_time), 1.0):
                    print(f"{file}: starts from {first_time} instead of 1")
        else:
            print(f"{file}: less than 2 rows, skipped")


def inspect_hdf5_file(h5_path):
    print(f"Inspecting HDF5 file: {h5_path}")
    
    with h5py.File(h5_path, 'r') as hf:
        print("\n📂 Datasets:")
        for key in hf.keys():
            data = hf[key]
            print(f" - {key:25s} shape: {data.shape} dtype: {data.dtype}")

        print("\n🔖 Attributes:")
        for attr in hf.attrs:
            value = hf.attrs[attr]
            if isinstance(value, bytes):
                value = value.decode()
            print(f" - {attr:25s} value: {value}")



# ----------- New function: find_blank_dynamics_in_folder -----------
def find_blank_dynamics_in_folder(dyn_folder):
    """
    Check all meta CSVs in the given folder.
    Report which files have 'blank' in their dynamic_mark column,
    and detail the beat_index ranges with 'blank' dynamics.
    """
    csv_files = sorted(f for f in os.listdir(dyn_folder) if f.endswith('.csv'))
    print(f"Checked {len(csv_files)} CSV files.")

    for file in csv_files:
        df = pd.read_csv(os.path.join(dyn_folder, file))

        if 'beat_index' in df.columns and 'dynamic_mark' in df.columns:
            beats = df['beat_index'].tolist()
            dynamics = df['dynamic_mark'].astype(str).str.strip().str.lower().tolist()

            if 'blank' in dynamics:
                print(f"\n{file} contains 'blank' dynamics:")
                ranges = []
                start = None

                for i, dyn in enumerate(dynamics):
                    if dyn == 'blank':
                        if start is None:
                            start = beats[i]
                    else:
                        if start is not None:
                            end = beats[i-1]
                            ranges.append((start, end))
                            start = None

                if start is not None:
                    end = beats[-1]
                    ranges.append((start, end))

                for s, e in ranges:
                    print(f"  blank from beat {s} to {e}")
        else:
            print(f"Warning: {file} missing 'beat_index' or 'dynamic_mark'. Skipped.")


def check_pid_consistency(beat_time_csv, discography_txt):
    """
    Check which PIDs in a beat_time CSV exist in the discography.txt,
    but only within the same opus.
    Report matching and mismatching PIDs.
    """

    # Load beat_time CSV header
    df = pd.read_csv(beat_time_csv, nrows=1)
    pids = [col for col in df.columns if col.startswith('pid')]
    stripped_pids = [pid.replace('pid', '') for pid in pids]

    # Infer opus from CSV filename: e.g., M17-4beat_time.csv -> '17-4'
    opus_match = re.search(r'M(\d+-\d+)', os.path.basename(beat_time_csv))
    opus = opus_match.group(1).replace('-', '.') if opus_match else None

    if opus is None:
        print(f"Could not infer opus from filename: {beat_time_csv}")
        return

    # Load discography.txt and filter to this opus only
    df_disc = pd.read_csv(discography_txt, delimiter='\t')
    df_disc_this_opus = df_disc[df_disc['opus'].astype(str) == opus]

    disc_pids = df_disc_this_opus['pid'].astype(str).tolist()

    matching = [pid for pid in stripped_pids if pid in disc_pids]

    print(f"\nChecked {beat_time_csv} against {discography_txt} for opus {opus}")
    print(f"Total pids in CSV: {len(pids)}")
    print(f"Matching pids: {matching}")

    extra_in_csv = [pid for pid in stripped_pids if pid not in disc_pids]
    extra_in_disc = [pid for pid in disc_pids if pid not in stripped_pids]

    print(f"[Error] PIDs in CSV but NOT in discography: {extra_in_csv}")
    print("\nWe allow PIDs in discography but not in CSV")
    print(f"[Allow] PIDs in discography but NOT in CSV: {extra_in_disc}")


def compare_sones_diff_methods(csv_files, titles):
    """
    Draw a 3x1 subplot to compare Ntot from multiple CSV files.
    
    Parameters:
        csv_files (list): list of CSV file paths
        titles (list): list of subplot titles
    """
    assert len(csv_files) == len(titles) and len(csv_files) >= 2, "Number of CSV files and titles must match, and at least 2."
    
    dfs = [pd.read_csv(f) for f in csv_files]

    fig, axes = plt.subplots(len(csv_files), 1, figsize=(12, 2.5 * len(csv_files)), sharex=True)
    # If only one file, axes is not a list, so wrap it
    if len(csv_files) == 1:
        axes = [axes]

    for ax, df, title in zip(axes, dfs, titles):
        ax.plot(df.iloc[:, 0], df.iloc[:, 1], label='Ntot Energy')
        ax.set_title(title)
        ax.set_ylabel('Ntot')
        ax.grid(True)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    
def batch_cosine_similarity_stat(dyn_folder, beat_folder, audio_folder, discography_file, sones_folder, bin_width=0.1):
    """
    Loop over all Mazurka-PID pairs in dyn_folder and calculate cosine similarity.
    Aggregate counts per bin and compute per-Mazurka average.
    """

    csv_files = sorted(f for f in os.listdir(dyn_folder) if f.endswith('.csv'))
    print(f"Found {len(csv_files)} dyn CSV files.")

    pid_cosine_list = []  # (mazurka_id, pid, cosine_sim)

    for file in csv_files:
        mazurka_id = file.replace('markings.csv', '')
        df = pd.read_csv(os.path.join(dyn_folder, file), header=None)

        if df.shape[0] >= 3:
            # Get pid columns from beat_folder files
            beat_time_file = os.path.join(beat_folder, f"{mazurka_id}beat_time.csv")
            beat_df = pd.read_csv(beat_time_file)
            pid_cols = [col for col in beat_df.columns if col.startswith('pid')]

            for pid in pid_cols:
                try:
                    # Reuse existing plot function but disable plotting (wrap with try-except)
                    measures, dynamics = load_dyn_markings(mazurka_id, dyn_folder)
                    beat_times = load_beat_times(mazurka_id, beat_folder, pid)
                    opus, performer, true_end = load_discography_pid_metadata(discography_file, pid)

                    valid_seconds = []
                    for b in measures:
                        idx = b - 1  # Convert measure number to 0-based index
                        if 0 <= idx < len(beat_times):
                            valid_seconds.append(beat_times[idx])
                        else:
                            print(f"[WARN] Beat index {idx} out of range for {mazurka_id} - {pid}, skipping this PID.")
                            valid_seconds = []
                            break  # Skip this PID if any measure is out of range

                    if not valid_seconds:
                        continue

                    full_seconds, full_dynamics = valid_seconds + [true_end], dynamics + [dynamics[-1]]

                    x_curve, y_curve = [], []
                    for i, dyn in enumerate(full_dynamics[:-1]):
                        midi_val = dmap.get(dyn.lower(), np.nan)
                        if not np.isnan(midi_val):
                            x_curve += [full_seconds[i], full_seconds[i+1]]
                            y_curve += [midi_val, midi_val]

                    sones_path = os.path.join(sones_folder, mazurka_id, f"{pid}Ntot.csv")
                    if not os.path.exists(sones_path):
                        print(f"Warning: Sones file not found for pid {pid} of {mazurka_id}. Skipping.")
                        continue
                    sones_df = pd.read_csv(sones_path, header=None, names=['time', 'sonic_value'])

                    cos_sim = calculate_cosine_similarity_sones_vs_dynamics(sones_df, x_curve, y_curve)
                    pid_cosine_list.append((mazurka_id, pid, cos_sim))

                except Exception as e:
                    print(f"Error processing {mazurka_id}-{pid}: {e}")
                    continue
        else:
            print(f"Warning: {file} has less than 3 rows. Skipped.")

    # --- Bin count ---
    bins = np.arange(-1.0, 1.0 + bin_width, bin_width)
    bin_labels = [f"{round(b,2)}~{round(b+bin_width,2)}" for b in bins[:-1]]
    bin_counter = Counter()

    for _, _, cos_val in pid_cosine_list:
        bin_idx = np.digitize([cos_val], bins) - 1
        bin_label = bin_labels[bin_idx[0]] if 0 <= bin_idx[0] < len(bin_labels) else 'out_of_range'
        bin_counter[bin_label] += 1

    # --- Mazurka average ---
    mazurka_avg = {}
    for mazurka_id in set(m for m, _, _ in pid_cosine_list):
        vals = [cos for m, _, cos in pid_cosine_list if m == mazurka_id]
        mazurka_avg[mazurka_id] = np.mean(vals) if vals else np.nan

    # --- Output ---
    print("\n=== Cosine Similarity Bin Counts ===")
    for bin_label in sorted(bin_counter.keys()):
        print(f"{bin_label}: {bin_counter[bin_label]}")

    print("\n=== Per Mazurka Average Cosine Similarity ===")
    for mazurka_id, avg_val in sorted(mazurka_avg.items()):
        print(f"{mazurka_id}: {avg_val:.4f}")

    return pid_cosine_list, bin_counter, mazurka_avg
"""
We found some data issues in the MazurkaBL-master dataset, refer to problem_in_data.ipynb

- Issue1: Unusual dynamic markings occur in some mazurkas, 
    e.g., M63-2.
    @ Policy clarification:
        We exclude these unusual mazurkas from the dataset.

- Issue2: Some pid columns in beat_time and beat_dyn CSVs are incorrect, 
    e.g., "pid9070b-01" should be "pid9070b-09" in M41-1.
    @ Policy clarification:
        We manually fix this by renaming the pid matching the correct performance ID.

- Issue3: Some mazurkas first annotated dynamic_marking is not starting from the first beat, 
    e.g., M17-3
    @ Policy clarification:
        1. Frames BEFORE the first *beat time* are BLANK
        2. Beat indices from first beat up to the first annotated dynamic beat are treated as 'mf' (default dynamic in common rules).

These issues are fixed in this script.
"""
import os
import sys    
import argparse

import pandas as pd
import numpy as np
import librosa
import h5py
from hydra import initialize, compose
from tqdm import tqdm
from utils import create_folder, float32_to_int16, create_logging, get_filename, read_midi, load_discography_pid_metadata,

# ---- Dynamic level mapping (Mazurka: 5 labels; optional 8). 'blank' covers padding/silence.

DYNAMIC_LEVEL_MAPS = {
    '8-level': {'blank': 0, 'ppp': 1, 'pp': 2, 'p': 3, 'mp': 4, 'mf': 5, 'f': 6, 'ff': 7, 'fff': 8},
    '5-level': {'blank': 0,           'pp': 1, 'p': 2,          'mf': 3, 'f': 4, 'ff': 5}
}

def fix_mazurka_pid(root, mazurka_id, old_pid, new_pid):
    """
    Fix incorrect pid column names in both beat_time and beat_dyn CSVs for a given mazurka_id.

    Args:
        root (str): Root path to the MazurkaBL-master dataset.
        mazurka_id (str): The Mazurka opus ID, e.g., "M41-1".
        old_pid (str): Incorrect pid column name.
        new_pid (str): Corrected pid column name.

    Found:
        "M41-1", "pid9070b-01" should be corrected to "pid9070b-09".
    """
    print("Dont repeat the run of fix_mazurka_pid")

    for subfolder in ["beat_time", "beat_dyn"]:
        filename = f"{mazurka_id}{'beat_time.csv' if subfolder == 'beat_time' else 'beat_dynNORM.csv'}"
        filepath = os.path.join(root, subfolder, filename)
        if not os.path.exists(filepath):
            print(f"Failed to correct. Can't find: {filepath}")
            continue
        df = pd.read_csv(filepath)
        if old_pid not in df.columns:
            print(f"Failed to correct. Can't find '{old_pid}' in {filepath}")
            continue
        df.rename(columns={old_pid: new_pid}, inplace=True)
        df.to_csv(filepath, index=False)
        print(f"[Solved Issue2] Fixed: {filepath}")


def cleanup_meta_csv(cfg):
    """
    Generate meta.csv for each mazurka, with optional exclusion list support.

    Args:
        cfg: Configuration object containing dataset paths and exclude list.
        raw_meta_dir: MazurkaBL-master
        exclude_opus: List of mazurka_ids to exclude from processing, e.g., ["K040-01", "K055-03"].
                      None or [] means no exclusions.

    If a mazurka_id is in exclude_opus list, it will be skipped.
    """
    # Input dirs
    beat_folder = f"{cfg.dataset.mazurka.raw_meta_dir}/beat_time"
    dyn_folder = f"{cfg.dataset.mazurka.raw_meta_dir}/markings_dyn"

    # Output dir
    meta_folder = cfg.dataset.mazurka.cln_meta_dir
    os.makedirs(meta_folder, exist_ok=True)

    beat_files = sorted(f for f in os.listdir(beat_folder) if f.endswith('beat_time.csv'))
    print(f"Found {len(beat_files)} beat_time files.")

    exclude_list = cfg.dataset.mazurka.exclude_opus
    if exclude_list is None:
        exclude_list = []

    for beat_file in tqdm(beat_files, desc='Generating meta CSVs', ncols=100):
        mazurka_id = beat_file.replace('beat_time.csv', '')

        # Skip excluded (Issue1)
        if mazurka_id in exclude_list:
            tqdm.write(f"[Issue1 Solved] Skipped {mazurka_id}: unusual dynamic markings.")
            continue

        beat_path = os.path.join(beat_folder, beat_file)
        dyn_path = os.path.join(dyn_folder, f"{mazurka_id}markings.csv")

        # Load beat_time and dynamic markings
        df_beat = pd.read_csv(beat_path)
        df_dyn = pd.read_csv(dyn_path, header=None)

        # Map beats to dynamic markings
        beats_list = df_dyn.iloc[1].dropna().astype(int).tolist()
        dynamics_list = df_dyn.iloc[2].dropna().astype(str).tolist()
        beat_to_dyn = {beat: dyn for beat, dyn in zip(beats_list, dynamics_list)}        

        dyn_column = []
        cp_column = []
        current_dyn = 'mf' # Handle the Issue3

        if beats_list and min(beats_list) > 1:
            first_annotated = min(beats_list)
            tqdm.write(f"[Issue3 Solved] {mazurka_id} markings start from beat {first_annotated}. Filled 'mf' for beats 1..{first_annotated-1}")

        for idx in range(len(df_beat)):
            beat_idx_in_score = idx + 1
            previous_dyn = current_dyn
            if beat_idx_in_score in beat_to_dyn:
                current_dyn = beat_to_dyn[beat_idx_in_score]
            dyn_column.append(current_dyn)
            cp_column.append(1 if current_dyn != previous_dyn else 0)

        # Construct output DataFrame with beat and dynamic info
        df_out = pd.DataFrame({
            'beat_index': df_beat.index + 1,  # 1-based beat indexing
            'measure_number': df_beat.iloc[:, 1],
            'beat': df_beat.iloc[:, 2],
            'downbeat': (df_beat.iloc[:, 2] == 2).astype(int),
            'dynamic_mark': dyn_column,
            'change_point': cp_column
        })

        # Add pid columns from beat dataframe
        pid_cols = [col for col in df_beat.columns if col.startswith('pid')]
        for pid_col in pid_cols:
            df_out[pid_col] = df_beat[pid_col]

        out_path = os.path.join(meta_folder, f"{mazurka_id}meta.csv")
        df_out.to_csv(out_path, index=False)


def pack_mazurka_dataset_to_hdf5(cfg, sample_rate):
    """
    Pack Mazurka dataset into HDF5 files, including audio, MIDI, beat times,
    dynamic markings (string and integer labels), and change points.
    """
    audio_root = cfg.dataset.mazurka.audio_dir
    midi_root = cfg.dataset.mazurka.midi_dir
    meta_root = cfg.dataset.mazurka.cln_meta_dir
    discography_path = f"{cfg.dataset.mazurka.raw_meta_dir}/mazurka-discography.txt"
    exclude_list = cfg.dataset.mazurka.exclude_opus or []

    hdf5_root = os.path.join(cfg.exp.workspace, 'hdf5s', f'mazurka_sr{sample_rate}')

    create_logging(os.path.join(cfg.exp.workspace, 'logs', get_filename(__file__)), filemode='w')
    tqdm.write(f"Start packing Mazurka dataset: {audio_root}")

    for opus in tqdm(
        sorted(f for f in os.listdir(audio_root) if not f.startswith('.')),
        desc="Processing MazurkaID"):

        # Skip excluded opus
        mazurka_id = opus.replace('mazurka', '')  # strip prefix
        if f"M{mazurka_id}" in exclude_list:
            tqdm.write(f"[Issue1 Solved] Skipping excluded opus: M{mazurka_id}")
            continue

        opus_path = os.path.join(audio_root, opus)
        mid_path = os.path.join(midi_root, opus)
        meta_csv_path = os.path.join(meta_root, f"M{mazurka_id}meta.csv")
        meta_df = pd.read_csv(meta_csv_path)

        for perf_file in sorted(f for f in os.listdir(opus_path) if f.endswith('.wav')):
            pid = perf_file[:-4]  # remove ".wav" to get performance ID

            # Skip if pid not in metadata
            if pid not in meta_df.columns:
                tqdm.write(f"[Error] {pid} not in metadata. Skipped.")
                continue

            # Load discography metadata
            try:
                _, performer, duration = load_discography_pid_metadata(discography_path, pid, mazurka_id=mazurka_id)
            except ValueError as e:
                tqdm.write(str(e))
                performer, duration = "Unknown", -1

            # Load audio/MIDI
            audio, _ = librosa.load(os.path.join(opus_path, perf_file), sr=sample_rate, mono=True)
            midi = read_midi(os.path.join(mid_path, f"{pid}.mid"), mode="hpt")

            # Subset rows/cols for this performance
            select_df = meta_df[['beat_index', 'measure_number', 'downbeat', 'dynamic_mark', 'change_point', pid]].dropna()

            beat_time = select_df[pid].astype(np.float32).values  # measure_time: first beat per measure
            measure_numbers = select_df['measure_number'].astype(int).values
            measure_time = []
            seen = set()
            for mn, bt in zip(measure_numbers, beat_time):
                if mn not in seen:
                    measure_time.append(bt)
                    seen.add(mn)
            measure_time = np.array(measure_time, dtype=np.float32)

            dynmark_labels = select_df['dynamic_mark'].astype(str).values
            
            # Map labels to 5/8-level classes
            try:
                dynmark_5_class = np.array([DYNAMIC_LEVEL_MAPS['5-level'][m] for m in dynmark_labels], dtype=np.int64)
                dynmark_8_class = np.array([DYNAMIC_LEVEL_MAPS['8-level'][m] for m in dynmark_labels], dtype=np.int64)
            except KeyError as e:
                tqdm.write(f"[Error] {e} in opus: M{mazurka_id}, PID: {pid}, labels: {set(dynmark_labels)}")
                raise

            def time_filter(col):
                return beat_time[select_df[col].astype(int) == 1]

            downbeat_time = time_filter('downbeat')
            change_point_time = time_filter('change_point')

            dyn_beats = list(zip(beat_time, dynmark_labels))
            dyn_changes = [(t, m) for t, m, c in zip(beat_time, dynmark_labels, select_df['change_point'].astype(int)) if c == 1]

            # Prepare output dir and path
            out_dir = os.path.join(hdf5_root, opus)
            create_folder(out_dir)
            out_path = os.path.join(out_dir, f"{pid}.h5")

            with h5py.File(out_path, 'w') as hf:
                # Attrs
                hf.attrs.create('opus', data=mazurka_id.encode(), dtype='S10')
                hf.attrs.create('audio_filename', data=perf_file.encode(), dtype='S100')
                hf.attrs.create('midi_filename', data=f"{pid}.mid".encode(), dtype='S100')
                hf.attrs.create('duration_librosa', data=np.float32(len(audio) / sample_rate), dtype=np.float32)
                hf.attrs.create('performer', data=performer.encode(), dtype='S100')
                hf.attrs.create('duration_in_meta', data=np.float32(duration), dtype=np.float32)

                # Audio/MIDI
                hf.create_dataset('waveform', data=float32_to_int16(audio, mazurka_id=mazurka_id, perf_file=perf_file), dtype=np.int16)
                hf.create_dataset('midi_event', data=[e.encode() for e in midi['midi_event']], dtype='S100')
                hf.create_dataset('midi_event_time', data=midi['midi_event_time'].astype(np.float32), dtype=np.float32)

                # Beat annotations
                hf.create_dataset('beat_time', data=beat_time, dtype=np.float32)
                hf.create_dataset('downbeat_time', data=downbeat_time, dtype=np.float32)
                hf.create_dataset('measure_time', data=measure_time, dtype=np.float32)
                hf.create_dataset('change_point_time', data=change_point_time, dtype=np.float32)

                # Dynamics + beat annotations
                hf.create_dataset('dynmark_beats', data=[f"{t:.3f}:{m}".encode() for t, m in dyn_beats], dtype='S20')
                hf.create_dataset('dynmark_changes', data=[f"{t:.3f}:{m}".encode() for t, m in dyn_changes], dtype='S20')
                hf.create_dataset('dynmark_5_class', data=dynmark_5_class, dtype=np.int64)
                hf.create_dataset('dynmark_8_class', data=dynmark_8_class, dtype=np.int64)

    tqdm.write(f"Finished writing HDF5 files to {hdf5_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['fix_problem', 'cleanup_meta', 'pack_h5'], required=True,
                        help="Select which step to run: fix_problem, cleanup_meta, or pack_h5.")
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help="Sampling rate for audio loading and folder naming.")
    args, unknown = parser.parse_known_args()     # Parser took known args, hydra can get the rest

    initialize(config_path="./", job_name="train", version_base=None)
    cfg = compose(config_name="config", overrides=unknown)

    if args.mode == 'fix_problem':
        print("[Mode] Fixing specific pid column errors...")
        fix_mazurka_pid(
            root=cfg.dataset.mazurka.raw_meta_dir,
            mazurka_id="M41-1",
            old_pid="pid9070b-01",
            new_pid="pid9070b-09"
        )

    elif args.mode == 'cleanup_meta':
        print("[Mode] Generating cleaned meta CSVs...")
        cleanup_meta_csv(cfg)

    elif args.mode == 'pack_h5':
        print("[Mode] Packing Mazurka dataset to HDF5...")
        pack_mazurka_dataset_to_hdf5(cfg, sample_rate=args.sample_rate)
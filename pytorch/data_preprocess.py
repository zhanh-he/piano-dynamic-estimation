"""
We found some data issues in the MazurkaBL-master dataset, refer to problem_in_data.ipynb

- Issue1: Unusual dynamic markings occur in some mazurkas, 
    e.g., M63-2.
    @ Clarification:
        We exclude these unusual mazurkas from the dataset.

- Issue2: Some pid columns in beat_time and beat_dyn CSVs are incorrect, 
    e.g., "pid9070b-01" should be "pid9070b-09" in M41-1.
    @ Clarification:
        We manually fix this by renaming the pid matching the correct performance ID.

- Issue3: Some mazurkas first annotated dynamic_marking is not starting from the first beat, 
    e.g., M17-3
    @ Clarification:
        1. Frames BEFORE the first *beat time* are BLANK
        2. Beat indices from first beat up to the first annotated dynamic beat are treated as 'mf' (default dynamic in common rules).

These issues are fixed in this script.
"""
import argparse, os, sys, h5py, librosa, re
import pandas as pd
import numpy as np
from hydra import initialize, compose
from tqdm import tqdm
from utils import create_folder, float32_to_int16, create_logging, get_filename, read_midi, load_discography_pid_metadata
from score_features import extract_score_features, read_midi_any


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


# ----------------------------------------------------------------------
# Test-only datasets: Vienna4x22 and Batik-plays-Mozart
# ----------------------------------------------------------------------
def _pack_test_dataset(audio_path, midi_path, match_path, musicxml_path,
                       opus, performer, hf_out, sample_rate):
    """Write one performance into hf_out using the Mazurka HDF5 schema."""
    feats = extract_score_features(match_path, musicxml_path)

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration_librosa = float(len(audio) / sample_rate)
    midi = read_midi_any(midi_path)

    dynmark_5_class = np.array(
        [DYNAMIC_LEVEL_MAPS['5-level'].get(m, 0) for m in feats['dynmark_labels_5']],
        dtype=np.int64,
    )
    dynmark_8_class = np.array(
        [DYNAMIC_LEVEL_MAPS['8-level'].get(m, 0) for m in feats['dynmark_labels_8']],
        dtype=np.int64,
    )

    hf_out.attrs.create('opus', data=opus.encode(), dtype='S64')
    hf_out.attrs.create('audio_filename', data=os.path.basename(audio_path).encode(), dtype='S200')
    hf_out.attrs.create('midi_filename', data=os.path.basename(midi_path).encode(), dtype='S200')
    hf_out.attrs.create('duration_librosa', data=np.float32(duration_librosa), dtype=np.float32)
    hf_out.attrs.create('performer', data=performer.encode(), dtype='S100')
    hf_out.attrs.create('duration_in_meta', data=np.float32(feats['duration_perf_sec']), dtype=np.float32)

    hf_out.create_dataset('waveform', data=float32_to_int16(audio), dtype=np.int16)
    hf_out.create_dataset('midi_event', data=[e.encode() for e in midi['midi_event']], dtype='S100')
    hf_out.create_dataset('midi_event_time', data=midi['midi_event_time'].astype(np.float32), dtype=np.float32)

    hf_out.create_dataset('beat_time', data=feats['beat_time'], dtype=np.float32)
    hf_out.create_dataset('downbeat_time', data=feats['downbeat_time'], dtype=np.float32)
    hf_out.create_dataset('measure_time', data=feats['measure_time'], dtype=np.float32)
    hf_out.create_dataset('change_point_time', data=feats['change_point_time'], dtype=np.float32)

    beats = feats['beat_time']
    labels_5 = feats['dynmark_labels_5']
    # change-point per-beat boolean (re-derive from labels for the bytes block)
    is_change = np.zeros(len(labels_5), dtype=bool)
    if len(labels_5):
        is_change[0] = True
        is_change[1:] = labels_5[1:] != labels_5[:-1]
    hf_out.create_dataset('dynmark_beats',
                          data=[f"{t:.3f}:{m}".encode() for t, m in zip(beats, labels_5)], dtype='S20')
    hf_out.create_dataset('dynmark_changes',
                          data=[f"{t:.3f}:{m}".encode() for t, m in zip(beats[is_change], labels_5[is_change])], dtype='S20')
    hf_out.create_dataset('dynmark_5_class', data=dynmark_5_class, dtype=np.int64)
    hf_out.create_dataset('dynmark_8_class', data=dynmark_8_class, dtype=np.int64)


def pack_vienna_dataset_to_hdf5(cfg, sample_rate):
    """Pack Vienna4x22 into HDF5 (one .h5 per performance)."""
    root = cfg.dataset.vienna.root
    audio_root = os.path.join(root, 'audio')
    midi_root = os.path.join(root, 'midi')
    match_root = os.path.join(root, 'match')
    xml_root = os.path.join(root, 'musicxml')
    hdf5_root = os.path.join(cfg.exp.workspace, 'hdf5s', f'vienna_sr{sample_rate}')

    create_logging(os.path.join(cfg.exp.workspace, 'logs', get_filename(__file__)), filemode='w')
    tqdm.write(f"Start packing Vienna4x22 dataset: {root}")
    exclude_list = cfg.dataset.vienna.exclude_opus or []

    # Performance filename pattern: <piece>_p<NN>.wav   (.match + .mid share that stem)
    pat = re.compile(r"^(.+?)_p(\d+)\.wav$")
    # Map piece-stem -> musicxml file (Vienna xml stem omits the _pNN suffix)
    xml_files = sorted(f for f in os.listdir(xml_root) if f.endswith('.musicxml'))
    piece_to_xml = {os.path.splitext(f)[0]: os.path.join(xml_root, f) for f in xml_files}

    skipped, packed = [], 0
    audio_files = []
    for sub in sorted(os.listdir(audio_root)):
        sub_path = os.path.join(audio_root, sub)
        if not os.path.isdir(sub_path):
            continue
        for f in sorted(os.listdir(sub_path)):
            if f.endswith('.wav'):
                audio_files.append((sub, f, os.path.join(sub_path, f)))

    for sub, fname, audio_path in tqdm(audio_files, desc='Packing Vienna'):
        m = pat.match(fname)
        if not m:
            skipped.append((fname, 'unparseable filename'))
            continue
        piece, perf_num = m.group(1), m.group(2)
        if piece in exclude_list:
            tqdm.write(f"[exclude_opus] skipped {fname}")
            continue
        if piece not in piece_to_xml:
            skipped.append((fname, f'no musicxml for piece {piece}'))
            continue
        match_path = os.path.join(match_root, f"{piece}_p{perf_num}.match")
        midi_path = os.path.join(midi_root, f"{piece}_p{perf_num}.mid")
        if not os.path.isfile(match_path) or not os.path.isfile(midi_path):
            skipped.append((fname, f'missing match/midi for {piece}_p{perf_num}'))
            continue

        opus = piece                           # e.g. "Mozart_K331_1st-mov"
        stem = f"{piece}_p{perf_num}"
        out_dir = os.path.join(hdf5_root, opus)
        create_folder(out_dir)
        out_path = os.path.join(out_dir, f"{stem}.h5")

        try:
            with h5py.File(out_path, 'w') as hf:
                _pack_test_dataset(audio_path, midi_path, match_path, piece_to_xml[piece],
                                   opus=opus, performer=f"p{perf_num}", hf_out=hf,
                                   sample_rate=sample_rate)
            packed += 1
        except Exception as e:
            tqdm.write(f"[ERR] {stem}: {e}")
            skipped.append((fname, str(e)))
            if os.path.isfile(out_path):
                os.remove(out_path)

    tqdm.write(f"Vienna packed: {packed}, skipped: {len(skipped)}")
    for s in skipped:
        tqdm.write(f"  skip {s[0]} -> {s[1]}")
    tqdm.write(f"Finished writing HDF5 files to {hdf5_root}")


def pack_batik_dataset_to_hdf5(cfg, sample_rate):
    """Pack Batik-plays-Mozart into HDF5 (one .h5 per movement)."""
    prepared = cfg.dataset.batik.prepared
    hdf5_root = os.path.join(cfg.exp.workspace, 'hdf5s', f'batik_sr{sample_rate}')

    create_logging(os.path.join(cfg.exp.workspace, 'logs', get_filename(__file__)), filemode='w')
    tqdm.write(f"Start packing Batik dataset: {prepared}")
    exclude_list = cfg.dataset.batik.exclude_opus or []

    stems = sorted(d for d in os.listdir(prepared) if d.startswith('kv'))
    skipped, packed = [], 0
    for stem in tqdm(stems, desc='Packing Batik'):
        d = os.path.join(prepared, stem)
        try:
            opus = stem.split('_')[0]   # e.g. "kv279"
            if opus in exclude_list:
                tqdm.write(f"[exclude_opus] skipped {stem}")
                continue
            out_dir = os.path.join(hdf5_root, opus)
            create_folder(out_dir)
            out_path = os.path.join(out_dir, f"{stem}.h5")
            with h5py.File(out_path, 'w') as hf:
                _pack_test_dataset(
                    audio_path=os.path.join(d, 'audio.wav'),
                    midi_path=os.path.join(d, 'performance.mid'),
                    match_path=os.path.join(d, 'alignment.match'),
                    musicxml_path=os.path.join(d, 'score.musicxml'),
                    opus=opus, performer='Batik', hf_out=hf, sample_rate=sample_rate,
                )
            packed += 1
        except Exception as e:
            tqdm.write(f"[ERR] {stem}: {e}")
            skipped.append((stem, str(e)))
            if os.path.isfile(out_path):
                os.remove(out_path)

    tqdm.write(f"Batik packed: {packed}, skipped: {len(skipped)}")
    for s in skipped:
        tqdm.write(f"  skip {s[0]} -> {s[1]}")
    tqdm.write(f"Finished writing HDF5 files to {hdf5_root}")


def write_test_split_csv(hdf5_root: str, csv_path: str) -> None:
    """Emit an inference/eval CSV (cols: h5_name, opus, split) marking every
    packed file as the test split. The training pipeline never sees these."""
    rows = []
    for opus in sorted(os.listdir(hdf5_root)):
        opus_dir = os.path.join(hdf5_root, opus)
        if not os.path.isdir(opus_dir):
            continue
        for f in sorted(os.listdir(opus_dir)):
            if f.endswith('.h5'):
                rows.append((f, opus, 'test'))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w') as fh:
        fh.write('h5_name,opus,split\n')
        for r in rows:
            fh.write(','.join(r) + '\n')
    print(f"Wrote {len(rows)} test-split rows -> {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                        choices=['fix_problem', 'cleanup_meta', 'pack_h5',
                                 'pack_h5_vienna', 'pack_h5_batik'],
                        required=True,
                        help="Select which step to run.")
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

    elif args.mode == 'pack_h5_vienna':
        print("[Mode] Packing Vienna4x22 dataset to HDF5 (test split)...")
        pack_vienna_dataset_to_hdf5(cfg, sample_rate=args.sample_rate)
        write_test_split_csv(
            hdf5_root=os.path.join(cfg.exp.workspace, 'hdf5s', f'vienna_sr{args.sample_rate}'),
            csv_path=os.path.join(cfg.exp.workspace, 'split_csvs',
                                  f'vienna_sr{args.sample_rate}_test.csv'),
        )

    elif args.mode == 'pack_h5_batik':
        print("[Mode] Packing Batik-plays-Mozart dataset to HDF5 (test split)...")
        pack_batik_dataset_to_hdf5(cfg, sample_rate=args.sample_rate)
        write_test_split_csv(
            hdf5_root=os.path.join(cfg.exp.workspace, 'hdf5s', f'batik_sr{args.sample_rate}'),
            csv_path=os.path.join(cfg.exp.workspace, 'split_csvs',
                                  f'batik_sr{args.sample_rate}_test.csv'),
        )
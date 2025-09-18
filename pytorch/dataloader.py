import os
import csv
import numpy as np
import h5py
import math
try:
    # Relative import when used as a package
    from .utils import int16_to_float32, traverse_folder, data_5fold_split, data_random_split, pad_or_truncate_np
except Exception:
    # Fallback for top-level execution
    from utils import int16_to_float32, traverse_folder, data_5fold_split, data_random_split, pad_or_truncate_np

def pad_or_truncate(arr, target_length, axis=0, pad_value=0):
    """Alias to utils.pad_or_truncate_np (backward-compat)."""
    return pad_or_truncate_np(arr, target_length, pad_value=pad_value, axis=axis)

def time_to_frame_roll(time_array, start_time, frames_per_second, frames_num):
    """Times (s) -> binary frame roll of length `frames_num`."""
    frames_roll = np.zeros(int(frames_num), dtype=np.float32)
    fps = float(frames_per_second)
    for t in np.asarray(time_array).ravel():
        frame_idx = int(round((float(t) - float(start_time)) * fps))
        if 0 <= frame_idx < frames_num:
            frames_roll[frame_idx] = 1.0
    return frames_roll

def time_to_dynamic_roll(time_array, dyn_class_array, start_time, frames_per_second, frames_num, duration):
    """Beat times + classes -> frame-wise dynamic roll (int64)."""
    fps = float(frames_per_second)
    frames_num = int(frames_num)
    dynamic_roll = np.zeros(frames_num, dtype=np.int64)
    times = np.asarray(time_array).ravel()
    classes = np.asarray(dyn_class_array).ravel().astype(int)
    if times.size == 0 or classes.size == 0:
        return dynamic_roll

    first_beat_of_continue_segment = True if float(start_time) != 0.0 else False
    dur = float(duration)

    for i in range(len(times)):
        start_f = int(round((float(times[i]) - float(start_time)) * fps))
        if i < len(times) - 1:
            end_f = int(round((float(times[i + 1]) - float(start_time)) * fps))
        else:
            end_f = int(round((dur - float(start_time)) * fps))
        end_f = min(end_f, frames_num)
        # If current beat is within the segment
        if 0 <= start_f < frames_num:
            if first_beat_of_continue_segment:
                backfill = classes[i - 1] if i > 0 else 0
                dynamic_roll[:start_f] = backfill
                first_beat_of_continue_segment = False
            dynamic_roll[start_f:end_f] = classes[i]
    return dynamic_roll

class Mazurka_Dataset(object):
    """Mazurka HDF5 dataset with segment sampling (audio/midi/both)."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', f'mazurka_sr{cfg.feature.sample_rate}')
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        self.input_type = cfg.exp.input_type  # 'audio' or 'midi'
        self.random_seed = cfg.exp.random_seed

        # MIDI helper (if needed)
        if self.input_type in ['midi', 'both']:
            self.midi_processor = MIDIProcessor(
                cfg.feature.segment_seconds, 
                cfg.feature.frames_per_second)
            self.midi_feature = cfg.feature.midi_feature

        # Dynamic target helper
        self.dynamic_processor = DynamicProcessor(
            cfg.feature.frames_per_second,
            cfg.feature.segment_seconds,
            cfg.feature.dynamic_classes)

        # Collect all .h5 files
        self.all_h5 = []
        for root, _, files in os.walk(self.hdf5s_dir):
            for filename in files:
                if filename.endswith('.h5'):
                    self.all_h5.append((filename, os.path.join(root, filename)))

        # Optional: exclude PIDs before split
        ex_list = getattr(getattr(cfg.dataset, 'mazurka', None), 'exclude_pids', None)
        if ex_list:
            exclude_set = set(ex_list)
            filtered = []
            dropped = 0
            for fname, fpath in self.all_h5:
                opus = os.path.basename(os.path.dirname(fpath))
                full_pid = f"{opus}/{fname}"
                if full_pid in exclude_set:
                    dropped += 1
                    continue
                filtered.append((fname, fpath))
            self.all_h5 = filtered

        # Option A: 5-fold split (group by opus)
        if cfg.dataset.use_5fold:
            fold_index = cfg.dataset.fold_index
            groups = [os.path.basename(os.path.dirname(p)) for (_, p) in self.all_h5]
            uniq_groups = sorted(set(groups))
            if len(uniq_groups) < 5:
                raise ValueError(f"Not enough distinct opus groups for 5-fold CV: got {len(uniq_groups)} (<5). "
                                 f"Consider disabling use_5fold or reducing dataset.mazurka.exclude_opus / exclude_pids.")
            self.train_list, self.valid_list, self.test_list = data_5fold_split(
                self.all_h5, groups,
                seed=self.cfg.exp.random_seed,
                fold_index=fold_index,
                csv_dir=cfg.exp.workspace)
        # Option B: random Train/Valid/Test split; expect list e.g., [8,1,1]
        else:
            self.train_list, self.valid_list, self.test_list = data_random_split(
                self.all_h5,
                split_ratio=tuple(cfg.dataset.random_split_ratio),
                seed=self.cfg.exp.random_seed,
                csv_dir=cfg.exp.workspace)

    def _load_audio_input(self, hf, start_time):
        """Load audio segment; pad/shift to fit; apply augment if set."""
        start_sample = int(start_time * self.cfg.feature.sample_rate)
        end_sample = start_sample + self.segment_samples
        # Ensure valid segment within bounds
        total_samples = hf['waveform'].shape[0]
        if end_sample > total_samples:
            # Shift segment to fit within available data
            end_sample = total_samples
            start_sample = max(0, end_sample - self.segment_samples)
        waveform = int16_to_float32(hf['waveform'][start_sample:end_sample])
        # Pad if segment is shorter than required
        if waveform.shape[0] < self.segment_samples:
            pad_width = self.segment_samples - waveform.shape[0]
            waveform = np.pad(waveform, (0, pad_width), mode='constant')
        if self.cfg.feature.augmentor:
            waveform = self.cfg.feature.augmentor.augment(waveform)
        return waveform

    def _load_midi_input(self, hf, start_time):
        """Load MIDI segment and convert to piano-roll via MIDIProcessor."""
        midi_events = [e.decode() for e in hf['midi_event'][:]]
        midi_events_time = hf['midi_event_time'][:]
        target_dict, _ = self.midi_processor.process(
            start_time, midi_events_time, midi_events
        )
        velo_roll = target_dict["velocity_roll"]
        onset_roll = target_dict["onset_roll"]
        # Ensure returned roll is correct length (pad if needed)
        frames_needed = int(round(self.cfg.feature.segment_seconds * self.cfg.feature.frames_per_second)) + 1
        if velo_roll.shape[0] < frames_needed:
            pad_width = frames_needed - velo_roll.shape[0]
            velo_roll = np.pad(velo_roll, ((0, pad_width), (0, 0)), mode='constant')
            onset_roll = np.pad(onset_roll, ((0, pad_width), (0, 0)), mode='constant')
        elif velo_roll.shape[0] > frames_needed:
            velo_roll = velo_roll[:frames_needed]
            onset_roll = onset_roll[:frames_needed]
        if self.midi_feature in ["masked_velocity"]:
            # Onset-masked velocity
            velo_roll = velo_roll * onset_roll
        return velo_roll
    
    def __getitem__(self, meta):
        """Prepare one segment (inputs + frame targets)."""
        opus, hdf5_name, start_time = meta
        hdf5_path = os.path.join(self.hdf5s_dir, opus, hdf5_name)
        data_dict = {}

        with h5py.File(hdf5_path, 'r') as hf:
            # Samples per frame
            hop_size = int(self.cfg.feature.sample_rate / self.cfg.feature.frames_per_second)
            input_frames = int(self.segment_samples / hop_size)

            # Load input based on input_type
            if self.input_type == 'audio':
                data_dict['audio_input'] = self._load_audio_input(hf, start_time)
            elif self.input_type == 'midi':
                velo_roll = self._load_midi_input(hf, start_time)
                data_dict['midi_input'] = velo_roll
            elif self.input_type == 'both':
                data_dict['audio_input'] = self._load_audio_input(hf, start_time)
                velo_roll = self._load_midi_input(hf, start_time)
                data_dict['midi_input'] = velo_roll
            else:
                raise ValueError(f"Unknown input_type: {self.input_type}")

            # Frame-level targets
            dynamic_dict = self.dynamic_processor.process(start_time, hf)

            # Assign outputs
            end_time = start_time + self.cfg.feature.segment_seconds
            data_dict['meta'] = f"{opus}/{hdf5_name}/{start_time:.2f}s~{end_time:.2f}s"
            data_dict['start_time'] = start_time
            data_dict['beat_roll'] = dynamic_dict['beat_roll']
            data_dict['downbeat_roll'] = dynamic_dict['downbeat_roll']
            data_dict['measure_roll'] = dynamic_dict['measure_roll']
            data_dict['change_point_roll'] = dynamic_dict['change_point_roll']
            data_dict['dynamic_roll'] = dynamic_dict['dynamic_roll']
            return data_dict


class Sampler(object):
    """Infinite sampler yielding segment batches for training."""
    def __init__(self, cfg, file_list):
        self.cfg = cfg
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', f'mazurka_sr{cfg.feature.sample_rate}')
        self.segment_seconds = cfg.feature.segment_seconds
        self.segment_hop_seconds = cfg.feature.segment_hop_seconds
        self.batch_size = cfg.exp.batch_size
        self.mini_data = cfg.exp.mini_data

        # Collect segments from selected .h5 files
        self.segment_list = []
        file_count = 0
        for h5_name, h5_path in file_list:
            # Parent folder is opus
            opus = os.path.basename(os.path.dirname(h5_path))
            with h5py.File(h5_path, 'r') as hf:
                duration = float(hf.attrs['duration_librosa'])
                start_time = 0
                # Valid segment range
                while start_time + self.segment_seconds <= duration:
                    self.segment_list.append([opus, h5_name, start_time])
                    start_time += self.segment_hop_seconds
                file_count += 1
                if self.mini_data and file_count == 10:
                    break

        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        while True:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[self.pointer]
                self.pointer += 1
                if self.pointer >= len(self.segment_indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.segment_indexes)
                batch_segment_list.append(self.segment_list[index])
                i += 1
            yield batch_segment_list

    def __len__(self):
        return int(np.ceil(len(self.segment_list) / self.batch_size))

    def state_dict(self):
        return {
            'pointer': self.pointer,
            'segment_indexes': self.segment_indexes
        }

    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']


class EvalSampler(Sampler):
    """Finite sampler for quick evaluation (fixed batch count)."""
    def __init__(self, cfg, file_list):
        super().__init__(cfg, file_list)
        self.max_evaluate_batches = int(getattr(cfg.exp, 'eval_batches', 40))

    def __iter__(self):
        pointer = 0
        batch_count = 0
        while batch_count < self.max_evaluate_batches:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[pointer % len(self.segment_indexes)]
                pointer += 1
                batch_segment_list.append(self.segment_list[index])
                i += 1
            batch_count += 1
            yield batch_segment_list


def get_train_positive_weights(train_list, fps: int, widen_target_mask: int = 3) -> dict:
    """Estimate dataset-level pos weights (beat/downbeat/change_point)."""
    all_frames = 0
    all_frames_db = 0
    beat_frames = 0
    downbeat_frames = 0
    change_frames = 0
    for _, h5_path in train_list:
        with h5py.File(h5_path, 'r') as hf:
            duration = float(hf.attrs.get('duration_librosa', 0.0))
            frames = int(np.ceil(duration * fps))
            all_frames += frames
            if 'beat_time' in hf:
                beat_frames += int(hf['beat_time'].shape[0])
            if 'downbeat_time' in hf:
                all_frames_db += frames
                downbeat_frames += int(hf['downbeat_time'].shape[0])
            if 'change_point_time' in hf:
                change_frames += int(hf['change_point_time'].shape[0])

    widen = (widen_target_mask * 2 + 1)
    beat_neg = max(all_frames - beat_frames * widen, 1)
    db_neg = max(all_frames_db - downbeat_frames * widen, 1) if all_frames_db else 1

    beat_w = int(round(beat_neg / max(beat_frames, 1))) or 1
    db_w = int(round(db_neg / max(downbeat_frames, 1))) if downbeat_frames else 1
    db_w = db_w or 1

    # change_point uses the same all_frames since it's a general frame timeline
    ch_neg = max(all_frames - change_frames * widen, 1)
    ch_w = int(round(ch_neg / max(change_frames, 1))) or 1

    return {"beat": beat_w, "downbeat": db_w, "change_point": ch_w}


def collate_fn(list_data_dict):
    """Stack a list of segment dicts into batch dict (numpy)."""
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    return np_data_dict


class MIDIProcessor(object):
    def __init__(self, segment_seconds, frames_per_second):
        """Build MIDI -> piano-roll processor (HPT 2020 style)."""
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second

        # Piano note range: MIDI piano keys are 21 (A0) to 108 (C8), total 88 keys
        self.begin_note = 21  # Lowest piano key
        self.classes_num = 88  # Total piano keys

    def process(self, start_time, midi_events_time, midi_events, note_shift=0):
        """Parse events in segment -> onset/velocity rolls + note list."""
        for bgn_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time:
                break
        for fin_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time + self.segment_seconds:
                break

        note_events = []
        buffer_dict = {}

        _delta = int((fin_idx - bgn_idx) * 1.0)
        ex_bgn_idx = max(bgn_idx - _delta, 0)

        for i in range(ex_bgn_idx, fin_idx):
            attr = midi_events[i].split(' ')
            if attr[0] in ['note_on', 'note_off']:
                midi_note = int(attr[2].split('=')[1])
                velocity = int(attr[3].split('=')[1])
                if attr[0] == 'note_on' and velocity > 0:
                    buffer_dict[midi_note] = {
                        'onset_time': midi_events_time[i],
                        'velocity': velocity
                    }
                else:
                    if midi_note in buffer_dict:
                        note_events.append({
                            'midi_note': midi_note,
                            'onset_time': buffer_dict[midi_note]['onset_time'],
                            'offset_time': midi_events_time[i],
                            'velocity': buffer_dict[midi_note]['velocity']
                        })
                        del buffer_dict[midi_note]

        for midi_note in buffer_dict:
            note_events.append({
                'midi_note': midi_note,
                'onset_time': buffer_dict[midi_note]['onset_time'],
                'offset_time': start_time + self.segment_seconds,
                'velocity': buffer_dict[midi_note]['velocity']
            })

        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.classes_num))
        velocity_roll = np.zeros((frames_num, self.classes_num))

        for note_event in note_events:
            # MIDI note -> piano index
            piano_note = note_event['midi_note'] - self.begin_note + note_shift
            if 0 <= piano_note < self.classes_num:
                bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
                fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))
                if fin_frame >= 0:
                    velocity_roll[max(bgn_frame, 0): fin_frame + 1, piano_note] = note_event['velocity']
                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1

        # Use helper for padding/truncating
        velocity_roll = pad_or_truncate(velocity_roll, frames_num, axis=0)
        onset_roll = pad_or_truncate(onset_roll, frames_num, axis=0)

        target_dict = {
            'onset_roll': onset_roll,
            'velocity_roll': velocity_roll
        }

        return target_dict, note_events
    
   
class DynamicProcessor(object):
    """Beat-level times/labels -> frame-wise rolls aligned to FPS."""
    def __init__(self, frames_per_second, segment_seconds, dynamic_classes):
        self.frames_per_second = frames_per_second
        self.segment_seconds =   segment_seconds
        self.dynamic_classes =   dynamic_classes

    def process(self, start_time, hf):
        """Extract frame-wise rolls from HDF5 for one segment."""
        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1

        beat_roll = time_to_frame_roll(hf['beat_time'][:], start_time, self.frames_per_second, frames_num)
        downbeat_roll = time_to_frame_roll(hf['downbeat_time'][:], start_time, self.frames_per_second, frames_num)
        measure_roll = time_to_frame_roll(hf['measure_time'][:], start_time, self.frames_per_second, frames_num)
        change_point_roll = time_to_frame_roll(hf['change_point_time'][:], start_time, self.frames_per_second, frames_num)
        dynamic_roll = time_to_dynamic_roll(
            hf['beat_time'][:], hf[f'dynmark_{self.dynamic_classes}_class'][:],
            start_time, self.frames_per_second, frames_num, hf.attrs['duration_librosa'])


        dynamic_dict = {
            'beat_roll': beat_roll,
            'downbeat_roll': downbeat_roll,
            'measure_roll': measure_roll,
            'change_point_roll': change_point_roll,
            'dynamic_roll': dynamic_roll
        }
        return dynamic_dict

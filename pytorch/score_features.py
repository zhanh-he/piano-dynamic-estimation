"""Extract mazurka-schema features (beats/downbeats/measures/dynamics/change-points)
from a partitura `.match` + MusicXML pair. Used by data_preprocess.py to pack
Vienna4x22 and Batik-plays-Mozart into the same HDF5 layout as Mazurka so that
the existing inference/eval code can read them with no changes.

The notion of "beat" here is the integer beat positions of the (unfolded) score,
mapped into performance time via the alignment between score note ids and
performance note ids in the .match file.

Dynamic markings come from `ConstantLoudnessDirection` directions in the score.
Impulsive accents (sf/fp/sfp/sfz) and italian words (dolce, ...) are ignored
because they do not change the steady-state dynamic level.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import partitura as pt
import partitura.score as ps
from mido import MidiFile


# ---------- dynamic-level normalisation ----------
# 5-level (mazurka): blank, pp, p, mf, f, ff
_TO_5 = {
    "ppp": "pp", "pp": "pp",
    "p":   "p",  "mp": "p",
    "mf":  "mf",
    "f":   "f",  "fp": "f",
    "ff":  "ff", "fff": "ff",
}
# 8-level (richer)
_TO_8 = {
    "ppp": "ppp", "pp": "pp",
    "p":   "p",   "mp": "mp",
    "mf":  "mf",
    "f":   "f",   "fp": "f",
    "ff":  "ff",  "fff": "fff",
}

# strings that should NOT update the steady-state level
_IGNORE_DYN_TEXT = {"sf", "sfz", "sfp", "fz", "rfz", "dolce", "espressivo", "cantabile"}


def _normalise_label(label: str, target: str) -> str | None:
    """Return canonical level name or None when the marking should be ignored."""
    s = (label or "").strip().lower()
    if not s or s in _IGNORE_DYN_TEXT:
        return None
    table = _TO_5 if target == "5" else _TO_8
    return table.get(s)


# ---------- score-feature extraction ----------
def extract_score_features(match_path: str, musicxml_path: str,
                           default_dyn: str = "mf") -> dict:
    """Build mazurka-schema annotations for one performance.

    Returns dict with float32 arrays (in seconds) plus per-beat label arrays:
        beat_time, downbeat_time, measure_time, change_point_time,
        dynmark_labels (str), dynmark_5_class (int64), dynmark_8_class (int64),
        duration_perf_sec (float, last performed note offset)
    """
    perf, alignment = pt.load_match(match_path)
    score = pt.load_musicxml(musicxml_path)
    spart_unf = pt.score.unfold_part_maximal(score[0], ignore_leaps=False)

    # --- 1. alignment -> (score_beat, perf_time) pairs ---
    pna = perf.note_array()
    sna = spart_unf.note_array()
    perf_time = {nid: float(t) for nid, t in zip(pna["id"], pna["onset_sec"])}
    score_beat = {nid: float(b) for nid, b in zip(sna["id"], sna["onset_beat"])}

    # partitura's unfold_part_maximal appends "-<pass>" to ids (e.g. n1 -> n1-1).
    # Vienna match files use bare ids ("n1"); Batik match files already use the
    # unfolded form ("n1-1"). Try the bare id first, then the "-1" fallback.
    pairs = []
    for entry in alignment:
        if entry.get("label") != "match":
            continue
        s_id, p_id = entry.get("score_id"), entry.get("performance_id")
        if p_id not in perf_time:
            continue
        if s_id in score_beat:
            pairs.append((score_beat[s_id], perf_time[p_id]))
        elif f"{s_id}-1" in score_beat:
            pairs.append((score_beat[f"{s_id}-1"], perf_time[p_id]))
    if len(pairs) < 8:
        raise RuntimeError(f"Too few aligned notes ({len(pairs)}) in {match_path}")

    pairs.sort()
    beats = np.array([b for b, _ in pairs], dtype=np.float64)
    times = np.array([t for _, t in pairs], dtype=np.float64)

    # Average + enforce monotone non-decreasing times
    uniq_beats, inv = np.unique(beats, return_inverse=True)
    sums = np.zeros_like(uniq_beats); cnts = np.zeros_like(uniq_beats)
    np.add.at(sums, inv, times)
    np.add.at(cnts, inv, 1)
    mean_times = sums / np.maximum(cnts, 1)
    mean_times = np.maximum.accumulate(mean_times)

    def beat_to_sec(b_arr: np.ndarray) -> np.ndarray:
        # Clamp to known beat range so pickup measures / trailing rests don't
        # produce spurious time=0 outputs.
        clamped = np.clip(b_arr.astype(np.float64), uniq_beats[0], uniq_beats[-1])
        return np.interp(clamped, uniq_beats, mean_times)

    # --- 2. integer beats from 0 .. last_aligned_beat ---
    max_beat = int(np.floor(uniq_beats[-1]))
    all_beats = np.arange(0, max_beat + 1, dtype=np.float64)
    beat_time = beat_to_sec(all_beats).astype(np.float32)

    # --- 3. downbeats from unfolded measure boundaries (within range) ---
    measures = list(spart_unf.iter_all(ps.Measure))
    measure_starts_beat = np.array(
        [float(spart_unf.beat_map(m.start.t)) for m in measures], dtype=np.float64
    )
    measure_starts_beat = measure_starts_beat[measure_starts_beat <= max_beat]
    downbeat_time = beat_to_sec(measure_starts_beat).astype(np.float32)
    measure_time = downbeat_time.copy()  # one downbeat per measure

    # --- 4. dynamic marks from unfolded score ---
    dyn_events = []
    for d in spart_unf.iter_all(ps.ConstantLoudnessDirection):
        try:
            b = float(spart_unf.beat_map(d.start.t))
        except Exception:
            continue
        if b > max_beat:
            continue
        dyn_events.append((b, str(d.text)))
    dyn_events.sort()

    # Carry-forward dynamic level at each integer beat
    labels = []
    current = default_dyn
    i = 0
    for b in all_beats:
        while i < len(dyn_events) and dyn_events[i][0] <= b:
            norm = _normalise_label(dyn_events[i][1], target="8")  # canonical 8-level
            if norm is not None:
                current = norm
            i += 1
        labels.append(current)
    labels = np.array(labels, dtype=object)

    # 5-level requires further squashing
    labels_5 = np.array([_TO_5.get(lbl, "mf") for lbl in labels], dtype=object)

    # --- 5. change points (only on transitions of the normalised level) ---
    is_change = np.zeros(len(all_beats), dtype=bool)
    if len(all_beats):
        is_change[0] = True
        for k in range(1, len(all_beats)):
            if labels[k] != labels[k - 1]:
                is_change[k] = True
    change_point_time = beat_time[is_change]

    # --- 6. duration of the performance ---
    duration_perf = float(np.max(pna["onset_sec"] + pna["duration_sec"]))

    return {
        "beat_time": beat_time,
        "downbeat_time": downbeat_time,
        "measure_time": measure_time,
        "change_point_time": change_point_time.astype(np.float32),
        "dynmark_labels_8": labels.astype(str),
        "dynmark_labels_5": labels_5.astype(str),
        "duration_perf_sec": duration_perf,
    }


# ---------- MIDI -> events that work for arbitrary track layouts ----------
def read_midi_any(midi_path: str) -> dict:
    """Merge all tracks into a single (event_str, time_sec) stream.

    Works for both Batik (2 mixed tracks) and Vienna (1 track) MIDIs without
    relying on hpt/maps/smd track conventions. Returns:
        {'midi_event': np.array[str], 'midi_event_time': np.array[float32]}
    """
    mf = MidiFile(midi_path)
    tpb = mf.ticks_per_beat
    tempo = 500000  # default = 120 BPM
    # Find first tempo, if any
    for tr in mf.tracks:
        for msg in tr:
            if msg.is_meta and msg.type == "set_tempo":
                tempo = msg.tempo
                break
        else:
            continue
        break
    ticks_per_sec = tpb * (1e6 / tempo)

    # Convert each track to absolute (tick, msg) then merge
    abs_events = []
    for tr in mf.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            abs_events.append((t, str(msg)))
    abs_events.sort(key=lambda x: x[0])

    times = np.array([e[0] / ticks_per_sec for e in abs_events], dtype=np.float32)
    events = np.array([e[1] for e in abs_events], dtype=object)
    return {"midi_event": events, "midi_event_time": times}

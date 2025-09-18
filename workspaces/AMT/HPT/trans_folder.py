import os, sys, glob
from tqdm import tqdm
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import argparse
import warnings
import contextlib
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def trans_folder(input_folder, output_folder, device='cuda'):
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Load transcriptor
    transcriptor = PianoTranscription(device=device)

    # Find all .wav files in the input folder recursively
    audio_files = glob.glob(os.path.join(input_folder, '**', '*.wav'), recursive=True)

    # Process each file with tqdm progress bar
    for audio_path in tqdm(audio_files, desc="Transcribing audio files"):
        # Get relative path from input folder
        relative_path = os.path.relpath(audio_path, input_folder)
        relative_base = os.path.splitext(relative_path)[0]

        # Prepare output path (change .wav to .mid and maintain folder structure)
        output_path = os.path.join(output_folder, relative_base + '.mid')

        # Create output subdirectory if necessary
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Skip if the MIDI file already exists
        if os.path.exists(output_path):
            tqdm.write(f"Skipping existing: {relative_base}.mid")
            continue

        tqdm.write(f"Processing: {relative_base}.wav")

        # Load audio
        audio, _ = load_audio(audio_path, sr=sample_rate, mono=True)

        # Transcribe and save MIDI file, suppress unwanted prints
        with suppress_stdout():
            transcriptor.transcribe(audio, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch transcribe WAV to MIDI.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to input WAV folder')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to output MIDI folder')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run model on: cuda or cpu')

    args = parser.parse_args()

    trans_folder(args.input_folder, args.output_folder, device=args.device)
# Env Setup
1. Install the environment with "hpt_mamba.yml"
2. Download the github repo "https://github.com/bytedance/piano_transcription", then `pip install piano_transcription_inference`

# Run the inferenec
`python trans_folder.py --input_folder ../../mazurka_audio --output_folder ../../mazurka_mid --device cuda`
-- input_folder is the dataset audio path
-- output_folder is the transcribed midi saved path
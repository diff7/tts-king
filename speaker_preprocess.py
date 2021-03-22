from resemblyzer import VoiceEncoder, preprocess_wav
import os
from pathlib import Path

from tqdm import tqdm
import numpy as np

encoder = VoiceEncoder()


def preprocess_speaker(folder, speaker_name, target_dir):
    for fpath in Path(f'{folder}/{speaker_name}').rglob('*.wav'):
        wav = preprocess_wav(fpath)
        basename = fpath.split(".wav")[0]
        speaker_filename = "{}-speaker-{}.npy".format(speaker_name, basename)
        embed = encoder.embed_utterance(wav)
        np.save(
            os.path.join(target_dir, speaker_filename),
            embed,
        )


def preprocess_dataset(dir, target_dir):
    os.makedirs(f"{target_dir}/speakers_emb", exist_ok=True)
    target_dir = f"{target_dir}/speakers_emb"
    for speaker_name in tqdm(os.listdir(dir)):
        preprocess_speaker(dir, speaker_name, target_dir)

import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf as omg

from resemblyzer import VoiceEncoder, preprocess_wav

encoder = VoiceEncoder()


def preprocess_speaker(folder, speaker_name, target_dir):
    for fpath in Path(f"{folder}/{speaker_name}").rglob("*.wav"):
        fpath = str(fpath)
        wav = preprocess_wav(fpath)
        basename = fpath.split(".wav")[0]
        speaker_filename = "{}.npy".format(basename)
        embed = encoder.embed_utterance(wav)
        save_path = os.path.join(target_dir, speaker_filename)
        print("**", save_path)
        np.save(save_path, embed)


def preprocess_dataset(dir, target_dir):
    print(dir, target_dir)
    os.makedirs(f"{target_dir}/speakers_emb", exist_ok=True)
    target_dir = f"{target_dir}/speakers_emb"
    for speaker_name in tqdm(os.listdir(dir)):
        print(speaker_name)
        preprocess_speaker(dir, speaker_name, target_dir)


if __name__ == "__main__":
    cfg = omg.load("./multi_config/preprocess.yaml")
    raw_dir = cfg.path.raw_path
    out_dir = cfg.path.preprocessed_path
    preprocess_dataset(raw_dir, out_dir)
import os
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf as omg

from resemblyzer import VoiceEncoder, preprocess_wav


def get_embedding(wav_path):
    wav = preprocess_wav(wav_path)
    return encoder.embed_utterance(wav)


def make_emb_save_path(embedding_folder, wav_name):
    target_path = os.path.join(
        embedding_folder, wav_name.replace(".wav", ".npy")
    )
    return target_path


def make_embeddings(target_dir, source_dir):
    speaker_names = os.listdir(source_dir)
    for i, speaker_name in enumerate(speaker_names):
        print(f"{i+1} out of {len(speaker_names)} processing {speaker_name} ")
        speaker_folder = os.path.join(source_dir, speaker_name)
        embedding_folder = os.path.join(target_dir, speaker_name)
        os.makedirs(embedding_folder, exist_ok=True)

        for wav_name in tqdm(os.listdir(speaker_folder)):
            if not "wav" in wav_name:
                continue
            embedding = get_embedding(os.path.join(speaker_folder, wav_name))
            emb_path = make_emb_save_path(embedding_folder, wav_name)
            np.save(emb_path, embedding)


if __name__ == "__main__":

    encoder = VoiceEncoder()

    cfg = omg.load("./config.yaml")
    target_dir = os.path.join(
        cfg.preprocess_config.path.preprocessed_path, "speaker_emb"
    )
    os.makedirs(target_dir, exist_ok=True)
    source_dir = cfg.preprocess_config.path.raw_path

    make_embeddings(target_dir, source_dir)
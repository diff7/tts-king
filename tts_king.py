# IMPORTS FOR PREPROCESS
import os
import torch
import numpy as np
from string import punctuation
from fs_two.text import text_to_sequence

# OTHER IMPORTS
from omegaconf import OmegaConf
from fsapi import FSTWOapi

# from fs_two.preprocess import prepare_dataset_lj_speech
from hifiapi import HIFIapi

from input_process import preprocess_lang, preprocess_eng


class TTSKing:
    def __init__(self, config_path="./config.yaml"):
        self.cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.load(config_path)

        self.tts = FSTWOapi(self.cfg, self.cfg.device)
        self.vocoder = HIFIapi(self.cfg, self.cfg.device)

    def generate_mel(
        self,
        text,
        duration_control=1.0,
        pitch_control=1.0,
        energy_control=1.0,
        speaker=0,
    ):

        phonemes = self.text_preprocess(text)

        result = self.tts.generate(
            phonemes,
            duration_control,
            pitch_control,
            energy_control,
            speaker=speaker,
        )

        # mel, mel_postnet, log_duration_output, f0_output, energy_output
        return result

    def mel_to_wav(self, mel_spec):
        wav_cpu = self.vocoder.generate(mel_spec.transpose(1, 2))
        return wav_cpu

    def speak(
        self, text, duration_control=1.0, pitch_control=1.0, energy_control=1.0
    ):
        mel_specs_batch = self.generate_mel_batch(
            text, duration_control, pitch_control, energy_control
        )
        return self.vocoder(mel_specs_batch)

    def text_preprocess(self, text):
        return np.array([preprocess_lang(text, self.cfg.preprocess_config)])

    def text_preprocess_eng(self, text):
        return np.array([preprocess_eng(text, self.cfg.preprocess_config)])

    def to_torch_device(self, items):
        return [torch.tensor(t).to(self.cfg.gpu) for t in items]

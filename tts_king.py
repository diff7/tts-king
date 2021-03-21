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


def get_fsp_configs(
    config_folder, config_names=["preprocess.yaml", "model.yaml", "train.yaml"]
):
    configs = []
    for name in config_names:
        full_path = os.path.join(config_folder, name)
        configs.append(OmegaConf.load(full_path))
    return configs


class TTSKing:
    def __init__(self, config_path, config_folder="./multi_config/"):
        configs = get_fsp_configs(config_folder)
        self.preprocess_config_c, _, _ = configs

        cfg = OmegaConf.load(config_path)

        self.tts = FSTWOapi(cfg.tts, cfg.tts_weights_path, cfg.device, configs)
        self.vocoder = HIFIapi(cfg.hifi, cfg.hifi_weights_path, cfg.device)
        self.logger = None
        self.cfg = cfg

    def generate_mel(
        self, text, duration_control=1.0, pitch_control=1.0, energy_control=1.0, speaker=0
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
        return np.array([preprocess_lang(text, self.preprocess_config_c)])
    
    def text_preprocess_eng(self, text):
        return np.array([preprocess_eng(text, self.preprocess_config_c)])

    def to_torch_device(self, items):
        return [torch.tensor(t).to(self.cfg.device) for t in items]


# def get_class(main, module):
#     main = importlib.import_module(main)
#     return getattr(main, module)


# def init_from_config(main, module, params=None):
#     """
#     Example:
#         main: sklearn.linear_model
#         module: LinearRegression
#     """
#     class_con = get_class(main, module)
#     if not params is None:
#         return class_con(**params)
#     else:
#         return class_con()

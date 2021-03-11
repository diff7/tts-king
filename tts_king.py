# IMPORTS FOR PREPROCESS
import re
import torch
import numpy as np
from string import punctuation
from g2p_en import G2p
from fs_two.text import text_to_sequence

# OTHER IMPORTS
from omegaconf import DictConfig, OmegaConf
import yaml
from fsapi import FSTWOapi

# from fs_two.preprocess import prepare_dataset_lj_speech
from hifiapi import HIFIapi

from input_process import preprocess_english


class TTSKing:
    def __init__(self, preprocess_config, model_config, train_config, config_path):

        self.preprocess_config_c = yaml.load(
            open(preprocess_config, "r"), Loader=yaml.FullLoader
        )
        self.model_config_c = yaml.load(
            open(model_config, "r"), Loader=yaml.FullLoader)
        self.train_config_c = yaml.load(
            open(train_config, "r"), Loader=yaml.FullLoader)
        configs = (self.preprocess_config_c,
                   self.model_config_c, self.train_config_c)

        cfg = OmegaConf.load(config_path)

        self.tts = FSTWOapi(cfg.tts, cfg.tts_weights_path, cfg.device, configs)
        self.vocoder = HIFIapi(cfg.hifi, cfg.hifi_weights_path, cfg.device)
        self.logger = None
        self.cfg = cfg

    def generate_mel(
        self, text, duration_control=1.0, pitch_control=1.0, energy_control=1.0
    ):

        phonemes = self.text_preprocess(text)

        result = self.tts.generate(
            phonemes,
            duration_control,
            pitch_control,
            energy_control,
            speaker=None,
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

    # TODO :
    def train_tts(self, data_loader):

        # get val train data loader

        self.tts.train(train_data_loader, val_data_loader, self.vocoder)

    # TODO
    def train_vocoder(self, dataset, epochs):
        pass

    # TODO
    def init_data_loader_tts(self, data_folder_path):
        # return data_loader
        pass

    def prepare_dataset_tts(self, path_to_data):
        # TODO
        pass

    def prepare_dataset_lj_speech(self):
        prepare_dataset_lj_speech(self.cfg)

    def text_preprocess(self, text):
        return np.array([preprocess_english(text, self.preprocess_config_c)])

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

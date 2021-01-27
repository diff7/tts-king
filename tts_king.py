import importlib
from omegaconf import DictConfig, OmegaConf
from fsapi import FSTWOapi
from hifiapi import HIFIapi


class TTSKing:
    def __init__(self, config_path="config.yaml"):

        cfg = OmegaConf.load(config_path)

        self.tts = FSTWOapi(cfg.tts, cfg.tts_weights_path, cfg.device)
        self.vocoder = HIFIapi(cfg.higi, cfg.hifi_weights_path, cfg.device)
        self.logger = None

    def generate_mel(
        self, text, duration_control=1.0, pitch_control=1.0, energy_control=1.0
    ):

        phonemes = self.text_preprocess(text)
        result = self.tts.generate(
            self,
            phonemes,
            duration_control,
            pitch_control,
            energy_control,
            speaker=None,
        )

        # mel, mel_postnet, log_duration_output, f0_output, energy_output
        return result

    def mel_to_wav(self, mel_spec):
        wav_cpu = self.vocoder.generate(mel_spec)
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
        self.tts.train(data_loader, self.vocoder, self.logger)

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

    def text_preprocess(self, text):
        # TODO write processing function
        # return process_txt(text)
        pass


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
import importlib
from omegaconf import DictConfig, OmegaConf
from fs_two.modules_imports import FSTWOTrainable


class TTSKing:
    def __init__(self, config_path="config.yaml"):

        cfg = OmegaConf.load(config_path)

        self.tts = FSTWOTrainable(cfg.tts, cfg.tts_weights_path, cfg.device)
        self.vocoder = None
        self.logger = None

    # def __init_module(self, module_name):
    #     return module

    def train_tts(self, data_loader):
        self.tts.train(data_loader, self.vocoder, self.logger)

    def train_vocoder(self, dataset, epochs):
        pass

    def generate_mel_batch(
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

    def init_data_loader_tts(self, data_folder_path):
        # TODO

        return data_loader

    def prepare_dataset_tts(self, path_to_data):
        # TODO
        pass

    def generate_mel_one(self, text=[]):
        phoneme = self.text_preprocess(text)

        pass

    def text_preprocess(self, text):
        # TODO write processing function
        return process_txt(text)

    def vocoder_generate(specs=[]):
        pass

    def train_vocoder(
        self,
    ):
        pass

    def gen_audio(self, mel):
        audio = self.vocoder.gen.audio(mel)
        return audio


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
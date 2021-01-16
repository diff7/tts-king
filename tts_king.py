import importlib
from omegaconf import DictConfig, OmegaConf


def get_class(main, module):
    main = importlib.import_module(main)
    return getattr(main, module)


def init_from_config(main, module, params=None):
    """
    Example:
        main: sklearn.linear_model
        module: LinearRegression
    """
    class_con = get_class(main, module)
    if not params is None:
        return class_con(**params)
    else:
        return class_con()


class TTSKing:
    def __init__(self, tts="fs_two", vocoder="hifi", config_path="config.yaml"):

        config = OmegaConf.load(config_path)

        self.tts = self._init_module(config.imports[tts])
        self.vocoder = self._int_module(vocoderconfig.imports[vocoder])

    def __init_module(self, module_name):
        return module

    def train_tts(self, dataset, epochs):
        pass

    def train_vocoder(self, dataset, epochs):
        pass

    def tts_generate(text=[]):
        pass

    def vocoder_generate(specs=[]):
        pass
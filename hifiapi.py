import torch
from hifi.models import Generator


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class HIFIapi:
    def __init__(self, config, device="gpu"):
        if config.model_config["vocoder"]["use_cpu"]:
            device = "cpu"

        # Load checkpoint if exists
        weights_path = config.hifi.weights_path

        self.model = Generator(config.hifi)
        if weights_path is not None:
            checkpoint = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["generator"])

        self.cfg = config
        self.device = device

        self.model.to(device)
        self.model.remove_weight_norm()
        self.model.eval()

    # TODO:
    def train(self):
        raise NotImplemented(" Train for HiFi was not implemented yet")

    def __call__(self, x):
        x = x.to(self.device)
        # use call for compatablity with other vocoders or functions
        return self.model(x)

    def generate(self, mel_specs):
        """
        Converts mel spectrogramma into an audio file.
        Returns cpu audio files.
        mel_specs - a batch of mel spectrogramms
        """

        self.model.eval()
        with torch.no_grad():
            audio = self.model(mel_specs)
            audio = audio * self.cfg.hifi.MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")
        return audio

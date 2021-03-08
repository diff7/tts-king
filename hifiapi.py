import torch
from hifi.models import Generator


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class HIFIapi:
    def __init__(self, config, weights_path=None, device="gpu"):

        # Load checkpoint if exists
        self.weights_path = weights_path

        if weights_path is not None:
            checkpoint = torch.load(weights_path)

        self.model = Generator(config).to(device)
        self.model.load_state_dict(checkpoint["generator"])

        self.cfg = config
        self.device = device

        # TODO get the righ restore step
        self.restore_step = 0

        self.model.remove_weight_norm()

    # TODO:
    def train(self):
        raise NotImplemented(" Train for HiFi was not implemented yet")

    # TODO:
    def generate(self, mel_specs):
        """
        Converts mel spectrogramma into an audio file.
        Returns cpu audio files.
        mel_specs - a batch of mel spectrogramms
        """

        self.model.eval()
        with torch.no_grad():
            audio = self.model(mel_specs)
            audio = audio * self.cfg.MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")
        return audio

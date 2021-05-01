import torch
import numpy as np
import torch.nn as nn

from fs_two.model import FastSpeech2
from fs_two.model.loss import FastSpeech2Loss

# from train_fs_lighting import train_fs

"""

 python3 prepare_align.py config/LJSpeech/preprocess.yaml
 python3 preprocess.py config/LJSpeech/preprocess.yaml
 python3 train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml


"""


class FSTWOapi:
    def __init__(self, config, device=0):
        weights_path = config.tts.weights_path
        self.model = FastSpeech2(
            config.preprocess_config, config.model_config
        ).to(device)
        # Load checkpoint if exists
        self.weights_path = weights_path
        if weights_path is not None:
            checkpoint = torch.load(weights_path)
            self.model.load_state_dict(checkpoint["model"])

        self.cfg = config
        self.device = device

        # TODO get the righ restore step
        self.restore_step = 0

    def generate(
        self,
        phonemes,
        duration_control=1.0,
        pitch_control=1.0,
        energy_control=1.0,
        speaker=None,
    ):
        if speaker is not None:
            speaker = torch.tensor(speaker).long().unsqueeze(0)
            speaker = speaker.to(self.device)
        self.model.eval()
        src_len = np.array([len(phonemes[0])])
        result = self.model(
            speaker,
            torch.from_numpy(phonemes).long().to(self.device),
            torch.from_numpy(src_len).to(self.device),
            max(src_len),
            d_control=duration_control,
            p_control=pitch_control,
            e_control=energy_control,
        )

        (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        ) = result

        return postnet_output
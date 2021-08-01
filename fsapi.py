import os
import json
import torch
import numpy as np

from fs_two.model import FastSpeech2


class FSTWOapi:
    def __init__(self, config, device=0):
        weights_path = config.tts.weights_path
        model_folder = "/".join(weights_path.split("/")[:-1])
        config.preprocess_config.path.preprocessed_path = model_folder

        self.speakers_dict, self.speaker_names = load_speakers_json(
            config.preprocess_config.path.preprocessed_path
        )

        self.model = FastSpeech2(
            config.preprocess_config,
            config.model_config,
            len(self.speaker_names),
        ).to(device)
        # Load checkpoint if exists
        self.weights_path = weights_path
        if weights_path is not None:
            checkpoint = torch.load(weights_path, map_location="cpu")
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
        speaker_name=None,
    ):

        if speaker_name is not None:
            if not speaker_name in self.speakers_dict:
                raise Exception(
                    f"Speaker {speaker_name} was not found in speakers.json"
                )
            speaker_id = self.speakers_dict[speaker_name]
            speaker = torch.tensor(speaker_id).long().unsqueeze(0)
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
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            postnet_output,
        ) = result

        return postnet_output


def load_speakers_json(dir_path):
    json_paht = os.path.join(dir_path, "speakers.json")
    if os.path.exists(json_paht):
        with open(
            json_paht,
            "r",
        ) as f:
            speakers = json.load(f)
    else:
        print(f'Did not find speakers.josn at {dir_path}')

    return speakers, list(speakers.keys())

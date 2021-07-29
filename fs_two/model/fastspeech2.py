import os
import json

import torch.nn as nn
import torch.nn.functional as F

from fs_two.transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from fs_two.utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(
        self, preprocess_config, model_config, n_speakers=None, device="cpu"
    ):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(
            preprocess_config, model_config, device
        )
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        nn.init.xavier_normal_(self.mel_linear.weight)
        self.speaker_emb = None

        if model_config["multi_speaker"]:
            if n_speakers is None:
                n_speakers = get_speakers_number(preprocess_config)
            self.speaker_emb = nn.Embedding(
                n_speakers,
                model_config["transformer"]["encoder_hidden"],
            )

        self.postnet = PostNet()

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        e_targets=None,
        d_targets=None,
        pitches_cwt=None,
        pitches_mean=None,
        pitches_std=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(
            src_lens, max_src_len, device=texts.device
        )
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len, device=texts.device)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)
        if self.speaker_emb is not None:
            embedding = (
                self.speaker_emb(speakers).unsqueeze(1).expand(-1, 1, -1)
            )
        (
            output,
            pitch_cwt_prediction,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
            pitch_mean,
            pitch_std,
        ) = self.variance_adaptor(
            output,
            embedding,
            src_masks,
            mel_masks,
            max_mel_len,
            pitches_cwt,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            pitch_cwt_prediction,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            postnet_output,
            pitch_mean,
            pitch_std,
        )


def get_speakers_number(preprocess_config):
    speaker_json = os.path.join(
        preprocess_config["path"]["preprocessed_path"],
        "speakers.json",
    )
    if os.path.exists(speaker_json):
        with open(
            speaker_json,
            "r",
        ) as f:
            n_speakers = len(json.load(f))
    else:
        raise Exception(
            "Model is multispeaker but number of speakers was not provided explicitly"
        )
    return n_speakers

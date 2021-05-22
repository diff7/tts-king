import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from fs_two.transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor, RevGrad
from fs_two.utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        n_speaker = 0
        speakers_path = (
            os.path.join(
                preprocess_config["path"]["preprocessed_path"],
                "speakers.json",
            ),
        )
        if os.path.exists(speakers_path):
            with open(
                speakers_path,
                "r",
            ) as f:
                n_speaker = len(json.load(f))

        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        # self.postnet = PostNet(
        #     n_in_channels=model_config["transformer"]["decoder_hidden"])

    def forward(
        self,
        speakers_emb,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
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
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        if self.model_config["concat_speaker"]:
            speakers_emb = speakers_emb.unsqueeze(1).repeat(
                1, output.size(1), 1
            )
            output = torch.cat([output, speakers_emb], 2)
        else:
            output = output + speakers_emb.unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        output, mel_masks = self.decoder(output, mel_masks)

        output = self.mel_linear(output)

        return (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

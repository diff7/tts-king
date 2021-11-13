import os
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function

from fs_two.utils.tools import get_mask_from_lengths, pad
from fs_two.cwt.cwt_utils import inverse_batch_cwt


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config, device):
        super(VarianceAdaptor, self).__init__()
        self.device = device

        hidden_size = model_config["transformer"]["variance_hidden"]

        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()

        
        if model_config.use_cwt:
            self.use_cwt = True
            self.pitch_predictor = VariancePredictor(
                model_config, output_size=11, dropout=0.1
            )
        else:
            self.use_cwt = False
            self.pitch_predictor = VariancePredictor(model_config)

        # PitchPredictor(hidden_size, cwt_size=11)

        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_mean = CNNscalar(size_one=hidden_size, size_two=11)
        self.pitch_std = CNNscalar(size_one=hidden_size, size_two=11)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"][
            "energy"
        ]["feature"]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"][
            "pitch_quantization"
        ]
        energy_quantization = model_config["variance_embedding"][
            "energy_quantization"
        ]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "stats.json"
            )
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(
                        np.log(pitch_min), np.log(pitch_max), n_bins - 1
                    )
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(
                        np.log(energy_min), np.log(energy_max), n_bins - 1
                    )
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )


    def get_pitch_embedding_normal(self, x, target, mask, control=1):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding


    def get_pitch_embedding_cwt(self, x, pitch_target_cwt, mask, control=1):
        # batch, seq_len, 10 -> batch, 10 -> batch, 1
        mask = mask.unsqueeze(2)
        mask = mask.repeat(1, 1, 11)
        pitch_cwt_prediction = self.pitch_predictor(x, mask)

        # NOTE: Might be more stable if train on Ground Truth
        # if pitch_target_cwt is None:
        #     pitch_cwt = pitch_cwt_prediction
        # else:
        #     pitch_cwt = pitch_target_cwt

        pitch_cwt = pitch_cwt_prediction

        pitch_mean = self.pitch_mean(x.detach(), pitch_cwt.detach())
        pitch_std = self.pitch_std(x.detach(), pitch_cwt.detach())

        pitch = inverse_batch_cwt(pitch_cwt)

        # print(pitch.shape)
        # print(pitch_std.shape)
        # print(pitch_mean.shape)
        pitch = (pitch * pitch_std) + pitch_mean

        pitch_embedding = self.pitch_embedding(
            torch.bucketize(pitch * control, self.pitch_bins)
        )
        return pitch_cwt_prediction, pitch_embedding, pitch_mean, pitch_std

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(
                torch.bucketize(target, self.energy_bins)
            )
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        embedding,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_raw_target = None,
        pitch_cwt_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        x = x + embedding
        if self.use_cwt:
            (
                pitch_prediction,
                pitch_embedding,
                pitch_mean,
                pitch_std,
            ) = self.get_pitch_embedding_cwt(
                x,
                pitch_cwt_target,
                src_mask,
                p_control,
            )
        else:
            (
                pitch_prediction,
                pitch_embedding,
            ) = self.get_pitch_embedding_normal(
                x,
                pitch_raw_target,
                src_mask,
                p_control,
            )
            pitch_mean = None
            pitch_std = None
        
        x = x + pitch_embedding
        
        energy_prediction, energy_embedding = self.get_energy_embedding(
                x,
                energy_target,
                src_mask,
                e_control,
            )
        x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (
                    torch.round(torch.exp(log_duration_prediction) - 1)
                    * d_control
                ),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len, device=self.device)

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
            pitch_mean,
            pitch_std,
        )


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, model_config, output_size=1, dropout=None):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["variance_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"][
            "filter_size"
        ]
        if dropout is None:
            self.dropout = model_config["variance_predictor"]["dropout"]
        else:
            self.dropout = dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, output_size)
        nn.init.xavier_normal_(self.linear_layer.weight)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class CNNflat(nn.Module):
    def __init__(self, size, reduce=30):
        super(CNNflat, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(size, 1, 1),
            nn.AdaptiveAvgPool1d(reduce),
            nn.LayerNorm(reduce),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.net(x)


class CNNscalar(nn.Module):
    def __init__(self, size_one, size_two, reduce=30):
        super(CNNscalar, self).__init__()
        self.flat_one = CNNflat(size_one, reduce)
        self.flat_two = CNNflat(size_two, reduce)
        self.linear = nn.Linear(reduce, 1)
        self.relu = nn.ReLU()

    def forward(self, x_one, x_two):
        x_one = self.flat_one(x_one)
        x_two = self.flat_two(x_two)
        out = self.linear(x_one + x_two)
        return self.relu(out).squeeze(1)


# class CNNscalar(nn.Module):
#     # TODO change to attention or new MLP
#     def __init__(self, hidden_size, cwt_size):
#         super(CNNscalar, self).__init__()
#         self.avg = nn.AdaptiveAvgPool1d(1)
#         self.norm_cwt = nn.LayerNorm(cwt_size)
#         self.norm_hidden = nn.LayerNorm(hidden_size)

#         self.linear_cwt = nn.Linear(cwt_size, 1)
#         self.linear_hidden = nn.Linear(hidden_size, 1)

#         self.relu = nn.ReLU()

#     def forward(self, hidden, cwt):
#         print("hidden:", hidden.shape)
#         print("cwt:", cwt.shape)
#         hidden = hidden.transpose(1, 2)
#         cwt = cwt.transpose(1, 2)
#         out_hidden = self.norm_hidden(self.avg(hidden).squeeze(2))
#         out_cwt = self.norm_cwt(self.avg(cwt).squeeze(2))
#         out_hidden = self.linear_hidden(out_hidden)
#         out_cwt = self.linear_cwt(out_cwt)
#         return self.relu(out_hidden) + self.relu(out_cwt)

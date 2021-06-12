import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

class NLayerDiscriminator(nn.Module):
    def __init__(self,inut_dim,  ndf, n_layers, downsampling_factor, last=False):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(inut_dim, inut_dim//2, kernel_size=15),
            nn.LeakyReLU(0.2, True),
            WNConv1d(inut_dim//2, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )
        if last:
            model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
            )
        else:
            model["layer_%d" % (n_layers + 2)] = WNConv1d(
                nf, 80, kernel_size=3, stride=1, padding=1
            )

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self, inut_dim, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleList()
        for i in range(num_D-1):
            self.model.append(NLayerDiscriminator(
                inut_dim, ndf, n_layers, downsampling_factor
            ))
        self.model.append(NLayerDiscriminator(
                inut_dim, ndf, n_layers, downsampling_factor, True
            ))
        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        print('SHAPE', x.shape)
        results = []
        for i, disc in enumerate(self.model):
            results.append(disc(x))
            x = self.downsample(x)
        return results

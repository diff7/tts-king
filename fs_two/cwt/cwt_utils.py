import torch
import numpy as np
import pycwt as wavelet
from sklearn import preprocessing


def mse(a, b):
    return ((a - b) ** 2).mean()


# PREPROCESSING


def transform_cwt(lf0, J=10):
    mother = wavelet.MexicanHat()
    dt = 0.005
    dj = 1
    s0 = dt * 2
    Wavelet_lf0, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        np.squeeze(lf0), dt, dj, s0, J, mother
    )
    Wavelet_lf0 = np.real(Wavelet_lf0).T
    return Wavelet_lf0


def inverse_cwt(wavelet_coefs, scales=[]):
    lf0_rec = np.zeros([wavelet_coefs.shape[0], len(scales)])
    for i in range(0, len(scales)):
        lf0_rec[:, i] = wavelet_coefs[:, i] * ((i + 1 + 2.5) ** (-2.5))
    lf0_rec_sum = np.sum(lf0_rec, axis=1)
    lf0_rec_sum = preprocessing.scale(lf0_rec_sum)
    return lf0_rec_sum


# TO REVERSE ADD
# reverse  = inverse_batch_cwt(wavelet_coefs, scales=10)*std + mean


def scaler(tensor, axis=-1):
    mean = tensor.mean(axis)
    std = tensor.std(axis)
    res = (tensor - mean.reshape(-1, 1)) / std.reshape(-1, 1)
    return res


def inverse_batch_cwt(wavelet_coefs, num_scales=10):
    batch_size = wavelet_coefs.shape[0]
    length = wavelet_coefs.shape[1]
    lf0_rec = torch.zeros(
        [batch_size, length, num_scales)], dtype=torch.float32
    )
    for i in range(0, num_scales):
        lf0_rec[:, :, i] = wavelet_coefs[:, :, i] * ((i + 1 + 2.5) ** (-2.5))
    lf0_rec_sum = torch.sum(lf0_rec, axis=-1)
    lf0_rec_sum = scaler(lf0_rec_sum)
    print(lf0_rec_sum[0][:10])
    return lf0_rec_sum
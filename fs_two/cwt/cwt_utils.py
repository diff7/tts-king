from sklearn import preprocessing
import pycwt as wavelet
import numpy as np


def mse(a, b):
    return ((a-b)**2).mean()


# PREPROCESSING


def transform_cwt(lf0, J=10):
    mother = wavelet.MexicanHat()
    dt = 0.005
    dj = 1
    s0 = dt*2
    Wavelet_lf0, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        np.squeeze(lf0), dt, dj, s0, J, mother)
    Wavelet_lf0 = np.real(Wavelet_lf0).T
    return Wavelet_lf0


def inverse_cwt(Wavelet_lf0, scales=10):
    lf0_rec = np.zeros([Wavelet_lf0.shape[0], len(scales)])
    for i in range(0, len(scales)):
        lf0_rec[:, i] = Wavelet_lf0[:, i]*((i+1+2.5)**(-2.5))
    lf0_rec_sum = np.sum(lf0_rec, axis=1)
    lf0_rec_sum = preprocessing.scale(lf0_rec_sum)
    return lf0_rec_sum

import numpy as np
import scipy.signal as signal

def make_spectrogram(data, fs, t_res=0.5, f_res=0.5, log=True, return_bins=False):
    n_per_seg = round(fs / f_res)
    overlap = 1 - t_res * fs / n_per_seg
    noverlap = round(overlap*n_per_seg)

    f, t, spectro = signal.spectrogram(data, fs, nperseg=n_per_seg, noverlap=noverlap)
    spectro = spectro[::-1]

    if log:
        spectro = np.log10(spectro)

    if return_bins:
        return f, t, spectro
    else:
        return spectro
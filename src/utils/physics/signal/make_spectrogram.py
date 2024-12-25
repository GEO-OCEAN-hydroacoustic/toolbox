import numpy as np
import scipy.signal as signal

def make_spectrogram(data, fs, t_res=0.5, f_res=0.5, log=True, return_bins=False, vmin=None, vmax=None, normalize=False):
    n_per_seg = round(fs / f_res)
    overlap = 1 - t_res * fs / n_per_seg
    noverlap = round(overlap*n_per_seg)

    f, t, spectro = signal.spectrogram(data, fs, nperseg=n_per_seg, noverlap=noverlap)
    spectro = spectro[::-1]

    if log:
        spectro = 10*np.log10(spectro)
    vmin = vmin or np.min(spectro)
    vmax = vmax or np.max(spectro)
    spectro[spectro < vmin] = vmin
    spectro[spectro > vmax] = vmax
    if normalize:  # rescale between 0 and 1, using vmin and vmax as bounds if defined or local normalization otherwise
        spectro = (spectro - vmin) / (vmax - vmin)

    if return_bins:
        return f, t, spectro
    else:
        return spectro
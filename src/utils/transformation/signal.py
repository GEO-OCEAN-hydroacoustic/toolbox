import numpy as np
from scipy.signal import butter, sosfilt, filtfilt


def _butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band', analog=False, output='sos')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, pad_s=10):
    sos = _butter_bandpass(lowcut, highcut, fs, order=order)
    data = [np.mean(data[:pad_s*int(fs)])] * pad_s * int(fs) + list(data)
    y = sosfilt(sos, data)[pad_s * int(fs):]
    return y

def _butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = _butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def _butter_highpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='high', analog=False)

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = _butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

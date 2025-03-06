import numpy as np
from scipy import signal

def downsample_audio(data, original_fs, target_fs):
    """Downsample audio data to the target frequency"""
    # Calculate downsampling factor
    factor = int(original_fs / target_fs)
    # Apply anti-aliasing filter before downsampling
    b, a = signal.butter(5, target_fs/2, fs=original_fs, btype='low')
    filtered_data = signal.filtfilt(b, a, data)
    # Downsample by taking every 'factor' sample
    downsampled_data = filtered_data[::factor]
    return downsampled_data, target_fs

def dehaze_audio(data, fs, frame_size=1024, overlap=0.8):
    """Apply spectral subtraction for dehazing"""
    hop_size = int(frame_size * (1 - overlap))
    # Estimate noise profile from first few frames
    num_noise_frames = 5
    noise_estimate = np.zeros(frame_size // 2 + 1)
    frames = []
    for i in range(0, len(data) - frame_size, hop_size):
        frame = data[i:i+frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        frames.append(frame)

    # Estimate noise from first few frames
    for i in range(min(num_noise_frames, len(frames))):
        noise_frame = frames[i]
        noise_spectrum = np.abs(np.fft.rfft(noise_frame * np.hanning(frame_size)))
        noise_estimate += noise_spectrum / num_noise_frames

    # Apply spectral subtraction
    result = np.zeros(len(data))
    window = np.hanning(frame_size)

    for i, frame in enumerate(frames):
        windowed_frame = frame * window
        spectrum = np.fft.rfft(windowed_frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Subtract noise and ensure no negative values
        magnitude = np.maximum(magnitude - noise_estimate * 1.5, 0.01 * magnitude)

        # Reconstruct frame
        enhanced_spectrum = magnitude * np.exp(1j * phase)
        enhanced_frame = np.fft.irfft(enhanced_spectrum)

        # Overlap-add
        start = i * hop_size
        end = start + frame_size
        result[start:end] += enhanced_frame

    # Normalize
    result = result / np.max(np.abs(result))
    return result

def apply_butter_bandpass(data, fs, lowcut, highcut, order=5):
    """Apply Butterworth bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def energy_plot(data, fs, window_size=1.0):
    """Detect potential seismic events based on energy threshold"""
    window_samples = int(window_size * fs)
    energy = []
    # Calculate energy in sliding windows
    for i in range(0, len(data) - window_samples, window_samples // 2):
        window = data[i:i + window_samples]
        window_energy = np.sum(window ** 2) / len(window)
        energy.append(window_energy)

    # Set threshold as a factor of the median energy
    energy = np.array(energy)
    threshold = np.median(energy) * 2  # Adjustable multiplier
    event_indices = np.where(energy > threshold)[0]

    return event_indices, energy, threshold
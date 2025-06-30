"""
refine_detections.py

"""
import numpy as np
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.data_reading.sound_data.sound_file_manager import DatFilesManager
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob2
from pathlib import Path
from utils.detection.association import load_detections
from utils.data_reading.sound_data.station import StationsCatalog

# === CONFIGURATION ===
# Path to your associations .npy file
ASSO_FILE = "/media/rsafran/CORSAIR/Association/2018/grids/2018/s_-60-5,35-120,350,0.8,0.6.npy"

# Root directory containing subfolders for each station
DATA_ROOT = "/media/rsafran/CORSAIR/OHASISBIO"
# Subfolder (e.g. year) within DATA_ROOT
DATASET   = "2018"
# Where to save the refined detections
OUTPUT_FILE = "/home/rsafran/PycharmProjects/toolbox/data/detection/association/refined_-60-5,35-120,350,0.8,0.6.npy"
# Number of parallel workers (adjust to CPU cores)
WORKERS = 27

# Signal processing parameters
SAMPLING_RATE = 240    # Hz (override if your DatFilesManager provides a different rate)
SMOOTH_WINDOW_SEC = 5.0
SIGNAL_WINDOW_SEC = 2.0
NOISE_WINDOW_SEC  = 10.0

drift = {"ELAN"    :[-0.0222, 'ok'],
         "MADE"    :[-0.7980, 'ok'],
         "MADW"    :[ 0.0020, 'not_ok'],
         "NEAMS"   :[ 0.2601, 'not_ok'],
         "RTJ"     :[ 0.0300, 'not_ok'],
         "SSEIR"   :[-0.2861, 'not_ok'],
         "SSWIR"   :[ 0.0346, 'ok'],
         "SWAMSbot":[ 0.0048, 'ok'],
         "WKER2"   :[-0.0203, 'not_ok']}
drift_raw = {"MADW"    :[ 0.0020, 'not_ok'],
             "NEAMS"   :[ 0.2601, 'not_ok'],
             "RTJ"     :[ 0.0300, 'not_ok'],
             "SSEIR"   :[-0.2861, 'not_ok'],
             "WKER2"   :[-0.0203, 'not_ok']}

drift_ok = {"ELAN"    :[-0.0222, 'ok'],
            "MADE"    :[-0.7980, 'ok'],
            "SSWIR"   :[ 0.0346, 'ok'],
            "SWAMSbot":[ 0.0048, 'ok']}

def get_energy_envelope(data, fs=SAMPLING_RATE, window_sec=SMOOTH_WINDOW_SEC):
    """
    Compute a moving-average energy envelope of the squared signal.
    """
    win = int(window_sec * fs)
    if win < 1:
        raise ValueError(f"Window too short: {win} samples")
    kernel = np.ones(win) / win
    return np.convolve(data**2, kernel, mode="same")


def refine_single_detection(station_obj, det_time):
    """
    Refine one pick time by locating the max energy within ±SMOOTH_WINDOW_SEC.
    Also computes a simple SNR ratio.
    Returns a dict or None on failure.
    """
    # load waveform ±3 minutes
    start = det_time - timedelta(minutes=1)
    end   = det_time + timedelta(minutes=1)
    try :
        raw = 'raw' if station_obj.name in drift_raw else None
        mgr = DatFilesManager(f"{DATA_ROOT}/{DATASET}/{station_obj.name}", kwargs=raw)
        data = mgr.get_segment(start, end)
    except Exception as e:
        raw = 'raw' if station_obj.name in drift_raw else None
        mgr = DatFilesManager(f"{DATA_ROOT}/{2017}/{station_obj.name}",kwargs=raw)
        data = mgr.get_segment(start, end)
    # energy envelope
    fs = getattr(mgr, 'sampling_rate', SAMPLING_RATE)
    energy = get_energy_envelope(data, fs)

    # time vector relative to det_time
    n = len(data)
    dt = 1.0 / fs
    rel_times = np.arange(n) * dt - (det_time - start).total_seconds()

    # mask ±SMOOTH_WINDOW_SEC
    mask = (rel_times >= -SMOOTH_WINDOW_SEC) & (rel_times <= SMOOTH_WINDOW_SEC)
    if not mask.any():
        return None
    sub = energy[mask]
    idx0 = np.argmax(sub)
    idx = np.where(mask)[0][idx0]

    # refined time
    offset = rel_times[idx]
    refined = det_time + timedelta(seconds=offset)

    # SNR: signal vs preceding noise
    sig_win  = int(SIGNAL_WINDOW_SEC * fs)
    noise_win = int(NOISE_WINDOW_SEC * fs)
    s0 = max(0, idx - sig_win//2)
    s1 = min(n, idx + sig_win//2)
    sig_e = np.mean(energy[s0:s1])
    n1 = max(0, s0 - noise_win)
    noise_e = np.mean(energy[n1:s0]) if s0 > n1 else sig_e * 1e-3
    snr = sig_e / (noise_e + 1e-12)

    return {
        'station': station_obj.name,
        'orig_time': det_time,
        'refined_time': refined,
        'snr': snr
    }


def process_associations():
    """
    Load associations, refine every detection, and save results.
    """
    print(f"Loading associations from {ASSO_FILE}...")
    assoc = np.load(ASSO_FILE, allow_pickle=True).item()

    tasks = []
    for date, groups in assoc.items():
        for detections, _ in groups:
            for station_obj, det_time in detections:
                tasks.append((station_obj, det_time))

    # # right after you build tasks = [...]
    # toy = tasks[:10]
    # results = []
    # for station_obj, det_time in toy:
    #     # 1) load ±1 min
    #     raw = 'raw' if station_obj.name in drift_raw else None
    #     print(raw)
    #     start = det_time - timedelta(minutes=1)
    #     end = det_time + timedelta(minutes=1)
    #     mgr = DatFilesManager(f"{DATA_ROOT}/{2017}/{station_obj.name}",raw)
    #     data = mgr.get_segment(start, end)
    #
    #     # 2) compute envelope
    #     fs = getattr(mgr, 'sampling_rate', SAMPLING_RATE)
    #     energy = get_energy_envelope(data, fs)
    #
    #     # 3) build time axis relative to original pick
    #     n = len(data)
    #     t = np.arange(n) / fs - (det_time - start).total_seconds()
    #
    #     # 4) find the “repointed” offset
    #     mask = (t >= -SMOOTH_WINDOW_SEC) & (t <= SMOOTH_WINDOW_SEC)
    #     sub = energy[mask]
    #     idx0 = np.argmax(sub)
    #     idx = np.where(mask)[0][idx0]
    #     offset = t[idx]  # seconds relative to original pick
    #
    #     # 5) plot raw waveform + markers
    #     plt.figure(figsize=(8, 3))
    #     plt.plot(t, data, label="Raw signal")
    #     plt.axvline(0, color='k', linestyle='--', label="Original pick")
    #     plt.axvline(offset, color='r', linestyle='--', label="Refined pick")
    #     plt.title(f"{station_obj.name} raw signal\norig: {det_time}, refined: {det_time + timedelta(seconds=offset)}")
    #     plt.xlabel("Time (s) rel. to original pick")
    #     plt.ylabel("Amplitude")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    #
    #     # 6) plot energy envelope + markers
    #     plt.figure(figsize=(8, 3))
    #     plt.plot(t, energy, label="Energy envelope")
    #     plt.axvline(0, color='k', linestyle='--', label="Original pick")
    #     plt.axvline(offset, color='r', linestyle='--', label="Refined pick")
    #     plt.title(f"{station_obj.name} energy envelope")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Energy")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    print(f"Refining {len(tasks)} detections using {WORKERS} workers...")
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(refine_single_detection, s, t) for s, t in tasks]
        with tqdm(total=len(futures), desc="Refining detections") as pbar:
            for f in as_completed(futures):
                r = f.result()
                if r is not None:
                    results.append(r)
                pbar.update(1)

    # after you've collected `results` as a list of dicts:
    #   results_map[(station_name, orig_time)] = refined_time

    results_map = {
        (r['station'], r['orig_time']): r['refined_time']
        for r in results
    }

    refined_assoc = {}
    for date, groups in assoc.items():
        new_groups = []
        for detections, valid_points in groups:
            # detections is an array of shape (N,2): [(station_obj, det_time), …]
            # we'll rebuild it with the new times
            new_det = []
            for station_obj, det_time in detections:
                key = (station_obj.name, det_time)
                new_time = results_map.get(key, det_time)
                new_det.append((station_obj, new_time))
            # preserve the same numpy-array structure
            new_det_array = np.array(new_det, dtype=detections.dtype)
            new_groups.append((new_det_array, valid_points))
        refined_assoc[date] = new_groups


    print(f"Saving refined detections to {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, refined_assoc)
    print("Done.")


if __name__ == '__main__':
    process_associations()

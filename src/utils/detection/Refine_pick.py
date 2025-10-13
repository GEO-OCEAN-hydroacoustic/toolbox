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


# paths
CATALOG_PATH = "/media/rsafran/CORSAIR/OHASISBIO/recensement_stations_OHASISBIO_RS.csv"
# DETECTIONS_DIR = "/media/rsafran/CORSAIR/temp/2018"
DETECTIONS_DIR = "/media/rsafran/CORSAIR/detections_CTBT/"
ASSOCIATION_OUTPUT_DIR = "../../../data/detection/association"

# Detections loading parameters
RELOAD_DETECTIONS = False # if False, load files called "detections.npy" and "detections_merged.npy" containing everything instead of the raw detection output. Leave at True by default
MIN_P_TISSNET_PRIMARY = 0.5  # min probability of browsed detections
MIN_P_TISSNET_SECONDARY = 0.15  # min probability of detections that can be associated with the browsed one
MERGE_DELTA_S = 10 # threshold below which we consider two events should be merged
MERGE_DELTA = timedelta(seconds=MERGE_DELTA_S)

REQ_CLOSEST_STATIONS = 0  # The REQ_CLOSEST_STATIONS th closest stations will be required for an association to be valid



# Root directory containing subfolders for each station
DATA_ROOT = "/media/rsafran/CORSAIR/OHASISBIO"
# Subfolder (e.g. year) within DATA_ROOT
DATASET   = "OHASISBIO-2018"
# Where to save the refined detections
OUTPUT_FILE = "/home/rsafran/Bureau/tissnet/2018/cache/refined_detections_merged.npy"
# Number of parallel workers (adjust to CPU cores)
WORKERS = 1

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

drift_ok = {"ELAN"    :-0.0222,
            "MADE"    :-0.7980,
            "SSWIR"   : 0.0346,
            "SWAMSbot": 0.0048}


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
    start = det_time - timedelta(minutes=1.5)
    end   = det_time + timedelta(minutes=1.5)
    try :
        raw = 'raw' if station_obj.name in drift_raw else 'raw'
        mgr = DatFilesManager(f"{DATA_ROOT}/{DATASET}/{station_obj.name}", kwargs=raw)
        data = mgr.get_segment(start, end)
    except Exception as e:
        raw = 'raw' if station_obj.name in drift_raw else 'raw'
        mgr = DatFilesManager(f"{DATA_ROOT}/{'OHASISBIO-2017'}/{station_obj.name}",kwargs=raw)
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
    if station_obj.name in drift_ok:
        refined -= (refined - station_obj.date_start) *(drift_ok[station_obj.name]*10**-6)
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
    print(f"Loading associations from {DATA_ROOT}...")

    STATIONS = StationsCatalog(CATALOG_PATH).filter_out_undated().filter_out_unlocated()

    if RELOAD_DETECTIONS:
        det_files = [f for f in glob2.glob(DETECTIONS_DIR + "/*") if Path(f).is_file()]
        det_files = [f for f in det_files if "2018" in f]
        DETECTIONS, DETECTIONS_MERGED = load_detections(det_files, STATIONS, DETECTIONS_DIR, MIN_P_TISSNET_PRIMARY,
                                                        MIN_P_TISSNET_SECONDARY, MERGE_DELTA)
    else:
        # DETECTIONS = np.load(f"{DETECTIONS_DIR}/cache/detections.npy", allow_pickle=True).item()
        DETECTIONS_MERGED = np.load(f"{DETECTIONS_DIR}/cache/detections_merged.npy", allow_pickle=True)

    tasks = []

    for det_time, _, station_obj in DETECTIONS_MERGED:
        if "OHASISBIO" in station_obj.dataset :
            tasks.append((station_obj, det_time))
    print(len(tasks))
    # right after you build tasks = [...]
    toy = tasks[:10]

    for station_obj, det_time in toy:
        # 1) load ±1 min
        raw = 'raw' if station_obj.name in drift_raw else None
        print(raw)
        start = det_time - timedelta(minutes=1)
        end = det_time + timedelta(minutes=1)
        mgr = DatFilesManager(f"{DATA_ROOT}/{"OHASISBIO-2018"}/{station_obj.name}",raw)
        data = mgr.get_segment(start, end)

        # 2) compute envelope
        fs = getattr(mgr, 'sampling_rate', SAMPLING_RATE)
        energy = get_energy_envelope(data, fs)

        # 3) build time axis relative to original pick
        n = len(data)
        t = np.arange(n) / fs - (det_time - start).total_seconds()

        # 4) find the “repointed” offset
        mask = (t >= -SMOOTH_WINDOW_SEC) & (t <= SMOOTH_WINDOW_SEC)
        sub = energy[mask]
        idx0 = np.argmax(sub)
        idx = np.where(mask)[0][idx0]
        offset = t[idx]  # seconds relative to original pick

        # 5) plot raw waveform + markers
        plt.figure(figsize=(8, 3))
        plt.plot(t, data, label="Raw signal")
        plt.axvline(0, color='k', linestyle='--', label="Original pick")
        plt.axvline(offset, color='r', linestyle='--', label="Refined pick")
        plt.title(f"{station_obj.name} raw signal\norig: {det_time}, refined: {det_time + timedelta(seconds=offset)}")
        plt.xlabel("Time (s) rel. to original pick")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 6) plot energy envelope + markers
        plt.figure(figsize=(8, 3))
        plt.plot(t, energy, label="Energy envelope")
        plt.axvline(0, color='k', linestyle='--', label="Original pick")
        plt.axvline(offset, color='r', linestyle='--', label="Refined pick")
        plt.title(f"{station_obj.name} energy envelope")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy")
        plt.legend()
        plt.tight_layout()
        plt.show()

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


    results_map = {
        (r['station'], r['orig_time']): r['refined_time']
        for r in results
    }

    refined_detections_merged = DETECTIONS_MERGED.copy()
    for i, (det_time, _, station_obj) in enumerate(refined_detections_merged):
        key = (station_obj.name, det_time)
        new_time = results_map.get(key, det_time)
        refined_detections_merged[i][0]=new_time



    print(f"Saving refined detections to {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, refined_detections_merged)
    print("Done.")


if __name__ == '__main__':
    process_associations()

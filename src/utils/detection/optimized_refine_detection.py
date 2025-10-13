"""
refine_detections_optimized.py

Version optimisée pour la gestion de mémoire.
"""
import numpy as np
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.data_reading.sound_data.sound_file_manager import DatFilesManager
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gc
import psutil
import time
from itertools import islice

# === CONFIGURATION ===
# Path to your associations .npy file
ASSO_FILE = "/media/rsafran/CORSAIR/Association/2018/grids/2018/s_-60-5,35-120,350,0.8,0.6.npy"
# Root directory containing subfolders for each station
DATA_ROOT = "/media/rsafran/CORSAIR/OHASISBIO"
# Subfolder (e.g. year) within DATA_ROOT
DATASET   = "2018"
# Where to save the refined detections
OUTPUT_FILE = "/home/rsafran/PycharmProjects/toolbox/data/detection/association/refined_-60-5,35-120,350,0.8,0.6.npy"
# Temporary directory for intermediate results
TEMP_DIR = os.path.join(os.path.dirname(OUTPUT_FILE), "temp_refined")
# Number of parallel workers (adjust to CPU cores)
WORKERS = 27
# Maximum batch size of tasks to process at once
BATCH_SIZE = 1000
# Memory threshold in percentage (adjust processing when memory usage exceeds this)
MEMORY_THRESHOLD = 80
# Sleep time in seconds when memory is high
MEMORY_SLEEP = 5

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


def get_memory_usage():
    """Return the current memory usage as a percentage"""
    return psutil.virtual_memory().percent


def get_energy_envelope(data, fs=SAMPLING_RATE, window_sec=SMOOTH_WINDOW_SEC):
    """
    Compute a moving-average energy envelope of the squared signal.
    """
    win = int(window_sec * fs)
    if win < 1:
        raise ValueError(f"Window too short: {win} samples")
    kernel = np.ones(win) / win
    result = np.convolve(data**2, kernel, mode="same")
    return result


def refine_single_detection(station_obj, det_time):
    """
    Refine one pick time by locating the max energy within ±SMOOTH_WINDOW_SEC.
    Also computes a simple SNR ratio.
    Returns a dict or None on failure.
    """
    # load waveform ±1 minute
    start = det_time - timedelta(minutes=1)
    end   = det_time + timedelta(minutes=1)
    
    try:
        # Try to load data from the primary dataset
        raw = 'raw' if station_obj.name in drift_raw else None
        mgr = DatFilesManager(f"{DATA_ROOT}/{DATASET}/{station_obj.name}", kwargs=raw)
        data = mgr.get_segment(start, end)
    except Exception:
        try:
            # Try to load data from the 2017 dataset as fallback
            raw = 'raw' if station_obj.name in drift_raw else None
            mgr = DatFilesManager(f"{DATA_ROOT}/2017/{station_obj.name}", kwargs=raw)
            data = mgr.get_segment(start, end)
        except Exception as e:
            print(f"Error loading data for {station_obj.name} at {det_time}: {e}")
            # If both fail, return None
            return None
    
    # energy envelope
    fs = getattr(mgr, 'sampling_rate', SAMPLING_RATE)
    energy = get_energy_envelope(data, fs)
    
    # Clean up references to large data objects immediately after use
    del mgr
    
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
    
    # Clean up large arrays to free memory
    del data, energy, rel_times, mask
    
    return {
        'station': station_obj.name,
        'orig_time': det_time,
        'refined_time': refined,
        'snr': snr
    }


def create_temp_dir():
    """Create temporary directory if it doesn't exist"""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)


def save_batch_results(batch_results, batch_num):
    """Save batch results to a temporary file"""
    batch_file = os.path.join(TEMP_DIR, f"batch_{batch_num}.npy")
    np.save(batch_file, batch_results)
    return batch_file


def load_batch_results(batch_files):
    """Load and combine results from batch files"""
    results = []
    for batch_file in batch_files:
        batch_results = np.load(batch_file, allow_pickle=True)
        results.extend(batch_results)
        # Remove the temporary file after loading
        os.remove(batch_file)
    return results


def process_batch(batch_tasks, batch_num):
    """Process a batch of tasks using parallel workers"""
    batch_results = []
    
    # Dynamically adjust worker count based on memory usage
    current_memory = get_memory_usage()
    workers = max(1, int(WORKERS * (1 - current_memory / 100)))
    
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(refine_single_detection, s, t) for s, t in batch_tasks]
        with tqdm(total=len(futures), desc=f"Batch {batch_num}") as pbar:
            for f in as_completed(futures):
                r = f.result()
                if r is not None:
                    batch_results.append(r)
                pbar.update(1)
                
                # Check memory usage and sleep if necessary
                if get_memory_usage() > MEMORY_THRESHOLD:
                    time.sleep(MEMORY_SLEEP)
    
    # Force garbage collection after each batch
    gc.collect()
    
    return batch_results


def associations_iterator(assoc):
    """Iterator that yields individual detection tasks from the associations dict"""
    for date, groups in assoc.items():
        for detections, _ in groups:
            for station_obj, det_time in detections:
                yield (station_obj, det_time)


def process_associations():
    """
    Load associations, refine every detection in batches, and save results.
    """
    print(f"Loading associations from {ASSO_FILE}...")
    assoc = np.load(ASSO_FILE, allow_pickle=True).item()
    
    # Create temporary directory
    create_temp_dir()
    
    # Process in batches to manage memory
    all_tasks = associations_iterator(assoc)
    batch_num = 0
    batch_files = []
    
    while True:
        # Extract next batch
        batch_tasks = list(islice(all_tasks, BATCH_SIZE))
        if not batch_tasks:
            break
            
        batch_num += 1
        print(f"Processing batch {batch_num} with {len(batch_tasks)} tasks...")
        
        # Process batch
        batch_results = process_batch(batch_tasks, batch_num)
        
        # Save batch results
        if batch_results:
            batch_file = save_batch_results(batch_results, batch_num)
            batch_files.append(batch_file)
        
        # Clear memory
        del batch_tasks, batch_results
        gc.collect()
        
        # Wait if memory usage is high
        while get_memory_usage() > MEMORY_THRESHOLD:
            print(f"Memory usage high ({get_memory_usage()}%). Waiting...")
            time.sleep(MEMORY_SLEEP)
    
    # Load all batch results and combine
    print("Combining batch results...")
    results = load_batch_results(batch_files)
    
    # Create result map
    results_map = {
        (r['station'], r['orig_time']): r['refined_time']
        for r in results
    }
    
    # Clean up results list to free memory
    del results
    gc.collect()
    
    # Build refined associations object in memory-efficient manner
    print("Building refined associations...")
    refined_assoc = {}
    
    for date, groups in assoc.items():
        new_groups = []
        for detections, valid_points in groups:
            new_det = []
            for station_obj, det_time in detections:
                key = (station_obj.name, det_time)
                new_time = results_map.get(key, det_time)
                new_det.append((station_obj, new_time))
            
            # Preserve the same numpy-array structure
            new_det_array = np.array(new_det, dtype=detections.dtype)
            new_groups.append((new_det_array, valid_points))
        
        refined_assoc[date] = new_groups
    
    # Clean up mapping
    del results_map
    gc.collect()
    
    print(f"Saving refined detections to {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, refined_assoc)
    
    # Clean up temporary directory if empty
    if os.path.exists(TEMP_DIR) and not os.listdir(TEMP_DIR):
        os.rmdir(TEMP_DIR)
    
    print("Done.")


if __name__ == '__main__':
    process_associations()

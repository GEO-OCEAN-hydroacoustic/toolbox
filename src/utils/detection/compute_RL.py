from src.utils.data_reading.sound_data.sound_file_manager import DatFilesManager, WFilesManager
from utils.data_reading.sound_data.station import StationsCatalog
import os
from utils.physics.sound_model.ellipsoidal_sound_model import GridEllipsoidalSoundModel
from scipy.signal import butter, filtfilt
import numpy as np
import datetime
from datetime import timedelta, tzinfo, timezone
import pickle
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm



# paths
CATALOG_PATH = "/media/rsafran/CORSAIR/OHASISBIO/recensement_stations_OHASISBIO_RS_V2.csv"  # csv catalog files
DETECTIONS_DIR = "/media/rsafran/CORSAIR/T-pick_2.2"  # where we have detection pickles
ISAS_PATH = "/media/rsafran/CORSAIR/ISAS/extracted/2018"
YEAR = 2018
 # output dir
ASSOCIATIONS_DIR = f"../../../data/detection/T-pick_2.2/{YEAR}"


Path(ASSOCIATIONS_DIR).mkdir(exist_ok=True)

# delimitation of the detections we keep (this notebook actually associates the detections of 1 year and 4 hours)
DATE_START = datetime.datetime(YEAR, 1, 1) - datetime.timedelta(hours=2)
DATE_END = datetime.datetime(YEAR+1, 3, 1) + datetime.timedelta(hours=2)
#crise
# DATE_START = datetime.datetime(YEAR, 7, 10) - datetime.timedelta(hours=2)
# DATE_END = datetime.datetime(YEAR+1, 7, 15) + datetime.timedelta(hours=2)

# Detections loading parameters
# MIN_P_TISSNET_PRIMARY = 0.5 # min probability of browsed detections
# MIN_P_TISSNET_SECONDARY = 0.1  # min probability of detections that can be associated with the browsed one
# MERGE_DELTA_S = 0.25# threshold below which we consider two events should be merged

MIN_P_TISSNET_PRIMARY = 0.5 # min probability of browsed detections
MIN_P_TISSNET_SECONDARY = 0.1  # min probability of detections that can be associated with the browsed one
MERGE_DELTA_S = 5 # threshold below which we consider two events should be merged

# The REQ_CLOSEST_STATIONS th closest stations will be required for an association to be valid
# e.g. if we set it to 6, no association of size <6 will be saved (this is useful to save memory)
REQ_CLOSEST_STATIONS = 6

# sound model definition

arr = os.listdir(ISAS_PATH)
file_list = [os.path.join(ISAS_PATH, fname) for fname in arr if fname.endswith('.nc')]
SOUND_MODEL = GridEllipsoidalSoundModel(file_list)
STATIONS = StationsCatalog(CATALOG_PATH).filter_out_undated().filter_out_unlocated()
Path(f"{ASSOCIATIONS_DIR}/cache").mkdir(parents=True, exist_ok=True)
DET_PATH = f"{ASSOCIATIONS_DIR}/cache/detections_{MIN_P_TISSNET_SECONDARY}_{MERGE_DELTA_S}.pkl"
with open(DET_PATH, "rb") as f:
    DETECTIONS = pickle.load(f)

# do not keep detection entries for which the detection list is empty
to_del = []
for s in DETECTIONS.keys():
    if len(DETECTIONS[s]) == 0:
        to_del.append(s)
    if s.name == 'H04S1':# or s.name == 'H04N1' or s.name == 'H01W1':
        to_del.append(s)
for s in to_del:
    del DETECTIONS[s]

# assign an index to each detection
idx_det = 0
IDX_TO_DET = {}
for idx, s in enumerate(DETECTIONS.keys()):
    s.idx = idx  # indexes to store efficiently the associations
    DETECTIONS[s] = list(DETECTIONS[s])
    for i in range(len(DETECTIONS[s])):
        DETECTIONS[s][i] = np.concatenate((DETECTIONS[s][i], [idx_det]))
        IDX_TO_DET[idx_det] = DETECTIONS[s][i]
        idx_det += 1
    DETECTIONS[s] = np.array(DETECTIONS[s])
DETECTION_IDXS = np.array(list(range(idx_det)))

# only keep the stations that appear in the kept detections
STATIONS = [s for s in DETECTIONS.keys()]
FIRSTS_DETECTIONS = {s : DETECTIONS[s][0,0] for s in STATIONS}
LASTS_DETECTIONS = {s : DETECTIONS[s][-1,0] for s in STATIONS}

# list that will be browsed
DETECTIONS_MERGED = np.concatenate([[(det[0], det[1], det[2], s) for det in DETECTIONS[s]] for s in STATIONS if "IMS" not in s.dataset])
DETECTIONS_MERGED = DETECTIONS_MERGED[DETECTIONS_MERGED[:, 1] > MIN_P_TISSNET_PRIMARY]
DETECTIONS_MERGED = DETECTIONS_MERGED[np.argsort(DETECTIONS_MERGED[:, 1])][::-1]

# Root directory containing subfolders for each station
DATA_ROOT = "/media/rsafran/CORSAIR/OHASISBIO"
# Subfolder (e.g. year) within DATA_ROOT
DATASET   = "OHASISBIO-2018"


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    y = filtfilt(b, a, data)
    return y



def refine_single_detection(station_obj, det_time, lowcut=None, highcut=None):
    """
    Refine one pick time by locating the max energy within ±SMOOTH_WINDOW_SEC.
    Also computes a simple SNR ratio.
    Returns a dict or None on failure.
    """
    # load waveform ±3 minutes
    start = det_time - timedelta(minutes=5)
    end   = det_time + timedelta(minutes=5)
    if station_obj.dataset=="OHASISBIO-2018":
        raw = 'raw'
        mgr = DatFilesManager(f"{DATA_ROOT}/{DATASET}/{station_obj.name}", kwargs=raw)
        data = mgr.get_segment(start, end)
        sampling_f = mgr.sampling_f
    else :
        mgr = WFilesManager(f"/media/rsafran/CORSAIR/CTBT/CTBTO_2018/{station_obj.name}_2018/")
        data = mgr.get_segment(start, end)
        sampling_f = round(mgr.sampling_f)

    n = len(data)
    t = np.arange(n) / sampling_f - (det_time - start).total_seconds()
    if lowcut is not None and highcut is not None:
        data = bandpass_filter(data,lowcut, highcut, round(sampling_f))

    return data,t, sampling_f


def compute_RL(data, fs, window_sec=10, step_sec=1, signal_margin_sec=5):

    P_REF = 1e-6

    # -------------------------
    # Fenêtres glissantes
    # -------------------------
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    nwin = int((len(data) - win) / step) + 1

    time_centers = np.zeros(nwin)
    SPL_RMS = np.zeros(nwin)

    for i in range(nwin):
        start = i * step
        end = start + win
        window = data[start:end]
        time_centers[i] = (start) / fs

        # --- RMS ---
        p_rms = np.sqrt(np.mean(window**2))
        SPL_RMS[i] = 20 * np.log10(p_rms / P_REF)

    # -------------------------
    # Identification des zones (asymétrique)
    # -------------------------
    total_duration = len(data) / fs
    signal_start_time = total_duration/2  # Le signal COMMENCE au début


    # Zone étendue avec marges de sécurité
    signal_zone_start = signal_start_time - signal_margin_sec
    signal_zone_end = signal_start_time + window_sec+signal_margin_sec

    # Masques booléens
    is_signal = (time_centers >= signal_zone_start) & (time_centers <= signal_zone_end)
    is_noise = ~is_signal

    # -------------------------
    # Extraction des niveaux
    # -------------------------
    # SPL maximum dans la zone du signal
    if np.any(is_signal):
        SPL_signal_max = np.max(SPL_RMS[is_signal])
    else:
        SPL_signal_max = np.nan
        print("Attention : aucune fenêtre dans la zone du signal")

    # Niveau de bruit (médiane hors signal)
    if np.any(is_noise):
        noise_level = np.median(SPL_RMS[is_noise])
    else:
        noise_level = np.nan
        print("Attention : aucune fenêtre de bruit disponible")

    return SPL_signal_max, noise_level


# -------------------------------------------------------
# 1. Mapping inverse : idx_det → station
# -------------------------------------------------------
IDX_TO_STATION = {}
for s in STATIONS:
    for det in DETECTIONS[s]:
        idx_det = int(det[-1])          # dernier champ = index
        IDX_TO_STATION[idx_det] = s

# -------------------------------------------------------
# 2. Paramètres du filtre (adaptez selon vos données)
# -------------------------------------------------------
LOWCUT  = 5.0    # Hz
HIGHCUT = 100.0   # Hz
N_WORKERS = 20

# -------------------------------------------------------
# Fonction top-level (obligatoire pour pickle avec multiprocessing)
# -------------------------------------------------------
def process_one(args):
    idx_det, det, station, lowcut, highcut = args
    det_time = det[0]
    try:
        data, t, fs = refine_single_detection(station, det_time, lowcut=lowcut, highcut=highcut)
        spl_max, noise_level = compute_RL(data, fs)
        return idx_det, (spl_max, noise_level)
    except Exception as e:
        return idx_det, (np.nan, np.nan)

# -------------------------------------------------------
# Préparation des arguments (pas de globaux dans les workers)
# -------------------------------------------------------
args_list = [
    (idx, IDX_TO_DET[idx], IDX_TO_STATION[idx], LOWCUT, HIGHCUT)
    for idx in IDX_TO_DET.keys()
]


# -------------------------------------------------------
# Exécution
# -------------------------------------------------------
IDX_TO_RL   = {}
FAILED_IDXS = []

with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(process_one, args): args[0] for args in args_list}
    with tqdm(total=len(futures), desc="Computing RL") as pbar:
        for future in as_completed(futures):
            idx_det, result = future.result()
            IDX_TO_RL[idx_det] = result
            if np.isnan(result[0]):
                FAILED_IDXS.append(idx_det)
            pbar.update(1)

print(f"Terminé. {len(IDX_TO_RL)} entrées, {len(FAILED_IDXS)} échecs.")

# Sauvegarde
RL_PATH = f"{ASSOCIATIONS_DIR}/cache/rl_{MIN_P_TISSNET_SECONDARY}_{MERGE_DELTA_S}.pkl"
with open(RL_PATH, "wb") as f:
    pickle.dump(IDX_TO_RL, f)
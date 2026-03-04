from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import glob2
import datetime
from datetime import timedelta
from pathlib import Path
import pickle
from utils.data_reading.sound_data.station import StationsCatalog
from utils.detection.association_geodesic_ridges import compute_candidates, update_valid_grid, update_results, load_detections, compute_grids
import os
from utils.physics.sound_model.ellipsoidal_sound_model import GridEllipsoidalSoundModel

# paths
CATALOG_PATH = "/media/rsafran/CORSAIR/OHASISBIO/recensement_stations_OHASISBIO_RS.csv"  # csv catalog files
DETECTIONS_DIR = "/media/rsafran/CORSAIR/T-pick_2.2"  # where we have detection pickles
ISAS_PATH = "/media/rsafran/CORSAIR/ISAS/extracted/2018"
YEAR = 2018

 # output dir
ASSOCIATIONS_DIR = f"../../../data/detection/T-pick_2.2/{YEAR}"
Path(ASSOCIATIONS_DIR).mkdir(exist_ok=True)

# delimitation of the detections we keep (this notebook actually associates the detections of 1 year and 4 hours)
DATE_START = datetime.datetime(YEAR, 1, 1) - datetime.timedelta(hours=2)
DATE_END = datetime.datetime(YEAR+1, 3, 1) + datetime.timedelta(hours=2)

# Detections loading parameters
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

# GRID_TO_COORDS is a list giving, for each cell of the grid, the coordinates of its center (e.g. GRID_TO_COORDS[25] is a (lat,lon) tuple accounting for cell 25)
with open("../../data/T-pick/grid_to_coords_ridges.pkl", "rb") as f:
    GRID_TO_COORDS = pickle.load(f)

STATIONS = StationsCatalog(CATALOG_PATH).filter_out_undated().filter_out_unlocated()
# load detection files and keep all the ones that are included in the wanted year
FILES = {}
for f in glob2.glob(f"{DETECTIONS_DIR}/*.pkl"):
    det_files = [f for f in glob2.glob(DETECTIONS_DIR + "/*") if Path(f).is_file()]
    det_files = [f for f in det_files if "2018" in f ]
    dataset, s_name = f[:-4].split("/")[-1].split("_")
    s = STATIONS.by_dataset(dataset).by_name(s_name)
    if len(s) != 1:
        print(f"station {dataset}_{s_name} not found or not unique")
        continue
    FILES[s[0]] = f
FILES = {s : FILES[s] for s in FILES if (s.date_end > DATE_START and s.date_start < DATE_END)}



lat_min, lon_min = GRID_TO_COORDS.min(axis=0)
lat_max, lon_max = GRID_TO_COORDS.max(axis=0)
LAT_BOUNDS = (lat_min, lat_max)
LON_BOUNDS = (lon_min, lon_max)

FILES_2018 = {}
for key in FILES.keys() :
    if key.dataset == "OHASISBIO-2018" or key.dataset == "IMS-2018":
        FILES_2018[key] = FILES[key]
FILES = FILES_2018

# load properly the detection files


Path(f"{ASSOCIATIONS_DIR}/cache").mkdir(parents=True, exist_ok=True)
DET_PATH = f"{ASSOCIATIONS_DIR}/cache/detections_{MIN_P_TISSNET_SECONDARY}_{MERGE_DELTA_S}.pkl"
if True :#not Path(DET_PATH).exists():
    DETECTIONS = load_detections(list(FILES.values()), STATIONS, MIN_P_TISSNET_SECONDARY, merge_delta=datetime.timedelta(seconds=MERGE_DELTA_S))
    for s in DETECTIONS.keys():
        DETECTIONS[s] = DETECTIONS[s][(DETECTIONS[s][:,0] > DATE_START) & (DETECTIONS[s][:,0] < DATE_END)]

    with open(DET_PATH, "wb") as f:
        pickle.dump((DETECTIONS), f)
else:
    with open(DET_PATH, "rb") as f:
        DETECTIONS = pickle.load(f)

# do not keep detection entries for which the detection list is empty
to_del = []
for s in DETECTIONS.keys():
    if len(DETECTIONS[s]) == 0:
        to_del.append(s)
    # if s.name == 'H04S1':# or s.name == 'H04N1' or s.name == 'H01W1':
    #     to_del.append(s)
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



drift_ppm = {'ELAN': np.float64(-0.131710549912667),
             'H01W1': np.float64(0.08420817976058004),
             'H04N1': np.float64(0.005474507338212975),
             'H04S1': np.float64(0.04367580175236678),
             'H08S1': np.float64(0.0339800345578377),
             'MADE': np.float64(0.06750201481222673),
             'MADW': np.float64(0.13357977320875622),
             'NEAMS': np.float64(0.12430695166652732),
             'RTJ': np.float64(0.060539097101553196),
             'SSEIR': np.float64(-0.036162831789924375),
             'SSWIR': np.float64(0.02191309789424821),
             'SWAMS-bot': np.float64(0.0),
             'WKER2': np.float64(0.07208521358944064)}

offset = {'ELAN': np.float64(-1.2),
          'H01W1': np.float64(0),
          'H04N1': np.float64(6.5),
          'H08S1': np.float64(0.0),
          'MADE': np.float64(0.0),
          'MADW': np.float64(0.0),
          'NEAMS': np.float64(0.0),
          'RTJ': np.float64(0.0),
          'SSEIR': np.float64(0.0),
          'SSWIR': np.float64(0.0),
          'SWAMS-bot': np.float64(0.0),
          'WKER2': np.float64(0.0),
          'H04S1': np.float64(6.5)}



try :
    del DETECTIONS
except : pass
files = glob2.glob(f"{DETECTIONS_DIR}/cache/associations.pkl")
with open(files[0], "rb") as f:
    associations = pickle.load(f)
    print(len(associations))
# associations = associations[:5000]
# #
def process(i):
    station = list(map(lambda j: STATIONS[j].get_pos(), associations[i][0][:,0]))

    if len(station) < 8:
        return None

    det = list(map(lambda j: IDX_TO_DET[j][0], associations[i][0][:,1]))
    # drift_errors = list(map(lambda j,k : timedelta(seconds=STATIONS[j].get_clock_error(k, drift_ppm=drift_mesured_10[STATIONS[j].name]))+timedelta(seconds=6.8) if STATIONS[j].name =="H04N1" else timedelta(seconds=STATIONS[j].get_clock_error(k, drift_ppm=drift_mesured_10[STATIONS[j].name])) ,associations[i][0][:,0], det))
    drift_errors = list(map(lambda j,k :timedelta(seconds=offset[STATIONS[j].name]) + timedelta(seconds=STATIONS[j].get_clock_error(k, drift_ppm=drift_ppm[STATIONS[j].name])),associations[i][0][:,0], det ))    #correction des drifts
    det = list(map(lambda d, e : d-e,det,drift_errors))

    # drift = list(map(
    #     lambda s, d: 0.1 + STATIONS[s].get_clock_error(IDX_TO_DET[d][0],drift_ppm=0.28) if "not_ok" in STATIONS[s].other_kwargs.values()
    #                  else 0.1,
    #
    #     associations[i][0][:,0],
    #     associations[i][0][:,1]
    # ))
    drift = list(map(
        lambda s, d: 0.1 + STATIONS[s].get_clock_error(IDX_TO_DET[d][0],drift_ppm=0.28) if "not_ok"  in STATIONS[s].other_kwargs.values() #or ok
                     else 0.1,
        associations[i][0][:,0],
        associations[i][0][:,1]
    ))
    # drift = list(map(
    # lambda s, d: 0.25*1e-6*d if "not_ok" in STATIONS[s].other_kwargs.values()
    #              else 0,
    # associations[i][0][:,0],
    # associations[i][0][:,1]
    # ))
    drift =  np.abs(drift) / np.sqrt(3)
    # detections_uncertanty = [2]*len(det)
    detections_uncertanty = list(map(
        lambda s :  1 if "not_ok"  in STATIONS[s].other_kwargs.values()
                    else 0.5 if "ok"  in STATIONS[s].other_kwargs.values()
                    else 3, associations[i][0][:,0]
    ))

    c0 = list(map(lambda j: GRID_TO_COORDS[j], associations[i][1]))
    min_date = np.argmin(det)
    max_date = np.argmax(det)
    t0 = -1 * SOUND_MODEL.get_sound_travel_time(np.mean(c0, axis =0), station[min_date], det[min_date])
    x0 = [t0]+list(np.mean(c0, axis =0))
    # dmax= SOUND_MODEL.get_distance(np.mean(c0, axis =0), station[max_date])
    # detections_uncertanty = list(map(lambda x : 0.53+SOUND_MODEL.get_distance(np.mean(c0, axis =0), x)/dmax, station))

    res= SOUND_MODEL.localize_with_uncertainties(
        station, det,y_min=lon_min-6, x_min=lat_min-6,y_max=lon_max+6,x_max=lat_max+6, drift_uncertainties=drift,pick_uncertainties=detections_uncertanty, initial_pos=x0
    )

    return i, res


# Taille du chunk (param important à ajuster selon ton CPU/RAM)
CHUNK_SIZE = 50
results = {}
with mp.Pool(mp.cpu_count()-5) as pool, open("results.npy", "wb") as f:
    for r in tqdm(pool.imap(process, range(len(associations)), chunksize=CHUNK_SIZE),
                  total=len(associations)):
        if r is not None:
            i, res = r
            results[i] = res


with open(f"../../data/localisation/results_iter2.pkl", "wb") as f:
    pickle.dump(results, f)

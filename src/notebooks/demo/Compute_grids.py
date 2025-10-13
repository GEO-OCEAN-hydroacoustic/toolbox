from utils.physics.sound_model.spherical_sound_model import GridEllipsoidalSoundModel
from utils.data_reading.sound_data.station import StationsCatalog
import utils.physics.sound_model.ISAS_grid as isg
import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


PATH = "/media/rsafran/CORSAIR/ISAS/86442/field/2020"
DETECTIONS_DIR = "/media/rsafran/CORSAIR/detections_CTBT/"
lat_bounds = [-60, 5]
lon_bounds = [35, 120]
LAT_BOUNDS = [-60, 5]
LON_BOUNDS = [35, 120]
grid_size = 400
# Define start and end points
lat1, lon1 = -31.5758,83.2423    # Example: Station MADE
lat2, lon2 = -59.99254334995582,35.00354027003104  # Example: Station NEAMS
depth = 1200    # Depth in meters

method = 'min'
year = '2018'
PATH = f"/media/rsafran/CORSAIR/ISAS/86442/field/{year}"
out_dir = f"/media/rsafran/CORSAIR/ISAS/extracted/{year}/"
os.makedirs(out_dir, exist_ok=True)
result = isg.load_ISAS_extracted(out_dir, 4)
CATALOG_PATH = "/media/rsafran/CORSAIR/OHASISBIO/recensement_stations_OHASISBIO_RS.csv"
STATIONS = StationsCatalog(CATALOG_PATH).filter_out_undated().filter_out_unlocated()
ISAS_PATH = "/media/rsafran/CORSAIR/ISAS/extracted/2018"
arr = os.listdir(ISAS_PATH)
file_list = [os.path.join(ISAS_PATH, fname) for fname in arr if fname.endswith('.nc')]
SOUND_MODEL = GridEllipsoidalSoundModel(file_list)
STATIONS = StationsCatalog(CATALOG_PATH).filter_out_undated().filter_out_unlocated()
DETECTIONS_DIR_NAME = DETECTIONS_DIR.split("/")[-1]

if False:
    det_files = [f for f in glob2.glob(DETECTIONS_DIR + "/*") if Path(f).is_file()]
    det_files = [f for f in det_files if "2018" in f ]
    DETECTIONS, DETECTIONS_MERGED = load_detections(det_files, STATIONS, DETECTIONS_DIR, MIN_P_TISSNET_PRIMARY, MIN_P_TISSNET_SECONDARY, MERGE_DELTA)
else:
    DETECTIONS = np.load(f"{DETECTIONS_DIR}/cache/detections.npy", allow_pickle=True).item()
    # DETECTIONS_MERGED = np.load(f"{DETECTIONS_DIR}/cache/detections_merged.npy", allow_pickle=True)
    DETECTIONS_MERGED = np.load(f"{DETECTIONS_DIR}/cache/refined_detections_merged.npy", allow_pickle=True)

STATIONS = [s for s in DETECTIONS.keys()]
FIRSTS_DETECTIONS = {s : DETECTIONS[s][0,0] for s in STATIONS}
LASTS_DETECTIONS = {s : DETECTIONS[s][-1,0] for s in STATIONS}

import numpy as np
import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

def compute_latitude_line(lat_idx, lat, pts_lon, stations, sound_model, date):
    """
    Calcule tous les temps de trajet pour une ligne de latitude donnée
    Cette fonction sera exécutée en parallèle par chaque processus
    """
    line_results = {}

    # Pré-calcul des positions des stations pour éviter les appels répétés
    station_positions = {s: s.get_pos() for s in stations}

    for s in stations:
        station_pos = station_positions[s]
        line_times = np.zeros(len(pts_lon[lat_idx]))

        # Calcul vectorisé pour tous les points de longitude de cette latitude
        for lon_idx, lon in enumerate(pts_lon[lat_idx]):
            line_times[lon_idx] = sound_model.get_sound_travel_time(
                [lat, lon], station_pos, date=date
            )

        line_results[s] = line_times

    return lat_idx, line_results

def compute_grids_by_latitude(lat_bounds, lon_bounds, grid_size, sound_model, stations,
                             pick_uncertainty=5, sound_speed_uncertainty=2, n_workers=None, irregular_points=None):
    """
    Version optimisée qui traite ligne de latitude par ligne de latitude
    avec multiprocessing pour une performance maximale
    """
    if irregular_points is None:
        pts_lat = np.linspace(lat_bounds[0], lat_bounds[1], grid_size)
        pts_lon = np.linspace(lon_bounds[0], lon_bounds[1], grid_size)
    else :
        pts_lat = np.linspace(lat_bounds[0], lat_bounds[1], grid_size)
        pts_lon = irregular_points

    grid_max_res_time = (0.5 * np.sqrt(2) * (pts_lat[1] - pts_lat[0]) * 111_000) / (
                sound_model.constant_velocity - sound_speed_uncertainty)
    grid_tolerance = grid_max_res_time + pick_uncertainty
    print(f"Grid tolerance of {grid_tolerance:.2f}s")

    if n_workers is None:
        n_workers = min(len(pts_lat), mp.cpu_count())

    print(f"Processing {len(pts_lat)} latitude lines using {n_workers} workers")

    # Date fixe pour tous les calculs
    calc_date = datetime.datetime(year=2020, month=1, day=1)

    # Initialisation des structures de données
    grid_station_travel_time = {s: np.zeros((len(pts_lat), len(pts_lon))) for s in stations}

    # Préparation de la fonction pour multiprocessing
    compute_line_func = partial(
        compute_latitude_line,
        pts_lon=pts_lon,
        stations=stations,
        sound_model=sound_model,
        date=calc_date
    )

    # Traitement parallèle ligne par ligne
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Soumettre toutes les tâches
        futures = []
        for lat_idx, lat in enumerate(pts_lat):
            future = executor.submit(compute_line_func, lat_idx, lat)
            futures.append(future)

        # Récupération des résultats avec barre de progression
        completed = 0
        for future in futures:
            lat_idx, line_results = future.result()

            # Assemblage des résultats dans la grille finale
            for station in stations:
                grid_station_travel_time[station][lat_idx, :len(pts_lon[lat_idx])] = line_results[station]

            completed += 1
            if completed % max(1, len(pts_lat) // 10) == 0:
                print(f"Completed {completed}/{len(pts_lat)} latitude lines ({100*completed/len(pts_lat):.1f}%)")

    print("Computing station couple travel times...")
    # Calcul vectorisé des différences de temps de trajet
    grid_station_couple_travel_time = {}
    for s in stations:
        grid_station_couple_travel_time[s] = {}
        for s2 in stations:
            grid_station_couple_travel_time[s][s2] = (
                grid_station_travel_time[s2] - grid_station_travel_time[s]
            )

    print("Computing station max travel times...")
    # Calcul des temps de trajet maximum entre stations
    station_max_travel_time = {}
    for s in stations:
        station_max_travel_time[s] = {}
        for s2 in stations:
            station_max_travel_time[s][s2] = sound_model.get_sound_travel_time(
                s.get_pos(), s2.get_pos(), date=calc_date
            )

    return (pts_lat, pts_lon, station_max_travel_time, grid_station_travel_time,
            grid_station_couple_travel_time, grid_tolerance)


def compute_grids_chunked_latitude(lat_bounds, lon_bounds, grid_size, sound_model, stations,
                                  pick_uncertainty=5, sound_speed_uncertainty=2,
                                  n_workers=None, chunk_size=None):
    """
    Version avec chunking adaptatif pour optimiser l'équilibrage de charge
    Traite plusieurs lignes de latitude par chunk
    """
    pts_lat = np.linspace(lat_bounds[0], lat_bounds[1], grid_size)
    pts_lon = np.linspace(lon_bounds[0], lon_bounds[1], grid_size)

    grid_max_res_time = (0.5 * np.sqrt(2) * (pts_lat[1] - pts_lat[0]) * 111_000) / (
                sound_model.constant_velocity - sound_speed_uncertainty)
    grid_tolerance = grid_max_res_time + pick_uncertainty
    print(f"Grid tolerance of {grid_tolerance:.2f}s")

    if n_workers is None:
        n_workers = mp.cpu_count()

    if chunk_size is None:
        # Chunk size adaptatif basé sur le nombre de workers
        chunk_size =  max(1, grid_size // (n_workers // 2))

    print(f"Processing {len(pts_lat)} latitude lines in chunks of {chunk_size} using {n_workers} workers")

    calc_date = datetime.datetime(year=2020, month=1, day=1)

    def compute_latitude_chunk(lat_indices_chunk):
        """Traite un chunk de lignes de latitude"""
        chunk_results = {}
        station_positions = {s: s.get_pos() for s in stations}

        for lat_idx in lat_indices_chunk:
            lat = pts_lat[lat_idx]
            chunk_results[lat_idx] = {}

            for s in stations:
                station_pos = station_positions[s]
                line_times = np.zeros(len(pts_lon))

                for lon_idx, lon in enumerate(pts_lon):
                    line_times[lon_idx] = sound_model.get_sound_travel_time(
                        [lat, lon], station_pos, date=calc_date
                    )

                chunk_results[lat_idx][s] = line_times

        return chunk_results

    # Création des chunks
    lat_indices = list(range(len(pts_lat)))
    chunks = [lat_indices[i:i + chunk_size] for i in range(0, len(lat_indices), chunk_size)]

    # Initialisation des structures de données
    grid_station_travel_time = {s: np.zeros((len(pts_lat), len(pts_lon))) for s in stations}

    # Traitement parallèle par chunks
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        chunk_futures = [executor.submit(compute_latitude_chunk, chunk) for chunk in chunks]

        completed_chunks = 0
        for future in chunk_futures:
            chunk_results = future.result()

            # Assemblage des résultats
            for lat_idx, lat_results in chunk_results.items():
                for station, line_times in lat_results.items():
                    grid_station_travel_time[station][lat_idx, :] = line_times

            completed_chunks += 1
            completed_lines = completed_chunks * chunk_size
            print(f"Completed ~{min(completed_lines, len(pts_lat))}/{len(pts_lat)} latitude lines "
                  f"({100*min(completed_lines, len(pts_lat))/len(pts_lat):.1f}%)")

    # Calcul des différences et temps max (identique à la version précédente)
    print("Computing station couple travel times...")
    grid_station_couple_travel_time = {}
    for s in stations:
        grid_station_couple_travel_time[s] = {}
        for s2 in stations:
            grid_station_couple_travel_time[s][s2] = (
                grid_station_travel_time[s2] - grid_station_travel_time[s]
            )

    print("Computing station max travel times...")
    station_max_travel_time = {}
    for s in stations:
        station_max_travel_time[s] = {}
        for s2 in stations:
            station_max_travel_time[s][s2] = sound_model.get_sound_travel_time(
                s.get_pos(), s2.get_pos(), date=calc_date
            )

    return (pts_lat, pts_lon, station_max_travel_time, grid_station_travel_time,
            grid_station_couple_travel_time, grid_tolerance)


# Version avec monitoring de performance
def compute_grids_monitored(lat_bounds, lon_bounds, grid_size, sound_model, stations,
                           pick_uncertainty=5, sound_speed_uncertainty=2, n_workers=None,irregular_points=None):
    """
    Version avec monitoring détaillé des performances
    """
    import time

    start_time = time.time()

    result = compute_grids_by_latitude(
        lat_bounds, lon_bounds, grid_size, sound_model, stations,
        pick_uncertainty, sound_speed_uncertainty, n_workers, irregular_points=irregular_points
    )

    total_time = time.time() - start_time
    total_calculations = grid_size * grid_size * len(stations)

    print(f"\n=== Performance Report ===")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Total calculations: {total_calculations:,}")
    print(f"Calculations per second: {total_calculations/total_time:,.0f}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Number of stations: {len(stations)}")
    print(f"Workers used: {n_workers or mp.cpu_count()}")

    return result

def load_ridge_data(dorsal_db_path):
    """
    Charge les données des dorsales océaniques
    """
    dorsal_files = [f for f in os.listdir(dorsal_db_path) if f.endswith('.xy')]
    print(f"Loading {len(dorsal_files)} ridge files: {dorsal_files}")

    ridge_data = {}
    all_ridge_points = []

    for f in dorsal_files:
        ridge_name = f.replace('axe-', '').replace('-tout.xy', '')
        df = pd.read_csv(os.path.join(dorsal_db_path, f),
                        comment=">", sep=r'\s+', names=["lon", "lat", "n"])

        # Nettoyage des données
        # df = df.dropna()
        ridge_points = df[['lat', 'lon']].values

        ridge_data[ridge_name] = ridge_points
        all_ridge_points.append(ridge_points)

        print(f"  {ridge_name}: {len(ridge_points)} points")

    # Combinaison de toutes les dorsales
    all_ridge_points = np.vstack(all_ridge_points)
    print(f"Total ridge points: {len(all_ridge_points)}")

    return ridge_data, all_ridge_points

def create_regular_grid(lat_bounds, lon_bounds, grid_size):
    """
    Crée une grille régulière de base
    """
    lats = np.linspace(lat_bounds[0], lat_bounds[1], grid_size)
    lons = np.linspace(lon_bounds[0], lon_bounds[1], grid_size)

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    grid_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    return grid_points, lats, lons

def filter_points_near_ridges_chunk(grid_chunk, ridge_points, max_distance_deg):
    """
    Filtre un chunk de points de grille selon la distance aux dorsales
    """
    if len(grid_chunk) == 0:
        return []

    # Calcul des distances (approximation euclidienne rapide)
    distances = cdist(grid_chunk, ridge_points, metric='euclidean')
    min_distances = np.min(distances, axis=1)

    # Points à conserver (distance < seuil)
    valid_mask = min_distances <= max_distance_deg
    valid_points = grid_chunk[valid_mask]

    return valid_points.tolist()

def create_ridge_based_grid(lat_bounds, lon_bounds, grid_size, ridge_points,
                           max_distance_deg=4.0, n_workers=None):
    """
    Crée une grille irrégulière basée sur la proximité des dorsales océaniques

    Args:
        lat_bounds: [lat_min, lat_max]
        lon_bounds: [lon_min, lon_max]
        grid_size: taille de la grille régulière de base
        ridge_points: array numpy des points de dorsales [lat, lon]
        max_distance_deg: distance maximum en degrés des dorsales
        n_workers: nombre de workers pour le parallélisme

    Returns:
        irregular_points: array numpy des points sélectionnés
    """
    print(f"Creating ridge-based grid:")
    print(f"  Bounds: lat {lat_bounds}, lon {lon_bounds}")
    print(f"  Grid size: {grid_size}x{grid_size}")
    print(f"  Max distance from ridges: {max_distance_deg}°")

    # Création de la grille régulière de base
    grid_points, _, _ = create_regular_grid(lat_bounds, lon_bounds, grid_size)
    print(f"  Initial regular grid: {len(grid_points)} points")

    if n_workers is None:
        n_workers = mp.cpu_count()

    # Chunking pour le parallélisme
    chunk_size = max(1000, len(grid_points) // (n_workers * 4))
    chunks = [grid_points[i:i + chunk_size] for i in range(0, len(grid_points), chunk_size)]

    print(f"  Processing {len(chunks)} chunks using {n_workers} workers...")

    # Filtrage parallèle
    filter_func = partial(filter_points_near_ridges_chunk,
                         ridge_points=ridge_points,
                         max_distance_deg=max_distance_deg)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        chunk_results = list(executor.map(filter_func, chunks))

    # Combinaison des résultats
    irregular_points = []
    for chunk_result in chunk_results:
        irregular_points.extend(chunk_result)

    irregular_points = np.array(irregular_points)

    reduction_factor = len(irregular_points) / len(grid_points) * 100
    print(f"  Final irregular grid: {len(irregular_points)} points ({reduction_factor:.1f}% of original)")

    return irregular_points

def grid_by_latitude(lat_bounds, lon_bounds, grid_size,
                              irregular_points=None,
                              fill_missing='regular',
                              regular_lon_count=None,
                              tol=None):

    # 1) création des latitudes linéaires demandées (uniques)
    lats = np.linspace(lat_bounds[0], lat_bounds[1], grid_size)

    # 2) réglage des paramètres
    if regular_lon_count is None:
        regular_lon_count = grid_size

    if grid_size > 1:
        lat_spacing = lats[1] - lats[0]
    else:
        lat_spacing = 0.0
    if tol is None:
        tol = lat_spacing / 2.0 + 1e-12

    # cas sans irregular_points : retourner grille régulière complète
    if irregular_points is None or len(irregular_points) == 0:
        regular_lons = np.linspace(lon_bounds[0], lon_bounds[1], regular_lon_count)
        lons_per_lat = [regular_lons.copy() for _ in range(len(lats))]
        lat_col = np.repeat(lats, [len(regular_lons)] * len(lats))
        lon_col = np.tile(regular_lons, len(lats))
        grid_points_flat = np.column_stack([lat_col, lon_col])
        return lats, lons_per_lat, grid_points_flat

    # 3) associer chaque point irrégulier à la latitude la plus proche si dans tol
    irr_lats = irregular_points[:, 0]
    irr_lons = irregular_points[:, 1]

    # distances entre chaque point irrégulier et chaque lat cible (abs difference)
    # opération vectorisée efficace
    abs_diff = np.abs(irr_lats[:, None] - lats[None, :])   # shape (N, grid_size)
    idx_nearest = np.argmin(abs_diff, axis=1)              # index de la lat la plus proche
    dist_nearest = abs_diff[np.arange(len(irr_lats)), idx_nearest]
    assigned_mask = dist_nearest <= tol

    # préparer liste vide
    lons_per_lat = [np.array([], dtype=float) for _ in range(len(lats))]

    # remplir
    for i_pt, assigned in enumerate(assigned_mask):
        if not assigned:
            continue
        lat_idx = idx_nearest[i_pt]
        lons_per_lat[lat_idx] = np.append(lons_per_lat[lat_idx], irr_lons[i_pt])

    # unique & tri
    for i in range(len(lons_per_lat)):
        if lons_per_lat[i].size > 0:
            lons_per_lat[i] = np.unique(lons_per_lat[i])
        else:
            lons_per_lat[i] = np.array([], dtype=float)

    # 4) remplir les lat vides si demandé
    if fill_missing == 'regular':
        regular_lons = np.linspace(lon_bounds[0], lon_bounds[1], regular_lon_count)
        for i in range(len(lons_per_lat)):
            if lons_per_lat[i].size == 0:
                lons_per_lat[i] = regular_lons.copy()

    elif fill_missing == 'nearest':
        # trouver indices non vides
        non_empty_idxs = [i for i, arr in enumerate(lons_per_lat) if arr.size > 0]
        if len(non_empty_idxs) > 0:
            # pour chaque vide, copier les longitudes du plus proche non-vide
            for i in range(len(lons_per_lat)):
                if lons_per_lat[i].size == 0:
                    # distance en index (proxi en lat)
                    nearest = min(non_empty_idxs, key=lambda j: abs(j - i))
                    lons_per_lat[i] = lons_per_lat[nearest].copy()
        else:
            # aucun point existant : si aucun non-empty, on peut remplir par régulier si souhaité
            if fill_missing == 'nearest':
                regular_lons = np.linspace(lon_bounds[0], lon_bounds[1], regular_lon_count)
                lons_per_lat = [regular_lons.copy() for _ in range(len(lats))]

    # else fill_missing == 'none' -> laisser vides

    # 5) construire grid_points_flat
    lat_list = []
    lon_list = []
    for i_lat, lon_arr in enumerate(lons_per_lat):
        if lon_arr.size == 0:
            continue
        lat_list.append(np.full(lon_arr.shape, lats[i_lat]))
        lon_list.append(lon_arr)
    if len(lat_list) == 0:
        grid_points_flat = np.zeros((0, 2))
    else:
        lat_col = np.concatenate(lat_list)
        lon_col = np.concatenate(lon_list)
        grid_points_flat = np.column_stack([lat_col, lon_col])

    return lats, lons_per_lat, grid_points_flat



def visualize_ridge_grid(ridge_data, irregular_points, lat_bounds, lon_bounds,
                        max_distance_deg=4.0, figsize=(12, 8)):
    """
    Visualise la grille irrégulière par rapport aux dorsales
    """
    plt.figure(figsize=figsize)

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Plot des dorsales
    for i, (ridge_name, ridge_points) in enumerate(ridge_data.items()):
        plt.scatter(ridge_points[:, 1], ridge_points[:, 0],
                   c=colors[i % len(colors)], s=1, alpha=0.7,
                   label=f'{ridge_name} ridge')

    # Plot de la grille irrégulière
    plt.scatter(irregular_points[:, 1], irregular_points[:, 0],
               c='black', s=0.5, alpha=0.3, label='Grid points')

    plt.xlim(lon_bounds)
    plt.ylim(lat_bounds)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Ridge-based irregular grid (≤{max_distance_deg}° from ridges)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()

# Exemple d'utilisation
def example_usage():
    """
    Exemple complet d'utilisation
    """
    # Paramètres
    dorsal_db = "/home/rsafran/Documents/Database_geo/"
    lat_bounds = [-60, 5]
    lon_bounds = [35, 120]
    grid_size = 400  # Grille régulière de base
    max_distance_deg = 4.0

    # Chargement des dorsales
    ridge_data, all_ridge_points = load_ridge_data(dorsal_db)

    # Génération de la grille irrégulière
    irregular_points = create_ridge_based_grid(
        lat_bounds, lon_bounds, grid_size, all_ridge_points, max_distance_deg, n_workers=1
    )


    # # Visualisation (optionnel)
    fig = visualize_ridge_grid(ridge_data, irregular_points, lat_bounds, lon_bounds, max_distance_deg)
    plt.show()

    lats, lons_per_lat, grid_flat = grid_by_latitude(
        lat_bounds, lon_bounds, grid_size,
        irregular_points=irregular_points,
        fill_missing='nearest',  # ou 'regular' / 'none'
        regular_lon_count=200  # optionnel si 'regular'
    )

    # mp.set_start_method("spawn", force=True)
    # Calcul complet avec votre sound_model et stations
    result =  compute_grids_monitored(lat_bounds, lon_bounds, grid_size, SOUND_MODEL, STATIONS,
                            pick_uncertainty=2, sound_speed_uncertainty=1, n_workers=None,
                            irregular_points=lons_per_lat)

    return irregular_points, ridge_data,result



if __name__ == "__main__":

    irregular_points, ridge_data,result = example_usage()
    pts_lat, pts_lon, station_max_travel_time, grid_station_travel_time, grid_station_couple_travel_time, grid_tolerance = result


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
        line_times = np.zeros(len(pts_lon))

        # Calcul vectorisé pour tous les points de longitude de cette latitude
        for lon_idx, lon in enumerate(pts_lon):
            line_times[lon_idx] = sound_model.get_sound_travel_time(
                [lat, lon], station_pos, date=date
            )

        line_results[s] = line_times

    return lat_idx, line_results

def compute_grids_by_latitude(lat_bounds, lon_bounds, grid_size, sound_model, stations,
                             pick_uncertainty=5, sound_speed_uncertainty=2, n_workers=None):
    """
    Version optimisée qui traite ligne de latitude par ligne de latitude
    avec multiprocessing pour une performance maximale
    """
    pts_lat = np.linspace(lat_bounds[0], lat_bounds[1], grid_size)
    pts_lon = np.linspace(lon_bounds[0], lon_bounds[1], grid_size)

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
                grid_station_travel_time[station][lat_idx, :] = line_results[station]

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
                           pick_uncertainty=5, sound_speed_uncertainty=2, n_workers=None):
    """
    Version avec monitoring détaillé des performances
    """
    import time

    start_time = time.time()

    result = compute_grids_by_latitude(
        lat_bounds, lon_bounds, grid_size, sound_model, stations,
        pick_uncertainty, sound_speed_uncertainty, n_workers
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
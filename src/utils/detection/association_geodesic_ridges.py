import copy
import pickle
from multiprocessing import cpu_count, Pool
from pathlib import Path
from tqdm import tqdm
import geopy.distance
from scipy.interpolate import griddata

import numpy as np
import datetime
from matplotlib import pyplot as plt
import matplotlib as mpl


# mpl.rcParams.update({
#     "font.size": 20,
#     "axes.titlesize": 20,
#     "axes.labelsize": 20,
#     "xtick.labelsize": 20,
#     "ytick.labelsize": 20,
#     "legend.fontsize": 20,
#     "figure.titlesize": 20,
#     "font.family": "serif",
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42
# })

from utils.physics.geodesic.distance import distance_point_point


def squarize(coordinates, weights, lat_bounds, lon_bounds, size=1000):
    grid_lat = np.linspace(lat_bounds[0], lat_bounds[-1], size)
    grid_lon = np.linspace(lon_bounds[0], lon_bounds[-1], size)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

    # Interpolation des poids sur une grille régulière
    grid = griddata(
        coordinates, weights, (grid_lat2d, grid_lon2d),
        method='nearest'
    )

    return grid, grid_lat, grid_lon

# given a list of detections, gets the list of the ones that are closest enough to a given date
def find_detections(detection_list, target_date, tolerance_before, tolerance_after):
    res = []

    idx = np.searchsorted(detection_list, target_date + tolerance_before, side="left") - 1
    idx = max(idx, 0)
    while idx < len(detection_list) and detection_list[idx] < target_date + tolerance_after:
        if detection_list[idx] > target_date + tolerance_before:
            res.append(idx)
        idx += 1
    return res

to_tdelta = lambda t : datetime.timedelta(seconds=float(t))

# given two detections (station+date), get the positions on a grid that may be the source
def get_valid_grid(s1, s2, date1, date2, valid_grid, TDoA, TDoA_uncertainties):
    diff = (date2 - date1).total_seconds()
    grid = TDoA[s1][s2][date1.month-1]

    if valid_grid is not None:
        grid = np.where(valid_grid, grid, np.nan)

    differences = np.abs(diff - grid)
    valid_grid = (differences < TDoA_uncertainties[s1][s2][date1.month-1])

    return differences, valid_grid

# given an association and a new detection, update the grid of possible source locations
def update_valid_grid(current_association, current_valid_grid, new_station, new_date, TDoA, TDoA_uncertainties,
                      save_path=None, lon_bounds=None, lat_bounds=None, grid_to_coords=None, new_idx=None):
    difference_grids, valid_grids = [], []
    for si, (datei, idxi) in current_association.items():
        difference_grid_new, valid_grid_new = get_valid_grid(si, new_station, datei, new_date, current_valid_grid,
                                                             TDoA, TDoA_uncertainties)
        valid_grids.append(valid_grid_new)
        difference_grids.append(difference_grid_new)
        current_valid_grid = valid_grids[-1] if save_path is None else current_valid_grid
    grid = np.all(valid_grids, axis=0) if save_path is not None else current_valid_grid

    if save_path is not None and np.count_nonzero(~np.isnan(difference_grid := np.max(difference_grids, axis=0))) > 0:
        asso = list(current_association.items())
        res = f'{asso[0][0].name}-{asso[0][1][1]}_{len(asso):02d}'
        res = res + '_'.join(['']+[f'{s.name}-{idx}' for s, (date, idx) in asso[1:]])
        save_path_grid = f'{save_path}/{res}_{new_station.name}-{new_idx}.png'

        fig, ax = plt.subplots(figsize=(10, 8))
        sq = squarize(grid_to_coords, difference_grid, lat_bounds, lon_bounds)
        im = ax.imshow(sq[::-1], cmap="inferno", vmin=0, extent=(lon_bounds[0], lon_bounds[-1], lat_bounds[0], lat_bounds[-1]))
        xticks = np.arange(np.floor(lon_bounds[0] / 5) * 5, np.ceil(lon_bounds[-1] / 5) * 5 + 1, 5)
        yticks = np.arange(np.floor(lat_bounds[0] / 5) * 5, np.ceil(lat_bounds[-1] / 5) * 5 + 1, 5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        for s in current_association.keys():
            p = s.get_pos()
            ax.plot(p[1], p[0], 'yx', alpha=0.75, markersize=10, markeredgewidth=3)
            ax.annotate(s.name, xy=(p[1], p[0]),
                        xytext=(p[1] - (lon_bounds[1] - lon_bounds[0]) / 30, p[0] + (lat_bounds[1] - lat_bounds[0]) / 50),
                        textcoords="data", color='y', alpha=0.9, weight='bold')

        s, p = new_station, new_station.get_pos()
        ax.plot(new_station.get_pos()[1], new_station.get_pos()[0], 'gx', alpha=0.75, markersize=10, markeredgewidth=3)
        ax.annotate(s.name, xy=(p[1], p[0]),
                    xytext=(p[1] - (lon_bounds[1] - lon_bounds[0]) / 30, p[0] + (lat_bounds[1] - lat_bounds[0]) / 50),
                    textcoords="data", color='g', alpha=1, weight='bold')

        cbar = plt.colorbar(im, fraction=0.0415, pad=0.04)
        cbar.set_label('|expected TDoA - observed TDoA| (s)', rotation=270, labelpad=20)
        ax.set_title(f"TDoA differences grid")
        ax.set_xlabel("lon (°)")
        ax.set_ylabel("lat (°)")
        plt.savefig(save_path_grid, dpi=500, bbox_inches='tight')
        plt.close()
    return grid, difference_grids

def compute_candidates(stations_to_update, current_association, detections, max_TDoA, generic_tolerance):
    candidates = {}
    date1 = next(iter(current_association.values()))[0]
    for s in stations_to_update:
        c = []
        for station_anchor, (date_anchor, idx_anchor) in current_association.items():
            t1 = -to_tdelta(max_TDoA[s][station_anchor][date1.month-1])
            t2 = to_tdelta(max_TDoA[station_anchor][s][date1.month-1])
            t1 -= to_tdelta(generic_tolerance)
            t2 += to_tdelta(generic_tolerance)

            c.append(set(find_detections(detections[s][:,0], date_anchor, t1, t2)))
        candidates[s] = list(set.intersection(*c))
    return candidates

# check if the association is valid (if it was never seen) and save it in results
def update_results(date1, current_association, valid_points, results,
                   TDoA, TDoA_uncertainties, compute_costs=False):
    res = np.array([[s.idx, idx] for s, (d, idx) in current_association.items()])

    if compute_costs:
        valid_points_2 = []
        for i in valid_points:
            diffs = []
            for s1, d1 in res:
                for s2, d2 in res:
                    if s1 != s2:
                        diffs.append(abs(TDoA[s1][s2][date1.month-1][i] - (d2 - d1).total_seconds()) -
                                     TDoA_uncertainties[s1][s2][date1.month-1][i])
            valid_points_2.append([i, np.max(diffs)])
        valid_points = valid_points_2

    results.append((res, np.array(valid_points)))

def load_detections(det_files, stations, min_p_tissnet_secondary=0.1, merge_delta=datetime.timedelta(seconds=5)):
    detections = {}
    for det_file in det_files:
        station_dataset, station_name = det_file[:-4].split("/")[-1].split("_")
        station = stations.by_dataset(station_dataset).by_name(station_name)
        if len(station) != 1:
            print(
                f"Not finding a single station with dataset {station_dataset} and name {station_name} (query result : {station}).\nSkipping.")
            continue
        station = station[0]

        d = []
        with open(det_file, "rb") as f:
            while True:
                try:
                    d.append(pickle.load(f))
                except EOFError:
                    break
        d = np.array(d)
        d = d[:, :2]
        d = d[d[:, 1] > min_p_tissnet_secondary]
        d = d[np.argsort(d[:, 0])]

        # remove duplicates and regularly spaced signals
        new_d = [d[0]]
        for i in range(1, len(d)):
            # check this event is far enough from the previous one
            if d[i, 0] - new_d[-1][0] > merge_delta:
                new_d.append(d[i])
        d = np.array(new_d)
        detections[station] = d
        print(f"Found {len(d)} detections for station {station}")

    return detections


def _compute_parallel(lat, lat_step, coords, stations, corner_coordinates,
                     sound_model, pick_uncertainty, sound_speed_uncertainty, max_clock_drift):
    print(f"processing parallel {lat:0.2f}°")

    travel_times_local = {s: [[] for _ in range(12)] for s in stations}
    grid_to_coords_local = []
    travel_time_uncertainties_local = {s: [[] for _ in range(12)] for s in stations}
    corner_differences_local = {s: [[] for _ in range(12)] for s in stations}

    lons = sorted(coords[:,1])
    lon_step = lons[1] - lons[0] if len(lons) > 1 else lat_step

    for lon in lons:
        grid_to_coords_local.append((lat, lon))

        for s in stations:
            d = distance_point_point([lat, lon], s.get_pos(), fast=False)

            corners = []
            for dlat, dlon in corner_coordinates:
                lat_corner, lon_corner = lat + dlat * lat_step / 2, lon + dlon * lon_step / 2
                corners.append(distance_point_point([lat_corner, lon_corner], s.get_pos(), fast=False))

            for m in range(12):
                start_date = datetime.datetime(1999, 12, 15)
                date = datetime.datetime(2000, m + 1, 15)
                
                sound_speed = sound_model.get_sound_speed([lat, lon], s.get_pos(), date)
                #sound_speed = sound_speeds[m]

                travel_times_local[s][m].append(d / sound_speed)

                # account for detection time uncertainty
                uncertainty_pick = pick_uncertainty
                uncertainty_celerity = (d / (sound_speed - sound_speed_uncertainty) -
                                        d / (sound_speed + sound_speed_uncertainty))
                max_clock_drift = 0.1+s.get_clock_error(date,ref_date=start_date, drift_ppm=0.28) if 'clock_drift_ppm' in s.other_kwargs else 0 #if "not_ok" in s.other_kwargs.values() else 0.1

                travel_time_uncertainties_local[s][m].append(
                    uncertainty_pick + uncertainty_celerity + max_clock_drift)

                corner_differences_local[s][m].append([])
                for d_corner in corners:
                    corner_differences_local[s][m][-1].append(d_corner / sound_speed -
                                                        travel_times_local[s][m][-1])

    return grid_to_coords_local, travel_times_local, travel_time_uncertainties_local, corner_differences_local

def compute_grids(grid_to_coords, sound_model, stations,
                  pick_uncertainty=5, sound_speed_uncertainty=2, max_clock_drift=1):
    # station -> month -> cell_index -> travel time from this cell to this station at this month
    # this is not a 2D array because if we are closer to poles, we have less cells (to keep squares)
    travel_times = {s: [[] for _ in range(12)] for s in stations}
    travel_time_uncertainties = {s: [[] for _ in range(12)] for s in stations}
    # given each grid cells are not points, the real source may be at +/- step_m/2 along both lat and lon
    # given two stations s1 and s2, a grid cell of center I and a source M, we thus look for an upper bound of
    # d = |(T(M->s1)-T(M->s2)) - T((I->s1)-T(I->s2))|. We make the hypothesis that it happens on a corner.
    # To compute it, we store T(c->s)-T(I->s) for each corner c of each cells, and for each station s.
    corner_differences = {s: [[] for _ in range(12)] for s in stations}
    corner_coordinates = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

    # we will make a grid of nearly-squares and compute the side we will use, in km
    # for this, we use latitudes (whose step should be constant)
    lats = sorted(np.unique(grid_to_coords[:,0]))
    lat_step = abs(lats[1] - lats[0])
    per_lat = {lat : grid_to_coords[grid_to_coords[:,0] == lat] for lat in lats}

    # compute each parallel in... Parallel
    # (note: pool.starmap guarantees the results to be in the same order as the arguments)
    with Pool(processes=max(1,cpu_count()//2)) as pool:
        results = pool.starmap(_compute_parallel, [(lat, lat_step, per_lat[lat], stations, corner_coordinates,
                sound_model, pick_uncertainty, sound_speed_uncertainty, max_clock_drift) for lat in lats])

    grid_to_coords_new_order = []
    for result in results:
        grid_to_coords_new_order.extend(result[0])
        for s in stations:
            for m in range(12):
                travel_times[s][m].extend(result[1][s][m])
                travel_time_uncertainties[s][m].extend(result[2][s][m])
                corner_differences[s][m].extend(result[3][s][m])



    travel_times = {s: [np.array(travel_times[s][m]) for m in range(12)] for s in stations}
    travel_time_uncertainties = {s: [np.array(travel_time_uncertainties[s][m]) for m in range(12)] for s in stations}
    corner_differences = {s: [np.array(corner_differences[s][m]) for m in range(12)] for s in stations}
    # station s1 -> station s2 -> month -> cell_index -> expected Time Difference of Arrival (TDoA) from this cell
    # between s1 and s2 at this month. Asymmetric: we do the travel time of s2 minus the travel time of s1
    TDoA = {s: {s2: [[] for _ in range(12)] for s2 in stations} for s in
                                       stations}
    TDoA_uncertainties = {s: {s2: [[] for _ in range(12)] for s2 in stations} for s in
                                       stations}
    max_TDoA = {s: {s2: [[] for _ in range(12)] for s2 in stations} for s in
                                       stations}
    for s in stations:
        for s2 in stations:
            for m in range(12):
                TDoA[s][s2][m] = travel_times[s2][m] - travel_times[s][m]

                max_corner_diff = np.max(np.abs(corner_differences[s][m] - corner_differences[s2][m]), axis=1)
                TDoA_uncertainties[s][s2][m] = (travel_time_uncertainties[s][m] + travel_time_uncertainties[s2][m]
                                                + max_corner_diff)

                mad_idx = np.nanargmax(TDoA[s][s2][m])
                max_TDoA[s][s2][m] = TDoA[s][s2][m][mad_idx] + TDoA_uncertainties[s][s2][m][mad_idx]

    print("Grid processing finished")
    return grid_to_coords_new_order, TDoA, max_TDoA, TDoA_uncertainties, travel_times, travel_time_uncertainties
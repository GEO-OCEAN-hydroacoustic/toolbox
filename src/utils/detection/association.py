import copy
import pickle
from pathlib import Path

import numpy as np
import datetime
from matplotlib import pyplot as plt

# given a list of detections, gets the list of the ones that are closest enough to a given date
def find_detections(detection_list, target_date, tolerance):
    res = []
    idx = np.searchsorted(detection_list, target_date - tolerance, side="left") - 1
    idx = max(idx, 0)
    while idx < len(detection_list) and detection_list[idx] < target_date + tolerance:
        if detection_list[idx] > target_date - tolerance:
            res.append(idx)
        idx += 1
    return res

to_tdelta = lambda t : datetime.timedelta(seconds=t)

# given two detections (station+date), get the positions on a grid that may be the source
def get_valid_grid(s1, s2, date1, date2, valid_grid, grid_station_couple_travel_time, grid_tolerance):
    diff = (date2 - date1).total_seconds()
    grid = grid_station_couple_travel_time[s1][s2]

    if valid_grid is not None:
        grid = np.where(valid_grid, grid, np.nan)

    differences = np.abs(diff - grid)
    valid_grid = (differences < grid_tolerance)

    return differences, valid_grid

# given an association and a new detection, update the grid of possible source locations
def update_valid_grid(current_association, current_valid_grid, new_station, new_date, grid_station_couple_travel_time,
                      grid_tolerance, save_path=None, lon_bounds=None, lat_bounds=None):
    difference_grids, valid_grids = [], []
    for si, datei in current_association.items():
        difference_grid_new, valid_grid_new = get_valid_grid(si, new_station, datei, new_date, current_valid_grid,
                                                             grid_station_couple_travel_time, grid_tolerance)
        valid_grids.append(valid_grid_new)
        difference_grids.append(difference_grid_new)
        current_valid_grid = valid_grids[-1] if save_path is None else current_valid_grid
    grid = np.all(valid_grids, axis=0) if save_path is not None else current_valid_grid

    if save_path is not None and np.count_nonzero(~np.isnan(difference_grid := np.max(difference_grids, axis=0))) > 0:
        res = '_'.join([f'{s.name}-{date.strftime("%Y%m%d_%H%M%S")}' for s, date in current_association.items()])
        save_path_grid = f'{save_path}/{res}_{new_station.name}-{new_date.strftime("%Y%m%d_%H%M%S")}.png'

        fig, ax = plt.subplots()

        extent = (lon_bounds[0], lon_bounds[-1], lat_bounds[0], lat_bounds[-1])
        im = ax.imshow(difference_grid[::-1], cmap="inferno", extent=extent, interpolation=None, vmin=0,
                       vmax=10*grid_tolerance)
        for s in current_association.keys():
            ax.plot(s.get_pos()[1], s.get_pos()[0], 'rx')
            ax.annotate(s.name, xy=(s.get_pos()[1], s.get_pos()[0]), textcoords="data", color='r')
        ax.plot(new_station.get_pos()[1], new_station.get_pos()[0], 'gx')
        ax.annotate(new_station.name, xy=(new_station.get_pos()[1], new_station.get_pos()[0]), textcoords="data",
                    color='r')

        fig.colorbar(im, orientation='vertical')
        plt.tight_layout()
        plt.savefig(save_path_grid)
        plt.close()
    return grid, difference_grids

# given an association and a new detection, update the grid of possible source locations
def update_valid_grid_gt(current_association, current_valid_grid, new_station, new_date, grid_station_couple_travel_time,
                      grid_tolerance, save_path=None, lon_bounds=None, lat_bounds=None, gt=None):
    difference_grids, valid_grids = [], []
    for si, datei in current_association.items():
        difference_grid_new, valid_grid_new = get_valid_grid(si, new_station, datei, new_date, current_valid_grid,
                                                             grid_station_couple_travel_time, grid_tolerance)
        valid_grids.append(valid_grid_new)
        difference_grids.append(difference_grid_new)
        current_valid_grid = valid_grids[-1] if save_path is None else current_valid_grid
    grid = np.all(valid_grids, axis=0) if save_path is not None else current_valid_grid

    if save_path is not None and np.count_nonzero(~np.isnan(difference_grid := np.max(difference_grids, axis=0))) > 0:
        res = '_'.join([f'{s.name}-{date.strftime("%Y%m%d_%H%M%S")}' for s, date in current_association.items()])
        save_path_grid = f'{save_path}_{res}_{new_station.name}-{new_date.strftime("%Y%m%d_%H%M%S")}.png'

        fig, ax = plt.subplots()

        extent = (lon_bounds[0], lon_bounds[-1], lat_bounds[0], lat_bounds[-1])
        im = ax.imshow(difference_grid[::-1], cmap="inferno", extent=extent, interpolation=None, vmin=0,
                       vmax=10*grid_tolerance)
        for s in current_association.keys():
            ax.plot(s.get_pos()[1], s.get_pos()[0], 'rx')
            ax.annotate(s.name, xy=(s.get_pos()[1], s.get_pos()[0]), textcoords="data", color='r')
        ax.plot(new_station.get_pos()[1], new_station.get_pos()[0], 'gx')
        ax.annotate(new_station.name, xy=(new_station.get_pos()[1], new_station.get_pos()[0]), textcoords="data",
                    color='r')
        ax.plot(gt[1], gt[0], 'o', color="bisque")
        fig.colorbar(im, orientation='vertical')
        plt.tight_layout()
        plt.savefig(save_path_grid)
        plt.close()
    return grid, difference_grids

def compute_candidates(stations_to_update, current_association, detections, station_max_travel_time, generic_tolerance):
    candidates = {}

    for s in stations_to_update:
        c = []
        for station_anchor, date_anchor in current_association.items():
            c.append(set(find_detections(detections[s][:,0], date_anchor, to_tdelta(station_max_travel_time[s][station_anchor]) + to_tdelta(generic_tolerance))))
        candidates[s] = list(set.intersection(*c))
    return candidates

# check if the association is valid (if it was never seen) and save it in results
def update_results(date1, current_association, valid_points, results,
                   grid_station_couple_travel_time, compute_costs=False):
    res = np.array([[s, d] for s, d in current_association.items()])
    res = res[np.argsort(res[:, 1])]

    if compute_costs:
        valid_points_2 = []
        for i, j in valid_points:
            diffs = []
            for s1, d1 in res:
                for s2, d2 in res:
                    if s1 != s2:
                        diffs.append(abs(grid_station_couple_travel_time[s1][s2][i,j] - (d2 - d1).total_seconds()))
            valid_points_2.append([i, j, np.max(diffs)])
        valid_points = valid_points_2

    results.setdefault(date1, []).append((res, np.array(valid_points)))

# check if the association was never seen
def association_is_new(association, new_date, association_hashlist):
    res = [d for _, d in association.items()] + [new_date]
    s = np.sum([int(d.timestamp() * 1_000) for d in res])
    if s in association_hashlist:
        return False
    association_hashlist.add(s)
    return True

def load_detections(det_files, stations, detections_dir, min_p_tissnet_primary=0.1, min_p_tissnet_secondary=0.1,
                    merge_delta=datetime.timedelta(seconds=5)):
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
            if d[i, 0] - d[i - 1, 0] > merge_delta:
                # check this event is not part of a series of regularly spaced events (which probably means we encounter seismic airgun shots)
                if i < 3 or abs((d[i, 0] - d[i - 1, 0]) - (d[i - 1, 0] - d[i - 2, 0])) > merge_delta and abs(
                        (d[i, 0] - d[i - 2, 0]) - (d[i - 1, 0] - d[i - 3, 0])) > merge_delta:
                    new_d.append(d[i])
        d = np.array(new_d)

        detections[station] = d

        print(f"Found {len(d)} detections for station {station}")

        # we keep all detections in a single list, sorted by date, to then browse detections
    stations = list(detections.keys())
    detections_merged = np.concatenate([[(det[0], det[1], s) for det in detections[s]] for s in stations])
    detections_merged = detections_merged[detections_merged[:, 1] > min_p_tissnet_primary]
    detections_merged = detections_merged[np.argsort(detections_merged[:, 0])]

    Path(f"{detections_dir}/cache/").mkdir(parents=True, exist_ok=True)
    np.save(f"{detections_dir}/cache/detections.npy", detections)
    np.save(f"{detections_dir}/cache/detections_merged.npy", detections_merged)

    return detections, detections_merged

def compute_grids(lat_bounds, lon_bounds, grid_size, sound_model, stations,
                  pick_uncertainty=5, sound_speed_uncertainty=2):
    pts_lat = np.linspace(lat_bounds[0], lat_bounds[1], grid_size)
    pts_lon = np.linspace(lon_bounds[0], lon_bounds[1], grid_size)

    grid_max_res_time = (0.5 * np.sqrt(2) * (pts_lat[1] - pts_lat[0]) * 111_000) / (
                sound_model.sound_speed - sound_speed_uncertainty)
    grid_tolerance = grid_max_res_time + pick_uncertainty
    print(f"Grid tolerance of {grid_tolerance:.2f}s")

    grid_station_travel_time = {s: np.zeros((len(pts_lat), len(pts_lon))) for s in stations}
    for s in stations:
        for ilat, lat in enumerate(pts_lat):
            for ilon, lon in enumerate(pts_lon):
                grid_station_travel_time[s][ilat, ilon] = sound_model.get_sound_travel_time([lat, lon], s.get_pos())

    grid_station_couple_travel_time = {s: {s2: np.zeros((len(pts_lat), len(pts_lon))) for s2 in stations} for s in
                                       stations}
    for s in stations:
        for s2 in stations:
            grid_station_couple_travel_time[s][s2] = grid_station_travel_time[s2] - grid_station_travel_time[s]

    station_max_travel_time = {s: {s2: sound_model.get_sound_travel_time(s.get_pos(), s2.get_pos()) for s2 in stations}
                               for s in stations}

    return (pts_lat, pts_lon, station_max_travel_time, grid_station_travel_time,
            grid_station_couple_travel_time, grid_tolerance)
import datetime
import itertools
from pathlib import Path
import glob2
import numpy as np
import pandas as pd
import yaml

from src.utils.data_reading.sound_data.sound_file_manager import make_manager

class Station:
    @property
    def depth(self):
        return getattr(self, '_depth', 0.0)
    
    @depth.setter
    def depth(self, value):
        self._depth = float(value) if value is not None else 0.0

    def __init__(self, path, name=None, lat=None, lon=None, depth=None, date_start=None, date_end=None, dataset=None,
                 initialize_metadata=False, other_kwargs=None):
        other_kwargs = other_kwargs or {}
        
        # Récupérer la profondeur depuis other_kwargs si elle n'est pas fournie directement
        if depth is None and 'depth' in other_kwargs:
            depth = other_kwargs['depth']
            del other_kwargs['depth']  # Supprimer la profondeur de other_kwargs après l'avoir utilisée
        
        self.depth = depth  # Ceci utilisera le setter de la propriété
        self.manager = None
        self.path = path
        assert isinstance(path, str)
        assert not name or isinstance(name, str)
        assert not date_start or isinstance(date_start, datetime.datetime)
        assert not date_end or isinstance(date_end, datetime.datetime)
        assert not lat or isinstance(lat, float) or isinstance(lat, int)
        assert not lon or isinstance(lon, float) or isinstance(lon, int)
        assert not dataset or isinstance(dataset, str)
        self.name = name
        self.date_start = date_start
        self.date_end = date_end
        self.lat = lat
        self.lon = lon
        self.dataset = dataset
        self.other_kwargs = {}
        for k, v in other_kwargs.items():
            if v != "" and not (isinstance(v, np.float64) and np.isnan(v)):
                self.other_kwargs[k] = v
        if initialize_metadata:
            self.get_manager()
            self.name = name or self.manager.name
            self.date_start = date_start or self.manager.dataset_start
            self.date_end = date_end or self.manager.dataset_end

    def get_manager(self):
        self.load_data()
        return self.manager

    def load_data(self):
        if self.path and not self.manager:
            self.manager = make_manager(self.path, self.other_kwargs)

    def get_pos(self, include_depth=False):
        if not include_depth:
            return [self.lat, self.lon]
        return [self.lat, self.lon, self.depth]

    def light_copy(self):
        """ Make a copy of this station, only including metadata.
        :return: A copy of self containing the name, lat, lon, dates, path and dataset of the station.
        """
        return Station(self.path, self.name, self.lat, self.lon, self.depth, self.date_start, self.date_end, self.dataset,
                       other_kwargs=self.other_kwargs)

    def get_clock_error(self, date=None):
        """
        Calculate the accumulated clock error after a given time
        """
        time_elapsed_seconds = (date -  self.date_start).total_seconds()

        return self.other_kwargs["clock_drift_ppm"] * 1e-6 * time_elapsed_seconds

    def __str__(self):
        return f"{self.dataset}_{self.name}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (self.name == other.name and self.lat == other.lat and self.lon == other.lon and
                self.date_start == other.date_start and self.date_end == other.date_end)

    def __hash__(self):
        return hash((self.name, self.lat, self.lon, self.date_start, self.date_end))

class StationsCatalog():
    def __init__(self, file=None, min_depth=0, max_depth=3):
        self.stations = []

        # if a file was given, and it is a .yaml or .csv, read it
        # otherwise, if it is a directory, recursively try to find .yaml or .csv files
        if file:
            if ".yaml" in file:
                self.load_yaml(file)
            elif ".csv" in file:
                self.load_csv(file)
            if Path(file).is_dir():
                if min_depth < max_depth:
                    catalogs = []
                    for path in glob2.glob(f"{file}/*"):
                        catalogs.append(StationsCatalog(path, min_depth+1, max_depth))
                    self.stations = list(itertools.chain.from_iterable(catalogs))

    def load_yaml(self, yaml_file):
        with open(yaml_file, "r") as f:
            data = yaml.load(f, Loader=yaml.BaseLoader)

        for dataset, dataset_yaml in data.items():
            path = dataset_yaml["root_dir"]
            if path == "":
                continue
            if path[0] != "/":
                # relative path
                path = "/".join(yaml_file.split("/")[:-1]) + "/" + path
            for station_name, station_yaml in dataset_yaml["stations"].items():
                date_start, date_end, lat, lon = None, None, None, None
                if station_yaml["date_start"].strip() != "":
                    date_start = datetime.datetime.strptime(station_yaml["date_start"], "%Y%m%d_%H%M%S")
                    del station_yaml["date_start"]
                if station_yaml["date_end"].strip() != "":
                    date_end = datetime.datetime.strptime(station_yaml["date_end"], "%Y%m%d_%H%M%S")
                    del station_yaml["date_end"]
                if station_yaml["lat"].strip() != "":
                    lat, lon = float(station_yaml["lat"]), float(station_yaml["lon"])
                    del station_yaml["lat"], station_yaml["lon"]
                depth = None
                if station_yaml.get("depth", "").strip() != "":
                    depth = float(station_yaml["depth"])
                    del station_yaml["depth"] #YAML never had this
                st = Station(f"{path}/{station_name}", station_name, lat, lon, depth, date_start, date_end, dataset,
                             other_kwargs=station_yaml)
                self.stations.append(st)

    def load_csv(self, csv_file):
        data = pd.read_csv(csv_file, 
                          parse_dates=["date_start", "date_end"],
                          dtype={"depth": float})  # forcer le type float pour depth
        # parent_path = Path(csv_file).parent

        # for i in data.index:
        #     path = f"{parent_path}/{data.loc[i]['dataset']}/{data.loc[i]['station_name']}"
        #     data.loc[i, "path"] = str(path)

        #     start = data.loc[i]["date_start"].to_pydatetime()
        #     end = data.loc[i]["date_end"].to_pydatetime()

        #     start = None if pd.isnull(start) else start
        #     end = None if pd.isnull(end) else end

        #     # Gestion explicite de la profondeur
        #     depth_value = data.loc[i]["depth"]
        #     depth = float(depth_value) if pd.notnull(depth_value) else None

        #     # kwargs is used to transfer other information, e.g. sensitivity
        #     exclude_keys = ["station_name", "lat", "lon", "depth", "date_start", "date_end", "dataset"]
        #     kwargs = {c: data.loc[i][c] for c in data.columns if c not in exclude_keys}

        #     st = Station(
        #         path=data.loc[i, "path"],
        #         name=data.loc[i, "station_name"],
        #         lat=float(data.loc[i, "lat"]),
        #         lon=float(data.loc[i, "lon"]),
        #         depth=depth,  # Utilisation de la valeur traitée
        #         date_start=start,
        #         date_end=end,
        #         dataset=str(data.loc[i, "dataset"]),
        #         other_kwargs=kwargs
        #     )

        #     self.stations.append(st)
        parent_path = Path(csv_file).parent

        for i in data.index:
            path = f"{parent_path}/{data.loc[i]['dataset']}/{data.loc[i]['station_name']}"
            data.loc[i, "path"] = str(path)

            start = data.loc[i]["date_start"].to_pydatetime()
            end = data.loc[i]["date_end"].to_pydatetime()

            start = None if pd.isnull(start) else start
            end = None if pd.isnull(end) else end

            # Gestion explicite de la profondeur
            depth_value = data.loc[i]["depth"]
            depth = float(depth_value) if pd.notnull(depth_value) else None

            # kwargs is used to transfer other information, e.g. sensitivity
            exclude_keys = ["station_name", "lat", "lon", "depth", "date_start", "date_end", "dataset"]
            kwargs = {c: data.loc[i][c] for c in data.columns if c not in exclude_keys}

            st = Station(
                path=data.loc[i, "path"],
                name=data.loc[i, "station_name"],
                lat=float(data.loc[i, "lat"]),
                lon=float(data.loc[i, "lon"]),
                depth=depth,  # Utilisation de la valeur traitée
                date_start=start,
                date_end=end,
                dataset=str(data.loc[i, "dataset"]),
                other_kwargs=kwargs
            )

            self.stations.append(st)

    def add_station(self, station):
        self.stations.append(station)

    def load_stations(self):
        for st in self.stations:
            st.load_data()

    def by_dataset(self, dataset):
        res = StationsCatalog()
        for st in self.stations:
            if st.dataset == dataset:
                res.add_station(st)
        return res

    def by_name(self, name):
        res = StationsCatalog()
        for st in self.stations:
            if st.name == name:
                res.add_station(st)
        return res

    def by_date(self, date):
        res = StationsCatalog()
        for st in self.stations:
            if st.date_start and st.date_end and st.date_start < date < st.date_end:
                res.add_station(st)
        return res

    def by_dates_or(self, dates):
        res = StationsCatalog()
        for st in self.stations:
            for date in dates:
                if st not in res.stations and st.date_start < date < st.date_end:
                    res.add_station(st)
                    break
        return res

    def by_starting_year(self, year):
        res = StationsCatalog()
        for st in self.stations:
            if year == st.date_start.year:
                res.add_station(st)
        return res

    def starts_before(self, date):
        res = StationsCatalog()
        for st in self.stations:
            if st.date_start < date:
                res.add_station(st)
        return res

    def ends_after(self, date):
        res = StationsCatalog()
        for st in self.stations:
            if date < st.date_end:
                res.add_station(st)
        return res

    def by_date_propagation(self, event_pos, event_date, sound_model, delta=None):
        res = []
        times_of_prop = []
        delta = delta or datetime.timedelta(seconds=0)
        for st in self.stations:
            time_of_prop = sound_model.get_sound_travel_time(event_pos, st.get_pos(), event_date)
            if not time_of_prop:
                continue
            time_of_arrival = event_date + datetime.timedelta(seconds=time_of_prop)
            if st.date_start < time_of_arrival - delta < st.date_end and \
            st.date_start < time_of_arrival + delta < st.date_end:
                times_of_prop.append(time_of_prop)
                res.append((st, time_of_arrival))
        res = np.array(res)[np.argsort(times_of_prop)]
        return res

    def by_loc(self, min_lat, min_lon, max_lat, max_lon):
        res = StationsCatalog()
        for st in self.stations:
            if min_lat < st.lat < max_lat and min_lon < st.lon < max_lon:
                res.add_station(st)
        return res

    def filter_out_unlocated(self):
        res = StationsCatalog()
        for st in self.stations:
            if st.lat is not None:
                res.add_station(st)
        return res

    def filter_out_undated(self):
        res = StationsCatalog()
        for st in self.stations:
            if st.date_start is not None:
                res.add_station(st)
        return res

    def to_dataframe(self):
        df = pd.DataFrame(columns=["station", "lat", "lon", "depth"])
        for i, station in enumerate(self.stations):
            df.loc[i] = [station.name, station.lat, station.lon, station.depth]
        return df

    def get_coordinate_list(self):
        res = []
        for station in self.stations:
            res.append((station.lat, station.lon))
        return np.array(res)

    def __getitem__(self, number):
        return self.stations[number]

    def __len__(self):
        return len(self.stations)

    def __str__(self):
        return f'({", ".join([str(s) for s in self.stations])})'
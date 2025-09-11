import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from src.utils.data_reading.sound_data.station import StationsCatalog

def load_ridge_data(dorsal_db_path):
    """
    Charge les données des dorsales océaniques
    """
    dorsal_files = [f for f in os.listdir(dorsal_db_path) if f.endswith('.xy')]
    print(f"Loading {len(dorsal_files)} ridge files: {dorsal_files}")

    ridge_data = {}
    all_ridge_points = []

    for f in dorsal_files:
        ridge_name = f.replace('axe-', '').replace('.xy', '')
        df = pd.read_csv(os.path.join(dorsal_db_path, f),
                         comment=">", sep=r'\s+')

        ridge_points = df[['y', 'x']].values

        ridge_data[ridge_name] = ridge_points
        all_ridge_points.append(ridge_points)

        print(f"  {ridge_name}: {len(ridge_points)} points")

    # Combinaison de toutes les dorsales
    all_ridge_points = np.vstack(all_ridge_points)
    print(f"Total ridge points: {len(all_ridge_points)}")

    return ridge_data, all_ridge_points


def generate_events_near_ridges(n_events, ridge_points, std_km=50):
    """
    Génère des événements proches des dorsales océaniques
    """
    events = []
    for _ in range(n_events):
        # choisir un point aléatoire sur les dorsales
        ridge_idx = np.random.randint(0, len(ridge_points))
        base_point = ridge_points[ridge_idx]

        # ajouter du bruit normal autour du point (en degrés approximativement)
        # 1 deg ~ 111 km, donc std_deg = std_km / 111
        std_deg = std_km / 111.0
        evt_lat = base_point[0] + np.random.normal(0, std_deg)
        evt_lon = base_point[1] + np.random.normal(0, std_deg)
        events.append([evt_lat, evt_lon])

    return np.array(events)

def load_synthetic_detections(det_files, stations, detections_dir, min_p_tissnet_primary=0.1, min_p_tissnet_secondary=0.1,merge_delta=timedelta(seconds=5)):
    detections = {}
    for det_file in det_files:
        station_dataset, station_name = det_file.split("/")[-2],det_file.split("/")[-1].split("_")[-1].split('-')[0]
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


class RealStationDataGenerator:
    def __init__(self,
                 stations: StationsCatalog,  # Your real station data
                 sound_model,  # model of sound propagation
                 n_real_events=10,  # Number of real seismic events
                 n_noise_detections=100,  # Number of random noise detections
                 detection_probability=1,
                 ridge_data_path=None,  # Path to ridge data directory
                 ridge_std_km=50,  # Standard deviation for events around ridges
                 perfect_events=False,  # Generate perfect events (no noise)
                 apply_clock_drift=True,  # Apply clock drift errors
                 reference_time_years=1,  # Reference time for clock drift calculation
                 seed=0):  # p of detection of each event by each station
        """
        Parameters:
        stations: StationsCatalog object with real station data
        n_real_events: Number of real seismic events to simulate
        n_noise_events: Number of random noise detections
        sound_model: Sound speed in m/s
        ridge_data_path: Path to ridge data directory (if None, events are random)
        ridge_std_km: Standard deviation in km for events around ridges
        perfect_events: If True, no timing noise is added to detections
        apply_clock_drift: If True, applies clock drift errors to stations
        reference_time_years: Reference time in years for clock drift calculation
        seed: random seed used for RNG
        """
        self.stations = stations
        self.n_real_events = n_real_events
        self.n_noise_events = n_noise_detections
        self.sound_model = sound_model
        self.ridge_data_path = ridge_data_path
        self.ridge_std_km = ridge_std_km
        self.perfect_events = perfect_events
        self.apply_clock_drift = apply_clock_drift
        self.reference_time_years = reference_time_years
        self.events = None
        self.ground_truth = None
        self.detection_probability = detection_probability

        # Load ridge data if provided
        self.ridge_points = None
        if ridge_data_path and os.path.exists(ridge_data_path):
            try:
                _, self.ridge_points = load_ridge_data(ridge_data_path)
                print(f"Ridge data loaded successfully: {len(self.ridge_points)} points")
            except Exception as e:
                print(f"Warning: Could not load ridge data: {e}")
                print("Falling back to random event generation")

        np.random.seed(seed)

    def _generate_event_location(self):
        """Generate event location either near ridges or randomly"""
        if self.ridge_points is not None:
            # Generate event near ridges
            ridge_idx = np.random.randint(0, len(self.ridge_points))
            base_point = self.ridge_points[ridge_idx]

            # Add normal noise around the point
            std_deg = self.ridge_std_km / 111.0
            evt_lat = base_point[0] + np.random.normal(0, std_deg)
            evt_lon = base_point[1] + np.random.normal(0, std_deg)
        else:
            # Random event within station coverage area
            stations_coords = self.stations.get_coordinate_list()
            min_lat, max_lat = stations_coords[:, 0].min(), stations_coords[:, 0].max()
            min_lon, max_lon = stations_coords[:, 1].min(), stations_coords[:, 1].max()

            evt_lat = np.random.uniform(min_lat - 5, max_lat + 5)
            evt_lon = np.random.uniform(min_lon - 5, max_lon + 5)

        return evt_lat, evt_lon

    def _apply_station_clock_error(self, station, detection_time, origin_time):
        """Apply clock drift error to detection time if applicable"""
        if not self.apply_clock_drift:
            return detection_time

        # Check if station has clock drift information
        if not hasattr(station, 'other_kwargs') or 'clock_drift_ppm' not in station.other_kwargs:
            return detection_time

        # Check GPS sync status
        gps_sync = station.other_kwargs.get('gps_sync', True)

        if gps_sync:
            # GPS synchronized, no drift correction needed
            return detection_time
        else:
            # GPS not synchronized, apply clock drift error
            clock_drift_ppm = station.other_kwargs['clock_drift_ppm']

            # Calculate time elapsed since reference (in seconds)
            reference_time_seconds = self.reference_time_years * 365.25 * 24 * 3600

            # Get clock error using station method if available
            if hasattr(station, 'get_clock_error'):
                clock_error_seconds = station.get_clock_error(time_elapsed_seconds=reference_time_seconds)
            else:
                # Fallback calculation: ppm * time_elapsed
                clock_error_seconds = clock_drift_ppm * 1e-6 * reference_time_seconds

            # Apply clock error to detection time
            corrected_time = detection_time + timedelta(seconds=clock_error_seconds)
            return corrected_time

    def generate_events(self, start_time, duration_hours=24):
        """Generate synthetic events with ground truth using real stations"""
        # Generate real events
        real_events = []
        self.ground_truth = []

        for event_id in range(self.n_real_events):
            # Generate event location
            evt_lat, evt_lon = self._generate_event_location()

            # Random origin time within duration
            origin_time = start_time + timedelta(
                seconds=np.random.uniform(0, duration_hours * 3600))

            self.ground_truth.append({
                'event_id': event_id,
                'lat': evt_lat,
                'lon': evt_lon,
                'origin_time': origin_time
            })

            # Generate detections for this event
            for station in self.stations:
                # Calculate distance and travel time
                travel_time = self.sound_model.get_sound_travel_time(
                    (evt_lat, evt_lon),
                    (station.get_pos()[0], station.get_pos()[1])
                )

                # Detection probability check
                if np.random.random() < self.detection_probability:
                    # Base detection time
                    base_detection_time = origin_time + timedelta(seconds=travel_time)

                    # Add timing noise (unless perfect events requested)
                    if not self.perfect_events:
                        timing_noise = np.random.normal(0, 1.0)  # 1 second std
                        detection_time = base_detection_time + timedelta(seconds=timing_noise)
                    else:
                        detection_time = base_detection_time

                    # Apply clock drift error
                    final_detection_time = self._apply_station_clock_error(
                        station, detection_time, origin_time
                    )

                    real_events.append({
                        'datetime': final_detection_time,
                        'station': station,
                        'probability': np.random.uniform(0.3, 0.6),
                        'true_event': event_id
                    })

        # Generate noise events
        noise_events = []
        for _ in range(self.n_noise_events):
            station = np.random.choice(self.stations.stations)
            base_time = start_time + timedelta(
                seconds=np.random.uniform(0, duration_hours * 3600)
            )

            # Apply clock drift to noise events as well
            final_time = self._apply_station_clock_error(station, base_time, base_time)

            noise_events.append({
                'datetime': final_time,
                'station': station,
                'probability': np.random.uniform(0.3, 0.6),
                'true_event': -1
            })

        # Combine and shuffle
        all_events = real_events + noise_events
        # np.random.shuffle(all_events)

        self.events = pd.DataFrame(all_events)
        self.ground_truth = pd.DataFrame.from_dict(self.ground_truth).set_index("event_id")
        return self.events, self.ground_truth

    def plot_stations_and_events(self):
        """Visualize hydrophone stations and ground truth acoustic events"""
        import matplotlib.pyplot as plt

        if self.ground_truth is None:
            raise ValueError("No events generated yet. Call generate_events() first.")

        plt.figure(figsize=(12, 10))

        # Plot ridge points if available
        if self.ridge_points is not None:
            plt.scatter(self.ridge_points[:, 1], self.ridge_points[:, 0],
                        c='lightgray', s=1, alpha=0.5, label='Ridge Points')

        # Stations
        coords = self.stations.get_coordinate_list()
        plt.scatter(coords[:, 1], coords[:, 0], c='blue', s=100,
                    marker='^', label='Hydrophone Stations')

        # Ground truth events
        plt.scatter(self.ground_truth.lon, self.ground_truth.lat,
                    c='red', marker='*', s=200, label='Simulated Events')

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        title = "Hydrophone Network with Simulated Acoustic Events"
        if self.ridge_points is not None:
            title += " (Near Ocean Ridges)"
        if self.perfect_events:
            title += " - Perfect Timing"
        if self.apply_clock_drift:
            title += " - With Clock Drift"

        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def get_simulation_info(self):
        """Return information about the simulation parameters"""
        info = {
            'n_real_events': self.n_real_events,
            'n_noise_events': self.n_noise_events,
            'detection_probability': self.detection_probability,
            'perfect_events': self.perfect_events,
            'apply_clock_drift': self.apply_clock_drift,
            'ridge_based': self.ridge_points is not None,
            'ridge_std_km': self.ridge_std_km if self.ridge_points is not None else None,
            'reference_time_years': self.reference_time_years
        }
        return info


# Exemple d'utilisation
if __name__ == "__main__":
    # Example usage with enhanced features
    # Create generator with ridge-based events and clock drift
    generator = RealStationDataGenerator(
        stations=your_stations_catalog,
        sound_model=your_sound_model,
        n_real_events=50,
        n_noise_detections=200,
        ridge_data_path="../../../data/dorsales/",
        ridge_std_km=100,  # Events within 100km of ridges
        perfect_events=False,  # Add timing noise
        apply_clock_drift=True,  # Apply clock drift errors
        reference_time_years=2,  # 2 years reference time
        seed=42
    )

    # Generate events
    start_time = datetime(2023, 1, 1)
    events, ground_truth = generator.generate_events(start_time, duration_hours=48)

    # Plot results
    generator.plot_stations_and_events()

    # Show simulation info
    print("Simulation parameters:")
    for key, value in generator.get_simulation_info().items():
        print(f"  {key}: {value}")
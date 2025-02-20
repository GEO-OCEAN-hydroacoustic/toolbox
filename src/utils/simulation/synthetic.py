import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils.data_reading.sound_data.station import StationsCatalog


class RealStationDataGenerator:
    def __init__(self,
                 stations: StationsCatalog,  # Your real station data
                 sound_model,  # model of sound propagation
                 n_real_events=10,  # Number of real seismic events
                 n_noise_detections=100, # Number of random noise detections
                 detection_probability=1,
                 seed=0):  # p of detection of each event by each station
        """
        Parameters:
        stations: StationsCatalog object with real station data
        n_real_events: Number of real seismic events to simulate
        n_noise_events: Number of random noise detections
        sound_model: Sound speed in m/s
        seed: random seed used for RNG
        """
        self.stations = stations
        self.n_real_events = n_real_events
        self.n_noise_events = n_noise_detections
        self.sound_model = sound_model
        self.events = None
        self.ground_truth = None
        self.detection_probability = detection_probability
        np.random.seed(seed)

    def generate_events(self, start_time, duration_hours=24):
        """Generate synthetic events with ground truth using real stations"""
        # Generate real events
        real_events = []
        self.ground_truth = []

        for event_id in range(self.n_real_events):
            # Random event origin within station coverage area
            stations_coords = self.stations.get_coordinate_list()
            min_lat, max_lat = stations_coords[:,0].min(), stations_coords[:,0].max()
            min_lon, max_lon = stations_coords[:,1].min(), stations_coords[:,1].max()

            evt_lat = np.random.uniform(min_lat-5, max_lat+5)
            evt_lon = np.random.uniform(min_lon-5, max_lon+5)

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
                travel_time = self.sound_model.get_sound_travel_time((evt_lat, evt_lon), (station.get_pos()[0], station.get_pos()[1]))

                # 0.8 for 80% probability of detection
                if np.random.random() < self.detection_probability:
                    detection_time = origin_time + \
                                     timedelta(seconds=travel_time +
                                                       np.random.normal(0, 1.0))  # Add noise

                    real_events.append({
                        'datetime': detection_time,
                        'station': station,
                        'probability': np.random.uniform(0.3, 0.6),
                        'true_event': event_id
                    })

        # Generate noise events
        noise_events = []
        for _ in range(self.n_noise_events):
            station = np.random.choice(self.stations.stations)
            noise_events.append({
                'datetime': start_time + timedelta(
                    seconds=np.random.uniform(0, duration_hours * 3600)),
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
        """Visualize stations and events (requires matplotlib)"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))

        # Plot stations
        plt.scatter(self.stations.lon, self.stations.lat,
                    c='blue', s=100, label='Stations')

        # Plot real events
        true_events = self.events[self.events.true_event >= 0]
        plt.scatter(true_events.longitude, true_events.latitude,
                    c=true_events.true_event, cmap='tab10',
                    s=20, alpha=0.5, label='Real Detections')

        # Plot noise events
        noise = self.events[self.events.true_event < 0]
        plt.scatter(noise.longitude, noise.latitude,
                    c='gray', s=5, alpha=0.2, label='Noise')

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Seismic Network with Synthetic Events")
        plt.legend()
        plt.grid()
        plt.show()
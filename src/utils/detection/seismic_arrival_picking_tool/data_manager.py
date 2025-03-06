import pandas as pd
import numpy as np
from datetime import timedelta
from src.utils.data_reading.sound_data.sound_file_manager import DatFilesManager

class SeismicDataManager:
    def __init__(self, catalogue_path):
        # Load catalogue with additional columns for observed times if not exists
        self.catalogue = pd.read_csv(
            catalogue_path, 
            parse_dates=['time', 'arrival_time'], 
            infer_datetime_format=True
        )
        
        # Add columns for observed times if they don't exist
        if 'predicted_arrival_time' not in self.catalogue.columns:
            self.catalogue['predicted_arrival_time'] = self.catalogue['arrival_time']

        if 'observed_arrival_time' not in self.catalogue.columns:
            self.catalogue['observed_arrival_time'] = pd.NaT

        # Filter only candidate events
        self.candidate_events = self.catalogue[self.catalogue['candidate']].copy()
        
        # Track processed events
        self.processed_events = []
    
    def get_unique_events(self):
        """
        Get unique events from candidate events
        
        Returns:
        - List of unique datetime events
        """
        return sorted(self.candidate_events['time'].unique().tolist())
    
    def get_event_details(self, event_time):
        """
        Retrieve details for a specific event
        
        Args:
        - event_time (datetime): Time of the event
        
        Returns:
        - DataFrame with event details
        """
        event_details = self.candidate_events[self.candidate_events['time'] == event_time]
        return event_details
    
    def get_seismic_data(self, event_time, dat_file_manager):
        """
        Extract seismic data for a specific event
        
        Args:
        - event_time (datetime): Time of the event
        - dat_file_manager (DatFilesManager): File manager for .dat files
        
        Returns:
        - Tuple: (data, sampling_frequency, start_time, end_time)
        """
        event_details = self.get_event_details(event_time)
        first_arrival = event_details['predicted_arrival_time'].min()
        
        # Define time window: 10 minutes before and after first arrival
        start = (first_arrival - timedelta(minutes=10)).replace(tzinfo=None)
        end = (first_arrival + timedelta(minutes=10)).replace(tzinfo=None)
        
        # Get seismic data segment
        data = dat_file_manager.get_segment(start, end)
        sampling_freq = dat_file_manager.sampling_f
        file_numnber = dat_file_manager.find_file_name(start)
        return data, sampling_freq, start, end, file_numnber
    
    def update_arrival_times(self, event_time, phase, predicted_time, observed_time):
        """
        Update arrival times for a specific event and phase
        
        Args:
        - event_time (datetime): Time of the event
        - phase (str): Seismic phase
        - predicted_time (datetime): Predicted arrival time
        - observed_time (datetime): Manually picked observed arrival time
        """
        mask = (
            (self.catalogue['time'] == event_time) & 
            (self.catalogue['phase'] == phase)
        )

        # Update both predicted and observed times
        self.catalogue.loc[mask, 'predicted_arrival_time'] = predicted_time
        self.catalogue.loc[mask, 'observed_arrival_time'] = observed_time-600

        self.processed_events.append((event_time, phase))
    
    def save_updated_catalogue(self, output_path):
        """
        Save the updated catalogue to a CSV file
        
        Args:
        - output_path (str): Path to save the updated CSV
        """
        self.catalogue.to_csv(output_path, index=False)
        print(f"Updated catalogue saved to {output_path}")
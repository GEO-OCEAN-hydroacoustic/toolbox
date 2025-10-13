import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.optimize import least_squares
from pyproj import Geod
import xarray as xr
import os
from src.utils.data_reading.sound_data.sound_file_manager import DatFilesManager
from utils.physics.sound_model import ISAS_grid as isg


class DetectionValidator:
    def __init__(self, velocity_grid, geod, grid_index_to_coord):
        """
        Initialize the detection validator with necessary resources

        Args:
            velocity_grid: Dictionary mapping months to sound velocity datasets
            geod: Geodesic object for distance calculations
            grid_index_to_coord: Function to convert grid indices to coordinates
        """
        self.velocity_grid = velocity_grid
        self.geod = geod
        self.grid_index_to_coord = grid_index_to_coord

        # Constants
        self.DATA_ROOT = "/media/rsafran/CORSAIR/OHASISBIO"
        self.SAMPLING_RATE = 240  # Hz
        self.SAMPLE_INTERVAL = 1 / self.SAMPLING_RATE  # seconds
        self.WINDOW_SEC = 5.0
        self.WINDOW_SAMPLES = int(self.WINDOW_SEC * self.SAMPLING_RATE)
        self.DEFAULT_SOUND_SPEED = 1480  # m/s, for fallback calculations
        self.RMS_THRESHOLD = 2.0  # seconds

        # Geographic parameters
        self.LAT_BOUNDS = [-60, -12.4]
        self.LON_BOUNDS = [35, 100]

    def load_associations(self, file_path):
        """Load association data from file"""
        try:
            return np.load(file_path, allow_pickle=True).item()
        except (IOError, ValueError) as e:
            print(f"Error loading associations: {e}")
            return {}

    def save_validated_associations(self, validated_data, output_path, suffix=None):
        """Save validated associations with optional suffix"""
        if suffix:
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}_{suffix}{ext}"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, validated_data)
        print(f"Saved validated associations to {output_path}")

    def filter_associations(self, associations, min_stations=None, specific_date=None):
        """
        Filter associations based on criteria

        Args:
            associations: Dictionary of associations
            min_stations: Minimum number of stations required
            specific_date: Specific date key to process

        Returns:
            Filtered associations dictionary
        """
        filtered = {}

        for date, associations_list in associations.items():
            if specific_date and date != specific_date:
                continue

            filtered[date] = []
            for assoc in associations_list:
                detections, valid_points = assoc
                if min_stations and len(detections) < min_stations:
                    continue
                filtered[date].append((detections, valid_points))

        return filtered

    def get_refined_detections(self, detections):
        """
        Process detections to get precise times and positions

        Args:
            detections: List of station detections

        Returns:
            Tuple of (refined_detections, station_positions)
        """
        refined_detections = []
        station_positions = []

        for i in range(len(detections)):
            station = detections[i, 0]
            approx_time = detections[i, 1]
            print(f"Processing station {station.name} at {approx_time}...")

            # Get precise detection time by finding max energy
            start = approx_time - timedelta(seconds=6)
            end = approx_time + timedelta(seconds=6)

            data = None
            for dataset in [2017, 2018]:  # Try both datasets
                try:
                    managers = DatFilesManager(f"{self.DATA_ROOT}/{dataset}/{station.name}")
                    data = managers.get_segment(start, end)
                    break  # If successful, exit the loop
                except Exception as e:
                    if dataset == 2018:  # If we've tried both datasets and still failed
                        print(f"Error loading data for station {station.name}: {e}")
                        continue

            if data is None:
                continue  # Skip this station if data couldn't be loaded

            # Create time axis
            times = pd.to_datetime(start) + pd.to_timedelta(np.arange(len(data)) * self.SAMPLE_INTERVAL, unit='s')

            # Compute energy envelope
            energy = np.convolve(data ** 2, np.ones(self.WINDOW_SAMPLES) / self.WINDOW_SAMPLES, mode='same')

            # Find time of max energy
            max_energy_idx = np.argmax(energy)
            precise_time = times[max_energy_idx]

            # Get station position
            station_pos = station.get_pos()  # (lat, lon)
            station_positions.append(station_pos)

            refined_detections.append((station, precise_time))

        return refined_detections, np.array(station_positions)

    def get_initial_guess(self, refined_detections, station_positions, valid_points):
        """
        Determine initial guess for optimization

        Args:
            refined_detections: List of refined detections
            station_positions: Array of station positions
            valid_points: List of valid grid points

        Returns:
            Dictionary with initial parameters
        """
        # Get initial guess from grid points or station centroid
        if len(valid_points) > 0:
            # Use the first valid point as initial guess
            grid_init_lat = self.grid_index_to_coord(valid_points[0, :])[0]
            grid_init_lon = self.grid_index_to_coord(valid_points[0, :])[1]
            valid_coord_points = [self.grid_index_to_coord(valid_points[i, :]) for i in range(len(valid_points))]
        else:
            # Fallback to center of stations as initial guess
            grid_init_lat = np.mean(station_positions[:, 0])
            grid_init_lon = np.mean(station_positions[:, 1])
            valid_coord_points = []

        # Estimate travel time to nearest station for t0 guess
        distances = np.array([
            self.geod.inv(grid_init_lon, grid_init_lat, pos[1], pos[0])[2]
            for pos in station_positions
        ])
        nearest_idx = np.argmin(distances)
        nearest_time = refined_detections[nearest_idx][1]

        # Calculate approximate travel time from source to nearest station
        _, _, distance_m = self.geod.inv(
            grid_init_lon, grid_init_lat,
            station_positions[nearest_idx][1], station_positions[nearest_idx][0]
        )

        # Estimate origin time
        t0_guess = pd.to_datetime(nearest_time) - timedelta(seconds=distance_m / self.DEFAULT_SOUND_SPEED)

        return {
            'lat0': grid_init_lat,
            'lon0': grid_init_lon,
            't0_guess': t0_guess,
            'valid_points': valid_coord_points
        }

    def create_residual_function(self, refined_detections, station_positions, t0_guess, month):
        """
        Create the residual function for least squares optimization

        Args:
            refined_detections: List of refined detections
            station_positions: Array of station positions
            t0_guess: Initial guess for origin time
            month: Month for velocity data selection

        Returns:
            Residual function for optimizer
        """
        # Get the velocity dataset for this month
        ds = self.velocity_grid[month]

        def residual(params):
            lat, lon, t0_offset = params

            # Convert t0_offset (seconds) to datetime
            t0 = pd.to_datetime(t0_guess) + timedelta(seconds=t0_offset)

            errors = []
            weights = []  # For weighted least squares based on error estimates

            for i, (station, detection_time) in enumerate(refined_detections):
                station_lat, station_lon = station.get_pos()

                # Assume source is at average depth of stations for simplicity
                depth = 1250  # meters

                try:
                    # Use 2D propagation at constant depth for simplicity
                    travel_time_sec, tt_error, _ = isg.compute_travel_time(
                        lat, lon, station_lat, station_lon,
                        depth, ds,
                        resolution=20,  # km between points
                        verbose=False,
                        interpolate_missing=True
                    )
                except ValueError:
                    # If sound velocity data is unavailable, use simple estimate
                    _, _, distance_m = self.geod.inv(lon, lat, station_lon, station_lat)
                    travel_time_sec = distance_m / self.DEFAULT_SOUND_SPEED
                    tt_error = travel_time_sec * 0.1  # 10% error estimate

                # Expected arrival time
                expected_time = t0 + timedelta(seconds=travel_time_sec)

                # Calculate time difference in seconds
                time_diff = (detection_time - expected_time).total_seconds()

                # Add to residuals
                errors.append(time_diff)
                weights.append(1.0 / (tt_error + 1))  # Add small constant to avoid division by zero

            # Return weighted residuals
            return np.array(errors) * np.array(weights)

        return residual

    def optimize_source_location(self, residual_func, initial_params, valid_points):
        """
        Find the optimal source location using least squares

        Args:
            residual_func: Residual function for optimization
            initial_params: Dictionary with initial parameters
            valid_points: List of valid grid points to try

        Returns:
            Best optimization result or None if failed
        """
        # Set bounds for the parameters
        lat_bounds = (min(self.LAT_BOUNDS), max(self.LAT_BOUNDS))
        lon_bounds = (min(self.LON_BOUNDS), max(self.LON_BOUNDS))
        t0_bounds = (-3600, 3600)  # seconds offset from initial guess

        bounds = ([lat_bounds[0], lon_bounds[0], t0_bounds[0]],
                  [lat_bounds[1], lon_bounds[1], t0_bounds[1]])

        # Initial guess
        x0 = [initial_params['lat0'], initial_params['lon0'], 0.0]  # lat, lon, t0_offset(seconds)

        # Test multiple starting points from valid_points grid
        best_result = None
        min_cost = float('inf')

        # Try each valid point as starting point
        for point in valid_points:
            point_lat, point_lon = point
            x0_point = [point_lat, point_lon, 0.0]

            try:
                result = least_squares(residual_func, x0_point, bounds=bounds, method='trf', loss='soft_l1', verbose=0)

                if result.cost < min_cost:
                    min_cost = result.cost
                    best_result = result
            except Exception as e:
                print(f"Error with point {point}: {e}")
                continue

        # If no valid points worked, try the initial center guess
        if best_result is None:
            try:
                best_result = least_squares(residual_func, x0, bounds=bounds, method='trf')
                min_cost = best_result.cost
            except Exception as e:
                print(f"Error with initial guess: {e}")

        return best_result

    def validate_single_association(self, association, date):
        """
        Validate a single association

        Args:
            association: Tuple of (detections, valid_points)
            date: Date string for this association

        Returns:
            Validation result dictionary or None if validation failed
        """
        detections, valid_points = association

        # Get month for ISAS data selection
        month = pd.to_datetime(date).month

        # Process detections
        refined_detections, station_positions = self.get_refined_detections(detections)

        # Skip if too few stations have data
        if len(refined_detections) < 3:
            print(f"Skipping association with only {len(refined_detections)} stations")
            return None

        # Get initial parameters
        init_params = self.get_initial_guess(refined_detections, station_positions, valid_points)

        # Create residual function for optimization
        residual_func = self.create_residual_function(
            refined_detections, station_positions, init_params['t0_guess'], month
        )

        # Optimize source location
        result = self.optimize_source_location(residual_func, init_params, init_params['valid_points'])

        if result is not None:
            final_lat, final_lon, t0_offset = result.x
            final_t0 = pd.to_datetime(init_params['t0_guess']) + timedelta(seconds=t0_offset)

            # Calculate RMS error in seconds
            rms_error = np.sqrt(result.cost / len(refined_detections))

            # Check if RMS error is below threshold
            if rms_error < self.RMS_THRESHOLD:
                return {
                    'detections': refined_detections,
                    'source_point': (final_lat, final_lon),
                    'origin_time': final_t0,
                    'rms_error': rms_error,
                    'num_stations': len(refined_detections),
                    'optimization_result': result  # Store the full result object
                }

        return None

    def validate_associations(self, associations, min_stations=None, specific_date=None):
        """
        Validate multiple associations with filtering options

        Args:
            associations: Dictionary of associations by date
            min_stations: Minimum number of stations required (optional)
            specific_date: Specific date to process (optional)

        Returns:
            Dictionary of validated associations
        """
        # Filter associations based on criteria
        filtered_associations = self.filter_associations(
            associations, min_stations, specific_date
        )

        validated_associations = {}

        for date, associations_list in filtered_associations.items():
            print(f"Processing date: {date} ({len(associations_list)} associations)")
            validated_associations[date] = []

            for association in associations_list:
                validation_result = self.validate_single_association(association, date)
                if validation_result:
                    validated_associations[date].append(validation_result)

        return validated_associations


# Main execution code
def main():
    # Setup for geospatial calculations
    geod = Geod(ellps="WGS84")  # Use WGS84 ellipsoid for distance calculations

    # Geographic parameters
    lat_bounds = [-60, -12.4]
    lon_bounds = [35, 100]
    grid_size = 350

    # Create coordinate grids
    PTS_LAT = np.linspace(lat_bounds[0], lat_bounds[1], grid_size)
    PTS_LON = np.linspace(lon_bounds[0], lon_bounds[1], grid_size)

    # Function to convert grid indices to coordinates
    def grid_index_to_coord(indices):
        """Convert grid indices to (lat, lon) coordinates"""
        i, j = indices
        return [PTS_LAT[i], PTS_LON[j]]

    # Load the ISAS grid data for sound velocity
    def load_isas_data(month, lat_bounds, lon_bounds):
        PATH = "/media/rsafran/CORSAIR/ISAS/86442/field/2018"
        return isg.load_ISAS_TS(PATH, month, lat_bounds, lon_bounds, fast=False)

    # Load velocity grid data for all months
    velocity_grid = {}
    for month in range(1, 13):
        print(f"Loading month {month}...")
        ds = load_isas_data(month, lat_bounds, lon_bounds)
        velocity_grid[month] = ds

    # Create validator instance
    validator = DetectionValidator(velocity_grid, geod, grid_index_to_coord)

    # Load association data
    path_asso = "/home/rsafran/PycharmProjects/toolbox/data/detection/association/grids/2018/s_-60--12.4,35-100,350,0.8,0.5.npy"
    associations = validator.load_associations(path_asso)

    # Output base path
    output_base = "/home/rsafran/PycharmProjects/toolbox/data/detection/association/validated/2018/s_-60--12.4,35-100,350,0.8,0.5.npy"

    # Example 1: Validate all associations
    validated_all = validator.validate_associations(associations)
    validator.save_validated_associations(validated_all, output_base, suffix="all")

    # Example 2: Validate only associations with at least 4 stations
    validated_4plus = validator.validate_associations(associations, min_stations=4)
    validator.save_validated_associations(validated_4plus, output_base, suffix="min4stations")

    # Example 3: Validate only associations from a specific date
    specific_date = "2018-03-15"  # Change to your date of interest
    validated_specific_date = validator.validate_associations(associations, specific_date=specific_date)
    validator.save_validated_associations(validated_specific_date, output_base, suffix=f"date_{specific_date}")

    # Example 4: Validate associations from a specific date with at least 5 stations
    validated_date_5stations = validator.validate_associations(
        associations,
        min_stations=5,
        specific_date=specific_date
    )
    validator.save_validated_associations(
        validated_date_5stations,
        output_base,
        suffix=f"date_{specific_date}_min5stations"
    )

    print("Validation complete!")


if __name__ == "__main__":
    main()
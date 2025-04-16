import xarray as xr
import os
import warnings
import gsw
warnings.filterwarnings('ignore')
import numpy as np
from geopy.distance import geodesic

def calculate_sound_velocity(ds, fast=True):
    # Sound velocity formula: Chen & Millero (1977)
    if fast :
        C = 1449.2 + 4.6 * ds.TEMP - 0.055 * ds.TEMP**2 + 0.00029 * ds.TEMP**3 \
            + (1.34 - 0.01 * ds.TEMP) * (ds.PSAL - 35) + 0.016 * ds.depth

    else :
        # Create a 2D grid of depth (converted to negative z) and latitude.
        # Using 'ij' indexing means the first axis corresponds to depth and second axis to latitude.
        Z, LAT = np.meshgrid(-ds.depth.values, ds.latitude.values, indexing='ij')
        lon_array = ds.longitude.values
        Z_3d, LAT_3d, LON_3d = xr.broadcast(
            xr.DataArray(Z, dims=['depth', 'latitude']),
            xr.DataArray(LAT, dims=['depth', 'latitude']),
            ds.longitude
        )
        # Extract numpy arrays for use with GSW:
        Z_3d = Z_3d.values
        LAT_3d = LAT_3d.values
        LON_3d = LON_3d.values

        # --- Calculations ---
        # Calculate pressure in dbar; note that gsw.p_from_z expects z (negative depth)
        p = gsw.p_from_z(Z_3d, LAT_3d)
        # Convert Practical Salinity (PSAL) to Absolute Salinity (SA)
        SA = gsw.SA_from_SP(ds.PSAL.values, p, LON_3d, LAT_3d)
        # Convert in-situ temperature (TEMP) to Conservative Temperature (CT)
        CT = gsw.CT_from_t(SA, ds.TEMP.values, p)
        # Calculate the sound speed (in m/s) using gsw.sound_speed.
        C = gsw.sound_speed(SA, CT, p)
    return C

# Define the Mackenzie sound speed function
def mackenzie_sound_speed(ds):
    temp, psal, depth = ds.TEMP,ds.PSAL, ds.DEPTH
    C = (1448.96
         + 4.591 * temp
         - 5.304e-2 * temp**2
         + 2.374e-4 * temp**3
         + 1.340 * (psal - 35)
         + 1.630e-2 * depth
         + 1.675e-7 * depth**2
         - 1.025e-2 * temp * (psal - 35)
         - 7.139e-13 * temp * depth**3)
    return C

def load_ISAS_TS(ISAS_Repertory, month,lat_bounds,lon_bounds, fast=True):
    """ Function to load the ISAS Temperature and Salainity files.
    :param ISAS_Repertory: repertory were nc files are stored. it should be 24nc files per year
    :param month: month number from 1 to 12
    :param data_name: Name of the main variable in the netCDF, like "t_an" in WOA temperature.
    :param lat_bounds: Latitude bounds as a size 2 float array (°) (points outside these bounds will be ignored).
    :param lon_bounds: Longitude bounds as a size 2 float array (°) (points outside these bounds will be ignored).
    :return: xarray.Dataset containing temperature and salainity, sound velocity and metadata
    """
    arr = os.listdir(ISAS_Repertory)
    file_list = [os.path.join(ISAS_Repertory, fname) for fname in arr if fname.endswith('.nc')]
    file_list_temp = [file for file in file_list if file.endswith('TEMP.nc')]
    file_list_psal = [file for file in file_list if file.endswith('PSAL.nc')]
    file_list_temp = sorted(file_list_temp)
    file_list_psal = sorted(file_list_psal)

    ds_t = xr.open_dataset(file_list_temp[month-1], engine='netcdf4', decode_times=False)
    ds_s = xr.open_dataset(file_list_psal[month-1], engine='netcdf4', decode_times=False)
    # Subset the dataset
    lat_min, lat_max = lat_bounds[0], lat_bounds[1]
    lon_min, lon_max = lon_bounds[0], lon_bounds[1]
    ds_t = ds_t.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    ds_s = ds_s.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    ds = xr.align(ds_s, ds_t)
    ds = xr.merge(ds)
    # Calculate sound velocity for each point in the dataset
    sound_velocity = calculate_sound_velocity(ds,fast)
    # Add sound velocity as a new variable to the dataset
    if fast:
        sound_velocity = xr.Dataset(dict(SOUND_VELOCITY=sound_velocity))
        ds = xr.align(ds, sound_velocity)
        ds = xr.merge(ds)
        ds.SOUND_VELOCITY.attrs["units"] = r"$m.s^{-1}$"
        ds.SOUND_VELOCITY.attrs["long_name"] = 'Sound Velocity'
    else :
        ds = ds.assign(SOUND_VELOCITY=(ds.TEMP.dims, sound_velocity))
        ds['SOUND_VELOCITY'].attrs["units"] = "m s^{-1}"
        ds['SOUND_VELOCITY'].attrs["long_name"] = "Sound Velocity"
    return ds

def _generate_coordinates_with_fixed_resolution(lat1, lon1, lat2, lon2, resolution):
    """
    Generate coordinates along the great-circle path between two points with fixed resolution.
    Much faster than geopy version using pyproj.
    """
    # Compute forward and back azimuths, and total distance
    from pyproj import Geod
    geod = Geod(ellps="WGS84")  # use WGS84 ellipsoid

    az12, az21, total_distance = geod.inv(lon1, lat1, lon2, lat2)

    # Compute number of points
    num_points = int(np.floor(total_distance / resolution)) + 1
    distances = np.linspace(0, total_distance, num_points)

    # Use pyproj's fwd function to get coordinates efficiently
    lons, lats, _ = geod.fwd(
        np.full(num_points, lon1),
        np.full(num_points, lat1),
        np.full(num_points, az12),
        distances
    )

    # Ensure last point is exact
    lats[-1] = lat2
    lons[-1] = lon2

    coordinates = np.column_stack((lats, lons))
    return coordinates, distances, total_distance

def extract_velocity_profile(ds, coordinates, method ='nearest') :
    depth = ds['depth'].values
    temp_profile = np.full((len(depth), len(coordinates)), np.nan)
    for i, (lat, lon) in enumerate(coordinates):
        temp_at_point = ds.sel(latitude=lat, longitude=lon, method=method)['SOUND_VELOCITY']
        temp_profile[:, i] = temp_at_point.values
    return temp_profile, depth

def compute_travel_time(lat1, lon1, lat2, lon2, depth, ds, resolution=10, verbose=True):
    """
    Compute the travel time of a sound wave between two points at a given depth.

    Parameters:
    - lat1, lon1: Latitude and longitude of point A.
    - lat2, lon2: Latitude and longitude of point B.
    - depth: Depth (in meters) at which to compute sound velocity.
    - ds: xarray.Dataset containing 'SOUND_VELOCITY' with dimensions (depth, latitude, longitude).
    - resolution: Spacing in km between points along the geodesic path.
    - verbose: If True, prints diagnostics.

    Returns:
    - travel_time: Total travel time in seconds.
    - total_distance: Total distance in meters.
    """

    # Convert resolution to meters
    resolution_m = resolution * 1000

    # Generate coordinates along the great-circle path
    coordinates, _, total_distance = _generate_coordinates_with_fixed_resolution(
        lat1, lon1, lat2, lon2, resolution_m
    )

    # Get sound velocity profile at each coordinate point
    sound_velocity_profile = []
    for lat, lon in coordinates:
        c = ds.sel(latitude=lat, longitude=lon, method='nearest')['SOUND_VELOCITY'] \
              .sel(depth=depth, method='nearest').values

        if np.isnan(c) or c < 100 or c > 1700:  # sanity check
            raise ValueError(f"Unrealistic sound velocity ({c}) at lat={lat}, lon={lon}, depth={depth}")

        sound_velocity_profile.append(c)

    sound_velocity_profile = np.array(sound_velocity_profile)

    # Compute segment distances and use average velocity for each segment
    travel_time = 0
    segment_lengths = []

    for i in range(len(coordinates) - 1):
        pt1 = coordinates[i]
        pt2 = coordinates[i + 1]

        # Distance between points in meters
        segment_length = geodesic(pt1, pt2).meters
        segment_lengths.append(segment_length)

        # Use average velocity between the two points
        v1 = sound_velocity_profile[i]
        v2 = sound_velocity_profile[i + 1]
        avg_velocity = 0.5 * (v1 + v2)

        travel_time += segment_length / avg_velocity

    # Verbose diagnostics
    if verbose:
        print(f"Total distance: {total_distance/1000:.2f} km")
        print(f"Segments: {len(segment_lengths)}")
        print(f"Velocity: mean={np.mean(sound_velocity_profile):.2f} m/s, min={np.min(sound_velocity_profile):.2f}, max={np.max(sound_velocity_profile):.2f}")
        print(f"Travel time: {travel_time[0]:.2f} s ({travel_time[0]/3600:.2f} hr)")

    return travel_time[-1], total_distance



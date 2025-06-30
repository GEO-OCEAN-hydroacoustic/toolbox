import xarray as xr
import os
import warnings
import gsw
warnings.filterwarnings('ignore')
import numpy as np
from geopy.distance import geodesic
from scipy import interpolate

def calculate_sound_velocity(ds, fast=True):
    # Sound velocity formula: Chen & Millero (1977)
    if fast :
        C = 1449.2 + 4.6 * ds.TEMP - 0.055 * ds.TEMP**2 + 0.00029 * ds.TEMP**3 \
            + (1.34 - 0.01 * ds.TEMP) * (ds.PSAL - 35) + 0.016 * ds.depth

    else :
        # Create a 2D grid of depth (converted to negative z) and latitude.
        # Using 'ij' indexing means the first axis corresponds to depth and second axis to latitude.
        Z, LAT = np.meshgrid(-ds.depth.values, ds.latitude.values, indexing='ij')
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


def calculate_gsw_celerity_error(ds, sigma_P=0.1):
    """
    Compute error in sound speed (m/s) due to measurement uncertainty
    using partial derivatives and error propagation.

    Parameters:
    ds -- xarray Dataset containing:
          - TEMP: In-situ temperature (°C)
          - PSAL: Practical salinity
          - TEMP_ERR: Temperature uncertainty (°C)
          - PSAL_ERR: Salinity uncertainty
          - depth: Depth coordinates (m, negative downward)
          - latitude: Latitude coordinates (degrees)
          - longitude: Longitude coordinates (degrees)
    sigma_P -- Uncertainty in pressure (dbar), default = 0.1 dbar

    Returns:
    C_error -- Estimated sound speed error (m/s)
    """
    # Create mesh grids for coordinates
    Z, LAT = np.meshgrid(-ds.depth.values, ds.latitude.values, indexing='ij')
    Z_3d, LAT_3d, LON_3d = xr.broadcast(
        xr.DataArray(Z, dims=['depth', 'latitude']),
        xr.DataArray(LAT, dims=['depth', 'latitude']),
        ds.longitude
    )

    # Extract numpy arrays
    Z_3d = Z_3d.values
    LAT_3d = LAT_3d.values
    LON_3d = LON_3d.values

    # Calculate pressure from depth
    p = gsw.p_from_z(Z_3d, LAT_3d)

    # Define small deltas for finite difference calculations
    dT = 1e-2  # Small temperature difference (°C)
    dS = 1e-2  # Small salinity difference (g/kg)
    dP = 1e-1  # Small pressure difference (dbar)

    # Calculate temperature partial derivative
    # First convert T+dT to Conservative Temperature
    SA_same = gsw.SA_from_SP(ds.PSAL.values, p, LON_3d, LAT_3d)
    CT_plus = gsw.CT_from_t(SA_same, ds.TEMP.values + dT, p)
    CT_minus = gsw.CT_from_t(SA_same, ds.TEMP.values - dT, p)

    # Calculate sound speed at T+dT and T-dT
    C_T_plus = gsw.sound_speed(SA_same, CT_plus, p)
    C_T_minus = gsw.sound_speed(SA_same, CT_minus, p)

    # Partial derivative with respect to temperature
    dC_dT = (C_T_plus - C_T_minus) / (2 * dT)

    # Calculate salinity partial derivative
    # First convert S+dS and S-dS to Absolute Salinity
    SA_plus = gsw.SA_from_SP(ds.PSAL.values + dS, p, LON_3d, LAT_3d)
    SA_minus = gsw.SA_from_SP(ds.PSAL.values - dS, p, LON_3d, LAT_3d)

    # Convert to Conservative Temperature with the modified salinity
    CT_S_plus = gsw.CT_from_t(SA_plus, ds.TEMP.values, p)
    CT_S_minus = gsw.CT_from_t(SA_minus, ds.TEMP.values, p)

    # Calculate sound speed at S+dS and S-dS
    C_S_plus = gsw.sound_speed(SA_plus, CT_S_plus, p)
    C_S_minus = gsw.sound_speed(SA_minus, CT_S_minus, p)

    # Partial derivative with respect to salinity
    dC_dS = (C_S_plus - C_S_minus) / (2 * dS)

    # Calculate pressure partial derivative
    # Need to recalculate SA and CT for different pressures since they depend on pressure
    # For P+dP
    SA_P_plus = gsw.SA_from_SP(ds.PSAL.values, p + dP, LON_3d, LAT_3d)
    CT_P_plus = gsw.CT_from_t(SA_P_plus, ds.TEMP.values, p + dP)
    C_P_plus = gsw.sound_speed(SA_P_plus, CT_P_plus, p + dP)

    # For P-dP
    SA_P_minus = gsw.SA_from_SP(ds.PSAL.values, p - dP, LON_3d, LAT_3d)
    CT_P_minus = gsw.CT_from_t(SA_P_minus, ds.TEMP.values, p - dP)
    C_P_minus = gsw.sound_speed(SA_P_minus, CT_P_minus, p - dP)

    # Partial derivative with respect to pressure
    dC_dP = (C_P_plus - C_P_minus) / (2 * dP)

    # Get the actual uncertainties from the dataset
    sigma_T = ds.TEMP_ERR  # Temperature uncertainty
    sigma_S = ds.PSAL_ERR  # Salinity uncertainty
    # sigma_P is passed as a parameter (default = 0.1 dbar)

    # Apply error propagation formula including pressure uncertainty
    C_error = np.sqrt(
        (dC_dT * sigma_T) ** 2 +
        (dC_dS * sigma_S) ** 2 +
        (dC_dP * sigma_P) ** 2
    )

    return C_error

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
    # sv_err = calculate_celerity_error(ds, use_pctvar=False)
    sv_err = calculate_gsw_celerity_error(ds)
    ds = ds.assign(SV_ERR=(ds.TEMP.dims, sv_err.data))
    ds["SV_ERR"].attrs["units"] = "m s^{-1}"
    ds["SV_ERR"].attrs["long_name"] = "Estimated Uncertainty in Sound Velocity"
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


def extract_velocity_profile(ds, coordinates, method='nearest', interpolate_missing=False):
    depth = ds['depth'].values
    # temp_profile = np.full((len(depth), len(coordinates)), np.nan)
    # temp_err = np.full((len(depth), len(coordinates)), np.nan)
    # for i, (lat, lon) in enumerate(coordinates):
    #     temp_at_point = ds.sel(latitude=lat, longitude=lon, method=method)['SOUND_VELOCITY']
    #     temp_profile[:, i] = temp_at_point.values
    #     temp_err_at_point = ds.sel(latitude=lat, longitude=lon, method=method)['SV_ERR']
    #     temp_err[:, i] = temp_err_at_point.values
    # First extract all data points, even if they contain NaN values
    points = xr.Dataset({'latitude': (['points'], coordinates[:,0]),
                         'longitude': (['points'], coordinates[:,1])})
    temp_profile = ds['SOUND_VELOCITY'].sel(latitude=points['latitude'],
                                        longitude=points['longitude'],
                                        method='nearest').values[0]
    temp_err = ds['SV_ERR'].sel(latitude=points['latitude'],
                                        longitude=points['longitude'],
                                        method='nearest').values[0]

    # If interpolation is requested, perform horizontal interpolation
    if interpolate_missing:
        # Interpolate along each depth level (horizontally)
        for d in range(len(depth)):
            row = temp_profile[d, :]
            row_err = temp_err[d, :]
            # Find valid indices
            valid_indices = np.where(~np.isnan(row))[0]

            # Only interpolate if we have at least 2 valid points
            if len(valid_indices) >= 2:
                valid_distances = np.arange(len(coordinates))[valid_indices]
                valid_values = row[valid_indices]

                # Create interpolated values using numpy's interp function
                interp_values = np.interp(
                    np.arange(len(coordinates)),  # All x positions
                    valid_distances,  # Valid x positions
                    valid_values,  # Valid values
                    left=np.nan,  # Don't extrapolate left
                    right=np.nan  # Don't extrapolate right
                )
                temp_err_int = np.interp(
                    np.arange(len(coordinates)),  # All x positions
                    valid_distances,  # Valid x positions
                    row_err[valid_indices],  # Valid values
                    left=np.nan,  # Don't extrapolate left
                    right=np.nan  # Don't extrapolate right
                )

                temp_profile[d, :] = interp_values
                temp_err[d, :] = temp_err_int
    return temp_profile, temp_err, depth


def compute_travel_time(lat1, lon1, lat2, lon2, depth, ds, resolution=10, verbose=False, interpolate_missing=False, use_harmonic=True):
    """
    Compute the travel time of a sound wave between two points at a given depth.

    Parameters:
    - lat1, lon1: Latitude and longitude of point A.
    - lat2, lon2: Latitude and longitude of point B.
    - depth: Depth (in meters) at which to compute sound velocity.
    - ds: xarray.Dataset containing 'SOUND_VELOCITY' with dimensions (depth, latitude, longitude).
    - resolution: Spacing in km between points along the geodesic path.
    - verbose: If True, prints diagnostics.
    - interpolate_missing: If True, interpolates horizontally to fill missing values.

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

    lats, lons = zip(*coordinates)
    lats = np.array(lats)
    lons = np.array(lons)

    sound_velocity_profile = ds['SOUND_VELOCITY'].interp(
        latitude=("points", lats),
        longitude=("points", lons),
        depth=depth,  method="nearest"
    ).values

    # Apply horizontal interpolation if requested
    if interpolate_missing and np.any(np.isnan(sound_velocity_profile)):
        # print('interpolating missing values')
        # Find valid indices
        sound_velocity_profile = sound_velocity_profile.flatten()
        mask = ~np.isnan(sound_velocity_profile)
        indices = np.arange(len(sound_velocity_profile.flatten()))

        valid_indices = indices[mask.flatten()]
        # Only interpolate if we have at least 2 valid points
        if len(valid_indices) >= 2:
            valid_values = sound_velocity_profile[valid_indices]
            # Create interpolated values using numpy's interp function
            f = interpolate.interp1d(valid_indices, valid_values,
                                     bounds_error=False,  # Allow extrapolation
                                     fill_value="extrapolate")
            # Apply interpolation to the entire array
            sound_velocity_profile = f(indices).reshape(-1, 1)
    sound_velocity_profile = np.squeeze(sound_velocity_profile)
    # Check for any remaining NaN values or unrealistic sound velocities
    if np.any(np.isnan(sound_velocity_profile)) or np.any(sound_velocity_profile < 100) or np.any(sound_velocity_profile > 1700):
        for i, c in enumerate(sound_velocity_profile.flatten()):
            if np.isnan(c) or c < 100 or c > 1700:  # sanity check
                lat, lon = coordinates[i]
                raise ValueError(f"Unrealistic sound velocity ({c}) at lat={lat}, lon={lon}, depth={depth}")

    sv_error_profile = ds['SV_ERR'].interp(
        latitude=("points", lats),
        longitude=("points", lons),
        depth=depth,  method="nearest"
    ).values
    sv_error_profile = np.squeeze(sv_error_profile)

    # This could be vectorized as:
    segment_lengths = np.array([geodesic(coordinates[i], coordinates[i+1]).meters for i in range(len(coordinates)-1)])
    avg_velocities = 0.5 * (sound_velocity_profile[..., :-1] + sound_velocity_profile[..., 1:])
    travel_time = np.sum(segment_lengths / avg_velocities)
    avg_error = 0.5 * (sv_error_profile[...,:-1] + sv_error_profile[...,1:])
    tt_segment_err = np.square((segment_lengths / avg_velocities ** 2) * avg_error)
    tt_total_err = np.sqrt(np.nansum(tt_segment_err))

    if not hasattr(travel_time, '__len__'):
        travel_time = np.array([travel_time])

    # Verbose diagnostics
    if verbose:
        print(f"Total distance: {total_distance / 1000:.2f} km")
        print(f"Segments: {len(segment_lengths)}")
        print(
            f"Velocity: mean={np.mean(sound_velocity_profile):.2f} m/s, min={np.min(sound_velocity_profile):.2f}, max={np.max(sound_velocity_profile):.2f}")
        print(f"Travel time: {travel_time[-1]:.2f} s ({travel_time[-1] / 3600:.2f} hr)")
        print(f"Travel time error: {tt_total_err:.2f} s")
    return travel_time[-1], tt_total_err, total_distance


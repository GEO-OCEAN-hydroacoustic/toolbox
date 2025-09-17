#!/usr/bin/env python3
"""
ls_association_with_clock_drift.py

Seismic event localization with clock drift estimation.
Implementation of a multi-phase approach for handling stations with unknown clock drift:
1. Initial localization with increased uncertainty for affected stations
2. Global clock drift parameter estimation using all events

This implementation follows the model: observed_time = a * true_time + b
where 'a' represents the clock drift factor and 'b' the absolute timing offset.
"""

import os
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.optimize import least_squares
from utils.physics.sound_model import ISAS_grid as isg
from pyproj import Geod
from joblib import Parallel, delayed, Memory
import time
import psutil
import pickle
import warnings

# === CONFIGURATION ===
ASSO_FILE = "/home/rsafran/PycharmProjects/toolbox/data/detection/association/grids/refined_s_-60-5,35-120,400,0.8,0.6-part0.npy"
OUTPUT_DIR = "/media/rsafran/CORSAIR/Association/validated"
OUTPUT_BASENAME = "refined_s_-60-5,35-120,350,0.8,0.6"
BATCH_SIZE = 250  # checkpoint every N dates
N_JOBS = max(1, os.cpu_count() - 1)  # leave one core free
GRID_LAT_BOUNDS = [-60, 5]
GRID_LON_BOUNDS = [35, 120]
GRID_SIZE = 400
SOUND_SPEED = 1480  # m/s, rough constant
GTOL = 1e-5
XTOL = 1e-5
ISAS_PATH = "/media/rsafran/CORSAIR/ISAS/exctracted/2018"

# Enhanced error model parameters
PICKING_ERROR_BASE = 2  # Base picking error in seconds

# Clock drift parameters
MAX_DRIFT_PPM = 1  # Maximum clock drift in parts per million
DRIFT_UNCERTAINTY_FACTOR = 2  # Factor to increase uncertainty for stations with unknown drift

# Stations with unknown clock drift
UNKNOWN_DRIFT = {
    "MADW": ['not_ok'],
    "NEAMS": ['not_ok'],
    "RTJ": ['not_ok'],
    "SSEIR": ['not_ok'],
    "WKER2": ['not_ok']
}

# Optimization flags
USE_CACHE = True
PRECOMPUTE_TRAVEL_TIMES = True
VERBOSE_OPTIMIZATION = 0  # Set to 1 for verbose optimization output
CHUNK_SIZE = 10  # For parallel processing

# Cache directory for ISAS data
CACHE_DIR = os.path.join(OUTPUT_DIR, "isas_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# === PERFORMANCE MONITORING ===
start_time = time.time()


def log_progress(message):
    elapsed = time.time() - start_time
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    print(f"[{elapsed:.1f}s | {memory_usage:.1f}MB] {message}")

# === INITIALIZATION ===
# Precompute grid lat/lon
PTS_LAT = np.linspace(*GRID_LAT_BOUNDS, GRID_SIZE)
PTS_LON = np.linspace(*GRID_LON_BOUNDS, GRID_SIZE)

# Geod instance for geodesic calculations
geod = Geod(ellps="WGS84")
# Setup memory caching
memory_cache = Memory(location=os.path.join(OUTPUT_DIR, "joblib_cache"), verbose=0)


# === ISAS DATA LOADING ===
def get_isas_data(month):
    """Load ISAS data with file-based caching to avoid serialization issues"""
    cache_file = os.path.join(CACHE_DIR, f"isas_month_{month}.pkl")

    # Try to load from cache file first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            # If loading fails, regenerate
            pass

    # Generate the data
    print(f"Loading ISAS data for month {month}...")
    data = isg.load_ISAS_extracted(
        ISAS_PATH, month
    )

    # Cache the result to file
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Warning: Failed to cache ISAS data: {e}")

    return data


def grid_index_to_coord(indices):
    """Convert grid indices to geographic coordinates"""
    i, j = indices
    return [PTS_LAT[i], PTS_LON[j]]


def compute_travel_time(lat, lon, station_lat, station_lon, month, travel_time_cache):
    """Compute travel time with process-local dictionary caching"""

    # Using provided cache dictionary
    key = (lat, lon, station_lat, station_lon, month)
    if key not in travel_time_cache:
        try:
            ds = travel_time_cache.get('isas_' + str(month), None)
            if ds is None:
                ds = get_isas_data(month)
                travel_time_cache['isas_' + str(month)] = ds

            tt, err, dist_m = isg.compute_travel_time(
                lat, lon, station_lat, station_lon,
                ds,
                resolution=10,
                verbose=False,
                interpolate_missing=True
            )
        except Exception:
            # Fallback to simple calculation
            _, _, dist_m = geod.inv(lon, lat, station_lon, station_lat)
            tt = dist_m / SOUND_SPEED
            err = tt * 0.1

        picking_err = PICKING_ERROR_BASE
        total_err = np.sqrt(picking_err ** 2 + err ** 2)

        travel_time_cache[key] = (tt, total_err, dist_m)

    return travel_time_cache[key]


geod = Geod(ellps="WGS84")

def theoretical_derivatives(lat1, lon1, lat2, lon2):
    """
    Compute derivatives of geodesic distance with respect to endpoint 2
    All inputs must be scalars or same-sized arrays
    Returns: ds/dlat2, ds/dlon2 in meters per radian
    """
    # Get geodesic parameters - all inputs must have same shape
    fwd_azi, back_azi, distance = geod.inv(lon1, lat1, lon2, lat2)

    # Convert to radians
    lat2_rad = np.radians(lat2)
    # Use back azimuth at point 2 (azi2 in our notation)
    azi2_rad = np.radians(back_azi)

    # WGS84 parameters
    a = geod.a  # semi-major axis
    f = geod.f  # flattening
    e2 = f * (2 - f)  # first eccentricity squared

    # Trigonometric components
    sin_lat2 = np.sin(lat2_rad)
    cos_lat2 = np.cos(lat2_rad)
    sin_azi2 = np.sin(azi2_rad)
    cos_azi2 = np.cos(azi2_rad)

    # Radii of curvature at point 2
    nu2 = 1 - e2 * sin_lat2**2
    N2 = a / np.sqrt(nu2)  # Normal radius (east-west)
    M2 = a * (1 - e2) / (nu2**1.5)  # Meridional radius (north-south)

    # Derivatives in meters per radian
    ds_dlat2 = M2 * cos_azi2  # ρ₂ cos(α₂)
    ds_dlon2 = N2 * cos_lat2 * sin_azi2  # ν₂ cos(φ₂) sin(α₂)

    return ds_dlat2, ds_dlon2

def jacobian_tdoa(receivers_lat, receivers_lon, source_lat, source_lon):
    """
    Compute Jacobian matrix for TDOA system using vectorized operations
    """
    n_receivers = len(receivers_lat)

    # Create arrays with same size for geod.inv
    source_lat_array = np.full(n_receivers, source_lat)
    source_lon_array = np.full(n_receivers, source_lon)

    # Compute all derivatives at once using vectorization
    dsi_dlat, dsi_dlon = theoretical_derivatives(
        source_lat_array, source_lon_array,  # source positions (repeated)
        receivers_lat, receivers_lon         # all receiver positions
    )

    # Reference receiver derivatives (index 0)
    ds0_dlat = dsi_dlat[0]
    ds0_dlon = dsi_dlon[0]

    # Build Jacobian matrix
    J = np.zeros((n_receivers, 2))

    for i in range(n_receivers):
        if i == 0:
            # Reference receiver: s_0 - s_0 = 0
            J[i, 0] = 0.0
            J[i, 1] = 0.0
        else:
            # Difference derivatives: ∂(s_i - s_0)/∂source
            J[i, 0] = dsi_dlat[i] - ds0_dlat  # ∂(s_i - s_0)/∂lat
            J[i, 1] = dsi_dlon[i] - ds0_dlon  # ∂(s_i - s_0)/∂lon

    return J

def has_unknown_drift(station_name):
    """Check if station has unknown drift"""
    return station_name in UNKNOWN_DRIFT


def estimate_clock_drift(observed_times, predicted_times, station_name):
    """
    Estimate clock drift parameters for a station

    Model: observed_time = a * predicted_time + b
    where:
        a = clock drift factor (unitless)
        b = absolute timing offset (seconds)
    """
    # Filter out any problematic values
    valid_indices = ~np.isnan(observed_times) & ~np.isnan(predicted_times)
    if np.sum(valid_indices) < 5:
        # Not enough data points for reliable estimation
        return None, None, None

    x = np.array(predicted_times)[valid_indices]
    y = np.array(observed_times)[valid_indices]

    # Initial guess: no drift, no offset
    params0 = [1.0, 0.0]

    # Bounds: drift within MAX_DRIFT_PPM, offset within reasonable range
    # a = 1 ± MAX_DRIFT_PPM/1e6
    a_lower = 1 - MAX_DRIFT_PPM / 1e6
    a_upper = 1 + MAX_DRIFT_PPM / 1e6
    b_range = 5.0  # ±5 seconds offset

    bounds = ([a_lower, -b_range], [a_upper, b_range])

    def residual_func(params):
        a, b = params
        return y - (a * x + b)

    try:
        res = least_squares(residual_func, params0, bounds=bounds, method='trf', loss='soft_l1')
        a, b = res.x

        # Calculate residuals and RMS error
        residuals = residual_func(res.x)
        rms_error = np.sqrt(np.mean(residuals ** 2))

        return a, b, rms_error
    except Exception as e:
        print(f"Failed to estimate drift for station {station_name}: {e}")
        return None, None, None


def process_date(date, associations_list):
    """Process a single date of associations with clock drift estimation"""
    month = pd.to_datetime(date).month
    validated = []

    # Create a local travel time cache for this process
    travel_time_cache = {}

    # Create simplified associations list to avoid serialization issues
    simplified_associations = []
    for detections, valid_points in associations_list:
        simple_detections = []
        for station_obj, det_time in detections:
            # Extract only necessary data from station_obj
            lat, lon = station_obj.get_pos()
            station_name = station_obj.name  # Get station name
            simple_detections.append(((lat, lon), det_time, station_name))
        simplified_associations.append((simple_detections, valid_points))

    for detections, valid_points in simplified_associations:
        # Skip tiny clusters
        if len(detections) < 5:
            continue

        # Build refined detections & station positions
        station_positions = [pos for pos, _, _ in detections]
        station_names = [name for _, _, name in detections]
        detection_times = [t for _, t, _ in detections]
        ref_time = min(detection_times)

        # Keep track of which stations have unknown drift
        has_drift = [has_unknown_drift(name) for name in station_names]

        # Convert detection times to seconds from reference time
        detection_secs = [(pos, (t - ref_time).total_seconds(), name) for pos, t, name in detections]

        # Select reference station - prefer one without drift issues
        ref_candidates = [i for i, drift in enumerate(has_drift) if not drift]
        if ref_candidates:
            ref_station_idx = ref_candidates[0]
        else:
            ref_station_idx = 0

        ref_station_pos, ref_station_sec, ref_station_name = detection_secs[ref_station_idx]

        # Create TDOA pairs using reference station
        tdoa_pairs = []
        for i, (pos, sec, name) in enumerate(detection_secs):
            if i != ref_station_idx:
                # Time difference relative to reference station
                tdoa = sec - ref_station_sec
                # Keep track of which stations have unknown drift
                has_unknown_drift_flag = has_drift[i] or has_drift[ref_station_idx]
                tdoa_pairs.append((ref_station_pos, pos, tdoa, has_unknown_drift_flag))

        # Initial guess & bounds from valid_points
        if len(valid_points) > 0:
            coords = np.array([grid_index_to_coord(tuple(p)) for p in valid_points])
            lat0, lon0 = coords.mean(axis=0)
            margin = 1
            lat_min, lat_max = coords[:, 0].min() - margin, coords[:, 0].max() + margin
            lon_min, lon_max = coords[:, 1].min() - margin, coords[:, 1].max() + margin
        else:
            arr = np.array(station_positions)
            lat0, lon0 = arr[:, 0].mean(), arr[:, 1].mean()
            lat_min, lat_max = arr[:, 0].min() - 0.5, arr[:, 0].max() + 0.5
            lon_min, lon_max = arr[:, 1].min() - 0.5, arr[:, 1].max() + 0.5

        # TDOA-based residual function with uncertainty adjustment for stations with unknown drift
        def tdoa_residual(params):
            lat, lon = params  # Only location parameters, no timing bias

            residuals = np.zeros(len(tdoa_pairs))
            sigmas = np.zeros(len(tdoa_pairs))

            for i, (ref_pos, pos, observed_tdoa, has_unknown_drift_flag) in enumerate(tdoa_pairs):
                ref_lat, ref_lon = ref_pos
                sta_lat, sta_lon = pos

                # Calculate travel times to both stations
                tt_ref, err_ref, _ = compute_travel_time(lat, lon, ref_lat, ref_lon, month, travel_time_cache)
                tt_sta, err_sta, _ = compute_travel_time(lat, lon, sta_lat, sta_lon, month, travel_time_cache)

                # Predicted TDOA: difference in travel times
                predicted_tdoa = tt_sta - tt_ref

                # Residual is observed minus predicted
                residuals[i] = observed_tdoa - predicted_tdoa

                # Combined error (errors add in quadrature for independent measurements)
                # Increase uncertainty for stations with unknown drift
                if has_unknown_drift_flag:
                    sigma = np.sqrt(err_ref ** 2 + err_sta ** 2+DRIFT_UNCERTAINTY_FACTOR**2)
                    # sigma *= DRIFT_UNCERTAINTY_FACTOR
                else :
                    sigma = np.sqrt(err_ref ** 2 + err_sta ** 2)
                sigmas[i] = max(sigma, 0.01)

            # Return normalized residuals
            return residuals / sigmas

        def tdoa_jac(params):
            lat, lon = params
            n_receivers = len(tdoa_pairs)
            receivers_lat = np.array([pos[0] for _, pos, _, _ in tdoa_pairs])
            receivers_lon = np.array([pos[1] for _, pos, _, _ in tdoa_pairs])
            # Compute geometric derivatives (meters per radian)
            J_geom = jacobian_tdoa(receivers_lat, receivers_lon, lat, lon)
            return np.array(J_geom)

        # Initial parameter guess and bounds for TDOA approach (only position)
        x0 = [lat0, lon0]
        bounds = (
            [lat_min, lon_min],  # Lower bounds
            [lat_max, lon_max]  # Upper bounds
        )

        # Solve with robust method and better settings
        try:
            # Use a more efficient optimization strategy
            res = least_squares(
                tdoa_residual, x0, tdoa_jac, bounds=bounds,
                method='trf', loss='soft_l1',  # Robust against outliers
                f_scale=2.5,
                gtol=GTOL, xtol=XTOL,
                max_nfev=100,  # Limit number of function evaluations
                verbose=VERBOSE_OPTIMIZATION)
        except Exception as e:
            print(e)
            # Just skip problematic optimizations
            continue

        lat_f, lon_f = res.x

        # Now estimate the origin time given the final location
        travel_times = []
        for i, (pos, sec, name) in enumerate(detection_secs):
            slat, slon = pos
            tt, _, _ = compute_travel_time(lat_f, lon_f, slat, slon, month, travel_time_cache)
            # Origin time estimate based on this detection
            est_origin_time_offset = sec - tt
            travel_times.append((sec, tt, est_origin_time_offset, has_drift[i]))

        # Use median from stations without drift issues for better accuracy
        no_drift_offsets = [t[2] for t in travel_times if not t[3]]
        if no_drift_offsets:
            origin_time_offset = np.median(no_drift_offsets)
        else:
            # If all stations have drift, use median of all
            origin_time_offset = np.median([t[2] for t in travel_times])

        origin_time = ref_time + timedelta(seconds=origin_time_offset)

        # Calculate detailed residuals and prepare for clock drift estimation
        detailed_residuals = []
        tdoa_squared_errs = []
        abs_squared_errs = []

        # For clock drift estimation
        station_data = {}

        for i, (pos, sec, name) in enumerate(detection_secs):
            slat, slon = pos
            tt, terr, dist = compute_travel_time(lat_f, lon_f, slat, slon, month, travel_time_cache)

            # If this is the reference station, TDOA is 0 by definition
            is_reference = (i == ref_station_idx)

            # Predicted arrival time
            pred_arrival_sec = origin_time_offset + tt

            # Residual = observed - predicted (in seconds)
            residual_sec = sec - pred_arrival_sec
            abs_squared_errs.append(residual_sec ** 2)

            # For TDOA pairs, calculate the time difference residual
            if not is_reference:
                ref_pos, ref_sec, _ = detection_secs[ref_station_idx]
                ref_tt, _, _ = compute_travel_time(lat_f, lon_f, ref_pos[0], ref_pos[1], month, travel_time_cache)

                observed_tdoa = sec - ref_sec
                predicted_tdoa = tt - ref_tt
                tdoa_residual_val = observed_tdoa - predicted_tdoa
                tdoa_squared_errs.append(tdoa_residual_val ** 2)
            else:
                tdoa_residual_val = 0.0  # No TDOA for reference station

            # Calculate station weight based on distance (inverse weighting)
            station_weight = 1 / (dist + 1e-5)  # Avoid division by zero


            # Store data for drift estimation
            if has_drift[i]:
                if name not in station_data:
                    station_data[name] = {'observed': [], 'predicted': []}

                # Store absolute time for regression
                absolute_time_sec = (ref_time + timedelta(seconds=sec)).timestamp()
                predicted_abs_time_sec = (origin_time + timedelta(seconds=tt)).timestamp()

                station_data[name]['observed'].append(absolute_time_sec)
                station_data[name]['predicted'].append(predicted_abs_time_sec)
            origin_time_uncertainty = estimate_origin_time_uncertainty(
                res, lat_f, lon_f, month, travel_time_cache,
                station_positions, has_drift
            )

            detailed_residuals.append({
                'station_pos': pos,
                'station_name': name,
                'has_unknown_drift': has_drift[i],
                'is_reference_station': is_reference,
                'observed_time_sec': sec,
                'predicted_time_sec': pred_arrival_sec,
                'absolute_time': ref_time + timedelta(seconds=sec),
                'residual_sec': residual_sec,
                'tdoa_sec': None if is_reference else observed_tdoa,
                'predicted_tdoa_sec': None if is_reference else predicted_tdoa,
                'tdoa_residual_sec': tdoa_residual_val,
                'distance_m': dist,
                'travel_time_sec': tt,
                'estimated_error_sec': terr,
                'station_weight': station_weight,
                'detection_uncertainty': 2.0,
                'origin_time_uncertainty': origin_time_uncertainty if origin_time_uncertainty else 1.0,
            })

        # Standard RMS for both methods
        tdoa_rms = np.sqrt(np.mean(tdoa_squared_errs)) if len(tdoa_squared_errs) > 0 else None
        abs_rms = np.sqrt(np.mean(abs_squared_errs))

        # Store detailed optimization results
        optimization_details = {
            'res' : res,
            'x': res.x.tolist(),  # Final parameters
            'cost': res.cost,  # Final cost function value
            'fun': res.fun.tolist(),  # Final residuals
            'nfev': res.nfev,  # Number of function evaluations
            'njev': res.njev,  # Number of Jacobian evaluations
            'status': res.status,  # Termination status
            'message': res.message,  # Termination message
            'success': res.success# Boolean flag indicating success
        }

        validated.append({
            'station_positions': station_positions,
            'station_names': station_names,
            'detection_times': detection_times,
            'has_unknown_drift': has_drift,
            'reference_station_idx': ref_station_idx,
            'reference_station_name': ref_station_name,
            'source_point': (lat_f, lon_f),
            'origin_time': origin_time,
            'tdoa_rms_error': tdoa_rms,
            'absolute_rms_error': abs_rms,
            'num_stations': len(detection_secs),
            'optimization_success': res.success,
            'optimization_details': optimization_details,
            'detailed_residuals': detailed_residuals
        })

    # Clear the travel time cache to save memory before returning
    travel_time_cache.clear()
    return date, validated

def preload_isas_data():
    """Preload and cache all ISAS data"""
    for m in range(1, 13):
        get_isas_data(m)
    log_progress("All ISAS datasets preloaded and cached")

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import warnings
def enhanced_drift_analysis(df,name=None):
    """
    Perform enhanced drift analysis with weighted least squares regression
    and comprehensive residual diagnostics.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing drift analysis data

    Returns:
    --------
    dict : Dictionary with regression results and diagnostic information
    """
    # Set up a clean, modern plotting style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Custom color palette
    colors = {
        'data': '#3498db',      # Blue for data points
        'fit': '#e74c3c',       # Red for fit line
        'residuals': '#2ecc71', # Green for residuals
        'background': '#f9f9f9' # Light gray background
    }

    # Convert times to seconds (as float)
    x_abs = df['arrival_time_corr'].values.astype(np.float64) / 1e9
    y_abs = df['observed_arrival_time'].values.astype(np.float64) / 1e9
    e_sec = df['arrival_uncertainty'].values.astype(np.float64)  # uncertainty in seconds

    # Compute drift (observed - expected) in seconds
    drift = y_abs - x_abs

    # Create a relative time axis by subtracting the minimum expected arrival time
    t0 = {'ELAN' : '2018-01-16 17:34:34', 'NEAMS' : '2018-02-05 09:42:32' , 'WKER2': '2018-01-19 22:23:03' , 'MADW' :' 2018-01-06 14:43:32' , "SSEIR" : "2018-01-09 02:22:53", "SSWIR" :"2018-01-09 02:22:53", "SWAMSbot" :"2018-01-31 16:05:57"}
    try :
        x0 =  dt.datetime.strptime(t0[name],'%Y-%m-%d %H:%M:%S')
    except :
        warnings.warn('need T0 for this station')
        x0 = dt.datetime.strptime('2018-01-01 00:00:00','%Y-%m-%d %H:%M:%S')

    x0 = x0.timestamp()  # reference time (first expected arrival)
    x_rel = x_abs - x0  # now x_rel starts at 0

    # Use weights = 1/uncertainty for regression on drift
    weights = 1 / e_sec

    # Perform weighted regression: drift = slope * x_rel + offset
    (slope, offset), cov = np.polyfit(x_rel, drift, 1, w=weights, cov=True)
    var_slope, var_offset = np.diag(cov)
    sigma_slope = np.sqrt(var_slope)
    sigma_offset = np.sqrt(var_offset)

    # Compute drift in parts per million (ppm)
    drift_ppm = slope * 1e6
    ci_slope_ppm = 1.96 * sigma_slope * 1e6  # 95% CI for slope in ppm
    offset_us = offset * 1e6  # offset at x_rel=0 in microseconds

    # Compute fitted drift and residuals
    drift_fit = slope * x_rel + offset
    residuals = drift - drift_fit

    # Compute reduced chi-squared (χ²ᵣ)
    # Degrees of freedom = number of data points - number of fitted parameters
    dof = len(drift) - 2
    reduced_chi_sq = np.sum((residuals / e_sec)**2) / dof
    # Residual diagnostics
    # Compute standardized residuals
    std_residuals = residuals / np.sqrt(e_sec**2 + (sigma_slope * x_rel)**2)
    # Statistical test on residuals
    _, shapiro_p = stats.shapiro(std_residuals)


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4),
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.set_facecolor(colors['background'])

    # Top plot: Drift Data with Regression Line
    x_hours = x_rel / 3600.0
    drift_us = drift #* 1e6  # convert measured drift to microseconds
    drift_fit_us = drift_fit #* 1e6
    e_drift_us = e_sec #* 1e6  # uncertainty in microseconds

    # Plot data points with styled error bars
    ax1.errorbar(x_hours, drift_us, yerr=e_drift_us, fmt='o',
                 color=colors['data'], ecolor=colors['data'],
                 capsize=4, alpha=0.8, markersize=8,
                 label='Measured Drift (s)')

    # Add regression line with enhanced styling
    regression_line = ax1.plot(x_hours, drift_fit_us, '-',
                              color=colors['fit'], linewidth=2.5,
                              label=f'Regression: {drift_ppm:.3f} ppm')

    # Add shaded confidence interval
    x_plot = np.linspace(min(x_hours), max(x_hours), 100)
    y_fit = (slope * (x_plot * 3600) + offset) #* 1e6
    y_err = 1.96 * np.sqrt((x_plot * 3600 * sigma_slope)**2 + var_offset) #* 1e6
    ax1.fill_between(x_plot, y_fit - y_err, y_fit + y_err,
                     color=colors['fit'], alpha=0.2)

    # Add styled annotations for data points
    # if 'file_number' in df.columns :
    #     for i, fname in enumerate(df['file_number']):
    #         short_name = fname
    #         ax1.annotate(short_name, (x_hours[i], drift_us[i]),
    #                      xytext=(8, 0), textcoords="offset points",
    #                      fontsize=9, fontweight='bold',
    #                      bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7),
    #                      arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    # Enhance axis styling
    ax1.set_xlabel('Time from GPS Synchronization (hours)', fontsize=8)
    ax1.set_ylabel('Drift (observed - expected) in s', fontsize=8)
    # ax1.set_title(f"{name} Time Drift Analysis with 95% Confidence Interval",
    #              fontsize=16, fontweight='bold', pad=20)

    # Add a text box with statistics
    stats_text = (f"Drift: {drift_ppm:.3f} ppm\n"
                  f"95% CI: [{drift_ppm - ci_slope_ppm:.3f} to {drift_ppm + ci_slope_ppm:.3f}] ppm\n"
                  f"Offset: {offset:.3f} s")
    props = dict(boxstyle='square', facecolor='white', alpha=0.7)
    ax1.text(0.1, 0.97, stats_text, transform=ax1.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    # Enhanced legend
    ax1.legend(loc='best', frameon=True, fontsize=8)

    # Bottom plot: Residuals with enhanced styling
    # Color residuals based on their values (red for negative, blue for positive)
    residual_colors = ['#e74c3c' if r < 0 else '#3498db' for r in std_residuals]
    ax2.scatter(x_hours, std_residuals, c=residual_colors,
               s=60, alpha=0.7, edgecolor='gray')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5)

    # Add horizontal lines at ±1.96 standard deviations (95% confidence interval)
    ax2.axhline(y=1.96, color='gray', linestyle=':', linewidth=1.5)
    ax2.axhline(y=-1.96, color='gray', linestyle=':', linewidth=1.5)
    ax2.set_ylim([-4,4])
    # Style for residual subplot
    ax2.set_xlabel('Time from GPS Synchronization (hours)', fontsize=8)
    ax2.set_ylabel('Standardized Residuals', fontsize=8)
    # ax2.set_title('Residuals Analysis (Shapiro-Wilk p-value: {:.4f})'.format(shapiro_p),
    #              fontsize=12, fontweight='bold')

    # Add grid to both plots
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout and spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45)


    # Print detailed results
    print(f"\nWeighted Least Squares Regression Results {name}:")
    print(f"Drift slope: {slope:.12f} s/s, i.e. {drift_ppm:.3f} ppm")
    print(f"95% Confidence Interval for slope: [{drift_ppm - ci_slope_ppm:.3f}:{drift_ppm + ci_slope_ppm:.3f}]")
    print(f"Drift offset at reference time: {offset:.3f} s")

    # Style for residual subplot
    ax2.set_xlabel('Time from GPS Synchronization (hours)', fontsize=8)
    ax2.set_ylabel('Standardized Residuals')
    ax2.set_title(fr'Reduced $\chi^2$: {reduced_chi_sq:.2f}', fontsize=8, fontweight='bold', loc='left')

    # Save the figure with high resolution
    plt.savefig('drift_analysis_plot.png', dpi=300, bbox_inches='tight')

    plt.show()


    return {
        'slope': slope,
        'drift_ppm': drift_ppm,
        'offset': offset,
        'ci_slope_ppm': ci_slope_ppm,
        'shapiro_p': shapiro_p
    }
# --- Estimation de l'incertitude sur le temps d'origine via propagation d'erreur ---

def compute_travel_time_gradient(lat, lon, station_lat, station_lon, month, cache, epsilon=1e-4):
    """spatial gradient of travel time (∂t/∂lat, ∂t/∂lon)"""
    t0, _, _ = compute_travel_time(lat, lon, station_lat, station_lon, month, cache)
    t_lat, _, _ = compute_travel_time(lat + epsilon, lon, station_lat, station_lon, month, cache)
    t_lon, _, _ = compute_travel_time(lat, lon + epsilon, station_lat, station_lon, month, cache)

    dt_dlat = (t_lat - t0) / epsilon
    dt_dlon = (t_lon - t0) / epsilon
    return dt_dlat, dt_dlon

def estimate_origin_time_uncertainty(res, lat_f, lon_f, month, travel_time_cache, station_positions, has_drift_flags):
    try:
        J = res.jac
        cov = np.linalg.inv(J.T @ J)
        sigma_lat = np.sqrt(cov[0, 0])
        sigma_lon = np.sqrt(cov[1, 1])
    except Exception:
        print(f"WARNING: Jacobian and covariance are not compatible. ")
        return None
    good_stations = [pos for pos, drift in zip(station_positions, has_drift_flags) if not drift]
    if not good_stations:
        return None
    gradients = [
        compute_travel_time_gradient(lat_f, lon_f, sta_lat, sta_lon, month, travel_time_cache)
        for sta_lat, sta_lon in good_stations
    ]

    dt_dlat = np.mean([g[0] for g in gradients])
    dt_dlon = np.mean([g[1] for g in gradients])

    sigma_origin = np.sqrt((dt_dlat**2) * sigma_lat**2 + (dt_dlon**2) * sigma_lon**2)
    return sigma_origin


def main():
    """Main execution function with two-phase approach for clock drift estimation"""
    log_progress(f"Starting with {N_JOBS} workers")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Preload ISAS data before parallelization (creates cache files)
    if PRECOMPUTE_TRAVEL_TIMES:
        preload_isas_data()

    # Load input
    log_progress(f"Loading associations from {ASSO_FILE}")
    associations = np.load(ASSO_FILE, allow_pickle=True).item()
    items = list(associations.items())
    print("Nb associations chargées :", len(associations))

    total_items = len(items)
    log_progress(f"Found {total_items} date entries to process")

    #################################
    # PHASE 1: Initial localization #
    #################################
    log_progress("PHASE 1: Initial localization with increased uncertainty for unknown drift stations")

    # Process in batches with checkpoints - first pass
    validated_associations = {}

    for batch_start in range(0, total_items, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_items)
        batch = items[batch_start:batch_end]

        log_progress(f"Processing batch {batch_start + 1}-{batch_end} of {total_items}")

        # Process batch in parallel
        # Note: we use a smaller chunk_size when jobs > 1 for better load balancing
        effective_chunk = 1 if N_JOBS > 1 else CHUNK_SIZE

        results = Parallel(n_jobs=N_JOBS, verbose=5, batch_size=effective_chunk)(
            delayed(process_date)(date, lst) for date, lst in batch
        )

        # Store results
        for date, val in results:
            if val:  # Only store if we have validated results
                validated_associations[date] = val

        # Checkpoint
        chkpt_path = os.path.join(
            OUTPUT_DIR,
            f"{OUTPUT_BASENAME}_partial_{batch_end}.npy"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.save(chkpt_path, validated_associations)

        log_progress(f"Checkpoint saved: {chkpt_path}")

        # Memory optimization: save and reload larger checkpoints
        if len(validated_associations) > 5000:
            log_progress("Checkpointing and refreshing memory...")
            pickle_path = os.path.join(OUTPUT_DIR, "temp_checkpoint.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(validated_associations, f)

            # Clear and reload
            validated_associations.clear()
            with open(pickle_path, 'rb') as f:
                validated_associations = pickle.load(f)

            # Force garbage collection
            import gc
            gc.collect()

    # Save phase 1 results
    phase1_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_final.npy")
    np.save(phase1_path, validated_associations)
    log_progress(f"Phase 1 results saved to {phase1_path}")

    #######################################
    # PHASE 2: Global clock drift analysis #
    #######################################
    log_progress("PHASE 2: Analyzing clock drift across all events")

    # Collect all data for stations with unknown drift
    station_data_global = {name: {'observed': [], 'predicted': []} for name in UNKNOWN_DRIFT.keys()}

    for date, events in validated_associations.items():
        for event in events:
            if 'detailed_residuals' in event:
                origin_uncertainty = None
                if event.get("detailed_residuals"):
                    for residual in event["detailed_residuals"]:
                        station_name = residual.get("station_name")
                        if station_name in station_data_global:
                            station_data_global[station_name]['observed'].append(residual['absolute_time'].timestamp())
                            origin_time = event['origin_time'].timestamp()
                            predicted_time = origin_time + residual['travel_time_sec']

                            station_data_global[station_name]['predicted'].append(predicted_time)
                            station_data_global[station_name].setdefault('obs_error', []).append(
                                residual['detection_uncertainty'])
                            station_data_global[station_name].setdefault('origin_error', []).append(
                                residual['origin_time_uncertainty'])

    # Estimate drift parameters for each station
    log_progress("Estimating global clock drift parameters")
    global_drift_params = {}

    for station_name, data in station_data_global.items():
        if len(data['observed']) >= 10:  # Need sufficient data points
            log_progress(f"Estimating drift for {station_name} with {len(data['observed'])} observations")

            drift_factor, time_offset, rms_error = estimate_clock_drift(
                np.array(data['observed']),
                np.array(data['predicted']),
                station_name
            )

            if drift_factor is not None:
                drift_ppm = (drift_factor - 1) * 1e6  # Convert to PPM
                global_drift_params[station_name] = {
                    'drift_factor': drift_factor,
                    'time_offset': time_offset,
                    'rms_error': rms_error,
                    'drift_ppm': drift_ppm,
                    'num_data_points': len(data['observed'])
                }
                log_progress(
                    f"Station {station_name}: drift = {drift_ppm:.2f} ppm, offset = {time_offset:.3f} sec, RMS = {rms_error:.3f}")
            else:
                log_progress(f"Failed to estimate drift for {station_name}")

    # Save drift parameters
    drift_params_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_drift_params.npy")
    np.save(drift_params_path, global_drift_params)
    log_progress(f"Clock drift parameters saved to {drift_params_path}")

    # Visualisation individuelle de la dérive pour chaque station à dérive inconnue
    log_progress("Génération des graphiques de dérive individuels")

    import matplotlib.pyplot as plt

    def create_drift_dataframe(observed_times, predicted_times, obs_uncertainties, origin_time_uncertainties):
        total_uncertainty = np.sqrt(np.array(obs_uncertainties) ** 2 + np.array(origin_time_uncertainties) ** 2)
        return pd.DataFrame({
            'arrival_time_corr': pd.to_datetime(predicted_times, unit='s'),
            'observed_arrival_time': pd.to_datetime(observed_times, unit='s'),
            'arrival_uncertainty': total_uncertainty
        })

    visualization_dir = os.path.join(OUTPUT_DIR, "drift_plots")
    os.makedirs(visualization_dir, exist_ok=True)

    visual_results = {}

    for station_name, data in station_data_global.items():
        if len(data['observed']) >= 10:
            df_drift = create_drift_dataframe(
                data['observed'],
                data['predicted'],
                obs_uncertainties=data.get('obs_error'),
                origin_time_uncertainties=data.get('origin_error')
            )

            # Remplace le nom de fichier dans enhanced_drift_analysis
            try:
                result = enhanced_drift_analysis(df_drift, name=station_name)
                visual_results[station_name] = result

                # Enregistrer la figure manuellement avec un nom unique
                fig_filename = os.path.join(visualization_dir, f"{station_name}_drift_analysis.png")
                plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
                plt.close()
                log_progress(f"Figure enregistrée pour {station_name}: {fig_filename}")
            except Exception as e:
                log_progress(f"Échec de l'analyse pour {station_name}: {e}")

    # Résumé des résultats numériques
    summary_path = os.path.join(OUTPUT_DIR, "drift_analysis_summary.csv")
    summary_df = pd.DataFrame.from_dict(visual_results, orient='index')
    summary_df.to_csv(summary_path)
    log_progress(f"Résumé des résultats enregistré: {summary_path}")

    elapsed = time.time() - start_time
    log_progress(f"Total execution time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")


if __name__ == "__main__":
    main()
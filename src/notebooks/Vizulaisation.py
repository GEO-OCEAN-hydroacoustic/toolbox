#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.optimize import least_squares
from scipy import stats
import matplotlib.pyplot as plt
import time
import datetime as dt
from collections import defaultdict
import networkx as nx
from tqdm import tqdm

# === CONFIGURATION ===
INPUT_FILE = "/media/rsafran/CORSAIR/Association/validated/refined_s_-60-5,35-120,350,0.8,0.6_final.npy"
OUTPUT_DIR = "/media/rsafran/CORSAIR/Association/validated/processed"
OUTPUT_BASENAME = "refined_drift_analysis"

# Parameters
MAX_DRIFT_PPM = 2  # Maximum clock drift in parts per million
MIN_EVENTS_FOR_DRIFT = 10  # Minimum number of events required for drift estimation
OVERLAP_THRESHOLD = 1  # Minimum number of shared picks to consider events as overlapping
MIN_PICKS_FOR_EVENT = 3  # Minimum number of picks required for an event
MAX_ORIGIN_UNCERTAINTY = 5  # Maximum uncertainty for origin time (seconds)
MAX_NORMALIZED_RMS = 0.95  # Maximum normalized RMS error
N_JOBS = os.cpu_count() - 1 if os.cpu_count() > 1 else 1  # Leave one core free

# Stations with unknown drift
UNKNOWN_DRIFT = {
    "ELAN": True, "MADE": True, "MADW": True, "NEAMS": True, "RTJ": True,
    "SSEIR": True, "SSWIR": True, "SWAMSbot": True, "WKER2": True
}
UNKNOWN_DRIFT = {name : True if UNKNOWN_DRIFT[name] else False for name in UNKNOWN_DRIFT  }

# Station reference times (for relative time calculations)
STATION_REF_TIMES = {
    'ELAN': '2018-01-16 17:34:34',
    'NEAMS': '2018-02-05 09:42:32',
    'WKER2': '2018-01-19 22:23:03',
    'MADW': '2018-01-06 14:43:32',
    "SSEIR": "2018-01-09 02:22:53",
    "SSWIR": "2018-01-09 02:22:53",
    "SWAMSbot": "2018-01-31 16:05:57",
    "RTJ": "2018-01-01 00:00:00"  # Default if specific time not available
}

# === PERFORMANCE MONITORING ===
start_time = time.time()


def log_progress(message):
    """Log progress with elapsed time"""
    elapsed = time.time() - start_time
    print(f"[{elapsed:.1f}s] {message}")


# === EVENT PROCESSING FUNCTIONS ===

def event_score(event):
    """Calculate event score - lower is better"""
    num_picks = len(event.get('detection_times', []))
    cost = event.get('cost', float('inf'))

    try:
        uncertainty = event.get('uncertainty',
                                event.get('detailed_residuals', [{}])[0].get('origin_time_uncertainty', 1.0))
    except:
        uncertainty = 1.0  # fallback

    # Lower score is better
    return cost / (num_picks + 1e-6) + uncertainty


def process_events(validated_data):
    """
    Process events: flatten, filter, remove subsets, and select best events
    Returns a dictionary with the best events
    """
    # Step 1: Flatten and filter events
    flat_events = flatten_events(validated_data)

    # Step 2: Remove subset events
    unique_events = remove_subset_events(flat_events)

    # Step 3: Build overlap graph and select best events
    best_events = find_best_events(unique_events)

    # Convert back to date-keyed dictionary format
    result = {}
    for event in best_events:
        result[event['key']] = event['original_event']

    return result


def flatten_events(validated_data):
    """Flatten nested event data structure with initial filtering"""
    log_progress("Flattening events...")
    flat_events = []

    for key, associations in tqdm(validated_data.items(), desc="Flattening events"):
        for i, event in enumerate(associations):
            # Apply basic filtering criteria
            if not event.get('optimization_success', False):
                continue

            num_stations = event.get('num_stations', 0)
            if num_stations < MIN_PICKS_FOR_EVENT:
                continue

            rms = event.get('absolute_rms_error', float('inf'))
            normalized_rms = rms / num_stations if num_stations > 0 else float('inf')

            if normalized_rms > MAX_NORMALIZED_RMS:
                continue

            try:
                if event.get('detailed_residuals')[0].get('origin_time_uncertainty',
                                                          float('inf')) > MAX_ORIGIN_UNCERTAINTY:
                    continue
            except:
                continue

            # Create flattened event with necessary data
            det_times = sorted(set(event.get('detection_times', [])))
            flat_events.append({
                'uid': f"{key}_{i}",  # unique ID per event instance
                'key': key,  # original date/time key
                'source_point': event.get('source_point'),
                'detection_times': det_times,
                'cost': event.get('optimization_details', {}).get('res', {}).get('cost', float('inf')),
                'uncertainty': event.get('detailed_residuals', [{}])[0].get('origin_time_uncertainty', 1.0),
                'normalized_rms': normalized_rms,
                'num_stations': num_stations,
                'origin_time': event.get('origin_time'),
                'original_event': event
            })

    log_progress(f"Total flattened events: {len(flat_events)}")
    return flat_events


def remove_subset_events(flat_events):
    """Efficiently remove events that are subsets of other events"""
    log_progress("Removing subset events...")

    # Sort by length of detection_times (descending), then by first detection time
    flat_events.sort(key=lambda x: (-len(x['detection_times']),
                                    x['detection_times'][0] if x['detection_times'] else float('inf')))

    # Create a lookup by length for faster filtering
    events_by_length = defaultdict(list)
    for event in flat_events:
        events_by_length[len(event['detection_times'])].append(event)

    lengths = sorted(events_by_length.keys(), reverse=True)

    unique_events = []
    excluded_uids = set()

    # Process events from longest to shortest
    for length in tqdm(lengths, desc="Processing by length"):
        for current in events_by_length[length]:
            if current['uid'] in excluded_uids:
                continue

            current_set = frozenset(current['detection_times'])
            is_subset = False

            # Only compare with longer events (which are already in unique_events)
            for prev in unique_events:
                if len(prev['detection_times']) < len(current_set):
                    # Can't be a subset if prev has fewer elements
                    continue

                if current_set.issubset(prev['detection_times']):
                    excluded_uids.add(current['uid'])
                    is_subset = True
                    break

            if not is_subset:
                unique_events.append(current)

    log_progress(f"Subset events removed: {len(excluded_uids)}")
    log_progress(f"Events after subset removal: {len(unique_events)}")
    return unique_events


def find_best_events(unique_events):
    """Build overlap graph and select best events from each cluster"""
    # Build inverted index: detection_time -> list of events
    time_to_events = defaultdict(set)
    uid_to_event = {e['uid']: e for e in unique_events}

    # Populate the inverted index
    for event in tqdm(unique_events, desc="Building inverted index"):
        for time in event['detection_times']:
            time_to_events[time].add(event['uid'])

    # Initialize graph
    G = nx.Graph()
    G.add_nodes_from([e['uid'] for e in unique_events])

    # Create edges from time_to_events
    edges = set()
    for events_with_time in tqdm(time_to_events.values(), desc="Finding overlaps"):
        if len(events_with_time) > 1:  # Only consider times shared by multiple events
            for uid1 in events_with_time:
                for uid2 in events_with_time:
                    if uid1 < uid2:  # Ensure we don't add duplicate edges
                        edges.add((uid1, uid2))

    G.add_edges_from(edges)
    log_progress(f"Found {G.number_of_edges()} overlapping event pairs")

    # Find connected components (clusters of overlapping events)
    clusters = list(nx.connected_components(G))
    log_progress(f"Overlapping clusters found: {len(clusters)}")

    # Select best event in each cluster
    deduplicated_events = []
    for cluster in tqdm(clusters, desc="Selecting best events"):
        best_uid = min(cluster, key=lambda uid: event_score(uid_to_event[uid]))
        deduplicated_events.append(uid_to_event[best_uid])

    # Add non-overlapping events
    clustered_uids = set().union(*clusters) if clusters else set()
    non_clustered_uids = set(uid_to_event.keys()) - clustered_uids

    for uid in non_clustered_uids:
        deduplicated_events.append(uid_to_event[uid])

    log_progress(f"Final deduplicated event count: {len(deduplicated_events)}")
    return deduplicated_events


# === DRIFT ANALYSIS FUNCTIONS ===

def extract_drift_data(events_data):
    """
    Extract drift data from events for analysis
    Returns a dictionary with data for each station with unknown drift
    """
    station_data = {name: {'observed': [], 'predicted': [], 'uncertainty': [], 'origin_uncertainty': []}
                    for name in UNKNOWN_DRIFT.keys()}

    # Process each event
    for event_data in events_data.values():
        if 'detailed_residuals' not in event_data:
            continue

        origin_time = event_data.get('origin_time')

        for residual in event_data['detailed_residuals']:
            station_name = residual.get('station_name')

            # Skip if station not in our unknown drift list
            if station_name not in UNKNOWN_DRIFT:
                continue

            # Get observed arrival time
            observed_time = residual.get('absolute_time')
            if observed_time is None:
                continue

            # Calculate predicted arrival time
            travel_time = residual.get('travel_time_sec', 0)
            predicted_time = origin_time + timedelta(seconds=travel_time)

            # Store data
            station_data[station_name]['observed'].append(observed_time.timestamp())
            station_data[station_name]['predicted'].append(predicted_time.timestamp())
            station_data[station_name]['uncertainty'].append(residual.get('detection_uncertainty', 2.0))
            station_data[station_name]['origin_uncertainty'].append(residual.get('origin_time_uncertainty', 1.0))

    # Remove stations with insufficient data
    to_remove = [name for name, data in station_data.items()
                 if len(data['observed']) < MIN_EVENTS_FOR_DRIFT]

    for name in to_remove:
        del station_data[name]

    return station_data


def create_drift_dataframe(observed_times, predicted_times, uncertainties, origin_uncertainties):
    """Create a DataFrame for drift analysis"""
    # Combine uncertainties (add in quadrature)
    total_uncertainty = np.sqrt(np.array(origin_uncertainties) ** 2)

    return pd.DataFrame({
        'arrival_time_corr': pd.to_datetime(predicted_times, unit='s'),
        'observed_arrival_time': pd.to_datetime(observed_times, unit='s'),
        'arrival_uncertainty': total_uncertainty
    })


def zero_intercept_residuals(slope_param, x, y, weights):
    """Compute weighted residuals for zero-intercept line fit"""
    slope = slope_param[0]
    residuals = (y - slope * x)
    return residuals * weights/weights


def analyze_drift(df, station_name):
    """Perform drift analysis with weighted least squares regression"""
    # Convert times to seconds (as float)
    x_abs = df['arrival_time_corr'].values.astype(np.float64) / 1e9
    y_abs = df['observed_arrival_time'].values.astype(np.float64) / 1e9
    e_sec = df['arrival_uncertainty'].values.astype(np.float64)  # uncertainty in seconds

    # Compute drift (observed - expected) in seconds
    drift = y_abs - x_abs

    # Create a relative time axis by subtracting the reference time
    try:
        x0 = dt.datetime.strptime(STATION_REF_TIMES.get(station_name, '2018-01-01 00:00:00'),
                                  '%Y-%m-%d %H:%M:%S').timestamp()
    except:
        x0 = dt.datetime.strptime('2018-01-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp()

    x_rel = x_abs - x0  # relative time

    # Use weights = 1/uncertainty for regression on drift
    weights = e_sec / e_sec

    if True :
        # Perform zero-intercept least squares regression
        initial_slope = np.sum(weights * x_rel * drift) / np.sum(weights * x_rel ** 2)
        result = least_squares(
            zero_intercept_residuals,
            [initial_slope],
            args=(x_rel, drift, weights),
            method='lm'
        )

        # Extract the slope
        slope = result.x[0]

        # Compute variance and confidence intervals
        residuals = zero_intercept_residuals(result.x, x_rel, drift, weights)
        dof = len(drift) - 1  # degrees of freedom (only slope parameter)
        variance_estimate = np.sum(residuals ** 2) / dof
        slope_variance = variance_estimate / np.sum(weights * x_rel ** 2)
        sigma_slope = np.sqrt(slope_variance)

        # Compute drift in parts per million (ppm)
        drift_ppm = slope * 1e6
        ci_slope_ppm = 1.96 * sigma_slope * 1e6  # 95% CI for slope in ppm

        # Compute fitted drift and residuals
        drift_fit = slope * x_rel
        residuals = drift - drift_fit

        # Compute reduced chi-squared
        reduced_chi_sq = np.sum((residuals / e_sec) ** 2) / dof

    else :
        # Perform weighted regression: drift = slope * x_rel + offset
        (slope, offset), cov = np.polyfit(x_rel, drift, 1, w=None, cov=True)
        var_slope, var_offset = np.diag(cov)
        sigma_slope = np.sqrt(var_slope)
        sigma_offset = np.sqrt(var_offset)
        # Compute drift in parts per million (ppm)
        drift_ppm = slope * 1e6
        ci_slope_ppm = 1.96 * sigma_slope * 1e6  # 95% CI for slope in ppm
        ci_offset_s = 1.96 * sigma_offset
        offset_us = offset * 1e6  # offset at x_rel=0 in microseconds

        # Compute fitted drift and residuals
        drift_fit = slope * x_rel + offset
        residuals = drift - drift_fit

        # Compute reduced chi-squared (χ²ᵣ)
        # Degrees of freedom = number of data points - number of fitted parameters
        dof = len(drift) - 2
        reduced_chi_sq = np.sum((residuals / e_sec) ** 2) / dof
        # Residual diagnostics
        # Compute standardized residuals
        std_residuals = residuals / np.sqrt(e_sec ** 2 + (sigma_slope * x_rel) ** 2)

        # Statistical test on residuals
        _, shapiro_p = stats.shapiro(std_residuals)



    # Create the drift analysis plot
    fig = create_drift_plot(station_name, x_rel, drift, e_sec, drift_fit,
                            drift_ppm, ci_slope_ppm, reduced_chi_sq, residuals, sigma_slope)

    return {
        'station_name': station_name,
        'slope': slope,
        'drift_ppm': drift_ppm,
        'offset': 0.0,  # Always zero
        'ci_slope_ppm': ci_slope_ppm,
        'reduced_chi_sq': reduced_chi_sq,
        'num_data_points': len(drift),
        'figure': fig
    }


def create_drift_plot(station_name, x_rel, drift, e_sec, drift_fit,
                      drift_ppm, ci_slope_ppm, reduced_chi_sq, residuals, sigma_slope):
    """Create visualization for drift analysis"""
    # Set up plotting style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Custom color palette
    colors = {
        'data': '#3498db',  # Blue for data points
        'fit': '#e74c3c',  # Red for fit line
        'residuals': '#2ecc71',  # Green for residuals
        'background': '#f9f9f9'  # Light gray background
    }

    # Create the drift analysis plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.set_facecolor(colors['background'])

    # Top plot: Drift Data with Regression Line
    x_hours = x_rel / 3600.0  # Convert to hours

    # Plot data points with error bars
    ax1.errorbar(x_hours, drift, yerr=e_sec, fmt='o',
                 color=colors['data'], ecolor=colors['data'],
                 capsize=4, alpha=0.8, markersize=6,
                 label='Measured Drift (s)')

    # Add regression line
    ax1.plot(x_hours, drift_fit, '-',
             color=colors['fit'], linewidth=2.5,
             label=f'Regression: {drift_ppm:.3f} ppm (zero-intercept)')

    # Add shaded confidence interval
    x_plot = np.linspace(min(x_hours), max(x_hours), 100)
    y_fit = drift_ppm / 1e6 * (x_plot * 3600)
    y_err = 1.96 * np.sqrt((x_plot * 3600 * sigma_slope) ** 2)
    ax1.fill_between(x_plot, y_fit - y_err, y_fit + y_err,
                     color=colors['fit'], alpha=0.2)

    # Enhance axis styling
    ax1.set_xlabel('Time from Reference (hours)', fontsize=10)
    ax1.set_ylabel('Drift (observed - expected) in s', fontsize=10)
    ax1.set_title(f"{station_name} Time Drift Analysis (Zero-Intercept)", fontsize=14, fontweight='bold')

    # Add a text box with statistics
    stats_text = (f"Drift: {drift_ppm:.3f} ppm\n"
                  f"95% CI: [{drift_ppm - ci_slope_ppm:.3f} to {drift_ppm + ci_slope_ppm:.3f}] ppm\n"
                  f"Zero intercept enforced\n"
                  f"Reduced χ²: {reduced_chi_sq:.2f}")
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    # Add legend
    ax1.legend(loc='best', frameon=True, fontsize=9)

    # Bottom plot: Residuals
    std_residuals = residuals / np.sqrt(e_sec ** 2 + (sigma_slope * x_rel) ** 2)
    residual_colors = ['#e74c3c' if r < 0 else '#3498db' for r in std_residuals]
    ax2.scatter(x_hours, std_residuals, c=residual_colors,
                s=40, alpha=0.7, edgecolor='gray')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5)

    # Add horizontal lines at ±1.96 standard deviations (95% confidence interval)
    ax2.axhline(y=1.96, color='gray', linestyle=':', linewidth=1.0)
    ax2.axhline(y=-1.96, color='gray', linestyle=':', linewidth=1.0)
    ax2.set_ylim([-4, 4])

    # Style for residual subplot
    ax2.set_xlabel('Time from Reference (hours)', fontsize=10)
    ax2.set_ylabel('Standardized\nResiduals', fontsize=10)

    # Add grid to both plots
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    return fig

def analyze_station_drifts(station_data, output_dir):
    """Analyze drift for all stations and create visualizations"""
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "drift_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Results dictionary
    drift_results = {}

    for station_name, data in station_data.items():
        if len(data['observed']) < MIN_EVENTS_FOR_DRIFT:
            continue
        # Create DataFrame for this station
        df = create_drift_dataframe(
            data['observed'],
            data['predicted'],
            data['uncertainty'],
            data['origin_uncertainty']
        )

        # Analyze drift
        try:
            result = analyze_drift(df, station_name)

            # Store results
            drift_results[station_name] = {
                'drift_ppm': result['drift_ppm'],
                'drift_factor': 1 + (result['drift_ppm'] / 1e6),
                'offset': result['offset'],  # Will always be 0
                'ci_low_ppm': result['drift_ppm'] - result['ci_slope_ppm'],
                'ci_high_ppm': result['drift_ppm'] + result['ci_slope_ppm'],
                'reduced_chi_sq': result['reduced_chi_sq'],
                'num_data_points': result['num_data_points']
            }

            # Save plot
            fig_path = os.path.join(plots_dir, f"{station_name}_drift_analysis.png")
            result['figure'].savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(result['figure'])

            print(f"Station {station_name}: drift = {result['drift_ppm']:.3f} ppm, "
                  f"samples = {result['num_data_points']}")

        except Exception as e:
            print(f"Error analyzing drift for station {station_name}: {e}")

    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame.from_dict(drift_results, orient='index')
    summary_path = os.path.join(output_dir, "drift_analysis_summary.csv")
    summary_df.to_csv(summary_path)

    return drift_results

def visualize_results(best_events, output_dir):
    """Create visualization of event statistics"""
    log_progress("Visualizing event statistics...")
    rms = []
    pos = []
    ori_err = []

    for date, event in best_events.items():
        if 'normalized_rms' not in event:
            # Calculate normalized RMS if not present
            num_stations = event.get('num_stations', 0)
            rms_value = event.get('absolute_rms_error', 0) / max(num_stations, 1)
            event['normalized_rms'] = rms_value

        rms.append(event.get('normalized_rms'))
        pos.append(event.get('source_point'))
        ori_err.append(event.get('detailed_residuals')[0]["origin_time_uncertainty"])

    # RMS histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rms, bins=100)
    plt.title('Distribution of Normalized RMS Values')
    plt.xlabel('Normalized RMS')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'normalized_rms_histogram.png'), dpi=300)
    plt.close()

    # Event location map
    plt.figure(figsize=(12, 10))
    pos = np.array(pos)
    plt.scatter(pos[:, 1], pos[:, 0], c=ori_err, cmap='viridis')
    plt.colorbar(label='Origin Time Uncertainty (s)')
    plt.title('Event Locations Colored by Origin Time Uncertainty')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(output_dir, 'event_locations_map.png'), dpi=300)
    plt.close()


def main():
    """Main execution function"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load validation data
    log_progress(f"Loading data from {INPUT_FILE}")
    validated_data = np.load(INPUT_FILE, allow_pickle=True).item()

    # Step 1: Process and select best events
    log_progress("Starting event processing...")
    best_events = process_events(validated_data)
    log_progress(f"Event processing complete. Selected {len(best_events)} events.")

    # Visualize results
    visualize_results(best_events, OUTPUT_DIR)

    # Save the filtered events
    filtered_events_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_filtered_events.npy")
    np.save(filtered_events_path, best_events)
    log_progress(f"Filtered events saved to {filtered_events_path}")

    # Step 2: Extract drift data from filtered events
    log_progress("Extracting drift data from events")
    station_data = extract_drift_data(best_events)

    # Step 3: Analyze drift for each station
    log_progress("Analyzing station drifts")
    drift_results = analyze_station_drifts(station_data, OUTPUT_DIR)

    # Save drift results
    drift_results_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_drift_results.npy")
    np.save(drift_results_path, drift_results)
    log_progress(f"Drift results saved to {drift_results_path}")

    # Print summary of results
    log_progress("\nDrift Analysis Summary:")
    for station, result in drift_results.items():
        log_progress(f"{station}: {result['drift_ppm']:.3f} ppm ± "
                     f"{result['ci_high_ppm'] - result['drift_ppm']:.3f} ppm")

    elapsed = time.time() - start_time
    log_progress(f"Total execution time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")

if __name__ == "__main__":
    plt.close('all')
    main()
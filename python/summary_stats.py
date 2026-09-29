# VENDORED COPY -- do not edit here.
# Source: /home/karinako/projects/DO_emulation_calibration/summary_stats.py
# Commit: 0476046 (2026-05-26), copied unchanged on 2026-09-29.
# Used by the Julia EKS driver (via PythonCall) so that EKS and the emulator+MCMC
# approach share one observation operator. If the original changes, recopy it and
# rerun python/export_python_calibration_setup.py.

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import welch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xarray as xr

# =============================================================================
# Helper functions for DO event detection (matching Julia implementation)
# =============================================================================

def detect_stadials_adaptive(amoc_smooth, method='bimodal_gap', offset=2.5):
    """Detect threshold for stadial/interstadial classification."""
    if method == 'percentile':
        return np.percentile(amoc_smooth, 25)

    elif method == 'bimodal_gap':
        hist, edges = np.histogram(amoc_smooth, bins=50, density=True)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        mid_range = (len(hist) // 4, 3 * len(hist) // 4)
        local_min_idx = mid_range[0] + np.argmin(hist[mid_range[0]:mid_range[1]])
        return bin_centers[local_min_idx]

    elif method == 'clustering':
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(amoc_smooth.reshape(-1, 1))
        centers = kmeans.cluster_centers_.flatten()
        low, high = np.min(centers), np.max(centers)
        threshold = low + offset
        if threshold > high:
            threshold = np.mean(centers)
        return threshold

    else:
        raise ValueError(f"Unknown method: {method}")


def find_peaks_positive(signal):
    """Find local maxima in signal."""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)
    return np.array(peaks)


def filter_peaks_by_spacing(peak_indices, values, time, min_spacing):
    """Keep only the strongest peak within each min_spacing window."""
    if len(peak_indices) == 0:
        return peak_indices
    
    filtered = []
    i = 0
    
    while i < len(peak_indices):
        current_idx = peak_indices[i]
        current_time = time[current_idx]
        
        cluster = [current_idx]
        j = i + 1
        
        while j < len(peak_indices):
            if time[peak_indices[j]] - current_time < min_spacing:
                cluster.append(peak_indices[j])
                j += 1
            else:
                break
        
        cluster_values = values[cluster]
        best_idx = cluster[np.argmax(cluster_values)]
        filtered.append(best_idx)
        
        i = j
    
    return np.array(filtered)


def find_crossing_before_peak(residual, time, peak_idx, crossing_value):
    """Find last time residual crossed above crossing_value before peak."""
    for i in range(peak_idx, 1, -1):
        if residual[i] >= crossing_value and residual[i-1] < crossing_value:
            return i
    return None


def detect_do_events_simple(amoc, time, span=0.02, min_spacing=500, crossing_value=5.0,
                             detection_mode='peak_walkback', pre_smooth_win=0):
    """
    Detect DO events from smoothed residual.

    Parameters:
    -----------
    amoc : array
        AMOC time series
    time : array
        Time array
    span : float
        Smoothing span parameter (fraction of data for window size)
    min_spacing : float
        Minimum spacing between DO events (in time units)
    crossing_value : float
        Threshold for residual to identify significant peaks
    detection_mode : str
        'peak_walkback'   — find positive peaks above crossing_value, walk back to
                           threshold crossing.  Works well with positive thresholds.
        'upward_crossing' — find every upward crossing of crossing_value directly,
                           then apply spacing filter.  Works for any sign of
                           crossing_value, including negative (adaptive) values.
    pre_smooth_win : int
        Window size (years) for optional pre-smoothing applied before the LOESS
        residual step. Removes high-frequency noise so the residual is cleaner.
        0 = disabled.

    Returns:
    --------
    do_event_indices : array
        Indices of detected DO events
    do_times : array
        Times of detected DO events
    do_waiting_times : array
        Waiting times between consecutive DO events
    """
    from scipy.ndimage import uniform_filter1d

    amoc = amoc.astype(float)
    if pre_smooth_win > 1:
        amoc = uniform_filter1d(amoc, size=pre_smooth_win, mode='nearest')

    window_size = max(3, int(len(amoc) * span))
    if window_size % 2 == 0:
        window_size += 1

    smoothed = uniform_filter1d(amoc, size=window_size, mode='nearest')
    residual = amoc - smoothed

    if detection_mode == 'upward_crossing':
        # Every point where residual rises from below crossing_value to above it.
        # Spacing filter: keep the first crossing in each cluster.
        all_crossings = [i for i in range(1, len(residual))
                         if residual[i - 1] < crossing_value and residual[i] >= crossing_value]
        do_event_indices = []
        for idx in all_crossings:
            if not do_event_indices or time[idx] - time[do_event_indices[-1]] >= min_spacing:
                do_event_indices.append(idx)

    elif detection_mode == 'peak_walkback':
        all_peaks = find_peaks_positive(residual)
        significant_peaks = np.array([p for p in all_peaks if residual[p] > crossing_value])

        if len(significant_peaks) > 0:
            filtered_peaks = filter_peaks_by_spacing(significant_peaks, residual, time, min_spacing)
        else:
            filtered_peaks = np.array([], dtype=int)

        do_event_indices = []
        for peak_idx in filtered_peaks:
            crossing_idx = find_crossing_before_peak(residual, time, peak_idx, crossing_value)
            if crossing_idx is not None:
                crossing_time = time[crossing_idx]
                is_far_enough = all(
                    abs(crossing_time - time[prev_idx]) >= min_spacing
                    for prev_idx in do_event_indices
                )
                if is_far_enough:
                    do_event_indices.append(crossing_idx)
    else:
        raise ValueError(f"Unknown detection_mode: {detection_mode!r}. "
                         "Use 'peak_walkback' or 'upward_crossing'.")

    do_event_indices = np.array(do_event_indices, dtype=int)
    do_times = time[do_event_indices] if len(do_event_indices) > 0 else np.array([])
    do_waiting_times = np.diff(do_times) if len(do_times) > 1 else np.array([])

    return do_event_indices, do_times, do_waiting_times


# =============================================================================
# Main summary statistics function (updated to match Julia version)
# =============================================================================

def first_peak_location(pdf_vals, x_grid, prominence_frac=0.05):
    """
    Return the x-value (Sv) of the leftmost prominent peak in a KDE PDF.
    Falls back to argmax if no prominent peaks are found.

    Used to distinguish DO runs (stadial peak at ~10 Sv) from wild
    oscillators (first peak at 14+ Sv, no low-AMOC state).
    """
    from scipy.signal import find_peaks
    prominence = prominence_frac * pdf_vals.max()
    peaks, _ = find_peaks(pdf_vals, prominence=prominence)
    if len(peaks) == 0:
        return x_grid[np.argmax(pdf_vals)]
    return x_grid[peaks[0]]


def compute_summary_stats(amoc_data, time_data=None, remove_spinup=True, spinup_fraction=0.02,
                          smooth_win=0, adaptive_threshold=True, threshold_method='clustering',
                          threshold=None, x_grid=None, grid_points=100, quantiles=[0.1, 0.5, 0.9],
                          ignore_first_stadial=True, verbose=False,
                          # DO event detection parameters (matching Julia defaults)
                          loess_span=0.02, do_min_spacing=600, do_crossing_value=5.0,
                          detection_mode='peak_walkback', pre_smooth_win=0,
                          # DO variability classification
                          do_peak_threshold=14.0, pdf_prominence=0.05):
    """
    Compute summary statistics from AMOC time series.
    
    Parameters:
    -----------
    amoc_data : array
        AMOC time series data
    time_data : array, optional
        Time array (default: indices)
    remove_spinup : bool
        Whether to remove initial spinup period
    spinup_fraction : float
        Fraction of data to remove as spinup
    smooth_win : int
        Smoothing window size (0 = no smoothing)
    adaptive_threshold : bool
        Use adaptive threshold detection
    threshold_method : str
        Method for threshold detection ('percentile', 'bimodal_gap', 'clustering')
    threshold : float, optional
        Manual threshold value
    x_grid : array, optional
        Fixed evaluation grid for KDE PDF. When provided, all runs are
        evaluated on the same support so that PCA/comparison is consistent.
        If None, a per-run grid spanning [data.min(), data.max()] is used.
    grid_points : int
        Number of points for PDF estimation (used only when x_grid is None)
    quantiles : list
        Quantiles to compute
    ignore_first_stadial : bool
        Whether to ignore first stadial in analysis
    verbose : bool
        Print results
    loess_span : float
        LOESS span for DO event detection
    do_min_spacing : float
        Minimum spacing between DO events
    do_crossing_value : float
        Crossing threshold for DO event detection
    
    Returns:
    --------
    stats : dict
        Dictionary containing all computed statistics
    """
    amoc_data = np.asarray(amoc_data)
    if time_data is None:
        time_data = np.arange(len(amoc_data))
    else:
        time_data = np.asarray(time_data)

    # Remove spinup
    if remove_spinup:
        start_idx = int(len(amoc_data) * spinup_fraction)
        amoc_data = amoc_data[start_idx:]
        time_data = time_data[start_idx:]
        time_data = time_data - time_data[0]

    # Basic stats
    basic_stats = {
        'mean': np.mean(amoc_data),
        'std': np.std(amoc_data),
        'quantiles': np.quantile(amoc_data, quantiles)
    }

    # Smooth
    if smooth_win > 1:
        kernel = np.ones(smooth_win) / smooth_win
        amoc_smooth = np.convolve(amoc_data, kernel, mode='same')
    else:
        amoc_smooth = amoc_data.copy()

    # PDF using KDE — skip runs where AMOC is constant (e.g. collapsed circulation)
    if np.std(amoc_data) == 0:
        return None
    try:
        kde = gaussian_kde(amoc_data)
    except np.linalg.LinAlgError:
        return None
    if x_grid is None:
        x_grid = np.linspace(amoc_data.min(), amoc_data.max(), grid_points)
    pdf_vals = kde(x_grid)
    pdf_vals /= np.trapezoid(pdf_vals, x_grid)  # normalize

    # Threshold
    if adaptive_threshold:
        threshold_val = detect_stadials_adaptive(amoc_smooth, method=threshold_method)
    else:
        threshold_val = np.percentile(amoc_smooth, 25) if threshold is None else threshold

    # Detect stadials
    is_stadial = amoc_smooth < threshold_val
    transitions = np.diff(is_stadial.astype(int))
    stadial_starts = np.where(transitions == 1)[0] + 1
    stadial_ends = np.where(transitions == -1)[0] + 1

    if is_stadial[0] and not ignore_first_stadial:
        stadial_starts = np.insert(stadial_starts, 0, 0)
    if is_stadial[-1]:
        stadial_ends = np.append(stadial_ends, len(amoc_smooth))

    # Robust alignment
    if ignore_first_stadial and len(stadial_ends) > len(stadial_starts):
        if len(stadial_starts) == 0 or stadial_ends[0] < stadial_starts[0]:
            stadial_ends = stadial_ends[1:]
    if len(stadial_starts) > len(stadial_ends):
        stadial_starts = stadial_starts[:len(stadial_ends)]
    if len(stadial_starts) != len(stadial_ends):
        min_len = min(len(stadial_starts), len(stadial_ends))
        stadial_starts = stadial_starts[:min_len]
        stadial_ends = stadial_ends[:min_len]

    n_stadials = len(stadial_starts)

    # Stadial durations
    if n_stadials > 0:
        stadial_durations = time_data[stadial_ends - 1] - time_data[stadial_starts]
        avg_stadial_duration = np.mean(stadial_durations)
    else:
        stadial_durations = np.array([])
        avg_stadial_duration = 0.0

    # *** NEW: Detect DO events from residual and compute waiting times ***
    do_event_indices, do_times, do_waiting_times = detect_do_events_simple(
        amoc_data, time_data,
        span=loess_span,
        min_spacing=do_min_spacing,
        crossing_value=do_crossing_value,
        detection_mode=detection_mode,
        pre_smooth_win=pre_smooth_win,
    )
    n_do_events = len(do_event_indices)
    avg_waiting_time = np.mean(do_waiting_times) if len(do_waiting_times) > 0 else 0.0

    # Classify as DO-variability or wild oscillator based on first PDF peak location.
    # Wild oscillators have no low-AMOC stadial state, so their first peak sits at
    # high Sv values; true DO runs always have a stadial peak below do_peak_threshold.
    fp_loc = first_peak_location(pdf_vals, x_grid, prominence_frac=pdf_prominence)
    do_variability = (fp_loc <= do_peak_threshold) and (n_do_events > 0)

    # Amplitude
    interstadial_max = []
    stadial_min = []
    for i in range(n_stadials - 1):
        interseg = amoc_smooth[stadial_ends[i]:stadial_starts[i + 1]]
        stadseg = amoc_smooth[stadial_starts[i]:stadial_ends[i]]
        if len(interseg) > 0:
            interstadial_max.append(np.max(interseg))
        if len(stadseg) > 0:
            stadial_min.append(np.min(stadseg))
    min_len = min(len(interstadial_max), len(stadial_min))
    amplitude_vals = np.array(interstadial_max[:min_len]) - np.array(stadial_min[:min_len])
    avg_amplitude = np.mean(amplitude_vals) if len(amplitude_vals) > 0 else 0.0

    # PSD (optional)
    f, Pxx = welch(amoc_data, fs=1.0, nperseg=min(256, len(amoc_data)))
    psd_bins = np.log(Pxx[:10]) if len(Pxx) >= 10 else np.log(Pxx)

    # Zero out all DO event data for wild oscillators so they don't pollute SBI
    if not do_variability:
        n_do_events          = 0
        do_event_indices     = np.array([], dtype=int)
        do_times             = np.array([])
        do_waiting_times     = np.array([])
        avg_waiting_time     = 0.0
        avg_stadial_duration = 0.0

    # Combine all results
    stats = {
        'mean': basic_stats['mean'],
        'std': basic_stats['std'],
        'quantiles': basic_stats['quantiles'],
        'pdf': pdf_vals,
        'x_grid': x_grid,
        'threshold': threshold_val,
        'n_stadials': n_stadials,
        'stadial_starts': stadial_starts,
        'stadial_ends': stadial_ends,
        'avg_stadial_duration': avg_stadial_duration,
        'stadial_durations': stadial_durations,
        'n_do_events': n_do_events,
        'do_event_indices': do_event_indices,
        'do_times': do_times,
        'waiting_times': do_waiting_times,
        'avg_waiting_time': avg_waiting_time,
        'do_variability': do_variability,
        # Keep amplitude and PSD
        'avg_amplitude': avg_amplitude,
        'psd_bins': psd_bins
    }

    if verbose:
        print(stats)

    return stats

    """
AMOC plotting utilities for DO event analysis.
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


def plot_amoc_analysis(ensemble_stats, model_files, default_stats=None, default_file=None,
                       run_ids=None, n_runs=None, spinup_fraction=0.02,
                       figsize_per_row=3, width_ratios=(3, 2),
                       show_default=True, save_path=None):
    """
    Plot AMOC time series and PDFs for selected runs.
    
    Parameters:
    -----------
    ensemble_stats : list
        List of stats dictionaries from compute_summary_stats()
    model_files : list
        List of file paths corresponding to ensemble_stats
    default_stats : dict, optional
        Stats dictionary for default run
    default_file : str, optional
        File path for default run
    run_ids : list, optional
        Specific run indices to plot. If None, uses n_runs.
    n_runs : int, optional
        Number of runs to plot from start. Ignored if run_ids is provided.
        Default is 10 if neither run_ids nor n_runs specified.
    spinup_fraction : float
        Fraction of data removed as spinup (default: 0.02)
    figsize_per_row : float
        Figure height per row in inches (default: 3)
    width_ratios : tuple
        Relative widths of (time series, PDF) columns (default: (3, 2))
    show_default : bool
        Whether to show default run as first row (default: True)
    save_path : str, optional
        If provided, save figure to this path
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    
    # Determine which runs to plot
    if run_ids is not None:
        runs_to_plot = list(run_ids)
    elif n_runs is not None:
        runs_to_plot = list(range(n_runs))
    else:
        runs_to_plot = list(range(min(10, len(ensemble_stats))))
    
    # Validate run_ids
    valid_runs = []
    for idx in runs_to_plot:
        if idx < len(ensemble_stats) and ensemble_stats[idx] is not None:
            valid_runs.append(idx)
        else:
            print(f"Warning: Run {idx} is invalid or has no stats, skipping.")
    runs_to_plot = valid_runs
    
    if len(runs_to_plot) == 0:
        raise ValueError("No valid runs to plot!")
    
    # Calculate number of rows
    n_rows = len(runs_to_plot)
    if show_default and default_stats is not None:
        n_rows += 1
    
    # Create figure
    fig, axes = plt.subplots(n_rows, 2, 
                             figsize=(18, figsize_per_row * n_rows),
                             gridspec_kw={'width_ratios': list(width_ratios)})
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    current_row = 0
    
    # ----------------------------
    # Default run (if requested)
    # ----------------------------
    if show_default and default_stats is not None and default_file is not None:
        stats = default_stats
        
        ds = xr.open_dataset(default_file)
        amoc = ds.amoc26N.values
        time = ds.time.values
        ds.close()
        
        # Apply spinup removal
        start_idx = int(len(amoc) * spinup_fraction)
        amoc = amoc[start_idx:]
        time = time[start_idx:]
        time = time - time[0]
        
        _plot_single_run(axes[current_row], time, amoc, stats, 
                         title='AMOC - Default Run')
        current_row += 1
    
    # ----------------------------
    # Ensemble runs
    # ----------------------------
    for run_idx in runs_to_plot:
        stats = ensemble_stats[run_idx]
        file = model_files[run_idx]
        
        ds = xr.open_dataset(file)
        amoc = ds.amoc26N.values
        time = ds.time.values
        ds.close()
        
        # Apply spinup removal
        start_idx = int(len(amoc) * spinup_fraction)
        amoc = amoc[start_idx:]
        time = time[start_idx:]
        time = time - time[0]
        
        _plot_single_run(axes[current_row], time, amoc, stats,
                         title=f'AMOC - Run {run_idx}')
        current_row += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def _plot_single_run(ax_row, time, amoc, stats, title='AMOC'):
    """
    Plot a single run's time series and PDF.
    
    Parameters:
    -----------
    ax_row : array of 2 axes
        The row of axes [time_series_ax, pdf_ax]
    time : array
        Time array
    amoc : array
        AMOC values
    stats : dict
        Stats dictionary from compute_summary_stats()
    title : str
        Title for the time series plot
    """
    ax_ts, ax_pdf = ax_row
    
    # Time series plot
    ax_ts.plot(time, amoc, color='darkblue', label='AMOC')
    ax_ts.axhline(stats['threshold'], color='gray', linestyle='--', 
                  alpha=0.5, label='Threshold')
    
    # Add vertical red lines at DO event times
    for do_time in stats['do_times']:
        ax_ts.axvline(do_time, color='red', alpha=0.6, linewidth=1.5)
    
    ax_ts.set_ylabel('AMOC (Sv)')
    ax_ts.set_title(title)
    ax_ts.legend(loc='upper left')
    ax_ts.grid(alpha=0.3)
    
    # Add summary stats text
    do_var = stats.get('do_variability', True)
    box_color = 'wheat' if do_var else 'lightsalmon'
    ax_ts.text(
        0.98, 0.02,
        f"DO variability: {'Yes' if do_var else 'No'}\n"
        f"N DO events: {stats['n_do_events']}\n"
        f"Avg. waiting time: {stats['avg_waiting_time']:.1f} yr\n"
        f"Avg. stadial duration: {stats['avg_stadial_duration']:.1f} yr\n"
        f"Amplitude: {stats['avg_amplitude']:.2f} Sv",
        transform=ax_ts.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8)
    )
    
    # PDF plot
    ax_pdf.plot(stats['x_grid'], stats['pdf'], color='darkblue', label='PDF')
    ax_pdf.axvline(stats['threshold'], color='gray', linestyle='--', alpha=0.5)
    ax_pdf.set_ylabel('Density')
    ax_pdf.set_xlabel('AMOC (Sv)')
    ax_pdf.legend(loc='upper right')
    ax_pdf.grid(alpha=0.3)


def plot_filtered_runs(ensemble_stats, model_files, filter_mask,
                       default_stats=None, default_file=None,
                       max_runs=20, **kwargs):
    """
    Plot runs that pass a boolean filter.
    
    Parameters:
    -----------
    ensemble_stats : list
        List of stats dictionaries
    model_files : list
        List of file paths
    filter_mask : array-like of bool
        Boolean mask where True = include run
    default_stats : dict, optional
        Stats for default run
    default_file : str, optional
        File path for default run
    max_runs : int
        Maximum number of filtered runs to plot (default: 20)
    **kwargs : 
        Additional arguments passed to plot_amoc_analysis()
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    # Get indices where filter is True
    run_ids = np.where(filter_mask)[0][:max_runs]
    
    print(f"Plotting {len(run_ids)} runs out of {filter_mask.sum()} filtered runs")
    
    return plot_amoc_analysis(
        ensemble_stats, model_files,
        default_stats=default_stats, default_file=default_file,
        run_ids=run_ids, **kwargs
    )
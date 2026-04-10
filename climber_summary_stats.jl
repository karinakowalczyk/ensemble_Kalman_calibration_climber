using NCDatasets
using Statistics
using KernelDensity
using Clustering
using Loess
using MultivariateStats
using LinearAlgebra

# =============================================================================
# HELPER FUNCTIONS FOR DO EVENT DETECTION
# =============================================================================

"""
Detect stadial threshold adaptively using clustering
"""
function detect_stadials_adaptive(amoc_smooth; method="clustering", offset=2.5)
    if method == "percentile"
        return quantile(amoc_smooth, 0.25)
    elseif method == "bimodal_gap"
        hist = fit(Histogram, amoc_smooth, nbins=50)
        edges = hist.edges[1]
        counts = hist.weights
        bin_widths = diff(edges)
        density = counts ./ (sum(counts) .* bin_widths)
        bin_centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
        mid_start = length(density) ÷ 4 + 1
        mid_end = 3 * length(density) ÷ 4
        local_min_idx = mid_start + argmin(density[mid_start:mid_end]) - 1
        return bin_centers[local_min_idx]
    elseif method == "clustering"
        data_matrix = reshape(amoc_smooth, (1, length(amoc_smooth)))
        result = kmeans(data_matrix, 2; maxiter=100, display=:none)
        centers = result.centers[1, :]
        low, high = minimum(centers), maximum(centers)
        threshold = low + offset
        if threshold > high
            threshold = mean(centers)
        end
        return threshold
    else
        error("Unknown method: $method")
    end
end

"""
Find positive peaks in signal
"""
function find_peaks_positive(signal)
    peaks = Int[]
    for i in 2:(length(signal)-1)
        if signal[i] > signal[i-1] && signal[i] > signal[i+1]
            push!(peaks, i)
        end
    end
    return peaks
end

"""
Filter peaks by minimum spacing, keeping highest peak in each cluster
"""
function filter_peaks_by_spacing(peak_indices, values, time, min_spacing)
    if length(peak_indices) == 0
        return peak_indices
    end
    filtered = Int[]
    i = 1
    while i <= length(peak_indices)
        current_idx = peak_indices[i]
        current_time = time[current_idx]
        cluster = [current_idx]
        j = i + 1
        while j <= length(peak_indices)
            if time[peak_indices[j]] - current_time < min_spacing
                push!(cluster, peak_indices[j])
                j += 1
            else
                break
            end
        end
        cluster_values = values[cluster]
        best_idx = cluster[argmax(cluster_values)]
        push!(filtered, best_idx)
        i = j
    end
    return filtered
end

"""
Find crossing before peak where residual crosses threshold
"""
function find_crossing_before_peak(residual, time, peak_idx, crossing_value)
    for i in peak_idx:-1:2
        if residual[i] >= crossing_value && residual[i-1] < crossing_value
            return i
        end
    end
    return nothing
end

"""
Detect DO onsets as upward crossings of `cv` in a pre-computed LOESS residual,
with a minimum inter-event spacing filter.
Returns (onset_indices, onset_times, avg_waiting_time).
"""
function detect_do_upward_crossing(resid, time; cv=-0.8, min_spacing=500.0)
    all_cross = [i for i in 2:length(resid)
                 if resid[i-1] < cv && resid[i] >= cv]
    onsets = Int[]
    for i in all_cross
        if isempty(onsets) || time[i] - time[onsets[end]] >= min_spacing
            push!(onsets, i)
        end
    end
    do_times = isempty(onsets) ? Float64[] : time[onsets]
    avg_wt   = length(do_times) > 1 ? mean(diff(do_times)) : NaN
    return onsets, do_times, avg_wt
end

"""
Detect DO events using LOESS detrending and peak detection
"""
function detect_do_events_simple(amoc, time; span=0.02, min_spacing=500, crossing_value=5.0)
    model = loess(time, amoc, span=span)
    trend = predict(model, time)
    residual = amoc .- trend
    all_peaks = find_peaks_positive(residual)
    significant_peaks = [p for p in all_peaks if residual[p] > crossing_value]
    if length(significant_peaks) > 0
        filtered_peaks = filter_peaks_by_spacing(significant_peaks, residual, time, min_spacing)
    else
        filtered_peaks = Int[]
    end
    do_event_indices = Int[]
    for peak_idx in filtered_peaks
        crossing_idx = find_crossing_before_peak(residual, time, peak_idx, crossing_value)
        if !isnothing(crossing_idx)
            crossing_time = time[crossing_idx]
            is_far_enough = true
            for prev_idx in do_event_indices
                if abs(crossing_time - time[prev_idx]) < min_spacing
                    is_far_enough = false
                    break
                end
            end
            if is_far_enough
                push!(do_event_indices, crossing_idx)
            end
        end
    end
    do_times = time[do_event_indices]
    do_waiting_times = length(do_times) > 1 ? diff(do_times) : Float64[]
    return do_event_indices, do_times, do_waiting_times
end

# =============================================================================
# MAIN SUMMARY STATISTICS COMPUTATION
# =============================================================================

"""
Compute comprehensive summary statistics from AMOC timeseries
Returns: Dictionary with PDF, stadial info, DO events, etc.
"""
function compute_summary_stats(amoc_data; time_data=nothing, remove_spinup=true,
                               spinup_fraction=0.02, adaptive_threshold=true,
                               threshold_method="clustering", threshold=nothing,
                               grid_points=100, ignore_first_stadial=true,
                               loess_span=0.02, do_min_spacing=500, do_crossing_value=5.0,
                               do_method="loess")
    # do_method: "loess"       — original LOESS detrend + peak detection (default)
    #            "stadial_end" — use stadial→interstadial threshold crossings directly
    amoc_data = vec(amoc_data)
    if isnothing(time_data)
        time_data = collect(0:length(amoc_data)-1)
    else
        time_data = vec(time_data)
    end
    
    # Remove spinup
    if remove_spinup
        start_idx = Int(floor(length(amoc_data) * spinup_fraction)) + 1
        amoc_data = amoc_data[start_idx:end]
        time_data = time_data[start_idx:end]
        time_data = time_data .- time_data[1]
    end
    
    # Compute PDF using KDE
    kde_obj = kde(amoc_data)
    x_grid = range(minimum(amoc_data), maximum(amoc_data), length=grid_points)
    pdf_vals = pdf(kde_obj, x_grid)
    integral = sum((pdf_vals[1:end-1] .+ pdf_vals[2:end]) .* diff(x_grid)) / 2
    pdf_vals = pdf_vals ./ integral
    
    # Detect stadial threshold
    if adaptive_threshold
        threshold_val = detect_stadials_adaptive(amoc_data; method=threshold_method)
    else
        threshold_val = isnothing(threshold) ? quantile(amoc_data, 0.25) : threshold
    end
    
    # Identify stadials
    is_stadial = amoc_data .< threshold_val
    transitions = diff(Int.(is_stadial))
    stadial_starts = findall(==(1), transitions) .+ 1
    stadial_ends = findall(==(-1), transitions) .+ 1
    if is_stadial[1] && !ignore_first_stadial
        stadial_starts = vcat([1], stadial_starts)
    end
    if is_stadial[end]
        stadial_ends = vcat(stadial_ends, [length(amoc_data)])
    end
    if ignore_first_stadial && length(stadial_ends) > length(stadial_starts)
        if length(stadial_starts) == 0 || stadial_ends[1] < stadial_starts[1]
            stadial_ends = stadial_ends[2:end]
        end
    end
    if length(stadial_starts) > length(stadial_ends)
        stadial_starts = stadial_starts[1:length(stadial_ends)]
    end
    if length(stadial_starts) != length(stadial_ends)
        min_len = min(length(stadial_starts), length(stadial_ends))
        stadial_starts = stadial_starts[1:min_len]
        stadial_ends = stadial_ends[1:min_len]
    end
    
    n_stadials = length(stadial_starts)
    if n_stadials > 0
        stadial_durations = time_data[stadial_ends] .- time_data[stadial_starts]
        avg_stadial_duration = mean(stadial_durations)
    else
        stadial_durations = Float64[]
        avg_stadial_duration = 0.0
    end
    
    # Detect DO events
    if do_method == "stadial_end"
        # Each stadial→interstadial crossing is a DO event onset
        do_event_indices = copy(stadial_ends)
        do_times         = length(stadial_ends) > 0 ? time_data[stadial_ends] : Float64[]
        do_waiting_times = length(do_times) > 1 ? diff(do_times) : Float64[]
    elseif do_method == "upward_crossing"
        # LOESS residual upward-crossing at do_crossing_value
        model_uc = loess(time_data, amoc_data, span=loess_span)
        resid_uc = amoc_data .- predict(model_uc, time_data)
        do_event_indices, do_times, _ = detect_do_upward_crossing(
            resid_uc, time_data; cv=do_crossing_value, min_spacing=Float64(do_min_spacing))
        do_waiting_times = length(do_times) > 1 ? diff(do_times) : Float64[]
    else
        do_event_indices, do_times, do_waiting_times = detect_do_events_simple(
            amoc_data, time_data;
            span=loess_span,
            min_spacing=do_min_spacing,
            crossing_value=do_crossing_value
        )
    end

    n_do_events = length(do_event_indices)
    avg_waiting_time = length(do_waiting_times) > 0 ? mean(do_waiting_times) : 0.0
    
    return Dict(
        "pdf" => pdf_vals,
        "x_grid" => collect(x_grid),
        "threshold" => threshold_val,
        "n_stadials" => n_stadials,
        "stadial_starts" => stadial_starts,
        "stadial_ends" => stadial_ends,
        "avg_stadial_duration" => avg_stadial_duration,
        "stadial_durations" => stadial_durations,
        "n_do_events" => n_do_events,
        "do_event_indices" => do_event_indices,
        "do_times" => do_times,
        "waiting_times" => do_waiting_times,
        "avg_waiting_time" => avg_waiting_time
    )
end

# =============================================================================
# CLIMBER-X SPECIFIC FUNCTIONS
# =============================================================================

"""
Read AMOC from CLIMBER-X output file
"""
function read_climber_amoc(output_file::String)
    if !isfile(output_file)
        error("Output file does not exist: $output_file")
    end
    
    try
        ds = NCDataset(output_file)
        amoc = ds["amoc26N"][:]
        time = ds["time"][:]
        close(ds)
        
        return amoc, time
    catch e
        @error "Failed to read CLIMBER-X output: $output_file" exception=e
        rethrow(e)
    end
end

"""
Process CLIMBER-X output and extract summary statistics for calibration
Returns vector: [pca_comp_1, ..., pca_comp_5, avg_waiting_time, avg_stadial_duration]
"""
function process_climber_output(output_file::String, pca_model; 
                                remove_spinup=true, spinup_fraction=0.02, do_crossing_value = 5.0)
    # Read AMOC data
    amoc, time = read_climber_amoc(output_file)
    
    # Compute summary statistics
    stats = compute_summary_stats(amoc; 
                                  time_data=time,
                                  remove_spinup=remove_spinup,
                                  spinup_fraction=spinup_fraction,
                                  adaptive_threshold=true,
                                  threshold_method="clustering",
                                  grid_points=100,
                                  ignore_first_stadial=true,
                                  loess_span=0.02,
                                  do_min_spacing=500,
                                  do_crossing_value=do_crossing_value)
    
    # Transform PDF to PCA space
    pdf = stats["pdf"]
    pca_components = vec(MultivariateStats.transform(pca_model, pdf))
    
    # Extract calibration targets (5 PCA + waiting time + stadial duration)
    calibration_stats = vcat(
        pca_components[1:5],  # First 5 PCA components
        stats["avg_waiting_time"],
        stats["avg_stadial_duration"]
    )
    
    return calibration_stats, stats
end

"""
Fit PCA model from ensemble of PDFs
"""
function fit_pca_from_ensemble(pdf_matrix; n_components=5)
    # pdf_matrix should be (n_grid_points x n_ensemble)
    pca_model = fit(PCA, pdf_matrix; maxoutdim=n_components)
    
    explained_var = principalvars(pca_model)
    total_var = var(pca_model)
    explained_var_ratio = explained_var ./ total_var
    
    println("  PCA variance explained:")
    for i in 1:n_components
        println("    Component $i: $(round(explained_var_ratio[i]*100, digits=2))%")
    end
    println("    Total: $(round(sum(explained_var_ratio)*100, digits=2))%")
    
    return pca_model
end

"""
Initialize PCA model from default run and first ensemble
"""
function initialize_pca_model(default_file::String, ensemble_files::Vector{String}; 
                              n_components=5, remove_spinup=true, spinup_fraction=0.02)
    println("\n  Initializing PCA model...")
    
    # Read default run
    amoc_default, time_default = read_climber_amoc(default_file)
    stats_default = compute_summary_stats(amoc_default; 
                                         time_data=time_default,
                                         remove_spinup=remove_spinup,
                                         spinup_fraction=spinup_fraction)
    
    # Collect all PDFs
    all_pdfs = [stats_default["pdf"]]
    
    for file in ensemble_files
        if isfile(file)
            try
                amoc, time = read_climber_amoc(file)
                stats = compute_summary_stats(amoc; 
                                            time_data=time,
                                            remove_spinup=remove_spinup,
                                            spinup_fraction=spinup_fraction)
                push!(all_pdfs, stats["pdf"])
            catch e
                @warn "Skipping file $file due to error: $e"
            end
        end
    end
    
    println("  Using $(length(all_pdfs)) PDFs for PCA (including default)")
    
    # Create matrix: (n_grid_points x n_pdfs)
    pdf_matrix = hcat(all_pdfs...)
    
    # Fit PCA
    pca_model = fit_pca_from_ensemble(pdf_matrix; n_components=n_components)

    return pca_model, stats_default
end

"""
Estimate observation uncertainties by splitting the default run into blocks.

Block size should match the length of calibration runs so that the std across
blocks directly gives the uncertainty for one model evaluation.

Returns a Dict containing per-block data and derived uncertainty vectors.
Optionally saves the full result to `save_dir/block_uncertainty_analysis.jld2`.
"""
function estimate_block_uncertainties(default_file::String, pdf_grid;
                                       block_size=6000, min_do_events=2,
                                       remove_spinup=true, spinup_fraction=0.02,
                                       do_min_spacing=500, do_crossing_value=-0.8,
                                       do_method="upward_crossing",
                                       save_dir=nothing)
    println("  Reading default run: $default_file")
    amoc, time = read_climber_amoc(default_file)

    if remove_spinup
        start_idx = Int(floor(length(amoc) * spinup_fraction)) + 1
        amoc = amoc[start_idx:end]
        time = time[start_idx:end]
    end

    total_years = time[end] - time[1]
    dt          = mean(diff(time))
    block_steps = round(Int, block_size / dt)
    n_blocks    = floor(Int, length(amoc) / block_steps)

    println("  Run length after spinup: $(round(total_years, digits=0)) years")
    println("  Block size: ~$block_size years  →  $n_blocks blocks")

    n_pdf             = length(pdf_grid)
    block_pdfs        = zeros(n_pdf, n_blocks)
    block_wts         = zeros(n_blocks)
    block_sds         = zeros(n_blocks)
    block_n_do        = zeros(Int, n_blocks)
    block_start_times = zeros(n_blocks)
    block_end_times   = zeros(n_blocks)

    for b in 1:n_blocks
        idx_s = (b - 1) * block_steps + 1
        idx_e = b * block_steps
        amoc_b = amoc[idx_s:idx_e]
        time_b = time[idx_s:idx_e]
        block_start_times[b] = time_b[1]
        block_end_times[b]   = time_b[end]
        time_b = time_b .- time_b[1]

        block_pdfs[:, b] = compute_pdf_on_grid(amoc_b, pdf_grid; remove_spinup=false)

        stats_b = compute_summary_stats(amoc_b;
                                        time_data=time_b,
                                        remove_spinup=false,
                                        spinup_fraction=0.0,
                                        adaptive_threshold=true,
                                        threshold_method="clustering",
                                        loess_span=0.02,
                                        do_min_spacing=do_min_spacing,
                                        do_crossing_value=do_crossing_value,
                                        do_method=do_method)
        block_wts[b]  = stats_b["avg_waiting_time"]
        block_sds[b]  = stats_b["avg_stadial_duration"]
        block_n_do[b] = stats_b["n_do_events"]
    end

    # Report per-block DO event counts
    println("\n  Per-block DO event counts:")
    for b in 1:n_blocks
        flag = block_n_do[b] < min_do_events ? " ← excluded (< $min_do_events events)" : ""
        println("    Block $b  ($(round(Int, block_start_times[b]))–$(round(Int, block_end_times[b])) yr):  $(block_n_do[b]) DO events$flag")
    end

    valid_blocks = findall(block_n_do .>= min_do_events)
    n_valid = length(valid_blocks)
    println("\n  Valid blocks: $n_valid / $n_blocks")

    if n_valid < 2
        error("Too few valid blocks ($n_valid) — reduce min_do_events or increase block_size")
    end

    # Uncertainties: std across valid blocks
    # Since block_size ≈ calibration run length, no sqrt(N) scaling is needed.
    pdf_uncertainty = vec(std(block_pdfs[:, valid_blocks], dims=2))
    wt_uncertainty  = std(block_wts[valid_blocks])
    sd_uncertainty  = std(block_sds[valid_blocks])

    println("\n  Derived uncertainties (std across $n_valid valid blocks):")
    println("    PDF mean per-grid-point std: $(round(mean(pdf_uncertainty), digits=6))")
    println("    PDF max  per-grid-point std: $(round(maximum(pdf_uncertainty), digits=6))")
    println("    Waiting time std:            $(round(wt_uncertainty, digits=1)) years")
    println("    Stadial duration std:        $(round(sd_uncertainty, digits=1)) years")

    result = Dict(
        "block_pdfs"         => block_pdfs,
        "block_wts"          => block_wts,
        "block_sds"          => block_sds,
        "block_n_do"         => block_n_do,
        "block_start_times"  => block_start_times,
        "block_end_times"    => block_end_times,
        "valid_blocks"       => valid_blocks,
        "n_blocks"           => n_blocks,
        "block_size"         => block_size,
        "min_do_events"      => min_do_events,
        "pdf_uncertainty"    => pdf_uncertainty,
        "wt_uncertainty"     => wt_uncertainty,
        "sd_uncertainty"     => sd_uncertainty,
        "pdf_grid"           => collect(pdf_grid)
    )

    if !isnothing(save_dir)
        save_file = joinpath(save_dir, "block_uncertainty_analysis.jld2")
        @save save_file result
        println("  ✓ Block analysis saved: $save_file")
    end

    return result
end
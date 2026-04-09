using JLD2
using Plots
using Printf
using Statistics
using LinearAlgebra
using StatsPlots
using NCDatasets
using DSP
using MultivariateStats

# ============================================
# LOAD RESULTS
# ============================================

function load_final_results(output_dir="output")
    """Load the final results from a completed calibration"""
    final_file = joinpath(output_dir, "final_results.jld2")
    
    if !isfile(final_file)
        error("No final results found in $output_dir")
    end
    
    @load final_file final_data
    return final_data
end

function load_checkpoint_for_animation(checkpoint_file)
    """Load a specific checkpoint"""
    @load checkpoint_file checkpoint_data
    return checkpoint_data
end

function get_latest_checkpoint(output_dir)
    """Find and load the latest checkpoint"""
    checkpoint_dir = joinpath(output_dir, "checkpoints")
    
    if !isdir(checkpoint_dir)
        error("Checkpoints directory not found: $checkpoint_dir")
    end
    
    checkpoint_files = filter(f -> startswith(f, "checkpoint_iter_"), 
                             readdir(checkpoint_dir))
    
    if isempty(checkpoint_files)
        error("No checkpoints found in $checkpoint_dir")
    end
    
    iterations = [parse(Int, match(r"checkpoint_iter_(\d+)", f).captures[1]) 
                 for f in checkpoint_files]
    latest_iter = maximum(iterations)
    latest_file = joinpath(checkpoint_dir, "checkpoint_iter_$(latest_iter).jld2")
    
    return latest_file, latest_iter
end

# ============================================
# AMOC DATA READING AND PROCESSING
# ============================================

function read_amoc_data(output_file; remove_spinup=true, spinup_fraction=0.02)
    """Read AMOC data from NetCDF file"""
    ds = NCDataset(output_file)
    amoc = ds["amoc26N"][:]
    time = ds["time"][:]
    close(ds)
    
    if remove_spinup
        start_idx = Int(floor(length(amoc) * spinup_fraction)) + 1
        amoc = amoc[start_idx:end]
        time = time[start_idx:end]
        time = time .- time[1]
    end
    
    return amoc, time
end

function compute_psd(signal; fs=1.0, nperseg=256)
    """Compute power spectral density using Welch's method"""
    nperseg = min(nperseg, length(signal))
    
    # Use DSP.jl for Welch periodogram
    psd_result = welch_pgram(signal, nperseg, fs=fs)
    
    return freq(psd_result), power(psd_result)
end

# ============================================
# AMOC TIMESERIES VISUALIZATION
# ============================================

function plot_amoc_ensemble_iteration(output_dir, default_file, iteration; 
                                      n_members=10, climber_summary_stats_file="climber_summary_stats.jl")
    """
    Plot AMOC diagnostics for multiple ensemble members from a specific iteration
    Similar to your Python multi-panel figure
    """
    
    println("Plotting AMOC diagnostics for iteration $iteration...")
    println("  Number of members to plot: $n_members")
    
    # Load compute_summary_stats function
    include(climber_summary_stats_file)
    
    # Load default stats
    amoc_default, time_default = read_amoc_data(default_file)
    default_stats = compute_summary_stats(amoc_default; time_data=time_default)
    
    # Find ensemble member output files
    iter_dir = joinpath(output_dir, "iter_$(iteration)")
    
    if !isdir(iter_dir)
        error("Iteration directory not found: $iter_dir")
    end
    
    # Collect valid output files
    ensemble_files = []
    for member_id in 1:n_members
        output_file = joinpath(iter_dir, "member_$(member_id)", "ocn_ts.nc")
        if isfile(output_file)
            push!(ensemble_files, output_file)
        else
            @warn "Member $member_id output not found"
        end
    end
    
    n_actual = length(ensemble_files)
    println("  Found $n_actual valid ensemble members")
    
    if n_actual == 0
        error("No valid ensemble members found for iteration $iteration")
    end
    
    # Create multi-panel figure
    fig = plot(layout=(n_actual, 3), size=(1800, 300*n_actual),
              plot_title="AMOC Diagnostics - Iteration $iteration")
    
    for (i, file) in enumerate(ensemble_files)
        # Read ensemble data
        amoc_ensemble, time_ensemble = read_amoc_data(file)
        ensemble_stats = compute_summary_stats(amoc_ensemble; time_data=time_ensemble)
        
        row = i
        
        # Panel 1: Time series
        plot!(fig[row, 1], time_ensemble, amoc_ensemble, color=:darkblue, 
              label="", linewidth=1)
        plot!(fig[row, 1], time_default, amoc_default, color=:lightcoral, 
              label="", linewidth=1, alpha=0.5)
        hline!(fig[row, 1], [ensemble_stats["threshold"]], color=:gray, 
               linestyle=:dash, alpha=0.5, label="")
        ylabel!(fig[row, 1], "AMOC (Sv)")
        if i == 1
            title!(fig[row, 1], "AMOC Timeseries")
        end
        if i == n_actual
            xlabel!(fig[row, 1], "Time (years)")
        end
        
        # Add member label
        annotate!(fig[row, 1], 0.02*maximum(time_ensemble), 
                 0.98*maximum(amoc_ensemble), 
                 text("Member $i", :left, 9, :black))
        
        # Add stats text
        stats_text = @sprintf("Stadial: %.1f (Def: %.1f)\nWait: %.1f (Def: %.1f)",
                             ensemble_stats["avg_stadial_duration"], 
                             default_stats["avg_stadial_duration"],
                             ensemble_stats["avg_waiting_time"], 
                             default_stats["avg_waiting_time"])
        annotate!(fig[row, 1], 0.98*maximum(time_ensemble), 
                 minimum(amoc_ensemble)*1.1, 
                 text(stats_text, :right, 7))
        
        # Panel 2: PDF
        plot!(fig[row, 2], ensemble_stats["x_grid"], ensemble_stats["pdf"], 
              color=:darkblue, label="", linewidth=2)
        plot!(fig[row, 2], default_stats["x_grid"], default_stats["pdf"], 
              color=:lightcoral, label="", linewidth=2, alpha=0.5)
        vline!(fig[row, 2], [ensemble_stats["threshold"]], color=:gray, 
               linestyle=:dash, alpha=0.5, label="")
        ylabel!(fig[row, 2], "Density")
        if i == 1
            title!(fig[row, 2], "PDF")
        end
        if i == n_actual
            xlabel!(fig[row, 2], "AMOC (Sv)")
        end
        
        # Panel 3: PSD
        f_ensemble, psd_ensemble = compute_psd(amoc_ensemble, fs=1.0, 
                                               nperseg=min(256, length(amoc_ensemble)))
        f_default, psd_default = compute_psd(amoc_default, fs=1.0, 
                                            nperseg=min(256, length(amoc_default)))
        
        plot!(fig[row, 3], f_ensemble, psd_ensemble, color=:darkblue, 
              label="", linewidth=2, xscale=:log10, yscale=:log10)
        plot!(fig[row, 3], f_default, psd_default, color=:lightcoral, 
              label="", linewidth=2, alpha=0.5, xscale=:log10, yscale=:log10)
        ylabel!(fig[row, 3], "Power")
        if i == 1
            title!(fig[row, 3], "PSD")
        end
        if i == n_actual
            xlabel!(fig[row, 3], "Frequency")
        end
    end
    
    display(fig)
    println("✓ AMOC diagnostics plot displayed!")
    
    return fig
end

# ============================================
# PARAMETER DISTRIBUTIONS PER ITERATION
# ============================================

function plot_parameter_distributions_per_iteration(output_dir; param_names=nothing, 
                                                     prior_bounds=nothing, 
                                                     iteration=nothing)
    """
    Plot parameter distributions for a specific iteration (or all iterations as animation)
    """
    
    println("Loading results...")
    latest_file, latest_iter = get_latest_checkpoint(output_dir)
    checkpoint_data = load_checkpoint_for_animation(latest_file)
    param_history = checkpoint_data["param_history"]
    
    N_params = size(param_history, 1)
    N_iterations = size(param_history, 2) - 1
    
    if isnothing(param_names)
        param_names = ["Param $i" for i in 1:N_params]
    end
    
    # If specific iteration requested, plot it
    if !isnothing(iteration)
        if iteration < 0 || iteration > N_iterations
            error("Iteration $iteration out of range [0, $N_iterations]")
        end
        
        params_at_iter = param_history[:, iteration+1, :]
        
        n_cols = min(3, N_params)
        n_rows = Int(ceil(N_params / n_cols))
        
        p = plot(layout=(n_rows, n_cols), size=(400*n_cols, 300*n_rows),
                plot_title="Parameter Distributions - Iteration $iteration")
        
        for param_idx in 1:N_params
            subplot_idx = param_idx
            
            # Histogram
            histogram!(p[subplot_idx], params_at_iter[param_idx, :],
                      bins=20, alpha=0.6, color=:blue, label="Ensemble",
                      normalize=:probability, xlabel=param_names[param_idx],
                      ylabel="Probability")
            
            # Mean
            param_mean = mean(params_at_iter[param_idx, :])
            vline!(p[subplot_idx], [param_mean], 
                   color=:red, linewidth=3, label="Mean")
            
            # Prior bounds if provided
            if !isnothing(prior_bounds) && haskey(prior_bounds, param_names[param_idx])
                bounds = prior_bounds[param_names[param_idx]]
                vline!(p[subplot_idx], [bounds[1]], 
                       color=:gray, linewidth=2, linestyle=:dash, label="Prior")
                vline!(p[subplot_idx], [bounds[2]], 
                       color=:gray, linewidth=2, linestyle=:dash, label="")
            end
        end
        
        display(p)
        return p
    else
        # Create animation over all iterations
        println("Creating parameter distribution animation...")
        
        n_cols = min(3, N_params)
        n_rows = Int(ceil(N_params / n_cols))
        
        anim = @animate for iter in 0:N_iterations
            params_at_iter = param_history[:, iter+1, :]
            
            p = plot(layout=(n_rows, n_cols), size=(400*n_cols, 300*n_rows),
                    plot_title="Parameter Distributions - Iteration $iter/$N_iterations")
            
            for param_idx in 1:N_params
                subplot_idx = param_idx
                
                histogram!(p[subplot_idx], params_at_iter[param_idx, :],
                          bins=20, alpha=0.6, color=:blue, label="",
                          normalize=:probability, xlabel=param_names[param_idx],
                          ylabel="Probability")
                
                param_mean = mean(params_at_iter[param_idx, :])
                vline!(p[subplot_idx], [param_mean], 
                       color=:red, linewidth=3, label="")
                
                if !isnothing(prior_bounds) && haskey(prior_bounds, param_names[param_idx])
                    bounds = prior_bounds[param_names[param_idx]]
                    vline!(p[subplot_idx], [bounds[1]], 
                           color=:gray, linewidth=2, linestyle=:dash, label="")
                    vline!(p[subplot_idx], [bounds[2]], 
                           color=:gray, linewidth=2, linestyle=:dash, label="")
                end
            end
        end
        
        display(gif(anim, fps=2))
        println("✓ Parameter distribution animation displayed!")
        return anim
    end
end

# ============================================
# PARAMETER CONVERGENCE ANIMATIONS
# ============================================

function create_pairwise_convergence_animation(output_dir; fps=2, param_names=nothing)
    """
    Create animation showing pairwise parameter convergence over iterations
    """
    
    println("Loading results...")
    latest_file, latest_iter = get_latest_checkpoint(output_dir)
    checkpoint_data = load_checkpoint_for_animation(latest_file)
    param_history = checkpoint_data["param_history"]
    
    N_params = size(param_history, 1)
    N_iterations = size(param_history, 2) - 1
    N_ensemble = size(param_history, 3)
    
    if isnothing(param_names)
        param_names = ["Param $i" for i in 1:N_params]
    end
    
    println("Creating pairwise convergence animation...")
    println("  Number of parameters: $N_params")
    println("  Number of iterations: $N_iterations")
    println("  Ensemble size: $N_ensemble")
    
    # Create pairs (upper triangle only)
    param_pairs = []
    for i in 1:(N_params-1)
        for j in (i+1):N_params
            push!(param_pairs, (i, j))
        end
    end
    
    n_pairs = length(param_pairs)
    println("  Number of parameter pairs: $n_pairs")
    
    # Determine layout
    n_cols = min(3, n_pairs)
    n_rows = Int(ceil(n_pairs / n_cols))
    
    anim = @animate for iter in 0:N_iterations
        params_at_iter = param_history[:, iter+1, :]
        param_mean = vec(mean(params_at_iter, dims=2))
        
        p = plot(layout=(n_rows, n_cols), size=(400*n_cols, 350*n_rows),
                plot_title="Iteration $iter/$N_iterations", plot_titlevspan=0.05)
        
        for (plot_idx, (i, j)) in enumerate(param_pairs)
            subplot_idx = plot_idx
            
            scatter!(p[subplot_idx], params_at_iter[i, :], params_at_iter[j, :],
                    alpha=0.5, color=:blue, markersize=4,
                    xlabel=param_names[i], ylabel=param_names[j],
                    legend=false, framestyle=:box)
            
            # Add mean
            scatter!(p[subplot_idx], [param_mean[i]], [param_mean[j]], 
                    color=:red, markersize=8, markershape=:cross, linewidth=2)
            
            # Add covariance ellipse
            if iter > 0
                pair_params = params_at_iter[[i,j], :]
                cov_matrix = cov(pair_params')
                eigen_result = eigen(cov_matrix)
                angle = atan(eigen_result.vectors[2,1], eigen_result.vectors[1,1])
                width = 2 * sqrt(eigen_result.values[1])
                height = 2 * sqrt(eigen_result.values[2])
                
                θ_ellipse = range(0, 2π, length=100)
                ellipse_x = param_mean[i] .+ width * cos.(θ_ellipse) * cos(angle) - 
                                           height * sin.(θ_ellipse) * sin(angle)
                ellipse_y = param_mean[j] .+ width * cos.(θ_ellipse) * sin(angle) + 
                                           height * sin.(θ_ellipse) * cos(angle)
                
                plot!(p[subplot_idx], ellipse_x, ellipse_y, 
                     color=:blue, linewidth=1.5, linestyle=:dash)
            end
        end
    end
    
    display(gif(anim, fps=fps))
    println("✓ Pairwise convergence animation displayed!")
    
    return anim
end

function create_parameter_evolution_animation(output_dir; fps=2, param_names=nothing)
    """
    Create animation showing individual parameter evolution over time
    """
    
    println("Loading results...")
    latest_file, latest_iter = get_latest_checkpoint(output_dir)
    checkpoint_data = load_checkpoint_for_animation(latest_file)
    param_history = checkpoint_data["param_history"]
    
    N_params = size(param_history, 1)
    N_iterations = size(param_history, 2) - 1
    N_ensemble = size(param_history, 3)
    
    if isnothing(param_names)
        param_names = ["Param $i" for i in 1:N_params]
    end
    
    println("Creating parameter evolution animation...")
    
    # Layout
    n_cols = min(3, N_params)
    n_rows = Int(ceil(N_params / n_cols))
    
    anim = @animate for current_iter in 0:N_iterations
        p = plot(layout=(n_rows, n_cols), size=(400*n_cols, 300*n_rows))
        
        for param_idx in 1:N_params
            param_mean = vec(mean(param_history[param_idx, 1:current_iter+1, :], dims=2))
            param_std = vec(std(param_history[param_idx, 1:current_iter+1, :], dims=2))
            
            subplot_idx = param_idx
            
            # Confidence band
            if current_iter > 0
                plot!(p[subplot_idx], 0:current_iter, param_mean .+ param_std, 
                      fillrange=param_mean .- param_std, 
                      fillalpha=0.3, fillcolor=:blue, 
                      label="", linewidth=0)
            end
            
            # Sample trajectories
            for j in 1:min(10, N_ensemble)
                plot!(p[subplot_idx], 0:current_iter, 
                      param_history[param_idx, 1:current_iter+1, j], 
                      alpha=0.2, color=:blue, label="")
            end
            
            # Mean
            plot!(p[subplot_idx], 0:current_iter, param_mean, 
                  color=:black, linewidth=3, label="")
            
            xlabel!(p[subplot_idx], "Iteration")
            ylabel!(p[subplot_idx], param_names[param_idx])
            xlims!(p[subplot_idx], (0, N_iterations))
        end
        
        plot!(p[1], title=@sprintf("Iteration %d/%d", current_iter, N_iterations),
              titlelocation=:left, titlefontsize=10)
    end
    
    display(gif(anim, fps=fps))
    println("✓ Parameter evolution animation displayed!")
    
    return anim
end

# ============================================
# SUMMARY STATISTICS VISUALIZATION
# ============================================

function plot_summary_stats_convergence(output_dir; n_pca=5)
    """
    Plot convergence of normalised PCA observations + dynamical stats over iterations.
    Uses G_pca and y_obs_pca stored per iteration (correct for :pca mode).
    """

    println("Loading iteration results...")
    latest_file, latest_iter = get_latest_checkpoint(output_dir)

    G_pca_history  = []
    y_obs_pca_ref  = nothing
    iters_with_pca = Int[]

    for iter in 1:latest_iter
        iter_file = joinpath(output_dir, "iteration_$(iter)_results.jld2")
        isfile(iter_file) || continue
        @load iter_file iter_data
        G = get(iter_data, "G_pca", nothing)
        y = get(iter_data, "y_obs_pca", nothing)
        if !isnothing(G) && !isnothing(y)
            push!(G_pca_history, G)
            push!(iters_with_pca, iter)
            y_obs_pca_ref = y
        end
    end

    if isempty(G_pca_history)
        @warn "No G_pca data found — run in :pca mode or check iteration result files"
        return nothing
    end

    n_obs = length(y_obs_pca_ref)
    n_iters = length(G_pca_history)
    labels = vcat(["PCA $k" for k in 1:n_pca], ["Waiting time", "Stadial dur."])

    means = zeros(n_obs, n_iters)
    stds  = zeros(n_obs, n_iters)
    for (k, G) in enumerate(G_pca_history)
        valid = [j for j in 1:size(G, 2) if !any(isnan.(G[:, j]))]
        means[:, k] = vec(mean(G[:, valid], dims=2))
        stds[:, k]  = vec(std(G[:, valid],  dims=2))
    end

    n_cols = min(3, n_obs)
    n_rows = Int(ceil(n_obs / n_cols))
    p = plot(layout=(n_rows, n_cols), size=(420*n_cols, 300*n_rows),
             plot_title="Normalised observation convergence")

    for k in 1:n_obs
        plot!(p[k], iters_with_pca, means[k, :] .+ stds[k, :],
              fillrange=means[k, :] .- stds[k, :],
              fillalpha=0.25, fillcolor=:blue, linewidth=0, label="±1σ")
        plot!(p[k], iters_with_pca, means[k, :],
              color=:blue, linewidth=2, label="Ensemble mean")
        hline!(p[k], [y_obs_pca_ref[k]],
               color=:red, linewidth=2, linestyle=:dash, label="Target")
        xlabel!(p[k], "Iteration")
        ylabel!(p[k], labels[k])
    end

    display(p)
    println("✓ Summary statistics convergence plot displayed!")
    return p
end

# ============================================
# FINAL RESULTS SUMMARY
# ============================================

function print_calibration_summary(output_dir; param_names=nothing, prior_bounds=nothing)
    """
    Print summary table of calibration results
    """
    
    println("\n" * "="^80)
    println("CALIBRATION SUMMARY")
    println("="^80)
    
    final_data = load_final_results(output_dir)
    
    θ_optimal = final_data["θ_optimal"]
    θ_std = final_data["θ_std"]
    
    N_params = length(θ_optimal)
    
    if isnothing(param_names)
        param_names = ["Param $i" for i in 1:N_params]
    end
    
    println()
    @printf("%-25s | %15s | %12s | %20s\n", 
            "Parameter", "Optimal", "Std Dev", "Prior Bounds")
    println("-"^80)
    
    for i in 1:N_params
        if !isnothing(prior_bounds) && haskey(prior_bounds, param_names[i])
            bounds = prior_bounds[param_names[i]]
            @printf("%-25s | %15.6g | %12.6g | [%.6g, %.6g]\n", 
                    param_names[i], θ_optimal[i], θ_std[i], bounds[1], bounds[2])
        else
            @printf("%-25s | %15.6g | %12.6g | %20s\n", 
                    param_names[i], θ_optimal[i], θ_std[i], "N/A")
        end
    end
    println("-"^80)
    
    # Physical target statistics — read from checkpoint's y_obs_full / uncertainties_full
    latest_file, _ = get_latest_checkpoint(output_dir)
    cp = load_checkpoint_for_animation(latest_file)
    if haskey(cp, "y_obs_full") && haskey(cp, "uncertainties_full")
        y_full = cp["y_obs_full"] .* cp["uncertainties_full"]
        n_pdf  = get(cp, "n_pdf", length(y_full) - 2)   # fallback
        # n_pdf may not be stored in checkpoint; infer from length
        n_pdf = length(y_full) - 2
        wt_phys = y_full[n_pdf + 1]
        sd_phys = y_full[n_pdf + 2]
        println("\nTarget dynamical statistics (physical units):")
        println("-"^80)
        @printf("%-25s | %15.3f years\n", "Avg waiting time",       wt_phys)
        @printf("%-25s | %15.3f years\n", "Avg stadial duration",   sd_phys)
        println("-"^80)
    end
end

# ============================================
# UNCERTAINTY DIAGNOSTICS
# ============================================

function print_iteration_uncertainties(output_dir; n_pca=5)
    """
    Print observation uncertainties for each iteration.
    Also prints the block-level uncertainty analysis if available.
    """

    # Block analysis (computed once at startup)
    block_file = joinpath(output_dir, "block_uncertainty_analysis.jld2")
    if isfile(block_file)
        @load block_file result
        println("="^70)
        println("BLOCK UNCERTAINTY ANALYSIS")
        println("="^70)
        println("Block size: $(result["block_size"]) years  |  " *
                "N blocks: $(result["n_blocks"])  |  " *
                "Valid: $(length(result["valid_blocks"]))")
        println()
        println("Per-block DO event counts:")
        for b in 1:result["n_blocks"]
            flag = b in result["valid_blocks"] ? "" : "  ← excluded"
            @printf("  Block %2d  (%6.0f – %6.0f yr):  %d DO events%s\n",
                    b, result["block_start_times"][b], result["block_end_times"][b],
                    result["block_n_do"][b], flag)
        end
        println()
        @printf("  PDF uncertainty (mean per grid pt): %.6f\n", mean(result["pdf_uncertainty"]))
        @printf("  PDF uncertainty (max  per grid pt): %.6f\n", maximum(result["pdf_uncertainty"]))
        @printf("  Waiting time uncertainty:           %.1f years\n", result["wt_uncertainty"])
        @printf("  Stadial duration uncertainty:       %.1f years\n", result["sd_uncertainty"])
    else
        println("Block uncertainty file not found: $block_file")
    end

    # Per-iteration PCA uncertainties
    latest_file, latest_iter = get_latest_checkpoint(output_dir)
    labels = vcat(["PCA $k" for k in 1:n_pca], ["WaitingTime", "StadialDur"])

    println()
    println("="^70)
    println("PCA OBSERVATION UNCERTAINTIES PER ITERATION")
    println("="^70)
    @printf("  %-12s", "Component")
    for iter in 1:latest_iter
        @printf("  %10s", "Iter $iter")
    end
    println()
    println("  " * "-"^(12 + 12*latest_iter))

    # Collect all uncertainties_pca first
    all_unc = Vector{Union{Nothing, Vector{Float64}}}(nothing, latest_iter)
    for iter in 1:latest_iter
        iter_file = joinpath(output_dir, "iteration_$(iter)_results.jld2")
        isfile(iter_file) || continue
        @load iter_file iter_data
        all_unc[iter] = get(iter_data, "uncertainties_pca", nothing)
    end

    for (k, label) in enumerate(labels)
        @printf("  %-12s", label)
        for iter in 1:latest_iter
            u = all_unc[iter]
            if !isnothing(u) && k <= length(u)
                @printf("  %10.4g", u[k])
            else
                @printf("  %10s", "—")
            end
        end
        println()
    end
    println()
end

# ============================================
# PCA RESIDUAL CONVERGENCE
# ============================================

function plot_pca_residuals_convergence(output_dir; n_pca=5)
    """
    Plot mean and RMS of normalised residuals per PCA component and
    dynamical stat over iterations.  RMS ≈ 1 means the ensemble spread
    equals the assumed uncertainty for that component.
    """

    labels = vcat(["PCA $k" for k in 1:n_pca], ["WaitingTime", "StadialDur"])
    n_obs  = length(labels)

    latest_file, latest_iter = get_latest_checkpoint(output_dir)
    iters_found = Int[]
    all_mean    = Vector{Vector{Float64}}()
    all_rms     = Vector{Vector{Float64}}()

    for iter in 1:latest_iter
        iter_file = joinpath(output_dir, "iteration_$(iter)_results.jld2")
        isfile(iter_file) || continue
        @load iter_file iter_data
        res = get(iter_data, "pca_residuals", nothing)
        isnothing(res) && continue
        push!(iters_found, iter)
        push!(all_mean, vec(mean(res, dims=2)))
        push!(all_rms,  vec(sqrt.(mean(res .^ 2, dims=2))))
    end

    if isempty(iters_found)
        @warn "No pca_residuals found in iteration results"
        return nothing
    end

    mean_mat = hcat(all_mean...)   # (n_obs × n_iters)
    rms_mat  = hcat(all_rms...)

    n_cols = min(3, n_obs)
    n_rows = Int(ceil(n_obs / n_cols))
    p = plot(layout=(n_rows, n_cols), size=(420*n_cols, 280*n_rows),
             plot_title="Normalised residuals per component")

    for k in 1:n_obs
        plot!(p[k], iters_found, rms_mat[k, :],
              color=:blue, linewidth=2, marker=:circle, label="RMS")
        plot!(p[k], iters_found, abs.(mean_mat[k, :]),
              color=:orange, linewidth=2, marker=:diamond, linestyle=:dash, label="|Mean|")
        hline!(p[k], [1.0], color=:gray, linestyle=:dot, label="σ=1")
        ylabel!(p[k], labels[k])
        xlabel!(p[k], "Iteration")
    end

    display(p)
    println("✓ PCA residual convergence plot displayed!")
    return p
end

# ============================================
# ENSEMBLE PDF OVERLAY
# ============================================

function plot_ensemble_pdfs_per_iteration(output_dir; iterations=nothing)
    """
    For each requested iteration plot all ensemble PDFs against the target.
    Physical units throughout.  Grey lines = ensemble members, red = target,
    shaded band = ±1σ from block uncertainty analysis.
    """

    latest_file, latest_iter = get_latest_checkpoint(output_dir)
    iters = isnothing(iterations) ? (1:latest_iter) : iterations

    # Target PDF from checkpoint
    cp = load_checkpoint_for_animation(latest_file)
    y_full  = cp["y_obs_full"] .* cp["uncertainties_full"]
    n_pdf   = length(y_full) - 2
    pdf_target = y_full[1:n_pdf]

    # PDF uncertainty from block analysis (for shaded band)
    pdf_sigma = nothing
    block_file = joinpath(output_dir, "block_uncertainty_analysis.jld2")
    if isfile(block_file)
        @load block_file result
        pdf_sigma = result["pdf_uncertainty"]
    end

    plots_out = []
    for iter in iters
        iter_file = joinpath(output_dir, "iteration_$(iter)_results.jld2")
        isfile(iter_file) || continue
        @load iter_file iter_data

        pdf_grid   = iter_data["pdf_grid"]
        pdfs_phys  = iter_data["pdfs_physical"]   # n_pdf × N_ensemble

        p = plot(title="Ensemble PDFs — iteration $iter",
                 xlabel="AMOC (Sv)", ylabel="Density",
                 legend=:topright, size=(700, 400))

        # Ensemble members (thin grey)
        for j in 1:size(pdfs_phys, 2)
            any(isnan.(pdfs_phys[:, j])) && continue
            plot!(p, pdf_grid, pdfs_phys[:, j],
                  color=:grey, alpha=0.25, linewidth=0.8, label="")
        end

        # ±1σ band from block uncertainty
        if !isnothing(pdf_sigma)
            plot!(p, pdf_grid, pdf_target .+ pdf_sigma,
                  fillrange=pdf_target .- pdf_sigma,
                  fillalpha=0.2, fillcolor=:red, linewidth=0, label="Target ±1σ")
        end

        # Target PDF
        plot!(p, pdf_grid, pdf_target,
              color=:red, linewidth=2.5, label="Target")

        # Ensemble mean
        valid_cols = [j for j in 1:size(pdfs_phys,2) if !any(isnan.(pdfs_phys[:,j]))]
        if !isempty(valid_cols)
            pdf_mean = vec(mean(pdfs_phys[:, valid_cols], dims=2))
            plot!(p, pdf_grid, pdf_mean,
                  color=:blue, linewidth=2, linestyle=:dash, label="Ensemble mean")
        end

        display(p)
        push!(plots_out, p)
    end

    println("✓ Ensemble PDF plots displayed!")
    return plots_out
end

# ============================================
# SUMMARY STATS SCATTER / VIOLIN PER ITERATION
# ============================================

function plot_summary_stats_scatter(output_dir)
    """
    Box plots of waiting time and stadial duration across ensemble members
    for each iteration (physical units).  Target value shown as horizontal line.
    """

    latest_file, latest_iter = get_latest_checkpoint(output_dir)
    cp = load_checkpoint_for_animation(latest_file)

    y_full  = cp["y_obs_full"] .* cp["uncertainties_full"]
    n_pdf   = length(y_full) - 2
    wt_target = y_full[n_pdf + 1]
    sd_target = y_full[n_pdf + 2]

    iter_labels = String[]
    all_wt = Vector{Vector{Float64}}()
    all_sd = Vector{Vector{Float64}}()

    for iter in 1:latest_iter
        iter_file = joinpath(output_dir, "iteration_$(iter)_results.jld2")
        isfile(iter_file) || continue
        @load iter_file iter_data
        wt = filter(!isnan, iter_data["waiting_times"])
        sd = filter(!isnan, iter_data["stadial_durations"])
        isempty(wt) && continue
        push!(iter_labels, "Iter $iter")
        push!(all_wt, wt)
        push!(all_sd, sd)
    end

    if isempty(all_wt)
        @warn "No iteration data found"
        return nothing
    end

    p1 = plot(title="Waiting time per iteration",
              ylabel="Years", xlabel="", legend=false)
    p2 = plot(title="Stadial duration per iteration",
              ylabel="Years", xlabel="", legend=false)

    for (k, label) in enumerate(iter_labels)
        boxplot!(p1, [label], all_wt[k], fillalpha=0.5, color=:steelblue,
                 outliers=true, whisker_width=0.5)
        boxplot!(p2, [label], all_sd[k], fillalpha=0.5, color=:steelblue,
                 outliers=true, whisker_width=0.5)
    end

    hline!(p1, [wt_target], color=:red, linewidth=2, linestyle=:dash, label="Target")
    hline!(p2, [sd_target], color=:red, linewidth=2, linestyle=:dash, label="Target")

    p = plot(p1, p2, layout=(1, 2), size=(900, 400))
    display(p)
    println("✓ Summary statistics scatter plot displayed!")
    return p
end

# ============================================
# PARAMETER CONVERGENCE (STATIC)
# ============================================

function plot_parameter_convergence_static(output_dir; param_names=nothing, prior_bounds=nothing)
    """
    Static plot of parameter mean ± std over iterations (no animation).
    """

    latest_file, latest_iter = get_latest_checkpoint(output_dir)
    checkpoint_data = load_checkpoint_for_animation(latest_file)
    param_history = checkpoint_data["param_history"]

    N_params     = size(param_history, 1)
    N_iterations = size(param_history, 2) - 1

    if isnothing(param_names)
        param_names = ["Param $i" for i in 1:N_params]
    end

    iters = 0:N_iterations
    n_cols = min(3, N_params)
    n_rows = Int(ceil(N_params / n_cols))
    p = plot(layout=(n_rows, n_cols), size=(420*n_cols, 280*n_rows),
             plot_title="Parameter convergence")

    for idx in 1:N_params
        μ = vec(mean(param_history[idx, :, :], dims=2))
        σ = vec(std(param_history[idx, :, :],  dims=2))

        plot!(p[idx], iters, μ .+ σ,
              fillrange=μ .- σ, fillalpha=0.25, fillcolor=:blue,
              linewidth=0, label="±1σ")
        plot!(p[idx], iters, μ,
              color=:blue, linewidth=2, label="Mean")

        if !isnothing(prior_bounds) && haskey(prior_bounds, param_names[idx])
            lo, hi = prior_bounds[param_names[idx]]
            hline!(p[idx], [lo, hi],
                   color=:gray, linestyle=:dash, linewidth=1, label="Prior")
        end

        ylabel!(p[idx], param_names[idx])
        xlabel!(p[idx], "Iteration")
    end

    display(p)
    println("✓ Parameter convergence plot displayed!")
    return p
end

# ============================================
# WALL CLOCK TIME ANALYSIS
# ============================================

"""
Load wall clock timing fields from every iteration result file found in output_dir.
Returns a NamedTuple with:
  - iters               : Vector{Int}    — iteration numbers found
  - iter_hours          : Vector{Float64} — total iteration wall time in hours (NaN if missing)
  - member_hours        : Vector{Vector{Float64}} — per-member times in hours (NaN for failed jobs)
"""
function _load_wall_clock_data(output_dir)
    latest_file, latest_iter = get_latest_checkpoint(output_dir)

    iters        = Int[]
    iter_hours   = Float64[]
    member_hours = Vector{Vector{Float64}}()

    for iter in 1:latest_iter
        iter_file = joinpath(output_dir, "iteration_$(iter)_results.jld2")
        isfile(iter_file) || continue
        @load iter_file iter_data

        iter_s   = get(iter_data, "iter_duration_seconds",     NaN)
        member_s = get(iter_data, "member_wall_times_seconds", Float64[])

        push!(iters,        iter)
        push!(iter_hours,   iter_s   / 3600.0)
        push!(member_hours, member_s ./ 3600.0)
    end

    return (iters=iters, iter_hours=iter_hours, member_hours=member_hours)
end

"""
Print a table of wall clock times per iteration.
Columns: iteration | total (h) | N valid | mean (h) | min (h) | max (h) | std (h)
"""
function print_wall_clock_summary(output_dir)
    d = _load_wall_clock_data(output_dir)

    if isempty(d.iters)
        @warn "No timing data found in $output_dir"
        return
    end

    println("\n" * "="^80)
    println("WALL CLOCK TIME SUMMARY")
    println("="^80)
    @printf("%-6s | %10s | %7s | %10s | %10s | %10s | %10s\n",
            "Iter", "Total (h)", "N valid", "Mean (h)", "Min (h)", "Max (h)", "Std (h)")
    println("-"^80)

    for (k, iter) in enumerate(d.iters)
        total_h  = d.iter_hours[k]
        mh       = filter(!isnan, d.member_hours[k])
        n_valid  = length(mh)
        mean_h   = isempty(mh) ? NaN : mean(mh)
        min_h    = isempty(mh) ? NaN : minimum(mh)
        max_h    = isempty(mh) ? NaN : maximum(mh)
        std_h    = length(mh) < 2 ? NaN : std(mh)

        total_str = isnan(total_h) ? "       N/A" : @sprintf("%10.2f", total_h)
        @printf("%-6d | %s | %7d | %10.2f | %10.2f | %10.2f | %10.2f\n",
                iter, total_str, n_valid, mean_h, min_h, max_h, std_h)
    end
    println("-"^80)

    # Overall totals across all iterations
    valid_totals = filter(!isnan, d.iter_hours)
    if !isempty(valid_totals)
        @printf("%-6s | %10.2f | %7s | %10s | %10s | %10s | %10s\n",
                "Total", sum(valid_totals), "", "", "", "", "")
    end
    println("="^80)
end

"""
Plot wall clock times across iterations.
Panel 1: total iteration wall time per iteration (bar chart).
Panel 2: box plots of per-member wall times per iteration.
"""
function plot_wall_clock_times(output_dir)
    d = _load_wall_clock_data(output_dir)

    if isempty(d.iters)
        @warn "No timing data found in $output_dir"
        return nothing
    end

    iter_labels = ["Iter $(i)" for i in d.iters]

    # ── Panel 1: total iteration time ───────────────────────────────────────
    p1 = bar(iter_labels, d.iter_hours,
             title="Total iteration wall time",
             ylabel="Hours", xlabel="",
             legend=false, color=:steelblue, alpha=0.8,
             bar_width=0.6, framestyle=:box)

    # Annotate each bar with its value
    for (k, h) in enumerate(d.iter_hours)
        isnan(h) && continue
        annotate!(p1, k, h + 0.02 * maximum(filter(!isnan, d.iter_hours)),
                  text(@sprintf("%.1fh", h), :center, 8))
    end

    # ── Panel 2: per-member box plots ───────────────────────────────────────
    p2 = plot(title="Per-member wall time distribution",
              ylabel="Hours", xlabel="",
              legend=false, framestyle=:box)

    for (k, (label, mh)) in enumerate(zip(iter_labels, d.member_hours))
        valid_mh = filter(!isnan, mh)
        isempty(valid_mh) && continue
        boxplot!(p2, [label], valid_mh,
                 fillalpha=0.5, color=:steelblue,
                 outliers=true, whisker_width=0.5)
    end

    # Mean line across all members and iterations for reference
    all_member_h = filter(!isnan, vcat(d.member_hours...))
    if !isempty(all_member_h)
        hline!(p2, [mean(all_member_h)],
               color=:red, linestyle=:dash, linewidth=1.5, label="Overall mean")
    end

    p = plot(p1, p2, layout=(2, 1), size=(800, 600))
    display(p)
    println("✓ Wall clock time plots displayed!")
    return p
end

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

function visualize_all(output_dir, default_file; 
                       param_names=nothing, 
                       prior_bounds=nothing,
                       fps=3,
                       skip_animations=false,
                       plot_amoc_iterations=nothing,
                       n_amoc_members=10,
                       climber_summary_stats_file="climber_summary_stats.jl")
    """
    Generate all visualizations for CLIMBER-X EKI results
    
    Arguments:
    - output_dir: Path to EKI output directory
    - default_file: Path to default run NetCDF file
    - param_names: Array of parameter names
    - prior_bounds: Dictionary of prior bounds
    - fps: Frames per second for animations
    - skip_animations: Skip animations if true
    - plot_amoc_iterations: Array of iterations to plot AMOC for (e.g., [1, 3, 6])
    - n_amoc_members: Number of ensemble members to plot per iteration
    - climber_summary_stats_file: Path to climber_summary_stats.jl file
    """
    
    println("\n" * "="^80)
    println("VISUALIZING CLIMBER-X EKI RESULTS")
    println("="^80)
    println("Output directory: $output_dir")
    println("="^80)
    
    # Calibration summary table
    print_calibration_summary(output_dir; param_names=param_names, prior_bounds=prior_bounds)

    # Uncertainty diagnostics (printed, no plot)
    println("\n--- Uncertainty diagnostics ---")
    print_iteration_uncertainties(output_dir)

    # Wall clock timing
    println("\n--- Wall clock time analysis ---")
    print_wall_clock_summary(output_dir)
    plot_wall_clock_times(output_dir)

    # Static convergence plots
    println("\nGenerating static plots...")
    plot_parameter_convergence_static(output_dir; param_names=param_names, prior_bounds=prior_bounds)
    plot_summary_stats_convergence(output_dir)
    plot_pca_residuals_convergence(output_dir)
    plot_summary_stats_scatter(output_dir)

    # Ensemble PDF overlay for requested iterations
    if !isnothing(plot_amoc_iterations)
        println("\nPlotting ensemble PDF overlays...")
        plot_ensemble_pdfs_per_iteration(output_dir; iterations=plot_amoc_iterations)
    end

    # Parameter distributions for specific iterations
    if !isnothing(plot_amoc_iterations)
        for iter in plot_amoc_iterations
            println("\nPlotting parameter distributions for iteration $iter...")
            plot_parameter_distributions_per_iteration(output_dir;
                param_names=param_names,
                prior_bounds=prior_bounds,
                iteration=iter)
        end
    end

    # AMOC timeseries for specific iterations
    if !isnothing(plot_amoc_iterations)
        for iter in plot_amoc_iterations
            println("\nPlotting AMOC diagnostics for iteration $iter...")
            plot_amoc_ensemble_iteration(output_dir, default_file, iter;
                n_members=n_amoc_members,
                climber_summary_stats_file=climber_summary_stats_file)
        end
    end

    # Animations
    if !skip_animations
        println("\nGenerating animations (this may take a while)...")
        create_pairwise_convergence_animation(output_dir; fps=fps, param_names=param_names)
        create_parameter_evolution_animation(output_dir; fps=fps, param_names=param_names)
        plot_parameter_distributions_per_iteration(output_dir; param_names=param_names,
                                                   prior_bounds=prior_bounds)
    else
        println("\nSkipping animations (skip_animations=true)")
    end

    println("\n✓ All visualizations complete!")
end

# ============================================
# USAGE EXAMPLE
# ============================================

# Define parameter names and bounds for CLIMBER-X calibration
CLIMBER_PARAM_NAMES = [
    "diff_dia_min",
    "drag_topo_fac", 
    "slope_max",
    "diff_iso",
    "diff_gm",
    "diff_dia_max"
]

CLIMBER_PRIOR_BOUNDS = Dict(
    "diff_dia_min" => (7.5e-6, 1.25e-5),
    "drag_topo_fac" => (2.25, 3.75),
    "slope_max" => (7.5e-4, 1.25e-3),
    "diff_iso" => (1125.0, 1875.0),
    "diff_gm" => (1125.0, 1875.0),
    "diff_dia_max" => (1.125e-4, 1.875e-4)
)

# Run all visualizations
output_dir = "/p/tmp/karinako/eki_calibration_7000/output"
default_file = "/p/tmp/karinako/default_run_long/0/ocn_ts.nc"

# Visualize iterations 1, 3, and 6 with AMOC timeseries for 10 members each
visualize_all(output_dir, default_file;
              param_names=CLIMBER_PARAM_NAMES,
              prior_bounds=CLIMBER_PRIOR_BOUNDS,
              fps=3,
              skip_animations=false,
              plot_amoc_iterations=[1, 3, 6],
              n_amoc_members=10,
              climber_summary_stats_file="climber_summary_stats.jl")
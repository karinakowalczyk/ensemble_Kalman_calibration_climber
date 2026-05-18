using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Statistics
using JLD2
using Dates

# ============================================
# JOB MANAGEMENT SYSTEM
# ============================================

"""
Structure to track job status
"""
mutable struct JobTracker
    job_id::String
    member_id::Int
    iteration::Int
    status::Symbol  # :submitted, :running, :completed, :failed, :timeout, :oom
    submit_time::DateTime
    completion_time::Union{DateTime, Nothing}
    param_file::String
    output_file::String
end

"""
Check job status with retry logic
"""
function check_job_status(job_id; max_retries=3, initial_delay=5)
    for attempt in 1:max_retries
        try
            output = read(`sacct -j $(job_id) -o State -n --parsable2`, String)
            
            if isempty(strip(output))
                if attempt < max_retries
                    delay = initial_delay * (2 ^ (attempt - 1))
                    sleep(delay)
                    continue
                else
                    return :unknown
                end
            end
            
            status_line = strip(split(output, "\n")[1])
            
            if occursin("COMPLETED", status_line)
                return :completed
            elseif occursin("RUNNING", status_line)
                return :running
            elseif occursin("PENDING", status_line)
                return :submitted
            elseif occursin("TIMEOUT", status_line)
                return :timeout
            elseif occursin("OUT_OF_MEMORY", status_line)
                return :oom
            elseif occursin("FAILED", status_line) || occursin("CANCELLED", status_line)
                return :failed
            else
                return :unknown
            end
            
        catch e
            if attempt < max_retries
                delay = initial_delay * (2 ^ (attempt - 1))
                @warn "Failed to check job $job_id (attempt $attempt/$max_retries). Retrying in $delay seconds..."
                sleep(delay)
            else
                @warn "Failed to check job $job_id after $max_retries attempts: $e"
                return :unknown
            end
        end
    end
end

"""
Submit a single job with retry logic
"""
function submit_job(script_file; max_retries=3)
    for attempt in 1:max_retries
        try
            job_id = strip(read(`sbatch --parsable $(script_file)`, String))
            
            sleep(2)
            status = check_job_status(job_id, max_retries=2)
            
            if status != :unknown
                return job_id
            else
                @warn "Job submission verification failed (attempt $attempt/$max_retries)"
                if attempt < max_retries
                    sleep(5)
                end
            end
            
        catch e
            @warn "Failed to submit job (attempt $attempt/$max_retries): $e"
            if attempt < max_retries
                sleep(5)
            end
        end
    end
    
    error("Failed to submit job after $max_retries attempts: $script_file")
end

"""
Check available disk space
"""
function check_disk_space(path; min_gb_required=100, warn_gb=200)
    try
        df_output = read(`df -BG $(path)`, String)
        lines = split(df_output, "\n")
        
        if length(lines) >= 2
            fields = split(lines[2])
            available_str = fields[4]
            available_gb = parse(Int, replace(available_str, "G" => ""))
            
            if available_gb < min_gb_required
                @error "CRITICAL: Only $(available_gb)GB available on $path (minimum: $(min_gb_required)GB)"
                return false, available_gb
            elseif available_gb < warn_gb
                @warn "Low disk space: $(available_gb)GB available on $path"
                return true, available_gb
            else
                println("  Disk space OK: $(available_gb)GB available")
                return true, available_gb
            end
        else
            @warn "Could not parse df output"
            return true, -1
        end
        
    catch e
        @warn "Could not check disk space: $e"
        return true, -1
    end
end

"""
Wait for all jobs in an iteration to complete
"""
function wait_for_iteration_completion(job_trackers;
                                       check_interval_minutes=30,
                                       max_wait_days=10,
                                       output_dir=nothing,
                                       max_unknown_checks=3)

    println("\n  Waiting for jobs to complete...")
    println("  Checking every $check_interval_minutes minutes")

    start_time = now()
    max_wait = Dates.Day(max_wait_days)
    # Per-tracker count of consecutive :unknown SLURM responses
    unknown_counts = Dict(tracker.job_id => 0 for tracker in job_trackers)

    while true
        for tracker in job_trackers
            if tracker.status in [:submitted, :running]
                # Ground-truth check: valid output file means the run finished,
                # regardless of what SLURM reports.
                if validate_climber_output_file(tracker.output_file)[1]
                    tracker.status = :completed
                    tracker.completion_time = now()
                    println("    Member $(tracker.member_id): output file valid — marking completed")
                    continue
                end

                new_status = check_job_status(tracker.job_id, max_retries=3, initial_delay=5)

                if new_status == :completed
                    # SLURM says COMPLETED but the file check above failed —
                    # the run likely diverged and exited cleanly without producing output.
                    tracker.status = :failed
                    tracker.completion_time = now()
                    @warn "Member $(tracker.member_id): SLURM COMPLETED but output invalid — treating as failed"

                elseif new_status in [:failed, :timeout, :oom, :cancelled]
                    tracker.status = new_status
                    tracker.completion_time = now()
                    @warn "Member $(tracker.member_id) SLURM status: $new_status"

                elseif new_status == :running && tracker.status == :submitted
                    tracker.status = :running

                elseif new_status == :unknown
                    unknown_counts[tracker.job_id] += 1
                    if unknown_counts[tracker.job_id] >= max_unknown_checks
                        tracker.status = :failed
                        tracker.completion_time = now()
                        @warn "Member $(tracker.member_id): SLURM status unknown for $max_unknown_checks consecutive checks — treating as failed (likely diverged)"
                    end
                end
            end
        end

        n_completed = count(t -> t.status == :completed, job_trackers)
        n_failed    = count(t -> t.status in [:failed, :timeout, :oom, :cancelled], job_trackers)
        n_pending   = count(t -> t.status in [:submitted, :running], job_trackers)

        println("    [$(now())] Status: $n_completed completed, $n_failed failed, $n_pending pending")

        if output_dir !== nothing
            check_disk_space(output_dir, min_gb_required=50, warn_gb=100)
        end

        if n_pending == 0
            if n_failed == 0
                println("  ✓ All jobs completed successfully!")
                return :success
            else
                @warn "Jobs finished with failures: $n_completed completed, $n_failed failed"
                return :partial_failure
            end
        end

        elapsed = now() - start_time
        if elapsed > max_wait
            @warn "Maximum wait time exceeded ($max_wait_days days)"
            return :timeout
        end

        sleep(check_interval_minutes * 60)
    end
end

"""
Save job tracker information
"""
function save_job_trackers(job_trackers, iteration, output_dir)
    tracker_file = joinpath(output_dir, "job_tracking", "iter_$(iteration)_trackers.jld2")
    mkpath(dirname(tracker_file))
    
    @save tracker_file job_trackers
    
    log_file = joinpath(output_dir, "job_tracking", "iter_$(iteration)_log.txt")
    open(log_file, "w") do f
        println(f, "Iteration $iteration Job Tracking")
        println(f, "="^60)
        println(f, "Timestamp: $(now())")
        println(f, "")
        
        for tracker in job_trackers
            println(f, "Member $(tracker.member_id):")
            println(f, "  Job ID: $(tracker.job_id)")
            println(f, "  Status: $(tracker.status)")
            println(f, "  Submitted: $(tracker.submit_time)")
            if tracker.completion_time !== nothing
                println(f, "  Completed: $(tracker.completion_time)")
                duration = tracker.completion_time - tracker.submit_time
                println(f, "  Duration: $(duration)")
            end
            println(f, "  Output: $(tracker.output_file)")
            println(f, "")
        end
    end
end

# Helper save functions
function save_iteration_results(i, params, G_ensemble, mean_vals, std_vals, output_dir)
    results_file = joinpath(output_dir, "iteration_results.jld2")
    
    if isfile(results_file)
        @load results_file all_results
    else
        all_results = Dict()
    end
    
    all_results[i] = Dict(
        "params" => params,
        "G_ensemble" => G_ensemble,
        "mean" => mean_vals,
        "std" => std_vals,
        "timestamp" => now()
    )
    
    @save results_file all_results
end

function save_checkpoint(i, eksobj, prior, param_history, y_obs, obs_noise_cov, metadata, checkpoint_dir)
    checkpoint_file = joinpath(checkpoint_dir, "checkpoint_iter_$(i).jld2")
    
    checkpoint_data = Dict(
        "iteration" => i,
        "eksobj" => eksobj,
        "prior" => prior,
        "param_history" => param_history,
        "y_obs" => y_obs,
        "obs_noise_cov" => obs_noise_cov,
        "metadata" => metadata,
        "timestamp" => now()
    )
    
    @save checkpoint_file checkpoint_data
    println("  ✓ Checkpoint saved: $checkpoint_file")
end

function save_final_results(θ_optimal, θ_std, final_ensemble, y_obs, metadata, output_dir)
    final_file = joinpath(output_dir, "final_results.jld2")
    
    final_data = Dict(
        "θ_optimal" => θ_optimal,
        "θ_std" => θ_std,
        "final_ensemble" => final_ensemble,
        "y_obs" => y_obs,
        "metadata" => metadata,
        "timestamp" => now()
    )
    
    @save final_file final_data
    println("Final results saved: $final_file")
end
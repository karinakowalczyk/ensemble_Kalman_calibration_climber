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
Resubmit a failed/stalled member's job, in place on its tracker.

`resubmit_fn(member_id) -> (new_job_id, new_output_file)` does the actual
submission (CLIMBER-specific, supplied by the caller); this function just
owns the retry bookkeeping and tracker mutation that's common to every kind
of failure (stalled output, SLURM failure/timeout/oom, persistently unknown
status). Returns `true` if a resubmission was issued, `false` if the member
was instead marked permanently `:failed` (no resubmit_fn given, or retries
exhausted).
"""
function attempt_resubmit!(tracker, reason, retry_counts, resubmit_fn, max_retries_per_member)
    if resubmit_fn === nothing || retry_counts[tracker.member_id] >= max_retries_per_member
        tracker.status = :failed
        tracker.completion_time = now()
        @warn "Member $(tracker.member_id): $reason — giving up (retries: $(retry_counts[tracker.member_id])/$max_retries_per_member)"
        return false
    end

    retry_counts[tracker.member_id] += 1
    # Clear the stale/partial output so a fresh run starts clean and the
    # year-count staleness check next cycle can't be confused by leftover
    # data from the crashed attempt.
    stale_dir = dirname(tracker.output_file)
    isdir(stale_dir) && rm(stale_dir; recursive=true, force=true)

    new_job_id, new_output_file = resubmit_fn(tracker.member_id)
    tracker.job_id = new_job_id
    tracker.output_file = new_output_file
    tracker.status = :submitted
    tracker.submit_time = now()
    tracker.completion_time = nothing
    @warn "Member $(tracker.member_id): $reason — resubmitted as job $new_job_id (attempt $(retry_counts[tracker.member_id])/$max_retries_per_member)"
    return true
end

function wait_for_iteration_completion(job_trackers;
                                       check_interval_minutes=30,
                                       max_wait_days=10,
                                       output_dir=nothing,
                                       max_unknown_checks=3,
                                       expected_nyears=nothing,
                                       resubmit_fn=nothing,
                                       max_retries_per_member=1)

    println("\n  Waiting for jobs to complete...")
    println("  Checking every $check_interval_minutes minutes")

    start_time = now()
    max_wait = Dates.Day(max_wait_days)
    # Keyed by member_id (not job_id): a resubmission gives a member a new
    # job_id, so job_id-keyed bookkeeping would silently reset/orphan itself
    # across a retry.
    unknown_counts = Dict(tracker.member_id => 0 for tracker in job_trackers)
    # Year-count last seen while SLURM already reported :completed -- nothing
    # (not yet observed) vs an actual count. Once SLURM confirms the writing
    # process has exited, an unchanged count between two checks proves the
    # run stalled/diverged early (nothing left alive to write more), rather
    # than just being a transient filesystem write/flush lag.
    last_years_written = Dict{Int, Union{Int, Nothing}}(tracker.member_id => nothing for tracker in job_trackers)
    retry_counts = Dict(tracker.member_id => 0 for tracker in job_trackers)

    while true
        for tracker in job_trackers
            if tracker.status in [:submitted, :running]
                # SLURM's own accounting is the authoritative "has this job's process
                # exited" signal -- ask it first, rather than trying to independently
                # re-derive completion from the filesystem. Only once SLURM says the
                # job is done do we check the output file, purely to catch the case
                # of a job that exited "successfully" without actually finishing the
                # real simulation (e.g. silent early divergence). If the file isn't
                # valid yet, we just leave the tracker as-is and let the next poll
                # cycle retry; max_wait_days remains the overall backstop, but the
                # year-count comparison below usually catches a genuinely stalled
                # run much sooner than that.
                new_status = check_job_status(tracker.job_id, max_retries=3, initial_delay=5)

                if new_status == :completed
                    is_valid, valid_msg = validate_climber_output_file(tracker.output_file; expected_nyears=expected_nyears)
                    if is_valid
                        tracker.status = :completed
                        tracker.completion_time = now()
                        println("    Member $(tracker.member_id): SLURM completed and output valid")
                    else
                        n_years = get_n_years_written(tracker.output_file)
                        prev_years = last_years_written[tracker.member_id]
                        if n_years >= 0 && prev_years !== nothing && n_years == prev_years
                            attempt_resubmit!(tracker, "SLURM completed, output stuck at $n_years years (unchanged since last check)",
                                               retry_counts, resubmit_fn, max_retries_per_member)
                            last_years_written[tracker.member_id] = nothing
                            unknown_counts[tracker.member_id] = 0
                        else
                            if n_years >= 0
                                last_years_written[tracker.member_id] = n_years
                            end
                            @warn "Member $(tracker.member_id): SLURM completed but output not yet valid ($valid_msg) — will recheck next cycle"
                        end
                    end

                elseif new_status in [:failed, :timeout, :oom, :cancelled]
                    attempt_resubmit!(tracker, "SLURM status: $new_status", retry_counts, resubmit_fn, max_retries_per_member)
                    last_years_written[tracker.member_id] = nothing
                    unknown_counts[tracker.member_id] = 0

                elseif new_status == :running && tracker.status == :submitted
                    tracker.status = :running

                elseif new_status == :unknown
                    unknown_counts[tracker.member_id] += 1
                    if unknown_counts[tracker.member_id] >= max_unknown_checks
                        attempt_resubmit!(tracker, "SLURM status unknown for $max_unknown_checks consecutive checks (likely diverged)",
                                           retry_counts, resubmit_fn, max_retries_per_member)
                        last_years_written[tracker.member_id] = nothing
                        unknown_counts[tracker.member_id] = 0
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
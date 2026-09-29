# One-off repair of checkpoint_iter_0.jld2 for the 75 kyr EKS run.
#
# Background: after the first driver crash (Sep 2026) the relaunch found only
# checkpoint_iter_0, which did not trigger the resume prompt, so a fresh random
# initial ensemble was drawn and checkpoint_iter_0 was overwritten. 81 of the 100
# iter_1 outputs had already been produced by the ORIGINAL draw and were reused,
# so the checkpoint's parameters no longer match the outputs.
#
# Fix: rebuild the initial ensemble from the parameters each member actually ran
# with (read from iter_1/member_j/ocn_par.nml), and rewrite checkpoint_iter_0 with
# an EKS object built from those. Everything else (prior, y_obs, obs_noise_cov,
# uncertainties, ...) is kept from the existing checkpoint. The original file is
# backed up first.
#
# Usage (on the HPC, from the directory containing this script):
#   julia repair_checkpoint_iter0.jl                # dry run: checks only, writes nothing
#   julia repair_checkpoint_iter0.jl --write        # backs up and rewrites checkpoint_iter_0

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Distributions
using JLD2
using Printf

const OUTPUT_DIR = "/p/tmp/karinako/eki_calibration_75ky/output"
const ITERATION_DIR = joinpath(OUTPUT_DIR, "iter_1")
const CHECKPOINT_FILE = joinpath(OUTPUT_DIR, "checkpoints", "checkpoint_iter_0.jld2")
const BACKUP_FILE = CHECKPOINT_FILE * ".bak_sep21"

# Must match climber_x_calibration.jl exactly (checked against the checkpoint below).
const PARAM_NAMES = ["diff_dia_min", "drag_topo_fac", "slope_max", "diff_iso", "diff_gm", "diff_dia_max"]
const PRIOR_BOUNDS = Dict(
    "diff_dia_min" => (5.0e-6,  2.0e-5),
    "drag_topo_fac" => (2.5,    3.5),
    "slope_max"    => (5.0e-4,  2.0e-3),
    "diff_iso"     => (500.0,   2000.0),
    "diff_gm"      => (500.0,   2000.0),
    "diff_dia_max" => (1.0e-4,  2.0e-4)
)

normalise_param(θ, name)   = (θ - PRIOR_BOUNDS[name][1]) / (PRIOR_BOUNDS[name][2] - PRIOR_BOUNDS[name][1])
denormalise_param(p, name) = PRIOR_BOUNDS[name][1] + p * (PRIOR_BOUNDS[name][2] - PRIOR_BOUNDS[name][1])

function nml_value(file, name)
    for line in eachline(file)
        m = match(Regex("^\\s*$(name)\\s*=\\s*([-+0-9.eEdD]+)"), line)
        m === nothing || return parse(Float64, replace(m.captures[1], r"[dD]" => "e"))
    end
    error("Parameter $name not found in $file")
end

do_write = "--write" in ARGS

@load CHECKPOINT_FILE checkpoint_data
old_eksobj    = checkpoint_data["eksobj"]
prior         = checkpoint_data["prior"]
param_history = checkpoint_data["param_history"]
y_obs         = checkpoint_data["y_obs"]
obs_noise_cov = checkpoint_data["obs_noise_cov"]
N_ensemble    = size(param_history, 3)

# Sanity check: the bounds above reproduce the checkpoint's own physical parameters.
u_old = get_ϕ_final(prior, old_eksobj)
P_old = param_history[:, 1, :]
P_old_recomputed = [denormalise_param(u_old[i, j], PARAM_NAMES[i]) for i in 1:6, j in 1:N_ensemble]
isapprox(P_old_recomputed, P_old; rtol=1e-10) ||
    error("PRIOR_BOUNDS in this script do not match those used to write the checkpoint -- aborting")
println("✓ PRIOR_BOUNDS consistent with existing checkpoint")

# Parameters each member actually ran with (physical units; already clamped to the box).
P_real = zeros(6, N_ensemble)
for j in 1:N_ensemble
    nml_file = joinpath(ITERATION_DIR, "member_$(j)", "ocn_par.nml")
    for (i, name) in enumerate(PARAM_NAMES)
        P_real[i, j] = nml_value(nml_file, name)
    end
end

# Classify members: identical to checkpoint, differing only by clamping, or from the lost draw.
n_same, n_clamped, n_replaced = 0, 0, 0
for j in 1:N_ensemble
    matches = [isapprox(P_real[i, j], P_old[i, j]; rtol=1e-6) for i in 1:6]
    clamped_ok = all(matches[i] ||
                     isapprox(P_real[i, j], clamp(P_old[i, j], PRIOR_BOUNDS[PARAM_NAMES[i]]...); rtol=1e-6)
                     for i in 1:6)
    if all(matches)
        global n_same += 1
    elseif clamped_ok
        global n_clamped += 1
        @printf("  member %3d: matches checkpoint after clamping (%s)\n", j,
                join(PARAM_NAMES[.!matches], ", "))
    else
        global n_replaced += 1
    end
end
println("Members identical to checkpoint:        $n_same")
println("Members matching after box clamping:    $n_clamped")
println("Members from the lost original draw:    $n_replaced")

# Every value that was actually run must lie inside the box.
for (i, name) in enumerate(PARAM_NAMES)
    lo, hi = PRIOR_BOUNDS[name]
    all(lo * (1 - 1e-9) .<= P_real[i, :] .<= hi * (1 + 1e-9)) ||
        error("Some $name values in ocn_par.nml lie outside PRIOR_BOUNDS -- unexpected, aborting")
end
println("✓ All member parameters lie inside PRIOR_BOUNDS")

# Rebuild the EKS object from the real initial ensemble (no_constraint prior => u = ϕ).
u_real = [normalise_param(P_real[i, j], PARAM_NAMES[i]) for i in 1:6, j in 1:N_ensemble]
new_eksobj = EnsembleKalmanProcess(u_real, y_obs, obs_noise_cov, Sampler(prior); verbose=true)
isapprox(get_ϕ_final(prior, new_eksobj), u_real; rtol=1e-12) ||
    error("Rebuilt EKS ensemble does not reproduce the member parameters -- aborting")

new_param_history = copy(param_history)
new_param_history[:, 1, :] = P_real

println("\nPhysical-space ensemble mean  (old checkpoint -> real runs):")
for (i, name) in enumerate(PARAM_NAMES)
    @printf("  %-14s %12.5g -> %12.5g\n", name, sum(P_old[i, :]) / N_ensemble, sum(P_real[i, :]) / N_ensemble)
end

if do_write
    isfile(BACKUP_FILE) && error("Backup $BACKUP_FILE already exists -- refusing to overwrite it")
    cp(CHECKPOINT_FILE, BACKUP_FILE)
    println("\n✓ Backup written: $BACKUP_FILE")
    checkpoint_data["eksobj"] = new_eksobj
    checkpoint_data["param_history"] = new_param_history
    @save CHECKPOINT_FILE checkpoint_data
    println("✓ Repaired checkpoint written: $CHECKPOINT_FILE")
else
    println("\nDry run only -- rerun with --write to back up and rewrite the checkpoint.")
end

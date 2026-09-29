# Check that the vendored Python observation operator, called from Julia, reproduces
# the calibration target exported from calibrate_do_paper.ipynb
# (data/python_calibration_setup.json) on the default run.
#
#   julia python/check_python_operator.jl [path/to/default_run/ocn_ts.nc]
#
# Default path is the HPC default run used by climber_x_calibration.jl. Run
# python/setup_python_env.jl once first.

using PythonCall
using Printf

const REPO_DIR = dirname(@__DIR__)
const SETUP_FILE = joinpath(REPO_DIR, "data", "python_calibration_setup.json")
default_file = isempty(ARGS) ? "/p/tmp/karinako/default_run_long/0/ocn_ts.nc" : ARGS[1]

sys = pyimport("sys")
sys.path.insert(0, joinpath(REPO_DIR, "python"))
ss = pyimport("summary_stats")
np = pyimport("numpy")
xr = pyimport("xarray")
setup = pyimport("json").load(pybuiltins.open(SETUP_FILE))

x_grid     = np.array(setup["x_grid"])
components = pyconvert(Matrix{Float64}, np.array(setup["pca"]["components"]))
pca_mean   = pyconvert(Vector{Float64}, np.array(setup["pca"]["mean"]))
obs_keys   = pyconvert(Vector{String}, setup["obs_keys"])
targets    = pyconvert(Vector{Float64}, setup["obs_targets"])
sigmas     = pyconvert(Vector{Float64}, setup["obs_sigmas"])
n_pca      = pyconvert(Int, setup["pca"]["n_components"])

println("Default run: $default_file")
ds = xr.open_dataset(default_file)
amoc = ds.amoc26N.values
time = ds.time.values
ds.close()

kw = setup["default_stats_kwargs"]
stats = ss.compute_summary_stats(amoc, time; x_grid=x_grid,
                                 (Symbol(pyconvert(String, k)) => kw[k] for k in kw)...)

pdf = pyconvert(Vector{Float64}, stats["pdf"])
pca_scores = components * (pdf .- pca_mean)
wt = pyconvert(Float64, stats["avg_waiting_time"])
values = vcat(pca_scores, wt)
@assert length(values) == length(obs_keys) "obs_keys expects $(length(obs_keys)) values, got $(length(values))"

println("\n  statistic          recomputed        notebook target   |diff|/sigma")
max_z = 0.0
for (k, v, t, s) in zip(obs_keys, values, targets, sigmas)
    z = abs(v - t) / s
    global max_z = max(max_z, z)
    @printf("  %-16s %16.6f  %16.6f   %.2e\n", k, v, t, z)
end
println("  n_do_events: ", pyconvert(Int, stats["n_do_events"]),
        "   do_variability: ", pyconvert(Bool, stats["do_variability"]))

if max_z < 1e-3
    println("\n✓ Python operator reproduces the notebook target (max |diff| = $(@sprintf("%.1e", max_z)) sigma)")
else
    println("\n✗ MISMATCH: max |diff| = $(@sprintf("%.3g", max_z)) sigma -- check the default run file and package versions")
end

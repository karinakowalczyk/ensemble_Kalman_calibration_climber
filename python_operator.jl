# Observation operator of the emulator+MCMC approach (calibrate_do_paper.ipynb), used
# directly by the EKS driver so that both methods share one observation operator,
# target and observation uncertainty (only the inference method differs).
#
# - python/summary_stats.py: vendored, unchanged copy of the Python summary statistics
# - data/python_calibration_setup.json: exported by python/export_python_calibration_setup.py
#   (PDF grid, PCA components/mean, obs_targets/obs_sigmas, compute_summary_stats settings)
#
# netCDF files are read in Julia (NCDatasets) and the AMOC/time arrays are passed to
# Python, so Python never loads its own netCDF/HDF5 libraries into this process.
# Julia and xarray read identical Float64 arrays from CLIMBER-X ocn_ts.nc files.

# Loading PythonCall with its CondaPkg-managed Python calls CondaPkg.activate!(ENV),
# which puts the private conda env first on PATH (plus CONDA_PREFIX etc.) for the
# whole Julia process. Every external command the driver starts would then see that
# Python -- including `./runme` (a Python script needing the `runner` module) and
# the jobs it submits. So: snapshot ENV, load PythonCall, import the compiled
# modules summary_stats.py needs while the conda env is still active (PythonCall
# activates it for loading extension modules), then restore ENV exactly. The
# embedded interpreter is initialised by then and unaffected.
const ENV_BEFORE_PYTHONCALL = copy(ENV)
using PythonCall
for mod in ("numpy", "scipy.stats", "scipy.signal", "scipy.ndimage", "sklearn.cluster",
            "sklearn.decomposition", "pandas", "xarray", "matplotlib.pyplot")
    pyimport(mod)
end
for k in collect(keys(ENV))
    haskey(ENV_BEFORE_PYTHONCALL, k) || delete!(ENV, k)
end
for (k, v) in ENV_BEFORE_PYTHONCALL
    ENV[k] = v
end

const PYTHON_DIR = joinpath(@__DIR__, "python")
const PYTHON_SETUP_FILE = joinpath(@__DIR__, "data", "python_calibration_setup.json")

struct PythonObservationOperator
    summary_stats::Py
    x_grid::Vector{Float64}
    components::Matrix{Float64}      # (n_pca × n_grid)
    pca_mean::Vector{Float64}        # (n_grid,)
    obs_keys::Vector{String}         # "pca_0" ... "pca_4", "avg_waiting_time"
    targets::Vector{Float64}
    sigmas::Vector{Float64}
    member_kwargs::Dict{Symbol, Any}
    default_kwargs::Dict{Symbol, Any}
    setup::Dict{String, Any}         # provenance fields of the setup file, for metadata
end

n_pca(op::PythonObservationOperator) = size(op.components, 1)

pykwargs(d) = Dict{Symbol, Any}(Symbol(pyconvert(String, k)) => pyconvert(Any, d[k]) for k in d)

function load_python_operator(setup_file=PYTHON_SETUP_FILE)
    isfile(setup_file) || error("Python calibration setup not found: $setup_file")
    sys = pyimport("sys")
    PYTHON_DIR in pyconvert(Vector{String}, sys.path) || sys.path.insert(0, PYTHON_DIR)
    ss = pyimport("summary_stats")
    np = pyimport("numpy")
    s = pyimport("json").load(pybuiltins.open(setup_file))

    obs_keys = pyconvert(Vector{String}, s["obs_keys"])
    op = PythonObservationOperator(
        ss,
        pyconvert(Vector{Float64}, s["x_grid"]),
        pyconvert(Matrix{Float64}, np.array(s["pca"]["components"])),
        pyconvert(Vector{Float64}, s["pca"]["mean"]),
        obs_keys,
        pyconvert(Vector{Float64}, s["obs_targets"]),
        pyconvert(Vector{Float64}, s["obs_sigmas"]),
        pykwargs(s["member_stats_kwargs"]),
        pykwargs(s["default_stats_kwargs"]),
        Dict{String, Any}(
            "setup_file" => setup_file,
            # Dict{String,Any}, not Any: pyconvert(Any, dict) keeps a PyDict wrapper,
            # which JLD2 saves but can't read back without Python.
            "notebook"   => pyconvert(Dict{String, Any}, s["notebook"]),
            "summary_stats_py" => pyconvert(Dict{String, Any}, s["summary_stats_py"]),
            "exported"   => pyconvert(String, s["exported"]),
        ),
    )
    obs_keys == vcat(["pca_$(k)" for k in 0:n_pca(op)-1], ["avg_waiting_time"]) ||
        error("Unexpected obs_keys in $setup_file: $obs_keys (expected $(n_pca(op)) PCA scores + avg_waiting_time)")
    size(op.components, 2) == length(op.x_grid) == length(op.pca_mean) ||
        error("PCA components/mean do not match the PDF grid in $setup_file")
    return op
end

"""
Python compute_summary_stats on an AMOC series (full run incl. spinup -- the spinup
removal is part of the Python settings). Returns the full vector
[pdf on x_grid...; avg_waiting_time; avg_stadial_duration] plus a small stats Dict.
"""
function python_full_vector(op::PythonObservationOperator, amoc, time, kwargs)
    np = pyimport("numpy")
    stats = op.summary_stats.compute_summary_stats(
        np.asarray(Vector{Float64}(amoc)), np.asarray(Vector{Float64}(time));
        x_grid=np.asarray(op.x_grid), kwargs...)
    pdf = pyconvert(Vector{Float64}, stats["pdf"])
    wt  = pyconvert(Float64, stats["avg_waiting_time"])
    sd  = pyconvert(Float64, stats["avg_stadial_duration"])
    info = Dict{String, Any}(
        "avg_waiting_time"     => wt,
        "avg_stadial_duration" => sd,
        "n_do_events"          => pyconvert(Int, stats["n_do_events"]),
        # pytruth, not pyconvert(Bool, ...): for runs without DO variability
        # summary_stats.py returns a numpy.bool_ (`(fp_loc <= thr) and (...)` yields its
        # first, numpy, operand), which pyconvert refuses to turn into a Julia Bool.
        "do_variability"       => pytruth(stats["do_variability"]),
    )
    return vcat(pdf, wt, sd), info
end

"""
Observation-space vector(s) used by EKS: [PCA scores (Python basis); avg_waiting_time],
from full vector(s) [pdf...; wt; sd]. Same projection as sklearn PCA.transform
(no whitening): (pdf - mean) * components'.
"""
function project_full_to_obs(op::PythonObservationOperator, G_full::AbstractMatrix)
    n_grid = length(op.x_grid)
    pdfs = G_full[1:n_grid, :]
    return vcat(op.components * (pdfs .- op.pca_mean), G_full[n_grid+1:n_grid+1, :])
end
project_full_to_obs(op::PythonObservationOperator, g_full::AbstractVector) =
    vec(project_full_to_obs(op, reshape(g_full, :, 1)))

# One-time setup: gives Julia a private Python environment (via PythonCall.jl +
# CondaPkg.jl) with the packages python/summary_stats.py needs, pinned to the
# versions of the local `emulator` conda env the emulator+MCMC results were made
# with. Installs into the active Julia environment (the global one, which is what
# `julia climber_x_calibration.jl` uses). Needs internet access (conda-forge), so
# run it on a login node:
#
#   julia python/setup_python_env.jl

import Pkg
Pkg.add(["PythonCall", "CondaPkg"])

using CondaPkg
CondaPkg.add("python";       version="=3.11")
CondaPkg.add("numpy";        version="=2.0.2")
CondaPkg.add("scipy";        version="=1.16.0")
CondaPkg.add("scikit-learn"; version="=1.7.1")
CondaPkg.add("pandas";       version="=2.3.1")
CondaPkg.add("xarray";       version="=2025.4.0")
CondaPkg.add("netcdf4";      version="=1.7.2")
CondaPkg.add("matplotlib-base"; version="=3.10.0")   # imported at module level by summary_stats.py
CondaPkg.resolve()

using PythonCall
for mod in ["numpy", "scipy", "sklearn", "pandas", "xarray", "netCDF4", "matplotlib"]
    m = pyimport(mod)
    println(rpad(mod, 12), pyconvert(String, m.__version__))
end
println("✓ Python environment ready: ", pyconvert(String, pyimport("sys").executable))

# CLIMBER-X EKI Calibration

Ensemble Kalman Inversion calibration of CLIMBER-X ocean parameters using AMOC statistics.

## Files

1. **eks_job_management.jl** - Job submission, tracking, and management utilities (no changes from toy model)
2. **climber_summary_stats.jl** - AMOC analysis: PCA, DO event detection, stadial identification
3. **climber_x_calibration.jl** - Main calibration script

## Requirements

### Julia Packages
```julia
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Statistics
using Random
using JLD2
using Dates
using Printf
using NCDatasets
using KernelDensity
using Clustering
using Loess
using MultivariateStats
```

### System Requirements
- SLURM cluster with `sbatch` and `sacct` commands
- CLIMBER-X installed at `/home/karinako/climber-x/`
- Default run output at `/p/tmp/karinako/default_run_long/0/ocn_ts.nc`
- Sufficient disk space (>500 GB recommended)

## Configuration

Edit `climber_x_calibration.jl` to modify:

### Paths
```julia
const CLIMBER_X_DIR = "/home/karinako/climber-x"
const DEFAULT_RUN_OUTPUT = "/p/tmp/karinako/default_run_long/0/ocn_ts.nc"
```

### Calibration Parameters
6 ocean parameters with **uniform priors** (true uniform distributions, not truncated Gaussians):
- `diff_dia_min`: [7.5e-6, 1.25e-5]
- `drag_topo_fac`: [2.25, 3.75]
- `slope_max`: [7.5e-4, 1.25e-3]
- `diff_iso`: [1125, 1875]
- `diff_gm`: [1125, 1875]
- `diff_dia_max`: [1.125e-4, 1.875e-4]

**Note on priors**: While EKI updates use Gaussian-like covariance operations in unconstrained space, the initial ensemble is sampled from true uniform distributions in physical space, as specified by the bounds above.

### Observations
7 target statistics from default run:
- 5 PCA components (σ = 0.0189 each)
- Average waiting time (σ = 39.1 years)
- Average stadial duration (σ = 42.6 years)

### Fixed CLIMBER-X Parameters
- `ctl.nyears = 75000` (75k years simulation)
- `ctl.co2_const = 190` ppm
- Restart from: `/home/karinako/climber-x/output/DO/spinup_ensemble/CO2_190/restart_out/year_3000`
- Freshwater noise: `ocn.l_noise_fw = T`

### SLURM Resources
- Queue: `long`
- Walltime: 200 hours
- CPUs: 32 (OpenMP threads)
- Memory: 64 GB

## Usage

### Basic Run
```bash
julia climber_x_calibration.jl
```

### Custom Configuration
Edit the main call at the bottom of `climber_x_calibration.jl`:
```julia
eksobj, param_history, metadata, pca_model = run_climber_x_calibration(
    N_iterations=10,           # Number of EKI iterations
    N_ensemble=50,             # Ensemble size
    output_dir="/p/tmp/karinako/eki_calibration",
    work_dir="/p/tmp/karinako/eki_calibration",
    check_interval_minutes=30, # How often to check job status
    max_wait_days=10,          # Maximum wait time per iteration
    pca_components=5           # Number of PCA components
)
```

## Workflow

1. **Iteration 0**: Initialize ensemble from prior
2. **Iteration 1**: 
   - Submit N_ensemble CLIMBER-X jobs
   - Wait for completion (~200 hours)
   - Fit PCA model from ensemble PDFs
   - Extract target observations from default run
   - Update ensemble with EKI
3. **Iterations 2-N**: 
   - Submit new ensemble jobs
   - Wait for completion
   - Process outputs with fixed PCA model
   - Update ensemble with EKI

## Outputs

### Directory Structure
```
/p/tmp/karinako/eki_calibration/
├── checkpoints/
│   ├── checkpoint_iter_0.jld2
│   ├── checkpoint_iter_1.jld2
│   └── ...
├── iteration_results.jld2
├── final_results.jld2
├── job_tracking/
│   ├── iter_1_trackers.jld2
│   ├── iter_1_log.txt
│   └── ...
├── iter_1/
│   ├── params_ensemble.txt
│   ├── member_1/
│   │   └── ocn_ts.nc
│   ├── member_2/
│   │   └── ocn_ts.nc
│   └── ...
└── ...
```

### Key Files
- **final_results.jld2**: Optimal parameters, uncertainties, final ensemble
- **iteration_results.jld2**: Results from each iteration
- **checkpoints/**: Full state for restart capability
- **job_tracking/**: Job status and timing information

## Monitoring

### Check Job Status
```bash
squeue -u $USER | grep climber
```

### Check Progress
```julia
using JLD2
@load "/p/tmp/karinako/eki_calibration/iteration_results.jld2" all_results
println("Completed iterations: ", length(all_results))
```

### Resume from Checkpoint
```julia
using JLD2
@load "/p/tmp/karinako/eki_calibration/checkpoints/checkpoint_iter_5.jld2" checkpoint_data
# Restart from iteration 5...
```

## Key Features

### Robust Job Management
- Automatic retry on submission failures
- Job status monitoring with exponential backoff
- Disk space monitoring
- Failed job tracking and reporting

### PCA-Based Calibration
- Captures full PDF shape (bimodality, skewness)
- Dimensionality reduction (100 → 5 components)
- Fitted on iteration 1 ensemble for consistency

### DO Event Detection
- LOESS detrending
- Adaptive peak finding
- Automatic spacing filtering
- Stadial identification with clustering

### Checkpointing
- Save after each iteration
- Full restart capability
- Job tracking persistence

## Troubleshooting

### Jobs failing
- Check SLURM logs: `/p/tmp/karinako/eki_calibration/iter_X/member_Y_JOBID.err`
- Verify CLIMBER-X paths and modules
- Check disk space: `df -h /p/tmp/karinako/`

### PCA fitting issues
- Ensure iteration 1 has sufficient successful runs (>40 recommended)
- Check for NaN values in AMOC timeseries
- Verify spinup removal is appropriate

### Memory issues
- Increase `--mem=64G` in job script if needed
- Reduce ensemble size
- Check CLIMBER-X memory requirements

## Performance Notes

- **Per iteration**: ~200 hours (CLIMBER-X runtime) + overhead
- **Total time**: ~2000 hours for 10 iterations (83 days if sequential)
- **Parallelization**: N_ensemble jobs run simultaneously
- **Disk usage**: ~2-5 GB per member × N_ensemble × N_iterations

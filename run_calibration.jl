include("eks_climber_x_calibration.jl")

eksobj, param_history, metadata, pca_model = run_climber_x_calibration(
    N_iterations=6,
    N_ensemble=60,
    output_dir="/p/tmp/karinako/eki_calibration/output",
    work_dir="/p/tmp/karinako/eki_calibration/working",
    check_interval_minutes=60,
    max_wait_days=10,
    pca_components=5,
    prior_shift_fraction=0.2
)

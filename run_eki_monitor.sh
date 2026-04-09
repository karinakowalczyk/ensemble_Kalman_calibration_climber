#!/bin/bash
#SBATCH --job-name=eki_monitor
#SBATCH --time=10-00:00:00
#SBATCH --partition=standard
#SBATCH --qos=standby
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/p/tmp/karinako/eki_calibration/monitor_%j.out
#SBATCH --error=/p/tmp/karinako/eki_calibration/monitor_%j.err

module load julia
module load python/3.12.3

cd /home/karinako/calibration_climber

julia climber_x_calibration.jl

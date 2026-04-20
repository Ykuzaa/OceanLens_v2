#!/bin/bash
#SBATCH --job-name=ol2-preprocess
#SBATCH --partition=mono
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

set -eo pipefail
cd /scratch/emboulaalam/OceanLens_v2
mkdir -p logs

source /softs/Anaconda3/2024.06/etc/profile.d/conda.sh
set +u
conda activate oceanlens
set -u

export PYTHONPATH=/scratch/emboulaalam/OceanLens_v2:${PYTHONPATH:-}
python -u scripts/preprocess_glorys.py --config configs/v1.yaml

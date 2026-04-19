#!/bin/bash
#SBATCH --job-name=ol2-preprocess
#SBATCH --partition=mono
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

set -euo pipefail
cd /scratch/emboulaalam/OceanLens_v2
mkdir -p logs
conda activate oceanlens
python scripts/preprocess_glorys.py --config configs/v1.yaml


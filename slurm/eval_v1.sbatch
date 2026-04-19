#!/bin/bash
#SBATCH --job-name=ol2-v1-eval
#SBATCH --partition=training-large-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail
cd /scratch/emboulaalam/OceanLens_v2
mkdir -p logs
conda activate oceanlens
python scripts/infer.py \
  --config configs/v1.yaml \
  --cno_ckpt runs/v1/cno/checkpoints/last.ckpt \
  --fm_ckpt runs/v1/fm/checkpoints/last.ckpt \
  --ensemble_members 5 \
  --max_samples 1 \
  --output_dir results/v1_ens5
python scripts/evaluate.py --result_dir results/v1_ens5


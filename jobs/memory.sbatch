#!/bin/bash
#SBATCH --job-name=scicheck-memory
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python -m scicheck performance --sweep memory

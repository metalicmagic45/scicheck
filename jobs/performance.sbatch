#!/bin/bash
#SBATCH --job-name=scicheck-perf
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python -m scicheck performance --device cuda:0
python -m scicheck report

#!/bin/bash
#SBATCH --job-name=scicheck-acc
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

ARCHS=(mlp cnn ae)
python -m scicheck accuracy --archs ${ARCHS[$SLURM_ARRAY_TASK_ID]} --workers 4 --threads 1 --devices cuda:0

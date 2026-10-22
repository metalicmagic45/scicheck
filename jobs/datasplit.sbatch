#!/bin/bash
#SBATCH --job-name=scicheck-split
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python -m scicheck accuracy --config configs/datasplit.yaml --workers 4 --threads 1 --devices cuda:0
python -m scicheck report --config configs/datasplit.yaml

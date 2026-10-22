#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

python -m scicheck accuracy --archs mlp --workers 4 --threads 1 --devices cuda:0 > mlp.log 2>&1 &
python -m scicheck accuracy --archs cnn --workers 4 --threads 1 --devices cuda:0 > cnn.log 2>&1 &
python -m scicheck accuracy --archs ae  --workers 4 --threads 1 --devices cuda:0 > ae.log  2>&1 &
wait

python -m scicheck performance --device cuda:0
python -m scicheck report

python -m scicheck accuracy --config configs/datasplit.yaml --workers 4 --threads 1 --devices cuda:0
python -m scicheck report --config configs/datasplit.yaml

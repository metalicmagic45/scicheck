#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

CONFIG=configs/laptop.yaml
python -m scicheck info --config "$CONFIG"
python -m scicheck download --config "$CONFIG"
python -m scicheck accuracy --config "$CONFIG" --workers "${WORKERS:-1}"
python -m scicheck report --config "$CONFIG"
echo "Report: results/laptop/report/report.md"

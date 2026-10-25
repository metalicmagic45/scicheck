#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

IFS=, read -ra GPU_LIST <<< "${GPUS:-0}"
WORKERS=${WORKERS:-4}
mkdir -p logs
trap 'trap - INT TERM; kill 0' INT TERM

python -c '
import sys, torch
gpus = ",".join(sys.argv[1:])
need = max(map(int, sys.argv[1:])) + 1
have = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"python {sys.executable}, torch {torch.__version__} (CUDA {torch.version.cuda}), {have} GPU(s) visible")
if have < need:
    sys.exit(f"GPU check failed: GPUS={gpus} needs {need} GPU(s) but torch sees {have}.\n"
             "Activate the venv from the README Setup, and install a torch build whose CUDA version "
             "this machine'"'"'s driver supports (nvidia-smi shows the highest).")
' "${GPU_LIST[@]}"

declare -A PIDS

start() {
    local name=$1; shift
    echo "$(date '+%F %T') start  $name -> logs/$name.log"
    python -m scicheck "$@" > "logs/$name.log" 2>&1 &
    PIDS[$name]=$!
}

finish() {
    local failed=0
    for name in "$@"; do
        if wait "${PIDS[$name]}"; then
            echo "$(date '+%F %T') done   $name"
        else
            echo "$(date '+%F %T') FAILED $name, see logs/$name.log" >&2
            tail -n 15 "logs/$name.log" | sed 's/^/    /' >&2
            failed=1
        fi
    done
    if [ $failed -ne 0 ]; then
        echo "Waiting for running steps to finish, then stopping. Fix the error and run this script again." >&2
        wait
        exit 1
    fi
}

run() { start "$@"; finish "$1"; }

run download download

ARCHS=(mlp cnn ae)
for i in "${!ARCHS[@]}"; do
    start "accuracy_${ARCHS[$i]}" accuracy --archs "${ARCHS[$i]}" --workers "$WORKERS" --threads 1 \
        --devices "cuda:${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
done
start memory performance --sweep memory
finish accuracy_mlp accuracy_cnn accuracy_ae

run datasplit accuracy --config configs/datasplit.yaml --workers "$WORKERS" --threads 1 \
    --devices "cuda:${GPU_LIST[0]}"
run stepablation accuracy --config configs/stepablation.yaml --workers "$WORKERS" --threads 1 \
    --devices "cuda:${GPU_LIST[0]}"
finish memory

run runtime performance --sweep runtime --device "cuda:${GPU_LIST[0]}"

run report report
run report_datasplit report --config configs/datasplit.yaml
run report_stepablation report --config configs/stepablation.yaml
echo "Reports: results/{paper,datasplit,stepablation}/report/report.md"

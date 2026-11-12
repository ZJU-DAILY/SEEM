#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:?Usage: run_dataset.sh DATASET [run.py options]}"
shift
PYTHON="${PYTHON:-python3}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
for horizon in 96 192 336 720; do
  "$PYTHON" -u "$ROOT/run.py" --dataset "$DATASET" --horizon "$horizon" --device "${DEVICE:-cuda}" --resume "$@" 2>&1 | tee -a "$LOG_DIR/${DATASET}_96_${horizon}_seed42.log"
done

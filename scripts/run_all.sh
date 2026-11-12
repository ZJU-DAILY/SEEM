#!/usr/bin/env bash
set -euo pipefail
for dataset in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity Traffic; do
  bash "$(dirname "${BASH_SOURCE[0]}")/run_dataset.sh" "$dataset" "$@"
done

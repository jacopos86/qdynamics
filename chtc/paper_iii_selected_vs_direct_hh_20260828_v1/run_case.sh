#!/usr/bin/env bash
set -euo pipefail

CASE_KEY="${1:?case key required}"
if [[ ! "$CASE_KEY" =~ ^[a-z0-9_]+_nph[13]$ ]]; then
  echo "Invalid case key: $CASE_KEY" >&2
  exit 64
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "JOB START case_key=$CASE_KEY host=$(hostname)"
date -u +%Y-%m-%dT%H:%M:%SZ
python -m pipelines.exact_bench.paper_iii_selected_vs_direct_case \
  --case-key "$CASE_KEY" \
  --output-json case_result.json
date -u +%Y-%m-%dT%H:%M:%SZ
echo "JOB DONE case_key=$CASE_KEY"

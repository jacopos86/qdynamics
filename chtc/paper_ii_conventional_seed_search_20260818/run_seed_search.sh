#!/usr/bin/env bash
set -euo pipefail
JOB_ID="${1:?job_id required}"
BASE=chtc/paper_ii_conventional_seed_search_20260818
read -r _ LAYERS SEED <<< "$(awk -F'\t' -v id="$JOB_ID" '$1==id {print $1, $2, $3}' $BASE/input/jobs.tsv)"
[[ -n "${LAYERS:-}" ]] || { echo "job_id $JOB_ID not found" >&2; exit 3; }
OUT="raw_outputs/paper_ii_conventional_seed_search_20260818/${JOB_ID}"
mkdir -p "$OUT" logs
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
python3 pipelines/time_dynamics/runners/build_fixed_vqe_conditioning_seed.py \
  --output-dir "$OUT" \
  --construction-mode conventional_fixed_layered_v1 \
  --num-sites 2 --u 1.25 --g-ep 0.5 --n-ph-max 3 \
  --layer-counts "$LAYERS" --search-seed "$SEED" \
  --population-size 24 --generations 40 --vqe-restarts 6 \
  --delta-e-max 1.0e-4 --write-artifacts \
  --max-architecture-workers 4 --max-snapshot-workers 2

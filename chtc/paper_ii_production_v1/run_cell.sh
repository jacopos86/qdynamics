#!/usr/bin/env bash
set -euo pipefail
CELL_ID="${1:?cell_id required}"
BASE=chtc/paper_ii_production_v1
ARGV=$(awk -F"\t" -v id="$CELL_ID" '$1==id {print $2}' "$BASE/input/cell_argv.tsv")
if [[ -z "$ARGV" ]]; then echo "unknown cell_id $CELL_ID" >&2; exit 3; fi
mkdir -p logs raw_outputs
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
# shellcheck disable=SC2086
python3 pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py $ARGV

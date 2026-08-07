#!/bin/bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: execute_source_locked_job.sh JOB AUTHORIZATION OUTPUT" >&2
  exit 64
fi

exec python3 -B run_cell.py \
  --job "$1" \
  --execution-authorization "$2" \
  --output "$3"

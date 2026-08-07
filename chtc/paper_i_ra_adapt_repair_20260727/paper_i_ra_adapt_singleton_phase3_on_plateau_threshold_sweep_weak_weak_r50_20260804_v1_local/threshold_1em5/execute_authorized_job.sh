#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export STATIC_ADAPT_HH_POOL_CACHE=off
export STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 JOB AUTHORIZATION EXECUTION_ID" >&2
  exit 64
fi

job_path="$1"
authorization_path="$2"
execution_id="$3"
if [[ ! "$execution_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "unsafe execution id" >&2
  exit 65
fi
for required in \
  "$job_path" \
  "$authorization_path" \
  package_contract.py \
  package_manifest.json \
  protocol_bundle_manifest.json \
  source_locks_snapshot.json \
  source/source_locked.tar.gz \
  source/source_archive_manifest.json \
  run_cell.py
do
  if [[ ! -f "$required" || -L "$required" ]]; then
    echo "missing or unsafe input: $required" >&2
    exit 66
  fi
done

work_root="attempt_${execution_id}"
output_archive="${execution_id}.tar.gz"
if [[ -e "$work_root" || -L "$work_root" || -e "$output_archive" || -L "$output_archive" ]]; then
  echo "refusing to overwrite attempt output" >&2
  exit 67
fi
mkdir "$work_root"

package_attempt() {
  worker_status="$?"
  packaging_status=0
  trap - EXIT
  set +e
  printf '%s\n' "$worker_status" >"${work_root}/worker_exit_status.txt" || packaging_status=71
  tar -czf "$output_archive" -C "$work_root" . || packaging_status=71
  if [[ "$packaging_status" -ne 0 || ! -s "$output_archive" ]]; then
    echo "attempt packaging failed" >&2
    exit 71
  fi
  exit "$worker_status"
}
trap package_attempt EXIT

python3 -B run_cell.py \
  --run \
  --job "$job_path" \
  --execution-authorization "$authorization_path" \
  --output-dir "${work_root}/artifacts" \
  --receipt "${work_root}/worker_receipt.json"




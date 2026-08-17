#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export STATIC_ADAPT_HH_POOL_CACHE=off
export STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off

if [[ "$#" -ne 11 ]]; then
  echo "usage: $0 PACKAGE JOB AUTH SOURCE SOURCE_SHA RESUME RESUME_SHA IMAGE IMAGE_SHA EXECUTION_ID OUTPUT" >&2
  exit 64
fi

package_dir="$1"
job_spec="$2"
authorization="$3"
source_archive="$4"
expected_source_sha="$5"
resume_archive="$6"
expected_resume_sha="$7"
image_path="$8"
expected_image_sha="$9"
execution_id="${10}"
output_archive="${11}"

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

case "$execution_id" in
  ""|*/*|*".."*)
    echo "unsafe execution id" >&2
    exit 65
    ;;
esac
case "$output_archive" in
  transfer/"${execution_id}"__*.tar.gz) ;;
  *)
    echo "unsafe or mismatched output archive" >&2
    exit 65
    ;;
esac

for path in "$job_spec" "$authorization" "$source_archive" "$resume_archive" "$image_path"; do
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "missing or unsafe input: $path" >&2
    exit 66
  fi
done
if [[ "$(basename "$job_spec" .json)" != "$execution_id" ]]; then
  echo "job/execution identity mismatch" >&2
  exit 67
fi
if [[ "$(basename "$authorization" .json)" != "$execution_id" ]]; then
  echo "authorization/execution identity mismatch" >&2
  exit 67
fi
if [[ "$(hash_file "$source_archive")" != "$expected_source_sha" ]]; then
  echo "source archive hash mismatch" >&2
  exit 68
fi
if [[ "$(hash_file "$resume_archive")" != "$expected_resume_sha" ]]; then
  echo "resume archive hash mismatch" >&2
  exit 68
fi
if [[ "$(hash_file "$image_path")" != "$expected_image_sha" ]]; then
  echo "container image hash mismatch" >&2
  exit 68
fi

work_root="worker_outputs"
run_root="${work_root}/runs/${execution_id}"
receipt="${work_root}/worker_receipt.json"
if [[ -e "$work_root" || -L "$work_root" ]]; then
  echo "worker output root already exists" >&2
  exit 69
fi
mkdir -p "${work_root}/runs" "$(dirname "$output_archive")"

apptainer exec \
  --cleanenv \
  --bind "$PWD:$PWD" \
  --pwd "$PWD" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env STATIC_ADAPT_HH_POOL_CACHE=off \
  --env STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off \
  "$image_path" \
  python -B "${package_dir}/run_cell.py" \
    --job "$job_spec" \
    --run \
    --execution-authorization "$authorization" \
    --output-dir "$run_root" \
    --receipt "$receipt"

if [[ ! -f "$receipt" || -L "$receipt" || ! -d "$run_root" ]]; then
  echo "worker result closure is absent" >&2
  exit 70
fi
tar -C "$work_root" -czf "$output_archive" worker_receipt.json "runs/${execution_id}"
test -s "$output_archive"

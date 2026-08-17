#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export STATIC_ADAPT_HH_POOL_CACHE=off
export STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off

if [[ "$#" -ne 11 ]]; then
  echo "usage: $0 PACKAGE BASE JOB AUTH RESUME_MANIFEST RESUME_ARCHIVE RESUME_SHA IMAGE IMAGE_SHA EXECUTION_ID OUTPUT" >&2
  exit 64
fi

package_dir="$1"
base_package_dir="$2"
job_spec="$3"
authorization="$4"
resume_manifest="$5"
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

work_root="worker_outputs"
run_root="${work_root}/runs/${execution_id}"
receipt="${work_root}/worker_receipt.json"
mkdir -p "$work_root" "$(dirname "$output_archive")"

finalize_attempt() {
  exit_code=$?
  trap - EXIT
  set +e
  printf '{"execution_id":"%s","exit_code":%d}\n' "$execution_id" "$exit_code" > "${work_root}/attempt_status.json"
  tar -C . -czf "$output_archive" "$work_root"
  archive_code=$?
  if [[ "$archive_code" -ne 0 || ! -s "$output_archive" ]]; then
    echo "failed to create attempt archive" >&2
    exit 74
  fi
  exit "$exit_code"
}
trap finalize_attempt EXIT

{
  for path in "$job_spec" "$authorization" "$resume_manifest" "$resume_archive" "$image_path"; do
    if [[ ! -f "$path" || -L "$path" ]]; then
      echo "missing or unsafe input: $path" >&2
      exit 66
    fi
  done
  if [[ ! -d "$package_dir" || -L "$package_dir" || ! -d "$base_package_dir" || -L "$base_package_dir" ]]; then
    echo "missing or unsafe package directory" >&2
    exit 66
  fi
  if [[ "$(basename "$job_spec" .json)" != "$execution_id" || "$(basename "$authorization" .json)" != "$execution_id" ]]; then
    echo "job/authorization execution identity mismatch" >&2
    exit 67
  fi
  if [[ "$(hash_file "$resume_archive")" != "$expected_resume_sha" ]]; then
    echo "resume archive hash mismatch" >&2
    exit 68
  fi
  if [[ "$(hash_file "$image_path")" != "$expected_image_sha" ]]; then
    echo "container image hash mismatch" >&2
    exit 68
  fi
  if [[ -e "$run_root" || -L "$run_root" || -e "$receipt" || -L "$receipt" ]]; then
    echo "worker result destination already exists" >&2
    exit 69
  fi
  apptainer exec \
    --cleanenv \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env STATIC_ADAPT_HH_POOL_CACHE=off \
    --env STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off \
    "$image_path" \
    python -B "${package_dir}/run_cell.py" \
      --base-package-dir "$base_package_dir" \
      --job "$job_spec" \
      --run \
      --resume-materialization "$resume_manifest" \
      --resume-archive "$resume_archive" \
      --execution-authorization "$authorization" \
      --output-dir "$run_root" \
      --receipt "$receipt"
  if [[ ! -f "$receipt" || -L "$receipt" || ! -d "$run_root" ]]; then
    echo "worker result closure is absent" >&2
    exit 70
  fi
} >"${work_root}/attempt.stdout" 2>"${work_root}/attempt.stderr"

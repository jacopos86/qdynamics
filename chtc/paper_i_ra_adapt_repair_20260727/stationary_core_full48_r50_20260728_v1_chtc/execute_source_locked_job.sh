#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export STATIC_ADAPT_HH_POOL_CACHE=off
export STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off

if [[ "$#" -ne 7 ]]; then
  echo "usage: $0 PACKAGE_DIR JOB_SPEC JOB_SHA256 SOURCE_SHA256 IMAGE IMAGE_SHA256 OUTPUT_ARCHIVE" >&2
  exit 64
fi

package_dir="$1"
job_spec="$2"
expected_job_sha256="$3"
expected_source_sha256="$4"
image_path="$5"
expected_image_sha256="$6"
output_archive="$7"
source_archive="${package_dir}/source_locked.tar.gz"
worker_root="worker_outputs"
source_root="source_locked_checkout"

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

case "$output_archive" in
  /*|*".."*|*"//"*)
    echo "unsafe attempt output archive" >&2
    exit 68
    ;;
esac
if [[ ! -f "$job_spec" || -L "$job_spec" ]]; then
  echo "missing or symlinked worker job spec: $job_spec" >&2
  exit 65
fi
job_ad_path="${_CONDOR_JOB_AD:-}"
if [[ -z "$job_ad_path" || ! -f "$job_ad_path" || -L "$job_ad_path" ]]; then
  echo "HTCondor job ad is unavailable" >&2
  exit 70
fi
attempt_ordinal="$(
  awk -F= '$1 ~ /^[[:space:]]*NumJobStarts[[:space:]]*$/ {
    value=$2
    gsub(/[[:space:]]/, "", value)
    print value
    exit
  }' "$job_ad_path"
)"
if [[ ! "$attempt_ordinal" =~ ^[1-9][0-9]*$ ]]; then
  echo "HTCondor NumJobStarts is unavailable or invalid" >&2
  exit 70
fi
cluster_id="$(
  awk -F= '$1 ~ /^[[:space:]]*ClusterId[[:space:]]*$/ {
    value=$2
    gsub(/[[:space:]]/, "", value)
    print value
    exit
  }' "$job_ad_path"
)"
proc_id="$(
  awk -F= '$1 ~ /^[[:space:]]*ProcId[[:space:]]*$/ {
    value=$2
    gsub(/[[:space:]]/, "", value)
    print value
    exit
  }' "$job_ad_path"
)"
if [[ ! "$cluster_id" =~ ^[0-9]+$ || ! "$proc_id" =~ ^[0-9]+$ ]]; then
  echo "HTCondor ClusterId/ProcId is unavailable or invalid" >&2
  exit 70
fi
execution_id="$(basename "$job_spec" .json)"
expected_output_archive="transfer/${execution_id}__${cluster_id}__${proc_id}.tar.gz"
if [[ "$output_archive" != "$expected_output_archive" ]]; then
  echo "attempt output archive does not match the authenticated job ad" >&2
  exit 70
fi

# Condor may restart this executable in the same sandbox after an eviction.
# Authenticate the exact prior attempt before deleting only these two
# sandbox-owned fixed roots and the fixed, non-transferred eviction archive.
if [[ "$worker_root" != "worker_outputs" || "$source_root" != "source_locked_checkout" ]]; then
  echo "internal retry-cleanup target drifted" >&2
  exit 72
fi
prior_attempt_authenticated=0
attempt_marker="${worker_root}/attempt_identity.tsv"
if [[ -e "$worker_root" || -L "$worker_root" ]]; then
  if [[ ! -d "$worker_root" || -L "$worker_root" || ! -f "$attempt_marker" || -L "$attempt_marker" ]]; then
    echo "unauthenticated prior worker root" >&2
    exit 72
  fi
  IFS=$'\t' read -r prior_execution_id prior_cluster_id prior_proc_id prior_attempt_ordinal prior_extra <"$attempt_marker"
  if [[ -n "${prior_extra:-}" \
     || "$prior_execution_id" != "$execution_id" \
     || "$prior_cluster_id" != "$cluster_id" \
     || "$prior_proc_id" != "$proc_id" \
     || ! "$prior_attempt_ordinal" =~ ^[1-9][0-9]*$ \
     || "$prior_attempt_ordinal" -ge "$attempt_ordinal" ]]; then
    echo "prior worker root does not authenticate an earlier attempt" >&2
    exit 72
  fi
  prior_attempt_authenticated=1
fi
if [[ -e "$output_archive" || -L "$output_archive" ]]; then
  if [[ "$prior_attempt_authenticated" -ne 1 || ! -f "$output_archive" || -L "$output_archive" ]]; then
    echo "unauthenticated prior attempt archive" >&2
    exit 72
  fi
fi
if [[ "$prior_attempt_authenticated" -eq 1 ]]; then
  rm -rf -- "$worker_root"
  if [[ -e "$source_root" || -L "$source_root" ]]; then
    if [[ ! -d "$source_root" || -L "$source_root" ]]; then
      echo "unsafe prior source root" >&2
      exit 72
    fi
    rm -rf -- "$source_root"
  fi
  if [[ -e "$output_archive" ]]; then
    rm -f -- "$output_archive"
  fi
elif [[ -e "$source_root" || -L "$source_root" || -e "$output_archive" || -L "$output_archive" ]]; then
  echo "orphaned retry state is unauthenticated" >&2
  exit 72
fi

mkdir "$worker_root"
printf '%s\t%s\t%s\t%s\n' \
  "$execution_id" "$cluster_id" "$proc_id" "$attempt_ordinal" \
  >"$attempt_marker"

package_attempt() {
  local worker_status="$?"
  local packaging_status=0
  trap - EXIT
  set +e
  mkdir -p "$(dirname "$output_archive")" "$worker_root" || packaging_status=71
  printf '%s\n' "$worker_status" >"${worker_root}/worker_exit_status.txt" || packaging_status=71
  printf '%s\n' "$attempt_ordinal" >"${worker_root}/scheduler_attempt_ordinal.txt" || packaging_status=71
  if [[ ! -e "$output_archive" ]]; then
    tar -czf "$output_archive" "$worker_root" "$job_spec" || packaging_status=71
  else
    packaging_status=71
  fi
  if [[ ! -s "$output_archive" ]]; then
    packaging_status=71
  fi
  if [[ "$packaging_status" -ne 0 ]]; then
    echo "attempt packaging failed" >&2
    exit "$packaging_status"
  fi
  exit "$worker_status"
}
trap package_attempt EXIT

for required in \
  "$source_archive" \
  "$image_path" \
  "${package_dir}/package_manifest.json" \
  "${package_dir}/authority/submission_authorization_receipt.json"
do
  if [[ ! -f "$required" || -L "$required" ]]; then
    echo "missing or symlinked worker input: $required" >&2
    exit 65
  fi
done

if [[ "$(hash_file "$job_spec")" != "$expected_job_sha256" ]]; then
  echo "job-spec SHA-256 mismatch" >&2
  exit 66
fi
if [[ "$(hash_file "$source_archive")" != "$expected_source_sha256" ]]; then
  echo "source archive SHA-256 mismatch" >&2
  exit 66
fi
if [[ "$(hash_file "$image_path")" != "$expected_image_sha256" ]]; then
  echo "execution image SHA-256 mismatch" >&2
  exit 67
fi

apptainer exec \
  --cleanenv \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env STATIC_ADAPT_HH_POOL_CACHE=off \
  --env STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off \
  --bind "$(pwd -P):$(pwd -P)" \
  "$image_path" \
  python3 "${package_dir}/run_cell.py" \
    --mode execute \
    --source-root "$source_root" \
    --job-spec "$job_spec" \
    --job-spec-sha256 "$expected_job_sha256" \
    --scheduler-attempt-ordinal "$attempt_ordinal" \
    --scheduler-cluster-id "$cluster_id" \
    --scheduler-proc-id "$proc_id" \
    --verified-image-path "$image_path" \
    --verified-image-sha256 "$expected_image_sha256" \
    --output "$worker_root"

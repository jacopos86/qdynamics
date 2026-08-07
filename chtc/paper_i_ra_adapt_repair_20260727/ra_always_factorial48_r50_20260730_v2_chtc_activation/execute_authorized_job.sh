#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export STATIC_ADAPT_HH_POOL_CACHE=off
export STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off

if [[ "$#" -ne 10 ]]; then
  echo "usage: $0 ACTIVATION_DIR PACKAGE_DIR JOB JOB_FILE_SHA AUTH AUTH_FILE_SHA SOURCE_SHA IMAGE IMAGE_SHA OUTPUT_ARCHIVE" >&2
  exit 64
fi

activation_dir="$1"
package_dir="$2"
job_spec="$3"
expected_job_file_sha256="$4"
authorization="$5"
expected_authorization_file_sha256="$6"
expected_source_sha256="$7"
image_path="$8"
expected_image_sha256="$9"
output_archive="${10}"
source_archive="${package_dir}/source_locked.tar.gz"
activation_manifest="${activation_dir}/activation_manifest.json"
archive_builder="${activation_dir}/build_attempt_archive.py"
worker_root="worker_outputs"

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

for required in \
  "$job_spec" \
  "$authorization" \
  "$source_archive" \
  "$image_path" \
  "$activation_manifest" \
  "$archive_builder" \
  "${package_dir}/run_cell.py"
do
  if [[ ! -f "$required" || -L "$required" ]]; then
    echo "missing or unsafe worker input: $required" >&2
    exit 65
  fi
done
if [[ "$(hash_file "$job_spec")" != "$expected_job_file_sha256" ]]; then
  echo "job file SHA-256 mismatch" >&2
  exit 66
fi
if [[ "$(hash_file "$authorization")" != "$expected_authorization_file_sha256" ]]; then
  echo "authorization file SHA-256 mismatch" >&2
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
runtime_bin="$(command -v apptainer || command -v singularity || true)"
if [[ -z "$runtime_bin" ]]; then
  echo "apptainer/singularity runtime unavailable" >&2
  exit 69
fi

job_ad_path="${_CONDOR_JOB_AD:-}"
if [[ -z "$job_ad_path" || ! -f "$job_ad_path" || -L "$job_ad_path" ]]; then
  echo "HTCondor job ad is unavailable" >&2
  exit 70
fi
read_ad_integer() {
  local name="$1"
  awk -F= -v wanted="$name" '
    $1 ~ "^[[:space:]]*" wanted "[[:space:]]*$" {
      value=$2
      gsub(/[[:space:]]/, "", value)
      print value
      exit
    }
  ' "$job_ad_path"
}
num_job_starts="$(read_ad_integer NumJobStarts)"
cluster_id="$(read_ad_integer ClusterId)"
proc_id="$(read_ad_integer ProcId)"
if [[ ! "$num_job_starts" =~ ^[0-9]+$ \
   || ! "$cluster_id" =~ ^[0-9]+$ \
   || ! "$proc_id" =~ ^[0-9]+$ ]]; then
  echo "HTCondor attempt identity is unavailable" >&2
  exit 70
fi
attempt_ordinal="$((10#$num_job_starts + 1))"
execution_id="$(basename "$job_spec" .json)"
expected_output_archive="transfer/${execution_id}__${cluster_id}__${proc_id}.tar.gz"
if [[ "$output_archive" != "$expected_output_archive" ]]; then
  echo "attempt archive does not match the authenticated job ad" >&2
  exit 70
fi

prior_attempt_authenticated=0
attempt_marker="${worker_root}/attempt_identity.tsv"
if [[ -e "$worker_root" || -L "$worker_root" ]]; then
  if [[ ! -d "$worker_root" || -L "$worker_root" \
     || ! -f "$attempt_marker" || -L "$attempt_marker" ]]; then
    echo "unauthenticated prior worker root" >&2
    exit 72
  fi
  IFS=$'\t' read -r prior_execution prior_cluster prior_proc prior_attempt extra <"$attempt_marker"
  if [[ -n "${extra:-}" \
     || "$prior_execution" != "$execution_id" \
     || "$prior_cluster" != "$cluster_id" \
     || "$prior_proc" != "$proc_id" \
     || ! "$prior_attempt" =~ ^[1-9][0-9]*$ \
     || "$prior_attempt" -ge "$attempt_ordinal" ]]; then
    echo "prior worker root is not an authenticated earlier attempt" >&2
    exit 72
  fi
  prior_attempt_authenticated=1
fi
if [[ "$prior_attempt_authenticated" -eq 1 ]]; then
  rm -rf -- "$worker_root"
  if [[ -e "$output_archive" ]]; then
    if [[ ! -f "$output_archive" || -L "$output_archive" ]]; then
      echo "unsafe prior attempt archive" >&2
      exit 72
    fi
    rm -f -- "$output_archive"
  fi
elif [[ -e "$output_archive" || -L "$output_archive" ]]; then
  echo "orphaned attempt archive is unauthenticated" >&2
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
  mkdir -p "$(dirname "$output_archive")" "$worker_root" \
    || packaging_status=71
  printf '%s\n' "$worker_status" >"${worker_root}/worker_exit_status.txt" \
    || packaging_status=71
  "$runtime_bin" exec \
    --cleanenv \
    --bind "$(pwd -P):$(pwd -P)" \
    "$image_path" \
    python3 "$archive_builder" \
      --worker-root "$worker_root" \
      --job "$job_spec" \
      --authorization "$authorization" \
      --activation-manifest "$activation_manifest" \
      --output-archive "$output_archive" \
      --execution-id "$execution_id" \
      --cluster-id "$cluster_id" \
      --proc-id "$proc_id" \
      --attempt-ordinal "$attempt_ordinal" \
      --worker-exit-status "$worker_status" \
      --source-archive-sha256 "$expected_source_sha256" \
      --image-sha256 "$expected_image_sha256" \
    || packaging_status=71
  if [[ "$packaging_status" -ne 0 || ! -s "$output_archive" ]]; then
    echo "attempt packaging failed" >&2
    exit 71
  fi
  exit "$worker_status"
}
trap package_attempt EXIT

sandbox_root="$(pwd -P)"
worker_abs="${sandbox_root}/${worker_root}"
package_abs="${sandbox_root}/${package_dir}"
job_abs="${sandbox_root}/${job_spec}"
authorization_abs="${sandbox_root}/${authorization}"
image_abs="${sandbox_root}/${image_path}"

"$runtime_bin" exec \
  --cleanenv \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env STATIC_ADAPT_HH_POOL_CACHE=off \
  --env STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off \
  --bind "${sandbox_root}:${sandbox_root}" \
  --pwd "$worker_abs" \
  "$image_abs" \
  python3 "${package_abs}/run_cell.py" \
    --job "$job_abs" \
    --execution-authorization "$authorization_abs" \
    --output "${worker_abs}/result.json"

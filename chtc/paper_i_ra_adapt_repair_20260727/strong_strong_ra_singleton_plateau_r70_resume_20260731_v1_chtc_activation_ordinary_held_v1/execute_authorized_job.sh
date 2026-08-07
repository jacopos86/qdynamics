#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export STATIC_ADAPT_HH_POOL_CACHE=off
export STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 ACTIVATION_DIR RUNTIME_DIR IMAGE OUTPUT_ARCHIVE" >&2
  exit 64
fi

activation_dir="$1"
runtime_dir="$2"
image_path="$3"
output_archive="$4"
execution_id="core__strong_strong_u8__nph7__ra_singleton_plateau__r70"
job_spec="${runtime_dir}/jobs_v2/${execution_id}.json"
authorization="${activation_dir}/execution_authorization.json"
activation_manifest="${activation_dir}/activation_manifest.json"
archive_builder="${activation_dir}/build_attempt_archive.py"
resume_archive="${runtime_dir}/resume_inputs/${execution_id}.tar.gz"
source_archive="${runtime_dir}/effective_source_archives/stationary_core_v11_retention_v2/source_locked.tar.gz"
worker_root="worker_outputs"

expected_job_sha="6612335f1b36c3ec5685e54af743420ea2d76915db67428e801bdc28688f014a"
expected_authorization_sha="caca1a33b4e29fe3b3c18352a8ecf2b938c0a224a78d0ac8bfb1654c7fcee894"
expected_resume_sha="73d89d74f1630e5c40a2d63aef99edf5a69be15e038fd1a98374280ed930ac51"
expected_source_sha="bb4481a7e19b0dba92baa4cf6975ddee3eb184d77a337c538dea8408c7cf4496"
expected_image_sha="fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

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

for required in "$job_spec" "$authorization" "$activation_manifest" "$archive_builder" "$resume_archive" "$source_archive" "$image_path" "${runtime_dir}/run_cell_v2.py"; do
  if [[ ! -f "$required" || -L "$required" ]]; then
    echo "missing or unsafe worker input: $required" >&2
    exit 65
  fi
done
[[ "$(hash_file "$job_spec")" == "$expected_job_sha" ]] || { echo "job SHA-256 mismatch" >&2; exit 66; }
[[ "$(hash_file "$authorization")" == "$expected_authorization_sha" ]] || { echo "authorization SHA-256 mismatch" >&2; exit 66; }
[[ "$(hash_file "$resume_archive")" == "$expected_resume_sha" ]] || { echo "resume SHA-256 mismatch" >&2; exit 66; }
[[ "$(hash_file "$source_archive")" == "$expected_source_sha" ]] || { echo "source SHA-256 mismatch" >&2; exit 66; }
[[ "$(hash_file "$image_path")" == "$expected_image_sha" ]] || { echo "image SHA-256 mismatch" >&2; exit 67; }

runtime_bin="$(command -v apptainer || command -v singularity || true)"
[[ -n "$runtime_bin" ]] || { echo "apptainer/singularity runtime unavailable" >&2; exit 69; }
job_ad_path="${_CONDOR_JOB_AD:-}"
[[ -n "$job_ad_path" && -f "$job_ad_path" && ! -L "$job_ad_path" ]] || { echo "HTCondor job ad unavailable" >&2; exit 70; }
read_ad_integer() {
  local name="$1"
  awk -F= -v wanted="$name" '$1 ~ "^[[:space:]]*" wanted "[[:space:]]*$" { value=$2; gsub(/[[:space:]]/, "", value); print value; exit }' "$job_ad_path"
}
num_job_starts="$(read_ad_integer NumJobStarts)"
cluster_id="$(read_ad_integer ClusterId)"
proc_id="$(read_ad_integer ProcId)"
if [[ ! "$num_job_starts" =~ ^[0-9]+$ || ! "$cluster_id" =~ ^[0-9]+$ || ! "$proc_id" =~ ^[0-9]+$ ]]; then
  echo "HTCondor attempt identity unavailable" >&2
  exit 70
fi
attempt_ordinal="$((10#$num_job_starts + 1))"
expected_output="transfer/${execution_id}__${cluster_id}__${proc_id}.tar.gz"
[[ "$output_archive" == "$expected_output" ]] || { echo "attempt archive identity mismatch" >&2; exit 70; }

attempt_marker="${worker_root}/attempt_identity.tsv"
if [[ -e "$worker_root" || -L "$worker_root" ]]; then
  if [[ ! -d "$worker_root" || -L "$worker_root" || ! -f "$attempt_marker" || -L "$attempt_marker" ]]; then
    echo "unauthenticated prior worker root" >&2
    exit 72
  fi
  IFS=$'\t' read -r prior_execution prior_cluster prior_proc prior_attempt extra <"$attempt_marker"
  if [[ -n "${extra:-}" || "$prior_execution" != "$execution_id" || "$prior_cluster" != "$cluster_id" || "$prior_proc" != "$proc_id" || ! "$prior_attempt" =~ ^[1-9][0-9]*$ || "$prior_attempt" -ge "$attempt_ordinal" ]]; then
    echo "prior worker root is not an authenticated earlier attempt" >&2
    exit 72
  fi
  rm -rf -- "$worker_root"
  [[ ! -e "$output_archive" ]] || rm -f -- "$output_archive"
elif [[ -e "$output_archive" || -L "$output_archive" ]]; then
  echo "orphaned attempt archive is unauthenticated" >&2
  exit 72
fi

mkdir "$worker_root"
printf '%s\t%s\t%s\t%s\n' "$execution_id" "$cluster_id" "$proc_id" "$attempt_ordinal" >"$attempt_marker"

package_attempt() {
  local worker_status="$?"
  local packaging_status=0
  trap - EXIT
  set +e
  mkdir -p "$(dirname "$output_archive")" "$worker_root" || packaging_status=71
  printf '%s\n' "$worker_status" >"${worker_root}/worker_exit_status.txt" || packaging_status=71
  "$runtime_bin" exec --cleanenv --bind "$(pwd -P):$(pwd -P)" "$image_path" python3 "$archive_builder" \
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
    --source-archive-sha256 "$expected_source_sha" \
    --resume-archive-sha256 "$expected_resume_sha" \
    --image-sha256 "$expected_image_sha" || packaging_status=71
  if [[ "$packaging_status" -ne 0 || ! -s "$output_archive" ]]; then
    echo "attempt packaging failed" >&2
    exit 71
  fi
  exit "$worker_status"
}
trap package_attempt EXIT

sandbox_root="$(pwd -P)"
worker_abs="${sandbox_root}/${worker_root}"
runtime_abs="${sandbox_root}/${runtime_dir}"
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
  python3 "${runtime_abs}/run_cell_v2.py" \
    --job "$job_abs" \
    --execution-authorization "$authorization_abs" \
    --output-dir "${worker_abs}/payload" \
    --receipt "${worker_abs}/worker_receipt.json"

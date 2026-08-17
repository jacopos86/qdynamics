#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export STATIC_ADAPT_HH_POOL_CACHE=off
export STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off

if [[ "$#" -ne 14 ]]; then
  echo "usage: $0 PACKAGE JOB AUTH SOURCE SOURCE_SHA IMAGE IMAGE_SHA EXECUTION_ID OUTPUT JOB_SHA MANIFEST_SHA WRAPPER_SHA RUN_CELL_SHA CONTRACT_SHA" >&2
  exit 64
fi

package_dir="$1"
job_path="$2"
authorization_path="$3"
source_archive="$4"
expected_source_sha256="$5"
image_path="$6"
expected_image_sha256="$7"
execution_id="$8"
output_archive="$9"
expected_job_sha256="${10}"
expected_manifest_sha256="${11}"
expected_wrapper_sha256="${12}"
expected_run_cell_sha256="${13}"
expected_contract_sha256="${14}"
pinned_image_sha256="fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

if [[ ! "$execution_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "unsafe execution id" >&2
  exit 65
fi
case "$output_archive" in
  /*|*".."*|*"//"*)
    echo "unsafe output archive" >&2
    exit 65
    ;;
esac
if [[ "$job_path" != "${package_dir}/jobs/${execution_id}.json" \
   || "$source_archive" != "${package_dir}/source/source_locked.tar.gz" ]]; then
  echo "worker input paths do not match the sealed execution id" >&2
  exit 65
fi

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

for required in \
  "$job_path" \
  "$authorization_path" \
  "$source_archive" \
  "$image_path" \
  "${package_dir}/run_cell.py" \
  "${package_dir}/package_contract.py" \
  "${package_dir}/execute_authorized_job.sh" \
  "${package_dir}/package_manifest.json"
do
  if [[ ! -f "$required" || -L "$required" ]]; then
    echo "missing or unsafe worker input: $required" >&2
    exit 66
  fi
done
for expected_hash in \
  "$expected_job_sha256" \
  "$expected_manifest_sha256" \
  "$expected_wrapper_sha256" \
  "$expected_run_cell_sha256" \
  "$expected_contract_sha256"
do
  if [[ ! "$expected_hash" =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid expected SHA-256 argument" >&2
    exit 67
  fi
done
if [[ "$(hash_file "$job_path")" != "$expected_job_sha256" \
   || "$(hash_file "${package_dir}/package_manifest.json")" != "$expected_manifest_sha256" \
   || "$(hash_file "${package_dir}/execute_authorized_job.sh")" != "$expected_wrapper_sha256" \
   || "$(hash_file "${package_dir}/run_cell.py")" != "$expected_run_cell_sha256" \
   || "$(hash_file "${package_dir}/package_contract.py")" != "$expected_contract_sha256" ]]; then
  echo "worker control-plane SHA-256 mismatch" >&2
  exit 67
fi
if [[ ! -f "$0" || -L "$0" \
   || "$(hash_file "$0")" != "$expected_wrapper_sha256" ]]; then
  echo "executed worker wrapper SHA-256 mismatch" >&2
  exit 67
fi
if [[ "$(hash_file "$source_archive")" != "$expected_source_sha256" ]]; then
  echo "source archive SHA-256 mismatch" >&2
  exit 67
fi
if [[ "$expected_image_sha256" != "$pinned_image_sha256" ]]; then
  echo "caller image SHA-256 does not match the package pin" >&2
  exit 67
fi
if [[ "$(hash_file "$image_path")" != "$pinned_image_sha256" ]]; then
  echo "execution image SHA-256 mismatch" >&2
  exit 67
fi
runtime_bin="$(command -v apptainer || command -v singularity || true)"
if [[ -z "$runtime_bin" ]]; then
  echo "apptainer/singularity runtime unavailable" >&2
  exit 68
fi

job_ad_path="${_CONDOR_JOB_AD:-}"
if [[ -z "$job_ad_path" || ! -f "$job_ad_path" || -L "$job_ad_path" ]]; then
  echo "HTCondor job ad is unavailable" >&2
  exit 70
fi
job_ad_integer() {
  awk -F= -v key="$1" '
    {
      field=$1
      gsub(/[[:space:]]/, "", field)
      if (field == key) {
        value=$2
        gsub(/[[:space:]]/, "", value)
        print value
        exit
      }
    }
  ' "$job_ad_path"
}
cluster_id="$(job_ad_integer ClusterId)"
proc_id="$(job_ad_integer ProcId)"
num_job_starts="$(job_ad_integer NumJobStarts)"
if [[ ! "$cluster_id" =~ ^[0-9]+$ || "$cluster_id" == "0" \
   || ! "$proc_id" =~ ^[0-9]+$ \
   || ! "$num_job_starts" =~ ^[0-9]+$ ]]; then
  echo "HTCondor scheduler attempt identity is invalid" >&2
  exit 70
fi
attempt_ordinal="$((10#$num_job_starts + 1))"
expected_output_archive="transfer/${execution_id}__${cluster_id}__${proc_id}.tar.gz"
if [[ "$output_archive" != "$expected_output_archive" ]]; then
  echo "terminal output archive does not match scheduler identity" >&2
  exit 70
fi

work_root="attempt_${execution_id}__${cluster_id}__${proc_id}__${attempt_ordinal}"
if [[ -e "$work_root" || -L "$work_root" \
   || -e "$output_archive" || -L "$output_archive" ]]; then
  echo "refusing to overwrite attempt output" >&2
  exit 69
fi
mkdir "$work_root"
printf '{"schema":"paper_i_chtc_scheduler_attempt_identity_v1","execution_id":"%s","cluster_id":%s,"proc_id":%s,"attempt_ordinal":%s,"terminal_output_archive":"%s","transfer_policy":"on_exit_only_v1"}\n' \
  "$execution_id" "$cluster_id" "$proc_id" "$attempt_ordinal" \
  "$output_archive" >"${work_root}/attempt_identity.json"

package_attempt() {
  worker_status="$?"
  packaging_status=0
  trap - EXIT
  set +e
  mkdir -p "$(dirname "$output_archive")" || packaging_status=71
  printf '%s\n' "$worker_status" >"${work_root}/worker_exit_status.txt" \
    || packaging_status=71
  tar -czf "$output_archive" -C "$work_root" . || packaging_status=71
  if [[ "$packaging_status" -ne 0 || ! -s "$output_archive" ]]; then
    echo "attempt packaging failed" >&2
    exit 71
  fi
  exit "$worker_status"
}
trap package_attempt EXIT

sandbox_root="$(pwd -P)"
"$runtime_bin" exec \
  --cleanenv \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env STATIC_ADAPT_HH_POOL_CACHE=off \
  --env STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off \
  --bind "${sandbox_root}:${sandbox_root}" \
  --pwd "${sandbox_root}/${work_root}" \
  "$image_path" \
  python3 "${sandbox_root}/${package_dir}/run_cell.py" \
    --run \
    --job "${sandbox_root}/${job_path}" \
    --execution-authorization "${sandbox_root}/${authorization_path}" \
    --output-dir "${sandbox_root}/${work_root}/runs/${execution_id}" \
    --receipt "${sandbox_root}/${work_root}/worker_receipt.json" \
    --scheduler-cluster-id "$cluster_id" \
    --scheduler-proc-id "$proc_id" \
    --scheduler-attempt-ordinal "$attempt_ordinal"

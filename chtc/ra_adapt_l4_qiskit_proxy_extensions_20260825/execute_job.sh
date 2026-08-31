#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 13 ]]; then
  echo "usage: execute_job.sh PACKAGE SOURCE SOURCE_SHA PLAN AUTH IMAGE IMAGE_SHA EXECUTION_ID ARM SELECTOR_COST CPUS OUTPUT CLUSTER_PROC" >&2
  exit 64
fi

package_dir="$1"
source_archive="$2"
source_sha="$3"
plan_path="$4"
authorization_path="$5"
image_path="$6"
image_sha="$7"
execution_id="$8"
arm="$9"
selector_cost="${10}"
allocated_cpus="${11}"
output_archive="${12}"
cluster_proc="${13}"
pinned_image_sha="fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

if [[ ! "$execution_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "unsafe execution id" >&2
  exit 65
fi
case "$output_archive" in
  /*|*".."*|*"//"*) echo "unsafe output archive" >&2; exit 65 ;;
esac

hash_file() {
  sha256sum "$1" | awk '{print $1}'
}

for path in "$source_archive" "$plan_path" "$authorization_path" "$image_path" "${package_dir}/worker.py"; do
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "missing or unsafe input: $path" >&2
    exit 66
  fi
done
[[ "$(hash_file "$source_archive")" == "$source_sha" ]] || { echo "source hash mismatch" >&2; exit 67; }
[[ "$image_sha" == "$pinned_image_sha" && "$(hash_file "$image_path")" == "$pinned_image_sha" ]] || { echo "image hash mismatch" >&2; exit 67; }

runtime_bin="$(command -v apptainer || command -v singularity || true)"
[[ -n "$runtime_bin" ]] || { echo "container runtime unavailable" >&2; exit 68; }

attempt="attempt_${execution_id}_${cluster_proc}"
[[ ! -e "$attempt" && ! -L "$attempt" && ! -e "$output_archive" && ! -L "$output_archive" ]] || { echo "attempt output exists" >&2; exit 69; }
mkdir "$attempt"

package_attempt() {
  rc="$?"
  trap - EXIT
  set +e
  mkdir -p "$(dirname "$output_archive")"
  printf '%s\n' "$rc" >"${attempt}/worker_exit_status.txt"
  tar -czf "$output_archive" -C "$attempt" .
  [[ -s "$output_archive" ]] || exit 71
  exit "$rc"
}
trap package_attempt EXIT

mkdir "${attempt}/source"
tar -xzf "$source_archive" -C "${attempt}/source"
sandbox="$(pwd -P)"
mkdir -p "${sandbox}/diet/ledger_stream" "${sandbox}/diet/history_spill"

"$runtime_bin" exec --cleanenv \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONHASHSEED=0 \
  --env PYTHONPATH="${sandbox}/${attempt}/source" \
  --env OPENBLAS_NUM_THREADS=1 \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env VECLIB_MAXIMUM_THREADS=1 \
  --env NUMEXPR_NUM_THREADS=1 \
  --env BLIS_NUM_THREADS=1 \
  --env OMP_DYNAMIC=FALSE \
  --env MKL_DYNAMIC=FALSE \
  --env STATIC_ADAPT_HH_POOL_CACHE=off \
  --env STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off \
  --env STATIC_ADAPT_ALLOCATED_CPUS="$allocated_cpus" \
  --env STATIC_ADAPT_ESTIMATOR_LEDGER_STREAM_DIR="${sandbox}/diet/ledger_stream" \
  --env STATIC_ADAPT_RUN_HISTORY_SPILL_DIR="${sandbox}/diet/history_spill" \
  --env QISKIT_NUM_PROCS=1 \
  --env QISKIT_PARALLEL=FALSE \
  --env RAYON_NUM_THREADS=1 \
  --bind "${sandbox}:${sandbox}" \
  --pwd "${sandbox}" \
  "$image_path" \
  python3 "${sandbox}/${package_dir}/worker.py" \
    --source-root "${sandbox}/${attempt}/source" \
    --plan "${sandbox}/${plan_path}" \
    --authorization "${sandbox}/${authorization_path}" \
    --execution-id "$execution_id" \
    --arm "$arm" \
    --selector-cost "$selector_cost" \
    --output "${sandbox}/${attempt}/run"

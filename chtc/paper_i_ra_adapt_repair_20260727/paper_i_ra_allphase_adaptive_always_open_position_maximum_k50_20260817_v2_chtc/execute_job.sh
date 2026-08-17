#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 10 ]]; then
  echo "usage: execute_job.sh PACKAGE JOB PROTOCOL AUTH SOURCE SOURCE_SHA IMAGE IMAGE_SHA EXECUTION_ID OUTPUT" >&2
  exit 64
fi

package_dir="$1"
job_path="$2"
protocol_path="$3"
authorization_path="$4"
source_archive="$5"
source_sha="$6"
image_path="$7"
image_sha="$8"
execution_id="$9"
output_archive="${10}"
pinned_image_sha="fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

if [[ ! "$execution_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "unsafe execution id" >&2
  exit 65
fi
case "$output_archive" in
  /*|*".."*|*"//"*) echo "unsafe output archive" >&2; exit 65 ;;
esac

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

for path in "$job_path" "$protocol_path" "$authorization_path" "$source_archive" "$image_path" "${package_dir}/worker.py"; do
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "missing or unsafe input: $path" >&2
    exit 66
  fi
done
[[ "$(hash_file "$source_archive")" == "$source_sha" ]] || { echo "source hash mismatch" >&2; exit 67; }
[[ "$image_sha" == "$pinned_image_sha" && "$(hash_file "$image_path")" == "$pinned_image_sha" ]] || { echo "image hash mismatch" >&2; exit 67; }

runtime_bin="$(command -v apptainer || command -v singularity || true)"
[[ -n "$runtime_bin" ]] || { echo "container runtime unavailable" >&2; exit 68; }

attempt="attempt_${execution_id}"
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
  --env STATIC_ADAPT_ALLOCATED_CPUS=4 \
  --env QISKIT_NUM_PROCS=1 \
  --env QISKIT_PARALLEL=FALSE \
  --env RAYON_NUM_THREADS=1 \
  --bind "${sandbox}:${sandbox}" \
  --pwd "${sandbox}" \
  "$image_path" \
  python3 "${sandbox}/${package_dir}/worker.py" \
    --job "${sandbox}/${job_path}" \
    --protocol "${sandbox}/${protocol_path}" \
    --authorization "${sandbox}/${authorization_path}" \
    --output "${sandbox}/${attempt}/run"

#!/usr/bin/env bash
set -euo pipefail

JOB_MANIFEST="${1:?job manifest required}"
SOURCE_ARCHIVE="${2:?source archive required}"
EXPECTED_SOURCE_SHA256="${3:?source SHA-256 required}"
IMAGE="${4:?image required}"
EXPECTED_IMAGE_SHA256="${5:?image SHA-256 required}"
JOB_ID="${6:?job id required}"
BUNDLE_ID="paper_i_hh_fm_vs_append_fm_first_hit_weak_weak_20260720_v1_chtc"
OUTPUT_ROOT="raw_outputs/${BUNDLE_ID}/${JOB_ID}"
TRANSFER_ROOT="${OUTPUT_ROOT}/transfer"
TRANSFER_ARCHIVE="raw_outputs/${BUNDLE_ID}/${JOB_ID}_transfer.tar.gz"

[[ "$JOB_ID" == "weak_weak_fm_vs_append_fm_first_hit" ]] || {
  echo "unexpected job id" >&2
  exit 2
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

package_outputs() {
  local status="$?"
  trap - EXIT
  mkdir -p "$TRANSFER_ROOT" "$(dirname "$TRANSFER_ARCHIVE")"
  if [[ "$status" -ne 0 ]]; then
    local campaign_root="${OUTPUT_ROOT}/campaign"
    local relative
    for relative in \
      "campaign_manifest.json" \
      "settings_diff.json" \
      "source_lock/formal_manifold_config.json" \
      "source_lock/source_lock.json" \
      "weak-weak/pair_manifest.json" \
      "weak-weak/fm_snake/plan.json" \
      "weak-weak/fm_snake/current.json" \
      "weak-weak/fm_snake/result.json" \
      "weak-weak/projected_singleton_append_fm/plan.json" \
      "weak-weak/projected_singleton_append_fm/partial_result.json" \
      "weak-weak/projected_singleton_append_fm/adapt_iteration_progress.jsonl" \
      "weak-weak/projected_singleton_append_fm/result.json"; do
      if [[ -f "${campaign_root}/${relative}" ]]; then
        mkdir -p "${TRANSFER_ROOT}/recovery/$(dirname "$relative")"
        cp "${campaign_root}/${relative}" "${TRANSFER_ROOT}/recovery/${relative}"
      fi
    done
  fi
  tar -czf "${TRANSFER_ARCHIVE}.tmp" "$TRANSFER_ROOT" || status=74
  if [[ -f "${TRANSFER_ARCHIVE}.tmp" ]]; then
    mv "${TRANSFER_ARCHIVE}.tmp" "$TRANSFER_ARCHIVE"
  fi
  exit "$status"
}
trap package_outputs EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

for path in "$JOB_MANIFEST" "$SOURCE_ARCHIVE" "$IMAGE"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ "$(sha256_file "$SOURCE_ARCHIVE")" == "$EXPECTED_SOURCE_SHA256" ]] || {
  echo "source archive hash mismatch" >&2
  exit 3
}
[[ "$(sha256_file "$IMAGE")" == "$EXPECTED_IMAGE_SHA256" ]] || {
  echo "image hash mismatch" >&2
  exit 3
}

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v singularity)"
else
  echo "apptainer/singularity unavailable" >&2
  exit 127
fi

mkdir -p source_locked "$TRANSFER_ROOT"
tar -xzf "$SOURCE_ARCHIVE" -C source_locked
ROOT="$PWD"
QISKIT_PREFLIGHT='import qiskit; assert qiskit.__version__ == "2.3.1"; from pipelines.qiskit_backend_tools import load_local_fake_backend; backend, resolved = load_local_fake_backend("FakeMarrakesh"); assert backend is not None and "marrakesh" in str(resolved).lower()'
"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'set -euo pipefail; cd /work; export PYTHONPATH=/work/source_locked PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; python3 -c "$3"; python3 -u "$1" "$2" "$4"' \
  -- "chtc/phase3_optuna/input/${BUNDLE_ID}/run_job.py" "$JOB_MANIFEST" "$QISKIT_PREFLIGHT" "$OUTPUT_ROOT"

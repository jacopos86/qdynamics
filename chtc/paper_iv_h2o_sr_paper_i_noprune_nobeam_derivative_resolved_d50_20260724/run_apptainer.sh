#!/usr/bin/env bash
set -euo pipefail

STAGE="chtc/paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_d50_20260724"
INPUT="$STAGE/input"
IMAGE="${PROJECT_IMAGE:-chtc/phase3_optuna/image.sif}"
DELIVERABLE="deliverables/paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_from_zero_d50_20260724_v2"

mkdir -p \
  raw_outputs \
  logs \
  runtime_source \
  runtime_inputs \
  runtime_cache/candidate_records \
  "$DELIVERABLE"
printf '{"state":"worker_starting"}\n' > "$DELIVERABLE/worker_status.json"

python3 - "$INPUT/input_manifest.json" "$INPUT" "$0" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
input_dir = Path(sys.argv[2])
running_executable = Path(sys.argv[3])
for name, entry in manifest["files"].items():
    path = input_dir / name
    if not path.is_file() or sha256(path) != entry["sha256"]:
        raise SystemExit(f"Input hash validation failed: {path}")
stage_dir = input_dir.parent
for name, entry in manifest.get("runtime_files", {}).items():
    path = running_executable if name == "run_apptainer.sh" else stage_dir / name
    if not path.is_file() or sha256(path) != entry["sha256"]:
        raise SystemExit(f"Runtime-file hash validation failed: {path}")
if manifest.get("route_difference_audit_status") != "pass":
    raise SystemExit("Paper-I route difference audit did not pass.")
print(json.dumps({"schema": manifest["schema"], "status": "pass"}, sort_keys=True))
PY

test -f "$IMAGE"
tar -xzf "$INPUT/source_tree.tar.gz" -C runtime_source
cp "$INPUT/h2o_fixture.json" runtime_inputs/h2o_fixture.json
cp "$INPUT/command.json" runtime_inputs/command.json

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v singularity)"
else
  echo "Neither apptainer nor singularity is available." >&2
  exit 127
fi

export APPTAINERENV_H2O_HEARTBEAT_SEC="${H2O_HEARTBEAT_SEC:-300}"
export APPTAINERENV_STATIC_ADAPT_CANDIDATE_RECORD_CACHE=disk
export APPTAINERENV_STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR=/work/runtime_cache/candidate_records
export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_VECLIB_MAXIMUM_THREADS=1
export APPTAINERENV_NUMEXPR_NUM_THREADS=1
export SINGULARITYENV_H2O_HEARTBEAT_SEC="$APPTAINERENV_H2O_HEARTBEAT_SEC"
export SINGULARITYENV_STATIC_ADAPT_CANDIDATE_RECORD_CACHE=disk
export SINGULARITYENV_STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR=/work/runtime_cache/candidate_records
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_VECLIB_MAXIMUM_THREADS=1
export SINGULARITYENV_NUMEXPR_NUM_THREADS=1

ROOT="$PWD"
exec "$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  python3 -u "/work/$STAGE/run_payload.py" \
  --root /work \
  --command-json /work/runtime_inputs/command.json \
  --preflight "/work/$STAGE/runtime_preflight.py" \
  --validator "/work/$STAGE/validate_result.py"

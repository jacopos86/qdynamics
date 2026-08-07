#!/usr/bin/env bash
set -euo pipefail

STAGE="chtc/paper_iv_h2o_sr_source_locked_resume_d12_20260716"
INPUT="$STAGE/input"
IMAGE="${PROJECT_IMAGE:-chtc/phase3_optuna/image.sif}"

mkdir -p raw_outputs logs runtime_source runtime_inputs

python3 - "$INPUT/input_manifest.json" "$INPUT" <<'PY'
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
for entry in manifest["files"].values():
    path = input_dir / entry["name"]
    if not path.is_file() or sha256(path) != entry["sha256"]:
        raise SystemExit(f"Input hash validation failed: {path}")
print(json.dumps({"schema": manifest["schema"], "status": "pass"}, sort_keys=True))
PY

test -f "$IMAGE"
tar -xzf "$INPUT/source_tree_no_beam_ablation_v1.tar.gz" -C runtime_source
cp "$INPUT/h2o_fixture.json" runtime_inputs/h2o_fixture.json
cp "$INPUT/h2o_depth12_current.json" runtime_inputs/h2o_depth12_current.json
cp "$INPUT/command.full.json" runtime_inputs/command.full.json

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v singularity)"
else
  echo "Neither apptainer nor singularity is available." >&2
  exit 127
fi

export APPTAINERENV_H2O_HEARTBEAT_SEC="${H2O_HEARTBEAT_SEC:-300}"
export APPTAINERENV_STATIC_ADAPT_CANDIDATE_RECORD_CACHE=memory
export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_VECLIB_MAXIMUM_THREADS=1
export APPTAINERENV_NUMEXPR_NUM_THREADS=1
export SINGULARITYENV_H2O_HEARTBEAT_SEC="$APPTAINERENV_H2O_HEARTBEAT_SEC"
export SINGULARITYENV_STATIC_ADAPT_CANDIDATE_RECORD_CACHE=memory
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_VECLIB_MAXIMUM_THREADS=1
export SINGULARITYENV_NUMEXPR_NUM_THREADS=1

ROOT="$PWD"
exec "$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  python3 -u "/work/$STAGE/run_payload.py" \
  --root /work \
  --command-json /work/runtime_inputs/command.full.json \
  --validator "/work/$STAGE/validate_continuation.py"

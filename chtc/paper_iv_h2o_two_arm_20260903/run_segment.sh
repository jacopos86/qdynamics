#!/usr/bin/env bash
# paper_iv_h2o_two_arm_20260903 -- one bounded segment of one arm.
#
# Runs on the execute node inside the scratch directory HTCondor gives us.
# Contract: this script EXITS NORMALLY well before the 72h pool wall because
# h2o_application stops between controller rounds at --max-wall-seconds and
# returns. Only a normal exit stages `results/` out (when_to_transfer_output =
# ON_EXIT); a hold never does. Nothing here traps SIGTERM: there is no
# graceful-stop path in the engine, so the wall budget is the protection.
#
# usage (from the submit file):
#   run_segment.sh ARM SEG CANDIDATE_REPRESENTATION COST_MODE MAX_WALL_SECONDS \
#                  HORIZON_ROUNDS RESULTS_MODE RESUME_NAME
#   RESUME_NAME is `current.json` for segments > 00, `-` for seg00.
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 ARM SEG REP COST_MODE MAX_WALL_SECONDS HORIZON RESULTS_MODE RESUME_NAME" >&2
  exit 64
fi
ARM="$1"; SEG="$2"; REP="$3"; COST_MODE="$4"; MAX_WALL="$5"
HORIZON="$6"; RESULTS_MODE="$7"; RESUME_NAME="$8"
TAG="${ARM}_seg${SEG}"
FIXTURE="h2o_linear_fd_work373_ref484_fixture_20260826.json"
IMAGE="${PROJECT_IMAGE:-image.sif}"
ROOT="$PWD"
START_UTC="$(date -u +%FT%TZ)"
START_EPOCH="$(date +%s)"

case "$REP" in
  macro_generator_v1|sector_complete_pauli_block_v1) ;;
  *) echo "refusing candidate representation '$REP' (single_pauli_word_v1 is refused by h2o_application for H2O)" >&2; exit 65 ;;
esac
case "$COST_MODE" in proxy|marrakesh_graph_span_v1) ;; *) echo "unknown cost mode '$COST_MODE'" >&2; exit 65 ;; esac
case "$RESULTS_MODE" in home|staging) ;; *) echo "unknown results mode '$RESULTS_MODE'" >&2; exit 65 ;; esac
[[ "$MAX_WALL" =~ ^[0-9]+$ && "$MAX_WALL" -gt 0 ]] || { echo "bad MAX_WALL_SECONDS '$MAX_WALL'" >&2; exit 65; }
[[ "$HORIZON" =~ ^[0-9]+$ && "$HORIZON" -gt 0 ]] || { echo "bad HORIZON '$HORIZON'" >&2; exit 65; }

echo "[$TAG] start $START_UTC host=$(hostname) pwd=$ROOT"

# --- input integrity (sha256sum is coreutils; no host python assumed) -------
[[ -f input_manifest.sha256 ]] || { echo "missing input_manifest.sha256" >&2; exit 66; }
sha256sum -c --strict input_manifest.sha256
[[ -f "$IMAGE" ]] || { echo "missing Apptainer image: $IMAGE" >&2; exit 66; }
[[ -f "$FIXTURE" ]] || { echo "missing fixture: $FIXTURE" >&2; exit 66; }

mkdir -p runtime_source results audit_cache transfer resume
tar -xzf source.tgz -C runtime_source
[[ -f runtime_source/pipelines/static_adapt/ra_adapt/h2o_application.py ]] \
  || { echo "source.tgz does not contain h2o_application.py" >&2; exit 66; }

# --- resume staging: the checkpoint must never sit inside --output-dir -----
RESUME_ARGS=()
if [[ "$RESUME_NAME" != "-" ]]; then
  [[ -f resume_manifest.sha256 ]] || { echo "resume requested but resume_manifest.sha256 is missing" >&2; exit 66; }
  sha256sum -c --strict resume_manifest.sha256
  while read -r _sha name; do
    [[ -n "$name" ]] || continue
    mv -- "$name" "resume/$name"
  done < resume_manifest.sha256
  [[ -f "resume/$RESUME_NAME" ]] || { echo "resume checkpoint resume/$RESUME_NAME missing after staging" >&2; exit 66; }
  RESUME_ARGS=(--resume-checkpoint "/work/resume/$RESUME_NAME")
  echo "[$TAG] resuming from resume/$RESUME_NAME"
fi

# --- container runtime ----------------------------------------------------
if command -v apptainer >/dev/null 2>&1; then APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then APPTAINER_BIN="$(command -v singularity)"
else echo "Neither apptainer nor singularity is available." >&2; exit 127; fi

# Env pins mirror the lost run (d50pkg/run_d50.sh) plus the opt-in compact
# checkpoint layout from 3b1b86f4. Sector-audit cache goes to scratch, which
# is not transferred.
CONTAINER_ENV=(
  PYTHONPATH=/work/runtime_source
  PYTHONUNBUFFERED=1
  OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1
  VECLIB_MAXIMUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  STATIC_ADAPT_PAULI_SIGN_CACHE_MAX_NQ=19
  STATIC_ADAPT_PAULI_SIGN_CACHE_MAX_ENTRIES=4096
  STATIC_ADAPT_SECTOR_AUDIT_MEMO_MAX=262144
  STATIC_ADAPT_SECTOR_AUDIT_CACHE=disk
  STATIC_ADAPT_SECTOR_AUDIT_CACHE_DIR=/work/audit_cache
  STATIC_ADAPT_CHECKPOINT_JSON_LAYOUT=compact
  PAPER_IV_SEGMENT_TAG="$TAG"
)

CMD=(
  python3 -u -m pipelines.static_adapt.ra_adapt.h2o_application
  --fixture "/work/$FIXTURE"
  --output-dir /work/results
  --maximum-controller-rounds "$HORIZON"
  --insertion append_only
  --checkpoint-history-tail 1
  --candidate-representation "$REP"
  --cost-mode "$COST_MODE"
  --max-wall-seconds "$MAX_WALL"
)
if ((${#RESUME_ARGS[@]})); then CMD+=("${RESUME_ARGS[@]}"); fi

printf '%s\n' "${CMD[@]}" > results/command_argv.txt
echo "[$TAG] launching: ${CMD[*]}"

set +e
"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" --pwd /work/runtime_source "$IMAGE" \
  env "${CONTAINER_ENV[@]}" "${CMD[@]}"
RC=$?
set -e
END_UTC="$(date -u +%FT%TZ)"
END_EPOCH="$(date +%s)"
echo "[$TAG] python exit=$RC elapsed=$((END_EPOCH - START_EPOCH))s"

# --- receipt: written inside the container so no host python is assumed ----
IMAGE_SHA="$(sha256sum "$IMAGE" | cut -d' ' -f1)"
CLUSTER_ID=""; PROC_ID=""
if [[ -n "${_CONDOR_JOB_AD:-}" && -f "${_CONDOR_JOB_AD}" ]]; then
  CLUSTER_ID="$(awk -F' = ' '$1=="ClusterId"{print $2}' "$_CONDOR_JOB_AD" | head -1)"
  PROC_ID="$(awk -F' = ' '$1=="ProcId"{print $2}' "$_CONDOR_JOB_AD" | head -1)"
fi
export TAG ARM SEG REP COST_MODE MAX_WALL HORIZON RESULTS_MODE RESUME_NAME \
  START_UTC END_UTC START_EPOCH END_EPOCH RC IMAGE_SHA CLUSTER_ID PROC_ID
"$APPTAINER_BIN" exec --bind "$ROOT:/work" "$IMAGE" python3 - <<'PY'
import hashlib, json, os, socket
from pathlib import Path
results = Path("/work/results")
def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
files = {
    p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
    for p in sorted(results.iterdir())
    if p.is_file() and p.name != "segment_receipt.json"
}
summary = None
if (results / "summary.json").is_file():
    summary = json.loads((results / "summary.json").read_text())
receipt = {
    "schema": "paper_iv_h2o_two_arm_segment_receipt_v1",
    "campaign": "paper_iv_h2o_two_arm_20260903",
    "segment_tag": os.environ["TAG"],
    "arm": os.environ["ARM"],
    "segment": int(os.environ["SEG"]),
    "candidate_representation": os.environ["REP"],
    "cost_mode": os.environ["COST_MODE"],
    "max_wall_seconds": int(os.environ["MAX_WALL"]),
    "horizon_rounds": int(os.environ["HORIZON"]),
    "results_mode": os.environ["RESULTS_MODE"],
    "resumed_from": None if os.environ["RESUME_NAME"] == "-" else os.environ["RESUME_NAME"],
    "start_utc": os.environ["START_UTC"],
    "end_utc": os.environ["END_UTC"],
    "elapsed_seconds": int(os.environ["END_EPOCH"]) - int(os.environ["START_EPOCH"]),
    "python_exit_code": int(os.environ["RC"]),
    "image_sha256": os.environ["IMAGE_SHA"],
    "hostname": socket.gethostname(),
    "cluster_id": os.environ.get("CLUSTER_ID") or None,
    "proc_id": os.environ.get("PROC_ID") or None,
    "result_json_present": (results / "result.json").is_file(),
    "current_json_present": (results / "current.json").is_file(),
    "summary": summary,
    "files": files,
}
(results / "segment_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: receipt[k] for k in ("segment_tag", "python_exit_code", "elapsed_seconds", "result_json_present", "current_json_present")}))
PY

if [[ "$RESULTS_MODE" == "staging" ]]; then
  tar -czf "transfer/${TAG}_results.tar.gz" -C "$ROOT" results
  sha256sum "transfer/${TAG}_results.tar.gz"
fi

echo "[$TAG] done $END_UTC exit=$RC"
exit "$RC"

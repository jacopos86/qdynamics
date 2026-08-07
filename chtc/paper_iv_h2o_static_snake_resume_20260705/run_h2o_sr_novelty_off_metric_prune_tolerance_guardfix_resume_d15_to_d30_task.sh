#!/usr/bin/env bash
set -euo pipefail

RECORD_ID="${1:-paper_iv_h2o_sr_snake_novelty_off_metric_prune_tolerance_guardfix_powell_resume_d15_to_d30}"
ROOT="$PWD"
OUTDIR="$ROOT/raw_outputs/$RECORD_ID"
LOGDIR="$ROOT/logs/$RECORD_ID"
PROGRESS_DIR="$OUTDIR/progress"
RESULT_JSON="$OUTDIR/result.json"
CURRENT_JSON="$OUTDIR/current.json"
ESTIMATOR_LEDGER_JSON="$OUTDIR/estimator_call_ledger.json"
STDOUT_LOG="$OUTDIR/stdout.log"
STDERR_LOG="$OUTDIR/stderr.log"
COMMAND_TXT="$OUTDIR/command.txt"
MANIFEST_JSON="$OUTDIR/submit_manifest.json"
SHELL_STATUS_JSON="$PROGRESS_DIR/shell_status.json"

FIXTURE_JSON="${H2O_FIXTURE_JSON:-tmp/h2o_linear_fd_valence_psi4_optimized/h2o_linear_fd_sparse_fixture_nph1_ref2_reencoded_v2.json}"
RESUME_SCAFFOLD_JSON="${H2O_RESUME_SCAFFOLD_JSON:-chtc/paper_iv_h2o_static_snake_resume_20260705/input/seed_artifacts/paper_iv_h2o_sr_snake_guardfix_depth15_cluster8790143_current.json}"
CODE_ARCHIVE="${H2O_CODE_ARCHIVE:-chtc/paper_iv_h2o_static_snake_resume_20260705/input/h2o_sr_snake_novelty_off_metric_prune_tolerance_guardfix_code_20260714.tgz}"
EXPECTED_FIXTURE_SHA256="570690bd126787305b340bd2f7493499c0f3101e3e2820c2d355c55c16afa594"
EXPECTED_RESUME_SHA256="7e081032f63f30253725d46c7b552aee2ed38b993b9844b54b7edc6e8044f05e"
RESUME_DEPTH="${H2O_RESUME_DEPTH:-15}"
ADAPT_MAX_DEPTH="${H2O_ADAPT_MAX_DEPTH:-15}"
ADAPT_SEGMENT_ID="${H2O_ADAPT_SEGMENT_ID:-paper_iv_h2o_sr_snake_novelty_off_metric_prune_tolerance_guardfix_powell_resume_d15_to_d30}"
ADAPT_SEGMENT_MAX_NEW_ADMISSIONS="${H2O_ADAPT_SEGMENT_MAX_NEW_ADMISSIONS:-15}"
PREFLIGHT_ONLY="${H2O_PREFLIGHT_ONLY:-0}"

mkdir -p "$OUTDIR" "$LOGDIR" "$PROGRESS_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"
export STATIC_ADAPT_CANDIDATE_RECORD_CACHE="${STATIC_ADAPT_CANDIDATE_RECORD_CACHE:-memory}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

python3 - "$FIXTURE_JSON" "$EXPECTED_FIXTURE_SHA256" \
  "$RESUME_SCAFFOLD_JSON" "$EXPECTED_RESUME_SHA256" "$RESUME_DEPTH" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def digest(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"Missing required input: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


fixture_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[3])
fixture_sha = digest(fixture_path)
checkpoint_sha = digest(checkpoint_path)
if fixture_sha != sys.argv[2]:
    raise SystemExit(
        f"Fixture SHA-256 mismatch: {fixture_sha} != {sys.argv[2]}"
    )
if checkpoint_sha != sys.argv[4]:
    raise SystemExit(
        f"Checkpoint SHA-256 mismatch: {checkpoint_sha} != {sys.argv[4]}"
    )
fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
expected_fixture = {
    "family_key": "molecular_vibronic_h2o_linear_fd",
    "basis": "sto-3g",
    "active_space": "cas_8e_6o_valence",
    "num_particles": [4, 4],
    "n_total_qubits": 15,
    "n_ph_max_by_mode": [1, 1, 1],
}
observed_fixture = {
    "family_key": fixture.get("manifest", {}).get("family_key"),
    "basis": fixture.get("geometry", {}).get("basis"),
    "active_space": fixture.get("active_space", {}).get("active_space_kind"),
    "num_particles": fixture.get("active_space", {}).get("num_particles"),
    "n_total_qubits": fixture.get("register_layout", {}).get("n_total_qubits"),
    "n_ph_max_by_mode": fixture.get("physical_sector", {}).get("n_ph_max_by_mode"),
}
if observed_fixture != expected_fixture:
    raise SystemExit(
        "Unexpected H2O fixture contract: "
        + json.dumps({"expected": expected_fixture, "observed": observed_fixture}, sort_keys=True)
    )
observed_depth = checkpoint.get("adapt_vqe", {}).get("ansatz_depth")
if int(observed_depth) != int(sys.argv[5]):
    raise SystemExit(
        f"Resume depth mismatch: checkpoint has {observed_depth}, expected {sys.argv[5]}"
    )
print(
    "H2O_INPUT_CONTRACT "
    + json.dumps(
        {
            "fixture_sha256": fixture_sha,
            "checkpoint_sha256": checkpoint_sha,
            "starting_depth": int(observed_depth),
            **observed_fixture,
        },
        sort_keys=True,
    ),
    flush=True,
)
PY

append_wrapper_event() {
  local event="$1"
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"schema":"paper_iv_h2o_shell_event_v2","timestamp_utc":"%s","record_id":"%s","event":"%s"}\n' \
    "$ts" "$RECORD_ID" "$event" >> "$PROGRESS_DIR/wrapper_events.jsonl" || true
}

write_shell_status() {
  local state="$1"
  local code="$2"
  python3 - "$SHELL_STATUS_JSON" "$RECORD_ID" "$state" "$code" <<'PY' || true
from __future__ import annotations

import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(
        {
            "schema": "paper_iv_h2o_shell_status_v2",
            "record_id": sys.argv[2],
            "state": sys.argv[3],
            "returncode": int(sys.argv[4]),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

emit_heartbeat() {
  python3 - "$OUTDIR" "$RECORD_ID" <<'PY' || true
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
current_path = out / "current.json"
payload = {
    "schema": "paper_iv_h2o_condor_progress_v2",
    "record_id": sys.argv[2],
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "current_json_exists": current_path.exists(),
}
if current_path.exists():
    try:
        current = json.loads(current_path.read_text(encoding="utf-8"))
    except Exception as exc:
        payload["current_json_read_error"] = str(exc)
    else:
        adapt = current.get("adapt_vqe", {}) if isinstance(current, dict) else {}
        progress = adapt.get("progress", {}) if isinstance(adapt, dict) else {}
        segment = adapt.get("adapt_segment", {}) if isinstance(adapt, dict) else {}
        payload.update(
            {
                "ansatz_depth": adapt.get("ansatz_depth") or progress.get("depth"),
                "energy": adapt.get("energy") or progress.get("energy"),
                "abs_delta_e": adapt.get("abs_delta_e") or progress.get("delta_abs_current"),
                "segment_new_admissions": segment.get("new_admission_records"),
                "stop_reason_so_far": progress.get("stop_reason_so_far"),
            }
        )
print("H2O_PROGRESS " + json.dumps(payload, sort_keys=True, default=str), flush=True)
PY
}

CMD=(
  python3
  -u
  -m
  pipelines.static_adapt.adapt_pipeline
  --problem
  molecular_vibronic_h2o_linear_fd
  --molecular-vibronic-h2o-linear-fd-fixture-json
  "$FIXTURE_JSON"
  --L
  6
  --n-ph-max
  1
  --term-order
  sorted
  --ordering
  blocked
  --boson-encoding
  binary
  --boundary
  open
  --include-zero-point
  --skip-trajectory
  --skip-pdf
  --adapt-pool
  full_meta
  --static-route-id
  route_a
  --static-meta-feature-profile
  paper_i_production_v1
  --static-lane-route
  physical_operator_type
  --physical-lane-shortlist-aggressiveness
  3
  --adapt-continuation-mode
  phase3_v1
  --historical-singleton-coordinate-solve-policy
  supported_metric_whitened_eigh_v1
  --historical-singleton-coordinate-solve-scope
  phase3_only_v1
  --historical-singleton-trust-region-update-policy
  displacement_calibrated_unbounded_v2
  --sr-escape-mode
  disabled
  --phase0-no-pilot
  --phase0-algebraic-lane-mode
  off
  --phase1-shortlist-size
  24
  --phase2-shortlist-size
  12
  --phase2-shortlist-fraction
  0.25
  --phase2-no-batching
  --phase3-no-batching
  --allow-archival-phase3-runtime-split
  --phase3-runtime-split-mode
  shortlist_pauli_children_v1
  --phase3-runtime-split-selection-mode
  archival_child_set_forward_v1
  --phase3-runtime-split-max-subset-size
  1
  --phase3-runtime-split-subset-sizes
  1
  --phase3-runtime-split-child-set-symmetry-policy
  hard_guard
  --adapt-inner-optimizer
  POWELL
  --adapt-maxiter
  300
  --adapt-scipy-maxfev
  20000
  --adapt-max-depth
  "$ADAPT_MAX_DEPTH"
  --adapt-resume-scaffold-json
  "$RESUME_SCAFFOLD_JSON"
  --adapt-segment-id
  "$ADAPT_SEGMENT_ID"
  --adapt-segment-max-new-admissions
  "$ADAPT_SEGMENT_MAX_NEW_ADMISSIONS"
  --adapt-reopt-policy
  full
  --adapt-window-size
  99
  --adapt-window-topk
  0
  --phase3-geometry-window-size
  99
  --adapt-full-refit-every
  1
  --adapt-final-full-refit
  true
  --adapt-final-refit-maxiter
  300
  --adapt-insertion-mode
  always
  --adapt-no-finite-angle-fallback
  --adapt-allow-repeats
  --adapt-eps-grad
  5e-7
  --adapt-eps-energy
  1e-9
  --adapt-schur-warm-start-mode
  append-prune
  --adapt-seed
  7
  --adapt-state-backend
  compiled
  --adapt-beam-live-branches
  3
  --adapt-beam-children-per-parent
  2
  --adapt-beam-terminated-keep
  0
  --adapt-beam-terminal-archive-mode
  disabled
  --adapt-beam-lambda
  0.0
  --adapt-beam-parent-workers
  1
  --adapt-parallel-gradient-workers
  8
  --phase1-prune-enabled
  --phase1-prune-policy
  recoverability_ladder_v1
  --phase1-prune-mode
  both
  --phase1-prune-fraction
  0.25
  --phase1-prune-schur-nomination-route
  metric_regularized_v1
  --phase1-prune-metric-schur-mu
  0.01
  --phase1-prune-metric-schur-solve-mode
  stationary_gw_zero_v1
  --phase1-prune-metric-schur-cost-weighting
  ansatz_entry_denominator_v1
  --phase1-prune-amplitude-witness-optional
  --phase1-prune-max-candidates
  6
  --phase1-prune-checkpoint-period
  3
  --phase1-score-mode
  trust_region_v1
  --phase1-probe-max-positions
  999999
  --phase1-trough-margin-ratio
  1.0
  --phase1-compile-position-shift-weight
  0
  --phase2-compile-position-shift-weight
  0
  --phase2-family-repeat-cost-scale
  0
  --phase2-motif-bonus-weight
  0
  --phase2-duplicate-penalty-weight
  0
  --phase-live-hysteresis-disabled
  --phase2-rho
  0.25
  --phase2-novelty-mode
  collective_span_v1
  --phase3-novelty-ablation-mode
  all
  --phase2-selector-gain-mode
  trust_region_v1
  --phase3-selector-policy
  algebraic_nested_v1
  --phase3-selector-geometry-mode
  reduced
  --phase3-window-relaxation-mode
  reduced
  --phase3-no-rescue
  --phase3-lifetime-cost-mode
  off
  --phase3-backend-cost-mode
  proxy
  --phase3-batch-selection-mode
  reduced_plane
  --phase3-batch-order-selection-mode
  finite_step_v1
  --phase3-batch-prefilter-mode
  off
  --phase3-oracle-inner-objective-mode
  exact
  --hardware-resolution-mode
  ideal
  --shared-pauli-pool-mode
  off
  --adapt-child-pool-expansion-mode
  off
  --phase1-lambda-2q
  0.20
  --phase1-lambda-d
  0.20
  --phase1-lambda-1q
  0.05
  --phase1-lambda-theta
  0.05
  --phase1-lambda-shot
  0.15
  --phase2-lambda-2q
  0.20
  --phase2-lambda-d
  0.20
  --phase2-lambda-1q
  0.05
  --phase2-lambda-theta
  0.05
  --phase2-lambda-shot
  0.15
  --output-json
  "$RESULT_JSON"
  --adapt-current-json
  "$CURRENT_JSON"
  --adapt-current-json-every-depth
  1
  --adapt-estimator-call-ledger-json
  "$ESTIMATOR_LEDGER_JSON"
)

case "$PREFLIGHT_ONLY" in
  1|true|yes|on)
    CMD+=(
      --adapt-max-depth
      0
      --adapt-segment-max-new-admissions
      0
      --adapt-final-full-refit
      false
      --adapt-maxiter
      0
      --adapt-scipy-maxfev
      1
    )
    ;;
  0|false|no|off)
    ;;
  *)
    echo "Invalid H2O_PREFLIGHT_ONLY=$PREFLIGHT_ONLY" >&2
    exit 2
    ;;
esac

printf '%q ' "${CMD[@]}" > "$COMMAND_TXT"
printf '\n' >> "$COMMAND_TXT"

python3 - "$MANIFEST_JSON" "$COMMAND_TXT" "$RECORD_ID" "$RESULT_JSON" \
  "$CURRENT_JSON" "$ESTIMATOR_LEDGER_JSON" "$FIXTURE_JSON" \
  "$RESUME_SCAFFOLD_JSON" "$CODE_ARCHIVE" "$RESUME_DEPTH" "$ADAPT_MAX_DEPTH" \
  "$ADAPT_SEGMENT_MAX_NEW_ADMISSIONS" "$ADAPT_SEGMENT_ID" "$PREFLIGHT_ONLY" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


path = Path(sys.argv[1])
command_path = Path(sys.argv[2])
fixture_path = Path(sys.argv[7])
checkpoint_path = Path(sys.argv[8])
archive_path = Path(sys.argv[9])
payload = {
    "schema": "paper_iv_h2o_static_snake_chtc_submit_manifest_v4",
    "record_id": sys.argv[3],
    "run_class": "candidate_diagnostic_continuation_sr_snake_tolerance_guardfix",
    "preflight_only": sys.argv[14].strip().lower() in {"1", "true", "yes", "on"},
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "cwd": os.getcwd(),
    "problem": "molecular_vibronic_h2o_linear_fd",
    "implementation_repair": {
        "source_cluster": 8787626,
        "source_process": 0,
        "source_exit_code": 1,
        "source_failure": "Serialized runtime-split terms cancel below tolerance.",
        "repair": "thread_runtime_split_tolerance_through_serialized_reconstruction_v1",
        "runtime_split_tolerance": 1.0e-12,
        "changed_source_file": "pipelines/scaffold/hh_continuation_generators.py",
        "scientific_settings_changed": False,
    },
    "continuation_source": {
        "cluster": 8790143,
        "process": 0,
        "source_depth": 15,
        "source_energy": -74.99301223223274,
        "source_exact_gs_energy": -75.00416358577124,
        "source_exact_abs_delta_e": 0.011151353538664921,
        "source_stop_reason": "segment_max_new_admissions",
    },
    "model_contract": {
        "basis": "sto-3g",
        "active_space": "cas_8e_6o_valence",
        "num_particles": [4, 4],
        "mode_labels": ["bend", "symmetric_stretch", "antisymmetric_stretch"],
        "n_ph_max_by_mode": [1, 1, 1],
        "n_total_qubits": 15,
    },
    "route": {
        "route_family": "singleton_response_snake",
        "route_profile": "supported_whitened_adaptive_trust_v1",
        "static_route_id": "route_a",
        "meta_feature_profile": "paper_i_production_v1",
        "lane_route": "physical_operator_type",
        "phase0_pilot_enabled": False,
        "phase2_enable_batching": False,
        "phase3_enable_batching": False,
        "phase2_novelty_enabled": False,
        "phase3_novelty_enabled": False,
        "phase3_novelty_ablation_mode": "all",
        "coordinate_solve_policy": "supported_metric_whitened_eigh_v1",
        "coordinate_solve_scope": "phase3_only_v1",
        "trust_region_update_policy": "displacement_calibrated_unbounded_v2",
        "sr_escape_mode": "disabled",
        "prune_policy": "recoverability_ladder_v1",
        "prune_schur_nomination_route": "metric_regularized_v1",
        "prune_metric_schur_mu": 0.01,
        "runtime_split_mode": "shortlist_pauli_children_v1",
        "runtime_split_selection_mode": "archival_child_set_forward_v1",
        "runtime_split_subset_sizes": [1],
        "beam_width": 3,
        "beam_children_per_parent": 2,
    },
    "execution": {
        "starting_depth": int(sys.argv[10]),
        "max_controller_rounds": int(sys.argv[11]),
        "max_new_admissions": int(sys.argv[12]),
        "target_depth": 30,
        "segment_id": sys.argv[13],
        "optimizer": "POWELL",
        "adapt_maxiter": 300,
        "adapt_scipy_maxfev": 20000,
        "gradient_workers": 8,
        "beam_parent_workers": 1,
        "candidate_cache_mode": os.environ.get("STATIC_ADAPT_CANDIDATE_RECORD_CACHE"),
        "resume_boundary_refit_policy": "before_first_new_admission",
    },
    "inputs": {
        "fixture_json": str(fixture_path),
        "fixture_sha256": sha256_file(fixture_path),
        "resume_checkpoint": str(checkpoint_path),
        "resume_checkpoint_sha256": sha256_file(checkpoint_path),
        "code_archive": str(archive_path),
        "code_archive_sha256": sha256_file(archive_path),
        "command_txt": str(command_path),
        "command_sha256": sha256_file(command_path),
    },
    "outputs": {
        "result_json": sys.argv[4],
        "current_json": sys.argv[5],
        "estimator_call_ledger_json": sys.argv[6],
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "JOB START record_id=$RECORD_ID host=$(hostname) pwd=$PWD"
date -u +%Y-%m-%dT%H:%M:%SZ
append_wrapper_event "shell_start"
write_shell_status "starting" 0

HEARTBEAT_PID=""
START_EPOCH="$(date +%s)"
stop_heartbeat() {
  if [[ -n "${HEARTBEAT_PID:-}" ]]; then
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    wait "$HEARTBEAT_PID" 2>/dev/null || true
    HEARTBEAT_PID=""
  fi
}

start_heartbeat() {
  (
    while true; do
      now_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      now_epoch="$(date +%s)"
      elapsed=$((now_epoch - START_EPOCH))
      echo "HEARTBEAT record_id=$RECORD_ID timestamp_utc=$now_utc elapsed_sec=$elapsed"
      emit_heartbeat
      sleep "${H2O_SHELL_HEARTBEAT_SEC:-300}"
    done
  ) &
  HEARTBEAT_PID=$!
}

on_signal() {
  local sig="$1"
  local code=$((128 + sig))
  append_wrapper_event "shell_signal_${sig}"
  stop_heartbeat
  write_shell_status "terminated" "$code"
  exit "$code"
}
trap 'on_signal 15' TERM
trap 'on_signal 2' INT

append_wrapper_event "python_starting"
set +e
start_heartbeat
"${CMD[@]}" > >(tee "$STDOUT_LOG") 2> >(tee "$STDERR_LOG" >&2)
STATUS=$?
stop_heartbeat
set -e
trap - TERM INT
append_wrapper_event "python_exited"

if [[ "$STATUS" -eq 0 ]]; then
  SHELL_STATE="completed"
elif [[ "$STATUS" -ge 128 ]]; then
  SHELL_STATE="terminated"
else
  SHELL_STATE="failed"
fi
write_shell_status "$SHELL_STATE" "$STATUS"

date -u +%Y-%m-%dT%H:%M:%SZ
echo "JOB DONE record_id=$RECORD_ID status=$STATUS"
exit "$STATUS"

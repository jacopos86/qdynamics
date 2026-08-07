#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="paper_i_hh_strong_weak_snake_continue_from_current_best_20260608_v1"
OUT_ROOT="${1:-raw_outputs/${RUN_TAG}/snake}"
MAX_DEPTH="${2:-500}"
MAX_SEGMENTS="${3:-}"
METHOD="snake"
SNAP_PREFIX="${RUN_TAG}"
SOURCE_JSON="chtc/phase3_optuna/input/paper_i_hh_strong_weak_no_target_continue_20260607_v1/sources/snake_strong_weak_trial0011_result.json"
RUNNER="chtc/phase3_optuna/input/paper_i_hh_strong_weak_no_target_continue_20260607_v1/run_strong_weak_no_target_continue.py"
IMAGE="${PROJECT_IMAGE:-chtc/phase3_optuna/image.sif}"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/json" "$OUT_ROOT/logs"

if [[ -f "$SOURCE_JSON" ]]; then
  cp -f "$SOURCE_JSON" "${SNAP_PREFIX}.current.json"
else
  printf '{}\n' > "${SNAP_PREFIX}.current.json"
fi
printf 'utc\tmethod\tevent\tsegment_index\titeration\tdepth\tenergy\traw_abs_delta_e_to_ed5\tsource_json\tstop_reason\n' > "${SNAP_PREFIX}.delta_e_progress.tsv"
printf '' > "${SNAP_PREFIX}.delta_e_progress.jsonl"
cat > "${SNAP_PREFIX}.status.json" <<EOF
{"status":"starting","run_tag":"${RUN_TAG}","out_root":"${OUT_ROOT}","source_json":"${SOURCE_JSON}"}
EOF
cat > "${SNAP_PREFIX}.manifest.json" <<EOF
{
  "schema": "paper_i_hh_strong_weak_snake_continue_recoverable_submit_v1",
  "run_tag": "${RUN_TAG}",
  "table_label": "tab:hh_first_plateau_prefix_costs",
  "regime": "strong_weak",
  "method": "SNAKE",
  "baseline_source_json": "${SOURCE_JSON}",
  "baseline_raw_external_error": 0.0001875954084569753,
  "n_ph_work": 2,
  "n_ph_ref": 5,
  "target_stop_enabled": false,
  "manual_stop_policy": "user_decides_from_delta_e_logs_and_recoverable_current_json",
  "recoverability": "top-level current/status/delta files are mirrored while the segment runner executes",
  "out_root": "${OUT_ROOT}",
  "max_depth": ${MAX_DEPTH},
  "max_segments": "${MAX_SEGMENTS}"
}
EOF

sync_snapshot() {
  if [[ -f "$OUT_ROOT/json/current.json" ]]; then
    cp -f "$OUT_ROOT/json/current.json" "${SNAP_PREFIX}.current.json"
  fi
  if [[ -f "$OUT_ROOT/status.json" ]]; then
    cp -f "$OUT_ROOT/status.json" "${SNAP_PREFIX}.status.json"
  fi
  if [[ -f "$OUT_ROOT/logs/delta_e_progress.tsv" ]]; then
    cp -f "$OUT_ROOT/logs/delta_e_progress.tsv" "${SNAP_PREFIX}.delta_e_progress.tsv"
  fi
  if [[ -f "$OUT_ROOT/logs/delta_e_progress.jsonl" ]]; then
    cp -f "$OUT_ROOT/logs/delta_e_progress.jsonl" "${SNAP_PREFIX}.delta_e_progress.jsonl"
  fi

  latest_segment=""
  if compgen -G "$OUT_ROOT/segments/segment_*" >/dev/null; then
    latest_segment="$(ls -d "$OUT_ROOT"/segments/segment_* | sort | tail -n 1)"
  fi
  tar_args=("${SNAP_PREFIX}.manifest.json" "${SNAP_PREFIX}.current.json" "${SNAP_PREFIX}.status.json" "${SNAP_PREFIX}.delta_e_progress.tsv" "${SNAP_PREFIX}.delta_e_progress.jsonl")
  [[ -d "$OUT_ROOT/json" ]] && tar_args+=("$OUT_ROOT/json")
  [[ -d "$OUT_ROOT/logs" ]] && tar_args+=("$OUT_ROOT/logs")
  [[ -n "$latest_segment" && -d "$latest_segment" ]] && tar_args+=("$latest_segment")
  tar -czf "${SNAP_PREFIX}.latest_snapshot.tgz" "${tar_args[@]}" 2>/dev/null || true
}

sync_loop() {
  while true; do
    sync_snapshot
    sleep 30
  done
}

sync_snapshot

if [[ ! -f "$IMAGE" ]]; then
  echo "Missing Apptainer image: $IMAGE" >&2
  exit 2
fi
if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v singularity)"
else
  echo "Neither apptainer nor singularity is available on this execute node." >&2
  exit 127
fi

sync_loop &
sync_pid=$!
child_pid=""
cleanup() {
  rc="${1:-143}"
  sync_snapshot
  if [[ -n "$child_pid" ]]; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
    sync_snapshot
  fi
  kill "$sync_pid" 2>/dev/null || true
  wait "$sync_pid" 2>/dev/null || true
  exit "$rc"
}
trap 'cleanup 143' TERM INT

ROOT="$PWD"
cmd=(
  "$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE"
  bash -lc 'cd /work && export TABLE_I_STATIC_SUITE_PROFILE=paper_i_three_model_hh_symmetric_20260527_v1 && python -u chtc/phase3_optuna/input/paper_i_hh_strong_weak_no_target_continue_20260607_v1/run_strong_weak_no_target_continue.py "$@"' --
  "$METHOD" --output-root "$OUT_ROOT" --max-depth "$MAX_DEPTH"
)
if [[ -n "$MAX_SEGMENTS" ]]; then
  cmd+=(--max-segments "$MAX_SEGMENTS")
fi

"${cmd[@]}" &
child_pid=$!
set +e
wait "$child_pid"
rc=$?
set -e
child_pid=""
sync_snapshot
kill "$sync_pid" 2>/dev/null || true
wait "$sync_pid" 2>/dev/null || true
exit "$rc"

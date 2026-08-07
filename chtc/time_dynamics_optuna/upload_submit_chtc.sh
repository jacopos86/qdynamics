#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:-upload-only}"
REMOTE_HOST="${CHTC_HOST:-jsstrobel@ap2001.chtc.wisc.edu}"
REMOTE_BASE="${CHTC_REMOTE_BASE:-Holstein_time_dynamics_optuna_chtc}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS="chtc/time_dynamics_optuna"
SSH_CONTROL_PATH="${CHTC_SSH_CONTROL_PATH:-/tmp/chtc-%C}"
SSH_OPTS=(-o ControlMaster=auto -o ControlPersist=10m -o "ControlPath=$SSH_CONTROL_PATH")
RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPersist=10m -o ControlPath=$SSH_CONTROL_PATH"

preflight() {
  local list="$1"
  if [[ "$list" == "all" ]]; then
    python "$LOCAL_ROOT/$HARNESS/preflight_inputs.py" --records "$LOCAL_ROOT/$HARNESS/input/records.tsv" --repo-root "$LOCAL_ROOT" --all --keep-stale-staged
  else
    python "$LOCAL_ROOT/$HARNESS/preflight_inputs.py" --records "$LOCAL_ROOT/$HARNESS/input/records.tsv" --repo-root "$LOCAL_ROOT" --record-list "$LOCAL_ROOT/$HARNESS/input/$list" --keep-stale-staged
  fi
}

preflight_custom() {
  local records="$1"
  local list="$2"
  python "$LOCAL_ROOT/$HARNESS/preflight_inputs.py" \
    --records "$LOCAL_ROOT/$records" \
    --repo-root "$LOCAL_ROOT" \
    --record-list "$LOCAL_ROOT/$list" \
    --no-stage \
    --keep-stale-staged
}

case "$ACTION" in
  upload-only)
    ;;
  image-build)
    ;;
  smoke)
    preflight smoke_records.txt
    ;;
  pilot)
    preflight pilot_records.txt
    ;;
  pilot-dt321-append-prune)
    preflight pilot_dt321_append_prune_records.txt
    ;;
  full)
    preflight full_records.txt
    ;;
  full-dt321-append-prune)
    preflight full_dt321_append_prune_records.txt
    ;;
  retry)
    preflight retry_records.txt
    ;;
  hubbard-hh-prune-recoverability)
    preflight hubbard_hh_prune_recoverability_records.txt
    ;;
  hubbard-hh-energybias)
    preflight hubbard_hh_energybias_records.txt
    ;;
  hh-highdrive-append-forensics)
    preflight hh_highdrive_append_forensics_records.txt
    ;;
  paper-ii-all-algorithm-class-calibration-v1-smoke)
    preflight_custom \
      "$HARNESS/input/paper_ii_all_algorithm_class_calibration_v1_smoke_records.tsv" \
      "$HARNESS/input/paper_ii_all_algorithm_class_calibration_v1_smoke_record_ids.txt"
    ;;
  paper-ii-all-algorithm-class-calibration-v1-full)
    preflight_custom \
      "$HARNESS/input/paper_ii_all_algorithm_class_calibration_v1_records.tsv" \
      "$HARNESS/input/paper_ii_all_algorithm_class_calibration_v1_record_ids.txt"
    ;;
  *)
    echo "usage: $0 {upload-only|image-build|smoke|pilot|pilot-dt321-append-prune|full|full-dt321-append-prune|retry|hubbard-hh-prune-recoverability|hubbard-hh-energybias|hh-highdrive-append-forensics|paper-ii-all-algorithm-class-calibration-v1-smoke|paper-ii-all-algorithm-class-calibration-v1-full}" >&2
    exit 2
    ;;
esac

echo "Uploading CHTC bundle: $LOCAL_ROOT -> $REMOTE_HOST:$REMOTE_BASE"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_BASE/logs' '$REMOTE_BASE/raw_outputs'"
case "$ACTION" in
  image-build)
    echo "Removing remote image before rebuild."
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "rm -f '$REMOTE_BASE/chtc/time_dynamics_optuna/image.sif'"
    ;;
  smoke|pilot|pilot-dt321-append-prune|full|full-dt321-append-prune|retry|hubbard-hh-prune-recoverability|hubbard-hh-energybias|hh-highdrive-append-forensics|paper-ii-all-algorithm-class-calibration-v1-smoke|paper-ii-all-algorithm-class-calibration-v1-full)
    if [[ "${CHTC_CLEAR_REMOTE_SEED_CACHE:-0}" == "1" ]]; then
    echo "Removing remote staged seed cache for selected queue."
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "rm -rf '$REMOTE_BASE/chtc/time_dynamics_optuna/input/seed_artifacts'"
    else
      echo "Keeping remote staged seed cache; set CHTC_CLEAR_REMOTE_SEED_CACHE=1 to prune it."
    fi
    ;;
esac
rsync -az -e "$RSYNC_SSH" \
  --exclude '.git' \
  --exclude '.pytest_cache' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.venv' \
  --exclude 'src/quantum/chemistry/conda-env' \
  --exclude 'artifacts' \
  --exclude 'output' \
  --exclude 'tmp' \
  --exclude 'plots' \
  --exclude 'prompt-exports' \
  --exclude 'raw_outputs' \
  --exclude 'logs' \
  --exclude 'chtc/time_dynamics_optuna/image.sif' \
  "$LOCAL_ROOT/" "$REMOTE_HOST:$REMOTE_BASE/"
case "$ACTION" in
  upload-only)
    echo "Upload complete."
    ;;
  image-build)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && condor_submit chtc/time_dynamics_optuna/submit_image_build.sub"
    ;;
  smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/time_dynamics_optuna/image.sif && condor_submit chtc/time_dynamics_optuna/submit_smoke.sub"
    ;;
  pilot)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/time_dynamics_optuna/image.sif && condor_submit chtc/time_dynamics_optuna/submit_pilot.sub"
    ;;
  pilot-dt321-append-prune)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/time_dynamics_optuna/image.sif && condor_submit chtc/time_dynamics_optuna/submit_pilot_dt321_append_prune.sub"
    ;;
  full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/time_dynamics_optuna/image.sif && condor_submit chtc/time_dynamics_optuna/submit_full.sub"
    ;;
  full-dt321-append-prune)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/time_dynamics_optuna/image.sif && condor_submit chtc/time_dynamics_optuna/submit_full_dt321_append_prune.sub"
    ;;
  retry)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/time_dynamics_optuna/image.sif && condor_submit chtc/time_dynamics_optuna/submit_retry.sub"
    ;;
  hubbard-hh-prune-recoverability)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/time_dynamics_optuna/image.sif && condor_submit chtc/time_dynamics_optuna/submit_hubbard_hh_prune_recoverability.sub"
    ;;
  hubbard-hh-energybias)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/time_dynamics_optuna/image.sif && condor_submit chtc/time_dynamics_optuna/submit_hubbard_hh_energybias.sub"
    ;;
  hh-highdrive-append-forensics)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/time_dynamics_optuna/image.sif && condor_submit chtc/time_dynamics_optuna/submit_hh_highdrive_append_forensics.sub"
    ;;
  paper-ii-all-algorithm-class-calibration-v1-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/time_dynamics_optuna/image.sif && condor_submit chtc/time_dynamics_optuna/submit_paper_ii_all_algorithm_class_calibration_v1_smoke.sub"
    ;;
  paper-ii-all-algorithm-class-calibration-v1-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/time_dynamics_optuna/image.sif && condor_submit chtc/time_dynamics_optuna/submit_paper_ii_all_algorithm_class_calibration_v1.sub"
    ;;
esac

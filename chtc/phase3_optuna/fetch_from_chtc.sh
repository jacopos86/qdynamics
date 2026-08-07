#!/usr/bin/env bash
set -euo pipefail
REMOTE_HOST="${CHTC_HOST:-jsstrobel@ap2001.chtc.wisc.edu}"
REMOTE_BASE="${CHTC_REMOTE_BASE:-Holstein_phase3_optuna_chtc}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_CONTROL_PATH="${CHTC_SSH_CONTROL_PATH:-/tmp/chtc-%C}"
RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPersist=10m -o ControlPath=$SSH_CONTROL_PATH"
mkdir -p "$LOCAL_ROOT/raw_outputs/chtc_phase3_optuna" "$LOCAL_ROOT/logs/chtc_phase3_optuna"
rsync -az -e "$RSYNC_SSH" "$REMOTE_HOST:$REMOTE_BASE/raw_outputs/" "$LOCAL_ROOT/raw_outputs/chtc_phase3_optuna/" || true
rsync -az -e "$RSYNC_SSH" "$REMOTE_HOST:$REMOTE_BASE/logs/" "$LOCAL_ROOT/logs/chtc_phase3_optuna/" || true
echo "Fetched outputs. Run: bash chtc/phase3_optuna/check_outputs.sh"

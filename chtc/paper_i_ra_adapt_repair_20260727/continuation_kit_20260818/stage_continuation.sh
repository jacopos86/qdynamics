#!/bin/bash
# Stage and submit a continuation of a killed campaign cell on CHTC.
#
# Usage: stage_continuation.sh <campaign_package_dir_on_chtc> <execution_id> \
#            <local_snapshot_dir> <request_memory_mb>
#
# - campaign_package_dir_on_chtc: the sealed package dir the cell ran from
#   (its job/, protocol/, authorization files are reused verbatim; the
#   continuation carries the identical authority).
# - local_snapshot_dir: contains current.json.gz + estimator ledger .gz from
#   pull_checkpoint.sh for that cell.
#
# The submit wraps the package's execute_job.sh environment but launches
# continuation_worker.py with the staged checkpoint. Submit ONLY after the
# original job is confirmed dead (wave rule: never duplicate a running cell).
set -euo pipefail
SOCK="$HOME/.ssh/cm-chtc-9661863.sock"
PKG="$1"; EXEC_ID="$2"; SNAP="$3"; MEM="${4:-40960}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="continuation_${EXEC_ID}_$(date +%Y%m%dT%H%M%SZ)"

# Guard: refuse if a job for this execution id is still in the queue.
if ssh -S "$SOCK" chtc "condor_q -af Args 2>/dev/null | grep -q '$EXEC_ID'"; then
  echo "REFUSED: a queued/running job still carries $EXEC_ID"; exit 2
fi

ssh -S "$SOCK" chtc "mkdir -p ~/$STAGE/resume"
scp -o ControlPath="$SOCK" "$SNAP/current.json.gz" "$SNAP/estimator_ledger_checkpoint.json.gz" chtc:"~/$STAGE/resume/"
scp -o ControlPath="$SOCK" "$HERE/continuation_worker.py" chtc:"~/$STAGE/"
ssh -S "$SOCK" chtc "cd ~/$STAGE/resume && gunzip -f current.json.gz && \
  gunzip -f estimator_ledger_checkpoint.json.gz && \
  mv estimator_ledger_checkpoint.json \
     current.estimator_call_ledger_checkpoint.staged.json"

cat > /tmp/continuation_submit_note.txt <<EOF
Staged: ~/$STAGE on CHTC.
Next (manual, deliberate):
 1. Copy the cell's submit stanza from $PKG/submit_home.sub, single proc for
    $EXEC_ID, RequestMemory=$MEM.
 2. Replace the worker invocation with:
      python3 ~/$STAGE/continuation_worker.py \
        --package-worker <pkg>/worker.py \
        --resume-checkpoint ~/$STAGE/resume/current.json \
        --job ... --protocol ... --authorization ... --output ...
 3. condor_submit and watch the CONTINUATION line + loader authentication.
EOF
cat /tmp/continuation_submit_note.txt
echo "STAGED $STAGE"

#!/usr/bin/env bash
set -euo pipefail

BATCH_ID="paper_i_hubbard_weak_u025_snake_feature_ablation_depth10_20260620_v1"
ANCHOR_BATCH="holstein-paper-i-hubbard-weak-u025-snake-feature-ablation-depth10-20260620-v1-anchor"
NORMAL_BATCH="holstein-paper-i-hubbard-weak-u025-snake-feature-ablation-depth10-20260620-v1-normal"
NO_SHORTLIST_BATCH="holstein-paper-i-hubbard-weak-u025-snake-feature-ablation-depth10-20260620-v1-no-shortlisting"
ANCHOR_SUB="chtc/phase3_optuna/submit_${BATCH_ID}_anchor.sub"
NORMAL_SUB="chtc/phase3_optuna/submit_${BATCH_ID}_normal_after_anchor.sub"
NO_SHORTLIST_SUB="chtc/phase3_optuna/submit_${BATCH_ID}_no_shortlisting_after_anchor.sub"
ANCHOR_RECORD="${BATCH_ID}__hubbard_weak_u025__snake__native200__full_snake_anchor"
ANCHOR_RESULT="raw_outputs/${BATCH_ID}/${ANCHOR_RECORD}/json/result.json"
ANCHOR_AUDIT="raw_outputs/${BATCH_ID}/${ANCHOR_RECORD}/source_lock_command_audit.json"

queued_or_running() {
  local batch="$1"
  condor_q "$USER" -nobatch -af ClusterId ProcId JobBatchName JobStatus HoldReason | grep -F "$batch" || true
}

wait_batch_empty() {
  local batch="$1"
  while queued_or_running "$batch" >/tmp/"${batch}".q; do
    if [[ ! -s /tmp/"${batch}".q ]]; then
      break
    fi
    date -Iseconds
    cat /tmp/"${batch}".q
    sleep 120
  done
}

submit_if_absent() {
  local batch="$1"
  local submit_file="$2"
  if queued_or_running "$batch" >/tmp/"${batch}".q && [[ -s /tmp/"${batch}".q ]]; then
    echo "Already queued/running: ${batch}"
    cat /tmp/"${batch}".q
    return 0
  fi
  echo "Submitting ${submit_file}"
  condor_submit "$submit_file"
}

validate_anchor() {
  python - <<'PY'
import json
from pathlib import Path

batch_id = "paper_i_hubbard_weak_u025_snake_feature_ablation_depth10_20260620_v1"
record = f"{batch_id}__hubbard_weak_u025__snake__native200__full_snake_anchor"
root = Path("raw_outputs") / batch_id / record
audit_path = root / "source_lock_command_audit.json"
result_path = root / "json" / "result.json"
if not audit_path.exists():
    raise SystemExit(f"missing anchor audit: {audit_path}")
audit = json.loads(audit_path.read_text())
if audit.get("status") != "pass":
    raise SystemExit(f"anchor command audit failed: {audit_path}")
if not result_path.exists():
    raise SystemExit(f"missing anchor result: {result_path}")
doc = json.loads(result_path.read_text())
settings = doc.get("settings") or {}
adapt = doc.get("adapt_vqe") or {}
depth = int(adapt.get("ansatz_depth") or len(adapt.get("operators") or []))
history_count = int(adapt.get("history_count") or len(adapt.get("history") or []))
if str(settings.get("u")) not in {"0.25", "0.2500000000000000"} and float(settings.get("u", 0.0)) != 0.25:
    raise SystemExit(f"anchor used wrong U: {settings.get('u')}")
if depth < 10 and history_count < 10:
    raise SystemExit(f"anchor did not reach depth/history 10: depth={depth}, history_count={history_count}")
print(json.dumps({"anchor_status": "pass", "depth": depth, "history_count": history_count, "abs_delta_e": adapt.get("abs_delta_e")}, indent=2))
PY
}

mkdir -p "logs/${BATCH_ID}" "raw_outputs/${BATCH_ID}"
submit_if_absent "$ANCHOR_BATCH" "$ANCHOR_SUB"
wait_batch_empty "$ANCHOR_BATCH"
condor_history "$USER" -limit 200 -af ClusterId ProcId JobBatchName JobStatus ExitCode HoldReason | grep -F "$ANCHOR_BATCH" || true
validate_anchor
submit_if_absent "$NORMAL_BATCH" "$NORMAL_SUB"
submit_if_absent "$NO_SHORTLIST_BATCH" "$NO_SHORTLIST_SUB"
echo "Submitted fan-out batches:"
queued_or_running "$NORMAL_BATCH" || true
queued_or_running "$NO_SHORTLIST_BATCH" || true

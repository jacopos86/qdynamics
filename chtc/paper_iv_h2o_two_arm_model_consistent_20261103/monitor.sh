#!/usr/bin/env bash
# paper_iv_h2o_two_arm_model_consistent_20261103/monitor.sh -- hourly CHTC watch for the two
# Paper-IV arms. READ-ONLY against the pool: never submits, releases, removes,
# or deletes anything remote. Implements
# agent_guidance/shared/paper-iv-h2o-chtc-handoff-20260903.md section 5.2.
#
#   usage: monitor.sh <cluster>[,<cluster>...]
#   env:   PAPER_IV_LOCAL         local mirror root   (default ~/paper_iv_h2o_results)
#          PAPER_IV_REMOTE_ROOT   remote campaign dir (default chtc/paper_iv_h2o_two_arm_model_consistent_20261103, relative to $HOME on ap2001)
#          PAPER_IV_RESULTS_MODE  home|staging        (default home; staging pulls /staging/jsstrobel/<campaign>/ too)
#
# Authentication: the `ap2001` ControlMaster alias only. The user runs
# `ssh -MNf ap2001` once per session and answers Duo. This script never
# prompts, never takes a password, and exits 78 if the master is down.
#
# The monitor is a tripwire, not a safety net: once a hold fires nothing
# stages out. Its jobs are to catch a segment drifting toward its budget while
# a human can still act, confirm each segment landed, and keep a local copy.
set -uo pipefail

CLUSTERS="${1:?usage: monitor.sh <cluster>[,<cluster>...]}"
CAMPAIGN="paper_iv_h2o_two_arm_model_consistent_20261103"
LOCAL_DIR="${PAPER_IV_LOCAL:-$HOME/paper_iv_h2o_results}"
REMOTE_ROOT="${PAPER_IV_REMOTE_ROOT:-chtc/$CAMPAIGN}"
RESULTS_MODE="${PAPER_IV_RESULTS_MODE:-home}"
STAGING_ROOT="/staging/jsstrobel/$CAMPAIGN"
STATE="$LOCAL_DIR/.monitor_state"
WALL_BUDGET=172800          # 48h: the submit-file periodic_hold
WALL_WARN=144000            # warn at 40h
QUOTA_SOFT_MB=40960
QUOTA_WARN_MB=38000
RETRIEVE_EVERY=43200        # 12h
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20)
mkdir -p "$LOCAL_DIR"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOCAL_DIR/monitor.log"; }
remote() { ssh "${SSH_OPTS[@]}" ap2001 "$@"; }

# --- liveness: never fall back to a password/Duo prompt ---------------------
if ! ssh -O check ap2001 >/dev/null 2>&1; then
  log "FATAL ssh ControlMaster for ap2001 is down. Run once, interactively: ssh -MNf ap2001   (then re-run this monitor)"
  exit 78
fi

# --- per-cluster state ------------------------------------------------------
alerts=0
for c in ${CLUSTERS//,/ }; do
  line="$(remote "condor_q $c -af JobStatus RemoteWallClockTime MemoryUsage DiskUsage JobBatchName 2>/dev/null" | head -1)"
  read -r status runtime mem disk batch <<<"$line"
  if [[ -z "${status:-}" ]]; then
    hist="$(remote "condor_history $c -limit 1 -af JobStatus ExitCode RemoteWallClockTime LastHoldReason JobBatchName 2>/dev/null")"
    log "cluster $c LEFT QUEUE: ${hist:-<no history row>}"
    case "$hist" in
      4\ 0\ *) log "cluster $c completed with exit 0; verify results/ landed (see retrieval below)." ;;
      4\ *)    log "ALERT cluster $c completed with a NONZERO exit code; results/ still transfers on ON_EXIT -- retrieve and read segment_receipt.json."; alerts=$((alerts+1)) ;;
      3\ *)    log "ALERT cluster $c was REMOVED."; alerts=$((alerts+1)) ;;
    esac
    continue
  fi
  case "$status" in
    1) log "cluster $c IDLE batch=${batch:-?}" ;;
    2) rt=${runtime%.*}
       log "cluster $c RUNNING ${rt}s ($((rt/3600))h) mem=${mem:-?}MB disk=${disk:-?}KB batch=${batch:-?}"
       if (( rt > WALL_WARN )); then
         log "WARN cluster $c at ${rt}s -- past the ${WALL_WARN}s (40h) warning line toward the ${WALL_BUDGET}s (48h) periodic_hold."
         log "WARN the in-process --max-wall-seconds stop should already have fired; a hold transfers NOTHING. Escalate to the user now."
         alerts=$((alerts+1))
       fi ;;
    5) reason="$(remote "condor_q $c -af HoldReason 2>/dev/null")"
       log "ALERT cluster $c HELD: ${reason:-<no reason>}"
       log "ALERT do NOT condor_rm (destroys any spooled sandbox) and do NOT condor_release (restarts from round zero). Tell the user."
       alerts=$((alerts+1)) ;;
    *) log "cluster $c status=$status" ;;
  esac
done

# --- quota guard: a full /home loses a transfer the way a hold does ---------
used_mb="$(remote "quota -s 2>/dev/null" | awk '
  /md9/ {
    v=$2; u=substr(v,length(v),1); n=v; gsub(/[^0-9.]/,"",n);
    if (u=="G") n=n*1024; else if (u=="K") n=n/1024; else if (u=="T") n=n*1024*1024;
    printf "%d\n", n; exit
  }')"
if [[ -n "${used_mb:-}" ]]; then
  if (( used_mb > QUOTA_WARN_MB )); then
    log "ALERT /home at ${used_mb}MB of ${QUOTA_SOFT_MB}MB soft quota -- retrieve+verify, then ask the user to delete retrieved segments (or obtain /staging)."
    alerts=$((alerts+1))
  else
    log "quota /home ${used_mb}MB of ${QUOTA_SOFT_MB}MB"
  fi
else
  log "WARN could not read quota -s on ap2001"
fi

# --- 12-hourly retrieval (rsync mirror, never --delete) ---------------------
last="$(cat "$STATE" 2>/dev/null || echo 0)"
now="$(date +%s)"
if (( now - last >= RETRIEVE_EVERY )); then
  mirror="$LOCAL_DIR/mirror"
  mkdir -p "$mirror"
  log "retrieving results -> $mirror"
  rsync -av --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
    --include='*/' --include='logs/**' --include='results/**' --include='segment.json' --include='submit.sub' \
    --exclude='package/**' --exclude='*' \
    "ap2001:$REMOTE_ROOT/" "$mirror/home/" 2>&1 | tail -n 5 | tee -a "$LOCAL_DIR/monitor.log"
  if [[ "$RESULTS_MODE" == "staging" ]]; then
    rsync -av --partial "ap2001:$STAGING_ROOT/" "$mirror/staging/" 2>&1 | tail -n 5 | tee -a "$LOCAL_DIR/monitor.log"
  fi
  echo "$now" > "$STATE"
  # Retrieval is only trustworthy once verified (handoff 5.3).
  verifier="$(dirname "$0")/next_segment.py"
  py="${PAPER_IV_PYTHON:-/Users/lordskyler/Holstein_test/.venv/bin/python}"
  if [[ -x "$py" && -f "$verifier" ]]; then
    while IFS= read -r cur; do
      d="$(dirname "$cur")"
      # next_segment.py --verify-only: 0 = checkpoint verified and the stop
      # was wall_clock_budget_v1 (resumable); 3 = checkpoint verified but the
      # arm is finished (exact target / 50-round horizon) or the stop is not
      # a wall-clock stop; anything else = the retrieved results/ failed
      # verification and must not be trusted or deleted remotely.
      out="$("$py" "$verifier" --previous-results "$d" --verify-only 2>&1)"; vrc=$?
      case "$vrc" in
        0) log "verified RESUMABLE $d: $(printf '%s' "$out" | tr -d '\n' | cut -c1-200)" ;;
        3) log "verified NOT-RESUMABLE $d: $(printf '%s' "$out" | tail -n 1)" ;;
        *) log "ALERT verify FAILED (rc=$vrc) $d: $(printf '%s' "$out" | tail -n 1)"; alerts=$((alerts+1)) ;;
      esac
    done < <(find "$mirror" -name current.json -path '*/results/*' 2>/dev/null)
  fi
fi

log "monitor pass done alerts=$alerts"
exit 0

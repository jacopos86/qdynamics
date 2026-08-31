#!/bin/bash
# B5-scoped tender: cluster 9674089 only (Paper I single-evidence bundle).
# Fetches finished b3mr archives from staging, sha256-verifies, clears remote.
# Leaves exit-holds alone (receipt-path crash is the known finish mode; the
# archive on staging is the deliverable). Logs a round count per running lane.
set -u
SOCK="$HOME/.ssh/cm-chtc-agent.sock"
SSH="ssh -n -o ControlPath=$SOCK -o BatchMode=yes jsstrobel@ap2001.chtc.wisc.edu"
LOG="/private/tmp/claude-501/-Users-jakestrobel-Documents-Holstein-implementation-Holstein-test-fullclone-3/5a13905e-934f-499a-9643-f0796d15250f/scratchpad/fleet_tender_b5.log"
STAGE="paper_i_hh_b3_measured_residual_20260821"
DEST="/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_hh_b3_measured_residual_20260821/transfer"
CLUSTER=9674089

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$LOG"; }
log "b5 tender starting (pid $$)"
mkdir -p "$DEST"

for cycle in $(seq 1 24); do
  log "=== cycle $cycle ==="
  if ! $SSH true 2>/dev/null; then
    log "SOCKET-DEAD — sleeping, will retry"
    sleep 1800; continue
  fi

  q=$($SSH "condor_q $CLUSTER -af ProcId JobStatus 2>/dev/null")
  running=$(echo "$q" | awk '$2==2{print $1}' | tr '\n' ' ')
  held=$(echo "$q" | awk '$2==5{print $1}' | wc -l | tr -d ' ')
  nrun=$(echo "$running" | wc -w | tr -d ' ')
  log "STATUS running=$nrun held=$held procs_running=[$running]"

  if [ "$nrun" -gt 0 ]; then
    for p in $running; do
      n=$($SSH "condor_tail -maxbytes 300000 $CLUSTER.$p 2>/dev/null" | grep -c hardcoded_adapt_iter)
      log "ROUNDS $CLUSTER.$p at $n/75"
    done
  fi

  avail_kb=$(df -k / | awk 'NR==2{print $4}')
  if [ "${avail_kb:-0}" -lt 4194304 ]; then
    log "DISK-GUARD local avail ${avail_kb}KB < 4GB — skipping fetches"
    sleep 1800; continue
  fi

  files=$($SSH "ls /staging/jsstrobel/$STAGE/transfer/ 2>/dev/null")
  for f in $files; do
    case "$f" in *.tar.gz) ;; *) continue ;; esac
    rsize=$($SSH "stat -c %s /staging/jsstrobel/$STAGE/transfer/$f 2>/dev/null")
    [ -z "$rsize" ] && continue
    rsync --partial -e "ssh -o ControlPath=$SOCK -o BatchMode=yes" \
      "jsstrobel@ap2001.chtc.wisc.edu:/staging/jsstrobel/$STAGE/transfer/$f" \
      "$DEST/$f.part" >> "$LOG" 2>&1
    lsize=$(stat -f %z "$DEST/$f.part" 2>/dev/null || echo 0)
    if [ "$lsize" != "$rsize" ] || [ "$lsize" = "0" ]; then
      log "FETCH-INCOMPLETE $f remote=$rsize local=$lsize — will resume next cycle"
      continue
    fi
    rsha=$($SSH "sha256sum /staging/jsstrobel/$STAGE/transfer/$f" | awk '{print $1}')
    lsha=$(shasum -a 256 "$DEST/$f.part" | awk '{print $1}')
    if [ -n "$rsha" ] && [ "$rsha" = "$lsha" ] && gzip -t "$DEST/$f.part" 2>/dev/null; then
      mv "$DEST/$f.part" "$DEST/$f"
      log "FETCHED $f ($lsize bytes, sha256+gzip verified) — clearing remote"
      $SSH "rm /staging/jsstrobel/$STAGE/transfer/$f" >> "$LOG" 2>&1
    else
      log "VERIFY-MISMATCH $f rsha=$rsha lsha=$lsha — remote kept, .part kept"
    fi
  done

  if [ "$nrun" -eq 0 ]; then
    log "NO RUNNING LANES — final fetch pass done, tender exiting"
    break
  fi
  sleep 1800
done
log "b5 tender complete"

#!/bin/bash
# Overnight fleet tender: clusters 9664316 (costexp+depth v2), 9664281
# (AAVQE continuations), 9664259 (v4 k75 resumes). Every 30 min:
# snapshot, ladder OOM holds (34/26) 40960->65536->81920, fetch and
# clear completed transfer tarballs after byte-size verification.
set -u
# Pick whichever CHTC master socket is alive (the pinned one died 2026-08-20
# overnight and silently took the tender with it). Re-checked every cycle.
pick_sock() {
  for c in "$HOME/.ssh/cm-chtc-agent.sock" /tmp/chtc-* "$HOME"/.ssh/cm-chtc-*.sock; do
    [ -S "$c" ] || continue
    if ssh -n -o ControlPath="$c" -o BatchMode=yes -o ConnectTimeout=10 \
         jsstrobel@ap2001.chtc.wisc.edu true 2>/dev/null; then echo "$c"; return 0; fi
  done
  return 1
}
SOCK="$(pick_sock)" || { echo "NO LIVE CHTC SOCKET - tender cannot start" >&2; exit 1; }
SSH="ssh -S $SOCK -o BatchMode=yes jsstrobel@ap2001.chtc.wisc.edu"
LOG="/private/tmp/claude-501/-Users-jakestrobel-Documents-Holstein-implementation-Holstein-test-fullclone-3/91434114-4541-47ef-95e4-2b8bd90bd929/scratchpad/fleet_tender_restart.log"
LOCAL_ROOT="/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs"

# stage_dir|local_subdir pairs (bash 3.2: no associative arrays)
STAGE_PAIRS="ra_adapt_costexp_depthfix_sixregime_20260820_v2|ra_adapt_costexp_depthfix_sixregime_20260820_v2
ra_adapt_warmstart_sixregime_20260820_v1|ra_adapt_warmstart_sixregime_20260820_v1
paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2|paper_i_aavqe_r50_to_r75_tolmatch_20260820
paper_i_aavqe_r50_to_r75_continuation_20260819_v1|paper_i_aavqe_r50_to_r75_continuation_20260819
paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819_v4|paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819_v4_results
paper_i_hh_b3_measured_residual_20260821|paper_i_hh_b3_measured_residual_20260821"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$LOG"; }
log "tender starting (pid $$)"

for cycle in $(seq 1 24); do
  log "=== cycle $cycle ==="
  if ! ssh -n -o ControlPath="$SOCK" -o BatchMode=yes -o ConnectTimeout=10 jsstrobel@ap2001.chtc.wisc.edu true 2>/dev/null; then
    NEW="$(pick_sock)" && { SOCK="$NEW"; SSH="ssh -n -S $SOCK -o BatchMode=yes jsstrobel@ap2001.chtc.wisc.edu"; log "SOCKET-SWITCH -> $SOCK"; } || { log "SOCKET-DEAD: no live master; sleeping"; sleep 1800; continue; }
  fi
  $SSH 'condor_q 9664325 9665004 9665210 9664259 -af:jh ClusterId ProcId JobStatus RequestMemory MemoryUsage 2>/dev/null' >> "$LOG" 2>&1

  # OOM ladder on our clusters only
  held=$($SSH 'condor_q -hold 9664325 9665004 9665210 9664259 9674089 -af ClusterId ProcId HoldReasonCode RequestMemory 2>/dev/null' 2>/dev/null)
  while read -r cl proc code mem; do
    [ -z "${cl:-}" ] && continue
    if [ "$code" = "34" ] || [ "$code" = "26" ]; then
      next=0
      [ "$mem" -le 40960 ] && next=65536
      [ "$mem" -gt 40960 ] && [ "$mem" -le 65536 ] && next=81920
      [ "$mem" -gt 65536 ] && [ "$mem" -le 81920 ] && next=122880
      if [ "$next" -gt 0 ]; then
        log "LADDER $cl.$proc $mem -> $next"
        $SSH "condor_qedit $cl.$proc RequestMemory $next && condor_release $cl.$proc" >> "$LOG" 2>&1
      else
        # a hold whose cgroup limit is below the already-raised request
        # restarts fine at that request (the 9665210.17 misread, 2026-08-22)
        limit=$($SSH "condor_q $cl.$proc -af HoldReason" 2>/dev/null | grep -o "limit of [0-9]*" | grep -o "[0-9]*")
        if [ -n "${limit:-}" ] && [ "$limit" -lt "$mem" ]; then
          log "RELEASE-RAISED $cl.$proc (killed at ${limit}, request already $mem)"
          $SSH "condor_release $cl.$proc" >> "$LOG" 2>&1
        else
          log "CEILING $cl.$proc held at $mem at true ceiling - leaving for eyes"
        fi
      fi
    elif [ "$code" = "12" ] && { [ "$cl" = "9664325" ] || [ "$cl" = "9665004" ] || [ "$cl" = "9665210" ] || [ "$cl" = "9664259" ]; }; then
      log "RELEASE-QUOTAHOLD $cl.$proc (staging drained)"
      $SSH "condor_release $cl.$proc" >> "$LOG" 2>&1
    elif [ "$code" = "3" ]; then
      # exit-holds are REAL failures now (e.g. forced-admission drift guard
      # on plateau-costexp weak cells) - never auto-release; leave for eyes
      log "EXITHOLD-KEPT $cl.$proc - needs diagnosis, not release"
    else
      log "HOLD-OTHER $cl.$proc code=$code — not touching"
    fi
  done <<< "$held"

  # fetch + clear completed transfers (skip when local disk is tight;
  # staging holds results safely until morning)
  avail_kb=$(df -k / | awk 'NR==2{print $4}')
  if [ "${avail_kb:-0}" -lt 2097152 ]; then
    log "DISK-GUARD local avail ${avail_kb}KB < 2GB — skipping fetches this cycle"
    sleep 1800
    continue
  fi
  echo "$STAGE_PAIRS" | while IFS="|" read -r stage local_sub; do
    [ -z "$stage" ] && continue
    dest="$LOCAL_ROOT/$local_sub/transfer"
    mkdir -p "$dest"
    files=$($SSH "ls /staging/jsstrobel/$stage/transfer/ 2>/dev/null" 2>/dev/null)
    for f in $files; do
      case "$f" in *.tar.gz) ;; *) continue ;; esac
      rsize=$($SSH "stat -c %s /staging/jsstrobel/$stage/transfer/$f 2>/dev/null" 2>/dev/null)
      [ -z "$rsize" ] && continue
      rsync -az -e "ssh -S $SOCK -o BatchMode=yes" \
        "jsstrobel@ap2001.chtc.wisc.edu:/staging/jsstrobel/$stage/transfer/$f" \
        "$dest/" >> "$LOG" 2>&1
      lsize=$(stat -f %z "$dest/$f" 2>/dev/null || echo 0)
      if [ "$lsize" = "$rsize" ] && [ "$lsize" != "0" ]; then
        log "FETCHED $stage/$f ($lsize bytes, verified) — clearing remote"
        $SSH "rm /staging/jsstrobel/$stage/transfer/$f" >> "$LOG" 2>&1
      else
        log "FETCH-MISMATCH $stage/$f remote=$rsize local=$lsize — remote kept"
      fi
    done
  done
  sleep 1800
done
log "tender complete (24 cycles)"

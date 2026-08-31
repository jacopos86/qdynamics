#!/bin/bash
# Overnight fleet tender: clusters 9664316 (costexp+depth v2), 9664281
# (AAVQE continuations), 9664259 (v4 k75 resumes). Every 30 min:
# snapshot, ladder OOM holds (34/26) 40960->65536->81920, fetch and
# clear completed transfer tarballs after byte-size verification.
set -u
SOCK="$HOME/.ssh/cm-chtc-agent.sock"
SSH="ssh -S $SOCK -o BatchMode=yes jsstrobel@ap2001.chtc.wisc.edu"
LOG="/private/tmp/claude-501/-Users-jakestrobel-Documents-Holstein-implementation-Holstein-test-fullclone-3/4759e8af-c607-49b1-b24e-7240b4844aff/scratchpad/fleet_tender.log"
LOCAL_ROOT="/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs"

# stage_dir|local_subdir pairs (bash 3.2: no associative arrays)
STAGE_PAIRS="ra_adapt_costexp_depthfix_sixregime_20260820_v2|ra_adapt_costexp_depthfix_sixregime_20260820_v2
ra_adapt_warmstart_sixregime_20260820_v1|ra_adapt_warmstart_sixregime_20260820_v1
paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2|paper_i_aavqe_r50_to_r75_tolmatch_20260820
paper_i_aavqe_r50_to_r75_continuation_20260819_v1|paper_i_aavqe_r50_to_r75_continuation_20260819
paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819_v4|paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819_v4_results"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$LOG"; }
log "tender starting (pid $$)"

for cycle in $(seq 1 24); do
  log "=== cycle $cycle ==="
  $SSH 'condor_q 9664325 9665004 9665210 9664259 -af:jh ClusterId ProcId JobStatus RequestMemory MemoryUsage 2>/dev/null' >> "$LOG" 2>&1

  # OOM ladder on our clusters only
  held=$($SSH 'condor_q -hold 9664325 9665004 9665210 9664259 -af ClusterId ProcId HoldReasonCode RequestMemory 2>/dev/null' 2>/dev/null)
  while read -r cl proc code mem; do
    [ -z "${cl:-}" ] && continue
    if [ "$code" = "34" ] || [ "$code" = "26" ]; then
      next=0
      [ "$mem" -le 40960 ] && next=65536
      [ "$mem" -gt 40960 ] && [ "$mem" -le 65536 ] && next=81920
      if [ "$next" -gt 0 ]; then
        log "LADDER $cl.$proc $mem -> $next"
        $SSH "condor_qedit $cl.$proc RequestMemory $next && condor_release $cl.$proc" >> "$LOG" 2>&1
      else
        log "CEILING $cl.$proc held at $mem — leaving for morning"
      fi
    elif [ "$code" = "12" ] && { [ "$cl" = "9664325" ] || [ "$cl" = "9665004" ] || [ "$cl" = "9665210" ] || [ "$cl" = "9664259" ]; }; then
      log "RELEASE-QUOTAHOLD $cl.$proc (staging drained)"
      $SSH "condor_release $cl.$proc" >> "$LOG" 2>&1
    elif [ "$code" = "3" ] && [ "$cl" = "9664325" ]; then
      # known worker defect fixed on the AP 2026-08-20; reruns pick up
      # the corrected file at job start
      log "RELEASE-EXITHOLD $cl.$proc (fixed worker)"
      $SSH "condor_release $cl.$proc" >> "$LOG" 2>&1
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

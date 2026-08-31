#!/bin/bash
SSH="ssh -n -S $HOME/.ssh/cm-chtc-agent.sock -o BatchMode=yes jsstrobel@ap2001.chtc.wisc.edu"
TAILS=/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data/live_tails
jobs=$($SSH 'condor_q jsstrobel -constraint "JobStatus==2" -af ClusterId ProcId' 2>/dev/null)
while read -r cl proc; do
  [ -z "$cl" ] && continue
  case "$cl" in 9664325|9665210|9665004|9663698) continue ;; esac
  raw=$($SSH "condor_q $cl.$proc -af Args Arguments 2>/dev/null" | tr ' ' '\n' | grep "__nph" | tail -1)
  eid=$(basename "$raw" .json)
  [ -z "$eid" ] && { echo "NO-EID $cl.$proc" >&2; continue; }
  $SSH "timeout 60 condor_ssh_to_job -auto-retry $cl.$proc 'grep hardcoded_adapt_iter _condor_stdout 2>/dev/null'" 2>/dev/null > "$TAILS/$eid.tail.new"
  n=$(grep -c hardcoded_adapt_iter "$TAILS/$eid.tail.new" 2>/dev/null || echo 0)
  if [ "$n" -gt 0 ]; then
    mv "$TAILS/$eid.tail.new" "$TAILS/$eid.tail"
    echo "TAIL $cl.$proc $eid rounds=$n"
  else
    rm -f "$TAILS/$eid.tail.new"
    echo "EMPTY $cl.$proc $eid"
  fi
done <<< "$jobs"
echo TAILS_DONE

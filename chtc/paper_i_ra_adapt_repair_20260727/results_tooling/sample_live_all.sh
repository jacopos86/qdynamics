#!/bin/bash
SSH="ssh -n -S $HOME/.ssh/cm-chtc-agent.sock -o BatchMode=yes jsstrobel@ap2001.chtc.wisc.edu"
OUT=live_samples_fresh.jsonl
: > "$OUT"
jobs=$($SSH 'condor_q 9664325 9664259 9665210 -constraint "JobStatus==2" -af ClusterId ProcId' 2>/dev/null)
while read -r cl proc; do
  [ -z "$cl" ] && continue
  eid=$($SSH "condor_q $cl.$proc -af Arguments 2>/dev/null" | tr ' ' '\n' | grep -m1 "__nph")
  lines=$($SSH "timeout 30 condor_ssh_to_job -auto-retry $cl.$proc 'grep hardcoded_adapt_iter _condor_stdout 2>/dev/null | tail -90'" 2>/dev/null)
  n=$(echo "$lines" | grep -c hardcoded_adapt_iter)
  echo "SAMPLED $cl.$proc $eid rounds=$n" >&2
  echo "$lines" | while read -r line; do
    payload=${line#*AI_LOG }
    echo "{\"cluster\": $cl, \"proc\": $proc, \"eid\": \"$eid\", \"row\": $payload}" >> "$OUT"
  done
done <<< "$jobs"
echo HARVEST_DONE >&2

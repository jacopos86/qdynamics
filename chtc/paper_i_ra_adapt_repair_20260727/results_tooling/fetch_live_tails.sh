#!/bin/bash
# Pull partial traces for every RUNNING job so in-flight cells can be plotted.
# Running jobs have no .out on the access point (stream_output=False), but
# condor_tail reads their live stdout.
set -u
SOCK="$HOME/.ssh/cm-chtc-agent.sock"
SSH="ssh -n -o ControlPath=$SOCK -o BatchMode=yes -o ConnectTimeout=20 jsstrobel@ap2001.chtc.wisc.edu"
OUT="${1:?usage: fetch_live_tails.sh <outdir>}"
mkdir -p "$OUT"
$SSH true 2>/dev/null || { echo "SOCKET-DEAD"; exit 1; }
for cl in 9679133 9675078; do   # B9 continuation + B3 costexp continuation
  for p in $($SSH "condor_q $cl -af ProcId JobStatus 2>/dev/null" | awk '$2==2{print $1}'); do
    eid=$($SSH "condor_q $cl -af ProcId Args 2>/dev/null" | awk -v p="$p" '$1==p{print $2}')
    f="$OUT/$cl.$p.out"
    $SSH "condor_tail -maxbytes 4000000 $cl.$p 2>/dev/null" > "$f" 2>/dev/null
    n=$(grep -c hardcoded_adapt_iter "$f" 2>/dev/null || echo 0)
    if [ "${n:-0}" -gt 0 ]; then echo "$cl.$p  $n rounds  ${eid:-?}"; else rm -f "$f"; echo "$cl.$p  (no rounds yet)"; fi
  done
done

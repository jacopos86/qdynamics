#!/bin/bash
# One-command refresh of the Paper-I RA comparison PDF from completed CHTC cells.
#
#   bash refresh_paper_i_ra_comparison_20260817.sh
#
# Pulls any new completed archives from the active campaign staging dirs,
# extracts each cell's run/summary, and regenerates the family-paged PDF at
# output/pdf/paper_i_ra_allphase_adaptive_20260817/comparison_latest.pdf.
# Read-only against CHTC; incremental (already-fetched archives are skipped).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# Single-flight: overlapping refreshes double peak RAM; never allow two.
LOCK="/tmp/paper_i_ra_refresh.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another refresh is running ($LOCK exists); skipping"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
REMOTE=jsstrobel@128.105.68.112
SOCK=~/.ssh/cm-chtc-9661863.sock
DATA="$REPO/output/local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data"
OUT="$REPO/output/pdf/paper_i_ra_allphase_adaptive_20260817/comparison_latest.pdf"
# Only the live final-source clusters (9662333 / 9662334). The older banked
# v3/v4 archives are pre-fix cross-checks, not paper data — including them
# would collide with the v7 reruns of the same cells.
STAGING=(
  /staging/jsstrobel/paper_i_ra_allphase_adaptive_three_arm_maximum_k50_20260817_v7/transfer
  /staging/jsstrobel/paper_i_ra_allphase_adaptive_always_open_position_maximum_k50_20260817_v4/transfer
  /staging/jsstrobel/paper_i_ra_allphase_adaptive_forced_k50_four_arm_20260817_v2/transfer
  /staging/jsstrobel/paper_i_ra_allphase_adaptive_min_floors_four_arm_20260817_v2/transfer
)

mkdir -p "$DATA/archives"
for dir in "${STAGING[@]}"; do
  for path in $(ssh -S "$SOCK" "$REMOTE" "ls $dir/*.tar.gz 2>/dev/null" || true); do
    base="$(basename "$path")"
    if [ ! -f "$DATA/archives/$base" ]; then
      echo "fetching $base"
      scp -o ControlPath="$SOCK" "$REMOTE:$path" "$DATA/archives/$base"
    fi
  done
done

for archive in "$DATA"/archives/*.tar.gz; do
  base="$(basename "$archive" .tar.gz)"
  cell="x_${base%%__nph*}"
  if [ ! -f "$DATA/$cell/run/summary/summary.json" ]; then
    mkdir -p "$DATA/$cell"
    # Failure archives lack run/summary; tolerate and skip them at load time.
    tar -xzf "$archive" -C "$DATA/$cell" ./run/summary ./run/execution_manifest.json ./run/result/result.json 2>/dev/null || true
  fi
done

# Live partial trajectories: full AI_LOG tails of every running cell
# (fleet + continuation clusters), sequential, plain text only.
mkdir -p "$DATA/workdir" "$DATA/live_tails"
CLUSTERS="9662333 9662334 9662370 9662396 9662543 9662556 9662562 9662564 9662565 9662566 9662959 9663436 9663437"
ssh -S "$SOCK" "$REMOTE" "
for cl in $CLUSTERS; do
  condor_q \$cl -af ClusterId ProcId JobStatus Args 2>/dev/null | while read -r c proc status args; do
    [ \"\$status\" = \"2\" ] || continue
    exec_id=\$(echo \"\$args\" | grep -oE 'allphase_maxk50__[a-z0-9_]+__nph[37]' | head -1)
    [ -n \"\$exec_id\" ] && echo \"\$c.\$proc \$exec_id\"
  done
done" > "$DATA/workdir/running_jobs.txt" 2>/dev/null || true
: > "$DATA/workdir/live_depths.txt"
while read -r job exec_id; do
  ssh -n -S "$SOCK" "$REMOTE" "condor_tail -maxbytes 9000000 $job 2>/dev/null" \
    | grep "^AI_LOG" > "$DATA/live_tails/${exec_id}.tail" || true
  depth=$(grep -oE '"depth": [0-9]+' "$DATA/live_tails/${exec_id}.tail" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
  echo "$exec_id ${depth:-0}" >> "$DATA/workdir/live_depths.txt"
done < "$DATA/workdir/running_jobs.txt"
# Remove tails of cells that are no longer running (completed cells now have
# real archives; a stale tail must not shadow anything).
for tail in "$DATA/live_tails/"*.tail; do
  [ -f "$tail" ] || continue
  exec_id="$(basename "$tail" .tail)"
  grep -q " $exec_id\$" "$DATA/workdir/running_jobs.txt" || rm -f "$tail"
done

# Hard memory cap: a runaway build must die before it takes the machine.
( ulimit -v 6291456 2>/dev/null || true
  python3 "$HERE/build_paper_i_ra_phase123_qiskit_comparison_pdf_20260817.py" \
    --ra-cells-dir "$DATA" \
    --workdir "$DATA/workdir" \
    --live-tails-dir "$DATA/live_tails" \
    --output "$OUT" \
    --status-note "refreshed $(date '+%Y-%m-%d %H:%M %Z') (completed + live partial trajectories)"
)
echo "PDF: $OUT"

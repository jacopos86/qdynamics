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

python3 "$HERE/build_paper_i_ra_phase123_qiskit_comparison_pdf_20260817.py" \
  --ra-cells-dir "$DATA" \
  --workdir "$DATA/workdir" \
  --output "$OUT" \
  --status-note "refreshed $(date '+%Y-%m-%d %H:%M %Z') from completed CHTC cells"
echo "PDF: $OUT"

#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/Users/jakestrobel/Documents/Holstein_implementation/Latex"
DST_DIR="/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH"

files=(
  "adaptive_selection_staged_continuation.tex"
  "adaptive_selection_staged_continuation.pdf"
  "adaptive_selection_and_mclachlan_time_dynamics.tex"
  "adaptive_selection_and_mclachlan_time_dynamics.pdf"
)

for name in "${files[@]}"; do
  cp "${SRC_DIR}/${name}" "${DST_DIR}/${name}"
done

echo "Synced manuscript sources and PDFs into ${DST_DIR}"

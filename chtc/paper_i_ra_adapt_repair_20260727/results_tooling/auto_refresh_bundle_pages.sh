#!/bin/bash
# Refresh the Paper-I bundle-pages PDF whenever its inputs have changed.
#
# Idempotent and safe to run on a timer: it fingerprints every input the data
# layer reads, and only invokes pdflatex when that fingerprint moves. Logs one
# line per run either way, so a silent failure is visible.
set -uo pipefail

TOOLING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/Users/jakestrobel/local_repos/Holstein_test_fullclone_3
PDF="$REPO/output/pdf/paper_i_ra_allphase_adaptive_20260817/paper_i_bundle_pages_20260823.pdf"
STATE="$TOOLING/.bundle_pages_inputs.sha256"
LOG="$REPO/output/pdf/paper_i_ra_allphase_adaptive_20260817/auto_refresh.log"
mkdir -p "$(dirname "$LOG")"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Fingerprint: every result/harvest/stream the pages are built from, plus the
# builder sources themselves, so a code change also triggers a rebuild.
fingerprint() {
  {
    find "$REPO/output/local_runs" -maxdepth 3 \
         \( -name '*.harvest.json' -o -name '*RETENTION_MANIFEST.json' \) \
         -exec shasum -a256 {} + 2>/dev/null
    find "$REPO/output/local_runs" -maxdepth 3 -name '*.out' \
         -newermt '2026-08-01' -exec shasum -a256 {} + 2>/dev/null
    shasum -a256 "$TOOLING/_bundle_pages_data.py" "$TOOLING/build_bundle_pages_tex.py" 2>/dev/null
  } | sort | shasum -a256 | cut -d' ' -f1
}

NEW="$(fingerprint)"
OLD="$(cat "$STATE" 2>/dev/null || echo none)"

if [ "$NEW" = "$OLD" ] && [ -f "$PDF" ]; then
  echo "$(stamp) unchanged ($NEW)" >> "$LOG"
  exit 0
fi

cd "$TOOLING" || exit 1
OUT="$(python3 build_bundle_pages_tex.py 2>&1)"
RC=$?
if [ $RC -eq 0 ] && [ -f "$PDF" ]; then
  echo "$NEW" > "$STATE"
  echo "$(stamp) rebuilt: $(echo "$OUT" | tail -1)" >> "$LOG"
else
  echo "$(stamp) FAILED rc=$RC: $(echo "$OUT" | tail -3 | tr '\n' ' ')" >> "$LOG"
  exit $RC
fi

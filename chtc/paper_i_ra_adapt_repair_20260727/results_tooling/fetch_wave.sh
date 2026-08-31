#!/bin/bash
# Fetch the overnight wave: verify size+sha, then clear remote. B2 (retired) skipped.
set -u
SOCK=/tmp/chtc-38a9500f9d868895c19b7bbdb8225ead24f6170a
SSH="ssh -n -o ControlPath=$SOCK -o BatchMode=yes jsstrobel@ap2001.chtc.wisc.edu"
LR=/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs
declare -a PAIRS=(
 "ra_adapt_costexp_depthfix_sixregime_20260820_v2|$LR/ra_adapt_costexp_depthfix_sixregime_20260820_v2/transfer"
 "ra_adapt_warmstart_sixregime_20260820_v1|$LR/ra_adapt_warmstart_sixregime_20260820_v1/transfer"
)
for pair in "${PAIRS[@]}"; do
  stage="${pair%%|*}"; dest="${pair##*|}"; mkdir -p "$dest"
  # weak always-insertion first (completes the weak sector), then the rest
  files=$($SSH "ls /staging/jsstrobel/$stage/transfer/*.tar.gz 2>/dev/null")
  for order in "costexp_always_open__weak" "costexp_always_open__intermediate" "costexp_always_open__strong" ""; do
    for rf in $files; do
      base=$(basename "$rf")
      case "$base" in *a_costexp*) continue ;; esac            # Bundle 2: retired, skip
      [ -n "$order" ] && { case "$base" in *"$order"*) ;; *) continue ;; esac; }
      [ -f "$dest/$base" ] && continue
      rsha=$($SSH "sha256sum $rf 2>/dev/null | cut -d' ' -f1")
      rsize=$($SSH "stat -c %s $rf 2>/dev/null")
      [ -z "$rsha" ] && { echo "SKIP $base (no sha)"; continue; }
      scp -q -o ControlPath=$SOCK "jsstrobel@ap2001.chtc.wisc.edu:$rf" "$dest/$base.part" || { echo "FETCH-FAIL $base"; rm -f "$dest/$base.part"; continue; }
      lsha=$(shasum -a 256 "$dest/$base.part" | cut -d' ' -f1)
      lsize=$(stat -f %z "$dest/$base.part")
      if [ "$lsha" = "$rsha" ] && [ "$lsize" = "$rsize" ] && gzip -t "$dest/$base.part" 2>/dev/null; then
        mv "$dest/$base.part" "$dest/$base"
        $SSH "rm $rf" && echo "FETCHED+CLEARED $base ($lsize bytes)"
      else
        echo "VERIFY-FAIL $base (remote kept)"; rm -f "$dest/$base.part"
      fi
    done
  done
done
$SSH "get_quotas /staging/jsstrobel 2>/dev/null | tail -1"
echo WAVE_FETCH_DONE

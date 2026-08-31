#!/bin/bash
# fetch-verify-clear three k50-wave staging dirs (wanted locally for
# Bundle-1 lineage), then release the quota-held transfers 18-20.
set -u
SSH="ssh -n -S $HOME/.ssh/cm-chtc-agent.sock -o BatchMode=yes jsstrobel@ap2001.chtc.wisc.edu"
SCP_OPT="-q -o ControlPath=$HOME/.ssh/cm-chtc-agent.sock"
DEST=/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data/archives
for d in paper_i_ra_allphase_adaptive_forced_k50_four_arm_20260817_v2 paper_i_ra_allphase_adaptive_three_arm_maximum_k50_20260817_v7 paper_i_ra_allphase_adaptive_three_arm_maximum_k50_20260817_v3 paper_i_ra_allphase_adaptive_always_open_position_maximum_k50_20260817_v4 paper_i_ra_allphase_adaptive_min_floors_four_arm_20260817_v2 paper_i_ra_allphase_adaptive_append_remaining5_maximum_k50_20260817_v4; do
  files=$($SSH "ls /staging/jsstrobel/$d/transfer/*.tar.gz 2>/dev/null")
  for rf in $files; do
    base=$(basename "$rf")
    read -r rsha rsize <<< "$($SSH "sha256sum $rf | cut -d' ' -f1; stat -c %s $rf" | tr '\n' ' ')"
    [ -z "$rsha" ] && { echo "SKIP $base (stat failed)"; continue; }
    if [ -f "$DEST/$base" ]; then
      lsha=$(shasum -a 256 "$DEST/$base" | cut -d' ' -f1)
    else
      scp $SCP_OPT "jsstrobel@ap2001.chtc.wisc.edu:$rf" "$DEST/$base.part" || { echo "FETCH-FAIL $base"; rm -f "$DEST/$base.part"; continue; }
      lsha=$(shasum -a 256 "$DEST/$base.part" | cut -d' ' -f1)
      if [ "$lsha" = "$rsha" ] && gzip -t "$DEST/$base.part" && tar -tzf "$DEST/$base.part" > /dev/null 2>&1; then
        mv "$DEST/$base.part" "$DEST/$base"
      else
        echo "VERIFY-FAIL $base (remote kept)"; rm -f "$DEST/$base.part"; continue
      fi
    fi
    if [ "$lsha" = "$rsha" ]; then
      $SSH "rm $rf" && echo "CLEARED $base ($rsize bytes, sha ok)"
    else
      echo "SHA-MISMATCH $base (remote kept)"
    fi
  done
done
$SSH "get_quotas /staging/jsstrobel 2>/dev/null | tail -1"
for p in 18 19 20; do $SSH "condor_release 9664325.$p" && echo "released $p"; done
echo FREE_STAGING_DONE

#!/bin/zsh
cd /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2 || exit 111
exec > artifacts/logs/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1.log 2>&1
echo $$ > artifacts/runstate/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1.pid
python -u artifacts/runstate/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1.py
code=$?
echo "$code" > artifacts/runstate/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1.exit
exit "$code"

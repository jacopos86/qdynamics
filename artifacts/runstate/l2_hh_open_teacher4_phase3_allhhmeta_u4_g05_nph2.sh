#!/bin/zsh
cd /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2 || exit 111
exec > artifacts/logs/l2_hh_open_teacher4_phase3_allhhmeta_u4_g05_nph2.log 2>&1
echo $$ > artifacts/runstate/l2_hh_open_teacher4_phase3_allhhmeta_u4_g05_nph2.pid
python -u pipelines/hardcoded/hh_staged_noiseless.py \
  --L 2 \
  --t 1.0 \
  --u 4.0 \
  --dv 0.0 \
  --omega0 1.0 \
  --g-ep 0.5 \
  --n-ph-max 2 \
  --boson-encoding binary \
  --ordering blocked \
  --boundary open \
  --warm-ansatz hh_hva_ptw \
  --warm-reps 3 \
  --warm-restarts 8 \
  --warm-maxiter 6000 \
  --warm-method SPSA \
  --warm-seed 7 \
  --adapt-pool all_hh_meta_v1 \
  --adapt-continuation-mode phase3_v1 \
  --adapt-max-depth 128 \
  --adapt-maxiter 6000 \
  --adapt-reopt-policy full \
  --adapt-no-repeats \
  --phase1-no-prune \
  --adapt-drop-floor -1 \
  --adapt-drop-patience 0 \
  --adapt-drop-min-depth 0 \
  --adapt-grad-floor -1 \
  --adapt-finite-angle-fallback \
  --adapt-finite-angle 0.1 \
  --adapt-finite-angle-min-improvement 1e-12 \
  --phase3-runtime-split-mode shortlist_pauli_children_v1 \
  --final-reps 2 \
  --final-restarts 8 \
  --final-maxiter 6000 \
  --replay-seed-policy auto \
  --replay-continuation-mode phase3_v1 \
  --ecut-1 1e-2 \
  --ecut-2 1e-4 \
  --tag l2_hh_open_teacher4_phase3_allhhmeta_u4_g05_nph2 \
  --skip-pdf
code=$?
echo "$code" > artifacts/runstate/l2_hh_open_teacher4_phase3_allhhmeta_u4_g05_nph2.exit
exit "$code"

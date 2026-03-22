#!/usr/bin/env bash
set -euo pipefail
cd /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2
TAG="l2_nph2_hva_refine_b1_splitoff_resume1"
RUN_DIR="artifacts/agent_runs/l2_nph2_beam_depth_search/recovery"
LOG="$RUN_DIR/${TAG}.stdout.log"
STATUS="$RUN_DIR/${TAG}.status.json"
START_TS=$(date -u +%FT%TZ)
printf '{"tag":"%s","status":"running","started_utc":"%s"}\n' "$TAG" "$START_TS" > "$STATUS"
{
  echo "START $START_TS"
  python pipelines/hardcoded/hh_staged_noiseless.py \
    --L 2 \
    --n-ph-max 2 \
    --boundary open \
    --ordering blocked \
    --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 1.0 \
    --warm-ansatz hh_hva \
    --warm-reps 3 --warm-restarts 4 --warm-maxiter 1778 \
    --seed-refine-family uccsd_otimes_paop_lf_std \
    --seed-refine-reps 3 --seed-refine-maxiter 1778 --seed-refine-optimizer SPSA \
    --adapt-pool paop_lf_std \
    --adapt-continuation-mode phase3_v1 \
    --adapt-max-depth 80 --adapt-maxiter 2222 \
    --adapt-eps-grad 5e-7 --adapt-eps-energy 1e-9 \
    --final-reps 3 --final-restarts 4 --final-maxiter 1778 \
    --replay-continuation-mode phase3_v1 \
    --noiseless-methods suzuki2 \
    --t-final 20.0 --num-times 201 --trotter-steps 128 --exact-steps-multiplier 2 \
    --skip-pdf \
    --circuit-backend-name GenericBackendV2 \
    --circuit-use-fake-backend \
    --circuit-transpile-optimization-level 1 \
    --handoff-from-warm-checkpoint artifacts/json/l2_nph2_hva_refine_b1_splitoff_warm_cutover_state.json \
    --tag "$TAG"
  EC=$?
  END_TS=$(date -u +%FT%TZ)
  echo "END $END_TS ec=$EC"
  printf '{"tag":"%s","status":"finished","exit_code":%d,"finished_utc":"%s"}\n' "$TAG" "$EC" "$END_TS" > "$STATUS"
  exit "$EC"
} >> "$LOG" 2>&1 || {
  EC=$?
  END_TS=$(date -u +%FT%TZ)
  echo "END $END_TS ec=$EC" >> "$LOG"
  printf '{"tag":"%s","status":"finished","exit_code":%d,"finished_utc":"%s"}\n' "$TAG" "$EC" "$END_TS" > "$STATUS"
  exit "$EC"
}

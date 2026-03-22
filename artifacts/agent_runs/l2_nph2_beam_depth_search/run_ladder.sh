#!/usr/bin/env bash
set -euo pipefail
cd /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2

RUN_ROOT="artifacts/agent_runs/l2_nph2_beam_depth_search"
LOG_DIR="$RUN_ROOT/logs"
MANIFEST="$RUN_ROOT/manifest.tsv"
SUMMARY="$RUN_ROOT/summary.tsv"
: > "$MANIFEST"
printf 'tag\texit_code\tworkflow_json\treplay_json\tlog\n' > "$SUMMARY"

common=(
  python pipelines/hardcoded/hh_staged_noiseless.py
  --L 2
  --n-ph-max 2
  --boundary open
  --ordering blocked
  --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 1.0
  --warm-ansatz hh_hva
  --warm-reps 3 --warm-restarts 4 --warm-maxiter 1778
  --seed-refine-family uccsd_otimes_paop_lf_std
  --seed-refine-reps 3 --seed-refine-maxiter 1778 --seed-refine-optimizer SPSA
  --adapt-pool paop_lf_std
  --adapt-continuation-mode phase3_v1
  --adapt-max-depth 80 --adapt-maxiter 2222
  --adapt-eps-grad 5e-7 --adapt-eps-energy 1e-9
  --final-reps 3 --final-restarts 4 --final-maxiter 1778
  --replay-continuation-mode phase3_v1
  --noiseless-methods suzuki2
  --t-final 20.0 --num-times 201 --trotter-steps 128 --exact-steps-multiplier 2
  --skip-pdf
  --circuit-backend-name GenericBackendV2
  --circuit-use-fake-backend
  --circuit-transpile-optimization-level 1
)

run_one() {
  local tag="$1"; shift
  local log="$LOG_DIR/${tag}.log"
  echo "===== START $tag $(date -u +%FT%TZ) =====" | tee -a "$MANIFEST"
  set +e
  "${common[@]}" --tag "$tag" "$@" 2>&1 | tee "$log"
  local ec=${PIPESTATUS[0]}
  set -e
  local workflow_json=""
  local replay_json=""
  if [[ -f "$log" ]]; then
    workflow_json=$(grep -E '^workflow_json=' "$log" | tail -n1 | cut -d= -f2- || true)
    replay_json=$(grep -E '^replay_json=' "$log" | tail -n1 | cut -d= -f2- || true)
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$ec" "$workflow_json" "$replay_json" "$log" >> "$SUMMARY"
  echo "===== END $tag ec=$ec $(date -u +%FT%TZ) =====" | tee -a "$MANIFEST"
  return "$ec"
}

run_one l2_nph2_hva_refine_b1_splitoff \
  --adapt-beam-live-branches 1 \
  --phase3-runtime-split-mode off

for B in 2 3 4; do
  run_one "l2_nph2_hva_refine_b${B}_splitoff" \
    --adapt-beam-live-branches "$B" \
    --adapt-beam-children-per-parent 2 \
    --adapt-beam-terminated-keep "$B" \
    --phase3-runtime-split-mode off

  run_one "l2_nph2_hva_refine_b${B}_spliton" \
    --adapt-beam-live-branches "$B" \
    --adapt-beam-children-per-parent 2 \
    --adapt-beam-terminated-keep "$B" \
    --phase3-runtime-split-mode shortlist_pauli_children_v1

done

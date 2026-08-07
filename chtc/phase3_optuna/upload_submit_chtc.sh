#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:-upload-only}"
REMOTE_HOST="${CHTC_HOST:-jsstrobel@ap2001.chtc.wisc.edu}"
REMOTE_BASE="${CHTC_REMOTE_BASE:-Holstein_phase3_optuna_chtc}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_CONTROL_PATH="${CHTC_SSH_CONTROL_PATH:-/tmp/chtc-%C}"
SSH_OPTS=(-o ControlMaster=auto -o ControlPersist=10m -o "ControlPath=$SSH_CONTROL_PATH")
RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPersist=10m -o ControlPath=$SSH_CONTROL_PATH"

preflight_submit_file() {
  local submit_rel="$1"
  mkdir -p "$LOCAL_ROOT/tmp/phase3_optuna_preflight"
  local out_json="$LOCAL_ROOT/tmp/phase3_optuna_preflight/${ACTION}.json"
  echo "Preflighting CHTC submit contract: $submit_rel"
  python "$LOCAL_ROOT/chtc/phase3_optuna/preflight_submit.py" \
    --submit "$LOCAL_ROOT/$submit_rel" \
    --output-json "$out_json"
  echo "Preflight manifest: $out_json"
}

case "$ACTION" in
  generic-static-table-snake-routeA-repair-smoke)
    preflight_submit_file "chtc/phase3_optuna/submit_generic_static_table_snake_routeA_repair_smoke.sub"
    ;;
  generic-static-table-snake-routeA-repair-full)
    preflight_submit_file "chtc/phase3_optuna/submit_generic_static_table_snake_routeA_repair_full.sub"
    ;;
  weighted-nph2-ref3-routea-smoke)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_weighted_nph2ref3_bosonic_hh_smoke.sub"
    ;;
  weighted-nph2-ref3-routea-full)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_weighted_nph2ref3_bosonic_hh.sub"
    ;;
  routeA-phase0-nph2-ref3-oracle-smoke)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_phase0_nph2_ref3_oracle_smoke.sub"
    ;;
  routeA-phase0-nph2-ref3-oracle-full)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_phase0_nph2_ref3_oracle_full.sub"
    ;;
  routeA-phase0-phonon-repair-nph3-ref4-smoke)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_phase0_phonon_repair_nph3_ref4_smoke.sub"
    ;;
  routeA-phase0-phonon-repair-nph3-ref4-full)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_phase0_phonon_repair_nph3_ref4_full.sub"
    ;;
  routeA-phase0-phonon-repair-nph4-ref5-smoke)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_phase0_phonon_repair_nph4_ref5_smoke.sub"
    ;;
  routeA-phase0-phonon-repair-nph4-ref5-full)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_phase0_phonon_repair_nph4_ref5_full.sub"
    ;;
  generic-static-table-paper-i-clean-fermionic-benchmarks-smoke)
    preflight_submit_file "chtc/phase3_optuna/submit_generic_static_table_paper_i_clean_fermionic_benchmarks_smoke.sub"
    ;;
  generic-static-table-paper-i-clean-fermionic-benchmarks-full)
    preflight_submit_file "chtc/phase3_optuna/submit_generic_static_table_paper_i_clean_fermionic_benchmarks_full.sub"
    ;;
  routeA-paper-i-clean-fermionic-smoke)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_paper_i_clean_fermionic_smoke.sub"
    ;;
  routeA-paper-i-clean-fermionic-full)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_paper_i_clean_fermionic_full.sub"
    ;;
  routeA-paper-i-clean-phonon-ladder-nph2-ref4-smoke)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_paper_i_clean_phonon_ladder_nph2_ref4_smoke.sub"
    ;;
  routeA-paper-i-clean-phonon-ladder-nph2-ref4-full)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_paper_i_clean_phonon_ladder_nph2_ref4_full.sub"
    ;;
  generic-static-table-paper-i-clean-ladder-nph2-ref4-smoke)
    preflight_submit_file "chtc/phase3_optuna/submit_generic_static_table_paper_i_clean_ladder_nph2_ref4_smoke.sub"
    ;;
  generic-static-table-paper-i-clean-ladder-nph2-ref4-full)
    preflight_submit_file "chtc/phase3_optuna/submit_generic_static_table_paper_i_clean_ladder_nph2_ref4_full.sub"
    ;;
  generic-static-table-paper-i-hh-symmetric-comparators-full-meta-newpool-20260530-v1-smoke)
    preflight_submit_file "chtc/phase3_optuna/submit_generic_static_table_paper_i_hh_symmetric_comparators_full_meta_newpool_20260530_v1_smoke.sub"
    ;;
  generic-static-table-paper-i-hh-symmetric-comparators-full-meta-newpool-20260530-v1-full)
    preflight_submit_file "chtc/phase3_optuna/submit_generic_static_table_paper_i_hh_symmetric_comparators_full_meta_newpool_20260530_v1_full.sub"
    ;;
  routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-20260530-v1)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_20260530_v1.sub"
    ;;
  routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v2)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v2.sub"
    ;;
  routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v3)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v3.sub"
    ;;
  routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v4)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v4.sub"
    ;;
  routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v5)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v5.sub"
    ;;
  routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v6)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v6.sub"
    ;;
  routeA-paper-i-table-cutoff-nph2-ref5-selected-logical-repair-20260525-v3)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_paper_i_table_cutoff_nph2_ref5_selected_logical_repair_20260525_v3.sub"
    ;;
  routeA-paper-i-table-cutoff-hk-nph4-ref7-selected-logical-repair-20260525-v3)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_paper_i_table_cutoff_hk_nph4_ref7_selected_logical_repair_20260525_v3.sub"
    ;;
  routeA-paper-i-table-cutoff-spin-boson-strong-nph6-ref9-selected-logical-repair-20260525-v3)
    preflight_submit_file "chtc/phase3_optuna/submit_routeA_paper_i_table_cutoff_spin_boson_strong_nph6_ref9_selected_logical_repair_20260525_v3.sub"
    ;;
  *)
    ;;
esac

echo "Uploading CHTC bundle: $LOCAL_ROOT -> $REMOTE_HOST:$REMOTE_BASE"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_BASE/logs' '$REMOTE_BASE/raw_outputs' '$REMOTE_BASE/artifacts/agent_runs/20260425_phase3_historical_best_ledger_v1'"
rsync -az -e "$RSYNC_SSH" \
  --exclude '.git' \
  --exclude '.pytest_cache' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.venv' \
  --exclude 'src/quantum/chemistry/conda-env' \
  --exclude 'artifacts' \
  --exclude 'output' \
  --exclude 'tmp' \
  --exclude 'plots' \
  --exclude 'prompt-exports' \
  --exclude 'raw_outputs' \
  --exclude 'logs' \
  --exclude 'chtc/phase3_optuna/image.sif' \
  "$LOCAL_ROOT/" "$REMOTE_HOST:$REMOTE_BASE/"
if [[ -f "$LOCAL_ROOT/artifacts/agent_runs/20260425_phase3_historical_best_ledger_v1/historical_best_ledger.json" ]]; then
  rsync -az -e "$RSYNC_SSH" \
    "$LOCAL_ROOT/artifacts/agent_runs/20260425_phase3_historical_best_ledger_v1/historical_best_ledger.json" \
    "$REMOTE_HOST:$REMOTE_BASE/artifacts/agent_runs/20260425_phase3_historical_best_ledger_v1/historical_best_ledger.json"
fi
case "$ACTION" in
  upload-only)
    echo "Upload complete."
    ;;
  image-build)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && condor_submit chtc/phase3_optuna/submit_image_build.sub"
    ;;
  smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_smoke.sub"
    ;;
  full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_full.sub"
    ;;
  recovery-v4)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_recovery_fullmeta_spsa_costlex_v4.sub"
    ;;
  three-lane-spsa)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_global_spsa_three_lane_canonical.sub"
    ;;
  three-lane-powell)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_global_powell_three_lane_canonical.sub"
    ;;
  tripartite-spsa-algebraic)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_global_tripartite_spsa_algebraic.sub"
    ;;
  tripartite-qnspsa-algebraic)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_global_tripartite_qnspsa_algebraic.sub"
    ;;
  tripartite-spsa-qnspsa-algebraic)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_global_tripartite_spsa_algebraic.sub && condor_submit chtc/phase3_optuna/submit_global_tripartite_qnspsa_algebraic.sub"
    ;;
  tripartite-spsa-ab-l2nph1-reset-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_global_tripartite_spsa_ab_l2nph1_reset_smoke.sub"
    ;;
  tripartite-spsa-ab-l2nph1-reset-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_global_tripartite_spsa_ab_l2nph1_reset.sub"
    ;;
  fermionic-protected-corr)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_global_spsa_fermionic_protected_corr.sub"
    ;;
  canonical-debug-reruns)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_global_spsa_canonical_debug_reruns.sub"
    ;;
  static-hh-paper-l2)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_static_hh_paper_l2.sub"
    ;;
  generic-static-table-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_smoke.sub"
    ;;
  generic-static-table-nph2-ref3-value-noise-one-record-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_nph2_ref3_v1_value_noise_one_record_smoke.sub"
    ;;
  generic-static-table-nph2-ref3-value-noise-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_nph2_ref3_v1_value_noise_smoke.sub"
    ;;
  generic-static-table-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_full.sub"
    ;;
  generic-static-table-geo-qeb)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_geo_qeb.sub"
    ;;
  generic-static-table-pos-geo)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_pos_geo.sub"
    ;;
  generic-static-table-missing-event-v1)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_missing_event_v1.sub"
    ;;
  generic-static-table-nph2-ref3-calibrated-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_nph2_ref3_calibrated_smoke.sub"
    ;;
  generic-static-table-nph2-ref3-calibrated-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_nph2_ref3_calibrated_full.sub"
    ;;
  generic-static-table-nph2-ref3-snake-matched-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_nph2_ref3_snake_matched_smoke.sub"
    ;;
  generic-static-table-nph2-ref3-snake-matched-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_nph2_ref3_snake_matched_full.sub"
    ;;
  generic-static-table-nph2-ref3-snake-executable-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_nph2_ref3_snake_executable_smoke.sub"
    ;;
  generic-static-table-nph2-ref3-snake-executable-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_nph2_ref3_snake_executable_full.sub"
    ;;
  generic-static-table-nph2-ref3-snake-profile-small-e4minus-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_nph2_ref3_snake_profile_small_e4minus_smoke.sub"
    ;;
  generic-static-table-nph2-ref3-snake-profile-small-e4minus-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_nph2_ref3_snake_profile_small_e4minus_full.sub"
    ;;
  generic-static-table-nph2-ref3-snake-route-a-spsa-prior-depth12-weak-lanes-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_nph2_ref3_snake_route_a_spsa_prior_depth12_weak_lanes_full.sub"
    ;;
  generic-static-table-snake-routeA-repair-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_snake_routeA_repair_smoke.sub"
    ;;
  generic-static-table-snake-routeA-repair-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_snake_routeA_repair_full.sub"
    ;;
  generic-static-table-paper-i-clean-fermionic-benchmarks-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_paper_i_clean_fermionic_benchmarks_smoke.sub"
    ;;
  generic-static-table-paper-i-clean-fermionic-benchmarks-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_paper_i_clean_fermionic_benchmarks_full.sub"
    ;;
  routeA-paper-i-clean-fermionic-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_paper_i_clean_fermionic_smoke.sub"
    ;;
  routeA-paper-i-clean-fermionic-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_paper_i_clean_fermionic_full.sub"
    ;;
  routeA-paper-i-clean-phonon-ladder-nph2-ref4-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_paper_i_clean_phonon_ladder_nph2_ref4_smoke.sub"
    ;;
  routeA-paper-i-clean-phonon-ladder-nph2-ref4-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_paper_i_clean_phonon_ladder_nph2_ref4_full.sub"
    ;;
  routeA-paper-i-table-cutoff-nph2-ref5-selected-logical-repair-20260525-v3)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_paper_i_table_cutoff_nph2_ref5_selected_logical_repair_20260525_v3.sub"
    ;;
  routeA-paper-i-table-cutoff-hk-nph4-ref7-selected-logical-repair-20260525-v3)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_paper_i_table_cutoff_hk_nph4_ref7_selected_logical_repair_20260525_v3.sub"
    ;;
  routeA-paper-i-table-cutoff-spin-boson-strong-nph6-ref9-selected-logical-repair-20260525-v3)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_paper_i_table_cutoff_spin_boson_strong_nph6_ref9_selected_logical_repair_20260525_v3.sub"
    ;;
  generic-static-table-paper-i-clean-ladder-nph2-ref4-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_paper_i_clean_ladder_nph2_ref4_smoke.sub"
    ;;
  generic-static-table-paper-i-clean-ladder-nph2-ref4-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_paper_i_clean_ladder_nph2_ref4_full.sub"
    ;;
  generic-static-table-paper-i-hh-symmetric-comparators-full-meta-newpool-20260530-v1-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_paper_i_hh_symmetric_comparators_full_meta_newpool_20260530_v1_smoke.sub"
    ;;
  generic-static-table-paper-i-hh-symmetric-comparators-full-meta-newpool-20260530-v1-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_generic_static_table_paper_i_hh_symmetric_comparators_full_meta_newpool_20260530_v1_full.sub"
    ;;
  routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-20260530-v1)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_20260530_v1.sub"
    ;;
  routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v2)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v2.sub"
    ;;
  routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v3)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v3.sub"
    ;;
  routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v4)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v4.sub"
    ;;
  routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v5)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v5.sub"
    ;;
  routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v6)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v6.sub"
    ;;
  weighted-nph2-ref3-routea-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_weighted_nph2ref3_bosonic_hh_smoke.sub"
    ;;
  weighted-nph2-ref3-routea-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_weighted_nph2ref3_bosonic_hh.sub"
    ;;
  routeA-phase0-nph1-oracle-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_phase0_nph1_oracle_smoke.sub"
    ;;
  routeA-phase0-nph1-oracle-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_phase0_nph1_oracle_full.sub"
    ;;
  routeA-phase0-nph2-ref3-oracle-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_phase0_nph2_ref3_oracle_smoke.sub"
    ;;
  routeA-phase0-nph2-ref3-oracle-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_phase0_nph2_ref3_oracle_full.sub"
    ;;
  routeA-phase0-phonon-repair-nph3-ref4-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_phase0_phonon_repair_nph3_ref4_smoke.sub"
    ;;
  routeA-phase0-phonon-repair-nph3-ref4-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_phase0_phonon_repair_nph3_ref4_full.sub"
    ;;
  routeA-phase0-phonon-repair-nph4-ref5-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_phase0_phonon_repair_nph4_ref5_smoke.sub"
    ;;
  routeA-phase0-phonon-repair-nph4-ref5-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_phase0_phonon_repair_nph4_ref5_full.sub"
    ;;
  routeA-phase0-nph1-class-smoke)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_phase0_nph1_class_smoke.sub"
    ;;
  routeA-phase0-nph1-class-full)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_BASE' && test -f chtc/phase3_optuna/image.sif && condor_submit chtc/phase3_optuna/submit_routeA_phase0_nph1_class_full.sub"
    ;;
  *)
    echo "usage: $0 {upload-only|image-build|smoke|full|recovery-v4|three-lane-spsa|three-lane-powell|fermionic-protected-corr|canonical-debug-reruns|static-hh-paper-l2|generic-static-table-smoke|generic-static-table-nph2-ref3-value-noise-one-record-smoke|generic-static-table-nph2-ref3-value-noise-smoke|generic-static-table-full|generic-static-table-geo-qeb|generic-static-table-pos-geo|generic-static-table-missing-event-v1|generic-static-table-nph2-ref3-calibrated-smoke|generic-static-table-nph2-ref3-calibrated-full|generic-static-table-nph2-ref3-snake-matched-smoke|generic-static-table-nph2-ref3-snake-matched-full|generic-static-table-nph2-ref3-snake-executable-smoke|generic-static-table-nph2-ref3-snake-executable-full|generic-static-table-nph2-ref3-snake-profile-small-e4minus-smoke|generic-static-table-nph2-ref3-snake-profile-small-e4minus-full|generic-static-table-nph2-ref3-snake-route-a-spsa-prior-depth12-weak-lanes-full|generic-static-table-snake-routeA-repair-smoke|generic-static-table-snake-routeA-repair-full|weighted-nph2-ref3-routea-smoke|weighted-nph2-ref3-routea-full|routeA-phase0-nph1-oracle-smoke|routeA-phase0-nph1-oracle-full|routeA-phase0-nph2-ref3-oracle-smoke|routeA-phase0-nph2-ref3-oracle-full|routeA-phase0-phonon-repair-nph3-ref4-smoke|routeA-phase0-phonon-repair-nph3-ref4-full|routeA-phase0-phonon-repair-nph4-ref5-smoke|routeA-phase0-phonon-repair-nph4-ref5-full|routeA-phase0-nph1-class-smoke|routeA-phase0-nph1-class-full|routeA-paper-i-clean-phonon-ladder-nph2-ref4-smoke|routeA-paper-i-clean-phonon-ladder-nph2-ref4-full|generic-static-table-paper-i-clean-ladder-nph2-ref4-smoke|generic-static-table-paper-i-clean-ladder-nph2-ref4-full|generic-static-table-paper-i-hh-symmetric-comparators-full-meta-newpool-20260530-v1-smoke|generic-static-table-paper-i-hh-symmetric-comparators-full-meta-newpool-20260530-v1-full|routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v3|routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v4|routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v5|routeA-paper-i-hh-symmetric-snake-energy-geom-nocost-routefix-20260530-v6|tripartite-spsa-algebraic|tripartite-qnspsa-algebraic|tripartite-spsa-qnspsa-algebraic|tripartite-spsa-ab-l2nph1-reset-smoke|tripartite-spsa-ab-l2nph1-reset-full}" >&2
    exit 2
    ;;
esac

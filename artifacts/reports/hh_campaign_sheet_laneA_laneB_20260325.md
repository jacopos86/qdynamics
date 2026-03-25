# HH Campaign Sheet — Lane A / Lane B (2026-03-25)

## Status
- Source-code changes: none
- Execution status: **pending one user choice** on continuation mode
- Required choice before launch:
  - `CONT_MODE=phase3_v1` (canonical README / recommended)
  - or `CONT_MODE=phase1_v1` (literal current staged CLI default)

Use one of:

```bash
export CONT_MODE=phase3_v1
# or
export CONT_MODE=phase1_v1
```

## Core physics lock
Use identical physics unless the specific comparison says otherwise:

- `L=2` for first-batch selector comparisons
- `t=1.0`
- `U=4.0`
- `omega0=1.0`
- `g_ep=0.5`
- `n_ph_max=1`
- open boundary / blocked ordering / binary boson encoding
- local/offline only
- no runtime / hardware submission

## Primary scorecard
For each run, report:

1. best row reaching fixed thresholds:
   - `ΔE_abs <= 1e-3`
   - `ΔE_abs <= 1e-4`
   - `ΔE_abs <= 6e-5`
2. primary axis:
   - `measurement_groups_cumulative`
3. tie-break:
   - `compile_gate_proxy_cumulative`
4. supporting fields:
   - `measurement_shots_cumulative` (nominal only)
   - `selection_mode`
   - `runtime_split_mode`
   - `batch_size`
   - `num_parameters`
   - `stage_depth`
   - final replay `abs_delta_e`

Do **not** sell `measurement_reuse_*` as persistent measured-data reuse.

## Lane A — Upstream selector lane (headline science lane)

### A1. Narrow-core-first curriculum vs broad `full_meta`

#### A1a. Canonical staged narrow-core
```bash
python pipelines/hardcoded/hh_staged_noiseless.py \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-continuation-mode "$CONT_MODE" \
  --skip-pdf \
  --output-json artifacts/json/campaign_A1a_L2_staged_core_${CONT_MODE}.json
```

#### A1b. Broad depth-0 `full_meta`
```bash
python pipelines/hardcoded/hh_staged_noiseless.py \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-pool full_meta \
  --adapt-continuation-mode "$CONT_MODE" \
  --skip-pdf \
  --output-json artifacts/json/campaign_A1b_L2_full_meta_${CONT_MODE}.json
```

### A2. Pool/manifold comparison

#### A2a. `pareto_lean_l2`
```bash
python pipelines/hardcoded/hh_staged_noiseless.py \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-pool pareto_lean_l2 \
  --adapt-continuation-mode "$CONT_MODE" \
  --skip-pdf \
  --output-json artifacts/json/campaign_A2a_L2_pareto_lean_l2_${CONT_MODE}.json
```

#### A2b. `pareto_lean` at L=3
```bash
python pipelines/hardcoded/hh_staged_noiseless.py \
  --L 3 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-pool pareto_lean \
  --adapt-continuation-mode "$CONT_MODE" \
  --skip-pdf \
  --output-json artifacts/json/campaign_A2b_L3_pareto_lean_${CONT_MODE}.json
```

#### A2c. `full_meta` at L=3
```bash
python pipelines/hardcoded/hh_staged_noiseless.py \
  --L 3 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-pool full_meta \
  --adapt-continuation-mode "$CONT_MODE" \
  --skip-pdf \
  --output-json artifacts/json/campaign_A2c_L3_full_meta_${CONT_MODE}.json
```

### A3. Reoptimization triad

#### A3a. `append_only`
```bash
python pipelines/hardcoded/hh_staged_noiseless.py \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-reopt-policy append_only \
  --adapt-continuation-mode "$CONT_MODE" \
  --skip-pdf \
  --output-json artifacts/json/campaign_A3a_L2_append_only_${CONT_MODE}.json
```

#### A3b. `windowed`
```bash
python pipelines/hardcoded/hh_staged_noiseless.py \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-reopt-policy windowed \
  --adapt-continuation-mode "$CONT_MODE" \
  --skip-pdf \
  --output-json artifacts/json/campaign_A3b_L2_windowed_${CONT_MODE}.json
```

#### A3c. `full`
```bash
python pipelines/hardcoded/hh_staged_noiseless.py \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-reopt-policy full \
  --adapt-continuation-mode "$CONT_MODE" \
  --skip-pdf \
  --output-json artifacts/json/campaign_A3c_L2_full_${CONT_MODE}.json
```

### A4. Lifetime-cost ablation

#### A4a. lifetime on
```bash
python pipelines/hardcoded/hh_staged_noiseless.py \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-pool full_meta \
  --adapt-continuation-mode "$CONT_MODE" \
  --phase3-lifetime-cost-mode phase3_v1 \
  --skip-pdf \
  --output-json artifacts/json/campaign_A4a_L2_lifetime_on_${CONT_MODE}.json
```

#### A4b. lifetime off
```bash
python pipelines/hardcoded/hh_staged_noiseless.py \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-pool full_meta \
  --adapt-continuation-mode "$CONT_MODE" \
  --phase3-lifetime-cost-mode off \
  --skip-pdf \
  --output-json artifacts/json/campaign_A4b_L2_lifetime_off_${CONT_MODE}.json
```

### A5. Runtime split ablation

#### A5a. split off
```bash
python pipelines/hardcoded/hh_staged_noiseless.py \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-pool full_meta \
  --adapt-continuation-mode "$CONT_MODE" \
  --phase3-runtime-split-mode off \
  --skip-pdf \
  --output-json artifacts/json/campaign_A5a_L2_split_off_${CONT_MODE}.json
```

#### A5b. split on
```bash
python pipelines/hardcoded/hh_staged_noiseless.py \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-pool full_meta \
  --adapt-continuation-mode "$CONT_MODE" \
  --phase3-runtime-split-mode shortlist_pauli_children_v1 \
  --skip-pdf \
  --output-json artifacts/json/campaign_A5b_L2_split_on_${CONT_MODE}.json
```

### A6. Backend-conditioned parent search

#### A6a. proxy vs fixed-backend shortlist
```bash
python pipelines/hardcoded/hh_adapt_backend_shortlist.py \
  --backend-shortlist ibm_boston,ibm_miami \
  --include-proxy-baseline \
  --run-prefix campaign_A6_L2_backend \
  --L 2 \
  --adapt-pool full_meta \
  --adapt-inner-optimizer POWELL \
  --adapt-continuation-mode "$CONT_MODE" \
  --skip-pdf
```

### A7. Shortlisting logic study (direct ADAPT surface)

Why this is separate from beam/batching:
- this tests the selector's candidate-admission funnel itself:
  - phase-1 cheap shortlist
  - phase-2 admitted full-score shortlist
- it should be reported as shortlist-quality vs measurement burden, not as generic beam behavior

#### A7a. tight shortlist (winner-takes-all style)
```bash
python pipelines/hardcoded/adapt_pipeline.py \
  --problem hh \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --boson-encoding binary --ordering blocked --boundary open \
  --adapt-pool full_meta \
  --adapt-continuation-mode "$CONT_MODE" \
  --adapt-inner-optimizer POWELL \
  --adapt-reopt-policy windowed \
  --adapt-window-size 999999 \
  --adapt-window-topk 999999 \
  --adapt-full-refit-every 8 \
  --adapt-final-full-refit true \
  --phase1-shortlist-size 1 \
  --phase2-shortlist-fraction 1.0 \
  --phase2-shortlist-size 1 \
  --phase3-runtime-split-mode shortlist_pauli_children_v1 \
  --phase3-lifetime-cost-mode phase3_v1 \
  --phase3-enable-rescue \
  --skip-pdf \
  --output-json artifacts/json/campaign_A7a_L2_shortlist_tight_${CONT_MODE}.json
```

#### A7b. default shortlist
```bash
python pipelines/hardcoded/adapt_pipeline.py \
  --problem hh \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --boson-encoding binary --ordering blocked --boundary open \
  --adapt-pool full_meta \
  --adapt-continuation-mode "$CONT_MODE" \
  --adapt-inner-optimizer POWELL \
  --adapt-reopt-policy windowed \
  --adapt-window-size 999999 \
  --adapt-window-topk 999999 \
  --adapt-full-refit-every 8 \
  --adapt-final-full-refit true \
  --phase3-runtime-split-mode shortlist_pauli_children_v1 \
  --phase3-lifetime-cost-mode phase3_v1 \
  --phase3-enable-rescue \
  --skip-pdf \
  --output-json artifacts/json/campaign_A7b_L2_shortlist_default_${CONT_MODE}.json
```

#### A7c. wide shortlist
```bash
python pipelines/hardcoded/adapt_pipeline.py \
  --problem hh \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --boson-encoding binary --ordering blocked --boundary open \
  --adapt-pool full_meta \
  --adapt-continuation-mode "$CONT_MODE" \
  --adapt-inner-optimizer POWELL \
  --adapt-reopt-policy windowed \
  --adapt-window-size 999999 \
  --adapt-window-topk 999999 \
  --adapt-full-refit-every 8 \
  --adapt-final-full-refit true \
  --phase1-shortlist-size 256 \
  --phase2-shortlist-fraction 1.0 \
  --phase2-shortlist-size 128 \
  --phase3-runtime-split-mode shortlist_pauli_children_v1 \
  --phase3-lifetime-cost-mode phase3_v1 \
  --phase3-enable-rescue \
  --skip-pdf \
  --output-json artifacts/json/campaign_A7c_L2_shortlist_wide_${CONT_MODE}.json
```

## Lane B — Downstream executable honesty lane

Do **after** Lane A winners are identified.

### B1. Executable replay fidelity gate
- confirm candidate is an honest executable import before any live/backend claim
- if replay fidelity is not near 1, exclude from executable Pareto claims

### B2. Energy-only executable burden check
- use narrow energy-only surfaces
- keep logical Pareto and executable Pareto in separate tables

## First batch to launch first

If execution is approved after continuation-mode choice, start with:
1. A1a / A1b
2. A3a / A3b / A3c
3. A4a / A4b

Reason:
- highest information per local wall-clock
- all in staged wrapper
- directly answer novelty-vs-implementation questions without mixing in downstream executable issues

## Deferred / supporting controls
Run later unless a headline run is inconclusive:
- probe intensity: `--phase1-probe-max-positions 1` vs default
- batch/no-batch via direct `adapt_pipeline.py`
- beam/no-beam historical-style control
- seed-refine on/off
- pruning-only descendant studies

## Blocking note
Execution is intentionally paused pending the required continuation-mode choice because README/AGENTS and current staged CLI default diverge.

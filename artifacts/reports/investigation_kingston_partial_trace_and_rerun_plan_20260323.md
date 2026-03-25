# Investigation: Kingston partial trace + fixed-scaffold rerun plan

## Summary
- Partial recovery of the interrupted Kingston 6-term SPSA screen shows the best observed recovered job energy reached **0.3935622931602998** at trace row **3**.
- The next matched decision step is **select_backend_then_anchor_7term_energy_only_fixedtheta_baseline** because the recovered Kingston evidence and the existing candidate fixed-theta baseline are still on different backends.
- Existing anchor evidence continues to indicate the dominant residual is **gate_stateprep**, not readout.

## Partial Kingston recovery
- Candidate: `artifacts/json/hh_marrakesh_fixed_scaffold_6term_drop_eyezee_20260323T171528Z.json`
- Recovery granularity: `runtime_job`
- Trace rows recovered: 6 / 6
- Objective calls proven: not proven from job-only recovery
- Runtime jobs recovered: 6 / 6
- Total recovered quantum seconds: 75.0
- Best-so-far trace row: 3
- Best-so-far noisy energy: 0.3935622931602998

## Rerun plan
- Target backend inferred from recovered jobs: `ibm_kingston`
- Candidate omitted runtime term(s): ['eyezee']
- Candidate screen budget: 15 runtime jobs
- Anchor screen budget: 17 runtime jobs
- Backend selection required: True
- Recommended next submission: `select_backend_then_anchor_7term_energy_only_fixedtheta_baseline`
- Evidence gaps: ['anchor_energy_only_runtime_baseline_missing', 'candidate_runtime_backend_mismatch']

## Noise evidence
- Attribution artifact: `artifacts/json/hh_gate_pruned_fixed_scaffold_attr_20260323T013814Z.json`
- Saved-parameter eval: `artifacts/json/hh_gate_pruned_savedparam_eval_20260323T011240Z.json`
- Full delta mean: 0.17849474598433385
- Readout-only delta mean: 0.04855089832808385
- Gate/state-prep-only delta mean: 0.11921435047652135
- M3 residual delta mean: 0.11868750301318737
- Readout not primary limit: True

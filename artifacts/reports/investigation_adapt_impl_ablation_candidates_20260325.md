# Investigation: ADAPT/HH implementation-ablation candidates

## Summary
Initial hypothesis: beyond the already-ablated novelty score, the likely ablation-worthy contributions are cost-aware selector scoring, curated HH pool design, probe/insertion logic, beam/batch behavior, runtime split, and downstream pruning. Measurement/shot cost appears to be the primary reporting axis.

## Symptoms
- User wants implementation-level variables isolated beyond the already-tested novelty factor.
- User specifically wants batch vs no batch, beam vs no beam, probe vs no probe, and other contributions assessed against shot/measurement cost.
- No code changes requested.

## Investigation Log

### Phase 1 - Initial assessment
**Hypothesis:** The repo already contains several partially-isolated ablations and code-level toggles that can support at least 6 strong new runs.
**Findings:** Context builder indicates strong prior evidence for novelty ON/OFF, beam ON/OFF, seed-refine comparisons, lean-pool vs heavy-pool, and pruning. Direct evidence for pure batch/no-batch and pure probe/no-probe appears thinner.
**Evidence:** Pending exact file-line verification.
**Conclusion:** Confirmed broadly; now collecting precise evidence and literature context.

## Root Cause
There is no single failure root cause here. The investigation instead shows a prioritization problem: some implementation toggles are probably headline research contributions, while others are mainly controls. Repo evidence points to the strongest contributions being (1) HH-specific pool/curriculum design and (2) integrated cost-aware continuation scoring. In contrast, beam/no-beam, batching, and probe/insertion are important controls, but the literature already covers multi-operator adaptive selection, trough/insertion strategies, and measurement-/hardware-aware ADAPT variants.

Key evidence:
- `README.md:25-35` defines the intended HH staged contract as narrow core first, then `phase3_v1`, then optional residual/runtime-split follow-ons.
- `pipelines/hardcoded/adapt_pipeline.py:875-1020` implements both the heavy `full_meta` HH pool and the curated `pareto_lean` pool.
- `artifacts/reports/pareto_lean_l2_vs_heavy_L2_ecut1_comparison.md:1-158` shows the 11-op lean pool matching the 46-op heavy pool at essentially the same energy while preserving low honest cost.
- `pipelines/hardcoded/hh_continuation_scoring.py:141-193,313-330` shows measurement-aware and compile-aware selector scoring, but also makes clear that current measurement reuse is accounting-only, not persistent measured-data reuse.
- `pipelines/hardcoded/adapt_pipeline.py:3338-3527,3950-4109` shows probe-position logic, shortlist/full scoring, batching, and runtime-split-aware admission in the live selector path.
- `artifacts/reports/pi_novelty_ablation_briefing_20260324.md:1-88` already settles the novelty-factor ablation, so it should not dominate the next campaign.

## Recommendations
Second-pass conclusion: split the campaign into two explicit lanes.

### Lane A — Upstream selector lane (main science / novelty lane)
Use this lane to establish research novelty. The object of study is the parent scaffold built inside ADAPT. Primary axis: cumulative measurement burden at fixed error; tie-break: cumulative compile-gate proxy.

Headline six comparisons for Lane A:
1. staged narrow-core-first curriculum vs depth-0 `full_meta`
2. `full_meta` vs `pareto_lean` / `pareto_lean_l2`, with L=3 included because `pareto_lean` already shows same-order accuracy there
3. reoptimization policy triad: `append_only` vs `windowed` vs `full`
4. `phase3_lifetime_cost_mode=off` vs `phase3_v1`
5. proxy vs backend-conditioned parent search (`transpile_single_v1` / shortlist wrapper)
6. `phase3_runtime_split_mode=off` vs `shortlist_pauli_children_v1`

### Lane B — Downstream executable lane (honesty / deployment lane)
Use this lane to separate logical/design claims from executable-import claims. The object of study is exported descendants / fixed-scaffold executables, not upstream selector novelty. Primary axis: executable measurement burden or energy-only evaluation burden, with replay fidelity and backend-compiled 2Q burden as secondary axes.

Beam/probe/batching, pruning-only, and seed-refine/replay should remain supporting controls unless a later matched campaign shows they dominate the upstream selector story.

For reporting, primary axis should be measurement burden first, then final error, then compiled cost. Beam should be treated as a control, not a headline contribution, because existing artifacts already show it can improve error slightly while sharply increasing selector measurement burden.

A subtle remaining question is interaction rather than main effects: pool/curriculum claims may still be entangled with reoptimization policy. If one more targeted interaction check is needed after the headline six, prioritize a reopt-policy × pool/curriculum cross-ablation.

### New shortlist-specific evidence (2026-03-25)
A direct shortlist-width ablation on `adapt_pipeline.py` now suggests that shortlist breadth is not just an efficiency tweak; it materially changes whether the good scaffold is reachable under matched `phase3_v1` continuation.

Matched local runs (HH, L=2, `t=1.0 U=4.0 g_ep=0.5 omega0=1.0 n_ph_max=1`, `full_meta`, `windowed` reopt, `shortlist_pauli_children_v1`, `POWELL`) produced:
- tight shortlist (`phase1_shortlist_size=1`, `phase2_shortlist_fraction=1.0`, `phase2_shortlist_size=1`): `artifacts/json/campaign_A7a_L2_shortlist_tight_phase3_v1.json` → final `|ΔE| = 3.282923125e-01`, ansatz depth 14, stop `drop_plateau`
- default shortlist (`64`, `0.2`, `12`): `artifacts/json/campaign_A7b_L2_shortlist_default_phase3_v1.json` → final `|ΔE| = 3.283230633e-01`, ansatz depth 14, stop `drop_plateau`
- wide shortlist (`256`, `1.0`, `128`): `artifacts/json/campaign_A7c_L2_shortlist_wide_phase3_v1.json` → final `|ΔE| = 5.617823464e-05`, ansatz depth 14, stop `drop_plateau`

Interpretation: shortlist width appears to alter the selector trajectory enough to separate a bad scaffold basin from the good basin, even at the same final ansatz depth. This upgrades shortlisting from a minor control to a real candidate contribution.

Follow-on control now run: with runtime split disabled, the default shortlist remained bad (`artifacts/json/campaign_A7d_L2_shortlist_default_runtime_split_off_phase3_v1.json` → `|ΔE| = 3.283230633e-01`) while the wide shortlist remained good (`artifacts/json/campaign_A7e_L2_shortlist_wide_runtime_split_off_phase3_v1.json` → `|ΔE| = 5.618140840e-05`). This suggests the shortlist effect is not merely an artifact of `shortlist_pauli_children_v1`; shortlist breadth itself is changing which basin the selector reaches.

## Preventive Measures
- For future reports, separate headline contribution claims from control-variable claims.
- For every new ablation, report both selector-stage measurement burden and final replay/readout burden.
- Be explicit that current measurement reuse is accounting/scoring reuse unless live measured-data reuse is actually implemented.
- Set `--adapt-continuation-mode phase3_v1` explicitly in future staged HH runs because `README.md` describes `phase3_v1` as canonical while `pipelines/hardcoded/hh_staged_cli_args.py:52-59` currently defaults to `phase1_v1`.

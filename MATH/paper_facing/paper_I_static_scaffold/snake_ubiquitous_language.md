# SNAKE Ubiquitous Language

Created: 2026-05-25  
Scope: Paper-I SNAKE/static ADAPT support language.  
Authority: `MATH/Math.md`, `MATH/paper_details/static_adapt_paper_I.tex`, and executable policy fields in `pipelines/static_adapt/`.  
Purpose: map repo prose and Optuna/runtime field names to Unicode mathematical language before writing manuscript prose, run notes, or PDF provenance.

## Rules

- Use this file for support docs, provenance notes, and run-analysis language. Translate again before final journal prose if a phrase is too repo-native.
- Do not collapse separate mechanisms merely because their field names share `phase1`, `phase2`, or `phase3` prefixes.
- Distinguish three layers:
  1. **Mathematical SNAKE mechanism**: the object defined in `Math.md` or `static_adapt_paper_I.tex`.
  2. **Repo/Optuna knob**: the field that enables, disables, or tunes the mechanism.
  3. **Run-specific policy value**: what a particular JSON actually used.
- If a run disables a feature that the manuscript describes as part of the SNAKE feature surface, say so explicitly. Do not imply that every displayed row used every available feature.

## Core record and phase-score language

**Repo prose / knobs:** `candidate record`, `candidate-position record`, `phase score`, `phase0`, `phase1`, `phase2`, `phase3`, `SNAKE selector`.  
**Math language:** record `r=(m,p)`; generator `m`; insertion position `p`; candidate record family `𝓡ₖ(t)`; shortlist `𝓢ₖ(t)`; phase score `Sₖ(r;t)`.

Canonical support-language sentence:

> SNAKE scores candidate-position records `r=(m,p)`, not bare generators, through a staged chain `𝓡₀→𝓡₁→𝓡₂→𝓡₃` and corresponding shortlists `𝓢₁,𝓢₂,𝓢₃`.

Score form:

```text
Sₖ(r;t) = ΔEₖ(r;t) · 𝒩ₖ(r;t) / Kₖ(r;t)
```

where:

- `ΔEₖ(r;t)` is the phase-`k` resolved energy-gain proxy.
- `𝒩ₖ(r;t)` is the tangent/reduced-window novelty factor.
- `Kₖ(r;t)` is the resource-cost burden, not a geometric object.
- `𝒩₀ = 𝒩₁ = 1` for the pilot and Phase-I score.

Reader-facing translation:

> The selector ranks candidate-position records by conservative energy gain, tangent novelty when applicable, and explicit resource burden.

## Novelty ablation mode

**Repo prose / knobs:** `phase3_novelty_ablation_mode`, values `off`, `no_phase2`, `no_phase3`, `all`; sometimes described as `novelty ablation`.  
**Math language:** disables `𝒩₂`, `𝒩₃`, or both in `S₂`/`S₃`.

Use this wording nearly verbatim in support docs:

```text
Phase score form: Sₖ(r;t) = ΔEₖ(r;t) · 𝒩ₖ(r;t) / Kₖ(r;t)
For early phases: 𝒩₀ = 𝒩₁ = 1
Phase 2 novelty: 𝒩₂(r;t) ≈ 1 − qᵀQ⁺q/Fᵣ
Phase 3 reduced-window novelty: 𝒩₃(r;t) ≈ 1 − q*ᵀQ⁺q*/F*ᵣ
```

Meaning:

- `off`: use both `𝒩₂` and `𝒩₃`.
- `no_phase2`: set or bypass Phase-II novelty, so `𝒩₂` does not affect Phase-II ranking.
- `no_phase3`: set or bypass Phase-III novelty, so `𝒩₃` does not affect Phase-III ranking.
- `all`: ablate both tangent-novelty factors.

This is **not pruning**. Novelty ablation changes candidate-record scoring. Generator ablation/pruning acts later on already-admitted ansatz entries.

## Phase 0 pilot and algebraic lanes

**Repo prose / knobs:** `phase0_pilot_enabled`, `phase0_pilot_alpha`, `phase0_pilot_threshold`, `phase0_pilot_max_records`, `phase0_algebraic_lane_mode`, `phase0_lane_quota_pressure`.  
**Math language:** optimistic pilot gain `ΔE₀(r;t)=α₀G₀ᵘᵖᵖᵉʳ(r;t)` with `𝒩₀=1`, lane-wise filtering by algebraic relation class `ℓ∈𝔏`.

Support-language sentence:

> Phase 0 is a cheap optimistic gradient-cost pilot that protects algebraic lane diversity before expensive tangent geometry is evaluated.

Recommended route rule:

- For production SNAKE, Phase 0 should be treated as part of the canonical selector surface unless a run is explicitly labelled an ablation.
- Optuna may tune pilot amplitude, threshold, maximum records, and lane pressure.
- Whether Optuna may disable the pilot via `feature_phase0_pilot_enabled` should be a profile-level decision, not an accidental default.

## Phase 1 score versus Phase-1 position-probe cap

**Repo prose / knobs:** `phase1_probe_max_positions`, `phase1_trough_margin_ratio`, `phase1_plateau_patience`, `phase1_shortlist_size`; sometimes incorrectly shortened to `Phase 1 cap`.  
**Math language:** Phase-I trust-region score `S₁`; position-domain opening parameter `τ_pos(t)`; insertion-position cap `M_probe`; append-distance telemetry `d_app`.

Paper-I TeX definition:

```text
S₁(r;t) = ΔE₁(r;t) / K₁(r;t)     because 𝒩₁=1
ΔE₁(r;t) = max_|α|≤ρ/√Fᵣ [ g̃ᵣ(t)|α| − λFᵣα²/2 ]
```

Current source-of-truth interpretation:

```text
d_app(r;t) = |p−p_app(t)| / max(1,|𝒪_t|)
𝓟_m¹(t) ⊆ W_act(t),     |𝓟_m¹(t)| ≤ M_probe
S₁,rank(r;t) = S₁,geom(r;t)
```

Interpretation:

- `τ_pos(t)` and `d_app(r;t)` belong to position-domain construction: they describe how broadly the eligible/probed insertion positions may open around append and active-window positions before Phase-I quantities are computed.
- `phase1_probe_max_positions = M_probe` caps how many insertion positions are probed/retained. It is not the Phase-I score itself.
- The paper emphasizes the admissible insertion window `W(θ;t)`, the active probe positions, and lane-wise shortlisting. It should not present an explicit `M₁,pos` score multiplier as canonical production scoring.

Recommended resolution:

> Present insertion bias as pre-score set construction: `τ_pos(t)`, `W(θ;t)` / `W_act(t)`, lane exposure, and `M_probe` decide which candidate-position records are evaluated. Once a record is admitted to `𝓡₁(t)`, Phase-I ranking is `S₁ = ΔE₁/K₁` with `𝒩₁=1`, not a post-score append-vs-insertion multiplier.

Implementation note:

> I found executable support for `phase1_probe_max_positions = M_probe`, `phase1_trough_margin_ratio` as a probe-trigger/trough detector, and a compiled-resource component `position_shift_span` inside `Kₖ`; I did not find a direct executable `M₁,pos` score multiplier. That supports treating `M₁,pos` as retired/non-live wording, not as a feature we should claim in the PDF.

## Frontier ratios and shortlisting bias

**Repo prose / knobs:** `phase2_frontier_ratio`, `phase3_frontier_ratio`, near-frontier, crowded frontier, shortlist pressure.  
**Math language:** leading-score frontier ratio and near-best admission threshold.

Two related quantities must not be confused:

1. **Frontier flatness statistic** in `Math.md`:

```text
u_front(t) = (s₂ + ε)/(s₁ + ε)
```

where `s₁≥s₂` are the two largest raw scores. Larger `u_front` means a flatter, more crowded frontier.

2. **Repo shortlist frontier ratio** used as a near-best cutoff:

```text
keep r when Sₖ(r;t) ≥ η_front,k · Sₖ,best(t)
```

where `η_front,k` is represented by knobs such as `phase2_frontier_ratio` and `phase3_frontier_ratio`.

Support-language sentence:

> Frontier ratios control how tightly Phase-II/III shortlists hug the leading score: ratios near one keep only near-tied leaders, while lower ratios admit a broader frontier for later geometry, batching, or beam continuation.

## Live/null phase gates and retirement

**Repo prose / knobs:** `phase_live_hysteresis_enabled`, `phase2_null_nrem_high_threshold`, `phase2_live_nrem_low_threshold`, `phase3_null_nrem_high_threshold`, `phase3_live_nrem_low_threshold`, `phase2_hysteresis_steps`, `phase3_hysteresis_steps`.  
**Math language:** phase-live gate `Γₚˡⁱᵛᵉ(t)` for `p∈{2,3}` driven by useful-runway telemetry `N̂_rem(t)`.

Math form:

```text
Γₚˡⁱᵛᵉ(t⁺)=0 if N̂_rem^high(t) ≤ νₚ⁻ for hₚ consecutive steps
Γₚˡⁱᵛᵉ(t⁺)=1 if N̂_rem^low(t)  > νₚ⁺
Γₚˡⁱᵛᵉ(t⁺)=Γₚˡⁱᵛᵉ(t⁻) otherwise
```

with Phase 3 forced off if Phase 2 is off:

```text
Γ₃ˡⁱᵛᵉ(t) ← Γ₂ˡⁱᵛᵉ(t)Γ₃ˡⁱᵛᵉ(t)
```

Meaning:

- These are phase-retirement gates.
- They do not ask whether an individual candidate score is above threshold.
- They decide whether expensive Phase 2/3 scoring remains worth running based on the estimated useful remaining admissions.
- Hysteresis prevents one noisy step from flickering a phase on/off.

## Null/live thresholds

**Repo prose / knobs:** `*_null_nrem_high_threshold`, `*_live_nrem_low_threshold`; occasionally misheard as `Nolan live thresholds`.  
**Math language:** retirement/reactivation thresholds `νₚ⁻`, `νₚ⁺` applied to `N̂_rem^high(t)` and `N̂_rem^low(t)`.

Do not describe these as raw score cutoffs. They are thresholds on useful-runway telemetry, not on `Sₖ(r;t)`.

Support-language sentence:

> Null/live thresholds compare pessimistic and optimistic useful-runway estimates against phase-retirement thresholds `νₚ⁻,νₚ⁺`; they decide whether Phase 2 or Phase 3 remains live, not whether a particular candidate record passes a score threshold.

## Resource burden: compile and measure weights

**Repo prose / knobs:** `lambda_compile`, `lambda_measure`, `compile_cx_weight`, `compile_sq_weight`, `compile_rotation_step_weight`, `compile_position_shift_weight`, `compile_refit_active_weight`, `measure_groups_weight`, `measure_shots_weight`, `measure_reuse_weight`, `phase2_w_depth`, `phase2_w_group`, `phase2_w_shot`, `phase2_w_optdim`, `phase2_w_reuse`.  
**Math language:** resource-cost denominator `Kₖ(r;t)`.

Do not call these “geometric scoring cost.” The correct phrase is **resource-cost burden**.

Support-language sentence:

> Geometry enters through `ΔEₖ` and tangent novelty `𝒩ₖ`; compile and measurement terms enter through the burden denominator `Kₖ`.

Paper-I-compatible cost form:

```text
Kₖ(r;t) = 1 + λ₂q C̄₂q(r;k) + λ_d C̄_d(r;k) + λ_θ C̄_θ(r;k) + λ_shot C̄_shot(r;k)
```

`Math.md` also tracks robust excess features and structural burden terms. Treat those as implementation/support-detail unless the manuscript explicitly defines them.

## Reduced plane and Phase-3 batching

**Repo prose / knobs:** `phase3_batch_selection_mode=reduced_plane`, `phase2_enable_batching`, `phase2_batch_target_size`, `phase2_batch_size_cap`, `phase2_batch_near_degenerate_ratio`, `phase2_batch_rank_rel_tol`, `phase2_batch_additivity_tol`, `phase3_batch_prefilter_mode`.  
**Math language:** reduced batch plane after common inherited-window relaxation; in Paper-I TeX this is the batch reduced-plane geometry around `G_B*` / `G_B^⋆` and the expected batch-gain equations.

The reduced-plane object is the batch analogue of Phase-III Schur relaxation. For batch `B`, eliminate/refit the inherited window `W_B` before judging candidate compatibility:

```text
uᵣ^(B)(t) = dᵣ(t) − T_B(t) M_B(t)⁻¹ bᵣ^(B)(t)
Π_B(t) = span{uᵣ^(B)(t): r∈B}
(G_B)ᵣₛ = Re⟨uᵣ^(B),uₛ^(B)⟩
```

The batch is accepted by reduced geometry, not just by TETRIS-style disjointness:

```text
ΔE_B^TR(t) = max_{αᵀG_Bα≤ρ_B²} [ g_B,lcb(t)ᵀα − ½αᵀ(H̃_B)⁺α ]
δ_add(B;t) = [1 − ΔE_B^TR(t)/(Σ_{r∈B}ΔE_{r|B}^TR(t)+ε)]₊
```

with gates such as:

```text
λ_min(G_B) ≥ τ_rank · tr(G_B)/|B|
δ_add(B;t) ≤ τ_add
```

Answer to the equation-35 question:

> Yes: “reduced plane” refers to the same mathematical object as the Paper-I batch reduced-plane geometry around the reduced Gram `G_B^⋆` / `G_B` and expected batch gain. It is the residual tangent plane after inherited-window relaxation, not merely support-disjoint batching.

## Single admission versus ordered-batch admission

**Repo prose / knobs:** `beam_structural_mode`, `single_admission`, `stop_or_single_admission`, `ordered_batch_admission`, `batch_selection_mode`, `phase2_batch_selection_mode`, `phase3_batch_selection_mode`, `phase2_batch_target_size`, `phase2_batch_size_cap`, `adapt_beam_live_branches`, `adapt_beam_children_per_parent`, `beam_cost_K`, `survival_policy_version`, `prune_key_version`.  
**Math language:** singleton candidate admission versus ordered batch-family admission and beam survival over child continuations.

Do not identify `B_max=1` with the old singleton route. These are different
questions:

1. **How many records are finally admitted?** This is reported by
   `batch_size` and bounded by the target/cap knobs.
2. **Which admission path constructs, scores, orders, and survives proposals?**
   This is reported by `beam_structural_mode` and `batch_selection_mode`.

Thus a row with `batch_size=1` can still use the new ordered-batch route if its
history says:

```text
beam_structural_mode = ordered_batch_admission
batch_selection_mode = greedy_reduced_plane or combinatorial_reduced_plane
```

In that case, the run is not a pure reproduction of the old singleton
admission path. Ordered-batch admission can change a singleton trajectory before
any multi-record batch is accepted, because the candidate family is generated,
ranked, tied, and passed to beam survival through the batch-admission machinery.

Concrete `B=1` implication:

> A `B_max=1` run can still differ from the old Paper-I route because it may
> choose the singleton by first building an ordered batch proposal frontier and
> then applying beam survival. The admitted set has size one, but the singleton
> need not be the same singleton the old reduced-plane singleton path would
> have selected.

Observed/new-route wording should therefore mention all three active surfaces:

```text
metric-prune greedy ordered batch        = metric prune + greedy ordered-batch admission
metric-prune combinatorial ordered batch = metric prune + combinatorial ordered-batch admission
cost-weighted beam                       = beam survival/ranking uses beam_cost_K / lambda_beam
```

Do not abbreviate the current route as merely "batching" when diagnosing
regressions. In the current diagnostic line, the route can change through:

1. **metric prune**: deletion/recoverability logic on existing ansatz entries;
2. **ordered-batch admission**: greedy or combinatorial proposal construction
   before final admission;
3. **beam survival**: branch continuation/ranking, possibly cost-weighted.

Use these support names:

| Support name | Required observed evidence | Meaning |
|---|---|---|
| old/singleton admission route | `beam_structural_mode=single_admission` or `stop_or_single_admission`; usually `batch_selection_mode=reduced_plane`; observed `batch_size=1` | One record is selected through the older singleton admission path. |
| metric-prune greedy ordered batch | metric-prune fields active; `beam_structural_mode=ordered_batch_admission`; `batch_selection_mode=greedy_reduced_plane` | New metric-prune route with greedy ordered-batch proposal construction. |
| metric-prune combinatorial ordered batch | metric-prune fields active; `beam_structural_mode=ordered_batch_admission`; `batch_selection_mode=combinatorial_reduced_plane` | New metric-prune route with combinatorial ordered-batch proposal construction. |
| cost-weighted beam | `beam_cost_K` present and the run records a nontrivial beam-cost/survival comparator such as `lambda_beam` or `survival_policy_version` | Beam survival uses a cost-aware branch comparator. If these fields are absent or zero, say ordered-batch beam, not cost-weighted beam. |

Support-language sentence:

> `maxB=1` blocks multi-record admission, but it does not by itself restore the old singleton admission route; verify `beam_structural_mode` before claiming a batch/prune-only ablation.

Audit rule:

- If a trajectory changes before any accepted prune deletion and every observed
  `batch_size` is `1`, check `beam_structural_mode`, `batch_selection_mode`,
  candidate-family counts, and beam survival fields before attributing the
  change to metric prune.
- If the intended experiment is “metric prune only,” hold the admission route
  fixed as `single_admission`/`stop_or_single_admission`.
- If the intended experiment is “new batch only,” hold the prune route fixed
  and vary `beam_structural_mode`/`batch_selection_mode` separately from the
  final batch cap.
- If the intended experiment is “old batch plus new beam,” keep the old
  singleton/reduced-plane admission route and vary only the beam survival
  comparator.
- If the intended experiment is “old beam plus new batch,” keep the old/no-cost
  beam survival comparator and vary only `beam_structural_mode` and
  `batch_selection_mode`.

## Phase-3 budget into batching

**Repo prose / knobs:** `phase2_batch_target_size`, `phase2_batch_size_cap`, `phase3_frontier_ratio`, `phase2_shortlist_fraction`, `phase2_budget`, `phase1_budget`.  
**Math language:** Phase-III shortlist `𝓢₃(t)` plus near-degenerate batch shell `𝓝₃(t)` and batch cap `B_max(t)`.

Current implementation caveat:

- Batch target/cap are wired.
- Phase-II/III frontier ratios are wired.
- A distinct standalone Optuna field named `phase3_budget` or `phase3_candidate_budget` is not wired.
- Effective Phase-III batching input is inherited from the Phase-II/III shortlist/frontier machinery plus batch target/cap.

Support-language sentence:

> The current repo wires batch size and near-frontier gates, but not an independent Phase-III candidate-budget knob; the batch universe is the final Phase-III shortlist after inherited shortlist and frontier controls.

## Phase-1 prune versus Phase-I score

**Repo prose / knobs:** `phase1_prune_enabled`, `phase1_prune_policy`, `phase1_prune_mode`, `phase1_prune_fraction`, `phase1_prune_min_candidates`, `phase1_prune_max_candidates`, `phase1_prune_max_regression`, `phase1_prune_*`.  
**Math language:** generator ablation / rollback-safe deletion acting on existing coordinates `d_j∈𝒪_t`, not on candidate records `r=(m,p)`.

This is not the Phase-I score `S₁`. The name `phase1_prune` is repo-historical: it means the pruning pass is scheduled near the early/static ADAPT selector machinery, but it acts on already-admitted ansatz entries.

Prune pipeline language:

```text
𝓜_t = mature deletion-target universe
𝓝₁(t) = cheap deletion-target nomination set
R_j^cheap(t) = cheap recoverability-prior score
𝓒_trial(t) = capped exact remove-refit trial set
D_j = commit/rollback deletion decision
```

Paper-I TeX language:

```text
Λ_j^⊖ = Schur-predicted deletion loss
ΔE_j^{⊖,refit} = measured post-refit deletion loss
𝓓 = {d_j: history gate, Schur-loss gate, and rollback/refit guard pass}
```

Support-language sentence:

> Phase-1 prune is rollback-safe generator ablation: it nominates stale existing ansatz entries using cheap recoverability evidence, but deletion authority comes only after remove-refit safety checks.

Recommended route rule:

- Production SNAKE should keep rollback-safe pruning available unless the run is explicitly a no-prune ablation.
- Optuna may tune prune pressure, caps, tolerance mode, protected age, stale age, candidate fraction, and amplitude-witness requirements.
- If Optuna is allowed to disable `phase1_prune_enabled`, the resulting run must be labelled as a no-prune policy choice or ablation, not as evidence that pruning was used.

### Prune reporting taxonomy

When reporting SNAKE prune behavior, use two gate-exposure counts followed by
the four-way outcome classification. Do not call a rejected deletion trial a
rollback.

| Quantity | Count when |
|---|---|
| Prune enabled rows | The row/depth records `phase1_prune_enabled` or `post_admission_prune.enabled` as true. |
| Permission-open rows | The prune gate permits a deletion attempt, e.g. `post_admission_prune.permission_open` is true. |
| Eligible deletion trials | The prune gate produces at least one concrete deletion trial candidate. |
| Rejected trials | A deletion trial exists but is not accepted, e.g. `trial.accepted=false`; report `rollback_reason` values here as rejection reasons. |
| Accepted deletions | A deletion trial is accepted and an ansatz entry is removed after the configured safety check. |
| Rollbacks | A previously accepted deletion or ADAPT depth is undone, e.g. `post_admission_prune.rolled_back=true` or `depth_rollback=true`. |

Use this table shape for support reports:

```markdown
| Regime | Prune enabled rows | Permission-open rows | Eligible deletion trials | Rejected trials | Accepted deletions | Rollbacks |
|---|---:|---:|---:|---:|---:|---:|
| ... | ... | ... | ... | ... | ... | ... |
```

Support-language sentence:

> Prune reporting separates gate exposure from the four prune outcomes: eligible deletion proposals, rejected proposals, accepted deletions, and true rollbacks; a failed remove-refit safety trial is a rejected proposal, not a rollback.

If Schur information appears in prune telemetry, state whether it is
nomination/screening evidence or deletion-acceptance authority. For canonical
rollback-safe pruning, Schur-like evidence may nominate or rank stale entries,
but accepted deletion authority comes from the configured remove-refit safety
check, such as `remove_refit_energy_safety`. A field such as
`post_refit_executed=false` means no extra post-prune cleanup/full-refit pass was
run after the prune decision; it does not by itself mean that an accepted
deletion skipped the remove-refit safety trial.

## Phase-3 selector policy

**Repo prose / knobs:** `phase3_selector_policy`, choices `algebraic_nested_v1`, `hardware_resolvable_v1`, `legacy_phase3_v1`.  
**Math language:** which Phase-III scoring/shortlisting surface instantiates the final reduced-window selector.

Support meanings:

- `algebraic_nested_v1`: canonical Paper-I/SNAKE-facing policy. Uses exact algebraic lane metadata with nested refit-window accounting and reduced-window geometry. This is the best match to the paper’s staged algebraic shortlisting plus Phase-III Schur rerank language.
- `hardware_resolvable_v1`: selector variant that emphasizes gradient-resolution/backend-aware scoring. Use for hardware-resolution experiments, not as the default math name.
- `legacy_phase3_v1`: compatibility route for older Phase-III behavior. Use only when reproducing legacy artifacts or diagnosing route drift.

Recommended route rule:

> For Paper-I production SNAKE, `phase3_selector_policy=algebraic_nested_v1` should be mandatory unless the run is explicitly a selector-policy ablation or legacy reproduction.

## Phase-3 selector geometry mode

**Repo prose / knobs:** `phase3_selector_geometry_mode`, choices `reduced`, `proxy_reduced`, `raw_exact`; `phase3_window_relaxation_mode`, choices `reduced`, `no_relaxation`.  
**Math language:** whether Phase III ranks by reduced-window Schur geometry or a raw/unrelaxed/proxy surface.

Support meanings:

- `reduced`: use the reduced-window Schur geometry. This maps to `Fᵣ*`, `hᵣ*`, `qᵣ*`, `𝒩₃`, and the Phase-III score.
- `proxy_reduced`: use a proxy bridge for reduced geometry, currently restricted in code to HH-specific compatibility contexts. This is diagnostic/compatibility language unless promoted by evidence.
- `raw_exact`: select using the raw Phase-II-style score instead of the full reduced-window Phase-III score. This is an ablation/diagnostic setting relative to the Paper-I Schur-rerank claim.
- `phase3_window_relaxation_mode=reduced`: eliminate/refit the inherited window before judging the candidate.
- `phase3_window_relaxation_mode=no_relaxation`: ablate that inherited-window relaxation.

Recommended route rule:

> For Paper-I production SNAKE, `phase3_selector_geometry_mode=reduced` and `phase3_window_relaxation_mode=reduced` should be mandatory unless the run is explicitly a reduced-geometry ablation.

## Optuna feature toggles and route identity

**Repo prose / knobs:** `feature_*`, `static_meta_feature_profile`, `safe_core_v1`, route forcing, canonical policy.  
**Math language:** tunable policy map `φ` over thresholds, budgets, costs, live gates, batching, beam, and pruning controls.

Current Optuna surface includes both continuous/counted knobs and feature toggles. Examples:

- Tuned scalar/counted knobs: budgets, shortlist fractions, frontier ratios, batch target/cap, batch tolerances, cost weights, live/null thresholds, hysteresis steps, prune caps/tolerances, SPSA schedule.
- Tuned categorical/Boolean feature toggles in safe-core profiles: Phase-0 pilot enabled, Phase-3 batching enabled, Phase-1 prune enabled, amplitude witness required, selector policy, novelty ablation mode, window relaxation mode, batch selection mode, batch prefilter mode.
- Canonical route forcing may override some knobs to preserve route identity, e.g. problem-local `full_meta`, collective Phase-II novelty, recoverability-ladder prune policy, and `phase1_prune_mode=both`.

Support-language sentence:

> Optuna tunes a policy surface `φ`; production profiles should declare which mechanisms are mandatory route identity and which are legal ablation/tuning choices.

Recommended production split:

- **Mandatory route identity unless ablation:** Phase 0 pilot, Phase I score, Phase II novelty, Phase III reduced-window Schur geometry, `algebraic_nested_v1`, reduced-window relaxation, rollback-safe pruning availability.
- **Optuna-tunable within route identity:** thresholds, budgets, frontier ratios, cost weights, batch caps/tolerances, live/null thresholds, hysteresis, prune pressure/caps/tolerances, optimizer schedule.
- **Explicit ablation-only toggles:** disabling pruning, disabling Phase-II/III novelty, disabling reduced-window relaxation, using raw/legacy selector geometry, disabling batching when claiming a no-batch control.

## Current evidence caveat for Paper-I rows

The current three-Hamiltonian PDF/table analysis must not say that every SNAKE row used every mechanism above. Use row-specific policy JSONs.

Known caveat to carry forward:

- Existing spin-boson and HH visible policies are old-cutoff/default-like evidence surfaces and may show `(n_ph_work,n_ph_ref)=(2,4)`-style provenance.
- Paper-facing cutoff pairs and clean target evidence must be regenerated or explicitly justified before final claims.
- If a row has `phase2_enable_batching=false`, describe batching as part of the SNAKE feature surface, not as an active mechanism in that row.

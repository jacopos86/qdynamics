# Paper-I Final Correctness and Evidence-Completion Plan

> Correction, 2026-07-29: the planned raw `FullCommutationInsertion` surface
> was invalid and is superseded. Always-open insertion is now typed as
> `AlwaysCommutationReducedInsertion`, executes
> `full_commutation_reduced`, and retains only deterministic earliest
> representatives of exact commutation-equivalence classes. Raw `full`
> profiles, the former capped-domain CLI spelling `always`, and the invalid
> always-v1 package are not executable evidence.

Date: 2026-07-28 (task id 20260727).
Author role: read-only scientific auditor and planner (Claude Fable).
Status: audit + implementation-ready plan. Nothing here launches, submits,
edits, promotes, or demotes anything. CHTC submission is not authorized by
this plan; it requires explicit user authorization after review (§9).

Verified inputs (hashes recomputed this audit):

- `archive/paper_i_static_adapt_legacy_20260727/MANIFEST.json` —
  `f4268ac7e588b1ceec635daf032dfaae2712d73b82b398f4e9019f8fc1159f25` (matches
  declaration).
- `chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/ra_adapt_unification_post_refactor_v5/final_materialization_receipt.json`
  — `10c6fb37c540efbea5ece2a42e638a38106843710409867fe1644300a71fcda8`
  (matches; 58 validated cells per bundle, 116/116 loader checks,
  `execution_authorized=false`, `submitted=false`).
- ICM chain `lock -> refactor -> verify -> materialize-bundles -> analyze ->
  user-review` is internally hash-linked and ends `pending_user_decision`
  with no winner selected and no execution performed.

Claim-status vocabulary: **[FACT]** verified in source/artifact at the cited
anchor; **[SETTLED]** fixed by the governing handoff's settled decisions;
**[FINDING]** audit result; **[PLAN]** proposed action for the later
execution agent.

---

## 1. Executive scientific disposition

The refactored RA-ADAPT/Append-ADAPT implementation is in materially good
shape: the repaired geometry (exact ordered-insertion chart), projected
generalized Phase-III solve, source-Gram no-overlap trust, shared pools
(102/148 macro; 123/171 parent ancestry), closed `S_alg` accounting, and the
typed bundle machinery all exist, are tested, and are receipt-verified. The
two 58-cell Study-1 bundles are validated and unsubmitted.

**The single blocking correctness defect is in the Append-ADAPT comparator:
the executable `run_append_adapt` performs every accepted refit in RA's
supported-FS-whitened chart and charges the chart's gradient/metric
construction into Append's estimator ledger.** This contradicts the settled
comparator convention (Append refits are ordinary unwhitened ansatz-coordinate
Powell) and the historical comparator behavior, in both trajectory and
accounting. No Append cell may execute before this is repaired and the
affected protocols are rematerialized (§3.2-F1, §7 step 2).

Minimum path to a correct final paper:

1. Repair the Append accepted-refit convention (and its ledger charges);
   fix the always-insertion request-fidelity defect; add singleton-matrix
   materialization support (§3, §6). These are ordinary plumbing/protocol
   repairs within the refactor's own contracts.
2. Rematerialize the affected protocols; run local semantic smokes (§7
   step 3).
3. Execute the user-locked minimal Study-1 discriminator (two regimes at
   `nph=3`: `strong_weak_u8`, `strong_strong_u8`; §5.3) on CHTC after
   explicit authorization.
4. User selects the stationarity policy.
5. Materialize and execute the mandatory 48-cell core: 6 regimes × 2
   representations × 4 trajectories at regime-appropriate cutoffs
   (weak-Holstein 3, strong-Holstein 7) **[SETTLED]**. The currently
   materialized bundles cover only the macro half (at both cutoffs); the
   singleton half does not exist yet and its materializer support is a
   required plumbing addition (§3.2-F3).
6. Validate, aggregate neutrally, and hand the user the evidence-replacement
   and manuscript-synchronization decisions.

Core: the 48-cell matrix plus the Study-1 discriminator. Conditional:
Study 2 (late versus all-phase weighting) — the manuscript and the displayed
convention already commit to late weighting (Phase I unweighted,
`Paper_I.tex:1067-1069`), so Study 2 is needed only if the user wants to
change the final method definition (§9). Reuse-audit only (no new runs by
default): the later 12-cell Geo campaign, and the six retrieved
`append_projected_singleton` r50 archives (§4). Deferred: noise, pruning,
batching, beam, other-model appendix replacements, `L=3`, `L=4` (§5.7).

Major risks, ranked: (i) executing Append cells before the whitening repair
(invalidates the fairness core); (ii) selecting the stationarity winner from
under-validated discriminator cells; (iii) treating the 24 cross-cutoff macro
cells as claim-facing (they are not; §5.4); (iv) reusing historical singleton
Append rows without the matched-protocol audit (§4-E3); (v) manuscript
equations (stationary-source) versus displayed/replacement evidence policy
mismatch until the Study-1 decision resolves it (§3.1-M1).

---

## 2. Claim-to-evidence and adverse-interpretation matrix

Central reader-facing claims of the active `MATH/paper_details/Paper_I.tex`
(commit `a42de3b` line anchors). "AI" = strongest adverse interpretation.

### C1. Abstract singleton aggregate (l.110)

- Claim: matched single-Pauli-word pools; RA ≥1 order lower same-cutoff error
  in 5/6 regimes; at Append's lowest pre-plateau error RA reduces
  ΣN2q/D2q/Dc/S by 366 (37%), 364 (46%), 948 (27%), 2,764,244 (83%).
- Visible target: Fig. `fig:hh_singleton_iteration50_insertion` (l.1096) +
  prose l.1135-1140. Source:
  `MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723/…singleton_insertion_policy_overlay_plot.png`
  and its provenance tracker.
- Status: **provisional by its own marker** (`PROVISIONAL_EVIDENCE(20260727)`,
  l.1079-1081).
- AI: "The RA and Append rows were produced under different active-gradient
  policies, different reporting windows, and an Append baseline whose
  protocol was never source-matched to RA's; the aggregate is an artifact of
  mismatched protocols."
- Finding/remedy: replacement mandated **[SETTLED]** (full source-matched
  singleton matrix). Cells: §5.4 singleton half. Gate: G1-G13 (§8).
- User decision: evidence replacement + new aggregate numbers (post-run).

### C2. Abstract macro aggregate (l.110)

- Claim: matched undecomposed pools; ≥1 order in 3/6; reductions 3,212 (63%),
  2,614 (60%), 10,156 (50%), 16,790 (30%).
- Visible target: Fig. `fig:hh_macro_iteration50_insertion` (l.1088) + prose
  l.1126-1133.
- Status: provisional and **confounded** — two confirmed protocol defects in
  the displayed macro evidence:
  1. **[FACT]** interior insertion scored/admitted with append-chart
     derivatives: lock receipt
     `agent_guidance/static-adapt/ra-adapt-lock-receipt-20260727.json`
     (`displayed_evidence_checks.interior_insertion_with_append_chart.answer=true`;
     accepted record `strong_weak_u8` history index 12, `selected_position=1`
     at pre-admission depth 12, scored-record
     `coordinate_chart=append_candidate_after_current_ansatz_v1`).
  2. **[FACT]** pool mismatch: displayed macro RA used 102/148 while
     displayed macro Append used 123/171 (implementation-spec §1.3-1.4;
     rerun-handoff decision).
- AI: "The macro insertion advantage is an artifact of derivatives belonging
  to a different circuit than the one refitted, and the Append baseline
  searched a different pool."
- Remedy: macro replacement runs (all four trajectories, common 102/148 pool,
  exact ordered chart at every recorded position) **[SETTLED]**; §5.4 macro
  half. Gate: G5 interior-position receipt requirement.

### C3. Insertion-policy conclusions (l.1102-1124)

- Claim: macro — always-insertion strongest; singleton — plateau attains high
  accuracy without always-insertion's extra `S`; no-insertion is the lower-`S`
  alternative; strong-weak is the parity exception.
- AI: "Policy rankings derive from confounded macro geometry and are
  policy-mixed (displayed rows used measured-residual response while the
  manuscript's Phase-III model is stationary-source)."
- Remedy: the three policies are reported separately in the replacement
  matrix; no best-of-envelope headline **[SETTLED]**. Rankings re-derived
  from the 48-cell core only after the stationarity choice.

### C4. Phase-III model equations (l.307-360, `eq:local_quadratic`)

- Claim: the Phase-III expansion sets active-coordinate first derivatives to
  zero "under the stationary-source model" (l.329) with `g=(g_α, 0)`
  (l.354-359).
- **[FINDING M1]** The displayed trajectories were produced with the measured
  residual active gradient (protocol-alignment note, "Hessian and active
  gradient"); the manuscript equations therefore do not describe the evidence
  currently shown. No stationarity winner exists yet **[FACT]**
  (`user-review.json: stationarity_winner=null`).
- AI: "The paper's central selection model is not the model that produced its
  results."
- Remedy: Study 1 resolves the policy; then either the equations stand (if
  stationary-source wins — replacement evidence will match them) or the
  equations must be revised (user manuscript decision). Until then this is a
  known, deliberately staged inconsistency; do not patch it piecemeal.

### C5. RA-versus-Append fairness description (l.986, l.1050, config table l.2009-2041)

- Claim: same inner-optimization method (Powell, 200), same horizon (50),
  same seed (7), same encodings; RA's whitened accepted-refit chart is
  described as an RA feature (abstract l.110; table row "Admission and
  whitened refit").
- **[FINDING M2]** The manuscript nowhere states the comparator's
  accepted-refit coordinate convention. The settled convention (RA whitened /
  Append unwhitened, an intentional method difference) should be stated
  explicitly post-repair; more urgently, the current *implementation* gives
  Append the RA whitened chart (§3.2-F1), so today code, manuscript, and
  settled convention are three different things.
- AI: "Fairness is asserted but the refit conventions are undocumented and
  (in the current code) silently unequal in RA's favor via shared whitening
  plus inflated Append ledger charges."
- Remedy: F1 repair + rematerialization; post-run manuscript sentence (user
  decision).

### C6. Estimator-work and resource claims (`S`, l.1016-1027, App. `app:estimator_queries` l.1854)

- Claim: `S` accumulated under the logical scalar-query convention through
  the reported prefix; compiled resources at fixed basis/level-0/seed-7,
  topology-free, reference prep included.
- **[FACT]** Typed protocols pin the identical convention
  (`estimator_accounting_convention: s_alg_equals_n_h_outer_plus_n_h_refit_plus_n_grad_plus_n_metric_v1`;
  `compile_identity: table_i_basis_gate_transpile_v1`, level 0, seed 7, no
  coupling map, reference prep included).
- AI: "Append's `S` includes whitening-chart metric/gradient work it would
  never perform" — true in the current code (§3.2-F1 charges at
  `append.py:1048-1089`); repaired by F1.
- Remedy: F1 + G10 ledger-closure gate per cell; reader-facing `S` recomputed
  from replacement ledgers.

### C7. Reporting-prefix rule (l.1007-1014; table row "Reporting prefix" l.2036)

- Claim: 10%-of-minimum plateau; macro uses the shared window `K_∩`;
  singleton uses each method's own selected-plateau window.
- **[FINDING M9]** The two representations use different reporting rules;
  this is disclosed but asymmetric. Replacement aggregation must re-apply
  exactly these disclosed rules (or the user changes them explicitly);
  the aggregation tooling is `pipelines/reporting/paper_i_run_summary.py`
  (common-accuracy contract).
- AI: "Window rules were chosen per-representation to flatter RA."
- Remedy: reporting-only; re-derive both aggregates from replacement runs
  under the stated rules and record the rule ids in the aggregation receipt.

### C8. Geo-ADAPT claim (l.1116-1117)

- Claim: no-insertion single-Pauli-word RA attains lower error than Geo-ADAPT
  in every regime except weak-weak.
- Evidence: later 12-cell Geo campaign (6 macro + 6 projected-singleton, 50
  rounds). **[FACT]** verified this audit: archives present in the four named
  retrieval dirs; sampled receipts pass
  (`paper_i_hh_geo_completion_validation_receipt_v1`: `status=pass`,
  `same_cutoff_reference=true`, `sector_leak_flag=false`,
  `boson_truncation_leak_flag=false`, `ledger_closure=pass`, 50 iterations;
  e.g. strong-strong projected-singleton error 2.28×10⁻¹ at `nph=7`).
- AI: "The Geo comparison mixes the old compact-report conventions with the
  new accounting."
- Remedy: reuse-audit only (§5.6): recompute the Geo-versus-RA comparison
  from the retrieved archives under the final reporting conventions, against
  the *replacement* singleton no-insertion RA rows. The 20260711 compact
  report is superseded as evidence **[SETTLED]**. If the audit fails, the
  exact affected claim is this sentence; the user then chooses targeted Geo
  rerun versus claim removal/deferment.

### C9. `L=3` appendix (l.2163-2189)

- Claim: three-site intermediate-weak support check, `M=1`, RA-only,
  `k_pl=29`, error 1.689×10⁻⁴, no `S`.
- **[FINDING M7]** The caption states the data are
  "generator-ablation-enabled" (l.2181) while the two-site core states
  ablation is disabled (l.1168) and the appendix prose says only
  "full-ansatz reoptimization and batching disabled" (l.2170-2171). The run
  is therefore not protocol-matched to the two-site core and has no Append
  baseline and no estimator accounting.
- AI: "The scaling claim rests on a single unmatched RA-only run with a
  different mechanism set."
- Remedy: safest final-paper posture is the current one — an explicitly
  limited support check — with the caption additionally disclosing the
  enabled ablation. A matched Append `L=3` baseline (evidence-queue E5) and
  any `L=3`/`L=4` expansion remain deferred user decisions **[SETTLED]**; no
  cells here.

### C10. Other appendix claims (noise l.2043-2160; spin-boson/Bose-Hubbard/Hubbard transfer benchmarks l.2192+)

- Noise: Hubbard-only fixed-iteration-8 diagnostics, disclosed conventions.
  Transfer benchmarks: older data, `S` explicitly unreported.
- AI: "Appendix evidence uses older conventions than the core."
- Remedy: retained as disclosed diagnostics; replacements deferred
  **[SETTLED]**; register entries only (§5.7).

### C11. Same-cutoff ED reference claims (l.988-1003)

- Claim: cutoffs 3/7 per sector; documented `M=10` drifts; receipt
  `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_ed_cutoff_reference_six_regime_20260727.json`
  (`66a6409790affffd6ce8928d7fb46cc945b57d50e210d3cb215e8039a63c5573`).
- AI: "The reference cutoff differs from the working cutoff."
- Remedy: gate G2 (same-cutoff identity per cell) + G8 (exact reference never
  a controller input). No new evidence needed.

---

## 3. Correctness and fairness findings

### 3.1 Manuscript versus mathematics/implementation

- **M1 (high, staged)**: Phase-III equations commit to stationary-source
  (`Paper_I.tex:329`, `g=(g_α,0)` l.354-359) while displayed evidence used
  measured residual; resolution staged behind Study 1 (§2-C4). Smallest
  repair: none now; post-selection, either equations or evidence change —
  user decision.
- **M2 (high)**: Append refit-coordinate convention undocumented in the
  manuscript and violated by the implementation (§3.2-F1, §2-C5). Smallest
  manuscript repair (deferred, user-owned): one sentence in the comparison
  protocol paragraph stating RA-whitened versus Append-unwhitened refits as
  an intentional method distinction.
- **M7 (medium)**: `L=3` ablation-enabled caption inconsistency (§2-C9).
  Smallest repair: caption disclosure (user-owned manuscript edit).
- **M9 (low)**: asymmetric plateau-window rules disclosed but must be
  restated in replacement aggregation receipts (§2-C7).
- Verified consistent **[FACT]**: config table optimizer (Powell/200), seed 7,
  horizon 50, cost weights, lane protection (η₁=0.70, τ₁=0), shortlists
  (24,12,0.25 macro; 8,4,1/12 singleton), ρ₀=0.25, support tolerance 1e-6,
  novelty-off + fallback, compile identity — all match the typed protocols
  and `sr_snake_route_profile.py` contracts. Phase-I unweighted cost
  (l.1067-1069) matches `resource_weighting_scope=late_resource_weighting_v1`
  in every materialized protocol.

### 3.2 Comparator fairness (code-cited)

- **F1 (blocking): Append accepted refits are whitened and ledger-inflated.**
  - Executable: `pipelines/static_adapt/ra_adapt/append.py:1018`
    (`build_supported_fs_powell_chart(...)` per accepted refit) with chart
    origin `N_H_refit`/`N_grad`/`N_metric` charges at `append.py:1048-1089`;
    module docstring `append.py:3-8` declares the shared "accepted-refit
    chart" deliberately.
  - Protocol: every `*append_macro*.json` cell records
    `accepted_refit_coordinate_chart: supported_fs_whitened_fixed_v1`;
    constant wired at `append.py:281-283`.
  - Historical comparator (displayed Append rows) used plain unwhitened
    full-ansatz Powell (`pipelines/exact_bench/generic_static_adapt_variants.py:15`,
    "routes use Powell for the full refit"; no whitening machinery present).
  - Settled convention: Append remains unwhitened **[SETTLED]**.
  - Scientific consequence: Append trajectories differ from the conventional
    method (different refit optimum path under Powell's
    coordinate-sensitivity), and Append's `S` is inflated by
    `O(n + n(n+1)/2)` chart charges per round — both directions of unfairness
    are possible; the comparison is uninterpretable as "conventional Append".
  - Smallest repair **[PLAN]**: in `append.py`, replace the accepted-refit
    chart construction with an ordinary full-ansatz Powell refit over the
    native ansatz coordinates (chart id `native_v1`, already defined in
    `pipelines/static_adapt/accepted_refit.py`:
    `ACCEPTED_REFIT_CHART_NATIVE_V1`), starting from the inherited runtime
    parameters with the admitted coordinate at zero; delete the
    chart-origin gradient/metric ledger charges (keep `N_H_refit` objective
    charges); set the protocol constant to `native_v1`; update
    `test/test_ra_adapt_append_facade.py` (chart assertions near l.603-619)
    and `test/test_ra_adapt_bundles.py`; rematerialize both bundles.
    Do not touch RA's refit.
- **F2 (medium): always-insertion request-fidelity defect.**
  - `bundles.py:1190-1194` builds `PlateauCommutationInsertion()` for the
    `ra_macro_always` route; the executed mode is selected by an
    algorithm-id string match at `engine.py:142-146`
    (`"always_insertion" in algorithm_id` → the
    `…full_insertion_diagnostic_v1` route contract,
    `adapt_insertion_mode: full`). The serialized protocol `request` field
    therefore misdescribes the executed policy, and dispatch-by-name is
    brittle.
  - Consequence: execution is (apparently) correct, but the protocol is not
    self-describing and a future request-driven path could silently run
    plateau gating in an always cell.
  - Smallest repair **[PLAN]**: add a typed always-insertion policy (e.g.
    `FullCommutationInsertion` in `sr_snake/contracts.py` alongside
    `PlateauCommutationInsertion`/`AppendOnlyInsertion`), dispatch on it in
    `engine.py` (removing the string match), emit it in `_build_request`,
    and gate every always/plateau cell on at least one interior-position
    geometry receipt (G5). Note **[FACT]**: no singleton full-insertion route
    contract exists in `sr_snake_route_profile.py` (only
    `…macro_only_physical_lanes_full_insertion_diagnostic_v1`); the singleton
    always cells of the core matrix need a registered singleton
    full-insertion contract (§6).
- **F3 (required for the core): singleton matrix materialization support.**
  - **[FACT]** `bundles.py` scope is
    `paper_i_displayed_macro_rows_plus_targeted_singleton_preservation_v1`
    (l.85); `VALIDATION_ROUTE_IDS = MACRO_ROUTE_IDS + singleton_plateau`
    (l.107). There is no `append_singleton` or
    `ra_singleton_{append_only,plateau,always}` cell generator, and the two
    58-cell bundles contain no singleton-matrix cells (only the two
    preservation cells at horizon 23).
  - The executable seams exist: `SinglePauliWordCandidateAdapter`
    (`adapters.py:119`, incl. `global_executable_pool` l.166) and the Append
    facade's global-child scan
    (`append.py:148-157`, receipt
    `global_pool_constructed_before_gradient_selection` l.1264).
  - Smallest repair **[PLAN]**: extend `build_study1_cell_specs`/a sibling
    builder with the singleton route family at horizon 50 and
    regime-appropriate cutoffs; register the singleton full-insertion route
    contract; add loader/validation coverage mirroring the macro cells.
- **F4 (clarification): preservation-cell policy semantics.**
  - The `validation__*__singleton_plateau` cells carry the bundle's
    stationarity policy. Under `ra_repair_measured_late_v1` they can
    reproduce the historical measured-residual plateau route (T13
    preservation); under `ra_repair_stationary_late_v1` they *cannot* equal
    history by design — they are defect-sensitivity cells whose receipts
    must show zero active-gradient acquisitions.
  - Smallest repair **[PLAN]**: no code change required if
    `test/test_ra_adapt_singleton_plateau_preservation.py` already encodes
    this asymmetry; the execution agent must verify and, if absent, add the
    two distinct gates (G13a preservation-equality on the measured bundle;
    G13b expected-deviation + zero-acquisition on the stationary bundle).
    Do not reinterpret a stationary-bundle deviation as a failure.
- **F5 (reporting-only): inert RA fields in Append protocols.**
  - Append protocols record `trust_policy_id`, `phase3_solver_id`, and
    `phase3_multiplier_contract` that the conventional selector never uses
    (declared "common infrastructure", `append.py:275-279`).
  - Smallest repair **[PLAN]**: add an explicit
    `selector_scope: conventional_append_no_phase3_no_trust_v1` (or
    equivalent) field during F1 rematerialization so a reviewer cannot read
    the inert ids as executed behavior. Low priority; do not block on it.

### 3.3 Source/provenance and reporting correctness

- **[FACT]** Bundle cell source locks resolve to the retrieved r50 archives
  with member-level SHA-256 (e.g.
  `source_locks.json → raw_outputs/chtc_fetch_paper_i_hh_r50_full48_20260719/append_macro_8887541_completed_p4/…intermediate_strong…tar.gz`),
  and `source_materialization/source_validation_receipt.json` passed.
- **[FACT]** Implementation inventory is digest-stable across
  materialization (`preflight_sha256 == post_loader_sha256`, 144 files); the
  supersession chain v1→v5 preserves older materializations unchanged.
- **[FACT]** The Q2 scheduler precondition was verified read-only (lock
  receipt: no active/queued/held job invokes `paper_i_runner.py` or
  `paper_i_hh_powell_pareto.py`), satisfying the retirement precondition the
  user set.
- **[FINDING]** The 48 `full` macro cells cover **both** cutoffs for all six
  regimes. Claim-facing cells are the 24 at the manuscript's
  regime-appropriate cutoff (weak→3, strong→7; `Paper_I.tex:991-992`,
  caption l.1087). The other 24 are cross-cutoff sensitivity cells: useful
  diagnostics, not required for any visible claim, and not part of the
  mandatory 48-cell core (which is completed instead by the 24 missing
  singleton cells). See §5.4.

---

## 4. Existing-evidence classification

One class per family: `retained` / `superseded` / `unresolved` /
`requires rerun`. Promotion/demotion remain user decisions; this is audit
classification only.

| id | Evidence family | Class | Reason / conditions |
|---|---|---|---|
| E1 | Displayed macro insertion overlays + aggregates (Fig. l.1088, `figures/paper_i_hh_macro_common_accuracy_20260723/*macro*`) | `requires rerun` | Confirmed interior-position append-chart scoring (lock receipt) + macro pool mismatch (102/148 vs 123/171) + overlap-trust/whitened-eigh selector conventions; already marked provisional. Affects C2, C3, C6. |
| E2 | Displayed singleton insertion overlays + aggregates (Fig. l.1096) | `requires rerun` | Mandated by the settled full singleton matrix; also policy-consistency (C4) and Append-baseline protocol matching must be established by construction, not archaeology. Affects C1, C3, C6. |
| E3 | Retrieved r50 `append_projected_singleton` archives, six regimes (`raw_outputs/chtc_fetch_paper_i_hh_r50_full48_20260719/comparator_completed_…`, `…20260720/retrieval_…`) | `unresolved` | Passing 50-round receipts, closed ledgers, same-cutoff refs verified on samples (e.g. strong-strong err 2.15×10⁻⁵, `S_alg=1,020,483`). Conventional Append is unaffected by stationarity/insertion/trust/Phase-III, and the historical refit was unwhitened — so these rows are *candidates* to fill the six singleton-Append core cells without rerun **if** the §5.6 matched-protocol audit passes (pool ancestry/guards, unit-Pauli convention, Powell/200/seed 7/horizon 50, compile identity, ledger convention). Reuse versus rerun is a user decision after that audit. |
| E4 | Retrieved r50 `append_macro` archives (`…append_macro_8887541_completed_p4/…`) | `superseded` | Ran the unfiltered 123/171 macro pool; the settled comparison requires 102/148. They remain the immutable source locks/baselines for the replacement protocols. |
| E5 | Later 12-cell Geo campaign (6 macro + 6 projected-singleton r50; four retrieval dirs listed in the handoff) | `retained` (conditional) | Receipts verified passing with same-cutoff references, closed ledgers, no leakage flags. Condition: §5.6 convention-alignment audit against the final reporting rules before the C8 sentence is re-derived. |
| E6 | Compact comparator report 20260711 (`output/pdf/paper_i_adaptive_comparators_retrieved_compact_20260711/…pdf`) | `superseded` | Historical cutoff/accounting conventions; not final evidence for active two-site claims **[SETTLED]**; preserve immutably. |
| E7 | `L=3` intermediate-weak support artifacts (`figures/paper_i_hh_l3_intermediate_weak_physical_lanes_support_20260709/…`) | `retained` (as limited support check) | RA-only, cutoff 1, ablation-enabled, no `S`; supports only the "limited support check" framing (C9). Any stronger claim ⇒ deferred E5-queue Append baseline (user decision). |
| E8 | Noise appendix evidence (Hubbard fixed-depth-8 sweeps) | `retained` | Disclosed diagnostic conventions; replacement deferred **[SETTLED]**. |
| E9 | Supplementary transfer benchmarks (Hubbard, Bose-Hubbard, spin-boson/Rabi; `S` unreported) | `retained` | Disclosed; deferred. |
| E10 | ED cutoff receipt `paper_i_hh_ed_cutoff_reference_six_regime_20260727.json` | `retained` | Reporting reference; hash pinned in the manuscript comment (l.1001-1003). |
| E11 | 2026-07-25 v3 macro/singleton plateau-insertion CHTC campaigns (normalized manifests) | `superseded` (as claim evidence) / `retained` (as source locks) | They feed the displayed overlays (E1/E2) and the lock-receipt refit-equality check; their protocols carry the D1/D2/D3-era conventions. |
| E12 | Materialized v5 bundles (2 × 58 cells) | `unresolved` | Loader-validated, unsubmitted. Append cells and the always-cell request field require F1/F2 repair + rematerialization; macro RA cells are protocol-sound; singleton matrix absent. Do not rubber-stamp both 58-cell bundles (§5). |
| E13 | Historical singleton plateau trajectory fixtures (T13; `test/fixtures/…`, lock receipt characterization fixtures) | `retained` | Preservation-gate source for the measured-residual bundle. |

---

## 5. Minimal required run matrix

Run classes follow `paper-i-run` (`smoke`/`diagnostic`/`candidate`/
`paper_facing`). Every cell inherits the common contract of §6; per-cell
rows list only what varies. **No cell executes without the §7 gates and
explicit user authorization.**

### 5.1 Preflight / local semantic validation (class: smoke/diagnostic; local)

| id | Cell | Purpose / gate |
|---|---|---|
| P1 | Re-run `test/test_ra_adapt_*` (18 suites) + `test_static_adapt_commutation_metadata.py` + resume/accepted-refit adjacents | green baseline before any repair |
| P2 | Pool/hash smoke: `build_executable_macro_pool` → 102@3 = `1549f2e1…b17df`, 148@7 = `e30e879d…d14e`; parent inventory 123/171; singleton global-child pool construction receipt | pool identity (G3) |
| P3 | One-round semantic smoke per representation (macro RA plateau; singleton RA plateau; Append macro; Append singleton) at `nph=3`, 2-3 rounds, local | exact-chart receipts, ledger closure, checkpoint/resume round-trip (G5, G10, G11) |
| P4 | Packaged-worker smoke (one cell end-to-end through the execution template) | CHTC payload integrity before submission |

### 5.2 Append protocol repair + rematerialization (required; class: implementation + materialization)

- F1, F2, F5 repairs (§3.2) with their focused tests; then rematerialize both
  Study-1 bundles as a new revision (v6) with normalized
  baseline-versus-target diffs proving only: Append refit chart
  (`supported_fs_whitened_fixed_v1 → native_v1`), Append ledger-charge
  removal, always-cell typed-request fidelity, `selector_scope` field, and
  revision labels changed. Older materializations remain preserved
  (supersession-chain discipline already in place).
- Gate: cross-file loader validation equivalent to the current 116/116; diff
  receipt reviewed by the user with Study-1 authorization (§9).

### 5.3 Minimal Study-1 discriminator (class: candidate; CHTC after authorization)

User-locked scope (Q1 decision, restated in the implementation spec §7.2):
regimes `strong_weak_u8` and `strong_strong_u8` at `nph=3`, reduced horizon
(materialized at 23 rounds), both gradient-policy bundles.

| Cells | Count | Notes |
|---|---|---|
| RA macro {append_only, plateau, always} × 2 regimes × 2 policies | 12 | the stationarity discriminator proper |
| RA singleton plateau preservation × 2 regimes × 2 policies | 4 | measured bundle: preservation equality (G13a); stationary bundle: expected deviation + zero acquisitions (G13b) |
| Append macro × 2 regimes | 2 (not 4) | **[FINDING]** conventional Append is stationarity-invariant (no Phase-III/trust/active-gradient acquisition); running it once per regime serves both bundles. The current materialization duplicates it per bundle (4 cells); the rematerializer should either dedupe with a shared-cell reference or the executor should run 2 and link results into both bundles' completion matrices. |

Total: 18 executed cells (20 materialized minus 2 redundant Append
duplicates). This is the smallest decisive subset consistent with the user's
locked validation scope; do not shrink it further — both regimes exercise
interior insertion, which is exactly where the stationarity policies
separate. The remaining 96 materialized `full` cells are **not** executed in
Study 1.

Dependency stated plainly: every RA cell of the 48-cell core depends on the
stationarity choice, so no core RA cell can run before the user's Study-1
decision. Core Append cells are policy-invariant and *could* run alongside
Study 1 once F1 lands; keeping them with the core fan-out is still
recommended so the entire claim-facing matrix shares one implementation
inventory digest.

### 5.4 Selected-stationarity 48-cell two-site core (class: paper_facing; CHTC after separate authorization)

After the user selects the policy: rematerialize one core bundle
(`ra_repair_<winner>_late_core_v1`) containing exactly:

- 6 regimes × regime-appropriate cutoff (weak→`nph=3`:
  `weak_weak`, `intermediate_weak`, `strong_weak_u8`; strong→`nph=7`:
  `weak_strong`, `intermediate_strong`, `strong_strong_u8`);
- × 2 representations (macro on the 102/148 executable pool; singleton with
  123/171 parent ancestry, RA staged children / Append global children);
- × 4 trajectories (Append conventional; RA append-only/no-insertion; RA
  plateau; RA always) — 48 cells, horizon 50, Powell/200, seed 7.

Disposition of already-materialized cells: the 24 macro cells at the
regime-appropriate cutoff carry over (RA cells unchanged; Append cells from
the v6 rematerialization); the 24 cross-cutoff macro cells are reclassified
as optional sensitivity cells and are **not** executed unless the user asks
(they serve no visible claim; the manuscript's cutoff-drift analysis is
already covered by the ED receipt, C11). The 24 singleton cells are new
(F3). If the §5.6-A audit passes and the user approves reuse, the 6
singleton-Append cells may be filled by E3 evidence instead of rerun.

### 5.5 Conditional Study 2 (late versus all-phase weighting)

Only after the Study-1 decision and only if the user wants to change the
final method definition away from the manuscript's current late-weighting
commitment (l.1067-1069). If declined, `late_resource_weighting_v1` stands
and no cells run. If authorized: the winning-policy bundle re-run with
`all_phase_resource_weighting_v1` on the Study-1 discriminator scope first,
compared against the corresponding late-weighting cells; core fan-out only by
a further explicit decision. Never combined with the stationarity comparison.

### 5.6 Reuse-validation audits (class: diagnostic; local, no science reruns)

- **A. Singleton-Append reuse audit (E3)**: for each of six retrieved
  `append_projected_singleton` r50 archives, verify against the final
  convention: parent ancestry 123/171 with hash; guarded canonical
  unit-Pauli children; global-pool-before-selection receipt; Powell 200;
  seed 7; horizon 50; unwhitened full refit; compile identity; ledger
  closure and `S_alg` convention. Emit one pass/fail matrix. Pass ⇒ user
  decides reuse versus rerun for the 6 core cells; any fail ⇒ those cells
  run in §5.4.
- **B. Geo reuse audit (E5)**: for the 12 Geo cells, verify same-cutoff
  reference identity, 50-round completion, ledger closure, compile identity,
  and recompute the C8 comparison under the final reporting rules against
  replacement singleton no-insertion RA rows. Pass ⇒ C8 stands with retained
  evidence; fail ⇒ report the exact mismatch and put "targeted Geo rerun
  versus claim removal/deferment" to the user. No Geo cells in the core.

### 5.7 Non-executable deferred-evidence register

Noise replacements; metric/trust pruning efficacy; greedy/combinatorial
batching; prune+batch; beam continuation; `L=3` matched-Append baseline
(queue E5) and any `L=3`/`L=4` scaling matrix; other-model appendix
(Hubbard/Bose-Hubbard/spin-boson) replacements; cross-cutoff macro
sensitivity cells. Each stays classified in §4/§2 with its affected claim;
none enters the executable matrix here.

---

## 6. Typed protocol and bundle contract

The v5 bundle machinery already locks most of the required surface
**[FACT]**; the execution agent must keep all of it and add the deltas
marked (new).

Per-cell resolved protocol (schema `paper_i_ra_adapt_resolved_protocol_v1` /
`paper_i_append_adapt_resolved_protocol_v1`) must pin:

- problem identity (family `hh`, `L=2`, regime parameters, boundary, sector,
  encoding, `n_ph_max`, same-cutoff exact-target label), and the ED receipt
  pointer (C11);
- `candidate_representation` + `adapter_id`; parent inventory (count 123/171
  + ordered-rows sha) and executable pool (count 102/148 macro, or the
  guarded child-pool contract) — equality across compared cells (G3);
- `algorithm_id`, `selector_identity`
  (`ra_adapt_staged_phase_i_ii_iii_funnel_v1` /
  `append_adapt_largest_absolute_commutator_gradient_v1`), and (new)
  `selector_scope` for Append (F5);
- typed insertion policy matching the executed mode (F2), with
  `derivative_chart_id = exact_ordered_insertion_zero_angle_v1`;
- `phase3_solver_id = supported_metric_projected_generalized_trust_v1`,
  `trust_policy_id = supported_source_gram_no_endpoint_overlap_trust_v1`,
  `phase3_multiplier_contract` with separate `kappa`, `lambda`, additive
  `mu = kappa + lambda`, and boundary-activity rule (RA cells; explicitly
  out-of-scope for Append via `selector_scope`);
- `active_gradient_policy` and `resource_weighting_scope` echoes (G7);
- accepted-refit triple: RA `full_ansatz_v1` +
  `supported_fs_whitened_fixed_v1` + `expanded_runtime_projected_logical_v1`;
  Append `full_ansatz_v1` + `native_v1` (post-F1) (G4);
- optimizer (`powell`, maxiter 200), seeds `{adapt: 7, transpiler: 7}`,
  horizon (50 core / 23 validation), stopping rule, fresh-start/resume kind;
- `estimator_accounting_convention` and `compile_identity`
  (`table_i_basis_gate_transpile_v1`, level 0, seed 7, no coupling map,
  reference prep included);
- `lineage_authority` (candidate-inventory lineage hash; Append:
  `append_position_only`, `selection_with_replacement`,
  `ra_staged_funnel_invoked=false`);
- cell source-lock id + archive/member SHA-256; `bundle_id`,
  `bundle_manifest_sha256`, canonical-JSON self-digest;
  `execution_authorized=false` at materialization.

Bundle level (schema `ra_adapt_run_bundle_v1`): manifest canonical+file
digests; `source_locks.json` with per-cell archive/member hashes and
`all_required_files_verified`; `expected_artifacts.json` (manifest,
checkpoint every round with tail, estimator ledger, result, summary,
non-colliding output/retrieval paths); execution templates 1:1 with
protocols; `validation_report.json` with the loader checks; implementation
inventory digest (`6cc3d62b…` at v5 — recomputed and re-pinned at v6);
environment/dependency fingerprints (already emitted by
`_default_environment_fingerprint`/`_dependency_lock_provenance`);
final receipt with supersession chain and
`user_decision_required_after_study_1=true`.

Reuse after audit: RA macro protocol cells (both bundles) — reusable
unchanged; Append cells — repair + rematerialize (v6); always cells —
request-field fix at v6; singleton matrix and core bundle — new
materialization; the two v5 58-cell bundles are then superseded planning
artifacts for everything except their carried-over RA macro cells.

---

## 7. Staged execution procedure (plan only)

1. **Plumbing corrections + targeted tests** (local): F1 (Append native
   refit + ledger), F2 (typed always-insertion), F3 (singleton
   materializer + singleton full-insertion route contract), F4 gate
   clarification, F5 field; run P1 suites plus the updated
   append-facade/bundle tests. No scientific semantics beyond the settled
   convention change; anything further stops for the user.
2. **Append protocol correction + rematerialization (v6)**: regenerate both
   Study-1 bundles; produce normalized per-cell baseline-versus-target
   diffs proving only the intended fields changed; re-run loader
   validation (116/116-equivalent).
3. **Local smokes**: P2 pool/hash; P3 one-round semantic smokes with
   exact-chart, ledger, checkpoint/resume verification; P4 packaged-worker
   smoke; source-anchor re-verification of every cell lock.
4. **User review + authorization** of the minimal Study-1 cell list (§5.3)
   and the diff receipt; explicit CHTC authorization.
5. **CHTC Study 1**: package, submit via the CHTC workflow, verify
   scheduler acceptance once, adaptive monitoring per the run-guide cadence
   through *worker completion* (not queue acceptance), retrieve to new
   non-colliding paths, validate per §8.
6. **User stationarity decision** from neutrally reported Study-1 results;
   record the decision receipt (ICM `user-review` successor). Then the
   conditional Study-2 decision (§5.5) — only if the audit-stated condition
   applies.
7. **Core rematerialization**: `ra_repair_<winner>_late_core_v1` (§5.4) +
   local validation; include the §5.6-A reuse outcome (user-decided) for the
   six singleton-Append cells.
8. **User review + CHTC authorization of that exact 48-cell core.**
9. **CHTC core execution** with adaptive monitoring through completion.
10. **Safe retrieval** to new paths; never overwrite prior retrievals.
11. **Validation**: manifests, checkpoints, replay identity, estimator
    ledgers, compiled resources, sector/cutoff/reference integrity,
    cross-cell matrix completeness (`done/failed/missing/blocked/
    superseded`), §8 gates per cell.
12. **Neutral aggregation** (plateau/common-accuracy/aggregates under the
    disclosed C7 rules, via `pipelines/reporting/paper_i_run_summary.py`
    contracts) into candidate evidence; then user decisions on evidence
    replacement, manuscript synchronization (C1-C9 rows), and Code Math
    Bijection re-binding (readiness addendum sequence steps 3-7).

The execution agent may repair ordinary run plumbing at any stage; it asks
before touching scientific semantics, and it never falls back to a
compatibility route on failure.

---

## 8. Objective gates (fail-closed; no favorable-outcome conditions)

- **G1 physics/source-lock equality**: per cell, problem fields equal the
  cell source lock; source archive+member hashes verified at execution time.
- **G2 same-cutoff identity**: `n_ph_work == n_ph_reference`; exact target
  label matches the ED receipt; `same_cutoff_reference=true` in the
  completion receipt.
- **G3 pool identity**: macro cells — executable-pool count/hash equality
  with `1549f2e1…` (102@3) / `e30e879d…` (148@7) and RA=Append equality per
  regime; singleton cells — parent inventory 123/171 hash equality across
  RA and Append, guarded-child construction receipts present, RA
  staged-exposure versus Append global-pool receipts present.
- **G4 refit-coordinate convention**: RA cells record and execute
  `supported_fs_whitened_fixed_v1`; Append cells record and execute
  `native_v1`; any cross-contamination fails the cell.
- **G5 insertion-position correctness**: every Phase-II/III geometry receipt
  carries `exact_ordered_insertion_zero_angle_v1` with an explicit position;
  every always-insertion and plateau-insertion cell that claims interior
  capability produces ≥1 interior-position (`p < n_k`) scored receipt, and
  accepted interior admissions record the position; append-only cells record
  endpoint-only domains.
- **G6 Phase-III integrity (RA cells)**: support projection receipt
  (threshold 1e-6, eigenvalues, retained mask, provenance id); projected
  generalized solve with whitening recorded false and ridge 0; separate
  `kappa`/`lambda` with `mu=kappa+lambda` and the boundary-activity rule;
  trust transaction receipts with zero endpoint-overlap acquisitions.
- **G7 policy echo**: `active_gradient_policy` and
  `resource_weighting_scope` present in protocol and echoed in the result;
  stationary-source cells show empty active-gradient acquisition telemetry
  and zero corresponding ledger charges; measured-residual cells show the
  acquisitions and charges.
- **G8 exact-reference isolation**: no controller decision consumes the
  exact reference; reference appears only in reporting fields.
- **G9 numerical/physical integrity**: finite energies/parameters;
  `sector_leak_flag=false`; `boson_truncation_leak_flag=false`; accepted
  energy non-increase per accepted transition or an explicit typed rollback
  receipt.
- **G10 accounting closure**: estimator ledger closes;
  `S_alg = N_H_outer + N_H_refit + N_grad + N_metric` reconciles with the
  occurrence summary; Append cells contain no whitening-chart
  gradient/metric charges (post-F1).
- **G11 checkpoint/replay/resume**: checkpoint every controller round with
  signed prefixes; replay identity verified for ≥1 cell per method/regime
  family; resume authentication binds problem+route digest.
- **G12 compile identity**: `table_i_basis_gate_transpile_v1`, level 0,
  seed 7, no coupling map, reference prep included; compiled `N2q/D2q/Dc`
  present at the reporting prefix.
- **G13 preservation cells**: (a) measured bundle — trajectory equality with
  the T13 fixture within recorded tolerances; (b) stationary bundle —
  expected deviation with zero active-gradient acquisitions (F4).
- **G14 completeness**: per-bundle completion matrix with explicit
  `done/failed/missing/blocked/superseded`; folder presence is never
  completion; all expected artifacts present with hashes.

Scientific outcomes (which method is more accurate/cheaper, which policy
wins) are reported neutrally and are never pass conditions.

---

## 9. Exact user decisions and execution interfaces

Decisions, in order:

1. Approve or revise this plan and the exact §5.3 Study-1 cell list
   (including the Append dedupe recommendation).
2. Authorize the F1/F2/F3/F5 repairs and v6 rematerialization (ordinary
   implementation within settled conventions; flagged here because F1
   changes comparator trajectories by design).
3. Authorize CHTC submission of Study 1 (separate, explicit).
4. Select the stationarity policy from validated Study-1 results.
5. Authorize or decline Study 2 (only if changing the late-weighting method
   definition; otherwise decline).
6. Decide singleton-Append reuse versus rerun after the §5.6-A audit.
7. Decide Geo outcome if §5.6-B fails (targeted rerun versus claim
   removal/deferment of C8).
8. Authorize CHTC submission of the 48-cell core (separate, explicit).
9. Decide evidence promotion/replacement and the manuscript edits (C1-C9),
   including the M1 equation/policy resolution and the M2 fairness sentence;
   then Code Math Bijection re-binding.

Execution interfaces (verified present):

- Facades: `pipelines.static_adapt.ra_adapt.run_ra_adapt` (`engine.py`),
  `pipelines.static_adapt.ra_adapt.run_append_adapt` (`append.py`, lazy via
  `__init__.py:29`).
- Contracts/adapters: `ra_adapt/contracts.py` (typed request/protocol/
  receipts), `ra_adapt/adapters.py` (`MacroCandidateAdapter`,
  `SinglePauliWordCandidateAdapter` incl. `global_executable_pool`),
  `sr_snake/contracts.py` (shared policy types).
- Kernels: `ra_adapt/{engine,runtime,insertion_geometry,support,trust,
  pools}.py`, `pipelines/static_adapt/accepted_refit.py`
  (`ACCEPTED_REFIT_CHART_NATIVE_V1` for F1),
  `exact_geometry_backend.py`, `commutation_metadata.py`,
  `estimator_call_ledger.py`, `joint_linear_solve.py`,
  `route contracts in sr_snake_route_profile.py`.
- Materializer/validators: `ra_adapt/bundles.py`
  (`build_study1_cell_specs`, `normalize_and_verify_source_locks`, loader
  validation), the v5 bundle roots under
  `chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/…v5/`,
  `final_materialization_receipt.json`.
- Tests: the 18 `test/test_ra_adapt_*` suites (F1 touches
  `test_ra_adapt_append_facade.py`, `test_ra_adapt_bundles.py`,
  `test_ra_adapt_cost_scope.py`/`gradient_policy` unaffected), plus
  `test_paper_i_sr_snake_resume_adapter.py`,
  `test_static_adapt_insertion_commutation_plateau.py`.
- Reporting: `pipelines/reporting/paper_i_run_summary.py` and
  `agent_guidance/static-adapt/reporting/run-summary.md`.
- Run gates: `agent_guidance/skills/paper-i-run/SKILL.md`,
  `agent_guidance/shared/run-guide.md`,
  `agent_guidance/skills/shared/scripts/resolve_visible_settings.py`,
  `$chtc-direct` for scheduler operations, ICM receipts under
  `agent_guidance/static-adapt/icm/ra-adapt-repair-20260727/`.

Unresolved problems (stated per the response contract): the F4
preservation-cell semantics need a one-time confirmation in the test before
Study 1; the singleton full-insertion route contract does not exist yet and
is on the F3 critical path; whether the six singleton-Append core cells are
reruns or reuses is decision 6 and does not block anything else.

No manuscript, implementation, evidence, bundle, or run state was changed by
this audit.

Files to edit:

- `pipelines/static_adapt/ra_adapt/append.py` — F1 (native unwhitened
  accepted refit; remove chart-origin `N_grad`/`N_metric` charges; protocol
  constant → `native_v1`; F5 `selector_scope` field).
- `pipelines/static_adapt/sr_snake/contracts.py` — F2 typed
  always-insertion policy (`FullCommutationInsertion`).
- `pipelines/static_adapt/ra_adapt/engine.py` — F2 dispatch on the typed
  policy (remove the `algorithm_id` string match at l.142).
- `pipelines/static_adapt/ra_adapt/bundles.py` — F2 request fidelity
  (`_build_request`), F3 singleton cell-spec generation + core-bundle scope,
  §5.3 Append dedupe, v6/core rematerialization plumbing.
- `pipelines/static_adapt/sr_snake_route_profile.py` — F3 registered
  singleton full-insertion route contract.
- `test/test_ra_adapt_append_facade.py`, `test/test_ra_adapt_bundles.py` —
  updated chart/ledger/request assertions; new singleton-matrix coverage.
- `test/test_ra_adapt_singleton_plateau_preservation.py` — F4 dual-gate
  (G13a/G13b) if not already encoded.
- New: `chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/
  ra_adapt_unification_post_refactor_v6/…` (rematerialized Study-1 bundles)
  and later `ra_repair_<winner>_late_core_v1` (48-cell core), each with
  manifests, protocols, source locks, validation reports, and receipts.
- Manuscript, figures, and evidence files: none in this plan (user-gated,
  §9 decisions 9).

---

## 10. Approved v7 objective-gate correction (2026-07-28)

The user approved this versioned correction and the exact 18-direct-job
Study-1 submission on 2026-07-28.  This addendum supersedes only the
contradictory G1/G2/G3/G5/G8/G9/G11/G13 evidence semantics below.  It does not
change a Hamiltonian, regime, cutoff, optimizer, budget, seed, route, horizon,
stationarity policy, or the §5.3 cell count.

The immutable v1-v6 materializations and their receipts remain historical
bytes.  They are not execution authority.  A new immutable v7 successor must
bind the corrected contracts, current implementation inventory, and a signed
Study-1 objective-gate authority receipt before an authorization receipt can
exist.

### 10.1 Corrected gates

- **G1 v2 — source-lock equality:** materialization re-hashes every required
  historical archive and member once and emits a self-digested per-cell
  receipt.  The compact receipt, source-lock manifest, protocol, and current
  source archive are hash-bound into v7.  Each worker verifies those bound
  bytes; the hundreds-of-megabytes historical archives are not retransferred
  to every worker.
- **G2 v2 — same-cutoff identity:** every direct result and completion receipt
  explicitly records `n_ph_work`, `n_ph_reference`,
  `same_cutoff_reference=true`, exact-target label, and the bound ED receipt
  hash.  Exact energy remains reporting-only.
- **G3 v2 — pool identity:** stable membership is the pool count plus ordered
  label hash.  Macro membership is
  `102 / a8831528…d3d9` at `nph=3` and
  `148 / e6de9374…b03a` at `nph=7`.  The coefficient-bearing full pool hash is
  recomputed from each source-locked problem and must agree exactly between
  RA and Append only within that same problem/regime.  The former
  `1549f2e1…`/`e30e879d…` hashes remain characterization-fixture values, not
  universal Study-1 hashes.  Singleton RA/Append ancestry and staged/global
  exposure are proven by a signed construction-equivalence receipt; this does
  not add scientific jobs.
- **G5 v2 — insertion evidence:** relevant cells serialize the scored
  insertion-position population, not merely eligibility or the selected
  position, and prove at least one interior scored position.
- **G8 v2 — exact-reference isolation:** a source/dataflow regression receipt
  and runtime attestation are both required.  A result-only negative assertion
  is insufficient.
- **G9 v2 — numerical/physical integrity:** reporting emits explicit finite
  value, sector-leak, boson-truncation-leak, and accepted-energy
  monotonicity/typed-rollback diagnostics.  These observations never feed the
  controller.
- **G11 v2 — checkpoint/replay/resume:** signed per-round prefixes and all
  referenced resume/ledger sidecars are retained.  At least one cell per
  method/regime family performs a bounded deterministic same-cell replay
  inside its already-authorized physical job.  RA cells additionally perform
  an authenticated resume round trip.  Append cells instead emit and validate
  the typed `authenticated_reconstruction_only_v1` boundary because no public
  Append continuation contract is authorized; that explicit inapplicability
  is the G11 resume outcome for Append, not a fabricated resume execution.
  The direct-job count remains 18.
- **G13 v2 — same-physics preservation:** T13 is retained as a generic route
  characterization only at its own `U=2, g=1` physics.  It is not a numerical
  baseline for the `U=8` Study-1 cells.  Study-1 preservation requires a
  deterministic same-problem replay and a matched measured/stationary pair
  differing only in active-gradient policy.  Stationary cells additionally
  require zero active-gradient indices and zero associated charge.  The paired
  trajectory deviation is reported neutrally and is never a pass condition.

G4, G6, G7, G10, G12, and G14 retain their prior scientific meaning, but the
fetched-package validator must evaluate all fourteen gates fail-closed rather
than validating packaging alone.

### 10.2 Execution boundary

The only authorized CHTC scope is the §5.3 Study-1 discriminator: 20 logical
cells, 18 direct executions, and two hash-authenticated shared Append
references.  Packaging must pass the corrected P2 pool proof, all four P3
representation smokes, P4 packaged-worker smoke, v7 loader validation, and
authorization binding before `condor_submit`.  Submission does not authorize
the later 48-cell core, evidence promotion, stationarity winner selection, or
manuscript edits.

Unresolved problems: none in the corrected Study-1 semantics.  Runtime
scientific outcomes and the later stationarity choice remain intentionally
unknown until the 18 direct cells finish and G1-G14 validation closes.

Files to edit:

- `pipelines/static_adapt/ra_adapt/` and
  `pipelines/static_adapt/sr_snake/` reporting/checkpoint seams.
- `pipelines/reporting/paper_i_run_summary.py` and
  `pipelines/reporting/paper_i_append_run_summary.py` where required for
  reporting-only diagnostics.
- `pipelines/static_adapt/ra_adapt/bundles.py` and focused RA/Append/bundle
  tests.
- `chtc/paper_i_ra_adapt_repair_20260727/materialize_study1_v7.py` and new
  immutable
  `bundles/materializations/ra_adapt_unification_post_refactor_v7/`.
- `chtc/paper_i_ra_adapt_repair_20260727/study1_minimal_20260728_v1_chtc/`
  package contracts, workers, and validators.
- New append-only ICM correction/materialization/execution receipts under
  `agent_guidance/static-adapt/icm/ra-adapt-repair-20260727/`.
- Manuscript, figures, promoted evidence, and v1-v6 files: none.

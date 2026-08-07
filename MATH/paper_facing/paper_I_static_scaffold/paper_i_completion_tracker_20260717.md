# Paper I completion tracker

Date: 2026-07-17
Scope: the single active agent-facing execution and evidence queue for finishing
the Paper-I Hubbard--Holstein comparison. This document authorizes no run or
submission by itself.

## Deliverable boundary

This tracker ends when the complete run evidence and comparison PDF are built,
validated, and source-mapped. It does **not** include editing, copying,
recompiling, or replacing the Paper-I manuscript.

There is exactly one active Paper-I tracker: this file. Dated result JSONs,
source maps, audits, route contracts, and validation reports are evidence or
authorities, not competing work queues.

## Fixed run program

The full program contains 53 scientific jobs, excluding short implementation
smokes:

| Family | Variants | Regimes per variant | Full jobs |
|---|---:|---:|---:|
| Main SR-SNAKE | 1 | 6 | 6 |
| Geo-ADAPT comparators | macro and projected singleton | 6 | 12 |
| Append-ADAPT comparators | macro and projected singleton | 6 | 12 |
| Appendix SR-SNAKE ablations | pruning, batching, and beam | 6 | 18 |
| Visible weak--weak mechanism-table completion | 5 additional source-locked rows; anchor, pruning, and batching are reused | 1 | 5 |
| **Total** |  |  | **53** |

Every scientific job has an exact horizon of 50 completed controller
rounds/ADAPT iterations unless a run-level implementation or scientific error
aborts it. Plateau detection must not terminate a run early.

## Ordered work queue

| Order | Required work | Status | Completion gate |
|---:|---|---|---|
| 1 | Audit (S_{alg}) and fidelity across SR-SNAKE, Geo-ADAPT, and Append-ADAPT | complete | Definitions, receipts, replay recomputation, focused tests, and the four cache-off comparator smokes pass. |
| 2 | Materialize and freeze the main SR-SNAKE source lock | complete locally; remote gate pending | Route-profile document, normalized settings, route digest, immutable-parent-derived source archive, and archive-only preflight pass. |
| 3 | Run the main SR-SNAKE six-regime matrix to round 50 | pending | Items 1--2. |
| 4 | Run Geo-ADAPT with the projected-singleton pool for all six regimes to iteration 50 | bundle ready; pending submission | Item 1 and comparator preflight. |
| 5 | Run Geo-ADAPT with the macro pool for all six regimes to iteration 50 | bundle ready; pending submission | Item 1 and comparator preflight. |
| 6 | Run Append-ADAPT with the projected-singleton pool for all six regimes to iteration 50 | complete; component adopted 2026-07-29 | Twelve-cell stationary-core v6 validation and component-adoption receipt. |
| 7 | Run Append-ADAPT with the macro pool for all six regimes to iteration 50 | complete; component adopted 2026-07-29 | Twelve-cell stationary-core v6 validation and component-adoption receipt. |
| 8 | Build and validate the five-row main comparison | pending | Items 3--7. |
| 9 | Run the pruning-enabled SR-SNAKE appendix matrix for all six regimes to round 50 | bundle ready; pending after main route | Exact main source lock; only pruning changes. |
| 10 | Run the batching-enabled SR-SNAKE appendix matrix for all six regimes to round 50 | bundle ready; pending after main route | Exact main source lock; only batching changes. |
| 11 | Run the beam-enabled SR-SNAKE appendix matrix for all six regimes to round 50 | bundle ready; pending after main route | Exact main source lock; only beam changes. |
| 12 | Run the five additional source-locked weak--weak mechanism-table rows to round 50 | not built or submitted | Main-route source lock plus a passing per-row settings-difference audit. |
| 13 | Build and validate the refreshed weak--weak mechanism-ablation support matrix and source inventory | pending | Weak--weak cells from items 3, 9, 10, and all five rows from item 12 are complete and validated. This does not authorize a manuscript edit. |
| 14 | Build the final comparison PDF and machine-readable source inventory | pending | All 53 scientific jobs complete and validated. |

### 2026-07-29 Append component adoption

The later stationary-core v6 execution supersedes the prepared-but-unsubmitted
Append bundle as the current Paper-I Append comparator source. All six macro
and all six projected-singleton cells completed 50 controller rounds and
passed fetched validation without automatic attempt selection. The immutable
32/48 cross-revision progress report remains non-evidentiary because its RA
matrix is incomplete; only its twelve independently closed Append cells were
adopted.

Adoption authority and source locks are recorded in
`MATH/paper_facing/paper_I_static_scaffold/provenance/paper_i_append_stationary_core_v6_component_adoption_20260729.json`
(SHA-256
`3373d5d54d267a0f5f75af7efb63518463a11c308c69d595b40f3516983b8cfc`).
The projected-singleton runtime/reporting consumer is
`agent_guidance/static-adapt/reporting/canonical-append-registry-v1.json`
(SHA-256
`2d59ee3d92ccf79d7c8f5fa826516159576220011872e7d1142a6b5b612f722a`).
This update does not complete item 8, promote any RA cell, or make the
aggregate partial-progress report paper evidence.

## Main SR-SNAKE authority

Do not duplicate the full executable route settings in this tracker. The
separate route task owns the exact contract. Once frozen, fill these pointers:

| Authority field | Value |
|---|---|
| Route-profile document | `MATH/paper_facing/paper_I_static_scaffold/paper_i_sr_snake_no_prune_symmetric_cost_candidate_runtime_settings_20260717.md` |
| Requested and resolved profile | `sr_snake_no_prune_symmetric_cost_v1` -> `supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_v1` |
| Route-contract SHA-256 | `69dcf6d5711a05b54cc143562bc6f437511d781164319b2705857b22deda1538` |
| Source commit | `8a746d244a15e2cb16099a732e78e1110a8e59f2` (base ancestry metadata; complete archive inventory is executable authority) |
| Source-archive SHA-256 | `fa9014b9608ccb8a301df0429268482abf6dd10c91eb336111a15d00be256d35` |
| Normalized six-regime manifest | `chtc/phase3_optuna/input/paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v1_chtc/normalized_manifests/` |
| Smoke evidence | `raw_outputs/paper_i_hh_sr_snake_no_prune_symmetric_cost_weak_weak_smoke_20260717/weak_weak_8_admissions_cache_off_v3/`; archive-only focused suite: 504 passed, 9 skipped |
| Full result/evidence root | planned: `raw_outputs/paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v1_chtc/` (submission pending remote gate) |

The route contract must encode the user-settled choices from the route task,
including main-route pruning, batching, and beam off; ordinary novelty
multipliers off; conditional novelty fallback telemetry; undamped response;
Phase-II whitening off; Phase-III supported whitening and adaptive trust on;
full active-plus-singleton Phase-III response; first-order Phase I;
measured-required Phase-II curvature; full accepted-ansatz Powell refit after
each admission; HH preseed off; random seed 7; no finite-angle, saddle escape,
periodic refit, terminal refit, terminal prune, or plateau early stop; and the
symmetric arctangent hardware-cost policy. This sentence is an acceptance
checklist, not an independent settings source.

All six main SR-SNAKE regimes run fresh from round zero through round 50. Older
prefixes are not valid substitutes when the scoring contract has changed.

## Next geometry and trust studies

These are planned source-locked studies outside the fixed 53-job program above.
They do not redefine the canonical SR-SNAKE route or authorize submission until
their remaining numerical thresholds and exact manifests are reviewed.

### 1. No-overlap source-metric calibration of the trust radius

- The potential canonical controller uses the source-point supported Gram
  metric to compare the Phase-III proposed coordinate displacement with the
  displacement realized by the complete accepted Powell refit. It performs no
  endpoint-overlap evaluation and charges zero overlap queries.
- Define `chi_G = delta_real / (delta_pred + eps)`. Contract the radius by the
  inverse-square-root factor when `chi_G > 1`. Expand by the same inverse-
  square-root rule only when `chi_G < 1`, the proposed step was trust-boundary
  limited, and the accepted refit produced positive energy descent. Otherwise
  hold the radius. If no coordinate prediction exists on an exact fallback
  round, hold the radius with explicit no-query telemetry.
- Continue recording actual-versus-predicted energy agreement as a separate
  response-model diagnostic. It does not stand in for state displacement and
  does not trigger an overlap circuit in this route.
- Implementation and focused tests are complete. The source-value anchor is
  CHTC cluster `8911303`, batch
  `paper-i-hh-sr-no-overlap-trust-parent-anchor-ww-r50-20260720-v1`, source
  archive SHA-256
  `9f48b7ed5bd451090b93314d000bc23e80d937b629779d43dc5574cc9dc6635d`.
  The six-regime candidate fanout remains fail-closed until that anchor exactly
  reproduces the locked projected weak--weak terminal metric and ordered
  operator sequence.

### 2. Supported Phase III without explicit whitening

- Construct the Phase-III response from the full active logical ansatz plus the
  singleton candidate, then project onto the supported eigenspace of the FS
  Gram matrix and remove only genuine null/alias modes.
- In supported coordinates, solve the generalized FS trust problem directly,
  using the supported metric in the constraint and KKT system (schematically,
  `(H_s + lambda G_s) q = -g_s` with `q^T G_s q <= rho**2`). Do not apply the
  additional `G_s^(-1/2)` whitening/rescaling inside Phase-III candidate
  scoring.
- Retain supported-FS whitening for the complete accepted-ansatz Powell refit.
  That whitening remains optimizer preconditioning because Powell is
  coordinate-sensitive; removing it from the coordinate-invariant Phase-III
  trust solve does not imply removing it from the inner optimizer.
- Compare this route against the source-locked whitened Phase-III route with all
  other scientific settings fixed. Record supported ranks, selected operators,
  predicted steps/gains, realized gains, energy-error trajectories, and
  estimator work.
- The weak--weak source-value anchor reproduced the whitened parent result. The
  six-regime projected route is already running as cluster `8908614`; its first
  three regimes completed 50 scientifically valid rounds, while a stale
  post-run validator prevented only reporting sidecars. Preserve the completed
  science and validate it externally rather than rerunning it.

### 3. Explicit windowed-geometry query-reduction study

- Consider a separate, explicitly named geometry-window policy only after the
  full-response projected-generalized-trust route above is validated.
- This study must not restore the historical bug in which Phase-III response
  coordinates silently followed the Powell reoptimization window. Geometry
  scope must be an independent scientific control, while the accepted-ansatz
  refit remains the complete supported-FS-whitened refit.
- Test whether restricted geometry construction reduces measured quantum-oracle
  work without degrading attainment of the fixed target energy error. Report
  the complete energy-error trajectory, target-hit round, `S_alg` and its
  geometry contribution, selected operators, supported ranks, and compiled
  costs across the six regimes.
- Build this only after the full-geometry no-overlap candidate is validated.
  The initial window rule should retain every coordinate with material Gram
  overlap or Hessian coupling to the candidate, rather than coupling geometry
  to the Powell refit window. Use full-geometry refreshes when the supported
  rank changes or the retained-block telemetry fails its closure test. Exact
  thresholds will be chosen from the full-geometry telemetry before the bundle
  is built. The full active-plus-singleton response remains the comparison
  anchor.

## Physics grid and cutoff contract

The variational and exact-reference cutoffs are identical in every row:

| Regime | (U/t) | Holstein sector | `n_ph_work` | `n_ph_ref` | Horizon |
|---|---:|---|---:|---:|---:|
| weak--weak | 0.25 | weak | 3 | 3 | 50 |
| intermediate--weak | 1.25 | weak | 3 | 3 | 50 |
| strong--weak | 8 | weak | 3 | 3 | 50 |
| weak--strong | 0.25 | strong | 7 | 7 | 50 |
| intermediate--strong | 1.25 | strong | 7 | 7 | 50 |
| strong--strong | 8 | strong | 7 | 7 | 50 |

Primary error is always
`abs(E_method(n_ph_work) - E_exact(n_ph_work))`. A higher-cutoff reference must
not enter this comparison.

Changing the cutoff changes the physical Hilbert space. Every row is therefore
a fresh round-zero run; do not resume a prefix generated at another cutoff.

## Required pre-run (S_{alg}) and fidelity audit

This audit is a hard gate before any of the 53 full jobs are submitted.

### (S_{alg})

For SR-SNAKE, Geo-ADAPT, and Append-ADAPT, verify the executable accounting
implements the same definition:

\[
S_{alg}=N_{H,outer}+N_{H,refit}+N_{grad}+N_{metric}.
\]

The audit must prove:

- every energy, gradient, metric/geometry, warm-start guard, boundary refit,
  and final ordinary refit occurrence is charged exactly once under the stated
  convention;
- raw call occurrences and canonical same-state deduplication are both
  preserved;
- winning-lineage, discarded-branch, and total work are separated where the
  method can create discarded work;
- cache hits do not erase the underlying algorithmic measurement charge;
- exact diagonalization and post-run fidelity/reference evaluation are
  reporting work and do not enter (S_{alg}) or controller decisions;
- macro and projected-singleton comparator variants use the same accounting
  semantics; and
- per-round receipts sum exactly to the terminal ledger totals.

### Fidelity

For all three method paths, verify:

- fidelity is recomputed from the replayed variational state and the same-cutoff
  exact ground state, not copied from a stale summary;
- the reference degeneracy convention is explicit: prove the ground state is
  unique or use ground-space/projector fidelity rather than an arbitrary
  eigenvector from a degenerate eigenspace;
- state normalization, global phase, logical/runtime parameter projection,
  binary-padding projection, and fixed-sector enforcement are handled
  identically;
- terminal and selected-plateau fidelity values identify their exact prefix;
- fidelity/reference data remains reporting-only and never enters selection,
  stopping, or optimizer decisions; and
- independent replay recomputation agrees with the stored value within the
  established numerical tolerance.

Add focused cross-method tests and one small cache-off audit smoke per distinct
execution path. If any ledger does not close or fidelity cannot be reconstructed,
repair the shared instrumentation before generating full-run bundles.

Completed audit evidence:

- four cache-off comparator-path smokes and cross-method validation:
  `raw_outputs/paper_i_hh_completion_comparator_cacheoff_smokes_20260718_v3/validation_summary.json`;
- exact scalable Geo projected-singleton parity rerun:
  `raw_outputs/paper_i_hh_completion_comparator_cacheoff_smokes_20260718_v4_span_parity/validation.json`;
- the parity rerun reproduces both selected labels, final energy, same-cutoff
  error, fidelity, final projective state, and
  `S_alg=2635919`; its focused regression reports 79 passed tests; and
- frozen scalable-selector source hashes are
  `pipelines/exact_bench/generic_static_adapt_variants.py =
  33f0e8ffba1d532e86037077e99d8578423fc7a52842e2479a40afea1588ed3d`
  and `test/test_generic_static_adapt_variants.py =
  790fd3bea888c444883d7677ba0418b2112781fec65e8283fe6a88a9935d1c19`.

## Geo-ADAPT and Append-ADAPT comparator contract

The comparator matrix has four explicit variants, each covering all six
regimes through iteration 50:

| Display row | Method id | Candidate representation | Jobs |
|---|---|---|---:|
| Geo-ADAPT singleton | `static_geo_adapt_vqe` | projected singleton children derived immediately from `full_meta` | 6 |
| Geo-ADAPT macro | `static_geo_adapt_vqe` | unsplit `full_meta` macro generators | 6 |
| Append-ADAPT singleton | `static_full_meta_append_adapt_vqe` | projected singleton children derived immediately from `full_meta` | 6 |
| Append-ADAPT macro | `static_full_meta_append_adapt_vqe` | unsplit `full_meta` macro generators | 6 |

Both representations begin from the same unfiltered `full_meta` parent family
with HVA exposure. The singleton variant exposes the symmetry- and
padding-valid projected singleton children as the candidate pool immediately;
it must not silently mix in macro parents. The macro variant admits the parent
macro generators and must not silently substitute child candidates.

Before building the bundles:

1. resolve the best visible Paper-I Geo-ADAPT and Append-ADAPT settings to their
   source JSONs with
   `agent_guidance/skills/shared/scripts/resolve_visible_settings.py`;
2. preserve each comparator's method-defining repeat and insertion semantics;
3. match the final SR-SNAKE physics, encoding, symmetry/padding enforcement,
   optimizer family and budget, seed 7, HH-preseed-off initialization, fixed
   horizon, exact-reference convention, and compilation convention;
4. create a normalized manifest and settings-difference audit for all 24 rows;
   and
5. fail closed on mixed cutoffs, ambiguous legacy regime aliases, pool mixing,
   missing accounting receipts, or unreconstructable fidelity.

The older comparator bundle
`chtc/phase3_optuna/input/paper_i_hh_fullmeta_singleton_symmetry_corrected_parent_iter50_20260710_v1/`
is provenance only. Its mixed working/reference cutoffs do not satisfy this
3/7 same-cutoff contract.

Prepared Geo comparator authority:

- bundle:
  `chtc/phase3_optuna/input/paper_i_hh_geo_comparators_macro_projected_singleton_all_six_r50_20260718_v1_chtc/`;
- source-archive SHA-256:
  `8922435b176d635544f6fa2629da05ea7151f457e584c39e47a2ee161de94ecd`;
- bundle-manifest SHA-256:
  `50a3b39d7f7fa12b4b09424fc005efd571d527792ecc431bc4001a9dcb3d58df`;
- settings-audit SHA-256:
  `b8a44d825d0c5d6e0cb09bf87492d55f5857e7db74c3c9a56cbf7789d69b5fe0`;
- validation: 19 bundle tests and 112 extracted archive-only comparator
  regressions passed; all six visible Geo source locks resolve and hash-match,
  with only their obsolete iCloud paths repaired; the source archive, queue,
  normalized job manifests, and physics are unchanged by the fail-closed
  remote-gate plumbing; and
- status: prepared but not submitted; only the authenticated remote
  image/Qiskit gate and deliberate submission enablement remain.

Historical prepared Append comparator authority (superseded by the
2026-07-29 stationary-core v6 component adoption):

- bundle:
  `chtc/phase3_optuna/input/paper_i_hh_append_comparators_macro_projected_singleton_all_six_r50_20260718_v1_chtc/`;
- source-archive SHA-256:
  `8922435b176d635544f6fa2629da05ea7151f457e584c39e47a2ee161de94ecd`;
- bundle-manifest SHA-256:
  `c6278f8e7087f7a11403769416f302cadb7b4af0ace5c6f2774ec101425ba177`;
- validation: 18 bundle tests and 108 archive-only comparator/fidelity
  regressions passed; all six visible Append source locks hash-match and the
  settings-difference audit reports no unapproved drift; the source archive,
  queue, normalized job manifests, and physics are unchanged by the
  fail-closed remote-gate plumbing; and
- status: prepared but not submitted; only the authenticated remote
  image/Qiskit gate and deliberate submission enablement remain.

The cross-bundle preflight over the prepared main SR, Geo, Append, and
historical-beam bundles passes all 36 job contracts: every archive and per-file
inventory hash closes; every job is fresh round zero through 50 with seed 7,
Powell `maxiter=200`, and the exact 3/3 or 7/7 same-cutoff physics grid; all
scientific blocker lists are empty; and every submit description remains
fail-closed pending authenticated remote validation and deliberate enablement.

## Main five-row comparison

The main comparison contains exactly these rows:

1. SR-SNAKE;
2. Geo-ADAPT projected singleton;
3. Geo-ADAPT macro;
4. Append-ADAPT projected singleton; and
5. Append-ADAPT macro.

For every row and regime preserve and report:

- same-cutoff absolute energy-error trajectory through round/iteration 50;
- selected plateau prefix and round-50 terminal prefix;
- fidelity and infidelity at both reported prefixes;
- accepted ansatz depth and ordered operator/parameter history;
- (S_{alg}), including winning-lineage, discarded, and total work where
  applicable;
- fixed-sector and binary-padding leakage;
- optimizer `nfev`, termination details, and stopping reason;
- Qiskit `N2q`, `D2q`, and total circuit depth at both the selected plateau and
  round 50 under one compile convention; and
- strict replay, normalized manifest, source/settings hashes, and exact source
  inventory.

If no objective plateau rule selects a prefix, mark the plateau endpoint
unresolved rather than inventing one; the round-50 costs remain mandatory.

The previously completed ordinary-novelty comparison is supporting evidence,
not another tracker. Its immutable result and Qiskit provenance remains in
`paper_i_no_ordinary_novelty_sr_snake_evidence_copy_20260717.json`.

## Appendix SR-SNAKE ablations

Run all three as separate one-factor variants of the exact main SR-SNAKE source
lock. Each variant covers all six regimes from round zero through round 50.

### Pruning enabled

- Change only the approved pruning/deletion-model fields.
- Keep batching and beam in the main-route off state.
- Do not add terminal pruning, terminal refitting, admission rollback, or any
  other stopping-time mutation.
- Preserve nomination, trust-model, delete-and-refit, acceptance/rejection,
  leakage, and accounting telemetry.

Prepared pruning authority:

- bundle:
  `chtc/phase3_optuna/input/paper_i_hh_sr_snake_appendix_fs_prune_nodamping_nobeam_nobatch_no_ordinary_novelty_all_six_r50_20260718_v1_chtc/`;
- route-contract SHA-256:
  `272ede635558edb4acc2507ac3a9803d8ccec062b96c98634b8d6407df9fbc21`;
- source-archive SHA-256:
  `1d6e93bd59f97f74cc444c6c3559b15d48053b2c4914736a3c32b0e0869a196a`;
- bundle-manifest SHA-256:
  `8974b4dd5417c7e42bdb8f0077ba0420a011de874641d3747f94d57fd92628ba`;
- validation: the exact settings audit changes only the 12 approved pruning
  fields; damping is zero/off; live-only full-logical deletion trust begins at
  radius `0.125`; measured delete-and-refit is the acceptance authority; all
  10 live focused tests, 130 archive profile/route tests, 108 runtime tests
  (9 skipped), 6 bundle tests, six archive-only manifest parses, and all nine
  verifier gates passed; and
- status: prepared but not submitted; only the authenticated remote
  image/Qiskit gate and deliberate submission enablement remain.

### Batching enabled

- Change only the explicitly approved batching fields.
- Keep pruning and beam in the main-route off state.
- Enable batching only for the Phase-III post-shortlist admission step. Keep
  Phase II singleton and explicitly non-batched.
- Use `phase3_batch_selection_mode=combinatorial_reduced_plane`,
  `phase3_batch_target_size=3`, and `phase3_batch_size_cap=3`; the projected
  child subset cap remains one. The legacy Phase-II batching aliases must
  resolve to the same non-batched Phase-II contract and may not silently turn
  on a second batching stage.
- Record proposed/accepted batches, batch sizes, extra optimization and
  estimator work, trajectory changes, and compiled costs.
- Serialize the resolved Phase-III-only scope and fail closed on contradictory
  Phase-II/Phase-III batching flags.

Active repaired Phase-III-batching authority:

- combinatorial bundle:
  `chtc/phase3_optuna/input/paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260720_v9_chtc/`;
- combinatorial CHTC cluster: `8911235`;
- combinatorial route-contract SHA-256:
  `27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050`;
- combinatorial source-archive SHA-256:
  `be62f877537c4b88b154dbe0e802a1029197a94b84115997dca1820f2cedcc80`;
- greedy bundle:
  `chtc/phase3_optuna/input/paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260720_v8_chtc/`;
- greedy CHTC cluster: `8911236`;
- greedy route-contract SHA-256:
  `ed3afaeffa076cfce6b3eb63bf912dd182768fd109b3cc2be5878f23b335c865`;
- greedy source-archive SHA-256:
  `14a1fd9617634a8ae2ca1b1d5f3971ffe8d74bf4203ed3226ad54de72281771f`;
- validation: both immutable repairs pass from extracted archives locally and
  in the exact CHTC image. The repair restores the complete accepted coordinate
  order from the already certified selected subset and changes no selector,
  model, batching, cutoff, or horizon setting. Each cluster contains exactly
  six 50-round rows and entered idle with no holds.

### Beam enabled

- Change only the explicitly approved live-branch, children-per-parent, and
  terminated-branch-retention fields.
- Keep pruning and batching in the main-route off state.
- Use the settled historical beam profile: three live branches, two children
  per parent, at most six expanded children per controller round, retain at
  most three live branches, and `adapt_beam_lambda=0.005`.
- Preserve its exact historical terminal semantics: materialize each expanded
  parent's explicit stop/terminated child even when admission proposals exist,
  carry prior terminal children forward across rounds, and prune the cumulative
  terminal archive to three. This is the historical `stop_or_single_admission`
  contract, not a newer retention-only approximation.
- Record the complete frontier lineage, winning branch, discarded work,
  estimator charges, trajectories, fidelity, and compiled costs.
- Preserve the historical terminated-branch-retention semantics explicitly in
  the source lock; do not substitute a new retention policy or infer one from
  defaults.

Prepared historical-beam authority:

- bundle:
  `chtc/phase3_optuna/input/paper_i_hh_sr_snake_appendix_historical_beam3x2_full_response_symmetric_cost_noprune_no_ordinary_novelty_all_six_r50_20260718_v2_chtc/`;
- route-contract SHA-256:
  `f932974ad3cdbd3b1b38239794cc9e7ab96a94502b53238bcdf5c5760f814a80`;
- source-archive SHA-256:
  `942838bf460a1804e98c1b9893d89a8dd1001aa8beefe4bbac0c3b6a625dbe2e`;
- non-scientific `S_alg` accounting-repair overlay SHA-256:
  `7dd0532449388d7c9359c579fda2f25f57153f83c7e3ebbb1aeba66aa0019652`;
- validation: all six archive-only job parses passed, 535 source-locked
  regressions passed (9 skipped), and all 18 bundle/corruption-mutation tests
  passed; and
- status: submission-ready but not yet submitted; the scientific route digest
  is unchanged.

The byte-preserved v1 bundle remains an archival pre-accounting-repair source
lock only. Its archive SHA-256 is
`021a86952e5a69b1beda8592ba3b1b7bdb91ef75d1278912a0c1cfce9d1a666c`;
do not submit it or use it for `S_alg` evidence.

The live historical-beam replay profile now resolves to the same exact
`f932974ad3cdbd3b1b38239794cc9e7ab96a94502b53238bcdf5c5760f814a80`
contract as the immutable bundle. It explicitly materializes every expanded
parent's stop child, accumulates the terminated archive across rounds with cap
three, preserves ordered-batch telemetry precedence, and reports
`stop_or_single_admission` for the non-batched historical route. The focused
profile/engine/prune suite passed 223 tests in the final main-agent check, and
the actual two-round legacy-beam behavior test passed separately.

Do not combine pruning, batching, and beam in this program. Those combinations
would be different methods and are outside the current completion queue.

## Visible Paper-I weak--weak mechanism-ablation table refresh

The active rendered ablation target is the main-body table
`tab:weak_weak_snake_mechanism_ablation_terminal_metrics` in
`MATH/paper_details/Paper_I.tex`. It is a weak--weak-only table of differences
from an omitted source anchor at controller iteration `k=30`. Its current
support source is:

```text
output/pdf/paper_i_hh_weak_weak_snake_mechanism_ablation_20260708/
  paper_i_hh_weak_weak_snake_mechanism_ablation_20260708.json
```

That support JSON belongs to the older `batch_cap3_combinatorial` route and uses
the old estimator-work accounting. None of its numeric deltas may be carried
forward. The manuscript pointer records SHA-256 `07304f5b...`, while the current
support JSON hashes to
`a8c64412614f13149933e981d7e1047402dff39f95f1ecdca503554430e7be79`;
the stale pointer is provenance debt to repair only if the user later
authorizes a manuscript update.

The current canonical main route already has batching off, pruning off, and a
full active-plus-singleton Phase-III response. Therefore the old displayed
rows `No batching reference`, `No prune`, and `Full geometry window` are no
longer nontrivial ablations. The refreshed source-locked matrix is:

| Refreshed row | Relation to current canonical anchor | Execution source | Current status |
|---|---|---|---|
| Source anchor (omitted from delta table) | no settings change | weak--weak cell of ordered item 3 | pending |
| Phase-III batching enabled | reverse of the old no-batching row | weak--weak cell of ordered item 10 | bundle ready; not submitted |
| Pruning enabled | reverse of the old no-prune row | weak--weak cell of ordered item 9 | bundle ready; not submitted |
| No cost term | disable only the symmetric hardware-cost shaping | new weak--weak row in ordered item 12 | not built or submitted |
| No Phase III | disable the Phase-III selection stage through its explicit historical ablation contract | new weak--weak row in ordered item 12 | not built or submitted |
| Phase I only; macro pool | retain only Phase I and expose the macro parent pool | new weak--weak row in ordered item 12 | not built or submitted |
| Phase I only; projected-singleton pool | retain only Phase I and expose the symmetry/padding-valid projected singleton pool | new weak--weak row in ordered item 12 | not built or submitted |
| Legacy fixed local Phase-III response window | replace canonical full active-plus-singleton response scope with the explicit historical local-window policy | new weak--weak row in ordered item 12 | not built or submitted |

This is seven displayed perturbations plus one omitted anchor, matching the
current table shape without preserving three now-degenerate row definitions.
The beam-on route and the ordinary-novelty-on evidence remain separate unless
the user explicitly expands the rendered row set.

Every job still runs through exactly 50 completed controller rounds without a
terminal refit, terminal prune, finite-angle probe, or plateau stop. For direct
continuity with the current rendered table, the support matrix must reconstruct
the exact `k=30` accepted checkpoint and report deltas there; it must also retain
the round-50 endpoint as validation/context evidence. The anchor may be reused
from the main weak--weak run rather than executed twice.

Before any of the five new rows is built or submitted:

1. materialize the complete normalized main weak--weak manifest and hashes;
2. create a `source_locked_sensitivity_audit_v1`-compatible per-row diff whose
   approved changed-field set is exactly the named mechanism contract;
3. prove all non-ablated physics, cutoff `3/3`, seed, pool, symmetry/padding,
   optimizer and budget, response/refit, fallback, stopping, and compilation
   fields are identical to the main source lock;
4. require the repaired estimator ledger to close at every round and at both
   displayed endpoints;
5. when ordinary novelty is off, prove ordinary Phase-II/III novelty solves
   and query charges are zero rather than computing novelty and multiplying by
   one; preserve lazy all-energy-models-infeasible fallback telemetry and flag
   any activation; and
6. report same-cutoff error, fidelity, accepted depth, ordered history,
   winning/discarded/total `S_alg`, leakage, optimizer receipts, and Qiskit
   `N2q`, `D2q`, and total depth at `k=30` and `k=50`.

Existing displayed values remain in place until all eight refreshed endpoints
are validated and the user separately authorizes a manuscript/table update.

## Final comparison PDF and source inventory

Build one dedicated LaTeX comparison report after all 53 scientific rows pass
validation. It must:

- lead with energy-error trajectories and the compact five-row main comparison;
- show fidelity, (S_{alg}), and plateau/round-50 Qiskit costs prominently;
- contain separate appendix sections for pruning, batching, and beam efficacy;
- use a two-column or Paper-I-like reader-facing layout;
- place the parameter manifest and agent-facing source inventory at the end,
  not on the first page;
- preserve a machine-readable JSON/CSV source map for every plotted point and
  displayed cell; and
- receive a staged build, PDF render, and visual inspection before replacing
  the report target.

Completion stops here. Do not edit `Paper_I.tex`, rebuild `Paper_I.pdf`, create
a manuscript copy, or promote results from this tracker without a later
explicit user request.

## Settled secondary-prefix reporting rule

The selected plateau prefix is the first accepted history prefix whose
same-cutoff absolute energy error is at most `1.10` times the minimum
same-cutoff error observed over the complete 50-round trajectory:

\[
k_{\rm pl}=\min\left\{k:\left|\Delta E_k\right|\leq
1.10\min_{1\leq j\leq 50}\left|\Delta E_j\right|\right\}.
\]

This rule is reporting-only. It cannot change candidate selection, pruning,
refitting, beam retention, or stopping, and every job must still finish and
report round 50. Fidelity, energy error, ansatz, and Qiskit costs displayed for
`k_pl` must all come from that exact reconstructed prefix state.

By explicit user decision on 2026-07-29, the main Paper-I
Hubbard--Holstein manuscript comparison instead reports every circuit resource
and \(S\) at the fixed endpoint \(k=50\). The selected-plateau rule above is
retained only for secondary diagnostic artifacts that explicitly invoke it.

## Explicitly outside this completion queue

- damping-coefficient sweeps;
- negative-curvature, saddle-escape, or finite-angle exploration;
- JR-SNAKE and propagated-manifold/FM development;
- combined prune-plus-batch, prune-plus-beam, or batch-plus-beam routes;
- beam-width sweeps beyond the single approved beam-on profile;
- cross-optimizer matrices; and
- manuscript editing or result promotion.

## Remaining definitions before bundle construction

None. The main SR-SNAKE route authority pointers above are frozen. Remote
execution validation and submission remain operational gates, not unresolved
scientific definitions.

Do not create another tracker for these decisions. Update this file in place.

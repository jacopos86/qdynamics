# SR-SNAKE `S_alg` quantum-oracle accounting audit — 2026-07-18

## Current repair status

**SOURCE REPAIR AND THE IMMUTABLE HISTORICAL-BEAM SUCCESSOR BUNDLE PASSED.**
The live source now records the
two omitted decision-authoritative primitive classes, canonicalizes the
candidate/accepted tangent alias, atomically checkpoints the identity ledger,
and restores cumulative receipt state across continuation. The immutable v1
bundle audited below remains unchanged and must not be submitted; its v2
successor contains the repaired accounting contract and strict fetched-evidence
validator.

The corrected logical-oracle contract is:

- every active-coordinate gradient consumed by the full Phase-III response is
  recorded in `N_grad` with a physical-tangent v2 identity;
- every endpoint overlap used by adaptive trust is recorded once in
  `N_metric`, with formal query category `N_cross` and the explicit protocol
  `logical_projective_overlap_magnitude_oracle_v1`;
- candidate and zero-amplitude accepted-refit metric rows share the same
  physical-tangent identity, so the accepted refit reuses the Phase-III row;
- an authenticated resumed prefix initializes the receipt cursor from the
  restored ledger instead of zero; when historical artifacts lack the full
  per-round receipt list, one explicit compacted prefix receipt preserves exact
  cumulative closure;
- checkpoint ledgers are content-addressed and authenticated before the
  checkpoint that references them is published.

Ordinary Phase-II and Phase-III novelty under `fallback_only_v1` is genuinely
not computed and then neutralized. The ordinary projection solve is skipped,
its measured value and multiplier remain null, and its classical-solve/query
counts remain zero. Gram and Hessian construction remains because it is
independently required by curvature response, supported whitening, and the FS
trust constraint. Only the all-energy-models-infeasible fallback may lazily
solve for novelty; it reuses already acquired geometry and adds zero quantum
queries.

Focused live-source verification completed during the repair:

```text
45 passed
  estimator-ledger unit tests plus beam/non-beam accepted-refit accounting

4 passed
  Phase-II/III fallback-only novelty execution contract

124 passed, 4 skipped
  estimator ledger, accepted refit, and resume/checkpoint regressions

243 passed
  continuation scoring, historical singleton overlays, and SR-v4 runtime
  regressions
```

One diagnostic cache-off, eight-admission weak--weak smoke was launched. No
CHTC submission was made during this repair.

## Fresh repaired depth-eight verification

The repaired smoke is stored at
[weak_weak_8_admissions_cache_off_v1](../../../raw_outputs/paper_i_hh_sr_snake_s_alg_repair_smoke_20260718/weak_weak_8_admissions_cache_off_v1/).
It retained the same route-contract digest, selected the same eight operators
in the same order, used the same 1,588 optimizer evaluations, and reproduced
the frozen smoke energy exactly:

\[
E=-0.9173328578719391,\qquad
|E-E_{\mathrm{ref}}|=1.0480621228828868\times 10^{-3}.
\]

The repaired unique-oracle ledger closes as

\[
6609=1+1467+1070+4071,
\]

for `N_H_outer`, `N_H_refit`, `N_grad`, and `N_metric`, respectively. Its raw
occurrence count is 9,329. The apparent reduction from the frozen ledger's
7,312 unique primitives is a physical-identity repair, not missing work:

- the 28 previously omitted active-coordinate gradients are present and all
  are newly charged;
- the eight decision-authoritative endpoint overlaps are present and all are
  newly charged;
- all 120 accepted-refit Gram occurrences reuse already acquired physical
  tangent primitives, whereas the frozen ledger incorrectly charged 36;
- 148 newly exposed full-population Phase-III candidate-geometry occurrences
  are present as explicit reuse receipts and therefore add zero unique
  charges;
- label-free physical-tangent identities remove artificial duplication across
  Phase I, Phase II, Phase III, runtime-child, and accepted-refit consumers.

The raw occurrence increase is exactly `148 + 28 + 8 = 184`. The unique-count
decrease is exactly accounted for by the per-scope charged deltas; energy
components are unchanged. Collision-safety checks confirm that the identity
still includes the projective state, Hamiltonian/backend/precision contract,
primitive kind and formula, ordered derivative circuit and angles, generator
coefficients and qubit count, insertion position, and parameter tie map.

Novelty-off execution was also audited across the complete smoke result:

- both ordinary policies are `fallback_only_v1`;
- all 190 Phase-II and 195 Phase-III status records say
  `not_computed_for_ordinary_scoring`;
- every ordinary novelty value and multiplier remains null;
- ordinary novelty classical-solve count and query charge are both zero;
- the all-energy-models-infeasible safety fallback was enabled but fired zero
  times and charged zero queries.

The final sidecar passed strict `EstimatorCallLedger.from_payload`
reconstruction. Its SHA-256
`c10a7912efe8174e092ed93af33ff398d054740d31e9c78893077bbfb9632841`
matches the pointer serialized in `result.json`; its ledger fingerprint is
`5e0eedd40aacbfc7d5782a3a5ada69bfcb34334a4b2b54eb102e588aa1082140`.

## Original frozen-bundle verdict

**FAIL for complete physical quantum-oracle accounting.** The immutable route's ledger arithmetic, identity serialization, and fresh-run total closure are internally consistent, but the inventory of decision-authoritative estimator primitives is not complete and one physical Gram row is counted twice under two coordinate labels. The pending six-regime submission remains paused.

No source code was changed, no scientific job was launched, and no CHTC submission was made during this audit.

## Audited route and source authority

- Bundle: `paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v1_chtc`
- Batch: `paper-i-hh-sr-main-fullresp-symcost-noprune-nobeam-nonovelty-six-r50-20260718-v1`
- Requested profile: `sr_snake_no_prune_symmetric_cost_v1`
- Resolved profile: `supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_v1`
- Route-contract SHA-256: `69dcf6d5711a05b54cc143562bc6f437511d781164319b2705857b22deda1538`
- Immutable source archive SHA-256: `fa9014b9608ccb8a301df0429268482abf6dd10c91eb336111a15d00be256d35`
- Source commit metadata: `8a746d244a15e2cb16099a732e78e1110a8e59f2`

Authoritative artifacts:

- [Bundle manifest](../../../chtc/phase3_optuna/input/paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v1_chtc/bundle_manifest.json)
- [Source archive manifest](../../../chtc/phase3_optuna/input/paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v1_chtc/source_archive_manifest.json)
- [Immutable source archive](../../../chtc/phase3_optuna/input/paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v1_chtc/source_locked.tar.gz)
- [Eight-admission smoke result](../../../raw_outputs/paper_i_hh_sr_snake_no_prune_symmetric_cost_weak_weak_smoke_20260717/weak_weak_8_admissions_cache_off_v3/json/result.json)
- [Eight-admission estimator ledger](../../../raw_outputs/paper_i_hh_sr_snake_no_prune_symmetric_cost_weak_weak_smoke_20260717/weak_weak_8_admissions_cache_off_v3/json/estimator_call_ledger.json)
- [Bundle smoke-evidence record](../../../chtc/phase3_optuna/input/paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v1_chtc/source_lock/local_smoke_evidence.json)

## What `S_alg` currently means

The frozen ledger reports

\[
S_{\mathrm{alg}}
=N_{H,\mathrm{outer}}+N_{H,\mathrm{refit}}+N_{\mathrm{grad}}+N_{\mathrm{metric}}.
\]

It counts each unique scalar prepared-state/observable primitive once under an assumed perfect result cache. `N_metric` is a legacy bucket that also contains Hessian primitives; `primitive_kind` preserves the distinction. Every logical request is retained separately as an occurrence. Therefore:

- unique-ledger `S_alg` is an information count under perfect caching;
- occurrence count is the number of recorded logical estimator requests;
- neither is a shot count or compiled-circuit count.

The identity key includes the projective state, Hamiltonian/backend/precision, primitive kind, observable or derivative formula, and coordinate identities. Branch and scope labels are consumers rather than physical identity fields.

## Operations correctly recorded

| Operation | Ledger component | Audit result |
|---|---|---|
| Initial and outer-refresh Hamiltonian energies | `N_H_outer` | Recorded |
| Powell/refit objective energies | `N_H_refit` | Recorded |
| Final energy verification | `N_H_refit` | Recorded |
| Pool and candidate gradients | `N_grad` | Recorded |
| Candidate self metric | `N_metric` | Recorded |
| Phase-II/III candidate Gram and Hessian elements | `N_metric` | Recorded |
| Active-active scaffold Gram and Hessian elements | `N_metric` | Recorded |
| Accepted-refit full Gram elements | `N_metric` | Recorded, with the alias defect below |
| All shortlisted candidates rather than only the winner | all applicable | Recorded |
| Symmetric hardware-cost shaping | none | Correctly excluded; classical |
| Same-cutoff reference and post-run fidelity | none | Correctly excluded; reporting-only |
| Infeasible-model novelty fallback | zero incremental | Correct for this design because it reuses acquired geometry; it did not fire in the smoke |
| Pruning, beam, batching, finite-angle probing, Phase-III rescue | none | Correctly absent because all are disabled in this route |

Because this route is effective `1x1` with pruning off, all-search and winning-lineage `S_alg` should coincide and discarded branch work should be zero.

## Exact depth-eight smoke reconstruction

Strict deserialization of the smoke ledger with the immutable ledger implementation passed and reconstructed

\[
7312=1+1467+1156+4688.
\]

The four terms are `N_H_outer`, `N_H_refit`, `N_grad`, and `N_metric`. The raw recorded occurrence count was

\[
9145=10+1587+1220+6328,
\]

so the ledger reused 1,833 repeated same-identity requests.

Hamiltonian occurrence accounting also closes exactly:

- 1 initial energy evaluation;
- 1,587 optimizer/refit energy evaluations;
- 8 round-level outer refreshes;
- 1 final verification;
- 1,597 Hamiltonian-energy occurrences total.

The recorded optimizer/guard `nfev` total is 1,588: the initial evaluation plus the 1,587 optimizer/refit evaluations. This is internally consistent, but internal closure does not prove that every physical oracle primitive entered the inventory.

## Physical accounting defects

### 1. Phase-III active-coordinate gradients are uncharged

The full-response scorer constructs the active-coordinate gradient vector

\[
(g_A)_i=-2\,\operatorname{Re}\langle \partial_i\psi|H|\psi\rangle
\]

for every active logical coordinate, concatenates it with the candidate gradient, and uses the joint vector in the candidate response solve. The ledger records the candidate/pool gradients, Gram elements, and Hessian elements, but the active-gradient vector is absent.

Frozen-source anchors:

- `pipelines/scaffold/hh_continuation_scoring.py`, lines 5608–5622: active-gradient construction;
- same file, lines 10176–10182: active and candidate gradients enter the joint solve;
- `pipelines/static_adapt/adapt_pipeline.py`, lines 10493–10519: scaffold receipt records only Gram and Hessian primitives.

For one singleton admission per round through depth \(D\), the omitted active-gradient count is

\[
\sum_{k=0}^{D-1} k=\frac{D(D-1)}2.
\]

This is 28 missing scalar primitives at depth 8 and 1,225 at depth 50.

### 2. Adaptive trust uses an uncharged endpoint-overlap oracle

After every accepted refit, the adaptive-trust controller computes

\[
d_{\mathrm{FS}}=\arccos\!\left|\langle\psi_{\mathrm{before}}|\psi_{\mathrm{after}}\rangle\right|
\]

and uses it to update the next trust radius. This overlap is therefore decision-authoritative, not a reporting-only diagnostic, but no overlap primitive is written to the estimator ledger.

Source anchors:

- [route_a_trust_region.py](../../../pipelines/static_adapt/route_a_trust_region.py#L78), lines 78–91: exact endpoint Fubini–Study distance;
- same file, lines 1060–1067 and 1326–1410: the endpoint displacement enters the radius update;
- frozen `pipelines/static_adapt/adapt_pipeline.py`, lines 41255–41408: trust update is called without an overlap-ledger record.

The depth-eight smoke records `endpoint_fubini_study_distance_v1` as the displacement-ratio metric on every round. Round 8 changed \(\rho\) from `0.5967720126` to `0.5959433028`, proving controller authority.

At the ledger's scalar-oracle abstraction, at least one overlap primitive is missing per accepted round: 8 at depth 8 and 50 at depth 50. A hardware-faithful circuit/shot accounting may charge more, depending on the selected overlap-estimation protocol.

### 3. The accepted candidate's Gram row is charged twice under coordinate aliases

Phase III records the winning candidate metric row as `candidate:<generator-hash>`. Immediately after zero-initialized admission, the supported-FS accepted-refit path records the same physical tangent row at the same projective state as `active:<new-index>:<generator-hash>`. Because these strings are part of the primitive identity, the ledger cannot reuse the already acquired row.

Frozen-source anchors:

- `pipelines/static_adapt/adapt_pipeline.py`, lines 10437–10449: distinct candidate and active coordinate identities;
- same file, lines 10521–10581 and 36179–36212: Phase-III candidate Gram receipts;
- same file, lines 10381–10418, invoked at 10173–10194 and 40142–40153: accepted-refit full-Gram receipts.

At depth 8 the accepted-refit path records 120 Gram occurrences. The 84 old-old entries reuse prior identities correctly, while 36 entries in the newly admitted coordinate's row are charged again. Through depth \(D\), the overcharge is

\[
\sum_{d=1}^{D}d=\frac{D(D+1)}2,
\]

which is 36 at depth 8 and 1,275 at depth 50.

### 4. The numerical total is accidentally preserved

If one endpoint overlap is treated as one scalar primitive, then

\[
\underbrace{\frac{D(D-1)}2}_{\text{missing active gradients}}
+\underbrace{D}_{\text{missing overlaps}}
=\underbrace{\frac{D(D+1)}2}_{\text{duplicate Gram-row charge}}.
\]

Thus the depth-eight omissions \(28+8=36\) happen to cancel the 36 duplicate Gram charges. At depth 50, \(1225+50=1275\) would likewise cancel 1,275 duplicate charges.

This does **not** make the frozen accounting accurate. The primitive identities and component attribution are wrong; the equality depends on a one-scalar overlap convention and a complete one-admission-per-round path; and early stops or other controller paths can break it. The frozen aggregate should therefore not be used as evidence that the route's physical oracle workload is correctly recorded.

## Serialization, recovery, and fetched-validation audit

The ledger implementation itself is strict: its deserializer verifies schema, component contract, unique identities, contiguous occurrences, summaries, and the ledger fingerprint. The surrounding run plumbing does not consistently preserve that strength.

| Gate | Evidence | Result |
|---|---|---|
| Final sidecar emission | Complete ledger is written only after ADAPT returns | Abrupt failure can leave checkpoint receipts without a resumable identity ledger |
| Resume reconstruction | Resume requires and strictly reconstructs a sibling ledger sidecar | Good when the complete sidecar exists |
| Per-round receipt cursor after resume | Restored ledger is followed by unconditional reset of the receipt cursor/list | First resumed receipt is attributed against zero rather than the restored prefix |
| Bundle receipt arithmetic | Count, sequence, deltas, checkpoint equality, and final closure are checked | Strong aggregate/receipt validation |
| Fetched ledger identity validation | Validator checks totals, list shape, and nonempty fingerprint but does not call strict `EstimatorCallLedger.from_payload` | Fail-open to identity/fingerprint corruption |
| Ledger pointer integrity | Result pointer SHA is not compared with the fetched sidecar SHA | Fail-open to sidecar substitution or damage |

Mutation proof: changing the serialized ledger fingerprint and the first identity ID was still accepted by the bundle's aggregate validator, while the immutable ledger deserializer rejected it with `serialized ledger fingerprint mismatch`.

Relevant validators:

- [evidence_validation.py](../../../chtc/phase3_optuna/input/paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v1_chtc/evidence_validation.py#L145), lines 145–190 and 208–366;
- [validate_fetched.py](../../../chtc/phase3_optuna/input/paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v1_chtc/validate_fetched.py#L371), lines 371–380.

The immutable depth-eight smoke predates the active-prefix receipt overlay, so it cannot itself prove the new receipt overlay. The bundle's focused no-beam regression is the current evidence for that layer.

## Verification performed

- Strict reconstruction of the 19.8 MB smoke ledger with the immutable `EstimatorCallLedger.from_payload`: passed.
- Sidecar SHA-256: `43198f427025fee730284de08a3d2044b2019ed41a0c8a34a68d46b301350847`, matching the bundle smoke pointer.
- Ledger fingerprint: `20d4fb18fbdd00d1cef18adb195fb97271491de195ee604446911a2472b49b14`.
- Frozen focused regressions:

  ```text
  PYTHONPATH=. pytest -q -p no:cacheprovider \
    test/test_static_adapt_estimator_call_ledger.py \
    test/test_static_adapt_accepted_refit.py
  37 passed in 7.29s
  ```

- Recorded archive-only preflight: 504 passed, 9 skipped.

These checks establish that the implemented ledger behaves as specified; they do not erase the missing-oracle and coordinate-alias defects in what is supplied to it.

## Clearance status before CHTC submission

| Condition | Current status |
|---|---|
| Record every active-coordinate Phase-III gradient | Passed in source tests and the fresh smoke |
| Serialize and charge the adaptive-trust endpoint-overlap primitive | Passed in source tests and the fresh smoke |
| Canonicalize candidate/accepted tangent identity | Passed; accepted-refit charged zero new Gram primitives in the fresh smoke |
| Strict fetched deserialization plus result-pointer SHA/size verification | Passed in the immutable v2 successor; fingerprint, identity, pointer-SHA, and pointer-size mutations are rejected |
| Atomic checkpoint ledger plus resume receipt continuity | Passed focused checkpoint/resume regressions |
| Fresh inventory, receipt, identity, and closure smoke | Passed for a fresh cache-off eight-admission run; successor-bundle mutation tests passed; resume continuity is covered by focused checkpoint/resume regressions rather than a new scientific continuation |

The repaired live-source ledger is accurate for the audited fresh smoke. The
immutable v2 successor is
`chtc/phase3_optuna/input/paper_i_hh_sr_snake_appendix_historical_beam3x2_full_response_symmetric_cost_noprune_no_ordinary_novelty_all_six_r50_20260718_v2_chtc/`.
Its source archive SHA-256 is
`942838bf460a1804e98c1b9893d89a8dd1001aa8beefe4bbac0c3b6a625dbe2e`,
its non-scientific accounting overlay SHA-256 is
`7dd0532449388d7c9359c579fda2f25f57153f83c7e3ebbb1aeba66aa0019652`,
and its scientific route digest remains
`f932974ad3cdbd3b1b38239794cc9e7ab96a94502b53238bcdf5c5760f814a80`.
All six archive-only job parses passed; source-locked regressions report 535
passed and 9 skipped; bundle and corruption-mutation tests report 18 passed.
The pre-repair v1 bundle remains byte-preserved and blocked from submission.

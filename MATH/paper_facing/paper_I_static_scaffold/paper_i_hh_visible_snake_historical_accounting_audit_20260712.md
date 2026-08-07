| Regime | $k_{\rm pl}$ / active depth | Displayed / exact replay / fixed-prefix attainability $\lvert\Delta E\rvert$ | $N_{2q}/D_{2q}/D_c$: displayed → prune-aware generic → FakeMarrakesh-1 | $S_{\rm alg}$: displayed / legacy subtotal / current definition | Route match | Evidence classification | Exact blockers | Exact source |
|---|---:|---:|---:|---:|---|---|---|---|
| weak–weak | 13 / 13 | 4.524e-4 / blocked / 4.006e-4 | 48/34/183 → 48/34/183 → 87/62/223 | 5,009 / 4,469 / unresolved | historical route resolved; JR unmatched | validated numerical source; methodologically unmatched; prefix state unreconstructible | signed prefix θ absent; no primitive ledger; guard/padding skipped; compiler mismatch | `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/weak_weak/json/result.json` |
| intermediate–weak | 10 / 10 | 2.158e-4 / blocked / 2.113e-4 | 38/30/133 → 38/30/133 → 62/51/176 | 4,142 / 3,740 / unresolved | historical route resolved; JR unmatched | validated numerical source; methodologically unmatched; prefix state unreconstructible | signed prefix θ absent; no primitive ledger; guard/padding skipped; compiler mismatch | `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/intermediate_weak/json/result.json` |
| strong–weak (U=8) | 11 / 9 | 1.591e-6 / blocked / 1.578e-6 | 44/37/200 → 38/29/147 → 71/49/164 | 4,126 / 3,678 / unresolved | historical route resolved; JR unmatched | validated numerical source; displayed resources stale; methodologically unmatched; prefix state unreconstructible | two accepted prunes ignored by display compiler; signed prefix θ absent; no primitive ledger; guard/padding skipped; compiler mismatch | `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/strong_weak/json/result.json` |
| weak–strong | 16 / 16 | 1.841e-2 / blocked / 1.841e-2 | 70/61/206 → 70/61/206 → 112/90/315 | 17,305 / 16,627 / unresolved | historical route resolved; JR unmatched | validated numerical source; methodologically unmatched; prefix state unreconstructible | signed prefix θ absent; no primitive ledger; guard/padding skipped; compiler mismatch | `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/weak_strong/json/result.json` |
| intermediate–strong | 28 / 27 | 6.871e-4 / blocked / 4.664e-4 | 150/121/540 → 144/123/549 → 281/214/687 | 28,617 / 27,387 / unresolved | historical route resolved; JR unmatched | validated numerical source; displayed resources stale; methodologically unmatched; prefix state unreconstructible | one accepted prune ignored by display compiler; signed prefix θ absent; no primitive ledger; guard/padding skipped; compiler mismatch | `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/intermediate_strong/json/result.json` |
| strong–strong (U=8) | 13 / 11 | 4.683e-5 / blocked / 4.213e-5 | 48/39/188 → 42/32/185 → 66/42/148 | 5,153 / 4,613 / unresolved | historical route resolved; JR unmatched | validated numerical source; displayed resources stale; methodologically unmatched; prefix state unreconstructible | two accepted prunes ignored by display compiler; signed prefix θ absent; no primitive ledger; guard/padding skipped; compiler mismatch | `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/strong_strong/json/result.json` |

# Paper-I visible Hubbard–Holstein SNAKE historical-accounting audit

The displayed energy is source-mapped and arithmetically consistent in all six rows. That does not validate the complete displayed tuple. Exact saved-state replay at the displayed prefix fails closed for all six because the history does not preserve the signed, unwrapped optimized prefix parameters. Three resource tuples are stale because the July 8 compiler inserted every admission through $k_{\rm pl}$ without applying accepted prune deletions. None of the six July 9 $S_{\rm alg}$ values is reconstructible under the current unique-scalar-primitive definition.

The fixed-prefix values in the opening table are fresh Powell attainability checks with the operator order held fixed; they are not exact historical replay. Full admissions, coefficients, insertion positions, prune events, active-order trajectories, replay results, settings, blockers, and SHA-256 records are in the companion [JSON](paper_i_hh_visible_snake_historical_accounting_audit_20260712.json).

## Provenance resolution

The resolved chain is:

1. `MATH/paper_details/Paper_I.tex`, visible rows under `fig:hh_main_results_composite`;
2. `BEGIN_MACHINE_READABLE_HH_PHYSICAL_LANE_DUPLICATE_UPDATE_20260708`, followed by the later governing `BEGIN_MACHINE_READABLE_PAPER_I_S_ACCOUNTING_CORRECTION_20260709`;
3. root `Paper_I_provenance.json`;
4. `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_snake_nobatch_duplicate_promotion_20260707.json`;
5. `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/source_lock_manifest.json` and `commands.json`;
6. the six row `result.json` files;
7. ordered `adapt_vqe.history` admissions, serialized Pauli terms, insertion positions, and accepted `post_admission_prune.decisions`;
8. the historical Table-I Qiskit compiler and this audit's prune-aware rebuilds.

The root sidecar is not the final authority for (S): it still contains the pre-correction July 8 values 7,144; 5,799; 5,933; 20,108; 33,487; and 7,434. The later Paper-I block displays 5,009; 4,142; 4,126; 17,305; 28,617; and 5,153. The root sidecar's stored hashes for the July 7 static `.tex`/`.pdf` and current manuscript copies also no longer match the files now present. The six result hashes, commands hash, source-lock hash, run-status hash, and comparison JSON hash do match their preserved records.

## Historical route actually executed

All six commands resolve to Route A with physical-operator-type lanes, `full_meta`, HVA included by the source-lock contract, Powell `maxiter=200`, no separate `maxfev` cap, a three-parameter optimization window, a periodic full refit every eight admissions, and a final full refit capped at 200 iterations. Historical SciPy `xtol`/`ftol` values were not serialized; the executable passed only `maxiter`, so their exact historical values depend on the unrecorded SciPy environment defaults.

The source shortlist values 24 and 12 were divided by physical-lane aggressiveness 3, giving effective Phase-I/II caps 8 and 4 and an effective Phase-II fraction (1/12). Every displayed admission remained in `core` with `stay_core`; the seed stage was skipped or empty. Runtime splitting used `shortlist_pauli_children_v1`, `archival_child_set_forward_v1`, maximum subset size one, with Phase-II and Phase-III batching disabled. Thus the historical mechanism is singleton admission, not JR batching. Beam settings were three live branches, two children per parent, and λ=0.005. Pruning used `recoverability_ladder_v1`, mode `both`, with remove/refit energy safety as the deletion authority.

The later July 9 canonical-settings block says full refit every iteration and no eight-iteration window cadence. That describes a later canonical target, not these preserved six commands. It cannot be substituted into the historical reconstruction.

| Regime | $U/t$ | $g/t$ | Working / same-cutoff ED $M$ | Exact same-cutoff $E_{\rm ED}$ | Reference state |
|---|---:|---:|---:|---:|---|
| weak–weak | 0.25 | 0.353553390593 | 2 / 2 | -0.9183531194991743 | `00000101` |
| intermediate–weak | 1.25 | 0.353553390593 | 2 / 2 | -0.49499563910866023 | `00000101` |
| strong–weak | 8.0 | 0.353553390593 | 2 / 2 | 0.5264587007998404 | `00000101` |
| weak–strong | 0.25 | 0.790569415042 | 4 / 4 | -1.138579200359263 | `0000000101` |
| intermediate–strong | 1.25 | 0.790569415042 | 4 / 4 | -0.6239104048313422 | `0000000101` |
| strong–strong | 8.0 | 0.790569415042 | 4 / 4 | 0.5205762777107088 | `0000000101` |

No separate higher reference cutoff is serialized for these rows. Paper I separately reports ED cutoff sensitivity over $M=1,\ldots,10$, but those diagnostics are not the displayed-error reference and must not be substituted for the same-cutoff energies above.

## Admission history versus active prefix

$k_{\rm pl}$ counts accepted outer iterations. It is not the number of operators surviving deletion. The complete ordered admission and active-prefix sequences, including Pauli coefficients, appear in the JSON. The accepted deletions that change the displayed circuit are:

| Regime | Prune outer iteration | Deleted admission | Deleted identity | Active depth at $k_{\rm pl}$ |
|---|---:|---:|---|---:|
| strong–weak | 4 | 2 | `hh_fermionic_reusable::opp_spin_assist_current_nn_up(0,1)::child_set[0]` | 9 |
| strong–weak | 10 | 8 | `hh_fermionic_reusable::opp_spin_assist_current_nn_up(0,1)::child_set[2]` | 9 |
| intermediate–strong | 20 | 13 | `paop_full:paop_disp(site=1)::child_set[12]` | 27 |
| strong–strong | 4 | 2 | `hh_fermionic_reusable::opp_spin_assist_current_nn_dn(0,1)::child_set[0]` | 11 |
| strong–strong | 10 | 8 | `hh_fermionic_reusable::bond_charge_current_nn_dn(0,1)::child_set[2]` | 11 |

Weak–weak, intermediate–weak, and weak–strong have no accepted deletion through their displayed prefix, so their admission sequence and active sequence coincide.

## Energy replay

For every row, `history[k_pl-1].energy_after_opt` minus the same-cutoff ED value reproduces the source `delta_abs_current` exactly and rounds to the displayed value. This validates the preserved numerical history record, not an independently reconstructed saved prefix state.

The displayed history rows retain initial angles, aggregate optimizer counts, serialized operators, and absolute prune witnesses, but not the signed unwrapped optimized parameter vector for the active prefix. Exact replay is therefore blocked, with no invented parameters. As a control on the reconstruction machinery, the terminal `logical_optimal_point` is present: all six terminal saved states replay, with the largest absolute energy discrepancy against `adapt_vqe.energy` equal to $4.22\times10^{-15}$.

| Regime | Source-history $\lvert\Delta E\rvert$ | Primary fresh-prefix Powell $\lvert\Delta E\rvert$ | Independent zero-start cross-check $\lvert\Delta E\rvert$ | Interpretation |
|---|---:|---:|---:|---|
| weak–weak | 4.5236490925915085e-4 | 4.005912579575499e-4 | 4.005912579088111e-4 | prefix can attain comparable/lower error; not saved-state replay |
| intermediate–weak | 2.1581402163145524e-4 | 2.1126594417625322e-4 | 2.1126594417625322e-4 | same |
| strong–weak | 1.5912792076244742e-6 | 1.5778937580979147e-6 | 1.5778937577648477e-6 | same |
| weak–strong | 1.8409726432997653e-2 | 1.840972738294333e-2 | 1.8409727304316004e-2 | same order; optimizer termination changes the last digits |
| intermediate–strong | 6.870740365135797e-4 | 4.664314790632229e-4 | 4.726699024675263e-4 | fresh reoptimization is optimizer-convention sensitive; not historical replay |
| strong–strong | 4.682526561594624e-5 | 4.212677646409091e-5 | 4.21267765631228e-5 | prefix can attain comparable/lower error; not saved-state replay |

## Qiskit compilation

The values displayed in Paper I were produced by `table_i_basis_gate_transpile_v1`: basis gates `{id,x,sx,rx,ry,rz,h,s,sdg,cx,cz}`, optimization level 0, seed 7, no backend, no coupling map, and a structural nonzero angle. Recompiling all admissions reproduces every displayed tuple. This is not a Marrakesh transpilation despite the manuscript prose saying that the resource columns rely on the IBM Marrakesh backend.

The historical report builder inserted selected Pauli groups but did not apply accepted prune decisions. Applying the deletions changes three tuples. Intermediate–strong is a useful warning that a smaller operator count need not reduce every transpiled depth coordinate: its active-prefix generic result is 144/123/549 versus displayed 150/121/540.

For convention comparison only, the same active prefixes were also compiled using FakeMarrakesh, optimization level 1, seed 7, Qiskit 2.3.1, and the preserved Marrakesh backend configuration. Those results are the rightmost tuples in the opening table. They are not replacements for, or numerically interchangeable with, the displayed generic values.

## $S_{\rm alg}$ audit

The July 9 mechanism-formula components sum to the displayed integers, but arithmetic closure is not a unique-estimator-call audit.

| Regime | July 9 displayed $(N_H^{\rm outer},N_H^{\rm refit},N_{\rm grad},N_{\rm metric})$ | Legacy telemetry subtotal | Current-definition winning / discarded branch |
|---|---:|---:|---:|
| weak–weak | (0, 1,794, 2,522, 693) = 5,009 | 4,469 | unresolved / unresolved |
| intermediate–weak | (0, 1,681, 1,940, 521) = 4,142 | 3,740 | unresolved / unresolved |
| strong–weak | (0, 1,423, 2,134, 569) = 4,126 | 3,678 | unresolved / unresolved |
| weak–strong | (0, 13,146, 3,296, 863) = 17,305 | 16,627 | unresolved / unresolved |
| intermediate–strong | (0, 21,287, 5,768, 1,562) = 28,617 | 27,387 | unresolved / unresolved |
| strong–strong | (0, 1,792, 2,678, 683) = 5,153 | 4,613 | unresolved / unresolved |

The historical executable's Phase-0 screen consumes `gradients_now`, the already computed gradient array. The July 9 reconstruction nevertheless adds equal Phase-0 and Phase-I gradient counts; for weak–weak, for example, 1,261 + 1,261 becomes $N_{\rm grad}=2,522$. This is a demonstrable charge of derived reuse, not evidence of two distinct same-state primitives. The reconstruction also sets $N_H^{\rm outer}=0$, although each first history row records `initial_energy_nfev=1`. Its metric counts are window/cardinality formulas rather than state-keyed symmetric-entry identities.

No preserved result or stdout log contains the prepared-state fingerprints, observable identities, optimizer objective states, symmetric-pair keys, or consumer/branch lineage needed to deduplicate same-state reuse and separate discarded branches. Missing primitive classes include initial Hamiltonian evaluation; all distinct Powell states; Phase-0/Phase-I gradient reuse; metric/Gram/Hessian/coupling entries; warm-start guards; boundary, prune, and final refits; and branch attribution. The current-definition $S_{\rm alg}$ is therefore unresolved for every row.

## Historical enforcement defects

All 91 admissions through the six displayed prefixes are labeled under `hard_guard`, yet every one has `symmetry_spec=null` and a runtime-split symmetry gate with `checked=false`, `passed=true`, and skipped reason `runtime_split_symmetry_spec_missing`. Thus 91 identities were affected; their exact labels are recorded per row in the JSON.

The parent-pool legal-subspace filter was active, but legality was not re-enforced after singleton splitting and before child scoring. Direct legal-codeword action checks find 71 violating admissions: 10/13 weak–weak, 7/10 intermediate–weak, 7/11 strong–weak, 13/16 weak–strong, 25/28 intermediate–strong, and 9/13 strong–strong. This makes all six historical rows methodologically unmatched to a corrected hard-guard/padding-enforced route even where the preserved numerical history is internally consistent.

## Evidence conclusion

The six energy cells are validated as rounded values from their preserved histories. Weak–weak, intermediate–weak, and weak–strong retain their displayed generic circuit tuples because no accepted prune occurs by $k_{\rm pl}$, but they remain methodologically unmatched and their exact saved prefix states and current-definition $S_{\rm alg}$ remain unreconstructible. Strong–weak, intermediate–strong, and strong–strong additionally have stale circuit tuples because accepted deletions were ignored.

The table cannot establish that singleton admission is intrinsically superior to batching. Historical singleton and current JR campaigns differ in route mechanics, shortlisting, pool exposure, stopping, estimator accounting, and compiler convention, while the historical run also skipped two declared enforcement requirements. A controlled source-locked ablation would be required for that causal claim.

No manuscript or PDF was edited, regenerated, promoted, replaced, or deleted.

# Paper-I HH SNAKE Pareto Candidate Ledger

Status: active diagnostic ledger. This file records candidate policies and evidence. It does not promote a policy, change canonical settings, or authorize a manuscript edit. Policy promotion remains a user decision.

## Comparison contract

- Primary scientific coordinates: absolute energy error, Qiskit `N2q`, Qiskit `D2q`, total Qiskit circuit depth, and winning-branch `S_alg` under the Paper-I accounting contract.
- Winning-branch `S_alg` includes optimizer evaluations, gradient probes, and uniquely charged metric/Hessian measurements on the surviving lineage. All-expanded-branch work is retained as a diagnostic only; it is not the Pareto query coordinate.
- An early trajectory may remain a reasonable candidate even when its current query work is above Paper I, provided its completed trajectory extends the accuracy-resource Pareto front.
- Early-round comparisons are trajectory diagnostics, not rejection gates. Candidate classification is based on the completed terminal point or a documented plateau point against the locked Paper-I plateau front.
- A policy need not dominate every reference coordinate to remain a candidate. It must be nondominated or provide a scientifically useful accuracy extension at defensible cost.
- Qiskit dominance remains unknown until the exact completed ansatz prefix is compiled under the locked Paper-I convention.
- Shortlist width is regime-conditioned. A width that is dominated in one regime or at one early horizon may remain useful elsewhere. Maintain a Pareto front per regime and form the cross-regime candidate union only after matched-horizon evidence; do not name one globally best `L_search` from weak-weak alone.
- The active transfer family is `L_search in {13,15,20}`. No additional `L25` runs are planned; completed L25 evidence remains historical diagnostic context.

## Locked weak-weak references

Source: `output/pdf/paper_i_hh_corrected_vs_current_20260710/paper_i_hh_corrected_vs_current_onepage_20260710.json`.

| Reference | Prefix | Logical depth | `abs(Delta E)` | `N2q` | `D2q` | Circuit depth | Winning-branch `S_alg` | Role |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Paper-I SNAKE displayed plateau | 13 | 13 | 4.5236490925915085e-4 | 48 | 34 | 183 | 5,009 | Locked visible Paper-I row |
| Paper-I SNAKE source tail | 30 | 30 | 3.6992268778168746e-4 | pending prefix compile | pending | pending | pending prefix reconstruction | Longer source trajectory, not the displayed row |
| Matched Geo-ADAPT plateau | 6 | 6 | 1.1270545023809309e-3 | 148 | 93 | 632 | 30,623 | Corrected matched comparator |
| Matched append-only ADAPT plateau | 16 | 16 | 5.724630227665894e-4 | 1,202 | 1,000 | 4,589 | 63,601 | Corrected matched comparator |

## Same-round trajectory diagnostics

These rows measure early accuracy acceleration only. They can justify retaining a policy for a longer run, but they do not by themselves establish terminal Pareto dominance.

| Candidate checkpoint | Controller round | Ansatz depth | Candidate `abs(Delta E)` | Paper-I SNAKE error at same round | Error ratio candidate/Paper-I |
|---|---:|---:|---:|---:|---:|
| Wave-9 moderate B2/L50 stopped checkpoint | 7 | 14 | 1.7630361552867235e-4 | 1.4503708094799617e-3 | 0.1216 |
| Wave-10 B1/L10 terminal point | 12 | 12 | 3.743902451498471e-4 | 5.419818772831597e-4 | 0.6908 |
| Wave-10 B2/L10 terminal point | 5 | 6 | 2.2496187569972514e-3 | 1.24190002453759e-2 | 0.1811 |
| Wave-11 B2/L25 stopped checkpoint | 13 | 25 | 1.1883371849874536e-6 | 4.5236490925915085e-4 | 0.00263 |
| Wave-12 B2/L50 stopped checkpoint | 13 | 26 | 1.5106404050335698e-5 | 4.5236490925915085e-4 | 0.0334 |
| Wave-14 guarded B2/L25 terminal | 2 | 3 | 3.945067676406666e-2 | 4.335311949917564e-2 | 0.9100 |
| Wave-16 projected B2/L25 stopped checkpoint | 7 | 13 | 3.8701214220515645e-4 | 1.4503708094799617e-3 | 0.2668 |
| Wave-18 normalized projected B2/L25 uncapped | 7 | 13 | 4.202572748736033e-4 | 1.4503708094799617e-3 | 0.2898 |
| Wave-19 normalized projected B2/L25 `maxfev=200` | 7 | 12 | 5.019962818002544e-4 | 1.4503708094799617e-3 | 0.3461 |

## Regime status before Phase-II joint-response implementation

Snapshot taken before pausing scientific runs for the experimental `child_12_joint_response_v2` implementation. Completed artifacts remain authoritative diagnostics. Interrupted rows retain their `current.json` checkpoints and are not reported as completed results.

| Regime | Evidence state | Best or latest `abs(Delta E)` | Controller round | Ansatz depth | Competitor status | Geometry status |
|---|---|---:|---:|---:|---|---|
| weak-weak | Wave-27 complete, combinatorial `B2/L15` | 5.076214602436346e-4 | 7 | 11 | Beats Geo and Append at the matched round; corrected L15 has not yet beaten Append's final plateau | Not blocked |
| intermediate-weak | Wave-31 complete, combinatorial `B2/L15` | 2.1714140314815777e-4 | 7 | 12 | Beats both comparator final plateaus on energy error | Not blocked |
| strong-weak | Wave-41 complete singleton/all control | 2.5218463428811067e-4 | 9 | 9 | Yet to beat Geo or Append | Singleton/all route unblocked; corrected batched Waves 42-44 are joint-Gram blocked |
| weak-strong | Wave-45 implementation-interrupted checkpoint, `B2/L15` | 6.799670857463336e-2 | 4 | 6 | Incomplete; no terminal comparison | Not blocked at checkpoint |
| intermediate-strong | Not started | pending | pending | pending | Yet to test | Unknown |
| strong-strong | Wave-37 complete `B2/L15`; Wave-46 implementation-interrupted `B3/L15` checkpoint | 3.692745774368511e-4 complete; 6.523710861043419e-3 checkpoint | 9 complete; 4 checkpoint | 11 complete; 9 checkpoint | Completed B2/L15 has not beaten the matched round-9 comparators | Not blocked at checkpoint |

Strong-weak corrected joint-gate details:

| Policy | Subsets considered | Exact-child compatibility rejections | Joint-rank rejections | Terminal result |
|---|---:|---:|---:|---|
| Wave-42 `B3/L15` | 575 | 56 | 519 | round 2/depth 3, `abs(Delta E)=2.5765847317721358e-3` |
| Wave-43 `B2/L20` | 210 | 4 | 206 | round 2/depth 3, `abs(Delta E)=2.5765847317721358e-3` |
| Wave-44 `B3/L20` | 1,350 | 76 | 1,274 | round 2/depth 3, `abs(Delta E)=2.5765847317721358e-3` |

The queued Waves 47-51 were stopped before launch and are deferred until the joint-response implementation, deterministic ordering audit, and user review are complete.

## Phase-II joint-response six-regime evidence

Completed local candidate matrix for `child_12_joint_response_v2` with combinatorial `B_max=2`, `L_search=15`, `M1/M2=32/24`, `C1/C2=32/25`, Powell `maxiter=50`, and `maxfev=200`. `S_alg` is winning-lineage terminal work. Circuit costs use `FakeMarrakesh`, optimization level 1, and transpiler seed 7. The machine-readable comparison is `raw_outputs/paper_i_hh_joint_response_six_regime_20260711/comparison/six_regime_comparison.json`.

| Regime | New `abs(Delta E)` | Round / depth | `S_alg` | `N2q / D2q / Dc` | Versus preceding ledger evidence | Versus Paper-I HH references |
|---|---:|---:|---:|---:|---|---|
| weak-weak | 6.026608274340983e-4 | 7 / 12 | 16,406 | 91 / 79 / 259 | Wave-27 has better error, 5.076214602436346e-4; joint-response uses less query work and fewer compiled resources | Five-axis dominates Geo; error remains above displayed SNAKE and Append |
| intermediate-weak | 4.1102520726088443e-4 | 7 / 11 | 16,737 | 135 / 101 / 317 | Wave-31 has better error, 2.1714140314815777e-4; joint-response uses less query work and fewer compiled resources | Five-axis dominates Append and beats Geo on error; displayed SNAKE has better error and work |
| strong-weak | 2.5765847317721358e-3 | 2 / 3 | 1,879 | 10 / 8 / 39 | Worse error than the Wave-41 singleton/all control, 2.5218463428811067e-4; low resources reflect premature selector termination | Does not approach the SNAKE, Geo, or Append errors near 1.6e-6 to 2.0e-6 |
| weak-strong | 2.538679443019798e-2 | 9 / 16 | 32,179 | 176 / 120 / 374 | Improves the interrupted Wave-45 checkpoint error, 6.799670857463336e-2, by 62.7% and completes the requested horizon | Five-axis dominates Geo; SNAKE and Append retain lower error |
| intermediate-strong | 8.47716482049754e-3 | 9 / 14 | 42,451 | 142 / 110 / 404 | First completed ledger evidence for this route/regime | Better error than Geo and near the Append trajectory marker, 8.150300978024494e-3; SNAKE remains much more accurate |
| strong-strong | 2.0286553759074621e-4 | 9 / 11 | 33,367 | 92 / 80 / 236 | Improves Wave-37 error by 45.1% and `S_alg` by 42.5%; lowers `N2q` but increases `D2q` and total depth | Better resources than Geo/Append but worse error; displayed SNAKE remains better in error, work, and compiled resources |

The strong-weak joint-response cell remains geometry-blocked. At round 2 it exposed 15 singleton and 105 pair subsets; 21 violated exact-child compatibility and the remaining 99 failed the joint rank gate. This is not physical-population exhaustion and should be repaired before treating this route as regime-agnostic.

The new matrix is not a uniform replacement for the preceding route. It provides resource/error tradeoffs in both weak sectors, a completed weak-strong and intermediate-strong transfer, and a stronger strong-strong point, while regressing strong-weak and failing to beat the displayed Paper-I SNAKE error in every regime at these horizons.

One-page visual comparison: [six-regime error and FakeMarrakesh cost overlay](../../../output/pdf/paper_i_hh_joint_response_six_regime_overlay_20260711/paper_i_hh_joint_response_six_regime_overlay_20260711.pdf). The corresponding [machine-readable provenance](../../../output/pdf/paper_i_hh_joint_response_six_regime_overlay_20260711/paper_i_hh_joint_response_six_regime_overlay_20260711.json) pins every trajectory, terminal/plateau marker, and `N2q/D2q/Dc` endpoint. It overlays the preceding ledger route where evidence exists, the completed Phase-II joint-response route, and the locked Paper-I SNAKE/Geo/Append references for each regime.

## Candidate policies

| Candidate | Policy delta | Current evidence | Candidate status | Missing evidence |
|---|---|---|---|---|
| Wave-9 moderate B2/L50 | `M1/M2=64/48`, child caps open, combinatorial `B_max=2`, `L_search=50` | Stopped checkpoint: round 7, depth 14, `abs(Delta E)=1.7630361552867235e-4` | Strong accuracy candidate; beats the displayed SNAKE row and the known SNAKE source tail on error | Completed plateau, winning-branch `S_alg`, Qiskit costs, matched replay |
| Wave-10 B1/L10 | `M1/M2=64/48`, `C1/C2=128/64`, `B_max=1`, `L_search=10` | Complete terminal selector point: round 12, depth 12, `abs(Delta E)=3.743902451498471e-4`, winning-branch `S_alg=63,120`; stop `joint_geometry_selector_exhausted` | Reasonable candidate; improves the displayed SNAKE plateau error and dominates matched Append in error and winning-branch `S_alg`, but is slightly above the longer SNAKE source-tail error | Qiskit costs and matched replay |
| Wave-10 B2/L10 | Same caps, combinatorial `B_max=2`, `L_search=10` | Complete: round 5, depth 6, `abs(Delta E)=2.2496187569972514e-3`, winning-branch `S_alg=15,330`; stopped by joint-selector exhaustion | Lower-resource diagnostic | Wider-search comparison and Qiskit costs if retained |
| Wave-11 B2/L25 | Same caps, combinatorial `B_max=2`, `L_search=25` | User-stopped checkpoint: round 13, depth 25, `abs(Delta E)=1.1883371849874536e-6`; measured illegal-codeword probability approximately `2.8e-7` | Leading accuracy candidate; error is 0.00321 times the Paper-I SNAKE source-tail error and is lower than L50 at the same round | Settings-identical legal-child-guard replay, winning-branch `S_alg`, Qiskit costs, terminal/plateau evidence |
| Wave-12 B2/L50 | Same caps, combinatorial `B_max=2`, `L_search=50` | User-stopped checkpoint: round 13, depth 26, `abs(Delta E)=1.5106404050335698e-5`; measured illegal-codeword probability approximately `3.2e-7` | Strong diagnostic candidate, but accuracy-dominated by L25 at the same round and more expensive to search | Retain as pre-guard diagnostic; no continuation currently planned |
| Wave-14 guarded B2/L25 | Wave-11 settings plus `child_padding_policy=nph2_legal_codeword_hard_filter_v1` | Complete: round 2, depth 3, `abs(Delta E)=3.945067676406666e-2`, winning-branch `S_alg=2,570`; stop `joint_geometry_selector_exhausted`; final illegal-codeword probability `0` | The literal globally legal singleton filter is incompatible with the desired L25 route: it removed 1,408 of 1,748 deduplicated child-position records and left no useful boson-transition direction | Choose an exact legality-preserving child representation before further Pareto runs |
| Wave-16 projected B2/L25 | Exact projected/grouped `n_ph_max=2` children, `M1/M2=64/48`, `C1/C2=128/64`, combinatorial `B_max=2`, `L_search=25` | User-stopped checkpoint: round 7, depth 13, `abs(Delta E)=3.8701214220515645e-4`; final illegal-codeword probability `0` | Demonstrated that exact legal projection restores the strong trajectory | Pre-direction-normalization diagnostic only; superseded by Waves 18/19 |
| Wave-18 normalized projected B2/L25 uncapped | Wave-16 route plus deterministic unit-norm projective child representatives and complete parent provenance; uncapped Powell | Complete: round 7, depth 13, `abs(Delta E)=4.202572748736033e-4`, winning-lineage `S_alg=34,607`; FakeMarrakesh `N2q=164`, `D2q=113`, total depth `373`; final illegal-codeword probability `0` | Accuracy anchor; improves the displayed Paper-I plateau error at this shorter controller horizon, at higher resources | Later-horizon plateau evidence and matched replay |
| Wave-19 normalized projected B2/L25 `maxfev=200` | Settings-identical to Wave-18 except Powell `maxfev=200` | Complete: round 7, depth 12, `abs(Delta E)=5.019962818002544e-4`, winning-lineage `S_alg=32,371`; final illegal-codeword probability `0` | Nondominated short-horizon query/accuracy tradeoff: 6.46% lower `S_alg` with 19.5% larger error than Wave-18 | Locked Qiskit compile and shortlist-sensitivity evidence |
| Wave-20 narrow macro funnel | Wave-19 settings with `M1/M2=32/24` | Complete: round 7, depth 12, `abs(Delta E)=5.019962818002544e-4`, winning-lineage `S_alg=25,200`; FakeMarrakesh `N2q=121`, `D2q=97`, total depth `304`; identical operator-path hash to Waves 19/21 | Strictly supersedes the wider capped macro funnels on accuracy, path, depth, and query work; current resource-oriented new-route anchor | Later-horizon evidence and matched replay |
| Wave-21 moderate macro funnel | Wave-19 settings with `M1/M2=48/36` | Complete: round 7, depth 12, `abs(Delta E)=5.019962818002544e-4`, winning-lineage `S_alg=29,822`; identical operator-path hash to Waves 19/20 | Superseded by Wave-20: same final path and energy with 18.3% more query work | Retain as macro-sensitivity evidence; no replay planned |
| Wave-22 narrow child funnel | Wave-20 settings with `C1/C2=64/32` | Complete: round 7, depth 12, `abs(Delta E)=5.019962818002544e-4`, winning-lineage `S_alg=22,275`; identical operator path, so FakeMarrakesh remains `N2q=121`, `D2q=97`, total depth `304` | Strictly supersedes Waves 19-21 on the measured coordinates; current resource-oriented new-route anchor | Later-horizon evidence and matched replay |
| Wave-23 moderate child funnel | Wave-20 settings with `C1/C2=96/48` | Complete: round 7, depth 12, `abs(Delta E)=5.019962818002544e-4`, winning-lineage `S_alg=23,747`; identical operator-path hash to Waves 19-22 | Superseded by Wave-22: same path and energy with 6.20% more query work | Retain as child-sensitivity evidence; no replay planned |
| Wave-24 minimum full-L25 child funnel | Wave-20 settings with `C1/C2=32/25` | Complete: round 7, depth 12, `abs(Delta E)=5.019962818002544e-4`, winning-lineage `S_alg=20,803`; identical operator path, so FakeMarrakesh remains `N2q=121`, `D2q=97`, total depth `304` | Strictly supersedes Waves 19-23 on the measured coordinates; current shortlist anchor | Later-horizon evidence and matched replay |
| Wave-25 C1 sensitivity at full L25 | Wave-20 settings with `C1/C2=48/25` | Complete: round 7, depth 12, `abs(Delta E)=5.019962818002544e-4`, winning-lineage `S_alg=21,539`; identical operator-path hash to Waves 19-24 | Superseded by Wave-24: same path and energy with 3.54% more query work | Retain as child Phase-1 sensitivity evidence; no replay planned |
| Wave-26 narrow combinatorial search | Wave-24 settings with `L_search=10` | Complete: round 7, depth 10, `abs(Delta E)=6.228195999226083e-4`, winning-lineage `S_alg=14,665`; FakeMarrakesh `N2q=78`, `D2q=51`, total depth `174` | Nondominated low-resource point; reduced accuracy relative to L15/L25 but substantially lower query and compiled cost | Later-horizon plateau behavior and matched replay |
| Wave-27 balanced combinatorial search | Wave-24 settings with `L_search=15` | Complete: round 7, depth 11, `abs(Delta E)=5.076214602436346e-4`, winning-lineage `S_alg=18,009`; FakeMarrakesh `N2q=104`, `D2q=84`, total depth `284` | Nondominated balanced point; only 1.12% more error than L25 with 13.4% lower query work and lower compiled costs | Later-horizon plateau behavior and matched replay |
| Wave-28 singleton selector | Wave-27 settings with `B_max=1` | Complete: round 7, depth 7, `abs(Delta E)=1.405724904938177e-3`, winning-lineage `S_alg=11,149` | Low-query singleton diagnostic; batching materially improves early accuracy | Regime transfer only if a lower-resource endpoint is needed |
| Wave-29 greedy B2/L15 | Wave-27 settings with greedy reduced-plane batching | Complete: round 7, depth 13, `abs(Delta E)=5.044191859397973e-4`, winning-lineage `S_alg=290,697`, elapsed 2,465 s; 83,928 pair subsets were considered in the last selector summary | Accuracy is essentially tied with combinatorial L15, but query work is 16.1 times larger and wall time is excessive; removed from the active transfer ladder | Retain as the completed greedy comparison; no continuation planned |
| Wave-30 weak-weak L13 | Wave-27 settings with `L_search=13` | Complete: round 7, depth 11, `abs(Delta E)=1.0223510456918161e-3`, winning-lineage `S_alg=16,414` | Dominated by L10 at weak-weak round 7 only; retained as a regime-transfer candidate, not globally rejected | Matched evidence in other regimes |
| Wave-31 intermediate-weak L15 | Wave-27 settings transferred to intermediate-weak | Complete: round 7, depth 12, `abs(Delta E)=2.1714140314815777e-4`, winning-lineage `S_alg=18,735`; beats Geo error `9.91160135894178e-2` and Append error `2.7015712223815935e-2`, and uses less `S_alg` than Append | Strong regime-specific transfer evidence; no global-width claim | Round-9 continuation, Qiskit costs, matched replay |
| Waves 34/36 strong-weak raw-ranked L15/L20 | Strong-weak U8 transfer before rank-feasible-width repair | Both stopped at round 2/depth 3 with `abs(Delta E)=2.5765847317721358e-3`; every exposed singleton/pair failed the hard tangent-rank gate although 100 child Phase-2 records survived | Diagnoses raw `L_search` being consumed by infeasible tangent-redundant records; not physical-population exhaustion | Superseded by corrected rank-feasible L15 replay |
| Wave-38 strong-weak all-singleton rank audit | All child Phase-2 survivors, singleton-only, three rounds | Complete: round 3, depth 3, `abs(Delta E)=1.2212922636823942e-3`, winning-lineage `S_alg=2,305`; 75 records were exposed at the final round and the run continued normally | Proves valid directions exist below raw rank 20 and justifies rank-feasible search-width accounting | Corrected B2/L15 round-9 replay |

The Wave-11 and Wave-12 processes predate the canonical child-padding repair. Their trajectories remain useful diagnostics. Wave-14 established that literal global legal-codeword filtering is too restrictive for binary phonon transitions; Waves 16, 18, and 19 instead use exact projected/grouped legal children.

Wave-14 showed why literal hard filtering fails. For binary `n_ph_max=2`, exhaustive enumeration of the 256 Pauli patterns on the four boson qubits finds 16 globally legal patterns, all containing only `e/z`; no globally legal singleton Pauli with `x/y` survives. The exact projected/grouped policy repairs that representation without leakage, globally deduplicates the projected direction, and normalizes its scalar/sign convention before scoring.

## Promotion-candidate states

- `diagnostic`: useful mechanism evidence but not yet a policy candidate.
- `reasonable_candidate`: completed or strongly progressing trajectory that may extend the final Pareto front.
- `verified_candidate`: completed result with winning-branch `S_alg`, Qiskit costs, provenance checks, and a matched replay; all-expanded work may be retained as diagnostic telemetry.
- `user_promoted`: policy explicitly selected by the user for canonical settings or manuscript work.

## Evidence required before asking for promotion

1. Complete the approved horizon or document a mathematically valid terminal condition.
2. Report both controller rounds and final ansatz depth.
3. Reconstruct winning-branch `S_alg` as the primary query coordinate and retain all-expanded-branch work only as a diagnostic.
4. Qiskit-compile the exact completed or selected plateau prefix.
5. Compare against the displayed Paper-I SNAKE row, the longer SNAKE source trajectory, and matched Geo/Append plateaus.
6. Replay the candidate under identical settings to establish deterministic or tolerance-level stability.
7. Present the evidence to the user; do not promote automatically.

## Active search ladder

1. `B1/L10` baseline: complete; reasonable candidate pending Qiskit and replay.
2. `B2/L10`: complete.
3. `B2/L25`: stopped at round 13/depth 25; leading pre-guard diagnostic.
4. `B2/L50`: stopped at round 13/depth 26; no continuation planned.
5. Settings-identical guarded `B2/L25`: complete; hard singleton filtering exhausted at round 2 and is not a viable replacement for the unchecked L25 route.
6. Exact projected/grouped child representation: implemented and validated; Wave-16 is pre-normalization diagnostic evidence.
7. Normalized projected L25 uncapped versus `maxfev=200`: Waves 18/19 complete and nondominated in short-horizon accuracy/query coordinates.
8. Macro exposure at fixed `C1/C2=128/64`, `B_max=2`, `L_search=25`, and `maxfev=200`: Wave-20 (`32/24`) preserves the Wave-19/21 path and energy while reducing `S_alg` from `32,371` and `29,822` to `25,200`.
9. Child exposure at `M1/M2=32/24`: Wave-22 (`64/32`) preserves the path and energy while lowering `S_alg` to `22,275`; Wave-23 (`96/48`) is superseded.
10. Minimum full-L25 child exposure: Wave-24 (`C1/C2=32/25`) preserves the path and energy with `S_alg=20,803`; Wave-25 (`48/25`) is superseded.
11. Weak-weak round-7 search-width evidence is retained, but future transfer uses `L13/L15/L20`; no additional L25 cells are planned.
12. Wave-28 singleton-only and Wave-29 greedy B2/L15 are complete. Greedy achieved nearly the same weak-weak error as combinatorial L15 but used `S_alg=290,697` versus `18,009`; combinatorial is the active transfer route and no further greedy cells are planned.
13. Strong-weak raw-ranked L15/L20 exposed a selector-ordering problem. A temporary singleton rank-feasible prefilter could remove every Phase-2 record before combinatorial construction, contrary to the approved joint-subset contract. Canonical Route A now forms `C_search` from exactly the first `L_search` globally ranked child Phase-2 records and applies rank/conditioning gates only after each singleton, pair, or triple receives its joint ansatz-plus-batch model. `rank_feasible_fill_v1` remains diagnostic only.
14. Wave-41 strong-weak all-candidate singleton control completed nine rounds/depth 9 at `abs(Delta E)=2.5218463428811067e-4`. It demonstrates that the route can continue when the shortlist exposes the needed trajectory, but it is not the promoted batched policy.
15. Corrected strong-weak Wave-42 (`B3/L15`) considered `15` singletons, `105` pairs, and `455` triples. The singleton prefilter rejected none; `56` subsets violated exact-child compatibility and all remaining `519` failed the joint rank gate on that selected trajectory. Waves 43/44 continue the prescribed `B2/L20` and `B3/L20` checks.
16. Active local queues preserve two scientific slots and the combinatorial-only ladder. Strong-strong proceeds from `B3/L15` to `B2/L20` to `B3/L20`; weak-strong first establishes `B2/L15`, then runs `B3/L15`, `B2/L20`, and `B3/L20`. Every launch retains the 10 GiB disk floor.
17. Qiskit-compile each completed nondominated or near-nondominated prefix under the locked Paper-I convention.

Update this ledger after each completed cell and after each Qiskit compilation. Keep incomplete checkpoints clearly labeled.

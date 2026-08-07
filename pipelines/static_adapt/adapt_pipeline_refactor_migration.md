# Static ADAPT Pipeline Refactor Migration Log

Purpose: track behavior-preserving moves out of
`pipelines/static_adapt/adapt_pipeline.py` while Route A is being made the
first-class pipeline. This is an agent-facing audit log, not a deletion list.

Branch:

- `codex/static-adapt-beam-refactor`

## Scope Rules

- A moved helper was live or compatibility-reachable at the time it moved.
- Moving code does not mean the behavior is canonical Paper-I Route A.
- Route B/C, noise/oracle, telemetry/debug, and legacy comparator surfaces
  should become importable/quarantined modules instead of branches inside the
  noiseless Route-A hot path.
- Stage only files named in the current slice. Ignore unrelated dirty scaffold
  files unless the user explicitly pulls them into scope.

## Migration Entries

| Commit | Module | Moved from `adapt_pipeline.py` | Refactor role | Validation |
|---|---|---|---|---|
| `e838eb3` | `route_support.py` | route support helpers | Route/config support extraction | focused static-ADAPT tests |
| `3917bae` | `noise_routes.py` | noise payload helpers | Payload-only noise/oracle helpers, no runtime oracle construction | focused noise/static-ADAPT tests |
| `7898c22` | `selector_geometry.py` | selector geometry helpers | Phase-III selector geometry/proxy helper extraction | focused static-ADAPT tests |
| `2310e9e` | `runtime_split.py` | runtime split helpers | Runtime/logical parameter split helpers | focused static-ADAPT tests |
| `441d513` | `noise_routes.py` | noise guard helpers | Noise/oracle validation guard extraction | focused noise/static-ADAPT tests |
| `d507ae6` | `route_c_plateau.py` | Route-C QNGD trial helper | Route-C quarantine step | focused route-C/static-ADAPT tests |
| `f24b925` | `route_c_plateau.py` | Route-C telemetry helpers | Route-C quarantine step | focused route-C/static-ADAPT tests |
| `7e6a452` | `oracle_lifecycle.py` | `FinalNoiseAuditSnapshot`, Phase-III oracle cleanup guard, oracle-plan cache helper, oracle backend-info normalizer, final-noise-audit runner | Lazy noise/oracle lifecycle extraction; runtime/Qiskit objects remain caller-injected | `py_compile`, `test/test_static_adapt_noise_routes.py`, `test/optimization/test_phase3_oracle_gradient_config.py`, final-noise/raw-oracle integration subset |
| `7e6f955` | `oracle_lifecycle.py` | Phase-III oracle runtime context setup and binding-backed oracle construction | Moves raw/expectation oracle setup out of `_run_hardcoded_adapt_vqe`; canonical noiseless Route A remains off-path | `py_compile`, `test/test_static_adapt_noise_routes.py`, `test/optimization/test_phase3_oracle_gradient_config.py`, final-noise/raw-oracle integration subset; larger phase3-oracle subset still has known pre-existing selector expectation failures |
| `ee13b0e` | `controller_telemetry.py` | controller snapshot dict/payload serializers, controller telemetry summary, branch state summary | Payload-only controller telemetry extraction; checkpoint writer and controller behavior remain in `adapt_pipeline.py` | focused controller telemetry tests plus static-ADAPT smoke tests |
| `38077c0` | `controller_phase_state.py` | controller snapshot-from-record helper, phase live/shot/cap/threshold/terminal accessors | Pure controller phase-state accessor extraction; controller updates, measurement-work accounting, and gate policy remain in `adapt_pipeline.py` | focused controller phase-state tests plus checkpoint/current-json smoke tests |
| `463455c` | `prune_schur_payloads.py` | prune authority telemetry, nomination source rows, inactive Schur payloads, Schur nomination gate metadata, compact Schur row serialization | Payload-only prune/Schur diagnostic extraction; derivative propagation, Hessian construction, warm-start guard, refit, and deletion acceptance remain in `adapt_pipeline.py` / `hh_continuation_pruning.py` | focused prune Schur payload tests plus prune-risk/static-ADAPT smoke tests |
| `0a9e705` | `prune_derivatives.py` | runtime derivative propagation for per-runtime-coordinate Schur prune diagnostics | Moves derivative propagation out of `_run_hardcoded_adapt_vqe`; Hessian assembly, surrogate scoring, pruning authority, and deletion acceptance remain in `adapt_pipeline.py` / `hh_continuation_pruning.py` | focused prune derivative tests plus recoverability prune integration smoke tests |
| `d49107a` | `route_c_plateau.py` | Route-C active-old index, runtime state payload, candidate identity, candidate term, and sort-key helpers | Quarantines pure Route-C helper logic; active-dormant novelty, plateau scoring, zeroing, trial optimizer, and Route-A path remain unchanged | focused Route-C helper tests plus Route-C plateau integration smoke tests |
| `08c150b` | `optimizer_routes.py` | explicit ADAPT inner optimizer config, SPSA/QNSPSA dispatch, deterministic ROTOSOLVE/SciPy dispatch, SPSA payload/heartbeat helpers | Moves optimizer selection/concrete dispatch out of `_run_hardcoded_adapt_vqe`; fidelity construction, refit windows, optimizer memory routing, Route-C SP-QNGD, and result accounting remain in `adapt_pipeline.py` | focused optimizer route tests plus SPSA/QNSPSA/ROTOSOLVE smoke tests |
| `81880bd` | `selector_geometry.py` | proxy-reduced selector override policy, shadow legacy geometry depth gate, shadow/debug fail-open attachment wrapper | Quarantines legacy/proxy selector-geometry control flow while leaving bridge construction and caches injected from `adapt_pipeline.py`; canonical reduced Route-A scoring remains unchanged | focused selector geometry/legacy bridge tests plus phase-3 integration smoke tests |
| `dba5cc9` | `checkpoint_telemetry.py` | current-checkpoint JSON normalization, compact prune audit, compact history-tail rows, selected-record summary, surface-row summary | Moves telemetry-only current-checkpoint compaction helpers out of `_run_hardcoded_adapt_vqe`; checkpoint write scheduling, branch choice, replay payloads, and final output assembly remain in `adapt_pipeline.py` | focused checkpoint telemetry tests plus current-json/static-ADAPT checkpoint smoke tests |
| `161cb86` | `batch_ordering.py` | ordered batch admission identity helpers, finite-step batch-order proxy scoring, finite-step rescue fill, batch Schur context telemetry | Moves Route-A-live batch admission ordering out of `_run_hardcoded_adapt_vqe`; batch candidate scoring, beam child materialization, energy/noise/oracle objective evaluation, and payload assembly remain in `adapt_pipeline.py` | focused batch-ordering tests plus resume/beam integration smoke tests |
| `d70cefc` | `phase_shortlists.py` | Phase-1 active-score payload helpers, generic phase shortlist legacy hook, algebraic lane shortlist wrappers, Phase-3 tie-beam selection helper | Moves shortlist policy and legacy-hook compatibility out of `_run_hardcoded_adapt_vqe`; candidate scoring, controller updates, Phase-0 pilot screening, and admission remain in `adapt_pipeline.py` | focused phase-shortlist/algebraic/batch tests plus static-ADAPT shortlist/beam smoke tests |
| `5d5f327` | `selector_measurement_proxy.py` | controller measurement-work record/probe helper cluster: group-key resolution, candidate term/metadata/symmetry lookup, logical operator-probe counts, common-exposure manifest digest, live controller work event wrapper | Moves telemetry/accounting-only controller work plumbing out of `_run_hardcoded_adapt_vqe`; accumulator/schema ownership remains in selector measurement proxy; Route-A selection/scoring/admission unchanged | focused controller measurement-work tests plus existing proxy/pareto and phase-shortlist smoke tests |
| `45af550` | `checkpoint_telemetry.py` | final selected-scaffold / Phase-3 surface / active-HH-pool / optimizer-memory / runtime-boundary telemetry payload helpers | Final-output telemetry extraction; pure selected-scaffold and Appendix-A boundary payload builders move out of `_run_hardcoded_adapt_vqe`; Route-A scoring/admission and beam branch assembly unchanged | focused checkpoint telemetry tests plus controller telemetry wrapper tests |
| `14879ae` | `route_identity.py` | static route identity observed-component assembly and declared-route validation wrapper | Moves Route-A/B/C metadata assembly out of `_run_hardcoded_adapt_vqe` behind an explicit config object; route contracts and validation semantics remain in `route_identity.py`; Route-A execution/scoring/admission unchanged | focused route-identity tests plus py_compile |
| `79bae0c` | `optimizer_routes.py` | optimizer progress interval, effective optimizer key, and stochastic heartbeat event helper access | Replaces remaining heartbeat callback reads of raw SPSA/progress closure values with explicit `AdaptInnerOptimizerConfig` helper access; optimizer dispatch/objectives/fidelity/memory and emitted event names remain unchanged | focused optimizer-route tests plus SPSA/QNSPSA heartbeat integration tests |
| `d627753` | `run_control.py` | benchmark-target absolute-error helper and target-hit classification payload helper | Moves pure benchmark-target payload construction out of `_run_hardcoded_adapt_vqe`; live target stop decisions and the exact-final-state audit overlay remain in `adapt_pipeline.py` | focused run-control tests plus current-json/beam target-hit smoke tests |
| `ecf16f4` | `beam_search.py` | beam replay branch-summary payload helper | Moves compact per-branch beam replay summary construction out of `_run_hardcoded_adapt_vqe`; runtime-local prune-key policy remains callback-injected from `adapt_pipeline.py` | focused beam helper tests plus beam current-checkpoint replay smoke test |
| `4aab9ba` | `beam_search.py` | final beam branch-summary payload helper | Moves final per-branch beam diagnostics summary construction out of `_run_hardcoded_adapt_vqe`; generator registry lookup and runtime-local prune-key policy remain callback/input-driven from `adapt_pipeline.py` | focused beam helper tests |
| `this commit` | `beam_search.py` | final beam diagnostics payload helper | Moves final beam diagnostics/update payload construction out of `_run_hardcoded_adapt_vqe`; finalist selection, target classification, checkpoint writing, and winner copy-back remain in `adapt_pipeline.py` | focused beam helper tests |

## Current Open Questions

- Route-B pairwise behavior should remain importable for old artifacts/tests,
  but not first-class in `adapt_pipeline.py`.
- Route-C plateau behavior should be quarantined but retained enough for old
  artifacts/tests and possible agent-robustness diagnostics.
- TETRIS/QEB/CEO are legacy/deletion candidates; do not delete without a reach
  audit and explicit user approval.
- Geo-ADAPT and append-only ADAPT are benchmarking siblings, not Route-A
  internals.

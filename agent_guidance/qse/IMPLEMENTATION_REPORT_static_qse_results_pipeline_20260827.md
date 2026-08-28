# Implementation report: Paper III static-QSE results pipeline and SIAM benchmark

Date: 2026-08-27
Branch: `paper-ii-exchange-selector`
Scientific anchor: `e6b2c2cae35cd1b831d01e29a1177972b9a69ffc` (branch tip at start)
Governing handoff: `prompt-exports/HANDOFF_paper_iii_static_qse_results_pipeline_and_siam_framing_20260827.md`
Operating contract: `agent_guidance/qse/paper-iii-static-results-acquisition-contract.md`

## Commits by increment

1. `paper-iii: port SIAM scientific layer and add static-spectroscopy adapter`
   - Ported verbatim from `codex/paper-ii-search-profiles-20260824` commit
     `da7b26f3`: `src/quantum/siam_{model,observables,pool,reference}.py`,
     `test/test_siam_{model,pool,reference}.py` (no time-dynamics runtime).
   - Added `pipelines/qse_spectra/siam_static_adapter.py`,
     `test/test_paper_iii_siam_static_adapter.py`.
2. `paper-iii: typed static-QSE campaign, planner, and immutable ledger`
   - Added `pipelines/qse_spectra/static_campaign.py`,
     `pipelines/qse_spectra/static_campaign_artifacts.py`, their tests, and
     the tracked operating contract.
3. `paper-iii: one-action static-QSE producer and deterministic local package`
   - Added `pipelines/qse_spectra/static_campaign_package.py`,
     `pipelines/exact_bench/paper_iii_static_qse_cell.py`, their tests; also
     committed the pre-existing `--regimes` CLI filter on
     `paper_iii_growth_trace_campaign.py` (display filter only).
4. `paper-iii: exact-subset ingestion contract, source-map roles, matrix report`
   - Added `pipelines/reporting/paper_iii_static_results_contract.py` and its
     test; extended `pipelines/qse_spectra/source_maps.py` (static campaign /
     ledger / contract roles), `pipelines/reporting/build_paper_iii_qse_results_pdf.py`
     (`--static-contract-json` matrix rendering), and both existing test suites.
5. This report plus a one-line repair in the PDF builder's compile copy step
   (same-file guard, pre-existing defect exposed by co-located tex/pdf paths).

## Verification

Focused suite (Increment 5 list, 16 files): **187 passed** —
`test_siam_model` (25), `test_siam_pool` (16), `test_siam_reference` (16),
`test_paper_iii_siam_static_adapter` (15), `test_paper_iii_static_campaign`
(18), `test_paper_iii_static_campaign_artifacts` (11),
`test_paper_iii_static_campaign_package` (5), `test_paper_iii_static_qse_cell`
(10), `test_paper_iii_static_results_contract` (7),
`test_paper_iii_growth_trace_hybrid` (10), `test_paper_iii_qse_source_maps`
(15), `test_paper_iii_qse_results_pdf` (12), plus the untouched
green-function, spectral-function, record-selection, and adaptive-QSE suites.

### nph1 support behavior lock: HELD

- `record_selection.py`, `exchange_maintenance.py`, `core.py`,
  `adaptive_qse_benchmark.py`, `paper_iii_problem.py`: **diff-clean against
  the anchor** (bit-identical files).
- All **93** acquisition-prefix supports recomputed across the six `nph1`
  regimes (order lengths 15/16/16/15/16/15); all **48** stride-row supports
  stored in `output/diagnostics/paper_iii_record_first_adaptive_qse_v2/
  growth_trace_nph1.json` compared **bit-identical**; zero mismatches.

## Bounded local smoke (diagnostic only, not paper evidence)

Full chain executed through the real subprocess CLI
(`python -m pipelines.exact_bench.paper_iii_static_qse_cell`):
declare → plan → package → execute → ingest exact subsets → plan dependents
→ execute QSE products → qualify → results contract → report manifest + PDF.

Declared smoke campaign `paper_iii_static_qse_smoke_20260827_v1`:
SIAM `N_b=2` record-first (cap 16, reduced 61-point grid, exact-sector
diagnostic fixture), SIAM `N_b=2` direct Davidson (only its ground-state
import executed, demonstrating exact-subset partial ingestion), one HH
`weak_weak nph1` record-first cell through the provider-exact diagnostic
fixture, plus `not_planned` SIAM fixed-class and Krylov slots. 15 actions
executed and admitted across 5 dependency rounds.

Result matrix:

| case | method | status |
|---|---|---|
| weak_weak (nph1) | record_first_adaptive_qse | qualified (window err 6.3e-14) |
| siam_nb2_fixture | record_first_adaptive_qse | qualified (window err 1.5e-9) |
| siam_nb2_fixture | direct_davidson_qse | gate_pending (import only) |
| siam_nb2_fixture | fixed_linear_response_qse | not_planned |
| siam_nb2_fixture | real_time_krylov | not_planned |

Both record phases terminated by `record_residual_converged`; the Davidson
fallback never fired (exhaustion-only preserved). Green-function sum-rule
deficits at machine precision; spectral functions nonnegative.

Artifacts (local, gitignored diagnostics under
`output/diagnostics/paper_iii_static_qse_smoke_20260827_v1/`):

- `campaign.json` sha256 `046dda95a9941b34...`
- `ledger.json` sha256 `fc4eb5443a6aa49f...` (15 admitted attempts)
- `results_contract.json` sha256
  `f386f1816e5e1df8b6a939aa6c27aac74bbb2ec782589827db84936bfd034dfd`
- `static_source_map.json` sha256 `6c5c259a56d106e9...`
- `report/paper_iii_static_qse_smoke.pdf` — compiled; manifest hashes the
  campaign digest `70fe258872f8c709...` and ledger digest `55fb3d19cb2724b3...`
- `package/` — 22 declared actions (ACTION.json + result envelopes + payloads)

## Settled policies preserved

- Online score `S = Delta/K`; graph-span `two_qubit_only_v1`, `alpha=1`,
  floor `0.05`; double Gram--Schmidt; retained-frame novelty; pool-relative
  window stopping quantity — selector untouched.
- True atomic `1<->1` exchange with whole-patch certification — untouched.
- Davidson residual completion strictly on `pool_exhausted`.
- Exact spectra: post-run qualification and annotation only.
- Public method ids `record_first_adaptive_qse` / `direct_davidson_qse` /
  `real_time_krylov` / `fixed_linear_response_qse`; `record_first_adaptive_qse_v2`
  normalized at ingestion; `record_only` and `early_switch` remain ablations.
- SIAM production spectroscopy seed must be the declared interacting
  ground-state artifact; the Paper II decoupled quench seed is structurally
  rejected; SIAM fixed-class stays `not_planned`.
- Four resource axes reported separately (proxy, graph-span N2q/D2q/Dc,
  optional Qiskit compile, pencil estimator work).
- No manuscript edits; results PDF consumes the immutable ledger via the
  verified contract and never rediscovers runs when a contract is supplied.

## Intentionally left unexecuted

- The six-regime HH production matrix (nph1 + nph3, four methods) under the
  typed campaign.
- The `N_b=4` SIAM pilot, including its family-stratified graph-span vs
  Qiskit-compile ranking receipt (implemented, not run).
- Qiskit-compiled resource stage (`include_qiskit_compile`) on any cell.
- Any remote/CHTC submission; any push.

## Operational notes

- `git pull --rebase` was required before each commit: the branch has **no
  remote-tracking upstream**, so there is nothing to rebase onto; the command
  also aborts on the shared checkout's unrelated dirty files. No other
  session's files were moved, staged, or stashed.
- Producer product stages (transitions / Green function / spectral) operate
  on the exchange-maintained terminal **record** support; methods without a
  record support (direct Davidson, Krylov) complete their trace stage but
  their product stages are not yet implemented — they stay honestly
  `pending` in the contract rather than being faked.

## Addendum 2026-08-27: switch policy became part of the cell identity

Commit `paper-iii: make the record-to-Davidson switch a declared cell identity`.

The escalation rule was hard-coded, so changing it (the author's stated
intent: a threshold-based switch to adaptive growth) would have produced
cells with unchanged ids but changed behavior -- the ledger would have
hard-conflicted instead of letting old and new evidence coexist.
`MethodPolicy` now carries the trigger and its parameters and digests into
`StaticQSECellKey`. Canonical `pool_exhausted_only` is enforced for public
method ids; `window_residual_stall(stall_rounds, improvement_ratio)` runs as
an `early_switch` ablation cell. Trigger predicates read only the selector's
online round trace, structurally guarded by `assert_online_only_receipt`.

Because this changes the cell payload, **every digest above was regenerated**;
the smoke was re-run and the recorded digests are the post-change ones. No
production evidence existed, so nothing was invalidated.

Observed on the real SIAM `N_b=2` receipt: canonical does not fire
(`residual_converged` at 8 records), while `stall(2, 0.5)` hands off at 4
records and `stall(1, 0.9)` at 5 -- the escalation experiment is now a
declared run rather than a code edit. The online residual is **not
monotone** there (round 5 `0.531` rises to round 6 `0.717`), so the stall
rule compares against a running best rather than the previous round.

nph1 lock re-verified after the change: 93/93 prefix supports recomputed,
48/48 stored rows bit-identical, and `record_selection.py`,
`exchange_maintenance.py`, `core.py`, and the growth-trace campaign remain
untouched.

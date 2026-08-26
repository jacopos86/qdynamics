# Handoff — implement standard adaptive QSE as a Paper III benchmark arm

Author: Claude (planning). Executor: Codex (repo access).
Contract: read `agent_guidance/shared/agent-handoff-contract.md` first; this
document fills in only what is specific to the task.
Protocol this arm must satisfy: `agent_guidance/qse/paper-iii-comparison-protocol.md`.

## Anchors

| field | value |
|---|---|
| Checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` (NOT the `~/Documents/Holstein_implementation` iCloud mirror) |
| Branch / SHA at authoring | `paper-ii-exchange-selector` @ `927b34b9` |
| Test baseline | `python3 -m pytest test/test_qse_record_selection.py test/test_qse_compiled_cost_selection.py test/test_qse_spectra_core.py test/test_qse_spectra_imports.py test/test_qse_spectra_io_cli.py test/test_qse_compiled_costs.py test/test_qse_exchange_maintenance.py test/test_paper_iii_matched_accuracy_campaign.py -q` → **67 passed**. Repo-wide collection has ~55 pre-existing `docs.*` import errors unrelated to this lane; ignore them. |
| Entry point after work | `PYTHONUNBUFFERED=1 nice python3 pipelines/exact_bench/paper_iii_matched_accuracy_campaign.py --regime-set nph1 --output-json output/diagnostics/paper_iii_matched_accuracy_20260826_v1/nph1.json` |

## Why this exists (do not re-derive)

Paper III's comparison is a **three-way method comparison**:

1. **standard QSE with adaptive growth** — the external benchmark. **MISSING. This task builds it.**
2. **ours** — cost-weighted geometric selection + certified exchange (already implemented).
3. **real-time Krylov** — external benchmark, different construction family (already implemented in `pipelines/exact_bench/paper_iii_qse_comparator_matrix.py::_krylov_arm`).

The current campaign's other arms (`fixed_class`, `cheapest_first`,
`input_order`) are **controls, not methods**. They stay, but they are not the
benchmark.

**Settled decision, do not reopen:** an arm produced by reconfiguring our own
selector is an ABLATION, never a benchmark. A previous attempt added a
`residual_guided_adaptive` arm that was byte-identical to the ablation
`no_cost_discount`; it was removed. Do not implement this arm by setting
weights on `StaticRecordSelectionConfig`.

## Scope

In scope:
- new module `pipelines/qse_spectra/adaptive_qse_benchmark.py`
- new tests `test/test_adaptive_qse_benchmark.py`
- integration edit to `pipelines/exact_bench/paper_iii_matched_accuracy_campaign.py`
- integration edit to `pipelines/reporting/build_paper_iii_campaign_report.py` (add the arm to `_ARMS`)

Out of scope (reason given):
- `pipelines/qse_spectra/record_selection.py` and `exchange_maintenance.py` — our method; changing them invalidates committed evidence.
- `MATH/paper_details/**` — manuscript prose is the author's; this task produces evidence only.
- `pipelines/excited_dynamics/**` — Paper III dynamics lane is frozen pending a separate algorithm.

Shared-resource limits: **one heavy local job at a time; hard 10 GB agent RAM
ceiling** (shared machine). The `nph1` and `hubbard_l2` tiers are cheap
(minutes). Do not run `--regime-set paper_i` (nph7) as part of this task.

## The architecture to implement

Standard adaptive QSE in the quantum-Davidson sense. The defining difference
from our method: it **synthesizes a new direction from the residual** rather
than selecting an operator from a fixed alphabet, and it carries **no
hardware-cost term**.

Given reference `|psi0>`, Hamiltonian `H`, target root count `R`, and a
subspace `V_l = [v_1 ... v_l]` (physical vectors, not operator labels):

1. **Seed.** `v_1 = |psi0>` normalized. If `R > 1`, seed additionally with the
   q0-projected images of a declared small seed set (see "Seed policy" below).
2. **Solve.** Build `S_ij = <v_i|v_j>`, `M_ij = <v_i|H|v_j>`; solve the same
   canonically-orthogonalized pencil our code uses. Reuse
   `pipelines.qse_spectra.core.compute_qse_spectra`-equivalent stabilization
   (relative overlap cutoff) so the comparison is not a stabilization artifact.
   **Requirement:** the retained-support convention must be identical to ours.
3. **Residual.** For each target root `nu <= R` with Ritz value `eps_nu` and
   Ritz vector `|Psi_nu>`, form `r_nu = (H - eps_nu)|Psi_nu>`.
4. **Precondition (Davidson correction).** `t_nu = (D - eps_nu I)^{-1} r_nu`
   where `D = diag(H)` in the computational basis, with a guarded denominator:
   entries with `|D_kk - eps_nu| < delta` are floored to `delta`
   (`delta = 1e-8`, exposed as config). Preconditioning is what makes this
   Davidson rather than Lanczos; do not skip it.
5. **Orthogonalize and admit.** Project `t_nu` against the current retained
   frame (same projector convention as ours), normalize; admit if the
   orthogonal component fraction exceeds a linear-independence floor
   (`1e-12`, same value our floor uses). Otherwise try the next root; if no
   root yields an admissible direction, stop with `POOL_EXHAUSTED_EQUIVALENT`.
6. **Stop.** Terminate when `max_{nu<=R} ||r_nu|| <= eps_residual`, or on the
   safety cap `max_dimension`. Report which.

Iterate 2–6, admitting one direction per iteration.

### Seed policy (declare it; it is a fairness axis)

Use the **same q0 projection convention** as our arm. Seed with `|psi0>` plus
the q0-projected images of the declared fixed linear-response class records
(the same set `_LINEAR_RESPONSE_FAMILIES` used elsewhere), because a
single-vector seed cannot represent `R=6` roots at iteration 1. Record the seed
set size in the artifact. **Do not** seed from our selected support.

### Cost accounting (this is the subtle part — get it right)

This arm does **not** consume the record alphabet, so per-record compiled costs
do not apply. Cost it the way the Krylov arm is costed: each synthesized
direction requires preparing a state, and we charge the **same deterministic
graph-span proxy** used everywhere else, applied to the operator that generates
it. Concretely:

- Charge each admitted direction as one first-order Trotter step of `H`,
  costed by `MarrakeshGraphSpanCostOracle` under `two_qubit_only_v1`, exactly
  as `_krylov_arm` charges `step_2q` (see `paper_iii_qse_comparator_matrix.py`).
- Report the same triple as every other arm: `n2q`, `d2q`, `dc` built from
  `estimate.c_hat_2q`, `c_hat_d`, and `c_hat_d + c_hat_1q`.
- Put the costing convention string in the artifact so the manuscript can state
  it. **If you conclude this costing is unfair to either side, stop and report
  rather than inventing a different one** — the convention is a claim.

## Increments

Proceed through 1–3 autonomously; **pause and report before increment 4.**

### Increment 1 — module + unit tests

Goal: `adaptive_qse_benchmark.py` exists with a pure, tested core.

Implement `run_adaptive_qse_benchmark(hamiltonian, prepared_state, *, target_roots, eps_residual, max_dimension, seed_elements, ...) -> dict`. Return per-iteration rows: dimension, `max_root_residual`, root energies, admitted-direction novelty fraction, cumulative resource triple, and a terminal `stop_reason` in {`RESIDUAL_CONVERGED`, `MAX_DIMENSION`, `POOL_EXHAUSTED_EQUIVALENT`}.

Tests (`test/test_adaptive_qse_benchmark.py`) must include:
- a small analytic Hamiltonian (e.g. random Hermitian 16x16, fixed seed) where the method converges to the exact lowest `R` eigenvalues to `1e-10`;
- **preconditioner guard**: a case with `D_kk == eps_nu` does not produce NaN/inf;
- **independence floor**: feeding a direction already in the span is rejected rather than admitted;
- monotone non-increase of `max_root_residual` over iterations is NOT asserted (Davidson is not monotone) — instead assert the terminal residual meets the target when `RESIDUAL_CONVERGED`.

Expected result: new tests pass; the 67-test baseline stays green.

Stop condition: if the method does not converge on the analytic case, the
preconditioner or the stabilization convention is wrong — stop and report with
the residual trace, do not tune thresholds to force convergence.

### Increment 2 — campaign integration

Goal: the campaign emits a fourth arm, `adaptive_qse`, resolved by the same
`resolve_cell` logic.

In `paper_iii_matched_accuracy_campaign.py`: run the benchmark over the same
residual-rung ladder as ours (`RESIDUAL_RUNG_LADDER`), producing one rung row
per tolerance with `max_root_abs_error` measured against the **cached exact
reference** (use `load_exact_reference`; do not recompute), plus `resources`.
Add `"adaptive_qse": resolve_cell(adaptive_rungs, eps_e, extendable=True)` to
each cell.

Expected result: `--regime-set hubbard_l2` completes in minutes and every cell
has an `adaptive_qse` key with a `REACHED`/`NOT_REACHED_WITHIN_POOL` status.

Stop condition: if `adaptive_qse` reaches machine precision at trivial cost in
every regime, suspect it is being handed our support or the exact reference —
audit the seed path and report.

### Increment 3 — reporting

Goal: the arm appears in the gaps figure and cost tables.

In `build_paper_iii_campaign_report.py`: add `("adaptive_qse", "benchmark: adaptive QSE")`
to `_ARMS`, and plot its root energies in `_gap_figure` as a third series
(suggest colour `#2f8f4e`, marker `^`). Keep the "ours" vs "benchmark:" label
convention — the author requires method identity to be readable from the figure.

Expected result:
`python3 pipelines/reporting/build_paper_iii_campaign_report.py --campaign-json output/diagnostics/paper_iii_matched_accuracy_20260826_v1/nph1.json --output-dir output/pdf/paper_iii_campaign_nph1`
writes a PDF whose tables have five columns and whose figure has three method
series. **Render it to PNG and look at it** (`pdftoppm -png -r 80`); a table
running off the page has happened here before.

### Increment 4 — PAUSE, then cheap-tier runs

Report increments 1–3 with the artifact paths and one rendered page image.
After the author confirms the arm is sane, run `hubbard_l2`, `nph1`, and
`nph3` tiers and report the three reports.

## Traps specific to this task

- **`output/` is gitignored.** Evidence JSONs and PDFs will not appear in a
  commit. Commit code and tests; reference artifacts by path in the report-back.
- **Do not recompute the exact reference.** `load_exact_reference` is
  content-addressed under `output/reference_store/paper_iii_exact_sector/` and
  verifies identity on read. Recomputing per arm is a protocol violation and
  wastes the slot.
- **The safety cap must not silently redefine the method.** If `max_dimension`
  is hit before `eps_residual`, the row must carry `stop_reason = MAX_DIMENSION`
  and must NOT be reported as a converged rung (contract §5).
- **Equal pool caps do not prove equal alphabets.** This arm does not use the
  alphabet at all; state that explicitly in the artifact rather than implying a
  shared pool.
- Another agent commits in this checkout. Commit at increment boundaries.

## Report back

Use the contract §6 block per increment, with commit SHAs. Additionally
report, for `nph1`: the per-regime terminal dimension and `C*(1e-4)` triple for
`adaptive_qse` beside `ours`, so the author can judge whether the benchmark is
strong. A benchmark that looks trivially weak is more likely misimplemented
than genuinely weak — say so if you see it.

# Handoff — Paper I: Bundle 3 supersedes Bundle 5 as HH headline evidence

Written 2026-08-25 by the Claude session that ran the B7/B8/B9 campaigns and
the route audit. Executor: the Paper-I manuscript agent.

**This document supersedes `HANDOFF_PAPER_I_B5_SINGLE_EVIDENCE_20260822.md`.**
That handoff said "Bundle 5 only — do not reopen." Reopen it. The reason is
below and it is a method-fidelity problem, not a numbers problem.

## What changed

Bundle 5 (`b3mr`, cluster 9674089) does not execute the algorithm Paper I
describes. Read back from its own executed checkpoints:

- `split_parent_count: 8`, `split_child_count: 18-27`, `unsplit_singleton_count: 0`

B5's Phases 0-I shortlist ~123 **macro parents**, which are split into Pauli
children between Phase I and Phase II. Paper I specifies the flat 948/6508
deduplicated singleton pool screened through Phases 0-III with identity
exposure and no in-pipeline construction. B5 also carries no compile-cost term
and plain (not min-cost) commutation insertion.

The cause was a silent default: `RAAdaptRequest.adapter` defaults to
`SinglePauliWordCandidateAdapter`, the macro-splitting adapter. B5's campaign
never named an adapter, so it inherited macro splitting while announcing only
a gradient change. Bundle 6 inherits the same route.

## Use Bundle 3 instead

Bundle 3 (cluster 9664325), arms `b_depth_append_ra`,
`b_depth_costexp_always_open`, `b_depth_costexp_plateau_position`. Verified
against its executed receipts, not its declared config:

| property | B3 executed |
|---|---|
| pool | `split_parent_count: 0` — flat singleton throughout |
| compile cost | `explicit_depth_qiskit_cost_v1`, source `backend_transpile_v1` (exact transpiles, not the proxy) |
| insertion positions | `termwise_cross_component_commutation_min_cost_realization_v1` — lowest-cost, not first |
| inner optimizer | POWELL, `xtol=1e-4`, `ftol=1e-8`, `maxiter=200` — identical to the AAVQE tolmatch comparator |
| Phase-0 shortlist | `phase0_active_score_effective_competition_shortlist_v2` — adaptive |
| certification | `ledger_closure_passed: True`, `paper_i_table_work_status: ok`, `estimator_accounting_complete: True` |

Coverage: append arm 6/6 regimes; both insertion arms 3/6 (weak sector).
The six strong-sector insertion cells are running now as cluster 9675078
(resubmitted fresh 2026-08-23 from B3's own frozen protocols, horizon 75)
and are at rounds 13-45 of 75 as of this writing.

## Bundle 5 moves to the appendix

Per the user's decision (2026-08-25), B5 is retained as an **optional method**
in the appendix, not retracted. It is a legitimate run of a different
traversal: macro-parent shortlisting with post-Phase-I splitting. It must be
labelled as such wherever it appears, and must not be presented as the
Paper-I route. Its S_alg values remain duplication-corrected per
`FINDINGS_ESTIMATOR_LEDGER_DUPLICATION_20260821.md` §5, §13.

## Stationarity: a null result, consistent across both systems

Bundle 9 is B3's route with the stationary-source approximation dropped
(measured residual). On all 12 shared cells B3 and B9 are equivalent: 6 have
identical terminal dE and S_alg agrees to within ~1%. The one apparent
exception (`costexp_plateau_position | intermediate_weak`, B9 at half the
estimator cost) was an artifact of a run truncated at depth 37; at the matched
horizon of 50 it is 707213 vs 703294, a 0.6% difference.

The L=3 pure-Hubbard diagnostic reaches the same conclusion independently:
measured-residual and stationary-source produced identical accepted
trajectories through k=12 (common digest `508878e744...`), with stationary
saving 65 gradient queries (0.84%).

Note for the text: under the stationary-source approximation the active-only
predicted gain is zero, so "stationary/joint" and "stationary/marginal" denote
the same score and must not be presented as two independent runs.

## On whitening and metric-informed directions — not a confound

B3 and B9 refit in `supported_fs_whitened_fixed_v1`, and RA supplies
metric-informed initial Powell directions. Neither is an unmatched advantage
over AAVQE and neither should be "controlled for":

Whitening is RA using **its own Gram geometry** — the Fubini-Study metric it
already computes as part of the algorithm (the `N_metric` term in S_alg, which
is charged to RA's own estimator budget). AAVQE has no Gram and therefore no
such procedure available to it. There is nothing to match. The same holds for
the metric-informed Powell directions: they come from the same Gram.

What *is* matched, and what the comparison rests on, is the inner optimizer
contract: POWELL at `xtol=1e-4`, `ftol=1e-8`, `maxiter=200` for both RA and the
AAVQE tolmatch comparator. That parity is verified from executed receipts.

**Resolved: Holstein and L=3 do NOT use the same coordinate treatment.**
Both use RA's Fubini-Study metric, but apply it differently, and this was
confirmed from the implementation rather than from labels.

Holstein (B3, B9) sets `accepted_refit_coordinate_chart:
supported_fs_whitened_fixed_v1`. `build_supported_fs_powell_chart` in
`pipelines/static_adapt/accepted_refit.py` constructs a genuine change of
variables:

    whitened_to_logical = base_to_logical @ whitened_to_base
    logical_to_whitened = pinv(whitened_to_logical, rcond=1e-12)
    raw_metric_in_chart = whitened_to_base.T @ base_gram @ whitened_to_base

and then hard-asserts that `raw_metric_in_chart` equals the identity
("accepted-refit raw FS chart failed orthonormality"). Powell therefore
optimizes in coordinates where the FS metric is the identity. The CLI states
the same: one fixed raw-Fubini-Study orthonormal chart per Powell invocation,
as opposed to the `native_v1` default which preserves the optimizer chart.
`expanded_runtime_projected_logical_v1` is the base chart being whitened.

L=3 labels every RA arm `native_with_metric_informed_directions`: optimization
stays in native logical amplitudes and the retained Fubini-Study support
supplies only Powell's initial direction set - explicitly not a persistent
whitened chart.

**Second cross-system difference: the candidate pool.** The L=3 representative
runs the ordered 24-generator **intact macro** pool; B3 and B9 run the flat
deduplicated singleton Pauli pool. Note the tension with the B5 demotion above:
B5 is demoted precisely for shortlisting macro parents instead of the flat
singleton pool. The two situations are not identical -- 24 intact Hubbard
generators is a different regime from Holstein's 948/6508, and no in-pipeline
splitting occurs on L=3 -- but Paper I must not describe a single pool
treatment as if it covered both systems.

Otherwise the L=3 representative is B3-like: stationary active gradient, full
joint Phase-III gain, adaptive shortlisting at gamma=0.9, exact Qiskit
marginals in selection. It differs from B3 only in the pool and the refit
coordinate treatment.

So: same metric, two different uses. Holstein applies it as a coordinate
transformation; L=3 applies it as a preconditioner on the initial directions.
Neither is a fairness problem against AAVQE, which has no Gram either way, but
Paper I cannot describe RA's use of the Fubini-Study metric in one sentence
that covers both systems. Either align the two campaigns on one treatment, or
state the difference explicitly per system.

## Do not use

Bundle 8 (cluster 9675127) is held and unusable. Its completed cells recorded
`compile_cost_policy: explicit_depth_qiskit_cost_v1` while scoring on
`marrakesh_graph_span_v1` — the label claims exact marginals, the numbers are
the analytic graph-span proxy. It also uses `joint_minus_active_only` candidate
gain where B3 and B9 use `joint_total_gain_v1`, so it is not comparable on
selection semantics either. Bundle 7 was removed 2026-08-23 and not resubmitted.

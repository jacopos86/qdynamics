# Findings — the alpha-star candidate warm start does not pay for itself

Bundles 5 and 6 form a complete one-axis comparison: 18 matched cells, identical
in every respect except that B6 seeds each newly inserted candidate's coordinate
at ±α\* — the Schur trust step — instead of at zero.

Both bundles are complete (18/18), every cell self-certified
(`ledger_closure_passed: true`, `paper_i_table_work_status: ok`), and every
S_alg below is duplication-corrected under `staged_candidate_primitive_reuse_v1`.

## What α\* is

α = min(g_lcb / h_eff, ρ / √F_red), clipped at the trust boundary when h_eff ≤ 0,
with predicted gain g·α − ½h·α². In code this is `SchurTrustStep.alpha_abs`
(`pipelines/static_adapt/schur_warm_start.py`), signed by an evaluate-both policy.

B5 starts each new candidate coordinate at **zero angle**
(`candidate_geometry_chart: exact_ordered_insertion_zero_angle_v1`). B6 sets
`adapt_schur_warm_start_mode: append`, seeding that one coordinate at ±α\*.
Accepted parameters are untouched in both; the joint-step response warm start
rides both lanes identically.

**It acquires nothing new.** The seed is read from
`exact_decision.trust.proposed_coordinate_values`, a quantity phase-3 scoring
has already computed and already charged. So the only channel to S_alg is the
number of optimizer callbacks needed to converge — the expectation being that a
better starting point needs fewer.

Engagement was total: `static_adapt_schur_warm_start_v1` fires with
`status: seeded` in every round of every lane, zero guard rejections. Whatever
the result shows is a real effect of α\*, not a suppressed one.

## Result

| cell | k\* | B5 S_alg | B6 S_alg | ratio | B5 ΔE | B6 ΔE | ratio |
|---|---|---|---|---|---|---|---|
| append W–W | 11 | 6,277 | 6,749 | **1.075** | 6.604e-04 | 6.604e-04 | 1.000 |
| append I–W | 9 | 5,599 | 5,148 | **0.919** | 5.052e-04 | 5.271e-04 | 1.043 |
| append S–W | 10 | 5,790 | 10,436 | **1.802** | 3.205e-06 | 4.562e-06 | 1.424 |
| append W–S | 75 | 389,622 | 386,780 | **0.993** | 4.556e-07 | 3.830e-07 | 0.841 |
| append I–S | 75 | 427,501 | 461,845 | **1.080** | 2.798e-07 | 7.472e-07 | 2.671 |
| append S–S | 27 | 42,018 | 54,286 | **1.292** | 2.284e-05 | 2.282e-05 | 0.999 |
| plateau W–W | 11 | 8,736 | 9,188 | **1.052** | 6.604e-04 | 6.604e-04 | 1.000 |
| plateau I–W | 9 | 5,599 | 5,148 | **0.919** | 5.052e-04 | 5.271e-04 | 1.043 |
| plateau S–W | 10 | 10,294 | 13,425 | **1.304** | 5.988e-06 | 3.265e-06 | 0.545 |
| plateau W–S | 75 | 772,024 | 704,312 | **0.912** | 2.831e-06 | 1.542e-06 | 0.545 |
| plateau I–S | 75 | 747,963 | 851,719 | **1.139** | 2.258e-07 | 4.880e-07 | 2.162 |
| plateau S–S | 27 | 113,326 | 107,384 | **0.948** | 2.283e-05 | 2.290e-05 | 1.003 |
| always W–W | 11 | 17,776 | 18,603 | **1.047** | 5.170e-04 | 5.170e-04 | 1.000 |
| always I–W | 9 | 12,358 | 12,748 | **1.032** | 5.052e-04 | 5.052e-04 | 1.000 |
| always S–W | 10 | 14,031 | 16,085 | **1.146** | 2.078e-06 | 1.744e-06 | 0.839 |
| always W–S | 75 | 1,046,764 | 970,255 | **0.927** | 6.314e-06 | 1.458e-06 | 0.231 |
| always I–S | 75 | 1,006,016 | 1,035,917 | **1.030** | 2.172e-07 | 1.082e-07 | 0.498 |
| always S–S | 27 | 183,767 | 222,429 | **1.210** | 1.340e-07 | 3.029e-07 | 2.260 |

**S_alg k*   : median 1.049  worse 12/18  range 0.912–1.802
**S_alg term : median 1.068  worse 14/18  range 0.912–1.139
**dE k*      : median 1.000  worse 11/18  range 0.231–2.671

## Reading

**α\* costs more estimator work than it saves.** At the reported round k\* it is
worse in 12 of 18 cells with a median of 1.049; at the terminal round worse in
14 of 18 with a median of 1.068. The seeding does not pay for itself.

**But it is not uniformly harmful, and a claim that it is would not survive
review.** Accuracy at k\* is unchanged across the entire weak sector — bit-identical
in five of nine weak cells — and swings hard both ways in the strong sector:
always W–S is 4.3× *better* with α\* (1.458e-06 vs 6.314e-06), while append I–S is
2.7× worse. The honest claim is the cost one.

**Why the cost claim is the safe one.** Where ΔE is bit-identical, the optimizer
reached the same converged optimum from a different start, so S_alg is a
like-for-like measure of the work required to get there. Those cells still show
α\* costing more. Where ΔE differs the trajectories diverged and the comparison
is not like-for-like, but those cells split in both directions and do not rescue
the cost result.

## Scope limit

Both bundles run the **macro-to-singleton** traversal: Phases 0–I shortlist ~123
macro parents, which are split into guarded Pauli children between Phase I and
Phase II. Paper I's appendix specifies the flat 948/6508 singleton pool screened
throughout with no in-pipeline construction. So this result characterises α\*
**on the route B5/B6 executed, which is not the route the manuscript describes.**

It is a valid one-axis comparison and a usable negative result about α\*. It is
not evidence about α\* on the paper-faithful route, and should not be presented
as such. If α\* matters to the paper, the comparison to run is B7-with-α\*
against B7.

## Provenance

- B5 = cluster 9674089 (`b3mr`), B6 = cluster 9674592 (`b6mr`)
- Values: `results_tooling/kstar_tables.json`, keys `b5` and `b6`,
  fields `salg_star_corrected` / `salg_term_corrected` / `dE_star`
- ΔE convention per cell in `dE_energy_source`: checkpoint terminal accepted
  energy where k\* equals the terminal round, stream trace otherwise
- Regenerate this table from the tables with the snippet in the session
  scratchpad; no value here was transcribed by hand

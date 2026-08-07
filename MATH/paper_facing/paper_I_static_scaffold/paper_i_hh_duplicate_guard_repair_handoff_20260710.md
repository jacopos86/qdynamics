# Paper-I HH duplicate-guard repair handoff

## Purpose

This file hands the active implementation repair to a new coding agent.  The
current agent has stopped making code changes.  Continue from the existing
working tree; do not recreate, revert, or overwrite the work already present.

Do not use RepoPrompt for this repair.  A RepoPrompt search stalled for more
than an hour and was interrupted without producing a usable result or changing
files.  Use targeted local source inspection, telemetry, tests, and a native
review agent only if a second review is needed.

The full diagnostic record is:

```text
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_full_reopt_implementation_error_20260710.md
```

## Scope and constraints

- Repair only the Paper-I Hubbard--Holstein SNAKE full-window/full-reoptimization
  duplicate-exhaustion behavior.
- Preserve the physical operator lanes, Hamiltonian settings, pool, optimizer,
  cost model, seeds, scientific thresholds, and no-batching setting.
- Preserve full ansatz reoptimization at every adaptive step and the full
  insertion window.
- Do not edit `MATH/paper_details/Paper_I.tex`.
- Do not rebuild manuscript figures, tables, or PDFs during this diagnostic.
- Do not launch another six-regime matrix until the fixed-cap weak--weak route
  passes the checks below.
- Do not stop or alter unrelated local or CHTC runs.
- Use path-limited Git inspection; do not stage generated outputs.

## Work already completed

The following implementation work is present in the working tree:

```text
pipelines/static_adapt/adapt_pipeline.py
pipelines/static_adapt/lane_routes.py
test/test_static_adapt_full_reopt_duplicate_guard.py
test/test_static_adapt_lane_routes.py
```

Implemented behavior includes:

- final-full-refit support for `adapt_reopt_policy in {"windowed", "full"}`;
- coherent final-refit metadata when the last step already optimized the full
  prefix;
- exact runtime child/candidate identity tracking for repeated records;
- post-refit rollback when a previously committed exact identity is repeated
  with no meaningful realized energy descent;
- branch-local blocked-identity filtering in the beam route;
- ordinary and beam history telemetry under `zero_gain_duplicate_filter` and
  `zero_gain_duplicate_guard`;
- explicit stop reason `zero_gain_duplicate_guard_exhausted` when the filtered
  admission source is empty;
- Route-C plateau exclusion so the new guard does not replace its existing
  duplicate policy;
- fixed physical-lane controller caps through
  `clamp_controller_cap_pair_for_lane_route()`.

Targeted duplicate-guard tests last passed:

```text
pytest -q test/test_static_adapt_full_reopt_duplicate_guard.py -x
10 passed
```

The lane-route tests define the intended fixed-cap behavior but should be run
again by the receiving agent together with the duplicate-guard tests.

## Current code-policy boundary

At `2026-07-10 11:59:31 CDT`, the physical-lane cap policy changed in:

```text
pipelines/static_adapt/lane_routes.py
pipelines/static_adapt/adapt_pipeline.py
test/test_static_adapt_lane_routes.py
```

For the current less-aggressive route, the intended fixed effective caps are:

```text
N1 = 14
N2 = 7
N3 = 7
```

The required telemetry identifier is:

```text
physical_route_fixed_to_effective_shortlist_caps_v1
```

The first five rows of the paused matrix were started under the older
maturity-cap scheduler.  Strong--strong started after the cap-policy change and
was interrupted before producing `json/result.json`.  Therefore the paused
matrix is not a uniform comparison.

## Preserved diagnostic outputs

Paused matrix root:

```text
raw_outputs/paper_i_hh_fullreopt_duplicate_guard_repair_matrix_20260710/less_aggressive_1p75_fullwindow_fullreopt
```

Only the supervisor and worker associated with that root were stopped.  A
post-interrupt process check found no remaining associated worker.

Completed rows from the old scheduler all stopped through duplicate exhaustion:

```text
weak_weak:          history=6, ops=5, unique=5, abs_delta_e=2.2543954537745048e-2
intermediate_weak:  history=5, ops=4, unique=4, abs_delta_e=1.0961489240935435e-1
strong_weak:        history=6, ops=4, unique=4, abs_delta_e=1.4053442005785843e-3
weak_strong:        history=7, ops=5, unique=5, abs_delta_e=1.6002226059966174e-1
intermediate_strong: history=8, ops=5, unique=5, abs_delta_e=1.4535346507174074e-1
strong_strong:      partial run.log only; no result JSON
```

The weak--weak error is orders of magnitude above the current Paper-I value of
approximately `4.524e-4`, so these outputs are diagnostic evidence only.

Earlier narrow validation artifacts are preserved under:

```text
raw_outputs/paper_i_hh_fullreopt_duplicate_guard_repair_smoke_20260710/
raw_outputs/paper_i_hh_fullreopt_duplicate_guard_repair_validation_20260710/
```

The latest cached depth-30 weak--weak validation is:

```text
raw_outputs/paper_i_hh_fullreopt_duplicate_guard_repair_validation_20260710/weak_weak_depth30_v3
```

It stopped at history length 6 with five unique committed operators and
`zero_gain_duplicate_guard_exhausted`.  It proves that repeated zero-gain
children no longer accumulate, but it does not prove that exhaustion is
correct.

## Unresolved implementation question

The leading hypothesis is that blocked identities are filtered only after the
Phase-II/III shortlist or admission source has already been capped.  If every
record in that capped source is blocked, the guard declares exhaustion even
when lower-ranked, unblocked records may still exist in the upstream ranked
record population.

The intended narrow repair, if this hypothesis is confirmed, is:

1. Filter blocked exact identities from the broader ranked/scored candidate
   population before final shortlist/admission-source construction.
2. Refill the shortlist up to the unchanged phase/admission cap from the next
   eligible ranked records.
3. Declare `zero_gain_duplicate_guard_exhausted` only when the broader eligible
   population itself contains no unblocked record.
4. Preserve all existing scores, lane thresholds, lane protection, shortlist
   caps, optimizer behavior, and scientific tolerances.

Do not implement this from the hypothesis alone.  First prove where the
eligible lower-ranked records disappear in the current beam path.

## Required investigation

1. Confirm in source and telemetry that the physical route resolves
   `N1=14`, `N2=7`, `N3=7` at every iteration and reports
   `physical_route_fixed_to_effective_shortlist_caps_v1`.
2. At the weak--weak exhaustion step, compare the blocked identities against:
   the upstream full/scored records, Phase-II shortlist, Phase-III shortlist,
   and final admission source.
3. Determine whether unblocked lower-ranked records existed upstream but were
   discarded before duplicate filtering.
4. Audit the repeated candidate's Phase-II and Phase-III novelty, predicted
   gain, realized refit gain, insertion position, and position-dressed tangent.
5. Decide whether novelty is mathematically wrong or locally valid before a
   nonlinear full refit that realizes no descent.
6. Add a narrow regression test reproducing the confirmed fallback failure.
7. Implement only the smallest tested shortlist-refill or eligibility-ordering
   correction supported by that evidence.

Likely code anchors in `adapt_pipeline.py` are the helpers near lines
`1443-1595`, beam candidate filtering near `13087-13160`, beam materialization
near `14603-14657`, and the ordinary route near `18053-18103` and
`20149-20201`.  Re-resolve line numbers before editing because the file is
already modified.

## Validation sequence after the narrow repair

1. Run `py_compile` on the modified implementation and tests.
2. Run both targeted files:

```text
pytest -q test/test_static_adapt_full_reopt_duplicate_guard.py test/test_static_adapt_lane_routes.py -x
```

3. Run one short weak--weak execution-path smoke.  It must report the fixed
   cap policy and must not terminate by premature duplicate exhaustion.
4. If the smoke passes, run one complete depth-30 weak--weak fixed-cap
   diagnostic with the same scientific settings.
5. Compare its trajectory directly with the current Paper-I weak--weak SNAKE
   trajectory, including the early accepted operators, error progression, and
   stopping reason.
6. Stop and report before launching any other regime.

Do not proceed to the other five regimes if weak--weak still stops through
premature duplicate exhaustion or remains orders of magnitude worse than the
current Paper-I trajectory.

## Working-tree warning

At handoff time, the relevant tree is intentionally dirty:

```text
M  pipelines/static_adapt/adapt_pipeline.py
?? pipelines/static_adapt/lane_routes.py
?? test/test_static_adapt_full_reopt_duplicate_guard.py
?? test/test_static_adapt_lane_routes.py
?? MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_full_reopt_implementation_error_20260710.md
```

The untracked status does not mean these files are disposable.  Treat every
existing change as user/agent work that must be preserved and reviewed before
any additional patch.

## Files the receiving agent may edit

Only after completing the investigation:

```text
pipelines/static_adapt/adapt_pipeline.py
test/test_static_adapt_full_reopt_duplicate_guard.py
```

Edit `pipelines/static_adapt/lane_routes.py` or
`test/test_static_adapt_lane_routes.py` only if the fixed-cap contract itself
is demonstrably incorrect; do not change that contract as part of duplicate
fallback repair.


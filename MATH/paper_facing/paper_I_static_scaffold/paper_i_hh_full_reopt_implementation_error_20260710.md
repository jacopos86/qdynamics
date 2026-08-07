# Paper-I HH full-reopt/full-window implementation error investigation

## Scope

This note tracks a diagnostic-only repair line for the Paper-I Hubbard--Holstein
SNAKE reruns using physical operator lanes with less aggressive shortlisting,
full insertion window, and full reoptimization at each accepted step.

Target behavior requested by the user:

- physical-operator lane route;
- no batching;
- full ansatz reoptimization at every adaptive step;
- full insertion window;
- less aggressive shortlisting relative to the promoted physical-lane rows;
- rerun enough evidence after repair before any broader matrix rerun.

No manuscript table, figure, or PDF result should consume the failed runs below.

## Failed diagnostic runs

Output root:

```text
raw_outputs/paper_i_hh_physical_operator_lanes_1p75_fullwindow_fullreopt_powell200_nobatch_20260709_v1/less_aggressive_1p75_fullwindow_fullreopt
```

Comparison report:

```text
output/pdf/paper_i_hh_1p75_fullwindow_vs_current_paper1_20260710/paper_i_hh_1p75_fullwindow_vs_current_paper1_20260710.pdf
```

All six runs reached the requested adaptive horizon, but the results are
diagnostically invalid.  In weak--weak, energy improves for the first four
accepted steps and then stalls:

```text
k=1  delta_E=-1.132782218537319
k=2  delta_E=-0.9922177814626808
k=3  delta_E=-0.010380746699305665
k=4  delta_E=-0.010428418262123373
k>=5 delta_E approximately 0
```

Despite the zero realized gain, the run continues admitting the same child
operator:

```text
paop_full:paop_cloud_p(site=1->phonon=0)::child_set[4]
```

Weak--weak selected-operator count from `adapt_vqe.history`:

```text
paop_full:paop_cloud_p(site=1->phonon=0)::child_set[4]  26 selections
```

Weak--weak final `adapt_vqe.operators`:

```text
28 operators total
5 unique labels
24 copies of paop_full:paop_cloud_p(site=1->phonon=0)::child_set[4]
```

This is the main code-error signature: exact duplicate zero-gain records are
allowed to dominate the ansatz after the first useful descent.

## Settings mismatch

Failed weak--weak settings:

```text
adapt_reopt_policy=full
adapt_window_size=99
adapt_full_refit_every=1
adapt_final_full_refit=true
adapt_insertion_mode=always
phase3_geometry_window_size=99
allow_repeats=true
eps_energy_termination_enabled=false
adapt_rollback_mode=parameter
adapt_rollback_tolerance=0.0
```

Runtime metadata nevertheless reports:

```text
adapt_vqe.final_full_refit.requested=false
adapt_vqe.final_full_refit.executed=false
adapt_vqe.final_full_refit.nfev=0
```

The code path in `pipelines/static_adapt/adapt_pipeline.py` gates final
full-prefix refit on `adapt_reopt_policy_key == "windowed"`:

```text
21212  # -- Final full-prefix refit (windowed policy only) --
21215  adapt_reopt_policy_key == "windowed" and adapt_final_full_refit_val
21239  adapt_reopt_policy_key == "windowed"
```

This conflicts with the requested and serialized setting
`adapt_final_full_refit=true` when `adapt_reopt_policy=full`.

`pipelines/static_adapt/engine_support.py::_resolve_reopt_active_indices`
does support `policy_key == "full"` by returning all active indices, so the
failure is not simply that the full policy is unknown.  The concern is that
`full` is only partially wired through final/full-refit, periodic-refit, and
Phase-III scoring/window metadata.

## Additional suspicious evidence

The selected record for weak--weak `k=5` has:

```text
selected_op=paop_full:paop_cloud_p(site=1->phonon=0)::child_set[4]
selected_position=1
delta_energy=-4.440892098500626e-16
reopt_policy_effective=full
reopt_active_count=5
phase3_duplicate_penalty=0.0
novelty=0.9998088769638305
phase3_reduced_novelty=0.9998088769638305
family_repeat_cost=4.0
physical_operator_lane=phonon_displacement
```

The candidate is treated as highly novel even after it is effectively redundant
with the current optimized state.  The repeated exact child admission therefore
looks like a selection/novelty/cooldown failure, not only a final-refit metadata
problem.

One record also shows inconsistent nested-window metadata:

```text
nested_refit_window.policy_requested=full
nested_refit_window.policy_effective=full
phase3_geometry_nested_refit_window.policy_requested=windowed
phase3_geometry_nested_refit_window.policy_effective=windowed
```

This suggests Phase-III scoring may still be using a windowed alias in places
even when the runtime refit policy is full.

## Initial candidate fixes to validate

1. Final full-refit metadata and execution should be consistent for
   `adapt_reopt_policy=full`.  Either:
   - request and execute final full refit for `policy in {"windowed", "full"}`;
   - or, when the last accepted step already optimized the full active set,
     mark it as requested and skipped with
     `skipped_reason=last_depth_already_full_prefix`.

2. Periodic full-refit checks that are currently guarded by
   `adapt_reopt_policy_key == "windowed"` should be audited.  For
   `adapt_reopt_policy=full`, every step is already full, so periodic refit
   should not be required, but metadata should not imply that full refit was
   disabled.

3. Phase-III geometry/window prediction should preserve `full` policy rather
   than silently converting nested geometry metadata to `windowed`.

4. Add a narrow duplicate-admission guard or reject/cooldown path for exact
   repeated child records whose realized refit energy improvement is below the
   existing energy-step tolerance.  The guard should be conservative and should
   not change ordinary ADAPT repeats that still realize descent.

5. Add tests covering:
   - `adapt_reopt_policy=full` with `adapt_final_full_refit=true` produces
     coherent final-refit metadata;
   - full-policy Phase-III nested refit metadata remains full;
   - exact duplicate zero-gain records are structurally rolled back or otherwise
     prevented from repeated admission in the full-window route.

## Oracle questions

1. Is the main failure caused by incomplete `adapt_reopt_policy=full` plumbing,
   by Phase-III geometry/window scoring still using a windowed alias, by missing
   duplicate cooldown, or by a combination?

2. What is the smallest implementation change that preserves existing
   windowed/append-only behavior while repairing full-policy semantics?

3. Where should the duplicate zero-gain guard live: before admission, immediately
   after realized refit, or as a cooldown feature for subsequent ranking?

4. What narrow unit/integration tests should be added before rerunning the HH
   smoke?

## Oracle/context-builder findings

Oracle export:

```text
prompt-exports/oracle-plan-2026-07-10-103843-full-reopt-repair-56-88a4.md
```

The Oracle/context-builder investigation found a combined implementation issue:

1. `adapt_reopt_policy=full` is supported by
   `engine_support.py::_resolve_reopt_active_indices`, but the final full-refit
   metadata/execution block in `adapt_pipeline.py` only treats `windowed` as a
   supported final-refit policy.  This explains why the failed runs serialize
   `adapt_final_full_refit=true` while `final_full_refit.requested=false`.

2. Ordinary rollback only rejects energy regressions.  It does not reject exact
   repeated candidate identities that produce zero realized energy gain.  Since
   `allow_repeats=true`, a repeated child can remain structurally committed even
   when it adds no descent.

3. Existing duplicate machinery in `plateau_acquisition.py` is Route-C
   plateau-specific.  It is not applied to normal Phase-III candidate admission.

4. The apparent `phase3_geometry_nested_refit_window.policy_requested=windowed`
   mismatch is mostly telemetry from the fixed local Phase-III geometry scoring
   window.  The optimizer-side nested refit policy can still be `full`.  This
   repair should preserve W3/Wopt decoupling unless a later explicit scoring
   semantics change is requested.

The recommended repair is therefore narrow:

- treat `adapt_reopt_policy in {"windowed", "full"}` as final-full-refit
  policy-supported;
- for `full`, mark final refit as requested and skip redundant execution with
  `skipped_reason=last_depth_already_full_prefix` when the last step already
  optimized all active coordinates;
- add a conservative post-refit duplicate guard for singleton, non-Route-C
  admissions: if an exact candidate identity was previously committed and the
  new realized energy gain is at or below tolerance, structurally roll back the
  admission and block that identity from subsequent candidate source lists;
- keep productive repeats allowed.

Candidate identity should prioritize runtime child/candidate identity over
generic family identity.  The guard should not ban all repeats; it should only
act after a repeated identity realizes no meaningful descent.

## Validation plan

After code changes:

1. Run targeted unit/integration tests for reoptimization policy and duplicate
   admission behavior.
2. Run a weak--weak HH diagnostic smoke to depth 8--10 under the same failed
   settings.
3. Require the smoke to show:
   - coherent `final_full_refit` metadata;
   - no exact child operator dominating repeated zero-gain admissions;
   - energy does not freeze while accepting the same zero-gain record.
4. Only after that, rerun the six-regime less-aggressive matrix.

## 2026-07-10 repair checkpoint

Focused code changes have been started in
`pipelines/static_adapt/adapt_pipeline.py`, with tests in
`test/test_static_adapt_full_reopt_duplicate_guard.py`.

Validation completed so far:

```text
python3 -m py_compile pipelines/static_adapt/adapt_pipeline.py test/test_static_adapt_full_reopt_duplicate_guard.py
pytest -q test/test_static_adapt_full_reopt_duplicate_guard.py -x
```

Result:

```text
6 passed
```

However, the failed weak--weak row was not a purely single-branch route.  Its
`row_manifest.json` argv includes:

```text
--adapt-beam-live-branches 3
--adapt-beam-children-per-parent 2
--phase2-no-batching
--phase3-no-batching
```

Therefore "no batching" does not mean "no beam branching" for this diagnostic
run.  The repeated zero-gain child was selected inside the beam-enabled route.
The duplicate guard must therefore cover beam branch expansion/selection, not
only the ordinary non-beam adaptive path.

The exact failed weak--weak argv source is:

```text
raw_outputs/paper_i_hh_physical_operator_lanes_1p75_fullwindow_fullreopt_powell200_nobatch_20260709_v1/less_aggressive_1p75_fullwindow_fullreopt/weak_weak/row_manifest.json
```

The next implementation question was where to apply the conservative duplicate
guard in the beam route: before child proposal creation, after branch-local
optimizer/refit evaluation, or when constructing the next beam frontier.

## Beam-route Oracle follow-up

Beam-focused Oracle/context-builder export:

```text
prompt-exports/oracle-plan-2026-07-10-111056-beam-duplicate-guard-d559.md
```

The beam-focused review found that the original non-beam repair was
insufficient because the failed row used beam branching.  The recommended
placement was a post-refit guard inside `_materialize_beam_child()`, using
branch-local history rather than global history.  The reason is that the
implementation only knows whether a candidate actually produced descent after
the branch-local optimizer/refit has run.

The follow-up also recommended that a zero-gain duplicate rollback should not
kill the whole beam route.  Instead, the rolled-back branch should carry a
blocked-identity record forward so the next branch evaluation can filter that
identity from candidate source lists and try lower-ranked alternatives.  Route-C
plateau logic and productive repeated identities should remain untouched.

## Implemented repair

Implemented in:

```text
pipelines/static_adapt/adapt_pipeline.py
test/test_static_adapt_full_reopt_duplicate_guard.py
```

The repair adds:

- final-full-refit support for `adapt_reopt_policy in {"windowed", "full"}`;
- a helper that detects whether the last history row already performed a
  full-prefix reoptimization, so redundant final refits can be recorded as
  requested but skipped;
- a conservative zero-gain duplicate identity based first on runtime
  Pauli-child labels/ids, then candidate labels;
- a shared post-refit guard payload for repeated identities whose realized
  energy gain is at or below the existing energy/rollback tolerance;
- ordinary-route post-refit rollback and blocked-identity filtering;
- beam-route post-refit rollback inside `_materialize_beam_child()`;
- beam-route branch-local blocked-identity filtering before child proposal
  construction;
- beam-route exhaustion when duplicate filtering leaves no admissible source
  records, preventing rollback-only child cycles from continuing to the nominal
  depth;
- telemetry in history rows under `zero_gain_duplicate_filter` and
  `zero_gain_duplicate_guard`.

The repair deliberately does not ban all repeated generators.  It only blocks an
exact repeated identity after that identity has already been committed once and
a later repeat realizes no meaningful descent.

## Validation after implementation

Targeted tests:

```text
pytest -q test/test_static_adapt_full_reopt_duplicate_guard.py -x
```

Result:

```text
8 passed
```

First weak--weak depth-10 smoke:

```text
raw_outputs/paper_i_hh_fullreopt_duplicate_guard_repair_smoke_20260710/weak_weak_depth10
```

Result summary:

```text
history_len=6
operators_len=5
unique_operator_count=5
stop_reason=structural_rollback
final_full_refit.requested=true
final_full_refit.executed=true
final_full_refit.nfev=118
```

This smoke showed that the repeated zero-gain child no longer accumulated in
the committed ansatz.  It also showed that treating duplicate rollback as an
ordinary structural-rollback terminal condition was too blunt for the beam
route, because the frontier stopped instead of trying lower-ranked alternatives.

Second weak--weak depth-10 smoke after the beam branch-local filter and
non-terminal duplicate rollback:

```text
raw_outputs/paper_i_hh_fullreopt_duplicate_guard_repair_smoke_20260710/weak_weak_depth10_v2
```

Result summary:

```text
history_len=10
operators_len=5
unique_operator_count=5
stop_reason=max_depth
final_full_refit.requested=true
final_full_refit.executed=true
final_full_refit.nfev=318
```

Triggered duplicate guards:

```text
depth=6   paop_full:paop_cloud_p(site=1->phonon=0)::child_set[4]
depth=9   paop_lf_full:paop_dbl_p(site=1->phonon=0)::child_set[4]
depth=10  paop_full:paop_cloud_p(site=1->phonon=0)
```

The final committed ansatz had five operators and five unique labels:

```text
paop_lf_full:paop_dbl_p(site=1->phonon=0)::child_set[4]
uccsd_ferm_lifted::uccsd_sing(alpha:0->1)::child_set[0]
paop_full:paop_disp(site=1)::child_set[0]
paop_full:paop_cloud_p(site=1->phonon=0)::child_set[0]
uccsd_ferm_lifted::uccsd_sing(beta:2->3)::child_set[0]
```

This meets the immediate repair criterion: the exact repeated zero-gain child
does not dominate the committed operator list, final-full-refit metadata is
coherent under `adapt_reopt_policy=full`, and the beam route can continue to the
requested depth after duplicate rollback.

Full weak--weak depth-30 validation before beam-exhaustion stop:

```text
raw_outputs/paper_i_hh_fullreopt_duplicate_guard_repair_validation_20260710/weak_weak_depth30
```

Result summary:

```text
history_len=30
operators_len=4
unique_operator_count=4
stop_reason=max_depth
final_full_refit.requested=true
final_full_refit.executed=true
final_full_refit.nfev=110
duplicate_guard_triggered_count=23
```

This confirmed that the committed ansatz no longer accumulates the repeated
child.  It also exposed a second beam-control issue: after duplicate filtering
emptied the admissible source list, the beam loop kept generating rollback-only
children and reached `max_depth`.

Full weak--weak depth-30 validation after beam-exhaustion stop:

```text
raw_outputs/paper_i_hh_fullreopt_duplicate_guard_repair_validation_20260710/weak_weak_depth30_v2
```

Result summary:

```text
history_len=6
operators_len=5
unique_operator_count=5
stop_reason=zero_gain_duplicate_guard_exhausted
final_full_refit.requested=true
final_full_refit.executed=true
final_full_refit.nfev=118
duplicate_guard_triggered_count=1
```

Final committed operators:

```text
uccsd_ferm_lifted::uccsd_sing(alpha:0->1)::child_set[0]
paop_full:paop_cloud_p(site=1->phonon=0)::child_set[4]
paop_full:paop_disp(site=1)::child_set[0]
paop_full:paop_cloud_p(site=1->phonon=0)::child_set[0]
uccsd_ferm_lifted::uccsd_sing(beta:2->3)::child_set[0]
```

The weak--weak validation now satisfies the acceptance criteria:

- no exact child identity dominates repeated zero-gain admissions;
- `adapt_vqe.operators` does not contain the previous repeated-child buildup;
- final-full-refit metadata remains coherent;
- any zero-gain duplicate rollbacks are represented as guard telemetry, not as
  committed ansatz growth.

## Next validation step

Before rerunning all six regimes, get final Oracle review of the repair and
then rerun the six-regime less-aggressive matrix from the same source settings.

Final Oracle review export:

```text
prompt-exports/oracle-review-2026-07-10-113121-beam-duplicate-guard-06dd.md
```

Oracle agreed the production repair is narrow enough to proceed, with two quick
test/invariant additions before the matrix rerun:

- assert the duplicate-filter exhaustion behavior;
- assert that the duplicate guard policy remains inactive for Route-C plateau
  handling.

Those tests were added to:

```text
test/test_static_adapt_full_reopt_duplicate_guard.py
```

Final targeted test result:

```text
pytest -q test/test_static_adapt_full_reopt_duplicate_guard.py -x
10 passed
```

Final cached weak--weak depth-30 validation after helper-level test additions:

```text
raw_outputs/paper_i_hh_fullreopt_duplicate_guard_repair_validation_20260710/weak_weak_depth30_v3
```

Result summary:

```text
history_len=6
operators_len=5
unique_operator_count=5
stop_reason=zero_gain_duplicate_guard_exhausted
final_full_refit.requested=true
final_full_refit.executed=true
final_full_refit.nfev=118
```

This is the validation artifact to use before launching the six-regime rerun.

## Paused six-regime diagnostic matrix

The six-regime local matrix was paused after user authorization because it no
longer represented a uniform comparison.  The stopped diagnostic root is:

```text
raw_outputs/paper_i_hh_fullreopt_duplicate_guard_repair_matrix_20260710/less_aggressive_1p75_fullwindow_fullreopt
```

Only the supervisor and worker associated with this output root were
interrupted.  A post-interrupt process check found no remaining associated
`adapt_pipeline` worker.

The first five completed rows used the earlier maturity-cap scheduler behavior
and all terminated through the duplicate guard exhaustion path:

```text
weak_weak: stop=zero_gain_duplicate_guard_exhausted, history=6, ops=5, unique=5, abs_delta_e=0.022543954537745048
intermediate_weak: stop=zero_gain_duplicate_guard_exhausted, history=5, ops=4, unique=4, abs_delta_e=0.10961489240935435
strong_weak: stop=zero_gain_duplicate_guard_exhausted, history=6, ops=4, unique=4, abs_delta_e=0.0014053442005785843
weak_strong: stop=zero_gain_duplicate_guard_exhausted, history=7, ops=5, unique=5, abs_delta_e=0.16002226059966174
intermediate_strong: stop=zero_gain_duplicate_guard_exhausted, history=8, ops=5, unique=5, abs_delta_e=0.14535346507174074
strong_strong: interrupted before result JSON; preserve run.log as partial diagnostic only
```

The weak--weak diagnostic is especially important because its error
`2.2544e-2` is orders of magnitude worse than the current Paper-I weak--weak
SNAKE value near `4.524e-4`; therefore these rows are failure diagnostics, not
candidate comparison rows.

At `2026-07-10 11:59:31 CDT`, the physical-lane route policy changed in:

```text
pipelines/static_adapt/lane_routes.py
pipelines/static_adapt/adapt_pipeline.py
test/test_static_adapt_lane_routes.py
```

The current source fixes physical-operator lane controller caps to the explicit
effective shortlist cap rather than allowing the older maturity-cap schedule to
choose smaller or different caps.  For the less-aggressive `1.75` route command
with base caps `N1=42`, `N2=21`, and aggressiveness factor `3`, current code
therefore resolves fixed caps:

```text
N1=ceil(42/3)=14
N2=ceil(21/3)=7
N3=7
```

The source telemetry policy identifier is:

```text
physical_route_fixed_to_effective_shortlist_caps_v1
```

The completed weak--weak row from the paused matrix still shows old scheduler
telemetry in its per-depth measurement-work counts, including Phase-I/II/III
shortlist sizes such as `11/7/4`.  This is the code-policy boundary that makes
the paused matrix non-uniform.

## Next narrow investigation

Do not launch another six-regime matrix until weak--weak is understood under
the fixed-cap route.  The next investigation is:

1. Confirm a fresh weak--weak execution reports fixed physical-lane caps
   `N1=14`, `N2=7`, `N3=7` and policy
   `physical_route_fixed_to_effective_shortlist_caps_v1`.
2. At the duplicate-exhaustion step, determine whether unblocked lower-ranked
   candidates existed in the upstream full record set but were removed before
   the duplicate filter.
3. Determine whether the guard declares exhaustion too early instead of
   filtering blocked identities first and rebuilding the shortlist from the
   remaining eligible records.
4. Audit the repeated candidate's Phase-II/III novelty, predicted gain,
   realized refit gain, insertion position, and position-dressed tangent.
5. Distinguish a wrong novelty calculation from a locally valid tangent-novelty
   prediction that fails only after nonlinear full refit.
6. Consult Oracle on this ranking/fallback question before any broader rerun.
7. Implement only a narrow, tested repair; do not change Hamiltonian settings,
   pool, optimizer, physical lanes, cost model, or scientific thresholds.
8. Run one short weak--weak execution-path smoke, then one complete depth-30
   weak--weak fixed-cap diagnostic, and stop for review before other regimes.

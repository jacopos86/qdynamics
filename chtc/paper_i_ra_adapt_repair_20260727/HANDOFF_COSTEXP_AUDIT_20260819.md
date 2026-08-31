# HANDOFF — Audit of the costexp min-cost realization route (2026-08-19)

Read-only audit. You verify claims and report findings; you do not commit
fixes. Any defect you find is a finding with a reproduction, not a patch.

**Route classification:** this is a current RA-ADAPT cost-expanded
realization experiment. It is not a Paper-I matrix and it does not revive or
select the retired SNAKE route. Names containing `paper_i` or `sr_snake` below
are legacy source/package namespaces only; they are not scientific ownership
or execution-policy claims.

**Final audit correction (2026-08-20):** no P1 release defect remains.  The
accepted-round, terminal, and resume validators independently authenticate a
realized coordinate against the sealed class partition and replay the signed
argmin receipt.  The two earlier in-memory selection checks are narrower:
they trust member aliases derived by the production domain builder from its
source position plan; they do not independently read the later serialized
partition receipt.  Do not describe all four layers as independently
receipt-bound.  The remaining defense-in-depth item is to thread a sealed map
into those two live checks.  The remaining coverage item is a plateau-open,
off-representative checkpoint continuation; public always/plateau continuation
is green, but the exercised plateau rounds remained closed.

## Anchors

- **Working checkout (the only valid one):**
  `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/ra-refactor-stage2`
  Decoys: `~/Documents/Holstein_implementation/Holstein_test_fullclone_3`
  (iCloud mirror) and the base checkout
  `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` (does NOT
  contain this work). Verify you are in the worktree before anything else.
- **Commit under audit:** `699f662e`. Run `git rev-parse --short HEAD`
  first; if it differs, diff against `699f662e` before trusting line
  numbers — other agents share this tree.
- **Test baseline (must stay green, you run it read-only):**
  `python3 -m pytest test/test_static_adapt_costexp_window.py test/test_static_adapt_explicit_depth_cost.py -q`
  → 20 passed. Wider battery:
  `python3 -m pytest test/test_static_adapt_costexp_window.py test/test_static_adapt_explicit_depth_cost.py test/test_static_adapt_resume_insertion_integrity.py test/test_static_adapt_builders.py test/test_static_adapt_guarded_singleton_pool_route.py test/test_static_adapt_phase_shortlists.py test/test_static_adapt_estimator_call_ledger.py -q`
  → 178 passed (last measured; report the count you observe).
- **Live reproduction harness** (the empirical gate this route must pass):
  a 5-round always-costexp run on `intermediate_strong` that produces
  ~12 off-representative realizations. Template:

  ```python
  import sys
  sys.path.insert(0, "<worktree>")
  from pipelines.static_adapt.ra_adapt import (
      RAAdaptOperationalControls, run_ra_adapt)
  from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
      build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request,
      build_paper_i_ra_hh_regime_problem,
      materialize_paper_i_ra_semantic_protocol)
  from pipelines.static_adapt.sr_snake.contracts import FreshStart
  problem = build_paper_i_ra_hh_regime_problem("intermediate_strong")
  request = build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
      insertion_policy="always_commutation_reduced_costexp",
      maximum_controller_rounds=75)
  protocol = materialize_paper_i_ra_semantic_protocol(problem, request)
  result = run_ra_adapt(problem, protocol,
      operational_controls=RAAdaptOperationalControls(
          maximum_controller_rounds=5, resume=FreshStart()))
  ```

  Takes ~12 minutes; run it with no wall-clock cap below 20 minutes.

## What this route claims

Two insertion kinds (`always_commutation_reduced_costexp`,
`plateau_commutation_costexp`; sealed route-contract shas `a40e82af…`,
`37ea4f71…`; policy
`termwise_cross_component_commutation_min_cost_realization_v1`): the
response (gradient/metric/sigma/ΔE) is computed once per
commutation-equivalence class at the representative position; the routed
FakeMarrakesh compile cost is evaluated at every member position; the
minimum-`penalty_total` member is admitted (earliest on exact ties); one
population slot per class; the chosen position is materialized into the
record and phases II/III inherit it (single realization decision).

Companion context: `costexp_audit_brief.md` (this directory's session
scratchpad copy may be stale; this handoff supersedes it where they
disagree).

## Scope

- In scope (read): `pipelines/static_adapt/adapt_pipeline.py` (costexp
  functions listed below), `pipelines/static_adapt/sr_snake/_selection.py`,
  `pipelines/static_adapt/ra_adapt/insertion_geometry.py`,
  `pipelines/static_adapt/ra_adapt/engine.py`,
  `pipelines/static_adapt/sr_snake/_resume.py`,
  `pipelines/reporting/paper_i_run_summary.py`,
  `test/test_static_adapt_costexp_window.py`.
- Out of scope: committing to this tree (another agent is actively
  committing); the CHTC cluster (you have no login; local runs only);
  anything under `pipelines/excited_dynamics/**` or
  `pipelines/qse_spectra/**` (Paper III lane). One heavy local run at a
  time; the machine has a hard 10 GB ceiling per agent process.
- Do not run anything while modifying source: the sealed protocols embed
  a source-inventory sha computed at call time, and a mid-run source edit
  fails runs with "materialization authority drifted".

## Increments

### 1 — Verify the four audit-layer relaxations are class-strict

Goal: confirm each relaxation admits exactly the realized-member case and
nothing else.

Sites (line numbers valid at `699f662e`):
- `_assert_phase_lineage` + `domain_by_id` merge and alias build:
  `sr_snake/_selection.py` (merge in `_select_singleton`, ~line 1359).
- `_assert_phase0_root_descent` alias resolution: same file, ~line 1209.
- `validate_commutation_reduced_insertion_receipt`
  (`min_cost_realization_active`): `ra_adapt/insertion_geometry.py`
  ~line 479 — scored pairs project onto class representatives via the
  receipt's own `representative_by_position` before the closure checks.
- `_validate_reduced_accepted_round_admission`: `ra_adapt/engine.py`
  ~line 5586 — admitted position must be a member in the receipt's
  `members_by_representative` partition.

Method: for each, construct the adversarial case in a scratch pytest file
(do not add it to the repo tests): (a) a realized position that belongs to
a DIFFERENT class of the same generator, (b) a position in no class,
(c) two realized members of the same class in one phase population.
Expected: all three rejected at every layer that sees them. Also confirm
the flags derive only from typed policy / sealed contract, never from
runtime data.

Stop condition: any adversarial case passes a layer → report with the
minimal reproduction; do not patch.

### 2 — Resume-path interaction (the known untested gap)

Goal: determine whether a checkpoint written after an off-representative
admission hydrates and revalidates cleanly.

Method: run the reproduction harness with a checkpoint observation
(`SRObservationPolicy(checkpoint=CheckpointObservation(path=..., every_controller_rounds=1, keep_history_tail=75))`),
cap 5 rounds, confirm ≥1 realization moved (grep the stdout for
`hardcoded_adapt_costexp_class_realization` and compare
`chosen_position` vs `representative_position`). Then resume from the
checkpoint with `AcceptedStateResume` and cap 6, and let it run one
more round.

Expected: hydration succeeds; round 6 executes; no audit layer rejects
the replayed receipts. Known hazard for interpretation: an equality-cap
resume (cap == checkpoint round) has a pre-existing off-by-one and is NOT
part of this audit — always resume with cap strictly greater.

Stop condition: hydration or revalidation fails → capture the full
traceback; this is the highest-value finding this audit can produce.

### 3 — Plateau-lane parity

Goal: confirm the plateau costexp kind exercises the same realization
code with the same guarantees, by inspection (no live plateau realization
has fired yet; plateau realization requires a plateau-opened domain).

Method: diff the always vs plateau branches at every site in Increment 1
plus the two engine receipt-audit switches (~line 6046 and ~line 6118).
Confirm the plateau receipt (`insertion_commutation_plateau`) carries the
same plan fields the relaxations rely on
(`members_by_representative`, `representative_by_position`) in both
domain-open and domain-closed rounds.

Expected: no plateau-only path reaches a representative-membership check
without the realization flag.

### 4 — Summary/reporting layer sweep

Goal: confirm `pipelines/reporting/paper_i_run_summary.py` and the
result/payload serializers accept realized positions (they should treat
positions as opaque integers; verify no representative-set validation
exists there).

## Traps (each of these bit the authoring session)

- Four separate insertion isinstance enumerations exist
  (`SRMethodPolicy.__post_init__`, `_require_semantic_closure_request`,
  `_validate_supported_request`, and the engine receipt audit). If you
  find a fifth that predates `699f662e` and lacks the costexp kinds,
  that is a finding.
- The selection workspace kernel is `_DefaultSingletonSelectionKernel`,
  NOT `_DefaultNoPruneSelectionTransaction` — the first fix in this
  session landed on the wrong facade and passed unit tests while failing
  live. Prefer live-run evidence over unit-only evidence.
- Smokes take ~12 min; a 10-minute execution cap kills them silently
  mid-round. No cap below 20 min.
- The 3-round strong_strong smoke is NOT a realization test (all early
  classes tie); intermediate_strong at 5 rounds is the reproducer.

## Report back

```markdown
## Increment N — <goal>
Status: done | blocked | skipped
Commands run: <exact>
Result: <pass/fail per adversarial case, tracebacks verbatim>
Verification: <test counts, realization counts observed>
Deviations: <anything done differently, and why>
```

Include the HEAD sha you audited. Report surprises even in passing
increments. Findings ranked by severity at the top of the report.

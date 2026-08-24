# Handoff — decompose `adapt_pipeline.py` and stop Paper-I settings drift (2026-08-24)

**For:** Codex (repo-access executor)
**From:** Claude, session on Paper-I route comprehension
**Contract:** `agent_guidance/shared/agent-handoff-contract.md` — read first; not restated here.
**Companion docs (READ BEFORE STARTING):**
`pipelines/static_adapt/adapt_pipeline_refactor_plan.md`,
`pipelines/static_adapt/adapt_pipeline_refactor_migration.md`,
`pipelines/static_adapt/beam_refactor_migration.md`,
`pipelines/static_adapt/adapt_pipeline_inventory_20260817.md`,
`pipelines/static_adapt/adapt_pipeline_teaching_refactor_map.md`,
`pipelines/static_adapt/adapt_pipeline_route_alignment.md`

---

## Read this first: the refactor already happened once

**Do not start a fresh decomposition.** Branch `codex/static-adapt-beam-refactor`
(local and on `origin`) carries a disciplined 182-commit extraction effort from
2026-07-04, documented in the three migration logs above.

| | |
|---|---|
| `adapt_pipeline.py` on that branch | **23,701 lines** |
| `adapt_pipeline.py` on HEAD | **72,528 lines** |
| Divergence | HEAD-only: **173** commits · branch-only: **182** commits |

Sixteen of the eighteen modules that branch extracted **do exist on HEAD**
(`beam_search.py`, `optimizer_routes.py`, `checkpoint_telemetry.py`,
`phase_shortlists.py`, `selector_measurement_proxy.py`, `batch_ordering.py`,
`oracle_lifecycle.py`, `route_c_plateau.py`, and others). So the extraction was
partly merged and then HEAD grew past it.

**Two did not merge**, and one of them is actively breaking the test suite:

- `route_identity.py` — 723 lines on the refactor branch, **absent on HEAD**
- `route_support.py` — absent on both (renamed later on the branch)

**34 of HEAD's 55 test collection errors are `ModuleNotFoundError: No module
named 'pipelines.static_adapt.route_identity'`.** The tests were merged; the
module was not.

Given 173/182 divergence on a file that has since tripled, **merging that branch
is not the plan.** Treat it as a proven reference design: it demonstrates the
target shape and the migration logs are a ready-made map of which helpers are
extractable and what each extraction must preserve. Increment 1 below recovers
the one file that is cheap and high-value to bring across.

---

## Why this work exists

The author's symptom, verbatim: *"every time I try to do a run with a minimal
change, many changes occur at once and I keep experiencing drift in settings,
and we have so many things that are not algorithmic options I want."*

That has a specific mechanical cause, measured below. **It is not caused by file
size.** Increments 1–3 fix the symptom and touch almost no algorithmic code.
Increments 4–6 are decomposition. If time runs short, 1–3 are the ones that matter.

---

## Anchors

| | |
|---|---|
| Checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` (NOT the `~/Documents/Holstein_implementation` iCloud mirror) |
| Branch | `paper-ii-exchange-selector` |
| Commit | `7edf45e3` — verify with `git rev-parse --short HEAD` before starting |
| Test baseline | `python3 -m pytest test --collect-only -q` → **5624 collected, 55 collection errors** (34 of them `route_identity`). Record your own baseline before editing. |
| Reference branch | `codex/static-adapt-beam-refactor` — read-only; do not merge |

**Work in your own worktree.** Thirteen worktrees exist; three Codex branches
were committed to today. Create `Holstein_test_fullclone_3.worktrees/<your-branch>`
and stage with explicit paths only — never `git add -A` (incident `ba7f2ac9`).

---

## The measured diagnosis

Every number measured on `7edf45e3`. Reproduce before trusting.

### Option surface

| measurement | value |
|---|---|
| distinct `--flags` in `static_adapt/` | **467** |
| `add_argument` in `cli_config.py` | 409 |
| argparse `default=` | 397 |
| dataclass `field(default…)` | 88 |
| `os.environ` reads | 16 |

Three independent default sources; no materialized view of what a run resolved to.

### Profile inheritance

`sr_snake_route_profile.py` holds 31 module-level `*_SETTINGS` dicts composed by
`**` spread. Measured by AST — **regex under-reports**, because one link is a
*subtractive dict comprehension* that re-spreads V3 minus `adapt_max_depth`.

- **1 real root:** `CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS`, **116 keys**
- **max chain depth: 7**; deepest leaf inherits ~147 keys and adds 13
- 27 of 31 descend from that single root — **changing V1 changes every profile**

### Why one axis cannot be varied

`normalize_sr_route_profile_namespace()` (`sr_snake_route_profile.py:~4192`)
force-writes contract fields onto the argparse namespace and is **fail-closed**:
an explicit flag disagreeing with the profile raises `ValueError: SR-SNAKE route
profile conflicts with explicit or untracked scientific settings`.

That is correct design — it prevents silent drift. But it means "the canonical
route with one setting changed" requires **authoring a new frozen profile**.
Hence names like `canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1`.

**This is the entire symptom.** "Many changes at once" = you switched to a
different leaf of a 7-deep chain. "Drift" = that leaf's effective settings are
never materialized anywhere you can diff.

### The mega function

`_run_hardcoded_adapt_vqe`, `adapt_pipeline.py:14876–56085`:

| measurement | value |
|---|---|
| lines | **41,210** (57% of file) |
| **signature parameters** | **348** |
| `if` statements | 1,661 |
| `for` loops | 222 |
| max nesting depth | 15 |
| distinct locals assigned | 3,104 |
| nested `def`/`class` | **200** (27,543 lines) |

Callers: `sr_snake/_context.py:885`, `exact_bench/hh_static_ground_state_benchmark.py:975`;
kwargs built by `cli_config.py:3736`.

---

## File audit — the author's "likely nonsense" list, verified

Reachability computed by AST import closure from the Paper-I entry points
(`adapt_pipeline`, `sr_snake._context`, `exact_bench.hh_static_ground_state_benchmark`,
`scaffold.runtime_loader`). **101 of 116 `static_adapt` modules are reachable.**

### Confirmed unreachable — deletion candidates (10,433 lines)

| lines | status | module |
|---|---|---|
| 2,965 | test-only | `ra_adapt/campaign.py` |
| 1,914 | test-only | `sr_snake_modeled_minimum_runtime.py` |
| 1,422 | test-only | `hh_prune_nighthawk.py` |
| 921 | test-only | `sr_snake_active_manifold_distance.py` |
| 734 | test-only | `sr_snake_uniform_barrier.py` |
| 498 | **no refs at all** | `hh_prune_marginal_analysis.py` |
| 483 | test-only | `hh_pareto_tracking.py` |
| 301 | test-only | `runtime_heartbeat.py` |
| 301 | test-only | `ra_adapt/h2o_application.py` |
| 280 | **no refs at all** | `hh_adapt_backend_shortlist.py` |
| 58 | **no refs at all** | `hh_adapt_backend_single.py` |

The author's instinct on `sr_snake_uniform_barrier.py` was right: only string
literals `"uniform_barrier"` in `sr_snake_modeled_minimum.py` mention it — the
module itself is imported by nothing on the run path.

`ra_adapt/h2o_application.py` is Paper-IV molecular-vibronic material in the
Paper-I lane. Per root `AGENTS.md`, "Paper IV must not inherit Paper-I
Hamiltonian defaults, source locks, evidence, or manuscript contracts." Two
profile aliases in `sr_snake_route_profile.py` also carry it
(`sr_snake_h2o_derivative_resolved_v2`, `…_paper_i_v3`). Relocation is a
**scientific-ownership decision for the author**, not a mechanical move — flag it,
do not act.

### NOT nonsense — all reachable, leave alone

The `sr_*` hypothesis is only half right. The four largest are load-bearing:
`sr_snake_route_profile.py` (4,588 — the config system itself),
`sr_snake_modeled_minimum.py` (2,934), `sr_snake_escape_controller.py` (929),
`sr_snake_phase12_policy.py` (117).

Every `selector_*` and `route_*` file is imported by live product code:
`route_a_child_padding` and `route_a_shortlists` by 8 files each,
`route_a_schur_selector` by 5, `route_a_trust_region` by 4,
`selector_query_closure` and `selector_exact_query_geometry` by 2 each,
`route_c_plateau` by 2. **None are deletion candidates.**

### Empty directories

`pipelines/static_adapt/optimization/`, `pipelines/static_adapt/prototypes/`,
`pipelines/static_adapt/diagnostics/`, `pipelines/exact_bench/configs/` — contain
only `__init__.py`/`__pycache__`. Safe to remove once no import references them.

---

## Scope

**In scope:** `pipelines/static_adapt/{adapt_pipeline,cli_config,sr_snake_route_profile}.py`,
new modules under `pipelines/static_adapt/`, and `test/` files covering them.

**Out of scope, and why:**

- `pipelines/time_dynamics/**`, `excited_dynamics/**`, `qse_spectra/**` — Paper II/III lanes, other agents commit daily.
- `chtc/**`, `raw_outputs/**`, `history/**`, `archive/**` — preserved evidence and source locks.
- The 21 non-`route_identity` collection errors (`docs.reports.report_labels`, `chtc.phase3_optuna` symbols) — unrelated rot.
- **Scientific behavior.** Every increment is behavior-preserving. A changed run number means a mistake — stop.
- Relocating H2O/Paper-IV material — author's decision.

**Shared-resource limits:** 10 GB resident RAM across all agent work on this 16 GB
machine (`agent_guidance/shared/memory-budget.md`). One heavy local job at a time.
No CHTC submission under this handoff.

**Autonomy:** proceed through increments 1–4 without pausing. **Pause and report
after 4.** Increment 5 is where behavior risk begins.

---

## Increments

### 1. Restore `route_identity.py` — fixes 34 of 55 collection errors

```bash
git show codex/static-adapt-beam-refactor:pipelines/static_adapt/route_identity.py \
  > pipelines/static_adapt/route_identity.py
python3 -m pytest test --collect-only -q 2>&1 | tail -3
```

Expected: collection errors drop **55 → 21**; collected tests rise above 5624.

Stop if: the restored module imports symbols absent from HEAD. It was written
against a 23.7k-line `adapt_pipeline.py`. Report the missing symbols rather than
stubbing them — a stub here silently changes route identity metadata.

### 2. Materialize effective settings (fixes the stated symptom)

New module `pipelines/static_adapt/route_profile_effective.py` exposing:

```python
effective_settings(profile_name) -> dict[str, Any]        # flattened, all 7 levels
effective_settings_diff(a, b)    -> dict[str, tuple]      # field -> (a_value, b_value)
```

Wire its output into the run receipt beside the existing
`sr_route_profile_contract_sha256`.

Expected:
- `effective_settings("sr_snake")` returns **≥127 keys**, no nested indirection.
- `effective_settings_diff("sr_snake_v3", "sr_snake_v3_1")` returns exactly **1**
  differing field. More than one means this document's chain analysis is wrong —
  stop and report.
- A test asserts, for each of the 31 profiles, that the flattened dict equals what
  `normalize_sr_route_profile_namespace()` writes onto the namespace. **This test
  is the regression lock for every later increment.**

Stop if: any profile disagrees. That is a live bug, not a refactor target —
report the field names, do not fix.

### 3. Replace the reflective signature filter with an explicit allowlist

**Prerequisite for all decomposition.** `adapt_pipeline.py:72076`:

```python
_CANONICAL_SR_SNAKE_LEGACY_EXECUTOR_PARAMETER_NAMES = frozenset(
    inspect.signature(_run_hardcoded_adapt_vqe).parameters
)
```

`_canonical_sr_snake_legacy_executor_kwargs()` filters runtime kwargs against it.

```bash
python3 -c "
import ast,pathlib
t=ast.parse(pathlib.Path('pipelines/static_adapt/adapt_pipeline.py').read_text())
fn=next(n for n in t.body if getattr(n,'name','')=='_run_hardcoded_adapt_vqe')
names=[a.arg for a in fn.args.posonlyargs+fn.args.args+fn.args.kwonlyargs]
print(len(names)); print(repr(sorted(names)))
"
```

Expected **348** names. Freeze as a checked-in literal; add a test asserting it
still equals the live signature. Now a signature change fails a test instead of
silently dropping kwargs.

Stop if: count is not 348 — the file moved under you; re-verify the SHA.

### 4. Delete confirmed-dead modules

Work the table above in ascending line order. For each, **source must ratify the
index** (see Evidence standards) before deletion. Delete the three "no refs at
all" files first; for test-only modules, delete the module and its test together
only after confirming with the author that the capability is retired. Remove the
four empty directories last.

Expected: `static_adapt/` drops up to 10,433 lines; collection errors do not rise.

### 5. Beam-cluster state objects — PAUSE BEFORE STARTING

Of the 200 nested defs, **94 capture ≤2 outer variables with no `nonlocal`**
(2,890 lines) and are cleanly liftable. The remaining 106 hold 24,653 lines of
implicitly-shared state:

| captured vars | lines | name |
|---|---|---|
| 126 (`nonlocal`) | 5,493 | `_evaluate_beam_branch` (L29962) |
| 101 (`nonlocal`) | 2,737 | `_materialize_beam_child` (L36162) |
| 99 (`nonlocal`) | 626 | `_write_current_checkpoint` (L28876) |
| 86 | 1,730 | `_process_phase2_full_candidate_record_local` (L30980) |

`beam_refactor_migration.md` already lists `_evaluate_beam_branch` and
`_materialize_beam_child` as "intentionally not moved yet" — the prior effort
stopped at exactly this boundary. Propose the state object's fields in your
report-back **before** writing it.

### 6. Base + delta profiles

Only after 1–5. Rewrite the 31 dicts as one base plus explicit named deltas so an
ablation is `base + {one field}` rather than a new leaf on a 7-deep chain.
Increment 2's flattening test is what makes this safe.

---

## Evidence standards

- **An index proposes; source ratifies.** Every "unreachable" verdict here came
  from AST import-closure analysis, which does **not** follow callables passed as
  arguments. The contract records a prior incident where exactly this method
  marked three live cost symbols dead because they were invoked through closures.
  Before deleting anything: grep for the name, read the call sites, run the tests.
- **Behavior parity needs a lock.** Increment 2's per-profile flattening test is
  the lock for 3–6. It must pass unchanged. A deliberate change that breaks it is
  a scientific decision for the author, not the executor.
- **"Never used" needs a measurement.** To delete any of the 467 flags, produce a
  count of how many of the 31 profiles and how many `chtc/**` run manifests set
  it. Zero profiles *and* zero historical runs is a deletion candidate; anything
  else is not.

---

## Traps

- **The 348-parameter signature is a runtime filter, not just a signature.**
  Removing or renaming a parameter causes the corresponding kwarg to be **silently
  dropped** — no exception, just a different run. Do increment 3 before touching
  the signature. Largest hazard in this work.
- **Profile inheritance includes a subtractive link.**
  `CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS` spreads V3
  through a comprehension that *removes* `adapt_max_depth` deliberately, so each
  regime source lock supplies the horizon. A flattener handling only `**NAME`
  produces a wrong effective dict.
- **The fail-closed conflict check is a feature.** Do not "fix" the `ValueError`
  by making profiles overridable — it is currently the only thing preventing
  silent drift. Increment 6 replaces it; until then it stays.
- **The migration logs are authoritative about what was already tried.** Read them
  before extracting anything; several helpers are documented as deliberately left
  in place, with reasons.
- **Test baseline is already red** (55 collection errors). Record which fail before
  you start so you can prove you did not add to it.
- **`output/` and `prompt-exports/` are gitignored.** This handoff sits in
  `agent_guidance/static-adapt/` so it survives in version control.
- **Another agent's broad `git add` can sweep your uncommitted work.** Commit at
  increment boundaries with explicit paths.

---

## Report back

```markdown
## Increment N — <goal>
Status: done | blocked | skipped
Commands run:
Result:
Verification:
Deviations:
```

Include commit SHAs. Surface surprises even on success. Report immediately if
increment 2 shows any profile whose flattened settings disagree with what the
namespace normalizer writes, or if increment 1's restored `route_identity.py`
needs symbols that no longer exist.

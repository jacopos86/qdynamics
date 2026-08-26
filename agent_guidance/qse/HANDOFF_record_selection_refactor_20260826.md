# Handoff — refactor the Paper III selection method to match its specification

Author: Claude (planning). Executor: Codex (repo access).
Contract: read `agent_guidance/shared/agent-handoff-contract.md` first; this
document supplies only what is task-specific.
Behaviour specification: `MATH/paper_details/excited_spectra_dynamics_paper_III.tex`,
subsections *Selection score*, *Acquisition rule*, *Hardware-cost weighting*,
*Stopping rule*, *Certified exchange maintenance*.
Comparison protocol (binding on any evidence produced):
`agent_guidance/qse/paper-iii-comparison-protocol.md`.

## Anchors

| field | value |
|---|---|
| Checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` — NOT the `~/Documents/Holstein_implementation` iCloud mirror |
| Branch | **Work directly on `paper-ii-exchange-selector`. Do NOT create a new branch or worktree** (this overrides the contract's default). Run `git pull --rebase` before every commit, and commit only at increment boundaries. Several agents commit here daily. |
| Test baseline | `python3 -m pytest test/test_qse_record_selection.py test/test_qse_compiled_cost_selection.py test/test_qse_spectra_core.py test/test_qse_spectra_imports.py test/test_qse_spectra_io_cli.py test/test_qse_compiled_costs.py test/test_qse_exchange_maintenance.py test/test_paper_iii_problem_provider.py test/test_paper_iii_matched_accuracy_campaign.py test/test_adaptive_qse_benchmark.py -q` → **76 passed**. Repo-wide collection has ~55 pre-existing `docs.*` import errors unrelated to this lane; ignore them. |
| Behaviour lock | `python3 pipelines/exact_bench/paper_iii_growth_trace_campaign.py --regime-set nph1 --k-max 60 --stride 3 --krylov-k-max 40 --exchange-policy every_k` must produce **bit-identical** selected supports before and after every increment. This is the acceptance test for the whole task. |

## Why

`pipelines/qse_spectra/record_selection.py` is 996 lines and 24 defs supporting
**one** production path. It accumulated four selection modes, a five-term score
that is now two terms, and several hard-filter knobs, while the manuscript was
separately rewritten to describe what the code actually does. The code is
correct; it is overgrown. The goal is a module a reader can check against the
manuscript in one sitting.

**Settled — do not reopen.** The production score is exactly

    S(r,l) = [ w_N * N_qse(r,l) + w_R * C_res(r,l) ] / max(C_hat(r,l), C_floor)^alpha

with `w_N = 0.25`, `w_R = 1.0`, `alpha = 1.0`, `C_floor = 0.05`, plus a hard
metric-novelty floor at `1e-12` that rejects candidates outright. Ritz gain,
probe-transition visibility, and the explicit conditioning penalty were each
implemented, ablated, and removed as not improving selection; their defaults
are already zero or deleted. Do not reintroduce them, and do not re-derive the
weights.

## File ownership (concurrent agents, binding)

The author and this executor are on one branch, so ownership is by path, not by
branch. Two agents independently wrote the same exchange-activation paragraph
from the same evidence on 2026-08-26; one copy had to be deleted. Do not repeat
that.

| path | owner |
|---|---|
| `pipelines/qse_spectra/record_selection.py`, `pipelines/qse_spectra/__main__.py`, and their tests | **you** |
| `MATH/paper_details/**` (manuscript, figures, generated fragments) | **the author** |
| `pipelines/reporting/**` | the author |
| `pipelines/exact_bench/**` | the author |
| `pipelines/qse_spectra/exchange_maintenance.py`, `paper_iii_problem.py` | nobody this task; out of scope |

If an increment forces a change outside your column, **stop and report the
required change** rather than making it. Do not edit the manuscript: if the
refactor changes behaviour the manuscript describes, say so in the report-back
and the author will edit the text.

## Scope

In scope: `pipelines/qse_spectra/record_selection.py`, its CLI surface in
`pipelines/qse_spectra/__main__.py`, and the tests covering both.

Out of scope, with reasons:
- `pipelines/qse_spectra/exchange_maintenance.py` — clean at 481 lines and it
  carries the paper's strongest novelty claim; leave it alone.
- `pipelines/qse_spectra/paper_iii_problem.py` — the shared physics provider;
  its guarantees (one alphabet per regime, ordered pool digest, cached exact
  reference, granularity guard) are load-bearing for every comparison.
- `pipelines/exact_bench/**`, `pipelines/reporting/**` — evidence and reporting;
  change them only if an increment's public-API edit forces a call-site update,
  and say so in the report-back.
- `MATH/paper_details/**` — the manuscript is the author's.

Shared-resource limits: one heavy local job at a time; **hard 10 GB agent RAM
ceiling** on this shared machine. The `nph1` behaviour lock is cheap (minutes).
Do not run `--regime-set paper_i` (nph7).

## Increments

Proceed through 1–3 autonomously; **pause and report before increment 4.**

### Increment 1 — retire dead selection modes

Goal: `geometry_selected` and `cost_proxy` are the only modes.

`_STATIC_RECORD_SELECTION_MODES` currently holds four. `input_order` and
`compiled_cost` exist only as baselines that the evidence drivers now construct
themselves (they order the pool directly). `cost_proxy` **stays** — the author
uses it in published results.

Before deleting either mode, grep for callers across `pipelines/`, `test/`, and
`chtc/`, and report what you find. If any non-test caller exists, stop and
report rather than rewriting that caller.

Expected result: baseline still 76 passed (tests referencing removed modes are
updated or removed, and you state which); behaviour lock bit-identical.

### Increment 2 — remove unreachable configuration

Goal: every field on `StaticRecordSelectionConfig` is reachable from a
production path.

Evidence gathered by the author, verify it yourself before deleting:

| field | test files | non-selector uses | disposition |
|---|---|---|---|
| `geometry_cost_weight` | 0 | 0 | only reachable on the `alpha is None` branch, which nothing takes — **remove with the branch** |
| `max_term_count` | 1 | 0 | Paper III never sets it; propose removal, but confirm no other lane relies on it |
| `max_pauli_weight` | 3 | 10 | **KEEP** — live public surface used outside this lane |
| `min_retained_rank` | 3 | 4 | **KEEP** |
| `max_overlap_condition` | 2 | 6 | **KEEP** |

An index proposes and source ratifies (contract §4): confirm each count by grep
and source reading, and report any disagreement with the table above rather
than trusting it.

Expected result: baseline green; behaviour lock bit-identical.

### Increment 3 — separate the concerns inside the module

Goal: the file reads as the manuscript does, in this order — candidate
admissibility, retained-frame rebuild, per-candidate geometry, score assembly,
stopping rule, payload assembly — with each a named function of stated inputs
and outputs.

The single hot loop currently interleaves the retained-frame projection, the
two geometric quantities, the cost discount, and the stopping bookkeeping.
Three subtleties that must survive verbatim; each exists because of a specific
failure:

1. **The Gram–Schmidt projection runs twice.** A single pass lets numerical
   noise from near-dependent accepted directions read as zero novelty and
   permanently exclude records that still carry rank. Do not "optimize" it to
   one pass.
2. **Novelty is measured against the numerically retained frame**, rebuilt each
   round under the solver's own overlap cutoff — never against the exact span
   of accepted images. Acquisition and diagonalization must share one
   stabilization scale.
3. **`h11` is consumed by the stopping rule.** It is computed inside the
   per-candidate geometry block but feeds the window-closure test
   (`round_thetas`). A previous cleanup nearly deleted it with the Ritz term
   and would have silently broken the stop.

Expected result: no behaviour change — the lock is bit-identical — and the
module is materially shorter. Report the before/after line count and def count
(currently 996 lines, 24 defs).

Stop condition: if any increment changes a selected support, **stop and
report with the diff**; do not adjust thresholds to restore it.

### Increment 4 — PAUSE, then confirm evidence

Report increments 1–3 with the lock output. After the author confirms, re-run
`--regime-set nph1` and `--regime-set nph3` and confirm the committed evidence
JSONs are unchanged apart from timestamps.

## Traps

- `output/` and `prompt-exports/` are **gitignored**; artifacts there will not
  appear in a commit. Reference them by path in the report-back.
- The exact reference is content-addressed under
  `output/reference_store/paper_iii_exact_sector/` and verified on read.
  Recomputing it per run is a protocol violation.
- A safety cap must not silently redefine a threshold method: if the selector
  hits `max_records` before its stopping condition, that must remain visible in
  `geometry_stop.stop_reason`, not be reported as convergence.
- Another agent's broad `git add` can sweep your uncommitted work into their
  commit. Commit at increment boundaries.
- Call these things **methods**, not "arms", in code, comments, and report-back.

## Report back

Contract §6 block per increment, with commit SHAs, plus: before/after line and
def counts, and the behaviour-lock verdict (bit-identical supports, yes/no) for
every increment.

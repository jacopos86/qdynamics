# Handoff — Paper III static-QSE lane: orientation and open queue (2026-08-27)

**For:** Codex (repo-access executor)
**From:** Claude — session that built the static-QSE results-acquisition pipeline
**Contract:** `agent_guidance/shared/agent-handoff-contract.md` (read it; not restated here)

**Companion docs** — this file is the index that makes them executable in the
current world. Do not start from them cold; two predate changes below.

| doc | role | status |
|---|---|---|
| `paper-iii-static-results-acquisition-contract.md` | binding operating contract for the pipeline | current |
| `IMPLEMENTATION_REPORT_static_qse_results_pipeline_20260827.md` | what was built, digests, what was left unrun | current |
| `EVIDENCE_dark_state_contamination_of_the_energy_metric_20260827.md` | metric finding + declared window readings (read before touching `C*(eps_E)`) | current; **read its final CORRECTION section, which reverses the middle one** |
| `HANDOFF_record_selection_refactor_20260826.md` | Increment queue for the selector refactor | **still open, now has a new constraint — see §4** |
| `BUGREPORT_hybrid_seed_ceiling_20260826.md` | seeded-Davidson error ceiling | **still open, untouched** |
| `BUGREPORT_adaptive_qse_root_indexing_20260826.md` | items 1–3 resolved; item 2 (`hubbard_u8` dim-7 stall) open | **partially open** |
| `paper-iii-comparison-protocol.md` | growth-trace comparison protocol | current |
| `paper-iii-novelty-claims-boundary-20260818.md` | claims boundary (A-CASE/DA-CASE exclusion is FINAL) | current — comparator set and evidence inventory amended 2026-08-27 |

---

## 0. Orientation — optional for execution, read if you lack session context

There are **two layers** in this lane, and conversation about them jumps
between the two. Keeping them apart makes the rest of this document legible.

**Layer 1 — the results pipeline (bookkeeping).** How a number gets produced
and recorded reproducibly:

```text
declare a cell  ->  run its stages  ->  verify outputs  ->  admit to ledger
                ->  project onto the declared matrix  ->  render the report
```

No physics lives here. It exists so a result can be traced to the exact
Hamiltonian, pool, prepared state, method, cap, and policy that produced it,
and so a partially-finished campaign reports honest `pending` instead of
silently omitting rows. Built and smoke-tested this session.

**Layer 2 — the science the pipeline carries.** The QSE comparison: our
record-first method against three comparators, all grown to a fixed cap,
matched on an accuracy target, compared on cost.

**Why a downstream question reached back into Layer 2's core metric.** The
open scope question was "do the comparator methods need their own Green
functions / spectral products?" Answering it required measuring whether
*energy* accuracy and *spectral-weight* accuracy arrive together. They do
not — and the same measurement showed the comparison metric itself
(`max_r |E_r - E_r^exact|` over six target roots) is partly scoring states
that no spectrum can display. Transition strengths are what distinguish a
**bright** root (an observable connects it to the ground state; it makes a
peak) from a **dark** one (nothing connects; invisible to any measurement).
Two to three of the six target roots are dark, and in two of six regimes a
dark root is the last to converge and therefore sets the reported cost.

`pipelines/qse_spectra/target_window.py` now makes this an explicit
declaration: the same trace can be read over `all_roots` or over the
`weight_carrying` subset, with visibility derived from the declared
observable set only and `assert_shared_window` refusing to score methods
against different windows.

Measured with Paper III's **declared** observables (`site0_density`,
`site0_displacement`), the weight-carrying reading *reduces* our margin
over direct Davidson from 4.6--6.8x to 1.5--4.0x, because Davidson spends
most of its budget on invisible roots and gets much cheaper when they are
not charged. The recommendation is therefore to keep `all_roots` as the
headline and report `weight_carrying` as a disclosed sensitivity. **No
increment below acts on this**; it is the author's call, listed in §6.

---

## 1. Anchors

| | |
|---|---|
| Checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` — **not** the `~/Documents/Holstein_implementation` iCloud mirror |
| Branch | `paper-ii-exchange-selector` — **work directly on it; do NOT create a branch or worktree** (overrides the contract default). Several agents commit here daily. |
| Commit | `d1030fa2` — verify with `git rev-parse --short HEAD` before starting |
| Test baseline | command in §3 → **174 passed**. Repo-wide collection has ~55 pre-existing `docs.*` import errors unrelated to this lane; ignore them. |
| Behaviour lock | `nph1` selected supports must stay bit-identical — verification command in §3 |
| Upstream | The branch has **no remote-tracking upstream**, so `git pull --rebase` has nothing to rebase onto and aborts on the shared tree's unrelated dirty files. Commit at increment boundaries with explicit paths; **never `git add -A`**, never `git stash`. |

---

## 2. Scope

**In scope (yours to change):** `pipelines/qse_spectra/record_selection.py`,
`pipelines/qse_spectra/adaptive_qse_benchmark.py`,
`pipelines/qse_spectra/__main__.py`, and their tests.

**Out of scope:**

- `pipelines/qse_spectra/exchange_maintenance.py` — the atomic `1<->1`
  exchange is a load-bearing novelty claim (C3) and is verified firing; a
  refactor here invalidates evidence rather than improving it.
- `pipelines/qse_spectra/paper_iii_problem.py` — the shared provider is what
  makes the matched-alphabet claim checkable.
- `pipelines/qse_spectra/static_campaign*.py`,
  `pipelines/qse_spectra/switch_policy.py`,
  `pipelines/qse_spectra/target_window.py`,
  `pipelines/qse_spectra/siam_static_adapter.py`,
  `pipelines/exact_bench/paper_iii_static_qse_cell.py`,
  `pipelines/reporting/paper_iii_static_results_contract.py` — authored this
  session by the other agent; coordinate before editing.
- `MATH/**` — no manuscript edits in this task, under any increment.

**Shared-resource limits:** hard 10 GB agent RAM ceiling; one heavy local job
at a time; no remote/CHTC submission and no `git push` without the author's
explicit go-ahead.

**Autonomy:** proceed through Increments 1–3 without pausing. **Pause before
Increment 4** and report.

---

## 3. Baseline commands

```bash
cd /Users/jakestrobel/local_repos/Holstein_test_fullclone_3
python -m pytest test/test_qse_record_selection.py test/test_qse_compiled_cost_selection.py \
  test/test_qse_spectra_core.py test/test_qse_exchange_maintenance.py \
  test/test_paper_iii_problem_provider.py test/test_adaptive_qse_benchmark.py \
  test/test_paper_iii_switch_policy.py test/test_paper_iii_target_window.py \
  test/test_paper_iii_static_campaign.py \
  test/test_paper_iii_static_campaign_artifacts.py test/test_paper_iii_static_campaign_package.py \
  test/test_paper_iii_static_qse_cell.py test/test_paper_iii_static_results_contract.py \
  test/test_paper_iii_siam_static_adapter.py test/test_paper_iii_growth_trace_hybrid.py \
  test/test_paper_iii_qse_source_maps.py test/test_paper_iii_qse_results_pdf.py -q
```

Expected: `174 passed`.

The `nph1` support lock — the acceptance test for any selector change:

```bash
cd /Users/jakestrobel/local_repos/Holstein_test_fullclone_3
python pipelines/exact_bench/paper_iii_growth_trace_campaign.py \
  --regime-set nph1 --k-max 60 --stride 3 --krylov-k-max 40 --exchange-policy every_k \
  --output-json /tmp/nph1_lock_check.json
```

Selected supports must be bit-identical before and after. The author verified
93/93 acquisition-prefix supports and 48/48 stored baseline rows identical at
`4152de3a`; the stored baseline is
`output/diagnostics/paper_iii_record_first_adaptive_qse_v2/growth_trace_nph1.json`.
**Any support change is a scientific event: stop and report the exact
difference. Do not retune to recover it.**

---

## 4. What changed since your last handoff — new constraint on the refactor

`HANDOFF_record_selection_refactor_20260826.md` was written before the switch
policy existed. One new coupling now binds it:

**`record_selection`'s `geometry_stop["round_trace"]` is a public interface.**
`pipelines/qse_spectra/switch_policy.py` decides the record-to-Davidson
handoff from that trace, and `assert_online_only_receipt` **fails closed** on
any key outside a hard-coded vocabulary — deliberately, so exact references
can never reach an online decision. The frozen key sets are
`_ONLINE_STOP_KEYS` and `_ONLINE_ROUND_KEYS` in that module.

Consequences for your refactor:

- Renaming, dropping, or adding a field in `round_row` or `stop_info` breaks
  **every** record-first cell, including the canonical exhaustion-only path —
  not just the ablation.
- If a rename is genuinely warranted, update `_ONLINE_ROUND_KEYS` /
  `_ONLINE_STOP_KEYS` in the same commit and say so in the report.
- Never widen those sets to admit a field carrying exact-reference
  information (energy error against the exact spectrum, reference
  excitations, window error). That guard is the structural reason the method
  can claim its switch is reference-free.
- `test/test_paper_iii_static_qse_cell.py` catches this: it runs the real
  selector receipt through `resolve_switch_point`. It is in the §3 baseline.

A second, weaker coupling exists but is **not yours**: `pipelines/qse_spectra/target_window.py` reads `per_root_abs_error` from growth-trace rows. Those come from `compute_qse_spectra` eigenvalues in `paper_iii_growth_trace_campaign.py`, not from the selector, so the refactor cannot disturb it. Noted so you do not go looking.

Fields currently relied on: `stop_reason`, `final_max_target_residual_norm`,
`residual_converged_pending`, `residual_stop_threshold`, `round_trace`; and
per round `round`, `selected_count_before`, `retained_frame_size`,
`max_target_residual_norm`, `window_top`, `window_margin`, `window_pressure`,
`window_closed`, `novel_candidate_count`, `max_candidate_metric_novelty`,
`max_candidate_residual_capture`, `selected_original_index`,
`selected_score`, `selected_metric_novelty`, `selected_residual_capture`.

---

## 5. Increments

### 1. Selector refactor (standing work, now constrained by §4)

Execute `HANDOFF_record_selection_refactor_20260826.md` increments 1–4 as
written, with §4 above added to its Traps section.

Expected: its own stated per-increment results, **plus** §3 at `161 passed`
and the `nph1` support lock bit-identical.

Stop if: any selected support changes, or the refactor cannot preserve the
`round_trace` vocabulary without a rename → stop and report the exact diff.

### 2. Seeded-Davidson error ceiling

Execute `BUGREPORT_hybrid_seed_ceiling_20260826.md`: seeded growth pins to a
seed-independent `3.71e-01` ceiling in `intermediate_strong` at `n_ph_max=3`
for seeds `k = 8..32`, while a fixed-class seed reaches `4.77e-15`.

Expected: the four requested diagnostics answered, and a determination of
**defect vs. genuine Ritz-blindness**. The report says explicitly: do not tune
thresholds to make the symptom disappear, and Ritz-blindness is a finding, not
a bug.

Stop if: the fix would change `record_selection.py` selection behaviour →
that collides with Increment 1; report instead of interleaving.

### 3. `hubbard_u8` dimension-7 stall

Open item 2 of `BUGREPORT_adaptive_qse_root_indexing_20260826.md`: the
adaptive benchmark stalls at dimension 7 for `hubbard_u8`, with seed
directions dropped at `g = 0`.

Expected: root cause named, and either a fix with
`test/test_adaptive_qse_benchmark.py` extended, or a written argument that the
stall is correct behaviour for a degenerate `g = 0` pool.

Stop if: the cause lies in the shared provider (`paper_iii_problem.py`, out of
scope) → report with the evidence.

### 4. PAUSE — then, only if the author approves: SIAM `N_b=4` cost receipt

Bounded, mechanical, no physics results:

```bash
python -c "
from pipelines.qse_spectra.siam_static_adapter import graph_span_compiled_cost_receipt
import json; print(json.dumps(graph_span_compiled_cost_receipt('siam_nb4_pilot'), indent=1))"
```

Expected: Spearman rank correlation and top-decile overlap between the
graph-span proxy and Qiskit-compiled two-qubit counts on a deterministic
family-stratified sample. **Report the numbers; change nothing.** Low
agreement leaves SIAM rows `candidate` and is an author policy decision, not a
code fix. This is the first exercise of the transpiler path here, so a
Qiskit/backend availability failure is an expected outcome worth reporting
rather than working around.

---

## 6. Explicitly NOT in this handoff

These are live questions, already analysed, and **blocked on the author, not
on effort**. Do not start them:

- **Comparator spectral products** (transitions / Green functions for direct
  Davidson and real-time Krylov). Blocked on a design question —
  `QSEBasisElement` admits only Pauli strings and polynomials, while those
  methods' basis vectors are synthesized *states* — and on whether the paper
  claims cheaper *energies* (current; needs nothing) or cheaper *spectra*
  (would need these). See the evidence doc §Consequences.
- **A weight-aware accuracy crossing** beside `C*(eps_E)`. Requires the author
  to declare the spectroscopic observable set for Hubbard–Holstein first;
  which roots are "dark" is observable-dependent.
- **Changing `CANONICAL_METHOD_POLICY`** (exhaustion-only). A threshold switch
  runs today as an `early_switch` ablation cell; promoting it is an author
  decision that re-digests every public cell by design.
- **The six-regime HH production matrix** and the `N_b=4` SIAM pilot proper.
  Deliberately unrun while the method is unfrozen.
- **Manuscript edits.** The abstract and conclusion are known stale (they
  still claim Peierls / L=3 / older tiers); that is tracked separately.

---

## 7. Traps

- §4's `round_trace` coupling — the highest-value item in this document.
- `output/` and `prompt-exports/` are **gitignored**. Durable artifacts belong
  in tracked paths; this lane's docs live in `agent_guidance/qse/`.
- Exact references are post-run only. They annotate traces and score
  crossings; they must never select a record, trigger the Davidson handoff, or
  stop a phase. The producer enforces this structurally — do not route around
  it for convenience.
- Growth-trace runner defaults are diagnostic. The canonical dimension cap is
  64 and Krylov's is 40; capping Krylov low cripples the comparator instead of
  measuring it (this already happened once and produced a false result).
- Another agent's broad `git add` can sweep your uncommitted work into their
  commit. Commit at increment boundaries with explicit paths.
- `paper-iii-novelty-claims-boundary-20260818.md` was amended 2026-08-27
  (comparator set; evidence-inventory currency) and is now accurate. Its
  verdicts, never-claim lists, and the A-CASE/DA-CASE exclusion are
  authoritative and were not touched. **Do not edit it** — claims text is
  the author's. Two things it now records that bear on execution: the
  independent adaptive comparator is `direct_davidson_qse`
  (`adaptive_qse_benchmark.py`, the module Increment 2 debugs), and the
  Davidson/Krylov alphabet asymmetry is *declared*, not a defect to
  equalize away.
- The nph1 baseline JSON records `k_max=60, stride=2`; the current default cap
  is 64. Compare supports per `k`, not by row position.

---

## 8. Report back

Per the contract, one block per increment:

```markdown
## Increment N — <goal>
Status: done | blocked | skipped
Commands run:
Result:
Verification:   (include: §3 count, and nph1 support-lock outcome)
Deviations:
```

Include commit SHAs. For Increment 2, state the defect-vs-physics
determination explicitly — that is the deliverable, not the code change.
Surface surprises even on success.

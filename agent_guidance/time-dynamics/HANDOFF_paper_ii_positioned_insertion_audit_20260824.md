# Handoff — Paper II: audit positioned insertion for a scoring/realization mismatch

**Author:** Claude (session 2026-08-23/24) · **Executor:** Codex (repo access)
**Contract:** read `agent_guidance/shared/agent-handoff-contract.md` first. This
document supplies only what is specific to this task.

---

## 1. Anchors

| field | value |
|---|---|
| Checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` (**not** the `~/Documents/...` iCloud mirror) |
| Branch | `paper-ii-exchange-selector` |
| Base commit | `b12679c3501b04c576fb262bf363241e8929cc48` |
| Lane test baseline | `PYTHONPATH=. python3 -m pytest test/test_ap_mclachlan_*.py test/test_time_dynamics_*.py test/test_paper_ii_runs.py -q -p no:randomly` → **351 passed** |
| Run commands | `agent_guidance/time-dynamics/paper-ii-run-parameters.md` (generated; do not hand-edit) and `pipelines/time_dynamics/paper_ii_runs.py` |

---

## 2. The finding that motivates this

At **byte-matched numerics**, this route's growth-only arm and the AVQDS
comparator are indistinguishable:

| pairing | median mean-|ΔE| ratio (ours / AVQDS) |
|---|---|
| Euler + ridge 1e-7 + δθ 5e-3, both sides | **0.96×** |
| rk4 + ridge 1e-7, both sides | **0.98×** |

Six drives, one seed (`hh_snake_nph1`), t=10, 251 checkpoints, identical
125-word pool. Three drives each; ratios scatter 0.14×–2.33× with no systematic
direction.

That is parity, and it is surprising, because the two insertion rules differ
substantially by construction:

| | this route | AVQDS |
|---|---|---|
| placement | positioned at commutation-reduced cuts | appended at circuit end |
| ranking | captured drift ÷ Paper-I hardware cost | pure L² reduction |
| certification | materialize, gate, optional refit | none |

**Positioning is genuinely active** — this was checked, not assumed. Across the
exchange arm's 106 committed insertions the cut positions span 31 distinct
values from 0 to 41, e.g. cut 17 with 22 active parameters, cut 9 with 22, cut 2
with 32. Nothing is pinned at the end.

Two ablations that should have moved something, and did not (verified in run
provenance, not inferred):

- `--append-cost-alpha 0` → byte-identical results.
- `--no-certification-refit` → byte-identical results. (Refit is skipped for
  pure zero-angle insertions by design, so this one is explainable.)

**Hypothesis to test: the scoring geometry and the realized circuit disagree
about where the coordinate sits.** A coordinate inserted at cut `p` should have
tangent `U_{>p}^† A U_{>p}` — the generator conjugated by the suffix after `p`.
If candidate scoring uses an unconjugated (end-append) tangent while
`state_with_inserted_runtime_coordinates` materializes at `p`, then positioning
is real in the circuit but invisible in the score, and the route would behave
like end-appending with extra work. That matches every observation above.

`structural_cache.build_...` documents the intended behaviour explicitly:
"the ascending pass keeps one prefix state and applies each block's rotation to
the accumulated raw-column matrix, so each column added at cut `p` receives
exactly the suffix `U_{>p}`" (`structural_cache.py:128-130`). The audit is
whether the implementation does what that docstring says, end to end.

---

## 3. Scope

**In scope**
- `pipelines/time_dynamics/ap_mclachlan/structural_cache.py` (positioned tangents)
- `pipelines/time_dynamics/ap_mclachlan/geometry_eval.py` (frozen tangent matrix)
- `pipelines/time_dynamics/ap_mclachlan/state.py` (`state_with_inserted_runtime_coordinates`)
- `pipelines/time_dynamics/ap_mclachlan/exchange_structural.py` (scoring only)
- `test/test_ap_mclachlan_structural_cache.py`, `test/test_ap_mclachlan_insertion_state.py`

**Out of scope, with reasons**
- The selector's decision logic, debt policy, and guards. This task is about
  whether a candidate's *geometry* is computed correctly, not about which
  candidate is chosen. Changing both at once makes the result unreadable.
- `pipelines/static_adapt/**` (Paper I lane, another agent commits there).
- Manuscript edits. Report findings; the author edits the `.tex`.

**Shared-resource limits**: hard 10 GB ceiling for agent-launched processes on a
shared 8-core machine; one heavy local job at a time. No CHTC (needs the user's
interactive Duo login).

---

## 4. Increments

Proceed autonomously; commit at each boundary. Stop only on a stop condition.

### Increment 1 — an independent oracle for the positioned tangent

**Goal:** a test that builds the positioned tangent by brute force and compares
it against `structural_cache`.

Construct a small ansatz (4–8 coordinates, Hilbert dimension ≤ 64) and, for a
candidate generator `A` at cut `p`, compute the tangent two ways: (a) from the
structural cache; (b) by explicitly forming the circuit with a zero-angle `A`
inserted at `p` and differentiating that coordinate numerically (central
difference in its angle). They must agree to ~1e-8.

**Expected result:** agreement at every cut `p ∈ {0, …, N}`, not only at `p = N`.

**Stop condition:** if they agree **only** at `p = N`, the hypothesis is
confirmed — stop and report immediately with the failing cuts; do not proceed to
fix it in the same increment.

### Increment 2 — scoring/realization consistency

**Goal:** a test that the captured drift used to *score* a candidate equals the
captured drift *realized* after materializing it.

For several candidates at several cuts: record `q_of(removed, selection)` from
the selector's scoring path, then materialize with
`state_with_inserted_runtime_coordinates`, re-evaluate the geometry from
scratch, and compare captured drift. Agreement to solver tolerance.

**Expected result:** the two agree at every cut. Report the distribution of the
discrepancy; a systematic gap that grows with `N − p` is the signature of a
missing suffix conjugation.

**Stop condition:** any discrepancy above solver tolerance — report the pattern
before changing anything.

### Increment 3 — does positioning change the score at all?

**Goal:** quantify whether the cut position affects the score, on a real
checkpoint from the calibration seed.

For one checkpoint, score the *same* atom at every retained cut and report the
spread of captured drift across cuts. If the spread is at or near numerical
noise, positioning cannot affect selection regardless of correctness, and that
is a finding in itself — it would mean the commutation-reduced cut set is
degenerate for this Hamiltonian.

**Expected result:** a table of (atom, cut, captured drift). Either a real
spread, or a demonstrated non-spread with numbers.

### Increment 4 — report

If Increments 1–3 find a defect, propose the fix but **do not land it** together
with the tests; the author wants the defect characterized first.

---

## 5. Settled decisions — do not reopen

- **Pool cap 128**, above the 125-word deduplicated pool so it does not bind.
- **Conditioning gate off**; no setting of it was useful (5e7 never binds
  against observed κ ≈ 1e8; 3e7 costs accuracy; 1e7 starves the ansatz).
- **Subdivision budget 10**; at 4 a step could exhaust its budget, fail to cure a
  violation, and advance anyway (measured error 3.8e-3 → 2.1e-1).
- **Shared numerics ridge 1e-7**, author's decision.
- **The reverse-Schur identity** `Q(J) − Q(R) = r^T S^+ r ≥ 0` is exact,
  tested in `test_ap_mclachlan_schur_identity.py`, and must stay green.

## 6. Traps

- **An arm can silently run another arm's configuration.** Always verify from
  `summary.support_patch_config` in the emitted `run.json`, never from the
  command you believe you issued. Two ablations this session appeared to "have
  no effect" and the provenance check is what distinguished a genuine null
  result from a wiring failure.
- **`eval` of an empty string returns 0.** A command generator that fails
  silently produces a "successful" no-op run. Check the generator's exit status
  and that its output is non-empty.
- **`output/` and `prompt-exports/` are gitignored.** Durable artifacts go in
  tracked paths.
- **Another agent's broad `git add` can sweep uncommitted work.** Commit at
  increment boundaries.
- Route-parity goldens (`test/test_ap_mclachlan_route_parity.py`) must stay green
  or a deliberate change must appear as a justified diff.

## 7. Report back

One block per increment per the contract, with commit SHAs. For Increment 1 the
deliverable is the per-cut agreement table; for Increment 3, the per-cut
captured-drift spread. Numbers, not descriptions of numbers.

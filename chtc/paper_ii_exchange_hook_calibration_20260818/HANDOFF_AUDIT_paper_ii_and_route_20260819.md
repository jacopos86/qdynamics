# Audit handoff — drift, source locks, and run streamlining (2026-08-19)

**For:** Codex (repo access, executing)
**From:** Claude session that implemented the exchange route, simplified it, and
wrote today's manuscript sections
**Contract:** `agent_guidance/shared/agent-handoff-contract.md` — read first;
this document does not restate it.

## What this audit is for

Not re-deriving numbers. A number recomputed from the same configuration
reproduces the same error, so that check buys little. The exposures worth an
independent pass are the three places where a claim can be true at one layer and
false at the next:

1. **Math → code → run drift.** The manuscript states an equation; the code
   implements something; the run actually executed something else. Each link
   breaks independently, and today's session broke two of them (a scoring
   quantity that changed in code before the paper caught up; runs that used
   diagnostic integrator defaults while the method section described the
   accurate configuration).
2. **Source locks and comparator robustness.** A method comparison is only a
   comparison if every arm demonstrably shared the same physics, seed,
   integrator, grid, cutoff, and inverse policy. Today that was verified by
   reading launch scripts — which is how one "append-only" arm silently ran the
   exchange configuration for a whole batch.
3. **Streamlining regime/Hamiltonian matrices.** Repeated runs across regimes,
   cutoffs, and Hamiltonians should be declarative and locked, not rebuilt as
   ad-hoc shell each time.

## Anchors

| | |
|---|---|
| Checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` (NOT the `~/Documents/Holstein_implementation/...` iCloud mirror) |
| Branch | `paper-ii-exchange-selector` |
| Commit | `047e050a` — confirm with `git rev-parse --short HEAD`; a different SHA means another agent committed here (shared tree) — report the drift before proceeding |
| Test baseline | `python3 -m pytest $(ls test/test_ap_*.py) test/test_time_dynamics_campaign.py -q` → `226 passed` |
| Manuscript | `MATH/paper_details/time_dynamics_paper_II.tex` (13 pages, `pdflatex` twice) |
| Spec | `prompt-exports/paper_ii_noiseless_conditional_exchange_implementation_spec.md` |

## Scope

**In scope:** `MATH/paper_details/time_dynamics_paper_II.tex`,
`pipelines/time_dynamics/ap_mclachlan/**`,
`pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py`,
`pipelines/time_dynamics/campaign.py`, `test/test_ap_mclachlan_*.py`,
`agent_guidance/time-dynamics/AGENTS.md`.

**Out of scope:** `pipelines/excited_dynamics/**`, `pipelines/qse_spectra/**`
(Paper III lane, another agent commits there daily — a conflicting edit costs
more than the delay); `pipelines/static_adapt/**` read-only.

**Shared-resource limits:** one heavy local run at a time; other agents use this
machine. Wide or long runs belong on CHTC, and CHTC access needs a login only
the user can perform.

**Autonomy:** proceed through all increments without pausing; report at the end.
Stop early only if a finding invalidates later increments.

## Increment 1 — Math → code → run drift

For each mechanism the manuscript presents as part of the method, verify the
same object at three layers and report any layer that disagrees:

- **paper**: the equation and what it claims is minimized, thresholded, or ranked;
- **code**: the implementing expression, by file:line;
- **run**: what a completed `run.json` shows was actually used — provenance
  block, config echo, and decision payloads, not the launch command.

Mechanisms to check, in priority order:

1. Realized captured drift `Q = 2 f^T theta_dot - theta_dot^T K theta_dot` as
   *the* scoring authority. Legacy `Gamma = f^T theta_dot` survives as telemetry;
   any use of `Gamma` in a decision path is a drift bug. The paper defines both
   and says which governs — confirm the code agrees.
2. The operating score (insertion gain per cost + deletion cost per loss +
   weighted net drift change) as the complete criterion, with conditioning and
   history extensions at zero weight.
3. Deletion loss and insertion gain: same normalization in paper and code
   (`||b||^2 + eps` denominators), same positivity clipping.
4. Commit gates: Fubini–Study ray displacement and phase-aligned velocity
   smoothness, applied to deletion-containing patches only.
5. Measurement gating: below the residual threshold, insertions unenumerated and
   the pool empty; deletions still scored.
6. Numerical rule: the manuscript now claims one operating mechanism
   (state-displacement subdivision). Verify runs used it and not the wider
   candidate search.

**AVQDS comparator (`pipelines/time_dynamics/ap_mclachlan/avqds.py`) is a
drift case of its own.** It was implemented from the implementing agent's
understanding of Yao et al., PRX Quantum **2**, 030307 (2021), not a
line-by-line reading. Check the decision rule against the paper: is the
thresholded `L^2` the same quantity with the same normalization, does the greedy
append loop match, does AVQDS regularize its inverse the same way, and is
end-of-circuit placement faithful. A wrong rule biases the comparison in our
favor, which is the direction that matters.

Expected: a per-mechanism verdict (paper/code/run agree, or which layer
diverges), with file:line and the run artifact consulted.

## Increment 2 — Source locks and comparator robustness

The question: given only completed artifacts, can you *prove* that any two arms
being compared shared the same physics?

- Determine what a `run.json` currently pins: seed artifact identity and hash,
  Hamiltonian family and cutoff, drive profile, time grid, integrator, inverse
  policy, repair configuration, structural settings.
- Identify what it does **not** pin, such that two runs could differ without the
  artifact showing it. Same-cutoff comparison is a hard invariant here (binary-
  aligned phonon cutoffs 1/3/7; the exact reference must be the exact evolution
  of the same truncated Hamiltonian, never the untruncated model).
- Propose a comparison lock: a single record, emitted per campaign, that binds
  every arm to a shared physics/numerics fingerprint, plus a verifier that
  refuses to aggregate arms whose fingerprints differ.

Concrete failure to design against: today an "append-only" arm ran the exchange
configuration because a shell array substitution silently did not match, and the
two arms produced byte-identical results. Nothing in the artifacts flagged it —
only the suspicious equality did.

Expected: an inventory of what is and is not locked, plus a proposed lock format
and verifier entry point. Do not implement yet.

## Increment 3 — Streamlining regime/Hamiltonian matrices

`pipelines/time_dynamics/campaign.py` declares a matrix as
`SeedSpec x DriveSpec x HorizonSpec x PolicyArm`, resolving each cell to a runner
argv with canonical numerics, the four computational guards, per-seed sha256, and
binary-aligned cutoff enforcement. It is new and unproven beyond its own tests.

Assess it against the ICM architecture in
`agent_guidance/shared/icm-gitnexus-pilot-plan.md`, whose stated gap is that
Paper-II launch surfaces route by artifact paths rather than one neutral
accepted-ansatz export. Then answer:

- Does the campaign spec belong on top of the seed ledger
  (`chtc/generic_time_dynamics_table/input/paper_ii_seed_tracks_seed_ledger_v2.json`)
  so a seed is named by ledger key rather than by path?
- What is missing to run a matrix on CHTC from one declaration — submit-file
  generation, per-cell output routing, fetch and aggregation?
- Where should the lock from Increment 2 attach so every cell inherits it?

Expected: a concrete plan with named entry points, sequenced so each step is
independently testable. Do not implement yet.

## Deliberate, do not re-report

- Appendix A.3 is labeled diagnostic-only; its equations intentionally describe
  machinery not used for results.
- Conditioning and history score hooks exist at zero weight by design.
- `append_ladder_mode` is vestigial in this route but retained because
  `pipelines/excited_dynamics/paper_iii_promoted_ap_demo.py` passes it — a
  cross-lane coordination item.
- Short horizons (t<=1) and the comparator's non-binding threshold are known
  evidence limitations, already stated in the manuscript.

## Report back

Use the per-increment block from the contract. Additionally:

- Rank findings by whether they change a manuscript claim, invalidate a
  comparison, or are hygiene only.
- Propose fixes; do not apply them. The user decides what lands.
- Evidence standard from the contract applies: an index proposes, source
  ratifies. GitNexus missed three live closure-mediated calls out of four
  "unreachable" symbols today.
- If you find a trap worth carrying forward, add it to
  `agent_guidance/shared/agent-handoff-contract.md` §5.

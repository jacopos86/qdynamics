# Audit handoff — Paper II manuscript and the exchange route (2026-08-19)

**For:** Codex (repo access, executing)
**From:** Claude session that implemented the exchange route, simplified it, and
wrote today's manuscript sections
**Contract:** `agent_guidance/shared/agent-handoff-contract.md` — read it first;
this document does not restate it.

## Why this audit exists

Everything below was written by one agent over one long session: the route, the
simplification, the manuscript sections, and the numbers in them. The parts most
worth distrusting are the ones where that agent both produced a result and
described it. Three specific exposures:

1. **Numbers introduced today came from runs the same agent configured.** A
   misconfiguration (wrong integrator, missing guard, loose gate) produces a
   plausible number that survives review because the reviewer is the author.
   Several such misconfigurations *were* caught mid-session; assume others were
   not.
2. **The AVQDS comparator was implemented from the implementing agent's
   understanding of Yao et al., PRX Quantum 2, 030307**, not from a line-by-line
   reading during implementation. If its decision rule is wrong, the comparison
   table in the manuscript is wrong in our favor, which is the worst direction.
3. **Manuscript-code alignment was repaired by the agent that broke it.** One
   equation (Schur novelty) was found defined-but-unused only because the user
   asked. There may be more.

Audit for *whether the claims are true*, not for whether the work was done.

## Anchors

| | |
|---|---|
| Checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` (NOT the `~/Documents/Holstein_implementation/...` iCloud mirror) |
| Branch | `paper-ii-exchange-selector` |
| Commit | `047e050a` — confirm with `git rev-parse --short HEAD`; if it differs, another agent has committed here (they share this tree) and you should report the drift before proceeding |
| Test baseline | `python3 -m pytest $(ls test/test_ap_*.py) test/test_time_dynamics_campaign.py -q` → `226 passed` |
| Manuscript | `MATH/paper_details/time_dynamics_paper_II.tex`, builds to 13 pages via `pdflatex` run twice |
| Route entry | `pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py` |

## Scope

**In scope:** `MATH/paper_details/time_dynamics_paper_II.tex`,
`pipelines/time_dynamics/ap_mclachlan/**`,
`pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py`,
`pipelines/time_dynamics/campaign.py`, `test/test_ap_mclachlan_*.py`.

**Out of scope:** `pipelines/excited_dynamics/**`, `pipelines/qse_spectra/**`
(Paper III lane — another agent commits there daily; a conflicting edit costs
more than the delay). Also out of scope: `pipelines/static_adapt/**` except
read-only, and the Paper-I manuscripts.

**Shared-resource limits:** one heavy local run at a time; other agents use this
machine. Long or wide runs belong on CHTC, and CHTC access needs a login the
user must perform.

**Autonomy:** proceed through all increments without pausing. Report findings at
the end. Stop early only if an increment reveals a scientific error that would
invalidate later increments — say so rather than working around it.

## Increment 1 — Reproduce every number the manuscript states

The manuscript contains numbers introduced today. Reproduce each from committed
artifacts and report agreement or disagreement. Do not accept a number because
it appears in a caption.

| claim | stated value | where to verify |
|---|---|---|
| Comparator table (Table II) | append-only 1.7e-3 / 3.8e-3 / 46 params; adaptive append 1.8e-3 / 8.5e-3 / 68; exchange 3.5e-3 / 6.3e-3 / 24 | `output/three_way_hh_snake_20260818/*/run.json` via `output/three_way_report.py` |
| Mechanism figure captions | exchange at tau_ray=2e-3 conserves to 2.9e-4 vs append-only 1.3e-3; 616→607 vs 624 params | `output/paper_arms_rk4_20260818/` |
| Repair audit | 6 of 11,499 candidates applied; base rung on 883/885 steps; peak kappa 5.1e7 | `pipelines/time_dynamics/diagnostics/knob_audit.py` |
| Minimal-profile parity | 3.4735e-3 → 3.4738e-3, identical support, ~26% fewer candidate solves | `output/repair_profile_parity_20260818/` |
| Seed quality | HH L=2 nph=1 seed at 7.0e-5 static energy error | `chtc/generic_time_dynamics_table/input/paper_ii_seed_tracks_seed_ledger_v2.json` |

`output/` is gitignored, so these artifacts exist in the working tree but not in
git. If any is missing, say so rather than regenerating silently — a
regenerated number is a different measurement.

Expected: every stated value reproduces, or a precise list of those that do not.

## Increment 2 — AVQDS comparator fidelity

Read Yao et al., PRX Quantum **2**, 030307 (2021) and check
`pipelines/time_dynamics/ap_mclachlan/avqds.py` against it. The implementation
claims to reproduce AVQDS's *decision rule* on our geometry: McLachlan distance
`L^2 = ||b||^2 - Q` as trigger, greedy append of the operator maximally reducing
`L^2`, placed at the circuit end, no deletion, no cost weighting.

Check specifically:

- Is `L^2` as implemented the same quantity AVQDS thresholds, including
  normalization? Ours is absolute where our own route uses a normalized residual
  ratio; a normalization mismatch would make the threshold comparison
  meaningless.
- Does AVQDS append one operator or a set per step, and does our greedy loop
  match?
- Does AVQDS use a regularized inverse, and does ours match its convention?
- Is "append at the circuit end" faithful, or does AVQDS position operators?

Then judge the fairness question the manuscript already flags: at the reported
threshold the comparator saturates its per-checkpoint append budget, so the
threshold never binds. Is the manuscript's framing of that limitation honest, or
does it understate a comparison that favors us?

Expected: a fidelity verdict with specific line references, plus a
recommendation on whether Table II can stand as written.

## Increment 3 — Manuscript claims against live code

For every equation and mechanism the manuscript presents as part of the method,
confirm live code implements it and the route reaches it. The evidence standard
in the contract applies: a code index proposes, source ratifies — GitNexus
missed three live closure-mediated calls out of four "unreachable" symbols
today, so grep and read before concluding anything is dead.

Known-and-deliberate, do not re-report:

- Appendix A.3 is labeled diagnostic-only; its equations intentionally describe
  machinery not used for results.
- Conditioning and history score hooks exist at zero weight by design.
- `append_ladder_mode` is vestigial in this route but retained because
  `pipelines/excited_dynamics/paper_iii_promoted_ap_demo.py` passes it — a
  cross-lane coordination item, not an oversight.

Expected: a list of any remaining manuscript claims with no live implementation,
or confirmation there are none.

## Increment 4 — Route implementation audit

Read the selector stack for correctness against
`prompt-exports/paper_ii_noiseless_conditional_exchange_implementation_spec.md`:
`exchange_structural.py`, `exchange_certification.py`, `exchange_selector.py`,
`exchange_integration.py`, `structural_cache.py`.

Priorities, highest first:

1. **Scoring authority** — realized captured drift `Q = 2 f^T theta_dot -
   theta_dot^T K theta_dot` should be used everywhere the score is computed; any
   surviving use of the legacy `Gamma = f^T theta_dot` in a decision path is a
   bug.
2. **Deletion competition** — every guard-admitted deletion rung must be scored
   before the first certification attempt. This was fixed today; verify it holds.
3. **Measurement gating** — below the residual threshold the insertion pool must
   be empty and no insertion candidate enumerated. `test_ap_mclachlan_route_parity.py`
   locks this; confirm the lock actually binds.
4. **Guards** — the four computational guards should bound work without altering
   ranking. A guard that changes *which* candidate wins is a science bug.

Expected: defects with file:line, severity, and a regression test proposal each,
in the style of `chtc/paper_i_ra_adapt_repair_20260727/HANDOFF_REFACTOR_cost_path_20260819.md`.

## Report back

Use the per-increment block from the contract. Additionally:

- Rank findings by whether they change a manuscript claim, change a number, or
  are code hygiene only.
- For anything you would fix, propose the fix but do not apply it — this session
  is an audit, and the user decides what lands.
- If you find a trap worth carrying forward, add it to
  `agent_guidance/shared/agent-handoff-contract.md` §5 rather than only
  mentioning it here.

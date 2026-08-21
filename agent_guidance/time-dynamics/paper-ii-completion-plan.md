# Paper II completion plan

The route is implemented, simplified, audited, and locked. What remains is
evidence. This file is the loop we iterate through to get it, and the place to
record where we are. It is a plan, not an authorization: no phase here
authorizes a run, a commit, or an evidence promotion by itself.

Read with `agent_guidance/time-dynamics/AGENTS.md` (lane contract) and
`agent_guidance/shared/agent-handoff-contract.md` (delegating work).

## The loop

```
  P1 requirements ──> P2 run plan ──> P3 collection ──> P4 integration ──> P5 interpretation
        ^                                   │
        └───────────── method tweaks ───────┘   (tweaks return to P1, never skip ahead)
```

The back edge is the dangerous one. Collecting data will suggest changes to
three things in particular — the **ansatz seed**, the **drive**, and the
**AP-McLachlan parameters** (thresholds, tolerances, pool and guard settings).
Adjusting those is expected and healthy; that surface is what calibration is
for. What must not happen is choosing a seed, drive, or parameter set *because
it beat the comparator on the runs that will be reported*. The separation rule
below exists for that.

## The separation rule (read before tweaking anything)

Two disjoint sets of runs:

- **Calibration runs** — where thresholds, tolerances, guards, and method
  variants may be chosen freely by looking at results. Use the designated
  calibration seeds and regimes.
- **Production runs** — the runs whose numbers enter the manuscript. Their
  configuration is fixed *before* they are launched, recorded in the campaign
  manifest, and not adjusted afterward.

If a production run reveals a genuine defect, the fix returns to P1: the method
changes, the affected production cells are relaunched *whole*, and the
manifest records the supersession. Selecting the better of two production
configurations after seeing both is the failure this rule prevents, and it is
invisible in the final tables unless we refuse it here.

## P0.5 — Framing: what the paper claims to contribute

`MATH/paper_facing/paper_II_dynamics/novelty_and_prior_art_audit_20260718.md`
already records an external prior-art audit with a per-element claim ledger and
the closest prior-art families (AVQDS, pVQD and adaptive pVQD, AVQDS(T),
Pruned-ADAPT-VQE, open-system adaptive VQS). Its evidence boundary still holds:
those are research leads to verify against primary sources, and negative
searches are bounded by literature through 2026-07-18.

**That audit predates this route.** Several elements it classified as candidate
contributions have since been removed as vestigial: Schur novelty plus
augmented-solve confirmation, failed-append certificate reuse and secant retry,
the cost-weighted combinatorial batch ladder, and — as an operating claim —
bidirectional inverse repair with release hysteresis, now diagnostic-only.
History- and conditioning-aware prune nomination survives only at zero weight.

Conversely the route has since acquired elements the audit never scoped:

- **positioned insertion at commutation-reduced cuts**, with block-swap
  equivalence classes quotienting the placement choice;
- **measurement-gated structural maintenance** — insertions enumerated only
  above the residual threshold because they cost new tangent measurements,
  while deletions, being row and column selections of already-acquired
  geometry, are considered at every checkpoint;
- **realized captured drift** as the scoring authority in place of the
  idealized explained drift;
- **bounded trust-region refit** as what makes deletions certifiable at all.

So the framing question is live, and it is narrower than a general survey: what
is prior art for *positioned* insertion and for *measurement-economic* gating of
structural maintenance in variational dynamics? A refreshed literature pass
scoped to those two, plus a check that the retired elements are not still
claimed anywhere in the manuscript, is the useful next step. It is a pure
literature question needing no repo access, which makes it a good external
deep-research task rather than repo work.

Exit criterion: a claim ledger matching the current route, with each surviving
claim tied to a specific comparison the run matrix will support.

## P1 — Requirements: what data completes the paper

Current state: 191 `\tentative` markers, four placeholder regime figures, and
several structurally-fixed tables with dashes. The manuscript needs, per
element:

| element | status | data required |
|---|---|---|
| Mechanism table (Tab. I) | dashes | Policy arms (fixed support, append-only, exchange ± refit) on one seed per redundancy class, with structural miss, patch types, gain/loss, continuity, final support, compiled cost, measured-candidate count |
| Mechanism figures | preliminary, stress seed | Regenerate on a conventional VQE seed at production horizon |
| Comparator table (Tab. II) | stale twice | Rerun: published L² convention, uncapped append rule, shared pool, production horizon |
| Comparator figure | preliminary | Same runs as Tab. II |
| Four regime blocks | placeholders | Driven HH trajectories in weak--weak, strong--weak, weak--strong, strong--strong with exact reference and Qiskit-compiled costs |
| Qiskit comparator rows | placeholders | TrotterQRTE, PVQD, VarQRTE at matched seed/drive/grid/cutoff |
| Seed-track table | dashes | Append vs SNAKE seeds at matched static error |

Exit criterion for P1: every element above has a named owner cell in a campaign
manifest, or an explicit decision to cut it from the paper.

Open requirement decisions (these change what we run, so settle them here):

1. **Horizon.** Everything so far is `t <= 1`, where staleness and accumulated
   error barely appear. Production likely needs `t ~ 5-10`.
2. **Regimes and cutoffs.** Binary-aligned cutoffs only (1, 3, 7). Which
   coupling regimes get full blocks, and at which cutoff each.
3. **Comparator set.** AVQDS is essential (closest prior art). Whether the
   Qiskit trio stays or is cut to one non-variational reference.
4. **Pool cap.** The comparator cannot reach its threshold at a capped pool.
   Either uncap it for both policies and bound deletion width another way, or
   report at matched pool and say so.

## P2 — Run plan

Declare the matrix in `pipelines/time_dynamics/campaign.py`
(`SeedSpec x DriveSpec x HorizonSpec x PolicyArm`), write the manifest, and
review it before launching. The manifest is the contract: it fixes every
production configuration in advance, which is what makes the separation rule
enforceable.

Missing machinery to build here:

- CHTC submit generation from a campaign spec (the pattern exists in
  `chtc/paper_ii_exchange_hook_calibration_20260818/`, not yet driven from the
  spec);
- seeds named by ledger key rather than path
  (`chtc/generic_time_dynamics_table/input/paper_ii_seed_tracks_seed_ledger_v2.json`);
- per-cell fetch and aggregation, with `assert_comparable` called before any
  cell group is aggregated.

Exit criterion: a manifest whose cells cover every P1 element, dry-run verified
in an isolated transfer set.

## P3 — Collection

Launch, monitor, fetch. Runs stream progress; check liveness after launch and
report at intervals rather than waiting silently. Long or wide work goes to
CHTC; the local machine is shared and holds one heavy job at a time.

Exit criterion: every manifest cell has a returned artifact carrying a run lock,
and `assert_comparable` passes within each comparison group.

## P4 — Integration

Numbers and figures move into the manuscript from returned artifacts only, never
from a scratch rerun. Each element loses its `\tentative` marker only when its
cell is complete and locked. Captions state what was held fixed and what varied.

Exit criterion: no `\tentative` marker remains on an element backed by a
complete cell; markers that remain identify genuinely missing evidence.

## P5 — Interpretation

Read the completed matrix for what it actually shows, including where the method
loses. The current honest position is an accuracy-versus-support trade against
AVQDS, not dominance; if production data changes that in either direction, the
claims change with it.

Exit criterion: discussion and conclusion state claims bounded by the evidence
in hand, with scope limits named rather than implied.

## Where we are

- **P0 complete**: route implemented and spec-faithful; simplification audited;
  comparator matched to its source (published L² convention, no append cap,
  shared pool); run locks binding physics; canonical defaults; parity locks.
- **P0.5 in progress**: a 2026-07-18 prior-art audit exists but predates the
  simplification; it credits retired machinery and misses positioned insertion
  and measurement gating. Needs a scoped refresh.
- **P1 in progress**: table above drafted; the four open decisions are unsettled.
- **P2 not started**: campaign spec exists, CHTC generation does not.

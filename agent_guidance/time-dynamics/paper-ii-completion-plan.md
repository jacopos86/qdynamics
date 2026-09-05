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

## Framing decision (2026-08-20)

**The contribution is prune and exchange.** Online removal of support during
variational dynamics, with certification that removal is safe, is the claim;
everything else — positioned insertion, measurement gating, the numerical
repair lane, seed construction — is supporting machinery that makes the claim
testable. Seed construction in particular is settled work consumed from the
static lane, not a contribution here.

**The success criterion is reduced circuit cost at matched or better
accuracy.** That single sentence determines the run matrix: every production
cell exists to place a policy on the cost--accuracy plane against a comparator
that shares its physics.

### What the current evidence does and does not support

On a **redundant seed** (620 coordinates, driven, matched numerics) exchange
achieved both: mean energy error $1.31\times10^{-3}$ against append-only's
$1.54\times10^{-3}$, at 619 coordinates against 631. Better accuracy, smaller
circuit — the claim as stated.

On a **compact seed** (28 coordinates) it did not: exchange held
$3.5\times10^{-3}$ on 24 coordinates while the adaptive-append comparator
reached $1.0\times10^{-3}$ on 76. Cheaper, but less accurate — a trade, not the
claim.

The honest reading is that the claim holds where support carries redundancy and
fails where it does not, which is exactly what a removal mechanism should do.
That is a scope statement, not a caveat to bury: **the paper claims cost
reduction at matched-or-better accuracy for ansätze that carry redundant
support**, and reports the compact-seed trade as the boundary of that claim.

### What this demands of the run matrix

1. **Seeds must carry realistic redundancy.** A referee will ask whether
   redundancy was manufactured to be removed. Answer it with seeds from the
   static lane's own ledger at depth, where redundancy accumulates naturally,
   rather than only the deliberately bloated conditioning-stress seed. If a
   stress seed appears, it is labeled as a stress test, not as the result.
2. **Cost must be reported in compiled hardware terms**, not only in
   coordinate count, since the claim is about circuit cost.
3. **Accuracy must be matched-or-better against a comparator sharing physics**,
   verified by the run-lock fingerprint rather than asserted.
4. **The boundary must be mapped, not avoided**: include at least one compact
   seed so the trade is visible and quantified.

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

Under the framing decision above, the live literature question narrows further:
what is prior art for **online support removal during variational real-time
dynamics**, and for certifying that a removal is safe mid-trajectory? Pruned-
ADAPT-VQE is the nearest quantum precedent but prunes after ground-state
optimization, not online. Positioned insertion and measurement-economic gating
matter as supporting mechanisms and should be checked, but they are not the
claim. A refreshed literature pass
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

## arXiv preprint scoping audit (2026-09-04)

Trigger: proposal to restrict the main text to state-motion controllers and
move state-vs-parameter motion, plus whitened parameter motion, to the
appendix.

**Negative result first: state-motion-only main text costs six of the eight
matched pairs and the entire upper half of the headline range.** It also
removes Fig.~1 (no strong--fast state-motion cell exists in the locked 15) and
leaves zero main-text comparisons in which AVQDS runs its own published
controller.

### Measured consequence, locked Table I (11 completed cells)

Terminal compiled-resource reduction, realized-$\Gamma$ vs AVQDS, from
`generated/paper_ii_t20_q_compiled_frontier_table.tex`
(comparison_sha256 `a5e0e1a8e0f0cfff...`):

| drive | controller | matched | $N_{2q}$ | $D_{2q}$ | $D_c$ |
|---|---|---|---:|---:|---:|
| fast--weak | $\delta\theta$ | yes | 44.2% | 52.1% | 54.7% |
| fast--weak | combined | yes | 47.6% | 51.2% | 48.5% |
| fast--weak | state motion | yes | 35.6% | 33.3% | 33.1% |
| mid-resonant | $\delta\theta$ | yes | 32.9% | 37.3% | 36.0% |
| mid-resonant | state motion | yes | 29.6% | 29.7% | 28.6% |
| slow--strong | $\delta\theta$ | no | 25.5% | 29.9% | 33.4% |
| strong--fast | $\delta\theta$ | yes | **95.4%** | **96.1%** | **96.1%** |
| strong--fast | combined | yes | 43.1% | 45.2% | 47.4% |
| fast--fast | $\delta\theta$ | yes | 41.3% | 48.3% | 48.7% |
| fast--fast | combined | no | 38.4% | 39.6% | 36.3% |
| fast--fast | state motion | no | 61.8% | 62.2% | 60.7% |

Headline range by main-text controller restriction:

| Main-text scope | cells | matched pairs | $N_{2q}$ reduction |
|---|---:|---:|---|
| all three controllers (current) | 11 | 8 | 29.6--95.4% |
| $\delta\theta$ only | 5 | 4 | 32.9--95.4% |
| state motion + combined | 6 | 4 | 29.6--47.6% |
| **state motion only** | **3** | **2** | **29.6--35.6%** |

### Why state-motion-only weakens the comparator, not just the range

Each locked cell fixes one controller for *both* arms
(`paper_ii_final_matched_cells_v1.json`: the AVQDS arm of
`fastfast__state_motion_5e-5` is
`screen_avqds_...__avqds__state_motion_5e-5__fastfast__...`). A state-motion
main text therefore reports every AVQDS comparator under a controller this
paper introduced, not the raw-coordinate rule AVQDS publishes
(Eq.~\ref{eq:raw_coordinate_step_control}, Yao2021). The structural claim —
prune-and-exchange reduces compiled cost at matched accuracy — is strongest
when the comparator runs its own published numerics.

`\delta\theta` is also the best-populated controller in the lock: 6 cells, 5
complete, 1 pending (`weakslow`). State motion has 5 cells, 3 complete, and
`strongfast` has no state-motion cell at all.

### Adopted scoping recommendation

Keep the user's structure — one controller carries the main text, controller
comparison moves to the appendix — but invert which controller is primary:

1. **Main text**: $\delta\theta$ cells only. Table I becomes 5 rows, 4 matched
   pairs, 32.9--95.4%. Fig.~1 (strong--fast, $\delta\theta$) is unchanged.
   AVQDS runs its published controller in every reported comparison.
2. **Appendix controller section** (extend
   `app:metric_eigenmode_controller`): state motion vs parameter motion at
   common $\tau_{L^2}$ (the existing 53.9--60.6% step and 58.0--59.0%
   $W_{\rm geom}$ result), then whitened parameter motion (retained-metric
   eigenmode, 62.2%/34.6%). Both already have locked evidence.
3. The `combined` controller leaves the main text with state motion.

This costs 4 matched pairs relative to the current draft and nothing from the
headline range.

### Submission-readiness state of the manuscript

Current build is clean: 14 pages, revtex4-2 `pra reprint`, 45 inline
`\bibitem`s, 0 overfull boxes, no undefined references or citations.

Blockers for arXiv, in order:

1. `\placeholder{Add funding, compute resources, discussions, and advisor
   acknowledgments.}` at `time_dynamics_paper_II.tex:738` — the only live
   placeholder in the source.
2. `\graphicspath{{../../}{}}` escapes the source directory; an arXiv tarball
   needs flattened `figures/` paths.
3. **Source-of-truth split.** Three divergent `.tex` files exist and the newest
   is not in git:
   - `AP-McLachlan---Paper-II/Paper_II_overleaf_20260901/` @ `main`,
     uncommitted, sha256 `cd35061e...` (2026-09-04 16:06)
   - `Documents/Codex/2026-09-01/l-2-relates-to-eq-9/outputs/` sha256
     `f5b5e035...` (2026-09-04 18:21) — identical structure, renames
     $L^2\to m(\dot\theta)$ and $\tau_{L^2}\to\tau_m$ across 61 lines; **in no
     git repository**
   - worktree `codex/repaired-proxy-paper-ii-20260903` sha256 `e965398b...`
     (2026-09-03), 1376 lines divergent
   The $m(\dot\theta)$ rename must land in the manuscript repository before any
   scoping edit, or the edit will be made against superseded notation.

### Not adopted without user decision

Evidence status of the fresh 30/45 selector comparison remains `analysis only`;
its calibration gate is `INCOMPLETE` (`benchmark_point_sameness` and
`response_normalized_matched_accuracy` not run). Table I continues to draw on
the locked cells, not that comparison.

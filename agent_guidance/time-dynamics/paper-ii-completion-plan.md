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

### Correction (2026-09-04): the shot axis was omitted above

The paragraph previously here argued that running AVQDS under the state
controller weakens the comparator. That is wrong, and it ignored the axis the
state controller exists to move. Within-drive controller effect, from the
fresh comparison (`four_route_analysis_exact_20260904_partial45`):

| drive | route | steps $\delta\theta$ | steps state | $\Delta$ | $W_{\rm tot}$ $\delta\theta$ | $W_{\rm tot}$ state | $\Delta$ | burden |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mid-resonant | realized-$\Gamma$ | 1,084,151 | 499,904 | $-53.9\%$ | $1.715\times10^9$ | $0.879\times10^9$ | $\mathbf{-48.8\%}$ | 0.0503→0.0564 (+12%) |
| mid-resonant | AVQDS | 947,997 | 499,904 | $-47.3\%$ | $1.715\times10^9$ | $0.922\times10^9$ | $\mathbf{-46.2\%}$ | 0.0304→0.0381 (+26%) |
| fast--fast | realized-$\Gamma$ | 1,279,472 | 504,448 | $-60.6\%$ | $1.719\times10^9$ | $0.778\times10^9$ | $\mathbf{-54.8\%}$ | 0.0399→0.1016 (+154%) |
| fast--fast | AVQDS | 687,788 | 504,448 | $-26.7\%$ | $1.484\times10^9$ | $1.896\times10^9$ | $+27.7\%$ | 0.0486→0.0271 ($-44\%$) |

The state controller cuts realized-$\Gamma$ estimator work by 48.8--54.8% and
AVQDS estimator work by 46.2% on mid-resonant. It does not handicap the
comparator: on fast--fast AVQDS it buys 44% lower accuracy burden for 27.7%
more work, because that AVQDS support grows to 100 coordinates and $g(p)$ is
quadratic in support size.

#### Why the one exception is not a controller cost

The state norm $\sqrt{v^\top Gv}$ is classical once $G$, $f$, and
$\|\bar d\|_2^2$ are estimated (Sec.~\ref{sec:numerical_stabilization}), so it
adds zero estimator evaluations per right-hand-side evaluation. At fixed
support, fewer steps therefore means strictly less work. The fast--fast AVQDS
increase is entirely support growth, not controller overhead:

| drive | route | terminal $N_\theta$, $\delta\theta$ → state |
|---|---|---|
| mid-resonant | AVQDS | 60 → 60 |
| fast--fast | AVQDS | 65 → **100** |
| mid-resonant | realized-$\Gamma$ | 49 → 48 |
| fast--fast | realized-$\Gamma$ | 47 → 49 |

Append-only AVQDS grew 54% more support under state-motion control on
fast--fast; $g(p)=p(p+1)/2+p+2$ then outruns the 26.7% step saving. AP-McLachlan
holds its support in both drives, which is the prune--exchange claim doing the
work.

#### Units precision

These are logical geometry-entry counts for the dense-statevector
implementation, not hardware shots. Physical shot count is
`unavailable_statevector_no_shot_ledger` for every cell. Manuscript prose must
keep timestep count, estimator/query work, and terminal circuit resources
distinct and must not call $W_{\rm geom}$ a shot count.

**The circuit axis and the shot axis are different claims.** Terminal
$\boldsymbol C_{\rm comp}$ reduction is the structural (prune--exchange)
contribution; $W_{\rm geom}$ reduction is the numerical (controller)
contribution. A main text scoped to state motion leads with the second.

### The actual constraint on a state-motion main text

Not comparator fidelity — **cell coverage and one failed accuracy match**:

- Only 3 of the 5 locked `state_motion_5e-5` cells have a realized-$\Gamma$
  trajectory: fast--weak (matched), mid-resonant (matched), fast--fast
  (**unmatched**, burden $0.0399\to0.1016$ fails the factor-two rule).
  `slowstrong__state_motion_5e-5` and `weakslow__state_motion_5e-5` are locked
  and incomplete.
- `strongfast` has **no** state-motion cell in the locked 15. Fig.~1 is
  strong--fast/$\delta\theta$; a state-motion main text has no strong--fast
  counterpart without extending the lock, which is a production run under the
  separation rule.

So state-motion-only today is 3 cells / 2 matched pairs / 29.6--35.6%, and a
new Fig.~1 drawn from fast--weak or mid-resonant.

### Revised recommendation

Run the two locked-but-incomplete state-motion cells (`slowstrong`,
`weakslow`) before deciding the main-text spine. They are already in the
manifest, so completing them is collection, not a new configuration choice.
That takes state motion from 3 cells to 5 and settles whether the matched-pair
count supports carrying the main text. Hold the $\delta\theta$/state-motion
scoping decision until they return.

If the state-motion spine is adopted regardless of that outcome, the honest
main-text framing leads with $W_{\rm geom}$ (48.8--54.8% realized-$\Gamma$
estimator-work reduction) and reports terminal compiled resources second, with
the fast--fast accuracy-match failure stated rather than dropped.

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

## Which controller maximizes APM's margin over AVQDS (2026-09-04)

Question asked: on current results, which controller makes AP-McLachlan look
best relative to AVQDS.

**Answer: $\delta\theta$** — and the reason is that the state controller's work
saving largely *cancels* against AVQDS, because both arms of a cell share the
controller.

### Ranked on the paper's own success criterion (matched accuracy)

| controller | matched pairs | mean $N_{2q}$ reduction | mean $D_c$ reduction | mean $W$ reduction vs AVQDS |
|---|---:|---:|---:|---:|
| **$\delta\theta$** | **4** | **+53.4%** | **+58.9%** | +4.2% |
| combined | 2 | +45.4% | +48.0% | +1.5% |
| state motion | 2 | +32.6% | +30.8% | +4.7% |

Per-cell $N_{2q}$: $\delta\theta$ = fast--weak 44.2, mid-res 32.9,
**strong--fast 95.4**, fast--fast 41.3; state = fast--weak 35.6, mid-res 29.6.

### Why the work axis does not separate them

The controller applies to both arms, so the 48.8--54.8% realized-$\Gamma$
estimator-work saving recorded above is matched by a 46.2% AVQDS saving.
The *ratio* barely moves:

| drive | $W_{\rm APM}/W_{\rm AVQDS}$, $\delta\theta$ | same, state motion |
|---|---:|---:|
| mid-resonant | 1.000 | 0.953 |
| fast--fast | 1.158 | **0.410** |

State motion's headline "+31.8% mean work advantage over all completed cells"
comes almost entirely from fast--fast, where AVQDS's append-only support grew
to 100 coordinates. **That cell fails the factor-two accuracy match**
(burden ratio 3.74) and therefore cannot carry the claim. Excluding it, state
motion's advantage over AVQDS is +4.7% on one cell — indistinguishable from
$\delta\theta$'s +4.2% on three.

Restated: the state controller is the better controller in absolute terms
(verified, 46--55% work reduction on both policies), but it is not the
controller that maximizes APM's *relative* margin, because AVQDS collects the
same benefit.

$W$ figures are from the fresh analysis-only comparison
(`four_route_analysis_exact_20260904_partial45`), whose calibration gate is
`INCOMPLETE`. $N_{2q}$/$D_c$ and every match flag are from locked Table I.

### Selection-integrity note

Choosing the main-text spine after seeing which subset flatters is post-hoc.
It is defensible only if every completed cell still appears in the paper, with
the non-flattering ones in the appendix rather than dropped — which is the
structure already proposed. Under that constraint this is emphasis, not
selective reporting. Cells must not be removed from Table I on this basis.

## Mainlining state vs parameter motion (2026-09-04)

Directive: carry state-vs-parameter controller results in the main text rather
than only in the appendix.

**The locked comparison already supports four times what §5.3 reports.**
`four_route_analysis/comparison.json` (SHA-256 `a5e0e1a8e0f0cfff...`, the same
file Table~I cites) contains 8 within-drive state-vs-parameter pairs. §5.3
currently reports 2 of them, both realized-$\Gamma$. The 5 AVQDS-side pairs are
absent from the manuscript entirely.

$W_{\rm geom}$ = integration + checkpoint-base geometry work, matching
Appendix~\ref{app:estimator_query_accounting}. Reproduces §5.3's published
$-58.0\%$/$-59.0\%$ and $-53.9\%$/$-60.6\%$ exactly.

| drive | route | steps | $W_{\rm geom}$ | burden ratio | factor-2 | $N_\theta$ |
|---|---|---:|---:|---:|:---:|---|
| fast--weak | realized-$\Gamma$ | $-29.8\%$ | $-22.4\%$ | 1.30 | yes | 45→49 |
| fast--weak | AVQDS | $-37.5\%$ | $-38.4\%$ | 1.25 | yes | 61→60 |
| mid-resonant | realized-$\Gamma$ | $-53.9\%$ | $-58.0\%$ | 1.12 | yes | 49→48 |
| mid-resonant | AVQDS | $-47.3\%$ | $-46.3\%$ | 1.26 | yes | 60→60 |
| slow--strong | AVQDS | $-51.7\%$ | $-49.3\%$ | 1.91 | yes | 62→64 |
| fast--fast | realized-$\Gamma$ | $-60.6\%$ | $-59.0\%$ | 2.54 | **NO** | 47→49 |
| fast--fast | AVQDS | $-26.6\%$ | $+26.8\%$ | 0.56 | yes | 65→**100** |
| weak--slow | AVQDS | $-60.2\%$ | $\mathbf{-76.6\%}$ | **1.03** | yes | 945→**370** |

**7 of 8 reduce propagation work (22.4--76.6%); 7 of 8 hold factor-two
accuracy.**

### What this changes about the claim

1. **It is not an AP-McLachlan trick.** Five of the eight pairs are AVQDS. The
   controller reduces append-only AVQDS propagation work by 38.4--76.6% on four
   of five drives. A numerical construction that also helps the comparator is a
   contribution to McLachlan dynamics generally, and is far harder to dismiss
   than one that only helps the proposing method.
2. **The strongest single data point is unreported.** weak--slow AVQDS:
   $-76.6\%$ propagation work at burden ratio **1.03** — essentially unchanged
   accuracy — with terminal support falling 945→370.
3. **Both exceptions are fast--fast and both are explainable.**
   Realized-$\Gamma$ takes the largest work saving ($-59.0\%$) but fails the
   factor-two rule at 2.54. AVQDS is the only work *increase*, caused by support
   growth 65→100 against quadratic $g(p)$, and it buys a 0.56 burden ratio.
   Report both; neither is a defect to hide.

### Recommended main-text form

Replace §5.3's two prose paragraphs with the 8-row table above plus the
retained-metric-eigenmode paragraph as-is. This mainlines state vs parameter
motion on locked evidence, needs no new runs, and states the two fast--fast
exceptions in the same table rather than omitting them.

Caveat to state in the caption: the comparison is at equal nominal cap
($5\times10^{-5}$ in both controllers), and the two caps have different units,
so the burden-ratio column is what establishes accuracy comparability — not the
cap value. This is the same standard the retained-metric-eigenmode ablation
already meets by construction.

This supersedes the "both metric controllers in the appendix" recommendation
above: the state-vs-parameter comparison belongs in the main text.

## Correction: "7 of 8 hold factor-two accuracy" was the wrong test (2026-09-04)

The previous section applied the *aggregate* accuracy burden. The manuscript's
declared rule (§\ref{sec:matched_comparison}) is stricter: both arms must
qualify individually against the absolute safety ceilings **and** their
**componentwise** response burdens must agree within a factor of two, over the
three observable components (doublon, energy, max site occupation; infidelity
is excluded). Reproduced against the stored
`pairwise_accuracy.old_proxy__avqds.worst_component_ratio` for
`weakslow__delta_theta_5e-5`: computed 1.8580, stored 1.8580.

Under the declared rule the state-vs-parameter comparison is **4 of 8**, not
7 of 8:

| drive | route | both qualify | burden ratio | worst component | matched |
|---|---|:---:|---:|---:|:---:|
| fast--weak | realized-$\Gamma$ | yes | 1.30 | 1.66 | **yes** |
| fast--weak | AVQDS | yes | 1.25 | 1.28 | **yes** |
| mid-resonant | realized-$\Gamma$ | yes | 1.12 | **5.62** | no |
| mid-resonant | AVQDS | yes | 1.26 | 1.55 | **yes** |
| slow--strong | AVQDS | yes | 1.91 | 2.21 | no |
| fast--fast | realized-$\Gamma$ | yes | 2.54 | 2.98 | no |
| fast--fast | AVQDS | yes | 1.79 | 1.88 | **yes** |
| weak--slow | AVQDS | **no** | 1.03 | 1.06 | no |

### The manuscript's own published number fails this rule

§5.3 currently reports mid-resonant realized-$\Gamma$ at $-53.9\%$ steps and
$-58.0\%$ $W_{\rm geom}$. That pair has worst component ratio **5.62**, driven
by normalized energy error $0.0086$ ($\delta\theta$) against $0.0484$ (state
motion). Aggregate burden ratio is 1.12, which is why it looked matched. It is
not matched under the declared rule, and §5.3 does not currently claim matched
accuracy for it — it reports the infidelity change and stops. **The prose must
not be upgraded to a matched-accuracy claim.**

Note the rule's behavior here: it fails the pair because the $\delta\theta$ arm
is unusually *accurate* on energy, not because state motion is inaccurate. That
is a property of a ratio test and is not grounds for changing the rule
post hoc.

### What survives as a matched-accuracy claim

| drive | route | steps | $W_{\rm geom}$ |
|---|---|---:|---:|
| fast--weak | realized-$\Gamma$ | $-29.8\%$ | $-22.4\%$ |
| fast--weak | AVQDS | $-37.5\%$ | $-38.4\%$ |
| mid-resonant | AVQDS | $-47.3\%$ | $-46.3\%$ |
| fast--fast | AVQDS | $-26.6\%$ | $+26.8\%$ |

**Three of four matched pairs reduce propagation work by 22.4--46.3%; the
fourth increases it by 26.8%. Exactly one is realized-$\Gamma$.**

The equal-nominal-cap results ($-58.0\%$, $-59.0\%$, and the $-76.6\%$
weak--slow AVQDS point) remain reportable as equal-cap ablations, clearly
labeled as such, which is what §5.3 does today. They are not matched-accuracy
results.

### Consequence for mainlining

The 8-row table recommended above stays, but its caption must carry the
declared-rule column and say plainly that 4 of 8 pairs are matched. Mainlining
the comparison is still right; describing it as a matched-accuracy query
reduction is not. To obtain a matched state-vs-parameter comparison at the
strength the appendix ablation already has, the state cap would have to be
retuned per drive the way $\epsilon_y$ was — that is new production work under
the separation rule, not a re-read of existing artifacts.

## Model identity of the locked evidence (2026-09-04)

**Every Paper II locked cell is Hubbard--Holstein. There is no SIAM evidence in
the lock.** All 92 physics blocks across the 15 cells and every route agree:

```text
family_key           hh
operator_label       hh_spinful_onsite_density
n_ph_max             1
seed_sha256          7a36a2da06bba447...  (hh_snake_nph1.json, single seed)
```

The SIAM route *is* enabled on `main` and imports cleanly:
`siam_nb2_proof_v1` and `siam_nb4_pilot_v1` search profiles, arms
`exchange`/`avqds`, builders `build_siam_nb2_proof_run`,
`build_siam_nb4_screen_run`, `build_siam_nb4_continuation_run`
(`pipelines/time_dynamics/paper_ii_runs.py`, ported 2026-08-31 from
`codex/paper-ii-search-profiles-20260824`), backed by
`src/quantum/siam_model.py` and `pipelines/time_dynamics/siam_seed.py`.

SIAM screening artifacts exist but sit in other worktrees and are
Paper-III-facing (`codex-paper-iii-quantum-comparators-20260903`,
`codex-siam-repaired-proxy-20260903`). None feed Paper II.

**Preprint consequence.** The manuscript's entire evidence base is one model,
one seed, one lattice, one drive family. That is a scope limit a referee will
name, and §\ref{sec:evidence_scope} should state it explicitly rather than
leave it inferable from the seed name. The SIAM Nb=2 proof pair is the
ready-made generality answer, but it is new production work under the
separation rule and Paper III already consumes that lane.

## SIAM dimension, regime coverage, and the CHTC handoff (2026-09-04)

### SIAM is not larger at Nb=2; it is larger at Nb=4

`SIAMLayout`: `modes_per_spin = n_bath + 1`, `n_qubits = 2 * modes_per_spin`.

| system | qubits | Hilbert dim |
|---|---:|---:|
| Paper II locked Hubbard--Holstein ($L=2$, $n_{\rm ph}^{\max}=1$) | 6 | 64 |
| SIAM $N_b=2$ | 6 | 64 |
| SIAM $N_b=4$ | 10 | 1024 |

SIAM $N_b=2$ is the same size as the published system, so it is a *generality*
result, not a scale result. $N_b=4$ is 16$\times$ the state space.

### The locked evidence is one Hamiltonian regime

From `hh_snake_nph1.json` (`seed_sha256 7a36a2da...`), identical in all 15 cells:

```text
L = 2       t = 1.0      U = 4.0      g_ep = 0.5
omega0 = 1.0             n_ph_max = 1
boundary = open          boson_encoding = binary
hilbert_dim = 64         num_qubits = 6      hamiltonian.num_terms = 17
```

Six *drives* $(A,\omega)$ were varied. **One coupling regime.** This is
open decision #2 of P1, still unsettled.

### A regime bundle is already staged and has never been submitted

`chtc/paper_ii_regime_seeds_v1` — seed construction via
`runners/build_fixed_vqe_conditioning_seed.py`, 6 jobs, 4 CPU / 24 GB / 40 GB
disk / 24 h cap, `queue regime_id from input/regime_ids.txt`. No outputs exist
anywhere in the tree.

| regime_id | $U$ | $g_{\rm ep}$ | $n_{\rm ph}^{\max}$ | qubits |
|---|---:|---:|---:|---:|
| `weak_weak` | 0.25 | 0.3536 | 3 | 8 |
| `intermediate_weak` | 1.25 | 0.3536 | 3 | 8 |
| `strong_weak_u8` | 8.00 | 0.3536 | 3 | 8 |
| `weak_strong` | 0.25 | 0.7906 | 7 | 10 |
| `intermediate_strong` | 1.25 | 0.7906 | 7 | 10 |
| `strong_strong_u8` | 8.00 | 0.7906 | 7 | 10 |

**Defect: the grid does not contain the published regime.** $U=4.0$ falls
between `intermediate` (1.25) and `strong` (8.00); $g_{\rm ep}=0.5$ falls
between the two $g$ columns; $n_{\rm ph}^{\max}=1$ appears in no row. Nothing
in the new grid nests the existing evidence, so a referee cannot check that the
seed-construction path reproduces the published point.

**Decision (agent's call): add a seventh anchor row**
`anchor_u4_g0p5_nph1` at $U=4.0$, $g_{\rm ep}=0.5$, $n_{\rm ph}^{\max}=1$. It
is the cheapest job in the set (6 qubits) and it is the only row that makes the
new seeds comparable to the locked one.

### Blockers before this can be handed to Codex

1. **`chtc/time_dynamics_optuna/image.sif` is absent from this checkout.** The
   submit file names it in `transfer_input_files`. It is a container image and
   correctly not in git; `chtc/time_dynamics_optuna/build_image.sh` is present.
   The image must be built or fetched before submission.
2. **The handoff is two stages, not one.** `paper_ii_regime_seeds_v1` produces
   *seeds*. The dynamics campaign consumes them and cannot be materialized
   until the seed SHA-256s return. Codex monitors stage 1, then generates
   stage 2 from the returned seeds using an existing packager
   (`prospective_apm_exchange_chtc_package.py`,
   `prospective_avqds_trajectory_chtc_package.py`,
   `prospective_exact_reference_chtc_package.py`).
3. **Stage 2 must be segmented.** A pool hold at 72 h transfers nothing. The
   locked $t=0\to20$ runs at 6 qubits already needed screen/continuation
   staging; at 8--10 qubits with 16$\times$ the state space, whole-horizon jobs
   will not exit normally. Stage 2 jobs must be bounded time segments that
   terminate on their own, in the pattern of `SIAM_NB4_SCREEN_SEGMENT` /
   `SIAM_NB4_CONTINUATION_SEGMENT`.

### Separation-rule note

These are production runs. Their configuration is fixed by the manifest before
launch and is not adjusted after seeing results. Regime choice must be made now,
not after the first returns.

### Corrections after execution (2026-09-04, later session)

**Negative result first: the section above rests on premises that the CHTC
mirror and the source contradict.** Full evidence, measurements, and the
executable plan are in
`chtc/paper_ii_regime_seeds_v1/HANDOFF_MONITOR_REGIME_SEEDS_20260904.md`
(branch `fable/paper-ii-regime-seeds-chtc-20260904`; code in commit `48336a01`, the handoff in the commit that follows it).

| Claim above | Measured |
|---|---|
| Bundle never submitted | Cluster 9674800 ran 2026-08-22: nph=3 procs exited 0 after 13–25 h with **zero seed artifacts** (no driven-snapshot fit reached the 1e-6 gate; best 5.5e-4); nph=7 procs held at 24.3–24.5 GB, still held 2026-08-31. Cluster 9663698 (2026-08-18, same builder): zero artifacts. |
| Stage 2 uses the prospective packagers | They admit only the frozen matrix `paper_ii_parallel_benchmarks_20260828` (8 seeds / 39 references, digest-pinned, `n_ph_max_7` excluded, no `(U=4, g=0.5)` cell). Its six HH seeds already exist (cluster 9714490–95, ΔE ≤ 5.2e-8, in the rescue mirror). |
| Image just needs building | `image.def` was absent from git (restored); the pinned image `fa5c4ea8…` is installed from the mirror and bound in `input/runtime_image_identity.json`; the old remote dir holds the 2026-05-17 image. |

What replaced the plan: stage 1 now builds every row with the frozen
three-complete-layer VQE (the matrix seeds' protocol; the frozen worker's
missing executor gradient was ported from 267516ea). Local smokes:
anchor `(U, g, n_ph) = (4, 0.5, 1)` ΔE 1.7e-11 in 8 s; `intermediate_weak`
(the row the search could not seed in 14.6 h) ΔE 6.2e-8 in 190 s; the nph=7
row exceeded 8 GB before finishing. Stage 1.5 (new, local) builds each
regime's six exact references on the locked grid and registers its
exact-response scales (22 s at 6 qubits; the anchor's scales match the
calibration sidecar to four digits). Stage 2 is the staged screen/promotion
machinery parameterized by regime and by the arm that realizes APM
(`--regime R --apm-arm exchange_q_compiled`), t5 screen → t20 promotion under
the locked response-normalized matched-pair policy, not the packagers.

Settled 2026-09-05 (user): the survey keeps the matched-pair qualification
(D2), and the `report_refresh_required` block does not gate this campaign
(Q3). Measured the same day: the calibration seed's `dv = 0.25` is a uniform
site potential, a constant `-dv·N_e` at fixed filling, so the anchor at
`dv = 0` is the same dynamics; the frozen worker cannot seed `dv ≠ 0` (its VQE
objective and exact qualifier disagree on that term's sign, ΔE = 2·dv·N_e).

Open for the user (recommendations in the handoff §2): D1 which rows
(recommend all seven seeds, dynamics on the four 6–8-qubit rows first, nph=7
deferred); D3 which lineage runs stage 2, because the results PDF's route
(`Gamma(v*)` realized = `exchange_q_compiled`) exists only on
`codex/paper-ii-exact-estimator-telemetry-20260903` (recommend cutting the
survey branch from it and cherry-picking this branch's commits). Submission
also waits on the SSH master and the words "submit stage 1".

### SIAM moved onto the compiled-Q route (2026-09-05)

User correction: the SIAM results above ran on the older exchange rule
without the exact-estimator accounting; one more AVQDS run would not have
completed a current-method comparison. Done on
`fable/paper-ii-regime-survey-20260905` (commits fc2de280 … 67f3a05e):

- engine: SIAM starts from an empty ansatz, so the compiled-cost runtime read
  `nq = None` and every compiled-Q bundle died at the first checkpoint; fixed
  via `state_num_qubits`. `Arm.dynamics_policy` separates the runner policy
  (`exchange`/`avqds`) from the arm id; `exchange_q_compiled` is registered
  for the Nb=2 proof pair and the Nb=4 ladder.
- identity: every SIAM score (schema v4), ladder action id (route token
  `qc`), stitch, binding and estimator record carries the route bundle;
  `classify_siam_matched_pair(..., expected_apm_route=...)` refuses a
  cross-route pair.
- estimator accounting: `siam_estimator_work.py` gives
  `logical_total_estimator_work` with exact status for any registered arm and
  checks the run lock's route bundle.
- packager: `prepare_siam_nb4_chtc_action(route_id=...)` emits v5 documents
  for the compiled-Q lineage; the historical lineage's v4 bytes are frozen
  by test.
- evidence: Nb=2 compiled-Q pair `qualified_pair`, ratio 1.018, estimator
  work AVQDS 967,888 vs compiled-Q 928,409 (`chtc/siam_nb2_proof_qc_v1/`).
  Nb=4 r00 screen packages for both arms prepared, not submitted; the ladder
  procedure is §8 of the CHTC handoff.

## STOP: the two Paper II sources diverged (2026-09-04)

Landing the "notation revision" is **not a copy**. Both candidate sources branch
from committed `054451d` and were edited independently:

| source | adds over `054451d` | unique content lost if overwritten |
|---|---:|---:|
| `AP-McLachlan---Paper-II/…/time_dynamics_paper_II.tex` (repo, uncommitted) | 663 lines | **144** |
| `Documents/Codex/2026-09-01/l-2-relates-to-eq-9/outputs/…tex` (newest, ungit) | 589 lines | **151** |

The 144 figure survives normalizing $L^2\to m(\dot\theta)$ and
$\tau_{L^2}\to\tau_m$, so it is not a rename artifact — they are whole rewritten
paragraphs. Neither source is a superset.

They edit the **same sections in the same proportions**, which is the signature
of two agents rewriting the same passages in parallel:

| section | A-only lines | B-only lines |
|---|---:|---:|
| AP-McLachlan: joint support-patch dynamics | 41 | 48 |
| Generalized exchange support patch | 35 | 43 |
| Structural and numerical failure | 18 | 15 |
| Geometry-aware fixed-support time-step control | 14 | 7 |
| Time-point geometry | 12 | 11 |
| Operating decision rule / time-point algorithm | 12 | 12 |
| all others | 14 | 15 |

This is exactly the collision `PAPER_II_METRIC_WHITENED_AVQDS_EDITOR_HANDOFF_20260903.md`
warned about: it named both worktrees, selected neither, and authorized no
synchronization between them. **Choosing a base is a user decision; overwriting
either direction destroys ~150 lines of drafted prose.**

Supersedes the earlier recommendation in this file to "land the
$m(\dot\theta)$ rename" — that was written before the divergence was measured.

## Queued main-results changes (user-directed 2026-09-04)

These are generator changes and apply to whichever base wins:

1. Drop the `combined` controller (`state_motion_1e-3_plus_parameter_5e-5`)
   from the main results. Table I 11 rows → 8, matched pairs 8 → 6.
2. Table columns become $N_{2q}$, $D_{2q}$, $D_c$, $\mathcal E$ — Paper I's
   presentation — **plus the internal time-step count** $N_{\rm step}$.
3. **No fidelity.** Report total energy, doublon, and charge-imbalance
   $\Delta n$ errors instead of $\overline{1-F}$.
4. **Retire "$\Gamma$ route" naming.** It is the only route; call it APM.

Measured effect of (1) on the surviving claims:

| quantity | with `combined` | without |
|---|---|---|
| completed cells | 11 | 8 |
| matched pairs | 8 | 6 |
| $N_{2q}$ reduction range | 29.6--95.4% | 29.6--95.4% (unchanged) |
| mean infidelity range | $1.35\times10^{-6}$--$2.56\times10^{-5}$ | $1.35\times10^{-6}$--$8.03\times10^{-6}$ |
| terminal $N_\theta$ | 44--50 | 44--50 (unchanged) |

Dropping `combined` costs two matched pairs and **improves** the reported
infidelity ceiling by a factor of 3.2, because $2.56\times10^{-5}$ was the
strong--fast `combined` cell. The headline reduction range is untouched.

Note on (2): Paper I's $\mathcal E$ is the *total* logical scalar-estimator
count. Paper II's $W_{\rm geom}$ is propagation-only. To print $\mathcal E$
with Paper I's meaning, use `logical_total_estimator_work`, not $W_{\rm geom}$,
and carry its per-cell `exact` / `lower_bound` status.

Generator: `pipelines/time_dynamics/diagnostics/q_compiled_manuscript_assets.py`,
present only in worktree `codex-paper-ii-dual-selector-overnight-20260903`.
Tables are generated; do not hand-edit them.

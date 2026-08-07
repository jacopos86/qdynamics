# Paper I — RA-vs-ADAPT framing + ΔE-vs-cost figure spec

Created 2026-07-29. Two deliverables requested: (A) a ΔE-vs-cost (Pareto) figure spec,
(B) reframed Results prose. **No run launched; `Paper_I.tex` not edited.**

Data source: `output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/`
(partial progress, **32/48 validated, 16 pending**). Validated (`complete`) curves:
RA-none, RA-plateau, Append-ADAPT. **RA-always is `pending` in every regime** — do not headline it.

Regime codes: WW/IW/SW = weak/intermediate/strong Hubbard × weak Holstein;
WS/IS/SS = ... × strong Holstein.

---

## ⚠️ Reconcile before drafting final numbers

The current Results prose in `Paper_I.tex` (`~:1140`) says single-Pauli plateau-insertion
RA-ADAPT is **"lower in all six regimes."** The evolving-run table (page 2) shows
**Append-ADAPT more accurate in strong–strong** (ADAPT 1.16e-8 vs RA-plateau 4.07e-8) and a
near-tie in strong–weak. The paper text likely predates the current provisional RA snapshot
(see the `COMPONENT_EVIDENCE_UPDATE(20260729)` note at `~:1090`: Append is validated v6, RA is
provisional). **Do not assert "all six"** until the paper text and the evolving run agree.
The robust, snapshot-independent claim is weak/intermediate-Hubbard dominance with the
strong-Hubbard regimes as the close/hard cases.

---

## A. ΔE-vs-cost (Pareto / cost-to-target) figure spec

**Why.** Error-vs-iteration hides RA's joint win (lower error *and* lower cost) and lets both
methods "converge together" on a 2-site system at k=50. Plotting error against *cumulative
resource* shows the resource-aware advantage directly and gives a clean cost-to-target readout.

**Primary figure — estimator-work front.**
- y: same-cutoff `|ΔE_k|` (log). x: **cumulative logical estimator work `S` through iteration k** (log).
- Layout: 2×3 regime grid (WW/IW/SW; WS/IS/SS), matching the existing figures. One figure per
  representation: `macro` (undecomposed) and `singleton` (single-Pauli-word).
- Curves per panel (reuse existing colors): Append-ADAPT (blue), RA-none (light red),
  RA-plateau (red), RA-always (dark red, **dashed + "pending" until validated**).
- Plot each trajectory two ways: raw `(cumulative S, |ΔE_k|)` connected by k (faint), plus a
  **best-so-far envelope** `min_{j≤k}|ΔE_j|` (bold). The envelope makes "estimator work to reach
  accuracy ε" single-valued and unambiguous.
- **Target line:** horizontal rule at a stated accuracy target (use the paper's two-site scale,
  `2×10⁻⁴|t|` total / `10⁻⁴|t|` per site, `~:1007`). Cost-to-target = x-value where each
  envelope crosses the line. Report per method as "estimator queries to reach ε."
- Read: a method **dominates** where its envelope is below-and-left of another's.

**Companion figure — circuit front.** Same design, x = **cumulative two-qubit gate count
`N_2q`** (and optionally `D_c`). This is where undecomposed always-insertion looks best
(−16–36% circuit resources) even though it loses on `S`. Showing both fronts is the honest
representation-tradeoff picture.

**Data.** No new run. Per-iteration `|ΔE_k|`, cumulative `S` (Appendix E accounting is already a
sum over rounds), and per-k transpiled `N_2q`/`D_c` all exist in the evolving-run trajectories.
This is a **replot of existing curves with cost on x**, not new evidence.

**Caveats to encode in the caption/build.** Validated curves solid; pending dashed. Log–log.
Cumulative cost is monotone in k, so envelopes are monotone decreasing in error. State the
target ε and that cost-to-target is read off the envelope.

**Headline readout this figure should produce (validated rows, single-Pauli, primary front):**
RA reaches the two-site target at a fraction of Append-ADAPT's estimator work in WW/IW/WS/IS;
aggregate single-Pauli plateau `S` is **1.42×10⁶ vs 4.34×10⁶ queries (−67%)**.

---

## B. Reframed Results prose (draft — validated-first, accuracy-forward)

Style: manuscript-editor rules (no `not X but Y` frames; field-native; scope explicit;
"same-cutoff energy error"; separate accuracy from resources). Keep current names
`RA-ADAPT`/`Append-ADAPT` — the `RA`/`ADAPT` rename (worklist C6) is not yet approved.
Numbers are from the evolving run (provisional); bracketed items await validation.

### B1. Single-Pauli-word headline (validated, strongest)
> In the single-Pauli-word representation, plateau-insertion RA-ADAPT attains lower same-cutoff
> energy error than Append-ADAPT throughout the weak- and intermediate-Hubbard regimes, by at
> least an order of magnitude in weak–weak, intermediate–weak, weak–strong, and
> intermediate–strong. It reaches these accuracies at lower estimator cost: aggregated over the
> six regimes at the fixed 50-iteration horizon, it uses 1.42×10⁶ logical estimator queries
> against 4.34×10⁶ for Append-ADAPT (67% fewer), with two-qubit gate count, two-qubit depth, and
> total depth 4–5% lower and pretranspilation one-qubit work 2% higher. The strong-Hubbard
> regimes are the close cases, where Append-ADAPT remains competitive [strong–strong pending
> reconciliation, see file header].

### B2. Undecomposed headline (accuracy; PENDING validation, higher S)
> [PENDING — RA-always rows unvalidated (16/48).] In the undecomposed-generator representation,
> always-insertion RA-ADAPT attains lower or tied terminal same-cutoff error in five of six
> regimes and at least an order of magnitude lower error in intermediate–weak and weak–strong.
> Its two-qubit circuit resources are lower in aggregate (29% fewer two-qubit gates, 25% lower
> two-qubit depth, 16% lower total depth), while its logical estimator work is higher
> (1.27×10⁷ vs 4.78×10⁶ queries), reflecting the cost of scoring every insertion position.

### B3. Mechanism (insertion as the strong-correlation lever)
> The representation dependence follows from candidate granularity. Multi-term undecomposed
> generators carry structured blocks whose ordering constrains the reachable ansatz, so scoring
> all insertion positions materially improves the trajectory. Single-Pauli-word generators are
> finer-grained; positional search becomes valuable once append-only growth stalls, and
> plateau-insertion captures most of the available improvement while opening insertion positions
> only after a stall.

### B4. Cost framing (cite the new figure)
> Because RA-ADAPT ranks candidates by predicted energy decrease per resource, its advantage is
> a joint accuracy-and-resource statement. Figure~\ref{fig:hh_error_vs_cost} plots same-cutoff
> error against cumulative logical estimator work; single-Pauli-word RA-ADAPT reaches a given
> accuracy at lower estimator cost than Append-ADAPT in the weak- and intermediate-Hubbard
> regimes.

### B5. Scope + hard case
> These comparisons are noiseless, use two-site models, and terminate at a fixed 50-iteration
> horizon. Strong electronic correlation (strong–weak, strong–strong) is the open case; behavior
> past the 50-iteration horizon there is left to future work.

### B6. Abstract adjustment
The abstract "five of six / three of six / at least one order of magnitude" counts (`~:110`)
must be **recomputed from validated rows**. As written they may reflect the stale/pending
snapshot (esp. strong–strong). Recommend restating as weak/intermediate dominance with the
strong-Hubbard regimes named as the hard case, matching B1/B5.

---

## Open items (not done here)
- Extended-run (>50) plan — deferred per your selection; the diagnostic payoff is concentrated
  in SW/SS, and dominance there must be judged on the per-cost front (Part A), not per-iteration.
- Strong–strong accuracy reconciliation (file header) — blocks the "all six" claim.
- RA-always validation (16/48 pending) — blocks the undecomposed accuracy headline (B2).

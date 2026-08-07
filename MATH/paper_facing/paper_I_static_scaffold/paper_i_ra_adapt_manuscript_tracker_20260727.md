# Paper I (RA-ADAPT) — manuscript issue tracker

Created 2026-07-27. Tracks open manuscript-review and formalization issues for the
active paper `MATH/paper_details/Paper_I.tex` (RA-ADAPT).

Supersedes `paper_i_completion_tracker_20260717.md` **for manuscript-issue tracking only** —
that file is SNAKE-era (0 RA-ADAPT / 31 SNAKE mentions) and remains the SNAKE run/deliverable log.

Status legend: **DONE** · **PENDING** · **DEFERRED** · **HANDOFF(codex)**

---

## Resolved (committed `4f078f4` / `a42de3b`)
- **DONE** Naming collision `append-only RA-ADAPT` → **`no-insertion RA-ADAPT`** (triad with plateau-/always-insertion).
- **DONE** `retainment` → `retention`; eq label `snake` → `ra_adapt`.
- **DONE** Declutter: 11 inline provenance blocks (612 lines) → `provenance/Paper_I_inline_comment_archive_20260727.md` + pointers (3256→2655 lines).
- **DONE** Routing: `manuscript-editor` skill tracked and pointed at `Paper_I.tex`/`Paper_I.pdf`.

## Citation structure
- **DONE** Replaced the literal `(CITE)` placeholder with the current fermionic,
  vibrational, and bosonic ADAPT application citations.
- **DONE** Current source has no uncited rendered `\bibitem`, undefined citation,
  undefined cross-reference, or duplicate label.
- **PENDING** External source-by-source claim audit remains separate; structural
  citation completeness does not establish that each source supports every
  attached claim.

## Deferred (user)
- **DEFERRED** Which RA-ADAPT variant the headline reduction numbers describe (plateau-insertion vs always-enabled).
- **DEFERRED** Whether the new runs use unweighted Phase-I ranking
  \(S^{(1)}=\Delta E_1\) or resource-weighted ranking
  \(S^{(1)}=\Delta E_1/K_1\). Align the method definition, schematic, Results,
  configuration table, and Algorithm 1 only after the run protocol is fixed.
- **DEFERRED** General stopping-rule formalization. The displayed trajectories
  currently use the terminal \(k=50\) state; do not replace this with a
  plateau/patience or exact-reference stop until the next run protocol is fixed.

---

## Required three-axis Results interpretation

The final Results and abstract must distinguish the following three empirical
comparisons. They answer different questions and require separate statistics.

1. **Cheaper paths at common iteration and accuracy.** At a shared adaptive
   iteration \(k\), compare endpoints for which RA-ADAPT and Append-ADAPT attain
   the same or a prespecified comparable same-cutoff energy error. The claim is
   that RA-ADAPT reaches that common accuracy through a cheaper path. Report
   endpoint circuit resources \(N_{2q}\), \(D_{2q}\), \(D_c\), and \(W_{1q}\)
   separately from cumulative logical estimator work \(S\).
2. **Lower energy in fewer adaptive iterations.** For a common error target
   \(\varepsilon\), compare the first hitting iterations
   \(k_{\rm hit}(\varepsilon)=\min\{k:\lvert\Delta E_k\rvert\le\varepsilon\}\).
   The claim is that RA-ADAPT generally reaches shared accuracy targets earlier
   than Append-ADAPT.
3. **Lower energy at comparable resource cost.** For a common resource budget,
   compare the lowest same-cutoff energy error attained without exceeding that
   budget. The claim is that RA-ADAPT generally reaches lower energy than
   Append-ADAPT at similar cost. Construct this comparison separately for each
   circuit metric and for \(S\); these resource coordinates are not
   interchangeable.

---

## Insertion-policy formalization

Three distinct notions remain separate:

- **insertion activation**: online policy selecting the append-only or
  commutation-reduced domain;
- **execution horizon/stopping**: how an adaptive trajectory terminates;
- **reporting point**: the post-run iteration marked and compared in a figure.

### Manuscript synchronization

- **DONE** \(\mathcal P_k(A)\) is candidate dependent and
  \(\mathcal R_k=\bigcup_{A\in\mathcal G}\{A\}\times\mathcal P_k(A)\).
- **DONE** Always-insertion uses
  \(\mathcal P_k^{\rm all}(A)=\overline{\mathcal P}_k(A)\), not the raw cut set
  \(\mathcal C_k\).
- **DONE** Plateau-insertion opens the same reduced domain after a
  sub-\(\epsilon_{\rm ins}\) accepted decrease and returns to append-only after
  a super-threshold decrease.
- **DONE** The commutation equality is scoped to the retained numerical
  generator representations defined by the \(10^{-12}\) coefficient cutoff.
- **PRESERVED** Pauli children inherit the representative position retained
  with their Phase-I parent record; no child-specific position reduction is
  recomputed after splitting. This remains the manuscript contract unless the
  implementation and replacement-run protocol establish a different rule.
- **DONE** Algorithm 1 constructs the candidate-indexed position union and
  preserves the parent position through `REPRESENT`.
- **DONE** Symmetry-retained children are identified as the Phase-II population;
  only the Phase-II shortlist proceeds to Phase III.
- **DONE** The benchmark-configuration position row lists no-, always-, and
  plateau-insertion rather than “appended position.”

### Required implementation synchronization

- **PENDING** Always-insertion must enumerate
  \(\overline{\mathcal P}_k(A)\) for each candidate, not every raw cut.
- **PENDING** Verify that the result-producing implementation preserves each
  retained parent position through parent-to-child representation. Do not add
  child-specific requotienting without a separate controlled comparison.
- **PENDING** Apply the \(10^{-12}\) retained-component convention consistently
  to both the commutation certificate and the generator representation whose
  ordering is treated as equivalent.
- **PENDING** Add regression cases for: fully commuting adjacent blocks;
  a noncommuting block that creates a new class; parent-position inheritance
  through Pauli splitting; missing/empty algebraic metadata failing closed; and
  equality of the retained circuits at equivalent cuts.

### Run- and display-dependent work

- **PENDING** Replace the provisional six-regime overlays after the planned
  source-matched RA-ADAPT and ADAPT reruns.
- **PENDING** Use a visible marker at each reported point, decode every marker
  in the caption, and name in Results which insertion policy supplies each
  aggregate resource comparison.
- **PENDING** Replace run-label prose in regenerated plot titles with
  reader-facing representation labels, e.g. “Undecomposed generators” and
  “Single-Pauli-word generators.”
- **PENDING** Run the missing matched Geo-ADAPT comparisons before retaining
  the five-of-six Geo-ADAPT statement.
- **PENDING** Preserve the terminal-\(k=50\) display convention for the current
  draft; reconsider the reporting-point prose only after the new run protocol
  is fixed. No additional summary table is required when the plotted markers
  and costs remain legible.

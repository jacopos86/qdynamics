# Paper II enrichment digest — seven GPT threads, read 2026-09-07

Source material for enriching `AP-McLachlan---Paper-II`. Nothing here is evidence.
Every item carries a status against the repository's own receipts:

- **VERIFIED** — checked against run records, code, or the generated tables today.
- **CONTRADICTED** — the receipts say otherwise; do not use as written.
- **UNCHECKED** — plausible, not tested here; check before it reaches the manuscript.
- **PROPOSAL** — a design or framing suggestion, not a factual claim.

## Provenance

| # | Thread | Date | What it is good for |
|---|---|---|---|
| 1 | [Combinatorial Optimal Support Exchange](https://chatgpt.com/g/g-p-6a0efee07f7c8191a7928e98c5bb7abd/c/6a4ae977-7e50-83ea-90bd-ed32c2313c29) (project SNAKE) | Jul 5 | Deep-research brief on the exchange problem. **Report body not captured** (see Gap below) |
| 2 | [Schur novelty and videos](https://chatgpt.com/c/6a4b0f1a-7f78-83ea-a6af-ab3893071eab) | Jul 5 | Naming and exposition of the conditional objects; FWL framing |
| 3 | [State operation request](https://chatgpt.com/c/6a7f9dcd-88b0-83ea-8eb2-912e7bc15199) | Aug 14 | Classical-vs-quantum cost boundary; NP-hardness localization |
| 4 | [Equation Analysis Clarification](https://chatgpt.com/c/6a4a82d3-1f1c-83ea-a96a-d46b04a37ce0) | Jul 5 | One global patch objective; macro-parent scouting as frontier-only layer |
| 5 | [Deep Research in Sparse Optimization](https://chatgpt.com/g/g-p-69c006a8d6e481918b300dd25f65a547/c/6a4aed43-3634-83ea-b5fe-557ce26df30e) | Jul 5 | Related-work map |
| 6 | [Response Evaluation](https://chatgpt.com/c/6a4aee42-4c70-83ea-a7c3-2db2ecea19bf) | Jul 5 | Counterexamples and corrections to the exchange formalism |
| 7 | [Momentum-space operator notation](https://chatgpt.com/c/6a80b3f3-b09c-83ea-a36d-b3d1b54b87cf) | Aug 15 | Measurement-adaptive architecture; scalar-entry accounting; spec review |

Raw captures: `/Users/lordskyler/gpt_paper_ii_chats/`.

**Gap.** Thread 1's deep-research report (24,502 characters, titled *Combinatorial-Optimal
Support Exchange for Variational Least Squares*) renders inside a cross-origin sandbox
frame; page text, accessibility tree, React state, and the conversation endpoint all
refuse it. One click on that card's download icon puts it on disk and I can fold it in.
Thread 5 says the project had no formal deep-research run, which thread 1 shows is wrong;
the report exists.

---

## 1. Measurement accounting — the strongest corroboration

Thread 7 derives, independently of our audit, the same rule the manuscript adopted today.

> Beyond the baseline `G_JJ`, `f_J`, and `‖b̄‖²`, all one-add exchanges require only
> `{G_Ja, G_aa, f_a}` for `a ∈ C`. At the scalar-entry level this is
> `M_column = j·u + 2u`. There is no `G_ab` measurement for `a ≠ b`. The same measured
> column for `a` evaluates its exchange with every deletion set `D`.

**VERIFIED.** This is exactly the reuse-adjusted charge now in Table I, `m_r(p_r + 2)`
per selector call. Our audit found the implementation was paying `p·m + m(m+1)/2 + m`,
and that the `m(m−1)/2` candidate–candidate block is 96.8–97.0% of the executed candidate
charge in all eleven cells while nothing reads its off-diagonals at insertion batch size
one. Effect: APM's count falls 5.5–91.7%, the comparator's by at most 1.1%.

Thread 7's tiers, all **VERIFIED** as consistent with our ledger:

| Acquisition | Scalar entries | Our status |
|---|---|---|
| Deletion scouting | 0 beyond propagation | Matches: deletions are row/column selections of measured geometry |
| One-add exchange, whole pool | `j·u + 2u` | The adopted reuse-adjusted charge |
| Candidate interactions inside a frontier of `m` | `+ m(m−1)/2` | The executed charge we removed; needed only for `|A| ≥ 2` |
| Full block oracle | `j·u + 2u + u(u−1)/2` | Never run |

Why the off-diagonals are not merely wasteful but genuinely required for batches, with the
worked two-candidate case (thread 7):

> `Δ({a,b}|R) = (r₁² − 2ρ r₁ r₂ + r₂²)/(1 − ρ²)` … the individual data determine `r₁` and
> `r₂`, but the unmeasured candidate–candidate overlap `ρ` can materially alter the joint
> gain. Therefore exact multi-add selection genuinely requires interaction measurements;
> no classical optimizer can manufacture them.

**Use this.** It is the cleanest justification for why the appendix's reuse-adjusted count
is the right charge at `|A| ≤ 1` and would not be at `|A| ≥ 2`. Two sentences in the
appendix, or a footnote where batch size one is declared.

## 2. Classical hardness versus quantum cost — a distinction the paper still blurs

Thread 3 is the sharpest statement of something the manuscript asserts only in passing.

> The NP-hardness of `max_{D⊆J_k, A⊆P_k} U_k^exch(D,A)` is, first and foremost, a
> **classical combinatorial post-processing hardness**. … Evaluating one million support
> subsets does not require one million new quantum experiments when all relevant entries
> of `K_k` and `f_k` have already been measured.

Its four resource cases (**UNCHECKED** as a taxonomy, though each matches our code):

1. **Deletion scouting** — free; the active-support Gram is already required for propagation.
2. **Append scouting** — linear in the tested candidates.
3. **Batched append and true exchange** — needs candidate–candidate blocks; largest upfront burden.
4. **Commit certification** — state-dependent: refit, ray distance, patched velocity. Cannot
   be recovered from the original `K_k`, `f_k`.

Its proposed replacement for our complexity sentence, worth adapting nearly verbatim:

> Equation (64) is NP-hard as a classical support-selection problem **conditional on a fixed
> checkpoint measurement record**. Quantum estimator cost is governed separately by the
> tangent, force, candidate-cross, refit, and continuity quantities acquired to construct and
> certify that record.

And the warning that our own latency paragraph half-makes:

> "hardware cost" should not be represented by one undifferentiated scalar … the defensible
> resource axes are separately `T_classical`, `N_measured observables`, `N_shots`,
> `N_circuit executions`, `D_2Q` and `N_2Q`.

**PROPOSAL, recommended.** Sec. `comparison_protocol` already separates decision latency from
estimator count; this makes the separation principled instead of anecdotal. It also
strengthens case 4, which is precisely why our certification charge is per-materialized-finalist.

## 3. Corrections to the formalism — check these against our equations

From thread 6, an adversarial read of the exchange write-up. All **UNCHECKED** against the
current manuscript text; each is a concrete, testable claim.

**Do not conflate regularizers under one symbol.** Ridge `Γ_λ(S) = f_Sᵀ(K_SS + λI)⁻¹f_S` and a
hard spectral threshold `Γ_τ(S) = f_Sᵀ V_S diag(μ_i⁻¹ 1_{μ_i>τ}) V_Sᵀ f_S` are not the same
object and a threshold is "not generally representable as adding a principal subblock".
Schur update formulas, monotonicity, and rank-change behaviour differ. **Our appendix uses
ridge 1e-6 and cutoff 1e-10 together**, so this is directly relevant to the
retained-support/damped-inversion subsection.

**Monotonicity fails under a raw pseudoinverse.** Counterexample:

```
K = [[1,1],[1,1]],  f = (1,0)ᵀ   ⇒   Γ({1}) = 1,   Γ({1,2}) = fᵀK†f = 1/4
```

so adding a coordinate can *lower* the captured drift. Our Eq. (ideal_exchange_components)
asserts non-negativity of the Schur quadratic form, which holds under the range condition we
state; the manuscript should say which assumption carries it (`K = X̄ᵀX̄` by construction,
or the ridge), because we then write "adding a measured block to a support cannot lower the
captured drift" without qualification.

**Γ is neither submodular nor supermodular.** Counterexample:

```
K = [[1,ρ],[ρ,1]],  f = (1,1)ᵀ  ⇒  Γ({1}) = Γ({2}) = 1,  Γ({1,2}) = 2/(1+ρ)
```

diminishing returns for `ρ > 0`, increasing returns for `ρ < 0`. Any approximation guarantee
needs weak-submodularity or restricted-eigenvalue assumptions. **We claim no guarantee**, so
this is a citation-worthy caution rather than a defect.

**The NP-hardness reduction needs its identification stated.** It is clean only with
`K = XᵀX`, `f = Xᵀy`, so that `Γ(S)` is the explained sum of squares of best-subset
regression; and the one-line reduction requires `D = ∅` to be admissible. If exact swaps
`|A| = |D|` were enforced, a different construction is needed.

**Cost sign discipline.** Thread 6 recommends an explicit resource functional rather than an
ambiguous `cost(D,A)`:

```
U(D,A) = Γ_ρ(S(D,A)) − μ C(S(D,A)) − η T(D,A) − [Γ_ρ(J) − μ C(J)]
```

"Otherwise deletion may be accidentally penalized when it should receive a resource credit."
**Relevant to our shifted score**: `S(P) = (ΔΓ + τ)/K(P) − τ` multiplies by a population-relative
cost factor rather than subtracting a resource change, so relief enters as `K < 1`. Worth one
sentence in Appendix `compiled_cost_realization` distinguishing the two conventions.

## 4. Search architecture — one real gap in our method

Thread 7's central recommendation is a change we did **not** make, and it is the strongest
candidate for future work or an explicit limitation.

> It is stronger than the paper's separate append and prune scout construction because an
> addition is ranked by its best conditional exchange value
> `s_exch(a) = max_{D∈D} Φ_J(D,{a})`, rather than by its pure-append value `Φ_J(∅,{a})`.
> Thus an atom that becomes useful only after a stale direction is deleted is retained
> automatically. This directly repairs the delete–add screening failure **at no additional
> measurement cost**.

**VERIFIED as a gap.** Our insertion frontier `W = 8` is sealed from the already-scored
`d = 0` family, i.e. from pure-append scores (`exchange_structural.py`, "Sealed HERE … from
scores the d = 0 family has already paid for"). Thread 7's point is that the same measured
columns support the conditional ranking, so the stronger rule costs nothing extra in
measurement, only classical work. This is exactly the failure mode our campaign hit as
"true exchange never enumerated" before the frontier repair.

Two options, both honest:
- **Limitation sentence** in Sec. `bounded search`: the frontier is ranked by pure-append
  score; ranking by best conditional exchange value uses the same measured data and is left
  to future work.
- **Experiment**: re-rank the frontier by `s_exch(a)` on one or two cells and report whether
  the committed patches change. Cheap; no new measurement model needed.

Also from thread 7, **PROPOSALS** not in the paper:

- **Nested interaction-frontier ladder, not a cardinality ladder.** Once `G_{F_ℓ F_ℓ}` is
  acquired, every subset up to the cap is classically evaluable, so stopping after an
  accepted pair "saves no quantum acquisition". Our `rounds_per_checkpoint` loop is a
  cardinality-flavoured ladder; if we keep it, thread 7 says to declare it as a lexicographic
  policy ("prefer the smallest append cardinality that passes") and not call it maximization.
- **Policy-relative exactness language.** Three labels worth borrowing directly, because our
  Sec. `bounded search` currently says "complete" in a way the audit already had to qualify:
  `zero_one_add_policy_exact`, `restricted_frontier_policy_exact`, `declared_caps_policy_exact`.
  "This avoids conflating complete enumeration under the implemented policy with global
  maximization of the unweighted geometric utility."
- **Confidence-driven shot allocation.** Certify a patch as best when
  `Φ̂_B − β_B > max_{B'≠B}(Φ̂_{B'} + β_{B'})` under simultaneous confidence radii, and spend
  shots only on observables separating unresolved competitors. Natural finite-shot successor
  to our noiseless study; connects to best-arm elimination and variance-based ADAPT shot
  allocation.
- **Full-block oracle on small instances** to measure how often column-first screening misses
  an interaction-only multi-add patch, and how large the regret is. We have the instances.

## 5. One global objective, and macro scouting as a frontier layer

Thread 4 states the objective the manuscript should be seen to optimize:

> Swap should not be defined by an append score plus a prune score. It should be defined by
> one global support-patch objective over the final patched support
> `J_k^B = (J_k \ I_{B,k}^⊖) ∪ I_{B,k}^⊕` … `Φ_k(S) := Γ_k^K(S) − π_k(S)`.

**VERIFIED as already ours.** This is our complete-patch population with the shifted compiled
score. Useful as independent confirmation that the paper's framing is the defensible one;
the phrase "the combinatorial objective nominates candidates, and the jointly recomputed
patched support authorizes the edit" (thread 3) is a crisper statement of our
nomination-versus-authorization separation than the manuscript's own.

Thread 4 also proposes a **macro-parent scouting layer**: group children by logical generator,
scout at parent level to decide *where to spend child-level measurement*, then run the real
child-level patch optimization over the union of selected parents' children. **PROPOSAL,
untried.** Relevant because our pool is already logical generators expanding into several
Pauli coordinates, and every materialized patch in the campaign inserted exactly one
coordinate.

## 6. Naming and exposition

Thread 2 proposes naming the conditional objects and giving three readings of the same algebra:

```
G_{X|B} := G_XX − G_XB G_BB⁻¹ G_BX        f_{X|B} := f_X − G_XB G_BB⁻¹ f_B
Γ_P(v*_P) − Γ_C(v*_C) = f_{A|R}ᵀ G_{A|R}⁻¹ f_{A|R} − f_{D|R}ᵀ G_{D|R}⁻¹ f_{D|R}
```

- **residual**: `T̄_X − T̄_B G_BB⁻¹ G_BX` is what remains of `X` after projection;
- **Schur complement**: names the elimination that produces `G_{X|B}`;
- **quotient**: the space `V_{B∪X}/V_B`.

**PROPOSAL.** Our Sec. `structural_support` writes these blocks out but names only the Schur
complement. Adopting `f_{A|R}`, `G_{A|R}` would shorten Eqs. (18)–(22) noticeably and make
the gain/loss symmetry visible at a glance. The Frisch–Waugh–Lovell reading is already cited
in the manuscript's bibliography, so the connection costs nothing to state.

## 7. Related-work map (thread 5)

Nearest fields, for the introduction or a related-work paragraph:

- **Best subset selection / sparse least squares** — `Γ(S) = f_Sᵀ K_SS⁻¹ f_S` with `K = XᵀX`
  is classical best-subset regression; decades of branch-and-bound, MIO, cutting planes.
- **k-exchange local search theory** — guarantees for some objective classes, not arbitrary quadratics.
- **Submodular optimization** — does not apply directly (see the counterexample above), but the
  exchange inequalities and proof techniques do.
- **Experimental design** — "probably the mathematically richest neighboring field":
  `log det(K_SS)`, `tr(K_SS⁻¹)`, `λ_min(K_SS)`, Fedorov-type exchange, convex relaxations, duality.
- **Numerical linear algebra** — Woodbury, leverage scores, CUR, column subset selection,
  rank-revealing QR, pivoted Cholesky: the algebra of how the objective moves under swaps.
- **Sparse approximation** — OMPR, CoSaMP, Subspace Pursuit, FoBa as heuristic exchange methods.
- **MIQP / binary quadratic optimization** — relaxations, SDP bounds, branch-and-cut.

Its statement of what is new here, **UNCHECKED** as a novelty claim but a good skeleton for
the introduction: the objective is local in time, built on a variational McLachlan functional,
constrained to edits of an existing support, coupled in append and prune simultaneously,
augmented by hardware cost and conditioning penalties, and run online during dynamics.

## 8. What to do with this

Ordered by value per unit effort.

1. **Appendix, estimator accounting.** Add the batch-size justification from §1 (the `ρ`
   two-candidate formula) and the classical-versus-quantum sentence from §2. Both support
   text already in the paper.
2. **Sec. bounded search.** Add the conditional-exchange frontier as a stated limitation (§4),
   and consider the policy-relative exactness wording.
3. **Appendix, technical realization.** Say which assumption carries the non-negativity of the
   Schur gain, and distinguish ridge from spectral threshold (§3).
4. **Notation.** Consider `G_{A|R}`, `f_{A|R}` (§6). Manuscript-wide, so decide before more edits.
5. **Related work.** One paragraph from §7 if a referee asks where this sits.
6. **Experiments, in order of cheapness:** re-rank the frontier by conditional exchange value
   on one cell; full-block oracle regret on a small instance; finite-shot confidence allocation.

## 9. Do not use without checking

- Any numeric claim in these threads about our results. They predate the corrected campaign;
  thread 3 and thread 7 both read an August draft whose equation numbers no longer match.
- Thread 5's assertion that no deep research was run in the project. Thread 1 contains one.
- The "two-pass scout" description of our implementation in threads 3 and 4. That was the
  pre-repair route; the current selector seals one population per round and enumerates
  complete deletion rungs.

# Paper II: borrowable mathematics and references not currently in the manuscript

Scope filter applied: every reference below is absent from the manuscript's bibliography,
and every formula is absent from its equations. Already cited and therefore excluded:
Natarajan (NP-hardness), Bertsimas--King--Mazumder (MIO best subset), Das--Kempe and
Elenberg et al. (weak submodularity), Yuan--Lin (group lasso), Frisch--Waugh--Lovell,
Efron, Meinshausen--B\"uhlmann.

Notation follows the manuscript: active support $C$ (size $p$), candidate pool, retained
support $R = C \setminus D$, Gram $G$, force $f$, captured drift $\Gamma(S) = f_S^\top
G_{SS}^{\dagger} f_S$.

---

## 1. Every deletion loss from one factorization

The manuscript computes the deletion loss as a Schur complement conditioned on the retained
support,

$$\Gamma(C) - \Gamma(R) = r_{D|R}^\top S_{D|R}^{-1} r_{D|R}, \qquad
S_{D|R} = G_{DD} - G_{DR}G_{RR}^{-1}G_{RD},$$

which needs $G_{RR}^{-1}$, i.e. a different inverse for every deletion set $D$. The same
quantity is available from the *unretained* inverse, which is computed once:

$$\boxed{\;\Gamma(C) - \Gamma(R) \;=\; x_D^\top \bigl(M_{DD}\bigr)^{-1} x_D,\qquad
M = G_{CC}^{-1},\quad x = M f_C. \;}$$

Derivation, two lines from the block-inverse identity: $M_{DD} = S_{D|R}^{-1}$ and
$x_D = S_{D|R}^{-1} r_{D|R}$, hence $x_D^\top M_{DD}^{-1} x_D = r_{D|R}^\top S_{D|R}^{-1}
r_{D|R}$.

**Consequence for our bounded search.** One Cholesky of $G_{CC}$ plus one solve gives $M$ and
$x$; after that the loss of *any* deletion batch costs a $d \times d$ solve, with $d = |D|$.
Singletons are free: $\Gamma(C) - \Gamma(C \setminus \{i\}) = x_i^2 / M_{ii}$. This is the
identity behind the PRESS statistic and leverage scores.

This bears directly on the sentence in Sec. `bounded search` that the $d=4$ rung is
$\binom{42}{4} = 111{,}930$ sets and is refused whole by the joint-work budget: under this
identity that rung is 111,930 four-by-four solves against a single factorization, not
111,930 pseudoinverses. Whether it changes the admitted rungs is a measurable question, not
a rhetorical one.

The retained inverse itself, needed for the conditional append gain after a deletion, comes
from the same $M$ without refactorizing:

$$G_{RR}^{-1} = M_{RR} - M_{RD}\,M_{DD}^{-1}\,M_{DR}.$$

- Fuzhen Zhang (ed.), *The Schur Complement and Its Applications*, Springer (2005) — block
  inverse, quotient property, PSD consequences.
- W. W. Hager, "Updating the inverse of a matrix," *SIAM Review* **31**, 221 (1989) —
  Sherman--Morrison--Woodbury family, including block downdates.
- D. M. Allen, "The relationship between variable selection and data augmentation and a
  method for prediction," *Technometrics* **16**, 125 (1974) — the $x_i^2/M_{ii}$ form as
  leave-one-out prediction error.

## 2. Conditioning-aware selection criteria we do not use

Our score weights compiled cost only; the conditioning of the patched geometry enters as a
diagnostic and a disabled rejection test. The design-of-experiments literature optimizes the
information matrix itself, with the same submatrix algebra:

$$\log\det G_{SS}, \qquad \operatorname{tr} G_{SS}^{-1}, \qquad \lambda_{\min}(G_{SS}).$$

The one-in/one-out exchange has a closed-form determinant update (Fedorov exchange), which is
the design analogue of our true exchange: for swapping $j$ out and $a$ in,

$$\frac{\det G_{S'S'}}{\det G_{SS}} = 1 + \Delta(a,j),$$

with $\Delta$ built from the same conditional quantities $s_{a|R}$ and leverage terms. Useful
if a referee asks why the exchange is scored by captured drift rather than by information
content, and as the natural home for a conditioning-aware variant.

- V. V. Fedorov, *Theory of Optimal Experiments*, Academic Press (1972) — exchange algorithm.
- T. J. Mitchell, "An algorithm for the construction of D-optimal experimental designs,"
  *Technometrics* **16**, 203 (1974) — DETMAX, the exchange heuristic in practice.
- F. Pukelsheim, *Optimal Design of Experiments*, SIAM Classics (2006).
- S. Joshi and S. Boyd, "Sensor selection via convex optimization," *IEEE Trans. Signal
  Process.* **57**, 451 (2009) — convex relaxation of exactly our combinatorial selection, with
  a bound on the gap to the optimum. This is the cleanest citable relaxation if we ever want a
  certified bound rather than a bounded enumeration.

## 3. Replacement algorithms: the closest algorithmic relatives

The manuscript states that its exchange is not beam search and cites no replacement method.
Four exist and are worth naming, because each performs delete-and-add but none defines a
coupled objective, which is precisely our claim:

- P. Jain, A. Tewari, I. S. Dhillon, "Orthogonal matching pursuit with replacement,"
  *NeurIPS 2011*, 1215--1223 — adds one coordinate and removes one per iteration; a partial
  hard-thresholding operator gives a family of such algorithms.
- T. Zhang, "Adaptive forward--backward greedy algorithm for learning sparse
  representations," *IEEE Trans. Inf. Theory* **57**, 4689 (2011) — FoBa; backward steps are
  taken only when the loss increase is below a fraction of the preceding forward gain, which
  is structurally our deletion-loss allowance $\tau$.
- D. Needell and J. A. Tropp, "CoSaMP: iterative signal recovery from incomplete and
  inaccurate samples," *Appl. Comput. Harmon. Anal.* **26**, 301 (2009).
- W. Dai and O. Milenkovic, "Subspace pursuit for compressive sensing signal
  reconstruction," *IEEE Trans. Inf. Theory* **55**, 2230 (2009).

The last two prune a merged support each iteration under a restricted-isometry assumption.
Their guarantees do not transfer to our geometry, and saying so in one sentence is stronger
than not citing them.

## 4. Exact solution of the bounded problem

If the small-instance oracle is ever run, these are the right tools and the right citations:

- H. Hazimeh and R. Mazumder, "Fast best subset selection: coordinate descent and local
  combinatorial optimization algorithms," *Operations Research* **68**, 1517 (2020) — local
  combinatorial search over *swap* neighborhoods with convergence to points satisfying
  necessary optimality conditions; the closest existing formalization of "locally optimal
  under exchange."
- A. Frangioni and C. Gentile, "Perspective cuts for a class of convex 0--1 mixed integer
  programs," *Math. Program.* **106**, 225 (2006) — the perspective reformulation is the
  standard strengthening for an indicator-constrained convex quadratic, which is exactly the
  form our ridge-regularized support problem takes. A big-M MIQP without it is weak.
- G. M. Furnival and R. W. Wilson, "Regressions by leaps and bounds," *Technometrics* **16**,
  499 (1974) — the original branch-and-bound with residual bounds for subset regression.

The exact MIQP form for our objective, with indicators $z_i$ and ridge $\lambda$:

$$\min_{\beta,z}\ \tfrac12 \beta^\top (G + \lambda I)\beta - f^\top\beta + \mu\,C(z),
\qquad \beta_i = 0 \ \text{if}\ z_i = 0,$$

whose optimal value at fixed support $S$ equals $-\tfrac12 \Gamma_\lambda(S)$. Exchange
constraints are linear in $z$: $\sum_{i \in C}(1-z_i) \le d_{\max}$, $\sum_{i \notin C} z_i
\le a_{\max}$.

## 5. Screening scores with provable guarantees

Our insertion frontier keeps the $W=8$ highest-scoring positions. The linear-algebra
literature has screening scores with approximation guarantees for the analogous problem of
choosing $k$ columns:

- C. Boutsidis, M. W. Mahoney, P. Drineas, "An improved approximation algorithm for the
  column subset selection problem," *SODA 2009*, 968--977 — randomized selection by
  subspace-based probabilities followed by a deterministic pass, with a spectral-norm bound.
- P. Drineas, M. W. Mahoney, S. Muthukrishnan, "Relative-error CUR matrix decompositions,"
  *SIAM J. Matrix Anal. Appl.* **30**, 844 (2008) — leverage-score sampling.
- M. Gu and S. C. Eisenstat, "Efficient algorithms for computing a strong rank-revealing QR
  factorization," *SIAM J. Sci. Comput.* **17**, 848 (1996) — deterministic pivoting with a
  bounded loss in the smallest singular value; the natural comparator for any frontier rule.

None of these scores the drift; they score the geometry. The honest use is one sentence
placing our frontier among them, not a guarantee claim.

## 6. Guarantees we could state, and the one we cannot

Das--Kempe and Elenberg et al. are cited but their conclusions are not stated. If we want any
statement at all about greedy quality, the citable form is: with submodularity ratio $\gamma$
of the set function $\Gamma$, greedy returns $S$ with $\Gamma(S) \ge (1 - e^{-\gamma})
\Gamma(S^\star)$, reducing to the classical $1 - 1/e$ at $\gamma = 1$
[G. L. Nemhauser, L. A. Wolsey, M. L. Fisher, *Math. Program.* **14**, 265 (1978)], and
restricted strong convexity plus restricted smoothness lower-bound $\gamma$.

What we cannot claim, with a two-by-two counterexample each:

- **$\Gamma$ is not submodular.** $G = \begin{psmallmatrix}1 & \rho\\ \rho &
  1\end{psmallmatrix}$, $f = (1,1)^\top$ gives $\Gamma(\{1\}) = \Gamma(\{2\}) = 1$ and
  $\Gamma(\{1,2\}) = 2/(1+\rho)$: diminishing returns for $\rho > 0$, increasing returns for
  $\rho < 0$.
- **$\Gamma$ is not monotone under a raw pseudoinverse.** $G =
  \begin{psmallmatrix}1&1\\1&1\end{psmallmatrix}$, $f = (1,0)^\top$ gives $\Gamma(\{1\}) = 1$
  but $\Gamma(\{1,2\}) = f^\top G^\dagger f = 1/4$. Our non-negativity statement for the Schur
  gain is correct because the tangent construction supplies $G = \bar T^\top \bar T$ and
  $f = \bar T^\top \bar d$ in the same range; the manuscript should say which of the two
  carries it, since we also run a ridge and a spectral cutoff.

Both counterexamples are two lines and would sit naturally as a footnote where the paper
first calls the gain non-negative.

## 7. Finite-shot successor

For the "measure only what separates the leading patches" layer, the citable frame is
best-arm identification with elimination: stop when the leading patch's lower confidence bound
exceeds every rival's upper bound.

- E. Even-Dar, S. Mannor, Y. Mansour, "Action elimination and stopping conditions for the
  multi-armed bandit and reinforcement learning problems," *J. Mach. Learn. Res.* **7**, 1079
  (2006).

This composes with the measurement-grouping references already in our bibliography
(Verteletskyi, Yen, Ikhtiarudin).

---

## Where each item attaches

| Item | Manuscript location |
|---|---|
| Deletion loss from one inverse (§1) | Sec. `bounded search`, the rung-cost paragraph; Appendix `technical_realization` |
| Retained inverse by downdate (§1) | Same |
| Design criteria and Fedorov exchange (§2) | Discussion, where conditioning is called a diagnostic |
| OMPR, FoBa, CoSaMP, Subspace Pursuit (§3) | Sec. `bounded support-patch search`, the "not beam search" paragraph |
| Perspective cuts, Hazimeh--Mazumder, leaps and bounds (§4) | Same paragraph, oracle sentence |
| Column subset selection, leverage, RRQR (§5) | Sec. `bounded search`, the frontier declaration |
| Submodularity ratio bound; two counterexamples (§6) | Footnote at the first non-negativity claim |
| Best-arm elimination (§7) | Discussion, finite-shot extensions |

## Verification status

Bibliographic details for Zhang (FoBa), Jain--Tewari--Dhillon (OMPR), Hazimeh--Mazumder,
Frangioni--Gentile, and Boutsidis--Mahoney--Drineas were confirmed against the publisher or
arXiv record today. The remaining entries are standard references quoted from memory: check
volume and page numbers before they enter the `.bib`. Every formula in §1 is derived above and
can be checked numerically on any of our stored Gram matrices in a few lines: the deletion-loss
identity, its singleton shortcut, and the retained-inverse downdate were verified here to
machine precision (relative error $\le 9\times10^{-16}$) for $|D| = 1,2,3,4$ on random
positive-definite Gram matrices.

Still missing: the deep-research report in the SNAKE thread (24,502 characters, 15 citations)
is rendered in a cross-origin sandbox and could not be read. Its download control would put it
on disk.

# Macro First or Singleton First? Measurement Savings from Pauli-Child Shortlisting

## The governing comparison

Let $\rho=|\psi\rangle\langle\psi|$ be the current variational state and let the Hamiltonian be a real Pauli expansion

$$
H=\sum_{a=1}^{L}h_aQ_a,
$$

where each $Q_a$ is a Hermitian Pauli word and each $h_a\in\mathbb R$. A physical macro generator is likewise represented over a global child dictionary $\{P_j\}_{j=1}^{D}$ as

$$
G_m=\sum_{j=1}^{D}c_{mj}P_j,
\qquad
d_m=\left|\{j:c_{mj}\neq0\}\right|,
$$

where the $P_j$ are Pauli children, the $c_{mj}$ are fixed real coefficients, and $d_m$ is the number of nonzero children in macro $m$.

Two candidate-selection pipelines can now be stated precisely. Let $\mathcal I$ be the retained macro-index set, let $\mathcal J$ be the retained singleton-index set, and let $\Phi_2$ denote the input passed to Phase II. Macro-first selection follows

$$
\{G_m\}_{m=1}^{M}
\longrightarrow \{g(G_m)\}_{m=1}^{M}
\longrightarrow \{G_m:m\in\mathcal I\}
\longrightarrow \bigcup_{m\in\mathcal I}\{P_j:c_{mj}\neq0\}
\longrightarrow \Phi_2.
$$

Singleton-first selection follows

$$
\{P_j\}_{j=1}^{D}
\longrightarrow \{g(P_j)\}_{j=1}^{D}
\longrightarrow \{P_j:j\in\mathcal J\}
\longrightarrow \Phi_2.
$$

Here $g(A)$ denotes the zero-insertion energy gradient associated with a Hermitian generator $A$. The measurement question is whether evaluating all required $g(G_m)$ uses less quantum information than evaluating all required $g(P_j)$.

The answer depends on the estimator and its retained measurement record. Under a term-resolved commutator estimator with complete caching, macro-first screening often saves zero Phase-I measurement bases because the measurements used to form the macro scores can already determine the singleton scores. Its dependable savings arise when Phase II requests additional child-specific information only for children of retained macros. Parent screening can also save Phase-I work through algebraic cancellation, coarser precision, randomized estimation, or a direct whole-macro derivative circuit. Each of these mechanisms has a distinct cost model.

---

## The gradient is a linear map

**Micro-intuition:** A macro score is one linear functional of the vector of child scores.

Insert a generator $A$ through

$$
U_A(\theta)=e^{-i\theta A/2},
\qquad
E_A(\theta)=\operatorname{Tr}\!\left(\rho\,U_A(\theta)^\dagger H U_A(\theta)\right).
$$

The scalar $\theta\in\mathbb R$ is the new variational coordinate, $U_A(\theta)$ is its unitary action, and $E_A(\theta)$ is the resulting energy with the pre-insertion state held fixed. Differentiation at the zero coordinate gives

$$
\begin{aligned}
g(A)
&:=\left.\frac{dE_A}{d\theta}\right|_{\theta=0}\\
&=\frac{i}{2}\operatorname{Tr}\!\left(\rho[A,H]\right).
\end{aligned}
$$

The commutator map $A\mapsto[A,H]$ and the expectation map $B\mapsto\operatorname{Tr}(\rho B)$ are both linear. Consequently,

$$
\begin{aligned}
g(G_m)
&=\frac{i}{2}\operatorname{Tr}\!\left(\rho\left[\sum_jc_{mj}P_j,H\right]\right)\\
&=\sum_jc_{mj}\frac{i}{2}\operatorname{Tr}\!\left(\rho[P_j,H]\right)\\
&=\sum_jc_{mj}g(P_j).
\end{aligned}
$$

Collect the child scores into $x\in\mathbb R^D$, with $x_j=g(P_j)$, and the macro scores into $y\in\mathbb R^M$, with $y_m=g(G_m)$. Let $C\in\mathbb R^{M\times D}$ contain the macro coefficients, padding absent children with zero. Then

$$
y=Cx.
$$

The map $C$ usually compresses many child scores into fewer macro scores. Hence the image coordinate $y_m$ generally underdetermines its child vector $(x_j)_{j\in m}$: contributions of opposite sign can cancel, and contributions of the same sign can reinforce.

For example,

$$
G=P_1+P_2,
\qquad
g(G)=g(P_1)+g(P_2).
$$

The child-score pairs $(1,-1)$, $(10,-10)$, and $(0,0)$ all yield $g(G)=0$. A small parent gradient therefore supplies no general upper bound on the largest child gradient. Macro-first and singleton-first selection consequently define different scientific policies even when their measurement costs coincide.

---

## Three kinds of Pauli word

**Micro-intuition:** The Pauli word being measured is usually produced by a Hamiltonian--generator commutator.

Three operator roles enter the calculation:

1. A **Hamiltonian word** $Q_a$ is a term in the expansion of $H$.
2. A **generator word** $P_j$ is a Pauli child of a candidate macro.
3. A **score word** $R_r$ is a Hermitian Pauli observable whose expectation appears after expanding the commutator gradient.

Expanding one child gradient gives

$$
g(P_j)
=\frac{i}{2}\sum_{a=1}^{L}h_a\operatorname{Tr}\!\left(\rho[P_j,Q_a]\right).
$$

Two Pauli words either commute or anticommute. Their contribution has the form

$$
\frac{i}{2}[P_j,Q_a]
=
\begin{cases}
0, & P_jQ_a=Q_aP_j,\\
iP_jQ_a, & P_jQ_a=-Q_aP_j.
\end{cases}
$$

For an anticommuting pair, $iP_jQ_a$ equals a real sign times another Hermitian Pauli word. After combining equal products, every child gradient can therefore be written as

$$
g(P_j)=\sum_{r=1}^{R}A_{jr}z_r,
\qquad
z_r:=\operatorname{Tr}(\rho R_r),
$$

where $R_1,\ldots,R_R$ are the distinct score words and $A\in\mathbb R^{D\times R}$ contains the known Hamiltonian coefficients, commutator signs, and collected multiplicities. In vector form,

$$
x=Az,
\qquad
y=CAz.
$$

Let $\mathcal M_\rho$ denote the measurement map that sends the prepared state to the retained vector of score-word expectations. The complete information flow is

$$
\rho
\xrightarrow{\mathcal M_\rho}
z
\xrightarrow{A}
x
\xrightarrow{C}
y.
$$

Thus a commutator-based gradient calculation measures the nonzero Pauli products generated by the anticommutation relation. The bare Hamiltonian word $Q_a$ and bare generator word $P_j$ specify the two inputs whose product creates the score word. A direct shifted-energy estimator follows a different path and is considered below.

If a macro is executed as a product $\prod_j e^{-i\theta c_{mj}P_j/2}$, its derivative at $\theta=0$ yields the same linear sum because every factor equals the identity at the base point. Away from zero, this product and $e^{-i\theta G_m/2}$ can describe different unitaries when the children fail to commute. The zero-gradient identity survives; finite-shift identities and circuit costs require the chosen macro execution convention.

---

## A two-qubit instance

**Micro-intuition:** One shared basis can reveal two child gradients and their macro sum.

Consider

$$
H=h_1ZI+h_2IZ,
\qquad
G=c_1XI+c_2IX.
$$

The Hamiltonian words are $ZI$ and $IZ$; the generator children are $XI$ and $IX$. Their nonzero commutators give

$$
\begin{aligned}
g(XI)&=h_1\langle YI\rangle_\rho,\\
g(IX)&=h_2\langle IY\rangle_\rho,\\
g(G)&=c_1h_1\langle YI\rangle_\rho+c_2h_2\langle IY\rangle_\rho.
\end{aligned}
$$

Here $\langle R\rangle_\rho:=\operatorname{Tr}(\rho R)$. The score words are $YI$ and $IY$. Measuring both qubits in the $Y$ basis yields bitstrings from which both expectations can be estimated. The macro score and both child scores therefore use the same single measurement setting. If those bitstrings are retained, expanding $G$ after shortlisting creates no new gradient measurement at this state; it selects different classical combinations of the same data.

---

## When cached commutator data removes the Phase-I saving

**Micro-intuition:** If the macro score is computed from a retained vector $z$, the same vector may already contain every singleton score.

Let $\mathcal R_P$ be the set of distinct score words required by all child gradients under consideration. Let $\mathcal R_G$ be the set remaining after the macro combinations $CA$ have been algebraically consolidated. These sets satisfy

$$
\mathcal R_G\subseteq\mathcal R_P.
$$

Suppose Phase I measures every $R_r\in\mathcal R_G$, retains the corresponding bitstrings or expectation estimates, and uses the same state $\rho$ throughout Phase I and Phase II. If

$$
\mathcal R_G=\mathcal R_P,
$$

then the stored $z$ reconstructs both

$$
\begin{aligned}
x&=Az,\\
y&=CAz
\end{aligned}
$$

by classical arithmetic. Initializing the singleton children in Phase I would require no additional quantum measurement settings. Under these conditions, macro-first screening saves zero Phase-I observable acquisition. It postpones a classical expansion that the measurement record already supports.

This equality is common when every child participates in a screened macro, the commutator expansion has little exact cancellation, and the hardware measures a union of Pauli bases whose outcomes are cached globally. It becomes even more consequential when one basis yields several compatible expectations: the same shot record can populate multiple entries of $z$, and those entries can contribute to many children and many macros.

Total saving may remain positive because Phase II can request a richer child-specific object, such as higher-precision gradients, metric overlaps, response coefficients, curvature information, insertion-position comparisons, or batch-interaction terms. If only children of retained macros receive those measurements, rejected macros avoid that downstream work.

Write the total measurement difference schematically as

$$
\Delta N
=N_1(P)-N_1(G)+N_2(D_{\mathcal J})-N_2(D_{\mathcal I}).
$$

Here $N_1(P)$ is the singleton-first Phase-I cost, $N_1(G)$ is the macro-first Phase-I cost, $D_{\mathcal J}=|\mathcal J|$ is the number of children passed by singleton-first screening, and $D_{\mathcal I}$ is the number of distinct children exposed by the retained macros. The special case that sends every singleton to Phase II has $D_{\mathcal J}=D$. Under complete term-resolved reuse, the first difference can vanish and the second can remain positive.

If Phase II only recomputes the same child gradients at the same state and precision, complete caching makes the second difference vanish as well. In that limiting case, macro-first shortlisting saves essentially no quantum measurement work; its remaining effects are classical bookkeeping and a scientifically different selection rule.

---

## Pauli expectations, measurement settings, and shared shots

**Micro-intuition:** Compatible score words share a basis and therefore share shots.

A measurement protocol partitions score words into jointly measurable groups. Under qubit-wise commuting grouping, two words can share a tensor-product basis whenever every qubit position carries either the same nonidentity Pauli or an identity in at least one word. For example, using $I$ for the identity,

$$
ZI,
\qquad
ZZ
$$

are both obtained from a computational-basis measurement. One shot produces a bitstring from which both eigenvalue products can be calculated.

Let $\Gamma(\mathcal R)$ denote a chosen grouping of an observable set $\mathcal R$, and let $|\Gamma(\mathcal R)|$ be its number of measurement settings. Equality of raw word sets is sufficient for equality of group cost, yet it is stronger than necessary. Distinct sets may still obey

$$
|\Gamma(\mathcal R_G)|=|\Gamma(\mathcal R_P)|
$$

because the additional child score words fit inside bases already required by the macro scores. In that event, singleton initialization adds Pauli expectations and leaves the measurement-setting count unchanged.

Shot count introduces a further layer. Let group $b$ have per-shot contribution variance $v_b$, and allocate $n_b$ shots to that group. An aggregate score estimator has variance of the schematic form

$$
\operatorname{Var}(\widehat y_m)
=\sum_b\frac{v_{mb}}{n_b}.
$$

Estimating one macro scalar to tolerance $\epsilon$ can require fewer shots than estimating every child coordinate $x_j$ to its own tolerance, even when both calculations use the same bases. The macro estimator is allowed to distribute error according to the coefficients that affect $y_m$; singleton ranking requires enough coordinate resolution to distinguish the children. In another precision regime, a high-quality cached data set acquired for stable macro ranking may already make every relevant child estimate sufficiently precise. Measurement-setting equality and shot-count equality must therefore be checked separately.

---

## Genuine Phase-I savings mechanisms

Macro-first screening reduces Phase-I quantum work through any of the following mechanisms.

### 1. Algebraic consolidation and cancellation

The matrix product $CA$ can contain fewer nonzero columns than $A$. Equal score words arising from different children combine before measurement, and their coefficients may cancel exactly. Then

$$
\mathcal R_G\subsetneq\mathcal R_P.
$$

Physical savings occur when the removed score words also remove measurement groups or shot demand. Exact cancellation in the parent score simultaneously creates a selection hazard: an individually large child may disappear from the aggregate.

### 2. Aggregate precision and coordinate precision

One scalar $y_m$ can be estimated to a useful screening tolerance with fewer shots than the full child vector. This is a statistical saving even when the observable support is unchanged. The saving depends on the coefficients, covariances, threshold margin, and shot-allocation rule.

### 3. Randomized estimation of the sum

An unbiased estimator can sample terms or groups according to known weights and estimate the aggregate without deterministically visiting every score word in every round. The reduced per-round coverage is purchased with estimator variance. Singleton resolution later requires either retained coverage, additional sampling, or fresh top-up shots.

### 4. Direct whole-macro derivative circuits

One may estimate

$$
g(G_m)=\left.\frac{d}{d\theta}E_{G_m}(\theta)\right|_{\theta=0}
$$

from shifted or finite-difference energy evaluations. Those circuits measure Hamiltonian words $Q_a$ on states transformed by the whole macro unitary and return the whole-macro derivative without separately outputting each child commutator expectation. A simple two-shift rule applies to generators with the required two-eigenvalue spectrum; a general multi-term macro can require a generalized shift rule, an ancilla construction, or an approximation. The measurement-setting count may then depend primarily on Hamiltonian grouping and the number of shifted states; the macro decomposition moves into circuit depth, synthesis error, and the number of spectral shifts.

This estimator can make macro-first screening genuinely cheaper than separate child derivative circuits. Its cost must be compared in state preparations, basis settings, shots, and gate depth together.

### 5. Avoidance of richer Phase-II observables

Parent screening can prevent rejected families from ever requesting child metrics, cross terms, curvature, insertion tests, or higher-precision top-ups. This is the most robust source of savings because those quantities are absent from the parent screen itself.

---

## The exact zero-saving criterion

The macro-first Phase-I screen has zero physical measurement advantage over singleton initialization when all of the following conditions hold:

1. both routes use the same commutator-observable estimator at the same state;
2. the macro score-word groups already cover every group needed for the child gradients;
3. the retained bitstrings or expectation estimates allow all child gradients to be reconstructed;
4. the Phase-I shots provide the precision required for singleton ranking;
5. repeated uses of an expectation are deduplicated through a shared cache.

Assume that the vector $z$ is retained and that its estimated covariance meets the tolerances required by both selection policies. If the measurement-group cover also obeys

$$
\Gamma(\mathcal R_G)=\Gamma(\mathcal R_P),
$$

then

$$
N_1(G)=N_1(P).
$$

Here $\operatorname{Cov}(\widehat z)$ is the covariance matrix of the estimated score-word expectations. It records both the uncertainty of each estimate and the correlations created when several observables are extracted from the same shots.

Macro-first screening has a positive Phase-I advantage when at least one of those equalities fails in its favor: it uses fewer measurement groups, requires fewer shots for an aggregate decision, or obtains its derivative through fewer whole-circuit evaluations. Its total advantage can also remain positive through avoided Phase-II child work.

---

## The selection-semantic cost

Measurement economy and selection fidelity are separate objectives. Macro screening ranks

$$
|y_m|=\left|\sum_jc_{mj}x_j\right|,
$$

Singleton screening ranks the individual magnitudes $|x_j|$. The triangle inequality supplies only

$$
|y_m|\leq\sum_j|c_{mj}|\,|x_j|.
$$

A small $|y_m|$ remains compatible with large child magnitudes because of destructive cancellation. A large $|y_m|$ can result from several moderate children that add coherently. Accordingly, replacing macro-first selection with singleton-first selection changes the candidate policy even when every measured expectation and every shot is identical.

This distinction suggests two possible purposes for a parent screen:

- A **scientific family screen** deliberately asks whether the physical macro acts strongly as a coherent operator. Cancellation is then part of the intended criterion.
- A **measurement-saving prefilter** aims to exclude families and preserve individually valuable children. An aggregate signed gradient alone lacks the information needed for a cancellation-safe guarantee. Such a prefilter needs additional bounds, unsigned component information, randomized certificates, or structural constraints.

The purpose must be declared before measurement savings are interpreted. A family screen can be scientifically meaningful without lowering measurement cost. A measurement-saving prefilter must demonstrate both reduced physical acquisition and acceptable false-rejection behavior.

---

## The practical conclusion for a three-phase selector

The decisive accounting sequence is the marginal physical acquisition after global reuse: new basis settings determine the required state preparations and shots, and those shots determine the achieved estimator precision.

A count of logical parent scores assigns one record to $g(G_m)$; a count of logical child scores assigns several records to the coordinates $g(P_j)$. Those record counts describe algorithmic requests. They establish hardware savings only after each request has been expanded into unique score words, grouped into compatible measurement bases, deduplicated against cached data, and allocated shots to a stated tolerance.

For the specific comparison posed here, the strongest default expectation is:

> If Phase I forms exact macro gradients from term-resolved commutator measurements, retains those measurements, and already covers the singleton commutator-word basis set, initializing the singletons in Phase I costs no additional quantum measurements. Macro-first shortlisting then saves quantum work only through additional Phase-II quantities or higher-precision top-ups that are avoided for rejected macros.

The corresponding warning is equally exact:

> If Phase II measures only the same child gradients at the same state and precision, a macro-first stage can provide no physical measurement saving at all. It still changes the selection rule through coherent addition and cancellation among child gradients.

A defensible comparison should therefore report, for both pipelines on the same accepted state:

1. the unique commutator score words after coefficient consolidation;
2. the compatible measurement groups newly required after cache reuse;
3. the new shots and achieved covariance or error tolerance;
4. the extra Phase-II observables requested per retained child;
5. the false-rejection behavior caused by macro-level cancellation;
6. logical estimator counts as a separate algorithmic-work quantity.

Hence the parent shortlist possesses no universal measurement advantage. Its advantage is conditional upon information compression that survives Pauli expansion, grouping, caching, and precision requirements, or upon genuinely richer child work being deferred until Phase II.

---

## Unsolved transfer questions

1. Suppose two children have gradients $x_1=8$ and $x_2=-7.9$ with equal macro coefficients. Determine what the parent score says about the usefulness of either singleton and identify the selection risk.
2. Suppose the macro and singleton routes require different score-word sets and both sets fit into the same three qubit-wise commuting bases. Identify which cost counters can differ and which basis-setting counter remains equal.
3. Suppose Phase I caches all basis bitstrings and Phase II asks only for the same child gradients at the same state with a tighter tolerance. Describe precisely what Phase II needs to acquire.
4. Suppose a direct whole-macro shift rule uses four shifted states, and its six children each admit a two-shift rule. State the additional information needed before deciding which route has lower hardware cost.

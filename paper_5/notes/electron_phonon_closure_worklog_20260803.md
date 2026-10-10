# Electron--phonon closure worklog

Status: exploratory local analysis; no result in this file is promoted paper
evidence unless explicitly adopted later.

This log records the hypothesis, implementation, verification, numerical
result, and decision for each attempted repair of the 31-real-coordinate
archive moment equations. Exact truncated-Hamiltonian trajectories are used
only for the common initial contraction and offline scoring. They are never
queried by an autonomous right-hand side or the representability controller.

## Baseline entering this work

- The retained state is the 31-coordinate representation of
  \((\rho,B,N,A,C)\).
- The uncorrected archive moment equations lose joint-Gram positive
  semidefiniteness before the late coordinate blow-up.
- The minimum-Euclidean-norm controller preserves the electronic and joint
  Gram constraints, the correlation trace, and zero correction-induced energy
  flux, but it does not repair the inaccurate closure of \(\dot C\).
- The exact-reference audit attributes the dominant \(\dot C\) discrepancy to
  a connected electron--two-phonon contribution, a fixed-sector same-spin
  Pauli-algebra contribution, and an opposite-spin covariance. Supplying all
  three exact contributions offline nearly reconstructs the exact derivative.

## 2026-08-03: autonomous same-spin Pauli replacement

### Hypothesis

In the one-spin-up-electron sector, the same-spin covariance is reconstructed
exactly from the retained one-body density,

\[
P^q_{ij}=\delta_{iq}\rho_{qj}-\rho_{ij}\rho_{qq}.
\]

Replace the effective same-spin source already contained in archive Eq. (14d)
by \(-igP\). This is an autonomous algebraic repair: it requires only the
current \(\rho,N,A\) blocks and does not add a new dynamic variable.

### Implementation and verification

- Added an opt-in Pauli-repaired matrix and 31-coordinate right-hand side.
  The raw archive right-hand side remains the default.
- The replacement changes only the \(C\)-velocity before controller action.
- The production correction agrees with the independently assembled
  exact-reference audit at sampled exact states.
- The fixed-sector covariance formula, matrix/coordinate derivative parity,
  decoupled control, and four-lane analysis are covered by tests.
- Complete Paper V test result after implementation: `90 passed`.

The four matched lanes are:

1. raw archive EOM;
2. Pauli replacement only;
3. representability controller only;
4. Pauli replacement plus controller.

### Short-horizon gate, cutoff 16, \(t=4\), \(\Delta t=0.01\)

- Absolute time-RMS \(\ell_2\) defect of \(\dot C\):
  \(0.438313\rightarrow0.279007\) with the Pauli replacement.
- Residual-subtracted time-RMS \(\ell_2\) defect:
  \(0.166690\rightarrow0.178473\). Thus the replacement reduced the absolute
  mismatch, including the initial residual, but did not improve the remaining
  time-dependent defect over this horizon.
- Pauli-only \(C\)-trajectory RMS error:
  \(0.234830\rightarrow0.161892\).
- The Pauli-only lane did not preserve representability: its first sampled
  negative joint-Gram eigenvalue occurred at \(t=0.14\), compared with
  \(t=0.17\) for the raw lane.
- The Pauli replacement restored \(\operatorname{Tr}C^q=0\) to floating-point
  precision, but trace tangency alone did not preserve the Gram cone.
- Relative to controller-only, Pauli plus controller reduced the \(C\)-block
  RMS trajectory error from \(0.193801\) to \(0.075213\), and reduced maximum
  total-coordinate error from \(0.887985\) to \(0.257910\). Its RMS controller
  action increased from \(0.201131\) to \(0.228304\).
- The combined lane remained representable, with minimum joint-Gram
  eigenvalue \(1.8963\times10^{-5}\).

### RK4 step-convergence gate

Repeating \(t=4\) at \(\Delta t=0.005\) changed the combined trajectory by at
most \(7.65\times10^{-7}\) in coordinate \(\ell_2\) norm on the common sample
grid. The reported improvement ratios changed by less than \(0.1\%\).

### Persistence gate, cutoff 16, \(t=20\), \(\Delta t=0.01\)

- The Pauli replacement reduced the absolute and residual-subtracted
  exact-sample \(\dot C\) defects by factors \(0.679\) and \(0.865\),
  respectively.
- The combined lane remained representable, with minimum joint-Gram
  eigenvalue \(1.0356\times10^{-5}\).
- Relative to controller-only, the combined \(C\)-trajectory RMS error was
  only \(5.7\%\) lower, while maximum total-coordinate error was \(41.0\%\)
  higher and RMS controller action was \(77.4\%\) higher.
- From \(t=4\) to \(20\), maximum energy change from its \(t=4\) value was
  \(1.09\times10^{-6}\) for the combined lane, \(5.03\times10^{-6}\) for
  controller-only, and \(3.07\times10^{-6}\) for the exact cutoff reference.
  The loss of advantage was therefore a trajectory-accuracy issue rather than
  a post-pulse energy-drift failure.

### Decision

Keep the same-spin replacement as a tested algebraic ablation, not as the
accepted closure repair. It removes the correlation-trace error and gives a
large early accuracy improvement with the controller, but that advantage does
not persist across all retained blocks. A \(t=140\) or \(t=1000\) run is not
warranted for this variant.

Artifacts:

- `output/local_runs/paper_v_autonomous_pauli_repair_ablation_20260803_v2/`
- `output/local_runs/paper_v_autonomous_pauli_repair_ablation_dt005_20260803_v1/`
- `output/local_runs/paper_v_autonomous_pauli_repair_ablation_t20_20260803_v1/`

The directory ending in `ablation_20260803_v1` is an unadopted rendering
failure: the numerical calculation completed, but an over-escaped plot label
prevented artifact emission. `v2` is the valid \(t=4\) artifact.

## 2026-08-03: audit before a direct connected-moment repair

The previously implemented electronic-conditioned Gaussian closure is not a
direct 31-coordinate approximation to the connected electron--two-phonon
term. It closes the fifth moments required by the 82-coordinate fourth-order
hierarchy. Its gate reduced the terminal relative derivative defect from
\(2.128\) to \(1.598\), but failed the declared \(0.1\) threshold; autonomous
propagation was correctly deferred.

The conditional-packet and multi-coherent studies tested wavefunction
representations rather than an archive-EOM-native correction to \(\dot C\).
They are not candidates for the present direct repair.

### Current bounded objective

Construct the smallest autonomous, symmetry-preserving approximation to

\[
K^{qr}_{ij}
=\left\langle[O_{ij},n_{r\uparrow}]\,\delta X_r\,\delta b_q\right\rangle
-\left\langle[O_{ij},n_{r\uparrow}]\right\rangle
 \left\langle\delta X_r\,\delta b_q\right\rangle,
\]

using only retained 31-coordinate moments. Audit its contribution
\(-ig\sum_rK^{qr}_{ij}\) on exact cutoff-16 samples before allowing any
autonomous propagation. Required gates are exact algebraic symmetries,
cutoff-converged derivative improvement, no exact-reference access in the
online map, and a material reduction of the residual \(\dot C\) defect.

### Candidate: sequential right-action Pauli regression

Let \(s_a=\sigma_a-\langle\sigma_a\rangle I\) and let
\(y=(\delta b_0,\delta b_1,\delta b_0^\dagger,\delta b_1^\dagger)\). The
retained joint Gram matrix supplies the electronic Gram block
\(E_{ab}=\langle s_a s_b\rangle\) and cross block
\(Z_{\alpha a}=\langle y_\alpha^\dagger s_a\rangle\). A state-weighted
least-squares projection reconstructs each phonon direction on the electronic
Pauli support. Applying this projection sequentially from the right to
\(\delta X_r\delta b_q|\psi\rangle\) reverses the electronic product order:
the rightmost \(\delta b_q\) produces \(s_b\), then commuting
\(\delta X_r\) past \(s_b\) and projecting it produces \(s_b s_a\). This
fixes the ordering without fitting it to exact data.

The first cutoff-16, \(t=4\) prototype used no trainable coefficients and gave:

- raw residual-subtracted \(C\)-derivative RMS \(\ell_2\) defect:
  \(0.166690\);
- plus conditional-regression \(K\): \(0.133597\);
- plus conditional-regression \(K\) and autonomous Pauli repair:
  \(0.132180\);
- plus the exact offline \(K\) contribution: \(0.089345\).

The candidate therefore captures a material but incomplete part of \(K\).
The next gate is a source-level implementation with symmetry, decoupled,
cutoff-convergence, and exact-reference-isolation tests. Propagation remains
unauthorized at this point.

### Source implementation and pre-gate verification

The autonomous candidate is implemented in
`src/paper5/stability/connected_moment_closure.py`. It reconstructs the
conditional Pauli coefficients and the resulting connected-moment source
using only the current 31-coordinate matrix state and model parameters. No
exact-reference object is imported or accepted by the online map.

The offline scoring driver is
`src/paper5/stability/connected_moment_closure_analysis.py`. It evaluates the
same autonomous map along exact cutoff trajectories only for error scoring,
compares cutoffs 12, 16, and 20, and emits arrays, a diagnostic plot, a summary,
and source/artifact hashes. Propagation is authorized only if the algebraic,
material-improvement, and cutoff-convergence gates all pass.

Pre-gate tests verify the state-weighted normal equations, zero diagonal and
trace of the predicted connected-moment matrices, the zero-cross-correlation
and zero-coupling controls, and complete artifact emission. The focused
closure, matrix-parity, exact-reference, and electron--phonon analysis tests
passed 28/28; the dedicated connected-closure suite passed 6/6 after adding
the analysis-driver tests.

### Offline direct-K gate result

The cutoff-12/16/20 gate passed all declared conditions. At cutoff 16, the
residual-subtracted time-RMS l2 defect of the correlation derivative changed
from 0.166690 for the archive closure to 0.133597 after adding the autonomous
conditional-regression connected-moment source, a ratio of 0.80147. Adding the
autonomous same-spin Pauli replacement gave 0.132180, a ratio of 0.79297.

The connected-moment improvement ratio differed by only 7.92e-5 between
cutoffs 16 and 20. The predicted matrices had exactly zero diagonal and trace
in the tested representation; the maximum relative residual of the weighted
normal equations was 3.26e-15. At cutoff 16, the predicted source had mean
cosine alignment 0.588 with the exact offline connected-moment source and was
positively aligned at 75.5% of active samples. Its relative time-RMS source
error was 0.664, confirming that it is useful but incomplete.

Decision: authorize only a short matched propagation with the existing joint
representability controller. The exact trajectory remains an offline scoring
reference and is not available to the propagated right-hand side.

Artifact:

- `output/local_runs/paper_v_conditional_pauli_regression_k_gate_20260803_v1/`

### Short matched propagation result

The authorized cutoff-16 propagation compared four controller-stabilized
lanes through t=4 at RK4 step 0.01: archive, Pauli repair, conditional K, and
conditional K plus Pauli repair. All lanes used the same exact contracted
initial state; exact propagation was used only for post-run scoring.

Relative to the Pauli-plus-controller parent, conditional K plus Pauli:

- reduced the C-block RMS trajectory error from 0.075213 to 0.070548
  (ratio 0.93798);
- reduced RMS controller action from 0.228304 to 0.167045
  (ratio 0.73168);
- remained representable, with minimum joint-Gram eigenvalue 2.08e-5;
- increased the maximum total-coordinate error from 0.257910 to 0.338933
  (ratio 1.31415).

The degradation was concentrated in the phonon second moments. RMS errors in
A and N changed from 0.077940 and 0.025105 to 0.104940 and 0.087258,
respectively. The B, C, and rho RMS errors were lower or comparable. The
conditional-K-only lane also failed to improve its archive-plus-controller
parent: its C RMS error increased from 0.193801 to 0.211505 and its maximum
coordinate error increased from 0.887985 to 0.948235.

Decision: the autonomous conditional-Pauli regression is rejected as the next
closure repair in its present form. It improves the local C derivative audit
and part of the propagated C behavior but transfers error into N and A through
the coupled EOM. The predeclared half-step refinement is not authorized because
the material trajectory-accuracy gate failed.

Artifact:

- `output/local_runs/paper_v_conditional_pauli_regression_k_propagation_20260803_v1/`

### Verification status after the direct-K experiment

The complete `paper_5/tests` suite passed 100/100. The online implementation
module `connected_moment_closure.py` has no exact-reference import. Exact
propagation appears only in the two reporting/analysis drivers used to score
the autonomous map and to provide a common initial state. No half-step or
long-horizon artifact was generated after the propagated-accuracy gate failed.

### Follow-up feedback audit and opposite-spin prototype

The transfer into N and A is structurally expected: the correlation-source
terms in archive Eqs. (14b)--(14c) depend directly on C. The autonomous K
source changes only dC, but its cutoff-16 source has relative time-RMS error
0.664 and negative alignment with the exact K source at 24.5% of active
samples. The resulting phase/orientation error in C therefore enters the N and
A velocities during subsequent steps. A lower controller norm only shows that
the candidate trajectory is easier to keep inside the representability cone;
it does not establish greater accuracy.

A second offline prototype reconstructed the omitted opposite-spin covariance
by conditional independence through the retained phonon Gram block,
schematically as the real symmetric part of Z-dagger times the pseudoinverse
of M_B times Z. This uses only retained moments and is the smallest natural
joint-Gram estimate of the covariance between the two spin sectors. It failed:
the predicted opposite-spin covariance had relative time-RMS error 1.558,
norm 2.079 times the exact covariance, and mean directional cosine 0.595. Adding
its velocity contribution to conditional K plus Pauli changed the residual C
derivative defect from 0.132180 to 0.134263. It was therefore not implemented
as an online EOM term.

### Current scientific boundary

The completed tests establish that neither an isolated same-spin repair nor
the two smallest instantaneous joint-Gram regressions provide an accurate
autonomous closure. The remaining error is not a positivity-controller bug:
the controller preserves representability, while the moment hierarchy lacks
enough information to reconstruct the missing K and opposite-spin terms
accurately from the instantaneous 31-coordinate tuple. A next attempt would
need an explicitly dynamical memory/auxiliary model or a stronger
state-reconstruction principle, with a new offline gate before propagation.

## 2026-08-03: GPT-Pro Ultra theoretical-closure handoff

A standalone math-first dossier was prepared for high-level theoretical
analysis. It asks for a finished solution rather than a research plan:
formalize the non-identifiability of the missing correlation source from the
instantaneous 31-coordinate tuple, select one autonomous closure, derive its
finite auxiliary state and equations, account for energy and joint-Gram
representability, and prescribe derivative-first falsification gates.

The dossier supplies the matrix EOMs, exact \(K/P/D\) attribution, cone
controller, hierarchy gates, physically adapted closure failures, and the
latest direct \(K\), Pauli, and opposite-spin negative results. A
projection-derived memory realization is presented as the leading hypothesis,
not a predetermined answer. The accepted solution may not query or fit the
exact trajectory online; the exact cutoff calculation remains an offline
scorer.

No new propagation, manuscript edit, evidence promotion, or implementation
change was made while preparing this handoff.

## 2026-08-03: finite-cutoff non-identifiability witness

### Question

Can the exact correlation velocity be a globally single-valued instantaneous
function of the retained 31-coordinate tuple, even before choosing a practical
closure formula?

### Construction

The exact fixed-sector operator matrices were used to build the real Hermitian
span of every raw expectation determining \((\rho,B,N,A,C)\) for both spin
sectors. Exact derivative operators for the lower \((\rho,B,N,A)\) blocks were
also included as constraints. A centered \(C\)-velocity operator was projected
onto the orthogonal complement of this span.

At cutoff 3, the spin-symmetric interior reference was

\[
\varrho_{\rm ref}
=0.9|\psi_0\rangle\langle\psi_0|+0.1I_{64}/64.
\]

The normalized orthogonal residual \(\Delta\) generated two states

\[
\varrho_\pm
=\varrho_{\rm ref}\pm0.0023542726\Delta.
\]

Both states are trace one, spin symmetric, strictly positive, and satisfy the
tested joint-Gram condition. Their minimum full-density and joint-Gram
eigenvalues are \(8.59375\times10^{-4}\) and \(0.111898\), respectively.

### Result

- The maximum difference between the two retained 31-coordinate states is
  \(3.25\times10^{-19}\).
- Their combined exact \((\rho,B,N,A)\)-velocity difference is
  \(1.75\times10^{-18}\).
- Their exact \(C\)-velocity difference is \(2.90343\times10^{-2}\) in
  Euclidean norm, with maximum component difference
  \(2.67394\times10^{-2}\).
- The selected centered-\(C\) derivative operator has orthogonal residual norm
  \(5.67891\), or \(0.61207\) of its full norm.
- The witness persists at cutoffs 2, 3, and 4. The relative operator residuals
  are \(0.58461\), \(0.61207\), and \(0.63538\); the corresponding exact
  \(C\)-velocity separations are \(0.02467\), \(0.02903\), and \(0.03313\).

Because the archive right-hand side sees the same \(x\), it assigns the same
velocity to both states, while the exact Hamiltonian assigns different
correlation velocities. This is a computational counterexample to a globally
exact instantaneous 31-coordinate closure on the finite-cutoff physical mixed
state space.

The result does not rule out exact closure on a narrower pure-state or
trajectory-specific manifold. It strengthens the case for an auxiliary
memory state or an explicitly selected reconstruction manifold.

### Verification

The focused witness tests pass 3/3. The complete Paper V suite passes 103/103.
The tests check full-density positivity, joint-Gram positivity, spin symmetry,
constraint orthogonality, equality of retained coordinates, equality of lower
exact velocities, equality of the archive RHS, and separation of the exact
\(C\) velocities.

## 2026-08-03: lifted-Frobenius correction-metric ablation

The online representability correction had previously minimized the Euclidean
norm of the 31 independent real correction coordinates. The pedagogical and
advisor derivations identify a distinct fixed metric \(W\) induced by lifting
those coordinates into \((\Delta\dot\rho,\Delta\dot B,\Delta\dot N,
\Delta\dot A,\Delta\dot C)\), but no trajectory had minimized
\(w^{\mathsf T}Ww\).

The correction implementation now accepts an explicit Frobenius metric. It
constructs \(W\) from the five matrix-block lift, whitens the correction
coordinates exactly, and applies the unchanged energy, correlation-trace,
electronic-positivity, and joint-Gram barrier constraints in the whitened
coordinates. The existing Euclidean metric remains the default. Focused tests
verify that \(w^{\mathsf T}Ww\) equals the explicit sum of squared block
Frobenius norms, that both optimization routes recover the same
Frobenius-minimum correction, and that the default Euclidean result is
unchanged. The focused correction suite passes 22/22.

The completed diagnostic source-locked the existing Euclidean trajectory and
propagated matched Euclidean and lifted-Frobenius lanes through \(t=20\).  The
new Euclidean lane reproduces the saved \(t=1000\) source prefix with maximum
coordinate \(\ell_2\) difference \(2.01\times10^{-6}\), below the predeclared
\(10^{-5}\) anchor tolerance.  All \(8{,}000\) constrained RK4-stage solves in
each lane converged, both lanes retained the tested electronic and joint-Gram
constraints, and their maximum correction-induced energy fluxes were below
\(3.8\times10^{-16}\).

Against the same saved cutoff-16 exact reference, replacing the coordinate
Euclidean objective by the lifted-Frobenius objective changed the
dynamic-normalized time-RMS block errors by

\[
(\rho,B,N,A,C)=(-6.94\%,-20.01\%,-5.92\%,+4.18\%,-2.88\%).
\]

The combined normalized block error decreased only \(0.55\%\), while the
unnormalized combined block-Frobenius error increased \(0.85\%\).  The site
occupation RMS error decreased from \(0.06091\) to \(0.05502\), but the
internal-energy RMS error increased from \(0.06284\) to \(0.12788\).  Both
post-drive internal energies were numerically stationary to approximately
\(10^{-11}\); the larger Frobenius energy offset arose during the driven
interval rather than from later drift.

A common-state audit at 101 samples from each corrected trajectory confirmed
the optimization semantics: the Euclidean solution minimized the coordinate
norm and the Frobenius solution minimized the lifted block norm to
\(7.0\times10^{-18}\) numerical tolerance.  Halving the RK4 step from \(0.01\)
to \(0.005\) changed either trajectory by at most \(2.6\times10^{-6}\) in
coordinate \(\ell_2\) norm.  Thus the accuracy redistribution is not a solver
step artifact.  Cutoff-16 versus cutoff-20 discrepancies are smaller than the
metric effect for \(\rho\) and \(B\), but comparable to some \(N,A,C\) changes;
those smaller block-level improvements remain cutoff-limited evidence.

Decision: retain the Euclidean metric as the default.  The lifted-Frobenius
metric is a valid diagnostic ablation that changes how correction effort is
allocated, but it is not a uniformly more accurate controller.  The complete
analysis, plots, source-lock audit, and numerical summaries are under
`output/local_runs/paper_v_correction_metric_ablation_t20_20260803_v1/`.

## 2026-08-03: Hilbert--Schmidt block-Krylov memory closure, Gates A--B

The Ultra memorandum's proposed fixed-projector construction is being tested
as a falsifiable model.  The first implementation defines the 29 independent
uncentered raw moments, their exact nonlinear centering map into the existing
31-slot state, the analytic 31-by-29 Jacobian, and the normalized
Hilbert--Schmidt observable basis.  Exact trajectories remain unavailable to
the online coefficient object; a wavefunction is accepted only by offline
initial-contraction and diagnostic helpers.

The initial algebraic tests pass:

- 100 raw/centered round trips recover the input to floating-point precision;
- the analytic centering Jacobian agrees with centered finite differences;
- the whitened raw operators are orthonormal to (2\times10^{-13});
- exact ground-state operator contractions reproduce the existing centered
  matrix contractions; and
- a randomly selected spin-symmetric state's exact retained velocity agrees
  with the augmented projected velocity below (8\times10^{-13}).

The measured finite-cutoff force rank is not the nominal physical rank seven.
At local phonon cutoffs 2, 3, and 4 it is 19, and each of the first five static
Krylov shells also has rank 19.  Therefore the exact three-shell static model
has 57 auxiliaries, not 21.  The retained drive force
(Q\mathcal J_VR) is zero to roundoff, as proposed, and all projected
generators have relative symmetric leakage below (10^{-16}).  The centered
Hamiltonian lies in the retained span and its augmented generator null-vector
residual is below (10^{-15}).  At zero electron--phonon coupling the measured
force deflates to rank zero.

The drive nevertheless leaks strongly from the static auxiliary space.  For
the three-shell construction, the norm of the exterior drive action relative
to the full auxiliary drive action is 0.689, 0.657, and 0.632 at cutoffs 2, 3,
and 4.  This does not by itself apply the memorandum's integrated residual
rejection rule, but it is an early indication that the static-chain candidate
may require the preregistered joint-word fallback.  The next calculation is
the teacher-forced residual and derivative gate; no autonomous rollout is
authorized before it passes.

### Cutoff-16 teacher-forced Gate-B result: rejected

The static-chain candidate was evaluated at cutoff 16 on the preregistered
(t=0:0.01:4) grid.  Cubic Hermite interpolation of exact retained
coordinates and derivatives supplied only the offline teacher input to the
auxiliary equation.  The auxiliary state was initialized once from the exact
ground state and then integrated by RK4; exact auxiliary coordinates were not
used as model output.

The audit independently reproduces the inherited archive baseline:

- raw (C)-derivative RMS: 0.438312675;
- residual-subtracted (C)-derivative RMS: 0.166690230.

The projected candidate fails decisively:

| static order | auxiliaries | residual-subtracted (C) RMS | terminal relative RMS | integrated residual ratio |
|---:|---:|---:|---:|---:|
| 2 | 38 | 3.579279 | 6.40309 | 0.69204 |
| 3 | 57 | 5.612152 | 4.04757 | 0.87923 |
| 4 | 76 | 4.264533 | 6.09544 | 0.64800 |

The order-3-to-order-4 modeled-source difference is 1.38488 relative to the
order-4 source norm, versus the required maximum 0.1.  The order-3 modeled
missing-source norm is 12.98 times the exact missing-source norm.  Errors also
enter (B,N,A), so the failure is not confined to the correlation block.

The integrated drive-residual fraction is only 0.0154 at order three.  Thus the
memorandum's joint-word fallback trigger, which requires drive leakage to
dominate the residual, is not met; the static exterior residual and omitted
initial orthogonal dynamics dominate this failure.  Because a single required
Gate-B rejection is sufficient, the cutoff-12/20 repetitions, autonomous
rollout, representability controller, and long-horizon runs are not
authorized.

An independent numerical audit re-integrated the saved teacher-forced
auxiliary equations with adaptive DOP853 at relative tolerance (10^{-11})
and maximum step 0.0025.  The order-2/3/4 (C)-derivative RMS values were
3.579279508, 5.612151827, and 4.264533431, agreeing with RK4 to at least seven
significant figures.  The failure is therefore not an RK4-step artifact.

This is the precise amendment needed from the theoretical agent: the fixed
uniform-Hilbert--Schmidt projection measures rank 19 rather than seven and its
static Krylov sequence is neither accurate nor order convergent under the
prescribed teacher-forced test.  Any revised proposal must explain whether the
problem is the cutoff-boundary force sector, the projection metric, treatment
of the correlated initial orthogonal state, or the choice of reachable versus
observable reduction before recommending another rollout.

Artifact:

- `output/local_runs/paper_v_hs_krylov_gate_b_cutoff16_20260803_v1/`

Verification after the implementation and artifact run: the complete Paper V
test suite passes 119/119.  The saved residual PDF was rendered and inspected;
all three panels, labels, logarithmic scales, and legends are legible with no
clipping or overlap.

## 2026-08-03: preparation-conditioned unitary residual-Galerkin replacement

The amended theoretical recommendation rejects the failed static,
uniform-Hilbert--Schmidt, force-only polynomial chain and replaces it with a
preparation-conditioned unitary residual-Galerkin model (PURG).  The online
state is now one reduced ket.  Its Hamiltonian, raw-moment operators, and
initial state are direct Hermitian compressions in a deterministic
Hilbert-space basis; the 31 centered moment coordinates are algebraic outputs
of that ket rather than separately integrated variables.

The construction implementation now includes the fixed preparation,
drive-direction, and 29-observable seed; shifted inverse and positive-power
packets at shift 0.5; residual-greedy packet enrichment; nested caps
32/64/96/128; unitary exponential-midpoint construction trajectories; reduced
residual Grams; and the Duhamel state and centered-derivative certificate.  The
deployed model object contains only compressed arrays and cannot access a full
or exact driven trajectory.  The construction artifact stores the frozen
bases, compressed models, residual Grams, centering map, certificates, and
source hashes needed by later scorer-only gates.

Seven focused tests pass.  They cover deterministic basis construction,
preparation and drive-direction containment, common-ket moment contraction,
analytic derivative/norm/work identities, reduced-versus-explicit residual
agreement, the Duhamel bound against an independent short exact trajectory,
online dependency isolation, and machine-readable artifact emission.  The
complete Paper V suite passes 126/126 before the cutoff-16 gate run.

The fixed cutoff-16 construction-only Gates A--C are running over
\(0\le t\le4\) at caps 32, 64, 96, and 128.  The short startup audit measures
initial seed rank 30; cap 32 is therefore available.  Its ground-state and
shifted-solve residuals are \(1.85\times10^{-13}\) and
\(1.25\times10^{-15}\), while its orthogonality, initial-state containment,
and initial drive-direction containment residuals are all below
\(1.1\times10^{-14}\).  No exact driven trajectory, controller feedback, or
long-horizon rollout is authorized during this construction gate.

### Cutoff-16 construction-only Gates A--C: rejected

The frozen run completed at all four caps.  Gate A passed in full: the initial
rank was 30; ground-state and maximum shifted-solve residuals were
\(1.85\times10^{-13}\) and \(1.25\times10^{-15}\); map round-trip and
Jacobian residuals were \(1.13\times10^{-16}\) and
\(8.67\times10^{-16}\); the decoupled old force had absolute norm
\(5.33\times10^{-15}\) and declared rank zero; and the online dependency
audit found only the six allowed compressed-model fields.

Gate B failed at every cap:

| cap | final Duhamel state bound | projection-residual integral (trapezoidal audit) | bounded \(C\)-derivative RMS | bounded \(C\)-derivative max |
|---:|---:|---:|---:|---:|
| 32 | 0.264891 | 0.264615 | 48.0151 | 116.419 |
| 64 | 0.0763191 | 0.0760153 | 10.1194 | 22.0479 |
| 96 | 0.0315012 | 0.0311678 | 3.83995 | 8.14825 |
| 128 | 0.0141274 | 0.0137762 | 1.82285 | 3.49655 |

The fixed state-bound limit is 0.0025 and the fixed \(C\)-derivative limits
are 0.025 RMS and 0.075 max.  Thus rank 128 misses the state certificate by a
factor 5.65, and its failure is dominated by the projection residual rather
than the midpoint-integrator defect.  The full-space operator-norm derivative
intervals are much looser, missing their \(C\) budgets by factors 72.9 and
46.6.  The half-step repeat also fails its fixed tolerance, and the measured
continuous norm drift slightly exceeds \(10^{-13}\) at some caps; neither
secondary numerical issue explains the projection-residual failure.

Gate C fails for 32-to-64 and 64-to-96 but passes for 96-to-128.  At the last
pair the autonomous modeled \(C\)-derivative difference is 0.000758 RMS and
0.001791 max, with every lower block also within its preregistered limits.
Consequently the observable paths appear rank-stable while the rigorous global
state and derivative certificates remain too large.  Under the amendment's
fixed decision rule this does not authorize exact scorer Gates D--F, Gate G,
or any long-horizon rollout.  No threshold was changed and no exact driven
trajectory was opened.

Artifact:

- `output/local_runs/paper_v_purg_construction_gate_cutoff16_20260803_v2/`

The canonical v2 artifact distinguishes three genuinely RRQR-deflated initial
columns from two independent packet columns truncated at the rank-32 cap.  Its
72 saved numerical arrays and all Gate A/B/C values are bitwise identical to
v1; only this provenance classification and the synchronized source hashes
changed.  The original v1 artifact remains preserved as superseded evidence.
After the provenance repair, the complete Paper V suite passes 126/126 and
bytecode compilation succeeds for all Paper V sources and tests.

### Scope clarification retained for the theoretical handoff

Gabriele's dimer evidence is condition-specific.  In his Figure 34 at
(lambda=1.5) and (v/t_{\mathrm{hop}}=1), the non-Markovian equations
without the anomalous phonon correlator diverge from a Hartree--Fock electronic
density plus zero correlations but remain bounded when initialized from exact
correlated contractions.  His earlier slides separately show that adding the
anomalous correlator at (lambda=0.5) improves electron--phonon correlations
while destabilizing the propagation and producing negative phonon
populations.  Therefore neither his work nor the local reproduction supports
a universal claim that the hierarchy fails for every strong-coupling initial
condition.  The PURG follow-up handoff now explicitly distinguishes equation
set, retained correlators, initialization, and protocol before comparing any
stability conclusion.

## 2026-08-03: nearby-initial-condition and finite-time Lyapunov screen

The earlier protocol comparisons establish dependence on materially different
initializations, but they do not test sensitive dependence between nearby
states.  A separate two-trajectory Benettin diagnostic now performs that test
for the 31-coordinate closure.  It uses the same strong-coupling point
(λ=1.5, γ=0.5, drive amplitude 1), cutoff-16 exact ground-state
contractions, and the unmodified archive vector field or the established
Euclidean minimum-norm joint-Gram controller.  No exact trajectory enters
either online vector field.

Each random 31-coordinate perturbation is projected to have zero first-order
energy change and zero real and imaginary correlation-trace changes.  The
31-coordinate lift already enforces electronic unit trace, Hermiticity, and
covariance symmetry.  The perturbation is normalized in the combined lifted
Frobenius norm, integrated beside the base trajectory, and rescaled every
0.5 inverse-hopping units.  Both Euclidean and lifted-Frobenius separation
rates are recorded.  Cone eigenvalues and correlation traces are checked for
the base and rescaled shadow at every reset.

The raw closure is not eligible for a physical-chaos conclusion.  Its joint
Gram matrix crosses the PSD boundary at 0.1607116782, and its correlation-trace
identity is already violated during that short interval.  Extending the raw
mathematical ODE to t=20 gives a positive Frobenius finite-time rate of
0.07258, but by then the minimum joint-Gram eigenvalue is approximately
-1.012 and the electron and boson marginals are also nonphysical.  This is an
unstable unphysical continuation, not evidence of physical chaos.

For the joint-Gram-controlled flow, four independent numerical controls give:

| direction / numerical control | final t=100 Frobenius rate | post-t=4 rate | t=40--100 rate | t=50--100 rate | minimum joint-Gram eigenvalue |
|---|---:|---:|---:|---:|---:|
| seed 20260803, dt=0.02, epsilon=1e-5 | 0.0398103 | 0.0517393 | 0.0567787 | 0.0541338 | 2.4221e-5 |
| seed 20260804, dt=0.02, epsilon=1e-5 | 0.0586009 | 0.0639801 | 0.0567872 | 0.0552500 | 2.4850e-5 |
| seed 20260803, dt=0.02, epsilon=2e-5 | 0.0382676 | 0.0504964 | 0.0568588 | 0.0541962 | 1.9518e-5 |
| seed 20260803, dt=0.01, epsilon=1e-5 | 0.0446702 | 0.0570530 | 0.0568080 | 0.0547403 | 2.4221e-5 |

The two directions have different early transients but align by the late
window.  The t=40--100 estimates agree within 8.1e-5, and all four remain
strictly inside the sampled electronic, bosonic, and joint cones.  The largest
sampled correlation-trace residual is 1.15e-16.  Perturbation size 1e-4 was
rejected because the shadow left the cone.  A 1e-7 smoke perturbation was also
rejected because it lies too close to the 1e-8 cone-solver tolerance and
produces a spurious positive rate from the solver floor.

This establishes reproducible finite-time exponential sensitivity of the
representability-controlled vector field over the tested window.  It does not
establish chaos of the archive EOMs: the raw flow loses representability before
a long physical Lyapunov calculation is possible, while the controlled flow is
a different, piecewise-smooth dynamical system.  A chaos claim for that
controlled system would still require a longer-horizon exponent plateau (or a
converged spectrum), bounded invariant-set analysis, and confirmation that the
positive rate is robust to the controller's active-set transitions.

Artifacts:

- `output/local_runs/paper_v_nearby_sensitivity_archive_t20_dt001_eps1e5_seed20260803_v1/`;
- `output/local_runs/paper_v_nearby_sensitivity_archive_precone_t016_dt0001_eps1e5_seed20260803_v2/`;
- `output/local_runs/paper_v_nearby_sensitivity_joint_t100_dt002_eps1e5_seed20260803_v1/`;
- `output/local_runs/paper_v_nearby_sensitivity_joint_t100_dt002_eps1e5_seed20260804_v1/`;
- `output/local_runs/paper_v_nearby_sensitivity_joint_t100_dt002_eps2e5_seed20260803_v1/`;
- `output/local_runs/paper_v_nearby_sensitivity_joint_t100_dt001_eps1e5_seed20260803_v1/`;
- `output/local_runs/paper_v_nearby_sensitivity_convergence_20260803_v1/summary.json`.

The reusable diagnostic is implemented in
`paper_5/src/paper5/stability/initial_condition_sensitivity.py`.  Three focused
tests cover a known linear exponent, the constrained perturbation construction,
and dual-norm bookkeeping.  The complete Paper V suite passes 129/129.

## 2026-08-03: amended PURG certificate received

The post-failure theoretical amendment preserves the online PURG reduced-ket
equation and replaces the rejected global state/operator-norm output gate with
an offline preparation-conditioned primal error correction and enriched
reduced-adjoint certificate.  Its central decomposition writes the exact
state error as a represented correction plus an unresolved remainder.  The
global spectral width then multiplies only the squared unresolved remainder,
while output-relevant linear error is evaluated through direct Hermitian goals
and reduced adjoints.  No exact driven trajectory enters this construction.

The displayed finite-dimensional residual and Duhamel identities have been
checked for sign and consistency against the existing PURG conventions.  The
amendment does not, however, fully specify validated floating-point linear
algebra for its requested formal outward-rounded certificate.  Implementation
therefore begins with a separately labeled numerical a posteriori version on
the frozen rank-96 and rank-128 spaces.  Exact scorer gates remain closed.  A
new module and focused tests implement the primal correction equation, its
actual continuous-extension residual, the unresolved-error Duhamel bound, and
the shifted quadratic-goal interval before any larger rank/cutoff ladder is
built.

### Frozen rank-128 numerical certificate pilot

The first no-exact-data pilot keeps the online rank-128 PURG model fixed and
uses a rank-192 primal-correction space plus a rank-256 enriched dual space.
The spaces are preparation/residual/goal conditioned; the dual candidates
include terminal direct-goal directions, Hamiltonian leakage of their projected
directions, and correction-residual directions.  The pilot uses step 0.01 and
scores every 0.04, so it is not an authoritative amended Gate-B run.

The primal-correction residual identity holds to
\(3.76\times10^{-17}\).  Its final unresolved-state bound is 0.0113394.  The
cheap-plus-forward-DWR numerical derivative bounds are much tighter than the
rejected global operator-norm certificate:

| block | old rank-128 RMS bound | amended-pilot RMS bound | amended budget |
|---|---:|---:|---:|
| \(\rho\) | 0.103265 | 0.0218113 | 0.0001 |
| \(B\) | 0.456332 | 0.00882125 | 0.0025 |
| \(N\) | 2.53440 | 0.0160796 | 0.0025 |
| \(A\) | 3.53543 | 0.0319400 | 0.0025 |
| \(C\) | 1.82285 | 0.0496332 | 0.025 |

Thus the new output geometry is materially sharper, especially for \(C\), but
this frozen level still fails every registered block budget.  The uniform
dual-leakage envelope controls most of the remaining width.  A columnwise
backward adjoint for the most threatened electronic goal at \(t=4\) reduces
its coordinate bound from about 0.0345 to about 0.0102, but this remains far
above the \(10^{-4}\) electronic budget.  Promoting the rank-256 dual space to
the next correction level lowers the unresolved-state bound to 0.0086562; a
rank-320 leakage-enriched dual lowers the same explicit-adjoint direct radius
only to about 0.00909 before its bilinear term.  These are exploratory master-
level diagnostics, not a completed fixed-rank ladder or formal rejection of
the amendment.

Artifact:

- `output/local_runs/paper_v_purg_goal_certificate_pilot_cutoff16_20260803_v3/`

No exact driven scorer was opened.  The old failed Gate-B artifact and result
remain unchanged.

## 2026-08-03: post-pilot PURG decision and sealed-score route

The returned post-pilot decision supersedes the pilot's prospective ladder.
Do not build the rank-192/224/256 acceptance ladder and do not invest in
theorem-level interval linear algebra for this proposal.  The existing
rank-128 basis is frozen byte-for-byte from
`paper_v_purg_construction_gate_cutoff16_20260803_v2`.  The only permitted
new basis is one blind rank-160 audit formed by appending exactly 32
twice-reorthogonalized, time-tie-broken residual pivots from the rank-128
reduced path.  Rank 160 is audit-only and cannot rescue or replace rank 128.

The rank-128/rank-192/rank-256 pilot is now classified as a failed numerical
preflight for investment in the registered *formal certificate*.  Its
correction and dual intervals remain archived numerical diagnostics but are
retired from model acceptance.  The reported rank-256/rank-320 continuation
is prose-only exploratory evidence: no corresponding array or run artifact
exists, and it must not be rerun merely to manufacture provenance.

Canonical historical hashes are:

- construction-v2 `summary.json`:
  `eecaf86f2a74c9ddd6bc1de0c0a10c2b005ea0dbd27647acf80d94620fb3ead6`;
- construction-v2 `arrays.npz`:
  `6a8aa9847f36e6edea3f72ccc7081e15b069b73cbee8608e503be0ac0344111b`;
- construction-v2 `manifest.json`:
  `ed65aef62b92c1743e15ecaf07405eb16f88fe0715d0844aeb3758024712eccc`;
- pilot-v3 `summary.json`:
  `e5af35e5a0fde1d0b73d675ccb93d14e63d5058a34a417c7e3fd8b6ccc0b1700`;
- pilot-v3 `arrays.npz`:
  `3ec4b4393f1da2a43a611aea608832537676602ca7cf714f34994104d2d9104e`;
- returned post-pilot decision text:
  `4a5029d2c5c77809bb3edb8fc4a401db36df3d84dcd22e8efef18d46612f1fbb`.

The new implementation is intentionally separate from the retired
goal-certificate code.  Its construction command verifies and hashes the
frozen rank-128 artifact, builds the blind rank-160 audit, and writes a
complete pre-scorer manifest before any full driven reference is generated.
The scoring command then verifies that manifest and compares the full,
rank-128, and rank-160 paths with exponential midpoint at both registered
steps and tolerances plus independent DOP853.  Numerical resolution and norm
gates are evaluated first.  Exactly one fixed refinement fallback is allowed;
scientific output/derivative and quarter-budget rank-difference gates are
evaluated only after numerical consistency.  A pass serializes rank 128 only;
either stop serializes no model.

The permitted conclusion remains narrowly scoped:

| initialization | retained variables | protocol | permitted conclusion |
|---|---|---|---|
| correlated cutoff-16 ground ket | the 31 real slots for \(\rho,B,N,A,C\), including the frozen 14-real \(C\) packing | the current pulse on \(0\le t\le4\), scored at 1601 nodes | A pass supports only empirical, independently resolution-checked rank-128 fidelity at this preparation, cutoff, pulse, and horizon.  It is not a formal certificate, cutoff-convergence result, off-grid supremum, long-time claim, or universal closure statement. |

At the time of this entry, the full driven scorer remains unopened.  The next
state-changing step is the blind construction and immutable pre-scorer freeze;
the expensive sealed scorer is a separate production action.

### Blind construction and sealed-score outcome

The implementation was verified by 10 dedicated sealed-score tests, the 12
existing focused PURG/certificate tests, and the complete Paper V suite
(147/147 passed).  The canonical construction-v2 artifact was then reloaded
and its rank-128 compressed data were reproduced exactly before the blind
append.

The blind construction accepted exactly 32 residual pivots.  Its rank-160
spectral orthogonality residual was (4.49\times10^{-15}), and its rank-128
nesting residual was (4.45\times10^{-15}).  The immutable pre-scorer record
is:

- artifact:
  `output/local_runs/paper_v_purg_blind_w160_pre_scorer_cutoff16_20260803_v1/`;
- pre-scorer manifest SHA-256:
  `2c4bdbc74a9cc3eb6153930688ca95a6e1285a48dc02f5df48afc220cb94fcf5`;
- frozen rank-128 content SHA-256 under the manifest's typed-array convention:
  `d275bcac9c109da2fc2ffe586db8e1fc3f468a7dac72484f7ba14ee41dd0dfc9`;
- blind rank-160 content SHA-256:
  `405f5e19172b8998ae535ebded798d378535b715d3cc116f407f67b49cad20c3`.

The subsequently opened scorer failed the initial numerical-resolution gate
in nine electronic metrics, so the single registered refinement fallback was
activated.  After refinement, only three failures remained: the separately
resolved derivative-\(\rho\) RMS estimates for rank 128, rank 160, and the
rank difference were approximately (1.26317\times10^{-6}),
(1.26313\times10^{-6}), and (1.26308\times10^{-6}), respectively, against
the fixed (1.0\times10^{-6}) numerical gate.  Maximum norm-ratio drift over
all paths was below (5.7\times10^{-14}); tolerance-repeat discrepancies were
at roundoff scale.  The remaining failure was controlled by the prescribed
fine/coarse step comparison.

Status: `indeterminate_numerical_stop`.  In accordance with the frozen order,
no provisional scientific output or derivative classification is reported,
rank 160 does not rescue rank 128, and no model was serialized.  This is not a
scientific failure of PURG and implies no closure conclusion.  The fixed
fallback has been consumed; no same-reference rescore, rank change, threshold
change, or basis adjustment is permitted.

Sealed-score artifact:

- `output/local_runs/paper_v_purg_sealed_score_cutoff16_20260803_v1/`;
- `score_summary.json` SHA-256:
  `e6f6172ea04d5f41a28673f3c691caa53394c9a8b1c9f893c7f220019894d4a0`;
- `score_arrays.npz` SHA-256:
  `45cd42434b4ddb6a8c4aff809bc7211b6deb732347cdf5a351fe361898c87168`.

A post-run read-only implementation audit independently reconstructed the
controlling fallback value as the required sum of the full-space and reduced
step discrepancies, (6.31608\times10^{-7}+6.31565\times10^{-7}).  It found
no error in the grid restriction, RMS convention, or triangle-safe resolution
formula.  Three prospective sealing controls were then hardened without
reopening the scorer: future loads verify the frozen Python/NumPy/SciPy/thread
environment and score config; a prepared manifest is atomically consumed
before reference propagation; and non-finite coordinate paths are rejected
instead of reaching comparisons.  An explicitly after-the-fact consumption
receipt in the pre-scorer directory ties the already completed v1 score hashes
to the consumed manifest.  These changes do not modify or reclassify the v1
numerical stop.  The final hardened suite passes 149/149 tests.

## 2026-08-04 abandonment decision and multi-coherent pre-seal audit

The returned theoretical decision memorandum has SHA-256
`5b702147278cb5035e930cc79d3dc04721f18e14e8b4aaedf5a0c410cb9d6a59`.
It separates broad exact instantaneous closure from protocol-specific model
selection.  The finite-cutoff witness rules out a universal exact deterministic
velocity depending only on `(rho, B, N, A, C)` on any state class containing
the witness pair.  The accumulated representability, derivative, and matched-
sensitivity evidence supports retiring the present 31-coordinate archive drift
as the primary propagator for the tested correlated-initialization,
anomalous-retaining strong-coupling protocol.  Narrow preparation-manifold and
finite-memory models remain open.  The selected next representation is an
adaptive electron-conditioned multi-coherent ket, with the 31 moments retained
as contracted outputs; capped MPS--TDVP with a dynamical phonon basis is the
sole ranked fallback.  The PURG result remains
`indeterminate_numerical_stop`.

A read-only pre-seal audit found that the existing multi-coherent code is a
useful development implementation but is not yet the frozen model specified by
the memorandum:

- it removes the center mode and propagates electron-conditioned coherent
  packets only in the relative mode; under the local cutoff the recorded
  center/relative factorization is `0.9999999306128844`, so the reduction is
  numerically excellent but not an exact finite-cutoff identity;
- its current tangent least-squares solve uses the raw, unnormalized tangent
  matrix rather than the memorandum's explicitly normalized horizontal tangent;
- one adaptive spawn adds a zero-weight packet to every electronic branch, so
  the existing six-packets-per-branch run has 24 branch packets and 96 raw real
  coordinates; the memorandum instead states a 16-total-packet, 96-coordinate
  two-mode cap;
- the current driver implements the original single pulse and its comparison
  reports a degree-four 82-coordinate hierarchy norm, not yet the proposed
  double-pulse 31-output, fidelity, sensitivity, work, and cost scores.

The existing development evidence remains unchanged: cutoff 16, four initial
and six final packets per branch, maximum active tangent rank 84, minimum
fidelity `0.9904161075726595`, maximum hierarchy-coordinate relative error
`0.09116862456704118`, maximum relative tangent residual
`0.9279140922375625`, and maximum norm drift `0.0002008003384174062` through
`t=20`.  The focused multi-coherent suite passes 10/10 tests.  No double-pulse
exact holdout was generated or opened during this audit.  The next action is to
repair and validate the model-internal geometry, capacity convention, drive
interface, and scorers on already-open development and analytic controls before
freezing any prospective holdout.

## 2026-08-04 horizontal multi-coherent implementation and open development

The pre-seal implementation now uses the normalized projective ket and the
horizontal real-coordinate tangent
`(I - |psi><psi|) d|phi>/||phi||`.  The McLachlan solve therefore removes the
norm and global-phase gauge directions before regularization.  Parameter
states are deterministically retracted after initialization and every accepted
segment: the reconstructed ket has unit norm and the largest coefficient fixes
the global phase.  Finite-cutoff coherent packets are evaluated without their
common Gaussian prefactor, which cancels under normalization and otherwise
underflows for large displacements.

The model interface now also supplies:

- an analytic causal Gaussian-sine drive object, including delayed pulse sums
  and their time derivative;
- explicit capacity counts (packets per electronic branch, total branch
  packets, and raw real coordinates);
- contraction of every normalized relative-mode ket to the established 31-real
  `(rho, B, N, A, C)` packing;
- an exactly model-representable symmetric tangent pair generated from the
  preparation-normalized sum of the electronic-drive and relative-position
  generators;
- stored energy, external power, and the integrated work residual;
- reference-only 31-block errors, five-block sensitivity amplification, and
  numerical-resolution certificates implementing `u_Q <= 0.1 b_Q` and
  `Q + u_Q <= b_Q`.

The delayed-drive exact comparison is deliberately rejected by the ordinary
development runner.  It must pass through a future manifest-consuming scorer,
so no double-pulse exact holdout has been generated or opened.

Open single-pulse cutoff-16 development runs were repeated after the horizontal
geometry repair.  With adaptive growth from four to at most six packets per
electronic branch, the completed results are:

| tangent regularization | horizon | minimum fidelity | maximum 31-coordinate relative error | maximum normalized work residual |
|---|---:|---:|---:|---:|
| Tikhonov `3e-3` | 4 | `0.9994207388` | `0.0023758099` | `4.459285e-4` |
| Tikhonov `1e-3` | 4 | `0.9996529305` | `0.0010935870` | `4.459169e-4` |
| Tikhonov `3e-4` | 4 | `0.9997487022` | `0.0018212414` | `4.459151e-4` |
| Tikhonov `3e-3` | 20 | `0.9849213766` | `0.0526518164` | `4.459285e-4` |
| Tikhonov `1e-3` | 20 | `0.9872288657` | `0.0288549001` | `4.459169e-4` |
| Tikhonov `3e-4` | 20 | `0.9945500682` | `0.0220532612` | `4.459150e-4` |

The `3e-4` run completed at maximum active tangent rank 94, maximum parameter
speed `37.5643`, and roundoff-level norm drift.  It is the only tested
horizontal parameterization that clears the open `0.99` fidelity target at
`t=20`, so it is the retained development choice.  The nearly unregularized
truncated-SVD development attempt at relative cutoff
`1e-8` became numerically non-finite before completing `t=4`; it is rejected as
a parameterization.  The finite Tikhonov comparison was limited prospectively
to the three values above.

One cap translation required explicit resolution before sealing.  The exact-center
relative-mode implementation has four real coordinates per branch packet, so
six packets in each of four electronic branches give 24 branch packets and 96
raw coordinates.  The memorandum's unreduced two-mode counting assigns six
real coordinates per packet and states 16 total packets and 96 raw coordinates.
The raw-coordinate limits coincide, but the literal packet limits do not.  The
implemented holdout contract therefore records both native counts explicitly
and retains the 96-coordinate cap; it does not label 24 branch packets as 16.

### Pre-seal contract and scorer implementation

The retained `3e-4` development run was repeated without changing its model
or integrator on a denser `0.05` stored-output grid.  It completed with minimum
fidelity `0.9945435216`, maximum 31-coordinate relative error
`0.0220959109`, geometric tangent rank 94 at the declared Gram threshold
`1e-10`, and maximum normalized work residual `4.80754e-6`.  The dense path is
`output/local_runs/paper_v_multi_coherent_horizontal_open_dev_cutoff16_t20_tikh3e4_dense_20260804_v1/`.
The dense exact single-pulse contractions now determine the prospective
coordinate scales.  Their minimum and maximum values are `0.016` and
`1.9202293828`.

The complete pre-seal software path is implemented, while the real holdout
reference remains unopened:

1. The model-input freeze stores the exact-correlated central preparation, the
   normalized horizontal chart, the symmetric model-tangent pair, the dense
   development scales, every scientific and numerical gate, implementation
   source hashes, and the Python/NumPy/SciPy/thread environment.
2. The symmetric pair uses the projected, preparation-normalized sum of the
   electronic-drive and relative-position generators.  On the current
   development preparation its five-block initial distance is
   `1.0121308947e-4`; its projected physical kick norm is `0.9999987336` and
   its projection residual is `1.93158e-4`.
3. The exact-center implementation's native hard limits are recorded as six
   packets per each of four electronic branches, 24 total branch packets, 96
   raw real coordinates, and geometric tangent rank at most 96.  This is an
   explicit translation of the memorandum's two-mode 96-coordinate budget,
   not a relabeling of 24 packets as 16.
4. Six reference-blind model trajectories are required: central, plus, and
   minus at frozen coarse and fine numerical settings.  Three additional
   fine-central repetitions provide the model wall-time and resident-memory
   baseline.  Every run is audited for source separation, completion,
   physicality, work balance, rank, and capacity before the model batch can be
   sealed.
5. The one-shot scorer atomically consumes the sealed model batch before
   constructing any driven exact reference.  Independent adaptive DOP853 and
   unitary exponential-midpoint ket batches must first satisfy mutual
   infidelity and 31-output consistency.  One prospectively fixed exact
   refinement is available.  Only then are fidelity, electronic trace
   distance, block RMS/maximum error, sensitivity amplification, work, and
   model/direct cost gates interpreted.

The long-horizon runner now separates checkpoint segments from stored-output
sampling, reports the memorandum's geometric Gram rank independently of the
Tikhonov effective rank, and records process wall time and maximum resident
set size.  The reference-only scorer applies full coarse/fine and
two-propagator score spreads, robust upper bounds, and a one-shot consumption
receipt.  A cutoff-coordinate audit of the opened dense development path found
that projection of the center/relative product back into the local cutoff-16
space retained norm between `0.9996335865` and `0.9999968233`; this quantity is
reported by the sealed scorer and is not used to tune the model.

The pre-seal provenance audit also tightened the cost contract.  Each of the
three timed model repetitions must now prove that it used the frozen central
initial state and fine numerical settings.  Its runtime source and artifact
hashes are checked before timing data can be sealed, and the resulting cost
manifest is self-hashed and revalidated before either the six-run model batch
or scorer can load it.  Tampering tests cover both a false fine-run claim and
a modified timing manifest.  The focused multi-coherent suite passes 33/33
tests, the complete Paper V suite passes 174/174 tests, and `compileall` plus
the holdout CLI smoke check pass.  The real double-pulse exact holdout remains
unopened.

## 2026-08-04 frozen double-pulse batch and one-shot exact score

The six required reference-blind model trajectories completed through
`t=100`: central, plus, and minus initial conditions at both coarse and fine
settings.  Every blind audit passed.  The three isolated fine-central timing
repetitions took `634.2998361`, `737.4425812`, and `664.7038699` seconds, giving
a frozen median of `664.7038699` seconds; the maximum resident set size over
the repetitions was `117424128` bytes.  No additional model timing was run.

The timing records and model outputs were then frozen before the driven exact
reference was opened.  The principal frozen identifiers are:

- prepared-model manifest: `f9d2788c78d843550dcc7d3c607f205d29633bcd282ef32ed8d6176ed3c4bd80`;
- model-cost manifest identifier: `8c55303a604d1f8e75d22a2d85d649d210287537c862898a22e39babe4c54be6`;
- six-run model-batch manifest identifier: `b630403171029032cefe3655c2700c0714a1ad24c4b667ac9998f9533ee5398c`;
- one-shot consumption-receipt file hash: `e8c649a6175b0b4a04a2b8d525cde7d30495fb1213f9513a8cb3cd96f72304bf`.

The one-shot exact scorer was consumed exactly once.  Its initial DOP853 versus
unitary exponential-midpoint comparison had maximum mutual infidelity
`1.59965e-11` and maximum contracted-moment disagreement `4.22020e-6`.  The
single prospectively allowed refinement reduced these to `1.00053e-12` and
`1.05505e-6`, respectively.  The ket criterion therefore passed its frozen
`1e-9` ceiling, but the contracted-moment disagreement remained approximately
`5.5%` above its frozen `1e-6` ceiling.  The scorer consequently returned
`indeterminate_reference_stop`, issued no scientific pass or failure, and did
not evaluate the fidelity, observable-error, sensitivity, or cost gates.  No
threshold was changed and no rescore was attempted.  The selected score also
recorded minimum local-cutoff retained norms of `0.9949306843` on the coarse
model path and `0.9961432076` on the fine path.

This indeterminate stop leaves the separate result in
`paper_5/notes/corrected_archive_eom_chaotic_sensitivity_20260804.md`
unchanged: the representability-corrected archive moment field exhibits strong
closure-generated sensitivity while the matched exact trajectories remain
close, and the controller slightly suppresses rather than creates that
separation.  The sealed multi-coherent holdout was designed to test whether the
replacement ket model removes that excess sensitivity, but the exact-reference
agreement gate stopped the score before that comparison could be interpreted.

Frozen artifacts:

- model batch: `output/local_runs/paper_v_multi_coherent_double_pulse_blind_model_cutoff16_20260804_v1/`;
- consumed score: `output/local_runs/paper_v_multi_coherent_double_pulse_sealed_score_cutoff16_20260804_v1/`.

### Stored-array observable postmortem

No additional propagation or timing was performed after the consumed score.
The stored score arrays were analyzed in the same archive Eq. (22) observable
decomposition used by Figure 3 of the advisor divergence report: site
occupation, electronic energy, phonon energy, electron--phonon energy, and
total internal energy.  This comparison is natural because the multi-coherent
route contracts its variational ket to the same 31-coordinate
`(rho, B, N, A, C)` output space.  It does not integrate or correct the archive
moment vector field; it replaces that closure with a McLachlan projection of
the Hamiltonian ket velocity onto an adaptive electron-conditioned coherent
packet manifold.

The exact-reference gate miss was localized without rerunning either exact
solver.  The maximum equal-five-block distance occurs on the minus member at
`t=46.1` and equals `1.0550496e-6`.  At that sample, the largest root
contributions come from the anomalous-phonon `A` block (`1.77042e-6`) and the
electron--phonon `C` block (`1.40511e-6`), followed by `B`, `N`, and `rho`.
The largest individual scale-normalized coordinate discrepancy occurs instead
on the central member at `t=48.7` in
`correlation_1_diag_difference_imag`; its raw difference is only
`2.89095e-7`.  The stop is therefore a distributed late-time contracted-moment
resolution issue rather than one catastrophic coordinate or an early-pulse
failure.

For the central trajectory over the full stored interval `0 <= t <= 100`, the
fine multi-coherent path has time-RMS errors against DOP853 of `0.02261` in site
occupation, `0.07220` in electronic energy, `0.06672` in phonon energy,
`0.09783` in electron--phonon energy, and `0.001231` in total internal energy.
The corresponding maximum absolute errors are `0.10016`, `0.29964`, `0.26604`,
`0.46560`, and `0.003437`.  The small total-energy error coexists with much
larger component-energy errors because those errors cancel.  Coarse/fine
differences also become visible late in the trajectory, so the fine path does
not uniformly dominate the coarse path in every component.

The two exact observable curves differ by at most `4.61e-7` in site occupation
and `1.89e-6` among the component energies, far below the model-reference
differences and visually indistinguishable on the trajectory scale.  The plot
is nevertheless classified `exploratory_consumed_score_not_promoted` because
the frozen exact moment-consistency gate was not cleared.  It supports a
qualitative observable diagnosis but does not create a post-hoc scientific
pass.

The new double-pulse plot cannot by itself establish improvement over the
archive Figure 3 trajectories: the protocols and horizons differ.  A direct
method comparison would require the raw archive closure, its physicality
controller, the multi-coherent model, and the exact Hamiltonian to share the
same initial state, double-pulse drive, output grid, and horizon.

Postmortem artifacts:

- observable figure: `output/pdf/paper_v_multi_coherent_double_pulse_observables_20260804.pdf`;
- metrics and input hashes: `output/pdf/paper_v_multi_coherent_double_pulse_observables_20260804.json`;
- reproducible stored-data command: `pipelines/open_dynamics/analyze_multi_coherent_sealed_observables.py`.

## 2026-08-04 trajectory-local closure identifiability audit

No propagation, timing experiment, or scorer reopening was performed.  The
consumed double-pulse score already stores three cutoff-16 exact trajectories
and their independent DOP853 and exponential-midpoint ket references.  At
every fifth stored sample (`Delta t=0.25`, 1203 member-time samples), the
analytic Schrodinger velocity was contracted into the exact centered
electron--phonon-correlation derivative.  The established autonomous
same-spin Pauli repair was subtracted from that derivative, defining the
remaining source that an improved `C` closure must supply.  A focused test
compared the new batched contraction with the established scalar exact
contraction at `2e-13` absolute and relative tolerance.

The broad finite-cutoff witness already rules out a globally exact
instantaneous source depending only on the 31 retained coordinates.  The new
audit asked the narrower preparation-manifold question.  Cross-time nearest
neighbors were required to be separated by at least four hopping-time units,
so adjacent samples on a smooth trajectory could not masquerade as closure
prediction.  An inverse-distance eight-neighbor predictor from the current 31
moments had normalized source RMS error `0.991996`.  The independent exact
references differed by only `2.90582e-6` on the same normalization.  The
nearest retained-state distances were not asymptotically small (minimum
`0.106854`, median `0.177961` in the frozen equal-five-block metric), so this
is evidence against a simple local interpolation on the available trajectory
coverage, not a second non-identifiability theorem.

A causal one-lag history feature was then tested on the same stored data.  The
best declared lag was one hopping-time unit: its normalized prediction error
was `0.862851`, versus `0.998718` for the current moments on the matching
trimmed samples.  The approximately 14% reduction is real but leaves most of
the source variation unexplained.  One short lag therefore does not supply a
usable closure.

The missing source itself is much more compressible than it is predictable
from the current moments.  A development-scaled singular-vector basis trained
only on the central exact trajectory showed that five real source modes
explain `0.99975425` of its training variance and reconstruct all three exact
members with normalized RMS residual `0.0156817`.  Ten modes reconstruct the
source to numerical precision; the other four of the nominal 14 coordinates
are removed by the fixed-sector structural relations.  This identifies a
specific candidate state augmentation: add a small latent source directly to
the `C` equation rather than promoting every raw third- and fourth-order
moment.  It does not yet provide an autonomous evolution equation for those
latent coordinates.  Exploratory nearest-neighbor checks of the five-mode
source velocity from the augmented `(31 moments, source modes)` state did not
improve prediction, so no latent model has been serialized or propagated.

The closure-development decision is therefore:

- do not add another instantaneous 31-coordinate regression;
- retain the five-mode missing-source subspace as a compact target for a
  future causal latent or memory model;
- require that any such model predict the latent-source evolution on held-out
  preparations or drives before it is coupled back into the archive EOM;
- continue to treat exact or multi-coherent kets as offline sources of the
  omitted correlations, not as online inputs to an archive closure.

Artifacts and verification:

- analysis arrays, metrics, manifest, and figure:
  `output/local_runs/paper_v_trajectory_closure_identifiability_cutoff16_20260804_v1/`;
- implementation:
  `paper_5/src/paper5/stability/trajectory_closure_identifiability.py` and
  `pipelines/open_dynamics/analyze_trajectory_closure_identifiability.py`;
- focused tests: 9/9 passing, including the pre-existing finite-cutoff witness,
  metric equivalence, exact derivative contraction, causal-history behavior,
  and low-rank-source recovery; complete Paper V suite: 182/182 passing.

## 2026-08-04 derivative-first latent-source gate

No trajectory, timing experiment, exact solver, or consumed scorer was rerun.
This gate used the dense stored exact-source audit at `Delta t=0.05` to test
whether the five-mode missing source can be given a compact autonomous local
dynamics before it is coupled back into the archive equations.  The proposed
state has 41 real coordinates: the 31 retained moments, five source amplitudes
`z`, and their five rates `p`.  Its fitted latent equation is

```text
dot(z) = p
dot(p) = c + B_x x + A_z z + A_p p + B_v V(t).
```

The five source directions were fixed from the central exact member over
`0 <= t <= 8`; they reconstruct the source over all three members and the full
stored interval with normalized RMS residual `0.0163831`.  Five-fold blocked
time validation on the central member selected ridge penalty `1.0` from the
frozen candidate set.  Its normalized acceleration RMS was `0.452799`.  The
model was then fitted to the central member and scored without refitting on the
nearby plus and minus preparations.  Their full-interval normalized
acceleration errors were `0.345845` and `0.345815`, respectively; the central
value was `0.345580`.  The corresponding second-pulse errors were approximately
`0.3744`.

The unconstrained homogeneous latent fit had maximum pole real part
`+0.0327330`.  The declared analytic pole shift moved every homogeneous pole
left by `0.0427330`, giving maximum real part `-0.01`; only the forcing terms
were then refitted.  This stability operation increased the derivative error
only modestly.  Numerical differentiation is not the limiting error: changing
the stored grid from `0.05` to `0.1` changed the acceleration by normalized RMS
`0.00256805`, while the two independently generated exact references differed
by `3.05690e-6`.

All five exploratory derivative gates passed, but this does not establish an
accurate or stable propagated closure.  It establishes the narrower result
that a stable second-order five-mode latent equation captures substantially
more of the missing-source evolution than an instantaneous function of the 31
moments, with approximately one third of the normalized acceleration variation
still unresolved.  The preparation holdouts are deliberately local and use
the same drive, so protocol generalization is also untested.  Exact states are
used only to fit and score the frozen coefficients; an eventual online model
would evolve `z` and `p` without access to an exact trajectory.

Artifacts and verification:

- derivative-gate arrays, metrics, manifest, and diagnostic figure:
  `output/local_runs/paper_v_latent_source_closure_derivative_gate_cutoff16_20260804_v1/`;
- implementation:
  `paper_5/src/paper5/stability/latent_source_closure.py` and
  `pipelines/open_dynamics/analyze_latent_source_closure.py`;
- focused latent-model tests: 6/6 passing; complete Paper V suite: 188/188
  passing.

## 2026-08-04 autonomous latent-source propagation gate

The frozen second-order five-mode model was coupled back into the archive
equations and propagated through `t=20` under the same double-pulse protocol.
The online state contained the 31 retained moments, five missing-source
amplitudes, and their five rates.  At each of 8,000 RK4 stage evaluations, the
reconstructed source was added only to the `C`-velocity block before the
existing Euclidean minimum-norm representability controller was solved.  A
matched baseline used the autonomous same-spin Pauli repair and controller but
no latent source.  Neither online right-hand side queried the exact trajectory;
the stored cutoff-16 reference was used after both runs for scoring.  The
latent amplitudes and rates were initialized from exact preparation data, so
this test already gives the candidate a favorable initialization.

Both lanes remained physical at the sampled states and all 8,000 constrained
solves in each lane converged.  The latent lane retained a minimum sampled
joint-Gram eigenvalue of `1.04873e-5`, a maximum correlation-trace residual of
`3.35e-16`, and a post-second-pulse internal-energy range of `3.12e-6`.  The
controller therefore contained the candidate trajectory without hiding an
optimization or representability failure.

The autonomous accuracy result was negative.  Relative to the matched
Pauli-plus-controller baseline, the latent model changed the principal errors
as follows:

| score through `t=20` | baseline | latent | latent / baseline |
|---|---:|---:|---:|
| missing-source normalized RMS | `1.35240` | `1.37545` | `1.01705` |
| normalized `C`-block time-RMS | `0.298615` | `0.590635` | `1.97792` |
| equal-five-block time-RMS distance | `0.216655` | `0.680091` | `3.13905` |
| site-occupation time-RMS error | `0.0682803` | `0.0977501` | `1.43160` |
| internal-energy time-RMS error | `0.0296273` | `0.0403599` | `1.36225` |

The phonon-energy RMS alone decreased from `0.245305` to `0.221237`, but this
isolated redistribution does not offset the larger electronic, correlation,
and complete-state errors.  The latent controller was active in `93.98%` of
stage evaluations, versus `88.65%` for the baseline, while its RMS correction
norm was slightly smaller (`0.13567` versus `0.13771`).  The controller is not
the immediate failure mode.

This result separates source compression from autonomous closure.  Five modes
represent the exact-path missing source accurately, and the fitted stable
second-order law has moderate exact-path derivative error, but its rollout
leaves the training path and supplies an increasingly incorrect source to the
coupled moment equations.  Closed-loop distribution shift and structural
model error therefore defeat the affine latent oscillator.  Further ridge,
pole-shift, or integrator tuning is not justified before reconsidering the
latent dynamics mathematically.  The next external-theory request should ask
whether the five-dimensional source subspace can support a passive,
energy-compatible memory realization, requires state-dependent/nonlinear
dynamics or constrained training on rollouts, or should be abandoned in favor
of another physically adapted closure.

Artifacts and verification:

- arrays, metrics, and runtime manifest:
  `output/local_runs/paper_v_latent_source_closure_t20_gate_cutoff16_20260804_v1/`;
- implementation:
  `paper_5/src/paper5/stability/latent_source_propagation.py` and
  `pipelines/open_dynamics/run_latent_source_closure_gate.py`;
- focused propagation tests verify `C`-only source injection, controller order,
  and all four RK4 stage evaluations; complete Paper V suite: 190/190 passing.

## 2026-08-04: literal `g=2` coupling-quench controller stress test

A matched raw-versus-corrected 31-coordinate diagnostic was run at the literal
Holstein coupling `g=2`.  With `t_hop=1` and `omega_ph=0.5`, this corresponds
to `lambda_ep=16`.  The initial state was held fixed at the validated central
correlated cutoff-16 contraction from the `lambda_ep=1.5` calculation.  This
is therefore a sudden-coupling-quench test of the archive vector field and
representability controller, not a converged `g=2` ground-state or exact-
trajectory accuracy comparison.  The latter would require a substantially
larger phonon cutoff because the equilibrium displacement is extreme.

Both lanes used the double Gaussian-sine pulse at delays zero and eight and
fixed-step RK4 through `t=20`.  The raw lane used the archive 31-coordinate
EOM.  The corrected lane used the same raw velocity followed at every RK4
stage by the Euclidean minimum-norm electronic/joint-Gram, correlation-trace,
and zero-correction-energy-flux controller.

At step `0.01`, the raw trajectory first had a negative sampled joint-Gram
eigenvalue at `t=0.10`, a negative bosonic-moment eigenvalue at `t=0.30`, and
a negative electronic-density margin at `t=7.60`.  Its largest coordinate
crossed the declared `1e4` threshold at `t=19.0`, reaching `10899.0` on that
sample.  The corrected lane completed `t=20`; all `8000` constrained stage
solves converged, the controller was active at every stage, the largest
coordinate was `8.82412`, the maximum correlation-trace residual was
`3.35e-16`, and the post-second-pulse internal-energy range was `1.01e-7`.
The minimum sampled joint-Gram eigenvalue was `2.89208e-5`, attained at the
common initial state.

The complete run was repeated at step `0.005`.  The raw lane crossed the same
three sampled physicality boundaries at `0.10`, `0.30`, and `7.60` and crossed
the coordinate threshold at `t=19.0`.  All `16000` corrected stage solves
converged.  Between the two corrected trajectories, the maximum absolute
coordinate difference on the common sample grid was `1.70e-5`, the maximum
coordinate-vector l2 difference was `3.55e-5`, and the final l2 difference was
`2.61e-5`.

This establishes step-converged representability containment for this one
extreme coupling quench.  It does not establish physical accuracy at `g=2`,
because no cutoff-converged exact reference or `g=2` equilibrium preparation
was used.

Artifacts and verification:

- step `0.01`: `output/local_runs/paper_v_high_coupling_g2_raw_vs_corrected_t20_20260804_v1/`;
- step `0.005`: `output/local_runs/paper_v_high_coupling_g2_raw_vs_corrected_t20_dt005_20260804_v1/`;
- driver: `pipelines/open_dynamics/run_high_coupling_controller_comparison.py`;
- observable plot: `output/local_runs/paper_v_high_coupling_g2_raw_vs_corrected_t20_20260804_v1/observables_raw_vs_corrected.png`;
- plot generator: `pipelines/open_dynamics/plot_high_coupling_controller_observables.py`;
- complete Paper V suite: 190/190 passing.

## 2026-08-04: weak-regime corrected control through `t=100`

The raw and representability-corrected 31-coordinate EOMs were propagated at
the established weak point `lambda_ep=0.5`, `gamma=0.5`, corresponding to
`g=0.3535533905932738`.  Unlike the preceding `g=2` quench, both lanes began
from the exact cutoff-16 zero-field ground-state moment contraction for this
same weak parameter point.  The double Gaussian-sine pulse, fixed RK4 step
`0.01`, sample step `0.05`, and controller settings were held fixed.

The raw trajectory remained amplitude-bounded through `t=100`, with maximum
absolute coordinate `1.05616`, but it did not remain representable.  Its
minimum sampled joint-Gram, bosonic-moment, and electronic-density margins
were respectively `-0.958162`, `-0.652550`, and `-0.122453`; its maximum
correlation-trace residual was `0.0560027`.

The corrected trajectory completed `t=100` with maximum absolute coordinate
`0.960045`.  All `40000` constrained RK4-stage solves converged.  The minimum
sampled joint-Gram and bosonic-moment eigenvalues were `1.02e-13` at the nearly
singular common initial state, the electronic margin remained above
`0.00823686`, and the maximum correlation-trace residual was `1.73e-16`.  The
post-second-pulse internal-energy range was `2.27e-5`.  The correction-vector
RMS and maximum norms were `0.0222213` and `0.0528900`; maximum instantaneous
correction-energy flux was `6.09e-17`.

This establishes long-horizon representability containment at the weak point,
including a case where coordinate boundedness alone would have hidden a large
physicality failure.

A matched exact cutoff-16 wavefunction was then propagated through the same
double pulse with adaptive DOP853 (`rtol=1e-10`, `atol=1e-12`, maximum step
`0.05`) and contracted on the same sample grid.  The exact initial contraction
agreed with the controller-run initial state to `6.54e-14` in coordinate l2,
and the maximum exact state-norm error was `7.49e-13`.  Relative to this
reference, the correction reduced the complete 31-coordinate l2 time-RMS
distance from `0.585740` to `0.263960` (`54.9%`).  Observable time-RMS errors
changed as follows:

| observable | raw | corrected | reduction |
|---|---:|---:|---:|
| either site occupation | `0.250814` | `0.161536` | `35.6%` |
| electronic energy | `0.105957` | `0.0494000` | `53.4%` |
| phonon energy | `0.0501221` | `0.0388800` | `22.4%` |
| electron-phonon energy | `0.114639` | `0.0563757` | `50.8%` |
| total internal energy | `0.0483343` | `0.0113783` | `76.5%` |

The controller therefore improves all six scored weak-regime observables and
the complete retained state, but the nonzero residual errors confirm that
representability stabilization does not repair the approximate closure.

Artifacts:

- `t=20` precursor: `output/local_runs/paper_v_weak_lambda05_raw_vs_corrected_t20_20260804_v1/`;
- `t=100` run: `output/local_runs/paper_v_weak_lambda05_raw_vs_corrected_t100_20260804_v1/`;
- exact score: `output/local_runs/paper_v_weak_lambda05_raw_vs_corrected_t100_20260804_v1/exact_metrics.json`;
- observable plot: `output/local_runs/paper_v_weak_lambda05_raw_vs_corrected_t100_20260804_v1/observables_raw_vs_corrected.png`.

## 2026-08-04: lossless bilinear word-Hankel order gate

The post-five-mode decision memorandum proposed a preparation-initialized,
lossless bilinear Hilbert--Schmidt realization in the verified 29-coordinate
uncentered operator chart.  Its first required computation was a combined
reachability--observability order audit, before any new autonomous fit or
rollout.

The drive-aware force channel alone is already decisive.  Starting with the
unresolved component forces `Q A_i R`, every complete static/drive component
word was generated recursively in the unresolved operator space.  Because the
component Liouvillians are skew-adjoint, the force input and resolved-feedback
output word spans coincide up to transpose, sign, and word reversal.  The
cumulative word-space rank is therefore the force-channel generalized-Hankel
rank for equal complete left/right word sets, and is a lower bound on the full
preparation-augmented hidden order.

The measured ranks were identical at local phonon cutoffs 12 and 16 and at
relative rank tolerances `1e-10` and `1e-12`:

| maximum component-word depth | new rank | cumulative Hankel-rank lower bound |
|---:|---:|---:|
| 0 | 19 | 19 |
| 1 | 19 | 38 |
| 2 | 36 | 74 |
| 3 | 61 | 135 |

The smallest retained depth-three singular value was `0.0913` at cutoff 12
and `0.0805` at cutoff 16, so the additional 61 directions are far above both
rank thresholds.  The lower bound `135` exceeds the memorandum's practical
ceiling `r=96` before any correlated-preparation columns are included.  Such
columns can only leave this rank unchanged or increase it.

Decision under the memorandum's declared `r <= 96` budget: do not build that
specific small global bilinear realization.  The budget is a cost probe, not a
scientific upper bound.  The rank result does not exclude a larger
realization, a finite-horizon nonlinear manifold, or a compressed-state route;
none should be rejected merely because it exceeds 96 coordinates.  The
preparation-specific and time-limited Hankel constructions, autonomous
rollout, controller coupling, and exact trajectory score were not run for this
particular budgeted realization.

Artifacts and verification:

- implementation: `paper_5/src/paper5/stability/reachability_observability.py`;
- driver: `pipelines/open_dynamics/analyze_lossless_bilinear_hankel_gate.py`;
- artifact: `output/local_runs/paper_v_lossless_bilinear_word_hankel_gate_20260804_v1/`;
- focused reachability/Krylov tests: 17/17 passing;
- no exact reference or autonomous propagation was used by this gate.

## 2026-08-04: packet-derived missing-source bridge

The existing fine multi-coherent packet trajectories were contracted into
analytic moment velocities without launching a new propagation, timing run, or
sealed exact-reference score.  At each of 401 stored times for the central,
plus, and minus preparations, the audit evaluated three velocities:

1. the McLachlan-projected packet velocity;
2. the exact Schrodinger velocity at that same packet state, before tangent
   projection; and
3. the autonomous same-spin Pauli-repaired 31-coordinate archive velocity.

Subtracting the third velocity from the first two produced matched packet
versions of the missing closure source.  Reconstructing the hierarchy and
31-coordinate state directly from every stored packet ket agreed with the
stored arrays to maximum errors `2.84e-14` and `8.88e-16`, respectively.

The full-block audit supports a correlation-only source insertion on this
tested packet family.  In the fixed scaled coordinate metric, the `C` block
contained `99.815%` of the squared McLachlan residual and `99.972%` of the
same-state Schrodinger residual.  The remaining blocks were small but were
retained in the audit rather than assumed away.

The five-dimensional exact-source decoder transferred strongly.  Its
normalized reconstruction residual was `0.00452` for the packet McLachlan
source and `0.00447` for the same-state Schrodinger source, each normalized by
its own fluctuation scale.  The five principal angles between the frozen exact
source subspace and the independently fitted packet source subspaces were all
below `0.539` degrees.  Thus, the earlier exact `Q5` result is not merely a
coordinate pattern peculiar to the exact trajectory: the same five output
directions are present on the tested packet manifold.

Over the complete stored interval through `t=100`, the six-packet trajectory's
source amplitudes did not transfer with comparable accuracy.  Relative to the
exact-path source fluctuation, the packet McLachlan source had normalized RMS
error `0.64888`.  The error was strongly horizon dependent: `0.04291` through
the first pulse, `0.13361` through `t=20`, and `0.71364` over `20<t<=100` when
each window was normalized by its exact-source fluctuation.  The exact
Schrodinger source evaluated at the packet states had full-interval error
`0.77949`, while the McLachlan projection differed from that same-state source
by `0.30775`.  The latter defect correlated with the tangent residual at
`0.835`.  These errors are not orthogonal and partially cancel.

Decision: retain the frozen five-source basis as a compact decoder and retain
the packet route as a possible nonlinear, preparation-aware structural model.
The full-interval six-packet result does not justify a global packet-route
stop because the model had reached its imposed capacity ceiling at `t=4` and
remained accurate over the early horizon.  It instead motivates an explicit
capacity and tangent-parameterization study.  The previous rank-135 result
remains scoped only to the global lossless bilinear realization under its
declared budget and does not reject this nonlinear packet manifold.

Artifacts and verification:

- implementation:
  `paper_5/src/paper5/stability/packet_derived_closure.py`;
- driver:
  `pipelines/open_dynamics/analyze_packet_derived_closure_source.py`;
- diagnostic artifact:
  `output/local_runs/paper_v_packet_derived_closure_source_cutoff16_20260804_v4/`;
- the earlier `v1`--`v3` artifacts are superseded only because the final
  audit independently reconstructs the packet state and reports temporal
  windows; the underlying source arrays and full-horizon metrics agree;
- the `v1` reconstruction
  metric repacked stored hierarchy coordinates instead of independently
  contracting the packet ket;
- focused packet-source tests: 2/2 passing; complete Paper V suite: 199/199
  passing.

## 2026-08-04: packet-capacity and tangent-conditioning continuation

The exploratory six-packet hard ceiling in the segmented packet runner was
removed without changing the frozen six-packet batch.  Central trajectories
were rerun through `t=40` with identical initial state, drive, tolerances,
spawn criterion, and Tikhonov damping, while allowing 8, 10, and 12 packets.
Every run completed and used its full allowed capacity.

| maximum packets per branch | real packet coordinates | source NRMS, `0<=t<=40` | same-state projection NRMS | exact source at packet states vs exact path | tangent-residual RMS | source NRMS, `20<t<=40` |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 96 | `0.20427` | `0.09157` | `0.23316` | `0.11395` | `0.27037` |
| 8 | 128 | `0.16053` | `0.06561` | `0.19455` | `0.09101` | `0.21515` |
| 10 | 160 | `0.12695` | `0.02225` | `0.14034` | `0.06748` | `0.16440` |
| 12 | 192 | `0.15172` | `0.09054` | `0.17207` | `0.09471` | `0.20259` |

The six-packet cap therefore did obscure a promising direction: increasing
capacity to ten reduced the full `t=40` source error by `37.9%`, and reduced
the same-state McLachlan projection defect by `75.7%`.  The improvement was
not monotone beyond ten.  The twelve-packet trajectory matched the ten-packet
trajectory until the eleventh packet was spawned at `t=23`; afterward its
tangent rank fell as low as 103 while the geometric tangent rank reached 184.
By comparison, the ten-packet run retained as many as 156 of 158 geometric
directions.

The eleventh and twelfth spawns each reduced the instantaneous residual, so
their failure is not explained by a poor local residual fit alone.  Repeating
the derivative projection on the stored twelve-packet states with smaller
Tikhonov damping reduced the same-state source defect from `0.09054` to
`0.06296`, `0.04436`, and `0.03396` at relative damping `1e-4`, `3e-5`, and
`1e-5`, respectively.  The maximum parameter-speed norm simultaneously grew
from `27.35` to `57.19`, `130.23`, and `286.81`, and agreement with the
exact-path source did not improve.  Lower damping alone therefore exchanges
regularization error for large, ill-conditioned coordinate velocities.

Decision: no packet-count value tested here is a scientific stopping cap.
Ten packets are the best result under the current raw packet coordinates,
spawn rule, and fixed Tikhonov metric, but this is an observed optimum of that
parameterization rather than evidence that the physical packet manifold has
saturated.  The next construction should make spawning and propagation
conditioning-aware--for example by orthogonalizing or whitening the active
tangent directions, rejecting or merging nearly dependent packets, or using
a better-conditioned packet chart--before adding more packets or performing
the distinct-drive closure test.

Artifacts:

- capacity runs:
  `output/local_runs/paper_v_multi_coherent_capacity_k8_t40_20260804_v1/`,
  `output/local_runs/paper_v_multi_coherent_capacity_k10_t40_20260804_v1/`,
  and
  `output/local_runs/paper_v_multi_coherent_capacity_k12_t40_20260804_v1/`;
- matched source audits:
  `output/local_runs/paper_v_packet_derived_closure_source_k6_central_t40_20260804_v2/`,
  `output/local_runs/paper_v_packet_derived_closure_source_k8_central_t40_20260804_v3/`,
  `output/local_runs/paper_v_packet_derived_closure_source_k10_central_t40_20260804_v2/`,
  and
  `output/local_runs/paper_v_packet_derived_closure_source_k12_central_t40_20260804_v2/`;
- complete Paper V suite after the capacity and audit changes: 199/199 passing.

## 2026-08-04: archive-Gram and symmetry-adapted mixed-tangent pilot

Implemented an offline stored-state pilot comparing the seven archive
fluctuation tangents, the native packet tangent, and conditional
electron--relative-phonon products.  The pilot uses 41 samples through
`t=40` from each existing `K=6,8,10,12` strong-coupling double-pulse packet
trajectory.  It restores the factored center oscillator as a centered vacuum
and projects each packet state and tangent into the spin-exchange-symmetric
dimer sector before applying the archive operator frame.

The operator Gram and the Gram reconstructed from `(rho,N,A,C)` differed by
exactly `(33/2) * p_top` to within `2.4e-15`, where `p_top` is the highest
relative-Fock-level population.  This isolates the observed `2.6e-4` to
`4.7e-4` Gram differences as finite-cutoff commutator boundary terms.

The twelve local-site mixed products reduce without loss to six complex
relative-mode products, `delta a_rel^(dagger) delta sigma_a` for
`a in {x,y,z}`.  The compact and full local pools produced the same residual
to within `4.4e-16`.  Adding the compact mixed frame to the packet tangent
reduced the time-RMS same-state Hilbert residual by `40.4%`, `47.0%`, `55.4%`,
and `56.2%` for `K=6,8,10,12`.  The corresponding scaled `C`-velocity errors
fell from `2.94e-2`, `2.11e-2`, `7.44e-3`, and `2.15e-2` to `4.02e-12`,
`8.59e-11`, `8.73e-12`, and `1.39e-8`.  Adding only the archive frame reduced
the packet Hilbert residual by at most `1.13%`.

Interpretation: conditional relative-phonon--Pauli tangents are a reproducible
missing local mechanism, led by the `sigma_z` and `sigma_y` channels.  This is
not yet an autonomous closure or online propagation result.  The next
closure-specific construction is to extract their six coefficient paths over
multiple preparations and drives, then test a compact reciprocal hidden-state
law in free rollout.

Artifacts:

- implementation:
  `paper_5/src/paper5/stability/archive_gram_tangent_pilot.py`;
- pipeline:
  `pipelines/open_dynamics/analyze_archive_gram_tangent_pilot.py`;
- run:
  `output/local_runs/paper_v_archive_mixed_tangent_pilot_cutoff16_20260804_v3/`;
- verification: 3 focused tests and the complete Paper V suite, 202/202
  passing.

## 2026-08-04: mixed-tangent factorization and static-decoder stop

The six complex symmetry-adapted mixed tangents identified by the preceding
pilot were extracted at 81 stored states through `t=20` on five paths: the
central/plus/minus double-pulse preparations, the central single-pulse path,
and the central double-pulse `K_max=10` path.  No new propagation, exact
trajectory score, or online correction was performed.

At each state, the mixed frame was projected orthogonally away from the
archive fluctuation frame.  Its twelve real coefficients `eta` and its
fourteen-by-twelve contraction map `J_C(X)` gave

`target C source = archive-frame part + J_C(X) eta + residual`.

The maximum absolute residual over all coordinates, paths, and samples was
`2.62e-15`.  The mixed complement retained rank 12 throughout.  This verifies
an exact local factorization on the sampled packet states; it does not verify
an autonomous closure.

A fixed ridge decoder from `eta` to the source failed.  Leave-one-preparation-
out source NRMS was `0.774`--`0.798`.  A decoder using the current retained
state, drive, and `eta` reached `0.176`--`0.195` on same-drive preparation
holdouts and `0.178` on the `K_max=10` capacity holdout.  For the distinct
single-pulse path it agreed with the double-pulse result before the second
pulse (`0.159` through `t=8`) but failed after the intervention (`1.536` over
`8<t<=20`, `1.711` over `10<=t<=20`).

Interpretation: the second pulse changes unresolved correlations through their
prior evolution.  The six mixed directions identify the entrance channels,
but their current projection coefficients are not a sufficient memory state,
and the state-dependent map from those directions to `dot C` carries further
connected electron--two-phonon and opposite-spin information.  Static linear
or nonlinear decoder development is stopped.  The next route is a
state-adapted Mori--Zwanzig or auxiliary-operator construction derived from
the Liouvillian commutators, with lossless/skew-adjoint hidden evolution in its
physical metric and explicit reciprocal feedback into the retained blocks.

Artifacts:

- implementation:
  `paper_5/src/paper5/stability/mixed_tangent_closure_identifiability.py`;
- driver:
  `pipelines/open_dynamics/analyze_mixed_tangent_closure_identifiability.py`;
- diagnostic:
  `output/local_runs/paper_v_mixed_tangent_closure_identifiability_cutoff16_t20_20260804_v2/`.

## 2026-08-04: first state-adapted Liouvillian layer

The implemented relative-mode Hamiltonian was rewritten as

`H = -t(X_up+X_down) + omega a^dagger a + [V/2 + (g/sqrt(2))(a+a^dagger)](Z_up+Z_down)`.

Applying `L O = i[H,O]` to the six raw mixed operators
`a^(dagger) sigma_{x,y,z}` generates the mixed layer itself, resolved Pauli
terms, and two new operator families: electron-conditioned two-phonon products
`(a^2,n,a^dagger^2) sigma_{x,y}` and opposite-spin products
`Z_other sigma_{x,y,z}`.  These are respectively the operator sources of the
connected electron--two-phonon and opposite-spin terms found in the exact
correlation-derivative decomposition.

The six expanded identities were checked against direct matrix commutators.
At relative dimension 33 the maximum residual after including the finite-Fock
boundary projector was `9.33e-16`.  Omitting that projector produced relative
global Frobenius residuals `0.1456`--`0.3933`, equal to the independently
constructed boundary norms.  This separates the canonical commutator layer
from the known cutoff-space top-level term.

The resulting bridge uses the state-adapted archive projector `P_X`, its
complement `Q_X`, and the generated auxiliary operators.  Because `P_X` moves
with the state, the orthogonal generator contains `Q_X L - dot(P_X)`; a fixed
projection would omit the connection term.  At frozen state the projected
Hamiltonian generator is skew on the supported Hilbert tangent range.  In an
evolving nonorthogonal auxiliary chart, losslessness requires
`A^T M + M A + dot(M) = 0` together with reciprocal resolved/hidden blocks.

Artifacts and verification:

- derivation:
  `paper_5/notes/state_adapted_liouvillian_auxiliary_bridge_20260804.md`;
- implementation:
  `paper_5/src/paper5/stability/mixed_operator_commutator.py`;
- focused tests: 3/3 passing across the commutator and mixed-factorization
  modules;
- complete Paper V suite: 205/205 passing.

## 2026-08-04: four-route memorandum and reciprocal archive-section gate

The Ultra memorandum now treats packet conditioning/capacity, Archive-Gram-
guided packet growth, integrable mixed-operator enrichment, and reciprocal
auxiliary memory as concurrent routes.  The preferred archive-compatible
route uses the verified 29-coordinate uncentered Hilbert--Schmidt chart and
defines the archive EOM as an instantaneous section of a reciprocal hidden
operator realization.  Its first falsifiable requirement is algebraic: the
Pauli-repaired archive velocity must have a unique raw lift and its difference
from the in-span Liouvillian drift must lie in the hidden-to-resolved coupling
range.

That gate is now implemented.  The first unresolved force shell has 19
Hilbert--Schmidt-orthonormal operators.  Its static and drive component blocks
obey skew-adjointness and resolved/hidden reciprocity to at most `2.71e-16`.
At each of 81 stored times on the central/plus/minus double-pulse paths, the
single-pulse holdout, and the `K_max=10` capacity holdout, the code:

1. lifted the 31-coordinate Pauli-repaired archive velocity through the
   analytic `31 x 29` centering Jacobian;
2. subtracted the resolved Liouvillian drift;
3. solved for the minimum-Euclidean-norm hidden section; and
4. measured the remaining incompatibility in the centered 31-coordinate
   output.

At phonon cutoffs 12 and 16 and relative rank tolerances `1e-10` and `1e-12`,
the raw-lift rank stayed 29, the hidden coupling rank stayed 19, the maximum
raw-lift relative residual was `1.1355e-13`, and the maximum noise-floored
centered section residual was `9.2895e-15`.  The smallest retained coupling
singular-value ratio was `0.1822` at cutoff 12 and `0.1588` at cutoff 16.
Thus the archive field is algebraically an exact section of this first
reciprocal force shell on the audited states; the envelope does not need to be
extended merely to reproduce the instantaneous archive field.

This result is not an autonomous closure score.  It does not show that 19
hidden coordinates carry the required preparation and pulse history, nor that
their free rollout improves the archive observables.  The next Route-4 step is
to initialize the physical hidden contractions, propagate the archive-relative
memory coordinate with the reciprocal hidden block and the total derivative
of the archive section, and then score autonomous transfer.  Finite-horizon
reachability--observability reduction remains an order-selection step after
that fixed-frame parent is verified.

Artifacts and verification:

- Ultra memorandum:
  `/Users/jakestrobel/images/chapter/lease/2026-08-04-four-route-state-adapted-archive-repair-memorandum.md`;
- implementation:
  `paper_5/src/paper5/stability/archive_auxiliary_memory.py`;
- driver:
  `pipelines/open_dynamics/analyze_archive_auxiliary_section.py`;
- diagnostic:
  `output/local_runs/paper_v_archive_auxiliary_section_cutoff12_16_20260804_v1/`;
- focused tests: 5/5 passing.

## 2026-08-04: first autonomous reciprocal fixed-union pilot

The algebraic archive section was promoted to an autonomous fixed-frame model.
The physical hidden coordinates were contracted once from the exact correlated
ground-state preparation, the archive-relative memory was initialized as the
difference between that contraction and the instantaneous archive section, and
the coupled equations propagated only the current retained coordinates,
memory, `V(t)`, and `dV/dt`.  The total derivative of the archive section was
recomputed at every RK4 stage.  Setting the relative memory to zero reproduces
the Pauli-repaired archive field, and the projected Hilbert--Schmidt lossless
identity held along every tested trajectory to below `8.5e-16`.

The hidden operator frame was the uncapped nested static/drive component-word
envelope.  At cutoff 16 its layer dimensions were `19, 19, 36, 61`, giving
cumulative hidden dimensions `19, 38, 74, 135`.  The single-pulse strong-
coupling pilot used `dt=0.01` through `t=4` and was scored against the existing
cutoff-16 exact Hamiltonian trajectory.  No exact trajectory entered the
online right-hand side and no representability controller was applied.

| model | hidden dimension | 31-coordinate RMS error | minimum joint-Gram eigenvalue |
|---|---:|---:|---:|
| raw archive EOM | 0 | `0.11066` | `-0.51719` |
| Pauli-repaired archive EOM | 0 | `0.057742` | `-0.39979` |
| reciprocal word depth 0 | 19 | `0.92401` | `-7.39098` |
| reciprocal word depth 1 | 38 | `1.25506` | `-12.02415` |
| reciprocal word depth 2 | 74 | `0.58403` | `-4.25125` |
| reciprocal word depth 3 | 135 | `1.00865` | `-7.38733` |

The depth-2 result is the best fixed prefix but remains substantially worse
than the archive backbone.  The failure is not caused by an archive-section
range error, loss of reciprocal skew structure, or online exact-data leakage:
the maximum section residual stayed below `1.33e-14` and every norm identity
passed.  It shows that unweighted component-word prefixes are not the finite-
horizon, preparation-aware reachable--observable frame requested by the Ultra
construction.  Increasing prefix depth alone is therefore not the next
action.  The missing initial complement and terminal output-visible forcing
must enter the finite-horizon order construction, and the accepted route must
be scored with the representability controller as a separate safety layer.

A cutoff-2 smoke test was qualitatively misleading: its depth-2 RMS error was
`0.0911`, compared with `0.2718` for the Pauli archive.  The cutoff-16 result
is the relevant working reference and demonstrates why a small-cutoff pilot
cannot select the closure.

Artifacts and verification:

- implementation:
  `paper_5/src/paper5/stability/archive_auxiliary_memory.py` and
  `paper_5/src/paper5/stability/reachability_observability.py`;
- driver:
  `pipelines/open_dynamics/analyze_archive_auxiliary_autonomous_pilot.py`;
- diagnostic:
  `output/local_runs/paper_v_archive_auxiliary_autonomous_cutoff16_t4_20260804_v1/`;
- focused reciprocal/word-envelope tests: 14/14 passing;
- complete Paper V suite: 212/212 passing with `PYTHONPATH=src`;
- measured elapsed construction and rollout time: `459.6 s`.

## 2026-08-05: preparation slip, memory repair, and autonomous rerun

The first preparation-seeded implementation mixed one dense cutoff-16
preparation density into every force-shell column during joint SVD
orthogonalization.  Since the full two-local-mode Hilbert dimension is 1156,
this made many `1156 x 1156` operator columns dense and also materialized a
large `2 d_H^2` real-vectorization workspace.  A depth-3 construction was
stopped after twenty minutes before rollout, and the first depth-2 attempt was
stopped at the user's request when system RAM appeared to grow rapidly.  These
were construction stops, not closure scores.

The operator construction was repaired in two ways.  The sparse archive-force
word envelope and low-rank preparation-density branch are now orthogonalized
separately, preventing one dense preparation residual from densifying every
force direction.  Liouvillian component blocks are also projected one hidden
column at a time instead of retaining a duplicate action block.  A reusable
process-tree monitor was added and verified to stop a test allocation at its
declared RSS threshold.

The repaired cutoff-16 depth-2 run completed through `t=4` in `167.2 s`.  Its
peak process-tree RSS was `1.186 GiB`, the minimum observed system-free memory
was `32%`, and neither the `3.5 GiB` RSS stop nor the `15%` free-memory stop was
approached.  The preparation-aware layer dimensions were `20, 21, 40`, giving
cumulative hidden dimensions `20, 41, 81`.

| model | hidden dimension | 31-coordinate RMS error | energy RMS error | minimum joint-Gram eigenvalue |
|---|---:|---:|---:|---:|
| Pauli-repaired archive EOM | 0 | `0.057742` | `0.096327` | `-0.39979` |
| preparation depth 0 | 20 | `0.92401` | `0.14472` | `-7.39098` |
| preparation depth 1 | 41 | `1.25506` | `0.32202` | `-12.02415` |
| preparation depth 2 | 81 | `0.36237` | `0.011027` | `-5.06978` |

Supplying the unresolved correlated preparation and its descendants improves
the depth-2 coordinate error relative to the force-only depth-2 pilot
(`0.58403 -> 0.36237`) and substantially improves its energy error.  It does
not recover the archive observables or representability: the coordinate error
remains over six times the Pauli-repaired archive error and the joint Gram is
strongly indefinite.  This isolates initial slip as useful but insufficient.
The next Route-4 construction is the memorandum's finite-horizon,
preparation-aware reachable--observable selection with terminal-output
forcing; increasing a raw word prefix is neither the scientific nor the
memory-efficient next step.

Artifacts and verification:

- memory-safe implementation:
  `paper_5/src/paper5/stability/reachability_observability.py` and
  `paper_5/src/paper5/stability/archive_auxiliary_memory.py`;
- monitor:
  `pipelines/open_dynamics/run_with_memory_monitor.py`;
- completed run:
  `output/local_runs/paper_v_archive_auxiliary_autonomous_prepseed_cutoff16_t4_depth2_memorysafe_20260805_v1/`;
- memory record:
  `output/local_runs/paper_v_archive_auxiliary_autonomous_prepseed_cutoff16_t4_depth2_memorysafe_20260805_v1_monitor/`;
- focused reciprocal and envelope tests: 15/15 passing;
- complete Paper V suite: 213/213 passing with `PYTHONPATH=src`.

## 2026-08-05: finite-horizon reachable--observable order curve

The preparation-aware 81-dimensional depth-2 envelope was subjected to the
finite-horizon construction required by the four-route memorandum.  On the
cutoff-16 strong-coupling exact development path through `t=4`, midpoint
state transitions formed grid-consistent reachability and observability
Gramians at split times `0.5, 1.0, ..., 3.5`.  Reachability included both the
continuous resolved-to-hidden forcing and the propagated correlated-
preparation memory.  Observability measured the hidden-induced velocity in
all 31 retained coordinates with the frozen development scales.  Exact data
were used only for this offline construction and later scoring, never in an
autonomous right-hand side.

The hidden realization was not strongly compressible on this horizon.  All
81 directions were reachable and observable at the central splits; 70 of 81
aggregate Hankel values remained above `1e-2` of the leading value.  The
worst unconstrained Hankel-tail defect was `0.3256` after retaining 20 ideal
balanced singular directions and did not fall below `1e-3` until approximately
79 such directions.  Those ideal counts are not the orders of the implemented
reciprocal frame.  The Hilbert--Schmidt-orthogonal union of the mandatory
entrance, primal, and dual proposals reached the full 81-dimensional envelope
after 31 proposed pairs.  This union, rather than an oblique balanced
projection, keeps the trial and test frames identical and therefore preserves
skew-adjoint hidden dynamics and reciprocal coupling.

Every distinct union order, `r=20,22,...,80,81`, was then propagated
autonomously with `dt=0.01` through `t=4`.  The exact trajectory entered only
after rollout for scoring.  The best coordinate and energy candidate was
`r=78`:

| model | hidden dimension | 31-coordinate RMS error | energy RMS error | minimum joint-Gram eigenvalue |
|---|---:|---:|---:|---:|
| raw archive EOM | 0 | `0.11066` | `0.027449` | `-0.51719` |
| Pauli-repaired archive EOM | 0 | `0.057742` | `0.096327` | `-0.39979` |
| best finite-horizon union | 78 | `0.34729` | `0.010080` | `-4.50875` |
| complete finite-horizon union | 81 | `0.36237` | `0.011027` | `-5.06978` |

The order curve was highly nonmonotone below `r=76`; no reduced order
improved the archive coordinate trajectory or representability.  Near-full
orders improve total-energy agreement, but that isolated gain accompanies
much larger errors in every retained block and a strongly indefinite joint
Gram.  The section residual stayed below `7.5e-14` and the lossless identity
residual below `5.7e-13`, so this is not an algebraic range, reciprocity, or
projection-implementation failure.  It establishes that finite-horizon
selection inside this fixed depth-2 reciprocal envelope does not produce an
acceptable autonomous archive closure on its own.

The fixed-union Route-4 candidate is therefore not accepted.  A continuation
must change the model class rather than tune its order: the memorandum's
state-adapted atlas with explicit projection transport is the remaining
Route-4 variant, while the packet-capacity, Archive-Gram-guided packet, and
direct mixed-tangent routes remain independent concurrent branches.

Artifacts and verification:

- finite-horizon implementation:
  `paper_5/src/paper5/stability/finite_horizon_auxiliary.py`;
- reciprocal numerical-projection seam:
  `paper_5/src/paper5/stability/archive_auxiliary_memory.py`;
- construction driver:
  `pipelines/open_dynamics/analyze_archive_auxiliary_finite_horizon.py`;
- autonomous order-curve driver:
  `pipelines/open_dynamics/analyze_archive_auxiliary_finite_horizon_rollout.py`;
- synchronized construction run:
  `output/local_runs/paper_v_archive_auxiliary_finite_horizon_cutoff16_t4_20260805_v2/`;
- autonomous order curve:
  `output/local_runs/paper_v_archive_auxiliary_finite_horizon_rollout_cutoff16_t4_20260805_v1/`;
- peak monitored RSS: `1.105 GiB` for construction and `1.035 GiB` for the
  order curve, with at least `32%` system memory free;
- focused finite-horizon, reciprocal, and envelope tests: 19/19 passing;
- complete Paper V suite: 217/217 passing with `PYTHONPATH=src`.

## 2026-08-05: split-local moving-frame viability audit

The remaining Route-4 variant was tested before implementing a moving-atlas
propagator.  A common cutoff-16 preparation-aware depth-2 envelope was built
from the central, plus, and minus exact preparations.  It contained 95 hidden
operators in layers `22, 25, 48`; the 22 mandatory entrance coordinates are
the 19 static/drive force directions plus the three preparation residuals.
Split-local finite-horizon Gramians were then constructed through `t=20` for
four paths: central single pulse and central/plus/minus double pulse, at split
times `0.5, 2, 4, 8.5, 10, 12, 16, 19`.  The double-pulse DOP853 path was used
only as an offline construction input because its previous independent exact-
reference comparison missed the declared `1e-6` moment tolerance narrowly
(`1.05505e-6`).

For every distinct local reciprocal-union order, the audit measured the actual
root trace residuals of the reachability and observability Gramians, principal
angles between neighboring subspaces, the spectral gap of projector blending,
archive-section compatibility, omitted resolved forcing, and the normal
transport defect `(I-P)(A U-Udot)`.  No autonomous rollout or scientific
dimension cap was used.

The split-local construction is not a compact smooth atlas.  Keeping the worst
reachability/observability residual below `0.1` first required order 88 of 95;
below `0.05` required order 94; and below `0.01` required the full order 95.
The order-88 and order-94 subspaces still rotated by as much as `89.73` and
`88.19` degrees between neighboring splits, with minimum projector-blend gaps
`0.00471` and `0.03164`.  Their maximum normal-transport leakage ratios were
`0.3093` and `0.1940`.  Only the full 95-dimensional space had negligible
transport leakage and a stable unit blend gap, which is tautological because
its projector is the identity.  It reduces to the complete fixed-envelope
model already shown to have inaccurate autonomous observables and severe
representability loss.

The mandatory entrance space did retain the raw archive section and direct
resolved forcing at floating-point precision for every tested order: maximum
section residuals stayed below `3.6e-15` and input leakage below `2.3e-15`.
The failure is therefore not loss of the instantaneous archive source.  It is
the broad, rapidly rotating finite-horizon hidden dynamics needed beyond that
source.  A moving atlas inside this depth-2 envelope is not implemented: the
viability audit supplies no intermediate order that is both meaningfully
reduced and smoothly transportable.  This closes that particular Route-4
construction, not the independent packet-growth or direct mixed-tangent
branches and not every possible archive repair.

Artifacts and verification:

- implementation:
  `paper_5/src/paper5/stability/moving_frame_viability.py`;
- split-local Gramian persistence:
  `paper_5/src/paper5/stability/finite_horizon_auxiliary.py`;
- driver:
  `pipelines/open_dynamics/analyze_archive_auxiliary_moving_frame_viability.py`;
- completed audit:
  `output/local_runs/paper_v_archive_auxiliary_moving_frame_viability_cutoff16_t20_20260805_v2/`;
- memory record: peak process-tree RSS `1.712 GiB`, with at least `32%`
  system memory free, under
  `output/local_runs/paper_v_archive_auxiliary_moving_frame_viability_cutoff16_t20_20260805_v2_monitor/`;
- the retained `v1` directory records the harmless failed bitwise-time-grid
  preflight; the grids differed by only `3.55e-15`, and `v2` uses a declared
  `1e-12` grid tolerance;
- focused finite-horizon, moving-frame, reciprocal, and envelope tests: 20/20
  passing before the production audit;
- complete Paper V suite: 218/218 passing after the audit with
  `PYTHONPATH=paper_5/src`.

## 2026-08-05: integrable mixed-layer construction and autonomous pilot

The six complex relative-phonon--Pauli entrance directions were integrated
into the ordered analytic packet-union chart proposed by the four-route
memorandum.  A first stored-state audit exposed a finite-cutoff convention
error: the chart acts on analytic coherent states before projection, whereas
its declared annihilation tangent had been evaluated with the already
truncated ladder matrix.  The discrepancy was confined to the horizontally
projected top retained Fock level, was independent of finite-difference step,
and reached `0.21598` relatively at the `K=10`, `t=31` stored state.  Replacing
that tangent by the derivative of the actual analytic-before-cutoff chart made
all twelve real finite-difference checks pass, including a high-boundary
regression fixture.  The original `v1` gate is therefore retained only as the
failed convention audit.

The corrected `K=6,8,10,12` audit covered 164 stored double-pulse states.  The
maximum origin finite-difference error was `4.69e-9` and peak RSS was
`196 MB`.  The analytic mixed tangent no longer has the spuriously exact
cutoff-operator `C` response, but it retains a strong same-state gain:

| local native-plus-archive comparison | parent | plus analytic mixed layer | fractional reduction |
|---|---:|---:|---:|
| Hilbert relative residual, time RMS | `0.096755` | `0.050678` | `47.62%` |
| scaled 31-coordinate derivative RMS | `0.0083725` | `0.0021939` | `73.80%` |
| scaled correlation-derivative RMS | `0.011212` | `0.00032872` | `97.07%` |

An uncapped exact packet-union retraction was then implemented.  It selects
coherent centers from the analytic layer until a declared state tolerance is
met, with no externally imposed packet count.  The autonomous online step uses
only the current packet state, Hamiltonian, and mixed chart; cutoff-16 exact
data enter afterward for scoring.  The retraction itself remained accurate
below `7.2e-10`, but direct time propagation did not preserve the local gain:

| rollout through `t=4` | maximum `K` per branch | scaled 31-coordinate RMS | scaled `C` RMS | minimum exact-state fidelity |
|---|---:|---:|---:|---:|
| native adaptive packet parent | 6 | `0.0020405` | `0.0027282` | not rescored here |
| mixed Euler, `dt=0.01` | 29 | `0.038814` | `0.037302` | `0.98499` |
| mixed Euler, `dt=0.005` | 28 | `0.015952` | `0.018501` | `0.99761` |
| ambient-transport midpoint, `dt=0.01` | 28 | `0.027500` | `0.032527` | `0.99254` |

Euler step halving confirms substantial first-order integration error, while
the attempted ambient midpoint transport does not supply the fixed-chart
macrostep integration required by the memorandum.  More importantly, accurate
retraction naturally drives the realization to `K=28--29` in a relative space
of dimension 33.  The direct layer has therefore approached the cutoff-state
cost without improving the parent trajectory, satisfying its declared
diminishing-return condition.  The chart remains a valid local diagnostic;
the next use of its directions is state-continuous, Archive-Gram-guided packet
admission followed by the established adaptive packet propagator, rather than
another full-layer rollout.

Artifacts and verification:

- analytic chart and uncapped retraction:
  `paper_5/src/paper5/stability/mixed_exponential_layer.py`;
- autonomous construction pilot:
  `paper_5/src/paper5/stability/mixed_enriched_propagation.py`;
- corrected stored-state gate:
  `output/local_runs/paper_v_integrable_mixed_layer_gate_cutoff16_20260805_v2/`;
- Euler rollouts:
  `output/local_runs/paper_v_integrable_mixed_enriched_cutoff16_t4_dt001_20260805_v1/`
  and
  `output/local_runs/paper_v_integrable_mixed_enriched_cutoff16_t4_dt0005_20260805_v1/`;
- midpoint rollout:
  `output/local_runs/paper_v_integrable_mixed_enriched_midpoint_cutoff16_t4_dt001_20260805_v1/`;
- focused mixed-layer and propagation tests: 8/8 passing.

## 2026-08-05: mixed-guided state-continuous packet admission

The verified mixed layer was next used only to place zero-weight coherent
packets; the established packet McLachlan right-hand side remained the online
propagator.  Each admission fits one new center per electronic branch to the
mixed-only same-state velocity gain, imposes spin-exchange symmetry on the two
single-occupancy branches, and leaves all new coefficients equal to zero.
Consequently every admission changes tangent capacity without changing the
represented ket.  The current packet state and Hamiltonian are the only online
inputs; cutoff-16 data enter afterward for scoring.

Starting from the common fitted `K=4` state, two admissions at `t=0` gave a
fixed `K=6` model.  Through `t=4`, this reduced the scaled 31-coordinate RMS
error from `0.0020405` for the established adaptive parent to `0.00088761`, and
the scaled correlation-block RMS from `0.0027282` to `0.00097219`.  Repeating
the run with maximum DOP853 step `0.005` instead of `0.01` changed the aggregate
score by less than `1e-10`, resolving the numerical integration scale.

Through the complete double-pulse horizon `t=20`, the `t=0`-only basis retained
lower total 31-coordinate error but lost its correlation and electronic-
observable advantage after the second pulse.  At the model state at `t=8`,
the mixed-only unresolved speed had grown from `0.001466` at initialization to
`0.017740`.  A state-only admission curve showed that the first new packet
reduced the native local residual by `56.9%`, and a second reduced the remainder
by `27.3%`.  This motivated matched state-preserving readmissions at the second
pulse, without using exact data to choose their centers.

| autonomous rollout through `t=20` | final `K` | scaled 31-coordinate RMS | scaled `C` RMS | occupation RMS | phonon-energy RMS | electron--phonon-energy RMS | final exact fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary adaptive parent | 6 | `0.027843` | `0.021629` | `0.002708` | `0.015406` | `0.018650` | not rescored here |
| mixed-guided only at `t=0` | 6 | `0.021429` | `0.025740` | `0.004748` | `0.008200` | `0.014790` | `0.98936` |
| one readmission at `t=8` | 7 | `0.016380` | `0.012253` | `0.001356` | `0.006084` | `0.009046` | `0.99377` |
| two readmissions at `t=8` | 8 | `0.015052` | `0.011097` | `0.001693` | `0.006244` | `0.009165` | `0.99355` |

The state-adapted readmission restores the trajectory-level gain after the
second pulse.  `K=7` and `K=8` are currently nondominated: `K=8` has lower
aggregate moment, correlation, electronic-energy, and total-energy error,
whereas `K=7` has lower occupation, phonon-energy, electron--phonon-energy
error and slightly higher final fidelity.  This also shows that minimizing the
local mixed residual alone does not choose the globally best capacity.  The
next Route-2 step was therefore to add the memorandum's archive-Gram rate and
native Hilbert-residual terms to the admission score and trigger admissions
from the current model state rather than from a declared pulse time.

Artifacts and verification:

- state-continuous admission and projection:
  `paper_5/src/paper5/stability/mixed_enriched_propagation.py`;
- segmented autonomous runner:
  `pipelines/open_dynamics/run_mixed_guided_packet_rollout.py`;
- `t=20` baseline, `K=7`, and `K=8` runs:
  `output/local_runs/paper_v_mixed_guided_packet_k6_cutoff16_t20_20260805_v1/`,
  `output/local_runs/paper_v_mixed_guided_packet_k7_readmit_t8_cutoff16_t20_20260805_v1/`,
  and
  `output/local_runs/paper_v_mixed_guided_packet_k8_readmit_t8_cutoff16_t20_20260805_v1/`;
- newest-first observable summary:
  `output/pdf/paper_v_results_progression_20260804.pdf`;
- focused mixed-layer and propagation tests: 10/10 passing;
- complete Paper V suite: 228/228 passing with
  `PYTHONPATH=paper_5/src`.

### Full three-objective adaptive gate

The current-state admission audit now evaluates

1. the squared native Hilbert-space residual;
2. the joint-Gram rate defect whitened on the supported Gram range; and
3. the equal-scale 31-coordinate impact of the unresolved velocity projected
   onto the mixed range novel to the native-plus-archive observer tangent.

The first two guided packets remain declared preparation capacity at `t=0`.
Thereafter, an uncapped autonomous rollout checks the established native
residual trigger (`relative >= 0.05` and `absolute >= 0.02`) at half-unit
segment endpoints.  A zero-weight candidate is accepted only when the mean of
all active objectives, normalized by their pre-admission values, decreases,
the supported native tangent rank increases, and the state-continuity check
passes.  The recorded score also includes measured candidate-construction
time and the before/after condition-number ratio.  The coordinate scales are
frozen construction data; no exact state or future value enters the trigger,
candidate placement, acceptance, or propagation.

The gate selected `K=6 -> 7` at `t=8` and `K=7 -> 8` at `t=19`.  Their
normalized three-objective reductions were `89.82%` and `71.44%`; both had
zero state discontinuity.  No other checkpoint through `t=20` triggered.

| rollout | final `K` | scaled 31-coordinate RMS | scaled `C` RMS | occupation RMS | phonon-energy RMS | electron--phonon-energy RMS | final exact fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary adaptive parent | 6 | `0.027843` | `0.021629` | `0.002708` | `0.015406` | `0.018650` | not rescored here |
| autonomous three-objective gate | 8 | `0.015844` | `0.012534` | `0.001695` | `0.006266` | `0.009331` | `0.99364` |

The autonomous gate improves every reported whole-horizon observable relative
to the ordinary parent and reduces aggregate coordinate error by `43.10%`.
A matched ablation from the identical `t=19` ket removed only the newly
admitted zero-weight packet and propagated both branches with identical
half-unit restarts.  Retaining `K=8` reduced post-event aggregate coordinate,
correlation, electronic-energy, and electron--phonon-energy errors by `1.38%`,
`3.76%`, `6.31%`, and `2.77%`, respectively.  The `K=7` branch remained
slightly better for occupations, phonon energy, and total energy.  Thus the
late admission adds genuine useful capacity but is itself a Pareto tradeoff;
the local score cannot be interpreted as a universal downstream-observable
loss.

Additional artifacts:

- autonomous gate:
  `output/local_runs/paper_v_archive_gram_adaptive_packet_cutoff16_t20_20260805_v1/`;
- matched `t=19` event ablation:
  `output/local_runs/paper_v_archive_gram_adaptive_packet_t19_ablation_cutoff16_20260805_v1/`;
- event-ablation driver:
  `pipelines/open_dynamics/analyze_adaptive_packet_event_ablation.py`.

### Frozen-gate transfer across preparation and drive

The central double-pulse admission rule was next frozen and applied without
threshold or objective retuning to the nearby plus and minus physical
preparations and to the central preparation under the original single pulse.
All cells began from their matched `K=4` packet chart, received the same two
state-continuous mixed-guided packets at `t=0`, and thereafter used the same
half-unit current-state checks, native residual trigger, three-objective
candidate score, rank-gain requirement, and zero-discontinuity requirement.
No packet-count cap was imposed.

| transfer cell | selected post-initial admissions | final `K` | parent -> adaptive scaled 31 RMS | parent -> adaptive scaled `C` RMS |
|---|---|---:|---:|---:|
| plus preparation, double pulse | `11, 12` | 8 | `0.026621 -> 0.017536` | `0.027773 -> 0.013242` |
| minus preparation, double pulse | `13.5, 16.5, 18.5` | 9 | `0.030731 -> 0.017306` | `0.023912 -> 0.017998` |
| central preparation, single pulse | `8, 11, 18` | 9 | `0.017471 -> 0.009203` | `0.016724 -> 0.009756` |

The adaptive trajectory improved both aggregate moment scores and every
reported occupation, electronic-energy, phonon-energy, and electron--phonon-
energy RMS in all three cells.  Total-energy RMS also improved for the minus
and single-pulse cells; in the plus cell it increased from `3.91e-4` to
`6.70e-4` while the other observables improved.  The plus/minus exact state
overlays gave minimum fidelities `0.99345` and `0.99337`.  The existing
single-pulse artifact stores dense exact moment contractions but not exact
state vectors, so that cell was scored at the moment and observable levels
without a fidelity claim.

The different admission schedules are direct transfer evidence: the gate did
not replay the central double-pulse times `8, 19`, and its distinct-drive run
continued to improve after the absent second pulse.  This establishes a useful
state-adapted capacity controller over the tested local preparation manifold
and one drive holdout.  It does not yet establish universality over coupling,
pulse amplitude, or substantially different correlated preparations.

Artifacts:

- plus preparation:
  `output/local_runs/paper_v_archive_gram_adaptive_packet_plus_cutoff16_t20_20260805_v1/`;
- minus preparation:
  `output/local_runs/paper_v_archive_gram_adaptive_packet_minus_cutoff16_t20_20260805_v1/`;
- central single pulse:
  `output/local_runs/paper_v_archive_gram_adaptive_packet_single_pulse_cutoff16_t20_20260805_v1/`;
- updated newest-first observable summary:
  `output/pdf/paper_v_results_progression_20260804.pdf`;
- complete Paper V suite after the transfer-runner extension: 228/228 passing
  with `PYTHONPATH=paper_5/src`.

### Uncapped central extension through `t=40`

The frozen central double-pulse gate was rerun without a packet ceiling through
`t=40`.  It reproduced the prior admissions at `t=8` and `19`, then admitted
at `t=22`, `26.5`, and `38`, ending at `K=11` with time-average capacity
`8.166` packets per electronic branch.  Every admission again had zero state
discontinuity.  Peak observed process RSS remained about `122 MB`.

| model through `t=40` | final `K` | mean `K` | scaled 31 RMS | scaled `C` RMS |
|---|---:|---:|---:|---:|
| ordinary `K_max=6` | 6 | `5.826` | `0.053379` | `0.063880` |
| fixed `K_max=8` | 8 | `7.340` | `0.050692` | `0.045564` |
| fixed `K_max=10` | 10 | `8.491` | `0.039069` | `0.027691` |
| fixed `K_max=12` | 12 | `9.305` | `0.048589` | `0.038589` |
| uncapped adaptive gate | 11 | `8.166` | `0.033863` | `0.030244` |

The adaptive gate gives the lowest aggregate coordinate, occupation, phonon-
energy, and electron--phonon-energy errors, with lower average capacity than
the fixed `K_max=10` trajectory.  Fixed `K_max=10` retains slightly lower
correlation-block and electronic-energy error.  The adaptive total-energy RMS
is worse (`1.478e-3` versus `7.42e-4`--`7.99e-4` for the fixed paths), and its
minimum exact-state fidelity falls to `0.98139`.  Thus the state-only gate adds
useful capacity on the longer horizon but does not yet produce a uniformly
dominant accuracy point; energy weighting or a separate energy-consistent
candidate condition remains unresolved.

Artifacts:

- uncapped trajectory:
  `output/local_runs/paper_v_archive_gram_adaptive_packet_cutoff16_t40_20260805_v1/`;
- newest-first observable and accuracy--cost page:
  `output/pdf/paper_v_results_progression_20260804.pdf`.

#### Scoring-representation correction

The first mixed-guided score files contracted the new trajectory after
projection from the center/relative chart into the finite local cutoff, while
the ordinary and fixed-capacity packet trajectories were scored directly in
the native center/relative moment chart.  The propagation and all admission
decisions were valid and unchanged, but those two packet score conventions are
not directly comparable.  The apparent `t=40` total-energy penalty was a
consequence of mixing them.

All stored packet trajectories were therefore rescored without propagation.
The primary comparison now uses the same direct center/relative contraction
for every packet path.  The local-cutoff projection remains separately stored
for retained-norm and exact-state-fidelity diagnostics.  This correction
supersedes the numerical mixed-guided score tables immediately above; it does
not supersede their admission times, packet counts, or state-continuity tests.

Corrected transfer scores are:

| cell | parent -> adaptive scaled 31 RMS | parent -> adaptive scaled `C` RMS | parent -> adaptive total-energy RMS |
|---|---:|---:|---:|
| central double pulse, `t=20` | `0.027843 -> 0.018088` | `0.021629 -> 0.012915` | `0.000625 -> 0.000638` |
| plus double pulse, `t=20` | `0.026621 -> 0.020342` | `0.027773 -> 0.013081` | `0.000391 -> 0.000369` |
| minus double pulse, `t=20` | `0.030731 -> 0.017898` | `0.023912 -> 0.014660` | `0.001243 -> 0.000721` |
| central single pulse, `t=20` | `0.017471 -> 0.010729` | `0.016724 -> 0.009759` | `0.001515 -> 0.001507` |

At `t=40`, the corrected uncapped adaptive scores are `0.037626` for all 31
coordinates and `0.028850` for `C`.  They remain better than the ordinary
`K_max=6` path (`0.053379`, `0.063880`) and give the lowest aggregate error on
the fixed-capacity comparison.  Fixed `K_max=10` retains slightly lower `C`
error (`0.027691`) and electronic-energy RMS (`0.00979` versus `0.01009`).
The corrected adaptive total-energy RMS is `0.000654`, the lowest of the
displayed packet paths, while its phonon-energy RMS `0.02447` is worse than the
ordinary `K_max=6` value `0.01522`.  The result remains a Pareto improvement,
now stated under one common packet scoring representation.

Rescoring implementation and artifacts:

- future-run correction:
  `pipelines/open_dynamics/run_mixed_guided_packet_rollout.py`;
- provenance-preserving stored-trajectory rescorer:
  `pipelines/open_dynamics/rescore_mixed_guided_packet_rollout.py`;
- corrected trajectory directories use the suffix
  `_direct_20260805_v2` under `output/local_runs/`;
- the newest-first PDF was rebuilt from only the corrected score artifacts:
  `output/pdf/paper_v_results_progression_20260804.pdf`.

Post-correction verification:

- all eight `_direct_20260805_v2` artifact sets are complete, and every
  artifact SHA-256 digest agrees with its runtime manifest;
- the rebuilt progression PDF contains five pages and its first three pages
  identify the corrected `t=40`, transfer, and central `t=20` results;
- the complete Paper V test suite passes: `228 passed in 67.96s`, using
  `PYTHONPATH=paper_5/src python3 -m pytest paper_5/tests -q`.

### Adaptive packet trajectory as an archive-source teacher

The accurate adaptive-packet observables do not come from propagation of the
archive moment EOM.  The propagated state is the multi-coherent packet ket,
whose parameter velocity is obtained from the McLachlan projection and
integrated with DOP853.  Contractions into `(rho, B, N, A, C)`, the archive
joint Gram, and the mixed observer directions guide packet admission and
provide the archive comparison chart; they do not replace the packet
right-hand side.

The three adaptive double-pulse trajectories were evaluated offline against
the established exact missing-source artifact.  At each stored packet state,
the audit formed

`projected packet C velocity - same-spin-Pauli-repaired archive C velocity`

and compared it with the corresponding exact-Hamiltonian missing source.  The
Pauli-repaired baseline is intentional: this isolates the remaining closure
defect without counting the already identified same-spin algebraic repair a
second time.

| matched three-preparation audit, `0 <= t <= 20` | fixed-capacity parent | adaptive packet |
|---|---:|---:|
| packet-derived source to exact-source normalized RMS | `0.133606` | `0.065061` |
| packet projection to same-state Schrodinger-source normalized RMS | `0.065017` | `0.017580` |
| tangent relative-residual RMS | `0.100032` | `0.056131` |

For the adaptive path before and through the first pulse (`0 <= t <= 8`), the
packet-derived source error is `0.023642`.  Across the full `t=20` comparison,
the frozen exact five-direction decoder reconstructs the adaptive packet
source with normalized RMS `0.004818`; the largest principal angle between
the exact and packet rank-five source spaces is `0.538 degrees`.  Thus the
adaptive packet model improves both the integrated observables and the
instantaneous source that the archive correlation equation is missing.  The
five directions remain an output decoder, not yet an autonomous hidden state.

The first adaptive-source audit exposed a normalization error in the offline
velocity contraction.  Adaptive parameter propagation permits the raw packet
coefficient norm to carry a non-unit gauge scale, whereas the McLachlan state
tangent is already defined for the normalized ket.  The audit divided that
tangent by the raw norm a second time.  `packet_closed_velocity_pair` now
contracts both projected and Schrodinger tangents from the normalized state,
and its regression test compares both derivatives with direct finite
differences.  Adaptive source artifacts `v1`--`v2` are therefore superseded;
the retained result is:

- `output/local_runs/paper_v_adaptive_packet_derived_closure_source_cutoff16_t20_20260805_v4/`.

The matched normalized fixed-capacity comparison is:

- `output/local_runs/paper_v_packet_derived_closure_source_cutoff16_t20_normalized_20260805_v1/`.

The adaptive artifact stores the packet, same-state Schrodinger, and exact
five-direction coefficient trajectories with shape `(3, 81, 5)`.  These are
the immediate inputs for identifying preparation-initialized hidden dynamics;
using them online directly would still import teacher information rather than
produce an autonomous archive closure.

Post-fix regression: the complete Paper V suite passes, `228 passed in
67.15s`.

#### Adaptive source persistence through `t=40`

The central adaptive trajectory was next compared with the exact missing
source through `t=40` using only stored states.  This does not repeat the
previous auxiliary-memory or reachability--observability constructions.

| central source teacher through `t=40` | normalized source RMS | same-state packet-projection RMS |
|---|---:|---:|
| fixed `K_max=6` | `0.204268` | `0.091574` |
| fixed `K_max=8` | `0.160534` | `0.065609` |
| fixed `K_max=10` | `0.126951` | `0.022250` |
| fixed `K_max=12` | `0.151716` | `0.090543` |
| uncapped adaptive, final `K=11` | `0.113141` | `0.023117` |

Over the post-`t=20` window alone, the adaptive source error is `0.148637`,
versus `0.164401` for fixed `K_max=10`.  The frozen five-direction decoder
continues to reconstruct the adaptive packet source with normalized RMS
`0.004955`, and the largest exact-to-packet source-subspace angle remains
`0.538 degrees`.  Adaptive growth therefore gives the best tested integrated
source teacher at `t=40`, although fixed `K_max=10` has a marginally smaller
same-state packet-projection defect.

Artifact:

- `output/local_runs/paper_v_adaptive_packet_derived_closure_source_central_cutoff16_t40_20260805_v1/`.

This result strengthens the packet route as a representable parent model and
offline source teacher.  It does not reopen the already rejected static
decoder, five-mode oscillator, fixed reciprocal auxiliary, finite-horizon
reachable--observable, or moving-atlas constructions.  A further archive-
compatible attempt would have to change the model class to nonlinear,
gauge-invariant compression of the adaptive packet state; otherwise the
scientifically supported endpoint is the adaptive packet replacement plus the
documented negative closure results.

#### Adaptive packet continuation from `t=40` to `t=100`

The trusted central, double-pulse adaptive trajectory is being continued from
its stored `t=40` state rather than rerun.  This is valid because the existing
algorithm already restarts DOP853 at each half-unit adaptive segment.  The
continuation preserves the seven prior admissions and their random-seed
sequence, checks the previously terminal `t=40` admission boundary before the
first new segment, and retains the frozen residual thresholds and admission
objectives.  A one-segment `t=40` to `40.5` preflight completed successfully;
the `t=40` gate did not trigger, the packet count remained `K=11` per
electronic branch, and the physical ket was continuous.  The full run writes
half-unit progress and current-state checkpoints plus a partial trajectory
every two time units.  Exact cutoff-16 states remain absent from the online
velocity and admission decisions and enter only after propagation for scoring.

Planned artifact:

- `output/local_runs/paper_v_archive_gram_adaptive_packet_cutoff16_t100_resume40_20260805_v1/`.

The user stopped the continuation after the gate had supplied sufficient
long-horizon evidence.  The last complete full-trajectory checkpoint is
`t=98`; isolated current-state checkpoints at `98.5` and `99` are not used in
trajectory scores.  The scored stopped artifact is:

- `output/local_runs/paper_v_archive_gram_adaptive_packet_cutoff16_t98_user_stopped_20260805_v1/`.

The frozen gate reached `K=27` packets per electronic branch, with time-average
capacity `12.57`.  Admissions accelerated sharply after `t=90`, demonstrating
that the present native-residual trigger is aggressive at long times.  Peak
resident memory nevertheless remained `224.4 MB`, and every accepted admission
preserved the current ket exactly.

| exact cutoff-16 score, `0 <= t <= 98` | ordinary packet | adaptive packet |
|---|---:|---:|
| all-31 scaled RMS | `0.140791` | `0.063649` |
| site-0 occupation RMS | `0.045221` | `0.013600` |
| electronic-energy RMS | `0.067278` | `0.023320` |
| phonon-energy RMS | `0.061007` | `0.028350` |
| electron--phonon-energy RMS | `0.088403` | `0.033785` |
| total-internal-energy RMS | `0.001188` | `0.000579` |

The adaptive minimum and final exact-state fidelity is `0.923677`, and its
minimum local-cutoff retained norm is `0.998472`.  The minimum electronic
density eigenvalue is `0.058149`; the bosonic and joint-Gram minima are only
`-1.41e-15` and `-1.72e-15`, respectively, i.e. positive semidefinite to
floating-point roundoff.  The correlation-trace residual is exactly zero on
the stored samples.

No minimum-norm archive-EOM physicality correction is present on this route.
The packet ket is normalized before contraction, so representability is
inherited from the ket.  The archive moments and joint Gram are observers and
admission guidance, not the propagated state or an online correction.  The
updated six-page progression report places this result first:

- `output/pdf/paper_v_results_progression_20260804.pdf`.

Post-update verification: the complete Paper V suite passes, `228 passed in
78.66s`; the six-page PDF compiles without overflow or undefined-reference
warnings and mechanically exposes the new `t=98` page first.

#### First executable adaptive positive commutator-moment slice

The August 5 APCM design has been reduced to a first autonomous entrance-layer
prototype.  This implementation is deliberately narrower than the complete
adaptive construction: it propagates the exact 29-real raw archive chart plus
31 preparation-dependent relative-mode moments, evaluates the complete archive
matrix EOM at every right-hand-side call, replaces the identified same-spin
source and adds the connected electron--two-phonon and opposite-spin sources
to `dot C`, closes the degree-four frontier with a positive Pauli--Weyl moment
completion, and retains the established joint-Gram physicality projection.
Adaptive descendant promotion and the memorandum's full lexicographic
moment-metric tangent projection are not yet implemented.

The implementation audit corrected two points before propagation.  First, the
verified Pauli repair is the full archive-source replacement after subtracting
the factorized mixed source; it is not merely the `S0` term used in the design
memorandum's provisional formula.  Second, matrix and hidden initial moments
must be contracted from the same exact ground-state vector.  Independent
eigensolver calls can select inconsistent representatives.  The cutoff-16
contractions also differ slightly from canonical CCR moments at the cutoff
boundary, so the 29 archive coordinates are held fixed while only the hidden
moments receive a minimum scaled canonical-cone retraction (`7.45e-5`).

The frozen strong-coupling protocol uses `lambda=1.5`, `gamma=0.5`,
`g=0.6123724357`, drive amplitude `1`, cutoff `16`, and SSPRK(3,3) with
`h=0.0025`.  The scored autonomous run through `t=1` completed 400 steps and
1,200 archive-backed right-hand-side evaluations.  Its retained joint-Gram
minimum was `2.8919e-5`; the selected extended moment matrix remained feasible
to the declared conic backward-error tolerance (minimum reported eigenvalue
`-8.76e-7`).  Hidden stage retraction was nonzero on 122 accepted steps, with
maximum scaled norm `8.61e-5`.

| exact cutoff-16 score, `0 <= t <= 1` | APCM entrance-layer prototype |
|---|---:|
| all-coordinate scalar RMS | `4.1174e-4` |
| site-0 occupation RMS | `6.8549e-5` |
| static-energy RMS | `1.0606e-4` |
| electron-density Frobenius RMS | `4.7145e-4` |
| anomalous-phonon Frobenius RMS | `1.0587e-3` |
| electron--phonon-correlation Frobenius RMS | `1.8137e-3` |

Exact cutoff dynamics entered only after the autonomous rollout was frozen.
Artifacts:

- `output/local_runs/paper_v_apcm_strong_t025_h0025_20260805_v1/`;
- `output/local_runs/paper_v_apcm_strong_t1_h0025_20260805_v1/`.

Focused verification passes: `11 passed` across the raw chart, source decoder,
positive completion, initialization, and propagation tests.  The next
development run extends the same frozen prototype through `t=4`; its result
tests this specific fixed entrance layer and must not be reported as the full
adaptive APCM construction.

#### Matched `t=4` terminal-cone/controller ablation

The full-cone development run was preserved as baseline A and completed before
any solver optimization.  Two matched ablations then reused A's stored initial
60-vector exactly (`max difference = 0`) and held the Hamiltonian, drive,
cutoff, `h=0.0025`, SSPRK stages, hidden dictionary, and exact offline scorer
fixed:

| run | degree-four terminal rule | retained `G(X)` controller |
|---|---|---|
| A | positive `M4` completion plus hidden-stage retraction | on |
| B | zero-cumulant terminal prior, no `M4` optimization/retraction | on |
| C | zero-cumulant terminal prior, no `M4` optimization/retraction | off |

The independently generated exact arrays agree across scorers within
`2.01e-13`, and B/C share the exact same reference array.  The matched results
through `t=4` are:

| metric | A: full cone | B: prior + controller | C: prior only |
|---|---:|---:|---:|
| all-coordinate scalar RMS | `0.012580` | `0.046475` | `0.049952` |
| `C`-block Frobenius RMS | `0.047255` | `0.103232` | `0.106632` |
| site-0 occupation RMS | `0.006957` | `0.025898` | `0.024029` |
| static-energy RMS | `7.963e-4` | `0.026972` | `0.025087` |
| minimum retained joint-Gram eigenvalue | `2.889e-5` | `1.032e-5` | `-8.226e-3` |
| minimum extended `M4` eigenvalue | `-8.76e-7` | `-1.4734` | `-1.4866` |
| autonomous wall time | `3399.6 s` | `52.9 s` | `27.0 s` |

The small negative A `M4` minimum is inside the declared conic backward-error
allowance; it never crossed `-1e-6`.  In B and C the unoptimized prior was
already strongly indefinite at the shared initial state (`t=0`, about
`-0.983`) and remained so.  The retained controller prevented a joint-Gram
crossing in B.  Without it, C first crossed at `t=0.575` and later re-entered
the retained cone; a temporary return to positive eigenvalues does not restore
physical meaning to the intervening trajectory.  None of the three reached
the declared amplitude threshold by `t=4`.

This ablation separates the mechanisms.  The retained controller supplies
retained representability but does not systematically improve every observable
error.  The positive degree-four completion and its stage containment produce
the large accuracy improvement in A, at a roughly 64-fold runtime cost versus
B.  Solver optimization is therefore justified, but it must preserve the A
trajectory's selected-cone semantics; replacing the completion with the cheap
zero-cumulant prior is not a scientifically equivalent optimization.

Artifacts:

- `output/local_runs/paper_v_apcm_strong_t4_h0025_20260805_v1/`;
- `output/local_runs/paper_v_apcm_ablation_B_prior_controller_t4_h0025_20260805_v2/`;
- `output/local_runs/paper_v_apcm_ablation_C_prior_no_controller_t4_h0025_20260805_v2/`.

The B/C `v2` artifacts supersede `v1` only for the initial `M4` crossing
timestamp; the trajectories and scored errors agree to floating-point
precision.  Version 2 evaluates the terminal matrix at `t=0` instead of
initializing that diagnostic entry as `NaN`.

#### Fourth factorial cell: positive `M4` completion without retained controller

The missing matched cell D completed from baseline A's stored initial
60-vector without regenerating the preparation.  It used the same
strong-coupling single-pulse protocol, cutoff 16, and SSPRK(3,3) step
`h=0.0025`, with positive fourth-moment completion and hidden-stage retraction
enabled and the retained joint-Gram controller disabled.

| exact cutoff-16 score, `0 <= t <= 4` | D: extended `M4` cone only |
|---|---:|
| all-coordinate scalar RMS | `0.0125814` |
| `C`-block Frobenius RMS | `0.0474289` |
| site-0 single-spin occupation RMS | `0.0075912` |
| static-energy RMS | `8.20845e-4` |
| minimum retained joint-Gram eigenvalue | `2.88937e-5` |
| minimum extended `M4` eigenvalue | `-9.68037e-7` |
| maximum hidden-retraction scaled norm | `1.11923e-4` |
| steps with hidden retraction | `1024 / 1600` |
| autonomous wall time | `3268.6 s` |

The small negative `M4` minimum remained inside the existing `1e-6` conic
backward-error tolerance; no retained-joint crossing or amplitude-threshold
crossing was recorded.  The independently generated exact reference agrees
with the other matched scorers within `1.99e-13` in the stored coordinates.

Artifact:

- `output/local_runs/paper_v_apcm_ablation_D_positive_no_controller_t4_h0025_20260805_v1/`.

The first page of
`output/pdf/paper_v_results_progression_20260804.pdf` now presents exact,
raw 31-coordinate, retained-cone 31-coordinate, and all four higher-moment
cone combinations with descriptive labels.  The stored 31-coordinate native
grid is nested exactly in the APCM grid; it is interpolated only for the visual
overlay, while its tabulated errors use its native grid.  The rebuilt
seven-page PDF has no layout warnings, and its changed first page was rendered
and visually checked for clipping and legibility.  No route selection or
promotion was made from this ablation.

#### Exact spin-exchange cone optimization and parity decision

The GitNexus index was refreshed in index-only mode after admitting `paper_5`
to the repository graph.  The resolved call graph shows three ordered calls to
`ArchiveBackedAPCM.evaluate` and three ordered feasibility checks per
`integrate_apcm_ssprk3` step.  Each failed warm witness enters
`ensure_extended_feasible -> retract_hidden_state ->
PositiveFourthMomentCompletion.retract_lower_moments`.  Direct source tracing
supplies the composition edges that GitNexus cannot infer through
`self.completion`: `ArchiveBackedAPCM.evaluate` calls `complete`, while
`retract_hidden_state` calls `retract_lower_moments`.  SSPRK stages and time
steps are therefore mathematically dependent and were not parallelized.

The degree-two Pauli--Weyl word Gram has an exact spin-exchange involution.
Every moment coefficient commutes with that involution, so one fixed unitary
change of word basis gives

```text
U^dagger M4 U = M4_symmetric direct-sum M4_antisymmetric,
33 by 33        21 by 21                  12 by 12.
```

The transformed coefficient matrices are now assembled in one batched
contraction.  An opt-in candidate imposes the two exact PSD blocks and bounds
Clarabel's `faer` backend to four threads.  The serial full-cone implementation
remains the default.  A direct reconstruction test verifies the block identity
within `2e-13`; no approximation is made in the matrix representation.

Solver-level parity nevertheless failed.  On 23 stored Run-A states selected
uniformly across the horizon and at cone/retraction extrema, the ordinary
completion and retained joint-Gram correction agreed exactly.  The retraction
did not: maximum differences were `1.30925e-4` in the 31 adjustable lower
moments, `30.5366` in the weakly regularized fourth-order frontier,
`2.13762e-5` in the `M4` minimum eigenvalue, and `6.79212e-10` in the retraction
objective.  Thus the two conic formulations describe the same mathematical
feasible set but terminate at numerically different near-degenerate retraction
solutions under the frozen tolerance.

The matched `t=0.05` trajectories confirmed that the difference propagates.
The four-thread block candidate reduced autonomous wall time from `24.7029 s`
to `5.01542 s` (`4.925x`), and its plotted observables remained within
`3.94505e-7`.  It failed the state/cone contract: the maximum 60-coordinate
difference was `2.25666e-4`, and the maximum `M4` eigenvalue-history difference
was `3.41165e-6`.  The retained joint-Gram history agreed within `8.92e-15`.

For the full-horizon check, the stored `t=1` APCM artifact was verified to be
an exact prefix of Run A and supplied both the identical 60-coordinate state
and its 35-coordinate frontier.  Run A's prefix was preserved, and the block
candidate continued autonomously from `t=1` to `t=4`.  The candidate suffix
took `624.871 s`, versus `2901.05 s` inferred from the matched serial prefix
and full timings (`4.643x` suffix speedup).  The preserved-prefix composite
took `1123.41 s` versus `3399.59 s` for Run A (`3.026x`).  This speedup is not
valid for replacement because full parity failed:

| full `0 <= t <= 4` difference from Run A | maximum | RMS |
|---|---:|---:|
| 60-coordinate state | `0.245845` | `0.0239575` |
| 31 archive coordinates | `0.0393587` | `0.00534403` |
| six plotted observables | `0.107570` | `0.0170387` |
| extended-`M4` eigenvalue history | `4.21453e-4` | `9.35248e-5` |
| retained joint-Gram history | `5.75189e-9` | `1.44531e-9` |

Canonical parity artifacts:

- `output/local_runs/paper_v_apcm_m4_parallel_parity_20260805_v2/`;
- `output/local_runs/paper_v_apcm_m4_parity_short_serial_t005_20260805_v1/`;
- `output/local_runs/paper_v_apcm_m4_parity_short_block4_t005_20260805_v1/`;
- `output/local_runs/paper_v_apcm_m4_parity_full_block4_t1_t4_20260805_v2/`;
- `output/local_runs/paper_v_apcm_m4_full_parity_20260805_v4/`.

The first full candidate attempt reached `t=4` but stopped during offline
scoring because the exact-reference helper required a grid beginning at zero;
its partial artifact remains preserved as `...t1_t4_20260805_v1/`.  The runner
now serializes the autonomous trajectory before offline scoring and handles a
continuation grid by prepending the exact `t=0` sample.  The complete Paper V
test suite passes (`244 passed in 101.93 s`).  No optimized implementation was
promoted, no scientific route was selected, and the serial Run-A semantics
remain unchanged.

#### Accelerated higher-moment cone routes through `t=20`

The exact spin-exchange block representation was next treated as its own
exploratory numerical path rather than as a parity-preserving replacement for
serial Run A.  Two fresh, matched strong-coupling trajectories used the same
exact-correlated cutoff-16 initialization, single Gaussian pulse,
`lambda=1.5`, `gamma=0.5`, `v=1`, SSPRK(3,3), `h=0.0025`, and four bounded
Clarabel/faer threads:

- retained joint-Gram controller plus positive `M4` completion;
- positive `M4` completion without the retained joint-Gram controller.

Both autonomous trajectories reached `t=20` without an amplitude-threshold
event, a retained joint-Gram crossing, or an extended-`M4` violation beyond
the declared `1e-6` conic backward-error tolerance.  Exact cutoff-16 DOP853
propagation entered only after each rollout for scoring.

| metric over `0 <= t <= 20` | both cones | `M4` cone only |
|---|---:|---:|
| 31-coordinate scalar RMS error | `0.128607` | `0.128841` |
| `C`-block Frobenius RMS error | `0.147742` | `0.146016` |
| total site-0 occupation RMS error | `0.204690` | `0.192012` |
| internal-energy RMS error | `9.39628e-4` | `9.39629e-4` |
| minimum retained joint-Gram eigenvalue | `1.36819e-5` | `2.66504e-5` |
| minimum extended-`M4` eigenvalue | `-9.89271e-7` | `-9.82547e-7` |
| maximum retained-cone correction norm | `0.0702218` | `0` |
| maximum hidden-stage retraction norm | `0.00146835` | `0.00122659` |
| steps with hidden retraction | `5998 / 8000` | `5866 / 8000` |
| autonomous wall time | `5832.34 s` | `5954.77 s` |

The differences are mixed: both cones give the slightly smaller aggregate
31-coordinate error, while `M4` alone gives the smaller occupation and
correlation-block errors.  The physical trajectories separate substantially
from the exact observables after the pulse despite preserved cone margins, so
representability remains distinct from closure accuracy.

Artifacts:

- `output/local_runs/paper_v_apcm_spin_exchange_blocks_controller_t20_h0025_20260805_v1/`;
- `output/local_runs/paper_v_apcm_spin_exchange_blocks_no_controller_t20_h0025_20260805_v1/`.

The rebuilt eight-page
`output/pdf/paper_v_results_progression_20260804.pdf` adds exactly one new top
page containing matched observables, instantaneous 31-coordinate and
`C`-block errors, both cone-eigenvalue histories, controller/retraction action,
and the quantitative summary.  The `M4`-only trajectory is rendered as a dark
dotted line so the comparison does not depend on color.  The new page was
rendered and visually checked; all seven earlier pages remain in their prior
order.

#### Active overnight both-cones continuation

The completed both-cones state and exact completion frontier at `t=20` seed an
autonomous continuation to `t=240` with the same `h=0.0025`, exact
spin-exchange blocks, four-thread bound, and both physicality cones.  The
target was selected from the measured throughput to occupy approximately the
requested nine-hour overnight window.  The process is protected against
system sleep and writes progress plus atomic checkpoints under:

`output/local_runs/paper_v_apcm_spin_exchange_blocks_controller_t240_from_t20_h0025_20260805_v1/`.

No exact-reference data enter the active rollout.  Long-horizon scoring and a
new progression page remain pending completion; the route is not promoted by
starting this run.

#### Interrupted checkpoint at `t=31` and persistent continuation

The terminal-bound overnight process did not remain alive after its execution
session ended.  This was a process-lifecycle failure, not a numerical or
physicality failure.  Its last atomic checkpoint is finite at `t=31`, step
`4400`, with minimum retained joint-Gram and extended-`M4` eigenvalues
`1.224e-4` and `3.032e-5`.  The 60-coordinate state and 35-coordinate
completion frontier were preserved.  Intermediate display samples over
`20 < t < 31` were not serialized.

Offline exact cutoff-16 scoring of that endpoint gives:

| `t=31` endpoint metric | value |
|---|---:|
| 31-coordinate scalar RMS error | `0.292044` |
| `C`-block Frobenius error | `0.274058` |
| total site-0 occupation absolute error | `0.665627` |
| internal-energy absolute error | `9.52185e-4` |

The nine-page
`output/pdf/paper_v_results_progression_20260804.pdf` now begins with the
exact observable curves through `t=31`, the complete stored both-cones path
through `t=20`, and a disconnected dark checkpoint marker at `t=31`.  No curve
is fabricated across the missing sampling interval.  The page was rendered
and visually checked.

Continuation now runs through the private, allowlisted localhost runner rather
than a chat-owned terminal session.  Run
`20260806T140200Z-c00672ee` resumes directly from the preserved `t=31` state
and frontier, targets `t=240`, checkpoints atomically, and writes to:

`output/local_runs/paper_v_apcm_spin_exchange_blocks_controller_t240_from_t31_h0025_20260806_v1/`.

The first logged accepted step is `t=31.0025`, with joint-Gram and `M4`
margins `1.222e-4` and `3.097e-5`.  At launch the process used approximately
`121 MB` resident memory and the runner reported one active job.  Exact data
remain offline and do not enter the continuation.

A live check at 09:21 CDT confirmed that runner job
`20260806T140200Z-c00672ee` and Python process `21089` remain active.  The
process used about `189%` CPU and `101 MB` RSS after 19 minutes; the runner
reported one active job.  The latest serialized progress remains `t=31.0025`.
A read-only stack sample placed the active computation inside Clarabel's Faer
KKT factorization for the positive fourth-moment completion, so the process is
computing rather than idle, but no progress beyond the serialized value is
claimed.  No sustained memory growth is present.

The first two results pages now include the uncorrected archive EOM alongside
the exact reference and higher-moment trajectories in every observable panel.
The archive curve reuses
`paper_v_archive_observable_trajectories_t1000_20260803_v1/`
`raw_refined_rk4_dt005_trajectory.npz`; the `t=31` exact curve reuses the
stored checkpoint score.  Their initial 31 coordinates match the higher-moment
initialization within `2e-13`, and the reused raw trajectory matches the
independent stored `t=20` raw calculation within `6.1e-8`.  Neither reference
was repropagated for this update.  Both changed PDF pages were rendered and
visually checked.

The active persistent process was not restarted from `t=20`; it resumes from
the valid `t=31` state and therefore cannot recover the missing display samples
over `20 < t < 31`.  At 09:26 CDT it remained active at roughly `191%` CPU and
`103 MB` RSS, but `t=31.0025` was still the latest serialized progress.  A
continuous replacement should begin from the stored `t=20` state and serialize
partial trajectory samples at atomic checkpoint intervals, not merely the
current state and completion frontier.

The stored comparisons quantify the scientific objective.  Over `0 <= t <=
20`, the uncorrected archive versus both-cones errors are respectively:

| error | uncorrected archive | both cones |
|---|---:|---:|
| 31-coordinate scalar RMS | `0.193205` | `0.128607` |
| `C`-block Frobenius RMS | `0.390951` | `0.147742` |
| site-0 occupation RMS | `0.286170` | `0.204690` |
| internal-energy RMS | `0.0182361` | `0.000939628` |

At `t=31`, both-cones still improves the 31-coordinate error (`0.292044`
versus `0.439929`), `C` error (`0.274058` versus `0.457487`), occupation error
(`0.665627` versus `1.05629`), and internal-energy error (`9.52e-4` versus
`0.0150113`).  The uncorrected joint-Gram eigenvalue is `-0.489880`, while both
corrected cone margins are positive.  Individual electronic, phonon, and
electron--phonon energy-component errors at `t=31` are mixed and are not
uniformly improved.  The PDF now uses black for exact, red dash-dot for the
uncorrected archive, blue dashed for both cones, and green dotted for the
`M4`-only route; pages 1 and 2 were rebuilt and visually checked.

### Polarization peak and observed-width audit across coupling

A matched single-pulse weak-coupling trajectory was generated at
`g/t_hop=0.3535533905932738` (`lambda=0.5`), `gamma=0.5`, through `t=100`
from the exact-correlated cutoff-16 zero-field ground-state contractions. The
raw and retained-cone-corrected 31-coordinate moment EOM used `h=0.01`; the
exact cutoff-16 wavefunction was propagated afterward with DOP853 for offline
scoring. The matched run is stored at:

`output/local_runs/paper_v_g035355_single_pulse_raw_vs_corrected_t100_20260806_v1/`.

Polarization spectra for that run and the existing matched strong-coupling run
at `g/t_hop=0.6123724356957945` (`lambda=1.5`) were computed on the common
post-pulse interval `10 <= t <= 100`, sampled at `0.2`, after mean subtraction
and a Hann window. Peak centers and interpolated observed full widths at half
maximum were measured in `1.5 <= omega/t_hop <= 3.5`; the frequency-bin spacing
is `0.0696584 t_hop`. These widths include finite-window and window-function
broadening and are not deconvolved lifetimes.

| `g/t_hop` | trajectory | Hellinger distance | peak center | observed FWHM | electronic-band weight |
|---:|---|---:|---:|---:|---:|
| 0.3536 | exact cutoff-16 | 0 | 2.0901 | 0.0940 | 0.8795 |
| 0.3536 | uncorrected archive EOM | 0.1464 | 2.0872 | 0.1171 | 0.9059 |
| 0.3536 | regular EOM correction (31D joint-Gram) | 0.1958 | 2.0889 | 0.0944 | 0.6841 |
| 0.6124 | exact cutoff-16 | 0 | 2.3561 | 0.1145 | 0.0569 |
| 0.6124 | uncorrected archive EOM | 0.6381 | 2.9399 | 0.1557 | 0.5225 |
| 0.6124 | regular EOM correction (31D joint-Gram) | 0.1782 | 2.4811 | 0.1395 | 0.0026 |

The raw archive EOM therefore preserve the main electronic peak position at
weak coupling but give an observed width about 25 percent too large. At
strong coupling the raw peak shifts upward by `0.5838 t_hop`, is about 36
percent too broad, and carries far too much normalized high-frequency weight.
The cone controller improves the strong-coupling whole-spectrum distance but
nearly removes the exact electronic-band weight, confirming that
representability control is not an accuracy closure. At weak coupling it
recovers the observed width but worsens the whole-spectrum distance by moving
weight into the low-frequency band.

The generated diagnostic is:

`output/plots/paper_v_results_progression_20260804/archive_polarization_peaks_g_comparison.{pdf,png,json}`.

The extraction implementation is
`pipelines/open_dynamics/analyze_archive_polarization_peaks.py`; tested
quadratic peak interpolation and half-maximum width extraction are shared with
`pipelines/open_dynamics/analyze_archive_observable_spectra.py`.

### Explicit four-route spectral comparison

The phrase `31-coordinate cone correction` was ambiguous once the
higher-moment route also imposed cone conditions.  Spectral reporting now uses
four fixed names:

1. `exact_cutoff16`: exact cutoff-16 Hamiltonian propagation;
2. `archive_eom`: the uncorrected 31-coordinate archive moment EOM;
3. `regular_eom_correction`: those same EOM plus the minimum-Euclidean-norm
   31-coordinate joint-Gram velocity correction;
4. `apcm_m4_prototype`: the implemented 60-coordinate entrance-layer APCM
   prototype with positive `M4` completion, hidden-stage retraction, and the
   retained joint-Gram controller.

The fourth route is related to the proposed McLachlan-type APCM construction
but does not implement its full adaptive moment-metric projection.  It is
therefore labeled `prototype` in every reader-facing output.  The canonical
route definitions and one-command regeneration workflow are recorded in
`paper_5/notes/archive_spectral_comparison_routes_20260806.md`; the generated
JSON records model definitions, source hashes, parameters, validation, and
spectral metrics.

The four-route comparison uses the longest common post-pulse interval,
`4 <= t <= 20`, because the stored `M4` trajectory ends at `t=20`.  All four
routes share the same exact-correlated cutoff-16 initialization within
`3e-12`, and the exact arrays stored by the long archive run and by the `M4`
scorer agree on the analysis grid within the generated manifest's tolerance.
With common `0.2` sampling and a Hann window, the frequency spacing is
`0.387851 t_hop`; the following widths are consequently resolution-dominated
observed FWHM values:

| model | Hellinger distance | RMS amplitude / exact | peak center | observed FWHM | electronic-band weight |
|---|---:|---:|---:|---:|---:|
| exact cutoff-16 | 0 | 1.000 | 2.3411 | 0.4619 | 0.6741 |
| archive EOM | 0.1701 | 1.522 | 2.3549 | 0.5450 | 0.7404 |
| regular 31D EOM correction | 0.5868 | 0.873 | 2.6976 | 0.4636 | 0.0218 |
| APCM `M4` prototype | 0.3708 | 0.868 | 2.4863 | 0.7905 | 0.3116 |

On this short common horizon the raw archive EOM give the nearest peak and
smallest whole-spectrum distance, while overestimating the oscillation
amplitude by 52 percent.  The regular correction's width is numerically close
to exact only after it suppresses almost all electronic-band power.  The `M4`
prototype restores more of that band than the regular correction but remains
shifted and broader.  The longer `10 <= t <= 100` analysis remains the more
precise peak/broadening audit for the three routes that reach that horizon.

Artifacts:

- `output/plots/paper_v_results_progression_20260804/archive_m4_four_route_polarization_spectrum.{pdf,png,json}`;
- `pipelines/open_dynamics/analyze_archive_m4_polarization_spectra.py`.

### Full APCM implementation: resource failure and safe completion seam

Work began on the full moment-metric APCM construction from
`2026-08-05-adaptive-positive-commutator-moment-mclachlan-design.md`, rather
than extending the 60-coordinate `M4` prototype.  The new implementation adds
the exact inverse differential of the 29-real raw-moment reconstruction, the
lifted retained metric, the extended active-operator moment matrix, the
retained-first tangent projection, and coupled stage-retraction seams in:

- `paper_5/src/paper5/stability/adaptive_positive_moment.py`;
- `paper_5/src/paper5/stability/apcm_positive_extension.py`;
- `paper_5/src/paper5/stability/apcm_moment_projection.py`.

The first extended-completion backend attempted to pass a 40-by-40 complex
log-determinant matrix with 129 affine frontier variables through CVXPY.  Its
dense canonicalization exhausted local memory and contributed to a machine
crash.  The reconstructed size metrics were 129 scalar variables and 285,236
scalar data entries; the complex log-det realification has a conservative
working-set estimate of approximately 21 GiB for one process.  Two completion
attempts may have overlapped before the crash, consistent with the observed
roughly 38 GiB resident-memory report.  No scientific trajectory produced by
that process is retained.

The dense route is now opt-in and rejected before solver canonicalization when
its estimate exceeds a 512 MiB budget.  The default completion backend uses
single-thread direct Clarabel quadratic SDPs plus matrix-sized Newton work
arrays; it never constructs the dense CVXPY log-det graph.  A strictly
feasible product functional completes in the fast relative interior.  For the
strong-coupling correlated initial state, the construction identifies a
39-dimensional relative matrix face and 85 feasible frontier directions and
then converges to a positive proximal log-det completion with reduced-gradient
residual `8.58e-8` in the construction test (`tau_A=1e-3`,
`epsilon_A=1e-10`, conic tolerance `1e-7`).  This is a solver-construction
check, not a promoted physical result or trajectory.

Resource and geometry regressions are in
`paper_5/tests/test_apcm_resource_safety.py` and
`paper_5/tests/test_apcm_moment_projection.py`.  They verify preflight
rejection, absence of a dense graph in the default backend, a bounded physical
completion, bidirectionality of the raw-coordinate reconstruction
differential, and exact bypass when a requested velocity is already viable on
the selected relative face.  The full projected model is not yet wired into a
trajectory integrator; no further APCM scientific comparison should be run
until that replacement is complete and its remaining conic paths pass the
same memory gate.

### RAM-safe fixed-dictionary APCM and the first prescribed adaptation stop

The fixed-dictionary replacement is now wired independently of the earlier
60-coordinate `M4` prototype. Its default state contains the 29 raw archive
coordinates and only the 15 symmetry-reduced `T/U` entrance moments required
by the `K/P/D` source, for 44 real propagated coordinates. The entrance
extension has 24 Gram rows and 62 terminal frontier variables. The archive
matrix EOM are evaluated at every RHS call, the `K/P/D` increment changes only
the `C` target, every active auxiliary velocity is compiled from the exact
canonical commutator, and the retained-first/auxiliary-second conic projection
selects a common viable velocity. SSPRK(3,3) now retries a declared failed
stage by recursive time-step subdivision and records every attempted RHS.

All completion, velocity-projection, and stage-retraction paths now use
single-thread direct Clarabel problems. CVXPY's dense complex log-det route
is opt-in and rejected by a preflight estimate above 512 MiB. The default
entrance stage-retraction workspace estimate is below 8 MiB. A correlated
initialization plus one RHS used about 273 MB RSS; one complete
`h=0.0025` SSPRK step used about 331 MB RSS, three RHS calls, no retraction,
and approximately 7.8 seconds. The former 20--40 GiB canonicalization path is
not reachable from the default backend.

Preparation contractions were also corrected to follow the memorandum's
canonical embedding rule. Requested Weyl words are now reduced and normally
ordered in the canonical CCR algebra before their polynomials are evaluated
with cutoff-16 matrices. This changes the 15 entrance coordinates by up to
`1.16153e-5` relative to direct multiplication of truncated quadratures. The
old contraction was therefore not an acceptable APCM initialization even
though the difference was numerically small.

The one-additional-shell closure-graph diagnostic was then executed at the
canonical correlated initial state. It conditions the 15 RHS-facing frontier
values selected by the current completion, reopens the auxiliary frontier,
and adds candidate rows and their covariance products. The resulting
diagnostic matrix has 39 rows and 121 free frontier moments. Its reference
conic problem returns the declared
`nested_completion_failure: ... PrimalInfeasible` at `t=0`. This result is
unchanged for `0.1*tau_A`, `epsilon_A=1e-12`, and `epsilon_A=1e-8`; the
`10*tau_A` construction fails the base selector's KKT acceptance test before
the nested audit. Peak RSS across these isolated diagnostics was 0.69--0.90
GB, with no overlapping process.

The obstruction is not caused by the six bosonic descendants: their joint
30-row extension is feasible with scaled minimum eigenvalue within
`5.2e-16` of zero and KKT error `1.7e-9`. It appears when the nine
spin--phonon descendant rows are conditioned jointly. Smaller subsets and
individual rows can be PSD-extendible, so this is a collective covariance
incompatibility of the selected terminal values rather than one oversized
coordinate. Direct cutoff moments are not a substitute feasibility proof:
the finite-cutoff boundary commutator makes their canonically assembled
39-row matrix miss PSD by `3.71e-6` in scaled eigenvalue.

The Ultra design explicitly requires an empty nested fiber to stop as
`nested_completion_failure`; it forbids silently rebuilding another current
frontier. Therefore no long APCM rollout or exact-reference scoring was
started. Promotion, pruning, and trajectory claims would be unreachable for
this frozen first pilot unless the mathematical construction itself is
amended. The current evidence is an implementation-level rejection of this
specific all-RHS-conditioned nested selector, not a rejection of every
adaptive moment closure.

Regression status after these changes: 34 focused APCM/exact-reference tests
passed, followed by the complete Paper V suite at 264/264 passing. No APCM,
pytest, or scientific-run process remained active afterward.

### Theory-to-code audit after the nested-selector stop

A direct audit against the 2026-08-05 APCM memorandum found and repaired one
specific one-shell construction omission.  The first diagnostic treated all
Liouvillian descendant moments as scalar entries of its Gram matrix, but it did
not append the descendant operators themselves as diagnostic half-words.  The
memorandum requires those half-words so that the additional positivity shell
contains their covariances.  There are 21 such descendant half-words for the
initial entrance dictionary.  Appending them changes the formal diagnostic
matrix from 39 to 60 rows and its free frontier from 121 to 223 moments.  The
former 39-row diagnostic is a principal submatrix of the corrected 60-row
diagnostic, so its certified infeasibility already implies infeasibility of the
larger declared shell at the same conditioned values; the scientific stop is
unchanged, but the implementation now represents the stated shell correctly.

Two additional invariant tests now verify that the 15 symmetry-reduced entrance
moments reproduce exactly the `K/P/D` source obtained from the complete
degree-three chart and that the frozen-Hamiltonian gradient is the derivative
of the declared affine energy.  They also verify that the `K/P/D` source adds
zero instantaneous frozen-Hamiltonian energy flux.

The audit also identified remaining implementation scope that must not be
confused with a completed APCM rollout.  The existing
`adaptive_positive_moment_analysis` command still executes the older
60-coordinate fourth-moment completion/controller prototype; the new
29-plus-active projected model has an integrator but no production analysis
command.  Its 15-coordinate entrance basis is functionally verified for the
required source, but the memorandum's explicit rational `48 -> r_0` `T/U`
compiler and immutable affine-map hashes are not implemented.  The current
relative-mode extended Gram uses nine reduced half-words and enforces the
retained 8-by-8 matrix separately; consequently, the memorandum's claim that
the retained matrix is literally a principal submatrix of the extended matrix
has not yet been implemented or replaced by a proved symmetry-reduced
congruence.  Dictionary promotion, the mandatory unresolved queue,
twice-reorthogonalized support selection, transfer homotopy, immutable
admission receipts, and the accepted-step retraction/work audit also remain
unimplemented.  These are theory-fidelity gaps, not failed scientific tests.

### URPG projective-guard amendment: pre-rollout implementation

The 2026-08-06 projective-guard amendment supersedes coordinate-frozen nested
conditioning.  Its first pre-rollout tranche is implemented in
`paper_5/src/paper5/stability/apcm_projective_guard.py`.  The compiler keeps
the eight structural Hamiltonian words separate and forms the exact row space
of their terminal dependence on the active commutator targets.  For the
15-coordinate entrance dictionary, the current positive extension has 62
frontier moments but the invariant target image has rank 15 and a
47-dimensional exact nullspace.  The former diagnostic therefore fixed 47
directions that the online vector field did not identify.

The same module records the immutable 15-key entrance chart and verifies that
the `K/P/D` source is independent of every omitted hidden degree-three key.  It
also constructs the literal 11-row core
`(I,b0,b1,b0^dagger,b1^dagger,up XYZ,down XYZ)`.  Its first eight rows are
exactly the retained affine joint Gram, and its relative-mode restriction
agrees with a direct Pauli--Weyl construction.  Exact prefix restriction and
composition tests are included.

Reopening, rather than promoting, the 15 old RHS-facing values changes the
relative outer fixture from 223 to 238 free frontier moments.  A second
dimension correction is required for the unified cone: replacing the old
nine-row relative core by the literal 11-row core changes the current and
outer Gram dimensions from 24 and 60 to 26 and 62.  The amendment's phrase
"unified 60-row problem" is therefore not dimensionally compatible with its
declared literal core unless two other rows are explicitly removed.  The code
uses the mathematically literal 62-row count and provides the exact congruence
that restricts it back to the 60-row relative system.  The two omitted core
directions are the center-mode quadratures; their cross moments with the
noncore rows are genuine free completion data in a fully unified solve.

The new selector is a native single-thread Clarabel QP+PSD construction with
an explicit invariant target variable, followed by a minimum-norm lift.  It
does not use log determinant.  `AlmostSolved` is not accepted.  Fresh tighter
and relative-face retries preserve that rule.  On a strictly feasible
manufactured state, the current selector and all four relative outer problems
(boxed/unboxed, conditioned/reopened) solve and classify the target as
`boxed_conditioned_feasible`.  That 60-relative-row fixture required about
272 seconds and approximately 2.4 GiB peak resident memory, which confirms the
need for the amendment's resource preflight before a unified 62-row solve.

At the actual correlated cutoff-16 preparation, the current invariant target
stage remains numerically unresolved before the outer audit.  The final fresh
explicit-target result is `AlmostSolved`, with primal residual
`9.7868e-11`, dual residual `6.7990e-10`, and scaled minimum eigenvalue
`-1.44e-15`; the facial retry does not supply a `Solved` certificate.  The
nominal values satisfy small residuals, but the frozen amendment explicitly
classifies `AlmostSolved` as unresolved.  No outer classification, promotion,
time step, rollout, or exact-reference score was inferred from that point.

Nine focused projective-guard tests pass.  They cover the entrance partition,
source independence, exact target nullspace, literal retained prefix,
relative/core congruence, restriction-map composition, reopened outer shell,
unified/relative gluing, strict-fixture selection, and refusal to accept the
unresolved correlated-preparation status.

### Exploratory URPG continuation after relaxing status-only gates

The strict certification path above is preserved, but it was too restrictive
for exploratory construction: the correlated preparation had a physically
feasible iterate that was discarded solely because Clarabel returned
`AlmostSolved`.  `GuardConicResult` now distinguishes certified, provisional,
and rejected results.  A noncertified result is provisionally usable only when
Clarabel reports `AlmostSolved` and independently recomputed conic
feasibility, stationarity, complementarity, and relative duality-gap residuals
are all at most `1e-6`.  This tolerance changes neither the strict certificate
gate nor any scientific infeasibility claim.

With that exploratory profile, the actual cutoff-16 correlated preparation
completed the four-way relative outer audit in 820 seconds.  Peak resident
memory was approximately 2.7 GiB and repeatedly returned below 1 GiB between
sequential solves; there was no memory leak.  The current target stage was
provisionally feasible with independent feasibility `9.79e-11`, stationarity
`2.49e-9`, complementarity `7.32e-10`, and relative gap `3.86e-8`.  Both outer
problems conditioned on that target were `PrimalInfeasible`, whereas the boxed
and unboxed reopened outer problems were provisionally feasible.  The correct
classification is therefore `nonextendible_target_image`: minimizing the
invariant image on the current cone still chose a target outside the
projection of the next shell.

The selector was consequently changed to minimize the same invariant target
image directly on the one-shell outer cone, then restrict one common feasible
witness to the current variables.  On the cutoff-16 preparation, this
outer-aware target stage was provisionally accepted with independent
feasibility `3.21e-10`, stationarity `3.86e-8`, complementarity `9.23e-9`,
relative gap `7.46e-7`, and scaled minimum eigenvalue `-2.08e-14`.  Its target
image differs from the rejected current-only target by `0.5193246` in the
declared target Euclidean norm.  The separate minimum-norm lift conditioned on
that image ended in `NumericalError`; it is not required for a feasible
exploratory witness because the target-stage solution already contains one.
The code records that distinction as a noncanonical `outer_target_stage`
witness rather than silently describing it as the minimum-norm lift.

The literal 62-row unified Gram does not require another large conic solve at
this stage.  The 11-row literal core and 60-row relative extension are two PSD
cliques sharing the same nine-row relative core.  Their canonical chordal Gram
completion sets the center-to-outer block through the Moore--Penrose inverse
of that shared block.  For the outer-aware cutoff-16 witness, the resulting
62-row Gram restricts to the 60-row matrix with infinity-norm error
`3.50e-15` and has minimum eigenvalue `-2.89e-11`; the current restriction has
scaled minimum eigenvalue `-8.15e-16`.  These are roundoff-scale PSD defects,
not evidence of a missing literal-core constraint.

The preparation obstruction is therefore repaired at the exploratory level,
but no trajectory has been started.  The outer target solve took about 500
seconds even when the failed secondary lift was skipped.  Recomputing that
selector at every SSPRK/RK stage is not an executable controller.  The next
implementation problem is a fast, state-dependent evaluation of the same
outer-feasible selector and its guard-shadow velocity (for example through a
validated active-face/KKT continuation with fallback), followed by a short
fixed-dictionary rollout.  Promotion and deeper shells remain disabled until
that per-stage path is both faithful and computationally viable.

Regression status after the exploratory continuation: 36 affected
projective-guard/APCM tests passed, followed by the complete Paper V suite at
278/278 passing.  No scientific rollout was active at handoff.

### URPG per-stage acceleration audit

The exploratory target-stage gate was moved ahead of the expensive retry
ladder: a first-pass `AlmostSolved` point is used only when the independent KKT
audit passes.  On the same cutoff-16 outer selector this reduced preparation
time from approximately 500 seconds to 171 seconds without changing the
objective (`0.01126712315`), target witness, PSD margins, or independent
residuals.  This is a 2.9-fold improvement but remains unusable at every
Runge--Kutta/SSPRK stage.

Two local accelerations were then tested.  First, a fixed-face selector was
compiled from the accepted preparation witness.  The outer Gram has one
numerically null direction, but the resulting affine face map has singular
values extending from order one to order `1e-20`.  Consequently, roundoff in
the nominal witness is amplified into order `1e-3` frontier-coordinate error.
On the actual outer cone, the frozen-face solve took 117 seconds, returned
`PrimalInfeasible`, and produced minimum scaled eigenvalue `-0.04746`.  It is
rejected.  The compiled matrices and complete diagnostic are preserved under
`output/local_runs/paper_v_urpg_outer_face_cutoff16_20260806_v1/` so the
171-second preparation solve need not be repeated.

Second, a PSD cutting-plane QP was implemented.  Each iteration replaces the
matrix cone by all accumulated Rayleigh-quotient inequalities and then adds
every currently negative eigenmode.  The method certifies a simple strict
fixture in six cuts, but on the correlated preparation its minimum-target
objective leaves a large null lift space: the minimizer rotates through new
negative modes and remains indefinite after 512 cuts.  A small frontier
regularizer does not cure this without becoming large enough to change the
declared lexicographic selector.  This route remains a guarded diagnostic and
is not connected to propagation.

A warm-started CVXPY/SCS prototype was also tested on the exact same 60-row
outer problem.  At `eps=1e-5` it required 80.1 seconds and 16,550 iterations;
despite reporting `optimal`, the reconstructed Gram had minimum scaled
eigenvalue `-3.15e-5`.  It therefore fails both the runtime and physicality
requirements.  Its result is stored as `scs_prototype.json` beside the frozen
face artifact.

No autonomous rollout was started.  The scientific preparation obstruction is
repaired, but the literal one-shell selector and unique-lift tie-break are too
ill-conditioned and expensive for stagewise use with the available conic
backends.  Proceeding now requires a mathematical/numerical redesign of the
selector representation (not another integrator refinement), or an explicit
decision to weaken the amendment's every-stage outer-guard requirement.

Regression status after the acceleration audit: all 36 affected tests and the
complete 278-test Paper V suite pass.  No selector benchmark, test process, or
scientific trajectory remained active afterward.

### CWRMF carried-witness implementation and short radial pilot

The 2026-08-06 dynamically executable projective-guard amendment was
implemented as a new carried-witness radial moment flow in
`paper_5/src/paper5/stability/apcm_carried_witness.py`, with the executable
driver in `apcm_carried_witness_analysis.py`.  The causal state contains 44
retained archive-plus-entrance coordinates and 340 independent completion
coordinates.  The latter comprise 238 relative frontier moments and 102 real
center/relative cross moments, yielding a 384-real-coordinate state and one
literal affine 62-by-62 unified Gram.  This avoids recomputing a fresh global
positive completion at every stage.

The canonical cutoff-16 correlated preparation is contracted once through a
degree-nine normal-ordered hierarchy.  Its 62-row restriction residual is
`1.97e-15`, its center/relative factorization residual is `5.37e-16`, and its
shifted coefficient-scaled Gram lower bound is `8.99e-12` for the declared
`tau_psd=1e-11` guard.  The compiler identifies 231 completion rates whose
commutators are readable from the carried registry; only the remaining 109
directions receive the minimum-motion prior.

Every forward-Euler atom evaluates the complete archive retained velocity
first and leaves it fixed.  The initial implementation incorrectly treated all
readable completion rates as hard equalities.  That branch became numerically
unresolved near `t=0.0075`.  The amendment specifies a readable-rate residual
as the next lexicographic tier when exact locking conflicts with finite-step
positivity.  Implementing that tier repaired the stop: the local solve now
minimizes the readable-rate residual with a frozen `1e-8` physical-rate
regularizer on its numerically flat completion fiber, while the authoritative
acceptance remains a full eigendecomposition of the shifted 62-row Gram.  This
is an executable regularized realization of the intended lexicography, not a
full primal-dual QSDP certificate.

SSPRK2 rollouts were completed through `t=0.05` at `h=0.0025`, `0.00125`, and
`0.000625`.  To avoid cumulative nonlinear-solver memory growth, each run was
split into fresh-process chunks and checkpointed after every accepted SSP
step.  All restart state gaps were exactly zero; duplicated exact-coordinate
rows agreed within `4.44e-16`.  Peak resident memory per process was about
`0.62--0.78 GB`.

The relaxed 16-mode numerical realization gave:

| time step | accepted steps | minimum shifted Gram lower bound | maximum readable-rate residual | maximum endpoint completion correction |
|---:|---:|---:|---:|---:|
| `0.0025` | 20 | `2.22e-12` | `1.06e-7` | `1.47e-5` |
| `0.00125` | 40 | `1.46e-12` | `1.45e-7` | `7.33e-6` |
| `0.000625` | 80 | `1.03e-12` | `4.63e-7` | `3.67e-6` |

The correction quantity in the last column is an endpoint displacement,
`h ||s-s_desired||`, so its near-linear decrease with `h` corresponds to a
stable completion-velocity correction.  The retained archive velocity was
never modified.  The unshifted finite Gram reached only roundoff-scale
negativity (`-6.77e-12` at the coarse step), as expected for the explicitly
inflated operational cone.

The archive-coordinate refinement is second order on the common coarse grid.
The time-RMS L2 difference is `6.53e-8` between `h` and `h/2`, and `1.63e-8`
between `h/2` and `h/4`; the factor `4.00` is the expected SSPRK2 refinement
ratio.  Against the offline cutoff-16 reference, the all-coordinate scalar RMS
is approximately `5.84e-5` on all three grids.  This number is dominated by
the fixed canonical-embedding offset already present at `t=0`; after
subtracting that initial offset, the final dynamic coordinate L2 defect is
`3.08e-5`.  The site-0 occupation RMS decreases from `3.72e-8` to `9.24e-9`
to `2.30e-9`, again showing the expected factor-four refinement.

The memorandum's strict eight-critical-mode cap was also retained as a
separate classification.  With the readable residual tier it reached
`t=0.0225` at `h=0.0025`, but the next atom required a larger local critical
subspace and stopped with normalized local residual `5.39e-7`.  The 16-mode
run therefore establishes that the carried-witness construction is
numerically executable and step-converged on this short interval, while the
specific eight-mode pilot cap fails.  No long-horizon closure or exact-CCR
representability claim follows from this result.

Canonical merged artifacts are:

- `output/local_runs/paper_v_cwrmf_relaxed16_t005_h0025_cutoff16_20260806_v3/`
- `output/local_runs/paper_v_cwrmf_relaxed16_t005_h00125_cutoff16_20260806_v1/`
- `output/local_runs/paper_v_cwrmf_relaxed16_t005_h000625_cutoff16_20260806_v1/`
- `output/local_runs/paper_v_cwrmf_relaxed16_step_refinement_t005_cutoff16_20260806_v1.json`

Five dedicated carried-witness tests cover the literal dimensions, canonical
preparation, affine completion lift, compiler-readable rate partition, and an
interior fixed-archive radial atom.  Together with the affected projective and
moment-projection tests, 27/27 focused tests pass.  The complete Paper V suite
passes 283/283 when its subprocess test inherits `PYTHONPATH=src:..`; the first
plain invocation produced one environment-only `ModuleNotFoundError` in that
pre-existing subprocess test and no scientific or implementation failure.

### Spectrum-adaptive carried-witness continuation

The artificial 8- and 16-mode critical-subspace caps were removed from the
canonical carried-witness settings.  The PSD subspace now grows from the
spectrum of the actual 62-by-62 endpoint Gram and is bounded only by its
natural 61-dimensional nontrivial support.  Every candidate remains subject
to a full eigendecomposition of the shifted Gram before acceptance; the
compressed bundle is therefore a solver representation, not a weakened
physicality test.  The readable-rate discrepancy is minimized and reported as
the secondary completion objective rather than used as an independent
physical admissibility threshold.

The first dense full-cone formulation required about 3 GB for a single atom
and terminated numerically, while a scalar cutting-plane fallback was memory
safe but inefficient.  The current implementation uses a conditioned,
spectrum-adaptive semidefinite bundle with a terminal full-Gram polish.  Fresh
subprocess chunks reset solver memory and save the complete 384-coordinate
state.  The discovered bundle rank is now included in continuation metadata
so later chunks need not rediscover the same critical subspace; this changes
neither the state nor the admissible cone.

The strong-coupling cutoff-16 rollout at `h=0.0025` is in progress toward
`t=0.5`.  The canonical prefix is stored under
`paper_v_cwrmf_conditioned_bundle_t05_h0025_cutoff16_20260807_v4`, and the
continuation from its `t=0.0525` checkpoint is stored under
`paper_v_cwrmf_objective_residual_continuation_00525_05_h0025_cutoff16_20260807_v5`.
At the latest audited state (`t=0.0875`), the shifted full-Gram lower bound was
`7.65e-12`, the spectrum-selected bundle used 41 modes, and the largest child
process resident memory observed in this continuation was approximately
0.83 GB.  These are interim execution facts, not a completed accuracy result.
The exact Hamiltonian and archive baselines remain offline scoring inputs and
will be evaluated after the autonomous trajectory is frozen.

Nine focused carried-witness tests pass after adding the uncapped spectrum
selection and checkpointed bundle-rank continuation.  Full-suite regression,
merged-trajectory scoring, and final interpretation remain pending until the
`t=0.5` rollout completes.

A segment-stitching scorer was added in
`apcm_carried_witness_stitch.py`.  It verifies grid continuity and duplicate
checkpoint states, preserves the carried trajectory, and constructs a matched
raw archive-EOM baseline from the same initial 31 coordinates; the cutoff-16
Hamiltonian contractions remain offline reference data.  A `t=0.0525` smoke
stitch reports dynamic all-coordinate scalar RMS errors of `3.40e-6` for the
carried-witness trajectory and `2.26e-3` for the raw archive EOM.  This short
prefix confirms that the scoring path distinguishes the new route from the
archive backbone, but it is not a substitute for the pending `t=0.5` result.

### Completed uncapped carried-witness result through t=0.5

The spectrum-adaptive strong-coupling rollout completed 200 SSPRK2 steps at
`h=0.0025` through `t=0.5`.  The online retained velocity was the archive
matrix EOM plus the explicit K/P/D correction to the electron--phonon
correlation rate, evaluated from the carried higher moments; the finite-Gram
guard did not modify this retained velocity.  Thus the result is an
archive-backed higher-moment closure with a carried positive completion, not
the raw 31-coordinate archive EOM with only a positivity controller.

The active spectral bundle reached 57 of the 61 nontrivial directions (mean
50.15 over the `t=0.1--0.5` continuation), demonstrating directly that the
earlier fixed 8- and 16-mode caps were artificial obstructions.  The minimum
shifted 62-row Gram lower bound over the stitched trajectory was `1.41e-12`;
the retained joint-Gram minimum was `-1.32e-15`, consistent with roundoff.
The maximum completion correction was `4.90e-3`, the maximum normalized
readable-rate residual was `1.84e-3`, and the minimum radial velocity margin
was `0.9168`.  The matched raw archive trajectory reached joint-Gram minimum
`-6.28e-2` over the same interval.

After subtracting the fixed initial canonical-embedding offset, the cutoff-16
exact-reference comparison gave:

| metric | carried K/P/D witness | matched raw archive EOM |
|---|---:|---:|
| all-coordinate scalar RMS | `3.19e-5` | `2.13e-2` |
| C-block scalar RMS | `8.14e-6` | `3.04e-2` |
| site-0 occupation RMS | `2.06e-7` | `3.53e-4` |
| total-energy RMS | `1.10e-7` | `3.81e-4` |

Exact contractions were used only after the autonomous rollout.  The 51
scored chunk artifacts accumulated 43,692 seconds of autonomous solver time
(about 12.1 hours) and a maximum reported child-process resident size of
3.27 GB.  The canonical stitched evidence is in
`output/local_runs/paper_v_cwrmf_spectrum_adaptive_t05_h0025_cutoff16_20260807_v1/`,
including the trajectory, summary, and exact/raw/carried observable plot.
Ten focused carried-witness tests and the complete 288-test Paper V suite pass
after the final archive-versus-K/P/D velocity audit.

### Balanced carried-witness speed--accuracy experiment

A named `balanced` numerical profile was added without changing the strict
defaults.  It retains the authoritative eigendecomposition of the literal
62-row endpoint Gram, but declares an unshifted numerical floor of `-1e-8`,
uses `1e-8` conic tolerances, and admits only spectrum-resolved modes that
approach that floor.  A full-cone solve is available only as a fallback when
the adaptive bundle fails.

Short matched pilots through `t=0.05` established that `h=0.005` and `h=0.01`
were both physical.  The `h=0.01` pilot had dynamic all-coordinate scalar RMS
`3.36e-6`, compared with `3.24e-6` for the strict `h=0.0025` reference on the
same interval.  The coarse balanced trajectory then advanced through `t=0.4`.
Its next SSPRK2 atom at `t=0.41` returned a numerical error in both the adaptive
bundle and literal full-cone formulations.  The last accepted state remained
inside the declared cone, so this was a time-step/solver-resolution failure,
not a physicality failure.  Restarting that exact checkpoint with `h=0.005`
completed the trajectory through `t=0.5`.

The successful mixed-step trajectory (`h=0.01` through `t=0.4`, then
`h=0.005`) gave:

| metric | balanced mixed step | strict `h=0.0025` | matched raw archive EOM |
|---|---:|---:|---:|
| dynamic all-coordinate scalar RMS | `3.56e-5` | `3.19e-5` | `2.37e-2` |
| dynamic C-block scalar RMS | `9.33e-6` | `8.14e-6` | `3.37e-2` |
| dynamic site-0 occupation RMS | `3.26e-6` | `2.06e-7` | `4.48e-4` |
| dynamic total-energy RMS | `2.02e-6` | `1.10e-7` | `4.89e-4` |

The minimum unshifted 62-row Gram eigenvalue was `-9.41e-9`, within the
declared `-1e-8` numerical floor; the retained joint-Gram minimum was
`-2.20e-15`.  The successful autonomous segments accumulated 17,561 seconds
(4.88 hours), a `2.49x` speedup over the strict trajectory's 43,692 seconds.
The coarse phase eventually required all 61 critical modes, so the remaining
cost is structural rather than solely a consequence of excessive solver
precision.  The practical operating rule supported by this experiment is a
balanced cone profile with local step halving when a coarse endpoint becomes
numerically unresolved.

The stitched trajectory, summary, matched raw archive baseline, and observable
plot are stored under
`output/local_runs/paper_v_cwrmf_balanced_mixed_t05_cutoff16_20260807_v1/`.

### Results-progression report update

The stored strict, balanced, raw archive, and exact trajectories were added as
the new first page of
`output/pdf/paper_v_results_progression_20260804.pdf`.  The raw line is
adaptive DOP853 propagation of the 31 archive coordinates with the archive
EOM alone.  It remains amplitude-finite through
`t=0.5` (maximum absolute coordinate `1.2248`) but first has a negative sampled
retained joint-Gram eigenvalue at `t=0.1625` and reaches `-0.06284`.  The page
therefore reports both observable error and physical representability rather
than using amplitude divergence as the only failure criterion.
This line is distinct from the earlier 60-coordinate APCM no-cone ablation;
an otherwise matched no-cone ablation of the newer 384-coordinate
carried-witness model has not been executed.

The figure and its machine-readable metrics are generated by the existing
results-progression plotting workflow with `--only-cwrmf-balanced`, at
`output/plots/paper_v_results_progression_20260804/cwrmf_balanced_t05_observables.pdf`
and `cwrmf_balanced_t05_metrics.json`.  No trajectory or exact reference was
recomputed for this reporting update.  The rebuilt progression report has 12
pages; its new first page was rendered and checked for clipping and layout
errors.

### Matched 384-coordinate no-Gram-guard ablation

The current carried higher-moment model now has an explicit named no-guard
mode.  It leaves the 44 retained archive-plus-entrance velocity unchanged,
including the carried K/P/D contribution to `dot C`, and propagates all 231
commutator-readable completion rates directly.  The remaining 109 unresolved
completion rates retain the declared minimum-motion value zero.  The mode
bypasses the 62-row PSD optimization and endpoint rejection; strict and
balanced defaults are unchanged.

Starting from exactly row zero of the completed strict 384-coordinate
trajectory, the matched strong-coupling cutoff-16 run reached `t=0.5` with
`h=0.0025` in 18.93 seconds.  A half-step run at `h=0.00125` completed in
37.10 seconds.  Their retained-coordinate RMS difference on the common grid
was `7.76e-8`, and their full-state RMS difference was `9.93e-8`, establishing
the expected numerical refinement at the scale relevant to this comparison.

Against the offline cutoff-16 reference, the `h=0.0025` no-guard result gave:

| metric | no Gram guard | strict Gram guard |
|---|---:|---:|
| dynamic all-coordinate scalar RMS | `3.19169e-5` | `3.19153e-5` |
| dynamic C-block scalar RMS | `8.13262e-6` | `8.14148e-6` |
| dynamic site-0 occupation RMS | `2.07468e-7` | `2.05946e-7` |
| dynamic total-energy RMS | `1.06402e-7` | `1.10151e-7` |
| minimum retained joint-Gram eigenvalue | `-1.03e-15` | `-1.32e-15` |
| minimum extended 62-row Gram eigenvalue | `-1.27089e-2` | `-7.58e-12` |
| autonomous runtime | `18.93 s` | `43,692 s` |

The no-guard and strict retained trajectories differ by only `6.57e-7`
coordinate RMS.  Consequently, the expensive cone solve is not responsible
for the observed short-time retained accuracy.  Its demonstrated role is to
preserve representability of the higher-order completion: without it, the
62-row Gram is below `-1e-8` at the first accepted step and decreases to
`-1.27e-2` by `t=0.5`, although the retained joint Gram remains positive to
roundoff.  This is a strong speed/structure result but not evidence that the
indefinite hidden trajectory is a valid long-horizon physical closure.

Artifacts:

- `output/local_runs/paper_v_cwrmf_no_gram_guard_t05_h0025_cutoff16_20260807_v1/`;
- `output/local_runs/paper_v_cwrmf_no_gram_guard_t05_h00125_cutoff16_20260807_v1/`.

The no-guard comparison is included on the new first page of
`output/pdf/paper_v_results_progression_20260804.pdf`; the plotting source and
machine-readable metric file were updated rather than creating a standalone
reader-facing plot.

### Full-pulse no-guard failure and negative-mode coupling audit

The matched 384-coordinate no-Gram-guard experiment was extended through the
complete driven interval, `t=4`, with `h=0.0025`, the same strong-coupling
cutoff-16 Hamiltonian, and the stored exact-correlated initial completion.  It
finished in 153.53 seconds, but the short-time agreement did not persist.
Against the offline exact reference, its dynamic all-coordinate RMS error was
`0.48049`, its C-block RMS error was `0.47234`, its site-0 occupation RMS error
was `0.04864`, and its interaction-energy RMS error was `0.01388`.  The matched
raw 31-coordinate archive EOM gave `0.11090`, `0.07660`, `0.02961`, and
`0.02759`, respectively.  Thus the unconstrained carried moments improved the
reported interaction-energy error but were less accurate overall and in C and
occupation over the full pulse.

The no-guard trajectory reached minimum eigenvalues `-17.6151` for the retained
joint Gram and `-26.4455` for the extended 62-row Gram.  An offline mode audit
then evaluated each negative extended-Gram eigenvector, the minimum-coordinate
linearized repair that would arrest that mode, and its induced effect on the
retained velocity.  Because the completion enters C through the entrance
channels rather than as an instantaneous direct C correction, the audit also
propagated the induced entrance-rate change for a short
`0.05 t_hop^{-1}` interval before reevaluating the C velocity.

The resulting failure sequence was:

| event | first time |
|---|---:|
| extended Gram below `-1e-8` | `0.0025` |
| predicted retained-velocity effect above 1% | `0.10` |
| predicted retained-velocity effect above 10% | `0.50` |
| two-stage C-rate effect above 1% | `0.55` |
| instantaneous dynamic coordinate RMS above `1e-3` | `1.0825` |
| two-stage C-rate effect above 10% | `1.10` |
| retained joint Gram below `-1e-8` | `1.355` |
| instantaneous dynamic coordinate RMS above `1e-2` | `1.48` |
| instantaneous dynamic coordinate RMS above `1e-1` | `2.11` |

The largest predicted retained-velocity effect was `7.3859`, the largest
two-stage C-rate effect was `0.7074`, and the largest instantaneous dynamic
coordinate RMS was `1.6394`.  The extended-cone violation is therefore not an
immediate observable failure, but it is also not dynamically inert: its modes
eventually feed through the entrance-channel equations into C and the retained
blocks before the retained joint Gram itself becomes indefinite.  This supports
an event-triggered or active-mode guard, using the online mode-coupling measure,
rather than either an always-on full 62-row optimization or a completely
unconstrained long-horizon rollout.

Artifacts:

- `output/local_runs/paper_v_cwrmf_no_gram_guard_t4_h0025_cutoff16_20260808_v1/`;
- `output/local_runs/paper_v_cwrmf_no_gram_guard_t4_scored_cutoff16_20260808_v1/`;
- `output/local_runs/paper_v_cwrmf_no_gram_guard_mode_audit_t4_20260808_v2/`.

The combined observable, physicality, instantaneous-error, and mode-coupling
plot was prepended to `output/pdf/paper_v_results_progression_20260804.pdf`.

### Initial-condition sensitivity figures

The completed sensitivity artifacts were converted into two reader-facing
figures without rerunning either trajectory.  The first compares the lifted
Frobenius amplification of the same two physical perturbations under exact
cutoff-16 Hamiltonian propagation and the representability-corrected
31-coordinate EOM through `t=100`.  The second shows the cumulative and
trailing-250-unit finite-time Lyapunov estimates from the corrected post-pulse
Benettin calculation through `t=1000`.

The plots and their hashed metric record are:

- `output/plots/paper_v_results_progression_20260804/initial_condition_matched_amplification.pdf`;
- `output/plots/paper_v_results_progression_20260804/initial_condition_lyapunov_convergence.pdf`;
- `output/plots/paper_v_results_progression_20260804/initial_condition_sensitivity_metrics.json`.

They were prepended as the newest page of
`output/pdf/paper_v_results_progression_20260804.pdf`.  This page keeps the two
claims separate: the direct matched test attributes the excess separation to
the corrected moment vector field rather than the exact dynamics, while the
rescaled long-horizon test shows that the positive rate persists rather than
being only an early transient.

### Matched uncorrected-EOM initial-condition sensitivity

The same two exact-state-derived nearby initial conditions used in the matched
controller audit were propagated with the raw 31-coordinate archive EOM.  The
base and shadows therefore begin from the same contracted physical states as
the exact and corrected comparisons.  No perturbation rescaling or exact
trajectory information enters the raw rollout.  Fixed-step RK4 with `h=0.02`
reached `t=100` in 12.24 seconds.

The electronic-drive perturbation had final and maximum lifted-Frobenius
amplifications `43862.65` and `47748.31`; the relative-phonon-position
perturbation had final and maximum amplifications `25497.25` and `33395.73`.
The matched exact values were `0.3105` and `0.4649` at the endpoint, with
maxima `1.3848` and `1.4899`.

This strong raw separation occurs after loss of representability.  On the
stored `0.1` grid, the correlation-trace residual exceeds `1e-8` at the first
nonzero sample, and the joint Gram is negative at `t=0.2`, consistent with the
independently refined crossing `t=0.1607116782`.  Through `t=100`, the base
trajectory reaches a maximum correlation-trace residual `0.22856` and minimum
joint-Gram eigenvalue `-4.44934`.  The result therefore establishes sensitivity
of the formal uncorrected mathematical continuation, not physical chaos of a
representable electron--phonon trajectory.

Artifacts:

- `output/local_runs/paper_v_matched_exact_raw_sensitivity_t100_dt002_20260812_v1/`;
- `output/plots/paper_v_results_progression_20260804/initial_condition_raw_matched_amplification.pdf`;
- `output/plots/paper_v_results_progression_20260804/initial_condition_raw_physicality.pdf`.

The raw-EOM sensitivity page was prepended to the results-progression PDF ahead
of the corrected-EOM sensitivity page.

### Observable-resolved raw-EOM sensitivity

The matched raw and exact initial-condition trajectories were also contracted
into site occupations and the established electronic, phononic,
electron--phonon, and total-internal-energy components.  Four new pages at the
front of the progression report show, for each perturbation, first the actual
base and perturbed exact/raw trajectories and then
`|O_perturbed(t) - O_base(t)|`.  The actual trajectories expose the direct
observable disagreement between the raw EOM and exact Hamiltonian, while the
difference pages resolve the initial-condition sensitivity that is hidden by
the common observable scale.

For the electronic perturbation, the maximum raw differences were `0.10939`
in each site occupation, `0.17481` in electronic energy, `0.09040` in phonon
energy, and `0.15577` in electron--phonon energy.  For the phonon perturbation,
they were `0.07619`, `0.13647`, `0.07842`, and `0.17678`, respectively.  All
matched exact observable differences remained below `8.3e-6`.  The total
internal-energy difference stayed near its small initial value in both raw
pairs, showing that energy conservation does not prevent sensitivity in the
energy partition or occupations.

Plots:

- `output/plots/paper_v_results_progression_20260804/initial_condition_raw_electronic-drive_observables.pdf`;
- `output/plots/paper_v_results_progression_20260804/initial_condition_raw_electronic-drive_observable_separation.pdf`;
- `output/plots/paper_v_results_progression_20260804/initial_condition_raw_relative-phonon-position_observables.pdf`;
- `output/plots/paper_v_results_progression_20260804/initial_condition_raw_relative-phonon-position_observable_separation.pdf`.

## 2026-08-18: event-triggered carried-witness Gram guard

### Implementation

The event-triggered guard supported by the full-pulse no-guard mode audit is
now a declared CWRMF guard policy in
`paper_5/src/paper5/stability/apcm_carried_witness.py`.  `CWRMFSettings`
gains `guard_policy` (`always` or `event_triggered`) and three event
thresholds: `event_gram_floor=1e-8`, `event_gram_hard_floor=1e-3`, and
`event_coupling_threshold=1e-2` with `event_coupling_horizon=0.05`.  Under
the event policy each radial atom first evaluates the unconstrained
higher-moment predictor and the full eigendecomposition of its endpoint
unshifted 62-row Gram.  A minimum eigenvalue above `-event_gram_floor`
accepts the atom directly.  Between the floor and the hard floor, the online
negative-mode coupling measure decides: the atom is accepted while every
negative mode's predicted retained-velocity and two-stage C-rate relative
effects stay at or below the coupling threshold.  Below the hard floor, or
above the coupling threshold, the existing guarded conic correction runs for
that atom.  The measure is the same function the offline no-guard mode audit
uses; it was moved into the model module and re-imported by
`apcm_carried_witness_mode_audit.py` so the online trigger and the offline
audit cannot drift apart.  The SSPRK convex-node rejection is policy-aware:
the unified Gram is affine in the state, so the convex node cannot fall below
the weaker of the two accepted endpoint contracts, and the event policy
checks its declared hard floor there instead of the always-on shifted cone.
Strict and balanced defaults, the always-on guard, and the named no-guard
mode are unchanged.  The driver gains `--event-triggered-guard`, mutually
exclusive with `--no-gram-guard`, with model classification
`carried_witness_event_triggered_guard_flow`.

Seven focused tests cover policy validation, the fast path, tolerated
negativity, trigger routing into the guarded correction, shared-measure
identity with the audit module, an event-policy smoke trajectory, and the
driver flag exclusivity.  The carried-witness file passes 20/20 and the
complete Paper V suite passes 299/299.

### Strict-numerics pilot and the first trigger firing

The strong-coupling cutoff-16 event-guard pilot at `h=0.0025` completed
`t=0.05` in 6.43 seconds of autonomous time with dynamic all-coordinate
scalar RMS `3.236e-6`, matching the strict always-on reference (`3.24e-6`)
on the same interval.  The first atom used the fast path; every later atom
was accepted under tolerated negativity with the coupling measure growing
from `2.31e-5` to `6.67e-3` and the unshifted minimum reaching `-9.17e-5`,
consistent with the offline audit's predicted crossing near `t=0.1`.
Artifact:
`output/local_runs/paper_v_cwrmf_event_guard_t005_h0025_cutoff16_20260818_v1/`.

The continuation toward `t=0.5` under strict numerics fired the first
trigger at the second atom of the `t=0.0825` step, from unshifted minimum
near `-2.26e-4`.  The guarded bundle solve returned `AlmostSolved` at rank
61 with independent residual `2.622e-9`, which the strict acceptance
(`10 * solver_tolerance = 1e-9`) rejects, and the terminal full-cone polish
declared `PrimalInfeasible`; the run stopped at `t=0.08` after 933.6
seconds.  This is a numerical-acceptance stop of the strict profile at a
triggered repair, not a failure of the trigger logic: the balanced profile's
declared acceptance (`1e-7`) admits exactly this residual.  Artifact:
`output/local_runs/paper_v_cwrmf_event_guard_t05_h0025_cutoff16_20260818_v1/`
(truncated trajectory, success false).  A balanced-profile event-guard
rerun of the same interval is the immediate next experiment.

### Balanced stop, declared retraction, and the bounded-depth trigger

The balanced-profile event-guard rerun passed the first triggered conic
repair near `t=0.0825` (maximum readable-rate residual `6.34e-2`) and
continued under tolerated negativity, but the coupling measure stayed below
`1e-2` while the endpoint minimum drifted from `-2.5e-4` to `-5.9e-4`.  The
next trigger, at the second atom of the `t=0.2025` step, failed with
`NumericalError` in both the bundle and full-cone solves; the run stopped at
`t=0.2` after 2,184 seconds.  Artifact:
`output/local_runs/paper_v_cwrmf_event_guard_balanced_t05_h0025_cutoff16_20260818_v1/`.
The conic repair is therefore depth-limited: it succeeded from `-2.3e-4`
and failed from `-5.9e-4`.

Two responses are implemented.  First, a declared minimum-norm completion
retraction (`_event_completion_retraction`) now runs before the conic
fallback on every triggered atom.  Each Gauss--Newton iteration solves a
Lawson--Hanson least-distance program requiring every deficient mode of the
endpoint 62-row Gram to reach `+event_gram_floor` while no working mode
below that band sinks under `-event_gram_floor`; genuine boundary modes
with vanishing gradients keep feasible do-not-sink rows.  Equality-only
variants were rejected experimentally: the working-set equality system is
inconsistent at the many-zero-mode preparation boundary, and overshoot
factors above two diverge.  The retraction converges for shallow events
(minimum `-3.3e-6` restored to `-8.2e-9` in 30 iterations, 0.5 s, no conic
backend) but stalls from `-4e-5` and deeper, so it is a fast path for
hysteresis-scale retriggers, not a replacement for the conic repair.

Second, the event hard floor is now the depth bound for the conic problem:
`--event-gram-hard-floor` overrides the declared `1e-3` default, and the
next pilot uses `2e-4` so repairs always fire at a depth the balanced conic
solve has already handled.  Focused carried-witness tests cover the
retraction (22/22) and the full Paper V suite passes 301/301.

The local rerun with the bounded depth was stopped by the user (thermal
concern) before its first trigger; its checkpoint prefix is under
`output/local_runs/paper_v_cwrmf_event_guard_balanced_hardfloor2em4_t05_h0025_cutoff16_20260818_v1/`.

### First Paper V CHTC lane

`chtc/paper_v_cwrmf_event_guard_20260818/` is the first CHTC lane for
Paper V: apptainer wrapper and inner runner mirroring the Paper II
exchange-hook lane, a single-job vanilla submit file
(`submit_cwrmf_event_guard_t05_v1.sub`: 1 CPU, 8GB memory, 20GB disk,
12 h MaxRuntime, batch
`holstein-paper-v-cwrmf-event-guard-20260818-v1`), and a README with the
exact upload/extract/submit commands.  The inner runner bootstraps a
scratch venv with pinned numpy/scipy/cvxpy/clarabel only when the
time-dynamics image lacks the conic stack.  The lifecycle validator
passes on the final submit file.  The upload bundle
`/tmp/paper_v_cwrmf_event_guard_20260818.tgz` (405,467 bytes, SHA-256
`a1edccf60392c3468ad9675d644593c6a90b47f3ac4209a4fa2e249eee3ae8d8`)
contains `paper_5/src`, `paper_5/pyproject.toml`, and the lane.  Nothing
has been uploaded or submitted; login requires the user's Duo approval.

## 2026-09-14: overnight loop, parity anchors after the machine loss

Every `output/local_runs/paper_v_*` artifact cited above was lost with the
previous machine (2026-08-31).  The plan of record is
`agent_guidance/paper-v/paper-v-state-and-plan-20260914.md`; this block and
the ones that follow are its execution log on branch
`claude/paper-v-routes-20260914`.

### No-guard CWRMF anchor, t=0 to 0.5

Rerun of the matched 384-coordinate no-Gram-guard mode from the canonical
cutoff-16 correlated preparation (prepared in-process), strong coupling,
`h=0.0025`, 200 SSPRK2 steps, exact reference regenerated in-process.
Artifact: `output/local_runs/paper_v_cwrmf_no_gram_guard_t05_h0025_cutoff16_20260914_v1/`.

| quantity | 2026-08-07 record | 2026-09-14 rerun |
|---|---:|---:|
| dynamic all-coordinate scalar RMS against the exact cutoff-16 reference, t=0 to 0.5 | `3.19169e-5` | `3.19169e-5` |
| dynamic C-block scalar RMS | `8.13262e-6` | `8.13262e-6` |
| dynamic site-0 occupation RMS | `2.07468e-7` | `2.07468e-7` |
| minimum extended 62-row Gram eigenvalue (carried, unshifted) | `-1.27089e-2` | `-1.27089e-2` |
| autonomous wall time | `18.93 s` | `9.40 s` |
| peak resident | not recorded | `701 MB` |

The rerun reproduces the recorded accuracy and physicality numbers to every
printed digit on the replacement machine; the wall time halves.  This is the
parity anchor for every no-guard-path acceleration.  The strict always-on
prefix anchor over `t=0` to `0.05` is running separately.

### Strict always-on CWRMF anchor, t=0 to 0.05

Rerun of the strict spectrum-adaptive profile (always-on guard, adaptive
bundle tier, `enable_full_cone_fallback=False`) from the same preparation,
`h=0.0025`, 20 SSPRK2 steps.  Artifact:
`output/local_runs/paper_v_cwrmf_strict_t005_h0025_cutoff16_20260914_v1/`.

| quantity | 2026-08-18 record (strict always-on, t=0 to 0.05) | 2026-09-14 rerun |
|---|---:|---:|
| dynamic all-coordinate scalar RMS against the exact cutoff-16 reference | `3.24e-6` | `3.23592e-6` |
| minimum carried unshifted 62-row Gram eigenvalue | not printed | `-3.89e-12` |
| minimum carried shifted lower bound | not printed | `5.11e-12` |
| maximum completion correction norm | not printed | `1.0989e-4` |
| maximum active critical modes | not printed | 49 |
| autonomous wall time | `43,692 s` for 200 steps, i.e. about `4,370 s` per 20 steps | `1,471 s` |
| peak resident | `3.27 GB` (200-step run) | `3.17 GB` |

The strict prefix reproduces the recorded accuracy figure and is the parity
anchor for every guarded-path acceleration: a change on the guarded path is
kept only if it reproduces `autonomous_trajectory.npz` of this run bitwise,
or to a stated tolerance with the difference reported.  Per-step cost on this
machine is about 74 s under the strict always-on guard against about 0.047 s
without it.

## 2026-09-14: overnight loop, round 1 (three implementers, each independently reviewed)

Full reports with line references:
`agent_guidance/paper-v/overnight/{cwrmf-cone-speedup,cone-correction-speedup,projective-event-guard}-20260914.md`
and the matching `review-*` files.  All three reviews passed; every kept
change is a named option or a bitwise-neutral rewrite, and the strict and
balanced defaults are unchanged.  Numbers below are from this machine.

### Carried-witness guard: where the time goes

With per-tier accumulators below the radial atom (`--tier-timing`, off by
default), the strict always-on guard over `t=0` to `0.01` at `h=0.0025`
(4 SSPRK2 steps, 8 radial atoms, strong coupling, cutoff 16) spends
`142.350 s` of a `145.172 s` rollout in the Clarabel solve (36 calls,
98.1 percent), `2.345 s` in coefficient assembly and svec packing, `0.049 s`
in the 62-row eigendecomposition, and `0.149 s` in the unguarded velocity
evaluation; the completion retraction ran zero times.  Artifact:
`output/local_runs/paper_v_cwrmf_instr_strict_t001_h0025_20260914_v1/`.

Consequence: the plan's Python-level accelerations cannot make the strict
guarded atom fast.  Four were landed anyway because they are bitwise-neutral
and pay on the unguarded atom, which is the event-triggered guard's steady
state: vectorized `_clarabel_svec_upper` (same packed order and sqrt(2)
factor), the step-scaled `(340, 62, 62)` completion coefficient tensor cached
once per rollout instead of rebuilt per solver-tier entry, the
time-independent commutator structure precompiled with the same accumulation
order, and the center-cross reads hoisted.  Gate: every array of
`autonomous_trajectory.npz` and `trajectory.npz` equal under `np.array_equal`
to the pre-change run for each change separately, on both the no-guard
`t=0` to `0.05` and the strict `t=0` to `0.01` anchors.

| quantity | before | after |
|---|---:|---:|
| no-guard autonomous solver wall time, `t=0` to `0.05`, `h=0.0025` | `1.2864 s` | `0.6314 s` |
| no-guard unguarded velocity evaluation tier, same run | `0.7386 s` | `0.1139 s` |
| strict autonomous solver wall time, `t=0` to `0.01`, `h=0.0025` | `143.511 s` | `139.883 s` |
| all-coordinate scalar RMS against the exact reference, same runs | `5.836875e-5` / `5.827262e-5` | identical |
| ram_guard peak, strict `t=0.01` | `1980 MB` | `1609 MB` |

Two named options were added off by default: `batched_mode_gradients`
(`--batched-mode-gradients`; one contraction over all working modes in the
completion retraction and the coupling measure, 31.5x at 61 modes, but not
bitwise: maximum departure `2.220e-16` absolute, one ulp of the largest
gradient entry) and bounded local step halving in `integrate_cwrmf_ssprk2`
(`--maximum-step-subdivisions`, default 0 reproduces the current terminate-
on-failure behaviour; a forced-failure test shows depth 1 completing the step
with one failed full step and two accepted half steps).  Carried-witness
tests 30/30 (22 before).

### 31-coordinate structured cone correction: 24 percent, and where the rest is

Bitwise-neutral hoists on the default path (the 31-fold joint-Gram basis
built from the already-available matrix state and a module-constant velocity
basis; the identity- and Frobenius-metric Cholesky factors as module
constants; the assembled corrected barrier matrices cached on the exact
reduced point and shared by the constraint, the Jacobian and the exit test;
slice-copy assembly in place of `np.block`) reduce the energy-neutral
joint-Gram barrier from `145.62 s` to `110.74 s` over `t=0` to `0.2` at
strong coupling (16,700 right-hand-side evaluations, 318,122 half-space
projections), with the sampled states, times, eigenvalue series and energies
byte-identical.  The ten-coordinate energy-neutral matrix barrier reproduces
the working notes' pinned minimum bosonic eigenvalue `2.893142825017192e-5`
over `t=0` to `2` in `0.226 s`.  Harness:
`output/local_runs/paper_v_cone_ref_t2_20260914_v1/run_pinned_energy_neutral_barrier.py`
(the fixed-step `run_high_coupling_controller_comparison.py` cannot run the
DOP853 pinned protocol).

Three negative results.  Memoizing the correction on the exact (state,
derivative) pair never hits (0 of 749 and 0 of 218 evaluations), because
DOP853 reuses the previous step's final stage and retries rejected steps at
different stage points; kept as an off-by-default option with the zero rate
recorded.  The warm-started dual active-set minimum-norm-point backend
(`halfspace_backend='active_set'`, closed form for one half-space, SLSQP as
declared fallback) avoids 287,941 of 317,277 SLSQP calls on an identical-
operand replay of 16,646 recorded corrections (`109.91 s` to `92.55 s`,
correction norms agreeing to `3.08e-10` relative), but integrated it moves
the joint-Gram trajectory by `6.286e-9` relative to trajectory scale over
`t=0` to `0.2` and takes 636 more right-hand-side evaluations, so it is not
the default (`DEFAULT_HALFSPACE_BACKEND='slsqp'`); SLSQP's own `ftol=1e-12`
means no exact-minimum backend can match it to `1e-12`.  Fusing the
`eigvalsh`/`eigh` pair was dropped as not bitwise-neutral.  After the change
47 percent of the joint-Gram route's profiled time is in
`matrix_reference.py` (`electron_phonon_moment_derivative`, 142,552
`ndarray.trace` calls over 206 evaluations), which this round did not own.
Cone-correction tests 29/29 (22 before).

### Projective route: executable, quiet, and 24.5 times less accurate than the carried witness

The event-triggered one-shell outer guard is a named policy on the
fixed-dictionary APCM integrator (`ProjectiveOuterEventGuardSettings`,
`projected_apcm.py`), sharing `negative_mode_coupling` with the carried-
witness route through an adapter; `frontier_weight` (default 0) turns the
two-solve selector into the single strictly convex solve
`0.5 q^T M q + 0.5 w ||z||^2` with no lift equality; driver
`paper5.stability.apcm_projective_event_guard_analysis`.  Rollout over `t=0`
to `0.05` at `h=0.0025`, 15 entrance keys, cutoff 16, strong coupling:
`50.58 s`, peak `300 MB`, 65 stages evaluated, none triggered (endpoint
unshifted 24-row Gram minimum of the unconstrained predictor never below
`-1.38e-13` against the declared floor `-1e-8`), 7 stage retractions
applied; all-coordinate scalar RMS against the exact cutoff-16 reference
`7.9318e-5` (blockwise rho `5.02e-7`, B `1.60e-6`, N `2.17e-4`, A
`9.10e-6`, C `2.02e-5`) against the carried-witness strict `3.236e-6` on the
same horizon.  Artifact:
`output/local_runs/paper_v_apcm_projective_event_guard_t005_h0025_cutoff16_20260914_v1/`.

Negative results at the cutoff-16 correlated preparation, `t=0`: the
outer-aware invariant target solve now ends in `explicit_target:NumericalError`
after about `46 s` under both gate settings (the record's `500 s` then
`171 s` were measured on a first-pass `AlmostSolved` that does not reproduce
here), so `accept_first_provisional` does not discriminate; the strictly
convex selector reaches `AlmostSolved` for every frontier weight from `1e-5`
to `1e-1` but its relative duality gap at `w=1e-5` is `1.119e-6` against the
exploratory acceptance `1e-6`, so no witness is accepted; a forced trigger
(floors lowered to roundoff) costs `45` to `365 s` and about `5.3 GB` per
triggered stage.  Objective values of the strictly convex selector are within
0.35 percent of the recorded `0.01126712315` over the weights tried.
Projective tests 30/30 (20 before).  Full `paper_5` suite at the end of the
round: 326 passed.

### Bounded-depth event-triggered guard, balanced profile, first attempt (killed by the memory guard)

Plan step 5: `--numerical-profile balanced --event-triggered-guard
--event-gram-hard-floor 2e-4 --maximum-step-subdivisions 2 --tier-timing`,
`t=0` to `0.5` at `h=0.0025`, from the canonical cutoff-16 correlated
preparation, under `ram_guard --limit-mb 5000`.  The unguarded flow ran under
tolerated negativity through `t=0.1175` (step 47; unshifted minimum
`-1.54e-4` and coupling measure `5.3e-3` at `t=0.10`, bundle rank hint 60),
then the first triggered conic repair pushed resident memory past the guard
(`5288 MB > 5000 MB`) and the process tree was killed.  Checkpoint:
`output/local_runs/paper_v_cwrmf_event_guard_balanced_hardfloor2em4_t05_h0025_cutoff16_20260914_v1/checkpoint.npz`
(`t=0.1175`, 384-coordinate state, bundle rank hint 60).  The run was resumed
from that checkpoint under a `7000 MB` guard into
`..._resume0p1175_20260915_v1/`; the round-1 projective agent measured about
`5.3 GB` for one forced triggered stage of the one-shell selector, so a
triggered repair on the 62-row cone at rank 60 appears to need more than the
`3.27 GB` the record reported for the strict always-on bundle.

### Projective fixed-dictionary APCM with the event-triggered outer guard through the full pulse, t=0 to 4

Plan step 13 (second half), under the exploratory reading of D1 and D2 with
the recorded lexicographic selector (`frontier_weight 0`) and the default
event floors (`event_gram_floor 1e-8`, `event_gram_hard_floor 1e-3`,
`event_coupling_threshold 1e-2`): fixed-dictionary APCM, 15 entrance keys,
SSPRK(3,3) at `h=0.0025`, containment before every right-hand-side call,
strong coupling, cutoff 16, exact-correlated preparation, scored offline
against the exact cutoff-16 driven reference.  Artifact:
`output/local_runs/paper_v_apcm_projective_event_guard_t4_h0025_cutoff16_20260914_v1/`.

| quantity | this run, t=0 to 4 | raw 31-coordinate archive EOM (record, t=0 to 4, h=0.0025) | carried moments, no Gram guard (record) |
|---|---:|---:|---:|
| all-coordinate scalar RMS against the exact cutoff-16 reference (31 closed scalar coordinates; the fixed canonical-embedding offset is `8.1e-16`) | `0.026049` | `0.11090` | `0.48049` |
| C-block dynamic scalar RMS | `0.020175` | `0.07660` | `0.47234` |
| site-0 occupation RMS | `0.046452` | `0.02961` | `0.04864` |
| blockwise dynamic scalar RMS rho / B / N / A | `0.05919` / `0.02602` / `0.01550` / `0.01389` | not printed | not printed |
| minimum unshifted retained joint-Gram eigenvalue over the run | `2.892e-5` | `-0.5173` | `-17.6151` |
| minimum scaled positive-extension eigenvalue over the run | `-9.3e-16` | --- | `-26.4455` (extended 62-row Gram) |
| autonomous wall time / peak resident | `3,229.8 s` / `284 MB` | --- | `153.53 s` |

The outer guard evaluated 5,559 stages, classified 4,948 quiet and 611
tolerated (endpoint unshifted 24-row Gram minimum down to `-1.02e-4`, never
below the hard floor; coupling measure at most `1.07e-3`), and fired no
selector solve.  The physicality work was done entirely by the stage
retraction (the proximal log-det completion, `direct_clarabel:Solved/Solved`)
on 1,068 of the 5,559 stages, the other 4,491 being already feasible; the
per-stage wall time including that retraction sums to `2,706.8 s` of the
`3,229.8 s` (mean `0.487 s`, maximum `19.2 s` at `t=3.094`), while the event
test itself costs about `0.01 s` per stage.  The driver's static-energy RMS
(`0.036201`) is not the record's interaction-energy RMS column and is not
compared to it here.

Reading: this is the first moment-hierarchy trajectory through the full
driven interval that stays representable (retained joint Gram positive to
`2.9e-5`) and beats the raw archive EOM on the 31-coordinate and C-block
errors, at the cost of a worse site-0 occupation error.  Against the
60-coordinate positive-M4 record over the same interval (cell A, full cone
plus controller: `0.012580` at `3,399.6 s`; cell B, zero-cumulant prior plus
controller: `0.046475` at `52.9 s`; worklog:2608-2664) it sits between the
two in accuracy at cell A's cost, and the cost is the stage retraction, not
the guard.  Next: make the stage retraction cheaper (warm-started completion
is under test in round 2) and extend the same configuration to `t=20` for the
comparison against the both-cones line (`0.128607` over `t=0` to `20`,
worklog:2781-2792).

### The same projective configuration toward t=20: fails between t=5 and t=6

Same driver and settings as the `t=0` to `4` run above, `--final-time 20`.
The rollout passed `t=4` (progress line: base Gram minimum `7.613e-4`,
extension minimum `1.992e-4`, one stage retraction in the fourth unit) and
`t=5` (`7.700e-4`, `2.022e-4`, none), then stopped with
`ProjectedAPCMFailure: moment_extension_failure persisted after 8 stage
subdivisions: retracted lower moments did not admit the declared selector:
moment_extension_degeneracy: conic Newton step failed descent after full cone
interior`.  The driver writes no trajectory on this failure path, so only
`progress.jsonl` survives under
`output/local_runs/paper_v_apcm_projective_event_guard_t20_h0025_cutoff16_20260915_v1/`;
the failure time lies in `5 < t < 6`, after the drive has ended.  Two things
to change before the next attempt: the driver must save the partial
trajectory and the failing stage's lower moments on every failure path, and
the post-pulse degeneracy of the proximal log-det completion needs its own
diagnosis (the fixed dictionary's positive extension approaches a face of
the cone once the drive is off).

## 2026-09-15: overnight loop, round 2 (four implementers, each independently reviewed)

Full reports: `agent_guidance/paper-v/overnight/{cone-solve-instances,projective-gap,boson-barrier-t100,matrix-reference-speedup}-20260915.md`
and the matching `review-*` files.  Three reviews passed; the cone-solve
review failed on one material reporting error (a balanced-acceptance count
produced by a degenerate gate) and was repaired and re-measured.  Defaults
unchanged throughout.

### Carried-witness conic solve: 36 recorded instances, no faster strict-passing solver, and the untried lever

A named, off-by-default instance recorder (`--record-conic-instances`) saves
every Clarabel problem the guard solves (P, q, A, b, cone specification,
settings, tier, atom, the returned solution, and the operands to rebuild the
62-row Gram of any candidate).  The strict `t=0` to `0.01` rollout with
recording on reproduces the round-1 anchor bitwise and yields 36 instances:
33 adaptive-bundle-tier iterations and 3 critical-Schur SQP iterations; the
positive-semidefinite cone side dimension grows from 12 to 80 (twice the
spectral bundle rank) and Clarabel's own `solve_time` from `0.006 s` to
`14.462 s` at 14 to 28 interior-point iterations per call, so the cost is the
per-iteration factorization of a growing block.  26 of the 33 bundle calls
are enlargement iterations whose solution is discarded, and the strict
acceptance test passes on exactly 7 of the 36 recorded solutions, one per
bundle atom; `_bundle_rank_hint` starts at 0, so the first atom spent
`11.24 s` walking the side dimension from 12 to 52 before the atom that used
it started at 52.  Artifact:
`output/local_runs/paper_v_cwrmf_record_strict_t001_h0025_20260915_v1/`.

Offline replay over the 36 instances (`cwrmf_solver_replay_bench`; recorded
baseline `148.407 s`, machine spread `0.077 s` on a `135.7 s` repeat):

| candidate | wall over 36 instances | strict acceptance / verdict agreement | disposition |
|---|---:|---|---|
| Clarabel tolerances loosened one / two decades | `139.4 s` / `133.9 s` | 35/36, 33/36 agreement | dropped, inside the spread and loses agreement |
| `direct_solve_method='qdldl'` | `760.6 s` | --- | dropped (auto already picks faer) |
| `equilibrate_enable=False` / `static_regularization_enable=False` | `107.9 s` / `103.7 s` | move a returned completion velocity by `1.561` / `1.005` absolute; Gram minimum `-4.08e-3` / `-2.58e-3` | dropped |
| `chordal_decomposition_enable=False` | neutral, 36/36 bitwise | --- | dropped (one dense PSD block) |
| warm-started SCS at `1e-9` | `768.4 s`; `720.5 s` with a balanced-gated Clarabel polish firing on 19 of 36 | strict 6/36; substituted-inflation balanced 15/36 | dropped |
| reduced bundle (drop directions above `1e-1` ... `1e-6` of the largest base eigenvalue) | `2.4x` to `46.6x` faster | strict 0/36 at every threshold; maximum cone violation `6.1e-3` to `1.5e-2` | dropped |
| `presolve_enable=False` | `135.7 s`, 36/36 bitwise | 7/7 | kept as named profile `presolve_off`; no speedup |
| `max_threads=8` | `82.6 s` (`1.64x`) | not bitwise (6 of 36 exact; worst completion-velocity difference `2.255e-4`); balanced 29/29; fails strict on 1 of 7 | kept as named profile `max_threads_8`, off by default |

Rollout reruns of the strict `t=0` to `0.01` anchor: `presolve_off`
`135.880 s`, bitwise equal to the anchor; `max_threads_8` `92.418 s` against
`139.883 s` (`1.51x`), all-coordinate scalar RMS `5.827261740618476e-5`
against `5.827261740618364e-5`, the 31 reported archive coordinates agreeing
to `2.1e-15` and the 384 carried coordinates differing by at most `3.8e-6`.
Reading: under the strict acceptance test no candidate solves one instance
faster; the strict guard gets faster only by solving fewer instances.  The
enlargement iterations are the untried lever (round 3).  The balanced
acceptance test on strict-profile operands accepts nothing without
substituting the balanced `psd_inflation` (`1e-8` for `1e-11`); every
balanced number above uses that substitution, stated.

### Projective route: the gap is the conic stage retraction, and carrying the completion closes most of it

Attribution over `t=0` to `0.05` at `h=0.0025` (recorded path rerun:
`7.9318e-5`, 7 stage retractions, `54.0 s`; blockwise rho/B/N/A/C
`5.02e-7` / `1.60e-6` / `2.17e-4` / `9.10e-6` / `2.02e-5`): the N-block error
against the exact reference is `2.250e-8` at `t=0.0175` with no retraction
applied, steps to `2.282e-4` at `t=0.02` where one is applied, is identical
to every printed digit on every later step that applies none, and steps up
on each that does; halving the step raises the retraction count from 7 to 46
and the N block from `2.17e-4` to `4.39e-4` (all-coordinate `1.5805e-4`), the
opposite of a truncation error.  With the retraction disabled, or
containment disabled on non-triggered stages (named options), the error is
`1.9408e-6` over `t=0` to `0.015` (N `4.26e-9`) and the rollout stops after 6
of 20 steps with `moment_extension_failure`.  With the exact K/P/D entrance
correction off the error is `2.1281e-3` (C `3.17e-3`).  Prefixes of 5, 10
and 12 entrance keys are inadmissible at this preparation; 13 and 14 change
the score only through how often the retraction fires.

Carrying the previous stage's completion coordinates as the lift when they
stay positive semidefinite inside the frozen box envelope, with the proximal
log-det completion as the counted fallback (`--carried-completion`, off by
default): all-coordinate scalar RMS `3.5137e-5` at `h=0.0025` (N `8.98e-5`,
1 retraction, `42.0 s`, 36 of 66 stage lifts carried), `1.2875e-5` at
`h=0.00125` (N `1.83e-5`, 1 retraction, `14.2 s`, 113 of 120 carried),
`1.4635e-5` at `h=0.000625`; against the carried-witness strict `3.2359e-6`
the factor falls from 24.5 to 3.98.  A control that solves the declared
completion from the origin instead of warm-starting it gives `5.1100e-5`
with 4 retractions.  The carried-lift rollout toward `t=0.5` at `h=0.00125`
stops at `t=0.11375` with "extended operator moment matrix numerical face is
unstable under null-tolerance refinement" (`5.8243e-5` over the horizon
reached, 293 of 300 lifts carried).  Reading: carrying the lift removes the
retraction but not the eventual degeneracy of the per-stage completion
problem; a long-horizon projective rollout needs the completion coordinates
propagated by their own velocity and repaired conically, which is the
carried-witness construction.  Artifacts under
`output/local_runs/paper_v_apcm_projective_{ablation_*,hybrid_*,control_*}_20260915_v1/`.

Selector: with a named exploratory acceptance tolerance of `2e-6` (default
`1e-6` unchanged) the strictly convex outer target at the cutoff-16
preparation, `w=1e-5`, is accepted as `provisional_independent_kkt` (gap
`1.1194e-6`, objective `0.01127104613`, reconstructed outer Gram minimum
scaled eigenvalue `-4.2e-14`); at a forced-trigger stage's own lower moments
all three SSPRK stages return `explicit_target:NumericalError` (`46.4`,
`43.1`, `58.8 s`; peak resident `3.90 GB`), which no acceptance tolerance
admits.  A Jacobi-scaled frontier term makes it worse (gap `6.6e-4`,
`NumericalError`); a second Clarabel pass warm-started from the first gives
gap `3.48e-6` at twice the wall time.  Projective tests 44 (34 before).

### Ten-coordinate energy-neutral matrix barrier through t=79.95: containment, not accuracy

Plan step 9, new driver
`pipelines/open_dynamics/run_paper_v_energy_neutral_barrier_long_horizon.py`
over `paper5.stability.cone_long_horizon` (a DOP853 stepping loop with a
wall budget that reproduces `solve_ivp` byte for byte when unbudgeted).
Strong coupling, cutoff-16 exact ground-state contractions, DOP853 rtol
`1e-9` atol `1e-11` maximum step `0.05`, single pulse, scored against exact
driven-dimer propagation at cutoff 16.  The corrected lane was stopped by the
25-minute rule at `t=79.95` (`1,500 s`, 328,358 right-hand-side evaluations,
`4.57 ms` each); both uncorrected lanes reached `t=100` in about `4.4 s`.
Artifact: `output/local_runs/paper_v_boson_barrier_t100_20260915_v1/`.

| quantity, `t=0` to `79.95` unless stated | energy-neutral barrier | uncorrected Eq. (112) archive EOM | archive EOM without the initial-residual subtraction |
|---|---:|---:|---:|
| complete 31-coordinate l2 time-RMS distance from exact | `1.016703` | `1.549827` | `2.050020` |
| all-31 scaled RMS (adaptive-packet convention; the packet route's `t=0` to `98` line is `0.063649`) | `0.700816` | `0.713667` | `1.585195` |
| site-0 occupation / electronic / phonon / electron--phonon / total internal energy RMS (packet line: `0.013600` / `0.023320` / `0.028350` / `0.033785` / `0.000579`) | `0.330411` / `0.230908` / `0.302944` / `0.422100` / `0.043971` | --- | --- |
| minimum bosonic eigenvalue / minimum electronic density eigenvalue | `2.8931e-5` / `8.51e-3` | `-4.708e-1` / `2.30e-2` | `-2.481` / `-9.26e-2` |
| polarization spectrum `10 <= t <= 79.8`: Hellinger / electronic-band weight (exact `0` / `0.0729`) | `0.2610` / `0.1186` | `0.2813` / `0.0381` | `0.5008` / `0.3973` |

The working notes' pinned `t=0` to `20` table is reproduced in every printed
column (barrier active on exactly 221 of 401 samples, 55.11 percent).  The
step-9 question is answered: the record's electronic-band suppression
(`0.0026` against exact `0.0569`) is a property of the 31-coordinate
joint-Gram controller, not of the closure, since this ten-coordinate
controller puts 1.63 times the exact weight in the band.  Definitional
finding: the record's polarization "raw" lane is the archive EOM without
the initial-residual subtraction (Hellinger `0.63807`, band weight
`0.52250`, exact `0.05691`, reproducing the record's `0.6381` / `0.5225` /
`0.0569`).  The joint-Gram lane over `10 <= t <= 100` remains out of reach
on this machine (`16,700` evaluations for `t=0` to `0.2`).

### matrix_reference.py: bitwise-neutral assembly, 35 percent off the joint-Gram barrier

Pauli monomial trace tables built once at import replace
`trace(M @ sigma_a)` and `trace(M @ sigma_a @ sigma_b)`; the correlation
Pauli block contracts both correlation matrices in one expression; the
electronic means are cached on the exact bytes of their operand (31 of 32
recomputations per structured correction are repeats); every `np.block` is
replaced by slice assignment into a preallocated array.  Proved an identity
by `np.array_equal` (200,000 random draws per identity, now 2,000 in
`test_matrix_reference_moment_assembly.py`) and by the three gates: the
joint-Gram energy-neutral barrier over `t=0` to `0.05` (`0.92371 s` to
`0.59877 s` integration wall time, states byte-identical), the ten-coordinate
barrier over `t=0` to `2` (byte-identical, `2.893142825017192e-5`), and the
no-guard carried-witness anchor (every array equal).  Profile of the
joint-Gram route: `electron_phonon_moment_derivative` `0.714 s` of `1.559 s`
to `0.145 s` of `0.975 s`; `ndarray.trace` calls 155,008 to 448.  Impact
analysis on the main index returned HIGH for five of the eight edited symbols
(`_validate_matrix_state` 155 impacted, `boson_moment_matrix` 70,
`electron_phonon_moment_matrix` 62, `_correlation_pauli_block` 48,
`_electron_fluctuation_moment_matrix` 44); the gates answer it.  Two facts
for later readers: `_boson_moment_derivative_from_blocks` now returns
complex128 for every input (every in-repository caller already passes
complex), and `_PAULI_MEAN_CACHE` is module-global, bounded at 8, returning
read-only arrays.

### Bounded-depth event-triggered guard, balanced profile: completes t=0.5

Resumed from the `t=0.1175` checkpoint under a `7000 MB` guard, same
settings (`balanced`, `--event-triggered-guard`, `--event-gram-hard-floor
2e-4`, `--maximum-step-subdivisions 2`), `h=0.0025`, strong coupling, cutoff
16.  Artifact:
`output/local_runs/paper_v_cwrmf_event_guard_balanced_hardfloor2em4_t05_h0025_cutoff16_resume0p1175_20260915_v1/`.

| quantity | value |
|---|---:|
| horizon completed | `t=0.1175` to `0.5`, 153 SSPRK2 steps, 306 radial atoms, no step subdivision used |
| dynamic all-coordinate scalar RMS against the exact cutoff-16 reference over `t=0.1175` to `0.5` (record's strict always-on figure over `t=0` to `0.5`: `3.19e-5`) | `2.4544e-5` |
| dynamic site-0 occupation RMS / C-block scalar RMS, same horizon | `2.34e-7` / `6.83e-6` |
| minimum carried unshifted 62-row Gram eigenvalue (declared hard floor `-2e-4`) | `-1.672e-4` |
| minimum retained joint-Gram eigenvalue | `3.95e-15` |
| maximum completion correction norm / maximum readable-rate residual | `3.86e-3` / `1.34e-2` |
| autonomous wall time (this segment) / peak resident | `14,722 s` (4.09 h) / `6.05 GB` |
| tier seconds: Clarabel solve / retraction / eigendecomposition / unguarded velocity | `14,636 s` over 381 calls / `51.6 s` over 82 calls / `1.06 s` over 1,072 / `0.83 s` over 306 |
| maximum atom wall time | `381.5 s` (bundle rank 48 to 61 on triggered atoms; mean active critical modes 32.3) |

Reading: the depth bound does what the record prescribed.  The balanced
always-on attempt of 2026-08-18 stopped at `t=0.2` with `NumericalError`
and the strict one at `t=0.08`; with repairs forced to fire at `2e-4` every
triggered conic repair solved (`Solved` or `AlmostSolved` within the
balanced acceptance), the retained joint Gram stayed positive, and the
carried Gram never went below the floor.  The cost is still 99.4 percent
Clarabel: about 4.1 h for the guarded `0.38` time units against the
record's `12.1 h` for the strict always-on `0.5`, i.e. roughly `2.9x` on the
guarded part and not the order of magnitude the route needs.  Round 3's
bundle-rank seeding (26 of 33 recorded bundle solves were discarded
enlargement iterations) and the `max_threads_8` profile (`1.51x`, balanced
only) are the two levers left on this path; the memory of a triggered atom
at rank 60 (`6.05 GB` peak) bounds concurrency to two such runs on this
machine.

## 2026-09-15: overnight loop, round 3 (four implementers, each independently reviewed)

Full reports: `agent_guidance/paper-v/overnight/{bundle-rank-seed,projective-face,joint-gram-t20,outer-selector-formulation}-20260915.md`
and the matching `review-*` files.  Three reviews passed; the joint-Gram
review failed on the lane's base velocity and census (below) and its repair
is recorded separately.  Defaults unchanged; `paper_5` suite 428 passed on
the round-3 tree.  Round 3 started at 03:04 and its longest task, the two
150-minute joint-Gram runs, is why the batch took six hours; that budget was
this session's choice, not the task's need.

### Bundle-rank seeding: the round-2 premise does not survive measurement

The 26 discarded adaptive-bundle solves are not waste: they are how the
working set finds the directions in which the correction drives the 62-row
Gram negative, and those directions are not in the predictor's own low
spectrum.  Seeding the first bundle atom with the 26 lowest eigenvectors of
its shifted 62-row predictor Gram (the rank at which the recorded walk
accepted) gives a full-Gram minimum eigenvalue of `-6.5304e-9` against the
acceptance floor `1.9e-12` and does not accept; the lowest seeded rank that
accepts there is 40 at `12.841 s` against the whole walk's `11.240 s`, and
seeded ranks 46 and 52 return `NumericalError`.  Per atom over the strict
`t=0` to `0.01` rollout (starting rank, accepting rank, discarded solves):
atom 1 (6, 26, 12), atom 2 (26, 31, 5), atom 3 (31, 35, 4), atom 4 (35, 36,
1), atom 5 (36, 38, 2), atom 6 (38, 39, 1), atom 7 (39, 40, 1); the 26
discarded solves carry `86.224 s` of the tier's `145.817 s`.  The compressed
problem's own dual cannot certify the full cone: all 26 discarded solves
report compressed residuals at or below `4.9e-11` while their full-Gram
minima reach `-5.6e-8`.

What is recoverable is continuation: carrying the previous atom's accepted
working set (not its rank) makes atoms 2 to 7 accept on their first solve.
Six named seed policies (`--bundle-seed-policy`, default `monotone_hint`,
the recorded rule, bitwise on the anchor), a growth factor, and an interior
overshoot were added, all off by default.  Strict `t=0` to `0.01`, Clarabel
tier seconds against the same-session default bracket `152.448 s` / 36 calls:
`carried_accepted_basis` margin 2 `104.227 s` / 27 (`1.46x`), with
`max_threads_8` `74.899 s` / 28 (`2.04x`, the 31 reported archive
coordinates agreeing with the anchor to `1.7e-15`); `spectral_threshold 1e-4`
`130.913 s` / 14 (`1.16x`, fewer solves but each costlier);
`previous_accepted_rank_margin 3` `1.14x`; growth factor 1.5 with trigger
`1e-7` `1.08x`, and without the trigger the working set runs to rank 61,
Clarabel returns `NumericalError` and the rollout stops at `t=0.005`.  In the
balanced profile over `t=0` to `0.05` the carried working set is `3.42x`
slower (`879.775 s` against `257.439 s`), because 16 of the 21 balanced
bundle atoms already accept on their first solve.  Reading: the guarded
carried-witness atom has no lever left below a factor of about two on this
machine; it is a CHTC-scale computation.  Artifacts under
`output/local_runs/paper_v_cwrmf_seed_*_strict_t001_h0025_20260915_v1/`.

### One-shell outer selector: the formulation is not the cause

The recorded one-shell outer target program, reassembled unchanged and run
at the recorded routine's own tightened retry profile (`retry_profile=2`:
conic tolerance `1e-10`, `max_iter 500`, 20 iterative refinement passes at
`1e-14`, equilibration off), returns an accepted witness under the unchanged
`1e-6` exploratory acceptance at both operand sets: relative duality gap
`7.4634e-7` at the cutoff-16 correlated preparation (objective
`0.011267123150627062`, reconstructed outer Gram minimum scaled eigenvalue
`-2.1e-14`, `75.86 s`) and `7.5469e-7` at the forced-trigger SSPRK(3,3)
stage-one trial point (`74.91 s`), against the recorded first pass's
`NumericalError` at `3.5740e-4` and `6.7813e-4`.  The recorded ladder never
reaches that profile because its retry trigger set is `{AlmostSolved,
InsufficientProgress, MaxIterations}` (`apcm_projective_guard.py:1493`) and
the first pass ends in `NumericalError`.  Round 2's frontier weight `1e-5`
was already at the tightened profile (a positive weight makes the first pass
`AlmostSolved`) and is what pushed the gap to `1.1194e-6`; at weight 0 the
same profile gives `7.4634e-7`.  The invariant target image quadratic
`R^T M R` is exactly the orthogonal projector onto the row space of `R` (15
unit eigenvalues, 223 zeros of 238); whitening the image improves the first
pass by a factor 109 on the gap and nothing at the tightened profile.  All
six requested alternatives (equilibration, PSD inflation at `1e-11` to
`1e-8`, the dual, a first-order pass with polish, the nullspace
parameterization, homogeneous frontier scaling) were measured on both operand
sets: at the first pass none is admitted; at the tightened profile the
formulation is immaterial (10 percent spread) except that Ruiz equilibration
is the one row rejected at the stage operands.  At the post-pulse failing
stage saved by the projective-face agent (`t=5.086270`) the outer target is
`PrimalInfeasible` at both profiles with minimum scaled eigenvalue `-0.2554`:
those retracted lower moments do not extend to the one-shell outer Gram
inside the box, and no formulation, profile or tolerance touches that.  New
module `apcm_outer_selector_formulations.py` with
`solve_outer_target_with_tightened_retry` (drop-in `GuardConicResult` shape;
first pass then the tightened profile when not accepted) and 14 tests; no
existing file edited.  The accepted witness sits on the box face (outer
envelope ratio `0.99946`, current `0.99274`, margin `7.3e-3` against the
guard's `1 - 1e-8` envelope fallback).  Artifacts under
`output/local_runs/paper_v_outer_selector_formulations_{prep,stage,t5086}_20260915_v1/`.

### Projective route: failure paths saved, the face refusal answered, the post-pulse wall unmoved

The driver now saves the partial trajectory, the exact reference over the
horizon reached, the accuracy metrics, the last accepted state and the
failing stage's state, lower moments and frontier moments on every failure
path (`stage_failure.npz`, a failure block in `summary.json`, nonzero exit),
with a forced-failure test, and gains `--initial-state-file` /
`--initial-time`.  Its first use, a continuation of the default policy from
the stored `t=4` state, reproduced the recorded post-pulse failure at
`t=5.085` in `250.5 s` (434 steps, 122 stage retractions, dynamic
all-coordinate scalar RMS `4.3845e-2` over `t=4` to `5.085`) and made it
diagnosable.

The stage retraction fires because the declared proximal log-det completion
returns `success=False` at a lift that is inside its cone (an uncertified
completion maps to minus infinity, `apcm_moment_projection.py:1348-1351`),
and which declared setting certifies is a property of the stage: at the
failing `t=5.0863` stage the declared settings are uncertified at KKT
backward error `6.39e-7` after the 40-iteration Newton cap, while
`logdet_shift` `1e-9`, `1e-8`, `1e-7` certify in 4 to 5 iterations and
`logdet_weight` `1e-1`, `1e-2` certify too.  A declared completion retry
ladder (`completion_retry_shifts` / `completion_retry_weights`, empty by
default; shifts `1e-8`, `1e-7` and weight `1e-1` tried) removes all seven
stage retractions of the recorded `t=0` to `0.05` rollout and takes its
all-coordinate scalar RMS from `7.9318e-5` to `1.2277e-5`, the N block from
`2.1728e-4` to `3.1027e-8`, at `112.5 s` against `54.4 s`: round 2's
attributed gap closed at its cause.

The "numerical face is unstable under null-tolerance refinement" refusal
that stopped the carried-completion rollout at `t=0.11375` is a collision of
two declared constants: the carried lift is accepted at
`carried_completion_margin_multiple x conic_tolerance = 1e-8` while the face
is refused for an eigenvalue in `[5e-9, 2e-8]`; the refused eigenvalue was
`1.9998e-8`.  Two named face policies, `continuation` and `inflated`
(default `refuse`), both carry the rollout to `t=0.5` at `h=0.00125` with
byte-identical trajectories (`9.5474e-4` over `t=0` to `0.5`, `1.2875e-5`
over `t=0` to `0.05`); raising the carried-completion margin multiple to 30
does the same with no face policy (`1.0437e-3` over `t=0` to `0.5`, 9 stage
retractions, 1,175 of 1,204 lifts carried, `56.9 s`).

Past the pulse nothing reaches `t=6` inside a 40-minute budget: from the
stored `t=4` state the committed path stops at `t=5.085`, PSD inflation of
the completion cone (`logdet_shift 1e-8`) at `5.1025`, carried completion at
`5.0975`, a larger proximal weight (`1e-2`) at `4.83`; the best horizon is
`t=5.235` with the carried lift plus face continuation (494 steps, 269 stage
retractions, dynamic RMS `4.9241e-2` over `t=4` to `5.235`), stopped by the
budget rather than a refusal, and the retry ladder plus face continuation
reaches `5.12`.  Past `t=5` the ladder certified 115 refusals and was
exhausted on 149.  Reading: after the drive ends the fixed dictionary's
positive extension approaches a face of the cone (the outer target at the
failing stage is `PrimalInfeasible`, above), and the route as formulated has
no completion that stays certified there.  At `t=0.5` the projective route
is a factor 30 behind the carried-witness flow without its Gram guard
(`9.5474e-4` against `3.19169e-5`).  Projective tests 63 (44 before).
Artifacts under `output/local_runs/paper_v_apcm_projective_t4restart_*_t8_20260915_v1/`
and `paper_v_apcm_projective_carried_*_t05_h000125_20260915_v1/`.

### 31-coordinate joint-Gram energy-neutral barrier toward t=20: no long horizon (review failed; see the repair note)

A named lane (`--corrected-controller joint_gram_barrier`, default
unchanged) was added to the long-horizon driver with `--halfspace-backend
{slsqp,active_set}`.  Under a 150-minute wall budget toward `t=20` at the
pinned DOP853 protocol the default lane reached `t=0.30` (`9,000 s`, 650,243
right-hand-side evaluations, 26,324,792 half-space projections, 46,034
accepted steps, `148 MB`) and the `active_set` lane `t=0.35` (688,322
evaluations, 28,457,232 projections).  The obstruction is a progressive
step-size collapse under the corrected velocity, not the cost per
correction: the mean accepted step falls from `1.24e-2` over `t=0` to `0.05`
to `9.0e-6` over `t=0.22` to `0.23` while half-space projections per accepted
step stay of order `300` to `800`; horizon gained per wall minute falls from
`0.0144` to `0.0006` and was still falling.  Over `t=0` to `0.30` the lane's
complete 31-coordinate l2 time-RMS distance from exact is `0.06420598`
(`coordinate_scalar_rms` `0.01184757`, all-31 scaled RMS `0.10810603`),
minimum bosonic eigenvalue `2.0685e-5`.  The reviewer's blocking findings:
the lane's base velocity is the archive EOM without the initial-residual
subtraction (`closed_electron_phonon_cone_projected_rhs` is built with
`subtract_initial_residual=False`), so the driver's Eq. (112) `raw` lane
(`0.00065244` on the same samples) is not the velocity it corrects, and the
census was solved at that other velocity (at the lane's own velocity the
barrier is active at 7 of 7 samples from `t=0`); the headline ratio mixed
two conventions.  Definitional finding that stands: the record's "uncorrected
archive EOM" `0.193205` over `t=0` to `20` is the plain archive EOM in the
`coordinate_scalar_rms` convention (round 2's stored plain lane scores
`0.193252`, its Eq. (112) lane `0.184065`, the ten-coordinate barrier
`0.165865`).  Reading: the joint-Gram barrier is retired as a long-horizon
route on this machine (extrapolating its final rate gives about 23 days to
`t=20`); the record's both-cones line was a fixed-step SSPRK(3,3) run, and a
fixed-step joint-Gram lane would be a different scheme from the pinned
protocol.

### Joint-Gram lane, review repair: the headline reverses against the right base velocity

The lane's base velocity is the plain archive EOM (`closed_scalar_rhs`,
because `closed_electron_phonon_cone_projected_rhs` is built with
`subtract_initial_residual=False`), not the Eq. (112) lane the driver had
been comparing against.  On the matched window `t=0` to `0.30` (7 samples of
the 0.05 grid) the barrier's complete 31-coordinate l2 time-RMS distance
from exact is `0.06420598` against the plain archive EOM's `0.07152712`
(ratio `0.898`) and the Eq. (112) archive EOM's `0.00065244`; the two
uncorrected velocities differ by the Eq. (112) initial residual, a fixed
31-vector of l2 norm `0.40968606`.  Re-solved at the lane's own velocity the
barrier is active at 7 of 7 samples from `t=0` with maximum correction norm
`0.2501716` (the pre-fix census, solved at the Eq. (112) velocity, read
`0.857` and `0.0647`); the plain archive EOM reaches minimum joint moment
eigenvalue `-0.02284502` over the window while the corrected lane holds
`2.0675e-5`.  The driver now builds the census base velocity from the
corrected lane's own `subtract_initial_residual`
(`cone_long_horizon.barrier_census_base_rhs`, recorded as
`corrected_lane.census_base_velocity`), reports
`protocol.corrected_derivative_coordinates` `[0, 31]` for the joint-Gram
controller, and puts the matched-horizon table behind the named
`--score-matched-horizon`; default path bitwise unchanged against HEAD at
`--final-time 2`.  The step-size collapse and the horizon (`t=0.30` in
150 minutes) stand; the lane is retired as a long-horizon route on this
machine.

## 2026-09-15: corrections from the conventions audit

Source: `agent_guidance/paper-v/overnight/conventions-audit-20260915.md`
(two tables: every accuracy convention with its defining function, and every
route's `t=0` to `4` and `t=0` to `0.05` numbers recomputed in one
convention from the saved arrays; recomputation scripts under
`output/local_runs/paper_v_review_conventions_20260915/`).  The blocks above
are left as written; the corrections below supersede them where they differ.

1. Every carried-witness accuracy figure above is the offset-subtracted
   ("dynamic") all-coordinate scalar RMS: the carried-witness 31-coordinate
   readout of the canonical cutoff-16 preparation sits `3.2442e-4` from
   exact at `t=0` (A block `3.2025e-4`), so its `t=0` to `0.05` figure is
   `3.235917e-6` offset-subtracted and `5.836875e-5` raw.  The projective
   route's offset is `8.08e-16`, so its figures are the same either way.  The
   factors quoted above (24.5, 3.98, 30) hold in the offset-subtracted
   convention.  The constant `_RECORDED_CWRMF_STRICT_ALL_COORDINATE_SCALAR_RMS
   = 3.24e-6` in `apcm_projective_event_guard_analysis.py` is stored under a
   raw key name and copied into every projective `summary.json`; to be
   renamed.
2. The plan file's overnight row that paired `1.2277e-5` (N block
   `3.1027e-8`) with `14 s` mixed two configurations: `1.227728e-5` is the
   completion retry ladder alone (`carried_completion: false`, `face_policy:
   refuse`, `112.46 s`); the carried lift plus face continuation gives
   `1.287512e-5` (N block `1.8296e-5`) at `14.18 s`.  No run combines the
   two over `t=0` to `0.05`, and their internal-energy RMS differ by six
   orders of magnitude (`5.4153e-11` against `2.1479e-5`).  Corrected in the
   plan file.
3. `max_threads_8` `1.51x` was measured on the strict `t=0` to `0.01`
   rollout; "balanced" describes the acceptance test it satisfies, not the
   speedup.  Carrying the accepted working set is `1.46x` faster in strict
   `t=0` to `0.01` (`2.04x` with threads) and `3.42x` slower in balanced
   `t=0` to `0.05`.  Corrected in the plan file.
4. The "roughly `2.9x`" claim for the bounded-depth event guard against the
   strict always-on run is withdrawn: `43,692 / 14,722` compares two
   horizons on two machines.  Per step on this machine the strict anchor is
   `73.6 s` and the balanced event-triggered resume `96.2 s`; the factor is
   unmeasured.
5. At the `t=4` restart, the carried lift plus face continuation scores
   `4.337435e-2` over the matched window `t=4` to `5.085`, more accurate than
   the committed path's `4.384520e-2` on the same window; the `4.9241e-2`
   quoted above is over its longer horizon to `5.235`.
6. The driver's "static energy" is the record's internal-energy column
   (Eq. (22) internal energy, one key).  Over `t=0` to `4`: projective
   `3.6201e-2`, plain archive EOM `2.7587e-2`, carried witness without the
   Gram guard `1.3879e-2`.  The projective route is the worst of the three
   on this quantity; the withholding above is reversed.
7. Over `t=0` to `0.05` the strict, balanced and no-guard carried-witness
   flows agree to seven figures (`3.235917e-6` / `3.235916e-6` /
   `3.235917e-6`) at `1,471 s` / `260 s` / under `1 s`, so the plan's step-1
   accuracy gate does not discriminate the guard on that horizon.  On the
   same horizon the Eq. (112) archive EOM alone scores `5.456228e-7`, better
   than every moment route, at no cost.
8. The record's "matched raw 31-coordinate archive EOM" is `closed_scalar_rhs`
   with no initial-residual subtraction (it reproduces `0.11090` /
   `0.07660` / `0.02961` / `0.02759` over `t=0` to `4` and `2.13e-2` over
   `t=0` to `0.5`).  Eq. (112) on the same scheme gives `6.082336e-2` over
   `t=0` to `4` and is absent from the record at that horizon.
9. Smaller: the three unstated horizons (`2.1281e-3`, `5.1100e-5`, the
   step-refinement triple) are all `t=0` to `0.05`; the step-refinement
   triple mixed 21/41/81-sample grids (common-grid values `3.513696e-5` /
   `1.300620e-5` / `1.488741e-5`); "N-block error" named both an
   instantaneous l2 and a horizon RMS in one paragraph; `all31_scaled_rms` is
   not comparable to the packet line's `0.063649` because the scales are
   frozen per run; `coordinate_scalar_rms` is a script-local alias of
   `all_coordinate_scalar_rms_error`; "site-0 occupation" is `rho[0,0]` in
   one driver family and `2*rho[0,0]` in the other; stage retractions at
   `t=4` are 1,068 by the guard log and 1,041 by the summary counter.

## 2026-09-15: CWRMF checkpoint resume completed, and the CHTC acceleration lane

Branch `claude/paper-v-routes-20260914`, from HEAD `d1f4eb83`.  Full report,
with every command and every artifact path:
`agent_guidance/paper-v/overnight/cwrmf-chtc-20260915b.md`.  Nothing is
committed, nothing is uploaded or submitted, and no default changed.

### The accepted working set now survives a checkpoint resume

Round 3's continuation note was that `restore_bundle_rank_hint` restores the
monotone rank hint across a resume but the accepted working set is not saved,
so a resumed rollout under `carried_accepted_basis` rebuilds its first atom's
working set from the recorded rule.  `checkpoint.npz` now carries
`bundle_accepted_rank` always and the 62-row `bundle_accepted_basis` whenever
the active seed policy is one that reads it, and `--initial-state-file`
restores both.  Older checkpoints without the keys still load.

Strict profile, always-on guard, strong coupling `lambda = 1.5`,
`gamma = 0.5`, drive amplitude 1, phonon cutoff 16, exact-correlated
preparation, `--bundle-seed-policy carried_accepted_basis
--bundle-seed-rank-margin 2`, `t = 0` to `0.005` at `h = 0.0025` run as one
rollout and again as two driver invocations joined by the checkpoint: with the
accepted working set carried the resumed segment reproduces the unbroken
rollout's `t = 0.005` row of `autonomous_trajectory.npz` exactly, every array,
maximum absolute difference `0.0`, in `30.4 s`.  From the same checkpoint with
the `bundle_accepted_basis` key removed -- the behaviour before this change --
the 384 carried coordinates differ by `2.026859e-08`, the completion
correction norm by `1.361523e-07` and the velocity margin by `1.735998e-07`,
the atom accepts at 36 active critical modes against the unbroken 35, and the
segment takes `44.4 s`.  Artifacts
`output/local_runs/paper_v_cwrmf_resume_20260915_v1/`.

A second, unrelated defect on the resume path was found while rehearsing the
lane and fixed: the driver decided whether a rollout starts at zero with
`np.isclose(times[0], 0.0)`, whose default absolute tolerance is `1e-8`, while
`exact_holstein_driven_trajectory` rejects a first sample time above `1e-15`,
so a resume starting at or below `1e-8` completed its rollout and then crashed
in scoring.  The branch now applies the reference's own `1e-15` test.

`paper_5/tests/test_apcm_carried_witness.py` goes 41 -> 45 tests.

### Balanced event-triggered guard, t = 0 to 0.05, with the round-3 flags

Balanced profile, `--event-triggered-guard --event-gram-hard-floor 2e-4
--maximum-step-subdivisions 2 --bundle-seed-policy carried_accepted_basis
--bundle-seed-rank-margin 2 --solver-setting-profile max_threads_8`,
`h = 0.0025`, 20 SSPRK2 steps: `2.646 s`, dynamic all-coordinate scalar RMS
against the exact cutoff-16 reference `3.235917e-06`, maximum readable-rate
residual `0.0`, peak resident `651 MiB`, zero Clarabel solve-tier calls.
Resuming from its checkpoint for one more step reaches `t = 0.0525` in
`0.497 s`.  Artifacts
`output/local_runs/paper_v_cwrmf_event_guard_accel_smoke_20260915_v1/`.

Both accelerations are inert over this interval and the run says so: the
adaptive bundle tier is never entered before the guard's first trigger, so
there is no working set to carry and no factorisation to thread.  The run
proves the flag set reaches the settings and that the resume runs with it; it
is not evidence of speed.

### The lane

`chtc/paper_v_cwrmf_event_guard_accel_20260915/`: apptainer wrapper, inner
runner, README, and eight bounded-segment submit files covering `t = 0` to `4`
at `0.5` per segment, `h = 0.0025`, hard floor `2e-4`.  Argument contract
`run_id final_time time_step event_gram_hard_floor initial_state_file`.  8
CPUs, 12 GB, 30 GB disk, `+MaxRuntime = 172800`.  Segment length follows the
recorded local rate: `14,722 s` for `0.38` time units of guarded propagation
is `10.8 h` per unit, so `0.5` per segment is `5.4 h` locally and `21.5 h` at
a 4x execute-node derate, inside the `48 h` bound and far inside the `72 h`
pool hold that transfers nothing.

The submit files do **not** carry `carried_accepted_basis`.  Round 3 measured
that pair on the balanced profile at `879.775 s` of Clarabel tier against the
balanced default's `257.439 s` over `t = 0` to `0.05`; the `2.04x` it earns is
a strict-profile result.  `max_threads_8` alone is this lane's acceleration,
and the two carried options are named environment overrides that no submit
file sets.

All eight submit files pass `chtc/validate_condor_submit_lifecycle.py`.  The
inner runner was rehearsed end to end outside Apptainer at `h = 1e-8`: a first
segment, a second segment resuming from the first's checkpoint, a receipt-named
archive for each, and a refusal with exit status 2 when the declared checkpoint
is missing.  Bundle
`output/local_runs/paper_v_cwrmf_event_guard_accel_20260915/paper_v_cwrmf_event_guard_accel_20260915.tgz`,
SHA-256 `edffe887de74267dd557973df4532383524e559d9dba13d098bfea5844d5110f`,
`520921` bytes (rebuilt with the wall budget below; the pre-wall-budget
bundle was `3110777f...`, `500713` bytes and must not be uploaded).  Upload
and submission need the user's Duo login and are not done here.

### The segment is now bounded in wall time as well as in horizon

Segment length bounds the propagated interval, not the wall clock, and the
cost per unit of propagated time beyond `t = 0.5` has never been measured, so
segment length alone could not keep a job inside `+MaxRuntime`.
`apcm_carried_witness_analysis` gains `--maximum-wall-seconds`, off by
default: when it is set the rollout stops at the first accepted SSPRK2 step
that ends past the budget, `checkpoint.npz` holds that step, the trajectory
arrays and `summary.json` are written for the steps taken with
`stop_reason = wall_budget` beside `maximum_wall_seconds`, and the process
exits 0, so a CHTC segment ends normally and stages out instead of being held
at the pool's 72 h wall with nothing transferred.  The integrator takes the
stop as a caller-supplied `stop_requested` predicate, consulted after every
accepted step except the last and after the checkpoint has recorded it; at
its default `None` nothing is consulted and no number moves.  The lane's
inner runner passes `138240 s`, which is `0.8 x 172800`, overridable per
segment through `CWRMF_MAXIMUM_WALL_SECONDS`, and `receipt.json` now carries
`maximum_wall_seconds`, `stop_reason` and `last_time` so a returned segment
says whether the next submission is the next segment or the same one again.

`paper_5/tests/test_apcm_carried_witness.py` goes 45 -> 48 tests: the stop
predicate stops at an accepted step whose state is bitwise the uninterrupted
rollout's state at that step; a budget of `600 s` over `t = 0` to `2e-8` at
`h = 1e-8` leaves every recorded array bitwise equal to the unbudgeted run
and records `stop_reason = final_time`, while a run without the option writes
neither key; a budget of `1e-9 s` over `t = 0` to `3e-8` stops after one
accepted step, exits 0, and leaves a `checkpoint.npz` a second invocation
resumes from.

## 2026-09-15: cheap projection route, first pass (no conic solve)

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`.
New modules `cheap_projection_route.py`, `cheap_projection_route_analysis.py`
and 16 tests; no existing file edited; full `paper_5` suite 458 passed.

Enabling fact, verified against the public builders: the retained joint
Gram (8 rows, 29 raw coordinates), the 24-row positive extension (44 state
plus 62 frontier coordinates) and the 62-row unified Gram (all 384 carried
coordinates) are exactly affine in the propagated coordinates
(reconstruction errors `1.1e-16`, `1.1e-16`, `2.8e-17`), so a physicality
repair is eigendecompose, clip, and one least-squares map back through a
pseudo-inverse factored once per rollout.  Named policies `eigenvalue_clip`,
`higham_nearest_psd`, `dykstra N`; targets `extension`, `retained`, `both`;
cadence `stage` or `step`.  Strong coupling, cutoff 16, exact-correlated
preparation, `h=0.0025` throughout.

Fixed-dictionary state (44 coordinates, frozen lift, no containment, no
stage retraction), `t=0` to `4`:

| policy | all-coordinate scalar RMS against exact (31 closed coordinates) | minimum scaled 24-row extension eigenvalue | minimum unshifted retained joint-Gram eigenvalue | wall |
|---|---:|---:|---:|---:|
| none | `0.13304` | `-1.5159` | `-1.5578` | `11.03 s` |
| Higham nearest PSD, both targets | `0.13460` | `-2.28e-3` | `-2.32e-4` | `13.06 s` |
| Dykstra 30, both targets | `0.13134` | `-1.04e-5` | `-7.92e-7` | `39.44 s` |
| record: projective event guard with stage retraction | `0.026049` | `-9.3e-16` | `2.89e-5` | `3,229.8 s` |
| record: raw archive EOM | `0.11090` | --- | `-0.5173` | --- |

The repair buys five to six decades of physicality and 1.3 percent of
accuracy.  Why `0.131` and not `0.026`: at six sampled times the frozen
lift's entrance-moment commutator velocity differs from the declared
proximal log-det completion's by 37 to 101 percent while the retained
velocity is unaffected exactly; the limiter is the frozen lift, not the
cone, and re-selecting the completion costs `0.83` to `1.34 s` per solve.
Over `t=0` to `0.05` every policy including the control scores `1.272e-5`
(the state leaves neither cone by more than `1.6e-11`) in `0.14 s`, against
the recorded retry-ladder `1.2277e-5` at `112.5 s`; over `t=0` to `0.015`
the route scores `1.7231e-6` in `0.057 s` where the recorded no-retraction
ablation scored `1.9408e-6` and then refused.  Negatives: the declared
motion scales are inert (the 15-dimensional null space of the affine map is
entirely propagated-state directions); frontier-only repair is harmful
(24-row Gram `-0.921`, retained Gram `-2.407`, frontier at 4.87 times its
frozen envelope).

Carried-witness state (384 coordinates, no-guard flow, repair on the 62-row
unified Gram after each step), `t=0` to `4`:

| quantity | no repair | Dykstra 30 per step | Dykstra 200 per step |
|---|---:|---:|---:|
| all-coordinate scalar RMS against exact | `0.48048` | `0.18430` | `0.18409` |
| minimum scaled 62-row unified Gram eigenvalue | `-26.4455` | `-3.66e-5` | `-4.25e-7` |
| minimum unshifted retained joint-Gram eigenvalue | `-17.6151` | `-2.5e-16` | `-2.5e-16` |
| wall (propagation / repair) | `26.01 s` | `69.32 s` (`26.21` / `43.06`) | `300.76 s` |

The control reproduces the record's no-guard line in every printed digit.
This is the first moment closure that reaches `t=4` with its 62-row Gram at
the cone: `69 s`, against `43,692 s` for the strict always-on guard over
`t=0` to `0.5` and `14,722 s` for the balanced event guard over 0.38 time
units, neither of which reached `t=4`.  It is still 1.66 times worse than
the raw archive EOM on the 31-coordinate RMS, so plan step 6 is not closed.
Next: the same repair to `t=20` (the record's both-cones horizon,
`0.128607`), and a hybrid that re-selects the completion every N steps with
the cheap repair between.

## 2026-09-15: projective route, task 1, with the reviewer's corrections

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`;
review: `review-routes-20260915b.md` (evidence under
`output/local_runs/paper_v_review_routes_20260915b/`).  The review failed
the report on its stage-retraction counts, which are corrected here; the
commit message of `59e3dc04` carries the wrong counts.

Kept (defaults unchanged, default `t=0` to `0.05` path bitwise equal to the
round-3 artifact): `select_outer_feasible_projective_guard(...,
target_solver=None)` as the single seam for the target stage;
`ProjectiveOuterEventGuardSettings.outer_target_solver` (`recorded` default,
`tightened_retry`); driver flags `--outer-target-solver`,
`--outer-target-single-pass`, `--outer-target-time-limit-seconds`,
`--initial-exact-time` (the exact cutoff-16 driven state at a declared time
contracted into the projected state and completed; a diagnostic initial
condition only, no exact information afterwards), `--phonon-envelope`.

Diagnosis of the post-pulse wall.  The exact state at `t=5.0`, contracted,
has a certified declared completion with no retry (14 interior Newton
iterations, KKT backward error `2.623e-9`, extension minimum `-6.2e-15`,
box envelope ratio `0.246` against the frozen envelope 1, retained
joint-Gram minimum `1.88e-5`).  Continued toward `t=8` at `h=0.0025` with
the carried lift, face continuation and the completion retry ladder it
reached `t=5.985` in 394 steps, stopped by its `1,080 s` budget, with 31
stage retractions (`0.079` per step; the report's "2" was the last step's
own count), five of them in `t=5.1325` to `5.1775`; all-coordinate scalar
RMS against exact over `t=5.0` to `5.985` `6.4865e-3` (rho `7.47e-3`, B
`7.12e-4`, N `5.10e-4`, A `1.46e-3`, C `8.95e-3`).  Every continuation from
the propagated `t=4` state (default policy, 1,068 retractions over `t=0` to
`4`, coordinate error `2.5037e-1` at `t=4`) stops between `t=4.83` and
`5.235` at `0.28` to `0.55` retractions per step at the same `h=0.0025`.
Reading: the wall is accumulated driven-interval error, not the frozen box;
the claim holds at equal step size only.

Halving the step from `t=4` (retry ladder plus face continuation,
`h=0.00125`, `1,020 s` budget) reached `t=5.1075` in 886 steps with 64
stage retractions (`0.072` per step; the report's "6" was again a last-step
count) at dynamic RMS `4.4778e-2` over `t=4` to `5.1075`, against `t=5.12`,
109 retractions, `4.5320e-2` and `2,583.8 s` at `h=0.0025`: the step buys
cost, not horizon; 800 of the 886 steps ran in about `120 s` and the last 86
took `995 s` with the ladder exhausted 78 times, all past `t=5`.  The wired
selector has not yet solved a real triggered stage inside a rollout, and no
trigger has fired on its own in any rollout to date (0 of 1,182 stages on
the exact-start run).  Next: a full rollout from `t=0` under the best cheap
policy, which has not been run.

### Reviewer corrections to the cheap projection block above

The Dykstra 30 per-step repair on the carried-witness no-guard flow makes
the Eq. (22) internal-energy RMS over `t=0` to `4` 51 times worse
(`0.0138792` unrepaired to `0.7092441` repaired) while improving the
31-coordinate RMS 2.6 times; the block's reading that the repair buys
physicality and accuracy is qualified by that column.  The rerun reproduces
the artifact bitwise (raw `0.18429968`, dynamic `0.18430033`; 62-row Gram
`-3.6615e-5`; retained `-2.50e-16`); the control is bitwise the record's
no-guard flow; the affine-map claim holds to `1.8e-15` at independent
probes.

### Reviewer findings on the CHTC acceleration lane

Gates pass (7 of 7 resume and wall-budget tests, bundle `edffe887` /
`520,921` bytes / 101 entries, lifecycle validator exit 0, the inner runner
deletes nothing and cannot overwrite a remote archive), but the
`wall_budget` resume the wall budget exists to enable is not executable as
shipped: every submit file names the previous segment's checkpoint (segment
01 names `none`), so a stopped segment cannot be resubmitted at its own
`final_time` without editing three lines the README says do not change, and
a reused run id would let stage-out overwrite
`raw_outputs/<batch>/<run_id>/checkpoint.npz`.  The bundle was also built
from an uncommitted working tree: five modules inside it differ from HEAD
`4510bd01`.  Both are being repaired before anything is uploaded.

## 2026-09-15: cheap projection route, second pass: the hybrid with periodic lift re-selection

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(second heading).  Strong coupling, cutoff 16, exact-correlated
preparation, `h=0.0025`; RMS figures are the all-coordinate scalar RMS
against the exact cutoff-16 reference on the 31 closed coordinates (raw and
offset-subtracted agree to every printed digit on this route); wall times
were measured with other agents' jobs on the machine and are upper bounds.

Negative result first: neither cheap-repair route alone is a long-horizon
closure.  Over `t=0` to `20`: the carried-witness no-guard flow with
Dykstra 30 per step scores `12.9677` (Eq. (22) internal-energy RMS
`47.2251`; 62-row Gram `-3.36e-2` after repair; the error is the
phonon-occupation block, N-block scalar RMS `35.566`, while the site-0
occupation stays at `0.210` against the record's `0.204690`; the Gram is
lost progressively after `t=4`: `-9.9e-6` at 4, `-2.4e-4` at 8, `-2.9e-2` at
20; `343.9 s`), and the fixed dictionary with the frozen lift and Dykstra 30
scores `1.41587` (internal energy `6.79887`; extension `-1.5e-4`; envelope
ratio `1.940`; `200.0 s`), against the record's both-cones `0.128607`
(internal energy `0.000939628`, `5,832.34 s`) and plain archive EOM
`0.193205` (`0.0182361`).

The hybrid: the declared proximal log-det completion (the lift) re-selected
every N steps, warm-started on the carried frontier, with Dykstra 30 on
both targets between re-selections, fixed-dictionary state, `t=0` to `4`:

| N | all-coordinate scalar RMS, `t=0` to `4` | wall |
|---|---:|---:|
| frozen lift (no re-selection) | `0.131335` | `39.4 s` |
| 64 | `0.0412601` | `49.8 s` |
| 16 | `0.0268423` | `89.6 s` |
| 4 | `0.0262899` | `137.9 s` |
| 1 | `0.0273992` | `702.2 s` |
| record: projective event guard with stage retraction | `0.026049` | `3,229.8 s` |
| record: raw archive EOM | `0.11090` | --- |

The frozen lift was the whole gap between the cheap route and the projective
route; N=16 gives the projective route's accuracy at one thirty-sixth of its
cost.  N=1 is 4 percent worse than N=4 at 5.1 times the wall (250 of 1,600
re-selections refuse and the carried lift is kept there); refusals are
ordinary (1 of 25, 9 of 100, 59 of 400, 250 of 1,600) and never stop the
route.  No post-pulse wall: N=16 to `t=8` completes all 3,200 steps in
`195.7 s` at `0.0577300` over `t=0` to `8` (extension minimum `-9.1e-6`,
retained `-8.3e-7`); N=64 gives `0.121366` at `98.4 s`.  Nothing happens at
`t=5.09` to `5.24`, where every propagated projective continuation stops;
the instantaneous 31-coordinate error is `0.3025` at `t=4`, `0.2867` at
`t=5` and `0.7041` at `t=8`, and what degrades after the drive is the phonon
occupation (N-block `0.0159` to `0.107`), not representability.  The
internal-energy column for the hybrid was not reported and is requested
next, given the Dykstra-only repair's 51-fold penalty on it.

Code: `CheapProjectionRoute.reselect_lift`, `lift_reselection_interval` and
`lift_reselection_warm_start` on the integrator, five trajectory fields,
driver flags `--lift-reselection-interval` and `--lift-reselection-start`;
21 tests in the file (5 new); `paper_5` suite 466 passed.  Nine artifacts
under `output/local_runs/paper_v_cheap_projection_*`, peak `861 MB`, longest
run 12 minutes.  Next: N=16 and N=4 to `t=20` against the record's
`0.128607`, with the internal-energy column.

### First Paper V CHTC submission: accelerated event-guard lane, segment 1

Cluster `10554184`, submitted 2026-09-15T15:45:55Z from `ap2001`, batch
`paper_v_cwrmf_event_guard_accel_20260915_v1-accel_s01_t0000_t0500`,
arguments `accel_s01_t0000_t0500 0.50 0.0025 2e-4 none none`
(`run_id final_time time_step event_gram_hard_floor initial_state_file
expected_initial_time`); 8 CPUs, 12 GB, 30 GB, 48 h `MaxRuntime` with the
driver's `--maximum-wall-seconds 138240`; bundle `eed32f13` (`502,619`
bytes) built from `git archive` of `0f9d5f35` plus the lane; the remote
image is a symlink to the Paper II campaign's `image.sif` (digest prefix
`fa5c4ea89a19ad2f`, identical to the local copy).  Lifecycle validator exit 0
on the eight submit files remotely; dry run 1 job.  Receipt:
`chtc/paper_v_cwrmf_event_guard_accel_20260915/submission_receipt_10554184.json`.
Physical protocol: balanced profile, event-triggered guard with hard floor
`2e-4`, step subdivisions 2, `max_threads_8`, default bundle seed policy,
`h=0.0025`, strong coupling, cutoff 16, exact-correlated preparation; the
local reference for this segment is the `t=0.1175` to `0.5` resume above
(`14,722 s` on one thread).  Segment 2 is submitted only from segment 1's
returned receipt through `make_resume_submit.py`.

## 2026-09-15: cheap projection route, third pass: the hybrid at `t=20` and what the repair is for

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(third heading).  Same configuration as the second pass: strong coupling,
cutoff 16, exact-correlated preparation, `h=0.0025`, the unguarded
fixed-dictionary predictor, Dykstra 30 on both targets at stage cadence
between re-selections, the declared proximal log-det completion re-solved
every N steps warm-started on the carried frontier.  Scores are the
all-coordinate scalar RMS on the 31 closed coordinates against the exact
cutoff-16 driven reference on the stated horizon; internal energy is the
sample RMS of the time-independent energy of Eq. (22) on that horizon; the
fixed-dictionary `t=0` offset is `8.08e-16`, so raw and offset-subtracted
agree to every digit printed.  No module was edited; nothing committed.

Negative result first: neither hybrid cadence is a `t=20` closure.  Over
`t=0` to `20`, `N=4` scores `0.259573` (internal energy `0.804515`,
`884.2 s`) and `N=16` scores `0.579332` (`2.59371`, `486.2 s`), against the
record's both-cones `0.128607` (`0.000939628`, `5,832.34 s`) and plain
archive EOM `0.193205` (`0.0182361`).  `N=4` is 2.02 times the both-cones
score and 1.34 times the archive EOM.  The `t=0` to `4` ordering reverses on
the long horizon: the two cadences were within 2 percent of each other at
`T=4`, and at `T=20` `N=4` is 2.23 times more accurate than `N=16`.

New and positive: both cones are held for the whole 20 time units, which
neither task-2 cheap route manages.  Minimum scaled 24-row extension
eigenvalue `-2.401e-5` (`N=16`) and `-5.553e-6` (`N=4`), retained joint Gram
`-8.296e-7` and `-8.975e-8`, standardized frontier inside its frozen box
throughout (`0.9105` and `0.6356`), against the frozen lift's `-1.51e-4` at
envelope `1.940` and the carried witness's `-3.36e-2`.  Re-selecting the lift
is what keeps the frontier in the box.  The whole long-horizon error is the
phonon-occupation block (N-block scalar RMS `1.36235` of `N=16`'s
`0.579332`); the site-0 occupation stays at the record's scale (`0.199933`
and `0.203657` against the both-cones `0.204690`).  Refusals stay ordinary
and never stop a rollout: 22 of 500 and 154 of 2,000, spread over the whole
interval.

The Eq. (22) internal-energy column, added to every hybrid row by rescoring
saved artifacts (nothing propagated), over `t=0` to `4`: no-repair frozen-lift
control `0.0292046`, Dykstra-only frozen lift `0.780101`, `N=64` `0.110171`,
`N=16` `0.0420873`, `N=4` `0.0376080`, `N=1` `0.0474351`, against the record's
projective event guard `0.036201` and plain archive EOM `0.027587`; over
`t=0` to `8`: `1.90816`, `0.870186`, `0.226943`, `0.0915587`.  **The hybrid
does not inherit the Dykstra-only repair's internal-energy penalty.**  That
penalty is 26.7-fold on the fixed dictionary (`0.0292046` to `0.780101`),
the same defect the reviewer measured at 51.1-fold on the carried witness;
re-selection removes 95 percent of it, and at `N=4` the hybrid matches the
projective route on the internal energy (`0.0376080` against `0.036201`) as
well as on the coordinate score.

The ablation that separates the hybrid's two ingredients -- `N=16` over `t=0`
to `4` with `policy="none"`, so the lift is re-selected every 16 steps and
nothing is repaired at any of the 4,800 stage opportunities -- scores
`0.0394447` at internal energy `0.0305130` and site-0 occupation `0.0199981`,
with the 24-row extension at `-0.323490` and the retained joint Gram at
`-0.759345`, 52 of 100 completions refused, `119.7 s`.  So re-selection alone
delivers 88 percent of the hybrid's reduction in the score, and the repair
supplies the physicality (six and seven decades), a further 1.47 times of
accuracy, and the drop in refusals from 52 to 9 per 100 solves; its price is
1.38 on the internal energy and 2.31 on the site-0 occupation, with the C
block moving the other way (`0.0485123` to `0.0226584`).

Code: no module edited, `cheap_projection_route.py` and
`cheap_projection_route_analysis.py` unchanged from `b04ec23b`; one new test
in `paper_5/tests/test_cheap_projection_route.py` (22 in the file),
`test_lift_reselection_without_a_repair_moves_the_lift_only_on_cadence`;
`paper_5` suite 467 passed in `250.29 s`.  Artifacts under
`output/local_runs/paper_v_cheap_projection_hybrid_{n16,n4}_t20_20260915_v1`
(both launched by the previous instance of this key, both completed, peak
`773` and `776 MiB` of their own `peak_resident_bytes`) and
`paper_v_cheap_projection_hybrid_n16_norepair_t4_20260915_v1` (this pass,
`351 MiB`, `ram_guard --limit-mb 3000`, `119.7 s`).  Next: `N=1` past `t=4`
to see whether its `t=0` to `4` ordering also reverses, and a repair that
projects in the Eq. (22) metric to see whether the residual 1.38-fold energy
cost is removable.

## 2026-09-15: projective route, task 2: the full rollout from `t=0`

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`,
second heading.  Fixed dictionary APCM, 15 entrance keys, cutoff 16, strong
coupling, `h=0.0025`, carried lift (margin multiple 10) with face continuation
and the declared completion retry ladder (`--completion-retry-shifts 1e-8
1e-7 --completion-retry-weights 1e-1`); every count is the run's own total
from `summary.json`.  No module was edited and no default changed.

Negative result first: a single projective rollout does not pass `t=6` and
does not reach `t=4`.  Three chained 20-minute segments from `t=0` reached
`t=3.115` (1,246 of the 3,200 steps `t=8` needs) in `3,182.0 s` with 117 stage
retractions (7, 35, 75 by segment), all-coordinate scalar RMS `1.936806e-2`
against the exact cutoff-16 driven reference on the 31 closed scalar
coordinates over `t=0` to `3.115` (rho `3.60e-2`, B `2.07e-2`, N `1.62e-2`, A
`9.53e-3`, C `1.78e-2`); the declared ladder was certified 189 times and
exhausted 132 at the stage lift, 56 and 12 inside a retraction.  Over that same
interval the recorded default policy scores `2.030594e-2`, so the named policy
buys 4.6 percent.  There is no `t=4` coordinate error to set beside the
default policy's `2.503676e-1`.

The cost wall is the trajectory's, not the policy's.  The rolling 20-step
median per-step wall sits at about `1 s` to `t=3.0`, exceeds `5 s` at
`t=3.0419` and `10 s` at `t=3.0913`, and the last 21 steps cost `52.54 s`
each.  In `t=3.00` to `3.45` the endpoint unshifted 24-row Gram minimum is
negative (chain minimum `-1.240e-4`, against the declared hard floor `1e-3`, so
0 of 3,780 stages triggered the outer guard), the carried lift stops being
accepted (0 of 93 past `t=3.0625`, against 1,804 of 2,508 over `t=0` to
`2.085`), the declared ladder is exhausted 84 of 85 times, and a stage with a
retraction applied costs `10.25 s` mean against `0.459 s` for one without.
The recorded default policy pays the same episode in the same window ---
`1,589.0 s` of its `3,229.8 s`, 322 retracted stages of 749 --- and then
leaves it at `t=3.45` and finishes `t=4`; inside the episode the carried lift
costs `8.034 s` per stage against the default's `2.121 s`.  The episode opens
as the pulse switches off: the field difference `2 sin(pi t/4) exp(-t^2/2)` is
`2.707e-1` at `t=2`, `1.571e-2` at `t=3`, `1.674e-3` at `t=3.5`.

Halving the step from `t=0` makes the error worse, not better.  One 20-minute
segment at `h=0.00125` reached `t=1.7075` in `1,052.1 s` with 6 stage
retractions and scored `8.818433e-3` over `t=0` to `1.7075` against the
`h=0.0025` chain's `8.361075e-3` on the same coordinates, and it carries 1.04
to 1.16 times the coordinate error at every shared sample time from `t=0.25`
to `1.5`.  So the driven-interval error is the fixed 15-key dictionary's
closure defect, not the time discretization, and `2.503676e-1` at `t=4` is
within a few percent of the floor for this dictionary; halving it is a
dictionary question, not a projection one.

Against the cheap projection route's hybrid (`t=8` at `5.773e-2` in `195.7 s`,
no post-pulse wall), the projective route costs 42 times the wall per unit
time for comparable accuracy and less horizon.  What it still produces is a
per-step certificate: the chain's accepted steps carry a declared proximal
log-det completion certified inside the frozen box (minimum scaled
positive-extension eigenvalue `-4.246e-13` to `3.006e-7` by segment, minimum
retained joint moment eigenvalue `7.228e-6` to `3.401e-4`), where the hybrid's
`t=8` state carries `-9.124e-6` and `-8.296e-7`; and it measures the distance
to the outer cone's face at every stage, which the cheap route does not
compute.  Artifacts:
`output/local_runs/paper_v_apcm_projective_t0_carried_ladder_facecont_t8_20260915c_seg{1,2,3}`,
`..._h000125_20260915c_v1`, and `paper_v_projective_t0_chain_20260915c`
(scoring scripts and logs), peak `360 MB`, every run under
`ram_guard --limit-mb 4500`.  Next, if the route is continued: the outer
guard has now evaluated 7,882 stages across this round without one trigger,
so whether the declared event floors should be tightened is still the open
scientific-acceptance question, and it is the user's.

## 2026-09-15: cheap projection route, fourth pass: the energy-neutral repair

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(fourth heading).  Same configuration as the second and third passes: strong
coupling, cutoff 16, exact-correlated preparation, `h=0.0025`, the unguarded
fixed-dictionary predictor, Dykstra 30 on both targets at stage cadence
between re-selections, the declared proximal log-det completion re-solved
every N steps warm-started on the carried frontier.  Scores are the
all-coordinate scalar RMS on the 31 closed coordinates against the exact
cutoff-16 driven reference on the stated horizon; internal energy is the
sample RMS of the time-independent energy of Eq. (22) on that horizon.  No
default changed; nothing committed.

Enabling fact, verified against the public chain the driver scores: Eq. (22)
is an exactly linear functional of the 29 raw moment coordinates (offset
zero, reconstruction `2.2e-16`), carried by seven of the 106 parameters --
`-2 t_hop` on the electronic Bloch x coordinate, `2 g` on each
coherent-phonon real part and on two raw cross coordinates, `omega_ph` on
each diagonal raw normal coordinate -- and vanishing on the anomalous, the
entrance and all 62 frontier coordinates.  The map back is one linear solve,
so the energy is also a linear functional of the realified Gram features,
and the hyperplane on which it equals a declared reference joins the cones
and the affine Gram set as a third declared set of the same Dykstra
alternation.  Two declared reference modes: `post_pulse` (off until a
declared drive-end time, then the energy that trajectory itself carried
there) and `drive_power` (on from the start, the reference advanced with the
SSPRK(3,3) weights by `gradient . (velocity at t - velocity at 0)`, which is
the exact drive power because the Hamiltonian's only time dependence is
`drive_difference` and that vanishes at `t=0`).  This is not the record's
`energy_neutral` control restriction, which confines the correction to the
null space of the energy-gradient row and so leaves the velocity's own
energy drift; holding the state on a reference removes both.

Which post-pulse quantity is a law, measured on the exact cutoff-16
reference over `t=4` to `20`: the Eq. (22) energy moves by `3.07e-6` (value
`-3.571397982` at `t=4`) and the phonon number, the trace of the uncentered
phonon normal moment, moves by `0.716` (value `4.017037038`), against an
energy swing of `0.286911` over the driven interval.  So task (c)'s
phonon-number-neutral variant is not admissible and was not run: `g n (b +
b^dagger)` does not commute with `b^dagger b`.

Negative result first: the constraint buys the phonon-occupation block by
giving back the cone and by making the declared completion refuse.  At
`N=16` over `t=0` to `20` the minimum scaled 24-row extension eigenvalue
moves from `-2.401e-5` to `-3.311e-4` and the retained joint Gram from
`-8.296e-7` to `-8.776e-3`, with 107 of 500 completions refusing against 22
of 500; at `N=4` refusals reach 503 of 1,163 at `0.917 s` a solve against
`0.396 s`, and that run stopped at `t=11.63` after 4,652 of 8,000 steps on
its `1,150 s` budget.  The constrained `N=16` score `0.239172` is still 1.86
times the record's both-cones `0.128607` and 1.24 times the plain archive
EOM `0.193205`.

What it delivers, `N=16` over `t=0` to `20`, post-pulse reference against
task 3: score `0.239172` against `0.579332`, internal energy `0.069384`
against `2.59371`, N-block scalar RMS `0.267235` against `1.36235`, site-0
occupation `0.247970` against `0.199933`, `669.8 s` against `486.2 s`.  The
instantaneous 31-coordinate error at `t=20` falls from `6.4271` to `2.0538`,
of which the N block falls from `5.7088` to `0.8312`, and the post-pulse
excursion of the propagated phonon number falls from `7.042` to `0.813`
against the exact `0.716` -- without ever constraining that number.  At
`N=4` over `t=0` to `11.63`: `0.079973` against `0.138634`, internal energy
`0.051374` against `0.382336`, N block `0.101002` against `0.256683`, site-0
occupation `0.068658` against `0.097051`.  The energy is held on its
reference to `1.78e-15` over 18,939 constrained repairs and `2.22e-15` over
9,156, and the hyperplane was representable at every one.

The drive-power reference, `N=16` to `t=20` in `719.6 s`: internal energy
`0.026000`, which is the best on this route and within 1.43 of the plain
archive EOM's `0.0182361`; site-0 occupation `0.186909`, better than the
record's both-cones `0.204690`; score `0.328674`, worse than the post-pulse
`0.239172`; extension `-2.331e-3` and retained joint Gram `-5.284e-2` with
276 of 500 completions refusing.  Over `t=0` to `4` it is the best row this
route has produced, `0.023744` at internal energy `0.020406`, against the
record's projective event guard's `0.026049` and `0.036201` at `3,229.8 s`.

Build-time fact, which answers task 3's third next step: the 24-row
extension does not constrain the phonon normal block -- one of the four raw
normal coordinates lies entirely in the 15-dimensional null space of its
affine map and two more lie 83 percent in it -- while the two targets
together, which every run here repairs, have full column rank 106 of 106
with the Eq. (22) gradient direction fully visible.

Code: `InternalEnergyFunctional`, the `energy_reference_mode` and
`energy_tolerance` settings, `energy_feature_functional`, the
`energy_reference` argument of `AffineGramCoordinates.repair`, the
`hyperplane` argument of the Dykstra alternation, the integrator's
`energy_reference_time` and seven trajectory fields; driver flags
`--energy-reference-mode`, `--energy-reference-time`, `--energy-tolerance`
and an `internal_energy` summary block; 28 tests in the file (6 new);
`paper_5` suite 473 passed in `239.12 s`.  The default path is unchanged:
with the mode at `"none"` the post-pulse `N=16` rollout is bitwise the
task-3 rollout over the whole driven interval `t=0` to `4`.  Four artifacts
under `output/local_runs/paper_v_cheap_projection_energy_*`, each wrapped in
`ram_guard --limit-mb 3000`, one at a time, largest own peak `782 MiB`, no
run over 20 minutes, walls measured at load average 4.3 to 4.7 and therefore
upper bounds.  Next: the completion retry ladder or a nonzero
`energy_tolerance` at `N=4`, which is what stands between this route and a
completed `t=20` run at that cadence.

## 2026-09-15: projective route, task 3: is the `t=4` error a dictionary floor?

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`, third
heading.  Task 2 concluded that the `2.503676e-1` coordinate error at `t=4` is
the fixed 15-key entrance dictionary's closure floor, because halving the step
raises it.  This round enlarged the dictionary and measured what moves.
Configuration common to every rollout: strong coupling, cutoff 16,
exact-correlated preparation, SSPRK(3,3) at `h=0.0025`, the unguarded
fixed-dictionary predictor, Dykstra 30 on both targets at stage cadence, the
declared proximal log-det completion re-selected every `N` steps warm-started
on the carried frontier -- the cheap route's hybrid, imported at its committed
API from the tree at `d31a0a9e`, because another agent was editing that module.
No module was edited and no default changed.

Four declared dictionaries exist, all as the `active_keys` of a
`SymmetryReducedPositiveExtension`, whose Gram rows are nine base half-words
plus one per active key: 15 entrance keys (44 state coordinates, 24 extension
rows, 62 frontier keys, 29 lower keys), 19 (48/28/111/33), 27 (56/36/70/41) and
31 (60/40/129/45).  `INITIAL_PROMOTION_CANDIDATE_KEYS` is the entire declared
enlargement -- the four relative-mode moments of the identity and three boson
coordinates and the twelve of two spin operators and one boson coordinate, all
degree three -- so `15 -> 31` is the only declared step and those two blocks its
only sub-steps.  The 27-key dictionary is the one that does not construct
through `prepare_projected_apcm_initial_state`: the declared completion refuses
at the contracted preparation (40 Newton iterations, relative face support
35/36, directions 19/70) and that function fails closed without consulting a
retry; the first rung of the declared completion retry ladder
(`logdet_shift 1e-8`) certifies there in 12 iterations, and the 27-key rollouts
use it.  The record's 60-coordinate positive-`M4` route (cell A, `0.012580` over
`t=0` to `4`) declared the same 31-key set, hard-wired at eight places in
`adaptive_positive_moment_propagation.py`, but closes its frontier with
`PositiveFourthMomentCompletion` (`phonon_envelope 16.0`, `logdet_shift 1e-5`),
so its score is not a line these rows can sit beside.

Result, first: the error does fall with dictionary size, but only on 27 of the
31 closed scalar coordinates.  At `N=16` over `t=0` to `4` the 27-key
dictionary scores `1.630660e-2` against the 15-key's `2.684227e-2` -- 1.65
times smaller, and below the recorded projective event guard's `0.026049` on
the same horizon, at `749.1 s` against `3,229.8 s`.  On the 27 coordinates
outside the phonon-occupation block the scalar RMS falls monotonically with
every key added (`2.810405e-2`, `2.491478e-2`, `1.319865e-2` at 15, 19, 27
keys); on the four phonon-occupation coordinates it rises monotonically
(`1.589107e-2`, `1.716470e-2`, `2.974702e-2`).  At `N=32`, the one cadence
where all four dictionaries complete `t=4`, the outside-block figure goes
`3.275830e-2`, `2.632542e-2`, `1.769154e-2`, `1.719508e-2` at 15, 19, 27, 31
keys (a factor `1.905`) while the phonon-occupation figure goes `3.086475e-2`,
`3.916403e-2`, `5.574190e-2`, `8.644553e-2` (a factor `2.80` the other way).
The two added blocks do not compose: the twelve two-spin-and-one-boson moments
carry 46.0 of the 47.5 percent reduction alone, and the four
identity-and-three-boson moments add `1.5` further points on top of them while
costing the phonon-occupation coordinates another factor `1.55`.  At the
loosest cadence the rising block wins outright: at `N=64` over `t=0` to `4` the
31-key dictionary scores `6.232531e-2` against the 15-key's `4.126006e-2`.

On the `t=4` claim itself: at the same propagation and cadence the Euclidean
norm of the closed scalar coordinate difference at `t=4` falls from
`3.025108e-1` at 15 keys to `2.350741e-1` at 27, and from `2.946822e-1` to
`1.829419e-1` on the coordinates outside the phonon-occupation block.  So the
floor is a dictionary floor and the dictionary moves it, by `1.29` on the whole
chart and `1.61` on the part that improves; the one declared enlargement does
not halve it, and everything here is measured at the cheap hybrid's
propagation, whose 15-key `t=4` value is `3.025108e-1` and not the projective
route's `2.503676e-1`.

Cost: the enlargement is paid entirely in the completion solve, `0.71`, `1.57`,
`7.10`, `16.28 s` mean per solve at 15, 19, 27, 31 keys, against `8.7` to
`14.2 s` of velocity for a whole 1,600-step rollout.  It does not track the
frontier dimension -- 27 keys has 70 frontier keys against 19 keys' 111 and
costs `4.5` times as much per solve.  The 31-key rollout at `N=16` therefore
does not reach `t=4` in twenty minutes: `t=2.52`, 1,008 of 1,600 steps,
`1,053.1 s` of which `1,025.9 s` is 63 solves.  Physicality holds throughout:
minimum scaled positive-extension eigenvalue `-1.969e-7`, `-5.824e-7`,
`-6.347e-6` at 15, 19, 27 keys, retained joint Gram at `1e-8`, standardized
frontier inside the frozen box at `0.4366`, `0.4433`, `0.3646`.

The residual-driven dictionary enlargement layer runs and admits nothing.  One
checkpoint of `APCMClosureGraphAnalyzer.analyze` at the 15-key dictionary, at
the preparation and at the `t=1` state, conditions 15 candidates -- nine
relative-mode moments of two spin operators and one boson coordinate and six
degree-four moments of one spin operator and three boson coordinates, taken
from `rhs_frontier_keys`, not from the promotion candidates -- builds a nested
30-active-key, 223-frontier-key diagnostic extension, and returns
`nested_completion_failure: moment_extension_degeneracy: reference completion:
PrimalInfeasible` at both times, so zero candidates are scored, `50.8` to
`53.7 s` at peak `3.29 GB`.  It also has no caller anywhere under `paper_5`, is
absent from `stability/__init__.py`, and nothing rebuilds a model from an
admitted key, so declaring the dictionary by hand is the only way a larger one
reaches a rollout today.

Artifacts under `output/local_runs/`:
`paper_v_dictionary_entrance15_n{16,32,64}_t4_20260915d`,
`paper_v_dictionary_entrance15_plus_boson_cubic_n{16,32,64}_t4_20260915d`,
`paper_v_dictionary_entrance15_plus_two_spin_boson_n{16,32}_t4_20260915d`,
`paper_v_dictionary_hidden31_n{16,32,64}_t4_20260915d`,
`paper_v_dictionary_adaptation_20260915d`, and
`paper_v_projective_dictionary_20260915d` (nine scripts and the logs).  Every
run under `ram_guard --limit-mb 4500`, one at a time, peak `3,293 MB`, none over
twenty minutes.  The 15-key rows reproduce the record in every printed digit
(`0.026842268` at `N=16`, `0.0412600573` at `N=64`).  Next, if the route is
continued: the phonon-occupation block is the only part of the chart the
enlargement hurts, and whether that is the cutoff-16 truncation or the
enlarged extension failing to constrain those four coordinates is a build-time
question about the affine map, not a run.

## 2026-09-15/16: projective route, task 4: 27 keys with the energy-neutral repair

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`
(headings "(a) The chained rollouts" through "(c) Is it the best Paper V
moment-closure trajectory on record?"); the agent was killed by an API
disconnect after writing the report and before the worklog block, which
this block supplies from the report; an independent review of the chain is
in progress (`review-27key-chain-20260916.md`).

Configuration: 27 entrance keys (56 state coordinates, 36 extension rows,
70 frontier keys; retry-ladder preparation), hybrid cadence N=16 (the
declared proximal log-det completion re-selected every 16 steps, warm
started on the carried frontier), Dykstra 30 on both targets between
re-selections with the Eq. (22) energy hyperplane in `drive_power` mode
(reference advanced by the Hamiltonian's drive power; no exact information),
`h=0.0025`, strong coupling, cutoff 16, exact-correlated preparation, six
chained `ram_guard --limit-mb 4500` segments each stopped by a `1,000 s`
budget or its horizon; 8,000 accepted steps; `4,794.4 s` total autonomous
wall, of which `4,416.9 s` (92 percent) is the 500 completion solves
(`8.83 s` mean, 144 refused); 23,959 repairs applied with the energy held
to `1.78e-15`; peak resident `631 MB`.  RMS figures are the all-coordinate
scalar RMS on the 31 closed scalar coordinates against the exact cutoff-16
reference.

| horizon from `t=0` | all 31 | 27 outside the phonon-occupation block | 4 phonon-occupation | Eq. (22) internal energy | site-0 occupation | minimum retained joint Gram | minimum scaled extension |
|---|---:|---:|---:|---:|---:|---:|---:|
| `t=4` | `1.142716e-2` | `1.164e-2` | `9.836e-3` | `4.597e-3` | `1.431e-2` | `-4.5e-4` | `-4.9e-5` |
| `t=8` | `2.593079e-2` | `2.475e-2` | `3.283e-2` | `5.752e-3` | `3.028e-2` | `-2.2e-3` | `-1.9e-4` |
| `t=12` | `4.250925e-2` | `4.385e-2` | `3.199e-2` | `6.088e-3` | `5.278e-2` | `-2.5e-3` | `-2.0e-4` |
| `t=20` | `8.519313e-2` | `9.046e-2` | `3.187e-2` | `6.345e-3` | `1.126e-1` | `-2.5e-3` | `-2.0e-4` |

Against the record (same convention): at `t=4` the recorded projective event
guard `2.6049e-2` (internal energy `3.620e-2`, `3,229.8 s`) and the
60-coordinate positive-M4 cell A `1.2580e-2` (internal energy `7.963e-4`,
retained joint Gram minimum `2.889e-5`, `3,399.6 s`; a different
construction, `PositiveFourthMomentCompletion` at `logdet_shift 1e-5`); at
`t=20` the both-cones run `1.286070e-1` (internal energy `9.396e-4`, both
cones held with retained minimum `1.368e-5`, `5,832.34 s`) and the plain
archive EOM `1.932050e-1` (`1.824e-2`).  The 15-key energy-neutral hybrids
over `t=0` to `20`: `post_pulse` `2.3917e-1` (internal energy `6.938e-2`,
`669.8 s`), `drive_power` `3.2867e-1` (`2.600e-2`, `719.6 s`); unconstrained
`5.7933e-1` (`2.594`, `486.2 s`).

Readings.  (1) On the 31-coordinate RMS this chain is the best Paper V
moment-closure trajectory on record at every horizon it reached (`0.908` of
cell A at `t=4`, `0.662` of the both-cones line at `t=20`, at `0.82` of the
both-cones wall); on the Eq. (22) internal energy it is `5.8` times worse
than cell A at `t=4` and `6.8` times worse than the both-cones line at
`t=20`; and it carries no physicality statement, since its repaired moments
sit `2.46e-3` outside the retained cone and `2.04e-4` outside the scaled
positive extension where both recorded cone routes stayed inside.  (2) The
energy constraint reverses the enlargement's phonon-occupation penalty: on
the four phonon-occupation coordinates over `t=0` to `4`, 15 keys
unconstrained `1.589e-2`, 27 keys unconstrained `2.975e-2`, 27 keys
energy-neutral `9.836e-3`; the constraint costs the electronic density block
(`1.36x`), the coherent-phonon block (`1.53x`) and the site-0 occupation
(`1.42x`) at `t=4`, and those are the blocks that grow after the pulse (at
`t=20` rho `1.358e-1`, B `1.873e-1`, N `3.187e-2`, A `4.397e-2`, C
`3.128e-2`), while the phonon-occupation block stops growing after the pulse
(`3.258e-2` at `t=6`, `3.187e-2` at `t=20`).  (3) A build-time probe of the
scaled positive extension's sensitivity to the 29 raw moment coordinates:
one of the four raw phonon normal coordinates (`raw_normal_01_imag`) has
exactly zero derivative at both 24 and 36 rows, and the other three are seen
`3.1` to `6.6` percent as strongly as the electronic Bloch coordinates at 27
keys (`2.2` to `4.5` at 15), while the retained Gram sees all 29 equally; the
Eq. (22) hyperplane is a statement about exactly that block, which is why it
repairs it.  Artifacts:
`output/local_runs/paper_v_dictionary_energy_power_27key_n16_20260915e_seg1`
to `_seg6`, `paper_v_dictionary_energy_postpulse_27key_n16_20260915e_seg1`
to `_seg4`, `paper_v_projective_dictionary_energy_20260915e/`.

## 2026-09-15/16: cheap projection route, fifth pass: the chained `N=4` energy-neutral runs and the retry ladder

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(fifth heading).  Same configuration as the fourth pass at `N=4`: strong
coupling, cutoff 16, exact-correlated preparation, `h=0.0025`, the unguarded
fixed-dictionary predictor, Dykstra 30 on both targets at stage cadence, the
declared proximal log-det completion re-solved every 4 steps warm-started on
the carried frontier, the energy-neutral option in the named mode.  Scores
are the all-coordinate scalar RMS on the 31 closed coordinates against the
exact cutoff-16 driven reference on the stated horizon; internal energy is
the sample RMS of the time-independent energy of Eq. (22) on that horizon.
No default changed; nothing committed.  The previous instance launched the
runs and was killed while scoring; this pass propagated nothing but the
exact reference on the chained sample times.

Three additions to the owned modules, each a declared option whose absence
is the earlier form: the record's completion retry ladder, built by the
record's `build_completion_ladder` and walked only at a refused lift
re-selection; an `energy_tolerance` above zero; and `--resume-from`, which
continues a rollout stopped on its wall budget from its last accepted state,
carried frontier, time and the first segment's declared reference (a
segment whose parameters, cutoff, dictionary or step differ is refused).
Six tests cover them; 34 pass in the file.  Two chained segments reproduce
one rollout to `1e-12`.

Negative result first: neither completed `N=4` chain stands next to the
both-cones line, and both are worse over `t=0` to `20` than the
unconstrained `N=4` hybrid.  Post-pulse, tolerance `1e-3`, three segments,
`2,467.4 s`: `0.318110`, internal energy `0.0539267`, site-0 `0.212252`,
N block `0.4805`, extension `-1.153e-3`, retained joint Gram `-2.632e-2`,
frontier envelope `0.9966`, 1,270 of 2,000 completions refused, all 173 of
the last segment (`t=18.27` to `20`).  Drive-power, tolerance `1e-3`, three
segments, `3,179.8 s`: `0.283805`, internal energy `0.0437311`, site-0
`0.290898`, N block `0.3089`, rho / B blocks `0.3435` / `0.5254`, extension
`-6.815e-4`, retained joint Gram `-1.777e-2`, envelope `0.9938`, 1,165 of
2,000 refused.  Against the unconstrained `N=4` `0.259573`, the `N=16`
post-pulse `0.239172`, the record's both-cones `0.128607` /
`0.000939628` / retained `1.36819e-5` at `5,832.34 s` and the plain archive
EOM `0.193205` / `0.0182361`.  Once the energy is held the tighter cadence
is the worse one, because the constrained completion refuses at 58 to 64
percent of re-selections at `N=4` against 21 percent at `N=16`.  The
tolerance never bound: no sample is outside `1e-3` (post-pulse within
`8.92e-6`, drive-power within `3.91e-14`); what reached `t=20` was the
chaining.

The ladder is not the refusal fix.  Post-pulse, tolerance `0`, shifts
`1e-8`, `1e-7`, weight `0.1`: it certified 119 of 493 refused
re-selections, all before `t=8.19`, and 0 of 289 after at `3.87 s` a solve,
`1,408.3 s` of retries in `2,322.5 s`; the chain stopped at `t=11.09` on
its second segment's budget and was not continued, scoring `0.163635`
there, the worst of the three variants.

The post-pulse `N=4` row past about `t=8` is one sample of a sensitive
trajectory.  The tolerance chain and the task-4 run are the same declared
configuration up to repairs skipped while the energy drift was below
`8.9e-6`; they coincide over `t=0` to `4`, separate to `1e-2` in the
31-coordinate Euclidean norm at `t=6.93`, `0.1` at `t=8.38` and order one
at `t=11`, and score `0.110625` against `0.072875` over `t=0` to `11.09`
(ladder chain `0.163635`; unconstrained `0.127335`).  The task-4 `t=0` to
`11.63` advantage (`0.079973` against `0.138634`) is therefore not a
reproducible factor: the tolerance chain scores `0.132357` on that horizon.
The `t=0` to `8` scores of all three agree to three digits
(`0.046854`, `0.046722`, `0.046972`).  The amplifier was not measured; the
candidate is the completion's refusal, a discontinuous function of the
state.

Over `t=0` to `4` against the projective event guard's `0.026049` /
`0.036201`: post-pulse `0.0262899` / `0.037608` (bitwise task 3), ladder
`0.0263373` / `0.0382408`, drive-power `N=4` `0.0476697` / `0.0373856`,
against the `N=16` drive-power row's `0.023744` / `0.020406` and the
projective agent's 27-key `N=16` drive-power chain's `0.011427` /
`0.004597`.  That 27-key chain (`0.085193` / `0.0063446` / retained
`-2.463e-3` at `t=20`, `4,794.4 s`) is the only row below the both-cones
score, at 6.8 times its internal energy and without a physicality
statement.  Every 15-key row is a containment.

Artifacts under `output/local_runs/`:
`paper_v_cheap_projection_energy_{ladder,tol,power_tol}_n4_t20_20260915_v1_seg*`,
`paper_v_cheap_projection_resume_smoke_{a,b}_20260915_v1`, and the scores
in `paper_v_cheap_projection_task5_score_20260915_v1/` (`score_segments.py`,
`chain_score_*.json`).  Every segment under `ram_guard --limit-mb 3000`,
one at a time, `1,150 s` budget, longest `1,172.5 s`, largest own peak
`507 MiB`.  Next: nothing at `N=4` on this route; settle the sensitivity
before comparing any `t>8` row at the third digit; the dictionary lever
moves the score more than anything measured here.

### Reviewer corrections to the 27-key chain block above

Review: `agent_guidance/paper-v/overnight/review-27key-chain-20260916.md`
(artifacts under `output/local_runs/paper_v_review_27key_chain_20260916/`).
The chain's own numbers reproduce digit for digit: the six segments tile
with no gap, overlap or discarded sample; the exact reference is bitwise the
record's; the energy is held to `1.78e-15`; no exact information enters the
propagation; 85 projective tests and the committed cheap-route tests pass.
Three corrections.

1. "Best on record at every horizon it reached" is withdrawn at `t=20` and
   `t=16`.  The route's own 27-key `post_pulse` chain (constraint off during
   the drive, then the energy held to its drive-end value; segments 2 to 4
   on disk from the same queue, unscored until the review) reads
   `6.990927e-2` over `t=0` to `20` against the `drive_power` chain's
   `8.519313e-2`, at `3,886.9 s` against `4,794.4 s`, and ties it at `t=16`
   (`6.385168e-2` against `6.377276e-2`); its Eq. (22) internal-energy RMS
   is 26 times worse (`1.648e-1` against `6.345e-3`) and its cones further
   violated (retained `-3.12e-3`, extension `-3.43e-4`).  Restated: the
   `drive_power` chain is best on the 31-coordinate RMS at `t=4`, `6`, `8`
   and `12`, tied at `16`, second at `20`; the best other rows are `6.0758e-2`
   at `t=12` and `6.3852e-2` at `t=16` (both the 27-key `post_pulse` chain)
   and `6.9909e-2` at `t=20` (the same), then the both-cones `1.2861e-1`.
2. The site-0 occupation column crossed conventions by a factor 2: the
   record's both-cones `0.204690` and plain archive EOM `0.286170` are the
   progression scorer's `2 rho[0,0]` trapezoidal time-RMS, while the chain's
   `1.125899e-1` is `rho[0,0]`; doubled it is `0.2252`, worse than the
   both-cones line.  The same crossing sits in the cheap-route report's and
   worklog blocks' site-0 columns wherever a record value is quoted.
3. `4,794.4 s` against `5,832.34 s` is two machines; the wall ratio is
   unmeasured.  Smaller: the 15-key unconstrained `t=0` to `4` wall is
   `89.6 s`, not `93.1 s`; peaks are MiB; "sees all 29 equally" means within
   a factor `1.41`.

Open decision, the user's: whether the 27-key `post_pulse` chain becomes the
`t=20` line on record for this route (best 31-coordinate RMS, `0.0699`, with
its internal-energy `0.165` and cone figures beside it) or the `drive_power`
chain does (`0.0852`, internal energy `0.0063`).  Both are subject to the
sensitivity study in progress before any third-digit reading.

### Cluster 10554184: terminated after 14 s, exit 2, no output

Read on 2026-09-16 once the SSH master was reopened.  The execute node
bootstrapped the conic stack, then `python -m
paper5.stability.apcm_carried_witness_analysis` failed at import:
`paper5/__init__.py` imports `paper5.repo`, whose `find_repo_root()` raises
`RuntimeError("Could not locate parent repo root with src/quantum.")` because
the bundle ships `paper_5/src` and the lane without the parent repository's
`src/quantum`.  The 2026-08-18 lane's README statement that the import chain
is self-contained predates `paper5/repo.py`; the local rehearsals passed
because the parent repository exists on this machine.  The run id
`accel_s01_t0000_t0500` is consumed; the resubmission carries a new one.
Job log: `logs/paper_v_cwrmf_event_guard_accel_20260915_v1.10554184.0.log`
and `.err` under the remote root.

## 2026-09-16: cheap projection route, sixth pass: the sensitivity of the recorded `N=16` drive-power row to one declared setting

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(sixth heading).  Configuration of every row is the recorded
`paper_v_cheap_projection_energy_power_n16_t20_20260915_v1` (fourth pass):
strong coupling, cutoff 16, exact-correlated preparation, 15 entrance keys,
`h=0.0025`, the unguarded fixed-dictionary predictor, Dykstra 30 on both
targets at stage cadence, the declared proximal log-det completion re-solved
every 16 steps warm-started on the carried frontier, the energy-neutral
option in `drive_power` mode, 8,000 steps to `t=20`, `719.6 s`.  Scores are
the all-coordinate scalar RMS on the 31 closed coordinates against the
exact cutoff-16 driven reference on the stated horizon; internal energy is
the sample RMS of the time-independent energy of Eq. (22) on that horizon;
a separation is the Euclidean distance between two rollouts' 31 closed
coordinates at the same accepted step.  One declared setting added to the
owned modules, `--completion-conic-tolerance` / `reselection_extension`, a
second extension on the same dictionary for the re-selection solves alone;
one test (35 in the file); no default changed; nothing committed.

Negative result first: the preparation's completion is not certified at
`conic_tolerance = 1e-10` (Clarabel `AlmostSolved`, `relative_lambda
2.0e-7`), so the tighter decade could be applied to the re-selection solve
only, and its first act at `t=0.04` is to refuse, on identical inputs, the
lift the `1e-9` solve certifies.  And no horizon of this configuration is
resolved at the second digit.  Three reruns, each one setting away from the
recorded run and each completed under `ram_guard --limit-mb 3000` one at a
time: `energy_tolerance` `1e-3` (`702.4 s`) scores `0.024240` / `0.044530`
/ `0.130946` / `0.325970` over `t=0` to `4` / `8` / `12` / `20`, internal
energy `0.020400` to `0.026136`; Dykstra 60 (`984.9 s`) `0.026837` /
`0.040677` / `0.114863` / `0.260123`, energy `0.025140` to `0.031797`,
192 of 500 completions refused against 276, both cones better (`-5.9e-4`,
`-1.4e-2` against `-2.3e-3`, `-5.3e-2`); re-selection `conic_tolerance`
`1e-10` (`831.0 s`) `0.027953` / `0.040450` / `0.108195` / `0.300340`,
energy `0.028639` to `0.036590`, 267 refused at 1.25 times the mean solve.
Recorded: `0.023744` / `0.045571` / `0.125139` / `0.328674`, energy
`0.020406` to `0.026000`.  Spread of the four: 18, 13, 21, 26 percent of
the score at `T=4`, `8`, `12`, `20`, and 40 percent of the internal energy.
The fourth pass's "best row this route has produced" over `t=0` to `4`
(`0.023744` against the projective event guard's `0.026049`) stands on the
internal energy only: two of its three twins score worse than the guard.

Separation from the recorded run: first nonzero `5.0e-16` at `t=0.0075`
(tolerance), `2.7e-15` at `t=0.0525` (Dykstra 60), `9.1e-7` at `t=0.0425`
(tolerance `1e-10`); `1e-3` at `t=0.1475`, `0.145`, `0.125`; `1e-2` at
`0.7525`, `0.36`, `0.48`; `1e-1` at `7.01`, `4.0025`, `3.085`; maxima
`0.41`, `0.99`, `0.72` near `t=12` to `13`.  The amplifier is the lift
re-selection's completion: in every rerun the carried frontiers jump apart
by `0.33` to `0.40` in one re-selection step on inputs equal to `1e-14`
(or exactly), the 44-coordinate state follows one step later at `1.2e-3`
to `2.3e-3`, and every later one-step growth is at most `4.0`.  Two
mechanisms: a refusal flip (Dykstra 60 at step 32, the recorded solve
refused and the rerun's certified; tolerance `1e-10` at step 16, the
`1e-9` solve certified and the `1e-10` solve refused), which is the fifth
pass's named candidate; and two certified solves returning lifts `0.40`
apart on inputs equal to `1e-14` (tolerance flag at step 16).

The exact reference is inert beside this.  The record's helper
`exact_reference_sensitivity` at its smallest amplitude `5e-6` (the
matched sensitivity study's `drive` and `relative_phonon_position`
perturbations) on the same 8,001 sample times: 31-coordinate distance
`1.19e-5` and `1.28e-5` at `t=0`, maximum over initial `1.000` and
`1.163` over `t=0` to `20`, never `1e-3`; at amplitude `5e-4`, the size of
the route's first crossing, `1.19e-3` and `1.28e-3` at `t=0`, maximum
`1.49e-3`, never `1e-2`.  Two decades in three to seven time units are the
closure's, none the physics'.

What is meaningful at the second digit for this configuration: nothing, at
any horizon.  At the first digit, a comparison is resolved where the other
row is more than about 1.3 times away: over `t=0` to `4` the 27-key chain
(`0.011427`), chain C (`0.0476697`) and the raw archive EOM (`0.11090`),
not the projective event guard (`0.026049`) or the `N=4` post-pulse chain
(`0.0262899`); over `t=0` to `20` both cones (`0.128607`), the 27-key chain
(`0.085193`), the third pass's `N=16` (`0.579332`) and at the edge the plain
archive EOM (`0.193205`), not the `N=16` post-pulse row (`0.239172`) or any
`N=4` row (`0.259573` to `0.318110`); every ordering among the 15-key `t=20`
rows of the fifth pass is inside one setting's spread.

Artifacts under `output/local_runs/`:
`paper_v_cheap_projection_energy_power_n16_t20_{etol1e3,dykstra60,conic1e10}_20260916_v1`,
and the scripts and their outputs in
`paper_v_cheap_projection_task6_sensitivity_20260916_v1/` (`run_chain.zsh`,
`score_sensitivity.py`, `separation_jumps.py`, `refusal_flip.py`,
`exact_sensitivity.py`, their `.json`, the run logs).  Longest rerun
`984.9 s`, largest own peak `803 MiB`.  Gates: 35 tests in
`test_cheap_projection_route.py`; full suite 480 passed in `264.86 s` with one untracked foreign test module (`test_paper5_import_without_parent_repo.py`, collection `ImportError`, not this route's) ignored.  Next:
`--record-stages` once on this configuration so the flips are counted, not
inferred; a re-selection rule that cannot flip on a `1e-14` input change;
the same three perturbations on the 27-key chain and the both-cones run.

### Cluster 10737278: retry of segment 1 after the import fix

Submitted 2026-09-16T17:40:11Z as run id `accel_s01_t0000_t0500_r01` from
the regenerated retry submit file; bundle `c717b8ca` (`519,591` bytes, 108
entries) built by `build_bundle.py` from `git archive` of `1023ef39` plus
the lane, and passed the new parentless rehearsal gate
(`BUNDLE_REHEARSAL_OK`); on the submit host `import paper5` succeeds with no
parent repository (`locate_repo_root()` is `None`); lifecycle validator exit
0 on the nine submit files; dry run 1 job; same resources and physical
protocol as cluster 10554184.  Receipt:
`chtc/paper_v_cwrmf_event_guard_accel_20260915/submission_receipt_10737278.json`.

## 2026-09-16: projective route, task 5: sensitivity of the 27-key `N=16` drive-power chain to the completion's conic tolerance

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`
(heading "projective-route, task 5 (2026-09-16)").  Configuration: the
task-4 27-key `N=16` `drive_power` energy-neutral chain (56 state
coordinates, 36 extension rows, ladder preparation at rung 0, Dykstra 30 on
both targets at stage cadence, `h=0.0025`, strong coupling, cutoff 16,
`energy_tolerance 0`, imports from `head_src` at `2d8f0efb`), rerun with
one setting changed: the re-selection completion
(`SymmetryReducedPositiveExtension.complete` every 16 accepted steps) solved
at `APCMExtensionSettings.conic_tolerance = 1e-10` in place of the committed
`1e-9`, through a driver-level subclass of `CheapProjectionRoute` whose
`reselect_lift` is the committed body with the solver swapped; preparation,
model extension (whose tolerance also enters the carried completion the
velocity accepts), repair, velocity and integrator unchanged.  Four chained
`ram_guard --limit-mb 4500` segments (`0` to `3.74`, `3.72` to `7.52`,
`7.52` to `8`, `8` to `12`; longest `1,001.3 s`; peak `646 MiB`), `2,114.8 s`
to `t=8` and `3,075.9 s` to `t=12`, 93 percent in the completion solves
(`9.51 s` mean).  Scores are the all-coordinate scalar RMS on the 31 closed
scalar coordinates against the exact cutoff-16 driven reference from `t=0`;
a separation is the Euclidean distance between the two chains' 31
coordinates at equal sample times.  No module edited; nothing committed; no
default changed.

Negative result first: the whole-extension form (preparation at `1e-10`
too) exhausts the declared completion retry ladder at `t=0` --- KKT backward
errors `6.209e-8` (base), `9.076e-8` (shift `1e-8`), `1.459` (shift `1e-7`),
`2.447` (weight `0.1`) against `50 tol = 5e-9`; at `1e-9` the base setting
refuses by the same check (`6.209e-8 > 5e-8`) and the `1e-8` rung certifies
at `4.270e-8`, a 15 percent margin; the scaled minimum eigenvalue is
`-1e-15` throughout, so the cone check never binds.  Gates: the patched
driver with no tolerance flag, and the subclass at `1e-9`, are each bitwise
the recorded segment 1 over `t=0` to `0.4` (161 samples, every array,
`0.0`), with the same 10-entry re-selection ledger.

Results.  (a) `t=0` to `4`: `1.292283e-2` against the recorded
`1.142716e-2` (`+13.1` percent); internal energy `3.687e-3` against
`4.597e-3`; phonon-occupation block `8.277e-3` against `9.836e-3`, the other
27 `1.348e-2` against `1.164e-2`.  `t=0` to `8`: `2.709352e-2` against
`2.593079e-2` (`+4.5`); internal energy `4.703e-3` against `5.752e-3`;
retained joint Gram `-2.201e-3` against `-2.152e-3`, extension `-1.980e-4`
against `-1.922e-4`; 200 completions re-solved, 97 refused against 75.
(b) `t=0` to `12`: `4.331967e-2` against `4.250925e-2` (`+1.9`; `t=10`:
`3.356926e-2` against `3.332970e-2`, `+0.7`); internal energy `4.995e-3`
against `6.088e-3`; retained `-2.374e-3` against `-2.463e-3`; 300 re-solved,
151 refused (recorded 107 by `t=11.84`).  The internal-energy difference is
the reference: the tightened chain's drive-power reference
`-3.565863978942048` sits `0.005534` above the exact `-3.571397982` where
the recorded `-3.564687762684483` sits `0.006710`, and the two chains'
energies differ by exactly `1.176e-3` at every sample after `t=4`.
Separation: identical at `t=0`; the first difference is the `t=0.2`
re-selection, certified by the recorded chain (conic certificate `Solved`,
reduced gradient `1.247e-8 <= 5e-8`) and refused by the tightened one
(`> 5e-9`), a `0.353` jump in the 70 standardized frontier coordinates;
`1e-4` at `t=0.2025`, `1e-3` at `t=0.5275`, `1e-2` at `t=1.7375`, `1e-1`
never through `t=12` (maximum `0.0937` at `t=11.7475`; 56-state distance
`1e-1` at `t=2.485`, maximum `0.337`); window means `0.0162`, `0.0346`,
`0.0532` over `(0,4]`, `(4,8]`, `(8,12]`, that is 21, 16 and 16 percent of
the chains' own distance from the exact reference at `t=4`, `8`, `12`.  Of
the 151 refusals, 69 are interior-Newton certificates over the KKT
threshold, 41 conic-Newton descent failures, 15 conic `Solved` certificates
over it, 18 infeasible reference completions, 7 empty facial reductions, 1
`AlmostSolved`; 27 of 234 interior-Newton solves have reduced gradients in
`(5e-9, 5e-8]`, the band the decade moves.

(c) Second digit, against the reviewer's corrected comparators: `t=4`, no
(`1.1e-2` to `1.3e-2`; cell A's `1.258e-2` lies between the two members, so
"best on record" fails there); `t=8`, one unit (`2.6e-2` to `2.7e-2`;
survives, the best other row `4.2542e-2` is `1.57` times the worse member);
`t=12`, yes (`4.3e-2`; survives, the 27-key `post_pulse` chain's
`6.0758e-2` is `1.40` times); `t=16` and `t=20`, unmeasured, and already
withdrawn by the review (tie with, then second to, the 27-key `post_pulse`
chain); the 18 percent gap between the two 27-key chains at `t=20` is the
size of the sensitivities measured (13 percent at `t=4` here, 26 percent at
`t=20` for the 15-key `N=16` chain), so that ordering is unresolved until
both are measured there.  The cumulative score's sensitivity shrinks with
horizon (13.1, 10.7, 4.5, 0.7, 1.9 percent at `t=4`, `6`, `8`, `10`, `12`)
while the pointwise separation is at its maximum at `t=11.75`; the former
is not evidence about the latter.

Artifacts under `output/local_runs/`:
`paper_v_dictionary_energy_power_27key_n16_tightresel_20260916f_seg{1,2,3,4}`
(each with `reselections.json`),
`paper_v_dictionary_energy_prefixgate_{tightdriver,reselroute}_20260916f`,
and in `paper_v_projective_dictionary_energy_20260915e/`: the drivers
(`run_energy_segment.py` with `--conic-tolerance`,
`--reselection-conic-tolerance`, `--record-reselections`; `run_chain.py`;
`compare_prefix.py`), the new `compare_chains.py`, `tabulate_task5.py`,
`probe_preparation_tolerance.{py,sh}`, `queue_tight.sh`,
`prefixgate_{tightdriver,reselroute}.sh`, and their outputs
(`preparation_tolerance_probe.json`, `prefix_gate_{tightdriver,reselroute}.json`,
`compare_tightresel_{t8,t12}.json`, `tightresel_chain_score_{t8,t12}.json`,
`power_chain_score_t10.json`, the logs).  Next: two more segments to carry
the tightened chain to `t=20`; the same lever on the 27-key `post_pulse`
chain; a decade looser (`1e-8`).

## 2026-09-16: projective route, task 6: the tightened 27-key `N=16` drive-power chain to `t=20`, and the `t=20` ordering of the two 27-key chains

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`
(heading "projective-route, task 6 (2026-09-16)").  The task-5 tightened
member (27-key `N=16` `drive_power` energy-neutral chain with the
re-selection completion solved at `conic_tolerance=1e-10` against the
committed `1e-9`, everything else identical, imports from `head_src` at
`2d8f0efb`) carried from `t=12` to `t=20` in two chained `ram_guard
--limit-mb 4500` segments (`12.0` to `16.6` on budget, 1,840 steps,
`1,004.4 s`, 115 completions re-solved / 50 refused; `16.6` to `20.0`
reached, 1,360 steps, `709.5 s`, 85 / 25; peaks `638` / `627 MiB`).  Chain
totals over `t=0` to `20`: `4,789.8 s` (recorded `4,794.4 s`), 500 re-solved
with 226 refused (recorded 144), `92.1` percent in the solves, energy held to
`1.78e-15` over 24,001 constrained repairs, reference `-3.565863978942047`
unchanged.  The recorded and the 27-key `post_pulse` chains were rescored
on the same seven horizons from the cached exact reference; 8,001 samples
on one grid.  No module edited; nothing committed; no default changed.

Negative result first: the `t=20` "best on record" ordering of the two
27-key chains is outside the sensitivity measured here, and the drive-power
chain is second under either setting --- members `8.519313e-2` (recorded)
and `9.088037e-2` (tightened) on the all-coordinate scalar RMS of the 31
closed scalar coordinates over `t=0` to `20`, a `6.7` percent band, against
the `post_pulse` chain's `6.990927e-2` (factors `1.22` and `1.30`).  The
measurement is one-sided: the `post_pulse` chain's own response to the lever
is unmeasured, and the branch's only same-horizon sensitivity (15-key `N=16`
drive-power, `26` percent at `t=20`) would cover the `22` percent gap.  At
`t=16` the members read `6.377276e-2` / `6.719628e-2` (`+5.4` percent) and
the `post_pulse` chain's `6.385168e-2` is inside the band: a tie.  The
cumulative sensitivity is not monotone (`+13.1`, `+10.7`, `+4.5`, `+0.7`,
`+1.9`, `+5.4`, `+6.7` percent at `t=4, 6, 8, 10, 12, 16, 20`); task 5's
"shrinking" reading is withdrawn past `t=12`.

Chart at `t=16` / `t=20`, recorded then tightened: internal-energy RMS
`6.250e-3` / `5.135e-3` and `6.345e-3` / `5.218e-3` (the `1.176e-3`
reference difference, as before); phonon-occupation block `3.188e-2` /
`2.844e-2` and `3.187e-2` / `2.882e-2` (the tightened member better there,
worse on the other 27: `7.117e-2` vs `6.722e-2`, `9.675e-2` vs `9.046e-2`);
site-0 `rho[0,0]` sample RMS `8.393e-2` / `8.742e-2` and `1.126e-1` /
`1.190e-1`, doubled to the record's `2 rho[0,0]` convention `0.168` / `0.175`
and `0.225` / `0.238` (both cones `0.205`); physicality minima over `t=0` to
`20` retained `-2.463e-3` / `-2.374e-3`, extension `-2.040e-4` /
`-2.098e-4`, attained before `t=12`; refusals cumulative `198/400` at `t=16`
and `226/500` at `t=20` (recorded `136/408` to `16.32`, `144/500`); 33 of 401
interior-Newton reduced gradients in `(5e-9, 5e-8]`.  Separation: the
31-coordinate distance between the chains is `0.0833` at `t=16` and `0.0771`
at `t=20`, maximum `0.0955` at `t=13.845`, never `0.1`; window means
`0.0532`, `0.0818`, `0.0733` over `(8,12]`, `(12,16]`, `(16,20]` while the
56-state distance grows (`0.279`, `0.289`, `0.398`; maximum `0.528` at
`t=19.94`); the members are `12.7` and `8.0` percent of their own distance
from exact apart at `t=16` and `t=20`.

Final ledger row of the chain (recorded / tightened, band width, comparator):
`t=4` `1.143e-2` / `1.292e-2`, 13.1 percent, cell A `1.258e-2` inside; `t=6`
`2.091e-2` / `2.314e-2`, 10.7, 27-key `post_pulse` `3.336e-2` (factor 1.44);
`t=8` `2.593e-2` / `2.709e-2`, 4.5, 15-key `post_pulse` `4.254e-2` (1.57);
`t=10` `3.333e-2` / `3.357e-2`, 0.7, 27-key `post_pulse` `4.956e-2` (1.48);
`t=12` `4.251e-2` / `4.332e-2`, 1.9, `6.076e-2` (1.40); `t=16` `6.377e-2` /
`6.720e-2`, 5.4, `6.385e-2` inside; `t=20` `8.519e-2` / `9.088e-2`, 6.7,
`6.991e-2` ahead by 1.22 to 1.30.  Plan-file row updated with this.

Artifacts under `output/local_runs/`:
`paper_v_dictionary_energy_power_27key_n16_tightresel_20260916f_seg{5,6}`
(each with `reselections.json`); in
`paper_v_projective_dictionary_energy_20260915e/`: `queue_tight_t20.sh`,
`tabulate_task6.py`, `score_tight_t20.sh`, `rescore_recorded_all.sh`,
`rescore_postpulse_all.sh`, `monitor_tight_t20.sh`, `comparators_task6.json`,
and the outputs `tightresel_chain_score_t20.json`,
`compare_tightresel_t20.json`, `power_chain_score_all.json`,
`postpulse_chain_score_all.json`, `tabulate_task6.log`, the segment logs.
Next: the same lever on the 27-key `post_pulse` chain, which is the one
measurement that settles the `t=20` ordering against both bands; a decade
looser (`1e-8`) on the drive-power chain.

## 2026-09-16: cheap projection route, seventh pass: a continuous lift update, and whether it resolves the second digit

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(seventh heading).  Configuration of every row is the recorded
`paper_v_cheap_projection_energy_power_n16_t20_20260915_v1` (fourth pass)
and its three one-setting twins of the sixth pass (`energy_tolerance`
`1e-3`; Dykstra 60; re-selection `conic_tolerance` `1e-10`): strong
coupling, cutoff 16, exact-correlated preparation, 15 entrance keys,
`h=0.0025`, the unguarded fixed-dictionary predictor, Dykstra 30 on both
targets at stage cadence, the declared proximal log-det completion
re-solved every 16 steps warm-started on the carried frontier, the
energy-neutral option in `drive_power` mode, 8,000 steps to `t=20`.
Scores, internal energy and separations are the sixth pass's; a lift is a
point of the 62 standardized frontier coordinates.  Added to the owned
modules: the declared lift re-selection rule `LiftReselectionRule`
(proximal anchor at the carried lift, trust radius with `carried` / `ball`
policy, deterministic acceptance margin on the completion's own objective,
`certified` / `measured` candidate scope), five driver flags, five
counters, the enlarged stage record; nine tests (44 in the file); no
default changed (a record-stages rerun of the recorded configuration
equals it in every array); nothing committed.

Negative result first: under the recorded rule the recorded `t=20` row is
a hybrid only to `t=11.76`.  Of its 500 re-selections 224 certify and 276
refuse, the last certificate at `t=11.76` and 234 of the refusals the
reference completion reporting the state's 29 lower moments
primal-infeasible (no positive 24-row extension exists), so the row's last
8.24 time units are the frozen-lift route with the Dykstra repair.  The
declared completion is proximal to the zero frontier and not to the
carried lift (its warm frontier is a starting point, not a term of the
objective), so every rule acts on the answer.  223 of the 224 certified
answers are full-face minimizers a median `0.012` (90th percentile
`0.037`) from the carried lift; the `t=0.04` answer on the 23-of-24 face
(move `0.334`) is the one certified answer whose declared objective is
above the carried lift's.  Screens over `t=0` to `2` on the four twins
(20 rollouts, one at a time): the margin `1e-3` alone declines that answer
in every run and cuts the spread from 29.1 to 11.2 percent at a cost,
`0.0101` to `0.0113` against the recorded base's `0.0079`; the trust
radius `0.05` is inert; anchor `mu = 1` gives 4.5 percent at `0.0089` to
`0.0093`; `measured` candidates remove the refusal flip (0.5 percent,
`conic1e10` identical to its base); `measured` with anchor `mu = 1` gives
0.4 percent at `0.0093`.

That rule to `t=20` on the four (walls `843.7`, `822.6`, `998.7`,
`908.5 s`, one at a time under `ram_guard --limit-mb 3000`): scores base /
`etol1e3` / `dykstra60` / `conic1e10` `0.031566` / `0.031582` / `0.031594`
/ `0.031566` over `t=0` to `4`, `0.075551` / `0.073518` / `0.076622` /
`0.076060` to `8`, `0.143618` / `0.120502` / `0.130712` / `0.127896` to
`12`, `0.264357` / `0.256831` / `0.273564` / `0.254707` to `20`; spreads
0.1, 4.2, 19.2, 7.4 percent against the recorded rule's 17.7, 12.7, 21.0,
26.4; internal energy `0.0237`, `0.0291`, `0.0307`, `0.0320` on the four
horizons (three runs to five digits; `etol1e3` `3e-4` above); scaled
extension minima `-3.2e-4` to `-5.8e-4` and retained joint Gram minima
`-7.6e-3` to `-1.4e-2` over `t=0` to `20` (recorded rule `-2.3e-3` to
`-5.9e-4`, `-5.3e-2` to `-1.4e-2`).  Separation from the base crosses
`1e-3` / `1e-2` / `1e-1` at `1.7875` / `4.1` / `8.415` (`etol1e3`),
`2.305` / `5.65` / `8.6025` (`dykstra60`), `6.4525` / `6.8` / `8.545`
(`conic1e10`), against the recorded rule's `0.125` to `0.1475` / `0.36` to
`0.7525` / `3.085` to `7.01`.  The second digit is resolved over `t=0` to
`4` (at the third digit; cost 33 percent, `0.0316` against the recorded
`0.0237`, 21 percent above the projective event guard) and to within three
units over `t=0` to `8` (cost 66 percent, `0.0756` against `0.0456`); not
at `t=12` (19.2 percent) nor at `t=20`, where the 7.4 percent of the
cumulative RMS hides twins `1.37` to `1.65` from the base in the 31
coordinates, though the score is 20 percent below the recorded row
(`0.264` against `0.329`) with the internal energy 23 percent worse and
re-selection continuing to the end (94 of 125 accepted in `t=15` to `20`
against none).  The ingredients' costs were not separated.

Artifacts under `output/local_runs/`:
`paper_v_cheap_projection_energy_power_n16_t20_recorded_stages_20260916_v1`,
the 24 `paper_v_cheap_projection_task7_<rule>_<twin>_t2_20260916_v1`
screens, the four
`paper_v_cheap_projection_task7_measured_anchor1_<twin>_t20_20260916_v1`,
and the scripts and outputs in
`paper_v_cheap_projection_task7_continuity_20260916_v1/` (`run_one.zsh`,
`screen.zsh`, `screens_all.zsh`, `screen_scores.py`,
`reselection_stats.py`, their `.json`, the run logs, `chain.log`, the
suite logs).  Longest rollout `998.7 s`, largest own peak `874 MiB`.
Gates: 44 tests in `test_cheap_projection_route.py`; full suite
499 passed in `257.04 s` under `ram_guard --limit-mb 5000`,
peak `2,033 MB`.  Next: separate the cost (`measured` without the
anchor, the anchor without the margin, four twins each to `t=8`); the
anchor weight `0.5` and `2`; the chosen rule on the 27-key `N=16`
drive-power chain.

## 2026-09-16: projective route, task 7: the same lever on the 27-key `N=16` post-pulse chain to `t=20`, and the `t=20` ordering against both bands

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`
(heading "projective-route, task 7 (2026-09-16)").  The lever of tasks 5
and 6 (re-selection completion solved at `conic_tolerance=1e-10` against
the committed `1e-9`, everything else identical, imports from `head_src` at
`2d8f0efb`) on the 27-key `N=16` `post_pulse` energy-neutral chain from
`t=0` to `20` in five chained `ram_guard --limit-mb 4500` segments (`0` to
`4` in mode `none`, `708.0 s`, 100 completions re-solved / 30 refused; `4`
to `8` in `post_pulse` mode, `721.8 s`, 100 / 35; `8` to `12.76` on budget,
`1,002.2 s`, 119 / 60; `12.76` to `18.115` on budget, resume `18.08`,
`1,000.0 s`, 133 / 37; `18.08` to `20` reached, `336.0 s`, 48 / 8; peaks
`649` to `717 MiB`).  The recorded driven interval could not be reused as
task 4 reused it: the lever enters its 100 re-selections, so it was rerun
under the lever, and the gate that the driver in that mode at `1e-9` is
bitwise the task-3 rollout over `t=0` to `0.4` (161 samples, `0.0`, seven
arrays) is what makes the rerun differ from the record by the lever alone.
Chain totals: `3,768.0 s` (recorded `3,887.0 s`), 500 re-solved with 170
refused (recorded 77), `92.5` percent in the solves, energy held to
`2.22e-15` over 19,130 constrained repairs, reference `-3.36598170532346`
unchanged to `9e-16`.  All four 27-key members scored on one grid of 8,001
samples against the cached exact reference.  No module edited; nothing
committed; no default changed.

Result first: the `t=20` ordering of the two 27-key chains is settled
against both measured bands --- the `post_pulse` chain scores
`6.990927e-2` / `7.563847e-2` (recorded / tightened, band 8.2 percent) on
the all-coordinate scalar RMS of the 31 closed scalar coordinates over
`t=0` to `20`, the `drive_power` chain `8.519313e-2` / `9.088037e-2` (6.7
percent); the bands are disjoint by 12.6 percent and the `post_pulse` chain
leads by 1.13 (closest members) to 1.30 (farthest) under every pairing.
The gap exceeds both band widths at `t=20` but not the `post_pulse` chain's
own response to the lever at `t=4` (20.9 percent) and `t=12` (24.1), so the
statement is on the four members, not a bound.  On the Eq. (22) internal
energy the ordering is reversed under both settings (`post_pulse`
`1.648e-1` / `1.876e-1` against `drive_power` `6.345e-3` / `5.218e-3`: 26 to
36 times), and on physicality neither chain holds a cone under either
setting (retained joint Gram minima `-3.124e-3` / `-2.993e-3` against
`-2.463e-3` / `-2.374e-3`; scaled extension `-3.429e-4` / `-3.295e-4`
against `-2.040e-4` / `-2.098e-4`).

The lever's effect on the `post_pulse` chain has no sign: tightened minus
recorded over recorded `+20.9`, `+4.0`, `-8.3`, `-10.7`, `-24.1`, `-15.6`,
`+8.2` percent at `t=4, 6, 8, 10, 12, 16, 20` (the `drive_power` chain:
`+13.1`, `+10.7`, `+4.5`, `+0.7`, `+1.9`, `+5.4`, `+6.7`); the `post_pulse`
band is the wider at every horizon but `t=6`.  Band table (recorded /
tightened; the `drive_power` band as comparator): `t=4` `1.631e-2` /
`1.972e-2`, `drive_power` ahead 1.26 to 1.73 (disjoint by 26.2 percent;
cell A `1.258e-2` ahead of the better member by 1.30); `t=8` `4.538e-2` /
`4.159e-2`, `drive_power` ahead 1.53 to 1.75, and the 15-key `post_pulse`
row `4.254e-2` lies inside the band (task 4's "worse at 27 keys" at `t=8`
is a tie); `t=12` `6.076e-2` / `4.612e-2`, `drive_power` ahead 1.06 to
1.43, disjoint by only 6.5 percent against a 24.1 percent band; `t=16`
`6.385e-2` / `5.386e-2`, the bands overlap on `6.377e-2` to `6.385e-2`: no
ordering; `t=20` as above.  The `t=0` to `4` row is task 3's 27-key
unconstrained row, so its band `1.63e-2` to `1.97e-2` is that number's
sensitivity; both members stay below the 15-key unconstrained `2.684e-2`
and the projective event guard's `2.605e-2`.

The internal-energy penalty of the tightened member (14 to 20 percent at
every horizon) is its reference: the `post_pulse` reference is the
trajectory's own Eq. (22) energy at `t=4`, `-3.3905733452008326` (recorded,
`0.180825` above exact) against `-3.3659817053234624` (tightened,
`0.205416` above), and the difference `2.459e-2` is `|E_tight - E_rec|` at
every constrained sample; the `drive_power` members' references differ by
`1.176e-3`.  Separation: identical through `t=0.0775`; the first difference
is the `t=0.08` re-selection, certified at the committed setting (reduced
gradient `9.016e-9` after 18 interior-Newton iterations; the gate's ledger
stands in for the recorded rollout, which recorded none) and refused at the
tightened one (`9.327e-9` at the 40-iteration cap), a `0.344` frontier
jump; the 31-coordinate distance reaches `1e-2` at `t=1.54`, `0.1` at
`5.39`, `0.2` at `7.09`, maximum `0.486` at `11.695` (window means
`0.0232`, `0.1411`, `0.3128`, `0.3452`, `0.3312` over `(0,4]` to
`(16,20]`), and at `t=12` is 147 percent of the better member's own
distance from exact; the 56-state distance reaches 1 at `t=7.82`, maximum
`2.71`.  Five times the `drive_power` pair's maximum `0.0955`.  Refusals of
the tightened chain, 170 of 500: 96 interior-Newton certificates over the
KKT threshold, 45 conic-Newton descent failures, 19 conic `Solved`
certificates over it, 8 infeasible reference completions, 2 `AlmostSolved`;
15 of 447 reduced gradients in `(5e-9, 5e-8]`; by window `15/50`, `15/50`,
`19/49`, `16/51`, `24/50`, `25/50`, `21/50`, `13/50`, `14/50`, `8/50`.
Site-0 in the record's `2 rho[0,0]` convention at `t=20`: `0.125` /
`0.187` against both cones `0.205` and `drive_power` `0.225` / `0.238`.
Plan-file row updated with this.

Artifacts under `output/local_runs/`:
`paper_v_dictionary_energy_postpulse_27key_n16_tightresel_20260916g_seg{0,1,2,3,4}`
(each with `reselections.json`),
`paper_v_dictionary_energy_prefixgate_postpulse_reselroute_20260916g`; in
`paper_v_projective_dictionary_energy_20260915e/`:
`queue_tight_postpulse.sh`, `prefixgate_postpulse_reselroute.sh`,
`score_tight_postpulse.sh`, `monitor_tight_postpulse.sh`,
`tabulate_task7.py`, and the outputs
`tightresel_postpulse_chain_score_t20.json`,
`compare_tightresel_postpulse_t20.json`,
`compare_tightresel_postpulse_driven.json`,
`prefix_gate_postpulse_reselroute.json`, `tabulate_task7.log`,
`tightresel_postpulse_chain.log`, the segment and gate logs.  Next: a
decade looser (`1e-8`) on both 27-key chains, a third member in every band;
the cheap route's continuous re-selection rule, which removes the
amplifier.

## 2026-09-16: cheap projection route, eighth pass: the continuous rule's continuity is the measured acceptance's and its cost is the anchor's

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(eighth heading, "Task 8").  Eleven rollouts of the recorded `N=16`
drive-power configuration (fourth pass: strong coupling, cutoff 16,
exact-correlated preparation, 15 entrance keys, `h=0.0025`, the unguarded
fixed-dictionary predictor, Dykstra 30 on both targets at stage cadence,
the declared proximal log-det completion re-solved every 16 steps
warm-started on the carried frontier, the energy-neutral option in
`drive_power` mode) to `t=8` (3,200 steps), one at a time under `ram_guard
--limit-mb 3000`, walls `244.7` to `384.7 s`, peaks `410` to `486 MiB`:
set A, the seventh pass's rule without its anchor (margin `1e-3`,
`measured` candidates) on the base and the sixth pass's three one-setting
twins (`energy_tolerance` `1e-3`; Dykstra 60; re-selection
`conic_tolerance` `1e-10`); set B, the anchor `mu = 1` on certified
candidates with no margin, on the four; the seventh pass's rule with anchor
`0.1` and `0.01`, base only; margin `1e-3` with anchor `1` on certified
candidates, base only.  Scores, internal energy, separations and spreads
are the sixth and seventh passes'; the physicality minima are over `t <= T`
(the joint Gram recomputed per sample from the states), so the `t=0` to
`20` rows of the sixth and seventh passes are restricted to the same
horizons.  No module edited, no new option (every rule is a setting of
`LiftReselectionRule`), no default changed, nothing committed; set A's
four runs reproduce the seventh pass's `t=2` screens with the same flags
bitwise over the 801 common samples.

Negative result first: the anchor pays the accuracy, and the seventh
pass's "the anchor buys accuracy back" holds only to `t=2`.  Set A scores
`0.027191` / `0.027215` / `0.027214` / `0.027191` (base / `etol1e3` /
`dykstra60` / `conic1e10`) over `t=0` to `4` and `0.038538` / `0.038617` /
`0.038605` / `0.038538` to `8`, spreads 0.1 and 0.2 percent (recorded rule
17.7, 12.7; seventh pass's rule 0.1, 4.2), 15 percent above the recorded
`0.023744` at `t=4` (inside the recorded rule's band `0.0237` to `0.0280`,
4 percent above the projective event guard's `0.026049`) and 15 percent
below the recorded `0.045571` at `t=8` (below the band `0.0404` to
`0.0456`); the `conic1e10` twin is within `1.9e-8` of the base at every
sample, the other two part at `t=0.16` by `0.0064` in the frontier and
cross `1e-3` / `1e-2` at `2.1` / `5.8`, never `1e-1`; 196 of 200
re-selections accepted in every run (4 declined, the wrong-face answers of
`t=0.04` to `0.16`; none infeasible), largest move `0.039`.  Set B scores
`0.030650` / `0.027274` / `0.031232` / `0.035728` and `0.071711` /
`0.062545` / `0.067015` / `0.070631`, spreads 31.0 and 14.7 percent: the
twins part at `t=0.04` to `0.08` by `0.17` to `0.24` (the halved wrong-face
answer; `conic1e10` on its certificate at `t=0.2`), cross `1e-3` / `1e-2`
/ `1e-1` at `0.17` to `0.29` / `0.71` to `0.85` / `3.0` to `5.0`; 112 to
157 accepted, 43 to 88 refused, 1 to 17 of 50 accepted in `t=6` to `8`
(set A 50 of 50; recorded rule 40; seventh pass's rule 10 with 40
`infeasible`).  So the continuity is the measured acceptance's (the
objective test under `measured` scope) and the cost the anchor's: at `t=8`
the anchor alone costs 57 percent, the seventh pass's rule 66, the measured
acceptance alone gains 15; at `t=4` 29, 33 and 15.  Margin `1e-3` with the
anchor on certified candidates scores `0.032761` / `0.064289` with 50
refusals, so the margin is not the cost.  The anchored lift lags the answer
and the state leaves the set admitting a positive 24-row extension four
units earlier than under the whole answer; over `t <= 8` the recorded base
holds the tightest cones of every set (scaled extension `-8.6e-5`, joint
Gram `-2.1e-3`), set A `-4.8e-5` to `-1.1e-4` / `-1.4e-3` to `-3.3e-3`,
the anchored rules `-1.7e-4` to `-3.6e-4` / `-4.1e-3` to `-8.7e-3` -- the
seventh pass's "cones an order of magnitude better" was a `t=0` to `20`
statement against a recorded row frozen-lift past `t=11.76`.  A weaker
anchor recovers the accuracy in proportion and never improves on none:
anchor `0.01` `0.027221` / `0.038647`, `0.1` `0.027426` / `0.040099`,
against `0` `0.027191` / `0.038538` and `1` `0.031566` / `0.075551`; the
`0.01` base sits `0.020` from the no-anchor base at `t=8` (one set-A
twin's size, `0.018` to `0.023`), the `0.1` base `0.21` (ten twins); the
twins' spread under a weak anchor was not measured.  Internal energy:
set A `0.0331` to `0.0333` / `0.0382` to `0.0384`, set B `0.0132` to
`0.0176` / `0.0170` to `0.0221`, recorded `0.0204` / `0.0241`; past `t=4`
the column is the declared drive-power reference minus the exact Eq. (22)
energy (`-3.571398`, constant to `3e-6` over `t=4` to `8`): `0.0272`
recorded, `0.0337` seventh pass's rule, `0.0426` set A, `0.0258` set B,
set by the drive-power integral along the run's own trajectory over the
pulse; the two ingredients trade the coordinate score against the energy
in opposite directions.  Recommendation recorded: drop the anchor (margin
`1e-3`, `measured` candidates, anchor `0`).

Artifacts under `output/local_runs/`:
`paper_v_cheap_projection_task8_{measured,anchor1}_{base,etol1e3,dykstra60,conic1e10}_t8_20260916_v1`,
`paper_v_cheap_projection_task8_measured_anchor0p1_base_t8_20260916_v1`,
`paper_v_cheap_projection_task8_measured_anchor0p01_base_t8_20260916_v1`,
`paper_v_cheap_projection_task8_margin_anchor1_base_t8_20260916_v1`, the
smoke `paper_v_cheap_projection_task8_smoke_anchor1_t005_20260916_v1`, and
the scripts and outputs in
`paper_v_cheap_projection_task8_attribution_20260916_v1/` (`run_one.zsh`,
`chain.zsh`, `task8_scores.py`, `task8_extras.py`,
`energy_reference_check.py`, `weak_anchor_separation.py`, their `.json`
and `.md`, the run logs, `chain.log`, the suite logs).  Gates: 44 tests in
`test_cheap_projection_route.py` (`24.82 s`); full suite 499 passed in
`270.42 s` under `ram_guard --limit-mb 5000`, peak `1,984 MB`.  Next: the
rule without the anchor to `t=20` on the four; the same on the 27-key
`N=16` drive-power chain; if the Eq. (22) energy is to be scored, the
drive-power reference's error over the pulse is the quantity to move.

## 2026-09-16: cheap projection route, ninth pass: the measured acceptance without the anchor to `t=20` -- continuity to `t=12`, the gain grows to `t=20`, and the large uncertified moves branch the twins after

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(ninth heading, "Task 9").  Four rollouts of the recorded `N=16`
drive-power configuration (fourth pass: strong coupling, cutoff 16,
exact-correlated preparation, 15 entrance keys, `h=0.0025`, the unguarded
fixed-dictionary predictor, Dykstra 30 on both targets at stage cadence,
the declared proximal log-det completion re-solved every 16 steps
warm-started on the carried frontier, the energy-neutral option in
`drive_power` mode) to `t=20` (8,000 steps) under the eighth pass's set-A
rule, margin `1e-3`, `measured` candidates, no anchor, on the base and the
sixth pass's three one-setting twins (`energy_tolerance` `1e-3`; Dykstra
60; re-selection `conic_tolerance` `1e-10`), one at a time under
`ram_guard --limit-mb 3000`: walls `900.2` / `851.5` / `841.1` / `848.7 s`
(budget `1,140 s`), peaks `782` to `784 MiB`, all reaching `t=20`; equal
to the eighth pass's set-A `t=8` runs in every array over the 3,201
common samples.  Scores, internal energy, separations, spreads and
physicality minima are the sixth to eighth passes' conventions, on the
horizons `t=0` to `4`, `8`, `12`, `16`, `20`; the recorded rule's four
runs and the seventh pass's combined rule's four are rescored on the same
five horizons by one script.  No module edited, no new option, no default
changed, 44 tests pass, nothing committed.

Negative result first: the continuity does not survive to `t=16` and
`t=20` in the `etol1e3` twin, and what breaks it is the rule's acceptance
of `0.4` to `0.8` frontier moves from `t=10` on.  Scores base / `etol1e3`
/ `dykstra60` / `conic1e10`: `0.027191` / `0.027215` / `0.027214` /
`0.027191` over `t=0` to `4`; `0.038538` / `0.038617` / `0.038605` /
`0.038538` to `8`; `0.098187` / `0.101367` / `0.100824` / `0.098186` to
`12`; `0.136367` / `0.161977` / `0.146397` / `0.136288` to `16`;
`0.150102` / `0.223420` / `0.160803` / `0.150285` to `20`; spreads 0.1,
0.2, 3.2, 18.8, 48.8 percent (the recorded rule 17.7, 12.7, 21.0, 28.1,
26.4; the combined rule 0.1, 4.2, 19.2, 21.6, 7.4), and without the
`etol1e3` twin 0.1, 0.2, 2.7, 7.4, 7.1.  Separation from the base crosses
`1e-3` / `1e-2` / `1e-1` at `2.1075` / `5.83` / `11.2275` (`etol1e3`; `2.27`
at `t=20`), `2.105` / `5.76` / `12.905` (`dykstra60`; `0.534`), `14.205`
/ `18.6425` / never (`conic1e10`; `0.089` at most), against the combined
rule's `1e-1` crossings at `8.4` to `8.6` and the recorded rule's `3.1` to
`7.0`.  To `t=10` no accepted move exceeds `0.16`; from `t=10.04` the base
accepts `0.716` (uncertified, a conic-Newton descent failure's last
iterate passing the measured feasibility test, objective improvement
`0.175`), then `0.686` certified at `10.2`, and 13 to 15 moves above
`0.1` in `[15, 20)` (`dykstra60` 5); the frontier envelope reaches `1.0`
at `t=10.04` (base, `conic1e10`), `10.2` (`dykstra60`), `18.96`
(`etol1e3`).  The branch is an answer branch between states `0.051` apart
at `t=10`, not the certificate: the `conic1e10` twin accepts the base's
`0.716` iterate to `0.0022` and stays within `0.09` to the end.  The
`infeasible` decisions begin at `t=8` to `10` (base 0 / 34 / 29 / 56 per
five-unit quarter; `dykstra60` 0 / 32 / 1 / 1, which accepts 124 of 125
in each of the last two quarters).

The gain survives and grows: from `t=8` on every one of the four runs is
below every run of the recorded and the combined rule, with the bands
disjoint at `8`, `12`, `16`, `20`; the base is 22 / 47 / 54 percent below
the recorded row at `t=12` / `16` / `20` (`0.0982` / `0.136` / `0.150`
against `0.125` / `0.258` / `0.329`), 42 percent below the recorded rule's
best twin at `t=20` (`0.260`), 41 percent below the combined rule's best
(`0.255`), 1.17 times the both-cones `0.128607`, and 37 percent below the
fourth pass's 15-key `post_pulse` row (`0.239`); the worst twin `0.223` is
below the best of the other eight runs.  Window RMS `(8, 12]` / `(12,
16]` / `(16, 20]`: base `0.161` / `0.213` / `0.196` against the recorded
`0.207` / `0.469` / `0.522` (frozen-lift past `t=11.76`).  Cost unchanged:
15 percent at `t=4` (`0.0272` against `0.0237`), and the Eq. (22)
internal energy `0.0331` / `0.0382` / `0.0397` / `0.0404` / `0.0409`
against the recorded `0.0204` / `0.0241` / `0.0252` / `0.0257` / `0.0260`
(the drive-power reference `0.0426` above exact from `t=4` on).
Physicality over `t <= 20`: scaled extension minimum `-2.6e-4` /
`-6.6e-4` / `-1.2e-4` / `-2.6e-4` (recorded `-2.3e-3` to `-5.9e-4`),
joint Gram `-7.9e-3` / `-1.6e-2` / `-3.7e-3` / `-7.9e-3` (recorded
`-5.3e-2` to `-1.4e-2`; both cones `+1.4e-5`); the frontier box exceeded
by at most `1.6e-4`; site-0 occupation RMS `0.158` to `0.201` (both cones
`0.205`).  Re-selections of 500: accepted (uncertified) / declined /
`infeasible`, solver refusals: 377 (39) / 4 / 119, 161; 385 (51) / 5 /
110, 163; 462 (35) / 4 / 34, 72; 380 (88) / 4 / 116, 208.  Ledger row
for the plan file's 15-key cheap-projection row (score, band of the four,
spread): `0.0272` (0.1 %), `0.0385` (0.2 %), `0.0982` (`0.0982` to
`0.1014`, 3.2 %), `0.136` (`0.136` to `0.162`, 18.8 %), `0.150` (`0.150`
to `0.223`, 48.8 %; three of four within 7.1 %) at `t=4`, `8`, `12`,
`16`, `20`; the plan file was not edited (shared with the projective
agent, which commits to it).

Artifacts under `output/local_runs/`:
`paper_v_cheap_projection_task9_measured_{base,etol1e3,dykstra60,conic1e10}_t20_20260916_v1`
(each with `repairs.jsonl`), the smoke
`paper_v_cheap_projection_task9_smoke_measured_t005_20260916_v1`, and the
scripts and outputs in
`paper_v_cheap_projection_task9_measured_t20_20260916_v1/` (`run_one.zsh`,
`chain.zsh`, `task9_scores.py`, `windows.py`, `reselection_moves.py`,
their `.json` and `.md`, the run logs, `chain.log`).  Gates: 44 tests in
`test_cheap_projection_route.py` (`24.03 s`); the full suite not re-run
(no module changed since the eighth pass's 499-test pass at the same
contents, `HEAD` `4de52486`; the projective agent's chain was running).
Next: the trust radius `0.1` (`ball` and `carried`) on the four to
`t=20`, the declared instrument for the `0.4` to `0.8` moves; then the
27-key `N=16` drive-power chain under the rule.

## 2026-09-17: cheap projection route, tenth pass: the trust radius `0.1` on the frontier move -- `ball` improves the base to `1.03` times the both-cones line and closes no band; `carried` is a freeze from `t=8.68`; the certified screen ties the base with two large moves

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(tenth heading, "Task 10").  Nine rollouts of the recorded `N=16`
drive-power configuration (fourth pass: strong coupling, cutoff 16,
exact-correlated preparation, 15 entrance keys, `h=0.0025`, the unguarded
fixed-dictionary predictor, Dykstra 30 on both targets at stage cadence,
the declared proximal log-det completion re-solved every 16 steps
warm-started on the carried frontier, the energy-neutral option in
`drive_power` mode) to `t=20` (8,000 steps), one at a time under
`ram_guard --limit-mb 3000` on 2026-09-16 (`19:07` to `21:12`): the ninth
pass's rule (margin `1e-3`, `measured` candidates, no anchor) with the
declared trust radius `0.1` on the Euclidean move of the standardized
frontier in one re-selection under the `ball` policy (the move scaled to
the radius on the segment toward the accepted lift) and the `carried`
policy (the carried lift kept when the cap binds), each on the base and
the sixth pass's three one-setting twins (`energy_tolerance` `1e-3`;
Dykstra 60; re-selection `conic_tolerance` `1e-10`); and, base only, the
certified screen (margin `1e-3`, `certified` candidates, no anchor, no
radius: an uncertified return refused outright).  Walls `761.6` to
`879.5 s` (budget `1,140 s`), peaks `778` to `872 MiB`, all at `t=20`.
The agent instance that launched the chain was lost to a network outage
after it completed; the scoring, the report section and this block were
written on 2026-09-17 from the artifacts, no rollout repeated.  Scores,
internal energy, separations, spreads and physicality minima are the
sixth to ninth passes' conventions on the horizons `t=0` to `4`, `8`,
`12`, `16`, `20`; a cap-bound re-selection is one whose candidate's move
exceeded the radius (`cap_bound_ball`, counted among the accepted since
the lift moves; `cap_bound_carried`, counted apart).  No module edited
(the radius and both policies are the seventh pass's `LiftReselectionRule`
fields at `a81b7ecb`; `HEAD` `3ea96121` throughout), no default changed,
44 tests pass before and after, nothing committed.

Negative result first: the radius closes neither the `t=16` nor the
`t=20` band below 10 percent under either policy.  `ball` scores base /
`etol1e3` / `dykstra60` / `conic1e10` `0.098867` / `0.104307` /
`0.103216` / `0.098867` over `t=0` to `12`, `0.132318` / `0.151238` /
`0.157814` / `0.132318` to `16`, `0.132542` / `0.163027` / `0.206990` /
`0.132542` to `20` (the `t=4` and `t=8` rows are the ninth pass's: the cap
first binds at `t=8.68` in every run, on the certified answer `0.125` to
`0.133`, and the eight runs equal the ninth pass's four in every array to
there): spreads 5.5, 19.3, 56.2 percent (ninth pass 3.2, 18.8, 48.8;
three of four within 4.4, 14.3, 23.0).  `carried` scores `0.135734` /
`0.145165` / `0.141612` / `0.135734`, `0.187509` / `0.206528` / `0.196694`
/ `0.187509`, `0.216450` / `0.251668` / `0.236386` / `0.216450`: spreads
6.9, 10.1, 16.3 percent, the narrowest past `t=12` on record, at 37 to 63
percent above the `ball` base, because from `t=8.68` the policy keeps the
carried lift at all but 7 of the remaining 283 re-selections (175
cap-bound, 128 of them on answers below `0.2`, minimum `0.105`, median
`0.183`; 116 `infeasible`): the frozen-lift route with the Dykstra repair
from `t=8.68`, cones `-5.3e-4` / `-1.2e-2` attained at `t=10.26` to
`10.30`, envelope at most `0.60`; not a candidate for the ledger row.
What the cap buys: the `ball` base's `0.132542` at `t=20` is 12 percent
below the ninth pass's base (`0.150102`), 60 percent below the recorded
row (`0.328674`) and `1.03` times the both-cones `0.128607`, with the
`(16, 20]` window `0.133` against `0.196`, the site-0 occupation RMS
`0.1245` against the both-cones `0.205`, the `conic1e10` twin identical
within `6.0e-5` at every sample (ninth pass `0.089`), the envelope never
at the box (`0.748`, `0.819`, `0.910`, `0.748`; the ninth pass's runs sat
at `1.0`), the same cones as the ninth pass's base (`-2.6e-4` / `-7.9e-3`,
attained at `t=8.37` before the first cap) and the internal energy
unchanged (`0.0331` / `0.0382` / `0.0397` / `0.0404` / `0.0409`, the
reference being set over the pulse).  On the base's trajectory the cap
never meets a `0.4` to `0.8` answer: its 72 answers above `0.4` are 69
`infeasible` and 3 declined by the margin (the wrong-face answers of
`t=0.08`, `0.16`, and the `0.894` uncertified iterate at `t=10.04` where
the ninth pass's base accepted `0.716`); the eight cap-bound re-selections
(per quarter 0, 2, 5, 1; `t=8.68` to `16.88`) are answers of `0.13` to
`0.25`, seven certified.  The two caps at `t=8.68` and `9.64` move the
state by enough that the `t=10.04` iterate no longer improves the
declared objective; that is the gain and the branch: the `etol1e3` twin's
trajectory admits and caps `0.670` and `0.511` at `t=10.08` and `10.12`
(28 caps, 8 uncertified), the `dykstra60` twin's `0.429` at `10.28` and
`0.567` at `16.44` (20 caps), after which `dykstra60` enters the
`infeasible` regime it avoided in the ninth pass (38 of 125 in `[15, 20)`
against 1) and scores `0.339` on `(16, 20]`, the outlier at `0.207`,
while `etol1e3` improves from `0.223` to `0.163`; separations from the
`ball` base cross `1e-1` at `11.78` (`etol1e3`; ninth pass `11.23`) and
`10.30` (`dykstra60`; `12.91`) and reach `0.626` and `1.41` at `t=20`
(`2.27`, `0.534`); which twin enters the regime is the trajectory's at
`t=10`, and no radius acts on it.  The `infeasible` regime begins in
`[5, 10)` at 33 to 34 of 125 in every measured-candidate run whatever the
rule.  The certified screen, base only: `0.027635` / `0.046342` /
`0.097956` / `0.123294` / `0.149326`; 65 refused (12 / 19 / 14 / 19 per
quarter, the first at `t=0.08`; its 14 answers above `0.4` all refused, in
runs of consecutive re-selections), 1 declined, no `infeasible`, 434
accepted with two moves above `0.1` (`0.290`, `0.316` at `t=11.12`,
`11.2`; ninth pass's base 21); it leaves the ninth pass's base at `t=0.24`
and costs 20 percent at `t=8` (`0.0463`; `(4, 8]` window `0.0594` against
`0.0472`), is the best run of the pass at `t=16` (`0.1233`) and ties the
ninth pass's base at `t=20` (13 percent above the `ball` base), with the
tightest cones past `t=8` (`-1.9e-4` / `-4.4e-3`), envelope `0.710`,
internal energy `0.0413` at `t=20` (reference `0.0430` above exact); its
band is unmeasured.  Ledger row under the `ball` rule (score, band of the
four, spread; three of four): `0.0272` (0.1 %), `0.0385` (0.2 %),
`0.0989` (`0.0989` to `0.1043`; 5.5 %), `0.132` (`0.132` to `0.158`;
19.3 %; 14.3), `0.133` (`0.133` to `0.207`; 56.2 %; 23.0) at `t=4`, `8`,
`12`, `16`, `20`, joint Gram `-7.9e-3` (`-5.4e-3` to `-7.9e-3`), extension
`-2.6e-4` (`-2.2e-4` to `-2.6e-4`), site-0 `0.1245` (`0.1245` to
`0.1705`), walls `762` to `847 s`: the second digit resolved to `t=12`
under either rule (all eight runs `0.0982` to `0.1043`), the first digit
past it (`0.13` to `0.16` at `t=16`, `0.13` to `0.21` at `t=20`; `0.13` to
`0.22` across the eight); the plan file not edited (shared; this route
does not commit); whether the row moves from the ninth pass's rule to
this one is an adoption call.

Artifacts under `output/local_runs/`:
`paper_v_cheap_projection_task10_{ball,carried}_{base,etol1e3,dykstra60,conic1e10}_t20_20260916_v1`,
`paper_v_cheap_projection_task10_certified_base_t20_20260916_v1` (each
with `repairs.jsonl`), the smoke
`paper_v_cheap_projection_task10_smoke_ball_t005_20260916_v1`, and the
scripts and outputs in
`paper_v_cheap_projection_task10_trust_t20_20260916_v1/` (`run_one.zsh`,
`chain.zsh`, `smoke_check.py`, `analyse.zsh`, `task10_scores.py`,
`windows.py`, `cap_events.py`, `cap_summary.py`, `large_answers.py`,
`test_gate.zsh`, their `.json`, `.md` and `.log`, the run logs,
`chain.log`).  Gates: 44 tests in `test_cheap_projection_route.py`
(`23.95 s` before the chain, `25.20 s` after the scoring); the full suite
not re-run (no module changed since the eighth pass's 499-test pass at
the same contents).  Next: the certified screen's three twins to `t=20`
(its band); the 27-key `N=16` drive-power chain under the ledger row's
rule; the `infeasible` regime against the Dykstra iteration count.

## 2026-09-17: projective route, task 8: the anchor-free measured-acceptance rule on the 27-key `N=16` drive-power chain, base and three twins to `t=8`

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`
(heading "projective-route, task 8 (2026-09-17)").  The cheap route's
task-8 rule --- `LiftReselectionRule(acceptance_margin=1e-3,
candidates="measured", anchor_weight=0)`, every field a setting of the
committed class at `a81b7ecb`, unchanged through `HEAD 3ea96121`; imports
from `head_src_e3374f6f`, identical to the tree --- on the task-4 27-key
`N=16` `drive_power` energy-neutral chain (56 state coordinates, 36
extension rows, ladder preparation at rung 0, Dykstra 30 on both targets
at stage cadence, `h=0.0025`, strong coupling, cutoff 16,
`energy_tolerance 0`), base and the three one-setting twins
(`energy_tolerance 1e-3`, Dykstra 60, re-selection `conic_tolerance
1e-10`), `t=0` to `8` in two chained `ram_guard --limit-mb 4500` segments
each (resumes at `t=3.84`, `3.56`, `3.48`, `3.72`; longest `1,006.8 s`;
peaks `625` to `651 MiB`; walls `1,886.3`, `1,901.9`, `1,962.8`,
`1,883.4 s` against the recorded chain's `2,058.0 s`, `88` to `92` percent
in the completion solves at `8.6` to `8.8 s` mean).  The eight segments
were run by the previous instance (killed by a network outage after
scoring them); this instance repeated none, fixed the ledger merge in
`compare_chains.py` (boundary-inclusive, as `tabulate_task5.py`), ran the
comparisons, the tabulation and a chaining gate, and wrote the record.
Scores are the all-coordinate scalar RMS on the 31 closed scalar
coordinates against the exact cutoff-16 driven reference from `t=0` on one
grid of 3,201 samples.  No module edited; nothing committed; no default
changed.

Negative results first.  (1) At `t=8` the rule is behind the recorded
chain by more than either band: the four members score `3.181485e-2` /
`3.179926e-2` / `3.224344e-2` / `3.179926e-2` (base / `etol1e3` /
`dykstra60` / `conic1e10`; band 1.40 percent) against the recorded chain's
`2.593079e-2` / `2.709352e-2` (task-5 band, 4.5 percent): behind by 1.17
to 1.24, bands disjoint by 17.4 percent; the loss is the
phonon-occupation block (`4.778e-2` against `3.283e-2`) and the A block
(`3.880e-2` against `1.994e-2`), B better and rho, C equal; on the window
`(4, 8]` alone `4.387e-2` against `3.485e-2`.  The rule that gained 15
percent at `t=8` at 15 keys loses 17 to 24 percent there at 27 keys, and
the 27-key `t=8` row on record stays the recorded chain's.  (2) Chaining
under the rule is not bitwise: the chain (`0` to `0.14`, resume at
`0.12`, then to `0.2`) agrees with a single rollout to `2.5e-14` through
three re-selections and flips the fourth (`t=0.16`) from `infeasible` (a
cone-failure iterate at scaled minimum eigenvalue `-1.76e-8`, below the
feasibility test's `-1e-8`) to `accepted` (another face's iterate at
`-9.78e-9`, a `0.8926` move, returned by `0.8923` at `t=0.2`; `d31`
`3.3e-4` at `t=0.2`); the members' bands are measured across four
different boundaries and include this perturbation.

Result: a sensitivity-resolved 27-key number exists at both horizons.
`t=4`: `1.001172e-2` / `1.001172e-2` / `9.959755e-3` / `1.001172e-2`, band
`9.96e-3` to `1.001e-2` (0.52 percent), every member below cell A's
`1.258000e-2` (cell A is 1.256 times the worst member) and below the
recorded chain's whole band `1.143e-2` to `1.292e-2` (ahead by 1.14 to
1.30, disjoint by 14.1 percent): the first 27-key `t=4` number that beats
cell A outside every measured band, where the recorded chain only tied
it.  Ahead of the combined-rule base (`1.891654e-2` / `3.417762e-2`,
anchor weight 1, superseded) by 1.89 and 1.06, of the 15-key row under
the same rule (`0.027191` / `0.038538`) by 2.7 and 1.2.  Internal energy
`3.922911e-3` / `4.781299e-3` (`dykstra60` `4.133866e-3` / `5.020505e-3`)
between the recorded and tightened members (references
`-3.5658908873520327`, `5.507e-3` above exact; recorded `6.710e-3`,
tightened `5.534e-3`).  Physicality over `t<=8`: retained joint Gram
`-1.301e-3` / `-1.300e-3` / `-5.361e-4` / `-1.300e-3` (recorded
`-2.152e-3`), scaled extension `-1.404e-4` / `-1.374e-4` / `-5.747e-5` /
`-1.374e-4` (recorded `-1.922e-4`); nothing held.  Decisions of 200:
accepted 188 (36 uncertified) / 188 (30) / 193 (35) / 188 (66),
`infeasible` 12 / 12 / 7 / 12, declined by the margin 0 in every member;
solver refusals 48 / 42 / 42 / 78 (recorded 75, tightened 97); every
member accepts the `t=0.04` whole-face answer (`0.3447` move, KKT
`1.07e-7`) that the earlier form refused, which is where the base and the
recorded chain part (`d70` `0.1` at `0.04`; `d31` `1e-2` at `1.84`, `0.1`
at `4.88`, `0.267` at `t=8`).

Separation: `etol1e3` and `conic1e10` are one trajectory with the base to
`3.2e-8` through `t=6.6` (and with each other to `5.1e-9` to `t=8`); at
the `t=6.6` re-selection the base's completion took the facial-reduction
path (`facial reduction removed every frontier direction`) and returned
an iterate `0.9565` away that passes the feasibility test and improves
the objective by `1.04`, applied, then `0.9428` back at `6.64`; the twins'
completions on states `1e-7` away took the interior-Newton path to its
cap (`0.0117`); cost `1.2e-3` in `d31` at most, `1.1e-3` at `t=8` (0.4
percent of the distance from exact), 0.05 percent of the score, base
envelope `0.772` against `0.387` --- the cheap route's task-9 event at
`t=6.6` and a `0.96` move, the trust radius's job.  `dykstra60` parts at
`t=0.16` (base `infeasible`, twin accepted `0.0011`), `d31` `1e-2` at
`3.835`, maximum `0.0349`, never `0.1`; score `-0.5` / `+1.3` percent.
The certificate tolerance moves nothing under this rule at 27 keys
either (the `conic1e10` twin accepts 66 uncertified iterates and stays on
the base).

Gates: driver without the flags bitwise the recorded chain over `t=0` to
`0.4` (161 samples, `0.0`); native `reselection_extension` path bitwise
the task-5 subclass; boundary resumes at 96, 89, 87, 93 intervals; energy
held to `1.78e-15` over 9,426 to 9,550 constrained repairs; one grid.
Artifacts under `output/local_runs/`:
`paper_v_dictionary_energy_power_27key_n16_measured_{base,etol1e3,dykstra60,conic1e10}_20260916i_seg{1,2}`
(each with `reselections.json`),
`paper_v_dictionary_energy_measured_chaingate{S,A,B}_20260917j`; in
`paper_v_projective_dictionary_energy_20260915e/`:
`queue_task8_measured.sh`, `score_task8m.sh`, `compare_task8m_extra.sh`,
`chaingate_task8m.sh`, `tabulate_task8.py`, `compare_chains.py`
(boundary-inclusive merge), `task8m_*_score_t8.json`,
`compare_task8m_*.json`, `chaingate_task8m_compare.json`,
`tabulate_task8.log`, the segment and gate logs.  Plan-file row updated.
Next: the trust radius (`0.1`, `ball` and `carried`) on the four to `t=8`,
then the base to `t=20` under the rule and the cap; and whether the `t=8`
loss against the earlier form is the `t=0.04` acceptance or the `(4, 8]`
re-selections.

## 2026-09-17: cheap projection route, eleventh pass: the certified-only rule's band, and the rule with the ball radius -- the certificate is the branch (132 percent at `t=20`, two twins frozen), the `conic1e10` twin is the recorded rule's twin bitwise, and the radius on the certified base moves `0.149` to `0.152`

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(eleventh heading, "Task 11").  Four rollouts of the recorded `N=16`
drive-power configuration (fourth pass: strong coupling, cutoff 16,
exact-correlated preparation, 15 entrance keys, `h=0.0025`, the unguarded
fixed-dictionary predictor, Dykstra 30 on both targets at stage cadence,
the declared proximal log-det completion re-solved every 16 steps
warm-started on the carried frontier, the energy-neutral option in
`drive_power` mode) to `t=20` (8,000 steps), one at a time under
`ram_guard --limit-mb 3000` on 2026-09-17 (`20:27` to `21:25`): the
certified-only rule (margin `1e-3`, `certified` candidates, no anchor, no
trust radius: an uncertified return refused outright, a certified answer
accepted when its declared-objective improvement exceeds the margin) on
the sixth pass's three one-setting twins (`energy_tolerance` `1e-3`;
Dykstra 60; re-selection `conic_tolerance` `1e-10`), completing the band
whose base is the tenth pass's certified screen; and the same rule with
the trust radius `0.1` under the `ball` policy, base only.  Walls `914.8`,
`869.8`, `840.7`, `802.4 s` (budget `1,140 s`), peaks `748` to `783 MiB`,
all at `t=20`.  Scores, internal energy, separations, spreads and
physicality minima are the sixth to tenth passes' conventions on the
horizons `t=0` to `4`, `8`, `12`, `16`, `20`.  No module edited (the
fields are the seventh pass's `LiftReselectionRule` at `a81b7ecb`; `HEAD`
`f971a284` throughout, the projective route's task-8 commit of `20:26`, no owned module changed), no default changed, 44 tests pass before and
after, nothing committed.

Negative result first: the certified-only rule has the widest band on
record and the certificate sets it.  Base / `etol1e3` / `dykstra60` /
`conic1e10`: `0.027635` / `0.028102` / `0.027735` / `0.027953` over `t=0`
to `4`, `0.046342` / `0.051402` / `0.058107` / `0.040450` to `8`,
`0.097956` / `0.095070` / `0.181008` / `0.108195` to `12`, `0.123294` /
`0.144537` / `0.290131` / `0.228471` to `16`, `0.149326` / `0.203554` /
`0.345743` / `0.300340` to `20`: spreads 1.7, 43.7, 90.4, 135.3, 131.5
percent (three of four 1.2, 25.4, 13.8, 85.3, 69.9; the ninth pass 0.1,
0.2, 3.2, 18.8, 48.8; the `ball` form 0.1, 0.2, 5.5, 19.3, 56.2; the
recorded rule 17.7, 12.7, 21.0, 28.1, 26.4).  The twins leave the base at
a refusal flip at `t=0.6` to `0.8` (a re-selection the base accepts at a
`0.008` move and the twin refuses, on states `5e-16` to `3e-5` apart;
`1e-3` crossed at `0.6375`, `0.81`, `0.6375`), cross `1e-1` at `5.175`,
`6.1375`, `3.9225` (ninth pass `11.2`, `12.9`, never; recorded `7.0`,
`4.0`, `3.1`) and end `0.772`, `1.63`, `1.60` from it.  Refusals of 500:
65 / 125 / 274 / 267 (per quarter `dykstra60` 13 / 29 / 106 / 125,
`conic1e10` 26 / 34 / 81 / 125, `etol1e3` 17 / 25 / 23 / 60); the
`dykstra60` twin's completion refuses 202 consecutive re-selections from
`t=11.96` (last accepted `t=11.92`) and the `conic1e10` twin's 159 from
`t=13.68`, so both are the frozen-lift route with the Dykstra repair over
their last 6 to 8 time units, with the recorded base's late windows
(`(12, 16]` `0.49`, `0.42`; `(16, 20]` `0.51`, `0.49`; recorded `0.47`,
`0.52`), joint Gram minima `-2.3e-2`, `-3.6e-2` attained at `t=19.8` to
`20`, and the `dykstra60` twin's frontier envelope above the box from
`t=18.78` (`1.018`).  The rule is the recorded rule plus the margin: the
`conic1e10` twin, whose margin never declines, is the sixth pass's
`conic1e10` twin bitwise (zero separation over 8,001 samples), and the
base, `etol1e3` and `dykstra60` runs leave the recorded rule's same twin
at `t=0.0425`, the sample after the `t=0.04` re-selection, where the
margin declines the certified wrong-face answer (frontier move `0.30` to
`0.33`, declared-objective improvement `-0.40` to `-0.46`) that the
recorded rule accepted (`etol1e3` declines two more, at `t=0.08`, `0.12`);
that one decline takes `t=20` from `0.329` to `0.149` (base), `0.326` to
`0.204` (`etol1e3`) and `0.260` to `0.346` (`dykstra60`), and the
recorded rule's band (`0.260` to `0.329`) sits inside this one (`0.149`
to `0.346`).  Internal energy at `t=20` `0.0413` / `0.0367` / `0.0368` /
`0.0366` (references `0.0430` / `0.0384` / `0.0385` / `0.0383` above the
exact post-pulse energy; the twins' trajectories are `0.05` to `0.10`
from the base's at `t=4`, when the reference is set); site-0 occupation
RMS `0.146` / `0.207` / `0.195` / `0.183`; physicality over `t <= 20`:
extension `-1.9e-4` / `-6.3e-4` / `-1.0e-3` / `-1.6e-3`, joint Gram
`-4.4e-3` / `-1.5e-2` / `-2.3e-2` / `-3.6e-2`, envelope `0.710` / `0.851`
/ `1.018` / `0.893`.

The certified rule with the `ball` radius, base only: `0.027635` /
`0.046342` / `0.097938` / `0.124542` / `0.151644`, equal to the certified
screen to `t=11.12` (the first cap: the certified `0.290` answer,
improvement `0.741`, scaled to `0.1`), within `6.7e-3` to `t=12` and
`0.107` to `t=16`, then 1.0 and 1.6 percent above it at `t=16` and `20`
(refusals in `[15, 20)` 35 against 19) and 14 percent above the `ball`
base (`0.132542`); seven caps, all on certified answers of `0.10` to
`0.29`, none on the `0.4` to `0.8` answers, which the certificate refuses
(33 above `0.4`, 20 of them in `[15, 20)`); envelope `0.616`, cones
`-2.1e-4` / `-5.0e-3`; its band unmeasured and, by the above, the
certificate's.  The tenth pass's item 1 is answered: the certified
screen's band past `t=12` is the recorded rule's certificate branch made
wider by the `t=0.04` decline, not a band below the ninth pass's, and the
measured acceptance stands.  Tightest bands at an accuracy below `0.21`
at `t=16` and `0.25` at `t=20`: the measured rule without the radius
(ninth pass), 18.8 percent at `0.136` to `0.162` and 48.8 at `0.150` to
`0.223` (three of four 7.4, 7.1); tighter only at worse accuracy
(`carried` 10.1 / 16.3 at `0.188` to `0.207` / `0.216` to `0.252`;
combined 7.4 at `0.255` to `0.274`); best base the `ball` form (`0.132`,
`0.133`; 19.3, 56.2).  Ledger row unchanged: the `ball` row, `0.0272`
(0.1 %), `0.0385` (0.2 %), `0.0989` (`0.0989` to `0.1043`), `0.132`
(`0.132` to `0.158`), `0.133` (`0.133` to `0.207`; three of four within
23 percent; `0.133` to `0.223` across the eight measured-candidate runs)
at `t=4`, `8`, `12`, `16`, `20`, the certified-only rule excluded on its
band; the sentence for the plan file's cheap-projection row is in the
report ("The 15-key ledger row"); the plan file not edited (shared; this
route does not commit).

Artifacts under `output/local_runs/`:
`paper_v_cheap_projection_task11_certified_{etol1e3,dykstra60,conic1e10}_t20_20260917_v1`,
`paper_v_cheap_projection_task11_certball_base_t20_20260917_v1` (each
with `repairs.jsonl`), the smoke
`paper_v_cheap_projection_task11_smoke_certball_t005_20260917_v1` (equal
to the certified screen over the 21 common samples), and the scripts and
outputs in `paper_v_cheap_projection_task11_certified_t20_20260917_v1/`
(`run_one.zsh`, `chain.zsh`, `smoke_check.py`, `smoke_check.zsh`,
`test_gate.zsh`, `dry_run.zsh`, `analyse.zsh`, `task11_scores.py`,
`windows.py`, `refusals.py`, the tenth pass's `cap_events.py` and
`large_answers.py` applied, their `.json`, `.md` and `.log`, the run
logs, `chain.log`).  Gates: 44 tests in `test_cheap_projection_route.py`
(`25.08 s` before the chain, `24.66 s` after); the full suite not re-run
(no module changed since the eighth pass's 499-test pass at the same
contents).  Next: the 27-key `N=16` drive-power chain under the `ball`
rule, base to `t=20`; the `infeasible` regime against the Dykstra
iteration count (60, 120) on the base; nothing further on the certified
screen.

## 2026-09-17: projective route, task 9: the ball trust radius `0.1` on the 27-key `N=16` drive-power chain, four members to `t=8`, then the base to `t=20`

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`
(heading "projective-route, task 9 (2026-09-17)").  The cheap route's
task-10 rule --- `LiftReselectionRule(acceptance_margin=1e-3,
candidates="measured", anchor_weight=0, trust_radius=0.1,
trust_policy="ball")`, every field a setting of the committed class;
imports from `head_src_e3374f6f`, identical to the tree at `HEAD
f971a284` outside `__pycache__` --- on the task-4 27-key `N=16`
`drive_power` energy-neutral chain (56 state coordinates, 36 extension
rows, ladder preparation at rung 0, Dykstra 30 on both targets at stage
cadence, `h=0.0025`, strong coupling, cutoff 16, `energy_tolerance 0`):
(a) base and the three one-setting twins (`energy_tolerance 1e-3`,
Dykstra 60, re-selection `conic_tolerance 1e-10`), `t=0` to `8` in two
chained `ram_guard --limit-mb 4500` segments each (resumes at `t=3.6`,
`3.6`, `3.88`, `3.72`; walls `1,977.2`, `1,882.8`, `1,835.1`, `1,858.7 s`;
peaks `627` to `636 MiB`); (b) the base chained from its `t=8` segment
to `t=20` in three more (`8.0` to `11.8`, `11.8` to `15.44`, `15.44` to
`20.0`; `4,844.7 s` in all, `0.924` in the 500 completion solves at
`8.95 s` mean).  Smoke gate first: `t=0` to `0.08` under the five flags
records the rule in `summary.json` and the `t=0.04` re-selection as
`cap_bound_ball` (solved `0.344669`, applied `0.1`).  Scores are the
all-coordinate scalar RMS on the 31 closed scalar coordinates against
the exact cutoff-16 driven reference from `t=0` on one grid
(`exact_t8.npz`, 3,201 samples; `exact_t20.npz`, 8,001).  No module
edited; nothing committed; no default changed.

Negative results first.  (1) At 27 keys the radius binds only inside
`(0, 0.64]` --- 3, 5, 5, 4 times of 200, the first at `t=0.04` in every
member (the whole-face `0.3447` answer task 8 applied whole, scaled to
`0.1`), then `0.16` (`0.244`), `0.20` (`0.145`) and, on the twins,
`0.56`/`0.60` (`0.285`, `0.115`), `0.60`/`0.64` (`0.881`, `0.103`),
`0.36` (`0.584`) --- and never in `(0.64, 8]`: every later answer above
`0.1` is `infeasible` (`0.53` to `0.55` at `0.36` to `0.64` and `6.44` to
`6.52`) or declined by the margin (`1.319` at `0.72`, `1.330` at `0.12`).
Task 8's `t=6.6` move (`0.9565`) is not capped because it does not occur:
the four capped trajectories' `t=6.6` completions return `0.0240` /
`0.0128` / `0.0105` / `0.0243`, all accepted; the ball chain left the
anchor-free base's trajectory at `t=0.04` (`d70` `0.2447`).  (2) The
`t=8` standing does not change: `3.180974e-2` / `3.151679e-2` /
`3.215530e-2` / `3.143476e-2` (band 2.29 percent; the anchor-free four
`3.180e-2` to `3.224e-2`, 1.40 percent, bases equal to `0.02` percent,
bands overlapping), behind the recorded chain's `2.593079e-2` to
`2.709352e-2` by 1.16 to 1.24, bands disjoint.  (3) At `t=4` the ball
costs 4 percent and widens the band: `1.043201e-2` / `1.003847e-2` /
`1.005305e-2` / `1.024635e-2` (band 3.92 percent) against the anchor-free
`9.96e-3` to `1.001e-2` (0.52 percent), disjoint and behind (base ratio
`1.042`), still ahead of the recorded band `1.143e-2` to `1.292e-2`
(worst member `0.913` of its bottom) and below cell A in every member
(`0.829`).  The cap at `t=0.04` puts each member on its own trajectory
from `t<=0.36` (`etol1e3` parts at `0.36`, `accepted` against
`infeasible`; `dykstra60` at `0.12`, `infeasible` against
`declined_by_margin`; `conic1e10` at `0.36`, where its `1e-10` completion
returned an uncertified `0.584` answer that passed the measured test and
was capped, so the certificate branch the anchor-free rule had removed
is back), `d31` at `t=8` `2.46e-2`, `3.91e-2`, `1.68e-2` (5.8 to 13.1
percent of the distance from exact; under task 8's rule `1.1e-3`,
`3.3e-2`, `1.1e-3`).  Internal energy `4.413844e-3` / `5.360816e-3` on
the base at `t=4` / `8` (anchor-free `3.922911e-3` / `4.781299e-3`; the
base's `drive_power` reference `-3.5652343570276206` lies `6.164e-3`
above exact where the anchor-free base's lay `5.507e-3`); physicality
over `t<=8` retained `-1.298e-3` / `-1.324e-3` / `-5.212e-4` /
`-1.316e-3`, extension `-1.369e-4` / `-1.384e-4` / `-5.279e-5` /
`-1.384e-4` (task 8's to two figures; recorded `-2.152e-3` /
`-1.922e-4`), envelope at most `0.421` (anchor-free base `0.772`);
solver refusals 47 / 47 / 47 / 80 (task 8 48 / 42 / 42 / 78).

Result, (b): the two rules tie at `t=8` on the base (`0.9998`; bands
overlap), so the ball --- the declared instrument for the moves past
`t=8` the cheap route measured, and the 15-key ledger row's rule --- was
chained to `t=20`, one run: `3.797419e-2` / `4.820897e-2` / `6.353308e-2`
at `t=12` / `16` / `20` against the recorded `drive_power` chain's
`4.250925e-2` / `6.377276e-2` / `8.519313e-2` (bands to `4.331967e-2`,
`6.719628e-2`, `9.088037e-2`; ahead by 1.12 to 1.14, 1.32 to 1.39, 1.34
to 1.43, outside every band), the `post_pulse` chain's `6.075829e-2` /
`6.385168e-2` / `6.990927e-2` (bands `4.611524e-2` to `6.075829e-2`,
`5.386180e-2` to `6.385168e-2`, `6.990927e-2` to `7.563847e-2`; ahead by
1.21 to 1.60, 1.12 to 1.32, 1.10 to 1.19) and the both-cones `0.128607`
(`0.494`); `t=10` `3.608325e-2`, behind the recorded `3.332970e-2` to
`3.356926e-2` by 1.07 to 1.08, so the crossing is in `(10, 12]`; per
window `(8,12]`, `(12,16]`, `(16,20]` `4.80e-2`, `7.05e-2`, `1.04e-1`
against the recorded `6.39e-2`, `1.04e-1`, `1.42e-1` and the
`post_pulse` `8.34e-2`, `7.23e-2`, `9.02e-2`.  The gain is in the rho
and B blocks (`1.006e-1` / `1.252e-1` against the recorded `1.358e-1` /
`1.873e-1` at `t=20`); the chain is behind the recorded one in the
phonon-occupation block at every horizon past `t=4` (`4.69e-2` against
`3.19e-2` at `t=20`), task 8's `(4, 8]` loss never recovered.  Internal
energy `5.641305e-3` / `5.776465e-3` / `5.856071e-3` (recorded
`6.088286e-3` / `6.249722e-3` / `6.344620e-3`, tightened `4.995294e-3` /
`5.135424e-3` / `5.217702e-3`, `post_pulse` `0.153` to `0.165`, both
cones `0.00094`); retained joint Gram minimum `-2.377e-3` (at `t=10.08`)
and extension `-2.202e-4` (at `9.545`) over `t<=12`, `16`, `20` alike
(recorded `-2.463e-3` / `-2.040e-4`); envelope `0.573`; nothing held;
energy to `1.78e-15` over 23,955 constrained repairs.  Decisions of 500:
accepted 441 (91 uncertified), cap-bound 14 (11 past `t=8`, at `9.08` to
`11.32` and `14.36`, answers `0.133` to `0.793`), declined 1,
`infeasible` 44; solver refusals 136 (recorded 144); of 38 answers
above `0.4`, 36 `infeasible`, 1 declined, 1 capped (`0.793` at `11.04`);
`(16, 20]` accepts all 100 with no answer above `0.0112`.

The 27-key ledger row: `t=4` task 8's anchor-free `1.00e-2` (band 0.5
percent; the ball does not displace it); `t=8` the recorded chain's
`2.59e-2` (band 4.5 percent; both rules behind); `t=12` / `16` / `20`
this round's ball base `3.80e-2` / `4.82e-2` / `6.35e-2` if adopted,
one run each, no band --- the leads to beat are 12, 12 and 10 percent
of the base (over the `drive_power`, `post_pulse`, `post_pulse` bands).
The three twins' continuations from their `t=8` segments to `t=20`
were queued at `23:32` (`queue_task9_t20_twins.sh`) to supply the band.

Gates: smoke (`cap_bound_ball` at `t=0.04`, the rule in `summary.json`);
the driver without the radius is task 8's (flags default to not given;
task 8's bitwise gates stand); same first decision in every member;
boundary resumes at 90, 90, 97, 93 intervals and, past `t=8`, at `8.0`,
`11.8`, `15.44`; one grid, every score from the caches.  Artifacts under
`output/local_runs/`: `paper_v_dictionary_energy_ball_smoke_20260917k`,
`paper_v_dictionary_energy_power_27key_n16_ball_{base,etol1e3,dykstra60,conic1e10}_20260917k_seg{1,2}`,
`..._ball_base_20260917k_seg{3,4,5}` (each with `reselections.json`); in
`paper_v_projective_dictionary_energy_20260915e/`:
`smoke_task9_ball.sh`, `check_task9_smoke.py`, `queue_task9_ball.sh`,
`score_task9.sh`, `tabulate_task9.py`, `queue_task9_t20.sh`,
`tabulate_task9b.py`, `tabulate_task9b.sh`, `queue_task9_t20_twins.sh`,
`task9_*_score_t8.json`, `task9_ball_base_score_t20.json`,
`compare_task9_*.json`, `tabulate_task9.log`, `tabulate_task9b_ball.log`,
the chain and segment logs.  Plan-file row updated.  Next: the twins'
`t=12`, `16`, `20` band; the anchor-free base from `t=8` to `t=20`
(`queue_task9_t20.sh measured`); the phonon-occupation loss of `(4, 8]`.

## 2026-09-18: projective route, task 9, the band: the three ball twins to `t=20`

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`
(subsection "(b) The band, landed 02:06 on 2026-09-18" under the task-9
heading; `tabulate_task9_band.py`, `tabulate_task9_band.log`).  The
`etol1e3`, `dykstra60` and `conic1e10` twins of the 27-key `N=16`
drive-power chain under the ball rule (margin `1e-3`, `measured`
candidates, anchor `0`, radius `0.1`, policy `ball`) were chained from
their own `t=8` segments to `t=20` (`queue_task9_t20_twins.sh`; three,
four and three `ram_guard --limit-mb 4500` segments of at most
`1,007.7 s`; walls `4,897.5`, `4,984.4`, `4,689.5 s` to `t=20`; peaks
`704`, `639`, `698 MiB`), one at a time, `23:32` to `02:06`, and scored
on `exact_t20.npz`.  No module edited; nothing committed; no default
changed.

Result: every member is ahead of every recorded 27-key chain at `t=12`,
`16`, `20`, outside every recorded band, with the rule's own band 2.9,
5.9, 9.7 percent.  Base / `etol1e3` / `dykstra60` / `conic1e10`:
`t=12` `3.797419e-2` / `3.831339e-2` / `3.906086e-2` / `3.799833e-2`
(band `3.797e-2` to `3.906e-2`; recorded `drive_power` `4.251e-2` to
`4.332e-2`, ahead by 1.09 to 1.14; `post_pulse` `4.612e-2` to
`6.076e-2`, 1.18 to 1.60); `t=16` `4.820897e-2` / `4.860629e-2` /
`4.588696e-2` / `4.825014e-2` (`4.589e-2` to `4.861e-2`; `drive_power`
`6.377e-2` to `6.720e-2`, 1.31 to 1.46; `post_pulse` `5.386e-2` to
`6.385e-2`, 1.11 to 1.39); `t=20` `6.353308e-2` / `6.330860e-2` /
`5.792205e-2` / `6.310984e-2` (`5.792e-2` to `6.353e-2`; `drive_power`
`8.519e-2` to `9.088e-2`, 1.34 to 1.57; `post_pulse` `6.991e-2` to
`7.564e-2`, 1.10 to 1.31; both cones `0.128607`, worst member `0.494`);
`t=10` band `3.580e-2` to `3.737e-2`, behind the recorded `3.333e-2` to
`3.357e-2` by 1.07 to 1.12.  The narrowest lead, the worst member over
the `post_pulse` band's bottom at `t=20`, is 1.10.  The `dykstra60` twin
is the best member at `t=16` and `20` and the one on which the cap
acts: seven uncertified answers above `0.4` (`0.720` to `1.138` at
`10.12` to `13.40`) pass the measured test on its repaired states and
are applied at `0.1` (the other members' answers above `0.4` are
`infeasible` in all but one case each), its `(12,16]` and `(16,20]`
windows `6.20e-2` and `9.14e-2` against the base's `7.05e-2` and
`1.04e-1`, its cones the tightest (joint Gram `-1.085e-3`, extension
`-9.84e-5`; base `-2.377e-3`, `-2.202e-4`), `d31` from the base `0.1` at
`t=13.1`, `9.3e-2` at `t=20`; the `etol1e3` and `conic1e10` twins stay
within 0.9 percent of the base from `t=12` on (`d31` at most `3.1e-2`),
the `conic1e10` twin accepting 204 uncertified iterates of its 250
refusals.  Cap-bound past `t=8`: 11 / 9 / 14 / 12 (14 / 14 / 19 / 16 of
500); `infeasible` 44 / 47 / 25 / 46; every member's `(16,20]` accepts
all 100 with no answer above `0.0115`.  Internal-energy RMS at `t=20`
`5.856e-3` / `5.262e-3` / `5.411e-3` / `5.441e-3` (recorded `5.22e-3` to
`6.34e-3`; both cones `0.00094`), held to `2.22e-15`; physicality over
`t<=20` joint Gram `-2.377e-3` / `-2.282e-3` / `-1.085e-3` / `-2.318e-3`,
extension `-2.202e-4` / `-2.151e-4` / `-9.84e-5` / `-2.162e-4`, all
attained in `(9.5, 11.1]`; envelope `0.52` to `0.57`.

The 27-key ledger row now carries a four-member band at every horizon:
`t=4` task 8's `1.00e-2` (0.5 percent), `t=8` the recorded chain's
`2.59e-2` (4.5), `t=12` / `16` / `20` the ball rule's base `3.80e-2` /
`4.82e-2` / `6.35e-2` with bands `3.80e-2` to `3.91e-2`, `4.59e-2` to
`4.86e-2`, `5.79e-2` to `6.35e-2`.  Adoption is the user's call; the
plan-file row was updated in place.  Next: the anchor-free base from
`t=8` to `t=20` (`queue_task9_t20.sh measured`); Dykstra 120 on the ball
base past `t=8`; the `(4, 8]` phonon-occupation loss.

## 2026-09-18: projective route, task 10: the radius-free rule and Dykstra 120 from `t=8` to `t=20` on the 27-key `N=16` drive-power chain

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`
(heading "projective-route, task 10 (2026-09-18)").  Two chains of the
27-key `N=16` `drive_power` energy-neutral base from a stored `t=8` state
to `t=20`, `1,000 s` `ram_guard --limit-mb 4500` segments, one at a time,
`02:13` to `04:01`: (a) task 8's anchor-free rule without the radius
(`LiftReselectionRule(acceptance_margin=1e-3, candidates="measured",
anchor_weight=0)`, Dykstra 30) continued from
`..._measured_base_20260916i_seg2` in three segments (`8.0` to `11.4`,
`11.4` to `15.4`, `15.4` to `20.0`; `2,901.8 s`; `4,788.1 s` in all);
(b) the task-9 ball base (`trust_radius=0.1`, `ball`) continued from
`..._ball_base_20260917k_seg2` with `--dykstra-iterations 120` in four
(`8.0` to `11.04`, `11.04` to `14.24`, `14.24` to `18.3925` with the
resume at `18.36`, `18.36` to `20.0`; `3,416.5 s`; `5,393.7 s` in all)
under the prefix `..._ball_base_dyk120_20260918l_seg` whose `seg1`,
`seg2` are links to the ball base's own.  Imports from
`head_src_e3374f6f`, identical to `paper_5/src/paper5` at `HEAD c14bf1bd`
outside `__pycache__`; drivers unchanged from task 8; no module edited;
nothing committed; no default changed.  Scores on `exact_t20.npz` (8,001
samples); physicality minima over `(8,20]` exclude the shared `t=8.0`
sample (`-1.2218e-3` in both (b) and the ball).

Negative results first.  (1) The radius owns none of the gain past
`t=8`: (a) scores `3.851781e-2` / `4.848061e-2` / `6.291440e-2` at `t=12`
/ `16` / `20` (ratios to the ball base `1.014` / `1.006` / `0.990`),
inside the ball rule's four-member band at every horizon from `t=8`
(`3.797e-2` to `3.906e-2`, `4.589e-2` to `4.861e-2`, `5.792e-2` to
`6.353e-2`), with windows `(8,12]`, `(12,16]`, `(16,20]` `4.926e-2` /
`7.036e-2` / `1.019e-1` against the ball's `4.799e-2` / `7.050e-2` /
`1.043e-1` (ratios `1.027` / `0.998` / `0.977`; the recorded chain's
`1.331` / `1.477` / `1.356`); so the gain over the recorded chain is the
measured-candidate acceptance's; the radius moves six applied
displacements of `0.105` to `0.228` (excess above `0.1` `0.261` of
`4.465` applied past `t=8`), every answer above `0.4` (30, `0.56` to
`0.79`) being `infeasible` on the radius-free chain (the ball's one
capped `0.793` at `t=11.04` lands `infeasible` on the other side of the
measured test from a state `3e-2` away); the 15-key finding (12 percent
at `t=20`) does not carry to 27 keys.  (2) Dykstra 120 does not touch
the internal energy (identical RMS `5.641305e-3` / `5.776465e-3` /
`5.856071e-3`: the same pinned reference) and worsens the
phonon-occupation block (`5.391e-2` against the ball's `4.692e-2` and
the recorded `3.187e-2` at `t=20`; per window `7.685e-2` / `5.633e-2` /
`2.680e-2` against `6.762e-2` / `3.658e-2` / `1.917e-2`).

Results.  (b) scores `3.834289e-2` / `4.594082e-2` / `5.784276e-2`
(ratios `1.010` / `0.953` / `0.910`); the `t=20` number is the best
27-key number on record, `0.14` percent below the `dykstra60` twin's
`5.792205e-2` (a tie), `0.450` of the both-cones `0.128607`, ahead of
the `drive_power` band by `1.473` to `1.571` and the `post_pulse` band
by `1.209` to `1.308`; windows `4.886e-2` / `6.350e-2` / `9.104e-2`
(ratios `1.018` / `0.901` / `0.872`); over `(8,20]` retained joint Gram
`-4.420e-4` at `11.047` and extension `-4.187e-5` at `9.498` against the
ball's `-2.377e-3` at `10.08` and `-2.202e-4` at `9.545` (`5.4` and `5.3`
times shallower; mean negative part `-1.635e-4` / `-1.426e-5` against
`-9.940e-4` / `-8.243e-5`; the `dykstra60` twin `-1.085e-3` /
`-9.84e-5`), the retained cone still violated at all 4,800 samples past
`t=8` in every chain; envelope `0.515` (ball `0.573`); first repaired
sample `-2.939e-5` against `-3.531e-4` from the shared state; the other
27 coordinates `5.840e-2` against `6.564e-2` at `t=20`; `d31` from the
ball `4.733e-2` / `8.681e-2` / `7.591e-2` / `9.784e-2` at `t=10` / `12`
/ `16` / `20` (first difference `t=8.0025`; first differing decision
`t=8.16`, decisions equal at 257 of 300); past `t=8` accepted 281 (71
uncertified), cap 5 (`8.28` `0.139u`, `8.32` `0.101`, `10.24` `0.259u`,
`10.28` `0.151`, `14.28` `0.281u`), `infeasible` 14, refusals 85, 7
answers above `0.4` all `infeasible` (ball 255 / 11 / 34 / 89 / 30);
`(16,20]` accepts all 100 in every chain (largest answer `0.0112`);
repair `549.3 s` past `t=8` against `142.7 s` (`3.85` times; 16.1
percent of the wall), solves `9.30 s` mean against `8.83 s`; peaks
`630` to `668 MiB`.  (a): internal-energy RMS `5.035053e-3` /
`5.157268e-3` / `5.229233e-3`, the ball's scaled by `0.893` (the two
declared references `5.507e-3` and `6.164e-3` above the exact post-pulse
energy), held to `2.22e-15` over 23,935 constrained repairs ((b) 23,994),
0 hyperplane refusals; retained joint Gram `-2.292e-3` at `11.08`,
extension `-2.145e-4` at `9.547`; past `t=8` accepted 260 (49),
`infeasible` 40, refusals 89, over-radius moves at `9.08` `0.105u`,
`9.20` `0.150`, `11.00` `0.124u`, `11.08` `0.228`, `14.24` `0.120u`,
`14.40` `0.135`; `d31` from the ball `2.012e-2` / `3.357e-2` / `1.719e-2`
/ `2.523e-2` at `t=8` / `12` / `16` / `20` (3 percent of the distance
from exact); its `seg4` overran the budget by `38.6 s` (`1,038.6 s`);
peaks `633` to `649 MiB`.  Gates: (a)'s `seg3` summary records the
resumed state, task 8's rule and Dykstra 30; (b)'s `d31 = d70 = 0`
against the ball at `t=8.0`, `dykstra_iterations 120` in `seg3` to
`seg6`; source tree identical to HEAD; one grid; ledgers merge to 500;
seven sequential segments, longest `1,042.2 s` plus `18.9 s` build.

The 27-key ledger row's numbers and standing do not change (`1.00e-2` /
`2.59e-2` / `3.80e-2` / `4.82e-2` / `6.35e-2` at `t=4` / `8` / `12` / `16`
/ `20`, bands unchanged, the radius-free base inside each from `t=8`);
its rule may be stated as the anchor-free measured rule without the
radius (the radius costs 4 percent at `t=4` and moves nothing past
`t=8`); the Dykstra count is the setting that moves the score past
`t=12` (30 / 60 / 120: `6.353e-2` / `5.792e-2` / `5.784e-2` at `t=20`)
and the cones (`-2.38e-3` / `-1.09e-3` / `-4.42e-4` past `t=8`), and the
Dykstra-120 member is the `t=20` low edge (`5.78e-2`, one run) if
adopted.  Artifacts: `..._measured_base_20260916i_seg{3,4,5}`,
`..._ball_base_dyk120_20260918l_seg{3,4,5,6}` (+ `seg1`, `seg2` links),
each with `reselections.json`; in
`paper_v_projective_dictionary_energy_20260915e/`: `queue_task10.sh`,
`tabulate_task10.py`, `crosscheck_task10_caps.py`,
`update_plan_row_task10.py`, `task9_measured_base_score_t20.json`,
`task10_ball_base_dyk120_score_t20.json`, `tabulate_task10.log`,
`crosscheck_task10_caps.log`, `task10_chain.log`, the segment logs.
Plan-file row updated in place.  Next: the anchor-free rule without the
radius with Dykstra 120 from `t=0` to `t=20`; the `(4,8]`
phonon-occupation loss.

## 2026-09-18: projective route, task 11: the candidate final configuration --- the anchor-free measured rule without the radius with Dykstra 120 --- from `t=0` to `t=20`, with three one-setting twins to `t=8`

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`
(heading "projective-route, task 11 (2026-09-18)").  The two ingredients
task 10 measured separately, run together from `t=0` on the 27-key `N=16`
`drive_power` energy-neutral chain: `LiftReselectionRule(acceptance_margin
=1e-3, candidates="measured", anchor_weight=0)` with no trust radius
(task 8's rule) and `--dykstra-iterations 120` (task 10's repair depth).
Base `t=0` to `20` in six `ram_guard --limit-mb 4500` segments (boundaries
`3.92`, `8.32`, `11.76`, `15.2`, `19.24`; `5,183.6 s`, longest `1,006.6 s`
plus `18.9 s` of build) and three one-setting twins to `t=8`
(`energy_tolerance 1e-3` in two segments, `1,977.0 s`; Dykstra 240 in
three, `2,196.6 s`; re-selection `conic_tolerance 1e-10` in two,
`1,880.4 s`), one at a time, `04:16` to `07:30`, nothing else heavy on the
machine.  Smoke gate first (`t=0` to `0.08`): `summary.json` records
`dykstra_iterations 120` and the rule with `trust_radius None`, both
re-selections accepted (`0.344683`, `0.863514`, uncertified), no cap.
Imports from `head_src_e3374f6f`, identical to `paper_5/src/paper5` at
`HEAD 31136ae9` outside `__pycache__`; `run_chain.py`,
`run_energy_segment.py` and `score_energy_chain.py` are task 8's
unchanged; no module edited; nothing committed; no default, rule, radius,
Dykstra count or ladder changed.  Scores are the all-coordinate scalar RMS
of the 31 closed scalar coordinates against the exact cutoff-16 driven
reference from `t=0` on one grid (`exact_t8.npz`, 3,201 samples;
`exact_t20.npz`, 8,001).

Negative results first.  (1) The combination is behind every 27-key chain
at `t=8`, `10` and `12`: the base scores `3.304213e-2` at `t=8` (behind the
recorded `drive_power` band `2.593e-2` to `2.709e-2` by 1.22 to 1.27, task
8's anchor-free four `3.180e-2` to `3.224e-2` by 1.025 to 1.039 and the
ball four `3.143e-2` to `3.216e-2` by 1.028 to 1.051), `3.856445e-2` at
`t=10` (behind the ball four `3.580e-2` to `3.737e-2` by 1.03 to 1.08) and
`4.059694e-2` at `t=12` (behind the ball band `3.797e-2` to `3.906e-2` by
1.04 to 1.07), so the deeper repair applied from `t=0` costs where task
10's from-`t=8` chain cost nothing.  (2) It re-opens the one-setting
sensitivity the rule had closed: the four members spread 7.14 percent at
`t=4` (`9.289800e-3` to `9.953452e-3`) and 9.78 at `t=8` (`3.009869e-2` to
`3.304213e-2`) against task 8's 0.52 and 1.40 at Dykstra 30 (14.76, 12.16,
12.33 percent at `t=1`, `2`, `6` against 3.83, 0.99, 1.35), every member
branching at the `t=0.08` re-selection where the completion's answer is
`0.8635` (base, accepted uncertified), `1.2690` and `1.3304` (declined by
the margin) and `1.4150` (`infeasible`) --- the same re-selection at which
task 8's Dykstra-30 base found a diverged answer `infeasible`; the base
then applies four more answers above `0.4` in `(0, 0.8]` (up to `1.092` at
`0.76`) and carries a frontier envelope maximum of `0.8825` at `t=0.7575`.
(3) It does not touch the `(4, 8]` phonon-occupation loss: the four
phonon-occupation coordinates score `6.7792e-2` over that window against
the recorded chain's `4.5378e-2` (1.494), where the same rule at Dykstra 30
gives `6.6901e-2`, at 240 `6.4449e-2` and the ball base `6.8097e-2`.

Results.  The base is the best 27-key number on record at `t=4` and at
`t=20`: `9.289800e-3` over `t=0` to `4`, ahead of task 8's whole band by
1.072 to 1.078 (and every task-11 member ahead of it by 1.001 to 1.078,
the worst tying task 8's best by 0.06 percent), ahead of the ball band by
1.08 to 1.12, the recorded `drive_power` band `1.143e-2` to `1.292e-2` by
1.23 to 1.39 and the `post_pulse` band by 1.76 to 2.12, `0.7385` of cell
A's `0.01258`; and `5.603962e-2` over `t=0` to `20`, ahead of the whole
ball band `5.792e-2` to `6.353e-2` by 1.034 to 1.134, of task 10's
Dykstra-120-from-`t=8` member `5.784276e-2` by 1.032, of the
`drive_power` band `8.519e-2` to `9.088e-2` by 1.52 to 1.62 and the
`post_pulse` band `6.991e-2` to `7.564e-2` by 1.25 to 1.35, `0.4357` of
the both-cones `0.128607`.  At `t=16` `4.610355e-2`, inside the ball band
and `1.0047` times its low edge (a tie with the `dykstra60` twin's
`4.588696e-2`); at `t=6` `1.984642e-2`, ahead of task 8's whole band by
1.002 to 1.015.  Windows `(0,4]`, `(4,8]`, `(8,12]`, `(12,16]`, `(16,20]`,
`(8,20]`: `9.2927e-3`, `4.5803e-2`, `5.2546e-2`, `5.9650e-2`,
`8.4857e-2`, `6.7131e-2` (ratios to the ball base `0.891`, `1.047`,
`1.095`, `0.846`, `0.813`, `0.863`; to the Dykstra-120-from-`t=8` member
`0.891`, `1.047`, `1.075`, `0.939`, `0.932`, `0.959`).  The gain is
outside the phonon-occupation block: the 27 coordinates outside it score
`4.268e-2` and `5.556e-2` at `t=16` and `20` against the ball's
`4.769e-2` and `6.564e-2` and the recorded chain's `6.722e-2` and
`9.046e-2`, while the four phonon-occupation coordinates are worse at
every horizon past `t=4` (`5.919e-2` at `t=20` against the ball's
`4.692e-2` and the recorded chain's `3.187e-2`); blockwise at `t=20` rho /
`B` / `N` / `A` / `C` `8.711e-2` / `9.768e-2` / `5.919e-2` / `3.964e-2` /
`3.045e-2` (ball `1.006e-1` / `1.252e-1` / `4.692e-2` / `4.258e-2` /
`2.974e-2`); site-0 occupation `6.805e-2` in the `rho[0,0]` convention at
`t=20`, `1.361e-1` doubled (both cones `0.205`, the `post_pulse` chain
`1.250e-1`).  The cones are the shallowest of any 27-key chain measured
from `t=0`: retained joint Gram minimum `-4.460e-4` at `t=11.04` and
extension `-3.936e-5` at `9.4975` over `t<=20`, 5.1 and 5.4 times
shallower than the same rule at Dykstra 30 (`-2.292e-3` / `-2.145e-4`),
mean negative part `-1.069e-4` / `-9.425e-6` against `-6.748e-4` /
`-5.631e-5`; past `t=8` the retained cone is violated at all 4,800 samples
as in every chain and the extension cone at 4,293 of 4,800 (task 8's base
4,363).  Internal-energy RMS `4.032812e-3` / `4.887963e-3` / `5.141528e-3`
/ `5.263751e-3` / `5.335746e-3` at `t=4` / `8` / `12` / `16` / `20`, the
declared reference `-3.565784077173663` lying `5.614e-3` above the exact
post-pulse energy `-3.571397982`, held to `1.78e-15` over 23,802
constrained repairs with 0 hyperplane refusals.  Decisions of 500: 484
accepted (101 uncertified), 16 `infeasible`, no cap, no decline, 117
solver refusals (task 8's base 137, the ball's 136, the recorded chain's
144); eight applied moves above `0.1`, five in `(0, 0.8]` and three past
`t=8` (`0.127u` at `10.40`, `0.174u` at `15.28`, `0.117u` at `15.32`); of
15 answers above `0.4`, four accepted and 11 `infeasible`; `(16,20]`
accepts all 100 with no answer above `0.0115`.  Wall `5,183.6 s`, `0.803`
in the 500 completion solves at `8.32 s` mean and `0.173` in the repair
(`895.4 s`, 3.69 times task 8's `242.4 s`, at `+8.3` percent of wall);
peaks `626` to `699 MiB`.  Twins at `t=8`: `3.108092e-2` / `3.009869e-2` /
`3.167023e-2`, retained joint Gram / extension minima `-1.917e-4` /
`-1.625e-5`, `-5.512e-5` / `-7.602e-6`, `-1.943e-4` / `-1.673e-5` (base
`-2.096e-4` / `-2.138e-5`), refusals 36 / 36 / 73 of 200 (base 43), the
`energy_tolerance 1e-3` twin holding its reference to `4.44e-15` over
8,975 constrained repairs.

The `(4, 8]` phonon-occupation loss, unchanged by any setting of this
rule, is entirely in the second half of the window and is a bias: over
`(4, 6]` the base scores `4.8357e-2` against the recorded chain's
`5.4693e-2` (ahead by 1.131) and over `(6, 8]` `8.2784e-2` against
`3.3571e-2` (behind by 2.466), with per-coordinate signed mean errors over
`(4, 8]` of `-7.348e-2`, `-7.436e-2`, `+7.332e-2`, `-8.343e-4` against
per-coordinate RMS `7.788e-2`, `7.916e-2`, `7.776e-2`, `2.120e-3`, while
all 100 completions of the window are accepted at a median applied
frontier displacement of `0.0093` and a maximum of `0.0145` (the twins
100, 101, 100 accepted, maxima `0.0149`, `0.0149`, `0.0147`).  So neither
the acceptance test nor the repair depth is implicated; the loss is in the
closed flow of the phonon-occupation coordinates over `(6, 8]`.

The 27-key ledger row's standing changes at `t=4` and `t=20` and nowhere
else, and the row now names three configurations across the five horizons:
this configuration's `9.29e-3` at `t=4` (band `9.29e-3` to `9.95e-3`, 7.1
percent) and `5.60e-2` at `t=20` (one run, no band of its own), the
recorded `drive_power` chain's `2.59e-2` at `t=8` (band `2.59e-2` to
`2.71e-2`), and the ball rule at Dykstra 30 `3.80e-2` at `t=12` (band
`3.797e-2` to `3.906e-2`) and `4.82e-2` at `t=16` (band `4.589e-2` to
`4.861e-2`).  Adoption is the user's call; the plan-file row was updated
in place.  Gates: smoke gate records the combination; source tree
identical to `HEAD 31136ae9`; every segment's `summary.json` records its
`resumed_from`, rule, `dykstra_iterations`, `energy_tolerance` and
`reselection_conic_tolerance`; energy held; one grid (8,001 and 3,201
samples); ledgers merge to 500 and 200; thirteen sequential rollouts under
`4,500 MB`.  Artifacts:
`paper_v_dictionary_energy_measured_dyk120_smoke_20260918m`,
`paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg{1..6}`,
`..._measured_dyk120_{etol1e3,dykstra240,conic1e10}_20260918m_seg*` (each
with `reselections.json`); in
`paper_v_projective_dictionary_energy_20260915e/`: `smoke_task11.sh`,
`check_task11_smoke.py`, `queue_task11.sh`, `tabulate_task11.py`,
`peek_task11_details.py`, `cone_counts_task11.py`,
`update_plan_row_task11.py`, `task11_base_score_t20.json`,
`task11_{etol1e3,dykstra240,conic1e10}_score_t8.json`,
`tabulate_task11.log`, `peek_task11_details.log`,
`cone_counts_task11.log`, `task11_chain.log`, `queue_task11.log`, the
segment logs.  Next: the two `t=4` to `8` reruns that locate the
phonon-occupation loss (task 8's rule from the recorded chain's `t=3.96`
state; this base with `certified` candidates from its `t=3.92` state); the
three twins of this configuration from `t=8` to `t=20` for the `t=20`
band.

## 2026-09-18: projective route, task 12: four `t=4` to `8` reruns that locate the `(4, 8]` phonon-occupation loss of the 27-key `N=16` drive-power chain

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`
(heading "projective-route, task 12 (2026-09-18)").  Four reruns of the
window, each resumed from a stored state and run to `t=8.0` in one
`ram_guard --limit-mb 4500` segment with an `1,100 s` autonomous budget,
one heavy process at a time, `10:47` to `11:47`, nothing else heavy on the
machine; imports from `head_src_e3374f6f`, identical to
`paper_5/src/paper5` at `HEAD c3b9de60` outside `__pycache__`;
`run_energy_segment.py`, `run_chain.py` and `compare_chains.py` are task
8's unchanged; no module edited, nothing committed, no default, rule,
radius, Dykstra count, reference mode or ladder changed.  (a)
`a_rule30_recorded`: task 8's rule
(`LiftReselectionRule(acceptance_margin=1e-3, candidates="measured",
anchor_weight=0)`, no radius) at Dykstra 30 continued from the recorded
`drive_power` chain's own `t=3.96` state, which separates the rule from the
trajectory it inherits; `a_certified_t11`: the task-11 base's configuration
(the same rule at Dykstra 120) with `certified` candidates from its
`t=3.92` state, task 8's standing proposal.  (b) from the same `t=3.92`
state under the base's own rule and repair depth, `b_energy_none` with the
energy hyperplane disabled and `b_postpulse` with the `post_pulse`
reference --- the trajectory's own Eq. (22) internal energy at the declared
drive end `t=4` --- in place of `drive_power`; both use only the
trajectory's own state and the Hamiltonian's declared drive.  Window scores
are the RMS over the 1,600 samples of `(4, 8]` against the exact cutoff-16
driven reference on one grid (`exact_t8.npz`), on the 31 closed scalar
coordinates and restricted to the four phonon-occupation coordinates and
the 27 outside them.  Smoke gate first: each member `t=3.92` (or `3.96`) to
`4.08`, `summary.json` recording the resumed time, the repair depth, the
reference mode, the rule, a declared reference in the three constrained
members and none in `b_energy_none`; `SMOKE PASSED`.

Negative results first.  (1) **The energy reference's offset accounts for
none of the bias.**  The `post_pulse` reference sits `+1.345e-2` above the
exact post-pulse energy `-3.571397982`, `2.40` times the `drive_power`
reference's `+5.614e-3`, and gives a *smaller* bias of the *same* sign
(per-coordinate signed means `-5.959e-2`, `-6.267e-2`, `+6.066e-2`,
`-6.917e-4`, `0.81` to `0.84` of the task-11 base's `-7.348e-2`,
`-7.436e-2`, `+7.332e-2`, `-8.343e-4`), and the recorded chain, whose
reference sits `+6.710e-3` above, carries the smallest bias of all
(`-4.963e-2`, `-5.219e-2`, `+4.649e-2`).  The offset sets the Eq. (22)
internal-energy RMS exactly --- every chain's window energy RMS equals its
own offset to four digits (`5.6148e-3`, `1.3448e-2`, `6.7113e-3`) --- and
nothing else.  (2) **The loss is not what the window's re-selections do.**
The quietest window on record (the task-11 base's 100 accepted completions,
maximum applied frontier displacement `0.0145`, 10 solver refusals) carries
the largest loss, `1.494`; the window with two whole-face answers applied
whole (`1.0492` at `t=6.72` uncertified, `1.0769` at `6.80` certified), 14
`infeasible` decisions on answers `0.5582` to `1.4108` and 33 refusals ---
task 8's rule on the recorded chain's trajectory --- carries `1.149`.  (3)
**Disabling the hyperplane does not recover the recorded chain's block.**
With it off the four phonon-occupation coordinates score `5.1131e-2` over
`(4, 8]`, still `1.127` times the recorded chain's `4.5378e-2` (and
`2.104` over `(6, 8]`), the bias reverses sign (`+5.106e-2`, `+2.322e-2`,
`-3.612e-2`, `+1.546e-3`; the share carried in the mean falls from `0.94`
to `0.55`--`0.69`), and the Eq. (22) internal-energy RMS rises from
`5.6148e-3` to `1.3790e-1` (signed mean `+1.225e-1`, `24.6` times) for a
4 percent all-coordinate gain (`4.4003e-2` against `4.5803e-2`) with the 27
coordinates outside the block worse (`1.031`, the `A` block `7.193e-2`
against `5.966e-2`).

Results.  (a) The loss follows the trajectory the window is handed, through
that state's history rather than its error, with the rule contributing
`1.149` of the `1.494` inside the window and the candidate test 1.7
percent.  Window scores on the four phonon-occupation coordinates over
`(4, 8]`: `a_rule30_recorded` `5.213841e-2` (`1.149` of the recorded
chain's `4.537824e-2`, `0.769` of the task-11 base's `6.779204e-2`);
`a_certified_t11` `6.665237e-2` (`0.983` of the base's; all 31
`4.444605e-2` against `4.580275e-2`, `0.970`; 11 of 100 completions refused
on answers `0.0067` to `0.0119`); over `(6, 8]` `5.013417e-2` (`1.493`) and
`7.948436e-2` (`2.368`) against the base's `8.278358e-2` (`2.466`).  The
recorded chain and task 8's rule enter `t=4` on the identical state (the
block's RMS `2.8916e-2`, signed errors `-3.3954e-2`, `-3.3537e-2`,
`+3.2623e-2`, `-1.6111e-3`) and leave `t=8` at `3.0562e-2` and `4.6569e-2`
(growth `1.06` against `1.61`), while the task-11 base enters with the
smallest block error of any chain (`2.0471e-2`, `0.71` of the recorded
chain's) and leaves with the largest (`1.0302e-1`, growth `5.03`; task 8's
base `2.2021e-2` to `9.2727e-2`, the ball `2.3268e-2` to `9.5453e-2`).  So
the rule's action over `t<=4` buys a better block at `t=4` and a worse one
at `t=8`, and `(6, 8]` growth, not the entering error, is where the row is
lost.  (b) The hyperplane sets the bias's *sign* --- it is the one setting
measured that reverses it --- and trades the internal-energy error against
the phonon-occupation error: the two `(b)` members are one chain to `t=4`
(identical coordinates there), and removing the constraint over the `0.08`
units from the stored state to the drive end already takes 23 percent off
the block's error (`1.5670e-2` against the base's `2.0471e-2`).
`b_postpulse` scores `5.678120e-2` on the block (`0.838` of the base's,
`1.251` of the recorded chain's) and `4.526223e-2` on all 31 (`0.988`),
energy RMS `1.344759e-2`, cones `-8.560e-5` / `-8.400e-6`.  **One quantity
the hyperplane owns outright is the cones**: with it off the Dykstra repair
reaches both over the whole window --- minimum unshifted retained joint
Gram `-1.666e-12`, minimum scaled positive-extension eigenvalue
`-1.498e-09` --- against the task-11 base's `-2.096e-4` / `-2.138e-5` and
the recorded chain's `-2.152e-3` / `-1.922e-4`, where every constrained
chain on record violates the retained cone at every sample; it costs `2.6`
to `2.7` times the repair time (`68.9 s` of a `722.3 s` segment against
`180.5 s` of `890.5 s`) and 23 percent of the segment's wall.  (c) The
scaled positive-extension matrix is affine in the state, so the probe's
derivative holds over the window: the three biased closed scalar
coordinates are built from `raw_normal_00` and `raw_normal_11` (extension
derivative `0.146282`, 3.1 to 3.3 percent of the electronic Bloch
coordinates' `4.4729` to `4.6915`) and `raw_normal_01_real` (`0.292565`,
6.2 to 6.5 percent), and the fourth from `raw_normal_01_imag`, whose
derivative is exactly zero.  The unbiased fourth is small in absolute terms
only because the exact coordinate's own RMS over `(4, 8]` is `8.4742e-4`
against the others' `0.4212` to `0.4218`: relative to its own size its
error is `2.501` against `0.185` to `0.188`, the worst-tracked of the four
by `13.5`.  All four phonon normal coordinates are weakly seen or unseen,
and the retained Gram --- which sees all 29 raw coordinates within a factor
`1.41` --- is the only cone touching the block.  Enlarging from 15 keys (24
extension rows) to 27 (36) raised the extension's derivative with respect
to the three by `1.394` (`0.104966` to `0.146282`; `0.209933` to
`0.292565`) and left `raw_normal_01_imag` at exactly zero at both sizes,
while the block is worse at 27 keys: more entrance keys buy visibility of
three coordinates and none of the fourth, so the dictionary needs a key
whose extension row carries the phonon normal moments rather than more of
the same.

Walls: `929.7` / `915.9` / `722.3` / `890.5 s` for 1,616 / 1,632 / 1,632 /
1,632 steps (repair `49.3` / `187.3` / `68.9` / `180.5 s`, re-selection
`854.7` / `702.7` / `640.1` / `696.5 s`); peaks `622` to `653 MiB`.  Gates:
smoke gate records each member's flags; source tree identical to `HEAD
c3b9de60`; every `resumed_from` recorded and every segment `completed` with
exit `0`; the re-declared `drive_power` references agree with the parent
chains' to `6.1e-09` (`a_rule30_recorded`) and `1.6e-08`
(`a_certified_t11`); the energy held to `2.22e-15` over 4,848, `1.33e-15`
over 4,866 and `1.78e-15` over 4,795 constrained repairs with 0 hyperplane
refusals, and 0 constrained repairs where disabled; the two `(b)` members
identical at `t=4`; one grid (1,600 samples in `(4, 8]`); 100
re-selections per window; eight sequential rollouts under `4,500 MB`.  The
ledger row's numbers do not move (no rerun is a chain from `t=0`); what is
added is the attribution and the cones result.  Artifacts:
`paper_v_dictionary_energy_window48_27key_n16_task12_smoke_*_20260918n` and
`paper_v_dictionary_energy_window48_27key_n16_task12_{a_rule30_recorded,a_certified_t11,b_energy_none,b_postpulse}_20260918n_seg1`
(each with `reselections.json`); in
`paper_v_projective_dictionary_energy_20260915e/`: `smoke_task12.sh`,
`check_task12_smoke.py`, `queue_task12.sh`, `tabulate_task12.py`,
`peek_task12_{states,resume,walls,decisions,entry}.py`,
`update_plan_row_task12.py`, `tabulate_task12.log`, `smoke_task12.log`,
`smoke_task12_check.log`, `peek_task12_*.log`, `task12_chain.log`,
`queue_task12.log`, the segment logs.  Next: the task-11 configuration with
the hyperplane disabled from `t=0` to `t=20` (the first 27-key chain that
would hold both cones); a dictionary key the extension sees the phonon
normal moments through; the three twins of the task-11 configuration from
`t=8` to `t=20` for the `t=20` band.

## 2026-09-18: projective route, dictionary probe: the `(4, 8]` phonon-occupation loss is one coordinate --- the relative-mode population --- and no dictionary can reach the other three

Task 12 ended on the instruction to find *a dictionary key that the positive
extension sees the phonon normal moments through, in particular
`raw_normal_01_imag`, whose extension derivative is exactly zero at both 15 and
27 keys*.  This light probe --- affine algebra, stored trajectories, and three
conic solves (the preparation's declared completion at three dictionaries),
eight processes under `ram_guard --limit-mb 2000` one at a time, no module
edited, nothing committed, no default changed --- answers it, and the answer is
negative on both halves.  **(a)** `raw_normal_01_imag` has zero derivative
because of the chart, not a missing row: the extension's online operator
algebra is the exact spin-symmetric relative-mode reduction (the dimer's centre
phonon is exactly decoupled in the fixed-particle sector), so the four raw
phonon normal coordinates reach it only through
`relative_moments_from_matrix_state`, which forms `v^dagger N v` with the real
relative-mode vector `v = (1,-1)/sqrt(2)`, that is
`0.5*(raw_normal_00 + raw_normal_11) - raw_normal_01_real`; the coefficient of
`Im N[0,1]` in a Hermitian form with a real `v` is identically zero.  Reading
the block in the phonon density's two normal modes --- the relative-mode
population `(phonon[0,0]+phonon[1,1])/2 - Re phonon[0,1]`, the centre-mode
population `(phonon[0,0]+phonon[1,1])/2 + Re phonon[0,1]`, and the two parts of
the relative--centre coherence `(phonon[0,0]-phonon[1,1])/2` and
`- Im phonon[0,1]` --- the extension sees exactly one of the four, and the row
that could carry the other three would have to be a word in a second boson
mode, which the `MomentKey` form `(spin_up, spin_down, x_power, p_power)`
cannot express.  **(b)** Eleven sets measured --- the four declared active
dictionaries (15, 19, 27, 31 keys) and seven with additional declared
`additional_halfword_keys`, 24 to 45 extension rows, 62 to 139 frontier keys,
state dimension 44 to 60: the rank of the extension's derivative map over the
four raw phonon normal coordinates is `1` in every one (second singular value
at most `3.2e-17`) and over all 29 raw coordinates `14` in every one, and the
directional derivative along the centre-mode population and along both parts of
the coherence is exactly `0` in every one.  The best available gain on the one
seen direction is `1.178` (`raw_normal_01_real` `0.292565` to `0.344586`, 27
keys plus the boson-quadratic and spin-boson-linear half-words, 45 rows);
`hidden31` and the four boson-cubic half-words buy `1.0005` while taking the
frontier from 70 to 129 and 133, which is the measured reason the boson-cubic
active keys do nothing for this block.  The probe reproduces
`extension_sensitivity_{15,27}key.json` to every digit printed.  Certification
at the exact-correlated preparation: `entrance15` directly (`1.28 s`), the
plain 27-key on the declared ladder's first rung (`20.35 s`), **27 keys plus
the six spin-boson-linear half-words directly** (`30.07 s`, Newton 11, face
`41/42`, scaled minimum eigenvalue `-8.47e-16`, state dimension unchanged at
56, rows 36 to 42, frontier 70 to 74); the boson-quadratic half-words did not
return inside the four-minute budget (stopped at `5 min 49 s`, peak
`1,558 MiB`) and are unmeasured.  **(c) The decisive measurement is the
decomposition, and it contradicts the premise**: the task-11 base's `(4, 8]`
phonon-occupation block RMS `6.779204e-02` is `6.765951e-02` (`0.9980`) from
the relative-mode population alone --- the one direction the extension already
sees --- and `1.883245e-03` (`0.0278`) from the three it cannot see; over
`(6, 8]`, where the whole loss sits, `8.278358e-02` is `0.9977` from it.  The
relative-mode population's signed mean error is `-1.472355e-01` with RMS
`1.562529e-01` against an exact coordinate of RMS `8.430089e-01` (`0.185` of
its own size), while the centre-mode population's error is `8.774705e-04` and
the coherence's `2.134963e-03` and `2.119634e-03`.  In the site basis that one
error of `-0.1472` appears as `-0.0736`, `-0.0736`, `+0.0736`, which is what
the per-coordinate table was reading as three separate biases.  The `1.494`
between the task-11 base and the recorded `drive_power` chain in this block is
`1.513` in that single coordinate over `(4, 8]` (`1.562529e-01` against
`1.032811e-01`) and `2.539` over `(6, 8]` (`1.907462e-01` against
`7.513319e-02`); and the recorded chain tracks the three blind directions
*worse* (`2.853940e-03` against `1.883245e-03`, centre-mode population
`5.038008e-03` against `8.774705e-04`, `5.7` times), so the dictionary's
blindness is not costing this chain anything measurable.  The Eq. (22) energy
hyperplane is cleared too: its normal is `0.5` on `raw_normal_00`, `0.5` on
`raw_normal_11` and `0` on both off-diagonal coordinates --- `0.5` on each of
the two populations and `0` on the coherence in the normal modes --- so it
moves the two populations identically and cannot produce a `178`-fold
asymmetry; its norm in the block is `0.7071068` of `3.2403703` over all 29 raw
coordinates and `0` on all 27 entrance coordinates.  **Recommendation**: do not
spend a rollout on a dictionary enlargement for this block; if two are run,
27 keys plus the six spin-boson-linear half-words (justified by the direct
certification, not by the derivative; no module edit needed, since
`additional_halfword_keys` is an existing declared argument and
`run_energy_segment.py`'s `prepare_with_ladder` builds the extension itself)
and `entrance15` under the task-11 configuration as the control, whose
visibility is *lower* (`0.209933`, `0.718`) while its block is better.
Cheapest decisive test: task 12's `t=4` to `8` rerun from the task-11 base's
stored `t=3.92` state, one segment to `t=8.0` under `--limit-mb 4500`, about
15 minutes, scored on the `exact_t8.npz` grid; improving means the four
phonon-occupation coordinates below `6.779204e-02` over `(4, 8]` and
`8.278358e-02` over `(6, 8]` *and* the relative-mode population below
`1.562529e-01` and `1.907462e-01`, with `1.032811e-01` and `7.513319e-02` the
recorded chain's line; under `1.10` is inside task 11's own twin spread of
`9.78` percent and is not a result.  Ledger row unchanged (no horizon score
this round).  Artifacts in
`paper_v_projective_dictionary_energy_20260915e/`:
`probe_dictionary_keys_20260918.py`,
`probe_extension_dictionary_20260918.py`,
`sweep_extension_dictionary_20260918.sh`,
`probe_normal_mode_bias_20260918.py`,
`probe_energy_normal_direction_20260918.py`,
`append_dictionary_probe_20260918.py`,
`extension_dictionary_sweep_20260918.{json,log}`,
`extension_dictionary_prepare_a_20260918.json`,
`normal_mode_bias_w48_20260918.{json,log}`,
`normal_mode_bias_w68_20260918.{json,log}`,
`energy_normal_direction_20260918.json`.  Next: the loss is in the closed flow
of the relative-mode population over `(6, 8]`; the next measurement belongs on
that flow, not on the dictionary.

## 2026-09-18: projective route, relative-mode flow: the `(6, 8]` relative-mode population gap is the energy-neutral Dykstra repair's displacement of that coordinate, not its closed flow

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`
(heading "projective-route, relative-mode flow (2026-09-18)").  The
dictionary probe ended on the instruction that *the loss is in the closed
flow of the relative-mode population over `(6, 8]`; the next measurement
belongs on that flow*.  This light round --- stored trajectories and affine
algebra only, five processes under `ram_guard --limit-mb 2500` one at a
time, peaks `2 MiB`, nothing propagated, no conic solve, no module edited,
nothing committed, no default changed --- makes that measurement on the
task-11 base, the recorded `drive_power` chain and the four task-12
`t=4` to `8` reruns, at 1,600 samples in `(4, 8]` and 800 in `(6, 8]`.
The exact cutoff-16 driven reference is used to compute reference
velocities, which is a reporting diagnostic and never an online controller
input.

Negative results first.  **(1) There is no closure defect in the equation
this coordinate's flow comes from.**  The cheap route propagates
`CheapProjectionRoute.unguarded_velocity`, that is
`FixedDictionaryProjectedAPCM.targets(...).augmented_retained_velocity`,
whose phonon-density block is Eq. (14b) at the chain's own matrix state
with no completion, no frontier and no KPD term in it (the KPD correction
enters only the electron--phonon correlation block through
`_with_correlation_increment`).  Eq. (14b) is exact and closed in the
phonon density and the correlation, and the measurement confirms it: the
Eq. (14b) velocity at the exact state integrates over `(4, 8]` to the
exact relative-mode population's own change `2.322816e-02` with residual
`5.009643e-04`, in a window where the task-11 base's error moves by
`1.901549e-01`.  **(2) The flow difference is not what separates the two
chains.**  Over `(6, 8]` the time integral of the signed difference
between the closure's own velocity of the relative-mode population and the
exact reference's velocity of the same quantity is `-6.514211e-02` for the
task-11 base and `-6.126227e-02` for the recorded chain (`1.063`), and the
RMS of that difference is `4.977392e-02` against `4.726448e-02` (`1.053`),
on a window where the coordinate's error stands at `1.907462e-01` against
`7.513319e-02` (`2.539`).  **(3) The entering state is not the gap
either**: the base enters `t=4` at `-4.710809e-02` and the recorded chain
at `-6.636815e-02`, so the chain that ends `2.539` times worse enters
`1.41` times better.

Results.  **(a)** The difference is already open at `t=4` --- over
`(0, 4]` it integrates to `-1.139146e-02` (base) and `-2.749505e-02`
(recorded), quiet until `t=2` --- and of the base's `-4.710809e-02` error
at `t=4`, `-3.570375e-02` was the repair's.  It is negative at 1,139 of
1,600 samples over `(4, 8]` and 600 of 800 over `(6, 8]`, minimum
`-9.782983e-02`, maximum `+6.038075e-02`, and it changes sign: positive at
all 200 samples of `(5.5, 6.0]` on both chains.  It is not one burst: the
base's running integral from `t=4` is `-5.120606e-02` at `5.5`,
`-2.664839e-02` at `6.0`, `-8.209275e-02` at `7.0` (of which
`-4.282017e-02` is the two quarter-unit bins `(6.5, 6.75]` and
`(6.75, 7.0]`, signed means `-8.256313e-02` and `-9.036871e-02`, negative
at every sample) and `-9.147453e-02` at `8.0`; the recorded chain's
largest loss is in `(7.0, 7.5]`, where the base's is over.  **(b)** The
equation is Eq. (14b), assembled in
`matrix_reference.py::_phonon_rhs_terms` and summed in `matrix_dimer_rhs`.
`eq14b_free_rotation` is `-i(omega_q - omega_q') phonon[q,q']` and both
dimer modes carry `omega_ph = gamma = 0.5`, so it is exactly `0` at every
sample of both trajectories; the whole velocity is
`eq14b_correlation_source` and its two halves carry the discrepancy in
shares `0.47` and `0.53` (`(4, 8]` integrals `-4.327125e-02` and
`-4.857776e-02` on the base, `-3.887930e-02` and `-4.236393e-02` on the
recorded chain).  Their sum reduces exactly, the shared trace cancelling,
to `-2 g (correlation_0_diag_difference_imag -
correlation_1_diag_difference_imag)` --- reproducing the module's assembly
to `1.4e-17` at every sample --- so the flow is driven by two
electron--phonon correlation coordinates and no phonon-occupation
coordinate, and the two chains' error in that antisymmetric combination
agrees to `1.053` over `(6, 8]` (`4.064024e-02` against `3.859129e-02`).
**(c)** It is the repair.  Over `(6, 8]` its displacement of the
relative-mode population is `-7.083106e-02` on the base and `+7.017512e-02`
on the recorded chain --- same size, opposite sign, the recorded chain's
very nearly cancelling its flow difference.  From `t=0` to `8` the
integrated velocity differences are `-1.032405e-01` and `-1.087383e-01`
(`0.949`) while the repairs are `-1.343725e-01` and `+4.048406e-02`: of the
`1.693122e-01` by which the two errors differ at `t=8`, the repair accounts
for `1.748566e-01` (`1.033`) and the flow difference for `-0.032` of it, the
base's being `+5.497800e-03` less negative than the recorded chain's.
Across the six chains the five constrained integrated velocity differences
lie in `-7.018901e-02` to `-9.482451e-02` (spread `1.35`, over two rules,
two repair depths, two candidate tests and two references) while the
repairs run `-1.000373e-01` to `+7.934428e-02`.  On the task-11 base over
`(6, 8]` the repair displaces the coordinate `1.988844e-04` per step in
mean absolute value (signed mean `-8.853883e-05`, maximum `6.490602e-04`)
against the closure's own `8.421646e-05` and the velocity difference's
`9.156545e-05`: `2.36` and `2.17` times.  `b_energy_none` identifies the
hyperplane's role --- with it disabled the repair displaces the coordinate
`+2.713256e-01` in the same direction at all 1,600 steps (per-step signed
mean `+1.695785e-04`, mean absolute `+1.695786e-04`), ending
`+2.697136e-01` past the exact coordinate: the cone projections move this
coordinate and the Eq. (22) hyperplane bounds and can reverse that motion.

Gates: the assembled terms equal `matrix_dimer_rhs`'s phonon block to
`0.000000e+00`; the closed form to `1.4e-17` and `6.9e-18`; error at `t=4`
plus the integrated velocity difference plus the repair's displacement
reproduces the error at `t=8` to `3.2e-04` to `3.6e-04` on every chain; the
state is written only by the stage increments and `repair(...)`, the
re-selection writing the carried frontier and never the state; the window
RMS values reproduce the dictionary probe's `1.562529e-01`,
`1.032811e-01`, `1.907462e-01`, `7.513319e-02` and the exact coordinate's
`8.430089e-01` and `8.372025e-01` to every digit; sample spacing `0.0025`
throughout; the chains' `correlation_shared_trace_*` are `0` at every
sample.  The ledger row does not move (no rollout this round).  Artifacts
in `paper_v_projective_dictionary_energy_20260915e/`:
`probe_relative_mode_flow_20260918.py`,
`append_relative_mode_flow_20260918.py`,
`relative_mode_flow_{w48,w68,w04,task12_w48}_20260918.{json,log}`.  Next:
which block of the repair sets the sign --- replay the declared repair
(`AffineGramCoordinates.repair`) on the stored `(6, 8]` samples of the task-11 base and the recorded
chain, at each stored state with its stored `frontier_standardized` and
`internal_energy_reference`, under the four declared settings
`target="both"`, `"retained"`, `"extension"` and
`energy_reference_mode="none"`, and record the displacement of the
relative-mode population each produces (about `0.19 s` per repair, no
solve, no propagation).

## 2026-09-18: projective route, repair displacement: the relative-mode population's motion is the retained joint Gram projection redirected by the Eq. (22) hyperplane, one unit of coordinate per unit of suppressed energy

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`
(heading "projective-route, repair displacement (2026-09-18)").  The
relative-mode flow measurement ended on the instruction to *replay
`AffineGramCoordinates.repair` on the stored `(6, 8]` samples of the task-11
base and the recorded `drive_power` chain under `target="both"`, `"retained"`,
`"extension"` and `energy_reference_mode="none"` beside the chain's own, and
record the displacement of the relative-mode population each produces*.  This
light round --- stored trajectories and affine algebra only, five processes
under `ram_guard --limit-mb 2500` one at a time, peaks `1 MiB` to `172 MiB`,
nothing propagated, no conic solve, no module edited, nothing committed, no
default changed --- is that replay, one repair at each of the 800 stored
samples of the window on each chain (27 keys, 56 state coordinates, 36
extension rows, 70 frontier keys, `N = 16`, strong coupling, cutoff 16, step
`0.0025`, `dykstra_iterations` 120 on the base and 30 on the recorded chain,
`energy_tolerance` and `repair_tolerance` and `eigenvalue_floor` all `0`).  The
rollout's `stage` cadence applies three repairs per step, so a per-repair
figure below multiplied by three is the per-step figure it would produce.

Negative results first.  **(1) The replay does not reproduce the
chain-to-chain sign difference and cannot.**  At each chain's own setting the
replayed repair moves the coordinate negatively on both chains
(`-1.502784e-04` and `-5.264001e-04` per repair) while the rollouts ran
`-8.853883e-05` and `+8.771890e-05` per step: the replay's input is the chain's
own post-repair sample, whose Eq. (22) energy its own repair drove onto the
reference to `3.469447e-16` and `3.702594e-16`, while the rollout's repair saw
a stage state at an offset that is not stored (`reselections.json` records the
lift re-selection only).  **(2) Raising the Dykstra count does not help**:
on the base at `target="both"` the signed mean is `-8.853781e-05` at the
declared 120, `-1.141342e-04` at 480 and `-1.149300e-04` at 2,000, at `0.040`,
`0.158` and `0.398` seconds per repair, and at `target="retained"` the
alternation halts by 86.6 iterations and is identical to twelve digits at all
three --- the displacement is the alternation's limit, not its stopping point.
**(3) `state_motion_scale` is exactly inert** at `0.1`, `1.0` and `10.0`
(identical to every printed digit, both chains, all three targets), which
confirms the module's own claim that the affine Gram map's null space lies
inside the propagated-state group.  **(4) Repairing only on an actual violation
buys nothing**: a cone is violated at `0.990` of the base's 800 stored samples
and `1.000` of the recorded chain's (extension below zero at `0.835` and
`0.969`, retained at `0.990` and `1.000`).

Results.  **(a)** With the hyperplane in the alternation the retained joint
Gram is the mover: `2.130842e-04` per repair in mean absolute value on the base
(signed mean `-2.130842e-04`, maximum `6.090969e-04`, p99 `5.975330e-04`,
negative at `0.991` of the 800, integrated `-1.704674e-01`), which is `2.53`
times the closure's own per-step displacement of `8.421646e-05` and `1.07`
times the rollout's own per-step `1.988844e-04`; and `3.842450e-03` on the
recorded chain (`63.0` and `10.0` times, negative at `1.000`, integrated
`-3.073960e+00`), against window RMS errors of `1.907462e-01` and
`7.513319e-02` on an exact coordinate of RMS `8.372025e-01`.  `0.970` and
`0.980` of the Euclidean length of that block's correction to the 56
propagated coordinates lands on this one direction.  The scaled positive
extension moves it the other way and one tenth as far (`+2.091033e-05`,
`+4.205861e-04`).  **(b)** Without the hyperplane the retained block is inert
on this coordinate --- `8.310995e-08` and `9.025583e-07` per repair, `2,564`
and `4,258` times smaller, negative at only `0.136` and `0.589`, share of the
state correction `0.0005` and `0.0003` --- the three targets are additive to
`2.2` and `2.3` percent, and `both` *is* `extension` (correlation `+1.0000` and
`+0.9997`, ratio of signed means `0.980` and `0.978`).  With it on the residual
`both - retained - extension` is `28` percent of `both`, `both` correlates with
`retained` at `+0.9090`, and `retained` and `extension` carry opposite signs at
`0.809` and `0.966` of the samples.  So it is the alternation, and specifically
the retained joint Gram alternating with the Eq. (22) hyperplane.  **(c) The
hyperplane redirects, it does not bound.**  On these samples it demands no
energy change at all (offset `3.5e-16`, energy change `6.4e-16`), yet it
reverses the sign of `target="both"` at `0.970` and `0.983` of the samples and
changes the magnitude by `3.21` --- *larger* --- on the base and `0.562` on the
recorded chain.  What it suppresses is an energy change of `1.8006e-04` and
`3.3023e-03` per repair, and the exchange rate is one for one: the signed
displacement is linear in a deliberately opened energy offset with slope
`+1.009268` (base, `both`; `+1.333239` retained, `+1.333945` extension) and
`+0.448470` (recorded; `+1.448553`, `+0.962088`), which is exactly what the
Eq. (22) gradient's `0.5` on `raw_normal_00` and `0.5` on `raw_normal_11` must
give against the relative-mode population's identical `0.5`, `0.5` on the same
diagonal --- the hyperplane and the quantity being lost are the same row of the
same block.  It costs the cone too: `target="both"` reaches the retained floor
at `0.010` of samples with it on and `0.388` with it off.  **(d) The cheapest
change is a declared per-coordinate scale in the least-squares map back**,
small on the four raw phonon normal coordinates: the correction is already
effectively one-dimensional along this direction (`0.970`), and `3.1622777` of
the Eq. (22) gradient's `3.2403703` over the 29 raw coordinates lies outside the
phonon block, so the energy-neutral constraint has `0.976` of its normal
available elsewhere.  A smaller step per repair is not available (the repair is
a projection sized by how far outside the cones the state sits, `-2.096395e-04`
and `-2.152324e-03`, and the alternation already fails to reach both cones at
`0.990` and `1.000`), and turning the hyperplane off is already refuted by
`b_energy_none` (`+2.713256e-01` over `(4, 8]`, one-signed at all 1,600 steps,
ending `+2.697136e-01` past the exact coordinate).  Cost: one change in
`cheap_projection_route.py` (`factorization` already builds `scales` as a
per-coordinate vector; `CheapProjectionSettings` would carry the override and
the caches key on it), locked to another agent this round; then a free replay
on the same 800 samples, about `100` s per chain, passing only if the
per-repair displacement falls below `8.421646e-05` with the both-cones fraction
not below `0.010` and the energy still on the reference to `1e-15`; then task
12's `t=4` to `8` rerun from the base's stored `t=3.92` state, one segment
under `--limit-mb 4500`, about 15 minutes, improving only below `1.907462e-01`
and closing the gap only at `7.513319e-02`.

Gates: the coordinate read from the stored 56 propagated coordinates through
the scorer's chart equals the one read from the stored `archive_coordinates` to
`0.000000e+00` and the affine Eq. (22) energy equals the stored
`internal_energy` to `0.000000e+00` at every sample; the window reproduces
`1.907462e-01`, `7.513319e-02`, `8.372025e-01`, `8.421646e-05`, `6.097830e-05`
and the two in-rollout repair rows to every digit of the relative-mode flow
measurement's tables; the replayed settings carry each chain's own recorded
dimensions (36 extension rows, 8 retained rows, 56 state, 70 frontier); sample
spacing `0.0025` throughout, 800 samples and 800 steps per chain.  The ledger
row does not move (no rollout this round).  Artifacts in
`paper_v_projective_dictionary_energy_20260915e/`:
`probe_repair_displacement_20260918.py`,
`tabulate_repair_displacement_20260918.py`,
`append_repair_displacement_20260918.py`,
`repair_displacement_{base,recorded}_w68_20260918.{json,log}`,
`repair_displacement_{base,recorded}_sweep_20260918.{json,log}`,
`repair_displacement_base_dykstra_20260918.{json,log}`,
`repair_displacement_tables_20260918.log`.  Next: the per-coordinate
projection scale, in that order --- module change, free replay on the stored
`(6, 8]` samples, then the one 15-minute `t=4` to `8` segment.

## 2026-09-18: projective route, task 13: the task-11 configuration with the Eq. (22) energy hyperplane disabled, `t=0` to `t=20` --- the first 27-key chain that holds both cones

Report: `agent_guidance/paper-v/overnight/projective-route-20260915b.md`
(heading "projective-route, task 13 (2026-09-18)").  The task-11
configuration --- `LiftReselectionRule(acceptance_margin=1e-3,
candidates="measured", anchor_weight=0)` with no trust radius and
`--dykstra-iterations 120` --- on the 27-key `N=16` chain with
`--energy-reference-mode none`, base `t=0` to `20` in four `ram_guard
--limit-mb 4500` segments of `1,000 s` autonomous budget (boundaries
`4.72`, `10.64`, `17.0`; `3,514.0 s`, longest `1,006.2 s` plus `18.8 s` of
build) and its re-selection `conic_tolerance 1e-10` twin to `t=8` in two
(`1,589.1 s`) --- the `energy_tolerance 1e-3` twin measures nothing with
the hyperplane off --- one heavy process at a time, `11:57` to `13:27`,
nothing else heavy on the machine.  Smoke gate first (`t=0` to `0.08`):
`summary.json` records `energy_reference_mode none`, `dykstra_iterations
120`, ladder rung `0`, the radius-free measured rule, no declared
reference, 0 constrained repairs and 0 hyperplane refusals.  Imports from
`head_src_e3374f6f`, identical to `paper_5/src/paper5` at `HEAD 45e2ac0a`
outside `__pycache__`; `run_chain.py`, `run_energy_segment.py` and
`score_energy_chain.py` are task 8's unchanged; no module edited, nothing
committed, no default, rule, radius, Dykstra count, reference mode or
ladder changed.  The chain uses only its own state and the Hamiltonian's
declared drive; no exact or classical quantity enters the propagation or
the repair.  Scores are the all-coordinate scalar RMS of the 31 closed
scalar coordinates against the exact cutoff-16 driven reference from `t=0`
on one grid (`exact_t20.npz`, 8,001 samples; `exact_t8.npz`, 3,201).

Result first, because it is the one the round was run for: **a cheap-route
chain holds both cones over the whole horizon.**  The minimum unshifted
retained joint Gram eigenvalue over `t<=20` is `-1.549433e-12` at
`t=16.9575` and the minimum scaled positive-extension eigenvalue
`-1.349691e-09` at `t=0.0375`, both at the scale of the arithmetic, against
the shallowest constrained chain on record (the task-11 base, `-4.460e-4`
at `11.04` and `-3.936e-5` at `9.4975`); the retained cone is held at 5,387
of 8,001 samples and the extension cone at 5,993 (task-11 base 204 and
1,493), and past `t=8` at 3,442 and 3,666 of 4,800, where every constrained
chain on record violates the retained cone at all 4,800; mean negative part
`-6.535e-15` / `-4.698e-12`.  The twin holds both to `t=8` as well
(`-6.957e-13` / `-9.832e-08`).

Negative results, which are the price.  (1) **It is the worst 27-key chain
on record at every horizon from `t=8`**: `1.205891e-2` / `4.852761e-2` /
`1.142902e-1` / `2.148351e-1` / `3.563862e-1` at `t=4` / `8` / `12` / `16`
/ `20` (`t=1` `1.427025e-3`, `t=2` `3.159332e-3`, `t=6` `2.893705e-2`,
`t=10` `7.846593e-2`), behind the same configuration with the hyperplane on
by 1.298, 1.469, 2.815, 4.660 and 6.360, behind the ball band by 2.93 to
3.01 at `t=12` and 4.42 to 4.68 at `t=16`, behind the recorded
`drive_power` band at `t=20` by 3.92 to 4.18 and the `post_pulse` band by
4.71 to 5.10, and `2.771` times the both-cones `0.128607` at `t=20`: not a
closure there under any pairing.  At `t=4` it is still inside the recorded
`drive_power` band `1.143e-2` to `1.292e-2` and `0.959` of cell A's
`0.01258`, behind the task-11 four by 1.212 to 1.298.  (2) **The
internal-energy cost over the whole horizon is 257 times, not the 24.6
times task 12 measured over one window from a constrained state**: Eq. (22)
internal-energy RMS `4.565370e-2` / `1.440467e-1` / `4.594017e-1` /
`9.585083e-1` / `1.370935e+0` at the five horizons, `11.3` / `29.5` /
`89.4` / `182.1` / `256.9` times the hyperplane-on base's and `1,459` times
the both-cones `0.000939628` at `t=20`; the chain's own Eq. (22) internal
energy drifts from `-3.775326553` at `t=0` to `-0.621581991` at `t=20`
(`+3.153745`) where the constrained chains drift `+0.2095` and `+0.2106`.
(3) **Task 12's `(4, 8]` window result does not carry to a chain run free
from `t=0`**: the four phonon-occupation coordinates score `1.2926e-1` over
`(4, 8]` here against `5.1131e-2` for task 12's `b_energy_none` member,
which entered that window on the constrained chain's `t=3.92` state (2.53
times), and `2.849` times the recorded chain's `4.5378e-2` where task 12's
member was at `1.127`; the bias has task 12's sign (`+`, `+`, `-`) and ten
times its size.

The whole excess is the phonon-occupation block's.  That block scores
`1.773e-2` / `9.225e-2` / `2.855e-1` / `5.547e-1` / `9.136e-1` at the five
horizons (hyperplane on `8.425e-3` / `4.830e-2` / `6.531e-2` / `6.462e-2` /
`5.919e-2`); per window `(0,4]` to `(16,20]` `1.7740e-2`, `1.2926e-1`,
`4.7699e-1`, `9.9321e-1`, `1.7154e+0` against the exact reference's own RMS
in those four coordinates `3.8315e-1`, `3.6511e-1`, `3.2926e-1`,
`2.4762e-1`, `2.0570e-1`, so by `(16, 20]` the error is `8.34` times the
coordinate it is an error in; the signed means there are `+2.2555`,
`+1.6962`, `-1.8259`, `+0.5398` and at `t=20` the four coordinate errors
are `+2.5306`, `+2.2952`, `-2.2247`, `+0.7328` against exact values
`+0.2399`, `+0.2478`, `-0.2373`, `+0.0018`.  The 27 coordinates outside the
block are behind the hyperplane-on chain by only 1.166 to 3.158 per window
(2.816 over `(8, 20]`), and the site-0 occupation at `t=20` is `1.600e-1`
in the record's `2 rho[0,0]` convention (hyperplane on `1.361e-1`, ball
`1.624e-1`, recorded `2.252e-1`, both cones `0.205`).

What removing the hyperplane returns, besides the cones: **500 accepted of
500** re-selections (22 uncertified), no `infeasible`, no decline, no cap,
22 solver refusals (task-11 base 484 / 16 / 117), no answer anywhere above
`0.4` (task-11 base 15, of which 11 `infeasible`), three applied moves
above `0.1` all in `(0, 0.16]`, frontier envelope maximum `0.4541`
(task-11 base `0.8825`), and `3,514.0 s` to `t=20` against `5,183.6 s`,
**32.2 percent less wall** (solves `6.19 s` mean against `8.32`, repair
`355.5 s` against `895.4`).  The median applied frontier displacement rises
monotonically through the quarters (`0.0030`, `0.0098`, `0.0384`, `0.0419`,
`0.0498`) where the hyperplane-on chain's peaks at `0.0127` and falls: the
free flow leaves the completion more to do at every quarter.  The
two-member band is `1.206e-2` to `1.330e-2` at `t=4` (10.29 percent) and
`4.853e-2` to `4.946e-2` at `t=8` (1.92), narrowing with the horizon rather
than widening; the twin accepts all 200 at four times the base's solver
refusals (72 against 17) and never separates from it by `0.1` in the 31
coordinates over `t<=8`.

The 27-key ledger row's numbers do not move at any of the five horizons ---
this chain is behind the standing entry at every one.  What the row gains is
the measured price of the cones: the hyperplane is the only thing standing
between the Dykstra repair and its two targets, and buying the cones with
it costs 6.36 times the all-coordinate score and 257 times the Eq. (22)
internal-energy RMS at `t=20` while returning 32 percent of the wall and
every re-selection decision.  Holding the cones and holding the energy are
not independent settings of this route at 27 keys.  Gates: smoke gate
records the hyperplane off; imports from the `e3374f6f` snapshot, byte-identical
to `HEAD 45e2ac0a` and untouched by another session's `13:15` working-tree
edit to `cheap_projection_route.py`; every
`resumed_from`, rule, repair depth, reference mode and re-selection
tolerance recorded; 0 constrained repairs and 0 hyperplane refusals in every
segment; one grid (8,001 and 3,201 samples); ledgers merge to 500 and 200;
seven sequential rollouts under `4,500 MB`, longest `1,006.2 s`, peaks `624`
to `646 MiB`.  Artifacts:
`paper_v_dictionary_energy_none_measured_dyk120_smoke_20260918o` and
`paper_v_dictionary_energy_none_27key_n16_measured_dyk120_{base_seg1..4,conic1e10_seg1,conic1e10_seg2}_20260918o`
(each with `reselections.json`); in
`paper_v_projective_dictionary_energy_20260915e/`: `smoke_task13.sh`,
`check_task13_smoke.py`, `queue_task13.sh`, `tabulate_task13.py`,
`peek_task13_scale.py`, `update_plan_row_task13.py`,
`task13_base_score_t20.json`, `task13_conic1e10_score_t8.json`,
`tabulate_task13.log`, `smoke_task13.log`, `smoke_task13_check.log`,
`peek_task13_scale.log`, `task13_chain.log`, `queue_task13.log`, the
segment logs.  Plan-file row updated in place.  Next: a declared reference that is a
slab rather than a point.  The existing `energy_tolerance` cannot supply it
--- in `cheap_projection_route.py` it gates only whether a repair runs when
both cones are already satisfied, and the hyperplane handed to `_dykstra` is
still the equality `reference - energy_before`, which is why task 11
measured a maximum deviation of `4.44e-15` at `energy_tolerance 1e-3` --- so
the missing measurement needs the projection target to be
`|E - reference| <= tau` with `tau` of the size of the cone violation the
hyperplane now forces (`1e-4` to `1e-2`): a named, default-off option and a
default-path test on the repair, outside this round's writable surface.
Then a dictionary key whose extension row carries the phonon normal moments;
and the three twins of the task-11 configuration from `t=8` to `t=20` for
the `t=20` band.


## 2026-09-18: cheap projection route: the declared per-coordinate metric of the least-squares map back -- implemented as a named option, swept on the stored `(6, 8]` samples, and refused: the affine Gram map is injective at `target="both"`, and where the metric can act it triples the displacement it was meant to cut

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(heading "The declared per-coordinate metric of the map back: implemented,
swept, and refused at the running configuration (2026-09-18)").  The
repair-displacement measurement ended on the instruction *the per-coordinate
projection scale, in that order --- module change, free replay on the stored
`(6, 8]` samples, then the one 15-minute `t=4` to `8` segment*.  This round is
the module change and the free replay; the replay refuses the fix and the
segment is not spent.  Light round: stored trajectories and affine algebra
only, eight processes under `ram_guard --limit-mb 2500` one at a time, peaks
`1 MiB` to `465 MiB`, nothing propagated, no conic solve, nothing committed,
no default, rule, radius, Dykstra count, reference mode, tolerance or ladder
changed, `projected_apcm.py` and `apcm_projective_event_guard_analysis.py`
untouched.  Configuration is tasks 11's and 12's: 27 active keys, 56
propagated state coordinates, 70 standardized frontier coordinates, 36
extension rows, 8 retained rows, `N=16`, strong coupling, cutoff 16, step
`0.0025`, `policy="dykstra"`, `cadence="stage"`, `target="both"`,
`dykstra_iterations` 120 on the task-11 base and 30 on the recorded
`drive_power` chain, 800 stored samples in `(6, 8]` per chain, one replayed
repair at each.  The quantity is the relative-mode population
`(phonon[0,0] + phonon[1,1])/2 - Re phonon[0,1]` of the centred phonon
density, read through the scorer's chart.

**(a) What was implemented.**
`CheapProjectionSettings.raw_coordinate_motion_scales`, default `()`, each
entry naming one of the 29 raw moment coordinates and a positive factor that
multiplies `state_motion_scale` on that coordinate;
`RAW_PHONON_NORMAL_COORDINATE_NAMES` names the four raw phonon normal ones.
It sets two things: the metric of the least-squares map back
(`AffineGramCoordinates.state_coordinate_scales`, the map back still
`increment = scales * (pseudo_inverse @ residual)` with `pseudo_inverse =
pinv(design * scales)`), and the direction inside the affine Gram set along
which the Eq. (22) energy hyperplane is met
(`AffineGramCoordinates.energy_correction_direction`, `scales**2 * gradient`
carried through the same affine map and reduced onto the range basis, handed
to `_dykstra` as `hyperplane_direction`; `None` for an empty declaration, and
the committed branch then runs).  It leaves alone the propagated velocity,
both cones and their declared floors, the eigenvalue clip, the affine Gram set
and its range basis (built from the unscaled design), the Dykstra count and
its correction bookkeeping, the declared reference and the tolerance at which
a repair fires, the lift re-selection, and the energy the repaired coordinates
carry (measured on the reference to `6.1e-16` per repair at every declared
value).  The alternation stops being single-metric -- the cone is still
projected onto in the Frobenius metric -- so cone attainment is measured, not
guaranteed.  The carried-witness lane refuses a declared metric.  Four tests
added, 48 pass (44 before).  Gate: a 4-step `drive_power` rollout (15
entrance keys, cutoff 16, `dykstra_iterations` 30, 8 repairs, all 8
energy-constrained) is bitwise identical under the changed module and under
the module at `94cf93b5`, which is the module at `e990cbb0` unchanged --- all
seven stored arrays, maximum difference `0.000000e+00`.

**(b) The sweep, and the two negative results it returns.**  *The map back has
no freedom to weight at this configuration*: the realified design over the 126
movable coordinates has rank 126 at `target="both"`, so the correction is
unique and every positive weight returns it.  (At `target="retained"` rank 29,
the 97 unseen directions being the 27 entrance and 70 frontier coordinates the
retained Gram does not read; at `target="extension"` rank 111, three of the 15
unseen directions lying wholly inside the four raw phonon normal
coordinates.)  With the hyperplane off every declared value reproduces the
empty declaration to every printed digit: base `4.681371e-05` per repair at
`target="both"` (correction `2.096e-04`, both cones `0.134`) and
`8.310995e-08` at `target="retained"`, recorded chain `9.368395e-04`
(correction `3.790e-03`).  This is the same class of result as
`state_motion_scale`'s inertness and a stronger reason for it.  *Where the
metric can act it makes the displacement larger*: on the base over `(6, 8]`,
`1.502784e-04` per repair committed against `2.532811e-04`, `4.021687e-04`,
`4.481513e-04`, `4.486544e-04`, `4.486596e-04` at declared `10`, `1`, `0.1`,
`0.01`, `0.001` (saturated by `0.1`; signed means turn positive at `1` and
below, integrated `+3.199701e-01` to `+3.589276e-01`), and on the recorded
chain `5.264001e-04` against `4.078098e-03` to `8.756030e-03`.  The share of
the correction landing on this one direction does fall, `0.771` to `0.362`,
and both-cone attainment rises `0.010` to `0.080`--`0.100` with the retained
cone `0.010` to `0.126`--`0.149`; but the correction lengthens `2.131e-04` to
`8.628e-04` (`4.05` times), because meeting the energy along a
coordinate-cheap direction is far in the Frobenius metric the cones are
projected in, and the product is `2.99` times the displacement the fix was
meant to cut by `1.78`.  Declaring the metric over eight names --- the four
raw phonon normal plus the four coherent coordinates the centring
`N = raw_normal - alpha alpha^dagger` brings in --- gives `4.580947e-04`, so
the coordinate set is not what fails.  Method note: a stride dividing the
chain's `lift_reselection_interval` 16 samples one phase of the re-selection
cadence; at stride 16 the committed setting reads `3.145754e-06` per repair
against `1.502784e-04` at stride 1, a factor `47.8`.  Every figure here is
stride 1 over all 800 samples; the earlier round's stride-8 sweeps compare
settings correctly but their absolute per-repair figures do not transfer to
the window.

**(c) Recommendation: adopt no value, keep the empty declaration.**  The
pre-stated pass condition --- the per-repair displacement below the closure's
own `8.421646e-05`, the both-cone fraction not below `0.010`, the energy still
on the reference to `1e-15` --- fails on the first by `2.99` (base) and `16.6`
(recorded) at every value while the other two improve.  The decisive rollout
(task 12's `t=4` to `8` rerun from the base's stored `t=3.92` state, one
segment under `--limit-mb 4500`, about 15 minutes on the `exact_t8.npz` grid,
improving only below `1.907462e-01` for the relative-mode population's
`(6, 8]` RMS and `8.278358e-02` for the four phonon-occupation coordinates)
is not spent, and the option is deliberately not exposed on the analysis
driver's command line.  What the measurement points at instead is a slab
rather than a hyperplane: correct the Eq. (22) energy only when it is further
than a declared band from the reference, and then only to the boundary.
`energy_tolerance` cannot do it --- it gates whether a repair runs, and a cone
is violated at `0.990` of the base's stored samples, so the repair runs and
the hyperplane lands the energy exactly on the reference, which is the reading
the projective route's task 13 records.  It is one place, the hyperplane step
of `_dykstra`, the offset soft-thresholded by the band; the two ends are
already measured (the hyperplane at every repair, `1.502784e-04` per repair at
both cones `0.010`; off, `4.681371e-05` at `0.134`, and task 13's whole-horizon
price of `6.360` at `t=20` and `256.9` in internal-energy RMS), and the free
replay of the stored `(6, 8]` samples scores the band at about `100 s` per
chain before any rollout.

Gates: the relative-mode population read from the stored 56 propagated
coordinates through the scorer's chart equals the one read from the stored
`archive_coordinates` to `0.000000e+00` and the affine Eq. (22) energy equals
the stored `internal_energy` to `0.000000e+00` at every sample of both chains;
the committed replay reproduces the repair-displacement measurement's table to
every digit (`1.502784e-04` / `-1.502784e-04` / `-1.202227e-01` / maximum
`5.846746e-04` / p99 `4.916299e-04` / negative `0.990` / both cones `0.010`,
and `4.681371e-05`, `8.310995e-08`, `5.264001e-04`, `9.368395e-04`); sample
spacing `0.0025` throughout, 800 samples per chain, stride 1.  The ledger row
does not move (no rollout this round).  Files: the module
(`raw_coordinate_motion_scales`, `RAW_PHONON_NORMAL_COORDINATE_NAMES`,
`state_coordinate_scales`, `energy_correction_direction`, `_dykstra`'s
`hyperplane_direction`, the two cache signatures, the carried-witness
refusal), its tests (four added), and under
`paper_v_projective_dictionary_energy_20260915e/`:
`probe_repair_displacement_20260918.py` extended with `--phonon-normal-scale`
and `--metric-coordinate` (additive; the first row of every table checks that
the earlier path is unchanged), `tabulate_phonon_scale_20260918.py`,
`repair_displacement_base_phononscale_{a,b}_20260918.{json,log}`,
`repair_displacement_base_phononscale_mapback_20260918.{json,log}`,
`repair_displacement_base_metricset8_20260918.{json,log}`,
`repair_displacement_recorded_phononscale_20260918.{json,log}`,
`repair_displacement_phonon_scale_tables_20260918.log`.  Next: the slab, in
that order --- the one-place module change, the free replay at a band of
`1e-3`, then the same 15-minute `t=4` to `8` segment.

## 2026-09-18: cheap projection route: the Eq. (22) reference as a slab -- a named band, default off, swept from `1e-5` to `1e-1` on two chains and run to `t=8` at `3e-4`: it cuts the replayed per-repair displacement `4.6` times and the rollout keeps `3` percent of that, because in a rollout the slab binds at every repair

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(heading "The Eq. (22) reference as a slab rather than a point: implemented,
swept, and run to `t = 8` at a band of `3e-4` (2026-09-18)").  The
declared-metric round ended on the instruction *a slab rather than a
hyperplane -- one place, the hyperplane step of `_dykstra`; free replay
first, then the same `t = 4` to `8` segment*.  This round is the module
change, the replay at nine declared band widths on two chains, and -- the
replay passing its pre-stated condition -- one `t = 0` to `8` chain of the
task-11 configuration at the band the replay selects.  Configuration is tasks
11's and 12's: 27 active keys, 56 state and 70 frontier coordinates, 36
extension rows, 8 retained rows, `N=16`, strong coupling, cutoff 16, step
`0.0025`, `policy="dykstra"`, `cadence="stage"`, `target="both"`,
`dykstra_iterations` 120 on the task-11 base and 30 on the recorded chain,
800 stored samples in `(6, 8]` per chain, stride 1.  The exact trajectory is
a reporting input only.

**(a) What was implemented.**  `CheapProjectionSettings.energy_reference_band`,
default `0.0`, the declared half-width of a band around the Eq. (22)
reference; zero is the declared equality of every earlier pass and a positive
value makes the third set of the alternation the slab
`|E - reference| <= band`.  It changes exactly one quantity: the offset
`AffineGramCoordinates._dykstra`'s energy step takes up, soft-thresholded by
the band -- zero inside it, the excess alone outside it, so the iterate lands
on the nearer boundary and no further -- taken along the same direction as
before and divided by the same pairing, so the slab is met exactly.  It
leaves alone the velocity, both cones and their floors, the clip, the affine
Gram set and its range basis, the map back and its metric, the factorization
and energy-functional cache signatures (the band is not a metric and does not
key them), the Dykstra count and bookkeeping, the declared reference, the
firing tolerance and the lift re-selection; what changes about the repaired
coordinates is that their internal energy is on the reference to the band
(`3.000000e-04` maximum over the chain below) rather than to the arithmetic
(`1.776357e-15`).  A negative band is refused, so is a positive band with no
`energy_reference_mode`, and the carried-witness lane refuses it outright.
Unlike the declared metric it is exposed on the command line
(`--energy-reference-band`), because the replay passes.  Five tests added, 53
pass (48 before).  Gate: a 4-step `drive_power` rollout is bitwise identical
under the changed module and under the module at `HEAD efade34d` -- all seven
stored arrays, maximum difference `0.000000e+00`.

**(b) The sweep.**  On the task-11 base the per-repair displacement of the
relative-mode population runs `1.502784e-04` (the equality), `1.399951e-04`,
`1.214396e-04`, `6.829136e-05`, `3.235795e-05` at bands `1e-5`, `3e-5`,
`1e-4`, `3e-4`, and `4.681371e-05` from `1e-3` upward, which is the
hyperplane-free repair to every printed digit; a refinement puts the sign
crossing between `2e-4` (`-4.519099e-06`) and `2.5e-4` (`+1.415659e-05`) and
the family's minimum at `2.5e-4` (`3.010093e-05`).  The family therefore
passes *below* both ends the record had measured, at `0.36` of the closure's
own per-step `8.421646e-05`.  Both-cone attainment rises monotonically
`0.010` to `0.121` at `3e-4` and `0.134` at saturation (retained cone `0.010`
to `0.311` and `0.388`), and the energy after the repair sits on the declared
band exactly.  On the recorded `drive_power` chain the same shape sits an
order of magnitude out -- `5.264001e-04`, `4.817265e-04`, `3.923560e-04`,
`1.446854e-04` at `1e-4`, `3e-4`, `1e-3`, then `9.368395e-04` from `1e-2` --
and never reaches the base's figure; each chain's crossing sits near the
energy its own unconstrained repair opens (`1.8006e-04` and `3.3023e-03`).
Method note: the offset the hyperplane step reads is not the stored sample's
own (`3.469447e-16`) but the energy the cone projection opens inside the
alternation, which is why the small bands separate at all.

**(c) The rollout, and what it keeps.**  The task-11 configuration at band
`3e-4`, `t=0` to `8` in two `ram_guard --limit-mb 4500` segments
(`1,935.3 s`), scored on `exact_t8.npz`: `9.314847e-03` at `t=4` and
`3.312528e-02` at `t=8` against the task-11 base's `9.289800e-03` and
`3.304213e-02` (`1.003` at both, that is `0.25` percent worse); the four
phonon-occupation coordinates' block `6.526520e-02` over `(4, 8]` and
`8.004316e-02` over `(6, 8]` against `6.779204e-02` and `8.278358e-02`
(`0.963`, `0.967`); the relative-mode population `1.844331e-01` over `(6, 8]`
against `1.907462e-01` (`0.967`), where the recorded chain stands at
`7.513319e-02`; Eq. (22) internal-energy RMS `4.472609e-03` at `t=8` against
`4.887963e-03` (`0.915`); retained minimum `-1.754855e-04` and extension
minimum `-1.587110e-05` against `-2.096395e-04` and `-2.137939e-05` (`0.837`,
`0.742`); the same rate of advance to within `0.7` percent (`4.13e-3` against
`4.16e-3` units per second).  So a `4.64`-times cut in the replayed
displacement buys `3` to `4` percent of the block, better physicality and a
better internal-energy RMS, at no wall, and closes none of the gap to the
recorded chain.  The chain's own samples say why: over `(6, 8]` its stored
internal energy is `3.000000e-04` from its reference in both mean absolute
value and maximum at all 800 samples, so in a rollout the slab binds at every
repair and the route meets an equality displaced by the band; its in-rollout
displacement per step falls only to `1.529396e-04` from `1.988844e-04` while
its signed mean grows to `-1.219112e-04` from `-8.853883e-05`.

Gates: the default path bitwise unchanged against `HEAD efade34d`; the
`head_src_efade34d` snapshot differs from `paper_5/src/paper5` in exactly the
two files this round edits; the one edit made while the chain ran compiles to
identical bytecode in all 77 code objects with one differing string constant,
a class docstring; the replay reproduces the declared-metric round's table to
every digit and reads the record's coordinate to `0.000000e+00` on all three
chains; the smoke gate records `energy_reference_band 0.0003`, 95 constrained
repairs, 0 refusals and maximum deviation `3.000000000006331e-04`; 53 tests
pass in `23.6 s`; five replay processes under `--limit-mb 2500` and three
rollout processes under `--limit-mb 4500`, one at a time, peaks to
`701 MiB`.  The ledger row does not move.  Files: the module, the analysis
driver, the tests, and under
`paper_v_projective_dictionary_energy_20260915e/` the probe's `--energy-band`,
`tabulate_energy_band_20260918.py`,
`tabulate_energy_band_chain_20260918.py`,
`gate_energy_band_default_path_20260918.py`,
`gate_docstring_only_20260918.py`, `queue_band3e4.sh`, the two run drivers'
flag, and the outputs; run artifacts
`paper_v_dictionary_energy_band3e4_measured_dyk120_smoke_20260918p` and
`paper_v_dictionary_energy_band3e4_27key_n16_measured_dyk120_base_20260918p_seg{1,2}`.
Nothing committed.  Next: record the Eq. (22) offset each repair actually sees
in the rollout (a declared, default-off counter), size a band from that
distribution, and run one `t=0` to `8` chain at it.

## 2026-09-18: cheap projection route: the repair's displacement of the relative-mode population is *not* the closure's own Eq. (22) energy drift -- there is no drift (`4.09e-16` per step against a reference the closure's flow carries exactly), the repair is asked to move no energy at any of 2,400 stage repairs per chain, and the displacement is the cone chase redirected

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(heading "Whether the repair's displacement of the relative-mode population is
the closure's own Eq. (22) energy drift, carried at the measured exchange
rate: measured on both chains and refuted (2026-09-18)").  The band round
ended on the instruction *record the Eq. (22) offset each repair actually sees
in the rollout*.  The identity this round was asked to test is that the
energy-neutral Dykstra repair's displacement of the relative-mode population
per step is how much Eq. (22) internal energy the closure's own flow loses or
gains against the declared reference over that step, divided by the measured
exchange rate of one unit of coordinate per unit of Eq. (22) energy -- which
if it held would put the remedy in the closure and nowhere else.
Configuration is tasks 11's and 12's: 27 active keys, 56 state and 70 frontier
coordinates, 36 extension rows, 8 retained rows, `N=16`, strong coupling,
cutoff 16, step `0.0025`, `policy="dykstra"`, `cadence="stage"`,
`target="both"`, `energy_reference_mode="drive_power"`, band `0.0`, metric
empty, `dykstra_iterations` 120 on the task-11 base and 30 on the recorded
chain, 800 declared steps in `(6, 8]` per chain, stride 1.  Stored
trajectories, one declared step propagated from each, and affine algebra only;
no module edited, no lift re-selected, no completion solved, nothing
committed.

**(a) The closure's own Eq. (22) energy per declared step.**  Propagating each
stored sample by one declared SSPRK(3,3) step of `0.0025` with the repair
disabled, the closure's velocity changes the Eq. (22) internal energy by
`1.783386e-12` per step in mean absolute value on the task-11 base and
`2.049781e-12` on the recorded `drive_power` chain, against a declared
reference the rollout advanced by `1.783207e-12` and `2.049630e-12`.  The
drift between them -- the quantity the hypothesis names -- is `4.091172e-16`
and `4.890532e-16` per step, maximum `1.776357e-15` and `2.220446e-15`, signed
means `+4.052314e-17` and `-3.497203e-17`, and it opens in no quarter-window
bin.  The reason is exact and is measured here for the first time: the
closure's own drive-free velocity changes the Eq. (22) energy by `4.352421e-17`
and `5.805252e-17` (maxima `2.775558e-16`, `3.330669e-16`) at every sample, so
`gradient . velocity(0, x, f)` vanishes and the declared `drive_power`
reference *is* the energy the closure's unrepaired flow carries.

**(b) The identity, refuted.**  The energy the chain's own repair actually
supplied over the step is `6.145084e-16` and `6.211698e-16`, so the
displacement the exchange rate predicts is `6.202037e-16` and `2.785760e-16`
against a measured `1.989131e-04` and `3.825613e-04`: ratios `3.207222e+11`
and `1.373274e+12`, correlations `+0.118448` and `+0.045096`, the same sign at
`0.395` and `0.4125` of the 800 steps, and a fitted slope through zero that
explains `-0.116248` and `-0.021179` of the measured variance.  The one-for-one
arithmetic is right; the quantity it multiplies is zero.  Replaying the step
with the declared repair at each of the three stages, at the stage reference
the rollout advanced, the offset each repair sees is `4.157785e-16`,
`3.652634e-16`, `3.202993e-16` (base) and `4.013456e-16`, `3.913536e-16`,
`3.924638e-16` (recorded), maximum `1.776357e-15` at all 2,400 stage repairs
of each chain, with the hyperplane applied at `0.9950`/`0.9975`/`0.9938` and
`1.0000` throughout and the energy changed by `6.257957e-16` and
`6.679842e-16`.  That corrects the band round's reading that the stage
increment had already opened more energy than `3e-4`: it opens nothing.

**(c) The sign, and what the per-step displacement is.**  It does not follow
the closure's energy drift, there being none: the recorded chain displaces the
coordinate `+8.755517e-05` per step (integrated `+7.004414e-02`) and the base
`-8.869670e-05` (`-7.095736e-02`) while both drifts sit at the floor.  What
sets it is the balance of the three stage repairs at the SSPRK(3,3) weights
`1/6`, `2/3`, `1`: stage signed means `+9.233469e-05`, `-1.982742e-04`,
`+2.826374e-05` on the base and `+1.006025e-03`, `-5.634084e-04`,
`+2.956400e-04` on the recorded chain, whose weighted combination reproduces
the in-rollout per-step displacement at correlation `+0.9999998` and
`+0.9999999`, slope `+1.000041` and `+1.000143`, residual `2.466784e-07` and
`2.431999e-07`, that is `9.58e-04` and `4.24e-04` of the series' own RMS.  The
second stage moves the coordinate negatively on both chains and the first and
third positively; the recorded chain's first stage is `1.79` times its second
and the base's `0.47` times.  The repair-displacement replay could not reach
this because it replayed at the accepted state, which is none of the three
stage inputs.

**(d) Repair-side remedies, and what is left.**  On the energy channel they
are exhausted definitively: there is no Eq. (22) offset for any repair-side
setting to trade against, at any stage, on either chain, and the declared
reference's level is a one-time cost rather than a per-step one (the
drive-power reference's `+5.614e-3` above the exact post-pulse energy).  The
closure's own drift is not a target either -- it is already exactly zero.
Running the same repair at the same stage inputs with the hyperplane out of
the alternation and discarding it, the cone chase opens `9.731785e-04` and
`4.320573e-03` of Eq. (22) energy per stage repair, positive at every one of
the 2,400, displacing the coordinate by `+1.081507e-03` and `+2.238899e-03`;
the hyperplane's annulment of that energy costs `-1.107399e-03` and
`-1.992811e-03` in the coordinate, tracking minus the energy opened at
correlation `+0.9802659` and `+0.9125455` and slope `+1.106743` and
`+0.453826` -- the offset sweep's `+1.009268` and `+0.448470`, now at the
working point.  The constrained net (`2.331650e-04` and `6.800476e-04` mean
absolute) is a near-cancellation of two terms `4.6`--`4.8` and `2.9`--`3.3`
times larger.  So the remaining lever is how far outside the two cones the
stage increment puts the state: the scaled retained joint Gram before the
stage repair is `-1.345722e-05` (minimum `-5.433718e-05`, below zero at
`0.9808`) on the base and `-2.643057e-04` (`-4.466120e-04`, `1.0000`) on the
recorded chain, the scaled positive extension `-2.245342e-04`
(`-1.059994e-03`, `0.8075`) and `-4.335943e-04` (`-1.138028e-03`, `0.9846`),
and both cones are reached after the repair at `0.0083` and `0.0000` of the
2,400 stage repairs.

Gates: the probe reads the record's coordinate and energy to `0.000000e+00` at
every step of both chains; the repaired step reproduces the chain's own next
stored state to `1.509903e-14` and `1.421085e-14` and its carried frontier to
`4.761816e-16` and `5.551115e-16` in the median, above `1e-9` only at the
`0.0625` and `0.0312` of steps carrying a lift re-selection; the endpoint
definition of the displacement differs from the relative-mode flow
measurement's quadrature definition by at most `5.08e-07` per step and that
definition reproduces its table to every digit; the repair's energy correction
plus the closure's drift is `2.664535e-15` and `2.220446e-15` at every step;
sample spacing `2.500000e-03` exactly; 53 tests pass in `22.9 s`; six probe
processes under `ram_guard --limit-mb 3000`, one at a time, peaks to
`161 MiB`, longest `191.1 s`.  The ledger row does not move.  Files: under
`paper_v_projective_dictionary_energy_20260915e/`,
`probe_closure_energy_drift_20260918.py`,
`tabulate_closure_energy_drift_20260918.py` and
`closure_energy_drift_{base,recorded}_w68_20260918.{json,log}`,
`closure_energy_drift_tables_20260918.log`; this report and the worklog block.
No module; nothing committed.  Next: the share of the cone projection's own
correction landing on the four raw phonon normal coordinates, measured at the
stage inputs rather than at the accepted samples (free, about `190 s` and
`70 s` per chain); then a dictionary key whose positive-extension row carries
the phonon normal moments.

## 2026-09-18: cheap projection route: the cone chase's own correction is `0.70` and `0.93` on the four raw phonon normal coordinates but the retained cone's `0.9864` displaces nothing, and reducing the declared step does not reduce the per-step displacement -- the retained violation is the accepted state's and does not scale, and the per-step quantity is a residue whose cancellation does not either

Report: `agent_guidance/paper-v/overnight/cheap-projection-20260915b.md`
(heading "Where the cone chase's own correction lands, and whether reducing the
declared step reduces the repair's displacement of the relative-mode
population: measured at the stage inputs on both chains (2026-09-18)").  The
closure energy-drift round named two things: the share of the cone projection's
own correction landing on the four raw phonon normal coordinates, taken at the
stage inputs rather than at the accepted samples, and the depth of the cone
violation the stage increment opens as the remaining lever.  This round
measures both.  Configuration is tasks 11's and 12's: 27 active keys, 56 state
and 70 frontier coordinates, 36 extension rows, 8 retained rows, `N=16`, strong
coupling, cutoff 16, `policy="dykstra"`, `cadence="stage"`, `target="both"`,
`energy_reference_mode="drive_power"`, band `0.0`, metric empty,
`dykstra_iterations` 120 on the task-11 base and 30 on the recorded chain, 800
declared steps in `(6, 8]` per chain, stride 1.  Stored trajectories, one
declared step or one stage increment propagated from each, and affine algebra
only; no module edited, no lift re-selected, no completion solved, nothing
committed.

**(a) Where the cone chase's correction lands.**  At the three stage inputs,
with the Eq. (22) hyperplane out of the alternation, the chase puts `0.7022` of
its squared correction length on the four raw phonon normal coordinates on the
task-11 base and `0.9336` on the recorded `drive_power` chain, against `0.0690`
and `0.0229` on the other 25 raw moment coordinates, `0.2288` and `0.0436` on
the 27 active dictionary coordinates and `0.0123` and `0.0024` on the 70
standardized frontier ones (per stage `0.6990`, `0.8952`, `0.6979` and
`0.9036`, `0.9772`, `0.9430`).  At `target="retained"` the figure is `0.9864`
with `0.0136` on the four coherent coordinates and exactly `0.0000` elsewhere,
at every stage on both chains -- the affine eight-row Gram of
`(1, b, b^dagger, sigma)` reads no other coordinate.  At `target="extension"`,
`0.6811` and `0.7742`.  With the hyperplane in, the declared repair's
correction puts only `0.0536` and `0.2198` on that block and `0.8091` and
`0.6872` on the dictionary coordinates: the annulment is taken up there.

**The share does not predict the displacement.**  The retained chase, `0.9864`
of it on the four raw phonon normal coordinates and on the recorded chain
`1.60` times longer than the extension chase's correction, displaces the
relative-mode population by `7.525961e-08` and `8.989114e-07` per stage repair
-- `2,556` times less than the extension chase on the recorded chain.  The
whole displacement is the positive extension's: `1.113000e-03` and
`2.297563e-03` against the both-cone `1.081507e-03` and `2.238899e-03`, `1.03`
times it on both chains.  Direction, not block, is what separates a correction
that moves this coordinate from one that does not.

**(b) The depth of the cone violation, at three declared steps.**  Taking one
first-stage increment from each stored sample at `0.0025`, `0.00125` and
`0.000625`, and the stored state itself as the control, the scaled retained
joint Gram before the stage repair reads `-1.468337e-05`, `-1.468297e-05`,
`-1.468279e-05` against `-1.468261e-05` with no increment added at all on the
base and `-2.651646e-04`, `-2.651628e-04`, `-2.651619e-04` against
`-2.651610e-04` on the recorded chain: observed power `+0.0000` on both chains
over both halvings.  That violation is the accepted state's, not the stage
increment's.  The scaled positive extension's is the increment's, at `+1.0628`
and `+1.0234` (base) and `+0.8802` and `+0.7234` (recorded):
`-5.216931e-04`, `-2.240551e-04`, `-8.481588e-05` and `-7.704522e-04`,
`-4.165343e-04`, `-2.506473e-04`.  The energy the chase opens falls at
`+0.8891`/`+0.7733` and `+0.3254`/`+0.1865`, its displacement at
`+0.9916`/`+0.9516` and `+0.7037`/`+0.5140`.

**(b) The per-step displacement does not follow.**  Taking the whole declared
step with the three stage repairs chained from the same stored samples -- which
at the chain's own `0.0025` reproduces the chain's own next stored state to
`0.000000e+00` and `8.437695e-15` -- the SSPRK(3,3)-weighted per-step
displacement is `1.988869e-04`, `1.494577e-04`, `1.616498e-04` on the base
(power `+0.4122` then `-0.1131`) and `3.826816e-04`, `4.811923e-04`,
`7.531224e-04` on the recorded chain (`-0.3305` then `-0.6463`), while the
closure's own per-step displacement is first order in the step at `+1.0002` and
`+1.0005` (`8.422905e-05`, `4.210889e-05`, `2.105304e-05` and `6.104977e-05`,
`3.051404e-05`, `1.525431e-05`).  The mechanism: the per-step quantity is the
residue of a cone chase and an annulment that each scale with the step but
whose cancellation ratio drifts through `1` (`0.958`, `1.030`, `1.161` on the
base), so the residue has no power of its own, and on the base the three stage
repairs stop opposing each other as the step falls (signed stage means `+`,
`-`, `+` at `0.0025`; all three negative at `0.000625`).

**(c) The step is not a lever.**  One halving buys a factor `1.33` on the base
where proportionality would buy `2`, the second gives it back, and on the
recorded chain every halving costs.  The ratio of the repair's per-step
displacement to the closure's own runs `2.36`, `3.55`, `7.68` and `6.27`,
`15.77`, `49.37`.  No declared step brings the per-step displacement below
`8.421646e-05`: the family's minimum is `1.494577e-04` at `0.00125` on the
base, `1.77` times it.  That step would cost `6,400` declared steps for
`t = 0` to `8` at the task-11 configuration's measured `0.647950 s` per step
(`5,183.6 s` for 8,000 steps at `0.0025`), that is `4,146.9 s` (`1.15 h`)
against `2,073.4 s`, for a displacement `3.55` times the closure's own at that
step against `2.36` at `0.0025`.  No such chain was run.

Gates: the chained step reproduces the chain's own next stored state to
`0.000000e+00` and `8.437695e-15` at all 800 steps; the stage-input replay
reproduces it to `1.509903e-14` and `1.421085e-14` and reproduces the closure
energy-drift round's stage and twin displacements to every digit; the
zero-increment control reads the per-coordinate metric round's `4.681371e-05`
and `9.368395e-04` with the hyperplane off as `4.681370e-05` and
`9.368393e-04`, and its committed `1.502784e-04` and `5.264001e-04` as
`1.497421e-04` and `5.255878e-04`, that round having replayed on a strided
subsample; the weighted per-step displacement is `1.988869e-04` and
`3.826816e-04`, the previous round's figures; the closure's own per-step
displacement reads `8.422905e-05` against the relative-mode flow measurement's
`8.421646e-05`, the difference being the trapezoidal quadrature's; sample
spacing `2.500000e-03` exactly; 53 tests pass in `22.71 s`; thirteen probe and
tabulator processes under `ram_guard --limit-mb 3000`, one at a time, peaks to
`160 MiB`, longest `243.0 s`.  The ledger row does not move.  Files: under
`paper_v_projective_dictionary_energy_20260915e/`,
`probe_cone_correction_share_20260918.py`,
`tabulate_cone_correction_share_20260918.py` and
`cone_correction_share_{base,recorded}_w68_20260918.{json,log}`,
`cone_step_scaling_{base,recorded}_w68_20260918.{json,log}`,
`cone_chained_step_{base,recorded}_w68_20260918.{json,log}`,
`cone_correction_share_tables_20260918.log`; this report and the worklog block.
No module; nothing committed.  Next: a dictionary key whose positive-extension
row carries the phonon normal moments, sized first at the same 2,400 stage
inputs by the extension twin's displacement and its depth below the extension
floor, against `1.113000e-03` and `2.297563e-03` and `-2.245342e-04` and
`-4.335943e-04`.

### Cluster 10737278 returned: the first complete guarded carried-witness trajectory, t=0 to 0.5

The retry of segment 1 finished on CHTC with `JobStatus 4`, `ExitCode 0`,
`13,175 s` of remote wall (`3.66 h`), `stop_reason final_time` at `t=0.5`.
Archive `881c8dec61e692bd443b0b5447dbbb41b6d423db6e14529462f214188319133d`
(`1,126,507` bytes) fetched by receipt and verified locally at
`output/local_runs/paper_v_cwrmf_accel_chtc_10737278/`.  Balanced profile,
event-triggered guard at hard floor `2e-4`, two step subdivisions,
`max_threads_8`, default bundle seed policy, `h=0.0025`, strong coupling,
cutoff 16, exact-correlated preparation.

| quantity, `t=0` to `0.5` | value | comparison |
|---|---:|---|
| dynamic all-coordinate scalar RMS against the exact cutoff-16 reference | `3.195379e-5` | the record's strict always-on figure over the same horizon is `3.19169e-5` |
| dynamic blockwise rho / B / N / A / C | `4.89e-7` / `1.57e-5` / `4.38e-6` / `7.02e-5` / `8.47e-6` | --- |
| dynamic site-0 occupation RMS | `3.19e-7` | --- |
| minimum retained joint-Gram eigenvalue | `-7.27e-16` | positive to roundoff |
| minimum carried unshifted 62-row Gram eigenvalue | `-1.8386e-4` | inside the declared hard floor `2e-4` |
| maximum completion correction norm / readable-rate residual | `3.857e-3` / `1.540e-2` | --- |
| autonomous wall / peak resident | `13,154 s` / `4.41 GB` | the local resume needed `14,722 s` for `0.38` time units |
| tier seconds: Clarabel / retraction / eigendecomposition | `12,957` over 372 calls / `82.1` over 80 / `1.85` over 1,199 | 98.5 percent Clarabel |
| maximum atom wall / mean active critical modes | `367.5 s` / `25.9` | maximum 62 |

Reading: this is the first guarded carried-witness trajectory to run `t=0` to
`0.5` in one job from the preparation.  The record's local attempts stopped at
`t=0.08` (strict) and `t=0.2` (balanced) before the depth bound existed, and
the local bounded-depth run reached `t=0.5` only as a resume from `t=0.1175`.
Per time unit the remote job costs `26,309 s` against the local
`38,742 s`, a factor `1.47` that is the `max_threads_8` profile.

Segment 2 (`accel_s02_t0500_t1000`, `t=0.5` to `1.0`, checkpoint transferred
from this segment) was generated from the returned receipt, validated, dry
run and submitted as cluster `10988383` at 2026-09-20T17:07:09Z; receipt
`chtc/paper_v_cwrmf_event_guard_accel_20260915/submission_receipt_10988383.json`.
The shipped planned grid `submit_cwrmf_accel_s02.sub` to `s08.sub` was retired
to `planned_grid_retired/` on both this checkout and the submit node: each
names the failed planned run id's checkpoint, which never existed, and each
blocked the generator's run-id reuse check.

## 2026-09-20: does the corrected archive closure ever beat exact propagation?

Four measured studies, a crossover model and three adversarial checks per
claim (reports `agent_guidance/paper-v/overnight/scaling-{dimension-growth,
conic-cost-law,exact-cost-law,repair-overhead}-20260920.md`; scripts and
outputs under `output/local_runs/paper_v_scaling_20260920/`).  The question
is the one the route's cost raises: at two sites the exact cutoff-16 driven
reference reaches `t=20` in `3.064 s` measured while the corrected closure
takes `5,183.6 s`, so the construction earns its cost only if its own cost
grows slowly in the site count `L` while the exact state grows as
`L^2 (M+1)^L`.

Measured, and not in dispute: the exact three-site driven reference was built
and propagated (state dimension `44,217`, `300,840` nonzeros) and reaches
`t=4` in `19.02 s` end to end, `137 s` to `t=20` under its fitted law; the
conic program's dense block is `74.8 (n(2n+1))^2` bytes at `n` extension
rows, which caps the `18 GB` aggregate contract at `87` rows; and the
two-site closure wall decomposes exactly as `500` completion solves at
`8.3249 s` over a re-selection share of `0.8030`, reproducing the recorded
`5,183.569 s`.

Modelled, and explicitly not measured: there is no three-site closure.  Every
Paper V module is dimer-specific, `MomentKey(spin_up, spin_down, x_power,
p_power)` cannot name a word in a second boson mode, and the one-term Pauli
product law is the `su(2)` identity `d_abc = 0`, which fails at `L>=3`.  The
`L>=3` dimensions are a counting model written this round
(`count_l_site_construction.py`), whose `L=2` row reproduces sixteen
committed quantities and whose counted and closed-form blocks agree at
`L=2,3,4,5`; the conic wall at those dimensions extrapolates a law fitted
over cone side `32` to `104` whose local exponent rises `3.24` to `4.17` to
`4.87` inside that window.  Five of the synthesis's six claims were refuted
on exactly this ground by the verifiers, who upheld the direction and
rejected the precision: the memory law and its `87`-row cap stand, the
three-site exact cost stands as measured, and the closure side at `L>=3` is
extrapolation.

Reading.  Under every cost law on record the corrected closure is `1.8e3` to
`5.1e4` times more expensive than exact propagation at every site count
either method can be run, and the earliest crossing any law supports is
`L=10` at a closure wall of `5.6e12 s`.  The cause is the dictionary, not the
solver: the entrance predicate pairs every electronic generator across both
spin species with every degree-two Weyl monomial over all `L-1` modes and
refers to locality nowhere, so the extension rows go quartic to quintic in
`L` (`36` at two sites, a counted `281` at three, `1,192` at four).  Holding
rows linear in `L` instead moves the modelled crossing to `L=5` or `6` with
the conic block at `2.4`, `7.6` and `18.5 GiB` at three, four and five sites,
inside or at the edge of the memory contract at exactly the site count where
the crossing happens; one verifier holds that rows cannot be held linear in
this construction, which is the open question.

Next, and cheap: re-run the same enumeration with a locality-restricted
entrance predicate, keys supported on a single site and its chain neighbour
only, at `L=2,3,4,5`, reporting extension rows, frontier keys and whether the
frontier closes under the full structure-constant product.  Counting only, no
rollout, no conic solve.  It decides the route either way without the
rewrite.

## 2026-09-20: can the entrance predicate be restricted to one bond?

The counting the previous block named as next, done (report
`agent_guidance/paper-v/overnight/scaling-locality-20260920.md`; scripts
`count_locality_restricted.py`, `summarize_locality.py` and outputs
`locality_counts_L2_L5.json`, `locality_reading.json` under
`output/local_runs/paper_v_scaling_20260920/`).  Counting only: nothing
propagated, prepared, solved or promoted, no module under `paper_5/src`
edited, no default changed.

The restriction, stated as a restriction of the committed predicate: write
the electronic one-body Hermitian basis bond-adapted --- identity,
`S_ij = E_ij + E_ji` and `A_ij = -i E_ij + i E_ji` for `i < j`, and
`D_q = E_qq - E_{q+1,q+1}` for each bond, which is `(I, X, Y, Z)` at two
sites --- and keep only the keys `ENTRANCE_RELATIVE_MOMENT_KEYS` admits for
which one bond carries both spin-species generators and the Weyl monomial
uses no coordinate of any other mode, interacting mode `q` identified with
bond `q`.  At `L = 2` the restriction removes nothing: archive-resolved keys
`14`, entrance keys `15`, two-spin/one-boson promotion keys `12`, the 27-key
dictionary `27`, hidden keys `31`, base half-words `9`, extension rows
`24 / 36 / 40`, lower keys `29 / 41 / 45`, frontier keys `62 / 70 / 129`,
propagated state coordinates `44 / 56 / 60` and realified positive-extension
cone side `48 / 72 / 80` all equal the committed numbers, with no departure.
Products carry the full structure constants (an exact linear solve against
the whole `L**2`-element basis); the maximum number of basis elements in a
product of two generators is `1`, `3`, `4`, `5` at `L = 2` to `5`, so the
one-term law of `_pauli_product` is confirmed as the `su(2)` identity
`d_abc = 0` and is not used above two sites.

Counted, 27-key analogue: active keys `27 (L - 1)` = `27`, `54`, `81`, `108`;
extension rows `35 L - 34` = `36`, `71`, `106`, `141`; realified
positive-extension cone side `70 L - 68` = `72`, `142`, `212`, `282`;
propagated state coordinates `56`, `137`, `260`, `437`; frontier keys `70`,
`1,011`, `2,418`, `4,334`.  So extension rows *can* be held linear in `L`,
and the verifier's claim as stated is refuted.

Decisive, and against the route: the restricted dictionary is not invariant
under the Holstein commutator.  One commutator step with the `8 (L - 1)`
chain words demands `84`, `173` and `267` moments of degree at most three at
`L = 3`, `4`, `5` that the restricted dictionary does not carry, and every
one of them is supported on more than one bond.  The transitive hull of that
demand is `27`, `228`, `975`, `2,988` keys, successive exponents `5.26`,
`5.05`, `5.02` --- the declared dictionary's own quintic law --- and a stable
`81 %` of the whole declared enlargement at `L = 3`, `4`, `5`.  At two sites
the same iteration on the 15-key entrance dictionary adds exactly the twelve
two-spin/one-boson keys and stops at `27`, reproducing the committed
promotion block.  The Gram frontier agrees from the other side: `0 %`,
`85.4 %`, `91.0 %`, `93.2 %` of it is supported on more than one bond, the
defect being `541`/`1,336`/`2,397` cross-mode Weyl monomials,
`207`/`588`/`1,158` electronic factors spanning three or more sites,
`100`/`246`/`438` generator-and-monomial-on-different-bonds and
`15`/`30`/`45` two-species-generators-on-different-bonds at `L = 3`/`4`/`5`.

Memory settles it independently.  At `74.8 (n (2 n + 1))**2` bytes the
`18 GB` aggregate contract admits `87` extension rows (`16.15 GiB`; `88` rows
is `16.91 GiB`).  The counted linear law fits at three sites (`71` rows,
`7.18 GiB`) and nowhere above: `106` rows and `35.52 GiB` at four,
`141` rows and `110.96 GiB` at five.  The modelled crossing under that same
law moves to `L = 6` (single-power exponent `3.274`) or `L = 7` (two-term
law), needing `176` and `211` rows, `16x` and `33x` the contract; the
previous block's `L = 5` or `6` crossing does not survive, and even the
`18 L` law it was stated for needs `90` rows and `18.49 GiB` at `L = 5`,
above the ceiling.  At the one runnable site count above two, three sites,
the modelled closure wall is `4.79e4` to `1.64e5 s` against the measured
exact three-site driven reference's `137 s` to `t = 20`, `3.5e2` to `1.2e3`
times behind.

Reading.  The route is dead at this construction and locality does not
revive it.  A dictionary with linear rows is not a closure but a truncated
equation of motion; closing it restores the quintic count; and the memory
contract caps the construction three or four sites short of any crossing on
record.  Nothing further to size before an `L`-site closure, because an
`L`-site closure of this construction is not worth writing.

## 2026-09-20: the cheap repair on the initialization that actually diverges

Every run of the 2026-09-14 to 2026-09-19 rounds prepared from the exact
correlated ground-state contractions.  The divergence the repair of the
archive moment closure exists for is the Hartree--Fock electronic density
matrix with every other correlation variable zero, which
`stability_quantum_workflow_working_notes.md` records as losing electronic
positive semidefiniteness at `t=15.5177` and reaching state magnitude `1e4`
at `t=130.47`.  This round runs the cheap repair from that initialization.
Findings, artifacts and reproduction are in
`agent_guidance/paper-v/overnight/divergent-initialization-20260920.md`;
drivers and rollouts are under
`output/local_runs/paper_v_divergence_20260920/` and
`output/local_runs/paper_v_divergence_hf_dyk120_20260920_seg{1..5}`.  Nothing
here is promoted evidence, no default or acceptance rule changed, and no
module was edited.

The divergent initialization is a pure product state --- the Hartree--Fock
determinant times one coherent state per local oscillator at `-g/omega_ph` ---
and building it as a state vector in the same truncated Fock space the exact
preparation uses reproduces `hartree_fock_closed_scalar_coordinates` to
`1.02e-10`.  Propagated unrepaired, the 13-coordinate Fan--Migdal system
reproduces the record exactly (`15.5177447`, `130.4674620`), and in the 31
closed scalar coordinates the same initialization loses the retained joint
Gram below `-1e-8` at `t=0.0124`, the boson-moment cone at `t=1.3227` and the
electronic cone at `t=27.9634`, reaching `1e4` at `t=54.4613`, where the exact
correlated initialization never reaches it through `t=140`.  The harness also
reproduces the record's two Eq. (112) exact-contraction rows to every printed
digit (`1.8592243`, `0.0251665`, `2.9068e-5`; and `-3.05033e-1`,
`5.51945e-2`, `1.97874`).

Decisive, and against the route as it stands: **the cheap repair cannot be
prepared at that initialization.**  Its one conic solve, the declared proximal
log-det completion at `t=0`, refuses at 15 and at 27 keys and at every rung of
the declared retry ladder (`logdet_shift` `1e-10`, `1e-8`, `1e-7`, `1e-6`,
`1e-4`, `1e-2`, `1e-1`; `logdet_weight` `1e-1`) with
`moment_extension_degeneracy: facial reduction removed every frontier
direction`.  The preparation is pure, so its retained joint Gram is exactly
singular --- rank 6 of 8, smallest eigenvalues `-4.44e-16` and `-4.21e-16`
against a largest of `8.405` --- and facial reduction leaves the completion no
free frontier direction; the log-determinant shift cannot restore one because
the reduction acts before the barrier is seen.  Because the preparation is a
state vector, every frontier moment is an expectation in it, and the
extension's frontier keys reach degree 5, so `moment_hierarchy(5)` supplies all
70 exactly; that exact lift is what the rollout below uses at `t=0` and it is
a genuine positive extension (unshifted retained joint Gram `-4.44e-16`,
scaled positive extension `-1.58e-15`).  Every re-selection after `t=0` is the
committed solve and the first one, at step 16, already moves the retained Gram
from `-4.10e-6` to `-5.79e-8`.

Run that way in the task-11 configuration (27 entrance keys, SSPRK(3,3) at
`h=0.0025`, re-selection every 16 steps, Dykstra 120 on the affine Gram sets
and the two cones, the Eq. (22) reference in `drive_power` form, the
anchor-free measured rule at margin `1e-3`), five chained segments reach
`t=20` in `4,296.9 s` with 500 re-selections, 44 solver refusals and a peak of
`654 MiB`.  **The electronic one-body density matrix never loses positive
semidefiniteness**: `-2.220e-16` at `t=0` and a minimum of `+1.447e-6` over
`0 < t <= 20`, with maximum `|x_i|` `1.9431`.  Neither Gram cone is held: the
retained joint Gram reaches `-5.561e-4` and the scaled positive extension
`-6.352e-5`, against the recorded exact-correlated chain's `-4.460e-4` and
`-3.936e-5` in the same configuration.  The all-coordinate scalar RMS of the
31 closed scalar coordinates against the exact driven trajectory started from
the same Hartree--Fock state is `1.0803e-2` at `t=4`, `3.7209e-2` at `t=8` and
`1.5368e-1` at `t=20`, with Eq. (22) energy RMS `1.1888e-3`; the recorded
exact-correlated chain's own figures are `9.290e-3`, `3.304e-2`, `5.604e-2`
and `5.336e-3` at `5,183.6 s`.

On the same initialization and horizon the unrepaired closure scores
`1.2957e-1` / `3.3200e-1` at `t=4` / `20` with the retained joint Gram at
`-0.55727`; the recorded energy-neutral joint electronic--bosonic matrix
barrier scores `2.0232e-1` / `3.4704e-1` at `793.2 s`, holds its own two cones
(`0` and `-5.103e-16`) and lets the retained joint Gram reach `-0.36128`; and
Eq. (112) alone scores `2.2891e-1` at `t=4` and then diverges, reaching `1e4`
at `t=15.3724` against the unrepaired `t=54.4613`.

Reading.  On the initialization the repair exists for, the cheap repair buys
exactly what the record asks of it in the electronic sector --- the positivity
loss at `t=15.5177` does not happen, and neither does the amplitude blow-up
--- at `12.0` times the unrepaired closure's accuracy at `t=4` and `2.16`
times at `t=20`, and it costs `2.74` times its accuracy at `t=20` relative to
the correlated preparation.  What it does not do is hold either Gram cone, and
that failure starts at the first step and is the Eq. (22) hyperplane's, not
the initialization's.  Two things are settled negatively and should not be
retried as stated: the declared log-det completion at a pure preparation, and
Eq. (112) as a regularization of the uncorrelated initialization.

## 2026-09-20: the cheap repair carried past both decisive horizons --- `t = 60` from the divergent initialization, and the basin test that separates prevention from postponement

Branch `claude/paper-v-routes-20260914`, worktree
`Holstein_test.worktrees/claude-paper-v-routes-20260914`, HEAD `fcbd391a`.
Exploratory local work, not promoted evidence; no default, acceptance rule or
module was changed.  Report
`agent_guidance/paper-v/overnight/divergent-initialization-20260920.md`
sections 8 to 12.

The round above stopped at `t = 20`, which is before every unrepaired failure
from the Hartree--Fock/zero-correlation initialization except the two Gram
crossings, so nothing in it separated preventing the divergence from being
measured short of it.  The same chain --- 27 entrance keys, SSPRK(3,3) at
`h = 0.0025`, the declared proximal log-det completion re-selected every 16
accepted steps, the Dykstra repair at 120 iterations, the Eq. (22) drive-power
reference, the anchor-free measured rule at margin `1e-3`, the `exact_moments`
lift at `t = 0` --- was resumed from its stored `t = 20` state and carried in
nine further segments to `t = 60`: 24,001 samples on one grid, `12,990.1 s`
autonomous wall, 1,500 lift re-selections of which 156 were solver refusals,
peak `656 MiB`.

**It prevents both failures.**  The electronic one-body density matrix is
`-2.220e-16` at `t = 0` and its minimum over `0 < t <= 60` is `+1.4466e-6`,
attained at the first step `t = 0.0025` --- the same sample and the same
number the `t = 20` rollout reported, so nothing between `t = 20` and `t = 60`
comes nearer the cone than the preparation's own first step.  The unrepaired
closure loses that cone at `t = 27.9634` and reaches magnitude `1e4` at
`t = 54.4613`; the largest absolute coordinate anywhere in the repaired chain's
`t = 0` to `60` is `2.1827`, at `t = 51.0675`, and the state's own maximum
absolute coordinate at `t = 60` is `1.3080`, below its `t = 20` value.  Its all-coordinate scalar RMS against the
exact driven trajectory from the same product state is `1.9752e-1` at
`t = 28`, `2.1606e-1` at `t = 40` and `2.5235e-1` at `t = 60`, a factor `1.64`
over a factor `3` in horizon, with the Eq. (22) energy RMS flat at `1.19e-3`
to `1.21e-3`.  Neither Gram cone is held, at `t = 60` exactly as at `t = 20`:
retained joint Gram `-8.511e-4`, scaled positive extension `-9.532e-5`,
boson-moment eigenvalue `-3.391e-3`, each within a factor `1.6` of its
`t = 20` value and none deepening over `t = 55` to `60`.

**The basin test, which is what separates prevention from postponement.**
Started from the repaired chain's own `t = 60` state, the unrepaired archive
closure --- no repair, no lift, no re-selection --- stays bounded through
`t = 200`, 140 further time units, with minimum electronic eigenvalue
`+0.0685`, maximum `|x_i|` `1.8256` and no amplitude event, in `2.2 s`.  The
same closure from the *exact* trajectory's `t = 60` state is likewise bounded
(`+0.0302`, `1.7989`), and by the retained joint Gram the repaired state is
the less violating of the two (`-0.5250` against `-0.9567`).  From the
Hartree--Fock preparation at `t = 0` the same closure reaches `1e4` at
`t = 54.4613`.  The repair is therefore not holding a divergent flow back step
by step; it has delivered the trajectory to a state whose own closed flow does
not diverge.  That test propagates 140 time units of the unrepaired closure in
`2.2 s`, that is `0.0157 s` per time unit, against `217.3 s` per time unit for
the repair over `t = 20` to `60`, a factor of `13,800`, and it is the cheapest test of
whether the claim fails after `t = 60`: continue the chain and propagate the
unrepaired closure 140 units from each stored segment boundary; the claim
fails at the first boundary whose continuation fires the `1e4` event.

**Against the other two vector fields at the same horizons.**  The
energy-neutral joint electronic--bosonic matrix barrier keeps holding exactly
the two cones it constrains through `t = 39` (`+7.525e-14` and `-5.103e-16`)
while its retained joint Gram, not in its constraint set, goes to `-0.53867`
--- but **it cannot be carried to the magnitude horizon**: resumed from its
`t = 20` state it covers `t = 20` to `34.13` in `7.3 s`, then `2` units in
`315 s`, `2` more in `589 s`, and `t = 38.5` to `39.0` in a whole `1,000 s`
segment, about `0.5 s` per time unit before `t = 34` against about `2,000 s`
per time unit after `t = 38.5`.  The wall begins inside a segment, so it is
the trajectory's, not a resume artifact; `t = 54.4613` is roughly `31,000 s`
away and the chain was stopped at `t = 39.0`.  The unrepaired closure at
`t = 28` has electronic eigenvalue `-4.387e-3`, retained joint Gram `-1.0522`
and RMS `3.8429e-1`; at `t = 40`, `-1.6647`, `-11.064` and `4.5096e-1`.  At
`t = 28` the cheap repair is `1.95` times the unrepaired closure's accuracy
and `1.62` times the barrier's; at `t = 40`, `2.09` times the unrepaired
closure's.

Reading.  On the initialization the repair exists for, the question the
`t = 20` round left open is answered: the cheap repair prevents the
representability loss and the amplitude divergence rather than postponing the
horizon at which they are measured, and the basin test says why --- the
repaired state at `t = 60` is outside the divergent basin, as the exact state
at `t = 60` is.  Electronic positivity remains a measured outcome and not a
construction, since the electronic one-body density matrix is not one of the
sets the Dykstra repair projects onto.  The barrier's cost wall at `t ~ 34` is
new and should be treated as a limit on that route, not a transient.

## 2026-09-20: the basin test at all fourteen stored boundaries --- it measures the time to the `1e4` magnitude event inside a window, not membership of a bounded set, and the repaired `t = 60` state fires at `t = 410.7827`

Branch `claude/paper-v-routes-20260914`, worktree
`Holstein_test.worktrees/claude-paper-v-routes-20260914`, HEAD `6590840f`.
Exploratory local work, not promoted evidence; no default, acceptance rule or
module was changed.  Report
`agent_guidance/paper-v/overnight/divergent-initialization-20260920.md`
sections 13 to 15.

The round above ran the basin test at one boundary --- the unrepaired archive
closure, no repair, no lift, no re-selection, started from the repaired
chain's own `t = 60` state and propagated 140 further time units with the
`1e4` terminal event --- found no event, and read that as the repaired chain
having left the divergent basin.  This round ran the identical test at every
one of the fourteen stored segment boundaries (`t = 4.76`, `9.60`, `13.82`,
`18.60`, `20.00`, `24.44`, `29.12`, `34.43`, `38.6325`, `43.12`, `48.48`,
`52.71`, `56.92`, `60.00`), and from the exact driven trajectory's own state
at the same fourteen times, on the same DOP853 protocol at `rtol 1e-10`,
`atol 1e-12`, sample step `0.0025`, `lambda_ep = 1.5`, `gamma = 0.5`, drive
amplitude `1.0`, phonon cutoff `16`.  At `t = 60` both reproduce the previous
round's numbers to every reported digit.

**At 140 time units the event fires at none of the fourteen boundaries from
the repaired chain's state and at three of the fourteen from the exact
trajectory's state** (`t = 20` at `147.3861`, `t = 29.12` at `74.8594`,
`t = 43.12` at `152.9483`).  **But the `140`-unit verdict is not
boundedness**: from the repaired chain's `t = 4.76` state the maximum absolute
coordinate reaches `2807.5440` at the last sample of the window, the
electronic one-body density matrix minimum is `-4.6266e+01` and the retained
joint Gram minimum `-6451.5882`; that continuation is still growing when the
window ends, and at `400` time units it fires at `t = 150.9977`.

**Carried to 400 time units the test refutes the basin reading.** From the
repaired chain's `t = 60` state the unrepaired closure fires the `1e4` event
at `t = 410.7827`; from the exact driven trajectory's `t = 60` state it fires
at `t = 433.8964`.  Over all fourteen boundaries the event fires at seven from
the repaired state and at thirteen from the exact state, the sole exception
being `t = 24.44`.  A set the exact trajectory's own state does not lie in is
not a basin whose complement the repair could deliver the trajectory into, so
the quantity the test measures is the time from a state to the `1e4` magnitude
event under the unrepaired closure, and the previous round read a window
shorter than that time as membership.  Membership is also not monotone in the
boundary time --- from the repaired state at `400` units, `t = 52.71` and
`t = 60.00` fire while `t = 56.92` between them does not --- so there is no
time from which the chain is outside, at either window length.

**What the repaired state is, by count.** It fires less often than the exact
trajectory's state at both window lengths (none of fourteen against three at
`140`, seven of fourteen against thirteen at `400`), but the time from
boundary to event is mixed row by row: of the seven boundaries where the
repaired state fires within `400` units it takes longer than the exact state
at three, less long at three, and at `t = 24.44` it fires at `380.5341` where
the exact state does not fire at all.

**When the repair does its work, on this measurement.** At `140` units the
earliest boundary with no event is the second, `t = 9.60`, the first firing
once the window is opened; at `400` units it is `t = 20.00`.  Both readings
place the whole of the change early in the rollout, before either unrepaired
failure at `t = 27.9634` and `t = 54.4613`, and the `12,990.1 s` spent
carrying the chain from `t = 20` to `t = 60` changed this measurement no
further.

Cost `75.9 s` of propagation for the `140`-unit sweep at all fourteen
boundaries from both starting states, `182.3 s` for the two `400`-unit
sweeps, and one `1.5 s` propagation of the exact driven reference for all
fourteen of its states; peak resident `259 MB`.

Reading.  The previous round's measured statements about the repaired chain
over `t = 0` to `60` are untouched --- electronic one-body density matrix
minimum `+1.4466e-6`, maximum absolute coordinate `2.1827`, against the
unrepaired closure's electronic loss at `t = 27.9634` and `1e4` at
`t = 54.4613`.  What is withdrawn is the third argument it offered for calling
that prevention rather than postponement.  The repair is measured to hold the
electronic cone and the magnitude over the horizon it was run to; it is not
measured to have moved the state into a region where the unrepaired closure is
stable, because this test exhibits no such region, not even at the exact
trajectory's own states.  Any future use of the basin test must state its
window, and a window that does not exceed the times reported here decides
nothing.

## 2026-10-10: the archive right-hand side as an explicit polynomial, and the Carleman lifting layer measured against it

Branch `claude/paper-v-routes-20260914`, worktree
`Holstein_test.worktrees/claude-paper-v-routes-20260914`, HEAD `b9f99271`.
Exploratory local work, not promoted evidence; no default changed, no
existing module edited, nothing committed.  Report
`MATH/paper_facing/paper_V_high_u_gkba/quantum_integration_feasibility_20261010.md`;
new module `paper_5/src/paper5/stability/quantum_target_form.py`, tests
`paper_5/tests/test_quantum_target_form.py` (12 passed); scripts and outputs
under `output/local_runs/paper_v_quantum_target_form_20261010/`.

The committed `closed_scalar_rhs` was interpolated exactly on 5,984 integer
points (one per monomial of degree at most three) at the declared
configuration (`lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude `1`).

**The form is exactly quadratic.**  It reconstructs the committed right-hand
side on 1,500 fresh random states to `1.7e-14` relative.  The form is

`dx/dt = F_0 + (A_1 + f(t) D) x + A_2 (x (x) x)`.

- **No cubic part.**  The normal particle and normal hole sources of
  Eq. (14d) sum to `-i N [g^T, rho]`, so their `rho g rho` products cancel.
  `A_3` is zero to `1e-16`.
- **The drive enters only through a skew-symmetric `D`.**  The drive's
  constant and quadratic parts are `0.0` exactly.
- **Nonzero entries:**
  - `F_0`: 5;
  - `A_1`: 83;
  - `D`: 10;
  - `A_2`: 69 monomials, only in the `rho` equation (`(rho)(B)`) and the `C`
    equation (`(B)(C)`, `(rho)(N)`, `(rho)(rho)`, `(rho)(A)`).
- **Spectral norms:** `0.906`, `4.047`, `1` and `1.995`.

**The linear part is conservative, not dissipative.**

- **At the origin of the closed coordinates:**
  - every eigenvalue of `A_1` has `|Re lambda| <= 2.2e-16`;
  - `A_1` is semisimple, with eigenvector condition `55.4`;
  - it is skew-adjoint in a metric of condition `3,066`;
  - its Euclidean symmetric part is indefinite, from `-1.598` to `+1.598`.
- **At every state:** the field is exactly divergence-free.  `tr A_1`,
  `tr D` and the gradient of `tr J` are all `0.0`, so no expansion point and no
  linear change of coordinates gives every eigenvalue a negative real part.

**The Liu et al. condition fails.**  Their premise needs
`Re lambda_1 < 0`, and `R < 1` needs `Re lambda_1 < -4.952` at the exact
correlated preparation (`-4.100` at Hartree--Fock).  The measured value is
`+1.05e-16`.  Here `R` is
`(||x_0|| ||A_2|| + ||F_0|| / ||x_0||) / |Re lambda_1|`, the definition checked
against arXiv:2011.03185.  The paper's authors are Liu, Kolden, Krovi,
Loureiro, Trivisa and Childs.

**The truncation converges only over a short window.**  Two measurements
from the exact correlated preparation, integrated by RK4 at `h = 0.0025`
(step-checked to `1.8e-8`):

- **The dropped term is not small.**  At the stored exact states, the part of
  block `K`'s lifted velocity that the truncation drops has median `0.72` of
  that velocity over `t <= 4`.  This holds for `K = 2, 3, 4` alike.
- **The order-`K` solution error.**  Measured as the maximum absolute
  coordinate error against the closure's own trajectory:

  | `K` | `t <= 1` | `t <= 4` | `t <= 20` |
  |---|---|---|---|
  | 1 | `0.261` | `0.957` | `1.66` |
  | 2 | `0.0513` | `0.981` | `12.3` |
  | 3 | `0.0076` | `1.036` | `72.1` |
  | 4 | `0.0012` | `1.001` | `387` |

  At `K = 4` the lift has dimension `954,304` (Kronecker) or `52,359`
  (distinct monomials); 276 s, 214 MB.

The truncation converges in `K` up to about `t = 2`, stagnates at about `t = 4`
and diverges with `K` beyond.  The closure trajectory used as the comparison
reproduces the record's plain-archive RMS `1.10890e-1`, `1.38174e-1` and
`1.93196e-1` at `t = 4, 8, 20` to every printed digit.

Reading.  The fault-tolerant lifting layer is not admissible for the archive
equations as they stand.  The failing property is dissipativity, and the
failure is structural: the field is divergence-free.  The cubic term the draft
carries in Eq. (`eq:paper_v_force_polynomial`) and in the lifting layer is
identically zero.  The manuscript was not edited.

## 2026-10-10: the bare closure's instability, linearized --- converged, volume-preserving, chaotic on an unbounded energy shell, and not along the relative-mode population

Branch `claude/paper-v-routes-20260914`, worktree
`Holstein_test.worktrees/claude-paper-v-routes-20260914`, HEAD `b9f99271`.
Exploratory local work, not promoted evidence; no default changed, no module
under `paper_5/src` edited, nothing committed.  Report
`agent_guidance/paper-v/overnight/closure-instability-20261010.md`; scripts
and outputs under `output/local_runs/paper_v_instability_20261010/`
(`summary.json` holds every quoted number).  "The bare closure" is
`closed_scalar_rhs` at `lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude `1`.

Negative results first.  **The bare closure from the exact correlated
preparation does fire the `1e4` event, at `t = 141.4057`**, one and a half
units past the 140-unit window in which the 2026-09-20 report says "never"
(maximum `24.2421`).  **The record's `433.8964` from the exact trajectory's
`t = 60` state is the DOP853 `rtol 1e-10` value**; tightened it is
`434.1811` (`1e-12`) and `434.1878` (`1e-13`).  **The leading unstable
direction is not the relative-mode population**: it carries a mean `0.074`
(median `0.022`) of the leading Jacobian eigenvector's squared norm along
that trajectory, above one half at `1.65` percent of samples; the direction
that actually grows (the leading forward Lyapunov vector) is `0.355`
anomalous phonon density, `0.333` electron--phonon correlation and `0.045`
relative-mode population.

Not numerical, by the record's own criterion.  The `1e4` event converges
under RK4 at `0.01`, `0.005`, `0.0025`, `0.00125`, DOP853 at four
tolerances and Radau at all three starts: `54.4612958` from Hartree--Fock
(last tightening `1.8e-10`), `141.405885` from the exact correlated
preparation (`1.2e-6`), `434.19` from the exact `t = 60` state (`6.7e-3`).
What is left at `t = 60` is the trajectory's sensitivity: admissible
perturbations of `1e-12` and `1e-10` spread the event by `7.1e-3` and
`1.31`, against `2.2e-9` and `2.2e-7` from the Hartree--Fock preparation.
The existing Benettin machinery gives a largest finite-time exponent of
`0.0463`, `0.0467`, `0.0468` over `t = 60` to `400` at steps `0.01`,
`0.005`, `0.0025`.

What it is.  `Tr J` is zero at every state (at most `6.1e-13`; the
explicit-polynomial block above finds the same) and the Eq. (22) energy is
conserved through the event, so the flow is volume-preserving and has no
attractor; the energy is linear in the phonon density and the diagonal of the
correlation and does not contain the off-diagonal correlation entries at
all, and those entries are what reach `1e4` on all four bare starts.  On
that unbounded shell the motion is chaotic (largest exponent `0.047`, its
31 exponents summing to `-8.9e-8`), the orbit's amplitude creeps from `1.56`
to `3.14` over 350 units, and across 64 admissible `1e-4` perturbations of
the `t = 60` states the time to the event is `293` to `478` units (exact) and
`296` to `779` (repaired), never below `293`.  Past `1e4` the magnitude
reaches `1e10` in `0.25` to `0.33` units on three of four starts at both
tolerances; no single power law fits and the integration stalls past `1e10`
to `1e12`, so a finite-time singularity is not established.  The leading
real part is positive at every sample except the Hartree--Fock product state
at `t = 0` (purely imaginary spectrum), stationary at ten-unit means of
`0.15` to `0.22` over `t = 60` to `330`, and a poor predictor: `78.0`
integrated e-foldings over `70` to `425.6` against a realized `20.9`.

The repair, in `t = 0` to `20`.  The bare closure's Jacobian has the same
unstable mode at the repaired chain's states as at the exact trajectory's
(leading real part `0.177` against `0.211` averaged over `0` to `20`; bare
Hartree--Fock `0.206`), and the repaired route's velocity departure from the
bare field has a median share of at most `0.022` in the plane of the leading
eigenvector.  What differs is the growing direction: along the bare
trajectory it moves into the correlation block between `t = 10` and `20`
(share `0.20` to `0.49`, `0.57` over `20`--`30`) and grows at a mean `0.11`;
along the repaired chain it stays in the phonon blocks (correlation `0.05`
to `0.23`) and grows at `0.045`, while the relative-mode population is a
*larger* share of it there (`0.09` to `0.28`) than along the bare trajectory
(`0.01` to `0.05`).

For the quantum-feasibility thread, at `dt = 0.0025`: the condition number
of `I - dt J` is at most `1.038` wherever the maximum absolute coordinate is
below `5`.  The linearized flow amplifies by `11` to `14` over `T = 4`,
`17.6` to `51` over `T = 20` and `54` (exact trajectory) to `3,170` (bare
from the exact correlated preparation) over `T = 60`, supremum over all
start and end times in the window; from the exact `t = 60` state `5.2e4` over
200 units and `2.3e9` over 350.  The block system stacking those
backward-Euler steps has condition number `1.1e4` to `1.2e4`, `7.9e4` to
`2.0e5` and `5.2e5` to `1.16e7` at `T = 4`, `20`, `60`, against `2,038`,
`10,187`, `30,559` with `J = 0`.

Reading.  The divergence is a genuine dynamical instability of the closed
equations: converged, conservative, transiently chaotic on an energy shell
that the closure's own conservation laws do not bound, ending in unbounded
growth of the correlation entries the energy does not contain.  Only the
positivity of the state bounds those entries, and the closure does not carry
it.  The repair does not act on the bare closure's unstable mode; it keeps
the state, and with it the growing direction, out of the correlation block.

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

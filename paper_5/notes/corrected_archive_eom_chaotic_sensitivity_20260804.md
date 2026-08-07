# Chaotic Sensitivity of the Representability-Corrected Archive Moment EOM

Date: 2026-08-04

Status: completed exploratory diagnostic. The numerical artifacts retain the
status `exploratory_local_not_promoted`.

## Result

The autonomous post-pulse, representability-corrected 31-coordinate moment
equations exhibit bounded chaotic sensitivity under the tested strong-coupling
protocol. Two independent perturbation directions converge to the same
positive largest finite-time Lyapunov exponent through
\(t=1000\,t_{\mathrm{hop}}^{-1}\):

\[
\Lambda_{1000}^{(\mathrm{drive})}=0.02845896,
\qquad
\Lambda_{1000}^{(\mathrm{ph})}=0.02846732.
\]

The final two 250-unit windows remain positive:

\[
\Lambda_{500:750}\simeq 0.01802,
\qquad
\Lambda_{750:1000}\simeq 0.01891.
\]

The late rate \(0.01891\) corresponds to one perturbation-growth e-fold in
approximately \(52.9\,t_{\mathrm{hop}}^{-1}\). The trajectory remains bounded,
positive semidefinite under every monitored representability condition,
energy stable, and correlation-trace preserving.

## Vector field being tested

The retained state is the real 31-coordinate representation of

\[
X=(\rho,B,N,A,C),
\qquad x=\mathcal R(X)\in\mathbb R^{31}.
\]

The archive closure supplies the raw velocity \(f_{31}(t,x)\). At every
Runge--Kutta right-hand-side evaluation, the constrained quadratic program
adds its minimum-Euclidean-norm correction \(w(t,x)\):

\[
\dot x=F_{\mathrm c}(t,x)
=f_{31}(t,x)+w(t,x).
\]

The correction enforces the electronic and joint-Gram positivity barriers,
the fixed-particle-number correlation-trace identities, and zero instantaneous
correction-energy flux. This defines a piecewise-smooth corrected vector field
because the active positivity inequalities can change along the trajectory.

The original Gaussian pulse is applied through \(t=4\). The drive is then set
exactly to zero, so the system used for the Lyapunov calculation is autonomous:

\[
t\ge4:\qquad \dot x=F_{\mathrm c}(x).
\]

The raw archive flow loses joint-Gram positive semidefiniteness at
\(t=0.1607116782\) in this protocol. Its long uncorrected continuation therefore
does not supply the bounded representable trajectory required for this test.

## Initial state and perturbation directions

The initial 31-coordinate state is obtained by contracting the cutoff-16 exact
ground state at \(\lambda=1.5\), \(\gamma=0.5\), hopping one, and drive amplitude
one. Two nearby states are generated from orthogonalized wavefunction tangents:

1. the electronic site-imbalance operator coupled to the drive;
2. the relative phonon-position operator.

These exact-state perturbations have wavefunction amplitude \(5\times10^{-6}\).
The base and both nearby contractions are propagated by the corrected moment
EOM through the driven interval \(0\le t\le4\). At \(t=4\), each difference is
projected into the local tangent space defined by

\[
\delta E=0,
\qquad
\operatorname{Re}\operatorname{Tr}\delta C^0=0,
\qquad
\operatorname{Im}\operatorname{Tr}\delta C^0=0,
\]

and normalized to \(\varepsilon=10^{-5}\) in the Frobenius norm induced by the
matrix lift. For a coordinate difference \(\delta x\) with lifted blocks
\(\delta X=(\delta\rho,\delta B,\delta N,\delta A,\delta C)\), this norm is the
combined block norm

\[
\|\delta x\|_{\mathrm{lift}}^2
=\|\delta\rho\|_{\mathrm F}^2
+\|\delta B\|_2^2
+\|\delta N\|_{\mathrm F}^2
+\|\delta A\|_{\mathrm F}^2
+\sum_q\|\delta C^q\|_{\mathrm F}^2.
\]

The minimum retained fraction under each tangent projection is
\(0.9999999999983\). Hence the repeated projection removes only roundoff-scale
normal components and does not generate the measured stretching.

## Benettin calculation

The base state and two shadows are advanced with fixed-step RK4 using

\[
\Delta t=0.02,
\qquad
\tau_{\mathrm r}=0.5,
\qquad
4\le t\le1000,
\]

where \(\tau_{\mathrm r}\) is the perturbation-renormalization interval. If
\(d_k\) is the lifted Frobenius separation at the end of interval \(k\), the
local and cumulative estimates are

\[
\ell_k=\frac{1}{\tau_{\mathrm r}}
\log\!\left(\frac{d_k}{\varepsilon}\right),
\qquad
\Lambda_n=\frac{1}{n\tau_{\mathrm r}}
\sum_{k=1}^{n}\log\!\left(\frac{d_k}{\varepsilon}\right).
\]

After recording \(d_k\), the evolved difference is tangent-projected and
rescaled to \(\varepsilon\). The two unrelated starting directions must align
with the dominant expanding direction if one persistent largest exponent
governs the flow.

## Exponent history

| Global-time window | Drive direction | Relative-phonon direction |
|---|---:|---:|
| \(4\) to \(100\) | 0.06021798 | 0.06031898 |
| \(100\) to \(250\) | 0.02869649 | 0.02869405 |
| \(250\) to \(500\) | 0.03610694 | 0.03610523 |
| \(500\) to \(750\) | 0.01801749 | 0.01801471 |
| \(750\) to \(1000\) | 0.01891447 | 0.01891493 |
| cumulative \(4\) to \(1000\) | 0.02845896 | 0.02846732 |

The two directions agree to \(8.35\times10^{-6}\) in the final cumulative
estimate and to \(4.58\times10^{-7}\) in the final window. The positive final
window excludes an explanation based only on the early post-pulse transient.

## Boundedness and invariants

| Diagnostic over base and shadows | Recorded value |
|---|---:|
| Maximum absolute coordinate | 2.09826061 |
| Minimum physicality margin | \(9.07734\times10^{-5}\) |
| Maximum base-energy drift | \(6.75539\times10^{-10}\) |
| Maximum shadow-energy drift | \(6.91329\times10^{-10}\) |
| Maximum correlation-trace residual | \(8.67362\times10^{-19}\) |

The positive exponent therefore accompanies bounded motion within the tested
representability domain. It is distinct from the late amplitude threshold and
from the raw closure's early cone crossing.

## Controller attribution

The electronic lower and upper positivity inequalities never bind in the
post-pulse run. The joint-Gram inequality binds at \(5.168\%\) of sampled base
states and changes binding status 101 times. Classifying a segment as
cone-active when the joint inequality binds at either endpoint gives:

| Segment class | Segments | Fraction | Mean local exponent, drive | Mean local exponent, phonon |
|---|---:|---:|---:|---:|
| No cone inequality binding at either endpoint | 1839 | 0.92319 | 0.0254261 | 0.0253907 |
| Joint-Gram inequality binding at either endpoint | 153 | 0.07681 | 0.0649133 | 0.0654474 |

The no-binding intervals contribute \(23.38\) of the drive direction's total
\(28.35\) accumulated log-growth. Cone-active intervals stretch more rapidly,
yet sustained positive growth remains when no positivity inequality is active.
The correlation-trace equality correction remains part of the corrected vector
field in both classes.

A separate matched-pair calculation through \(t=100\), performed without
perturbation rescaling, decomposes the late instantaneous separation growth in
the lifted Frobenius metric. For the drive and relative-phonon directions,
respectively,

\[
\begin{array}{c|cc}
&\text{drive}&\text{relative phonon}\\ \hline
\text{raw archive component}&0.0567658&0.0572579\\
\text{controller component}&-0.0005936&-0.0005945\\
\text{total corrected field}&0.0561722&0.0566634
\end{array}
\]

Thus the raw archive component supplies the measured local stretching along
the corrected trajectory, whereas the controller contribution slightly
suppresses it. Active cone switching enhances selected intervals without
creating the positive long-time exponent.

## Exact-state perturbation control

The same two wavefunction perturbations were propagated under the common
cutoff Hamiltonian and contracted to the 31 retained coordinates through
\(t=100\). At cutoff 16, the maximum contracted amplifications were 1.3848 and
1.4899; the final amplifications were 0.3105 and 0.4649. The late endpoint log
rates were negative. The cutoff-12 and cutoff-20 controls gave the same bounded
pattern, and state fidelity was conserved to approximately \(10^{-14}\).

This control establishes that the selected perturbation directions and their
31-coordinate contractions do not themselves impose exponential growth. The
positive exponent is a property of the corrected moment vector field under the
tested protocol.

## Scientific conclusion

The tested corrected archive moment EOM satisfy the operational numerical
criterion for bounded chaos: an autonomous bounded trajectory carries a
positive, direction-independent largest finite-time Lyapunov exponent that
persists through the final long window. The positivity controller preserves
representability and slightly reduces the local separation rate; it does not
repair the closure-generated sensitivity.

The result does not yet classify the invariant set. A full Lyapunov spectrum,
phase-space-divergence measurement, and basin or recurrence analysis would
distinguish a dissipative strange attractor from a conservative chaotic region.
The more immediate modeling test is whether a physically adapted replacement
for the inaccurate highest-order closure reduces both the exact-reference
derivative defect and the positive exponent.

## Numerical controls and evidence boundary

Earlier \(t=100\) Benettin runs tested independent random directions,
\(\Delta t=0.01\) and \(0.02\), and perturbation sizes \(10^{-5}\) and
\(2\times10^{-5}\). Their late \(t=40\) to \(100\) exponents agreed within
\(8.1\times10^{-5}\). The completed \(t=1000\) calculation uses
\(\Delta t=0.02\), \(\tau_{\mathrm r}=0.5\), and two exact-state-induced
directions. The complete Paper V regression suite passes 165 tests.

The current evidence is a numerical classification for the stated corrected
vector field, parameter point, initialization, metric, and hard post-pulse
switch. It is neither a theorem nor a parameter-space survey.

## Artifacts and implementation

- Long-horizon plan and source hashes:
  `output/local_runs/paper_v_postpulse_lyapunov_t1000_dt002_20260804_v1/plan.json`.
- Long-horizon summary:
  `output/local_runs/paper_v_postpulse_lyapunov_t1000_dt002_20260804_v1/summary.json`.
- Complete segment record:
  `output/local_runs/paper_v_postpulse_lyapunov_t1000_dt002_20260804_v1/trajectory.npz`.
- Exact-state sensitivity control:
  `output/local_runs/paper_v_exact_reference_sensitivity_t100_cutoff12_16_20_20260803_v1/summary.json`.
- Matched exact/controller sensitivity and vector-field decomposition:
  `output/local_runs/paper_v_matched_exact_controller_sensitivity_t100_dt002_20260803_v1/summary.json`.
- Earlier finite-time convergence screen:
  `output/local_runs/paper_v_nearby_sensitivity_convergence_20260803_v1/summary.json`.
- Long-horizon implementation:
  `paper_5/src/paper5/stability/postpulse_lyapunov.py`.
- Exact-state control implementation:
  `paper_5/src/paper5/stability/exact_reference_sensitivity.py`.
- Matched-field attribution implementation:
  `paper_5/src/paper5/stability/matched_exact_controller_sensitivity.py`.
- Focused long-horizon tests:
  `paper_5/tests/test_postpulse_lyapunov.py`.

The dated earlier chronology remains in
`paper_5/notes/electron_phonon_closure_worklog_20260803.md`; its open request
for a longer exponent plateau is resolved by the present note.

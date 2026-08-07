# Non-Markovian closure diagnostic study

## Decision

The next Ping-group-facing result is an evidence study, not another software
architecture.  It asks whether the repository's independent 31-coordinate
electron--phonon closure improves shared observables over its coherent-only
Ehrenfest limit, and whether loss of joint electron--phonon representability is
an informative predictor of observable error.

This is an exploratory, repository-owned benchmark.  It is source-anchored by
the driven Holstein dimer used by Riva, Simoni, and Ping, but it is not a
reproduction of their spin-reduced equations or curves.

## Locked questions

1. At matched parameters, time samples, and initial lower moments, does the raw
   31-coordinate closure reduce electronic-density and coherent-phonon error
   relative to coherent-only dynamics?
2. Does the first loss or subsequent severity of the joint electron--phonon
   Gram certificate predict electronic-density error across the bounded grid?
3. When a velocity barrier restores the certificate, does it also improve the
   exact-trajectory error, or does admissibility remain distinct from accuracy?

No causal or universal claim is permitted from this small grid.  Correlations
are reported as exploratory diagnostics and stratified by coupling before any
pooled association is interpreted.

## Immutable evidence source

The analysis extends the completed diagnostic run
`output/local_runs/paper_v_electron_phonon_analysis_20260801_v3/`.
Its runtime manifest records a complete run, hashes the trajectory artifacts,
and states that exact data were used only for offline comparison.  The source
run already contains:

- exact truncated-Hamiltonian trajectories;
- raw and joint-Gram-corrected 31-coordinate trajectories;
- fixed-step and phonon-cutoff convergence checks; and
- a 12-point Cartesian grid over
  `lambda_ep in {0.5, 1.5}`, `gamma in {0.25, 0.5, 1.0}`, and
  `drive_amplitude in {0.5, 1.0}` through `t = 4`.

The extension must verify the recorded SHA-256 hashes before reading these
arrays.  It does not rerun or alter the immutable source directory.

## Matched methods

| Method | Role | Online access to exact data |
|---|---|---|
| Truncated explicit Holstein evolution | Offline reporting reference | Not applicable |
| Five-coordinate Ehrenfest evolution | Coherent-only comparator | Initial shared lower moments only |
| Raw 31-coordinate closure | Independent non-Markovian candidate | Initial shared reduced moments only |
| Joint-Gram velocity barrier | Admissibility-corrected candidate | None |

The coherent-only trajectory starts from the exact trajectory's electronic
one-body density matrix and coherent phonon amplitude, just as the 31-coordinate
candidate starts from the exact contracted moments.  This is a matched-initial-
moments diagnostic, not a claim that the projected state is a stationary state
of the coherent equations.

Adaptive McLachlan is deliberately outside this first comparison.  It may be
added only after a finite explicit-mode problem uses the same Hamiltonian,
drive, initial-state artifact, sampled observables, and seed provenance.  A
separate Paper-II seed or a differently initialized variational trajectory is
not a matched comparator.

## Metrics

For the electronic density `rho` and coherent amplitude `B`, each method reports

```text
rms_error_over_exact_dynamic_rms
maximum_frobenius_error
final_frobenius_error
```

The main comparative statistic is

```text
raw_closure_rms / coherent_only_rms
```

where values below one favor the raw closure.  The same ratio is reported for
the corrected closure, but a physicality correction is never presumed to be an
error estimator.

For the raw closure, the study also records

```text
minimum_joint_gram_eigenvalue
first_time_joint_gram_below_-1e-8
first_time_instantaneous_rho_error_exceeds_0.1 * exact_dynamic_rms
```

The `0.1` scale is inherited from the completed run's exploratory materiality
threshold.  It is not presented as a universal physical tolerance.

## Evidence gates

The result memo may say that the closure improves a shared observable only when
the matched error ratio is below one.  It may say that the Gram certificate is
not sufficient as a predictive error diagnostic if certificate loss occurs
without a corresponding material electronic error or if within-coupling trends
are inconsistent.  It may not say that representability is irrelevant: a
negative Gram eigenvalue still makes the reduced state inadmissible.

The output remains `exploratory_local_not_promoted`.  It is suitable for asking
Professor Ping and Jacopo whether the observed separation between
representability and accuracy is scientifically useful; it is not collaborator
validation, source reproduction, or paper-facing final evidence.

## Deliverables

1. A hash-verified machine-readable summary and coherent-only trajectories.
2. A source-point trajectory figure at
   `(lambda_ep, gamma, drive) = (0.5, 0.5, 1.0)`.
3. A bounded-grid figure comparing closure/coherent error and Gram severity.
4. A one-page discussion memo written only from the measured results.

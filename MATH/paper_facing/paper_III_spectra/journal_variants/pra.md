# Paper III Physical Review A Variant

Created: 2026-05-09  
Role: venue-specific planning notes, not a manuscript copy.

## Fit thesis

Physical Review A is the best Paper III target if the paper is written as a physics-motivated quantum-science methods article: mixed fermion--boson spectra, transition response, method detail, and class-wise benchmarks.

## Title direction

Preferred:

> Geometry-Aware Quantum Subspace Expansion for Mixed Fermion--Boson Spectra

Backup:

> Geometry-Aware QSE and Excited-State Dynamics for Mixed Fermion--Boson Hamiltonians

## Abstract shape

1. Begin with electron--phonon / mixed fermion--boson response, not with internal algorithm phases.
2. Name QSE as the baseline excited-state object.
3. State the selection mechanism compactly.
4. State benchmarks by Hamiltonian class and observable type.
5. Keep claims proportional to completed tables.

## Main-text structure

1. Physical motivation and Hamiltonian families.
2. Existing excited-state quantum algorithms.
3. Prepared seed and excitation records.
4. Geometry-aware QSE selection.
5. Generalized eigenproblem, spectral observables, and regularization.
6. Frozen-subspace and live excited-state McLachlan propagation.
7. Benchmark suite.
8. Static spectra and ablations.
9. Driven excited-state response and ablations.
10. Discussion, limitations, and classical/quantum resource interpretation.

## PRA-specific writing guidance

- Keep derivations visible enough for a physics reader to audit the method.
- Do not make the introduction a generic VQE survey.
- Treat ED, tensor networks, qEOM, QSE, and VQD as serious context, not straw men.
- Put placeholder-free benchmark tables before making comparative claims.
- Use appendices for alphabet registry, matrix-measurement estimates, root tracking, and regularization details.

## Evidence bar

Minimum:

- class-wise spectra table;
- mechanism ablation table;
- frozen QSE vs live excited-state dynamics table;
- transition-strength or spectral-function figure;
- overlap-conditioning figure;
- clear exact-reference/measurement-compatible data-flow statement.

PRA can tolerate technical detail, but not unsupported broad NISQ or quantum-advantage language.

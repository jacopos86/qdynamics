# Paper III Theory TODOs

Created: 2026-06-24  
Role: future-theory capture for Paper III support work. This file is not manuscript prose and does not assert implemented evidence.

## Sketch-and-project QSE versus nonlinear variational training

Static QSE already has a sketch-and-project structure: the selected excitation manifold

\[
\Phi_{\mathcal A}=[\,\Omega_a|\psi_0\rangle\,]_{a\in\mathcal A}
\]

defines the sketch, and QSE projects the Hamiltonian into that nonorthogonal span through

\[
S_{\mathcal A}=\Phi_{\mathcal A}^\dagger\Phi_{\mathcal A},
\qquad
M_{\mathcal A}=\Phi_{\mathcal A}^\dagger H_0\Phi_{\mathcal A},
\qquad
M_{\mathcal A}c^{(\nu)}=\varepsilon_\nu S_{\mathcal A}c^{(\nu)}.
\]

This step is linear Rayleigh--Ritz optimization over coefficients \(c\). It is distinct from SR, MinSR, or SPRING-style nonlinear training of wavefunction parameters.

The variational leverage point is QSE-root compression or live excited-root refinement. Given a selected QSE root

\[
|\Psi_\nu^{\rm QSE}\rangle
=
\sum_{a\in\mathcal A}c_a^{(\nu)}\Omega_a|\psi_0\rangle,
\]

compress the root into a live ansatz state \( |\psi_\nu(\vartheta)\rangle \) by overlap maximization, energy or variance minimization, and lower-root protection. Candidate objectives include

\[
\min_\vartheta
\left[
1-
\left|
\langle\psi_\nu(\vartheta)|\Psi_\nu^{\rm QSE}\rangle
\right|^2
\right],
\]

or an excited-root energy/variance objective with penalties that prevent collapse into lower roots.

The corresponding update family could use SR, MinSR, or sketched natural-gradient steps on the live ansatz tangent metric,

\[
F_\nu(\vartheta)\,\delta\vartheta=-g_\nu(\vartheta),
\]

with \(F_\nu\) estimated from the tangent vectors of \( |\psi_\nu(\vartheta)\rangle \) and \(g_\nu\) determined by the chosen root-refit objective.

## Manuscript boundary

Do not claim that the current Paper III method implements SR, MinSR, SPRING, or nonlinear sketch-and-project training unless runs and method text are added. The safe current claim is:

> Static QSE supplies an operator-selected spectral sketch and a projected generalized eigenproblem; future live-root compression can use sketched natural-gradient training to variationally realize selected QSE roots as circuits.


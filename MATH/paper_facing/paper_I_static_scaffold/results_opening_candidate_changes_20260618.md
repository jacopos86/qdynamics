# Paper I Results Opening Candidate Changes

Date: 2026-06-18

Purpose: handoff note for continuing the Paper I Results-section cleanup. No manuscript edit has been approved or applied from this note yet.

## Current focus

Target manuscript:

`MATH/paper_details/static_adapt_paper_I.tex`

Target section:

`\section{Results}`, especially the opening paragraphs before the main benchmark tables/plots.

The goal is a minimally invasive cleanup of the Results opening:

- improve paragraph separation;
- reduce overloaded first-paragraph scope;
- remove stale four-regime Hubbard--Holstein framing;
- keep fidelity as a diagnostic;
- avoid making the plateau-prefix convention look like a fake precise algorithmic rule.

## User decisions already made

- Do not collapse the \(E_T\) discussion and \(|\Delta E|\) discussion into one paragraph.
- \(E_T\) is mostly archaic and should be demoted to an orientation scale for convergence plots.
- The plateau prefix \(k_{\rm pl}\) is selected by visual fairness from the error-versus-iteration traces, not by a rigid formal rule.
- The manuscript should say the reader can verify fairness from the error-versus-iteration plots.
- Fidelity should stay, but not as a target fidelity or convergence criterion.
- Preserve draft placeholders unless explicitly doing final-polish cleanup.
- Do not redesign figures/tables in this pass.

## Candidate change 1: Results opening scope paragraph

Replace the overloaded first paragraph with a cleaner scope paragraph:

> The benchmark suite compares SNAKE against existing ADAPT variants and fixed-structure VQE baselines across Hubbard, spin-boson/Rabi, and Hubbard--Holstein instances. All lattice rows use \(L=2\), half filling, and open boundary conditions. The main Hubbard--Holstein stress test reports six regimes crossing \(U/t\in\{0.25,1.25,8\}\) with \(\lambda\in\{0.25,1.25\}\), with compiled resources and same-cutoff energy errors summarized in Table~\(\ref{tab:hh_first_plateau_prefix_costs}\).

Intent:

- state the Results scope first;
- explicitly use six HH regimes;
- avoid stale four-regime language;
- avoid burying comparator/protocol details in the first paragraph.

## Candidate change 2: comparator paragraph separation

Keep the existing comparator details, but move them into their own paragraph after the scope paragraph.

Comparator content to preserve:

- `Qiskit` AdaptVQE and VQE baselines;
- append-only ADAPT;
- fixed-structure VQE;
- `Qiskit` transpilation for compiled resource estimates;
- matched implementations of TETRIS-ADAPT, Qubit/QEB-ADAPT, and Geo-ADAPT.

Intent:

- paragraph separation only;
- no semantic change unless required for grammar.

## Candidate change 3: shared protocol paragraph separation

Keep the shared protocol details as a separate paragraph.

Content to preserve:

- with the exception of fixed-VQE and Qubit/QEB-ADAPT, all ADAPT variants use the same problem-local generator pool from Appendix~\(\ref{app:pools}\);
- displayed rows use the SPSA refit protocol from Appendix~\(\ref{app:results_discussion}\);
- Geo-ADAPT keeps its Fubini--Study metric score but evaluates the candidate tangent metric with a Moore--Penrose pseudoinverse;
- fixed-structure VQE ansatz choices;
- Qubit/QEB-ADAPT as a mapped-qubit excitation-pool comparator.

Intent:

- keep comparator protocol clear;
- avoid making the Results opening a single dense block.

## Candidate change 4: demote \(E_T\) but keep it separate

Replace the current \(E_T\) paragraph with:

> For orientation in convergence plots, we mark the energy scale \(E_T=10^{-4}LE_0=2\times10^{-4}\), with \(E_0=1\) set by the model unit \(t\), \(J\), or \(\omega_0\) as appropriate. This scale is used only as a visual reference for reading convergence histories, not as the primary reported observable.

Intent:

- keep \(E_T\) available for plot reading;
- stop making it sound like the central benchmark target;
- keep this paragraph separate from \(|\Delta E|\).

## Candidate change 5: keep \(|\Delta E|\) paragraph separate

Keep the \(|\Delta E|\) paragraph separate and close to the existing wording:

> The tables report raw absolute same-cutoff energy error \(|\Delta E|=|E_{\rm alg}-E_{\rm ref}|\) against the relevant ED reference. Here \(E_{\rm alg}\) is the energy obtained from a VQE method and \(E_{\rm ref}\) is the corresponding exact-diagonalization energy evaluated at the same cutoff as the displayed ansatz when a bosonic cutoff is present. Cutoff-response diagnostics are kept separate from the displayed energy-error convention.

Intent:

- make \(|\Delta E|\), not \(E_T\), the main reported observable;
- preserve same-cutoff convention.

## Candidate change 6: replace formal plateau rule with visual-fairness convention

Replace the current plateau-definition paragraph with:

> For the Hubbard--Holstein stress test, several adaptive histories enter persistent plateaus before the terminal iteration. Table~\(\ref{tab:hh_first_plateau_prefix_costs}\) therefore reports a representative plateau prefix \(k_{\rm pl}\), chosen to compare methods at the earliest visually stable error plateau rather than at an arbitrarily long terminal prefix. The accompanying error-versus-iteration traces display the full histories, so the reader can verify that the reported prefix is not selected from an isolated transient improvement. The fixed-prefix Hubbard--Holstein comparison is retained in Appendix~\(\ref{app:hh_stress_diagnostics}\).

Intent:

- remove fake precision from the current "no later accepted prefix lowers the same-cutoff error by more than five percent" rule;
- state the real fairness criterion;
- point the reader to the convergence traces as the visible check.

## Candidate change 7: keep fidelity as diagnostic, not target

Replace or extend the current fidelity sentence with:

> The table fidelity diagnostic is \(1-F=1-\left|\braket{\psi_{\rm alg}}{\psi_{\rm ref}}\right|^2\), evaluated against the exact reference state for the displayed benchmark row. We report it as a state-overlap diagnostic, not as a convergence target.

Intent:

- retain fidelity;
- prevent target-fidelity framing.

## Candidate change 8: fix HH regime statement

Replace the stale four-regime statement:

\[
(U/t,\lambda)\in\{0.25,1.25\}\times\{0.25,1.25\}.
\]

with:

\[
U/t\in\{0.25,1.25,8\},\qquad \lambda\in\{0.25,1.25\}.
\]

Use the six manuscript-facing display labels:

- `weak-weak`;
- `intermediate-weak`;
- `strong-weak`;
- `weak-strong`;
- `intermediate-strong`;
- `strong-strong`.

Implementation/provenance note:

- Some overlay/source artifacts use `strong-weak-u8` and `strong-strong-u8`.
- In manuscript prose, those are the strong-Hubbard display regimes.
- Preserve `u8` only where provenance or artifact labels are being discussed.

## Explicit non-goals for this pass

- Do not redesign the main Results figure.
- Do not decide whether all six regimes must be plotted in the main body.
- Do not update table cells or source maps.
- Do not remove placeholders.
- Do not rewrite the full Results section.
- Do not change manuscript claims beyond the Results-opening scope cleanup.

## Build rule if edits are applied

After editing `MATH/paper_details/static_adapt_paper_I.tex`, rebuild the PDF from `MATH/paper_details/` with `latexmk` if available, otherwise:

```bash
tectonic --keep-logs --reruns 2 static_adapt_paper_I.tex
```

Report exact build failures if the rebuild fails.

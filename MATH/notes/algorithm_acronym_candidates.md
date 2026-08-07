# Algorithm Acronym Candidates

This note collects candidate names for the adaptive variational algorithms used
across the static ADAPT, checkpoint McLachlan dynamics, and excited-state
subspace manuscripts. It is a naming aid, not a manuscript source of truth.

## Naming Boundary

Use named acronyms sparingly. A manuscript should first be legible as a
scientific method:

- static paper: geometry- and cost-aware ADAPT ansatz construction;
- dynamics paper: checkpoint-adaptive McLachlan dynamics with append/prune/stay
  decisions;
- spectra paper: geometry-selected QSE and excited-state dynamics.

If an acronym is used, define it once and keep the long-form scientific
description nearby. Do not make readers decode the paper through the acronym.

## Current Recommendation

### Use SNAKE for the Static ADAPT Controller Only, Unless Reframed

The current static manuscript already uses SNAKE. That is defensible if SNAKE
names the geometry-aware ADAPT controller: candidate-position selection,
state-space novelty, Schur reranking, beam/batch options, and rollback-safe
ablation.

Recommended static expansion:

\[
\mathrm{SNAKE}
:=
\text{State-space Novelty and Ablation-aware ADAPT Kernel Engine}.
\]

Rationale:

- "State-space" matches the Fubini--Study / tangent-metric emphasis.
- "Novelty" matches Phase II.
- "Ablation-aware" foregrounds the prune/recoverability contribution.
- "ADAPT" keeps the method discoverable.
- "Kernel" is acceptable if explained as the tangent Gram / metric kernel, not
  as machine learning.
- "Engine" is broad enough to include optional beam, batch, Schur, and pruning
  extensions.

Avoid using "Krylov" for the static ADAPT paper unless the method is explicitly
reframed as a Krylov or Krylov-like subspace construction. The static selector
uses tangent geometry and reduced-window Schur logic; calling it Krylov may
invite unnecessary reviewer objections.

### Use a Descriptive Dynamics Name by Default

For the dynamics paper, the safest title/abstract wording is not necessarily
SNAKE. Prefer:

\[
\text{checkpoint-adaptive McLachlan dynamics}
\]

or

\[
\text{bidirectional checkpoint-adaptive McLachlan dynamics}.
\]

If an acronym is wanted for the dynamics controller, the strongest SNAKE-style
expansions are:

\[
\mathrm{SNAKE}
:=
\text{State-space Navigation by Adaptive Kinematic Events},
\]

\[
\mathrm{SNAKE}
:=
\text{Stabilized Nonorthogonal Adaptive Kinematic Evolution},
\]

or

\[
\mathrm{SNAKE}
:=
\text{Selective N-body Ansatz Kinematic Engine}.
\]

The first is the best conceptual fit for checkpoint stay/append/prune/repair
events. The second foregrounds metric-stabilized McLachlan propagation. The
third is more presentation-friendly but less mathematically precise.

Avoid "Krylov" in the dynamics acronym unless the method explicitly builds and
propagates in a Krylov subspace. McLachlan projection and Schur append gain are
not automatically Krylov methods.

### Do Not Force One Acronym Across All Three Papers

The static, dynamics, and spectra methods share geometric ideas but have
different mathematical authorities:

- static deletion authority is remove-refit energy recoverability;
- dynamic deletion authority is live McLachlan recoverability;
- spectra authority is the conditioned generalized eigenproblem and transition
  response in a selected nonorthogonal basis.

A single umbrella acronym risks blurring those distinctions. A better hierarchy
is:

\[
\text{SNAKE-static}
\quad\text{only if needed internally,}
\]

\[
\text{checkpoint-adaptive McLachlan}
\quad\text{for dynamics,}
\]

\[
\text{geometry-selected QSE}
\quad\text{for spectra.}
\]

## SNAKE Candidate List

### Static ADAPT Candidates

Best candidates:

1. \(\mathrm{SNAKE}:=\) State-space Novelty and Ablation-aware ADAPT Kernel
   Engine.
2. \(\mathrm{SNAKE}:=\) Selection-Novel ADAPT Kernel Engine.
3. \(\mathrm{SNAKE}:=\) State-space Novelty ADAPT with Kernelized
   Energy-screening.
4. \(\mathrm{SNAKE}:=\) Structured Novelty and Ablation-aware Kinetic-free
   Energy ADAPT.

Notes:

- Candidate 1 is the strongest.
- Candidate 2 is short but "Selection-Novel" is less idiomatic.
- Candidate 3 is descriptive but too long for a title.
- Candidate 4 should probably be rejected; "kinetic-free" is awkward and not
  the scientific point.

Rejected or risky:

\[
\mathrm{SNAKE}:=\text{Selection-Novel ADAPT Krylov Engine}.
\]

This sounds good but is technically risky unless the paper explicitly proves or
uses a Krylov interpretation.

### Dynamics Candidates

Best candidates:

1. \(\mathrm{SNAKE}:=\) State-space Navigation by Adaptive Kinematic Events.
2. \(\mathrm{SNAKE}:=\) Stabilized Nonorthogonal Adaptive Kinematic Evolution.
3. \(\mathrm{SNAKE}:=\) Selective N-body Ansatz Kinematic Engine.
4. \(\mathrm{SNAKE}:=\) Structural Navigation by Adaptive Kinematic Events.
5. \(\mathrm{SNAKE}:=\) Scaffold-Nucleated Adaptive Kinematic Engine.

Notes:

- Candidate 1 best matches checkpoint event logic.
- Candidate 2 best matches stabilized McLachlan solves.
- Candidate 3 is closest to the user's proposed "Selective N-body Ansatz
  Krylov Engine" while avoiding "Krylov."
- Candidate 5 should be used cautiously in manuscripts because "scaffold" is
  not standard journal language unless defined.

### Umbrella Candidates

If SNAKE names the whole project rather than one paper:

1. \(\mathrm{SNAKE}:=\) State-space Novelty and Kinematic Adaptation Engine.
2. \(\mathrm{SNAKE}:=\) Structural Novelty for Adaptive Kinematic Enrichment.
3. \(\mathrm{SNAKE}:=\) State-space Navigation and Kinematic Enrichment.

These are less precise than paper-specific names. Use only for slides or a
high-level project label.

## WORM Candidates

WORM is not a good umbrella name for the whole algorithm, but it is useful for
the pruning/recoverability subsystem.

Best WORM expansion:

\[
\mathrm{WORM}
:=
\text{Windowed Operator-Removal and Refit Minimization}.
\]

This matches the static pruning object: delete one operator coordinate, allow a
typed local window to refit, and test whether the variational energy is
recoverable.

Other acceptable WORM expansions:

\[
\mathrm{WORM}
:=
\text{Windowed Operator Recoverability Metric},
\]

\[
\mathrm{WORM}
:=
\text{Windowed Operator-Removal Mechanism},
\]

\[
\mathrm{WORM}
:=
\text{Windowed Orthogonal Residual Minimization}.
\]

Use cases:

- Static paper: WORM can label the prune/recoverability ladder, not the entire
  ADAPT selector.
- Dynamics paper: WORM can label optional pruning diagnostics only if the live
  McLachlan authority remains explicit.
- Spectra paper: avoid WORM unless a true windowed basis-removal procedure is
  introduced.

Recommendation:

\[
\text{Use SNAKE for the static ADAPT controller if the paper keeps acronymic
branding; use WORM only as a pruning subsystem label.}
\]

## Manuscript-Safe First-Definition Sentences

Static paper:

> We refer to the resulting static controller as SNAKE, a state-space novelty
> and ablation-aware ADAPT kernel engine: it ranks candidate--position records
> by geometric gain, cost, tangent novelty, optional Schur reranking, and
> recoverability-based ablation.

Dynamics paper:

> The dynamics method is a checkpoint-adaptive McLachlan controller: it advances
> on the current variational ansatz and opens stay, append, prune, or repair
> events only when live tangent-geometry diagnostics warrant a structural
> change.

Pruning subsystem:

> The optional WORM stage, windowed operator-removal and refit minimization,
> nominates a coordinate for deletion only when a typed refit window can recover
> the variational objective within the prune tolerance.

## Current Decision

Do not rename Papers II or III around SNAKE unless the manuscript is explicitly
being branded as a SNAKE-family paper. The cleanest current split is:

- Paper I: SNAKE, if desired.
- Paper II: checkpoint-adaptive McLachlan dynamics.
- Paper III: geometry-selected QSE / excited-state dynamics.


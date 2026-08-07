# Holstein Research Program Context Map

This is one Git repository containing separate paper-owned scientific lanes.
Each lane owns its method language, execution workflows, provenance, evidence,
and paper-facing support.

## Language

**Paper-owned lane**:
A scientific context that owns one paper's methods, runs, provenance, evidence,
and reporting semantics.
_Avoid_: Paper folder, feature area

**Shared scientific core**:
Paper-neutral mathematical and quantum-computing primitives that are used by
more than one paper-owned lane but do not own paper evidence.
_Avoid_: Paper-I utilities, common dumping ground

**Receiving-lane import**:
A receiving paper-owned lane's transformation of a locked output from another
lane into its own native input. The producing lane remains unchanged; the
receiver owns the import receipt and all downstream provenance.
_Avoid_: Cross-paper handoff, shared run, copied provenance

**Lane-local benchmark**:
A comparator retained for evidence within its owning paper lane but excluded
from downstream paper imports.
_Avoid_: Shared baseline, retired route when retirement is undecided

## Contexts

- [Paper I: Static ADAPT and SNAKE](agent_guidance/static-adapt/CONTEXT.md) —
  adaptive ansatz construction, static optimization, and static resource
  evidence.
- **Paper II: AP-McLachlan time dynamics** — physical time evolution and
  checkpoint-local manifold maintenance. Its detailed context glossary is
  deferred.
- **Paper III: QSE and excited dynamics** — excitation manifolds, spectra,
  transition observables, and excited-state evolution. Its detailed context
  glossary is deferred.
- **Paper IV: Molecular-vibronic water** — the H2O application lane. Its
  detailed context glossary is deferred.
- **Paper V: High-U regularization and GKBA** — the high-interaction
  exploratory lane. Its detailed context glossary is deferred.
- **Shared scientific core** — paper-neutral Hamiltonian, Pauli, encoding,
  state, optimization, and validation primitives.

## Relationships

- **Paper I to Paper II**: Paper I produces Paper-I static results. A
  future Paper-II-owned import route may transform a locked Paper-I result into
  a Paper-II-native dynamics seed. Paper II owns the import receipt and every
  subsequent trajectory, run, artifact, and evidence record. Detailed import
  semantics are deferred until the Paper-II refactor.
- **Geo-ADAPT locality**: Geo-ADAPT remains a Paper-I-local benchmark. It is not
  an eligible Paper-II seed or a dependency of another paper-owned lane; its
  final retirement status remains undecided.
- **Shared scientific core to paper-owned lanes**: Lanes consume neutral
  scientific primitives through stable interfaces; the core never adopts a
  paper's provenance or defaults.
- **Paper-owned lane isolation**: One lane may cite or import another lane's
  locked output, but it does not write downstream results into the producing
  lane's provenance namespace.

# Paper I Claim Boundaries

Created: 2026-05-09  
Target: `MATH/paper_details/static_adapt_paper_I.tex`  
Paper identity: geometry- and cost-aware static ADAPT scaffold construction for mixed fermion--boson systems.  
Recommended title: **Geometry- and Cost-Aware ADAPT Ansatz Construction for Mixed Fermion--Boson Systems**.  
Strongest framing: **budgeted variational scaffold acquisition by geometry- and cost-aware candidate-position selection**.

## Core novelty sentence

Use this as the default Paper I novelty sentence, then tune to the local paragraph:

> Rather than selecting bare pool operators by local ADAPT gradients, we select candidate-position records under a joint geometric and hardware-cost objective, then rerank them by reduced-window Schur relaxation and remove stale scaffold elements by rollback-safe pruning.

This distinguishes Paper I from original ADAPT-VQE, qubit-ADAPT-VQE, QEB-ADAPT-VQE, TETRIS-ADAPT-VQE, Overlap-ADAPT-VQE, CEO-style ADAPT, Hessian-recycling subroutines, and Geo-ADAPT-VQE without claiming that all of ADAPT or geometry-aware selection is new.

## Safe claims

These are safe as method-definition or scope claims when grounded in `MATH/Math.md`:

- Candidate-position records extend ADAPT selection beyond bare generator labels.
- Geometry and cost enter acquisition before the circuit is purchased.
- Fubini--Study/state-space scaling and tangent-span novelty are used to compare candidate directions.
- Reduced-window Schur reranking estimates marginal value after local refit geometry is considered.
- Batching and beam continuation address early branch/order sensitivity.
- Rollback-safe static pruning turns the scaffold from append-only into append-and-verify-delete construction.
- Mixed fermion--boson systems make encoding, cutoff, layout, compiled depth, and measurement burden part of the resource surface.
- Compact scaffolds may seed dynamics, but online McLachlan maintenance belongs to Paper II.

## Claims that need data

Require a locked table, figure, ablation, benchmark protocol, or reproducibility record:

- Any percentage improvement.
- Any superiority over all ADAPT variants.
- Any claim of lower shot cost or measurement rounds.
- Any claim of lower compiled two-qubit depth at fixed error.
- Any claim that pruning improves resource Pareto behavior.
- Any claim that Schur reranking, novelty, batching, beam, or cost terms are necessary.
- Any hardware-readiness claim.
- Any cross-family generalization beyond the benchmark suite.

## Avoid / rewrite

| Avoid | Safer replacement |
|---|---|
| first geometry-aware ADAPT method | candidate-position geometry/cost selector distinct from Geo-ADAPT-VQE's QIM/natural-gradient operator selection |
| better ADAPT | improves a specified metric against a specified ADAPT baseline under a specified protocol |
| solves electron-phonon simulation on NISQ devices | evaluates compact scaffold construction for mixed fermion--boson benchmark instances |
| new operator pool | new acquisition policy over candidate-position records from a problem-local pool |
| hardware-ready | compiled/noisy/fake-backend calibrated under the stated benchmark conditions, if evidence exists |

## Required boundaries with prior work

- ADAPT-VQE: prior work established gradient-selected adaptive growth; Paper I changes the acquisition object and score structure.
- Geo-ADAPT-VQE: prior work established QIM/natural-gradient ADAPT operator selection; Paper I must emphasize insertion-position records, FS span novelty, Schur reranking, cost, batching/beam, and pruning.
- TETRIS-ADAPT-VQE: prior work established batching/disjoint additions; Paper I includes batching as one component, not as the entire novelty.
- CEO-ADAPT-VQE/Hessian recycling: prior work reduces resources and measurement/optimization cost; Paper I cannot claim state-of-practice resource dominance without direct comparisons.
- Electron--phonon encoding work: prior work establishes mixed boson/fermion encodings; Paper I is not an encoding paper unless new encoding data are explicitly added.

## Geo-ADAPT metric-rank boundary

Do not claim that UCCSD operators cannot cause metric ill-conditioning or rank deficiency. The safer distinction is local and conditional: a de-duplicated UCCSD singles/doubles excitation pool acting on the Hartree--Fock reference gives distinct first-order determinant tangents, while redundant, symmetry-forbidden, repeated, or later-state-dependent directions can still weaken a QGT/QIM block.

Geo-ADAPT-VQE supports the statement that QIM/Fubini--Study natural-gradient operator selection uses a metric solve. It does not appear to solve general rank-deficient inverse problems by broad operator deduplication: the algorithm chooses operators with replacement, skips only an immediately repeated selected operator, uses block-diagonal metric approximations for fixed UCCSD VQE, and its convergence proof assumes a uniformly positive-definite pool metric.

For Paper I, the distinct issue is mixed fermion--boson/truncated-phonon rank collapse. Low-cutoff bosonic pool elements can become identical or zero on the encoded truncated space, e.g. at \(n_{\rm ph}=1\), \(X^2=P^2=I\), \(n^2=n\), and \(n(n-I)=0\). Phrase this as a cutoff-induced tangent-redundancy mechanism in our heterogeneous pool, not as a literature claim that Geo-ADAPT already documents phonon-generator rank deficiency.

## Evidence gate before strong venue submission

- Pool-matched controls against canonical ADAPT and feasible modern variants.
- Phase-by-phase ablations.
- Proxy-vs-final compiled cost calibration.
- Frozen tuning/evaluation split or equivalent anti-overfitting protocol.
- Cross-family benchmark matrix.
- Encoding/cutoff/layout panel for bosonic/mixed systems.

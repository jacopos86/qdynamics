# Paper I: RA-ADAPT Static Construction

This glossary defines the domain language used when planning, implementing,
testing, and reporting the Paper-I RA-ADAPT static-construction method.

## Physical calculation

**Resolved physical problem**:
A complete physical calculation definition: Hamiltonian, register layout,
symmetry sector, reference state, cutoff, and exact-comparison space.
_Avoid_: Problem config, Hamiltonian request

**Static ansatz construction**:
Application of a static adaptive construction method to one resolved physical
problem. Paper I and Paper IV are peer producers that may use the same method
with different Hamiltonians and family-owned physical capabilities.
_Avoid_: Paper-I algorithm, Hubbard--Holstein route

**Exact-ED target**:
A predefined exact-diagonalization energy tied to the same resolved physical
problem, sector, and cutoff as the adaptive calculation.
_Avoid_: Oracle energy, correct route

**Canonical provenance baseline**:
The locked Paper-I run contract whose complete recorded settings define an
ordinary RA-ADAPT run unless an explicit later decision replaces one policy.
_Avoid_: Current documentation, registry default

## RA-ADAPT construction

**Controller round**:
One complete cycle that selects an admission proposal and produces the next
accepted state. A batch may add several generators within one controller round.
_Avoid_: Depth, iteration when the intended unit is ambiguous

**Candidate-position record**:
One admissible generator paired with the ansatz position at which it could be
admitted, together with its stable physical and lineage identities.
_Avoid_: Candidate, operator when position or lineage matters

**Admission proposal**:
The singleton candidate-position record or explicitly constructed batch that
Phase III offers for accepted-state expansion.
_Avoid_: Candidate when singleton versus batch matters

**Physical macro pool**:
The unfiltered `full_meta` collection of physical operator families, including
HVA families, from which Phase-I and Phase-II records are formed.
_Avoid_: Pauli pool, winning pool

**Symmetry-retained Pauli child**:
An exact-cardinality-one Pauli term projected from a shortlisted physical macro
and admitted to Phase III only after fixed-sector and binary-padding guards.
_Avoid_: Macro, arbitrary Pauli word

**Hard symmetry guard**:
The mandatory exclusion of a proposed Pauli child that violates the resolved
physical sector or binary-padding contract.
_Avoid_: Symmetry preference, mitigation

**Admission decision**:
The selected admission proposal and the Phase-I/II/III and trust-solve receipts
that justify selecting it. It has not yet changed the accepted state.
_Avoid_: Accepted candidate, new ansatz

**Accepted state**:
The ordered ansatz, parameter identities and values, and energy that have
completed the authorized full refit and any enabled pruning transition.
_Avoid_: Working state, trial state

**Accepted ansatz export**:
An immutable, problem-bound accepted prefix prepared for a downstream paper
lane while preserving its construction method, inclusive accepted-construction
cutoff and source-native index kind, state identity, and source provenance.
_Avoid_: Paper-I seed, artifact JSON

**Accepted transition**:
The transformation from one accepted state to the next by applying an
admission decision, full refit, and any enabled pruning verification.
_Avoid_: Optimizer step, append

**RA-ADAPT request**:
The candidate-representation adapter, scientific method, execution (including
stopping and resume), and observation choices for applying RA-ADAPT to a
resolved physical problem.
_Avoid_: Problem, route flags

**Route family**:
The stable scientific method identity shared by compatible RA-ADAPT profiles.
_Avoid_: Route name, CLI route

**Route profile**:
An immutable resolved realization of a route family, including its active
scientific policies and digest.
_Avoid_: Settings bundle, mode

**Compatibility route**:
A historical, benchmark, ablation, or research implementation reachable only
through an explicit versioned identity and never considered by canonical route
resolution.
_Avoid_: Alternative default, fallback

**Controlled ablation**:
An explicitly requested run that changes one named canonical policy while
preserving and recording the provenance baseline for everything else.
_Avoid_: Mode, fallback route

**One-off experiment**:
A source-locked controlled comparison that does not create a permanent public
policy, route, adapter, or configuration seam.
_Avoid_: Experimental architecture

## Optional method policies

**Conditional policy interview**:
A pre-execution decision branch that is silent while an optional policy is
disabled and reveals only that policy's required choices when enabled.
_Avoid_: Fallback, recovery path

**Progressive-disclosure request**:
An RA-ADAPT request assembled from canonical defaults plus only the conditional
policy interviews activated by user intent.
_Avoid_: Flag bundle, fallback chain

**Admission policy**:
The choice between singleton admission, greedy batching, and combinatorial
batching.
_Avoid_: Batching flag

**Pruning policy**:
The choice between no pruning, metric-nominated measured delete-and-refit, and
trust-region-nominated measured delete-and-refit. The two enabled policies are
peers until evidence selects a default.
_Avoid_: Prune flag

**Beam policy**:
The choice between one accepted continuation and explicit competition among
fork-local accepted lineages.
_Avoid_: Beam flag

**Lineage fork**:
The common accepted state from which two or more accepted continuations begin
to diverge.
_Avoid_: Global root, frozen parent

**Dominant lineage**:
The surviving accepted continuation after a fork-local energy-and-cost
comparison terminates its competitor.
_Avoid_: Correct route, exact branch

## Accounting and stopping

**Estimator primitive**:
A uniquely identified logical estimator quantity counted once in algorithmic
measurement work regardless of repeated consumption.
_Avoid_: Shot, circuit

**Fork ledger prefix**:
The closed estimator-ledger state at a lineage fork that identifies work shared
by all descendants and therefore excluded from their fork-local work.
_Avoid_: Beam cost root, mutable snapshot

**Fork-local estimator work**:
The realized unique estimator primitives attributable to the divergent
portion of a lineage after its common fork.
_Avoid_: Total branch cost when shared history is included

**Stop policy**:
The finite controller-round horizon and any optional predefined exact-ED target
that may terminate a run.
_Avoid_: Convergence flags, source-locked horizon

**Observation policy**:
Checkpoint, logging, diagnostic, and artifact choices that may observe a run
but cannot alter its accepted trajectory.
_Avoid_: Execution policy

## Run reporting

**Accepted error trace**:
The accepted energy and same-cutoff absolute energy error recorded for every
completed controller round of one run.
_Avoid_: Optimization trace, time trajectory

**Effective plateau prefix**:
The earliest accepted prefix within the named reporting tolerance of the best
error observed over the available accepted trajectory.
_Avoid_: Final iteration, converged ansatz

**Common-accuracy target**:
An error threshold that both compared methods attain within their shared
pre-plateau observation window.
_Avoid_: Equal final energy, exact match

**Selected-prefix resource observation**:
Compiled circuit resources and separately accounted algorithmic work for one
identified accepted prefix.
_Avoid_: Run cost, Qiskit score

**Paper-I run summary**:
An additive post-run observation containing the accepted error trace and the
standard selected-prefix resource observations. It cannot affect the accepted
trajectory it describes.
_Avoid_: Analysis run, controller report

## Cross-paper handoff and campaign control

**Static-construction producer lane**:
A paper-owned lane that resolves its physical problem, applies a static
construction method, and owns the resulting evidence. Paper I and Paper IV are
peer producer lanes.
_Avoid_: Parent paper, upstream algorithm

**Downstream consumer lane**:
A paper-owned lane that imports a declared neutral export while owning its new
calculation and evidence. Paper II may consume Paper-I or Paper-IV accepted
ansatz exports; Paper III may later consume declared static or dynamics exports.
_Avoid_: Child paper, inherited route

**Paper-lane adapter**:
A provenance-preserving translation between lane-owned evidence and a neutral
cross-paper input or output. It does not transfer ownership of the source
evidence.
_Avoid_: Schema fallback, artifact search

**Source-locked Paper-I study**:
An explicitly requested remaining Paper-I comparison, such as pruning or
batching, whose unchanged scientific settings remain bound to a named evidence
source.
_Avoid_: Canonical campaign, new default

**Campaign stage receipt**:
A machine-readable record of one campaign stage's authoritative inputs,
outputs, checks, state, and supersession lineage. Folder contents alone do not
determine stage completion.
_Avoid_: Status folder, done marker

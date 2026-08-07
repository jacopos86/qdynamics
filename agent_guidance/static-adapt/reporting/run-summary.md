# Canonical Paper-I Run Summary

This is the only canonical accepted-run summary seam:

```python
from pipelines.reporting.paper_i_run_summary import summarize_paper_i_run

summary = summarize_paper_i_run(
    result.run,
    requested_controller_rounds=(),
)
```

It accepts the typed canonical `SRRunResult` embedded in a `RAAdaptResult`.
The retained receipt type name is not an alternate execution interface. The
ordinary append comparator is selected with the typed
`CANONICAL_APPEND_REFERENCE` marker; it does not accept mappings, raw sentinel
strings, historical schemas, artifact paths, artifact-tree searches, or
recovery fallbacks.

## Standard output

`PaperIRunSummary` owns:

1. every accepted controller-round energy and same-cutoff absolute error;
2. the effective-plateau prefix under `paper_i_effective_plateau_v1`;
3. diagnostic compiled resources for that exact prefix;
4. append-matched common-accuracy observations resolved automatically from the
   exact problem-bound canonical append registry;
5. diagnostic exact accepted-prefix observations for requested controller
   rounds;
6. closed canonical all-work and prefix `S_alg` receipts;
7. problem, route, reference-state, optimizer, candidate-representation, and
   compile-convention provenance;
8. whether the available history is naturally terminal or a deliberately
   stopped prefix.

One batch is one controller round even when it adds several generators, so
every trace row records controller round and active ansatz depth separately.

## Effective plateau

`paper_i_effective_plateau_v1` selects the earliest accepted prefix whose
same-cutoff absolute energy error is within 10 percent of the best error
observed over the available accepted trajectory.

The selected exact prefix is compiled under
`table_i_basis_gate_transpile_v1`. This prefix compilation supports trajectory
analysis and diagnostics. It is not the paper-facing cost row.

## Paper-facing fixed-round resources

Paper-I cost reporting always uses the accepted controller-round-50 prefix for
both SNAKE and Append-ADAPT. The resource tuple is:

```text
(N2q, D2q, Dc, W1q, S_alg) at controller round k = 50
```

Here `N2q` is compiled two-qubit count, `D2q` is compiled two-qubit depth,
`Dc` is compiled total circuit depth, `W1q` is Pauli one-qubit work, and
`S_alg` is closed canonical algorithmic work. Compile the exact round-50
prefix under `table_i_basis_gate_transpile_v1`. Do not substitute the
effective-plateau prefix, a common-accuracy crossing, the first tolerance
crossing, or another earlier prefix in a manuscript table, figure, caption, or
paper-facing PDF.

The canonical Append registry retains all 50 accepted prefixes so plateau and
common-accuracy selectors remain reproducible. Each adopted record also carries
one validated `fixed_controller_round_50_v1` reporting block. Its `S_alg` and
absolute error must match the terminal accepted-prefix receipt; its four
Qiskit-derived fields must come from that same round-50 ansatz.

## Append-matched common accuracy

Comparison requires the same:

- resolved physical problem and cutoff;
- optimizer and optimizer budget;
- seed;
- candidate representation;
- Qiskit compile convention.

The shared window ends at the earlier effective plateau. The common target is
the larger of the best errors achieved by the two methods within that window.
Select the first crossing from each method. Crossing-prefix resource
observations are diagnostic only; paper-facing resources remain fixed at round
50.

The typed default canonical marker resolves the source-locked projected-singleton
append comparator by exact problem-request hash, same-cutoff energy,
reference-state fingerprint, optimizer contract, candidate representation, and
compile convention. A typed `PaperIAppendReferenceResolver` or
`PaperIAppendRunSource` remains available for explicit report tests and locked
adapters. The summary never accepts an artifact path as the ordinary comparison
interface.

## Requested rounds

Pass one or more positive accepted controller rounds:

```python
summary = summarize_paper_i_run(
    result.run,
    requested_controller_rounds=(10,),
)
```

The summary reconstructs the exact accepted prefix and uses the same compiler
and cache key as automatic plateau and common-accuracy observations. A
requested round outside the complete accepted history is an error. Requested
round resources are diagnostic unless the requested round is the canonical
paper-facing round 50.

## Accounting and failure semantics

Canonical algorithmic work is:

```text
S_alg = N_H_outer + N_H_refit + N_grad + N_metric
```

It counts executed logical scalar-estimator occurrences, including repeated
callbacks and measured rejected proposals. Exact-reference evaluation,
fidelity, plotting, and Qiskit compilation are reporting-only.

A Qiskit/compiler defect returns a retryable observation failure. It does not
alter or invalidate the accepted scientific result; repair the tooling and
rerun the same summary.

## Automatic wiring

`run_ra_adapt` attaches this strict summary as
`result.run.paper_i_summary` only after controller finalization for every
completed or deliberately stopped canonical plateau-insertion run. Qiskit
compilation is a post-run observation: a compiler failure is recorded as
retryable and cannot alter or invalidate the accepted scientific result.
`SRObservationPolicy(resource_rounds=(...))` inside a `RAAdaptRequest`
selects any additional accepted prefixes for the automatic summary. Calling
`summarize_paper_i_run` explicitly later uses the same reconstruction,
compiler, cache, and accounting contract.

Explicit append-only compatibility/replay runs remain runnable but do not
receive a canonical plateau summary.

The typed default append marker loads
`canonical-append-registry-v1.json` lazily, verifies its fixed SHA-256, and
returns `canonical_append_reference_not_found` when the exact resolved problem
and comparison contract are absent. The explicit registry builder re-derives
that compact file only from the six adopted v6, hash-locked
projected-singleton archives and their component-adoption receipt. The partial
48-cell report is not adopted as a whole, and no RA cell is promoted by this
Append-only receipt. Ordinary completion never scans those archives or
artifact trees.

For explicit source-maintenance only, run
`PYTHONPATH=. python3 pipelines/reporting/build_paper_i_canonical_append_registry.py`
from the active checkout, then update the exact digest in
`pipelines/reporting/paper_i_append_registry.py` and rerun
`test/test_paper_i_append_registry.py`. This is a reporting derivation, not a
scientific rerun.

Campaign and PDF builders must consume `PaperIRunSummary` decisions directly.
When a source-locked historical input cannot provide an `SRRunResult`, its
explicit typed adapter must use the exported plateau selector and canonical
four-component accounting constructor. Builders must not duplicate plateau,
common-accuracy, prefix reconstruction, or accounting selectors.

Paper-facing cost builders must additionally select the exact accepted
controller-round-50 prefix and must reject an input that lacks that prefix or a
validated round-50 compilation. Plateau and common-accuracy selections do not
override this fixed reporting round.

# Canonical Paper-I RA-ADAPT Interface

This is the minimal agent-facing guide for an ordinary Paper-I RA-ADAPT
request. It defines an interface, not a CLI recipe and not a compatibility
registry.

## Ordinary call

The caller supplies one already resolved physical problem. The canonical
request is silent:

```python
from pipelines.static_adapt.ra_adapt import run_ra_adapt

result = run_ra_adapt(problem)
```

The interface has exactly two arguments:

```python
run_ra_adapt(problem, request=None)
```

Do not add route names, artifact paths, optimizer flags, or dormant policy
settings to an ordinary call. `problem` must be a
`ResolvedProblemContext` for the canonical Hubbard--Holstein `L=2` calculation.

## Silent canonical resolution

The no-request call resolves one typed protocol through
`build_resolved_ra_protocol`. That protocol and its canonical digest are the
sole executable source for scientific defaults. Do not copy its pool,
geometry, trust, refit, admission, insertion, pruning, stopping, or accounting
values into agent Markdown.

Tests under `test/test_ra_adapt_*` lock the facade and scientific contracts.
Historical append-only evidence keeps its original route identity and is never
relabeled. Folder presence is not authority; a materialized bundle does not authorize execution.
Retired executable source is discoverable only through the compact archive
manifest named by the lane router, never through this ordinary guide.

## Explicit request shape

Create `RAAdaptRequest` only when user intent changes the candidate
representation or a named public policy. It contains exactly:

```text
adapter
method
execution
observation
```

Use `RAAdaptRequest` and the candidate adapters exported by
`pipelines.static_adapt.ra_adapt`. Shared policy types retain their stable
qualified names in `pipelines.static_adapt.sr_snake.contracts`; that contracts
module is not a route registry. Do not translate the request into a legacy flag
union.

| Intent | Typed choice | Conditional guide |
|---|---|---|
| batch admission | `GreedyBatchAdmission` or `CombinatorialBatchAdmission` | `policies/batching.md` |
| pruning | `MetricPruning` or `TrustRegionPruning` | `policies/pruning.md` |
| always-open insertion | `AlwaysCommutationReducedInsertion` | `policies/insertion.md` |
| insertion ablation | `AppendOnlyInsertion` | `policies/insertion.md` |
| stopping | `SRStopPolicy` and optional `ExactEDStop` | `policies/stopping.md` |
| bounded beam | `ForkLocalBeam` | `policies/beam.md` |
| accepted-state resume | `AcceptedStateResume` | `policies/resume.md` |
| checkpoint observation | `CheckpointObservation` | only when observation mechanics are requested |
| estimator-ledger observation | `EstimatorLedgerObservation` | only when observation mechanics are requested |
| requested-prefix resources | `SRObservationPolicy(resource_rounds=...)` | `reporting/run-summary.md` |

Disabled policies reveal no subtype settings. Progressive disclosure selects
one complete request before execution; it never retries another scientific
route after a failure.

## Current executable boundary

The direct canonical controller executes singleton, greedy-batch, and
combinatorial-batch admission; canonical plateau insertion; always-open
commutation-reduced insertion; explicit append-only insertion; metric and
trust-region pruning; bounded fork-local
beam; and authenticated accepted-state resume. These policies compose in the
direct controller without entering a compatibility loop. Historical
recoverability pruning remains a private explicit
singleton-plus-append-only-plus-beam-off compatibility identity; it is not an
ordinary typed policy.

Resume is route-exact: the authenticated checkpoint must bind the same
problem, policy-composed route contract, accepted history, signed numerical
prefixes, and estimator ledger. Read `policies/resume.md` before constructing
an accepted-state resume.

## Compatibility wall

An ordinary request never reads or selects:

- `route_a`, Route-A/B/C, JR-SNAKE, or FM-SNAKE;
- Geo-ADAPT, append-ADAPT, H2O, or another paper lane;
- versioned historical SR profiles or their registry aliases;
- historical amplitude, hysteresis, terminal, or mixed pruning;
- raw unreduced `full` insertion or its retired route-profile aliases;
- frozen-parent or unchanged-parent beam archives;
- a legacy adaptive-insertion trigger union;
- `full_meta_minus_hva`, intact-macro admission, no-guard behavior, or
  unwhitened accepted refits.

If the user explicitly names a preserved compatibility identity, stop using
this canonical guide and follow that exact identity's provenance path. Never
fall back from the canonical resolver to compatibility code.

## Result and completion

`run_ra_adapt` returns one typed `RAAdaptResult`. Its `run` field is the
scientific `SRRunResult` receipt containing the resolved problem, route,
accepted trajectory, transition/refit receipts, estimator accounting,
observation receipts, stop receipt, and canonical reporting inputs. The
retained receipt type name is not an alternate public execution seam.

Canonical plateau-insertion runs attach the strict `paper_i_summary`
to `result.run` automatically. Use
`agent_guidance/static-adapt/reporting/run-summary.md` for its contract or for
an explicit later summary call. Reporting is observational and cannot alter a
completed scientific result.

# Conditional Pruning Policy

Read this file only when the user enables or asks about pruning.

## Peer choices

Pruning is off in the ordinary request. Enabling it requires one peer policy:

| Policy | Type | Nomination source |
|---|---|---|
| metric pruning | `MetricPruning` | regularized local metric/response model |
| trust-region pruning | `TrustRegionPruning` | full-logical trust-domain model |

Neither peer falls back to the other.

```python
request = RAAdaptRequest(
    method=SRMethodPolicy(
        pruning=MetricPruning(),
    )
)
```

## Shared acceptance authority

Both peers only nominate a deletion. Acceptance is always the measured
delete-and-complete-refit result. A modeled improvement alone cannot delete an
accepted coordinate.

The transition order is:

```text
admit singleton or batch
-> complete accepted refit
-> nominate deletion
-> measure deletion trial
-> complete deletion refit
-> accept or reject deletion
-> close transition accounting
```

All executed work counts in canonical `S_alg`, including rejected measured
deletion trials. Reporting, exact references, fidelity, and Qiskit compilation
remain outside algorithmic accounting.

`RecoverabilityPruning` is the preserved historical trust-region spelling. It
remains available for explicit compatibility replay but is not an ordinary
peer-policy name. Historical amplitude, hysteresis, terminal, and mixed modes
remain quarantined.

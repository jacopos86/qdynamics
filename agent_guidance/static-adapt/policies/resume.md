# Conditional Accepted-State Resume Policy

Read this file only when the user requests continuation from a canonical
accepted-state checkpoint.

## Typed request

Use `AcceptedStateResume` inside the `SRExecutionPolicy` carried by a
`RAAdaptRequest`:

```python
request = RAAdaptRequest(
    execution=SRExecutionPolicy(
        stop=SRStopPolicy(maximum_controller_rounds=requested_rounds),
        resume=AcceptedStateResume(
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
        ),
    )
)
```

The maximum round must exceed the authenticated checkpoint round. A checkpoint
that already satisfies the requested exact-ED stop is terminal and is not
resumed.

## Authentication contract

Resume accepts only a regular canonical current-checkpoint envelope with:

- the caller-supplied SHA-256;
- exact same-problem and same-cutoff binding;
- the same policy-composed route profile and contract digest;
- complete accepted history and signed active-prefix replay;
- exact operator order, parameter blocks, coordinates, state fingerprint, and
  energy;
- fixed-sector and padding-guard evidence;
- complete trust, maturity, selection-count, and optional prune state;
- a sibling estimator-ledger sidecar whose pointer, SHA-256, fingerprint,
  component counts, unique counts, and prefix receipts all close.

The reader does not call a historical resume scaffold, infer aliases, search
artifacts, repair partial evidence, or fall back to another route.

## Result semantics

The direct session reconstructs the accepted numerical state, verifies it
again, restores controller and ledger state, and executes only later rounds.
The returned `RAAdaptResult.run` is a contiguous `SRRunResult`: accepted
trajectory, transitions, scientific replay, accepted-prefix `S_alg`, and
automatic summary include the authenticated prefix followed by the new
segment.

For beam, the checkpoint must use
`canonical_terminal_winning_lineage`; the global sidecar retains all executed
branches while the resumed controller extends only the authenticated winner.

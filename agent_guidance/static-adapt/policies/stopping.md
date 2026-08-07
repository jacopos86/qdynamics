# Conditional Stopping Policy

Read this file only when the user asks to change the run horizon or add an
exact-ED stop.

## Finite horizon

`SRStopPolicy` and the resolved RA protocol own the current finite horizon. A
controller round is one complete accepted singleton or batch transition, not
one admitted generator and not one optimizer evaluation.

Use another positive integer only when the user explicitly changes the
horizon. Do not copy the default horizon into Markdown.

## Optional exact-ED stop

An `ExactEDStop` must bind:

- a finite target energy;
- a positive absolute tolerance;
- an `ExactEDSourceReceipt` for the same resolved physical problem, sector,
  cutoff, and comparison space.

The check occurs only after a complete accepted refit and at least one accepted
controller round. Exact data cannot enter screening, candidate scoring, trust
solves, optimization, pruning, beam survival, or any online controller choice.
The finite round horizon remains active.

```python
request = RAAdaptRequest(
    execution=SRExecutionPolicy(
        stop=SRStopPolicy(
            maximum_controller_rounds=requested_rounds,
            exact_ed_target=exact_stop,
        )
    )
)
```

A deliberate user stop is valid only at a complete accepted state. Its summary
must label the available horizon as a deliberately stopped prefix and must not
imply that an unobserved later round could not improve the result.

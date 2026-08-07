# Conditional Batching Policy

Read this file only when the user enables or asks about batching.

## Choice

Batching replaces singleton admission with one typed admission policy:

- greedy: `GreedyBatchAdmission`;
- combinatorial: `CombinatorialBatchAdmission`.

The typed constructors and resolved route contract own all sizes, ceilings,
and search-window defaults. Markdown must not restate them.
`FullCombinatorialSearchWindow()` explicitly selects the complete ranked
Phase-III population for combinatorial search.

```python
request = RAAdaptRequest(
    method=SRMethodPolicy(
        admission=GreedyBatchAdmission(),
    )
)
```

## Semantics

- Slice the ranked Phase-III population before admission.
- Selected candidate-position records are ordered, nonempty, and
  generator-distinct.
- Combinatorial search examines subsets, not permutations and not new
  insertion placements.
- One admitted batch is one atomic controller round: one admission, remap,
  commit, complete accepted refit, trust update, ledger close, checkpoint, and
  controller increment.
- A batch can increase active ansatz depth by more than one; reporting must not
  equate controller round with ansatz depth.
- Measured off-diagonal batch geometry counts once physically and records both
  logical metric and Hessian occurrences.

Batching does not change canonical insertion, pruning, beam, stopping, or
reporting unless the request explicitly enables those independent policies.
When batching is absent, do not expose maximum-size or search-window settings.

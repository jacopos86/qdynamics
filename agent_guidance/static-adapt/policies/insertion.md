# Canonical Insertion Policy

Read this file when the user asks about insertion or requests an insertion
ablation. Otherwise the canonical policy is silent.

## Typed canonical policy

The resolved RA protocol owns the canonical insertion policy and its numerical
threshold. Its current typed behavior is:

1. use append-only position construction while accepted progress is adequate;
2. after an accepted transition whose marginal accepted-state energy decrease,
   divided by the mean accepted-state energy decrease before that transition,
   is strictly below the resolved plateau ratio threshold, widen the next
   candidate domain to all admissible logical insertion positions;
3. collapse exactly commuting-equivalent positions to one deterministic
   representative before ranking;
4. return to append-only construction after accepted progress resumes.

If the accepted decreases are
\(d_j=E_{j-1}-E_j\), the domain for round \(k+1\) opens when
\[
\frac{d_k}{\frac{1}{k-1}\sum_{j<k}d_j}
\]
is strictly below the typed threshold. This historical-mean denominator avoids
making the trigger progressively easier merely because more rounds have
elapsed. The ratio is undefined after only one accepted transition, so that
round remains append-only. Exact-ED error, reporting metrics, absolute energy
origin, and future trajectory information cannot open the insertion domain.
The threshold is an internal typed-profile value, not a Markdown default.

Accepted transition and state receipts record the actual insertion position.
Generator identity and generator-position identity remain distinct.

## Explicit always-open policy

Use `AlwaysCommutationReducedInsertion()` when the user requests insertion at
every controller round. Every round begins from the full logical position
domain `range(append_position + 1)`. The controller then uses the same exact
termwise cross-component commutation reducer as an open plateau round and
scores only the earliest representative of each equivalence class.

The runtime mode is `full_commutation_reduced`; the typed discriminator is
`always_commutation_reduced`. The former raw `full` mode, raw-full route
profiles, `full_commutation` typed discriminator, and capped-domain CLI mode
named `always` are retired and must fail closed. There is no executable bypass
around the reducer.

## Explicit append-only ablation

Use `AppendOnlyInsertion()` only when the user explicitly requests append-only
RA-ADAPT insertion or replays preserved historical evidence:

```python
request = RAAdaptRequest(
    method=SRMethodPolicy(
        insertion=AppendOnlyInsertion(),
    )
)
```

This restores the frozen parent insertion behavior. It does not relabel an old
trajectory as canonical, and it does not invoke the separate append-ADAPT
comparator method.

The legacy union of plateau, flatness, repeated-family, and escape triggers is
compatibility-only.

# Strong--strong cumulative-relative RA plateau continuation

This is a one-row, source-locked CHTC runtime package for
`core__strong_strong_u8__nph7__ra_singleton_plateau__r70`.

It resumes the canonical-validated round-50 checkpoint at SHA-256
`b8186aabb56c8fee9ff71d5a6a9c6f5a7c18ea42e36431b65d54fca245386811`.
The only scientific protocol change is the maximum controller horizon from
50 to 70.  The cumulative-relative plateau threshold (`1e-4`), stationary
source gradients, late resource weighting, singleton candidate supply,
POWELL/200 optimizer, and seed 7 are preserved exactly.

The exact round-50 source protocol is also bundled directly under
`protocols/` and bound by the package manifest, job, plan, and audit.  Runtime
and validation load that package-relative copy; they do not resolve the
producer checkout's `output/local_runs/` tree.

The effective source archive may differ from the original round-50
implementation inventory only in the separately attested estimator-ledger
validation-performance repair and occurrence-stable checkpoint-writer repair.
Neither delta changes the protocol, controller decisions, accepted state, or
scientific accounting semantics.

The package is inert: it contains no execution or submission authorization and
no submit descriptor.  The sibling ordinary-held activation supplies the
single-cell execution authorization and remains initially held for an exact
proc release after external CHTC collision and quota checks.

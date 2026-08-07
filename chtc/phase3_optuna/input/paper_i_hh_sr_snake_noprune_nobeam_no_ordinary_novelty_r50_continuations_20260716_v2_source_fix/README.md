# SR-SNAKE round-30 continuation source repair v2

Status: prepared locally; not submitted.

This immutable source-only revision derives from the submitted v1 continuation
archive. It changes only `pipelines/static_adapt/adapt_pipeline.py`: per-round
empty batch telemetry and the inactive joint-selector flag are initialized
before the Phase-II live gate, so a restored Phase-I-only controller snapshot
cannot read uninitialized local state after a successful singleton refit.

The patch does not change route settings, controller state, candidate scoring,
selection, pruning, batching, beam policy, optimizer configuration, or resume
inputs. A future submission must use a separately generated submission bundle
whose Condor arguments and artifact manifests name and hash this revised source
archive. This package itself must not be submitted as the old v1 bundle.

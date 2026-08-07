# Stale preflight bundle

Do not submit this v1 bundle.  Archive-only validation found that the worker's
split-literal `PROFILE` and `SOURCE_LOCK_STATE` constants retained their parent
values even though the manifests and frozen route contract were correctly
derived.  No scientific work was run.  Immutable v2 repairs only those worker
identity constants.

# Stale preflight bundle

Do not submit v3.  Exact-image validation found that its source-only derivation
renamed an internal route constant still imported by `cli_config`.  No
scientific work was run.  Immutable v4 retains that stable symbol and changes
only the resolved profile value and Phase-III selector mode.

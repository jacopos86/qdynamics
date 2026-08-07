# historical_beam_3x2 Phase-live-hysteresis-disabled successor

This immutable v4 bundle supersedes stopped cluster `8887539`.
It preserves that family’s exact v3 source archive and all prior operational
repairs, then applies one scientific correction required by the canonical
full-response contract:

```text
phase_live_hysteresis_enabled = false
```

The route profile now owns the setting, explicit attempts to enable hysteresis
fail closed, and every job remains a fresh six-regime 50-controller-round row.
This bundle was built and locally preflighted but was **not submitted** by the
builder.

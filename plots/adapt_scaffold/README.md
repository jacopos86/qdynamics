# ADAPT scaffold plots

Standalone, read-only plotting utilities for persisted ADAPT artifact JSONs.

## Scope

- Reads existing exported/raw ADAPT JSON artifacts.
- Does **not** modify core files or pipeline behavior.
- Writes plot outputs into a user-specified output directory.

## Views

1. **Energy vs generator iteration**
   - step-based
   - uses persisted `history[*].energy_before_opt` and `history[*].energy_after_opt`

2. **Generator usefulness ranking**
   - ranks admitted generator instances, not shortlist candidates
   - primary metric: `delta_abs_drop_from_prev`
   - fallback metric: exact-error improvement if exact GS energy is present
   - last fallback: raw energy drop `E_before - E_after`
   - batch admissions are allocated by equal split across admitted records in the step

3. **Final scaffold trajectory**
   - uses final persisted `parameterization` + final theta only
   - current path mode is a surrogate:
     - `runtime_prefix_final_theta`
   - this is **not** the true optimizer trajectory because per-step theta snapshots are not persisted

## CLI

```bash
python -m plots.adapt_scaffold.main \
  --input-json path/to/adapt_artifact.json \
  --output-dir output/adapt_plots
```

Optional flags:

- `--ranking-top-k 30`
- `--include-landscape`
- `--compute-surrogate-energy`
- `--orqviz-overlay`

## Optional advanced landscape helper

The package can optionally try an `orqviz` overlay for the final-scaffold landscape.
That dependency is intentionally isolated and not required for the first two plots.


# Held-out-five expansion command

Do not run this command until the CHTC `1e-4` anchor comparison passes and
both predeclared CHTC variant results have completed R50.

```bash
python3 -B \
  chtc/paper_i_ra_adapt_repair_20260727/paper_i_ra_adapt_singleton_phase3_on_plateau_threshold_sweep_weak_weak_r50_20260804_v1_local/materialize_heldout_five.py \
  --anchor-comparison <fetched-chtc-anchor-comparison.json> \
  --tau-1em5-result <fetched-tau-1em5-result.json> \
  --tau-1em6-result <fetched-tau-1em6-result.json> \
  --output-dir \
    chtc/paper_i_ra_adapt_repair_20260727/<new-noncolliding-heldout5-package-id>
```

The command selects the lower terminal round-50 same-cutoff absolute energy
error and refuses materialization unless it is strictly below the locked
weak--weak Append target, `9.416688540042628e-10`. It then reuses the winning
variant's sealed source archive unchanged for these five rows:

- `intermediate_weak`, `nph=3`;
- `strong_weak_u8`, `nph=3`;
- `weak_strong`, `nph=7`;
- `intermediate_strong`, `nph=7`;
- `strong_strong_u8`, `nph=7`.

The output is inert: it contains no authorization overlay and performs no
staging or submission. Activate and submit it through the ordinary CHTC
workflow only after validating the new five-row package.

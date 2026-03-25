# PI Briefing: Phase-3 Novelty Ablation (2026-03-24)

## Verdict

The matched ablation strongly supports novelty weighting as a real contributor to the Phase-3 HH operator selector.

Turning novelty weighting off by setting `phase2_gamma_N = 0.0` worsened the final energy error by about `5.8e3x` in both tested pools:

- `pareto_lean_l2`: `5.62e-05 -> 3.28e-01`
- `full_meta`: `5.62e-05 -> 3.28e-01`

So, in these runs, novelty is not passive telemetry. It materially changes which operators are selected and whether the search reaches the correct low-energy scaffold.

## Artifacts

- [pi_ablate_pareto_lean_l2_novelty_on.json](../json/pi_ablate_pareto_lean_l2_novelty_on.json)
- [pi_ablate_pareto_lean_l2_novelty_off.json](../json/pi_ablate_pareto_lean_l2_novelty_off.json)
- [pi_ablate_full_meta_novelty_on.json](../json/pi_ablate_full_meta_novelty_on.json)
- [pi_ablate_full_meta_novelty_off.json](../json/pi_ablate_full_meta_novelty_off.json)

Historical anchors for context:

- [adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_plateau_20260321T030400Z.json](../json/adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_plateau_20260321T030400Z.json)
- [adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_currentcode_20260321T171615Z.json](../json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_currentcode_20260321T171615Z.json)

## Provenance / Honesty

- All four ablations are fresh direct `adapt_pipeline` runs from `2026-03-24`.
- Same physics in all four runs:
  - `problem=hh`, `L=2`, `t=1.0`, `u=4.0`, `omega0=1.0`, `g_ep=0.5`, `n_ph_max=1`
  - `boson_encoding=binary`, `ordering=blocked`, `boundary=open`
- Same selector/reopt scaffold in all four runs except for:
  - pool choice: `pareto_lean_l2` vs `full_meta`
  - novelty weight: `phase2_gamma_N = 1.0` vs `0.0`
- Same seed, same compiled backend path, same windowed reoptimization, same prune settings, same rescue/symmetry/lifetime settings.
- Important caveat:
  - these are clean matched on/off ablations
  - they are not byte-for-byte reproductions of every March 21/22 auto-resolved runtime knob
  - the command surface left some knobs implicit, so the pipeline resolved staged HH defaults at runtime, notably `adapt_maxiter=300` and automatic drop-policy values
  - that does not weaken the on/off comparison, because those resolved settings were shared within each matched pair

## Compact Table

| Pool | `gamma_N` | Final energy | `abs_delta_e` | Depth | Params | Prune kept | Novelty avg | Novelty min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pareto_lean_l2` | `1.0` | `0.15872408236037028` | `5.6178234644e-05` | `14` | `25` | `4 / 5` | `0.22346` | `3.9996e-06` |
| `pareto_lean_l2` | `0.0` | `0.48696021436469644` | `3.2829231024e-01` | `14` | `23` | `4 / 5` | `0.14287` | `4.9999e-07` |
| `full_meta` | `1.0` | `0.15872408236037047` | `5.6178234644e-05` | `13` | `28` | `5 / 5` | `0.22059` | `2.1016e-06` |
| `full_meta` | `0.0` | `0.48699090951182056` | `3.2832300539e-01` | `13` | `22` | `5 / 5` | `0.14287` | `9.9999e-07` |

Useful ratios:

- `pareto_lean_l2`: novelty-off error is `5843.76x` larger than novelty-on
- `full_meta`: novelty-off error is `5844.31x` larger than novelty-on

## Interpretation

- In both pools, novelty-on lands on the low-error line near `5.62e-05`.
- In both pools, novelty-off converges to a much worse final energy near `0.487`, far above the exact filtered reference near `0.158668`.
- Depth alone does not explain the effect:
  - the on/off pairs stop at the same logical depth in each pool
  - the bad result comes from a different selection path, not from simply stopping earlier
- Pruning stayed active in all runs:
  - the effect is not “novelty on had pruning and novelty off did not”
  - instead, novelty changed which scaffold was built before pruning/refit
- Novelty telemetry still exists in the `gamma_N = 0` runs:
  - the geometry was still computed
  - the selector just stopped using novelty as a multiplicative ranking factor
  - this makes the ablation especially clean

## Relation To The Pruning Story

This also clarifies the pruning claim:

- live pruning is implemented and used
- it ranks removal candidates mainly by small `|theta|`
- it accepts a removal only if post-removal local refit stays within the allowed regression threshold
- estimated remaining depth is part of the Phase-3 lifetime-burden scoring, not the prune accept/reject rule itself

So the novelty result is not secretly a pruning-only effect. Novelty is affecting the upstream operator-ranking trajectory.

## Suggested PI Claim Paragraph

In the current Phase-3 HH selector, I can ablate novelty cleanly by setting `phase2_gamma_N = 0` while leaving the geometric telemetry itself intact. On matched L=2 Hubbard-Holstein runs in both the heavy `full_meta` pool and the pruned `pareto_lean_l2` pool, novelty-on reaches about `5.6e-05` final energy error, whereas novelty-off degrades to about `3.28e-01`. Since the runs share the same physics, seed, reoptimization policy, pruning machinery, and continuation settings, this is strong evidence that reduced-tangent-space novelty is not just recorded metadata; it is doing real work in the ADAPT operator-ranking rule.

## If The PI Asks For One More Check

The next clean follow-on is not another novelty run. It is a single `full_meta` pair with `phase3_lifetime_cost_mode = off` vs `phase3_v1`, to separate novelty from the lifetime-burden term. That is only needed if someone wants to factor the full selector into subcomponents after seeing the much larger novelty signal above.

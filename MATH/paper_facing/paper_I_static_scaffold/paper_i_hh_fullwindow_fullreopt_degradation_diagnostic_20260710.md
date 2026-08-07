# Paper-I HH Full-Window Full-Refit Degradation Diagnostic

Date: 2026-07-10

Scope: diagnostic note only.  This file records plausible causes for the
unexpectedly worse Hubbard--Holstein SNAKE results from the full-window,
full-refit, physical-lane route.  It does not promote or demote manuscript
results.

## Compared Run Roots

- New full-window/full-refit less-aggressive route:
  `raw_outputs/paper_i_hh_physical_operator_lanes_1p75_fullwindow_fullreopt_powell200_nobatch_20260709_v1/less_aggressive_1p75_fullwindow_fullreopt`
- Earlier physical-lane no-batch factor-3 route:
  `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708`

## Current Diagnosis

The bad full-window route should not be interpreted as a clean scientific
result yet.  The strongest code-level explanation is that the candidate universe
was enlarged from append-only position probing to many generator-position
records, while the expensive phases still received a small maturity/controller
capped shortlist.  In addition, selected records in the failed route were
heavily concentrated at insertion positions 0 and 1 and heavily concentrated in
one or two physical lanes, rather than exposing the dressed electron-phonon
response that dominated many earlier successful trajectories.

## Code Evidence

- `pipelines/static_adapt/adapt_pipeline.py` divides the base Phase-I and
  Phase-II shortlist caps by `--physical-lane-shortlist-aggressiveness` when
  `--static-lane-route physical_operator_type` is active.
- `pipelines/static_adapt/lane_routes.py` clamps maturity cap min/max to the
  already-effective shortlist cap for the physical-lane route.
- `pipelines/static_adapt/controller_phase_state.py` uses the stage-controller
  phase cap when present; the shortlist calls then pass `_controller_cap(...)`
  rather than the raw requested cap.
- `pipelines/scaffold/hh_continuation_stage_control.py` schedules phase caps
  between maturity cap min/max as a function of the controller's early/late
  coordinates.
- `pipelines/static_adapt/algebraic_metadata.py` defines record identity using
  candidate label, generator id, pool index, and position id.  The same physical
  generator at different insertion positions can therefore occupy multiple
  record slots.

## Run Telemetry

For the full-window 1.75 route, average insertion positions considered per
accepted step were about 13 to 15, with max 27 to 29.  The earlier factor-3
route considered one append position per accepted step.

The full-window route still retained only about 4.6 to 5 records into the later
shortlist stages in most regimes, with Phase caps usually `(11, 7, 5)` or
`(10, 7, 4)`.  The earlier route used stable caps `(8, 4, 4)`.

Selected insertion positions in the full-window route were almost entirely
positions 0 and 1:

| Regime | `abs_delta_e` | selected-position concentration | selected-lane concentration |
| --- | ---: | --- | --- |
| weak--weak | `2.2544e-2` | positions 0/1 only | 28 phonon-displacement, 2 UCCSD-single |
| intermediate--weak | `1.0961e-1` | positions 0/1 only | 27 electronic-current, 2 UCCSD-single, 1 phonon-displacement |
| strong--weak | `1.4053e-3` | positions 0/1 only | 27 phonon-displacement, 2 UCCSD-single, 1 electronic-current |
| weak--strong | `1.6002e-1` | positions 0/1 only | 28 phonon-displacement, 2 UCCSD-single |
| intermediate--strong | `1.4535e-1` | positions 0/1 only | 28 phonon-displacement, 2 UCCSD-single |
| strong--strong | `7.2878e-3` | positions 0/1 only | 27 phonon-displacement, 2 UCCSD-single, 1 UCCSD-double |

The earlier factor-3 route selected a much broader mixture including dressed
electron-phonon records, phonon-squeeze relaxation, phonon displacement,
electronic-current records, and UCCSD records.

Pruning is not the first explanation.  A prune summary appears at every step,
but actual prune execution counts were small in the failed route: 1 to 3
executed prune events per regime.

## Plausible Causes To Track

| Priority | Plausible cause | Why it fits the evidence | Diagnostic/fix |
| --- | --- | --- | --- |
| P0 | Maturity/controller caps remained active | Full position probing enlarged the record universe, but Phase caps stayed small and scheduled by maturity logic. | Canonical settings should use nonbinding maturity cap sentinels, freeze maturity shots at one, and disable phase-live hysteresis. |
| P0 | Generator-position duplication crowded the shortlist | Record identity includes `position_id`, so one physical operator can appear at multiple positions and consume multiple retained slots. | Audit per-step retained records by unique generator id and lane; consider generator-level protected representatives before position expansion. |
| P0 | Full insertion changed the selected trajectory rather than adding a safe superset | Failed runs selected only positions 0/1, while old append-only probing used one append position per step. | Compare a short run with full refit but append-only insertion, then full insertion with generator-level de-duplication. |
| P1 | Physical-lane protection is too weak after global refill | Failed runs selected almost no dressed electron-phonon records. | Check live-lane representatives and global refill composition; test larger protected-lane fraction or one-record-per-live-lane retention before refill. |
| P1 | Powell 200 may be too low for full-coordinate reoptimization | Full refit increases active dimension to about 13 to 29 parameters; several steps use hundreds of objective evaluations. | For one regime, rerun with a higher Powell budget and compare first 10 accepted steps. |
| P1 | Cost/novelty calibration may be mismatched for insertion positions | Position-shift and refit-active costs can change the relative score under full insertion. | Log top-20 records by raw gain, novelty, cost, lane, and position for a short diagnostic run. |
| P2 | Prune interaction | Prune executed only a few times in the failed route, so it is not the leading explanation. | Keep a no-prune smoke as a control after maturity and position effects are checked. |

## Canonical Settings Correction

The forward canonical settings should disable maturity scheduling as a separate
adaptive route.  Because the physical-lane implementation clamps maturity caps
to the effective shortlist caps, use high sentinel values:

```text
--phase1-maturity-cap-min 999999
--phase1-maturity-cap-max 999999
--phase2-maturity-cap-min 999999
--phase2-maturity-cap-max 999999
--phase3-maturity-cap-min 999999
--phase3-maturity-cap-max 999999
--phase-maturity-shot-min 1
--phase-maturity-shot-max 1
--phase1-maturity-shot-cap 1
--phase2-maturity-shot-cap 1
--phase3-maturity-shot-cap 1
--phase-live-hysteresis-disabled
```

This does not remove the explicit shortlist caps.  It removes the extra
maturity/runway route from deciding a smaller or different cap.

## Next Minimal Diagnostics

1. One short weak--weak run with full refit, full geometry, physical lanes,
   append-only insertion, and maturity-disabled settings.
2. One short weak--weak run with full refit, full geometry, full insertion, and
   maturity-disabled settings.
3. Compare the first 5 to 10 selected records by lane, generator id, position,
   raw gain, novelty, and cost.
4. Only after the maturity and position effects are separated, test a no-prune
   control.

# Investigation: HH site3 checkpoint probe summary

## Scope
Checkpoint-only replay/probe of the shared packet for the repaired guarded d-shape lane (`v18`) and the fidelity-first lane (`v19`), plus a `v19` counterfactual with forced stays at checkpoints `3,7,61`.

Run bundle:
- `artifacts/agent_runs/20260416_hh_l2_high_amp_site3_checkpoint_packet_probe_v1/`
- `json/site3_v18_checkpoint_probe.json`
- `json/site3_v19_checkpoint_probe.json`

Shared probe checkpoints:
- `3,7,11,15,31,35,61,67,102,105,115,117,118,119,120`

## Headline findings
- `v18` probe parity is clean at the late rescue window: the replay reproduces the original protected-horizon append decisions at checkpoints `102,115,117,118,119,120`.
- `v19` has an explicit two-surface contradiction early: at checkpoints `11,15,31,35,67` the leading candidate is **admitted** with `admission_reason=below_floor_componentwise_aspiration`, but the final action is forced back to `stay` with `decision_override_reason=exact_forecast_nonimproving_tracking_score`.
- Forcing away the early `v19` appends at checkpoints `3,7,61` does **not** restore the late `v18` rescue behavior. The late window still lands in `no_tracking_win_vs_stay` at `102+`.
- The `v18` late rescue labels are absent from the probed top candidate set of both `v19` actual-prefix and `v19` force-stay in the late window, so the current branch difference is not only final override; it also changes the late candidate/shortlist geometry.

## Evidence table

### 1. `v18` late protected-horizon parity
| checkpoint | selection reason | final action | winning candidate | parity |
|---|---|---|---|---|
| 102 | `d_shape_barrier_protected_horizon` | `append_candidate` | `paop_full:paop_disp(site=0)` | exact match |
| 115 | `d_shape_barrier_protected_horizon` | `append_candidate` | `paop_full:paop_cloud_p(site=1->phonon=0)` | exact match |
| 117 | `d_shape_barrier_protected_horizon` | `append_candidate` | `paop_full:paop_disp(site=1)` | exact match |
| 118 | `d_shape_barrier_protected_horizon` | `append_candidate` | `paop_full:paop_cloud_p(site=1->phonon=0)` | exact match |
| 119 | `d_shape_barrier_protected_horizon` | `append_candidate` | `paop_full:paop_cloud_p(site=1->phonon=0)` | exact match |
| 120 | `d_shape_barrier_protected_horizon` | `append_candidate` | `paop_full:paop_cloud_p(site=1->phonon=0)` | exact match |

### 2. `v19` actual-prefix contradiction rows
| checkpoint | selection reason | proposed/favored candidate | candidate admitted? | final action | override |
|---|---|---|---|---|---|
| 11 | `below_floor_componentwise_aspiration` | `paop_lf_full:paop_dbl_p(site=1->phonon=1)` | yes | `stay` | `exact_forecast_nonimproving_tracking_score` |
| 15 | `below_floor_componentwise_aspiration` | `paop_lf_full:paop_dbl_p(site=1->phonon=1)` | yes | `stay` | `exact_forecast_nonimproving_tracking_score` |
| 31 | `below_floor_componentwise_aspiration` | `paop_lf_full:paop_dbl_p(site=1->phonon=1)` | yes | `stay` | `exact_forecast_nonimproving_tracking_score` |
| 35 | `below_floor_componentwise_aspiration` | `paop_lf_full:paop_dbl_p(site=1->phonon=1)` | yes | `stay` | `exact_forecast_nonimproving_tracking_score` |
| 67 | `below_floor_componentwise_aspiration` | `paop_lf_full:paop_dbl_p(site=0->phonon=1)` | yes | `stay` | `exact_forecast_nonimproving_tracking_score` |

Additional actual-prefix rows:
- checkpoint `3`: `below_floor_componentwise_aspiration -> append_candidate`
- checkpoint `7`: `below_floor_componentwise_aspiration -> append_candidate`
- checkpoint `61`: `below_floor_componentwise_aspiration -> append_candidate`
- checkpoints `102,105,115,117,118,119,120`: `no_tracking_win_vs_stay -> stay`

### 3. `v19` force-stay counterfactual (`force_stay_3_7_61`)
| checkpoint | selection reason | final action | top candidate / rejection |
|---|---|---|---|
| 3 | `below_floor_componentwise_aspiration` | forced `stay` | candidate still admissible |
| 7 | `below_floor_componentwise_aspiration` | forced `stay` | candidate still admissible |
| 11 | `below_floor_componentwise_aspiration` | `append_candidate` | append survives when earlier appends are removed |
| 15 | `below_floor_componentwise_aspiration` | `stay` | overridden again by `exact_forecast_nonimproving_tracking_score` |
| 31 | `outside_exact_turn_window` | `stay` | top candidate rejected: `outside_energy_safe_window` |
| 35 | `outside_exact_turn_window` | `stay` | top candidate rejected: `outside_energy_safe_window` |
| 61 | `outside_exact_turn_window` | `stay` | top candidate rejected: `outside_energy_safe_window` |
| 67 | `outside_exact_turn_window` | `stay` | top candidate rejected: `outside_energy_safe_window` |
| 102+ | `no_tracking_win_vs_stay` | `stay` | late rescue still absent |

Interpretation:
- Removing the early `v19` appends changes the middle-window gate story (`checkpoint 11` now appends, `31-67` fall back to `outside_energy_safe_window`), but it does **not** restore the late `v18` rescue regime.
- So the early `v19` appends are not the sole cause of the late failure basin.

### 4. Late-window candidate-family mismatch (`v18` label presence inside `v19` top candidates)
| checkpoint | `v18` winning label | present in `v19` actual top set? | present in `v19` force-stay top set? |
|---|---|---|---|
| 102 | `paop_full:paop_disp(site=0)` | no | no |
| 115 | `paop_full:paop_cloud_p(site=1->phonon=0)` | no | no |
| 117 | `paop_full:paop_disp(site=1)` | no | no |
| 118 | `paop_full:paop_cloud_p(site=1->phonon=0)` | no | no |
| 119 | `paop_full:paop_cloud_p(site=1->phonon=0)` | no | no |
| 120 | `paop_full:paop_cloud_p(site=1->phonon=0)` | no | no |

Caveat:
- This statement is only about the probed top candidate set (`candidate_rank_limit=8`, or fewer when shortlist/confirmed counts are smaller). It is evidence of shortlist/ranking divergence, not yet proof that the full pool lacks the `v18` family.

## Baseline/stay hints
Top baseline margins from the probe:
- `v18` margins are mostly tie-like (`0.0`) at early/mid checkpoints and small even at `102` (`~8.94e-4`), suggesting brittle/tie-dominated stay selection for much of the trajectory.
- `v19` margins are also tie-like at many early checkpoints, but become visibly nontrivial by the late failing window (`115: ~9.22e-2`, `120: ~1.60e-1`), suggesting the late stay basin is already more structurally separated there.

Representative chosen baseline tuples:
- `v18@61`: `tangent_secant_exact_v1`, step `1.15`, gain `1.0`
- `v18@102`: `norm_locked_blend_v1`, blend `1.0`, step `2.5`, gain `1.3`
- `v19@61`: `norm_locked_blend_v1`, blend `0.5`, step `1.0`, gain `1.0`
- `v19@102`: `norm_locked_blend_v1`, blend `-0.5`, step `0.0`, gain `1.0`
- `v19@120`: `norm_locked_blend_v1`, blend `-0.5`, step `1.2`, gain `1.4`

## Current read
1. Best immediate local seam remains the shared guarded-policy mismatch: `below_floor_componentwise_aspiration` can admit a candidate that the final law still cancels with `exact_forecast_nonimproving_tracking_score`.
2. Best global-cause candidate remains upstream stay/baseline basin construction: even after forcing away the early `v19` appends, the late window still never reaches the `v18` rescue family.
3. Append-family insufficiency is still not the first default story, but it is back on the table as a **shortlist/ranking divergence** question because the `v18` late labels are absent from the probed top `v19` candidate set.

## Line-level code seam for the contradiction
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:4236-4267`
  - `_exact_v1_componentwise_aspiration_result(...)` uses `_exact_v1_guarded_turn_window_core_score(...)` and rejects only when `selected_score >= stay_score` or when downstream barrier/energy checks fail.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:3832-3850`
  - `_exact_v1_guarded_turn_window_core_score(...)` uses **core-only** scoring: `fidelity_first_core_score` for `fidelity_first_barrier_v1`, otherwise `live_d_shape_core_score`.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:4477-4505`
  - `_exact_v1_forecast_override_reason(...)` later compares `selected_score = _forecast_tracking_score(selected_forecast)` against `stay_score = _forecast_tracking_score(stay_forecast)` and returns `exact_forecast_nonimproving_tracking_score` if the selected append no longer wins.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:3876-3898`
  - in the fidelity-first branch, `_forecast_tracking_score(...)` dispatches to `_exact_v1_fidelity_first_score(...)`, which adds the live d-barrier penalty back on top of the fidelity-first core.

Plain-language interpretation:
- admission can be decided on **core-only guarded score**,
- but final commit can still be decided on **core + barrier total score**,
- which is exactly the two-surface mismatch seen at checkpoints `11/15/31/35/67`.

## Next seams to test
- Single-surface commit law: make the same guarded comparison surface govern both tentative admission and final commit-vs-stay.
- Stay-basin audit: explain why `v18@102+` and `v19@102+` choose materially different stay baselines and why the late `v18` rescue family drops out of the `v19` top candidate set.
- Only after those: broaden append-family investigation if the shortlist/ranking audit shows the needed family is truly absent, not just demoted.

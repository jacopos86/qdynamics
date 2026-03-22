# HH L=2 `phase3_v1` plateau-only rerun summary

- generated_utc: 2026-03-09T18:24:55Z
- source_runs: 6 staged noiseless HH workflow JSONs from the 2026-03-09 plateau-only rerun batch

## Definitions

- `E_exact`: exact filtered-sector HH ground-state energy for the same truncated HH problem, with the same `L, U, omega0, g_ep, n_ph_max, boundary, ordering`.
- `ΔE := |E - E_exact|`. In this report, `ΔE_replay` means `|E_replay - E_exact|`.

## Common run configuration

- model: Hubbard-Holstein (HH)
- lattice: `L=2`
- physics baseline: `t=1.0`, `U=4.0`, `dv=0.0`, `omega0=1.0`, `boundary=open`, `ordering=blocked`
- points:
  - weak: `g_ep=0.5`, `n_ph_max=1`
  - knee: `g_ep=1.25`, `n_ph_max=2`
  - strong: `g_ep=1.5`, `n_ph_max=2`
- warm-start ansätze compared: `hh_hva_ptw`, `hh_hva`
- continuation setup:
  - `adapt_continuation_mode=phase3_v1`
  - `phase3_lifetime_cost_mode=phase3_v1`
  - `phase3_runtime_split_mode=shortlist_pauli_children_v1`
  - `phase3_symmetry_mitigation_mode=off`
- optimizers:
  - warm start: `SPSA`
  - ADAPT inner: `SPSA`
  - replay: `SPSA`
- workflow-scaled defaults selected by the staged wrapper:
  - warm/replay reps=`2`, restarts=`4`, maxiter=`1778`
  - ADAPT maxiter=`2222`, max_depth=`240`
  - dynamics: `trotter_steps=128`, `t_final=10.0`, `num_times=135`
- plateau-only ADAPT stop configuration:
  - `adapt_drop_floor=0.0005`
  - `adapt_drop_patience=3`
  - `adapt_drop_min_depth=12`
  - `adapt_grad_floor=-1.0`
- stopping outcome: all 6 ADAPT runs stopped with `drop_plateau`; none stopped with `max_depth`.

## Top-line conclusions

- PTW replay `ΔE` by point: weak `0.0069856490`, knee `0.0383602593`, strong `0.1209236844`.
- Layerwise replay `ΔE` by point: weak `1.6207674408`, knee `0.3135713618`, strong `0.2837240138`.
- `hh_hva_ptw` is better than `hh_hva` at weak, knee, and strong.
- Plateau-only stopping changed the stop reason as intended, but it did not change the ordering of warm-start quality.
- Weak PTW is the only point close to the practical `1e-2` quality band; knee is moderate; strong remains far from exact.
- Practical choice for future HH L=2 work: keep `hh_hva_ptw` as the warm start.

## PTW warm start (`hh_hva_ptw`)

### Weak point

- physics: `g_ep=0.5`, `n_ph_max=1`
- `E_exact = 0.1586679041`
- `E_warm = 0.1797385781`
- `ΔE_warm = 0.0210706739`
- `E_adapt = 0.1656535531`
- `ΔE_adapt = 0.0069856490`
- `adapt_depth = 16`
- `adapt_stop = drop_plateau`
- `E_replay = 0.1656535531`
- `ΔE_replay = 0.0069856490`
- `replay_stop = converged`
- warm gate `ecut_1 <= 0.1`: `True`
- final gate `ecut_2 <= 0.0001`: `False`
- final static dynamics abs-energy error vs ground state:
  - `suzuki2 = 0.0070028032`
  - `cfqm4   = 0.0069856490`
- artifacts:
  - workflow_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_weak_ptw.json`
  - adapt_handoff_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_weak_ptw_adapt_handoff.json`
  - replay_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_weak_ptw_replay.json`
  - replay_md: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/useful/L2/20260309_plateau_v3_weak_ptw_replay.md`
  - replay_log: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/logs/20260309_plateau_v3_weak_ptw_replay.log`

### Knee point

- physics: `g_ep=1.25`, `n_ph_max=2`
- `E_exact = 0.0602872922`
- `E_warm = 0.1774929603`
- `ΔE_warm = 0.1172056681`
- `E_adapt = 0.1016136050`
- `ΔE_adapt = 0.0413263128`
- `adapt_depth = 20`
- `adapt_stop = drop_plateau`
- `E_replay = 0.0986475515`
- `ΔE_replay = 0.0383602593`
- `replay_stop = converged`
- warm gate `ecut_1 <= 0.1`: `False`
- final gate `ecut_2 <= 0.0001`: `False`
- final static dynamics abs-energy error vs ground state:
  - `suzuki2 = 0.0384851546`
  - `cfqm4   = 0.0383602593`
- artifacts:
  - workflow_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_knee_ptw.json`
  - adapt_handoff_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_knee_ptw_adapt_handoff.json`
  - replay_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_knee_ptw_replay.json`
  - replay_md: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/useful/L2/20260309_plateau_v3_knee_ptw_replay.md`
  - replay_log: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/logs/20260309_plateau_v3_knee_ptw_replay.log`

### Strong point

- physics: `g_ep=1.5`, `n_ph_max=2`
- `E_exact = -0.0318802757`
- `E_warm = 0.1109568753`
- `ΔE_warm = 0.1428371510`
- `E_adapt = 0.0926942045`
- `ΔE_adapt = 0.1245744803`
- `adapt_depth = 18`
- `adapt_stop = drop_plateau`
- `E_replay = 0.0890434087`
- `ΔE_replay = 0.1209236844`
- `replay_stop = converged`
- warm gate `ecut_1 <= 0.1`: `False`
- final gate `ecut_2 <= 0.0001`: `False`
- final static dynamics abs-energy error vs ground state:
  - `suzuki2 = 0.1184391976`
  - `cfqm4   = 0.1209236844`
- artifacts:
  - workflow_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_strong_ptw.json`
  - adapt_handoff_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_strong_ptw_adapt_handoff.json`
  - replay_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_strong_ptw_replay.json`
  - replay_md: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/useful/L2/20260309_plateau_v3_strong_ptw_replay.md`
  - replay_log: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/logs/20260309_plateau_v3_strong_ptw_replay.log`

## Layerwise warm start (`hh_hva`)

### Weak point

- physics: `g_ep=0.5`, `n_ph_max=1`
- `E_exact = 0.1586679041`
- `E_warm = 1.7794353449`
- `ΔE_warm = 1.6207674408`
- `E_adapt = 1.7794353449`
- `ΔE_adapt = 1.6207674408`
- `adapt_depth = 13`
- `adapt_stop = drop_plateau`
- `E_replay = 1.7794353449`
- `ΔE_replay = 1.6207674408`
- `replay_stop = converged`
- warm gate `ecut_1 <= 0.1`: `False`
- final gate `ecut_2 <= 0.0001`: `False`
- final static dynamics abs-energy error vs ground state:
  - `suzuki2 = 1.6104105623`
  - `cfqm4   = 1.6207674408`
- artifacts:
  - workflow_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_weak_layerwise.json`
  - adapt_handoff_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_weak_layerwise_adapt_handoff.json`
  - replay_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_weak_layerwise_replay.json`
  - replay_md: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/useful/L2/20260309_plateau_v3_weak_layerwise_replay.md`
  - replay_log: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/logs/20260309_plateau_v3_weak_layerwise_replay.log`

### Knee point

- physics: `g_ep=1.25`, `n_ph_max=2`
- `E_exact = 0.0602872922`
- `E_warm = 1.2659841312`
- `ΔE_warm = 1.2056968389`
- `E_adapt = 0.6197686217`
- `ΔE_adapt = 0.5594813295`
- `adapt_depth = 16`
- `adapt_stop = drop_plateau`
- `E_replay = 0.3738586540`
- `ΔE_replay = 0.3135713618`
- `replay_stop = converged`
- warm gate `ecut_1 <= 0.1`: `False`
- final gate `ecut_2 <= 0.0001`: `False`
- final static dynamics abs-energy error vs ground state:
  - `suzuki2 = 0.3142296916`
  - `cfqm4   = 0.3135713618`
- artifacts:
  - workflow_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_knee_layerwise.json`
  - adapt_handoff_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_knee_layerwise_adapt_handoff.json`
  - replay_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_knee_layerwise_replay.json`
  - replay_md: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/useful/L2/20260309_plateau_v3_knee_layerwise_replay.md`
  - replay_log: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/logs/20260309_plateau_v3_knee_layerwise_replay.log`

### Strong point

- physics: `g_ep=1.5`, `n_ph_max=2`
- `E_exact = -0.0318802757`
- `E_warm = 0.9273600326`
- `ΔE_warm = 0.9592403084`
- `E_adapt = 0.2518437381`
- `ΔE_adapt = 0.2837240138`
- `adapt_depth = 21`
- `adapt_stop = drop_plateau`
- `E_replay = 0.2518437381`
- `ΔE_replay = 0.2837240138`
- `replay_stop = converged`
- warm gate `ecut_1 <= 0.1`: `False`
- final gate `ecut_2 <= 0.0001`: `False`
- final static dynamics abs-energy error vs ground state:
  - `suzuki2 = 0.2844453504`
  - `cfqm4   = 0.2837240138`
- artifacts:
  - workflow_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_strong_layerwise.json`
  - adapt_handoff_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_strong_layerwise_adapt_handoff.json`
  - replay_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_strong_layerwise_replay.json`
  - replay_md: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/useful/L2/20260309_plateau_v3_strong_layerwise_replay.md`
  - replay_log: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/logs/20260309_plateau_v3_strong_layerwise_replay.log`

## Artifact index

- `20260309_plateau_v3_weak_ptw`
  - workflow_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_weak_ptw.json`
  - replay_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_weak_ptw_replay.json`
  - replay_md: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/useful/L2/20260309_plateau_v3_weak_ptw_replay.md`
- `20260309_plateau_v3_knee_ptw`
  - workflow_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_knee_ptw.json`
  - replay_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_knee_ptw_replay.json`
  - replay_md: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/useful/L2/20260309_plateau_v3_knee_ptw_replay.md`
- `20260309_plateau_v3_strong_ptw`
  - workflow_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_strong_ptw.json`
  - replay_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_strong_ptw_replay.json`
  - replay_md: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/useful/L2/20260309_plateau_v3_strong_ptw_replay.md`
- `20260309_plateau_v3_weak_layerwise`
  - workflow_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_weak_layerwise.json`
  - replay_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_weak_layerwise_replay.json`
  - replay_md: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/useful/L2/20260309_plateau_v3_weak_layerwise_replay.md`
- `20260309_plateau_v3_knee_layerwise`
  - workflow_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_knee_layerwise.json`
  - replay_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_knee_layerwise_replay.json`
  - replay_md: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/useful/L2/20260309_plateau_v3_knee_layerwise_replay.md`
- `20260309_plateau_v3_strong_layerwise`
  - workflow_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_strong_layerwise.json`
  - replay_json: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/json/20260309_plateau_v3_strong_layerwise_replay.json`
  - replay_md: `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/useful/L2/20260309_plateau_v3_strong_layerwise_replay.md`

## Short recommendation

- Freeze `hh_hva_ptw` as the default HH L=2 warm start.
- If you want the next accuracy check, run the strong PTW point at `n_ph_max=3` under the same plateau-only stop rule.

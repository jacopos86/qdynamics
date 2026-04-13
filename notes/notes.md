# Residual Notes From Prompt Exports, Handoffs, And Notes Folder

This file keeps only the prompt-export, handoff, and notes-folder intents that still look live but are not yet fully implemented in the repo.

## 1. Real-QPU raw-path consolidation is still incomplete

### Source prompts
- `prompt-exports/2026-03-26-qpu-vqe-raw-gpt-pro-prompt.md`
- `prompt-exports/gpt_pro_qpu_vqe_adapt_handoff.md`

### Intended direction
- Make raw measured circuits the canonical real-QPU surface for HH VQE / ADAPT.
- Quarantine `EstimatorV2` as legacy compatibility.
- Keep durable raw-shot / raw-job artifacts as the research-facing contract.

### Current repo state
- The raw path exists:
  - `raw_measurement_v1`
  - `SamplerV2` runtime transport
  - fake-backend `backend.run(counts)` support
- But the runtime expectation / energy-only path still exists and still uses `EstimatorV2` in:
  - [noise_oracle_runtime.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/exact_bench/noise_oracle_runtime.py)
- The later note [QPU_RAW_VQE_IMPLEMENTATION_SPEC.md](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/notes/QPU_RAW_VQE_IMPLEMENTATION_SPEC.md) already corrects the older `backend.run()` assumption and says current IBM production transport should be `SamplerV2`, not literal `backend.run()`.

### Residual
- Finish the canonical-path cleanup so the repo has one clearly documented real-QPU default:
  - IBM runtime raw path = `SamplerV2`
  - raw artifacts = canonical research surface
  - `EstimatorV2` = explicit legacy / compatibility-only lane

## 2. L3 soft-opening / family-prior schedule is still a live gap

### Source prompt
- `prompt-exports/2026-04-05-l3-soft-opening-gpt-pro-handoff.md`

### Intended direction
- Keep the `phase3_v1` selector math, but add a separate family-opening prior / schedule so simpler HH families dominate early and broader families open later.
- Main scientific motive: recover the good `L=3` strong-coupling basin instead of falling into the early bad basin.

### Current repo state
- The repo has the current `phase3_v1` selector and the child-generation fix.
- I did not find an actual soft-opening / family-schedule control in the active static ADAPT path:
  - [adapt_pipeline.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/static_adapt/adapt_pipeline.py)
  - [hh_continuation_stage_control.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/scaffold/hh_continuation_stage_control.py)

### Residual
- If we want to preserve one mathematically motivated open implementation target for `L=3`, this is the one to keep:
  - family-opening schedule / prior
  - without redefining novelty or replacing `phase3_v1`

## 3. The live-pruning remaining-work heuristic still looks unfinished

### Source prompts
- `prompt-exports/2026-03-24-121803-plan-phase3-adapt-live-pruning.md`
- `prompt-exports/2026-03-24-133622-plan-phase3-live-pruning-heuristic.md`

### Intended direction
- Replace the shallow `max_depth - current_depth` style remaining-work proxy with a more realistic estimate of remaining useful search horizon.
- Use that estimate in live-pruning / lifetime-aware scoring.

### Current repo state
- The current active selector still exposes only:
  - `auto`
  - `none`
  - `remaining_depth`
- This is still wired through:
  - [adapt_pipeline.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/static_adapt/adapt_pipeline.py)
  - [hh_continuation_types.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/scaffold/hh_continuation_types.py)
- The prompt’s desired richer remaining-work model does not appear to have landed.

### Residual
- Preserve this as an open ADAPT-selector improvement:
  - better remaining-work / remaining-useful-depth estimate
  - then wire that into live pruning and lifetime-cost scoring

## 4. Do not preserve these as active residuals

- `notes/IMPLEMENT_SOON.md`
  - This is mostly the old ADAPT-to-replay continuation target and no longer matches the current direction well enough to keep as the central residual note.
- The fixed-scaffold mitigation-matrix prompt
  - This one appears implemented in the local fake-backend path already.
- The spectra prompt
  - The repo now has an implemented spectra stack in [hh_time_dynamics_spectra.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/time_dynamics/hh_time_dynamics_spectra.py).
- The raw-symmetry additive slice prompt
  - The additive symmetry-diagnostic slice appears implemented and tested in the raw-measurement path.

## 5. Kingston state-prep ranking remains a live residual

### Source note
- `notes/HH_KINGSTON_STATE_PREP_IMPROVEMENT_SPEC.md`

### Intended direction
- Add schedule-aware prepared-state ranking and diagnostics for hardware-facing HH state preparation.
- The note specifically calls for signals such as:
  - `scheduled_duration_dt`
  - `max_qubit_idle_dt`
  - empty-circuit thermal checks
  - rep-delay sensitivity
  - prefix-mirror survival / return diagnostics

### Current repo state
- The current shortlist / compile-aware surfaces do have lighter compile proxies, for example:
  - [hh_adapt_backend_shortlist.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/static_adapt/hh_adapt_backend_shortlist.py)
- But the stronger Kingston-style schedule-aware diagnostics do not appear as first-class active surfaces in the current pipeline stack.

### Residual
- Keep exactly this one note-folder residual alive:
  - hardware schedule-aware state-prep ranking
  - idle-time / thermal / rep-delay diagnostics
  - mirror-style prepared-state quality probes

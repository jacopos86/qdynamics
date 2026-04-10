# HH Real-QPU Raw-Measurement Brief

Scope:
- HH only
- real QPU architecture only
- audience: superior repo agent, then GPT-Pro planner
- objective: make raw measured circuits the canonical research surface

## Short answer

Our current real-QPU path is not the right long-term research surface. The repo's main real `runtime` route centers `EstimatorV2`, while the next things we care about, realtime dynamics and adaptive VQE, want raw shot-level, basis-level, and job-level data that can be reused, regrouped, and audited later.

The desired direction is simple: raw measured circuits via `backend.run()` should become the canonical real-hardware surface. `SamplerV2` is an acceptable stepping stone, but raw counts and raw job records are the real target.

## 1. What I want

- Real QPU work should not center on `EstimatorV2`.
- The canonical research surface should be raw measured circuits via `backend.run()`.
- `SamplerV2` is acceptable as an intermediate path, but not the final architectural target.
- The reason is not stylistic. Realtime dynamics and adaptive VQE need shot-level data, basis grouping, reuse, and a durable raw audit trail.

## 2. What the repo does now

- [noise_oracle_runtime.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/exact_bench/noise_oracle_runtime.py) uses `EstimatorV2` for the real `runtime` path.
- The same file already contains a runtime grouped-sampling path through `SamplerV2`.
- [hh_realtime_measurement.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/hh_realtime_measurement.py) already lets the checkpoint controller consume raw grouped samples on runtime-like paths.
- [adapt_pipeline.py](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/adapt_pipeline.py) still treats phase-3 oracle gradients as a local-surrogate path. It does not expose a real-runtime raw-data mode.

## 3. Why this is insufficient

- `EstimatorV2` returns compressed expectation values, not the canonical raw dataset.
- That is weak for measurement reuse, basis-level bookkeeping, offline re-analysis, and a true raw audit trail.
- The repo is therefore split across two incompatible ideas:
- Realtime/controller work already points toward raw grouped measurement data.
- ADAPT scouting is still framed as local oracle energies rather than a real raw-measurement contract.

## 4. Desired architecture direction

- Build one shared raw-QPU execution layer.
- That layer should own measured-circuit compilation, basis grouping, counts capture, job metadata, and cache/reuse behavior.
- Realtime checkpoint control and ADAPT phase-3 should both sit on top of that same layer.
- The first principle should be raw-first. Mitigation can come later as offline or post-processing logic.
- Local modes should remain intact. This brief is about the real-QPU surface.

## 5. What I am not asking for yet

- I am not asking to implement this in this handoff.
- I am not asking for full mitigation parity in the first pass.
- I am not asking to replace every legacy estimator route in one shot.
- I am not asking for a generic Qiskit rewrite. This should stay tightly repo-aware.

## 6. What I want GPT-Pro to plan

- Exact interface and mode names for the new raw-QPU layer.
- The cache and job-record contract.
- The migration path for the realtime checkpoint controller.
- The migration path for HH ADAPT phase-3 oracle gradients.
- How legacy `EstimatorV2` routes should be quarantined, kept, or deprecated.
- The tests and acceptance criteria for saying the architecture is correct.

## Planning questions for GPT-Pro

1. What should the canonical new surface be called in this repo: a new `noise_mode`, a new execution mode, or a separate raw-QPU service layer?
2. Should `backend.run()` be the direct long-term contract, with `SamplerV2` only as a temporary compatibility layer, or should both remain first-class?
3. What is the minimal raw job record we must persist per measurement group so later agents can reproduce, audit, and re-aggregate results without rerunning hardware?
4. How should grouped-basis caching be represented so both the realtime controller and ADAPT phase-3 can share the same reuse semantics?
5. Where should the architecture draw the boundary between online execution and offline post-processing, especially for readout mitigation and other future corrections?
6. How should legacy `EstimatorV2` energy-only routes be isolated so they remain usable without continuing to define the main research architecture?
7. What is the smallest safe migration slice that proves the architecture on real use cases: realtime controller first, ADAPT phase-3 first, or both together?


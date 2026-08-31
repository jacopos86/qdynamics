# Paper-II Reusable Seed Tracks Plan Critique

Scope: critique only of `docs/plans/paper-ii-reusable-seed-tracks-2026-05-28.md` against `prompt-exports/oracle-plan-2026-05-28-112123-seed-track-plan-d420-b260.md`; no broad codebase exploration beyond named docs and mandatory run/calibration guidance.

## 1. Top 3 under-specified seams

1. **Static-source selection/regeneration seam.** Item 4 asks for 12 revised-HH seed sources and says missing/wrapper-only strict Geo/append sources are regenerated (`docs/plans/paper-ii-reusable-seed-tracks-2026-05-28.md:86-95`), but not how TSV record IDs map to actual static result artifacts, what selector chooses among multiple completed artifacts, or what command/report proves regeneration succeeded. The export flags the key risk: existing strict Geo/append outputs may be wrapper-only and need regeneration after `runtime_seed.json` support (`prompt-exports/oracle-plan-2026-05-28-112123-seed-track-plan-d420-b260.md:443-449`).
2. **Runtime-seed schema seam.** The plan requires `settings`, `adapt_vqe`, `initial_state`, and `ansatz_input_state` (`docs/plans/...:62-68`) and a normalizer (`docs/plans/...:74-80`), but drops the export’s concrete `runtime_seed.json` framing and extra schema fields (`ground_state`, `paper_ii_static_seed_export`) (`prompt-exports/...:244-267`). Implementers would guess whether to extend existing static result JSON, add a sidecar, or normalize only in the dynamics builder.
3. **Dynamics case/method/drive seam.** The plan says “two drives” and fixed `t_final=8.0`, `num_times=321` (`docs/plans/...:42`) but omits the export’s explicit `A=0.2 or 0.6`, case-id pattern, and visible method IDs (`prompt-exports/...:219-242`, `:318-326`). That matters before generating records, smoke selectors, or summaries.

## 2. Specificity balance

- **Over-specified:** forcing a shared `paper_ii_seed_track_common.py` and v2 helper refactor as Item 1/6 (`docs/plans/...:50-56`, `:110-120`) may pre-decide architecture; an implementation agent could first build the HH v1 path with local validation, then extract common helpers if reuse is clean.
- **Dropped useful framing:** the export’s file-by-file impact lists stale PosGeo prose correction, `generic_static_benchmark.py` propagation of the runtime seed path, configurable closeout knobs, and duplicate-row blocking (`prompt-exports/...:374-442`). The plan keeps some as tests but not as owned implementation work (`docs/plans/...:146-150`).
- **Dropped useful framing:** the export forbids `position_policy=best_insert_refit` for Geo (`prompt-exports/...:196-202`); the plan only forbids PosGeo identity/source paths (`docs/plans/...:27-31`).

## 3. Contradictions or missing dependencies

- Item 4 should depend on Item 3’s validation/loadability helper, not only on Item 2 “if” regeneration is needed (`docs/plans/...:86-96`); otherwise source selection may precede the validator that defines valid.
- Item 10’s “local dry-run rows cover all revised HH regimes and all three seed tracks” (`docs/plans/...:158-168`) is stricter than the export’s smoke minimum of at least one HH regime across all tracks (`prompt-exports/...:538-547`). Decide whether this is smoke or a larger candidate dry-run.
- Table/report updates are listed as key files in Item 11 (`docs/plans/...:170-176`) while the plan says placeholders remain until validated; this should be a downstream dependency, not part of candidate batch implementation.

## 4. Risk of over-planning

- Cut or defer Item 12 and the non-HH Open Questions (`docs/plans/...:182-208`); they do not affect the visible HH path.
- Merge Item 6 into Items 3/5/9 as a “preserve v2 behavior” acceptance criterion.
- Compress Validation and rollout (`docs/plans/...:194-203`) because it mostly repeats Items 9-11.

## 5. Questions whose answers would change implementation order

1. Are `A=0.2/0.6` the locked visible HH drives, or must the drive lock be re-read before generator work?
2. Do current strict Geo/append HH static artifacts already contain replayable state, or must runtime sidecar support and static reruns happen first?
3. Is duplicate-row/closeout hardening required before any local dry-run, or only before post-fetch/table-facing aggregation?
4. Which exact visible dynamics method IDs define “visible-method rows” for HH seed tracks?
5. May the HH builder emit partial diagnostic manifests, or only missing-source reports until all 12 seeds validate?

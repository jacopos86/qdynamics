# Oracle Review

## Summary

The implementation matches the approved safest path: static ADAPT now owns the QEB construction, `uccsd_qeb` is Hubbard-only, UCCSD-first structural dedup is guarded by a surviving-QEB audit, and Hubbard physical lanes now include `qeb_excitation`. I do **not** see a code/science-settings blocker for a **single local Hubbard weak run** using the source-locked 1.75 physical-lane settings, assuming the Oracle/source-lock gate is explicitly recorded before launch.

## P0 — Must Fix / Launch Gates

- **`agent_guidance/skills/paper-i-run/SKILL.md`; `agent_guidance/skills/source-locked-sensitivity/SKILL.md` — Oracle/source-lock approval artifact not shown**
  - The provided evidence shows good source-lock reconstruction and pool audit, but the run gates still require explicit Oracle/user approval and a machine-readable settings-diff record before launch.
  - **Suggestion:** before starting the tmux/local supervisor, write/attach the launch audit with:
    - source JSON/root + SHA256,
    - exact reused weak command/settings,
    - science setting changed: `adapt_pool: uccsd → uccsd_qeb`,
    - output path changed separately,
    - expected provenance consequences: Hubbard classifier/route variant v2,
    - `pool_label_audit.other_count == 0`,
    - `qeb_excitation > 0`,
    - child/runtime/shared split modes all `off`.

## P1 — Should Fix

- **`pipelines/exact_bench/generic_static_adapt_variants.py` — unrelated comparator/fidelity changes are mixed into the launch patch**
  - The QEB delegation is appropriate, but the same diff also adds dense exact-state fidelity and runtime-seed behavior changes unrelated to the Hubbard weak static-ADAPT run.
  - **Suggestion:** not a local-launch blocker if this path is not invoked, but split or explicitly annotate these as unrelated/non-invoked in the provenance note before any merge/promotion.

- **`pipelines/static_adapt/adapt_pipeline.py` / supervisor manifest — ensure the audit is over the final emitted pool**
  - The in-pipeline guard appears correctly placed after pool resolution, but the launch manifest should prove the same final labels the run will score.
  - **Suggestion:** have the supervisor persist `pool_label_audit.json` from the resolved `PoolResolution.pool` after all filters/expansions. Since source split modes are off, this should match the custom smoke counts.

## P2 — Consider

- **`pipelines/exact_bench/generic_static_adapt_variants.py` — old local QEB helper functions are now dead code**
  - After delegating to `build_qeb_pool_specs`, the old exact-bench QEB helper functions remain.
  - **Suggestion:** remove later to avoid two apparent QEB implementations.

- **`pipelines/contracts/static_provenance.py` — UCCSD matching is prefix-based**
  - QEB labels are anchored, but UCCSD labels still use `startswith`.
  - **Suggestion:** optional hardening: use anchored regexes for `uccsd_sing(...)` and `uccsd_dbl(...)` too. Not required for this run because emitted UCCSD labels are canonical.

- **Later plateau-cost work — `first_eps_energy_termination_condition` naming**
  - This is fine only if it means first prefix crossing the source-locked benchmark target, not optimizer `eps_energy` stagnation.
  - **Suggestion:** for later cost extraction, define it as first batch-safe prefix with `abs_delta_e < 1e-5`; since this weak launch has no batching, prefix handling is simpler.

## Launch Verdict

No additional narrow code fix is required before the local Hubbard weak run **if** the Oracle/source-lock audit is recorded. Do not change split modes, batching, optimizer, seed, or shortlist settings; change only `--adapt-pool uccsd_qeb` and the output root.
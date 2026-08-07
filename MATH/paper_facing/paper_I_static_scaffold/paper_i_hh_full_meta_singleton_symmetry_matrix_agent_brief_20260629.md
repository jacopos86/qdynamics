# Paper I HH Full-Meta Singleton Symmetry Matrix Agent Brief

Created: 2026-06-29

Purpose: hand this file to a run agent before preparing or launching the next
Paper-I Hubbard--Holstein SNAKE / Geo-ADAPT / append-only ADAPT matrix.

This is an execution brief, not manuscript wording. GPT-Pro's response was used
only as advisory direction. The corrected user decisions below are authoritative
for this matrix.

## Source Anchors

Before building commands, read these gates and settings sources:

1. `MATH/AGENTS.md`.
2. `agent_guidance/skills/paper-i-run/SKILL.md`.
3. `agent_guidance/shared/run-guide.md`.
4. `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_snake_canonical_runtime_settings_draft_20260627.md`.
5. `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_snake_canonical_runtime_settings_draft_20260627.pdf`.

Use the canonical settings audit as the baseline for SNAKE algorithmic settings:
Route A, `paper_i_production_v1`, ROTOSOLVE overlay, depth cap 30, maxiter 200,
final refit maxiter 200, phase liveness/hysteresis off, collapse witness off,
and the suggested canonical cost/backend/window/batching/prune/maturity values.

This brief intentionally overrides older helper text that assumes
`full_meta_minus_hva`. For this matrix, the active parent pool is unfiltered
`full_meta` for SNAKE, Geo-ADAPT, and append-only ADAPT.

## Fixed Matrix Contract

Run the six canonical Hubbard--Holstein regimes:

| Regime | `U/t` | `lambda` | Working cutoff |
| --- | ---: | ---: | --- |
| `weak-weak` | 0.25 | 0.25 | `n_ph_work=2` |
| `intermediate-weak` | 1.25 | 0.25 | `n_ph_work=2` |
| `strong-weak` | 8 | 0.25 | `n_ph_work=2` |
| `weak-strong` | 0.25 | 1.25 | `n_ph_work=4` |
| `intermediate-strong` | 1.25 | 1.25 | `n_ph_work=4` |
| `strong-strong` | 8 | 1.25 | `n_ph_work=4` |

Run exactly these methods unless the user separately requests diagnostics:

| Display method | Method id / route |
| --- | --- |
| SNAKE | `static_family_native_adapt_phase3` / Route A / `paper_i_production_v1` |
| Geo-ADAPT | `static_geo_adapt_vqe` |
| append-only ADAPT | `static_full_meta_append_adapt_vqe` |

Fixed across all rows:

- Parent pool: `full_meta`.
- Pool class filter: none. Do not apply `agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json`.
- Inner optimizer: ROTOSOLVE for all three methods.
- Optimizer budget: `adapt_maxiter=200` and final/refit maxiter 200.
- SNAKE canonical settings: inherit from the canonical runtime settings audit except for the child/symmetry policy varied below.
- Child-set size: singleton only, `max_subset_size=1`.
- Do not mix in TETRIS, QEB, HEA, PosGeo, family-informed VQE, or old selected-logical routes.

## Matrix Rows

| Label | Role | SNAKE policy | Geo/append policy | Symmetry |
| --- | --- | --- | --- | --- |
| `A_native_staged_singleton_hard_guard` | Main strongest disclosed SNAKE route | Native Phase-III singleton child split after macro/record shortlisting | Singleton child candidates exposed through generic comparator split | hard guard |
| `A_native_staged_singleton_no_guard` | Symmetry ablation for main route | Same native staged singleton split | Same comparator singleton split | off / no guard |
| `B_common_phase0_singleton_hard_guard` | Strict common-exposure fairness control | Same parent-plus-singleton pool exposed at Phase 0; no later Phase-III split | Same parent-plus-singleton pool exposed at Phase 0 | hard guard |
| `B_common_phase0_singleton_no_guard` | No-guard strict fairness control | Intended same parent-plus-singleton pool exposed at Phase 0; no later Phase-III split | Intended same parent-plus-singleton pool exposed at Phase 0 | off / no guard |
| `C_macro_only` | Macro-generator control | Macro generators only | Macro generators only | not applicable |

The main interpretive comparison is layered evidence: row A shows native SNAKE
with full disclosure; row B controls for identical candidate-exposure timing;
row C measures the effect of child resolution; no-guard rows measure the effect
of removing the symmetry guard across all methods.

## Policy Details To Encode

### A: native staged singleton rows

SNAKE:

- `--adapt-pool full_meta`
- no `--adapt-pool-class-filter-json`
- `--phase3-runtime-split-mode shortlist_pauli_children_v1`
- `--allow-archival-phase3-runtime-split`
- `--phase3-runtime-split-selection-mode archival_child_set_forward_v1`
- `--phase3-runtime-split-max-subset-size 1`
- `--adapt-child-pool-expansion-mode off`
- `--shared-pauli-pool-mode off`
- hard-guard row: use the SNAKE runtime-split symmetry setting that enforces the fixed-sector guard if supported by the runner.
- no-guard row: use the SNAKE runtime-split symmetry setting with no hard guard; do not silently leave hard guard on.

Geo-ADAPT and append-only ADAPT:

- `base_pool_name=full_meta`
- no HVA/class-filter profile
- `generic_adapt_runtime_split_mode=shortlist_pauli_children_v1`
- `generic_adapt_runtime_split_max_subset_size=1`
- hard-guard row: `generic_adapt_runtime_split_symmetry_policy=hard_guard`
- no-guard row: `generic_adapt_runtime_split_symmetry_policy=off`

### B: common Phase-0 singleton rows

The intended row-B contract is identical parent-plus-singleton exposure before
the outer selection rule acts. For SNAKE, disable Phase-III runtime split in row
B; do not combine Phase-0 child exposure with later SNAKE child exposure.

Preferred hard-guard implementation if the runner supports it:

- `--shared-pauli-pool-mode shared_pauli_child_sets_v1`
- `--shared-pauli-pool-symmetry-policy hard_guard`
- `--shared-pauli-pool-max-subset-size 1`
- `--phase3-runtime-split-mode off`
- `--adapt-child-pool-expansion-mode off`
- equivalent shared-pool env/metadata overlays for Geo-ADAPT and append-only ADAPT.

No-guard Phase-0 warning:

- Current shared/global child-pool code may require `hard_guard` when Phase-0 child expansion is enabled.
- The requested scientific row is still `B_common_phase0_singleton_no_guard` for all three methods.
- If current code cannot express an identical Phase-0 singleton pool with symmetry off, fail closed and report this row as blocked. Do not run hard guard and label it no guard. Do not silently fall back to SNAKE native Phase-III exposure.

### C: macro-only row

All methods:

- `adapt_pool=full_meta`
- no HVA/class-filter profile
- no child expansion of any kind
- SNAKE: `--phase3-runtime-split-mode off`, `--adapt-child-pool-expansion-mode off`, `--shared-pauli-pool-mode off`
- Geo/append: `generic_adapt_runtime_split_mode=off`, `shared_pauli_pool_mode=off`

## Required Preflight Before Launch

Before any candidate or paper-facing launch:

1. Materialize the planned command/manifest rows without running the full matrix.
2. Verify every planned row records:
   - method id;
   - regime;
   - optimizer `ROTOSOLVE`;
   - parent pool `full_meta`;
   - empty/no class-filter path;
   - child policy row label;
   - child-set cap `1`;
   - symmetry policy `hard_guard`, `off`, or `not_applicable`;
   - SNAKE route/profile ids;
   - settings hash or normalized command hash.
3. Verify no row contains the active class-filter path
   `agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json`.
4. Run command parsing or the smallest safe smoke first. Do not submit a full
   matrix if any command-generation path silently inserts `full_meta_minus_hva`,
   mixed optimizers, mixed child caps, or an unrequested hard guard.

## Reporting Expectations

Use compact status tables grouped by row label, method, and regime. Report:

- status;
- depth / accepted prefix;
- `abs(Delta E)`;
- first plateau prefix when available;
- fidelity when available;
- `N2q`, `D2q`, and `D_c` when available;
- estimator-work / operator-probe count when available;
- parent-pool hash;
- expanded-pool hash when child exposure is enabled;
- selected parent / singleton child counts.

Do not call any row paper-promotable. Report objective evidence, missing fields,
failed gates, blocked rows, hashes, and risk notes. The user decides what to
promote, defer, rerun, or edit.

## Agent Notes From GPT-Pro Review, Corrected

Useful direction retained:

- Use layered evidence rather than one fake notion of fairness.
- Compare native SNAKE against strict common-exposure controls.
- Include macro-only controls.
- Compare hard symmetry guard against no symmetry guard.
- Keep ROTOSOLVE fixed so the experiment probes outer ansatz construction.

Corrections made by the user:

- Do not use `full_meta_minus_hva` for this matrix.
- Do not let GPT-Pro determine manuscript wording.
- Use singleton child sets, not cap-3 poly-child sets.
- Apply the no-symmetry-guard comparison to all methods, not SNAKE only.

## Success Criteria For The Next Agent

The run agent succeeds when it can show a preflight manifest or command table
with the rows above, all six regimes, all three methods, ROTOSOLVE fixed,
`full_meta` unfiltered everywhere, singleton child cap everywhere child-enabled,
and hard-guard/no-guard status recorded literally. If any row is unsupported by
the current code, the agent should report the exact unsupported row and the
blocking parser/generator rule instead of substituting a different policy.

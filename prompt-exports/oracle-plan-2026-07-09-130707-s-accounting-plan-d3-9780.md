## Final Prompt
<taskname="S Accounting Plan"/>

<task>
Build a concise, implementation-ready plan for safely updating Paper-I Hubbard--Holstein S-accounting after the shadow audit, without applying active manuscript/table changes. The plan should be actionable but must not instruct the next agent to overwrite active `MATH/paper_details/Paper_I.tex`, raw outputs, or support artifacts until explicit user approval and the required gates are satisfied.

Use the final audit convention: displayed-prefix logical scalar estimator-query accounting. Recommended HH SNAKE table S values are the mechanism-formula values: weak-weak 5009, intermediate-weak 4142, strong-weak 4126, weak-strong 17305, intermediate-strong 28617, strong-strong 5153. Runtime S is diagnostic only: weak-weak 4469, intermediate-weak 3740, strong-weak 3678, weak-strong 16627, intermediate-strong 27387, strong-strong 4613.
</task>

<architecture>
- `pipelines/reporting/build_paper_i_hh_s_accounting_shadow.py` is the non-destructive shadow builder. It reads current active `Paper_I.tex`, current HH comparison/provenance CSVs, raw SNAKE JSONs, and comparator support/override CSVs; writes shadow CSV/JSON/MD under `output/pdf/paper_i_hh_s_accounting_shadow_20260709/`; and writes duplicate TeX candidates `Paper_I_s_accounting_shadow_runtime_20260709.tex` and `Paper_I_s_accounting_shadow_mechanism_formula_20260709.tex`.
- `MATH/paper_details/Paper_I.tex` is the active source. Selected slices include the current HH table/provenance block, current old Appendix A S-accounting wording/formula, and appendix non-HH S rows that are inventoried but blocked.
- `Paper_I_s_accounting_shadow_mechanism_formula_20260709.tex` is the recommended shadow candidate: it adds a machine-readable shadow block, changes only the HH SNAKE S cells to mechanism-formula values, and updates Appendix A to P0/P1/P2/P3 scalar estimator-query wording.
- `Paper_I_s_accounting_shadow_runtime_20260709.tex` records runtime S values for diagnostics; do not recommend those as active values.
- `pipelines/exact_bench/snake_table_i_measurement_work.py` defines `snake_algorithmic_work_from_payload()` (runtime/display-prefix S_alg) and `snake_mechanism_resolved_work_from_payload()` (mechanism-formula reconstruction). The latter treats candidate exposure counts as diagnostics/formula operands, reconstructs phase2 window formula components when raw windows exist, preserves non-phase2 metric work, and publishes mechanism S only when checks pass.
- `pipelines/static_adapt/selector_measurement_proxy.py` provides `controller_proxy_from_history_rows()`, which builds display-prefix controller summaries from accepted history rows while carrying by_phase/by_scope and candidate ledger fields.
- `pipelines/exact_bench/generic_static_metric_enrichment.py` defines event-ledger S_alg summing: H_outer + grad + metric + H_refit with unit weights, rejecting legacy proxies for paper-facing S_alg.
- Guidance gates: root `AGENTS.md`, `MATH/AGENTS.md`, and `paper-i-run/SKILL.md` require Paper-I run gate plus Paper-I results gate for table/source-map/current-status updates. `agent_guidance/skills/paper-i-results/SKILL.md` is absent in the workspace, so active promotion must record this as a blocker unless the user explicitly overrides with replacement instructions.
</architecture>

<selected_context>
- `pipelines/reporting/build_paper_i_hh_s_accounting_shadow.py` (full): non-destructive builder; constants for current HH rows, recommended conventions, comparator-visible values, appendix S inventory, replacement Appendix A text/formula, shadow provenance block insertion, and output paths.
- `MATH/paper_details/Paper_I.tex` slices:
  - 1140-1669: active HH narrative, canonical settings guard, `BEGIN_MACHINE_READABLE_HH_PHYSICAL_LANE_DUPLICATE_UPDATE_20260708`, and current visible table values: SNAKE S = 7144, 5799, 5933, 20108, 33487, 7434.
  - 1718-1767: current Appendix A old P1/P2/P3 SNAKE accounting and compatible-Pauli grouping sentence.
  - 2190-2367: appendix non-HH S cells: spin-boson/Rabi 927 and 5237; Bose--Hubbard 6381 and 7283; Hubbard 867 and 1472.
- `MATH/paper_details/Paper_I_s_accounting_shadow_mechanism_formula_20260709.tex` slices: recommended duplicate with shadow block and HH SNAKE S cells = 5009, 4142, 4126, 17305, 28617, 5153; updated Appendix A scalar estimator-query convention and P0/P1/P2/P3 formula.
- `MATH/paper_details/Paper_I_s_accounting_shadow_runtime_20260709.tex` slice: runtime shadow block with diagnostic S cells = 4469, 3740, 3678, 16627, 27387, 4613.
- `MATH/paper_details/figures/paper_i_physical_lane_snake_duplicate_20260708/paper_i_physical_lane_snake_duplicate_20260708_append_parent_only_provenance.json` (full): current visible plot/table source JSON hashes, marker policies, visible methods, and strong-strong Append k=8 override source/hash.
- `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_snake_nobatch_duplicate_promotion_20260707.json` (full): prior duplicate-only HH promotion manifest; records preservation of Geo/Append rows, row policy, S_alg source, source hashes, and previous duplicate-only scope.
- `pipelines/exact_bench/snake_table_i_measurement_work.py` slices: SNAKE work constants, display-prefix runtime reconstruction, mechanism event parsing/classification, phase2 window formula reconstruction, mechanism publishability, and `mechanism_algorithmic_work` output used by the shadow builder.
- `pipelines/static_adapt/selector_measurement_proxy.py` slices: controller/candidate ledger constants, merge helpers, and `controller_proxy_from_history_rows()` prefix summary behavior.
- `pipelines/exact_bench/generic_static_metric_enrichment.py` slices: S_alg component aliases and `algorithmic_measurement_work_from_row()` event-ledger summing/rejection policy.
- `test/test_snake_table_i_measurement_work.py` slices: fixtures and tests covering diagnostic candidate exposure, by_scope/by_phase behavior, phase2 formula publication, non-phase2 metric preservation, final-prefix history scoping, and blocking on missing/nonfinite prefix data.
- `AGENTS.md`, `MATH/AGENTS.md`, `agent_guidance/skills/paper-i-run/SKILL.md`, `agent_guidance/skills.md`, `MATH/paper_facing/shared/journal_math_skill_supplement.md`: gate/approval/manuscript-planning rules.
</selected_context>

<relationships>
- Active HH table rows in `Paper_I.tex` -> shadow builder exact-string row replacement -> duplicate TeX candidates only.
- `build_paper_i_hh_s_accounting_shadow.compute_shadow_rows()` -> current comparison CSV visible S fields + SNAKE raw JSONs -> `snake_algorithmic_work_from_payload(scope="display_prefix", history_position=k_pl)` for runtime S -> `snake_mechanism_resolved_work_from_payload(scope="display_prefix", history_position=k_pl)` for mechanism-formula S.
- `snake_mechanism_resolved_work_from_payload()` -> `snake_algorithmic_work_from_payload()` first for coarse S_alg -> `controller_proxy_from_history_rows(prefix)` for scoped controller summary -> phase/event classification + phase2 window formula -> `mechanism_algorithmic_work` publishable S.
- `snake_algorithmic_work_from_payload()` -> `_scoped_runtime_s_norm_components_from_payload()` -> `algorithmic_measurement_work_from_row()` -> unit-weight S_alg component sum.
- Shadow builder verifies Append/Geo visible S against nested prefix-aligned support components; strong-strong Append uses the k=8 override sidecar.
- Active promotion path must route through Paper-I run/results gates and user approval; missing `paper-i-results/SKILL.md` is a blocker to record.
</relationships>

<ambiguities>
- The generated shadow CSV/JSON/MD directory `output/pdf/paper_i_hh_s_accounting_shadow_20260709/` was not selectable/readable in the RepoPrompt workspace. Its expected artifact paths are encoded in the builder, but contents are not included here.
- Absolute comparator support CSV/JSON paths under `/Users/jakestrobel/Documents/.../output/pdf/...` and current raw-output JSONs are referenced by provenance/builder but are not directly selected. Use selected provenance/source hashes and builder constants rather than inventing missing contents.
- `agent_guidance/skills/paper-i-results/SKILL.md` is referenced by gates but absent; treat active table promotion as blocked pending user override or restoration of that results skill.
- Appendix non-HH S rows are visible and inventoried, but source JSON mappings for `paper_i_alt_hamiltonian_1p75_20260709` figures are absent/unselectable; plan should leave those rows untouched.
</ambiguities>

<expected_output>
Return a concise implementation-ready plan, not code edits. Include: (1) safety boundaries/no-overwrite rule, (2) evidence/gate preflight including missing results skill blocker, (3) exact active HH table cells and Appendix A text/formula targets, (4) shadow artifact review steps using duplicate TeX and builder outputs, (5) validation/test checks to run before any approved active update, (6) active promotion steps that require explicit user approval, and (7) remaining blockers for appendix non-HH S rows and unavailable shadow CSV/JSON/MD contents.
</expected_output>

## Selection
- Files: 15 total (5 full, 10 slice)
- Total tokens: 79228 (Auto view)
- Token breakdown: full 26008, slice 53220

### Files
### Selected Files
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/
├── MATH/
│   ├── paper_details/
│   │   ├── figures/
│   │   │   └── paper_i_physical_lane_snake_duplicate_20260708/
│   │   │       └── paper_i_physical_lane_snake_duplicate_20260708_append_parent_only_provenance.json — 6,205 tokens (full)
│   │   ├── Paper_I.tex — 11,527 tokens (lines 1140-1669 (Active Paper-I HH results section: current visible HH table rows, canonical settings guard, HH physical-lane duplicate provenance block, and current pre-audit SNAKE S cells that any approved promotion would update.), 1718-1767 (Active Appendix A total estimator-query accounting: old ADAPT/Geo/SNAKE formulas and old logical expectation-value/compatible-Pauli wording targeted by shadow convention update.), 2190-2367 (Active non-HH appendix S rows for spin-boson/Rabi, Bose-Hubbard, and Hubbard; builder inventories these but blocks updates because source JSON mappings are absent.))
│   │   ├── Paper_I_s_accounting_shadow_mechanism_formula_20260709.tex — 2,696 tokens (lines 1538-1677 (Shadow mechanism-formula duplicate: machine-readable S-accounting shadow block and HH SNAKE table cells updated to recommended mechanism-formula S values.), 1733-1784 (Shadow mechanism-formula Appendix A: replacement P0/P1/P2/P3 scalar estimator-query formula and displayed-prefix logical scalar convention wording.))
│   │   └── Paper_I_s_accounting_shadow_runtime_20260709.tex — 213 tokens (lines 1541-1567 (Shadow runtime duplicate provenance block listing diagnostic runtime S values; runtime convention is diagnostic only, not the recommended active Paper-I S update.))
│   ├── paper_facing/
│   │   ├── paper_I_static_scaffold/
│   │   │   └── paper_i_hh_snake_nobatch_duplicate_promotion_20260707.json — 4,889 tokens (full)
│   │   └── shared/
│   │       └── journal_math_skill_supplement.md — 2,752 tokens (full)
│   └── AGENTS.md — 1,069 tokens (lines 86-145 (MATH paper-facing gate summary: run/results skills, visible-target gating, user-only promotion/demotion, and manuscript-refiner requirement before paper-edit plans.))
├── agent_guidance/
│   ├── skills/
│   │   └── paper-i-run/
│   │       └── SKILL.md — 2,480 tokens (lines 94-148 (Paper-I active paths, visible target/read-order gate, best-visible-settings baseline gate, and run class definitions.), 614-705 (Paper-I report/table handoff: table mutation belongs to paper-i-results, preservation-first rule, minimum table handoff bundle, and strict replay provenance requirements.))
│   └── skills.md — 1,548 tokens (full)
├── pipelines/
│   ├── exact_bench/
│   │   ├── generic_static_metric_enrichment.py — 4,819 tokens (lines 1-130 (Generic metric enrichment constants and S_alg component aliases; defines algorithmic_measurement_work schema and unit weights.), 742-1040 (Event-ledger to S_alg conversion and algorithmic_measurement_work_from_row: sums H_outer, grad, metric, and H_refit components as logical algorithmic estimator/probe work, rejecting legacy proxies.))
│   │   └── snake_table_i_measurement_work.py — 15,541 tokens (lines 1-130 (Module imports and constants defining SNAKE measurement-work schemas and logical estimator-request charge basis used by the shadow builder.), 1320-2539 (Display-prefix runtime S_alg reconstruction: history prefix row selection, explicit prefix outer-nfev policy, scoped runtime components, and public snake_algorithmic_work_from_payload start.; Mechanism-resolved SNAKE work: controller event parsing, by-scope/by-phase handling, phase2 window formula reconstruction, publishability rules, and mechanism_algorithmic_work output used for recommended mechanism-formula S.))
│   ├── reporting/
│   │   └── build_paper_i_hh_s_accounting_shadow.py — 10,614 tokens (full)
│   └── static_adapt/
│       └── selector_measurement_proxy.py — 4,444 tokens (lines 1-70 (Selector measurement proxy constants for controller work/candidate ledgers and logical estimator-request charge basis consumed by SNAKE measurement-work accounting.), 623-742 (Controller event merge and candidate-work ledger aggregation; preserves by_scope and candidate exposure diagnostics without promoting them directly to S_alg.), 803-862 (Controller summary merge helpers used when combining nested by_phase/by_scope summaries from history rows.), 1733-1847 (controller_proxy_from_history_rows: builds display-prefix controller summary from accepted history rows, carrying by_phase/by_scope and candidate ledger fields for scoped accounting.))
├── test/
│   └── test_snake_table_i_measurement_work.py — 9,133 tokens (lines 1-70 (Test imports and helper field constructors for candidate ledgers, common exposure, and actual probe count basis.), 153-352 (Runtime and prefix payload fixtures plus _add_by_scope helper used by SNAKE mechanism/prefix accounting tests.), 365-534 (Mechanism tests ensuring candidate exposure counts remain diagnostic, by-phase-only summaries stay partial, raw event exposure aliases stay operands, and missing/mismatched typed probe counts block reconciliation.), 565-639 (Prefix runtime tests for batch candidate evaluation handling; shows runtime S metric changes are scoped and diagnostic relative to mechanism formula accounting.), 793-967 (Display-prefix mechanism tests: prefix history-row source, phase2 window formula component reconstruction, publishable formula S, non-phase2 metric preservation, and final-prefix ignoring terminal summary bins.), 1172-1346 (Prefix runtime gate tests: history rows not terminal summaries, missing/nonfinite counts block, history_position required, and missing prefix outer-nfev blocks S_alg.))
└── AGENTS.md — 1,298 tokens (lines 122-204 (Global mandatory Paper run/results gates, user-only promotion authority, visible-target gate, and Paper-I results skill requirement.))


---

## Generated Plan

## Chat Send ✅
- **Chat**: `s-accounting-plan-D3CEB1` | **Mode**: plan

### Response
# Summary

Safely update Paper-I Hubbard--Holstein \(S\)-accounting by treating the shadow audit as evidence, not as an active promotion. The correct convention is displayed-prefix logical scalar estimator-query accounting, with HH SNAKE values taken from the mechanism-formula shadow candidate. This is a targeted manuscript/provenance update, not a code refactor: no raw outputs, existing support artifacts, or active `MATH/paper_details/Paper_I.tex` should be overwritten until explicit user approval is given and the Paper-I run/results/manuscript gates are satisfied.

# Current-state analysis

- `pipelines/reporting/build_paper_i_hh_s_accounting_shadow.py` is already the correct non-destructive extension point. It reads:
  - active `MATH/paper_details/Paper_I.tex`;
  - current HH comparison CSV;
  - raw SNAKE result JSONs;
  - comparator support/override CSVs.
- It writes shadow CSV/JSON/MD artifacts and duplicate TeX candidates only.
- `Paper_I_s_accounting_shadow_mechanism_formula_20260709.tex` is the recommended candidate:
  - changes only HH SNAKE \(S\) cells to mechanism-formula values;
  - inserts a shadow provenance block;
  - updates Appendix A to displayed-prefix scalar estimator-query accounting.
- `Paper_I_s_accounting_shadow_runtime_20260709.tex` is diagnostic only and must not drive active table values.
- SNAKE accounting data flow:
  - raw SNAKE payload → `snake_algorithmic_work_from_payload(scope="display_prefix")` → runtime diagnostic \(S\);
  - raw SNAKE payload → `snake_mechanism_resolved_work_from_payload(scope="display_prefix")` → mechanism-formula \(S\);
  - `controller_proxy_from_history_rows()` supplies display-prefix by-phase/by-scope controller summaries;
  - `algorithmic_measurement_work_from_row()` enforces unit-weight component sums and rejects legacy proxies.
- Active promotion is blocked unless:
  - the user explicitly approves active manuscript/table changes;
  - Paper-I run gate is followed;
  - Paper-I results gate is followed;
  - manuscript-refiner requirements are followed for `.tex` edits.
- `agent_guidance/skills/paper-i-results/SKILL.md` is referenced but absent in the provided workspace. Treat this as an active-promotion blocker unless the user explicitly restores the skill or provides replacement override instructions.

# Design

## Safety boundaries

Before explicit approval:

- Do **not** edit `MATH/paper_details/Paper_I.tex`.
- Do **not** overwrite raw outputs.
- Do **not** overwrite existing support/provenance artifacts.
- Do **not** copy a shadow duplicate over the active manuscript.
- Do **not** update root `Paper_I.pdf`.
- Do **not** use runtime \(S\) as active HH table values.

Allowed pre-approval work:

- read shadow artifacts;
- regenerate shadow artifacts only under a new stamp/output directory if the current shadow directory is missing;
- compare active TeX against the mechanism-formula duplicate;
- run tests and TeX validation on duplicates;
- prepare an approval packet.

## Final HH SNAKE table target

Only the HH SNAKE \(S\) cells change. All \(k_{\rm pl}\), \(|\Delta E|\), \(N_{2q}\), \(D_{2q}\), \(D_c\), Geo, Append, plot, and non-HH appendix values stay unchanged.

| Regime | Current active SNAKE \(S\) | Recommended mechanism-formula \(S\) | Runtime diagnostic \(S\) |
|---|---:|---:|---:|
| weak-weak | 7,144 | 5,009 | 4,469 |
| intermediate-weak | 5,799 | 4,142 | 3,740 |
| strong-weak | 5,933 | 4,126 | 3,678 |
| weak-strong | 20,108 | 17,305 | 16,627 |
| intermediate-strong | 33,487 | 28,617 | 27,387 |
| strong-strong | 7,434 | 5,153 | 4,613 |

## Appendix A target

Target convention:

- \(S\) is a displayed-prefix logical scalar estimator-query count.
- Count scalar expectation-value primitives used by the algorithm at the relevant prepared state.
- Apply same-state reuse and algebraic symmetry.
- Do not count raw Pauli decompositions, compatible-Pauli grouping, shot allocation, or physical shots in \(S\).
- Use the SNAKE \(P_0/P_1/P_2/P_3\) de-duplication formula:

\[
S_{\rm SNAKE}
=
\sum_{k=0}^{k_{\rm pl}}
\left[
|P_0(k)|
+ |P_1(k)\setminus P_0(k)|
+ |P_2(k)\setminus(P_0(k)\cup P_1(k))|
+ |P_3(k)\setminus(P_0(k)\cup P_1(k)\cup P_2(k))|
+ f_k
\right].
\]

Implementation rule: if active `Paper_I.tex` already contains this final convention, leave Appendix A unchanged and record that it already matches. If it still contains the older P1/P2/P3 or compatible-Pauli wording, update only that Appendix A accounting block after approval.

## Shadow artifact review

Before any active update, verify:

1. `Paper_I_s_accounting_shadow_mechanism_formula_20260709.tex` contains the six recommended \(S\) values above.
2. `Paper_I_s_accounting_shadow_runtime_20260709.tex` contains the runtime values only as diagnostics.
3. Shadow JSON/CSV/MD under `output/pdf/paper_i_hh_s_accounting_shadow_20260709/` exist and agree with the user-provided final values.
4. Mechanism rows show component-sum consistency.
5. Comparator rows validate visible Geo/Append \(S\) against prefix-aligned support components.
6. Strong-strong Append remains sourced from the explicit \(k=8\) override.
7. Appendix non-HH rows are inventoried only, not updated.

If the shadow CSV/JSON/MD directory is unavailable, regenerate to a **new** stamp/output directory and do not overwrite the existing 20260709 shadow outputs.

## Approved active-promotion design

Only after explicit approval and gate satisfaction:

- Edit active `MATH/paper_details/Paper_I.tex` directly rather than copying the whole shadow duplicate.
- Replace only the six HH SNAKE \(S\) cells.
- Add a new machine-readable active S-accounting overlay block after the existing HH physical-lane block.
- Do not rewrite the historical `BEGIN_MACHINE_READABLE_HH_PHYSICAL_LANE_DUPLICATE_UPDATE_20260708` block; the new block should explicitly supersede only its HH SNAKE \(S\) cells.
- Create a new, non-overwriting promotion manifest if needed, e.g. under `MATH/paper_facing/paper_I_static_scaffold/`, recording:
  - approval source;
  - gate status;
  - shadow artifact paths;
  - mechanism values;
  - runtime diagnostic values;
  - unchanged rows/cells;
  - non-HH appendix blocker status.

Illustrative active overlay shape:

```json
{
  "schema": "paper_i_hh_s_accounting_active_update_v1",
  "convention": "displayed_prefix_logical_scalar_estimator_query",
  "changed_snake_s_cells": {
    "weak-weak": 5009,
    "intermediate-weak": 4142,
    "strong-weak": 4126,
    "weak-strong": 17305,
    "intermediate-strong": 28617,
    "strong-strong": 5153
  },
  "runtime_s_diagnostic_only": {
    "weak-weak": 4469,
    "intermediate-weak": 3740,
    "strong-weak": 3678,
    "weak-strong": 16627,
    "intermediate-strong": 27387,
    "strong-strong": 4613
  },
  "non_hh_appendix_policy": "unchanged_source_mapping_blocked"
}
```

# File-by-file impact

| File/path | Pre-approval impact | Approved-only impact | Dependency |
|---|---|---|---|
| `MATH/paper_details/Paper_I.tex` | Read/diff only | Replace six HH SNAKE \(S\) cells; optionally update Appendix A if not already final; add active overlay block | User approval + gates |
| `MATH/paper_details/Paper_I_s_accounting_shadow_mechanism_formula_20260709.tex` | Read as recommended candidate | No edits | Existing shadow audit |
| `MATH/paper_details/Paper_I_s_accounting_shadow_runtime_20260709.tex` | Read as diagnostic reference | No edits | Existing shadow audit |
| `pipelines/reporting/build_paper_i_hh_s_accounting_shadow.py` | Reuse only | No code changes | Existing builder |
| `output/pdf/paper_i_hh_s_accounting_shadow_20260709/*` | Review if available | No overwrite | Missing contents are blocker |
| `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_s_accounting_active_update_<stamp>.json` | None | Optional new manifest, no overwrite | User approval |
| `MATH/paper_details/Paper_I.pdf`, root `Paper_I.pdf` | None | Rebuild/copy only after approved TeX edit | Successful TeX build |
| `test/test_snake_table_i_measurement_work.py` | Run tests only | No edits expected | Validation |

# Risks and migration

- **Missing results skill:** active promotion is blocked while `agent_guidance/skills/paper-i-results/SKILL.md` is absent.
- **Shadow directory unavailable:** do not assume contents; regenerate under a new stamp or block promotion.
- **Appendix A ambiguity:** active `Paper_I.tex` may already contain final P0/P1/P2/P3 wording. Treat Appendix A update as idempotent.
- **Old provenance block staleness:** avoid rewriting historical provenance; add a new overlay block that clearly supersedes only \(S\)-accounting cells.
- **Non-HH appendix \(S\) rows:** leave unchanged until source JSON mappings are available.

# Implementation order

1. Confirm no active edit is authorized yet; record current task as review/planning only.
2. Check required gates and record `paper-i-results/SKILL.md` absence as a blocker.
3. Review mechanism shadow duplicate and runtime duplicate.
4. Review or regenerate shadow CSV/JSON/MD under a new stamp if unavailable.
5. Verify mechanism values exactly match: 5009, 4142, 4126, 17305, 28617, 5153.
6. Verify runtime values are recorded only as diagnostics.
7. Diff active `Paper_I.tex` against the mechanism duplicate; classify changes as six table cells, Appendix A convention, and provenance block only.
8. Run relevant accounting tests, especially `test/test_snake_table_i_measurement_work.py`.
9. Prepare an approval packet listing exact cell changes, validation status, blockers, and unchanged appendix non-HH rows.
10. Stop and request explicit user approval before active promotion.
11. After approval and gate satisfaction only: edit active TeX atomically, rebuild PDF, verify visible rows, and write non-overwriting provenance/manifest records.


> 💡 Continue this plan conversation with ask_oracle(chat_id: "s-accounting-plan-D3CEB1", new_chat: false)
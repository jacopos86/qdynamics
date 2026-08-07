---
name: source-locked-sensitivity
description: Guard an actual one-variable sensitivity or fixed-settings replay batch that must hold an existing paper or table source fixed while changing exactly one requested variable. Use for rho, noise, shot, backend, parity, or one-flag sweeps only when a run or report claims source locking; do not use for algorithm implementation, general parameter discussion, exploratory multi-variable studies, unit tests, or mathematical handoffs.
---

# Source-Locked Sensitivity Runs

Use this skill after the relevant paper run skill whenever the task claims to
vary exactly one variable while preserving a current table, figure, or declared
diagnostic source. Examples: trust-region rho sweeps, noise sweeps, shot-cost
sweeps, backend/compile sweeps, one-flag ablations, and "same settings except
X" reruns.

## Core Rule

"Same settings" means the same executable state, not the same headline labels.
The source lock includes:

- source result/manifest/current JSON paths and SHA-256 values;
- command-line arguments or a complete normalized settings manifest;
- code/container route and runner mode;
- random seeds, optimizer settings, resume/current-state inputs, sidecars, and
  source-selected scaffolds;
- cutoff/reference policy, pool contract, route/profile ids, batching, shortlist
  sizes, Phase-I/II/III score modes, pruning, insertion/refit, cost weights, and
  stop rules.

If the requested variable is `x`, every non-`x` executable field must be either
identical to the source or explicitly approved by the user as a second changed
variable.

## Forbidden Shortcuts

- For candidate or paper-facing source-locked sweeps, do not use Optuna,
  oracle-grid, search, warm-start discovery, or trial-selection wrappers at all,
  even with `n_trials=1`. Use the original source command, strict replay path,
  or a dedicated source-lock replay runner. A wrapper may be used only for an
  explicitly user-approved diagnostic run, and the audit must label
  `wrapper_used: true`, `run_class: diagnostic`, and list every wrapper-added or
  wrapper-defaulted field.
- Do not reconstruct settings by mixing source fields with current code
  defaults. Do not use `setdefault`, current-code default merging, or
  missing-field backfilling for source-locked fields. A missing source field
  must remain `unresolved_source_field` in the audit and blocks candidate or
  paper-facing sweeps unless the user explicitly approves a diagnostic run with
  that field listed as an additional changed variable.
- Do not silently introduce newly added defaults such as `phase1_score_mode`,
  new shortlist caps, new batching toggles, new cost denominators, or changed
  stop policies when the source predates those fields.
- Do not accept a generated batch because its file name says `fixed_settings`.
  Trust only the machine-readable equality audit and the anchor result.
- Do not report or plot a sweep as sensitivity evidence when the source-value
  anchor fails. Label it diagnostic-invalid until repaired.

## Required Workflow

1. Resolve the visible or user-declared source.
   Use the paper run skill's resolver/source-map path when available. Record the
   table/figure label, method, regime/case, visible value, source path, source
   hash, source command/manifest path, and original value of the variable to be
   swept.

2. Materialize a complete baseline.
   Prefer the original source command or strict replay manifest. If only a
   result JSON exists, build a normalized complete settings snapshot and mark any
   missing executable field as unresolved. Unresolved fields block paper-facing
   or candidate sweeps unless the user explicitly accepts a diagnostic run.

3. Generate the source-value anchor first.
   Before submitting the full grid, run exactly one row with the swept variable
   set to the source value, for example `rho=0.25` when the current source uses
   `phase2_rho=0.25`.

4. Compare the anchor to the source.
   The anchor must pass all applicable checks:
   - normalized non-swept settings diff is empty;
   - runner mode and route/profile ids match;
   - source SHA and source/current/resume inputs match;
   - selected operator sequence or scaffold hash matches when the run is meant
     to replay an adaptive trajectory;
   - displayed/table metric matches within the table's provenance tolerance;
   - stopping condition or selected prefix semantics match the source contract.

5. Fan out only after the anchor passes.
   Before fan-out, materialize every planned row and compute a normalized
   settings diff against the source. Every row must have
   `changed_fields_vs_source == [<swept variable>]` except for explicitly
   approved diagnostic second variables. If any non-anchor row changes defaults,
   runner mode, shortlist behavior, score mode, batching, pruning, optimizer
   settings, seeds, or source inputs, do not submit the grid. Submit or run the
   remaining grid values only after the anchor audit says
   `anchor_reproduces_source: true` and every planned-row diff passes. If the
   anchor fails, stop and repair the generator or runner before producing more
   data.

6. Attach the audit to every report.
   Every PDF, table, or chat summary for the sweep must state the anchor status
   and link the audit JSON. If the anchor is missing or failed, the report must
   say the sweep did not isolate the requested variable.

## Audit JSON Minimum

Emit a machine-readable audit with at least:

```json
{
  "schema": "source_locked_sensitivity_audit_v1",
  "source": {
    "table_label": "tab:...",
    "method": "SNAKE",
    "regime_or_case": "...",
    "source_json": "...",
    "source_sha256": "...",
    "source_command_or_manifest": "...",
    "source_command_or_manifest_sha256": "...",
    "runner_mode": "direct_replay",
    "route_or_profile_id": "route_a/paper_i_production_v1",
    "settings_hash": "...",
    "source_variable_value": 0.25
  },
  "sweep": {
    "run_class": "candidate",
    "variable": "phase2_rho",
    "grid": [0.05, 0.1, 0.25, 0.5, 1.0],
    "runner_mode": "direct_replay",
    "wrapper_used": false,
    "wrapper_kind": null,
    "baseline_materialization_status": "complete",
    "unresolved_source_fields": [],
    "fields_added_by_current_defaults": [],
    "settings_changed": ["phase2_rho"]
  },
  "planned_rows": [
    {
      "value": 0.05,
      "settings_hash": "...",
      "changed_fields_vs_source": ["phase2_rho"],
      "non_swept_settings_diff": []
    }
  ],
  "anchor": {
    "value": 0.25,
    "anchor_result_json": "...",
    "anchor_reproduces_source": true,
    "metric_abs_diff": 0.0,
    "operator_sequence_match": true,
    "non_swept_settings_diff": []
  },
  "status": "pass"
}
```

If `non_swept_settings_diff` is nonempty, `status` must be `blocked` or
`diagnostic_invalid` unless the user explicitly approves those differences as
additional variables.

Missing `source_command_or_manifest`, unresolved executable fields, wrapper use
without explicit diagnostic approval, or any default-added non-swept field makes
the audit `blocked`.

## Rho-Sweep Specific Guardrails

For Paper-I SNAKE trust-region sensitivity:

- the intended swept field is the global trust-region scalar currently surfaced
  as `phase2_rho` or its renamed alias;
- do not change `phase1_score_mode` while claiming a rho-only sweep;
- do not change `phase2_shortlist_size`, `phase2_min_count`,
  `phase2_max_count`, batching, beam settings, optimizer constants, pruning, or
  stop policy;
- if the current code default differs from the source artifact because a field
  was added later, explicitly set the source-compatible value or use the source
  code/container. The anchor must prove the choice.

## Failure Response

When an anchor fails, say so plainly:

```text
The sweep did not isolate <variable>. The source-value anchor changed
<field list> and did not reproduce <source metric/operator hash>. I am stopping
the sweep/report until the generator is repaired or you approve a diagnostic
run with those additional variables.
```

Do not soften this into "path dependence" or "stochastic drift" unless the
non-swept settings audit is clean and the code path is actually stochastic.

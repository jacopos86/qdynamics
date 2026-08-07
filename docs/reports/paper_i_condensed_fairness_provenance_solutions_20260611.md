# Paper I Condensed Fairness/Provenance: Current Solutions

Generated: 2026-06-11  
Baseline audit: `docs/reports/paper_i_condensed_fairness_provenance_audit_20260611.md`  
Baseline audit JSON: `output/pdf/paper_i_condensed_fairness_provenance_audit_20260611.json`  
Scope: review/copy-paste solution file only. No manuscript edits, support-file edits, table edits, CHTC fetches, or runs were performed by this file.

## Current Status

The June 11 audit supersedes the June 10 audit for the active condensed TeX/PDF. The active document changed after the June 10 audit, and the refreshed audit now reports:

- `source_hash`: `match=296`, `mismatch=16`, `not_checked=51`, `directory_not_checked=9`, `external_not_checked=1`
- `metric_policy`: `ok=3`, `policy_divergence=1`
- `compiled_cost`: `ok=69`, `qualified=2`, `blocked=1`
- `work_proxy`: `qualified=54`, `blocked=2`
- `fairness`: `ok=4`, `qualified=2`, `blocked=1`

The straightforward first fix is the support/provenance hash chain in Patch A. It does not require scientific reruns.

## Patch A — no-run support/provenance hash repair

Use this only after approving non-manuscript support-file edits. It updates stale hashes inside `hh_tableiii_convergence_sources.json`, then updates `paper_i_snake_fairness_status_20260608.json` to the resulting source-map hash.

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib

root = Path.cwd()
source_map = root / "MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json"
fairness = root / "MATH/paper_facing/paper_I_static_scaffold/paper_i_snake_fairness_status_20260608.json"

source_map_replacements = [
    (
        "b049e7786ccf9e083ed9e81988bf599b03cd878fe469efe3c7124b770bdeddca",
        "d2811248e45961582be7e190b0cbc7fbc7e04e8d2f9a81fabb5d6b61d0cb756c",
        "strong_strong Append-ADAPT previous_source_sha256",
    ),
    (
        "ef93c95f18ca79213160fe314f87fc4466cf85ba49a5d56511ad718de8424457",
        "cbd89454ee2ce787c98ab8f810e70d759acd84b86eee17805475c2a2a41b036f",
        "strong_strong TETRIS-ADAPT previous_source_sha256",
    ),
    (
        "f6afaf3ef2bfe87a2bc6d7f8227153bfce61f15c0d93c0b6cc535c36a055b3e0",
        "05405d90d156ad73704c36b3e15beaf8a6e66f4b02989c494bdd92f515b0b745",
        "strong_weak Append-ADAPT previous_source_sha256",
    ),
    (
        "3e91ac6a619f0b7f86c42e2cf5fabece464f7b63e526f3c38896fb3ef2a1d5b5",
        "f025be3960f7c29ee19b3a6350431ec13be18b2781c8d39667bf85b2ed570112",
        "strong_weak SNAKE promotion_sha256",
    ),
    (
        "5597783f704935dd9b7b5b369efee1d17d91074cb49dd0cc233c8235a023b652",
        "1a7bb4c84e3f76c56aa210b71b79ddda2f6e8cd3e9cafdcf2dbcc2d29eb1faae",
        "strong_weak TETRIS-ADAPT previous_source_sha256",
    ),
    (
        "9396d770b5c132c691fb8f040e21570c0e07cefc823c8c30e087073343e7fbc2",
        "6d3e6583556b255a2fcf27975557672cb44e5ffb4904953cd11119a5484934d8",
        "weak_strong Append-ADAPT previous_source_sha256",
    ),
    (
        "6045d1314d50f2dd8c59e153414270c0a21a43a41515c1f1bff03d3cb83f0d0d",
        "774ef9ed116486d7acbac1d62b686af5fc89fe59e5b12566166b92fa07847345",
        "weak_strong TETRIS-ADAPT previous_source_sha256",
    ),
]

expected_fixed_source_map_hash = "b5e2110d306ccd015d8c41f18f82533dfae839815659d6a41a4dd23ff6f71938"
expected_fixed_fairness_hash = "c25b11cf0c3ee2cb6861c27bb7da953b13c12f12b0967c439a2d16a94eff52d0"

text = source_map.read_text()
for old, new, label in source_map_replacements:
    if old not in text:
        raise SystemExit(f"old hash not found for {label}; source map may already be edited")
    text = text.replace(old, new, 1)

fixed_hash = hashlib.sha256(text.encode()).hexdigest()
if fixed_hash != expected_fixed_source_map_hash:
    raise SystemExit(f"unexpected fixed source-map hash: {fixed_hash}")
source_map.write_text(text)

fairness_text = fairness.read_text()
old_fairness_source_map_hash = "72133a06e9ce996dd8efb04f9927fa04be57bac62ebf7cd32ed6a15a506599f8"
if old_fairness_source_map_hash not in fairness_text:
    raise SystemExit("old fairness source_map_sha256 not found; fairness status may already be edited")
fairness_text = fairness_text.replace(old_fairness_source_map_hash, expected_fixed_source_map_hash, 1)
fairness.write_text(fairness_text)

fairness_hash = hashlib.sha256(fairness.read_bytes()).hexdigest()
if fairness_hash != expected_fixed_fairness_hash:
    raise SystemExit(f"unexpected fixed fairness-status hash: {fairness_hash}")

print("source_map_sha256", fixed_hash)
print("fairness_status_sha256", fairness_hash)
PY
```

Expected result:

- `hh_tableiii_convergence_sources.json` SHA256 becomes `b5e2110d306ccd015d8c41f18f82533dfae839815659d6a41a4dd23ff6f71938`.
- `paper_i_snake_fairness_status_20260608.json` SHA256 becomes `c25b11cf0c3ee2cb6861c27bb7da953b13c12f12b0967c439a2d16a94eff52d0`.

Then rerun the audit locally with new output paths:

```bash
python3 pipelines/reporting/audit_paper_i_condensed_fairness_provenance.py \
  --repo-root /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3 \
  --condensed-tex MATH/paper_details/static_adapt_paper_I_condensed.tex \
  --condensed-pdf MATH/paper_details/static_adapt_paper_I_condensed.pdf \
  --non-condensed-tex MATH/paper_details/static_adapt_paper_I.tex \
  --output-json output/pdf/paper_i_condensed_fairness_provenance_audit_20260611_after_patchA.json \
  --output-md docs/reports/paper_i_condensed_fairness_provenance_audit_20260611_after_patchA.md \
  --output-csv-dir output/pdf/paper_i_condensed_fairness_provenance_audit_20260611_after_patchA
```

## Patch B — deferred active-manuscript hash refresh

Do not apply while manuscript edits are out of scope. If Patch A is applied and manuscript machine-readable comments are later allowed, update active comments in:

- `MATH/paper_details/static_adapt_paper_I.tex`
- `MATH/paper_details/static_adapt_paper_I_condensed.tex`

Replace active `source_map_sha256` occurrences of:

```text
62c617943ad10fbffe6ced1394b29922909331d0a6a269204569b56fc76ffe92
```

with:

```text
b5e2110d306ccd015d8c41f18f82533dfae839815659d6a41a4dd23ff6f71938
```

Also refresh the active `TABLE_I_II_ITERATION_PLOTS_20260610` machine-readable comment hash fields:

| Source key | Old hash | Current hash |
|---|---|---|
| `source` / `paper_i_tables_i_ii_repeat_enabled_iteration_plots_20260610.provenance.json` | `d45d961b8a4a045107a53df5a1155135a990100ab487b89da1a4b5be6ea62317` | `025465d5d2e5aeb12cb7103b89dfb2a8abe315034580a60b9a7fdfb50b96d440` |
| `table_source` / `paper_i_tables_i_ii_repeat_enabled_comparator_promotion_20260610.json` | `5ec244aae5205e7674663c02a509d9784e6ce582cc558f486db19f323f123285` | `3a54035c0e54bbe9c152ca0abe58cfd16333bee33e46eb6e4c089051e301afa0` |
| `main_body_strong_hubbard_plot` | `3c4d715422cf2e3221cf61a7fdd06ff5a0e55602f7984f967bc0fadb43961751` | `0ea5684b71309bd80c5a3f31575a4ac960bc07e7826a84f01fa1c306d4d33805` |
| `main_body_strong_hubbard_plot_pdf` | `8c22fbf9842a467a4b16b951b33ac4b52ff745abbfd698179339d3b95c27856e` | `0554fe61be0bfdee7a2db773db24ed78849cfc4418f4f84d9c40212bda05505a` |
| `appendix_hubbard_weak_plot` | `af8eda4e84758a9aba1ae9f0851c1af094b1d5a01e50d8d3973f65a0d4571845` | `aa31b6b2f1b8236339bae4cea726dfca401ecaac475fcaa470845bbcb4a91b52` |
| `appendix_spin_boson_plot` | `ef8f8c15ac9900155185e6c78798ea668dae3456f193906616249b2fa98ab441` | `841a9fe2c0068a93218a3ee8ac1f8a2c3cf21b5c59e7dbac5d61fff5165716f2` |

These are provenance-comment refreshes only; they do not change scientific values.

## Patch C — HH Table III weak-strong SNAKE cost decision

Current conservative visible row remains the cleanest evidence statement:

```tex
SNAKE & 42 & 1.92e-2 & -- & -- & -- \\
```

Optional retained-sidecar row, only if accepted as qualified cost evidence rather than strict replayable plateau-prefix evidence:

```tex
SNAKE & 42 & 1.92e-2 & 1592 & 1282 & 7194 \\
```

Evidence for the optional row:

- Sidecar: `output/pdf/paper_i_table_iii_snake_weak_strong_costpreserve_trial0004_qiskit_cost_20260531.json`
- SHA256: `b63650c97fda3eef11aa1778cddda689f39fb4b7e67b0ca59e342654a2d36170`
- `compiled_resource_qiskit_validated: true`
- `compiled_resource_source_kind: snake_qiskit_compiled_terminal_best_trial_ansatz_circuit`
- `history_len: 18`, `terminal_depth: 18`

Run/recovery boundary: the replayable continuation files are still missing locally:

```text
raw_outputs/routeA_paper_i_hh_snake_structural_continue_20260531_v1/weak_strong/json/current.json
raw_outputs/routeA_paper_i_hh_snake_structural_continue_20260531_v1/weak_strong/json/result.json
```

Next no-run step is CHTC artifact recovery. If recovery fails, a clean fix requires replay/rerun.

## Patch D — HH Table III strong-strong SNAKE cost decision

Current qualified visible row:

```tex
SNAKE & 13 & 7.95e-3 & 976 & 938 & 5559 \\
```

Run-emitted terminal-sidecar alternative, only if verified to be the same displayed ansatz:

```tex
SNAKE & 13 & 7.95e-3 & 956 & 918 & 5350 \\
```

Evidence for the terminal-sidecar alternative:

- Sidecar: `output/pdf/paper_i_table_iii_snake_strong_strong_costpreserve_trial0001_qiskit_cost_20260531.json`
- SHA256: `6e72c296cbb6d667459645f89704c6e3484d5e9030a0f48ed32c4f9c999b2a56`
- `compiled_resource_qiskit_validated: true`
- `compiled_resource_source_kind: snake_qiskit_compiled_terminal_best_trial_ansatz_circuit`
- `history_len: 12`, `terminal_depth: 12`

Prior audit rule: `MATH/paper_facing/paper_I_static_scaffold/paper_i_table_consistency_audit_20260604.json:176-192` says to use the run-emitted sidecar only if the plateau prefix is the same displayed ansatz; otherwise keep the qualified continuation provenance explicit.

Run/recovery boundary: the replayable continuation files are still missing locally:

```text
raw_outputs/routeA_paper_i_hh_snake_structural_continue_20260531_v1/strong_strong/json/current.json
raw_outputs/routeA_paper_i_hh_snake_structural_continue_20260531_v1/strong_strong/json/result.json
```

Next no-run step is CHTC artifact recovery. If recovery fails, removing the qualification requires replay/rerun.

## Patch E — `S_alg` policy and known component-backed values

Definition:

```text
S_alg = N_H_outer_eval + N_grad_probe + N_metric_probe + N_H_refit_eval
```

Known component-backed values from `MATH/paper_facing/paper_I_static_scaffold/table_i_snake_support_20260514_salg_mixed_recovered.json`:

| Surface | Status | `S_alg` | Components |
|---|---:|---:|---|
| `bosonic_snake_current_table_support` | `ok` | `228297.66666666666` | `N_H_outer_eval=198949.33333333334`, `N_grad_probe=546.6666666666666`, `N_metric_probe=265.5`, `N_H_refit_eval=28536.166666666668` |
| `fermion_boson_snake_current_table_support.aggregate` | `ok` | `34235.5` | `N_H_outer_eval=11391.5`, `N_grad_probe=31.5`, `N_metric_probe=39.5`, `N_H_refit_eval=22773.0` |
| `all_averaged_snake_current_table_support` | blocked | `null` | `S_alg_status=legacy_proxy_not_event_ledger`; missing all four components |
| `fermionic_snake_current_table_support` | blocked | `null` | `S_alg_status=legacy_proxy_not_event_ledger`; missing all four components |

Copy/paste-safe rule: do not relabel controller-shot proxies, legacy `S_norm`, or `--` cells as `S_alg`. A row can receive `S_alg` only when the source exposes finite nonnegative values for all four components under `algorithmic_measurement_work_v1`.

HH Table III SNAKE `weak_strong` and `strong_strong` still have missing work-proxy cells in the June 11 audit. If no event ledger exists in recovered continuation JSON/logs, the `S_alg` fix requires instrumented replay/rerun.

## Patch F — HH appendix metric-policy wording

Status: still a manuscript wording issue only. No rerun required, but not applied while manuscript edits are out of scope.

Problem: `tab:fixed_accuracy_hh_cartesian` still reads as same-cutoff wording, while the Paper-I results contract expects raw external-reference error for the fixed-prefix appendix audit.

Minimal future wording target if manuscript edits are approved:

```tex
The error column reports the raw external-reference error for the displayed working cutoff, using the higher ED reference cutoff recorded in the source contract; the Holstein-weak rows use $n_{\rm ph}^{\rm work}=2$ with ED5 diagnostics, and the Holstein-strong rows use $n_{\rm ph}^{\rm work}=4$ with ED7 diagnostics.
```

## Recommended Next Step

1. Apply Patch A only if non-manuscript support-file edits are approved.
2. Rerun the audit with the `after_patchA` output paths above.
3. If source-hash mismatches then remain only in active manuscript comments, decide whether to apply deferred Patch B.
4. For HH weak-strong/strong-strong SNAKE, try CHTC artifact recovery before any replay/rerun.
5. Do not change `S_alg`/`S` values without component-backed event ledgers.

# GitNexus Engineering Plan

> Task: Persist exact logical estimator-work telemetry for Paper II generalized exchange and AVQDS, and consume it in the comparison report without changing scientific decisions.
> Evidence verified at commit 74e820a3bdb8199aedddfe2bc7c67f89f0d78410; GitNexus index refreshed this session (`--index-only --pdg`) and current at that commit.
> Evidence provenance schema 2; global dirty digest fa2de2a5e2e27d667b32a6bfa5d534538dc869882d946db00075d4ddb2838b95; cited-path manifest 15 sorted entries; exact generated plan path excluded.

## 1. Objective

Add additive, exact telemetry for all candidate-acquisition and finalist-certification work performed by the two Paper II structural policies. New run artifacts must contain enough per-round information for `complete_estimator_work` to report exact logical estimator proxies rather than lower bounds. Preserve all physics, candidate populations, rankings, gates, integration, route identifiers, and existing serialized fields. Preserve lower-bound inference only as a compatibility path for legacy artifacts.

## 2. Current Behaviour

- [verified] `select_avqds_appends` performs a complete geometry evaluation for every materialized end-append candidate and retains only the aggregate successful count `candidates_scored`; greedy-round support dimensions and failure counts are discarded (`pipelines/time_dynamics/ap_mclachlan/avqds.py:83-207`).
- [verified] `build_selector_inputs` builds one positioned-tangent cache per generalized-exchange selector pass; its exact column width is available from `StructuralInsertionCache.column_count`, but the width and corresponding logical-entry count are not serialized (`pipelines/time_dynamics/ap_mclachlan/exchange_integration.py:1036-1271`; `pipelines/time_dynamics/ap_mclachlan/structural_cache.py:140-320`).
- [verified] `certify_finalist` may run a local ray refit, then exactly one full patched geometry and solve; refit callback evaluations are not counted and a combined geometry/solve exception makes legacy post-hoc inference ambiguous (`pipelines/time_dynamics/ap_mclachlan/exchange_certification.py:132-351`).
- [verified] generalized exchange may invoke the selector repeatedly at one checkpoint. Each call overwrites `decision`; only the last payload and aggregate accepted-kind labels survive (`pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py:1097-1173`).
- [verified] the internal deletion-to-insert fallback combines attempts and structural counts but loses the first pass's cache inputs (`pipelines/time_dynamics/ap_mclachlan/exchange_integration.py:1519-1537`).
- [verified] `complete_estimator_work` reconstructs work from the final payload. It marks AVQDS candidate work and generalized-exchange certification work as lower bounds when per-round/refit facts are missing (`pipelines/time_dynamics/diagnostics/paper_ii_four_route_comparison.py:596-746`).

## 3. Relevant Architecture

- [verified] `run_append_mclachlan_trajectory` owns checkpoint repetition and persists one `PatchDecision` per point. The point and trajectory serializers already carry decision metadata verbatim, so no runner schema rewrite is required (`adaptive_trajectory.py:635-671,792-838,906-1295`).
- [verified] `select_generalized_exchange_patch` is the route adapter: it builds inputs, owns internal fallback, closes compiled-cost resources, and creates the JSON-ready payload. This is the correct place to aggregate exact per-call work (`exchange_integration.py:1274-1693`).
- [verified] `select_exchange_patch` owns finalist attempt ordering and sees every `CertificationResult` before the public payload truncates the human-readable `attempts` list to 16 entries (`exchange_selector.py:191-468`; `exchange_integration.py:1576-1588`).
- [verified] the statevector geometry proxy convention is `g(p)=p(p+1)/2+p+2`; physical shots and commuting groups do not exist in current artifacts and must remain unavailable rather than reported as zero (`paper_ii_four_route_comparison.py:444-476,596-746`).
- [inferred] A compact nested receipt is preferable to serializing complete selector payloads for every repeated call: it preserves exact accounting without multiplying large ranking/cost receipts.

## 4. GitNexus Findings

- [graph] `mcp__gitnexus__impact(repo=<worktree>, target=run_append_mclachlan_trajectory, upstream, includeTests=true, summaryOnly=true)` reports CRITICAL risk, 3 direct dependents, 46 total affected symbols, and two Paper III execution flows. Quote: `risk: CRITICAL; direct: 3; processes_affected: 2`.
- [graph] The same impact query for `select_generalized_exchange_patch` reports HIGH risk, 5 direct dependents, 42 total affected symbols, and the same two Paper III flows. Quote: `risk: HIGH; direct: 5`.
- [graph] The same impact query for `select_avqds_appends` reports HIGH risk with 4 direct dependents and exact epistemic status. Quote: `impactedCount: 8; direct: 4`.
- [graph] The same impact query for `build_local_ray_refit` reports HIGH risk with 4 direct dependents; `CertificationResult` reports MEDIUM risk with 6 direct dependents. Defaults and the two-value callable contract therefore must be preserved.
- [graph] The same impact query for `complete_estimator_work` reports LOW risk with 3 direct dependents and 5 total affected symbols.
- [verified] Text search confirms the high-risk callers are the shared trajectory/runners and tests; changes must be additive metadata only and receive Paper III smoke coverage.

## 5. Statement-Level PDG Findings

- [graph] `mcp__gitnexus__pdg_query(mode=controls,target=select_avqds_appends)` shows geometry evaluation is controlled by the atom loop and the materialization exception branch. The count must increment immediately before evaluator invocation, while success increments only after return; materialization failures must not be counted as evaluator attempts.
- [graph] `mcp__gitnexus__pdg_query(mode=flows,target=select_avqds_appends,variable=scored)` shows the only persisted flow is the aggregate increment at the completed evaluation and the final `AVQDSDecision` construction. Per-round state must be accumulated alongside, not derived afterward.
- [graph] `mcp__gitnexus__pdg_query(mode=flows,target=run_append_mclachlan_trajectory,variable=decision)` shows each selector result flows into the loop checks, the final metadata replacement, and the single persisted point. A separate round ledger must be accumulated before `decision` is overwritten.
- [graph] `mcp__gitnexus__pdg_query(mode=flows,target=select_generalized_exchange_patch,variable=selection)` shows the internal fallback replaces `selection` and `inputs`, after which only combined attempts/counts flow to the payload. Cache-pass telemetry must be appended inside `select_once` before that replacement.
- [verified] The refit objective performs one state-and-all-tangents preparation per invocation; the exact callback count includes the explicit initial objective, optimizer callbacks, and explicit final objective (`exchange_certification.py:168-204`).
- [verified] The current L2 loop uses `range(max_rounds + 1)` while its existing cap test reads a nonexistent metadata key. This is a pre-existing behavioral defect. Telemetry must faithfully distinguish selector-call count and accepted-commit count; changing the cap is out of scope.

## 6. Proposed Changes

1. `pipelines/time_dynamics/ap_mclachlan/avqds.py`
   - Add an additive per-greedy-round record to `AVQDSDecision` and its JSON serialization.
   - In `select_avqds_appends`, record base/candidate support dimension, pool size, evaluator attempts/completions, finite results, chosen atom, L2 before/after, whether the choice committed, failure counts, and exact logical geometry entries.
   - Preserve `candidates_scored` as the successful-completed-geometry compatibility alias.

2. `pipelines/time_dynamics/ap_mclachlan/exchange_certification.py`
   - Add a defaulted nested estimator-work receipt to `CertificationResult` without changing existing constructor requirements or the refit callable's two-value return API.
   - Count production refit objective/tangent passes inside the actual callback. Snapshot the cumulative counter per certification call.
   - Separate full-geometry invocation/completion telemetry from the subsequent solve while preserving legacy failure reason prefixes.
   - Record patched support dimension, exact `g(p)` entries for completed full geometry, exact refit scalar-overlap entries, classical ray/velocity check counts, and a completeness flag. An uninstrumented injected refit or interrupted evaluator marks completeness false rather than fabricating work.

3. `pipelines/time_dynamics/ap_mclachlan/exchange_selector.py`
   - Add a defaulted tuple of compact certification-work receipts to `ExchangeSelection`.
   - Append the receipt for every actual call to `certify_finalist`, including rejected finalists; preserve it through all selector return paths.
   - Keep existing `AttemptRecord`, ranking, retry, and public attempt behavior unchanged.

4. `pipelines/time_dynamics/ap_mclachlan/exchange_integration.py`
   - Record every `select_once` cache pass before internal fallback can overwrite `inputs`: pre-support, positioned columns, and exact shared-cache entries `pM+M(M+3)/2`.
   - Combine certification receipts across fallback selections.
   - Emit a compact `estimator_work` receipt with exact component totals, full reason-count summary derived from untruncated attempts, attempt-list truncation disclosure, and completeness. Retain the existing last-16 `attempts` field for compatibility.

5. `pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py`
   - Accumulate one compact receipt per generalized-exchange selector call before `decision` is overwritten.
   - Persist ordered call index, pre/post support, accepted flag/kind/reason, and the selector's estimator-work receipt on the final decision metadata.
   - Persist both actual selector-call count and actual accepted-selector count. Preserve all existing L2 loop behavior and legacy metadata.

6. `pipelines/time_dynamics/diagnostics/paper_ii_four_route_comparison.py`
   - Prefer the exact new ledgers for AVQDS and generalized exchange.
   - Sum shared acquisition, AVQDS candidate full geometries, finalist geometry, and refit work as distinct components. Report exact only when every emitted receipt is complete.
   - Keep the current inference path and lower-bound statuses unchanged for legacy artifacts. Continue reporting physical shots as unavailable for statevector runs.

## 7. Implementation Sequence

1. Add AVQDS per-round records and unit tests; verify legacy fields and decisions are unchanged.
2. Add certification/refit receipts and tests for no refit, on-ray single callback, multi-callback refit, solve failure after completed geometry, evaluator failure, and uninstrumented injected refit.
3. Plumb compact certification receipts through `ExchangeSelection`; test accepted and rejected finalist paths.
4. Aggregate generalized-exchange cache/fallback/certification work in the adapter; test exact formula, fallback preservation, full reason counts, and serial/parallel equality.
5. Preserve every outer selector call in trajectory metadata; test a forced multi-round checkpoint and distinguish call count from commit count without fixing the separate off-by-one defect.
6. Update comparison aggregation and fixtures for exact new receipts plus legacy lower-bound fallback.
7. Run focused suites, route-parity suites, Paper III smoke tests, and the repository time-dynamics validation appropriate to the changed shared trajectory seam. Stage only explicit listed paths and run GitNexus detect-changes before each commit.

## 8. Test Strategy

- `test/test_ap_mclachlan_avqds.py`: initial below-threshold yields an empty ledger; one round records p→p+1 and exact attempted/completed counts; multiple rounds preserve dimensions and totals; materialization versus evaluator failures are distinguished; JSON total equals `candidates_scored`.
- `test/test_ap_mclachlan_exchange_certification.py`: exact full-geometry and refit counts across accepted, rejected, and failure paths; unchanged certification decisions and thresholds.
- `test/test_ap_mclachlan_exchange_selector.py`: every certifier call contributes one compact receipt; pre-certification rejections contribute none; receipt order matches certification attempt order.
- `test/test_ap_mclachlan_exchange_integration.py`: exact cache formula from the real column count; both internal fallback passes retained; aggregate reason counts use all attempts; compiled serial/parallel receipts match.
- `test/test_ap_mclachlan_route_parity.py`: scientific decisions, energies, support counts, and serial/parallel route parity remain bitwise identical while telemetry is present.
- `test/test_paper_ii_four_route_comparison.py`: exact AVQDS and generalized-exchange ledgers produce exact totals; incomplete receipts do not; legacy fixtures preserve lower-bound results.
- Verification commands:
  - `.venv/bin/python -m pytest -q test/test_ap_mclachlan_avqds.py test/test_ap_mclachlan_exchange_certification.py test/test_ap_mclachlan_exchange_selector.py`
  - `.venv/bin/python -m pytest -q test/test_ap_mclachlan_exchange_integration.py test/test_ap_mclachlan_route_parity.py test/test_paper_ii_four_route_comparison.py`
  - `.venv/bin/python -m pytest -q test/test_paper_iii_promoted_ap_demo.py test/test_paper_iii_regime_handoff.py` when present; otherwise run the located Paper III route smoke tests selected by `rg`.

## 9. Risk and Impact Analysis

- [graph] The shared trajectory seam is CRITICAL and reaches Paper III. Risk is controlled by additive metadata, defaulted dataclass fields, unchanged callable signatures, and explicit route/decision parity tests.
- [graph] Generalized-exchange selection and AVQDS selection are HIGH-risk. No ranking inputs, score calculations, iteration order, acceptance predicates, or budgets may change.
- [verified] JSON payloads will grow. Compact receipts avoid duplicating structural/ranking/compile payloads; the existing truncated human attempt list remains intact.
- [verified] New metadata may change whole-payload hashes if any consumer hashes serialized trajectory metadata. Scientific parity tests must compare decisions/energies/support directly; any receipt digest change must be identified as metadata-only.
- [inferred] Failed geometry evaluation can perform partial internal work before raising. Such a receipt must be marked incomplete; exact journal totals require successful/instrumented runs, not optimistic reconstruction.
- [verified] Current runs are statevector. The exact logical proxy must never be relabeled as physical shots or circuit-group counts.
- [verified] Unrelated untracked manuscript-asset files already exist in the worktree and must not be edited or staged.

## 10. Files Expected to Change

| File | Symbols | Reason |
| ---- | ------- | ------ |
| `pipelines/time_dynamics/ap_mclachlan/avqds.py` | `AVQDSDecision`, `select_avqds_appends` | Exact AVQDS greedy-round geometry ledger |
| `pipelines/time_dynamics/ap_mclachlan/exchange_certification.py` | `CertificationResult`, `build_local_ray_refit`, `certify_finalist` | Exact refit/finalist work receipts |
| `pipelines/time_dynamics/ap_mclachlan/exchange_selector.py` | `ExchangeSelection`, `select_exchange_patch` | Preserve every certification receipt |
| `pipelines/time_dynamics/ap_mclachlan/exchange_integration.py` | `select_generalized_exchange_patch` | Aggregate cache/fallback/certification work |
| `pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py` | `run_append_mclachlan_trajectory` | Preserve every selector-call receipt |
| `pipelines/time_dynamics/diagnostics/paper_ii_four_route_comparison.py` | `complete_estimator_work` | Exact-first, legacy-fallback report aggregation |
| `test/test_ap_mclachlan_avqds.py` | AVQDS tests | Round/failure serialization coverage |
| `test/test_ap_mclachlan_exchange_certification.py` | certification/refit tests | Work-counter coverage |
| `test/test_ap_mclachlan_exchange_selector.py` | selector tests | Receipt propagation coverage |
| `test/test_ap_mclachlan_exchange_integration.py` | integration tests | Cache/fallback aggregation coverage |
| `test/test_ap_mclachlan_route_parity.py` | route parity tests | No scientific drift and multi-round persistence |
| `test/test_paper_ii_four_route_comparison.py` | comparison tests | Exact and legacy paths |

## 11. Reusable Implementation Context

```yaml
implementation_context:
  task_summary: "Persist exact logical estimator-work telemetry for Paper II generalized exchange and AVQDS, then consume it exact-first in the comparison report."
  acceptance_criteria:
    - "New AVQDS artifacts contain every evaluated greedy round with exact support dimensions and evaluator attempt/completion counts."
    - "New generalized-exchange artifacts contain every internal cache pass, every certification/refit receipt, and every outer selector call."
    - "The comparison reports exact logical totals for complete new receipts and preserves lower-bound fallback for legacy artifacts."
    - "Physics, rankings, decisions, energies, support evolution, and parallel/serial parity are unchanged."
    - "Physical shots remain unavailable for statevector runs."
  evidence_provenance:
    schema_version: 2
    head_commit: "74e820a3bdb8199aedddfe2bc7c67f89f0d78410"
    generated_plan_path: "docs/plans/2026-09-03-gitnexus-plan-exact-estimator-telemetry.md"
    global_dirty_digest:
      algorithm: "sha256"
      canonicalization: "gitnexus-evidence-provenance-v2 NUL-framed UTF-8 records"
      value: "fa2de2a5e2e27d667b32a6bfa5d534538dc869882d946db00075d4ddb2838b95"
    cited_path_manifest:
      - {path: "AGENTS.md", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:e51841b577068c43932e7d4c40d814bc3ab84665f8314e06004edc2921993af5", index_digest: "sha256:e51841b577068c43932e7d4c40d814bc3ab84665f8314e06004edc2921993af5", worktree_digest: "sha256:e51841b577068c43932e7d4c40d814bc3ab84665f8314e06004edc2921993af5", untracked_digest: absent}
      - {path: "agent_guidance/shared/repository-work.md", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:cb496e9979a940698c10df55bb5950b7be87a8baab3bccfc9d6ec66aaa3e9746", index_digest: "sha256:cb496e9979a940698c10df55bb5950b7be87a8baab3bccfc9d6ec66aaa3e9746", worktree_digest: "sha256:cb496e9979a940698c10df55bb5950b7be87a8baab3bccfc9d6ec66aaa3e9746", untracked_digest: absent}
      - {path: "agent_guidance/shared/scientific-invariants.md", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:bf03c2ce0c72fe603f1244bccd5af75bdad988ca057929969facd95f50ec0a3e", index_digest: "sha256:bf03c2ce0c72fe603f1244bccd5af75bdad988ca057929969facd95f50ec0a3e", worktree_digest: "sha256:bf03c2ce0c72fe603f1244bccd5af75bdad988ca057929969facd95f50ec0a3e", untracked_digest: absent}
      - {path: "agent_guidance/time-dynamics/AGENTS.md", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:7701ce08dedb229b9fff74a07011af9401d5311341c984f190050a1a7e1064d7", index_digest: "sha256:7701ce08dedb229b9fff74a07011af9401d5311341c984f190050a1a7e1064d7", worktree_digest: "sha256:7701ce08dedb229b9fff74a07011af9401d5311341c984f190050a1a7e1064d7", untracked_digest: absent}
      - {path: "pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:c52c0a0d1b6b5ea49692fa3912bc43a095daf86c2d68fa8fe472f17a8760dc5e", index_digest: "sha256:c52c0a0d1b6b5ea49692fa3912bc43a095daf86c2d68fa8fe472f17a8760dc5e", worktree_digest: "sha256:c52c0a0d1b6b5ea49692fa3912bc43a095daf86c2d68fa8fe472f17a8760dc5e", untracked_digest: absent}
      - {path: "pipelines/time_dynamics/ap_mclachlan/avqds.py", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:1ca57d52be6d2e36e11f86a629b736330eab5f0d26b8e214f0a90b6f7b295fa1", index_digest: "sha256:1ca57d52be6d2e36e11f86a629b736330eab5f0d26b8e214f0a90b6f7b295fa1", worktree_digest: "sha256:1ca57d52be6d2e36e11f86a629b736330eab5f0d26b8e214f0a90b6f7b295fa1", untracked_digest: absent}
      - {path: "pipelines/time_dynamics/ap_mclachlan/exchange_certification.py", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:eed2aa591daa65954d3b256a3b5da25383ef680c38968b6d57c14ae722f8c46a", index_digest: "sha256:eed2aa591daa65954d3b256a3b5da25383ef680c38968b6d57c14ae722f8c46a", worktree_digest: "sha256:eed2aa591daa65954d3b256a3b5da25383ef680c38968b6d57c14ae722f8c46a", untracked_digest: absent}
      - {path: "pipelines/time_dynamics/ap_mclachlan/exchange_integration.py", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:2e5ec405f81e421554f71ae18d9ac11640ca93ba26ca618f363185ba3cb25be1", index_digest: "sha256:2e5ec405f81e421554f71ae18d9ac11640ca93ba26ca618f363185ba3cb25be1", worktree_digest: "sha256:2e5ec405f81e421554f71ae18d9ac11640ca93ba26ca618f363185ba3cb25be1", untracked_digest: absent}
      - {path: "pipelines/time_dynamics/ap_mclachlan/exchange_selector.py", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:81e187a1fe59cd70d8f70a4bce7bebc9b6c675e132861c1c0895a424b5698a69", index_digest: "sha256:81e187a1fe59cd70d8f70a4bce7bebc9b6c675e132861c1c0895a424b5698a69", worktree_digest: "sha256:81e187a1fe59cd70d8f70a4bce7bebc9b6c675e132861c1c0895a424b5698a69", untracked_digest: absent}
      - {path: "pipelines/time_dynamics/diagnostics/paper_ii_four_route_comparison.py", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:10dde0de180188f5b60acee00ae3ae1d3d74a479962ebe75e3e703a6dd044cae", index_digest: "sha256:10dde0de180188f5b60acee00ae3ae1d3d74a479962ebe75e3e703a6dd044cae", worktree_digest: "sha256:10dde0de180188f5b60acee00ae3ae1d3d74a479962ebe75e3e703a6dd044cae", untracked_digest: absent}
      - {path: "test/test_ap_mclachlan_avqds.py", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:ff1f6e4a55d8bd69f615000c4b23c4515b6d3ea423b099c88c15e802e911f283", index_digest: "sha256:ff1f6e4a55d8bd69f615000c4b23c4515b6d3ea423b099c88c15e802e911f283", worktree_digest: "sha256:ff1f6e4a55d8bd69f615000c4b23c4515b6d3ea423b099c88c15e802e911f283", untracked_digest: absent}
      - {path: "test/test_ap_mclachlan_exchange_certification.py", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:a389efcfba9b0da3c30fdae9065ccd649b90f295b1e55d0fe4a29575f95c4d5d", index_digest: "sha256:a389efcfba9b0da3c30fdae9065ccd649b90f295b1e55d0fe4a29575f95c4d5d", worktree_digest: "sha256:a389efcfba9b0da3c30fdae9065ccd649b90f295b1e55d0fe4a29575f95c4d5d", untracked_digest: absent}
      - {path: "test/test_ap_mclachlan_exchange_integration.py", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:57d24fad018c0b57d9f4039ef880f1c0e6bcb4055b0b180664d19ec3caf7f778", index_digest: "sha256:57d24fad018c0b57d9f4039ef880f1c0e6bcb4055b0b180664d19ec3caf7f778", worktree_digest: "sha256:57d24fad018c0b57d9f4039ef880f1c0e6bcb4055b0b180664d19ec3caf7f778", untracked_digest: absent}
      - {path: "test/test_ap_mclachlan_route_parity.py", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:2ace4aaa71aca4d6d1f3ab60c6998256ea06b7cec5ae10771805b4aea47cd377", index_digest: "sha256:2ace4aaa71aca4d6d1f3ab60c6998256ea06b7cec5ae10771805b4aea47cd377", worktree_digest: "sha256:2ace4aaa71aca4d6d1f3ab60c6998256ea06b7cec5ae10771805b4aea47cd377", untracked_digest: absent}
      - {path: "test/test_paper_ii_four_route_comparison.py", object_kind: {head: regular, index: regular, worktree: regular, untracked: absent}, state: clean, rename_from: null, rename_to: null, head_digest: "sha256:3757fb052a9b0079385f863087b8e6366c7a2802427d188c81ff2d36697a6251", index_digest: "sha256:3757fb052a9b0079385f863087b8e6366c7a2802427d188c81ff2d36697a6251", worktree_digest: "sha256:3757fb052a9b0079385f863087b8e6366c7a2802427d188c81ff2d36697a6251", untracked_digest: absent}
  primary_symbols:
    - {symbol: "select_avqds_appends", file: "pipelines/time_dynamics/ap_mclachlan/avqds.py", lines: "111-207", role: "AVQDS candidate geometry loop"}
    - {symbol: "certify_finalist", file: "pipelines/time_dynamics/ap_mclachlan/exchange_certification.py", lines: "207-351", role: "Finalist materialization, refit, geometry, and gates"}
    - {symbol: "select_exchange_patch", file: "pipelines/time_dynamics/ap_mclachlan/exchange_selector.py", lines: "191-468", role: "Full certification attempt loop"}
    - {symbol: "select_generalized_exchange_patch", file: "pipelines/time_dynamics/ap_mclachlan/exchange_integration.py", lines: "1274-1693", role: "Selector-pass and payload aggregation"}
    - {symbol: "run_append_mclachlan_trajectory", file: "pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py", lines: "906-1295", role: "Outer per-checkpoint selector repetition and persistence"}
    - {symbol: "complete_estimator_work", file: "pipelines/time_dynamics/diagnostics/paper_ii_four_route_comparison.py", lines: "596-746", role: "Human-facing exact/lower-bound aggregation"}
  related_symbols:
    - {symbol: "StructuralInsertionCache.column_count", relationship: "read by selector adapter", relevance: "exact positioned-column width"}
    - {symbol: "evaluate_mclachlan_geometry", relationship: "called by AVQDS and certification", relevance: "actual full geometry acquisition seam"}
    - {symbol: "AdaptiveTrajectoryPoint.to_json_dict", relationship: "serializes PatchDecision", relevance: "existing metadata persistence path"}
  execution_path:
    - "Checkpoint base geometry and solve are evaluated."
    - "The active policy evaluates candidate work: AVQDS full candidate geometries or generalized-exchange shared cache and classical structural enumeration."
    - "Generalized exchange materializes/refits/certifies ranked finalists; AVQDS reuses its winning candidate geometry."
    - "The selected patch is committed, possibly followed by another selector call at the same checkpoint."
    - "One final PatchDecision is serialized; new compact ledgers preserve all work that preceded it."
  pdg_constraints:
    - {description: "AVQDS evaluator attempts are downstream of successful materialization but precede evaluator completion.", affected_statements: ["pipelines/time_dynamics/ap_mclachlan/avqds.py:161-186"], implementation_consequence: "Increment attempt immediately before evaluate_mclachlan_geometry and completion immediately after return."}
    - {description: "Trajectory decision is overwritten on every selector call.", affected_statements: ["pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py:1109-1173"], implementation_consequence: "Append compact work receipt before overwrite/commit; serialize ordered list on final decision."}
    - {description: "Internal deletion fallback replaces inputs.", affected_statements: ["pipelines/time_dynamics/ap_mclachlan/exchange_integration.py:1519-1537"], implementation_consequence: "Record cache-pass receipt inside select_once before replacement."}
  architectural_patterns:
    - {pattern: "Additive JSON-ready nested telemetry", example_location: "pipelines/time_dynamics/ap_mclachlan/exchange_integration.py:1576-1634", usage_guidance: "Keep existing keys stable; add versioned compact receipts."}
    - {pattern: "Frozen result dataclasses with defaulted telemetry", example_location: "pipelines/time_dynamics/ap_mclachlan/exchange_certification.py:97-121", usage_guidance: "Append new default field so existing constructors remain valid."}
  files_to_modify:
    - {file: "pipelines/time_dynamics/ap_mclachlan/avqds.py", symbols: ["AVQDSDecision", "select_avqds_appends"], intended_change: "Per-round full-candidate geometry ledger"}
    - {file: "pipelines/time_dynamics/ap_mclachlan/exchange_certification.py", symbols: ["CertificationResult", "build_local_ray_refit", "certify_finalist"], intended_change: "Exact refit and full-geometry receipts"}
    - {file: "pipelines/time_dynamics/ap_mclachlan/exchange_selector.py", symbols: ["ExchangeSelection", "select_exchange_patch"], intended_change: "Retain every certifier receipt"}
    - {file: "pipelines/time_dynamics/ap_mclachlan/exchange_integration.py", symbols: ["select_generalized_exchange_patch"], intended_change: "Aggregate cache, fallback, and certification work"}
    - {file: "pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py", symbols: ["run_append_mclachlan_trajectory"], intended_change: "Preserve every outer selector-call receipt"}
    - {file: "pipelines/time_dynamics/diagnostics/paper_ii_four_route_comparison.py", symbols: ["complete_estimator_work"], intended_change: "Exact-first report aggregation with legacy fallback"}
  tests:
    - {file: "test/test_ap_mclachlan_avqds.py", scenarios: ["below threshold -> empty ledger", "one/multiple rounds -> exact dimensions and counts", "materialization/evaluation failures -> correct attempt/completion split"]}
    - {file: "test/test_ap_mclachlan_exchange_certification.py", scenarios: ["refit disabled/on-ray/multi-callback -> exact counts", "geometry/solve failures -> truthful completeness"]}
    - {file: "test/test_ap_mclachlan_exchange_selector.py", scenarios: ["all actual certification calls -> ordered receipts"]}
    - {file: "test/test_ap_mclachlan_exchange_integration.py", scenarios: ["cache formula", "fallback retains both passes", "untruncated reason summary", "parallel parity"]}
    - {file: "test/test_ap_mclachlan_route_parity.py", scenarios: ["scientific route parity unchanged", "multi-round selector work survives"]}
    - {file: "test/test_paper_ii_four_route_comparison.py", scenarios: ["new receipts -> exact", "incomplete/legacy receipts -> lower bound"]}
  verification_commands:
    - ".venv/bin/python -m pytest -q test/test_ap_mclachlan_avqds.py test/test_ap_mclachlan_exchange_certification.py test/test_ap_mclachlan_exchange_selector.py"
    - ".venv/bin/python -m pytest -q test/test_ap_mclachlan_exchange_integration.py test/test_ap_mclachlan_route_parity.py test/test_paper_ii_four_route_comparison.py"
  risks:
    - "Shared trajectory code reaches Paper III; metadata-only changes still require smoke verification."
    - "Whole-payload hashes may change due additive telemetry even when decisions are bitwise identical."
    - "Evaluator exceptions may leave unobservable partial work; mark incomplete."
  assumptions:
    - "Verify immediately before editing that all cited production/test files remain at the recorded digests and the branch still descends from 74e820a3."
    - "Verify current production refit comes from build_local_ray_refit and therefore exposes its counter; injected callbacks may be conservatively incomplete."
  open_questions:
    - "The off-by-one L2 round cap is a separate behavioral defect and is intentionally not corrected in this telemetry change."
    - "A physical shot ledger requires a future measurement/grouping/precision allocation protocol; logical statevector proxies are not shots."
  avoid:
    - "Do not change physics, selectors, rankings, candidate populations, gates, budgets, integrator behavior, or route identifiers."
    - "Do not fix the L2 round-cap defect in this change."
    - "Do not relabel logical geometry/refit entries as hardware shots."
    - "Do not remove legacy lower-bound inference."
    - "Do not edit or stage unrelated untracked manuscript-asset files."
    - "Do not regenerate PDFs or submit/restart CHTC jobs."
```

## 12. Assumptions and Open Questions

- [assumed] Production refit hooks continue to be built by `build_local_ray_refit`; verify at execution start. Arbitrary injected refits are supported but conservatively mark telemetry incomplete.
- [assumed] The logical refit proxy counts one overlap objective plus one derivative per active coordinate for each full state-and-tangent callback; verify the callback still prepares all indexed tangents in one invocation before relying on `p+1`.
- [verified] The L2 selector-call/commit cap discrepancy is real but outside this metadata-only task. Record actual calls/commits and file the finding in the implementation report; do not alter behavior.
- [verified] Physical shot accounting remains an open methodological extension because no measurement grouping, precision target, allocation rule, or shot ledger exists in these statevector runs.

## 13. Definition of Done

- Every new AVQDS candidate-evaluation round and every generalized-exchange selector/cache/certification/refit pass is persisted exactly or explicitly marked incomplete at its originating seam.
- Repeated outer selector calls and internal fallback passes survive serialization in order.
- `complete_estimator_work` returns exact status for complete new receipts, lower-bound status for legacy/incomplete data, and no `>=`-style ambiguity for the planned reruns.
- Existing serialized keys remain available; old artifacts still aggregate.
- Focused unit/integration/parity tests and shared-route Paper III smoke coverage pass.
- GitNexus pre-edit impacts are current, post-change detect-changes is reviewed, and only explicit task files are committed on `codex/paper-ii-exact-estimator-telemetry-20260903`.

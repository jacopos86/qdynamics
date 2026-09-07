# GitNexus Engineering Plan

> Task: curate the Paper II implementation before adding optional compiled-cost exchange policies.
> Evidence verified at commit 74ff226338caa56d29ef28004dd7b37615f5e511; GitNexus refreshed this session with --index-only --pdg, current schema-4 runner, incompleteReasons=[].
> Evidence provenance schema 2; global dirty digest 0a9c85780067d9afcd0764f307b60891e3cee927ee11eaeb5ec7826d10fd82cd; 16 sorted cited paths; exact generated plan path excluded.

## 1. Objective

Implement a bounded curation increment: delete confirmed dead machinery, isolate historical acquisition utility, and eliminate disabled-hook work. Preserve current route outputs, public configuration and scientific safeguards. Specify the optional protocol boundary without enabling or implementing it in this increment.

## 2. Current Behaviour

[verified] `exchange_structural.iter_structural_families` computes frozen-support captured drift, then `scored` (472–612) derives utilities, diagnostic Gamma and compiled inputs. `priority_value` (614) still uses historical `candidate.score`; pure d0 scores seal the insertion frontier (684–700). Batch acquisition also uses these scores. `exchange_integration.build_population_ranker` (564–843) subsequently ranks complete patches using the selected route, with compiled population normalization on the compiled routes.

[verified] Zero conditioning/history weights are defaults, but nonzero hooks are tested. Their calculations currently run even at zero. History recording in integration is separately observable resume state and is retained.

## 3. Relevant Architecture

[verified] Geometry/memoization belongs to `structural_cache`; family admission and acquisition belong to `exchange_structural`; compiled counts/placement belong to `compiled_patch_cost` and `CompiledCostRuntime`; population normalization belongs to `population_scoring`; certification is downstream. Keep all these authorities intact. [verified] Prospective source epochs compose historical base files with the explicit `_EXCHANGE_FIX_PATHS` overlay; new imports need that closure.

## 4. GitNexus Findings

- [graph] `impact(iter_structural_families, upstream, includeTests=true, depth=3)` reports MEDIUM, 40 affected, five direct. Direct dependents: `select_exchange_patch`, `enumerate_structural_candidates`, `_family_stream`, and two structural error/worker tests. Scoped structural, selector, integration and oracle suites cover them.
- [graph] `impact(CompiledCostRuntime, upstream, depth=3)` reports HIGH, 140 affected and six direct file dependents: adaptive trajectory, prospective worker, AP runner, SIAM package, AVQDS tests and integration tests. [inferred] Defer runtime extraction until its cache-lifetime extension; moving it now adds risk without removing obsolete behavior.
- [graph] `impact(support_frontier.py)`, `impact(build_append_support_frontier)`, `impact(gain_for_support)` and the four leaf prune helpers report UNKNOWN/no resolved callers. [verified] Targeted source/test/dynamic-string searches corroborate no live consumer. `_finite_or_none` has only three incoming helpers in the same dead island.
- [graph] `_EXCHANGE_FIX_PATHS` has no resolved callers. [verified] Source tuple expansion into `EXECUTOR_OVERLAY_PATHS` and source-composition tests prove this is a live dependency that the graph does not resolve.

## 5. Statement-Level PDG Findings

[graph] `pdg_query(target=iter_structural_families, mode=flows, variable=score)` records score definition at 527 feeding rank_primary (543), StructuralCandidate (586), and singleton priority stores (627/636). [verified] Source confirms utility feeds acquisition before compiled final ranking. Preserve this ordering, the serial replay after parallel memo warm, and exception timing for active callbacks. Preserve independent conditioning injection even when weighted metadata hooks are zero. Gamma monotonicity and candidate-local compile-failure branches remain unchanged.

## 6. Proposed Changes

[verified] Delete `support_frontier.py` and three stale `adaptive_trajectory.__all__` strings; delete unused `prune_cost` helpers and `support_patch.gain_for_support`. Retain public `PruneCostSettings`, support-patch payloads and tested numeric APIs. Update current module maps; preserve historical handoffs.

[inferred] Extract the arithmetic currently in `iter_structural_families.scored` into proposed `exchange_legacy_scoring.score_legacy_acquisition`, with a small immutable component result. Keep `StructuralScoreWeights` and `StructuralCandidate` at their current public import paths. The helper may invoke the existing q/cost hooks in their current order; it must not own enumeration, compiled ranking or certification. Preserve all score components and operation order for finite inputs. Do not introduce new user-facing knobs.

[inferred] Skip history-loss callback evaluation when lambda_hist is zero; skip memoized condition-number retrieval/log10 when both weighted conditioning terms are zero. Disabled diagnostics may therefore be unavailable without affecting scoring. Keep the unweighted deletion_conditioning hook active, and preserve history recording/serialization. Construct each removed-index set once per q_of call, rather than once per retained index.

[inferred] Add the extracted file to the prospective executor overlay and test that an older source tree plus the overlay carries the import. Document acquisition and final compiled ranking separately.

## 7. Implementation Sequence

1. Capture small pre-change candidate/family fingerprints for default, nonzero hooks, accuracy-debt and bounded/batch frontier cases; preserve the external receipt. Existing baseline: 113 structural/selector/integration/compiled/population tests passed.
2. Delete the verified dead surface and update its current references. Retain public/tested compatibility APIs.
3. Extract historical acquisition arithmetic and add zero-work guards. Add focused behavioral tests alongside existing exact nonzero-hook and serial/parallel tests.
4. Update prospective source composition and its regression; update module ownership descriptions.
5. Run the focused numerical, oracle, compatibility and packaging checks; compare baseline fingerprints exactly. Refresh GitNexus after relation changes, inspect staged detect_changes and git diff --check, then commit the coherent increment. Record actual validation and remaining extension work separately; do not rewrite this plan.

## 8. Test Strategy

[verified] Existing runnable baseline uses `/Users/lordskyler/Holstein_test/.venv/bin/python` under `pipelines/shell/ram_guard.py`.

- Zero-weight history/metadata callbacks that raise if called -> enumerate -> no call, same finite candidate/family fingerprints and telemetry.
- Nonzero conditioning/history -> retain existing exact formula tests and callback failures. Zero weighted metadata plus positive unweighted conditioning hook -> still changes deletion utility as declared.
- Default and nonzero hooks, batch=1/2, frontier sealed/unrestricted, debt tie ordering -> pre/post fingerprints identical.
- Whole-family guard, solve error serial timing, Q/Gamma shared population, signed marginals, compilation failures and timeouts -> existing structural/selector/integration/oracle/compiled suites stay green.
- Prospective historical-base overlay -> extracted source present and its import executable; tamper/hash tests still hold. Existing run/route and SIAM serialization tests preserve omission semantics.

## 9. Risk and Impact Analysis

[verified] Historic exchange remains executable in prospective and SIAM routes; all four bundles are registered comparison arms. Removing their zero-default hooks or replacing frontier scores would change scientific behavior. [inferred] The extraction is medium impact and must keep the default population/order unchanged. Disabled-hook short-circuiting intentionally stops callback side effects and nonfinite diagnostics from leaking through multiplication by zero. It does not relax numerical solver or certification checks. Do not cache population-normalized scores across checkpoints.

[verified] Larger `support_patch` numeric code is test-backed even where no production caller exists. Defer its removal until the scientific invariants are transferred to active authorities. A stale diagnostic-only Paper III demo does not justify resurrecting retired config; it is outside this increment.

## 10. Files Expected to Change

| File | Responsibility |
| --- | --- |
| `pipelines/time_dynamics/ap_mclachlan/support_frontier.py` | Delete the unreferenced retired module. |
| `pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py` | Remove only the three already-unbound APPEND_MACRO_SCOUT exports. |
| `pipelines/time_dynamics/ap_mclachlan/prune_cost.py` | Delete the private unused payload/conditioning helper island and unused imports. |
| `pipelines/time_dynamics/ap_mclachlan/support_patch.py` | Delete unused gain_for_support and its export; keep tested mathematical helpers and payload API. |
| `pipelines/time_dynamics/ap_mclachlan/exchange_structural.py` | Delegate historical acquisition arithmetic; skip inactive work; preserve candidate/weight API and enumeration. |
| `pipelines/time_dynamics/ap_mclachlan/exchange_legacy_scoring.py` | New bounded helper for current historical utility/rank arithmetic; no compiled-cost policy. |
| `pipelines/time_dynamics/prospective_apm_exchange_worker.py` | Include extracted module in source-epoch overlay closure. |
| `pipelines/time_dynamics/README.md` | Correct current module map. |
| `pipelines/time_dynamics/ap_mclachlan/README.md` | Correct current selector/scoring module map. |
| `agent_guidance/time-dynamics/AGENTS.md` | Remove stale module entry and distinguish acquisition utility from compiled population rank. |
| `test/test_ap_mclachlan_exchange_structural.py` | Disabled-hook and active-hook parity scenarios. |
| `test/test_prospective_apm_exchange_worker.py` | Overlay module presence and executable source closure. |

## 11. Reusable Implementation Context

```json
{
  "implementation_context": {
    "task_summary": "Paper II curation before optional compiled-cost protocol",
    "acceptance_criteria": [
      "Delete verified dead machinery",
      "Preserve current scoring/population outputs",
      "Disabled hooks do no work",
      "Prospective source closure and tests pass",
      "No new protocol enabled"
    ],
    "evidence_provenance": {
      "schema_version": 2,
      "head_commit": "74ff226338caa56d29ef28004dd7b37615f5e511",
      "generated_plan_path": "docs/plans/2026-09-07-gitnexus-plan-paper-ii-scoring-curation.md",
      "global_dirty_digest": {
        "algorithm": "sha256",
        "canonicalization": "gitnexus-evidence-provenance-v2 NUL-framed UTF-8 records",
        "value": "0a9c85780067d9afcd0764f307b60891e3cee927ee11eaeb5ec7826d10fd82cd"
      },
      "cited_path_manifest": [
        {
          "path": "agent_guidance/time-dynamics/AGENTS.md",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:7701ce08dedb229b9fff74a07011af9401d5311341c984f190050a1a7e1064d7",
          "index_digest": "sha256:7701ce08dedb229b9fff74a07011af9401d5311341c984f190050a1a7e1064d7",
          "worktree_digest": "sha256:7701ce08dedb229b9fff74a07011af9401d5311341c984f190050a1a7e1064d7",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/time_dynamics/README.md",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:9cff51f4958d1f6df534df444654a73988ea227e894c5bae519918bb0fb5a5a1",
          "index_digest": "sha256:9cff51f4958d1f6df534df444654a73988ea227e894c5bae519918bb0fb5a5a1",
          "worktree_digest": "sha256:9cff51f4958d1f6df534df444654a73988ea227e894c5bae519918bb0fb5a5a1",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/time_dynamics/ap_mclachlan/README.md",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:e747c3926d11a3fa47ede9fa4be5e6bde139ca61c9b5e1d6c8da91a519f3f948",
          "index_digest": "sha256:e747c3926d11a3fa47ede9fa4be5e6bde139ca61c9b5e1d6c8da91a519f3f948",
          "worktree_digest": "sha256:e747c3926d11a3fa47ede9fa4be5e6bde139ca61c9b5e1d6c8da91a519f3f948",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:bd460e2a20b672f56048cfcc057741d2275bc715675987b6a68a619172de9072",
          "index_digest": "sha256:bd460e2a20b672f56048cfcc057741d2275bc715675987b6a68a619172de9072",
          "worktree_digest": "sha256:bd460e2a20b672f56048cfcc057741d2275bc715675987b6a68a619172de9072",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/time_dynamics/ap_mclachlan/compiled_patch_cost.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:c3e4bff0ca503bca63cbf111281a093ee4e6e2c9ab6ca367394e950902d4c4b1",
          "index_digest": "sha256:c3e4bff0ca503bca63cbf111281a093ee4e6e2c9ab6ca367394e950902d4c4b1",
          "worktree_digest": "sha256:c3e4bff0ca503bca63cbf111281a093ee4e6e2c9ab6ca367394e950902d4c4b1",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/time_dynamics/ap_mclachlan/exchange_integration.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:0343eb5d478420d5b65031d1df3ec4014a1c17265c89ec37f9fb1c037250ddd3",
          "index_digest": "sha256:0343eb5d478420d5b65031d1df3ec4014a1c17265c89ec37f9fb1c037250ddd3",
          "worktree_digest": "sha256:0343eb5d478420d5b65031d1df3ec4014a1c17265c89ec37f9fb1c037250ddd3",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/time_dynamics/ap_mclachlan/exchange_legacy_scoring.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "absent"
          },
          "state": "absent",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/time_dynamics/ap_mclachlan/exchange_structural.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:ab258b483d3e1d29b3dffa1734071d75bde9b732ac9d8de923951afff0c69806",
          "index_digest": "sha256:ab258b483d3e1d29b3dffa1734071d75bde9b732ac9d8de923951afff0c69806",
          "worktree_digest": "sha256:ab258b483d3e1d29b3dffa1734071d75bde9b732ac9d8de923951afff0c69806",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/time_dynamics/ap_mclachlan/population_scoring.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:a0aa5b3e3b6ac4d101e30b37988f8726b1a23755cd2ea9ec0473616895c1a0f9",
          "index_digest": "sha256:a0aa5b3e3b6ac4d101e30b37988f8726b1a23755cd2ea9ec0473616895c1a0f9",
          "worktree_digest": "sha256:a0aa5b3e3b6ac4d101e30b37988f8726b1a23755cd2ea9ec0473616895c1a0f9",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/time_dynamics/ap_mclachlan/prune_cost.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:45251091235a4cab4505e39405ccadc35ae1ed85762d8f8a70705224e52488cf",
          "index_digest": "sha256:45251091235a4cab4505e39405ccadc35ae1ed85762d8f8a70705224e52488cf",
          "worktree_digest": "sha256:45251091235a4cab4505e39405ccadc35ae1ed85762d8f8a70705224e52488cf",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/time_dynamics/ap_mclachlan/support_frontier.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:23cfe338a65b0bcfe9c0089280ee42c81b4fd66689125a6a8c7d190bfb29e079",
          "index_digest": "sha256:23cfe338a65b0bcfe9c0089280ee42c81b4fd66689125a6a8c7d190bfb29e079",
          "worktree_digest": "sha256:23cfe338a65b0bcfe9c0089280ee42c81b4fd66689125a6a8c7d190bfb29e079",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/time_dynamics/ap_mclachlan/support_patch.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:1c474e3db3b093e9d75dab67abe13bc781cdecebc48feb3415d5841f2a350563",
          "index_digest": "sha256:1c474e3db3b093e9d75dab67abe13bc781cdecebc48feb3415d5841f2a350563",
          "worktree_digest": "sha256:1c474e3db3b093e9d75dab67abe13bc781cdecebc48feb3415d5841f2a350563",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/time_dynamics/prospective_apm_exchange_worker.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:4f82b337752424beb7918eabb40069e0a0eafcf7205ac953de3af920e8733c14",
          "index_digest": "sha256:4f82b337752424beb7918eabb40069e0a0eafcf7205ac953de3af920e8733c14",
          "worktree_digest": "sha256:4f82b337752424beb7918eabb40069e0a0eafcf7205ac953de3af920e8733c14",
          "untracked_digest": "absent"
        },
        {
          "path": "test/test_ap_mclachlan_exchange_integration.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:e1833d7ba370e036802190365d4621ea1c6a8c6c923b77503efcf4fafe87e5a4",
          "index_digest": "sha256:e1833d7ba370e036802190365d4621ea1c6a8c6c923b77503efcf4fafe87e5a4",
          "worktree_digest": "sha256:e1833d7ba370e036802190365d4621ea1c6a8c6c923b77503efcf4fafe87e5a4",
          "untracked_digest": "absent"
        },
        {
          "path": "test/test_ap_mclachlan_exchange_structural.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:049fdc43eea34cad32f61c16a003f61c110d410e8d31a220801f5e76186be355",
          "index_digest": "sha256:049fdc43eea34cad32f61c16a003f61c110d410e8d31a220801f5e76186be355",
          "worktree_digest": "sha256:049fdc43eea34cad32f61c16a003f61c110d410e8d31a220801f5e76186be355",
          "untracked_digest": "absent"
        },
        {
          "path": "test/test_prospective_apm_exchange_worker.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:efd8c9be8576555b689c4393d22d376711e91d74160b11aad8bdf5250b338412",
          "index_digest": "sha256:efd8c9be8576555b689c4393d22d376711e91d74160b11aad8bdf5250b338412",
          "worktree_digest": "sha256:efd8c9be8576555b689c4393d22d376711e91d74160b11aad8bdf5250b338412",
          "untracked_digest": "absent"
        }
      ]
    },
    "primary_symbols": [
      {
        "symbol": "iter_structural_families",
        "file": "pipelines/time_dynamics/ap_mclachlan/exchange_structural.py",
        "lines": "312\u2013933",
        "role": "enumeration and acquisition score seam"
      }
    ],
    "related_symbols": [
      {
        "symbol": "select_exchange_patch",
        "relationship": "CALLS",
        "relevance": "main consumer"
      },
      {
        "symbol": "build_population_ranker",
        "relationship": "downstream-ranker",
        "relevance": "compiled authority unchanged"
      }
    ],
    "execution_path": [
      "frozen geometry",
      "whole-family admission",
      "historical acquisition utilities",
      "candidate compiled inputs",
      "population ranking",
      "certification"
    ],
    "pdg_constraints": [
      {
        "description": "score feeds candidate and frontier priorities",
        "affected_statements": [
          "exchange_structural.py:527",
          "exchange_structural.py:543",
          "exchange_structural.py:586",
          "exchange_structural.py:627",
          "exchange_structural.py:636"
        ],
        "implementation_consequence": "retain arithmetic and ordering"
      }
    ],
    "architectural_patterns": [
      {
        "pattern": "explicit source overlay closure",
        "example_location": "pipelines/time_dynamics/prospective_apm_exchange_worker.py:_EXCHANGE_FIX_PATHS",
        "usage_guidance": "new imported module must be packaged"
      }
    ],
    "files_to_modify": [
      {
        "file": "pipelines/time_dynamics/ap_mclachlan/support_frontier.py",
        "symbols": [],
        "intended_change": "Delete the unreferenced retired module."
      },
      {
        "file": "pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py",
        "symbols": [],
        "intended_change": "Remove only the three already-unbound APPEND_MACRO_SCOUT exports."
      },
      {
        "file": "pipelines/time_dynamics/ap_mclachlan/prune_cost.py",
        "symbols": [],
        "intended_change": "Delete the private unused payload/conditioning helper island and unused imports."
      },
      {
        "file": "pipelines/time_dynamics/ap_mclachlan/support_patch.py",
        "symbols": [],
        "intended_change": "Delete unused gain_for_support and its export; keep tested mathematical helpers and payload API."
      },
      {
        "file": "pipelines/time_dynamics/ap_mclachlan/exchange_structural.py",
        "symbols": [],
        "intended_change": "Delegate historical acquisition arithmetic; skip inactive work; preserve candidate/weight API and enumeration."
      },
      {
        "file": "pipelines/time_dynamics/ap_mclachlan/exchange_legacy_scoring.py",
        "symbols": [],
        "intended_change": "New bounded helper for current historical utility/rank arithmetic; no compiled-cost policy."
      },
      {
        "file": "pipelines/time_dynamics/prospective_apm_exchange_worker.py",
        "symbols": [],
        "intended_change": "Include extracted module in source-epoch overlay closure."
      },
      {
        "file": "pipelines/time_dynamics/README.md",
        "symbols": [],
        "intended_change": "Correct current module map."
      },
      {
        "file": "pipelines/time_dynamics/ap_mclachlan/README.md",
        "symbols": [],
        "intended_change": "Correct current selector/scoring module map."
      },
      {
        "file": "agent_guidance/time-dynamics/AGENTS.md",
        "symbols": [],
        "intended_change": "Remove stale module entry and distinguish acquisition utility from compiled population rank."
      },
      {
        "file": "test/test_ap_mclachlan_exchange_structural.py",
        "symbols": [],
        "intended_change": "Disabled-hook and active-hook parity scenarios."
      },
      {
        "file": "test/test_prospective_apm_exchange_worker.py",
        "symbols": [],
        "intended_change": "Overlay module presence and executable source closure."
      }
    ],
    "tests": [
      {
        "file": "test/test_ap_mclachlan_exchange_structural.py",
        "scenarios": [
          "disabled hooks not called",
          "active hook exact weights",
          "serial/parallel bitwise parity",
          "pre/post branch fingerprints"
        ]
      },
      {
        "file": "test/test_prospective_apm_exchange_worker.py",
        "scenarios": [
          "new helper overlay closure for old epoch"
        ]
      }
    ],
    "verification_commands": [
      "/Users/lordskyler/Holstein_test/.venv/bin/python pipelines/shell/ram_guard.py --limit-mb 3000 -- /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q -p no:cacheprovider test/test_ap_mclachlan_exchange_structural.py test/test_ap_mclachlan_exchange_selector.py test/test_ap_mclachlan_exchange_integration.py test/test_ap_mclachlan_exchange_oracle.py test/test_ap_mclachlan_compiled_patch_cost.py test/test_ap_mclachlan_population_scoring.py test/test_ap_mclachlan_support_patch.py test/test_ap_mclachlan_support_atoms.py test/test_prospective_apm_exchange_worker.py test/test_paper_ii_runs.py test/test_siam_chtc_package.py",
      "git diff --check"
    ],
    "risks": [
      "Historical scoring still determines compiled acquisition",
      "Source epoch misses extracted files unless explicitly overlaid",
      "Do not remove Gamma/shared-population checks"
    ],
    "assumptions": [
      "Disabled hooks need not execute: verify raising-callback tests",
      "New overlay file may be absent in historical base: verify source-closure regression"
    ],
    "open_questions": [
      "Future conditional-frontier screening budget and realization population must be fixed before implementation"
    ],
    "avoid": [
      "Do not repeat broad repository discovery",
      "Do not change compiled runtime/cache lifetime or normalization in this increment",
      "Do not delete nonzero hooks/history recording/public config",
      "Do not touch manuscript/evidence or run campaigns",
      "No UI actions",
      "Do not edit the plan during execution"
    ]
  }
}
```

## 12. Assumptions and Open Questions

No unresolved choice blocks curation. [assumed] Disabled weights mean their callbacks need not execute; verify using explicit no-call tests while retaining active-hook behavior. [assumed] Extracted modules can be represented as new overlay files when absent in a source epoch; verify with the source-composition regression before accepting the extraction.

The subsequent optional protocol remains separate from reported defaults:

- [inferred] Given a declared deletion-conditioned patch population, rank complete patch merit with its own signed Qiskit marginal relative to the current full ansatz. Deletions are compiled hypothetical patches; they require no QPU deletion trial for this resource term.
- [inferred] Within each patch, minimize the final population-relative cost over certified equivalent insertion realizations. Use one explicitly sealed normalization population/receipt for both representative choice and outer ranking. Normalizing only already-chosen representatives introduces a circular selection problem; declare the population of realizations and multiplicities first.
- [inferred] Reuse frozen measured columns only for positions certified equivalent in that frozen geometry. Commutations enabled solely by removal may change a pre-deletion tangent and cannot inherit measurements without proof or extra acquisition.
- [inferred] Cache raw compiled resource triples by full structural word, reference preparation and compiler contract (backend/target, layout, passes, seed, versions, structural-angle policy). Recompute patch-relative differences when the base changes and population scores for every new population. Cache pruning must bound trajectory memory. Parameter-independent structural templates permit reuse; do not assume numeric-angle compilation is invariant.
- [inferred] Conditional frontier screening may inspect D times the measured candidate pool before keeping W; it needs a declared classical budget and whole-family admission policy. Reusing W to prefilter by pure score would reproduce the defect. Do not hide screened work in final-frontier accounting.
- [inferred] Parallel compilation/caching can reduce latency; no walltime-equality claim follows without measuring throughput. Keep acquisition, compilation, normalization, ranking and commit receipts separate.

## 13. Definition of Done

Dead code and stale current references are removed; historical arithmetic is isolated; disabled hooks perform no work; active hooks and current candidate populations/order remain verified; prospective overlay is closed; focused checks and graph gate pass; changes are committed on the isolated curation branch. No campaign, manuscript/evidence change, or new compiled protocol is represented as completed by this increment.

Files to edit: `pipelines/time_dynamics/ap_mclachlan/support_frontier.py`, `pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py`, `pipelines/time_dynamics/ap_mclachlan/prune_cost.py`, `pipelines/time_dynamics/ap_mclachlan/support_patch.py`, `pipelines/time_dynamics/ap_mclachlan/exchange_structural.py`, `pipelines/time_dynamics/ap_mclachlan/exchange_legacy_scoring.py`, `pipelines/time_dynamics/prospective_apm_exchange_worker.py`, `pipelines/time_dynamics/README.md`, `pipelines/time_dynamics/ap_mclachlan/README.md`, `agent_guidance/time-dynamics/AGENTS.md`, `test/test_ap_mclachlan_exchange_structural.py`, `test/test_prospective_apm_exchange_worker.py`

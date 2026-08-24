# Golden regression data — Increment 0 rescue (2026-08-24)

Executes Increment 0 of
`agent_guidance/static-adapt/HANDOFF_ADAPT_PIPELINE_DECOMPOSITION_20260824.md`,
under Gate 1 of
`/Users/jakestrobel/local_repos/ADAPT---Paper-I/PAPER_I_REFACTOR_BEHAVIORAL_CONTRACT.md`.

Gate 1 requires archives and provenance files to "remain byte-identical **or be
copied with recorded SHA-256 hashes**". That second clause is what this
directory implements. Nothing here was rewritten in place.

Verify:

```bash
cd agent_guidance/static-adapt/golden && shasum -a 256 -c MANIFEST.sha256
# and, from the repo root, for the in-place archives:
cd output/local_runs/paper_i_hh_b3_measured_residual_20260821/transfer && \
  shasum -a 256 -c <(sed -n 's/^\([0-9a-f]*\)  b3:/\1  /p' \
  ../../../../agent_guidance/static-adapt/golden/ARCHIVES.sha256)
```

## What is here

| file | source | why |
|---|---|---|
| `bundle3_final_results_manifest.json` | `output/pdf/paper_i_ra_allphase_adaptive_20260817/` | `output/` is gitignored (`.gitignore:49`) |
| `kstar_tables.json` | `ra-refactor-stage2` worktree, `chtc/paper_i_ra_adapt_repair_20260727/results_tooling/` | tracked there on `ra-refactor-stage2` @ `4334049f`, but not on this branch |
| `b9_mr_measured_costexp_20260824_package_manifest.json` | `ra-refactor-stage2` worktree, `…/paper_i_hh_b9_mr_measured_costexp_20260824_chtc/` | **was untracked** in that worktree |
| `ra_adapt_l3_manuscript_cost_20260824__RESULTS_SOURCES.md` | `appendix-demos-20260820` worktree | **was untracked** |
| `ra_adapt_l3_repaired_20260823__RESULTS_SOURCES.md` | `appendix-demos-20260820` worktree | **was untracked** |
| `bundle3_surviving_inputs/` | see below | the 4 of 24 Bundle-3 inputs that still exist |
| `ARCHIVES.sha256` | — | 27 hashes for the 13.5 GB of run archives left in place |

## Why the run archives were not copied into git

The handoff said "copy these into a tracked path … and commit". Measured, the
two `transfer/` trees are **11 GB (18 archives)** and **2.5 GB (9 archives)**.
Committing 13.5 GB of `.tar.gz` is not a viable rescue and is not what Gate 1
asks for. They remain at their original gitignored paths; `ARCHIVES.sha256`
records their identity so any later copy or move can be proven byte-identical.

**They are still unbacked.** Recording a hash proves corruption, it does not
prevent loss. An off-repo backup of these two trees is still outstanding.

## Correction to the handoff's stated risk

The handoff said `git worktree prune` would destroy the worktree-resident
files. That mechanism is wrong: `prune` only removes administrative entries for
worktrees whose directories are *already* gone, and neither `ra-refactor-stage2`
nor `appendix-demos-20260820` is prunable (`git worktree list` marks only the
two `/private/tmp` ones). The real exposure was different and narrower: three of
those files were **untracked**, so deleting the worktree directory would have
lost them. `kstar_tables.json` was already committed on `ra-refactor-stage2`
and was never at risk of loss.

Also: two divergent `kstar_tables.json` copies exist. The one saved here is the
`ra-refactor-stage2` copy the handoff names (`e7aa782c…`, 2026-08-24 11:31). The
`paper-i-factual-audit-20260822` worktree holds a different one (`bdea7469…`).

## STOP CONDITION — Bundle-3 provenance is already broken

The handoff's Increment 0 says: *"Stop if: any source is already missing.
Report which — do not reconstruct it from another bundle."* **That condition is
met.**

`bundle3_final_results_manifest.json` is a pointer file: it records 24 inputs
with SHA-256 for each. Checked on 2026-08-24:

- **4 of 24 inputs still exist, and all 4 match their recorded hash exactly.**
- **20 of 24 are gone.** They lived in the scratchpad of a *different* agent
  session, under the **iCloud checkout**, not this one:
  `/private/tmp/claude-501/-Users-jakestrobel-Documents-Holstein-implementation-Holstein-test-fullclone-3/91434114-…/scratchpad/`.
  That directory no longer exists and `/private/tmp` is not backed up.

Among the missing is `kstar_tables.json` @ `629c8c13…` — **the version that
actually generated the Bundle-3 PDF**. Neither surviving copy matches it
(`e7aa782c…`, `bdea7469…`), nor do two stale backups found in another dead
scratchpad (`b9528ef5…`, `51649572…`). A filesystem-wide search found no file
with hash `629c8c13…`.

Consequence: Bundle 3's reported numbers cannot currently be re-derived from
their recorded inputs. Contract Gate 1 ("importing each golden dataset
reproduces its trajectory length, k*, event markers, resource tuples, and
accounting scope exactly") is **not satisfiable for Bundle 3** as things stand.
The contract's out-of-scope list forbids filling Bundle-3 cells from another
bundle, so this is an evidence decision for the author, not something to patch.

The 20 missing inputs, with the hashes Bundle 3 expects:

| input | expected sha256 |
|---|---|
| `kstar_tables.json` | `629c8c13bdc6803d…` |
| `live_samples_fresh.jsonl` | `4c584ad24133aa9d…` |
| `stream:9664325.0-sixregime__a_costexp_always_open__weak_weak__nph3.out` | `1ee5de9addb82b26…` |
| `stream:9664325.1-sixregime__a_costexp_always_open__intermediate_weak__nph3.out` | `c9bbe4ef005e9911…` |
| `stream:9664325.12-sixregime__b_depth_append_ra__weak_weak__nph3.out` | `a9de4952f6626a0d…` |
| `stream:9664325.13-sixregime__b_depth_append_ra__intermediate_weak__nph3.out` | `255a92cbd8c8e681…` |
| `stream:9664325.14-sixregime__b_depth_append_ra__strong_weak_u8__nph3.out` | `3682aeb48b7d2bf5…` |
| `stream:9664325.15-sixregime__b_depth_append_ra__weak_strong__nph7.out` | `f0cf8b3f733f2432…` |
| `stream:9664325.16-sixregime__b_depth_append_ra__intermediate_strong__nph7.out` | `3e67ff60bf24bc5b…` |
| `stream:9664325.17-sixregime__b_depth_append_ra__strong_strong_u8__nph7.out` | `0565dc5961708534…` |
| `stream:9664325.18-sixregime__b_depth_costexp_always_open__weak_weak__nph3.out` | `8e61741f1dcbe1cd…` |
| `stream:9664325.19-sixregime__b_depth_costexp_always_open__intermediate_weak__nph3.out` | `ba63b6db7c65fbdf…` |
| `stream:9664325.2-sixregime__a_costexp_always_open__strong_weak_u8__nph3.out` | `1026bd62312528df…` |
| `stream:9664325.20-sixregime__b_depth_costexp_always_open__strong_weak_u8__nph3.out` | `d60b8a0ef17ed25e…` |
| `stream:9664325.24-sixregime__b_depth_costexp_plateau_position__weak_weak__nph3.out` | `ac22593fc591d7c7…` |
| `stream:9664325.25-sixregime__b_depth_costexp_plateau_position__intermediate_weak__nph3.out` | `98db093507e8b898…` |
| `stream:9664325.26-sixregime__b_depth_costexp_plateau_position__strong_weak_u8__nph3.out` | `8c43069fd31b7bf1…` |
| `stream:9664325.6-sixregime__a_costexp_plateau_position__weak_weak__nph3.out` | `5fcd47e5715c91e9…` |
| `stream:9664325.7-sixregime__a_costexp_plateau_position__intermediate_weak__nph3.out` | `5559011c42d6af69…` |
| `stream:9664325.8-sixregime__a_costexp_plateau_position__strong_weak_u8__nph3.out` | `0475f561da4dffb3…` |

The 4 that survive are preserved in `bundle3_surviving_inputs/`.

Note the same exposure applies going forward: run tooling wrote golden inputs
into session scratchpads under `/private/tmp`. Anything a manifest points at
should be written somewhere durable instead.

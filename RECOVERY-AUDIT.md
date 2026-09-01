# Holstein_test — Repository Recovery Audit

**Date:** 2026-08-31 · **Auditor:** Claude (read-only audit; nothing has been merged, pushed, or deleted)
**Context:** The computer holding the working copy was lost. This clone at `/Users/Patron/Holstein_test` was recovered from GitHub (`git@github.com:20jastrobel/Holstein_test.git`) and is currently the **only local copy**. 52 remote branches, 1 tag (`route-c-preimpl-2026-06-08`), 5 PRs (all pointing at branch tips we already have — nothing hidden there).
*(Superseded by §12, 2026-08-31: the active local copy is now `/Users/lordskyler/Holstein_test` on the replacement M4 Pro, cloned over HTTPS via gh; 54 remote branches after the recovery pushes. §§2–11 describe the recovery session's machine and moment.)*

---

## 1. Executive summary

- **`origin/paper-ii-exchange-selector` is the real main.** Last commit 2026-08-30, 653 commits ahead of stale `origin/main` (frozen 2026-03-25). It already fully contains `RA`, `Simplied-Refactored`, `golden-rescue-20260824`, `codex/paper-ii-route-diagrams`, `codex/hardcoded-test-retirement`, `perf/sector-audit-memoization`, `claude/paper-i-20260824`, and `codex/paper-iii-exchange-manuscript-20260826`. There is no branch literally named "paper-ii-refactor"; this is the one.
- **The trunk is healthy and runnable** once a Python 3.12 environment is installed (details §7). All 18,779 tracked files check out; 426 test files; only 2 files need Python ≥3.12 syntax; nothing looks code-broken.
- **But the trunk is NOT the whole story.** The agents worked in parallel worktrees and several branches are the *sole* holders of irreplaceable material — most critically the **Paper I evidence base (~2M lines of completed CHTC campaign results)** on `claude/paper-i-consolidated-20260826` / `ra-refactor-stage2`, and the **Paper II SIAM r00–r06 results + a newer (08-27) Paper II manuscript revision** on `codex/paper-ii-search-profiles-20260824`.
- **Security:** old `main` has a committed `.ibm_runtime.env` with a **filled-in IBM Quantum API token + instance CRN**. It is in GitHub history and was on the lost machine. **Revoke/rotate that token at IBM Quantum / IBM Cloud.** (File does not exist on the trunk; the history keeps it regardless.)
- **Time-sensitive:** the `infidelity-gate-shadow` branch records CHTC submit-node archive paths (e.g. `/home/jsstrobel/Holstein_paper_ii_staged_accuracy_euler_v1_resume_fix_20260826/paper_ii_t20_resume_fix_cluster9698264_12.tar.gz`). Raw run archives were only fetched to the lost machine — the copies on the CHTC submit node may still exist **until the cluster purges them**. Fetch soon if wanted.

## 2. How the repo got here (timeline)

| Era | What happened |
|---|---|
| Feb 2026 | Original `Fermi-Hamil-JW-VQE-TROTTER-PIPELINE` architecture; replaced by the new `pipelines/` architecture on 2026-02-28. `main` continues to 2026-03-25, then freezes. |
| Mar–Jun 2026 | "Snake" era: realtime time-dynamics routes, static-ADAPT registry, Route C plateau work on `Snake` → `route-c-plateau-acquisition-v1` (+ safety diff snapshots 06-29). |
| Jul 2026 | SR-SNAKE v4 era in a separate July worktree (unrelated-history root `476dbbdc`): `codex/sr-*`, `copilot/worktree-*`, formal-manifold experiments. |
| 2026-08-07 | Trunk ingests the Snake worktree ("Replace Snake snapshot with RA") — legacy code content reaches the trunk. |
| 2026-08-18 | Fork point `9f55e696`: the Paper I campaign lane (`ra-refactor-stage2`, `appendix-demos`, `noise-nonpositive-shortlist`, `paper-i-factual-audit`, `paper-i-consolidated`) splits off while the trunk pursues Paper II/III. |
| 2026-08-24 | Fork point `260e1531`: the Paper II campaign lane (`exchange-layer-cleanup` → `infidelity-gate-shadow`, and `paper-ii-search-profiles`) splits off. Trunk does a partial "golden rescue" of Paper I evidence. |
| 2026-08-25–28 | Agents seal retiring worktrees with "preserve … state" tip commits across many branches. |
| 2026-08-29–30 | Final activity: `search-profiles` seals HH APM preflight data (08-29); trunk records SIAM nb4 observables, best-verified Paper II PDF snapshot, scheduler note (08-30). |

## 3. Security finding

`origin/main:.ibm_runtime.env` contains filled `QISKIT_IBM_TOKEN`, `QISKIT_IBM_CHANNEL`, `QISKIT_IBM_INSTANCE` (CRN). Present in pushed GitHub history; was present on the lost machine. **Action (user): revoke/rotate the token.** Optional later: history scrub (destructive rewrite — separate decision, not needed if the token is revoked).

## 4. Promotion plan (approved by user, not yet executed)

1. Preserve old main: create branch `archive/main-pre-recovery` at current `origin/main` tip (`50474137`). Old main holds trunk-required modules and sole-copy artifacts — see §5a last row and §6.1.
2. Point `main` at `origin/paper-ii-exchange-selector` tip (`cde67194`) and push (non-fast-forward update of `main`; GitHub default branch stays `main`).
3. Delete nothing (user decision: keep all branches for now).
4. Then execute the rescue merges/cherry-picks below onto the new `main`.

## 5. Branch dispositions (audit verdicts)

### 5a. Rescue into trunk — highest value first

| Branch | Unique | What it uniquely holds | How to take it |
|---|---|---|---|
| `claude/paper-i-consolidated-20260826` | 190 | **Sole holder of the finished Paper I evidence base** (~2.05M lines): CHTC Bundles 3–9, B3-MR, B9 costexp, L=3 appendix v2–v10, noise + Powell-tolerance results, newest `kstar_tables.json` (22,640 lines vs trunk's stale golden 19,649), 6,215-line failure atlas, bundle-page LaTeX tooling, PDF_INVARIANT.md | **Graft the add-only paths** (all `chtc/paper_i_*`, `chtc/ra_adapt_*`, results_tooling, agent_guidance docs) — zero conflicts. Do NOT merge its code: its static_adapt refactor and the trunk's are two incompatible refactors (14 conflicts) → user decision §6. |
| `ra-refactor-stage2` | 131 | Same campaign lane, complementary: 668 files of CHTC checkpoint/history receipts whose upstream raw data was deliberately deleted (27GB local + 4.3GB CHTC freed) — only surviving copy; B3-supersedes-B5 ruling; stage-2/3 ra_adapt refactor | **Merge** (audit verdict): only 8 conflicts with clear per-file resolution — take TRUNK for monolith/CLI/integration-test files, take BRANCH for `semantic_closure_routes.py` (+852 lines powering B8/B9). Regenerate parity fixtures; run tests after. |
| `codex/paper-ii-search-profiles-20260824` | 156 | **Sole copy** of SIAM r00–r06 evaluated scores/receipts (clusters 9713618–9715147), most HH/Hubbard acquisition evidence, 263k-line Hubbard L4 exact-reference dataset, staged-campaign/acquisition-supervisor pipeline + ~104-file test suite, **08-27 Paper II manuscript revision (newer than trunk's)** | **Path-level cherry-picks**: branch-only `output/` evidence trees + pipeline modules/tests (clean); manual reconciliation of `time_dynamics_paper_II.tex` (§6). Trunk is authoritative for the 32 conflicting files (evolved exchange core, newer SIAM state JSONs). Preserve the ref — trunk manifests cite its commit hashes as provenance. |
| `codex/paper-ii-infidelity-gate-shadow-20260826` | 62 | Staged Paper II accuracy-campaign stack; CHTC t20/t5 receipts (5 clusters); **provisional t20 frontier numbers** (12 rows in `t20_provisional_author_override_receipt.json` + generated table + figure) — sole in-git record of that compute | **Cherry-pick** the results/provenance set (conflict-free adds) + optionally the staged-campaign modules. Do not merge wholesale (20 conflicts vs live trunk). Fetch CHTC submit-node archives before purge (§1). |
| `appendix-demos-20260820` | 117 | Tip-only (+10k lines): 08-23–25 L=3/L=4 campaign dirs (`l3_run_inventory.json`, `l3_pareto_frontiers.json`), prune-sweep results, and `marginal_qiskit_cost.py` — which the trunk's own golden doc names as an uncommitted manuscript-results dependency | **Merge its tip into `claude/paper-i-consolidated-20260826`** (verified conflict-free), then take everything via the consolidated graft. Not into trunk directly (14 conflicts). |
| `codex/noise-nonpositive-shortlist-20260824` | 117 | Tip-only (+1.8k lines): two L=3 noise campaign dirs incl. `RESULTS_SUMMARY.md` with the **measured noise-sweep numbers flagged as the manuscript noise-figure provenance**; nonpositive phase-shortlist feature + test | **Cherry-pick** the 2 chtc noise dirs + shortlist module/tests after the appendix merge (6 overlapping pipeline files to skip/resolve). |
| `codex/paper-i-factual-audit-20260822` | 99 | Top 3 commits only: 2,039-line Paper I factual/mathematical/QPU provenance audit — includes a **pre-release blocker** (Figure-2 provenance re-bound from authenticated cluster-9664325 evidence to the later b3mr family; "must be corrected before release") + AVQE estimator-count verification handoff | **Cherry-pick `0db00282`, `77c39791`, `83ccbc40`** — conflict-free, doc-only. Rest is shared history superseded by the consolidated branch. |
| `paper-i-l4-batch-debug-20260826` | 7 | Complete source-locked Hubbard L4 five-arm campaign code + tests (~2k lines); no data; trunk has none of it | **Merge — textually clean.** Run touched tests after. (Contains all of `paper-i-l4-five-arm`.) |
| `codex/paper-ii-positioned-tangent-audit-20260824` | 4 | 119-line positioned-insertion audit report + 134 lines of verification tests backing Paper II manuscript claims | **Merge — clean.** Tests target the 08-24 API; fix signatures if they break. |
| `claude/exchange-layer-cleanup-20260825` | 26 | Only tip `788efa48`: golden-run-parity-verified exchange-config/round-rule refactor (absent from trunk and sibling) | **Port tip if wanted** (generalized_exchange.py near-clean; 2 files need manual porting). Else keep as archive. |
| `codex/static-adapt-beam-refactor` | 182 | Terminal tip of the whole Feb–Jul static-adapt chain (7 other branches are interior ancestors). Unique: **PHYS 765 final report** (tex + PDF) + its 79 result files, 33 Apr-2026 notes, early Overleaf manuscript uploads | **Cherry-pick the named files** (e.g. into `archive/`); keep the ref as the archive of the whole chain. Never merge (247 conflicts). |
| **`main` (old)** | 13 | **The trunk's live code depends on it**: `pipelines/scaffold/imported_artifact_resolution.py` hardcodes artifact JSONs (nighthawk/pareto-lean/marrakesh) that exist only here, and trunk's `hh_staged_noise_workflow.py` imports 4 modules (`hh_staged_workflow`, `hh_realtime_checkpoint_{controller,types}`, `hh_realtime_measurement`) that exist **only on old main** — trunk tests currently `importorskip` and silently skip. Also: 1,978 tracked artifact files (~30M lines) of HH L2/L3 noisy-run outputs — sole surviving copies (trunk gitignores `artifacts/`) — and the CFQM time-propagation package (`src/quantum/time_propagation/` + tests), which has no counterpart anywhere in trunk. | **Cherry-pick**: (a) the 9 main-only `pipelines/hardcoded/` modules to un-break the staged-noise lane, (b) the curated `artifacts/json` inputs the trunk resolver names (needs `git add -f` or a `.gitignore` exception — policy decision), (c) the CFQM package if that line matters. Never full-merge (58 conflicts, trunk newer in all). Keep the ref as `archive/main-pre-recovery` per plan — it is the permanent archive of the 30M-line artifact corpus. |

### 5b. Keep as archive (unique content; do not merge, do not delete)

| Branch | Why |
|---|---|
| `codex/paper-i-worklog-audit-20260824` | Sole copy of the Q45 STOP record (ledger fingerprints the later Q53 diagnosis references). Optionally cherry-pick the worklog hunk. |
| `safety/codex-diff-snapshot-20260629T031325Z` | 6.5MB dirty-worktree patch blob (pre overnight run) + keeps the whole 111-commit Snake line reachable. |
| `safety/codex-diff-snapshot-20260629T040640Z` | Sibling 4.2MB patch + **3,145-line inventory of untracked/gitignored files that lived on the lost machine's disk** — useful map of what was lost. |
| `copilot/worktree-2026-07-23T21-51-12` | 83 files that exist nowhere else: formal-manifold pipeline family + tests, sr_snake campaign scripts, Paper I provenance exports. Also preserves the SR v4 chain. |
| `codex/sr-v4-phase3-lowgrad-repair-20260717` | Only ref holding `b2400825` (SR full-response phase-retirement fix). SR lane retired, but the fix exists nowhere else. |
| `codex/fm-sr-snake-20260714` | Only record of the qbroyd 20260714 CHTC campaign submission (build/submit scripts + matrix audit/manifest JSONs). |
| `codex/formal-manifold-snake-20260711` | Complementary seal (only 77/193 files overlap with the sibling): jr_snake canonical runtime settings + larger qbroyd audit. |
| `codex/wip-local-active-source-snapshot-20260706-120000` | Sole copy of IDEAS.md, IMPLEMENT_NEXT.md, 2,352-line exact-citations ledger, Nocedal trust-region derivation notes, 13 plan critiques. Easy to hand-copy into trunk later if wanted. |
| `backup/pre-delete-20260221_213720` | 157 files found nowhere else: earliest HH architecture/plan docs, L2_hh_smart result data, Obsidian vault. The designated pre-cleanup backup. |
| `snapshot/fullspace-20260306T031421` | Sole holder of 144 early-March HH L2/L3 noise-robustness/validation result files (plus 9,712 committed `.venv` junk files — a future history-filter candidate, destructive, user's call). |

### 5c. Needs your decision (§6)

| Branch | Decision |
|---|---|
| `codex/paper-i-q45-diagnosis-20260825` (16) | The corrected, *completed* Q45 implementation (global shortlists, shared cost oracle, Aer fail-closed, −4,300 lines) never landed on trunk, and trunk's worklog still says Q45 "in progress". Trunk rewrote the same files, so it must be **re-applied by hand** using the branch as reference — is Q45 still wanted on the current architecture? At minimum take the worklog closure (merges clean). |

### 5d. Safe to drop (fully contained elsewhere — kept anyway per your instruction)

`paper-i-l4-five-arm-20260826` (⊂ l4-batch-debug) · `codex/adaptive-qse-20260826` (engine ported to trunk and evolved to schema v2; **correction 2026-08-31**: its 14-line reporting arm in `build_paper_iii_campaign_report.py` had NOT been ported — cherry-picked to main as `65e3f044`, branch now truly fully absorbed) · `codex/paper-ii-gamma-symbol-definitions-20260823` (concept absorbed in newer trunk prose) · `codex/remove-step-overlap-20260824` (deleted code already absent from trunk) · `codex/adapt-generator-geometry-speedups-20260613`, `feature/static-adapt-problem-registry-phase1`, `Refactor`, `checkpoint/pre-generic-adapt-refactor-2026-04-17`, `<Best-ADAPT-EVER>`, `<Janitorial-Duties>` (all interior ancestors of `static-adapt-beam-refactor`; the last two are the *same commit*) · `Simplied-Refactored` (ancestor of trunk, 0 unique) · `Snake`, `route-c-plateau-acquisition-v1`, `backup/non-artifact-code-docs-20260328` (⊂ safety snapshots) · `codex/sr-snake-v3-foundation-20260716` (manuscript-editor skill exists newer on trunk), `codex/sr-v4-rankgate-repair-20260717`, `copilot/worktree-2026-07-23T21-51-02`, `codex/local-repo-health-20260706` (⊂ kept SR refs) · `codex/minimize-absolute-error-for-l=2`, `…-gd6bk8`, `codex/add-l=2,3-to-hh_hva_tw` (target the pipeline deleted 2026-02-28).

Fully merged into trunk (0 unique commits): `RA`, `Simplied-Refactored`, `golden-rescue-20260824`, `codex/paper-ii-route-diagrams`, `codex/hardcoded-test-retirement-20260824`, `perf/sector-audit-memoization`, `claude/paper-i-20260824`, `codex/paper-iii-exchange-manuscript-20260826`.

## 6. Open items

1. **Old `main` rescue scope** (audit complete — see §5a last row): its shared code is fully superseded, but the trunk *silently depends* on 9 modules and several curated artifact JSONs that exist only there, and it is the sole archive of ~30M lines of March HH noise artifacts. Decisions: re-track the curated artifacts in trunk against its `artifacts/` gitignore policy (recommended: yes, `git add -f` just the resolver-named files), and whether to bring the CFQM time-propagation package forward.
2. **Paper II manuscript reconciliation:** trunk's `time_dynamics_paper_II.tex` (08-26, + Paper III comparator fix, matches the 08-29/30 built PDFs) vs `search-profiles`' (08-27, matched-accuracy promotion, −551/+138 vs trunk) vs `infidelity-gate-shadow`'s t20-frontier rewrite. Three divergent versions — user must pick the canonical line; the others become archived variants.
3. **Paper I code lineage:** trunk's post-08-24 decomposition vs the campaign lane's run-axes/costexp/measured-lane refactor (consolidated + ra-refactor-stage2). The audit recommends: data grafts now; code lineage decision before the next Paper I rerun. *(Update 2026-09-01: the EXPRESSIBILITY half is resolved — `ef842070` ports compile_cost_policy additively, zero pin movement. The TRAJECTORY half is now precisely characterized and remains open: insertion-arm divergence is exact-score-tie resolution over disjoint candidate enumerations in the phase-0/I/II position-record pipeline — wholesale-merge territory; `paper-i-integration-20260831` @ `f041a5c1` is the reproduction lineage of record for those rows. See the CORRECTION in `docs/reports/PAPER_I_RERUN_PARITY_RESULTS_20260831.md`.)*
4. **Q45**: see §5c.
5. **Figure-2 provenance pre-release blocker** from the factual audit (§5a) — must be resolved before Paper I release.
6. **CHTC submit-node archives** — fetch before purge (§1).
7. **IBM Quantum token** — revoke (§3).

## 7. Getting it running on this Mac (fresh machine)

Machine currently has ONLY Apple's Python 3.9.6 — unusable here (numpy 2.3.5 needs ≥3.11; two reporting scripts need ≥3.12 f-strings). No Homebrew, no Node, no conda.

1. Install **Python 3.12** (python.org installer or Homebrew).
2. `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install pytest ijson` (pytest and ijson are missing from requirements.txt; `.venv/` is already gitignored).
3. From the repo root (imports require cwd = root): `python -m pytest test --collect-only -q`, then targeted (`python -m pytest test/test_ap_mclachlan_exchange_selector.py -q`) before the full suite — it contains heavy integration tests; respect the documented 10 GB RAM budget (`pipelines/shell/ram_guard.py`).
4. Smoke runs: `python pipelines/exact_bench/cross_check_suite.py --problem hh --L 2 --omega0 1.0 --g-ep 0.5 --n-ph-max 1` (no artifacts needed), then `python pipelines/static_adapt/adapt_pipeline.py --L 2 --problem hh … --skip-pdf --output-json artifacts/json/smoke.json`. Paper II dynamics runners need an ADAPT artifact JSON produced first — all pre-existing `artifacts/` were on the lost machine.

## 8. Repo-local agent tooling status

- **`CLAUDE.md`** imports **`AGENTS.md`** (the real router: 5 paper lanes under `agent_guidance/`, 10 GB RAM invariant, per-agent worktrees). **Drift:** AGENTS.md "Active checkout" points at the lost machine (`/Users/jakestrobel/local_repos/Holstein_test_fullclone_3`) — needs updating to this clone. README references several retired scripts (list in audit data; the legacy pipeline survives only as archived `.txt` and in `chtc/*/runtime_source/`).
- **`.claude/skills/gitnexus/*` (6 skills):** intact, but need Node.js + `npx gitnexus analyze` (rebuilds the gitignored `.gitnexus/` index) + the GitNexus MCP server registered in Claude Code. Until then the "MUST run impact before editing" rules in CLAUDE.md are unsatisfiable — treat as inoperative.
- **`agent_guidance/skills/manuscript-editor/SKILL.md`** exists on the trunk (newer than the July branch copy) — the manuscript-editor skill was NOT lost.
- **Lost with the machine (not in any branch):** user-level skills referenced as gates (`$journal-math-manuscript-refiner`, `$gpt-pro-handoff`, `$pedagogical-math-primer`, paper-specific run skills), all gitignored `artifacts/` evidence trees, `.gitnexus/` index, CHTC `.sif` images. The safety-snapshot branch's untracked-file inventory (§5b) maps what was there.

## 9. Execution log (Stage 1, 2026-08-31)

**Pushed:** `main` → trunk tip `cde67194` (old main preserved at `archive/main-pre-recovery`); new branch `paper-i-integration-20260831` (= consolidated + appendix + noise tips, holds the deferred Paper I code lineage and its tests).

**Built locally on `main`, NOT yet pushed — 13 rescue commits:** the two clean merges (L4 campaign, positioned-tangent audit); factual-audit docs cherry-picks; t20 evidence graft (seed.json duplicates excluded); stage2 + consolidated Paper I evidence grafts (**data only** — the stage2 code merge was attempted, failed test verification 26-vs-7, and was rebuilt as data-only; the code lineage decision of §6.3 stands); Q45 worklog closure; AGENTS.md checkout repoint; and the **full legacy staged-noise lane resurrection**: 9 + 18 modules, `report_labels`, CFQM package, `polaron_paop`, `checkpoint_measurement`, `json_parity`, 5 fixed-manifold modules, 6 curated `artifacts/json` files (force-added), with the July beam-refactor era's alias-shim modernization adopted wholesale.

**Verification** (Python 3.12.14 venv, this Mac): full-suite collection **6,225 tests / 35 errors** vs untouched-trunk baseline 5,540 / 57 (22 pre-existing errors fixed, +685 tests). Resurrected staged-noise suites: **406 pass / 7 fail** (were 100% silently skipped on trunk); the 7 are era drift (driven-neutral seam checks, oracle scheduler semantics). CFQM suite 22/22. cfqm4 is retired per user ruling 2026-08-31 — its tests no longer execute. Targeted rescue suites (L4, exchange selector/oracle/structural-cache): 22/22.

**Continuation (same day, pushed as commits `b1ee21da..e70397c8` + this file):** search-profiles Paper II evidence graft (678 branch-only files, 37MB — SIAM r00–r06 scores, Hubbard L4 exact reference, acquisition evidence, CHTC lineage; trunk-newer state files untouched); 28 sole-copy research docs (IDEAS/IMPLEMENT_NEXT, citations ledger, derivation notes, plan critiques); PHYS 765 report + 71 run files + 33 spring notes (under `archive/`) + early Overleaf upload sets; Q45 STOP record merged into the worklog; **27 retired-but-imported modules restored** (each verified to have live production importers; `plots/` package force-added past the gitignore that lost it) and pytest/ijson/clarabel/cvxpy added to requirements.txt. Final state: **6,519 tests collected / 10 errors** (trunk baseline 5,540 / 57). The 10 residual errors are missing data files and module drift; the 7 era-drift failures in the resurrected staged suites remain documented above. Full-suite *run* (not just collection) deliberately deferred to the new Mac — heavy integration tests vs the 10 GB RAM invariant.

## 10. qdynamics (checked 2026-08-31, user-provided)

`jacopos86/qdynamics` was the collaboration sync target (the March "Sync … to qdynamics" commits). Verdict: **nothing to recover.** Its `RA` (08-26) and `paper-ii-exchange-selector` (08-18) branches are strictly contained in the new Holstein main (0 unique commits; main is 86 and 610 ahead respectively). The other branches are the collaborators' own psi4/pyscf lineage, not Holstein work. If the collaboration repo should see the recovered main, that's a push to someone else's repo — user's call.

## 11. Paper II / III evidence status (traced 2026-08-31)

Full trace: `docs/reports/PAPER_II_III_EVIDENCE_TRACE_20260831.md`. Headlines:

- **Paper II — data substantially saved, provenance exceptional.** The 18-page best-verified results PDF (built 08-30, sha-verified, pages 4–17 the sole surviving copy of the campaign report), the 12-row t20 frontier receipt with per-run digests, the byte-verified t20 table+figure, and all SIAM r00–r07 scores (full time series each) are on `main`. Every number traces to a cluster id, source revision, digest-bound input, and a literal recorded `runner_argv` command line. Lost from git: raw trajectories (`run.json`) — **~53 MB of sha-bound archives on the CHTC submit node, receipts say `remote_raw_state: PRESERVED`** (see `CHTC-FETCH.md`; re-running instead means CHTC-scale campaigns, and the t20 lane cannot re-run as-is because it resumes from vanished t5 screens). Code gap: the 84-module Paper II reporting/scoring/campaign stack + 76 tests live only on `codex/paper-ii-search-profiles-20260824` (+ sibling). **Manuscript correction to §6.2:** `main`'s tex was the stale draft (331 `\tentative` markers); the *promoted* 08-27 version (4 markers) was the completed one. **Executed 2026-08-31 (user-approved):** the promoted manuscript was adopted onto `main`, and the 160-module reporting/scoring/campaign stack was restored with per-file reconciliation of the nine shared core files (details in commit `cc1400e4^`). Verification: all `time_dynamics` modules import (one exception needs a generated seed ledger, producer committed); trunk-core suites 62/62; grafted suites 66 pass with 8 data-missing failures on the vanished `output/paper_ii_staged` exact references. Suite: **7,535 tests / 9 errors** (trunk baseline 5,540 / 57). The branches' wider rewrites of `campaign.py` (46 conflicts) and `exchange_integration.py` (9) remain deliberately unmerged.
- **Paper III — the completed paper is the *static* manuscript** (byte-exact vs its locked sha, 0 placeholders); the dynamics manuscript was never completed (209 placeholders — scope fact, not a loss). All raw evidence JSONs and every rendered PDF are gone (gitignored), **but every lost artifact has a tracked producer, compute is local (seconds–minutes, deterministic, never CHTC), and all builders import and run today** — the static chain regenerates end-to-end in ~3 commands (needs LaTeX installed for PDFs). One true loss: the historical nph1 regression baseline — a rerun re-establishes a baseline but cannot detect drift introduced since `4152de3a`. The unported adaptive-qse report arm was cherry-picked (`65e3f044`).

## 12. Replacement-machine bootstrap (M4 Pro, executed 2026-08-31)

- **Machine:** fresh Apple M4 Pro (14-core CPU, 20-core GPU, 24 GB RAM, 1 TB
  SSD, macOS 26.6.2), user `lordskyler`. NOT migrated — nothing carried over;
  §7's environment description is the previous recovery Mac, not this one.
- **Toolchain** (all user-local under `~/.local`, no sudo, no Homebrew):
  gh 2.98.0 (HTTPS auth as `20jastrobel`, git credential helper wired),
  node 24.20.0, uv + CPython 3.12.14 venv from `requirements.txt`
  (17 pins installed clean), gitnexus 1.6.10 global.
- **Clone:** `/Users/lordskyler/Holstein_test` over HTTPS; HEAD `29f924bc`
  == `origin/main`; 54 remote branches present; rescue commits verified
  reachable (`cde67194`, `cc1400e4`, `65e3f044`, `b1ee21da..e70397c8`).
  `~/ADAPT---Paper-I` manuscript repo re-cloned alongside. Sources verified
  not APFS `compressed,dataless` (0 hits).
- **Verification vs recorded baselines:** collection **7,535 tests / 9
  errors** — exact match to §11. `test_ra_adapt_semantic_closure_routes.py`
  **7 failed / 85 passed** — exact match. Staged-noise smoke files 70/70.
- **Staged-suite re-measure + classification** (all 9 staged files, 27
  failures, diagnosed by parallel agents): **9 data-missing** (vanished
  `output/paper_ii_staged` exact references — §11 family; recoverable only
  via the CHTC fetch or regeneration), **16 regression** — one root cause:
  trunk `paper_ii_runs.py:build_run` lacks the `search_profile` kwarg that
  the wholesale-restored `staged_campaign.py:1699` passes (incomplete
  `e0e6d6cd` reconciliation; escaped that commit's own verification note),
  **1 doc/test mismatch** (`agent_guidance/skills/paper-ii-run/SKILL.md`
  never received branch-only commit `e2c96ba5`'s response-normalized
  wording — user decision pending: graft the doc or align the test),
  **1 env** (no LaTeX engine installed on this machine). The 7 era-drift
  failures did not reproduce **in the two smoke files**, but the family is
  NOT gone: the full-suite classification (below) finds 12 era-drift tests
  red elsewhere in the suite.
- **GitNexus rebuilt:** 103,520 nodes / 213,084 edges / 722 flows at
  `29f924b`; MCP server registered (`gitnexus setup -c claude`); skill
  layout migrated to the 1.6.10 flat scheme (`.claude/skills/<name>/`).
  Fresh-macOS fix recorded: `lbugjs.node` dynamically links
  `@rpath/libssl.3.dylib` with only package-manager rpaths — built OpenSSL
  3.6.4 libs into `~/.local/openssl3/lib`, `install_name_tool -add_rpath`
  + ad-hoc re-sign of both `lbugjs.node` copies (redo after any gitnexus
  reinstall/update).
- **Full-suite run** (first ever since recovery; this IS the baseline —
  no prior end-to-end result exists): `ram_guard --limit-mb 12000 --
  .venv/bin/python -m pytest test -q --continue-on-collection-errors` →
  **6,751 passed / 683 failed / 85 errors / 25 skipped** in 26:15, peak
  RSS 5,447 MB. (First attempt without
  `--continue-on-collection-errors` aborts at the 9 known collection
  errors — the flag is required.) Failure set: 768 failed/error ids
  across 166 files, recorded in
  `docs/reports/FULL_SUITE_BASELINE_20260831_failed.txt`; heavily
  clustered in CHTC-package/campaign-fixture families
  (test_adapt_vqe_integration 75, test_siam_chtc_package 52,
  test_generic_static_benchmark 36, prospective_* 61, test/chtc/* ~120).
  Root-cause classification (complete, id-by-id:
  `docs/reports/FULL_SUITE_BASELINE_20260831_classification.md`):
  incomplete-graft 346, data-missing-chtc 221, retired-interface 93,
  unknown 57, data-missing-regenerable 26, env 25. The 2026-08-31
  seam-fix commits `06a258de`/`b319ba9c` are verified CLEAN (pre/post
  worktree A/Bs). The one genuine recovery-day regression is merge
  `e5692712` (l4-batch-debug): the batch executor's no-phase0 path lacks
  the `phase1_event_start` correction its siblings have
  (`adapt_pipeline.py:2509/2744` vs `:3182`) — turned 9 green tests red
  (21 total in the family); §5a's "run touched tests after" step was
  never executed for that merge. Correction to the bullet above is
  incorporated per the classification (era-drift reproduces: 12 tests).

---
*Full per-branch agent reports: workflow `wf_3badde6b-c1a`; paper evidence trace: `wf_310612e2-038`. *

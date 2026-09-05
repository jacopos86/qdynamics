# HANDOFF — Paper II CHTC campaign: Hubbard–Holstein regimes and SIAM

**Executor:** Codex · **Written:** 2026-09-05 · **Contract:** `agent_guidance/shared/agent-handoff-contract.md`
**Branch to work on:** `fable/paper-ii-regime-survey-20260905`
(worktree `/Users/lordskyler/Holstein_test.worktrees/fable-paper-ii-regime-survey-20260905`)
for everything Hubbard–Holstein; `fable/siam-search-profiles-20260905`
(worktree `/Users/lordskyler/Holstein_test.worktrees/siam-search-profiles-20260905`)
for everything SIAM. Python is always `/Users/lordskyler/Holstein_test/.venv/bin/python`
with `PYTHONPATH=.` from the worktree root.

## 1. What this campaign produces

Paper II compares the APM route (the paper's realized Γ(θ̇*) selector, arm
`exchange_q_compiled`) against AVQDS. Today its evidence is one Hamiltonian
point. This campaign adds three more Hubbard–Holstein points and closes the
SIAM (single-impurity Anderson) model.

### 1a. Hubbard–Holstein (two sites, half filling, t = 1, ω₀ = 1)

| name | U | g | n_ph^max | qubits | status |
|---|---:|---:|---:|---:|---|
| `anchor_u4_g0p5_nph1` | 4.00 | 0.500 | 1 | 6 | the published point, rebuilt with the same seed recipe as the new rows |
| `weak_weak` | 0.25 | 0.354 | 3 | 8 | new |
| `intermediate_weak` | 1.25 | 0.354 | 3 | 8 | new |
| `strong_weak_u8` | 8.00 | 0.354 | 3 | 8 | new |

Each point goes through three steps:

1. **Seed** (CHTC, minutes): the fixed-ansatz ground state at t = 0, built by
   the frozen three-layer VQE with the hard gate |E − E₀| ≤ 1e-6. Output: one
   `seed.json` per row.
2. **Response scales** (local, ~30 s): the six exact driven trajectories of
   that seed, which define the accuracy target for step 3.
3. **Dynamics** (CHTC, hours to ~2 days per job): APM vs AVQDS, six drives ×
   three step controllers = 36 cells per point, run as a t = 0→5 screen and a
   t = 5→20 continuation, qualified under the same matched-pair accuracy
   policy as the published cells.

### 1b. SIAM

**SIAM is not yet on the current method.** Everything SIAM that exists,
including the two results below, ran on the older exchange scoring rule
(historical pilot lineage `origin/codex/paper-ii-search-profiles-20260824`),
not on the compiled-Q route (`exchange_q_compiled`), and without the
exact-estimator accounting the Hubbard–Holstein cells carry. Those results
are historical pilots. The SIAM comparison counts as complete only when both
arms have been run on the survey branch's compiled-Q route with the
estimator accounting, Nb = 2 locally and Nb = 4 on CHTC. Section 8 holds the
work needed to get there; it is being built now and this section will be
replaced when it lands.

| system | qubits | historical pilot state (old scoring rule) |
|---|---:|---|
| Nb = 2 proof pair | 6 | exchange burden 0.3564, AVQDS 0.3525, matched pair, ratio 1.011 (commit `e073fe48`, SIAM branch) |
| Nb = 4 pilot | 10 | exchange rung r07 to t = 5 qualified, burden 0.0554 (commit `8edc93a3`); AVQDS t = 5 at cut 1e-3 outside the broad guard |

## 2. What the user does (nothing else is needed from them)

1. The SSH master to `ap2001` is already up. Only when Codex reports it
   expired (it lasts 8 h): GlobalProtect on, then one Duo push,
   `ssh -MNf ap2001` (verify with `ssh -O check ap2001` → `Master running`).
2. Say **"submit"** when Codex reports a package is ready. Codex never submits
   without it.

## 3. Setup (Codex, once)

```bash
cd /Users/lordskyler/Holstein_test.worktrees/fable-paper-ii-regime-survey-20260905
git log --oneline -1                              # expect this branch's head
ls -la chtc/time_dynamics_optuna/image.sif        # 216,371,200 bytes; sha256 fa5c4ea8…
PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest test/test_paper_ii_runs.py test/test_paper_ii_staged_accuracy_campaign.py test/test_build_hh_fixed_layer_seed.py test/test_build_regime_response_scales.py test/test_compiled_ansatz.py test/test_paper_ii_regime_seed_install.py -q -p no:cacheprovider
```
Expected: 139 passed on the last line. Then stage the bundle remotely:
```bash
ssh ap2001 'quota -s | tail -3; mkdir -p Holstein_paper_ii_regime_seeds_v1_20260905/{logs,raw_outputs}'
rsync -azR --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude 'output' --exclude 'raw_outputs' --exclude 'logs' --exclude 'artifacts' --exclude 'chtc/time_dynamics_optuna/image.sif' \
  ./pipelines ./src ./chtc/paper_ii_regime_seeds_v1 ./chtc/time_dynamics_optuna ./chtc/validate_condor_submit_lifecycle.py \
  ap2001:Holstein_paper_ii_regime_seeds_v1_20260905/
# -R keeps the chtc/ prefix on the remote side; the submit file and scripts address chtc/... paths.
ssh ap2001 'for p in Holstein_siam_nb4_source_ba50339c_20260828 Holstein_siam_nb4_frontier_after_9712800_ebdf7ca5_20260827; do f=$HOME/$p/chtc/time_dynamics_optuna/image.sif; [ -f "$f" ] && { echo "$f"; sha256sum "$f"; break; }; done'
```
If a remote copy prints sha `fa5c4ea8…`, symlink it; otherwise rsync the
local image (216 MB):
```bash
ssh ap2001 'ln -sfn <that path> Holstein_paper_ii_regime_seeds_v1_20260905/chtc/time_dynamics_optuna/image.sif'
ssh ap2001 'cd Holstein_paper_ii_regime_seeds_v1_20260905 && sha256sum chtc/time_dynamics_optuna/image.sif && python3 chtc/validate_condor_submit_lifecycle.py chtc/paper_ii_regime_seeds_v1/submit_regime_seeds_v1.sub && echo LIFECYCLE_OK && condor_submit -dry-run - chtc/paper_ii_regime_seeds_v1/submit_regime_seeds_v1.sub | grep -c "^Iwd"'
```
Expected: the sha, `LIFECYCLE_OK`, and `4`. Stop and report if quota headroom
is under 2 GB or the count is not 4.

## 4. Step 1 — seeds (CHTC)

Submit after the user says "submit":
```bash
ssh ap2001 'cd Holstein_paper_ii_regime_seeds_v1_20260905 && condor_submit chtc/paper_ii_regime_seeds_v1/submit_regime_seeds_v1.sub'
```
Write `chtc/paper_ii_regime_seeds_v1/submission_receipt_<cluster>.json`
(cluster id, UTC time, remote dir, image sha) and commit it.

| proc | row | request | expected wall | expected memory |
|---:|---|---|---|---|
| 0 | anchor | 1 CPU, 2 GB | < 2 min | < 200 MB |
| 1–3 | nph = 3 rows | 1 CPU, 4 GB | 10–40 min | < 1 GB |

Monitor with the loop in §7. When a proc finishes, fetch and install:
```bash
mkdir -p ~/chtc_fetch/paper_ii_regime_seeds_v1_20260905
rsync -az ap2001:Holstein_paper_ii_regime_seeds_v1_20260905/raw_outputs ap2001:Holstein_paper_ii_regime_seeds_v1_20260905/logs ~/chtc_fetch/paper_ii_regime_seeds_v1_20260905/
PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python chtc/paper_ii_regime_seeds_v1/install_seeds.py --fetched-root ~/chtc_fetch/paper_ii_regime_seeds_v1_20260905
git add chtc/paper_ii_regime_seeds_v1/input/seeds && git commit -m "paper ii: install stage-1 regime seeds from cluster <cluster>"
```
Expected: `problems` empty and one file per returned row under
`chtc/paper_ii_regime_seeds_v1/input/seeds/`. A receipt with
`status: unqualified` is a result, not a failure to retry: report its ΔE.

## 5. Step 2 — response scales (local, per installed row R)

```bash
R=anchor_u4_g0p5_nph1
PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python pipelines/shell/ram_guard.py --limit-mb 8000 -- \
  /Users/lordskyler/Holstein_test/.venv/bin/python -m pipelines.time_dynamics.diagnostics.build_regime_response_scales --regime $R
git add pipelines/time_dynamics/campaigns/paper_ii_exact_response_scales_$R.json pipelines/time_dynamics/campaigns/paper_ii_exact_response_scales_by_regime.json
git commit -m "paper ii: exact response scales for regime $R"
```
Expected: JSON summary with six drives × {t5, t20}, ~30 s at 6 qubits,
minutes at 8. Any `ResponseScaleIntegrityError` → stop and report.

## 6. Step 3 — dynamics (CHTC, per row R)

Prepare the t = 0→5 screen (36 jobs):
```bash
R=anchor_u4_g0p5_nph1; D=$(date +%Y%m%d)
PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python -m pipelines.time_dynamics.campaigns.paper_ii_staged_accuracy_v1 \
  prepare-chtc --ledger output/paper_ii_regime_survey/$R/ledger.json --regime $R --apm-arm exchange_q_compiled \
  --package-dir chtc/paper_ii_regime_survey_v1_${R}_screen_t5_$D
/Users/lordskyler/Holstein_test/.venv/bin/python chtc/validate_condor_submit_lifecycle.py chtc/paper_ii_regime_survey_v1_${R}_screen_t5_$D/*.sub
git add chtc/paper_ii_regime_survey_v1_${R}_screen_t5_$D && git commit -m "paper ii: survey screen package for $R"
```
Expected: `"status": "PREPARED_NOT_SUBMITTED"`, 36 action ids, every APM
action's `runner_argv` containing `--exchange-route-bundle paper_ii_exchange_q_compiled_v1`.
Commit before preparing (a dirty tree stamps `-dirty`). Upload with the §3
rsync pattern into `Holstein_paper_ii_regime_survey_v1_<R>_<D>/`, submit
after "submit", monitor per §7, fetch to
`output/paper_ii_regime_survey/$R/fetched/cluster_<id>/`, ingest:
```bash
PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python -m pipelines.time_dynamics.campaigns.paper_ii_staged_accuracy_v1 \
  record-package --ledger output/paper_ii_regime_survey/$R/ledger.json --regime $R --apm-arm exchange_q_compiled \
  --actions-jsonl chtc/paper_ii_regime_survey_v1_${R}_screen_t5_$D/input/actions.jsonl \
  --artifact-root output/paper_ii_regime_survey/$R/fetched/cluster_<id>/raw_outputs \
  --seed-json chtc/paper_ii_regime_seeds_v1/input/seeds/$R.json \
  --reference-store-root output/paper_ii_regime_survey_$R/exact_references
```
Then `prepare-chtc` again with the same flags and a `..._promote_t20_$D`
package dir: the ledger emits the t = 5→20 continuations (376 points each,
resuming from the fetched screens). Same upload → "submit" → monitor → fetch
→ `record-package`. R is complete when `next` prints `stage: complete`.

Package resources are 2 CPU / 10 GB / 40 GB per job; the APM route compiles
patches, so submit with `request_cpus = 4` if the package exposes
`--compile-workers` (pass 3); otherwise leave the descriptor as generated.
Expected wall per job: screens 1–8 h; continuations 20–50 h for the 8-qubit
rows (the 6-qubit anchor is a few hours). Order: anchor, then the three
8-qubit rows.

## 7. Monitoring and repair (Codex, unattended between Duo pushes)

Heartbeat: 15 min after each submission, then every 2 h while seed jobs run
and every 6 h while dynamics jobs run. Each heartbeat is exactly one queue
query and one history query (never poll):
```bash
ssh -O check ap2001 || { echo "SSH master expired: ask the user to run ssh -MNf ap2001"; exit 0; }
C=<cluster>; ATTRS='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,HoldReasonCode,JobBatchName,RequestMemory,MemoryUsage,RemoteWallClockTime,EnteredCurrentStatus'
ssh ap2001 "condor_q $C -json -attributes '$ATTRS'"        > output/chtc_monitor/${C}_queue_$(date -u +%Y%m%dT%H%M%S).json
ssh ap2001 "condor_history $C -json -attributes '$ATTRS'"  > output/chtc_monitor/${C}_history_$(date -u +%Y%m%dT%H%M%S).json
```
Decide per proc from those two files:

| observation | action |
|---|---|
| finished, `ExitCode 0` | fetch, verify, install/ingest (§4, §6) |
| finished, `ExitCode 3` on a seed job | unqualified seed: record ΔE from its receipt, no retry |
| finished, other nonzero exit | fetch logs, report; no retry without the user |
| held, `HoldReasonCode 34` (memory) on a **seed** job | resubmit that one row once with `request_memory` doubled (edit only `input/regime_queue.txt`, same everything else), record a `resource_retry_<cluster>_<proc>.json` receipt; a second memory hold → report |
| held, `HoldReasonCode 34` on a **dynamics** job | report with `MemoryUsage`; the user decides |
| held, input-transfer failure (`HoldReasonCode 12`/`13`) | fix the named path (usually the image symlink) and resubmit once; second time → report |
| held, `HoldReasonCode 21` (72 h) | unrecoverable (no stage-out); report; never release |
| running past 48 h | report so the user can decide before 72 h |
| anything else held or idle > 24 h | report |

Never `condor_release`, never `condor_rm`, never change a scientific flag,
cut, seed, or grid between attempts. Every repair is a resubmission of an
identical job with only a resource request changed, and it gets a receipt.
Fetch before any other action on a finished job; never delete remote files.

## 8. SIAM — historical pilot lineage (superseded; kept until the compiled-Q SIAM lane lands)

Worktree `/Users/lordskyler/Holstein_test.worktrees/siam-search-profiles-20260905`
(branch `fable/siam-search-profiles-20260905`, base `d5590449` (`d55904498125ca087ee8541772410f66c50a92b7`) on
`origin/codex/paper-ii-search-profiles-20260824`, the only lineage whose
runner output the SIAM scorer accepts and the lineage all committed r00–r07
scores came from).

Done: Nb = 2 proof pair (`chtc/siam_nb2_proof_v1/`, commit `e073fe48`) and
the Nb = 4 exchange r07 t = 5 score (`output/siam_screening/nb4_pilot_v1/attempts/cluster_9716077/`,
commit `8edc93a3`).

Do not run new SIAM jobs from this section: they would extend the old
scoring rule, not the current method. The steps below stay only as the
record of how the pilot lineage is operated.

1. Rescore the committed AVQDS t = 5 run (cluster 9714524) on this machine so
   both arms share one exact-reference platform, then classify the pair.
   The raw run is under `~/chtc_rescue/home/Holstein_siam_nb4_source_446b75f3_20260828/`
   (find `*9714524*/run.json` or the `continue_*avqds*cut1em03` raw_outputs);
   stitch it with its screen (`stitch_siam_nb4_segments`, seed
   `chtc/siam_nb4_full_apm_screen_r00_sigma_v2_epoch_ba50339c_source_74da5166_20260828/input/seed.json`),
   `score_siam_raw_run` with `reference_store_root=output/siam_screening/nb4_pilot_v1/exact_references`,
   then `classify_siam_matched_pair(avqds_score, exchange_score)`. Expected:
   AVQDS still outside the broad guard (burden ≈ 1.8) → pair unmatched.
2. Prepare the AVQDS r03 (cut 3.33e-4) screen package and, after it returns,
   its continuation package; each is one `queue 1` job:
   ```bash
   PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python -c "
   from pipelines.time_dynamics.siam_chtc_package import prepare_siam_nb4_chtc_action as p
   r=p(repo_root='.', package_dir='chtc/siam_nb4_avqds_screen_r03_cut3em04_source_d5590449_$(date +%Y%m%d)',
       seed_path='chtc/siam_nb4_full_apm_screen_r00_sigma_v2_epoch_ba50339c_source_74da5166_20260828/input/seed.json',
       image_path='chtc/time_dynamics_optuna/image.sif', source_revision='d55904498125ca087ee8541772410f66c50a92b7',
       action_id='siam-nb4-avqds-screen-r03-cut3em04', activation_cut=0.000333333333333, arm='avqds',
       segment='screen_t0_t1_101', request_cpus=2, request_memory_gb=2, request_disk_gb=40, max_runtime_seconds=43200)
   print(r.package_dir, r.action_sha256)"
   ```
   The continuation call is the same with `segment='continue_t1_t5_401'`,
   `resume_from_run_json=<the fetched screen run.json>`,
   `resume_action_receipt_json=<its action_receipt.json>`, and
   `max_runtime_seconds=172800`. Expected wall: screen ~10 min, continuation
   ~1 h 20 m, memory < 250 MB. Upload, "submit", monitor per §7, fetch,
   stitch, score, classify. If the pair is then matched, the Nb = 4 result is
   complete; if AVQDS is still outside the guard, the next rung is r04
   (cut 1e-4), same procedure.

The SIAM packager requires a clean tree (`git diff --quiet <rev> -- pipelines src`
and no untracked files under `pipelines/` or `src/`); commit before packaging.

## 9. Report back

Append to this file under `## Log`, one block per step:
```markdown
## <step> — <date>
Status: done | blocked
Commands run: <exact>
Result: <cluster ids, sha256s, ΔE, burdens, classifications>
Deviations: <anything done differently, and why>
```

## Appendix — evidence and measurements

- Seeds branch history: `fable/paper-ii-regime-seeds-chtc-20260904` (commits
  `48336a01` code, `910d8738` first handoff, `716192f1` response scales and
  `--apm-arm`, `2968d60f` four-row trim); the survey branch is those commits
  cherry-picked onto `codex/paper-ii-exact-estimator-telemetry-20260903`
  (`c3ab7cf1`), which carries the paper's `exchange_q_compiled` arm.
- Why the earlier bundle could not have produced seeds: cluster 9674800
  (2026-08-22) returned no artifact from any row (its builder required a
  driven-snapshot fit at 1e-6 that never got below 5.5e-4) and its 10-qubit
  rows were held at the 24 GB memory limit.
- Local measurements on this branch: anchor seed ΔE 1.7e-11 in 8 s (87 MB);
  `intermediate_weak` seed ΔE 6.2e-8 in 190 s (664 MB); anchor response
  scales 22 s (75 MB), equal to the published seed's to four digits; anchor
  smoke dynamics run in the production runner for both arms.
- Calibration seed convention: its scalar `dv = 0.25` is a uniform site
  potential, a constant −0.5 at half filling; every regime seed is built at
  `dv = 0`; errors and response scales are differences and unaffected.
- Test baseline on the survey branch: 139 passed (six modules above plus the
  installer and staged-campaign files); on the seeds branch 161 passed / 2
  failed, both failures pre-existing on unmodified `main`.
- Image: sha256 `fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f`,
  216,371,200 B, from `~/chtc_rescue/home/Holstein_siam_nb4_source_ba50339c_20260828/chtc/time_dynamics_optuna/image.sif`.
- Remote: `/home` had 4.9 GB headroom on 2026-09-03; no `/staging`; 138 held
  jobs belong to other campaigns and are never touched.

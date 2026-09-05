# HANDOFF — Paper II regime seeds (stage 1) and regime survey (stage 2): monitor-only

**Author:** Claude (Fable 5.1) · **Executor:** Codex · **Date:** 2026-09-04
**Contract:** `agent_guidance/shared/agent-handoff-contract.md`
**Scoping it corrects:** `agent_guidance/time-dynamics/paper-ii-completion-plan.md`,
section "SIAM dimension, regime coverage, and the CHTC handoff (2026-09-04)"

Everything below was verified against source, the CHTC rescue mirror
(`~/chtc_rescue/`), or a local run in this session. Nothing was submitted.

---

## 0. Negative results first — the scoping rested on four false premises

| Scoping claim | Measured fact | Evidence |
|---|---|---|
| The bundle "has never been submitted" | Submitted 2026-08-22 as **cluster 9674800** (6 procs). The three nph=3 procs exited 0 after 13 h 00 m / 15 h 18 m / 25 h 10 m with **zero seed artifacts**; the three nph=7 procs were evicted at 24.3–24.5 GB after ~1 h and were **still held on 2026-08-31** (procs 3–5, `run_seed_apptainer.sh weak_strong / intermediate_strong / strong_strong_u8`). | `~/chtc_rescue/home/Holstein_paper_ii_regime_seeds_v1/logs/paper-ii-regime-seeds.9674800.*.log`; `~/chtc_rescue/scheduler/condor_q_20260831T215057.txt` |
| Resubmitting the bundle would produce seeds | The conditioning-stress builder has produced **0 artifacts in 10 CHTC jobs** (cluster 9663698, 4 jobs, 2026-08-18; cluster 9674800). Its Pareto front admits only architectures with a driven-snapshot fit at `d_FS <= 1e-6`; the best fit ever reached was 5.5e-4 (strong_weak_u8), median 2.9e-3. `intermediate_weak` never met even the 1e-4 ground-state gate (best 2.5e-4 in 825 architectures, 14.6 h). | `raw_outputs/paper_ii_regime_seeds_v1/*/search_batches.jsonl` in the mirror; `pipelines/time_dynamics/fixed_vqe_conditioning.py::pareto_front` |
| Stage 2 uses the three prospective packagers | The packagers admit only the frozen matrix `paper_ii_parallel_benchmarks_20260828` (exactly 8 seeds / 39 references, payload digest pinned in `parallel_benchmark_matrix.py`, seven test files). That matrix names `n_ph_max_7` in `excluded` and contains no `(U=4, g=0.5)` cell, so it cannot take this grid or the anchor. Its six HH seeds already exist (cluster 9714490–9714495, 2026-08-29, ΔE ≤ 5.2e-8). | `pipelines/time_dynamics/campaigns/paper_ii_parallel_benchmark_matrix_v1.json`; `prospective_apm_exchange_worker._registered_subjects`; mirror `Holstein_paper_ii_prospective_seeds_267516ea_20260829/source/raw_outputs/paper_ii_seed/*/seed_qualification_receipt.json` |
| Stage 2 can run on `main` | The route the results PDF reports (`paper_ii_current_best_compiled_vs_proxy.pdf`, "Gamma(v*) realized", internal key `q_compiled`, arm `exchange_q_compiled`) exists only on `codex/paper-ii-exact-estimator-telemetry-20260903` (worktree `codex-paper-ii-dual-selector-overnight-20260903`, 51 commits ahead of / 17 behind `main`, merge-base `557f20ae`). `main` has only the legacy `exchange` selector, which the PDF does not report, and the staged machinery hard-wired that arm. The plan's own queued change says the Γ route is the only route and is to be called APM. | `git rev-list --left-right --count main...HEAD` in that worktree; `paper_ii_runs.py` there registers `exchange_q_compiled`, `exchange_gamma_compiled`, `exchange_legacy_proxy_fixed` |
| The anchor row nests the published point | Only within the fixed-filling sector. The calibration seed carries a scalar site potential `dv = 0.25`, which the runtime applies uniformly as `-dv * sum n_{i,sigma}` (Frobenius difference of the static Hamiltonian 4.47; ground energy −0.401 vs 0.159), a constant `-dv*N_e = -0.5` that commutes with everything. The manuscript's model has no such term. The frozen seed worker cannot build a seed at `dv = 0.25` (its VQE objective and exact qualifier apply that term with opposite signs: measured ΔE = 2·dv·N_e = 1.0), so every regime seed is built at `dv = 0`; errors and response scales are differences and are unaffected. | `src/quantum/hubbard_latex_python_pairs.py::_parse_site_potential` (scalar → uniform); local `matrix_at(0)` comparison; `raw_outputs/.../anchor_u4_g0p5_nph1` rebuild at `dv = 0.25` |
| The image only needs building | `image.def` was not in git at all (added in d4ea8c37, absent since); local builds are impossible here (arm64, no apptainer). The pinned campaign image `fa5c4ea8…` (216,371,200 B, 2026-08-26) was in the mirror and is now installed locally; the remote `Holstein_paper_ii_regime_seeds_v1/` dir still holds the **older** 2026-05-17 image `68e8335f…`. | `CAMPAIGN_STATE_20260817.md:21`; `input/runtime_image_identity.json` |

Two more facts that gate submission:

- **Standing block.** `chtc/paper_ii_staged_accuracy_euler_v1_t20_resume_repair_d3301581_20260826/report_refresh_required.json`
  blocks `further_campaign_submission` for the Paper II lane and is still open
  (`docs/reports/OPEN_DECISIONS_20260901.md` C5). The user must lift it or
  override it in writing before anything is submitted.
- **Remote state is unverified.** `ssh -O check ap2001` found no control master
  and a direct connect timed out (GlobalProtect VPN + one Duo push are needed).
  Every remote number in this handoff is from the 2026-08-31 mirror.

## 1. What was built instead (branch `fable/paper-ii-regime-seeds-chtc-20260904`: commits `48336a01`, `910d8738`, and the 2026-09-05 commit that carries stage 1.5 and this revision)

The seed protocol is the one the six matrix seeds were built with: the frozen
three-complete-layer fixed-structure VQE
(`three_complete_ordered_hh_hamiltonian_layers_v2`, four seeded L-BFGS-B
restarts, exact compiled adjoint gradient, hard `|E - E_0| <= 1e-6`, no
conditioning search, no snapshot fits, no exact data in the optimizer). It ran
on CHTC in 13 s–26 min per seed at ≤ 394 MB.

| Change | Path | Why |
|---|---|---|
| Ported the executor reverse-mode gradient the frozen worker calls | `src/quantum/compiled_ansatz.py` (`energy_and_runtime_gradient`, `_runtime_steps`), `test/test_compiled_ansatz.py` | Main had the worker side of 267516ea but not the executor side: `compute_seed` raised `AttributeError` in 2 s. Two-file cherry-pick; 24 tests pass. |
| Regime seed runner | `pipelines/time_dynamics/runners/build_hh_fixed_layer_seed.py`, `test/test_build_hh_fixed_layer_seed.py` | Drives `prospective_hh_seed_worker.compute_seed` for an arbitrary `(U, g_ep, n_ph_max)`; writes a runtime-loadable artifact + sealed receipt; exit 3 and **no artifact** when the gate fails. |
| Stage-1 bundle rewritten | `chtc/paper_ii_regime_seeds_v1/{run_seed.sh, submit_regime_seeds_v1.sub, input/regimes.tsv, input/regime_ids.txt, input/regime_queue.txt, input/runtime_image_identity.json}` | Seventh anchor row; per-row `request_memory`; image sha bound; lifecycle validator passes. |
| Post-fetch installer | `chtc/paper_ii_regime_seeds_v1/install_seeds.py`, `test/test_paper_ii_regime_seed_install.py` | The only bridge from fetched `raw_outputs/` to the registry paths `input/seeds/<regime_id>.json`; refuses unsealed, unqualified, physics-mismatched, or tampered seeds. |
| Registry | `pipelines/time_dynamics/paper_ii_runs.py` (`_HH_REGIME_ROWS` + anchor) | `resolve_regime("anchor_u4_g0p5_nph1")` exists; `SeedNotBuiltError` until installed. |
| Stage-2 survey path | `pipelines/time_dynamics/campaigns/paper_ii_staged_accuracy_v1.py` (`--regime`, `--promotion-horizon`, `--apm-arm`, `campaign_id_for`), `pipelines/time_dynamics/staged_campaign.py` (`StagedCampaign.apm_arm`), tests | The proven staged screen/promotion machinery (the locked 15 cells came from it) runs for any registered regime with a suffixed campaign id, under the **same response-normalized matched-pair policy** once the regime's exact-response scales are registered, and with the APM method realized by any registered non-comparator arm (`--apm-arm exchange_q_compiled` on the dual-selector lineage). Ledger vocabulary, defaults, and the locked campaign are unchanged (12 tests). |
| Stage 1.5: per-regime exact response scales | `pipelines/time_dynamics/diagnostics/build_regime_response_scales.py`, `test/test_build_regime_response_scales.py`; registry `pipelines/time_dynamics/campaigns/paper_ii_exact_response_scales_by_regime.json` (written by the producer) | Builds the six content-addressed exact references of a regime's installed seed on the locked grid (126-point t5 screen + 376-point continuation, the same floats the record step keys on), derives `RMS_t[O(t)-O(0)]` with the loader's own formula, writes the v1-schema sidecar and a registry row, and round-trips it through the campaign loader. For the calibration seed it reproduces all six locked reference keys bit-exactly. |
| Seed settings identity | `build_hh_fixed_layer_seed.py` writes `t_prime = 0`, `v_nn = 0`, `initial_state_source` | The response-scale loader refuses a seed that leaves these implicit; the runtime loader already defaulted them to zero, so behaviour is unchanged. `regimes.tsv` gained an explicit `dv` column (all rows `0.0`). |
| Image definition restored | `chtc/time_dynamics_optuna/image.def` | `build_image.sh` / `submit_image_build.sub` work again. |

### Local measurements (Apple M4, `.venv` Python 3.12, RAM-guarded)

| Row | Qubits | Coordinates | ΔE | Wall | Peak RSS | Result |
|---|---:|---:|---:|---:|---:|---|
| `anchor_u4_g0p5_nph1` (U=4, g=0.5, nph=1) | 6 | 216 | 1.69e-11 | 8 s | 87 MB | qualified |
| `intermediate_weak` (U=1.25, g=0.354, nph=3) | 8 | 972 | 6.23e-08 | 190 s | 664 MB | qualified (the row the search could not seed in 14.6 h) |
| `weak_strong` (U=0.25, g=0.791, nph=7) | 10 | — | — | > 31 min, unfinished | > 8,000 MB | killed by the 8 GB RAM guard at 412 s (peak 8,014 MB during pool/architecture construction); rerun under a 14 GB guard held 2.3–5.4 GB RSS in the VQE phase and was stopped at 31 min, unfinished. The `request_memory = 16GB` row is sized from that peak; the true wall is unknown and is measured on CHTC, not here. |

Stage 1.5 on the anchor seed (six drives, 501-point grid, 40 substeps, 6
qubits): **22 s wall, 75 MB peak**; the resulting t20 scales agree with the
calibration seed's tracked sidecar to four digits (fastfast energy 0.2464 vs
0.24639, doublon 0.0413 vs 0.04133, site 0.1118 vs 0.11178), as they must for
the same physics up to the inert `dv` shift. At 8 qubits the cost is the same
order; at 10 qubits (nph=7) it is a 1024-dimensional `expm` per substep, still
minutes.

Dynamics smoke on the anchor seed (production runner, `smoke` horizon, 13 points,
`fastweak`): `exchange` ran 12 steps and accepted one generalized-exchange delete
at k=9 (peak 606 MB); `avqds` ran 12 steps (peak 280 MB). Both `run.json` carry a
`run_lock.physics.seed_sha256` equal to the artifact digest and a
`resume_state`, so continuation segments work on these seeds.

Survey dry run for the anchor regime: `next` emits 36 t5 screen actions under
campaign id `paper_ii_staged_accuracy_euler_v1__anchor_u4_g0p5_nph1__t10`;
`prepare-chtc` emits `PREPARED_NOT_SUBMITTED` and its descriptor passes
`chtc/validate_condor_submit_lifecycle.py`.

## 2. Decisions (user's; two settled on 2026-09-05, one open, one new)

Notation: the model is `H_HH(U, g, omega_0 = 1, t = 1)` at `n_ph^max`, the
survey compares the APM route (the manuscript's realized Γ(θ̇*) selector) with
AVQDS across regimes `(U, g, n_ph^max)`, six drives `(A, ω)`, three step
controllers, on the locked grid.

**D1 (open) — which Hamiltonian points get regime blocks.**
→ Recommendation: seeds for all seven rows (seconds to minutes each at
`n_ph^max ≤ 3`); dynamics first on the four 6–8-qubit rows
(`(4, 0.5, 1)`, `(0.25, 0.354, 3)`, `(1.25, 0.354, 3)`, `(8, 0.354, 3)`);
`n_ph^max = 7` (10 qubits) later. Basis: the locked 8-qubit 616-coordinate
cells needed 7–24 h for a whole t10 (cluster 9674614); a 972-coordinate seed
is ~1.5× that; the frozen matrix already excludes `n_ph_max_7`; the nph=7
seed build alone exceeded 8 GB here.

**D2 (settled 2026-09-05, user) — qualification policy.** The survey keeps
the locked campaign's response-normalized matched-pair policy (relative
target 0.05, guard 0.15, matched within ×2, infidelity ≤ 1e-4, absolute
guards 1e-2 / 5e-3 / 5e-3). The per-regime exact response scales are a local
exact-diagonalization step (stage 1.5 below, 22 s at 6 qubits), not a reason
to weaken the policy. `componentwise_1e-4` remains available only as the
legacy ingestion replay.

**Q3 (settled 2026-09-05, user: yes) — submit before the report refresh.**
The `report_refresh_required` block (OPEN_DECISIONS C5) does not gate this
campaign.

**D3 (new) — which code lineage runs stage 2.** The Γ(θ̇*) route is not on
`main`. Options: (a) cut the survey branch from
`codex/paper-ii-exact-estimator-telemetry-20260903` and cherry-pick the two
commits of this branch onto it (the touched files are identical between the
two lineages except `paper_ii_runs.py`, `compiled_ansatz.py`, and the runner,
none of which this branch edits in conflicting places), or (b) wait for the
queued landing of that lineage onto `main`. → Recommendation: (a), because
the seeds and the stage-1.5 scales are lineage-independent and stage 2 can
then start the day the seeds return. Stage 1 does not depend on D3.

**Gates only the user can open:** GlobalProtect VPN + `ssh -MNf ap2001` (one
Duo push), and the words "submit stage 1".

## 3. Anchors

| Field | Value |
|---|---|
| Checkout | `/Users/lordskyler/Holstein_test` (main at `0027ac5c` when this branch was cut) |
| Work here | worktree `/Users/lordskyler/Holstein_test.worktrees/fable-paper-ii-regime-seeds-20260904`, branch `fable/paper-ii-regime-seeds-chtc-20260904`; `git log --oneline main..HEAD` lists the commits (48336a01 code, 910d8738 handoff, then the 2026-09-05 stage-1.5/route revision) |
| Python | `/Users/lordskyler/Holstein_test/.venv/bin/python` (3.12). System `python3` is 3.9 and cannot import the package. |
| Test baseline | `PYTHONPATH=. .venv/bin/python -m pytest test/test_paper_ii_runs.py test/test_prospective_hh_seed_worker.py test/test_compiled_ansatz.py test/test_build_hh_fixed_layer_seed.py test/test_paper_ii_regime_seed_install.py test/test_paper_ii_staged_accuracy_campaign.py test/test_staged_campaign.py test/test_staged_campaign_chtc.py test/test_staged_campaign_artifacts.py test/test_staged_campaign_ingestion.py test/test_prospective_hh_seed_activation.py test/test_prospective_hh_seed_ingestion.py -q -p no:cacheprovider` plus `test/test_build_regime_response_scales.py test/test_exact_reference_store.py` → **161 passed, 2 failed**; the 2 failures (`test_staged_campaign_chtc.py::test_t20_package_copies_each_resume_source_and_preserves_global_grid`, `::test_package_can_exclude_in_flight_t5_actions_and_emit_ready_promotions`) fail identically on unmodified main. Every test file constructing `CompiledAnsatzExecutor` (19 files): **1061 passed, 10 failed, 14 skipped** in 6 m 18 s; all 10 failures (`test_hh_realtime_checkpoint_controller.py` ×7, `test_time_dynamics_resume.py::test_continuation_matches_an_uninterrupted_trajectory`, `test_ra_adapt_semantic_closure_routes.py` ×2) reproduce on unmodified main. |
| Image | `chtc/time_dynamics_optuna/image.sif`, sha256 `fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f`, 216,371,200 B (gitignored; copy in both the checkout and the worktree; source `~/chtc_rescue/home/Holstein_siam_nb4_source_ba50339c_20260828/chtc/time_dynamics_optuna/image.sif`) |
| Submit node | `jsstrobel@ap2001.chtc.wisc.edu` via GlobalProtect; alias `ap2001` (ControlMaster in `~/.ssh/config`). Never enter the password; one Duo push per login; read-only queries are one-shot. |
| Remote quota | `/home` 35.7 GB used of 40.96 GB soft / 51.2 GB hard on 2026-09-03; `/staging/jsstrobel` does not exist. |
| Remote dirs | old: `/home/jsstrobel/Holstein_paper_ii_regime_seeds_v1/` (2026-08-22 tree, wrong image, held procs); new (create): `/home/jsstrobel/Holstein_paper_ii_regime_seeds_v1_20260904/` |
| Held jobs not to touch | cluster 9674800 procs 3–5 (this lane's 2026-08-22 nph=7 rows) and everything else in the 138-job held list; removal only when the user names them. |

## 4. Scope

In scope: stage-1 submission preparation, monitoring, fetch, verification,
installation, the stage-2 survey packages after seeds exist, and the receipts
that record each step. Out of scope, with reasons: the prospective lane
(`pipelines/time_dynamics/prospective_*`; another design, digest-pinned, do
not bend it to this grid); any `condor_release`/`condor_rm`; the manuscript;
the `report_refresh_required` block itself (it belongs to the C5 owner). Local
RAM: every heavy process under `pipelines/shell/ram_guard.py`, aggregate
≤ 18 GB (`agent_guidance/shared/memory-budget.md`).

## 5. Increments — stage 1 (seeds)

Proceed autonomously through I2 and I3 except where marked **STOP**; I1 is the
user's.

### I1 — user opens the gates
GlobalProtect on, then one push:
```bash
ssh -MNf ap2001 && ssh -O check ap2001
```
Expected: `Master running`. Then the user states, in chat, (a) D1/D2 answers or
"yes", (b) that the C5 block is lifted or overridden for this campaign, and
(c) "submit stage 1".

### I2 — stage the bundle remotely (no submission yet)
Goal: a fresh remote tree with the pinned image and the seven-row bundle.
```bash
cd /Users/lordskyler/Holstein_test.worktrees/fable-paper-ii-regime-seeds-20260904
ssh ap2001 'quota -s | tail -3; mkdir -p Holstein_paper_ii_regime_seeds_v1_20260904/{logs,raw_outputs}'
rsync -az --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
  --exclude 'output' --exclude 'raw_outputs' --exclude 'logs' --exclude 'artifacts' \
  --exclude 'chtc/time_dynamics_optuna/image.sif' \
  pipelines src chtc/paper_ii_regime_seeds_v1 chtc/time_dynamics_optuna chtc/validate_condor_submit_lifecycle.py \
  ap2001:Holstein_paper_ii_regime_seeds_v1_20260904/
# Image: reuse an existing remote copy of fa5c4ea8… instead of spending 216 MB of quota.
ssh ap2001 'for p in Holstein_siam_nb4_source_ba50339c_20260828 Holstein_siam_nb4_frontier_after_9712800_ebdf7ca5_20260827 Holstein_paper_ii_staged_accuracy_euler_v1_frontier319_4774e3f3_20260827; do f=$HOME/$p/chtc/time_dynamics_optuna/image.sif; [ -f "$f" ] && { sha256sum "$f"; break; }; done'
```
Expected: one line ending in the sha above. Then symlink it (or `rsync` the
local file if none survives) and verify:
```bash
ssh ap2001 'ln -sfn <that path> Holstein_paper_ii_regime_seeds_v1_20260904/chtc/time_dynamics_optuna/image.sif && sha256sum Holstein_paper_ii_regime_seeds_v1_20260904/chtc/time_dynamics_optuna/image.sif && cd Holstein_paper_ii_regime_seeds_v1_20260904 && python3 chtc/validate_condor_submit_lifecycle.py chtc/paper_ii_regime_seeds_v1/submit_regime_seeds_v1.sub && echo LIFECYCLE_OK && condor_submit -dry-run - chtc/paper_ii_regime_seeds_v1/submit_regime_seeds_v1.sub | grep -c "^Iwd" '
```
Expected: sha `fa5c4ea8…`, `LIFECYCLE_OK`, and `7` (seven procs in the dry run).
**Stop condition:** quota headroom < 2 GB, a missing image, or a dry-run count
≠ 7 → report, do not submit.

### I3 — submit (only after I1(c)), then monitor
```bash
ssh ap2001 'cd Holstein_paper_ii_regime_seeds_v1_20260904 && condor_submit chtc/paper_ii_regime_seeds_v1/submit_regime_seeds_v1.sub'
```
Record the cluster id immediately in
`chtc/paper_ii_regime_seeds_v1/submission_receipt_<cluster>.json` (cluster,
UTC time, remote dir, image sha, bundle file sha256s from `git ls-files -s`)
and commit it with an explicit path.

Monitor with one-shot reads only, by exact cluster id: after 5 min, 30 min,
then every 6 h:
```bash
ssh ap2001 'condor_q <cluster> -nobatch; condor_history <cluster> -limit 7 -af ProcId ExitCode RemoteWallClockTime MemoryUsage HoldReasonCode'
```
Expectations per proc (from the measurements above and cluster 9714490–95):

| Proc | Row | Expected wall | Expected memory | Requested |
|---:|---|---|---|---|
| 6 | anchor (6q, 216 coords) | < 2 min | < 200 MB | 2 GB |
| 0–2 | nph=3 rows (8q, 972 coords) | 10–40 min | < 1 GB | 4 GB |
| 3–5 | nph=7 rows (10q) | unknown; > 7 min locally before the 8 GB kill | > 8 GB | 16 GB |

If a nph=7 proc is held with `HoldReasonCode 34` (memory), record the peak from
the `.log` (`MemoryUsage of job (MB)` lines), do **not** release it, and do not
resubmit with a different request without the user (separation rule: the
manifest is fixed before launch). Never `condor_release` or `condor_rm`
anything.

When a proc completes, fetch and verify without touching the remote tree:
```bash
mkdir -p ~/chtc_fetch/paper_ii_regime_seeds_v1_20260904
rsync -az ap2001:Holstein_paper_ii_regime_seeds_v1_20260904/raw_outputs ap2001:Holstein_paper_ii_regime_seeds_v1_20260904/logs ~/chtc_fetch/paper_ii_regime_seeds_v1_20260904/
cd /Users/lordskyler/Holstein_test.worktrees/fable-paper-ii-regime-seeds-20260904
PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python chtc/paper_ii_regime_seeds_v1/install_seeds.py --fetched-root ~/chtc_fetch/paper_ii_regime_seeds_v1_20260904
```
Expected: JSON with `installed` naming every returned row, `problems` empty
(exit 0), files under `chtc/paper_ii_regime_seeds_v1/input/seeds/` plus
`SHA256SUMS` and `install_receipt.json`. Commit those with explicit paths and
regenerate the parameters doc:
```bash
PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python -m pipelines.time_dynamics.write_paper_ii_manifest
git add chtc/paper_ii_regime_seeds_v1/input/seeds agent_guidance/time-dynamics/paper-ii-run-parameters.md && git commit -m "paper ii: install stage-1 regime seeds from cluster <cluster>"
```
**Stop conditions:** any receipt with `status: unqualified` (report its ΔE; the
row is "not seedable at 1e-6 by this protocol", not a retry); `problems`
non-empty from the installer; a remote `sha256sum` that differs from the
receipt.

Partial returns are fine: install the rows that came back; stage 2 can start on
them while nph=7 rows are still queued.

### I4 — stage 1.5: exact response scales per installed regime (local, minutes)

Goal: regime R can be qualified under the locked matched-pair policy.
```bash
cd /Users/lordskyler/Holstein_test.worktrees/fable-paper-ii-regime-seeds-20260904
R=anchor_u4_g0p5_nph1
PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python pipelines/shell/ram_guard.py --limit-mb 8000 -- \
  /Users/lordskyler/Holstein_test/.venv/bin/python -m pipelines.time_dynamics.diagnostics.build_regime_response_scales --regime $R
git add pipelines/time_dynamics/campaigns/paper_ii_exact_response_scales_$R.json pipelines/time_dynamics/campaigns/paper_ii_exact_response_scales_by_regime.json
git commit -m "paper ii: exact response scales for regime $R"
```
Expected: JSON summary with six drives × {t5, t20} scales, `verified_index_schema
= paper_ii_response_scale_index_v2`, references under
`output/paper_ii_regime_survey_$R/exact_references/` (six keys), and a registry
row whose `sha256` matches the sidecar. Wall: ~22 s at 6 qubits, minutes at 8–10.
**Stop condition:** any `ResponseScaleIntegrityError` (the loader re-derives
every scale from the reference data; a mismatch means the seed or the store
changed under it). Never edit a sidecar by hand.

## 6. Increments — stage 2 (regime survey), per installed regime R

Only after I3 installed R and I4 registered its scales. Configuration is
fixed here, identical to the locked calibration campaign except for the seed:
screen `t5` (126 points) → promotion `t20` (376 continuation points via
`--resume-from-run-json`; the response-scale loader binds exactly t5 and
t20), Euler ridge 1e-6, `mclachlan_l2_1e-3` gate, `paper_declared_v1` search
profile, three controllers × six drives × {AVQDS, APM} = 36 cells, thresholds
from the default extension seed (3e-6, 1e-6, 3e-7, 1e-7), the
response-normalized matched-pair policy, and the APM arm per D3
(`--apm-arm exchange_q_compiled` on the dual-selector lineage; on `main` only
the legacy `exchange` arm exists and the flag defaults to it).

```bash
cd <survey checkout per D3>
R=anchor_u4_g0p5_nph1; D=$(date +%Y%m%d); ARM=exchange_q_compiled
PYTHONPATH=. .venv/bin/python -m pipelines.time_dynamics.campaigns.paper_ii_staged_accuracy_v1 \
  next --ledger output/paper_ii_regime_survey/$R/ledger.json --regime $R --apm-arm $ARM \
  --output-root output/paper_ii_regime_survey/$R/raw \
  --worklist-json output/paper_ii_regime_survey/$R/worklist_screen.json
PYTHONPATH=. .venv/bin/python -m pipelines.time_dynamics.campaigns.paper_ii_staged_accuracy_v1 \
  prepare-chtc --ledger output/paper_ii_regime_survey/$R/ledger.json --regime $R --apm-arm $ARM \
  --package-dir chtc/paper_ii_regime_survey_v1_${R}_screen_t5_$D
.venv/bin/python chtc/validate_condor_submit_lifecycle.py chtc/paper_ii_regime_survey_v1_${R}_screen_t5_$D/*.sub
```
Expected: `"status": "PREPARED_NOT_SUBMITTED"`, `"stage": "screen_t5_frontier"`,
campaign id `paper_ii_staged_accuracy_euler_v1__<R>__t20__exchange_q_compiled`,
36 action ids, `input/seed.json` whose sha256 equals the installed seed's,
`input/code_revision.txt` = the committed HEAD (commit before preparing; a
dirty tree is stamped `-dirty`), and every APM action's `runner_argv` carrying
`--exchange-route-bundle paper_ii_exchange_q_compiled_v1`. The package
descriptor requests 2 CPU / 10 GB / 40 GB and `+MaxRuntime = 43200` (inert;
boundedness comes from the grids). The Γ(θ̇*) route compiles patches
(FakeMarrakesh, ~30 s per large-miss checkpoint at 4 compile workers); the
dual-selector campaign ran with `request_cpus = compile_workers + 1 = 4`
(cluster 9718529) — use the same and pass `--compile-workers 3` if the
lineage's prepare step exposes it; report, do not guess, if it does not.

Submit only with the user's explicit "submit <package dir>"; upload with the
I2 rsync pattern into `Holstein_paper_ii_regime_survey_v1_<R>_<D>/`, record
the cluster id beside the package, monitor as in I3. Wall-time basis for the
t5→t20 continuation at 8 qubits / 972 coordinates: the locked 616-coordinate
whole-t10 cells took 7–24 h, so expect roughly 20–50 h per continuation; that
is under the 72 h hold but not by much, so watch the first returns and report
any proc past 48 h. Fetch to
`output/paper_ii_regime_survey/$R/fetched/cluster_<id>/`, then ingest:
```bash
PYTHONPATH=. .venv/bin/python -m pipelines.time_dynamics.campaigns.paper_ii_staged_accuracy_v1 \
  record-package --ledger output/paper_ii_regime_survey/$R/ledger.json --regime $R --apm-arm $ARM \
  --actions-jsonl chtc/paper_ii_regime_survey_v1_${R}_screen_t5_$D/input/actions.jsonl \
  --artifact-root output/paper_ii_regime_survey/$R/fetched/cluster_<id>/raw_outputs \
  --seed-json chtc/paper_ii_regime_seeds_v1/input/seeds/$R.json \
  --reference-store-root output/paper_ii_regime_survey_$R/exact_references
```
The reference store root is the one stage 1.5 filled, so `record-package`
reuses the six references instead of rebuilding them. Then `next` again: the
ledger opens the `t20` promotion actions (each carrying
`resume_from_run_path` to its own t5 `run.json`), and the same `prepare-chtc`
→ submit → fetch → `record-package` cycle runs once more with a
`..._promote_t20_<D>` package dir. R is complete when `next` reports
`stage: complete`.

Order: `anchor_u4_g0p5_nph1` first (it nests the published point and is the
cheapest), then `weak_weak`, `intermediate_weak`, `strong_weak_u8`; nph=7 rows
only after the user reopens D1 with the measured seed cost in hand.

## 7. Traps (each bit this session)

- `python3` here is 3.9 and cannot import the package; scripts that call
  `python3` (`run_seed.sh`) must run with `PATH=/Users/lordskyler/Holstein_test/.venv/bin:$PATH`
  locally. On CHTC the image provides `python3`.
- The worktree needs its own copy of `image.sif`; a symlink out of the repo
  root makes `prospective_seed_chtc_package.py runtime-image-identity` refuse.
- `+MaxRuntime` is inert unless a `periodic_*` expression references it; the
  72 h pool hold transfers nothing (`chtc-72h-hold-no-stageout`). Both stage-2
  segments are bounded by their 126-point grids, not by wall clock.
- `_validated_revision` / `resolve_code_revision` stamp `-dirty` on an
  uncommitted tree; commit before `prepare-chtc` or pass `--code-revision`.
- `raw_outputs/`, `output/`, `logs/` are gitignored: the only durable seed copy
  is the one `install_seeds.py` writes under `chtc/.../input/seeds/`.
- The seeds built locally in this session (`raw_outputs/paper_ii_regime_seeds_v1/`
  in the worktree) are smokes on arm64, not evidence; production seeds come
  from CHTC (x86) and are installed only through the receipt path.
- `prepare-chtc` with an absolute `--package-dir` writes absolute paths into the
  descriptor; always pass a repo-relative `chtc/...` directory.
- The frozen seed worker is self-consistent only at `dv = 0`: its VQE
  objective and exact qualifier apply a scalar site potential with opposite
  signs (ΔE = 2·dv·N_e). Keep every regime row at `dv = 0`; the term is a
  constant at fixed filling.
- The response-scale loader binds exactly `t5` and `t20` and keys references on
  the combined grid (126-point screen + 376-point continuation); a plain
  `linspace(0, 20, 501)` differs in the last bit at 72 points and yields a
  different key. `build_regime_response_scales.locked_grid()` is that grid.
- A seed artifact without explicit `t_prime`, `v_nn`, `initial_state_source`
  settings loads in the runner but is refused by the response-scale loader.
- Two pre-existing failures in `test_staged_campaign_chtc.py` and one in
  `test_ra_adapt_semantic_closure_routes.py` (plus the eight in
  `test_hh_realtime_checkpoint_controller.py` / `test_time_dynamics_resume.py`)
  fail identically on unmodified main; do not "fix" them in passing.

## 8. Report back

One block per increment, in this file under a `## Log` heading appended at the
end (not in chat):
```markdown
## Increment N — <goal>
Status: done | blocked | skipped
Commands run: <exact>
Result: <cluster ids, sha256s, ΔE values, counts>
Verification: <installer JSON, test counts, remote sha256sum lines>
Deviations: <anything done differently, and why>
```

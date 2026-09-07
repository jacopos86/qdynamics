# HANDOFF — Paper II CHTC campaign: Hubbard–Holstein regimes and SIAM

**Executor:** Codex · **Written:** 2026-09-05 · **Contract:** `agent_guidance/shared/agent-handoff-contract.md`
**Branch to work on:** `fable/paper-ii-regime-survey-20260905`
(worktree `/Users/lordskyler/Holstein_test.worktrees/fable-paper-ii-regime-survey-20260905`)
for everything, Hubbard–Holstein and SIAM. Python is always
`/Users/lordskyler/Holstein_test/.venv/bin/python` with `PYTHONPATH=.` from the
worktree root. Nothing is run from the old SIAM branch
`fable/siam-search-profiles-20260905`.

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

### 1b. SIAM (single impurity, U = 1.5, Nb bath sites)

Both SIAM arms run on this branch's compiled-Q route with the same exact
estimator accounting as the Hubbard–Holstein cells: AVQDS vs compiled-Q
exchange (`exchange_q_compiled`, route bundle `paper_ii_exchange_q_compiled_v1`),
scored offline against the exact sector reference and classified by the
response-normalized burden ratio (ceiling 2.0). Every score, receipt and
estimator record now carries the route identity, so a compiled-Q arm cannot
be paired against or reported as the historical route.

| system | qubits | state |
|---|---:|---|
| Nb = 2 proof pair | 6 | **done on this branch** (`chtc/siam_nb2_proof_qc_v1/`, commit `2163171b`): both arms qualified; burden AVQDS 0.3525, compiled-Q 0.3589, ratio 1.018, `qualified_pair`; estimator work exact: AVQDS 967,888 (12 parameters, 61 rotations), compiled-Q 928,409 (11 parameters, 59 rotations); wall 55 s / 165 s, peak 105 / 166 MB |
| Nb = 4 ladder | 10 | **to run on CHTC** (§8): r00 screen packages for both arms are prepared and not submitted; climb the activation-cut ladder until the t = 5 pair is matched |

The earlier SIAM numbers (Nb = 2 ratio 1.011 at `e073fe48`; Nb = 4 exchange
r07 burden 0.0554 at `8edc93a3`, both on `fable/siam-search-profiles-20260905`)
came from the older exchange rule without estimator accounting. They are
history, not evidence for this comparison.

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
PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest test/test_paper_ii_runs.py test/test_paper_ii_staged_accuracy_campaign.py test/test_build_hh_fixed_layer_seed.py test/test_build_regime_response_scales.py test/test_compiled_ansatz.py test/test_paper_ii_regime_seed_install.py test/test_siam_compiled_q_route.py test/test_siam_chtc_package.py test/test_siam_nb4_ladder.py test/test_siam_offline_scoring.py test/test_siam_segment_stitching.py test/test_siam_estimator_work.py -q -p no:cacheprovider
```
Expected: 334 passed, 0 failed on the last line. Then stage the bundle remotely:
```bash
ssh ap2001 'quota -s | tail -3; mkdir -p Holstein_paper_ii_regime_seeds_v1_20260905/{logs,raw_outputs}'
rsync -azR --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude 'output' --exclude 'raw_outputs' --exclude 'logs' --exclude 'artifacts' --exclude 'chtc/time_dynamics_optuna/image.sif' \
  ./pipelines ./src ./chtc/paper_ii_regime_seeds_v1 \
  ./chtc/siam_nb4_qc_avqds_screen_r00_cut1em02_source_2163171b_20260905 \
  ./chtc/siam_nb4_qc_full_apm_screen_r00_cut1em02_source_2163171b_20260905 \
  ./chtc/time_dynamics_optuna ./chtc/validate_condor_submit_lifecycle.py \
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
and every 6 h while dynamics jobs run (SIAM screens and continuations are
dynamics jobs). Each heartbeat is exactly one queue
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
| finished, `ExitCode 0` | fetch, verify, install/ingest (§4, §6, §8) |
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

## 8. SIAM Nb = 4 — compiled-Q ladder (CHTC)

All from the survey worktree. Activation-cut ladder
(`SIAM_NB4_ACTIVATION_CUT_LADDER`): r00 = 1e-2, r01 = 3.33e-3, r02 = 1e-3,
r03 = 3.33e-4, r04 = 1e-4, r05 = 3.33e-5, r06 = 1e-5, … r10 = 1e-7. One rung
of one arm is two dependent `queue 1` jobs: a **screen** (t 0→1, 101 points)
and, if the screen qualifies, a **continuation** (t 1→5, 401 points) resumed
from the screen's `run.json`. Action ids carry the route token `qc`:
`siam-nb4-qc-<lane>-<screen|continue>-r<NN>-cut<tok>`, lane `avqds` or
`full_apm` (the compiled-Q exchange arm's ladder lane). Each job's outputs
land in `raw_outputs/<package>/<action id>/{run.json,action_receipt.json}`.

Prepared (commit `67f3a05e`, source-locked to
`2163171bbbf98015fe741b445fc9e1dd3e1794aa`, schema `siam_nb4_chtc_action_v5`,
2 CPU / 2 GB / 40 GB / `+MaxRuntime 43200`, `requirements = TARGET.HasSIF`):

| package | arm | action id | action sha256 |
|---|---|---|---|
| `chtc/siam_nb4_qc_avqds_screen_r00_cut1em02_source_2163171b_20260905` | `avqds` | `siam-nb4-qc-avqds-screen-r00-cut1em02` | `f127a0d4a37c178b291af986cc59ff2e65132abe9698381471a0da798bf6a1f5` |
| `chtc/siam_nb4_qc_full_apm_screen_r00_cut1em02_source_2163171b_20260905` | `exchange_q_compiled` | `siam-nb4-qc-full_apm-screen-r00-cut1em02` | `091a9e44aa10d7b0a866b2879519f29c7afb6fec15e31738a07035f9e75b102c` |

Both share seed sha `5752bc1e…` (copied into each `input/seed.json`) and the
compiled-Q route contract sha `7883d937…c356b1`.

**8.1 Submit the two r00 screens** (after §3 upload; `submit` from the user):
```bash
ssh ap2001 'cd Holstein_paper_ii_regime_seeds_v1_20260905 && for p in chtc/siam_nb4_qc_*_source_2163171b_20260905; do python3 chtc/validate_condor_submit_lifecycle.py $p/submit.sub && echo LIFECYCLE_OK $p; done'
ssh ap2001 'cd Holstein_paper_ii_regime_seeds_v1_20260905 && condor_submit chtc/siam_nb4_qc_avqds_screen_r00_cut1em02_source_2163171b_20260905/submit.sub && condor_submit chtc/siam_nb4_qc_full_apm_screen_r00_cut1em02_source_2163171b_20260905/submit.sub'
```
Record both cluster ids; monitor per §7. Expected: AVQDS screen ≈ 10 min,
< 300 MB; compiled-Q screen about 3× that (Nb = 2 ratio). Record the actual
`RemoteWallClockTime` and `MemoryUsage` — they size the continuation.

**8.2 Fetch, verify, score a finished segment** (`P` = its package dir,
`A` = its action id, `LANE` = `avqds` or `full_apm`, `ARM` = `avqds` or
`exchange_q_compiled`):
```bash
rsync -az ap2001:Holstein_paper_ii_regime_seeds_v1_20260905/raw_outputs/$P/ raw_outputs/$P/
rsync -az ap2001:Holstein_paper_ii_regime_seeds_v1_20260905/$P/logs/ $P/logs/
PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python - <<EOF
import json
from pipelines.time_dynamics.siam_chtc_package import validate_siam_nb4_completion_receipt
from pipelines.time_dynamics.diagnostics.siam_exact_reference_store import SIAMExactReferenceStore
from pipelines.time_dynamics.diagnostics.siam_offline_scoring import (
    score_siam_raw_run, SIAM_NB4_FULL_TIMES, SIAM_NB4_SCREEN_TIMES, SIAM_NB4_CONTINUATION_TIMES)
P, A, ARM = "$P", "$A", "$ARM"
seed = f"{P}/input/seed.json"; run = f"raw_outputs/{P}/{A}/run.json"
receipt = json.load(open(f"raw_outputs/{P}/{A}/action_receipt.json"))
validate_siam_nb4_completion_receipt(receipt, expected_source_revision="2163171bbbf98015fe741b445fc9e1dd3e1794aa", expected_arm=ARM)
store = SIAMExactReferenceStore("output/siam_screening/nb4_qc_v1/exact_references")
full = store.get_or_build(seed_path=seed, times=SIAM_NB4_FULL_TIMES)      # already built here (key bc163af8…, 3 s, 115 MB); reused
view = store.prefix_view(full, times=SIAM_NB4_SCREEN_TIMES) if "-screen-" in A else store.slice_view(full, times=SIAM_NB4_CONTINUATION_TIMES)
score = score_siam_raw_run(run_path=run, seed_path=seed, reference_store_root=store.root, verified_reference=view)
json.dump(score, open(f"raw_outputs/{P}/{A}/score.json", "w"), indent=2, sort_keys=True)
s = score["summary"]; print(A, s["classification"], "burden", s["aggregate_accuracy_burden"], "broad_guard", s["broad_response_guard_pass"], "route", score["method"]["exchange_route_bundle"])
EOF
```
The receipt validator and the scorer both fail closed on a wrong source
revision, seed, grid or route; a failure is reported, never worked around.

**8.3 Decide the next job for that arm** (each arm climbs on its own):

| segment result | next action for that arm |
|---|---|
| screen `qualified` | prepare and submit the **continuation** of the same rung (8.4) |
| screen not qualified (outside the broad guard or a target fails) | prepare and submit the **next rung's screen** (8.4, `r+1`) |
| continuation `qualified` | stitch, estimator work, and — once the other arm also has a qualified t = 5 — classify the pair (8.5) |
| continuation not qualified | next rung's screen for that arm |
| r10 reached without qualifying | stop; report |

**8.4 Prepare the next package** (commit first; the packager refuses a dirty
`pipelines/` or `src/`; `SEG` is `screen_t0_t1_101` or `continue_t1_t5_401`,
`CUT` the rung's cut from the ladder above, `REV` the branch head's full
hash):
```bash
PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python - <<EOF
from pathlib import Path
from pipelines.time_dynamics.siam_chtc_package import prepare_siam_nb4_chtc_action
from pipelines.time_dynamics.siam_nb4_ladder import siam_nb4_action_id
ROUTE = "paper_ii_exchange_q_compiled_v1"; LANE, ARM, SEG, CUT, REV = "$LANE", "$ARM", "$SEG", $CUT, "$REV"
aid = siam_nb4_action_id(arm=LANE, segment=SEG, activation_cut=CUT, route_id=ROUTE)
tok = aid.split("-")[-2] + "_" + aid.split("-")[-1]          # e.g. r01_cut3em03
kind = "screen" if SEG.startswith("screen") else "continue"
pkg = Path(f"chtc/siam_nb4_qc_{LANE}_{kind}_{tok}_source_{REV[:8]}_$(date +%Y%m%d)")
extra = {}
if kind == "continue":                                     # bind to the fetched screen
    extra = dict(resume_from_run_json="raw_outputs/<screen package>/<screen action id>/run.json",
                 resume_action_receipt_json="raw_outputs/<screen package>/<screen action id>/action_receipt.json")
p = prepare_siam_nb4_chtc_action(repo_root=".", package_dir=pkg,
    seed_path="chtc/siam_nb4_qc_full_apm_screen_r00_cut1em02_source_2163171b_20260905/input/seed.json",
    image_path="chtc/time_dynamics_optuna/image.sif", source_revision=REV, action_id=aid,
    activation_cut=CUT, arm=ARM, segment=SEG, route_id=ROUTE, request_cpus=2,
    request_memory_gb=2, request_disk_gb=40,
    max_runtime_seconds=43200 if kind == "screen" else 172800, **extra)
print(p.package_dir, p.action_sha256)
EOF
```
Set `request_memory_gb` to twice the screen's observed `MemoryUsage` (GB,
rounded up) when that exceeds 1 GB. Validate with
`chtc/validate_condor_submit_lifecycle.py <pkg>/submit.sub`, commit the
package directory, rsync it with the same `-azR` form as §3, then ask for
"submit". Continuations are 48 h bounded (`172800`), inside the 72 h hold.

**8.5 Close a rung pair** (both arms have a qualified continuation):
```bash
PYTHONPATH=. /Users/lordskyler/Holstein_test/.venv/bin/python - <<EOF
import json, subprocess
from pipelines.time_dynamics.diagnostics.siam_segment_stitching import stitch_siam_nb4_segments
from pipelines.time_dynamics.diagnostics.siam_offline_scoring import score_siam_raw_run, classify_siam_matched_pair
SEED = "chtc/siam_nb4_qc_full_apm_screen_r00_cut1em02_source_2163171b_20260905/input/seed.json"
OUT = "output/siam_screening/nb4_qc_v1"; scores = {}
for arm, screen_run, cont_run in (("avqds", "<avqds screen run.json>", "<avqds continuation run.json>"),
                                  ("exchange_q_compiled", "<full_apm screen run.json>", "<full_apm continuation run.json>")):
    stitched = f"{OUT}/{arm}_t0_t5_501.json"
    stitch_siam_nb4_segments([screen_run, cont_run], seed_path=SEED, output_path=stitched)
    scores[arm] = score_siam_raw_run(run_path=stitched, seed_path=SEED, reference_store_root=f"{OUT}/exact_references")
    json.dump(scores[arm], open(f"{OUT}/{arm}_t0_t5_501.score.json", "w"), indent=2, sort_keys=True)
    subprocess.run(["/Users/lordskyler/Holstein_test/.venv/bin/python", "pipelines/time_dynamics/diagnostics/siam_estimator_work.py",
                    "--trajectory-json", stitched, "--seed-json", SEED, "--arm", arm,
                    "--output", f"{OUT}/{arm}_t0_t5_501.estimator_work.json"], check=True, env={"PYTHONPATH": "."})
pair = classify_siam_matched_pair(scores["avqds"], scores["exchange_q_compiled"], expected_apm_route="paper_ii_exchange_q_compiled_v1")
json.dump(pair, open(f"{OUT}/matched_pair.json", "w"), indent=2, sort_keys=True)
m = pair["matched_accuracy"]; print(m["classification"], "ratio", m["burden_ratio"], {a: scores[a]["summary"]["aggregate_accuracy_burden"] for a in scores})
EOF
```
`qualified_pair` → the Nb = 4 result is complete: commit the stitched
trajectories, scores, estimator records and `matched_pair.json` under
`chtc/siam_nb4_qc_result_<date>/` and report the two estimator-work totals.
Ratio above 2.0 (pair unmatched) → the arm with the **larger** burden climbs
one rung (8.4) and the other arm's t = 5 result is kept; classify again when
the new t = 5 arrives.

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
- SIAM on the compiled-Q route, this branch: engine `fc2de280` (empty-ansatz
  `nq` fix, `selection_route_bundle`, `Arm.dynamics_policy`, SIAM arm
  registration), runner fields `e23f35fe`, route identity through scorer /
  ladder / stitcher / bindings `3d45cba1` `ccab8ada`, estimator adapter
  `a2ab82ac` `543d4d48` `c90d6847`, packager v5 route lineage `26fd877f`,
  Nb = 2 evidence `2163171b`, r00 packages `67f3a05e`.
- SIAM Nb = 2 compiled-Q pair, this machine: AVQDS 55 s / 105 MB, compiled-Q
  165 s / 166 MB (`--compile-workers 1`); exact reference key `03c64995…`,
  data sha `aae315a4…`; score schema `siam_offline_accuracy_score_v4`.
- Test baseline on the survey branch: 334 passed, 0 failed for the
  §3 list; `test/test_staged_campaign_chtc.py` has 2 failures and
  `test/test_siam_stable_bindings.py` 4, all pre-existing on unmodified `main`
  (the latter read an untracked planner receipt that exists only in the main
  checkout).
- Image: sha256 `fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f`,
  216,371,200 B, from `~/chtc_rescue/home/Holstein_siam_nb4_source_ba50339c_20260828/chtc/time_dynamics_optuna/image.sif`.
- Remote: `/home` had 4.9 GB headroom on 2026-09-03; no `/staging`; 138 held
  jobs belong to other campaigns and are never touched.

## 20260905T214114Z — 15-minute post-submission heartbeat (manual)

- `9720203`: {'running': 11}
- `9720204`: not in queue
- `9720205`: not in queue
- `9720069`: {'running': 1}

Decisions: none required. Fetches: none.

## 20260905T214344Z — SIAM r00 screens: startup failure and once-only resubmission

- `9720204.0` and `9720205.0` completed ExitCode 1 in 0.1–0.2 min, 0 MB: `_verify_source_tree_contract` found `pipelines/artifacts/__init__.py` missing on the remote. Cause: the staging rsync's `--exclude 'artifacts'` (from the §3 recipe, intended for output dirs) also dropped the source package `pipelines/artifacts/`. The HH package is unaffected (it extracts its own source tarball).
- Fix: re-uploaded `pipelines/artifacts/`; verified every file in both packages' source-tree contracts is present on the remote. Resubmitted each once with an identical submit description (§7 input-transfer class; attempt 2 of 2). Receipts: `output/chtc_monitor/resource_retry_9720204_0.json`, `resource_retry_9720205_0.json`.
- resubmitted:   siam_nb4_qc_avqds_screen_r00_cut1em02_so: cluster 9720211
- resubmitted:   siam_nb4_qc_full_apm_screen_r00_cut1em02: cluster 9720212
- `9720203`: 11/11 running; `.0` image 0.87→1.44 GB over 10 min (executing).

## 20260905T214824Z — SIAM r00 second startup failure: source-lock mismatch (no further retry)

- `9720211.0`, `9720212.0`: ExitCode 1 at 16 s / 11 s. `_verify_source_tree_contract` line 531: `SIAM source-tree bytes differ: pipelines/time_dynamics/campaigns/paper_ii_corrected_exchange_cells_v1.json`. The contract hashes file contents; the packages are locked to `a3d1cebe` while the uploaded tree is the branch head (campaign files changed at `a43bcd86`). Attempt 2 of 2 exhausted; not resubmitted. Fix is a rebuild of both packages on the uploaded tree's revision, not a retry.
## 20260905T214927Z — SIAM r00 rebuilt on the uploaded revision; remote pre-flight clean

- Packages `chtc/siam_nb4_qc_avqds_screen_r00_cut1em02_source_812d20c3_20260905` and `..._full_apm_..._source_812d20c3_20260905` (`46df0ab1`), source-locked to `812d20c3` whose `pipelines/src` equals the tree on ap2001. Exchange arm on `siam_nb4_pilot_exchange_v1` (frontier 8), AVQDS default.
- Uploaded. All 809 `(path, sha256)` rows of the source-tree contract verified on the remote: 0 mismatches. LIFECYCLE_OK both; dry-run 1 job each. Awaiting user "submit".
- `9720203`: 11 running. `9720069.3`: running.
## 20260905T215118Z — SIAM r00 rebuilt packages submitted (standing authorization)

- AVQDS screen → `9720216`; exchange screen (frontier 8) → `9720217`. Source lock `812d20c3`, byte contract pre-verified on the remote. These replace `9720204/9720211` and `9720205/9720212`, all of which failed at startup and hold no outputs.
## 20260905T230840Z — SIAM r00 screens complete, fetched, receipt-valid

- `9720216.0` AVQDS: exit 0, 5.3 min, 196 MB; 25,700 steps, terminal support 15, 11 accepted patches. Receipt valid vs `812d20c3`, profile `siam_nb4_pilot_v1`.
- `9720217.0` exchange (frontier 8): exit 0, 61.1 min, 977 MB; 25,700 steps, terminal support 25, 19 accepted, **0 true exchanges**. Receipt valid vs `812d20c3`, profile `siam_nb4_pilot_exchange_v1`.
- Frontier telemetry on the exchange screen:
  - decisions by face: {'insert_only': 1, 'delete_only': 81, 'full': 18}
  - true_exchange_face_open: {'False': 82, 'True': 18}
  - exchange_frontier present at decision: {'frontier set': 19, 'frontier None': 82}
  - admitted families: {'singleton_d0': 100, 'singleton_d1': 99, 'singleton_d2': 93, 'singleton_d3': 79, 'singleton_d4': 72, 'singleton_d5': 62, 'singleton_d6': 24, 'singleton_d7': 8, 'singleton_d8': 7, 'singleton_d9': 5, 'singleton_d10': 1}
  - guard rejections (family:size): {'singleton_d7:None': 15, 'singleton_d6:None': 35, 'singleton_d5:None': 9, 'singleton_d4:None': 5}
  - materialized attempt kinds: {'insert': 19}
  - committed reasons: {'accepted_generalized_exchange_insert': 19}
- Fetched to `raw_outputs/<package>/<action_id>/` in the campaign worktree (outputs land there, not under `archives/`). Not scored yet (§8.2 scoring is the next step, needs the Nb=4 exact-reference store).
- `9720203`: 11 running, peak 2.2 GB. `9720069.3`: running, 3.2 GB.
- Scheduled heartbeat dispatched 22:05:52Z, no transcript, no log block: presumed stalled on a permission prompt. Manual heartbeats continue from the session.
## 20260905T231400Z — heartbeat; first corrected HH cell back; SIAM r00 scored; zero-exchange mechanism

- `9720203.4` strong--mid $(1.2,2)$ $\delta\theta$: exit 0, 94 min, 1465 MB. Archived on the remote, fetched to `archives/paper_ii_corrected_exchange_v1_a43bcd86_20260905/`, receipt `code_revision a43bcd86`. **2 true exchanges** at $t=0.04$ ($|D|=2$, $|A|=1$) and $t=0.08$ ($|D|=1$, $|A|=1$); terminal $N_\theta=42$; 960,117 steps. None after $t=0.08$.
- `9720203` remaining 10: running, 1.4--2.1 GB, 1 h 41 min. `9720069.3`: running, 6 h 30 min, 3.2 GB.
- SIAM r00 §8.2 scores (`raw_outputs/<pkg>/<action>/score.json`, reference store key `bc163af8`): AVQDS `candidate_outside_broad_guard`, burden 3.855, broad guard fail; exchange `candidate_outside_broad_guard`, burden 5.659, broad guard fail. §8.3: neither screen qualifies at cut $10^{-2}$; both arms step to r01 (cut $3.33\times10^{-3}$) screens.
- Zero exchanges on the SIAM screen, mechanism (Paper II Eq. deletion_ray_bound): the permission test admits a deletion batch $D$ only when $\sin[\min(\pi/2,\sum_{\mu\in D}|\theta_\mu c_\mu|)]$ lies below the *post-refit* ray tolerance $2\times10^{-3}$ (`exchange_integration.py:202` reuses `certification.ray_distance_max`). Only near-zero-angle coordinates are ever deletable.
  - SIAM exchange screen: 1,929,795 deletion sets evaluated, 1,929,781 angle-rejected, 14 Schur-evaluated, 8 Schur-rejected, **6 permitted**, all at $t\le0.40$. At those 6 full-face checkpoints the pure append ranked first and certified, so no exchange finalist was materialized. Materialized attempts: 19 inserts.
  - HH strong--mid $\delta\theta$: 4,706,498 evaluated, 166 permitted (0.004%); both exchanges while the deleted coordinates were still dormant.
  - Singleton pass fraction of the angle bound over (checkpoint, coordinate) pairs at $c_\mu=1$: SIAM 2.1% at $2\times10^{-3}$, 22% at $5\times10^{-2}$, 40% at $10^{-1}$; HH 0.4%, 4.2%, 7.6%.
  - Consequence for the locked results: every deletion in Paper II's reported trajectories, maintenance included, removed a coordinate with $|\theta_\mu c_\mu|\lesssim2\times10^{-3}$.
- Added `exchange_permission_ray_distance_max` (`--exchange-permission-ray-distance-max`), default None = the historical coupling, provenance byte-identical when undeclared; a declared value changes only the pre-permission bound, the post-refit certificates (Eqs. patch_ray_distance, patch_velocity_continuity) unchanged. Local sweep on strong--fast $t=0\to2$: control, $2\times10^{-2}$, $5\times10^{-2}$, $10^{-1}$ (`/tmp/perm_sweep/`). Adoption of a value is the user's decision.
## 20260905T234500Z — pre-permission threshold sweep, strong--fast $t=0\to2$ (local, smoke controller)

$(2.4,3)$, 51 points, profile `paper_declared_exchange_v1`, head `2c25b0bf`; control = threshold undeclared ($2\times10^{-3}$ coupling). Errors against the exact reference (`/tmp/perm_sweep/exact_references`), units $10^{-4}$.

| $\tau_{\rm pre}$ | true exch. | at $t$ | $N_\theta$ | steps | $\overline{|\Delta E|}$ | $\overline{|\Delta D|}$ | $\overline{|\Delta n|}$ | permitted/evaluated | del.-bearing finalists tested/certified | post-refit ray rejections |
|---|---:|---|---:|---:|---:|---:|---:|---|---|---|
| $2\times10^{-3}$ (control) | 1 | 0.56 | 40 | 1130 | 91.2 | 15.5 | 20.2 | 29/469,720 | 8/8 | 0 |
| $2\times10^{-2}$ | 2 | 0.12, 0.56 | 37 | 912 | 516 | 77.7 | 124 | 91/1,313,341 | 21/11 | 10 |
| $5\times10^{-2}$ | 4 | 0.12, 0.16, 0.20, 0.60 | 34 | 792 | 571 | 143 | 145 | 1,431/1,002,149 | 147/16 | 131 |
| $10^{-1}$ | 4 | 0.12, 0.16, 0.20, 0.84 | 27 | 776 | 15.5 | 10.4 | 6.78 | 4,229/1,102,044 | 207/14 | 193 |

The post-refit ray certificate (Eq. patch_ray_distance, $2\times10^{-3}$) remains the authority: it rejected 131--193 deletion-bearing finalists once the pre-bound was widened. Accuracy is not monotone in $\tau_{\rm pre}$ on this horizon; one cell at $t=2$ under the smoke controller does not fix a value. Next: the same sweep at $t=20$ under the production $\delta\theta$ controller of the campaign cell (control = `9720203.8`, running on CHTC).
## 20260905T235500Z — SIAM AVQDS r01 screen submitted; $t=20$ threshold sweep launched

- `chtc/siam_nb4_qc_avqds_screen_r01_cut3em03_source_2c25b0bf_20260905` (`04994d0d`, cut $3.33\times10^{-3}$, 2 CPU / 2 GB / 40 GB / 43200 s). Tree `pipelines src` re-uploaded to the corrected staging dir (no excludes); all 809 contract rows verified on the remote; LIFECYCLE_OK; submitted → **`9720251`**. Running HH jobs transfer only `chtc/<pkg>` + image, so the tree refresh does not touch them.
- Exchange-arm r01 not prepared: at the coupled threshold it would repeat the zero-exchange screen; it waits on the $\tau_{\rm pre}$ decision.
- Local $t=20$ sweep on the production strong--fast $\delta\theta$ argv of `9720203.8` (workers 2/3 instead of 4/8, results unaffected) with $\tau_{\rm pre}\in\{2\times10^{-2},5\times10^{-2},10^{-1}\}$: `/tmp/perm_sweep/t20/sf20_<tag>.{json,log}`, ram_guard 4 GB each. Control is `9720203.8` itself.
- Watch list now: `9720203` (10), `9720069.3`, `9720251`.
## 20260905T233500Z — heartbeat: three corrected HH cells back; AVQDS r01 running

| action | exit | wall | MemoryUsage | true exchanges (time points) | $N_\theta$ | steps | permitted/evaluated deletion sets |
|---|---:|---:|---:|---|---:|---:|---|
| `midres__delta_theta_5e-5` | 0 | 94 min | 1465 MB | 2 ($t=0.04, 0.08$) | 42 | 960,117 | 166/4,706,498 |
| `fastweak__state_motion_5e-5` | 0 | 110 min | 1709 MB | 1 ($t=0.08$) | 43 | 499,108 | 209/5,117,416 |
| `midres__state_motion_5e-5` | 0 | 109 min | 1709 MB | 3 ($t=0.04, 0.08, 7.72$) | 40 | 501,412 | 200/4,591,935 |

- All three archived on the remote and fetched to `archives/paper_ii_corrected_exchange_v1_a43bcd86_20260905/`, receipts `code_revision a43bcd86`. The $t=7.72$ commit on strong--mid state motion is the first true exchange after the opening time points under the coupled threshold.
- `9720203`: 8 running (`.0 .1 .2 .6 .7 .8 .9 .10`), 1.5--2.2 GB. `9720069.3`: running. `9720251` (AVQDS r01): running, just started.
- Local $t=20$ sweep: 1--3 of 500 time points after 10 min at workers 2/3 — 25--40 h projected; too slow. Moving the sweep to CHTC.
## 20260906T000500Z — pre-permission threshold pilot submitted to CHTC; local $t=20$ sweep stopped

- Locks `pipelines/time_dynamics/campaigns/paper_ii_permission_threshold_{2em02,5em02,1em01}_cells_v1.json` (`aa863835`): the strong--fast $(2.4,3)$ $\delta\theta$ cell of the corrected lock, verbatim, with `extra_flags` declaring `--exchange-permission-ray-distance-max` at $2\times10^{-2}$, $5\times10^{-2}$, $10^{-1}$. Loader/packager accept exactly that declaration and assert it on every argv; the corrected lock's actions are unchanged (53 tests).
- Packages `chtc/paper_ii_permission_threshold_{2em02,5em02,1em01}_v1_2c769d52_20260905` (`dc32da45`), source-locked to `2c769d52` (tarball `66938ce6…`), same slot as the corrected campaign (9 CPU / 12 GB / 40 GB / 172800 s). Each argv differs from the corrected strong--fast $\delta\theta$ action only by the flag pair, output path, and code revision. Validated locally and on the remote with the container sha; LIFECYCLE_OK.
- Submitted: **`9720253`** ($10^{-1}$), **`9720254`** ($2\times10^{-2}$), **`9720255`** ($5\times10^{-2}$). Control is `9720203.8` (running). Local `/tmp/perm_sweep/t20/` runs killed (they were mine; 1--3 of 500 time points after 20 min).
- Watch list: `9720203` (8), `9720069.3`, `9720251`, `9720253`, `9720254`, `9720255`.
## 20260906T004500Z — what the stored certification telemetry says about the pre-permission threshold

`pipelines/time_dynamics/diagnostics/deletion_permission_response.py` replays each time point's selector rounds on the entry support and pairs every materialized deletion-bearing finalist's accumulated effective angle $a_D=\sum_{\mu\in D}|\theta_\mu c_\mu|$ (Eq. deletion_angle_sum) with the post-refit ray distance and velocity value its certification recorded. Ten runs: the four $t=2$ sweep runs, the three corrected cells back so far, three locked $t=20$ cells. 441 finalists; 43 unattributed (coefficient of a label absent from every final support). Entry angles are recorded for state-motion runs and first-order estimated ($\theta+\Delta t\,\dot\theta$) for $\delta\theta$ runs (15 rows).

| $a_D$ | finalists materialized | certified | post-refit ray $\le2\times10^{-3}$ | median post-refit ray | median ray / $a_D$ |
|---|---:|---:|---:|---:|---:|
| $=0$ (coordinate appended earlier in the same time point) | 3 | 3 | 3 | 0 | -- |
| $(0,\,2\times10^{-3})$ | 71 | 69 (97%) | 69 (100%) | $2.1\times10^{-4}$ | 0.35 |
| $[2\times10^{-3},\,5\times10^{-3})$ | 18 | 7 (39%) | 7 (39%) | $2.1\times10^{-3}$ | 0.70 |
| $[5\times10^{-3},\,10^{-2})$ | 15 | 2 (13%) | 2 (13%) | $3.4\times10^{-3}$ | 0.50 |
| $[10^{-2},\,2\times10^{-2})$ | 33 | 5 (15%) | 4 (12%) | $8.6\times10^{-3}$ | 0.59 |
| $[2\times10^{-2},\,5\times10^{-2})$ | 123 | 11 (9%) | 11 (9%) | $1.3\times10^{-2}$ | 0.39 |
| $[5\times10^{-2},\,10^{-1})$ | 135 | 3 (2%) | 3 (2%) | $3.4\times10^{-2}$ | 0.53 |

- The bounded local refit (Eq. patch_ray_refit) recovers a constant fraction, not orders of magnitude: post-refit ray $\approx0.35$--$0.7\,a_D$. The post-refit certificate at $2\times10^{-3}$ therefore admits deletions up to $a_D\approx4$--$6\times10^{-3}$ routinely and only exceptional coordinates beyond that. A pre-permission threshold above $\sim10^{-2}$ mostly manufactures rejected finalists (each costing a full patched geometry plus refit in $\mathcal E$), bounded by the 12-per-level / 2-per-branch certification caps.
- Every accepted exchange under the coupled bound removed a coordinate with $a_D<2\times10^{-3}$: strong--mid $\delta\theta$ 2 dormant, weak--fast state motion 3 dormant, strong--mid state motion 3 dormant; the $t=2$ control's single exchange removed a coordinate appended two rounds earlier in the same time point. The $t=2$ run at $10^{-1}$ is the only one whose exchanges removed rotated coordinates (4).
- Pilot extended with $\tau_{\rm pre}=5\times10^{-3}$ and $10^{-2}$ (locks `paper_ii_permission_threshold_{5em03,1em02}_cells_v1`), so the strong--fast $\delta\theta$ pilot spans $\{2\times10^{-3}$ control, $5\times10^{-3}, 10^{-2}, 2\times10^{-2}, 5\times10^{-2}, 10^{-1}\}$.
## 20260906T015500Z — heartbeat after the tunnel reopened: 8/11 corrected cells back; pilot at six values; SIAM r01 scored, r02 submitted

| corrected cell | exit | wall | MemoryUsage | true exchanges (time points) | $N_\theta$ | steps | permitted/evaluated |
|---|---:|---:|---:|---|---:|---:|---|
| `fastfast__delta_theta_5e-5` | 0 | 211 min | 1709 MB | 3 ($0.04, 0.24, 1.36$) | 44 | 1,390,854 | 356/5,403,601 |
| `fastfast__state_motion_5e-5` | 0 | 192 min | 1709 MB | 3 ($0.04, 0.24, 1.36$) | 51 | 505,956 | 314/6,845,879 |
| `fastweak__delta_theta_5e-5` | 0 | 156 min | 1709 MB | 3 ($0.64, 4.64, 8.68$) | 50 | 523,605 | 405/8,122,596 |
| `slowstrong__delta_theta_5e-5` | 0 | 127 min | 1954 MB | 4 ($0.84, 1.40, 2.88, 9.08$) | 50 | 748,635 | 233/7,291,139 |
| `strongfast__delta_theta_5e-5` (pilot control) | 0 | 131 min | 1465 MB | 4 ($0.84, 0.92, 3.36, 11.24$) | 42 | 1,589,477 | 129/4,758,651 |
| earlier: `midres` $\delta\theta$ / sm, `fastweak` sm | 0 | 94--110 min | 1.5--1.7 GB | 2 / 3 / 1 | 42 / 40 / 43 | | |

- Still running: `9720203.0` weak--slow $\delta\theta$, `.1` weak--slow state motion, `.7` strong--slow state motion (2.4--2.7 GB, > 9 h). Under the coupled bound every accepted exchange removed a coordinate the permission test had bounded at $a_D\le2\times10^{-3}$; the diagnostic's "rotated" labels on $\delta\theta$ cells (7 of 31 commits) are first-order entry-angle estimates, not recorded angles, and are not evidence against that bound.
- Pilot: `9720454` ($5\times10^{-3}$), `9720455` ($10^{-2}$) submitted (`1e7aa28f`, tarball `5e79c267…`); `9720253/54/55` running (1.7 GB). Six values now on the strong--fast $\delta\theta$ cell including the finished control.
- SIAM AVQDS r01 (`9720251`): exit 0, 7 min, 269 MB; receipt VALID vs `2c25b0bf`; `candidate_outside_broad_guard`, burden 0.916 (r00: 3.855), broad guard fail → r02 (cut $10^{-3}$) prepared on `1e7aa28f` (`d554d78a`), tree re-uploaded, 815 contract rows verified, LIFECYCLE_OK, submitted → **`9720456`**. Exchange-arm ladder still held for the $\tau_{\rm pre}$ decision.
- Codex anchors `9720069.3` (exit 0, 490 min, 3.4 GB) and `.4` fetched to `archives/codex_gapfill_weakslow_20260905/` (kept separate): source `426807d9`, no frontier, weak--slow state motion $N_\theta=59$ / state-motion-plus-parameter $N_\theta=62$, **0 true exchanges** each. The corrected `9720203.1` supersedes them for the pending Table I row.
- Watch list: `9720203` (3), `9720253`, `9720254`, `9720255`, `9720454`, `9720455`, `9720456`.

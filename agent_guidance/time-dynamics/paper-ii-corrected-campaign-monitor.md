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
## 20260906T021500Z — corrected cells against the locked Table I values (8 of 11)

Errors against the exact reference (`output/paper_ii_staged/exact_references`, 501 points), units $10^{-4}$; $\mathcal E$ in $10^{8}$ (all `exact`); $N_{\rm step}$ in $10^{3}$. "locked → corrected".

| cell | $\overline{\|\Delta E\|}$ | $\overline{\|\Delta D\|}$ | $\overline{\|\Delta n\|}$ | $N_\theta$ | $N_{2q}$ | $D_{2q}$ | $D_c$ | $\mathcal E$ | $N_{\rm step}$ | exch. |
|---|---|---|---|---|---|---|---|---|---|---:|
| weak--fast $\delta\theta$ | 8.18 → 16.2 | 1.81 → 3.03 | 6.31 → 4.08 | 45 → 50 | 130 → 152 | 91 → 115 | 309 → 402 | 10.8 → 9.37 | 709 → 524 | 3 |
| weak--fast state motion | 8.96 → 3.73 | 3.01 → 2.00 | 6.26 → 1.82 | 49 → 43 | 150 → 164 | 128 → 128 | 463 → 442 | 8.37 → 6.83 | 498 → 499 | 1 |
| weak--fastest $\delta\theta$ | 5.73 → 5.41 | 0.93 → 2.77 | 4.09 → 2.37 | 47 → 44 | 159 → 117 | 125 → 85 | 430 → 322 | 17.2 → 17.1 | 1279 → 1391 | 3 |
| weak--fastest state motion | 17.1 → 6.17 | 2.72 → 1.29 | 6.60 → 1.84 | 49 → 51 | 162 → 169 | 128 → 112 | 450 → 407 | 7.78 → 7.95 | 504 → 506 | 3 |
| strong--mid $\delta\theta$ | 1.03 → 6.52 | 1.89 → 2.23 | 3.53 → 1.25 | 49 → 42 | 145 → 170 | 116 → 131 | 426 → 437 | 17.2 → 10.5 | 1084 → 960 | 2 |
| strong--mid state motion | 5.80 → 2.71 | 1.87 → 2.59 | 2.86 → 1.44 | 48 → 40 | 152 → 154 | 130 → 117 | 478 → 406 | 8.79 → 5.92 | 500 → 501 | 3 |
| strong--slow $\delta\theta$ | 3.40 → 2.98 | 2.54 → 3.52 | 2.56 → 3.68 | 50 → 50 | 187 → 158 | 150 → 132 | 498 → 472 | 11.3 → 11.7 | 700 → 749 | 4 |
| strong--fast $\delta\theta$ | 3.50 → 4.30 | 1.44 → 1.45 | 4.71 → 2.32 | 44 → 42 | 144 → 132 | 113 → 98 | 432 → 365 | 14.0 → 16.7 | 1227 → 1589 | 4 |

Matched-accuracy status against the locked AVQDS comparators is not re-derived here; the table generator on `claude/paper-ii-main-results-20260904` does that once all eleven cells are in.
## 20260906T024500Z — corrected-vs-reported PDF (8 of 11 cells)

`agent_guidance/time-dynamics/reports/paper_ii_corrected_vs_reported_20260906.pdf` (+ `.tex`, `comparison_corrected_20260906.json`). Generator `pipelines/time_dynamics/diagnostics/paper_ii_corrected_campaign_report.py`; inputs: the locked comparison the manuscript tables came from (`four_route_analysis_exact_20260904_partial45/comparison.json`), a comparison built by `paper_ii_four_route_comparison.py` over `archives/paper_ii_corrected_exchange_v1_a43bcd86_20260905` plus the eleven locked AVQDS comparators (artifact roots: `chtc_rescue/home`, `codex-paper-ii-dual-selector-overnight-20260903`, `codex-paper-ii-coverage-automation-20260901`), terminal costs from the archives' `terminal_qiskit_cost.json`, and the manuscript's generated table for the recovered strong--fast AVQDS row. Contents: Table I columns as reported / corrected / AVQDS rows with the matched-accuracy verdict, generator-level Table II for both APM trajectories with exchange commits, and per-cell observable and error panels. Matched against AVQDS (factor two, componentwise): corrected weak--fast state motion, strong--mid both, weak--fastest both pass; weak--fast $\delta\theta$ and strong--slow $\delta\theta$ fail (worst factor 2.82 / 2.83); strong--fast has no AVQDS trajectory in either comparison (recovered row only).

Regenerate (after new cells are fetched and extracted under `archives/`):
```
PYTHONPATH=. .venv/bin/python pipelines/time_dynamics/diagnostics/paper_ii_four_route_comparison.py --repository-root . --direct-root archives/paper_ii_corrected_exchange_v1_a43bcd86_20260905 --direct-root /tmp/report/avqds_root --artifact-root /Users/lordskyler/chtc_rescue/home --artifact-root /Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-dual-selector-overnight-20260903 --artifact-root /Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-coverage-automation-20260901 --output-json /tmp/report/comparison_corrected.json
PYTHONPATH=. .venv/bin/python pipelines/time_dynamics/diagnostics/paper_ii_corrected_campaign_report.py --locked-comparison <partial45>/comparison.json --corrected-comparison /tmp/report/comparison_corrected.json --locked-cost-root <partial45>/terminal_costs --corrected-cost-root /tmp/report/terminal_costs_corrected --manuscript-table <Paper II>/generated/paper_ii_t20_q_compiled_frontier_table.tex --output-dir /tmp/report/out --compile
```
(`/tmp/report/avqds_root` is a copy of the eleven `<cell>__avqds` run directories of the locked telemetry campaign; `/tmp/report/terminal_costs_corrected/<cell>/q_compiled.json` is each archive's `terminal_qiskit_cost.json`.)
## 20260906T030500Z — archive collision on the remote; next-rung reruns, 2e-2 rerun, SIAM r03 submitted

- **Defect found.** Every campaign package stages `archives/<action_id>.{tar.gz,archive.json}` into the shared submit-directory `archives/`, and the pilot packages reuse the corrected campaign's action id `strongfast__delta_theta_5e-5__exchange_q_compiled`. The $5\times10^{-2}$ pilot (`9720255`, 148 min) overwrote the $2\times10^{-2}$ pilot (`9720254`, 150 min) and the corrected control's remote copy; the control had already been fetched (receipt `a43bcd86`), the $2\times10^{-2}$ archive is lost. Fixes: (i) a remote watcher `archive_watch.sh` (pid file `archive_watch.pid`) copies every `(tar, receipt)` arrival into `archives_by_receipt/<archive_sha256>/` within 5 s — 8 preserved at start; (ii) packages built from `e38eff4c` on name archives `<campaign_id>__<action_id>.*`. The still-running $10^{-1}$, $5\times10^{-3}$, $10^{-2}$ pilots (`9720253/9720454/9720455`) use the old names and rely on the watcher.
- $5\times10^{-2}$ pilot fetched to `archives/paper_ii_permission_threshold_5em02_v1_2c769d52_20260905/`: 8 true exchanges, $N_\theta=40$, 1,775,269 steps, 4,300/3,698,861 deletion sets permitted (control 129/4,758,651).
- Submitted (standing authorization): `9720482` next activation-threshold rung for the two corrected cells that failed the matched-accuracy rule — weak--fast $\delta\theta$ $\tau_m$ $3\times10^{-7}\to10^{-7}$, strong--slow $\delta\theta$ $3\times10^{-6}\to10^{-6}$ (lock `paper_ii_corrected_exchange_next_rung_cells_v1`, package `chtc/paper_ii_corrected_exchange_next_rung_v1_4b27eb00_20260906`); `9720483` the $2\times10^{-2}$ pilot rerun (`chtc/paper_ii_permission_threshold_2em02_rerun_v1_5926f4c9_20260906`); `9720484` SIAM AVQDS r03 (cut $3.33\times10^{-4}$, `96bad0b4`, 821 contract rows verified) after r02 scored `candidate_broad_guard` (burden 0.438, broad guard pass, individual target fail).
- Watch list: `9720203` (3), `9720253`, `9720454`, `9720455`, `9720482` (2), `9720483`, `9720484`. Fetch from `archives_by_receipt/` (by receipt) rather than `archives/`.
## 20260906T031500Z — first $t=20$ pilot point: $\tau_{\rm pre}=5\times10^{-2}$ is too loose

Strong--fast $(2.4,3)$ $\delta\theta$, errors against the exact reference in $10^{-4}$ ($\overline{|\Delta n|}$ as in the comparison tool, twice the mean maximal site-occupation error):

| run | true exchanges | $N_\theta$ | $N_{2q}$ | $D_{2q}$ | $D_c$ | $\mathcal E$ ($10^8$) | $N_{\rm step}$ ($10^3$) | $\overline{|\Delta E|}$ | $\overline{|\Delta D|}$ | $\overline{|\Delta n|}$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control, coupled bound (`9720203.8`) | 4 | 42 | 132 | 98 | 365 | 16.7 | 1589 | 4.30 | 1.45 | 4.64 |
| $\tau_{\rm pre}=5\times10^{-2}$ (`9720255`) | 8 | 40 | 111 | 84 | 301 | 16.9 | 1775 | 9.68 | 14.6 | 42.0 |

Twice the exchanges and an 18\% shallower terminal circuit, at a tenfold loss in doublon and charge-imbalance accuracy: deletions of rotated coordinates that pass the post-refit gates (ray $\le2\times10^{-3}$, velocity $\le10^{-3}$) still accumulate state error over 501 time points. The $5\times10^{-3}$ and $10^{-2}$ pilots (`9720454`, `9720455`) are the decisive ones.
## 20260906T032500Z — $10^{-1}$ pilot back; SIAM r03 scored; queue picture

- $\tau_{\rm pre}=10^{-1}$ (`9720253`, exit 0, 201 min, 1954 MB; preserved as `aa05498f`): **0 true exchanges**, $N_\theta=39$, $(N_{2q},D_{2q},D_c)=(110,76,271)$, $\mathcal E=21\times10^{8}$, $N_{\rm step}=2{,}441\times10^{3}$, errors $(21.7,\,12.3,\,35.4)\times10^{-4}$ — worse than the control on every accuracy axis and on $\mathcal E$ and $N_{\rm step}$. With $5\times10^{-2}$ (8 exchanges, tenfold accuracy loss) the wide-threshold branch is closed; $5\times10^{-3}$ and $10^{-2}$ (`9720454/9720455`, running) remain.
- SIAM AVQDS r03 (`9720484`, exit 0, 12 min): `candidate_outside_broad_guard`, burden 0.961 (r02 0.438 inside the guard, r01 0.916, r00 3.855) — not monotone in the cut; §8.3 → r04 (cut $10^{-4}$).
- Queue at 03:16Z (one accounting identity for every paper): 21 running, 5 idle, 144 held, 10 completed-in-queue. Ours: 8 running (`9720203` ×3, `9720454`, `9720455`, `9720482` ×2, `9720483`), 9 CPU each = 72 cores; nothing of ours idle. Codex's earlier Paper II jobs still running: `9718536` ×2, `9718542`, `9718545` (4 CPU), `9719355` ×4 (9 CPU), `9719356` ×4 (1 CPU; `.10` is the strong--fast AVQDS comparator) = 56 cores. Paper I: 5 idle (`9720239.0-4`, 4 CPU / 64 GB), last negotiator match attempt 22:15Z with 0 matching slots; 208 unclaimed slots with $\ge4$ CPU and $\ge64$ GB exist, and one machine running our job (`e4043`) has 220 free cores / 429 GB. Effective user priority 66221 with 132 cores in use, so every running job of this user, ours included, lowers the share under which the idle Paper I jobs are negotiated.
- Watch list: `9720203` (3), `9720454`, `9720455`, `9720482` (2), `9720483`, SIAM r04 once submitted.
## 20260906T033500Z — pre-repair gamma-route jobs removed (user direction)

`condor_rm 9719355.2 9719355.5 9719355.8 9719355.29 9718536.1 9718536.3 9718542.0 9718545.0` — Codex's ideal-$\Gamma$ (gamma-compiled) APM jobs on `426807d9` / `4bca80b0`, all pre-repair and on a route the manuscript no longer reports; 52 cores released for the idle Paper I jobs. Kept: `9719356.0/.1/.2/.10`, the fresh exact-estimator AVQDS comparators (weak--slow ×3, strong--fast $\delta\theta$; AVQDS has no exchange search). Their outputs land in `~/Holstein_paper_ii_final_estimator_telemetry_v1_426807d9_20260904/` (Codex's staging dir).
## 20260906T042000Z — SIAM AVQDS r04 scored; r05 submitted

- r04 (`9720495`, exit 0, 6 min, 269 MB, `4c7405bf`): `candidate_outside_broad_guard`, burden 0.981, $N_\theta=22$. Ladder so far: r00 3.855, r01 0.916, r02 **0.438 (inside the broad guard, individual target failed)**, r03 0.961, r04 0.981 — the burden bottomed at cut $10^{-3}$ and rises again with tighter cuts, so the activation cut alone is not driving the AVQDS arm to the target on this seed. §8.3 is followed to r05 (cut $3.33\times10^{-5}$) since each rung costs six minutes; if r05 does not improve on r02, the ladder decision for the AVQDS arm should be revisited rather than climbed to r10.
## 20260906T042500Z — $10^{-2}$ pilot back (`9720455`, exit 0, 143 min, 1709 MB; preserved as `6d3310f7`)

Strong--fast $(2.4,3)$ $\delta\theta$ at $t=20$, errors in $10^{-4}$ ($\overline{|\Delta n|}$ as twice the mean maximal site-occupation error):

| $\tau_{\rm pre}$ | true exch. | $N_\theta$ | $N_{2q}$ | $D_{2q}$ | $D_c$ | $\mathcal E$ ($10^8$) | $N_{\rm step}$ ($10^3$) | $\overline{|\Delta E|}$ | $\overline{|\Delta D|}$ | $\overline{|\Delta n|}$ | permitted / evaluated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| $2\times10^{-3}$ control | 4 | 42 | 132 | 98 | 365 | 16.7 | 1589 | 4.30 | 1.45 | 4.64 | 129 / 4.76M |
| $10^{-2}$ | 4 | 40 | 105 | 76 | 266 | 19.9 | 2068 | 5.92 | 2.12 | 16.5 | 784 / 4.12M |
| $5\times10^{-2}$ | 8 | 40 | 111 | 84 | 301 | 16.9 | 1775 | 9.68 | 14.6 | 42.0 | 4,300 / 3.70M |
| $10^{-1}$ | 0 | 39 | 110 | 76 | 271 | 21.0 | 2441 | 21.7 | 12.3 | 35.4 | -- |

At $10^{-2}$: same exchange count as the control, 27\% shallower terminal circuit, but $\overline{|\Delta n|}$ 3.6× worse, $\overline{|\Delta E|}$ 1.4× worse, $\mathcal E$ +19\%, $N_{\rm step}$ +30\%. Only $5\times10^{-3}$ (`9720454`) and the $2\times10^{-2}$ rerun (`9720483`) remain.
## 20260906T043000Z — strong--slow $\delta\theta$ at $\tau_m=10^{-6}$ back (`9720482.1`, exit 0, 83 min, 1709 MB; preserved as `161b3d04`)

| $\tau_m$ | $\overline{|\Delta E|}$ | $\overline{|\Delta D|}$ | $\overline{|\Delta n|}$ | $N_\theta$ | $\mathcal E$ ($10^8$) | $N_{\rm step}$ ($10^3$) | exchange commits | matched vs AVQDS (worst factor) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| $3\times10^{-6}$ (corrected) | 2.98 | 3.52 | 7.37 | 50 | 11.7 | 749 | 4 | no (2.83) |
| $10^{-6}$ | 4.52 | 2.05 | 2.64 | 46 | 9.15 | 621 | 4 | no (2.27) |

Doublon and charge-imbalance errors fall, the energy error rises; still outside the factor-two rule. Next rung $3\times10^{-7}$ submitted (lock `paper_ii_corrected_exchange_next_rung2_cells_v1`). Weak--fast $\delta\theta$ at $10^{-7}$ (`9720482.0`) still running.
## 20260906T043500Z — SIAM AVQDS r05 scored; r06 submitted; strong--slow rung $3\times10^{-7}$ running

- r05 (`9720664`, exit 0, 6 min, `7cf306c6`): `candidate_broad_guard`, burden 0.519, inside the broad guard, individual target failed; $N_\theta=32$. Normalized response components: $m_d$ 0.002, $D_d$ 0.007, $C_1$ 0.049, $C_2$ 0.044, **$C_3$ 2.07**, $C_4$ 0.081, fidelity 0.007 — one bath-correlator component carries the whole burden at every rung so far (r02 0.438, r03 0.961, r04 0.981, r05 0.519), and the activation cut does not move it monotonically. r06 (cut $10^{-5}$) submitted per §8.3; the remaining rungs cost six minutes each, but whether the AVQDS arm can reach the individual target on $C_3$ by the cut alone is a question for the user once r10 is reached or earlier.
- Strong--slow $\delta\theta$ at $\tau_m=3\times10^{-7}$: `9720666` (`85288674`), running.
## 20260906T045500Z — SIAM AVQDS r06 scored; r07 submitted

r06 (`9720667`, exit 0, 18 min, 293 MB, `864aece6`): `candidate_broad_guard`, burden 0.300, $C_3$ 1.20 (r05 2.07), $N_\theta=80$ (r05 32). The cut now moves $C_3$ at the price of a much larger support; r07 (cut $3.33\times10^{-6}$) submitted.
## 20260906T050000Z — weak--fast $\delta\theta$ at $\tau_m=10^{-7}$ back (`9720482.0`, exit 0, 115 min, 1954 MB; preserved as `c6afe073`)

| $\tau_m$ | $\overline{|\Delta E|}$ | $\overline{|\Delta D|}$ | $\overline{|\Delta n|}$ | $N_\theta$ | $\mathcal E$ ($10^8$) | $N_{\rm step}$ ($10^3$) | exchange commits | matched vs AVQDS (worst factor) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| $3\times10^{-7}$ reported | 8.18 | 1.81 | 6.31 | 45 | 10.8 | 709 | 0 | yes (1.42) |
| $3\times10^{-7}$ corrected | 16.2 | 3.03 | 8.17 | 50 | 9.37 | 524 | 3 | no (2.82) |
| $10^{-7}$ corrected | 15.4 | 2.88 | 8.20 | 45 | 11.0 | 581 | 2 | no (2.67) |

The energy error barely moves with the cut (AVQDS: 4.24); the corrected search commits different early patches on this drive than the reported one did. Third rung $3.33\times10^{-8}$ submitted (lock `paper_ii_corrected_exchange_next_rung3_cells_v1`) to follow the ladder; if it does not close the gap, this cell's corrected row stays unmatched and is reported as such.
## 20260906T052500Z — SIAM AVQDS r07 scored; r08 submitted

r07 (`9720671`, exit 0, 27 min, 293 MB, `34f89a50`): `candidate_broad_guard`, burden 0.324, $C_3$ 1.30 (r06 1.20), $N_\theta=90$; every other component $\le0.05$. $C_3$ has stopped responding to the cut. r08 (cut $10^{-6}$) submitted to close the ladder formally; r09/r10 follow the same way unless $C_3$ moves.
## 20260906T053500Z — $2\times10^{-2}$ pilot rerun back (`9720483`, exit 0, 148 min, 1709 MB; `47a6008b`)

$\tau_{\rm pre}=2\times10^{-2}$: 6 true exchanges, $N_\theta=36$, $(N_{2q},D_{2q},D_c)=(98,77,276)$, $\mathcal E=17.5\times10^8$, $N_{\rm step}=2{,}240\times10^3$, errors $(19.2,\,11.5,\,35.4)\times10^{-4}$; 1,238 / 18.1M deletion sets permitted. Same pattern as $5\times10^{-2}$ and $10^{-1}$: shallower circuit, order-of-magnitude accuracy loss. Only $5\times10^{-3}$ (`9720454`) is outstanding.
## 20260906T144000Z — morning consolidation: pilot complete at six values; 10/11 corrected cells; rungs; local runs stopped

**Pre-permission threshold pilot, strong--fast $(2.4,3)$ $\delta\theta$, $t=20$** (errors $10^{-4}$; $\overline{|\Delta n|}$ as twice the mean maximal site-occupation error):

| $\tau_{\rm pre}$ | cluster | true exch. | $N_\theta$ | $N_{2q}$ | $D_{2q}$ | $D_c$ | $\mathcal E$ ($10^8$) | $N_{\rm step}$ ($10^3$) | $\overline{|\Delta E|}$ | $\overline{|\Delta D|}$ | $\overline{|\Delta n|}$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $2\times10^{-3}$ (manuscript bound) | `9720203.8` | 4 | 42 | 132 | 98 | 365 | 16.7 | 1589 | 4.30 | 1.45 | 4.64 |
| $5\times10^{-3}$ | `9720454` (`3e871cf3`) | 4 | 31 | 103 | 89 | 321 | 11.5 | 1519 | 27.4 | 3.30 | 17.2 |
| $10^{-2}$ | `9720455` | 4 | 40 | 105 | 76 | 266 | 19.9 | 2068 | 5.92 | 2.12 | 16.5 |
| $2\times10^{-2}$ | `9720483` | 6 | 36 | 98 | 77 | 276 | 17.5 | 2240 | 19.2 | 11.5 | 35.4 |
| $5\times10^{-2}$ | `9720255` | 8 | 40 | 111 | 84 | 301 | 16.9 | 1775 | 9.68 | 14.6 | 42.0 |
| $10^{-1}$ | `9720253` | 0 | 39 | 110 | 76 | 271 | 21.0 | 2441 | 21.7 | 12.3 | 35.4 |

Every decoupled value loses at least a factor 3.6 on $\overline{|\Delta n|}$ and none keeps $\overline{|\Delta E|}$ within a factor two of the control; the post-refit certificates at $(2\times10^{-3},10^{-3})$ do not protect the trajectory once rotated coordinates become deletable. Exchange counts stay at 4--8 (0 at $10^{-1}$). On this cell the manuscript's coupled bound is the accuracy-preserving choice; adoption remains the user's decision.

**Corrected campaign** (10 of 11 back; strong--slow state motion `9720203.7` running, 16 h): weak--slow $\delta\theta$ (`9720203.0`, 505 min, 3.2 GB, `d2622832`): 32 true exchanges, $N_\theta=59$, errors $(2.17, 1.65, 7.45)$ vs locked $(3.15, 3.75, 8.08)$, $(N_{2q},D_{2q},D_c)=(185,139,502)$ vs $(188,139,505)$, $\mathcal E$ 27.1 vs 23.7, $N_{\rm step}$ 73.9 vs 66.5 ($10^3$). Weak--slow state motion (`9720203.1`, 380 min, 2.9 GB, `ae15a0f6`): 24 true exchanges, $N_\theta=59$, errors $(2.16, 2.84, 7.09)$ vs $(2.96, 3.70, 8.24)$, $(213,173,600)$ vs $(180,140,490)$, $\mathcal E$ 20.3 vs 19.7, $N_{\rm step}$ 49.8 vs 52.3.

**Activation-threshold rungs for the two non-matched corrected cells** (manuscript bound):

| cell | $\tau_m$ | $\overline{|\Delta E|}$ | $\overline{|\Delta D|}$ | $\overline{|\Delta n|}$ | $N_\theta$ | $\mathcal E$ | matched (worst factor) |
|---|---|---:|---:|---:|---:|---:|---|
| strong--slow $\delta\theta$ | $3\times10^{-6}$ | 2.98 | 3.52 | 7.37 | 50 | 11.7 | no (2.83) |
| | $10^{-6}$ | 4.52 | 2.05 | 2.64 | 46 | 9.15 | no (2.27) |
| | $3\times10^{-7}$ (`9720666`, `ca1be191`) | 4.93 | 2.93 | 3.34 | 42 | 8.48 | no (2.47) |
| weak--fast $\delta\theta$ | $3\times10^{-7}$ | 16.2 | 3.03 | 8.17 | 50 | 9.37 | no (2.82) |
| | $10^{-7}$ | 15.4 | 2.88 | 8.20 | 45 | 11.0 | no (2.67) |
| | $3.33\times10^{-8}$ (`9720676`, `f07c4d21`) | 13.1 | 2.92 | 5.93 | 50 | 11.1 | no (2.27) |

Fourth rungs submitted as `9720908` (weak--fast $10^{-8}$, strong--slow $10^{-7}$).

**SIAM AVQDS ladder**: r08 (`9720692`, 45 min) burden 0.285, $C_3$ 1.14, $N_\theta=124$; r09 (cut $3.33\times10^{-7}$) submitted `9720906`. The individual target is not reached by the cut alone on this seed.

**Local runs stopped**: the three local AVQDS ridge-sweep runs launched 2026-09-05 12:58 for the floor rule (`/tmp/ridge_sweep`, $\lambda\in\{10^{-5},10^{-6},10^{-7}\}$, weak--slow) were killed after 20 h with no output; if the floor rule is still wanted it goes to CHTC. No other local Paper II runs.

**Codex AVQDS comparators**: `9719356.0` and `.2` restarted at ~11:50Z (evicted after ~36 h; AVQDS has no checkpoint), `.1` and `.10` at 43--46 h. All four are inside the 72 h window only if they finish before ~2026-09-07 evening.
## 20260906T161000Z — strong--slow $\delta\theta$ at $\tau_m=10^{-7}$ (`9720908.1`, exit 0, 83 min, `804e2c7a`)

Errors $(4.27, 1.52, 4.54)\times10^{-4}$, $N_\theta=50$, $\mathcal E=8.38\times10^8$, $N_{\rm step}=444\times10^3$, 4 exchange commits; matched vs AVQDS: no (2.14; the energy component, AVQDS $1.99\times10^{-4}$). Rung sequence $3\times10^{-6}\to10^{-7}$: worst factor 2.83, 2.27, 2.47, 2.14. Fifth rung $3.33\times10^{-8}$ submitted (lock `paper_ii_corrected_exchange_next_rung5_cells_v1`).
## 20260906T163000Z — data figures regenerated from the corrected runs

`agent_guidance/time-dynamics/reports/corrected_assets_20260906/figures/`: `paper_ii_t20_response_spectra.pdf` (+ `spectra_manifest.json`; the four $\delta\theta$ cells with both arms, as in the manuscript: strong--slow, strong--mid, weak--fast, weak--fastest), `paper_ii_t20_q_compiled_weakslow_state_motion.pdf`, `paper_ii_t20_q_compiled_fastfast_state_motion.pdf` (+ manifests). Inputs: the corrected comparison, a flat raw root of corrected APM `run.json` plus the locked AVQDS `run.json` per cell (`/tmp/report/flat_root`), the HH exact-reference store, and the supplement for the weak--slow AVQDS arm. `paper_ii_t20_q_compiled_slowstrong_state_motion.pdf` waits on `9720203.7`. Spectral relative errors (APM, AVQDS): strong--slow doublon (0.0146, 0.0037), staggered (0.0065, 0.0018); strong--mid (0.0026, 0.0016), (0.0017, 0.0013); weak--fast (0.0140, 0.0031), (0.0052, 0.0016); weak--fastest (0.0694, 0.0058), (0.0013, 0.0011). PDFs are gitignored; manifests committed.
## 20260906T165500Z — weak--fast $\delta\theta$ at $\tau_m=10^{-8}$ (`9720908.0`, exit 0, 128 min, 2198 MB, `94921b13`): ladder stopped for this cell

| $\tau_m$ | $\overline{|\Delta E|}$ | $\overline{|\Delta D|}$ | $\overline{|\Delta n|}$ | $N_\theta$ | $\mathcal E$ | exchange commits | worst factor vs AVQDS |
|---|---:|---:|---:|---:|---:|---:|---:|
| $3\times10^{-7}$ | 16.2 | 3.03 | 8.17 | 50 | 9.37 | 3 | 2.82 |
| $10^{-7}$ | 15.4 | 2.88 | 8.20 | 45 | 11.0 | 2 | 2.67 |
| $3.33\times10^{-8}$ | 13.1 | 2.92 | 5.93 | 50 | 11.1 | 6 | 2.27 |
| $10^{-8}$ | 13.8 | 2.39 | 7.87 | 53 | 12.4 | 7 | 2.39 |

The energy error sits at $(13$--$16)\times10^{-4}$ across four rungs while the AVQDS comparator is at $4.24\times10^{-4}$; the activation threshold does not move it. No further rung for weak--fast $\delta\theta$; its corrected row is reported outside the matched-accuracy rule (the reported trajectory at $3\times10^{-7}$ was inside it at 1.42). Strong--slow $\delta\theta$ continues at $3.33\times10^{-8}$ (`9721087`).
## 20260906T172000Z — SIAM AVQDS r09 **qualified**; continuation and exchange-arm r01 submitted

- r09 (`9720906`, exit 0, 144 min, 342 MB, `3e1691c4`): `qualified` — burden 0.110, broad guard and individual target pass, $C_3$ 0.44, $N_\theta=161$. Ladder: r00 3.855, r01 0.916, r02 0.438, r03 0.961, r04 0.981, r05 0.519, r06 0.300, r07 0.324, r08 0.285, r09 **0.110**.
- §8.3/8.4: AVQDS continuation $t=1\to5$ (401 points) resumed from the r09 screen, 2 CPU / 2 GB / 40 GB / 172800 s → **`9721121`** (`chtc/siam_nb4_qc_avqds_continue_r09_cut3em07_source_2ef146d6_20260906`, `ee9095b8`; contract 825 rows verified).
- Exchange arm resumed at the manuscript's permission bound (the pilot found no adoptable decoupled threshold): r01 screen, cut $3.33\times10^{-3}$, profile `siam_nb4_pilot_exchange_v1` (frontier 8) → **`9721122`**. Its r00 screen had burden 5.659 outside the broad guard.
- Watch list: `9720203.7`, `9719356.*`, `9721087`, `9721121`, `9721122`.
## 20260906T173500Z — strong--slow $\delta\theta$ at $\tau_m=3.33\times10^{-8}$ (`9721087`, exit 0, 80 min, `1bd74f03`): ladder stopped for this cell

| $\tau_m$ | $\overline{|\Delta E|}$ | $\overline{|\Delta D|}$ | $\overline{|\Delta n|}$ | $N_\theta$ | $\mathcal E$ | exchange commits | worst factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| $3\times10^{-6}$ | 2.98 | 3.52 | 7.37 | 50 | 11.7 | 4 | 2.83 |
| $10^{-6}$ | 4.52 | 2.05 | 2.64 | 46 | 9.15 | 4 | 2.27 |
| $3\times10^{-7}$ | 4.93 | 2.93 | 3.34 | 42 | 8.48 | 4 | 2.47 |
| $10^{-7}$ | 4.27 | 1.52 | 4.54 | 50 | 8.38 | 4 | 2.14 |
| $3.33\times10^{-8}$ | 6.25 | 2.99 | 3.49 | 48 | 7.47 | 7 | 3.14 |

Not monotone in $\tau_m$; no rung reaches the factor-two rule (AVQDS energy error $1.99\times10^{-4}$). No further rung. Both corrected cells that left the rule (weak--fast $\delta\theta$, strong--slow $\delta\theta$) stay outside it at every rung tried; which rung to print for them is a reporting decision (the lock's cut keeps the row comparable to the reported one; $10^{-7}$ is the closest to the rule).

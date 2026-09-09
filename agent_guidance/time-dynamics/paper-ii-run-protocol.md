# Paper II — operational run protocol

How a reported number is actually produced, end to end, on CHTC. The scientific
contract a number must satisfy is `paper-ii-comparison-protocol.md`; this file is
the mechanics. Every step below has failed at least once in the way noted, so the
traps are part of the protocol, not asides.

Two lanes share the same skeleton and differ in their packagers:

- **Hubbard--Holstein (HH)**: locked cells, two arms per cell, one package, many procs.
- **SIAM $N_b=4$**: one action per package, comparator-first ladder, segments.

---

## 0. Where you work

Repo-changing work happens in your own git worktree under
`Holstein_test.worktrees/` on your own branch; other agents are live in the main
checkout. Stage commits with explicit paths, never `git add -A`.

The remote submit root is
`ap2001:~/Holstein_paper_ii_corrected_exchange_v1_4c9984b0_20260905/`.
The author's standing CHTC authorization covers connection, transfer and
submission when each campaign step becomes ready. A fresh request to connect
or a readiness question is not a prerequisite.

First check the shared master with `ssh -O check ap2001`. If it is unavailable,
start one interactive login:

```bash
ssh -o ControlPersist=12h -o ServerAliveInterval=60 -o ServerAliveCountMax=3 ap2001
```

Use the available authorized credential, respond promptly to the password and
Duo prompts, and select one Duo push. The author approves the notification.
Keep the PTY alive and verify the login and reusable master before continuing
with other work. Reuse that master for subsequent `ssh`/`rsync`/`scp` calls and
continue the due campaign step immediately after login. Store connection state,
never credentials.

The connection policy in `paper-ii-overnight-campaign.md` is authoritative,
including how to record an actually denied or timed-out push without repeated
pushes. An expired connection triggers this login procedure; it is not a reason
to defer an authorized operation before trying it.

## 1. Lock the cells

A lock is the scientific identity of a campaign: which cells, which activation
cuts, which route bundle, which extra flags. Never hand-edit one.

```python
from pipelines.time_dynamics.campaigns import paper_ii_permission_threshold_bundle as bundle
path = bundle.write_lock(
    literal=None,                     # None keeps the manuscript's permission bound
    scope="cells",                    # or "eleven" / "strongfast_dtheta"
    cell_ids=["midres__delta_theta_5e-5", ...],
    campaign_id="paper_ii_<name>_v1",
    purpose="one sentence a reader can check",
    apm_cut_overrides={"cell": 1e-7}, # only to move a cell to another rung
)
```

`write_lock` validates by loading the result through
`paper_ii_final_selector_comparison_v1.load_lock`, so a lock that survives the
call is structurally sound.

**Traps.** `adoption_status` accepts only `pair_matched_comparison` and
`individual_response_target`; inventing a third value fails the loader. Cell rows,
metrics and sources are copied verbatim from the parent lock, so edit only cuts
and flags.

## 2. Build the package

The package is a self-contained directory holding a `git archive` of one
revision, the lock, the action list, the submit files, and a validator.

```python
bundle.build_two_arm_package(                    # HH: APM + fresh AVQDS per cell
    lock_path=path,
    package_dir=Path(f"chtc/<campaign>_{REV[:8]}_{DATE}"),
    source_revision=REV, repo_root=Path("."),
    apm_memory="12GB", avqds_memory="6GB", max_runtime_seconds=172800,
)
```

For SIAM, one action per package:

```python
from pipelines.time_dynamics.siam_chtc_package import prepare_siam_nb4_chtc_action
from pipelines.time_dynamics.siam_nb4_ladder import siam_nb4_action_id, SIAM_NB4_ACTIVATION_CUT_LADDER
from pipelines.time_dynamics.paper_ii_runs import SIAM_NB4_SEARCH_PROFILES
prepare_siam_nb4_chtc_action(
    repo_root=".", package_dir=Path(f"chtc/siam_nb4_qc_{lane}_{kind}_{tok}_source_{REV[:8]}_{DATE}"),
    seed_path="chtc/siam_nb4_qc_full_apm_screen_r00_cut1em02_source_812d20c3_20260905/input/seed.json",
    image_path="chtc/time_dynamics_optuna/image.sif",
    source_revision=REV, action_id=siam_nb4_action_id(arm=..., segment=..., activation_cut=CUT, route_id=ROUTE),
    activation_cut=CUT, arm="exchange_q_compiled" | "avqds",
    segment="screen_t0_t1_101" | "continue_t1_t5_401", route_id="paper_ii_exchange_q_compiled_v1",
    search_profile=SIAM_NB4_SEARCH_PROFILES["siam_nb4_pilot_exchange_v1"],   # APM only; AVQDS keeps the implicit default
    request_cpus=2, request_memory_gb=3, request_disk_gb=40, max_runtime_seconds=172800,
    resume_from_run_json=f"raw_outputs/{SCREEN_PKG}/{SCREEN_ACTION}/run.json",      # continuations only
    resume_action_receipt_json=f"raw_outputs/{SCREEN_PKG}/{SCREEN_ACTION}/action_receipt.json",
)
```

**Traps.**

- **The tree must be clean and the lock committed.** `git archive` takes the
  revision, not the working tree, so an uncommitted lock produces
  `actions do not carry the requested source revision (dirty tree?)`. Commit first.
- `package_dir` and `image_path` must be inside the repo root, and `image_path`
  must be a real file: a symlink to another worktree resolves outside the root and
  is refused. Hard-link it instead.
- A continuation must be built **at the same revision as the segment it resumes**,
  or carry a revision-equivalence record (§7). Building it at HEAD is the failure
  that discarded 21 h of propagation in cluster 9721121.

## 3. Validate before shipping

```bash
python chtc/validate_condor_submit_lifecycle.py <pkg>/submit_apm.sub
PYTHONPATH=. python <pkg>/validate_package.py        # HH two-arm packages
```

The validator checks action count, routes per cell, campaign id, and the archive
digest. When a package is templated from another campaign, the routes-per-cell
count and the `raw_outputs/<campaign>/` path inside the validator must be
rewritten too; the bundle tool does this, hand-built packages do not.

## 4. Ship and prove the source tree

```bash
rsync -azR pipelines src <pkg> ap2001:Holstein_paper_ii_corrected_exchange_v1_4c9984b0_20260905/
```

Then prove the remote tree matches what the actions declare, before spending
cluster time:

```python
rows = [(f["sha256"], f["path"]) for f in json.load(open(f"{pkg}/input/action.json"))["source_tree"]["files"]]
open("/tmp/contract_check.sha","w").write("".join(f"{h}  {p}\n" for h, p in rows))
```

```bash
rsync -az /tmp/contract_check.sha ap2001:<root>/contract_check.sha
ssh ap2001 "cd <root> && sha256sum -c --quiet contract_check.sha && echo CONTRACT_OK"
```

Note `rsync -azR pipelines src` reverts any newer files on the AP to this
revision's contents; that is intended, and it is why the contract check runs after
the sync rather than before.

## 5. Submit

```bash
ssh ap2001 "cd <root> && python3 chtc/validate_condor_submit_lifecycle.py <pkg>/submit_apm.sub && condor_submit <pkg>/submit_apm.sub"
```

Record the cluster id in `agent_guidance/time-dynamics/paper-ii-corrected-campaign-monitor.md`
and add it to the poller's watch list (`/tmp/chtc_monitor_state/clusters`).

**The 72-hour wall.** The pool holds any job that has not finished in 72 h, and a
held job **transfers nothing**: its stdout, stderr and outputs are lost, and
`condor_transfer_data` does not apply unless the job was submitted with `-spool`.
Anything that cannot finish inside 72 h must be a bounded segment that exits
normally and resumes. Four AVQDS comparators died this way.

## 6. Collect by receipt, never by name

Every package stages `archives/<action_id>.tar.gz` into one shared directory, and
packages templated from a common parent reuse action ids, so a later stage-out
silently overwrites an earlier one. A watcher (`archive_watch.sh`, pid in
`archive_watch.pid`) copies each arriving pair into
`archives_by_receipt/<archive_sha256>/` within five seconds.

```bash
scp -q "ap2001:<root>/archives_by_receipt/<sha>/*" /tmp/report/arrivals/<sha>/
tar -xzf /tmp/report/arrivals/<sha>/*.tar.gz -C /tmp/report/arrivals/<sha>/
```

Restart the watcher after a tunnel or AP reboot. Kill it by pid from the pidfile,
never with `pkill -f archive_watch`, which also kills your own SSH shell.

## 7. Score

**HH, whole campaign:**

```bash
PYTHONPATH=. python pipelines/time_dynamics/diagnostics/paper_ii_four_route_comparison.py \
  --repository-root . --direct-root archives/<campaign> --direct-root /tmp/report/avqds_root \
  --artifact-root <legacy roots...> --output-json /tmp/report/comparison.json
```

**HH, one run against the exact reference** (named experiments the comparison
tool will not admit, e.g. a new controller):
`paper_ii_full_results_report._run_record(run_path, ExactReferenceStore(root))`.

**SIAM:**

```python
score = score_siam_raw_run(run_path=..., seed_path=..., reference_store_root=store.root,
                           verified_reference=store.prefix_view(full, times=SIAM_NB4_SCREEN_TIMES))
verdict = classify_siam_matched_pair(apm_score, avqds_score, revision_equivalence=record)
```

Segments are joined first with `stitch_siam_nb4_segments([screen, continuation],
seed_path=...)`, which verifies the boundary and writes the 0..5 trajectory.

**Revision equivalence.** Two arms whose recorded revisions differ are
incomparable unless a record proves the difference lies outside the trajectory
runner's import closure:

```python
from pipelines.time_dynamics.diagnostics.source_engine_surface import (
    TRAJECTORY_ENTRY_POINT, python_import_closure, declare_revision_equivalence)
surface = python_import_closure(TRAJECTORY_ENTRY_POINT, reader(A)) | python_import_closure(TRAJECTORY_ENTRY_POINT, reader(B))
record = declare_revision_equivalence(left_revision=A, right_revision=B,
                                      differing_paths=git_diff_names(B, A), engine_paths=surface)
```

It raises if any differing path is inside the closure. Report generators, campaign
locks, submit packages and guidance files are outside it; anything under `src/` or
the runner's imports is not.

## 8. Regenerate the manuscript assets

```bash
PYTHONPATH=. python pipelines/time_dynamics/diagnostics/q_compiled_manuscript_assets.py \
  --comparison-json /tmp/report/comparison.json --terminal-cost-root /tmp/report/terminal_costs_combined \
  --supplement-comparison-json <legacy comparison> --supplement-terminal-cost-root /tmp/paper_ii_supp/terminal_costs \
  --pending-cell drive:controller \
  --table <out>/paper_ii_t20_q_compiled_frontier_table.tex --figure <out>/fig.pdf \
  --manifest <out>/manifest.json --adoption-note "..."
```

Copy the table into the manuscript repository's
`Paper_II_overleaf_20260901/generated/`, build with
`tectonic --keep-logs --reruns 2 time_dynamics_paper_II.tex`, commit, and push;
Overleaf pulls from GitHub. Never edit `generated/` by hand: it is produced here.

## 9. What a finished result looks like

A number is reportable when all of these exist: a lock naming the cell, a package
whose archive digest matches the remote tree, a completion receipt with
`status: COMPLETED`, an archive preserved by receipt, a score computed against the
content-addressed exact reference, and an entry in the monitor log giving the
cluster id, wall time, memory, and the verdict. If any one is missing, the number
is not evidence yet.

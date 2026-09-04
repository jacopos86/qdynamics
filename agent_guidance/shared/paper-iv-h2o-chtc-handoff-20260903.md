# Handoff — Paper-IV H2O CHTC campaign: two arms, monitored, loss-proof

**Author:** Claude (Opus 5) · **Executor:** Codex · **Date:** 2026-09-03
**Contract:** `agent_guidance/shared/agent-handoff-contract.md`
**Post-mortem this replaces:** `agent_guidance/shared/paper-iv-worklog.md`

---

## Status — end of 2026-09-03 (read this first; §0–§8 below are the original plan plus corrections)

Campaign branch **`paper-iv/h2o-campaign-20260903`** at **`315b96e2`**
(worktree `Holstein_test.worktrees/paper-iv-h2o-campaign`), = `main c81247de`
+ every lane below, merged. Fast baseline over eleven suites: **237 passed,
21 skipped, 1 failed** — the failure is `test_ra_adapt_facade::test_frozen_ordinary_v1_protocol_remains_executable`,
pre-existing on main. Package rebuilt on this HEAD; `runtime_preflight.py`
**21/21 pass**.

| Blocker at the start of the day | State now | Evidence |
|---|---|---|
| Graph-span proxy unavailable at 19 qubits | **built** — 19q entry + `--cost-mode {proxy,marrakesh_graph_span_v1}`, default sha unchanged; fails closed in 0.4 s on unsupported sizes | `af222b69`, `3bcf490f`; 65 passed |
| Which 19q embedding | **user decision open** — the tuple is a choice (430 candidates); lane (10,8,8) vs B (8,7,6) vs F (1,2,1) boson-block spans | verifier + `3bcf490f` comment |
| "Singleton" = `single_pauli_word_v1` | **mapped** — `sector_complete_pauli_block_v1` | `h2o_application.py:76-80` |
| Singleton arm crashed after round 1 (`engine.py:3319`) | **fixed** — parallel fail-closed sector-block lineage branch; 2 rounds → exit 0 + `result.json` | `eecf2b8f`; 10 + 2 opt-in passed |
| Every H2O run exited 1 after its last round (`runtime.py:1969`) | **fixed** — H2O family skips the Paper-I canonical summary | `929f1b47` |
| No wall-clock stop | **built** — `--max-wall-seconds`, `wall_clock_budget_v1`, round-boundary only, protocol sha unchanged | `929f1b47` |
| H2O checkpoints could not be resumed (`_resume.py:1455`) | **admitted** — named `paper_iv_h2o_linear_fd_application` branch, fixture-sha attestation, HH/Hubbard untouched; macro arm chained seg1 → seg2 from a relocated fixture path | `73854cbc`; 14 passed |
| Ledger port broke every resume (`NameError occurrences`) | **fixed** in the same commit | `73854cbc` |
| b3 speed-up branch 679 commits behind main | **integrated the other way** — chunked encoder + compact layout picked, O(1) ledger ported with byte-identical parity | `d6d44af3`, `3b1b86f4`, `34c87900` |
| Fixture single-copy, untracked | **tracked** — `test_support/paper_iv_h2o/`, sha `190387449b…` | `fd37ec19`; 7 passed |
| `/home` quota 4.9 GB headroom, no `/staging` | **designed around** — results to `/home`, retrieve→verify→delete; `--results-mode staging` ready if provisioned | `216aec84` |

Still gated on the user: the embedding choice (§0.1), whether to install the
opus-only subagent hook, and removal of the 36 dead held jobs. The opt-in
end-to-end chain tests for both arms on the merged HEAD were running when this
was written; their result is recorded in the worklog.

Lane contract `agent_guidance/molecular-vibronic/AGENTS.md` now describes the
admitted resume state (on the campaign branch).

---

## 0. Read this before planning anything

Three findings block or reshape the requested plan. They are verified against
source, not inferred.

### 0.1 The graph-span cost proxy cannot run at 19 qubits

`marrakesh_graph_span_v1` resolves a physical layout from a hard-coded table:

```python
# pipelines/static_adapt/hh_backend_compile_oracle.py:128-135
_MARRAKESH_GRAPH_SPAN_EMBEDDINGS_V1: dict[int, tuple[int, ...]] = {
    4: (0, 1, 2, 3),
    6: (0, 1, 2, 3, 4, 16),
    8: (0, 1, 2, 3, 4, 16, 5, 23),
    9: (0, 1, 2, 3, 4, 16, 5, 23, 6),
    10: (0, 1, 2, 3, 4, 16, 5, 23, 6, 22),
    12: (0, 1, 2, 3, 4, 16, 5, 23, 6, 22, 7, 21),
}
```

`marrakesh_logical_embedding_v1(19)` raises
`unsupported_marrakesh_graph_span_embedding_v1_size:19`
(`hh_backend_compile_oracle.py:150-154`). The converged H2O model is
**19 qubits**. So graph-span is unavailable for this problem until a 19-qubit
FakeMarrakesh embedding is added.

**This is a scientific decision, not a mechanical one.** The embedding fixes
which physical qubits and couplings the cost is measured over; a different
connected 19-qubit subgraph gives different two-qubit counts and depths. It is
therefore the user's call, not the executor's. **Do not invent an embedding.**

Note also that the H2O route already runs the *cheap* path: the July
submission record's `application_overlay` sets
`"phase3_backend_cost_mode": "proxy"`
(`chtc/paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_d50_20260724/submission_record.json`).
`proxy` is the inexpensive analytic mode already; `marrakesh_graph_span_v1` is
the *Paper-I* analytic estimator, separately gated and locked to
`phase3_backend_name='FakeMarrakesh'`, optimization level 1, transpile seed 7
(`adapt_pipeline.py:28396-28405`). If "cheaper/quicker proxy" meant "not full
transpilation," that is already satisfied and no change is needed.

→ **Ask the user which they meant before building the graph-span arms.**

### 0.2 "Singleton only" must not use `single_pauli_word_v1`

For H2O that representation is refused by design:

```python
# pipelines/static_adapt/ra_adapt/h2o_application.py:76-80
if representation == CANDIDATE_REPRESENTATION_SINGLE_PAULI:
    raise ValueError(
        "Raw H2O single-Pauli children are not sector-complete. Use "
        f"{H2O_CANDIDATE_MODE_SECTOR_COMPLETE_PAULI_BLOCK!r}."
    )
```

| Requested arm | Actual flag |
|---|---|
| macro only | `--candidate-representation macro_generator_v1` |
| singleton only | `--candidate-representation sector_complete_pauli_block_v1` |

`sector_complete_pauli_block_v1` is the singleton-style arm and the route
default. Passing `single_pauli_word_v1` fails immediately at startup.

### 0.3 `/home` quota has 4.9 GB of headroom — a full disk loses results the same way a hold does

Measured on `ap2001` 2026-09-03:

| Filesystem | Used | Soft quota | Hard limit |
|---|---|---|---|
| `/dev/md9` (`/home`) | **35,719 M** | 40,960 M | 51,200 M |
| `/dev/md3` | 6,944 K | 128 M | 256 M |

**Correction (2026-09-03, verified `ls -ld`): `/staging/jsstrobel` does not
exist.** An earlier read counted a failed `ls` as "empty". CHTC provisions
per-user `/staging` directories on request to the facilitators — an external
action only the user can take. Until it exists, every segment's output lands in
`/home`, so the design must keep segment outputs small (`--checkpoint-history-tail 1`,
compact checkpoint layout, no estimator ledger unless resume needs it) and the
monitor must **retrieve, verify, then delete** each segment from `/home` promptly.
Gate every submission on a free-space check (§4, guard G3). If the user obtains
`/staging`, switch the package's output root to it — nothing else changes.

Housekeeping: the queue holds **138 held jobs** (2026-09-03), only one of them
Paper IV. Triage by hold reason:

| Count | Hold reason | Disposition |
|---|---|---|
| 50 | `OnExitHold` — exit code ≠ 0 | Paper I/II campaigns; owners decide |
| 36 | input transfer failure — `image.sif` no longer exists (`Holstein_jr_l10_…` and `Holstein_phase3_optuna_chtc/…`) | **unambiguously dead**; can never run |
| 16 | held by user | intentional |
| 11 | cgroup memory limit | owners decide |
| 7 | superseded by authenticated B3 continuations / preventive disk escalation | intentional |
| 4 | `request_disk` exceeded | owners decide |
| 3 | failed to complete in 72 h | one is 9715945 (Paper IV) |

By batch: 84 are the 2026-08-17 `allphase_maxk50` Paper-I jobs with
`JobBatchName` undefined; the rest are named Paper I/II batches. **These belong
to other agents' campaigns — do not remove any without the user naming them.**
The 36 missing-image jobs are the safe first ask.

---

## 1. Anchors

| Field | Value |
|---|---|
| Checkout | `/Users/lordskyler/Holstein_test` (M4 replacement machine; supersedes any `/Users/Patron/` or `/Users/jakestrobel/` path in older docs) |
| Campaign branch (work here) | **`paper-iv/h2o-campaign-20260903` at `315b96e2`**, worktree `Holstein_test.worktrees/paper-iv-h2o-campaign` |
| Base | `main` at `c81247de` |
| Lane branches (merged, kept for history) | `fix/paper-iv-h2o-resume-20260903`, `paper-iv/build-{port-ledger,wallclock,graphspan,resume,sector-lineage,embedding-claim}-20260903` |
| b3 speed-up source | `claude/paper-i-b3-local-tail-20260903` at `83690456` (base `f041a5c1`, 679 commits behind main; integrated b3 → main, see §2) |
| Python | `/Users/lordskyler/Holstein_test/.venv/bin/python` (3.12.14). System `python3` is 3.9 and **cannot import this package** (PEP-604 unions fail at `sr_snake/contracts.py:226`) |
| Submit node | `ap2001.chtc.wisc.edu`, user `jsstrobel`, remote root `~/chtc/` |
| Test baseline (fast, default mode) | eleven suites (`test_ra_adapt_h2o_application`, `test_paper_iv_h2o_resume_admissibility`, `test_ra_adapt_sector_block_lineage`, `test_marrakesh_graph_span_embedding_19q`, `test_paper_iv_compact_layout_resume`, `test_paper_iv_h2o_work373_fixture`, `test_estimator_ledger_occurrence_window`, `test_json_streaming`, `test_current_checkpoint_json_layout`, `test_ra_adapt_facade`, `test_paper_i_sr_snake_resume_adapter`) → **237 passed, 21 skipped, 1 failed** (the pre-existing `test_frozen_ordinary_v1_protocol_remains_executable`) |
| Opt-in end-to-end (production route, minutes each) | `PAPER_IV_H2O_CLI_RUNS=1 … -k "sector_arm or segment_chain or macro_arm or wall_budget_stops"` |
| Known-failing (pre-existing, do not "fix") | `test/test_paper_iv_h2o_static_route.py::test_h2o_electronic_control_activates_whitened_singleton_route` and `::test_h2o_derivative_resolved_pool_activates_whitened_singleton_route` — both fail on unmodified `main` at `c81247de` with `_historical_paper_i_route_compatibility_active() missing 2 required keyword-only arguments: 'phase2_enable_batching' and 'phase3_enable_batching'` |

### The fixture

`h2o_linear_fd_work373_ref484_fixture_20260826.json`, 28,025,185 bytes.

| Property | Value |
|---|---|
| Work cutoffs `n_ph_work` (sym stretch, bend, antisym stretch) | (3, 7, 3) |
| Reference cutoffs | (4, 8, 4) |
| Qubits | 19 |
| Sector dimension | 28,800 |
| `E_work` | −75.014463513376 Ha |
| `E_ref` | −75.015134943251 Ha |
| `abs(ΔE)` | 6.7143e−4 Ha (tol 1.6e−3) — pass |
| Total boundary weight | 6.2066e−3 (tol 1e−2) — pass |

Locations (**not git-tracked — single copy on infrastructure that has already
lost data once**):

- `~/chtc_rescue/home/chtc/paper_iv_h2o_reencode_20260826/results/h2o_linear_fd_work373_ref484_fixture_20260826.json`
- `ap2001:/home/jsstrobel/chtc/paper_iv_h2o_reencode_20260826/results/`

Increment 0 tracks it.

---

## 2. Branch integration — decide before building

`claude/paper-i-b3-local-tail-20260903` carries the serialization work the user
wants (`STATIC_ADAPT_CHECKPOINT_JSON_LAYOUT=compact`, chunked encoder, O(1)
ledger bookkeeping, byte-identical parity 45/45 vs `f041a5c1`). It does **not**
contain the resume flag. `main` does not contain the serialization work.

Commits on b3 not in main: `83690456`, `ef42dd04`, `60c93b26`, `d893a6b0`,
`f041a5c1`, `58c23b34`, `662644d0`, `5e9336e0`, `c2cd9379`, `de19458b`.
Commits on main not in b3 include `9c94f94c`, `ed8526aa`, `75d4c5d1`,
`c81247de`, `56eeb49d`, `dda15f1f`, `557f20ae`.

Both are needed: without the resume flag a segment cannot continue; without the
compact layout the per-round checkpoint write is the dominant serialization
cost on a 19-qubit, 28,800-dimension sector.

**Resolved 2026-09-03 — the integration runs the other direction.** Basing on
b3 was tried and abandoned: b3 is **679 commits behind main**, lacks
`f433b89d` (the commit that made append-only at per-mode cutoffs reachable),
and lacks 13 `ra_adapt`/`sr_snake` refactors; cherry-picking `9a43bfe2` onto
it conflicted immediately. Instead the campaign branch is
`paper-iv/h2o-campaign-20260903` (worktree
`Holstein_test.worktrees/paper-iv-h2o-campaign`) = `main` + `9a43bfe2` +
b3's serialization chain cherry-picked onto it:

| Picked | As | Content |
|---|---|---|
| `60c93b26` | `d6d44af3` | chunked JSON encoder, byte-identical to `json.dumps` |
| `ef42dd04` | `3b1b86f4` | chunked checkpoint digests; opt-in `STATIC_ADAPT_CHECKPOINT_JSON_LAYOUT=compact` |
| `83690456` | `6648ed5b` | the b3 implementation report |

`test_json_streaming.py` + `test_current_checkpoint_json_layout.py`: **61 passed**
on main's base.

**Not picked:** `d893a6b0` (O(1) ledger `occurrence_count`/`occurrence_rows`,
"6 pre-Phase-I materializations → 0") — 333 lines into `adapt_pipeline.py`
conflict with main's refactors. It is being **ported** as a build task, with
its 522-line `test_estimator_ledger_occurrence_window.py` as the acceptance
bar. Until that lands, only the checkpoint-write speed-up is on the branch;
the ledger re-materialization cost is not yet removed.

The b3 author's 45/45 byte-identical parity was measured against `f041a5c1`,
**not main**; it must be re-established on the campaign branch before the
compact layout is trusted for a paper-facing run.

---

## 3. What actually killed the last run — the constraint every design must respect

Cluster 9715945 ran **70.0 h** and returned **nothing**.

CHTC's 72 h limit is a `SYSTEM_PERIODIC_HOLD`, not a preemption:

```
012 (9715945.000.000) 2026-09-02 12:13:52 Job was held.
        Error from slot1_21@e2601.chtc.wisc.edu: Job failed to complete in 72 hrs
        Code 21 Subcode 0
```

`when_to_transfer_output = ON_EXIT_OR_EVICT` fires on normal exit and on
graceful eviction. **A policy hold terminates with no stage-out at all.**
Confirmed: `StageOutStart`, `StageOutFinish`, `SpooledOutputFiles` all
`undefined`; no spool directory; `condor_transfer_data` fails on a missing
`results/`.

The engine *was* writing a complete accepted-state checkpoint to
`results/current.json` after every controller round
(`h2o_application.py:171-175`, `adapt_pipeline.py:21488`) — onto execute-node
scratch nobody could reach.

**The load-bearing consequence:** monitoring cannot rescue a job at the wall.
By the time the hold fires the data is already unreachable. The only reliable
protection is that **the Python process exits normally before 72 h.** Every
design below follows from that.

Also note there is **no SIGTERM handler anywhere in the execution path**
(`kill_sig = SIGTERM`, `kill_sig_timeout = 900` buys nothing), so a timeout-kill
is not a graceful stop either.

---

## 4. Increments

Proceed autonomously through increments 0–2. **Pause and report before
increment 4 (submission).** Increment 3 is blocked on a user decision.

### Increment 0 — put the fixture under version control

**Goal:** the only copy of Paper IV's cutoff-convergence evidence is no longer
single-copy on infrastructure.

```bash
cd /Users/lordskyler/Holstein_test
mkdir -p test_support/paper_iv_h2o
cp ~/chtc_rescue/home/chtc/paper_iv_h2o_reencode_20260826/results/h2o_linear_fd_work373_ref484_fixture_20260826.json \
   test_support/paper_iv_h2o/
cp ~/chtc_rescue/home/chtc/paper_iv_h2o_reencode_20260826/results/reencode_summary_work373_ref484.json \
   test_support/paper_iv_h2o/
shasum -a 256 test_support/paper_iv_h2o/*.json
```

**Expected:** the fixture hashes to
`570690bd126787305b340bd2f7493499c0f3101e3e2820c2d355c55c16afa594` **only if**
it is byte-identical to the July fixture. It is a *different, later* fixture
(work373/ref484 vs the July nph1/ref2 lineage), so **expect a different
digest** — record whatever it is as the first anchor and do not "fix" a
mismatch.

**Stop condition:** if the repo forbids a 28 MB blob (check `.gitattributes`
and any LFS config first), stop and report rather than committing it anyway.

### Increment 1 — add a clean wall-clock stop (highest value) — **DONE (`929f1b47`)**

**Goal:** a segment always exits normally before the pool wall, whatever the
per-round cost turns out to be. This is the durable fix; round-count sizing
alone is a guess.

Add `--max-wall-seconds` to `h2o_application.py`. On expiry the controller loop
must **stop between rounds and exit 0** so `transfer_output_files = results`
fires. It must not raise, must not `os._exit`, and must write `current.json`
plus its estimator-ledger sidecar before returning.

**Expected:** a unit test proving that a run started with
`--max-wall-seconds 1` exits 0, produces `current.json` and its sidecar, and
does **not** produce `result.json` (the run is incomplete, not finished).

**Stop condition:** if the controller loop has no between-round hook that can
return cleanly, stop and report the seam you found rather than bolting a signal
handler onto it — that changes stopping semantics and is a scientific-surface
change.

### Increment 2 — calibration segment (measurement, not a guess)

**Goal:** real per-round wall times at 19 qubits / 28,800 dimension, so segment
sizing stops being folklore. Nobody knows how far 9715945 got; the depth trace
died with the sandbox.

Run **one** short segment per arm, 8 rounds, local or CHTC.

**Expected:** a table of per-round seconds for each arm, and the observed growth
in per-round cost with depth (append-only ADAPT grows per round; quantify it).

**Stop condition:** if a single round exceeds ~4 h, 50 rounds is unreachable in
any sane segmentation — stop and report, because that is a scope conversation,
not an execution detail.

### Increment 3 — graph-span arms — **code done; embedding choice open**

`--cost-mode marrakesh_graph_span_v1` works at 19 qubits (`af222b69`,
`3bcf490f`). What remains is the user's choice of embedding (§0.1 / status
table): the shipped tuple is a placeholder documented as a choice. Apply the
chosen tuple in BOTH `pipelines/static_adapt/hh_backend_compile_oracle.py`
and `pipelines/resource_estimation/pauli_graph_span.py`, update the pinned
spans in `test/test_marrakesh_graph_span_embedding_19q.py`, and note that
every graph-span cost reported for H2O is conditional on it. Measured cost:
the Steiner DP prices the (3,7,3) pool's 2,542 unique supports in ≈27 min per
process, cache per process, so each segment re-pays it once.

### Increment 4 — submit, monitored (PAUSE FOR APPROVAL FIRST)

Two arms. Segment length from increment 2, sized so each segment finishes in
**≤ 48 h** — a 24 h margin under the 72 h wall, because per-round cost grows
with depth and the last round of a segment is the most expensive.

| Arm | `--candidate-representation` |
|---|---|
| macro | `macro_generator_v1` |
| singleton | `sector_complete_pauli_block_v1` |

Both arms: `--insertion append_only`, `--checkpoint-history-tail 1`,
`--no-estimator-ledger` **only if** the estimator ledger is genuinely not
needed — note that resume validates a digest-matched estimator-ledger sidecar
(`sr_snake/_resume.py:4047-4059`), so **verify the sidecar contract still holds
with the ledger disabled before relying on it for a resumed segment.**

Submit-file requirements (the last run had none of these):

```
when_to_transfer_output = ON_EXIT
transfer_output_files   = results
+MaxRuntime             = 172800
periodic_hold           = (JobStatus == 2) && (time() - EnteredCurrentStatus > 172800)
periodic_hold_reason    = "segment exceeded its own 48h budget before the pool wall"
max_retries             = 0
request_memory          = 8GB
request_disk            = 10GB
```

`request_memory = 32GB` was 8× oversized — measured usage was 4,151 MB with a
34 MB/day creep; `request_disk = 30GB` against 293 MB used. Smaller requests
match faster.

**Stop condition:** any arm whose first segment does not return a `results/`
directory. Do not submit a second segment into an unexplained empty return.

---

## 5. The monitor

### 5.1 Authentication — never put the password in a file

The submit node requires password + Duo. **Do not store, echo, or type the
user's password, and do not accept it if offered.** Use SSH connection
multiplexing: the user authenticates **once**, and every later command reuses
the socket with no prompt.

Already configured in `~/.ssh/config`:

```
Host ap2001 ap2001.chtc.wisc.edu
    HostName ap2001.chtc.wisc.edu
    User jsstrobel
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 8h
    ServerAliveInterval 60
```

The user runs this once per session and answers Duo:

```bash
ssh -MNf ap2001
```

Check liveness before every automated action:

```bash
ssh -O check ap2001    # "Master running (pid=NNNN)" and exit 0
```

If the master is down, **stop and ask the user to re-run `ssh -MNf ap2001`.**
Never fall back to interactive password entry inside automation — it will hang
a background job forever waiting on a Duo prompt nobody sees.

`ControlPersist 8h` means a long campaign needs periodic re-authentication.
Design for it: the monitor must degrade to "cannot check, notify the user"
rather than blocking.

### 5.2 Cadence and duties

**Hourly:** job state, runtime against budget, quota.
**Every 12 h:** pull results locally.

`agent_guidance/skills/chtc-direct/SKILL.md` exists — read it before writing
automation; do not edit it.

```bash
#!/usr/bin/env bash
# scripts/paper_iv_h2o_monitor.sh — hourly CHTC watch for the Paper-IV arms.
# Read-only against the pool. Never submits, releases, or removes.
set -uo pipefail

CLUSTERS="${1:?usage: monitor.sh <cluster>[,<cluster>...]}"
LOCAL_DIR="${PAPER_IV_LOCAL:-$HOME/paper_iv_h2o_results}"
STATE="$LOCAL_DIR/.monitor_state"
WALL_BUDGET=172800          # 48h segment budget
WALL_WARN=144000            # warn at 40h
QUOTA_SOFT_MB=40960
QUOTA_WARN_MB=38000
mkdir -p "$LOCAL_DIR"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOCAL_DIR/monitor.log"; }

if ! ssh -O check ap2001 >/dev/null 2>&1; then
  log "FATAL ssh master down — user must run: ssh -MNf ap2001"
  exit 78
fi

for c in ${CLUSTERS//,/ }; do
  read -r status runtime mem disk <<<"$(ssh ap2001 \
    "condor_q $c -af JobStatus RemoteWallClockTime MemoryUsage DiskUsage 2>/dev/null" | head -1)"
  if [[ -z "${status:-}" ]]; then
    hist=$(ssh ap2001 "condor_history $c -limit 1 -af JobStatus ExitCode LastHoldReason 2>/dev/null")
    log "cluster $c LEFT QUEUE: $hist"
    continue
  fi
  case "$status" in
    1) log "cluster $c IDLE" ;;
    2) log "cluster $c RUNNING ${runtime}s mem=${mem}MB disk=${disk}KB"
       # Integer compare; RemoteWallClockTime is a float like 252164.0
       rt=${runtime%.*}
       if (( rt > WALL_WARN )); then
         log "WARN cluster $c at ${rt}s — approaching its ${WALL_BUDGET}s budget."
         log "WARN a pool hold at 72h transfers NOTHING. Escalate to the user now."
       fi ;;
    5) reason=$(ssh ap2001 "condor_q $c -af HoldReason 2>/dev/null")
       log "ALERT cluster $c HELD: $reason"
       log "ALERT do NOT condor_rm — removal destroys any spooled sandbox." ;;
    *) log "cluster $c status=$status" ;;
  esac
done

# --- quota guard (G3): a full /home loses a transfer like a hold does -------
used=$(ssh ap2001 "quota -s 2>/dev/null | awk '/md9/ {gsub(/[^0-9]/,\"\",\$2); print \$2}'")
if [[ -n "${used:-}" ]] && (( used > QUOTA_WARN_MB )); then
  log "ALERT /home at ${used}MB of ${QUOTA_SOFT_MB}MB soft quota — stage to /staging."
fi

# --- 12-hourly retrieval ---------------------------------------------------
last=$(cat "$STATE" 2>/dev/null || echo 0)
now=$(date +%s)
if (( now - last >= 43200 )); then
  log "retrieving results -> $LOCAL_DIR"
  for c in ${CLUSTERS//,/ }; do
    rsync -av --partial "ap2001:~/chtc/paper_iv_h2o_*/results/" \
      "$LOCAL_DIR/$(date -u +%Y%m%dT%H%M%SZ)_${c}/" 2>&1 | tee -a "$LOCAL_DIR/monitor.log"
  done
  echo "$now" > "$STATE"
fi
```

Schedule hourly (`cron`, `launchd`, or the `/loop` skill). **The monitor is a
tripwire, not a safety net** — §3 explains why nothing can save a job once the
hold fires. Its real jobs are: catch a segment drifting toward its budget while
a human can still act, confirm each segment landed, and keep a local copy.

### 5.3 Retrieval is only trustworthy once verified

A returned `results/` proves nothing until checked. After every retrieval:

```bash
python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('rounds', d.get('run',{}).get('stop',{}).get('completed_controller_rounds'))
print('stop_reason', d.get('run',{}).get('stop',{}).get('primary_reason'))
" <local>/current.json
```

Confirm `current.json` has its estimator-ledger sidecar beside it, and that the
sidecar's SHA-256 matches the pointer inside `current.json` — resume hard-fails
otherwise (`sr_snake/_resume.py:4047-4059`).

---

## 6. Traps

**T1 — never fresh-start into a directory holding `current.json`.** Before
`9a43bfe2` only `result.json` was guarded, so a restart silently overwrote the
checkpoint at its own round 1. The guard now exists; do not weaken it, and
never point a resumed run's `--output-dir` at the directory its
`--resume-checkpoint` lives in (also now refused).

**T2 — `condor_rm` on a held job destroys its spooled sandbox.** If a job is
held with data possibly in spool, retrieve first (`condor_transfer_data`), then
remove. Removal is irreversible and is what converts "maybe recoverable" into
"gone."

**T3 — `condor_release` restarts from round zero.** It does not continue a
held job's work. Releasing 9715945 would burn another 72 h into the same wall.

**T4 — `+MaxRuntime` alone enforces nothing.** It is a bare ClassAd attribute.
It only bites when a `periodic_hold`/`periodic_remove` expression references
it, which the old `d50pkg/submit.sub` did not have. The 72 h wall came from
pool policy, not from that line.

**T5 — 138 held jobs are in the queue.** Triage before adding more; a cluttered
queue makes monitoring output unreadable and some pools throttle on held
counts. Confirm with the user which are dead before removing any — some may
belong to other campaigns.

**T6 — use `.venv/bin/python`.** System `python3` is 3.9 and cannot import the
package.

**T7 — GitNexus MCP is unreachable on this host.** The repo mandates
`impact`/`detect_changes` before edits and commits. Substitute ripgrep caller
verification and **record the substitution in the commit message**, as
`9a43bfe2` does.

**T8 — do not edit files under `agent_guidance/skills/`.** Reading a skill to
follow it is required; changing one needs an explicit user request.

**T9 — evidence promotion is user-only.** Report numbers, validation outcomes,
and failures. Do not state that a result is or is not paper-promotable.

---

## 7. Out of scope

- `pipelines/excited_dynamics/**`, `pipelines/qse_spectra/**` — Paper III lane.
- Paper-I Hubbard–Holstein defaults, source locks, and comparators. Paper IV
  must not inherit them (`agent_guidance/molecular-vibronic/AGENTS.md`).
- The manuscript `MATH/paper_details/molecular_vibronic_h2o_paper_IV.tex`.
  Manuscript edits need `$journal-math-manuscript-refiner` and a rebuilt PDF —
  and **no TeX toolchain is installed on this machine** (`latexmk`, `tectonic`,
  `pdflatex`, `xelatex`, `lualatex` all absent). `brew install tectonic` is the
  approved fallback builder per `MATH/AGENTS.md`.
- Other agents' worktrees under `Holstein_test.worktrees/`. Stage commits with
  explicit paths only; never `git add -A` in a shared checkout.

Memory ceiling: all agent-launched local work combined stays under 18 GB
resident. Read `agent_guidance/shared/memory-budget.md` and wrap heavy local
processes in `pipelines/shell/ram_guard.py`. One heavy job at a time per
machine; many-job campaigns go to CHTC.

---

## 8. Report back

1. **Blockers first.** Anything in §0 that changed on inspection, stated in one
   sentence before anything else.
2. **Numbers, not narration.** Per-round seconds by arm from increment 2;
   completed rounds and stop reason per segment; energy against
   `E_ref = −75.015134943251 Ha`; whether the 1.6 mHa stop triggered.
3. **Commit SHAs and paths.** No restatement of work already recorded there.
4. **What you did not do, and why.** Especially anything skipped as blocked.
5. **Open decisions for the user,** at most two, each with a recommendation.

Update `agent_guidance/shared/paper-iv-worklog.md` in place. Do not create
parallel worklogs.

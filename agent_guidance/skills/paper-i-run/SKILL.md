---
name: paper-i-run
description: Plan and operate actual Paper-I RA-ADAPT campaigns and explicitly named compatibility/comparator studies, including source-locked materialization, local or CHTC launch, monitoring, repair, retrieval, validation, aggregation, and run-evidence reports. Trigger only when a request plans or changes run state, consumes completed run evidence, or produces a run report. Do not use for ordinary algorithm implementation, route mathematics, unit tests, conceptual review, agent handoffs, or manuscript prose.
---

# Paper-I Run Operations

Use this skill as the operational gate for actual Paper-I runs. It does not own
scientific defaults. The Paper-I lane, typed campaign plan, machine-readable
source locks, and `run_ra_adapt` own the scientific interface.

ICM describes which campaign stage is active. This skill governs how an agent
may safely cross the materialize, execute, monitor, repair, validate, analyze,
and evidence-handoff stages.

## Load order

Read only the smallest applicable surfaces:

1. root `AGENTS.md`;
2. `MATH/AGENTS.md`;
3. `agent_guidance/static-adapt/AGENTS.md`;
4. `agent_guidance/static-adapt/run-guide.md`;
5. `agent_guidance/shared/icm-gitnexus-pilot-plan.md` when materializing or
   executing a canonical campaign;
6. the exact visible target, source map, manifest, or compatibility support
   contract named by the request;
7. `agent_guidance/shared/run-guide.md` only for cross-paper routing.

Do not inspect manuscript source, artifact trees, route registries, or
historical handoffs merely to resolve an ordinary canonical request.

If the request becomes manuscript wording, a caption, a table edit, a `.tex`
change, or other PDF-facing work, stop this workflow and use
`$journal-math-manuscript-refiner`.

## Choose the run path

### Canonical campaign

For a new ordinary Paper-I Hubbard--Holstein campaign:

- materialize a `PaperICampaignPlan` from the campaign intent;
- resolve physics from the plan's hash-locked six-regime source;
- execute each cell through
  `pipelines.static_adapt.ra_adapt.run_ra_adapt(problem, request=None)`;
- use a typed `RAAdaptRequest` only for policies explicitly named by the user;
- consume the returned `PaperIRunSummary` for trajectory, plateau,
  matched-accuracy, Qiskit-resource, and `S_alg` observations.

Never enter `paper_i_runner.py`, Route A/B/C, JR-SNAKE, FM-SNAKE, Geo-ADAPT,
append-ADAPT, a versioned historical profile, or a legacy flag union from an
ordinary campaign.

Canonical scientific behavior comes from:

- `agent_guidance/static-adapt/run-guide.md`;
- the conditional policy document activated by user intent;
- the campaign plan's exact physics/source lock;
- the resolved typed route receipt.

Do not restate or override those defaults in this skill.

### Explicit compatibility or comparator work

Use a historical or comparator path only when the user names it or when an
explicit visible-target/source contract requires it. Preserve its exact route,
method, pool, optimizer, cutoff, seed, and source identity.

Compatibility work never changes the ordinary campaign default and never
falls back to another identity after failure.

### One-variable study

When the request claims “same settings except X,” also read and follow
`agent_guidance/skills/source-locked-sensitivity/SKILL.md`.

Materialize and validate the source-value anchor before generating or
submitting the remaining values. If a non-X scientific setting changes, stop
and report the difference unless the user explicitly expands the study.

## Materialization gate

Before execution, record:

```text
run_class
paper or explicit diagnostic target
campaign_id
method
ordered regimes/cases
typed request or exact compatibility identity
physics/source-lock path and SHA-256
route/profile digest
working cutoff and same-cutoff reference
optimizer and budget source
seed/reference-state source
expected manifest, checkpoint, ledger, result, and summary roles
execution target: local or CHTC
execution_authorized = false
```

Classify the run:

- `smoke`: a route/plumbing proof expected to run for less than three minutes;
- `diagnostic`: a named behavior or failure investigation;
- `candidate`: a complete run for scientific review;
- `paper_facing`: a user-approved run tied to a visible reader-facing target.

A smoke uses smaller diagnostic physics only when the request allows it. It
never substitutes for the locked candidate or paper-facing scale.

For a paper-facing rerun, repair, comparator extension, or CHTC submission,
resolve the current visible result to its source settings:

```text
python3 agent_guidance/skills/shared/scripts/resolve_visible_settings.py ...
```

Record the target, visible value, source map, source JSON, source hash,
settings reused, and settings changed. If the source settings cannot be
resolved, block materialization rather than inventing settings.

Materialization is non-executing. A valid plan is not run authorization.

## Execution authorization

Require current user authorization before:

- launching a local scientific run;
- submitting or resubmitting a CHTC job;
- starting an Optuna/settings campaign;
- regenerating run evidence that changes external or paper-facing state;
- promoting or replacing any result.

Ordinary implementation tests, plan validation, source-hash checks, and
read-only artifact inspection do not constitute a scientific run.

At execution:

1. verify the active non-iCloud checkout and local-file preconditions from root
   `AGENTS.md`;
2. verify the plan digest and every source lock again;
3. set `execution_authorized=true` in a new authorization receipt rather than
   modifying the locked plan;
4. create non-colliding output and log destinations;
5. emit a normalized machine-readable manifest;
6. invoke only the planned scientific seam;
7. preserve every effective runtime setting in the receipt.

Use `$chtc-direct` for direct CHTC submission, status, fetch, cleanup, or
storage operations.

## Monitoring

Follow root `AGENTS.md` adaptive monitoring.

### Paper-I status-completeness gate

When the active Paper-I objective has launched or submitted work, interpret a
bare `status` or `status?` as a request to reconcile every backend used by that
objective. Check local processes, the private remote runner, and CHTC before
answering. For CHTC, query the exact cluster IDs recorded at submission and use
`JobBatchName` only as supporting metadata because it can be undefined. An
existing authenticated CHTC session must be used automatically for a one-shot
read-only `condor_q`/`condor_history` check; do not stop at VPN reachability or
local process state. If no authenticated session exists, report `CHTC status
not queried` as a limitation and never collapse that partial observation into
`no Paper-I jobs are active`.

The response must name each applicable backend and its state. In particular,
`no local Paper-I process` and `remote runner idle` do not imply `Paper-I idle`
when CHTC work was submitted. Read-only status inspection is always permitted
by a status request; mutations remain separately gated.

For a local run:

- verify the command, settings, route, and output destination;
- observe one meaningful completed outer unit, such as one accepted controller
  round;
- then switch to cadence proportional to remaining runtime.

For CHTC:

- verify submission acceptance and expected scheduler records once;
- never continuously monitor the queue;
- reconcile scheduler state with meaningful progress artifacts at each
  scheduled poll.

At every poll inspect only:

```text
process or scheduler state
latest accepted checkpoint/progress receipt
error log
disk or quota when relevant
```

For status replies, lead with the owned scope and one compact row per
method/regime. Show operational state, current accepted round/depth, current or
final same-cutoff error, and relevant cost fields. Put scheduler IDs and paths
after the scientific status, not instead of it.

For each running Paper-I Hubbard--Holstein cell, let `k` be its latest accepted
controller iteration. Report that run's `|ΔE(k)|` together with the current
plateau-insertion RA-ADAPT and current Append-ADAPT `|ΔE(k)|` for
the same regime, cutoff, exact reference, and identical `k`; these are the
two baseline comparator families currently used in Paper I. Same-cutoff binds
each energy error to the same working cutoff and exact reference.
Same-iteration requires the exact same controller prefix `k` across all three
histories. Never substitute a terminal, plateau, or nearest-prefix value when
`k` differs; write
`unavailable at k=<k>` instead. If a comparator history omits the `k=0`
baseline, report it unavailable at `k=0`. Recommended compact columns are:
`method/regime | state | k/depth | run |ΔE(k)| | plateau RA-ADAPT |ΔE(k)| | Append-ADAPT |ΔE(k)| | cost`.

For a running CHTC cell, the accepted-state checkpoint is authoritative for
the live `(k, energy)` pair. Use one read-only `condor_ssh_to_job` inspection,
locate the execution-specific `cell_output/checkpoints/current.json`, and read
the checkpoint depth together with the final `energy_after_opt` from the same
opened file. These checkpoints can be multiple gigabytes, so do not copy them
or parse the full document with `jq` merely to report progress. A bounded-
output, single-pass extraction is:

```text
condor_ssh_to_job CLUSTER.PROC find /tmp -maxdepth 5 -type f \
  -path '*/cell_output/checkpoints/current.json' -print

condor_ssh_to_job CLUSTER.PROC 'awk '\''
  /"energy_after_opt":/ { energy=$2 }
  /^  "checkpoint": \{$/ { in_checkpoint=1; next }
  in_checkpoint && /^    "depth":/ { depth=$2 }
  in_checkpoint && /^  },$/ { in_checkpoint=0 }
  END {
    gsub(/,/, "", depth); gsub(/,/, "", energy)
    print depth, energy
  }
'\'' /tmp/.../cell_output/checkpoints/current.json'
```

Treat `hardcoded_adapt_iter` in `condor_tail` as a progress aid, not the
authoritative accepted-energy record. At event depth `d`, its `energy` is
`energy_before_refit`, which equals the accepted energy entering that round
(the accepted `k=d-1` state). Never label that field as the accepted energy at
`k=d`. If the checkpoint cannot be inspected, state the limitation explicitly
and apply only the documented `d -> k=d-1` fallback; do not invent a current
energy. After obtaining `(k, energy_after_opt)`, compute the run error against
the cell's same-cutoff exact reference and look up both comparator errors at
exactly that `k` in the current Paper-I Page-12 reference adapter.

## Failure and repair

A run failure is a repair trigger.

Repair implementation plumbing without changing scientific settings:

- missing directories;
- stale imports or entry points;
- path, schema, serialization, or manifest bugs;
- scheduler submit/progress wiring;
- report or Qiskit observation tooling.

After repair, run the narrowest proof and resume or rerun the same scientific
command.

Ask before changing:

- physical model, regime, cutoff, or exact comparison;
- method, route, pool, candidate representation, or optional policy;
- optimizer, budget, seed, stop condition, or accounting semantics;
- evidence meaning, visible target, or manuscript scope.

Never respond to a canonical failure by trying compatibility routes.

## Completion and validation

A completed cell must provide or explicitly classify:

```text
terminal execution receipt
normalized run manifest
typed route/problem receipt
accepted trajectory
checkpoint and replay identity
closed estimator ledger
PaperIRunSummary
result and sidecar paths with SHA-256
validation checks
```

The canonical `PaperIRunSummary` supplies:

- every accepted controller-round energy and same-cutoff error;
- effective-plateau prefix and compiled resources;
- canonical append-matched common-accuracy observations when available;
- requested-round compiled resources;
- closed occurrence-based
  `S_alg = N_H_outer + N_H_refit + N_grad + N_metric`.

One batch is one controller round even when active ansatz depth grows by more
than one.

A Qiskit/compiler defect is a retryable observation failure. It does not
invalidate an already accepted scientific trajectory. Repair the observation
tooling and rerun the same summary.

For a multi-cell campaign, produce a completion matrix with explicit
`done`, `failed`, `missing`, `blocked`, and `superseded` states. Folder
presence is never completion.

## Evidence and reports

Generate a run-support PDF only when the user requests one or the authorized
candidate/paper-facing workflow requires it. Build it from LaTeX and lead with
scientific trajectories, comparison results, and Qiskit/resource costs. Put the
parameter/provenance manifest in the final appendix by default.

Run evidence is preservation-first:

- never blank or replace a completed visible value merely because a newer run
  is queued, running, failed, or incomplete;
- report newer status additively;
- preserve source paths and hashes;
- do not edit manuscript tables unless the user separately authorizes the
  manuscript/evidence-transfer workflow.

Promotion and demotion are user decisions. Report objective validation,
provenance, metrics, and missing fields, then ask what the user wants to
promote, defer, rerun, or edit.

## Handoff

At final handoff, report:

- campaign and run class;
- method/regime status;
- final same-cutoff error and accepted round/depth;
- plateau and requested-prefix `N2q`, `D2q`, `Dc`, and `S_alg` when available;
- route/problem/plan digests;
- result, manifest, ledger, summary, validation, and report paths;
- exact blockers or retryable observation failures;
- whether any job remains active.

Do not require the user to open a PDF or JSON to learn the principal result.

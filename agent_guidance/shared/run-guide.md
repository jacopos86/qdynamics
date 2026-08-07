<!--
Agent-facing cross-paper run router.
Keep this file thin. Paper-specific scientific defaults belong in lane
interfaces and machine-readable source locks.
-->

# Shared Run Router

Use this file only after root `AGENTS.md` and `MATH/AGENTS.md` have identified
an actual run, evidence, or report workflow.

This file does not define scientific defaults, command catalogs, table values,
or compatibility routes.

## Route by paper

| Scope | Read next | State-changing gate |
|---|---|---|
| Paper I RA-ADAPT | `agent_guidance/static-adapt/AGENTS.md`, then `agent_guidance/skills/paper-i-run/SKILL.md` | use the typed Paper-I campaign plan or an explicitly named compatibility contract |
| Paper II AP-McLachlan dynamics | `agent_guidance/time-dynamics/AGENTS.md` when present, then `agent_guidance/skills/paper-ii-run/SKILL.md` | use the Paper-II run contract and calibration gate |
| Paper III QSE/excited dynamics | `agent_guidance/qse/ubiquitous-language.md` and `MATH/AGENTS.md` | no Paper-III run skill exists; paper-facing execution fails closed |
| Paper IV or V using an earlier method | the Paper-IV/V lane, then the underlying method's run skill | do not invent a paper-specific gate |

If a referenced lane file or skill is absent, do not reconstruct it from old
commands. Continue read-only inspection under root/lane contracts and fail
closed only when the missing contract is required for state-changing,
paper-facing work.

## Common sequence

For an actual run:

1. classify it as `smoke`, `diagnostic`, `candidate`, or `paper_facing`;
2. identify the explicit diagnostic question or visible reader-facing target;
3. resolve the authoritative source settings and hashes;
4. materialize an immutable plan with `execution_authorized=false`;
5. obtain current authorization before local execution or scheduler submission;
6. execute only the planned scientific seam;
7. validate startup through one meaningful outer unit, then use adaptive
   monitoring from root `AGENTS.md`;
8. repair implementation plumbing without changing scientific settings;
9. validate manifests, checkpoints, ledgers, results, and summaries;
10. report evidence status and leave promotion to the user.

A folder, queue record, PID, or generated file does not by itself prove a
successful run.

## Source-lock rule

For a paper-facing rerun, repair, comparator extension, or submission, begin
with the best current visible result for the same method and regime. Resolve it
to its source JSON/manifest and change only the user-requested variable.

Use:

```text
python3 agent_guidance/skills/shared/scripts/resolve_visible_settings.py ...
```

If the source settings or hash cannot be resolved, block the state-changing
operation. Do not substitute defaults, old commands, or reconstructed values.

For a claimed one-variable study, use
`agent_guidance/skills/source-locked-sensitivity/SKILL.md` in addition to the
paper run skill.

## Scale and evidence

Candidate and paper-facing runs use the scale and cutoff fixed by their
source-lock contract. Smoke settings never replace that scale or become
paper-facing evidence.

Every run emits a normalized machine-readable manifest. Preserve completed
evidence when newer work is queued, running, failed, missing, or incomplete.
New status is additive until a complete validated replacement exists and the
user decides what to adopt.

## Paper-I canonical seam

An ordinary Paper-I campaign uses:

```python
run_ra_adapt(problem, request=None)
```

Read `agent_guidance/static-adapt/run-guide.md` for the canonical scientific
interface, conditional policy routing, and materialization boundary.
Compatibility and implementation-history plans are not ordinary navigation
surfaces.

This shared router must not introduce Route A, legacy profiles, append-ADAPT,
Geo-ADAPT, pool filters, optimizer settings, pruning, batching, beam, or other
scientific choices into an ordinary request.

## Execution, monitoring, and repair

Do not interrupt a legitimate active run merely because it is heavy. Stop only
for an explicit request covering that run, a wrong physics point, failed
precondition, documented policy conflict, resource failure, debugging need, or
an immediate machine-safety emergency.

Jobs started from another chat, agent, terminal, notebook, remote-runner entry,
or scheduler submission are intentional and out of scope unless the user says
otherwise. Never cancel, pause, renice, replace, or delete their artifacts.

For a local job, verify the intended command, settings, output path, route, and
one meaningful outer unit before switching to adaptive checks. For CHTC,
confirm submission and expected scheduler records once; do not continuously
watch the queue.

At each check, inspect only the process/scheduler state, latest meaningful
checkpoint, error log, and storage/quota when relevant. Default intervals:

| Estimated time remaining | Check interval |
|---|---|
| under 10 minutes | remain with the job |
| 10--30 minutes | about 5 minutes |
| 30 minutes--2 hours | 10--20 minutes |
| 2--8 hours | 30--60 minutes |
| 8--24 hours | 2--5 hours |
| over 24 hours | about 10 hours |

Re-estimate from observed unit duration. Healthy unchanged state is not a
reason to poll sooner.

An implementation-plumbing failure is a repair trigger: diagnose, repair,
prove the fix narrowly, and resume the same authorized scientific command.
Ask before changing scientific semantics, settings, cutoffs, seeds, route
identity, evidence meaning, or scope. A status request does not cancel an
already-authorized run objective.

Use the matching available CHTC operation workflow when directly operating
CHTC.

## Handoff

At handoff, report the scientific status first:

```text
method and regime
queued/running/done/failed/blocked
accepted round and active depth
current or final primary error
resource/accounting fields when available
source and validation state
```

Scheduler identifiers and paths are secondary provenance.

No run, report, artifact, setting, or table value is promoted or demoted
without the user's explicit decision.

# Agent handoff contract (Claude author → Codex executor)

Applies when one agent plans work and another executes it inside this
checkout. The usual split: Claude holds long context — architecture, why a
design is the way it is, what the manuscript claims, which evidence is
load-bearing — and Codex executes plans, audits, and detail work with repo
access. Both agents read this file; it is the shared definition of a handoff
that can be executed cold.

This contract assumes the executor **has repo access** and can run commands,
tests, and commits. For an external model without repo access, the dossier
conventions in the GPT-Pro handoff workflow apply instead.

## The test a handoff must pass

Could a competent agent with no memory of the authoring conversation execute
this and produce work you would accept without rewriting? Everything below
exists to make that answer yes.

## 1. Anchors — state facts, never "as we discussed"

| field | why it matters |
|---|---|
| Checkout path | `/Users/jakestrobel/local_repos/<repo>` is where work happens. `~/Documents/Holstein_implementation/<repo>` is an iCloud mirror; running there silently produces work in the wrong tree. |
| Branch and commit SHA | Several agents share this working tree. Without a SHA the executor cannot tell whether files changed underneath it — and they do. |
| Test baseline | The command that must stay green and its current count, plus the file recording pre-existing failures. |
| Entry point | The full copy-pasteable command with every flag that matters. Partial commands cause the guard omissions in §4. |

## 2. Scope — say what is out of bounds, and why

A reason travels where a bare rule does not:

> Out of scope: `pipelines/excited_dynamics/**`, `pipelines/qse_spectra/**`
> (Paper III lane, another agent commits there daily). If the work appears to
> require touching them, stop and report — a conflicting edit costs more than
> the delay.

Also state shared-resource limits: RAM ceiling, one heavy local job at a time,
cluster access the executor cannot obtain alone.

## 3. Increments — each ends committable and verifiable

For every increment give four things:

1. **Goal** — one sentence on what is true afterward.
2. **Exact commands** — real paths and flags.
3. **Expected result** — a number, decision, or output that can be checked.
   "Tests pass" is weak; "226 passed, and `plot_rows` drops 16 macro-scout
   columns" is checkable.
4. **Stop condition** — what would mean the increment is wrong, and what to do
   then (usually: stop and report, do not improvise).

State explicitly whether the executor proceeds through all increments
autonomously or pauses between them. Silence on this wastes a cycle.

## 4. Evidence standards — prove before acting

Plausible reasoning produces confident mistakes. What has actually failed here:

- **An index proposes; source ratifies.** GitNexus reachability marked four
  cost symbols unreachable from the runner entry point; source review showed
  three were live, invoked through closures passed as callables — an edge the
  static call graph does not follow. Require grep for callers, confirmation in
  source, and green tests before any deletion.
- **"Never fires" needs a measurement.** The repair-ladder simplification was
  justified by an audit over 885 integration steps showing 6 of 11,499
  candidates applied (`pipelines/time_dynamics/diagnostics/knob_audit.py`).
  Ask for that kind of number, not an argument.
- **Behavior parity needs a lock.** `test/test_ap_mclachlan_route_parity.py`
  pins decisions, support evolution, and energies. It must pass unchanged, or a
  deliberate change must appear as a visible diff with justification.

## 5. Traps — write down what bit you

Every trap the author hit, the executor will hit blind:

- Computational guards are mandatory at scale. Omitting
  `--max-joint-patch-evaluations` turns a four-checkpoint run into an unbounded
  grind. Hand over the fully guarded command.
- Runner defaults are diagnostic, not canonical: Euler with loose repair caps
  gives 1.6e-2 energy error where rk4 with tight caps gives 1.3e-3.
- `output/` and `prompt-exports/` are gitignored — work saved there will not
  appear in a commit. Put durable artifacts in tracked paths and say where.
- Another agent's broad `git add` can sweep uncommitted work into their commit.
  Commit at increment boundaries rather than accumulating.

## 6. Report back — one block per increment

```markdown
## Increment N — <goal>
Status: done | blocked | skipped
Commands run: <exact>
Result: <numbers, decisions, diffs>
Verification: <test counts, run outputs, before/after values>
Deviations: <anything done differently, and why>
```

Include the commit SHAs produced. A blocked increment reports the blocker
rather than a workaround. Surprises get surfaced even when the increment
succeeded — a surprise usually means an assumption in the plan was wrong.

## Where handoffs live

Task-scoped handoffs sit beside the work they describe
(`chtc/<campaign>/HANDOFF_<TOPIC>_<YYYYMMDD>.md`); route- or lane-level
handoffs go in `prompt-exports/`. Cross-reference companions by filename. Note
`prompt-exports/` is gitignored: if the handoff must survive in version
control, place it in a tracked directory and say so in the document.

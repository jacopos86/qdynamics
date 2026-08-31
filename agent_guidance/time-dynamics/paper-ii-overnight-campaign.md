# Paper II overnight campaign supervisor

This is the desired unattended operating contract for the staged
matched-accuracy campaign. It becomes executable only after the live operations
adapter, scope-complete snapshot producer, stable lane bindings, and recurring
service preflight pass. It automates execution of already-settled decisions; it
does not authorize a new scientific method, a new numerical protocol, or
evidence deletion.

The deterministic orchestration authority is
`pipelines/time_dynamics/acquisition_supervisor.py`, configured by
`pipelines/time_dynamics/campaigns/time_dynamics_acquisition_registry_v1.json`.
See `agent_guidance/time-dynamics/acquisition-supervisor.md`. The numbered
requirements below define the Paper-II lane adapter; they must not be manually
reconstructed as an alternative turn-level workflow.

## Automatic loop

After production activation, each scheduled heartbeat must:

1. Check report freshness before any planning, monitoring, evaluation, or
   submission. Compare the canonical manifest against the immutable ledger and
   all verified completion/fetch receipts. If stale, immediately rebuild and
   compilation-check a lightweight LaTeX status view from receipt metadata. Show
   unevaluated verified trajectories as `gate_pending`; do not wait for the
   heavy exact-reference slot. A stale PDF is itself a material failure state.
2. Inspect active CHTC clusters once and identify every newly completed process
   not already covered by an incremental-ingestion receipt. Cluster completion
   is not the result-visibility boundary. In the same snapshot, identify every
   unfinished process whose running, idle, or held state or available progress
   telemetry materially changed since the last digest-bound partial-evidence
   receipt.
3. As soon as one or more processes finish, package only that newly completed
   delta: the exact per-process outputs and logs, its action rows, and a
   per-process disposition. Verify the submission receipt and the narrow remote
   archive. Never wait for the slowest process merely to form a whole-cluster
   bundle.
4. Run the shared storage guard before each incremental transfer or extraction.
   Fetch the narrow archive through one `.part` file, verify size, SHA-256,
   gzip, and tar integrity, then rename atomically. If the guard refuses,
   preserve the remote artifact and stop. A later whole-cluster archive is
   closure evidence only; it does not supersede or delay earlier increments.
5. Validate each `run.json` against its staged action and achieved live
   configuration. Exit status alone is never evidence.
6. Append every newly verified rung to the immutable ledger immediately,
   including discarded rungs and safeguard outcomes. Ingestion accepts any
   nonempty exact subset of the prepared package, is keyed by action ID and
   verified run path, and is idempotent when an already-recorded increment is
   replayed.
7. Compute the declared exact-response-normalized classification from the stable
   content-addressed full-horizon reference for that seed and drive. Generate
   that reference once per physical regime, reuse its `t=5` prefix and complete
   `t=20` grid for every method, threshold, controller, and ablation, and fail
   on a cache-key or grid mismatch. Never propagate an identical exact
   reference once per run. Normalize the three observable errors by the bound
   RMS exact excursions for that drive and horizon. Define each observable
   burden as `q_O=(MAE_O/RMS_O)/0.05` and fidelity burden as
   `q_F=mean_infidelity/1e-4`. The individual target is `q <= 1` componentwise;
   the comparison guard is `q <= 3` componentwise, including fidelity, plus
   the declared observable absolute-safety ceilings. The fidelity target is a
   normalized accuracy component, not an additional absolute-safety ceiling.
   A completed
   APM/AVQDS pair is also admissible when both arms pass that guard and their
   aggregate accuracy burdens agree within a factor of two. Every primitive
   normalized component burden must also agree within a factor of two after the
   declared deadband, `max(q, 0.10)`, is applied. Aggregate agreement never
   compensates for a primitive-component mismatch. An unmatched miss schedules
   the next registered method-owned rung in the direction prescribed by the
   frozen comparator policy. A qualified `t=5` arm
   may resume to `t=20` while unrelated calibration continues, and an already
   qualified stitched `t=20` result does not require redundant `t=5`
   recalibration.
8. After every ledger mutation, rebuild the canonical
   consolidated LaTeX PDF at
   `output/pdf/paper_ii_staged_accuracy_campaign/paper_ii_staged_report.pdf`
   and verify the successful LaTeX build plus manifest/source/PDF digests before
   submitting more work. Routine data-only refreshes must not spend time on
   page rendering or visual inspection when the established table and plot
   structures are unchanged. Render-inspect only a newly introduced or
   materially changed report structure. Paper-II manuscript promotions retain
   their separate changed-page visual check. Page one must show every recorded rung and its
   status across both horizons. Complete `t=20` APM/AVQDS pairs add comparison
   pages keyed by physical regime and seed provenance, drive, and time-step
   controller. Each page uses the same verified exact reference as both method
   arms and carries a digest-bound tuple identity. The pruning comparison additionally requires the matching
   APM insertion-face (`append_only`) ablation to pass independently. Its
   physics display uses total energy, mean doublon occupation, and charge
   imbalance; full APM is solid dark red, AVQDS dashed blue, and the shared
   exact reference dotted black. Accuracy errors remain in the table rather
   than replacing a physics panel. Qualified `t=5` pairs remain screening evidence. The absence
   of a `t=20` pair produces a status-only PDF, never no PDF. A stale report
   blocks submission.
   Independently of qualification, every complete exact-evaluated `t=20`
   APM/AVQDS pair adds a clearly labeled promotion-preview page using the latest
   evaluated rung per method. The page uses the final physics-panel and compact
   table contracts, lists the remaining failed gates, and is not a manuscript
   claim. Preserve and refresh these pages after each relevant ledger mutation
   so the author can inspect candidate results before promotion. This
   results-PDF tuple matrix is a hard precondition for the manuscript table.
   The matrix contains every planned physical-regime, drive, and time-step-
   controller cell even before both methods finish. Paper II mirrors the same
   exact/APM/AVQDS tuple scaffold. Every completed, exact-evaluated arm exposes
   its numerical values: `qualified` when the individual target or matched-pair
   rule passes, or `candidate` with
   the failed-gate/extension state when they do not. Only arms without a valid
   completed trajectory remain explicit `pending` placeholders. The manuscript
   table may be regenerated immediately afterward in the same campaign cycle,
   but it must consume that verified scaffold and its slot states rather than
   selecting runs independently.
   Independently, the results PDF contains an unfinished-job section for every
   active or held process. It reports the exact submitted action, scheduler
   state, cut, target horizon, and only those progress values actually captured
   in a verified log or serialized checkpoint. A runner that writes
   `run.json` only on graceful completion may leave a held job without
   observable errors; the report must say so rather than suppressing the
   attempt or fabricating values. These `partial_running` and
   `partial_held` rows are author-review evidence, not ledger classifications
   or automatic manuscript rows.
9. Submit only the exact next actions replayed from the ledger: the next
   per-cell threshold rung or a qualified arm's decision-identical continuation.
   For each drive/controller cell, calibrate AVQDS first. Its individual target
   or comparison-guard cut may open the structural comparator. Use that cut as
   the common initial rung for full APM and insertion McLachlan (IM);
   thereafter each APM arm extends its own insertion-cut ladder independently.
   The borrowed cut is calibration, not a matched-threshold requirement.

## Incremental result invariant

An active multi-process cluster is a stream of independently usable scientific
artifacts, not one indivisible result bundle. Every heartbeat computes

`newly completed = completed process IDs - process IDs in durable incremental receipts`.

If that set is nonempty, the supervisor must, in the same campaign cycle,
create and verify a delta archive, transfer it, validate and ingest its runs,
and rebuild plus compilation/digest-check the canonical PDF. Remaining running processes do
not block any of those actions. The incremental receipt records the cluster ID,
covered process IDs and action IDs, archive size and SHA-256, extraction root,
ledger digest before and after ingestion, and report-manifest digest after the
refresh. A final cluster archive may later prove full coverage, but reporting
must never wait for it.

### Snapshot-to-report transaction

For this already-authorized campaign, every scheduler snapshot is a transaction
boundary, including a snapshot taken because the author asked only for status.
When the snapshot discovers a nonempty `newly completed` set, the turn must not
end after merely reporting scheduler counts. It must finish the exact delta's
verified package/fetch/validation/evaluation/ledger/report/manuscript-table
pipeline first. The only permitted early exit is a concrete failed guard; in
that case write a durable process-scoped `incremental_refresh_blocked` receipt,
refresh the lightweight PDF with those processes shown as `gate_pending`, and
name the failed guard. “Completed but awaiting ingestion,” “the next heartbeat
will update it,” and an unchanged PDF are forbidden terminal states.

When `newly completed` is empty but an unfinished process has a material
scheduler or telemetry transition, the transaction still refreshes the
results PDF before ending. Package or capture only the narrow scheduler logs
and progress telemetry needed for those process IDs, verify and receipt them,
and update the partial-evidence section. Do not append partial attempts to the
immutable scientific ledger and do not regenerate the manuscript table unless
the verified completed-slot promotion contract changed.

The transaction reuses the campaign-construction arguments bound by the
submission package. In particular, `prior_root` is tri-state provenance:
absent, or present with its exact path and digest. Ingestion must preserve that
state exactly. Adding a prior root to a package prepared without one, omitting
one that was present, or substituting another root changes the threshold
schedule and is an artifact-validation failure rather than evidence that the
submitted action was stale.

Cumulative estimator-work accounting is postponed. It is neither an automatic
action nor a gate for accuracy evaluation, threshold extension, ablation runs,
or report freshness. The main Paper-II table reports full-horizon mean
infidelity; full-horizon mean absolute energy, doublon, and charge-imbalance
errors; accepted integration steps; terminal active coordinates; and the
routing-independent terminal graph tuple. It omits the reader-facing evidence
column: completed arms expose numbers, unfinished arms remain placeholders, and
qualification is read from the declared response-normalized and pair-matching
contract. Other available
counters remain in receipts rather than being mixed into that table.

Automated manuscript refreshes are table-only. Do not add or replace Paper-II
observable figures when a new tuple completes. Keep the complete set of
`E(t)`, `D(t)`, and `Delta n(t)` tuple plots in the campaign results PDF; the
author will select the small publication subset before submission. Preserve
the currently admitted representative manuscript figure unless the author
requests a specific replacement. Changed manuscript-table pages still receive
the manuscript visual check.

## Visibility invariant

Completed work must not be hidden behind the accuracy gate. The report
distinguishes evidence roles rather than omitting rows:

- `accuracy_qualified`: the individual five-percent response target passes, or
  the completed APM/AVQDS pair passes the fifteen-percent physical guard and
  factor-two burden-matching rule; the record reason identifies which route;
- `requires_threshold_extension`: the individual target is missed and no
  admissible matched pair is yet present, so the next tighter registered rung
  is scheduled;
- `safeguard_tripped`: the artifact is provenance only and the reason is shown;
- `gate_pending`: the verified run exists but exact-reference evaluation has
  not yet completed.

`gate_pending` is an operational report state, not a scientific ledger status.
It may be sourced from a verified ingestion receipt while the immutable ledger
awaits its exact metrics. The next heartbeat must prioritize resolving it. The
frontier PDF must show its action, method, drive, controller, horizon, cut, and
artifact-validation state; it must not show provisional errors as qualified
evidence.

Unfinished scheduler evidence has two additional results-PDF-only roles:

- `partial_running`: a job is still executing; show the achieved time,
  checkpoint, support size, and other directly captured progress telemetry when
  available, while leaving exact errors blank unless a valid checkpoint state
  was serialized and evaluated offline;
- `partial_held`: a job is held or evicted; show the exact action and concrete
  scheduler reason, plus any valid serialized progress that survived.

Neither role is a ledger status or a manuscript-table state. Both exist so that
unusual dynamics, support growth, or resource failures remain visible to the
author before a completion decision is made.

## Resource invariants

- Preserve at least 10 GiB free on the local data volume. Unknown JSON-heavy
  archives reserve twelve archive sizes for extraction.
- Put every heavy local process under the 8,000 MB RAM guard and run only one
  heavy local process at a time. If system free memory is below 30%, defer the
  work to CHTC or the next heartbeat.
- Keep resume blobs out of Git. Retain the verified compressed archive and
  uncompressed ledgers, receipts, summaries, and report inputs.
- Never infer supersession from `partial`, `live`, or `final` names. Produce a
  per-file SHA-256 coverage proposal; deletion still requires author approval.
- Never delete remote raw output before verified local ingestion, and never
  delete preserved or adopted evidence.

## Policy-transition ingestion bridge

Jobs submitted before an accuracy-policy transition remain bound to the
campaign constructor and policy recorded in their package. Ingest those jobs
against the preserved submission-era ledger with
`--submission-policy-replay componentwise_1e-4`; this option is ingestion-only
and must never prepare or submit new work. After each legacy increment, replay
the enlarged submission-era ledger through the canonical policy into
`output/paper_ii_staged/ledger_current.json`, then complete the normal
results-PDF-first promotion transaction. While this bridge is active, do not
submit canonical-policy work from a ledger that has diverged from the preserved
submission-era ledger. Raw evidence is retained under both classifications;
the policy transition changes admission and scheduling, not measurements.

## Submit automatically only when

- the action is emitted by the current ledger and registered campaign matrix;
- source revision, seed digest, resume digest, arm, controller, numerics, cut,
  horizon, and run lock all pass their existing audits;
- no equivalent action is queued, running, completed, or already recorded;
- local and remote storage guards pass; and
- the action changes neither algorithm mathematics nor the now-declared
  response-normalized matched-pair accuracy policy.

## Stop and report

Stop the affected lane, without guessing a repair, on a scheduler hold, receipt
or digest mismatch, duplicate action, run-lock mismatch, storage refusal,
memory-guard kill, exhausted declared threshold schedule, controller safeguard,
missing exact reference, or report failure. Other independent cells may
continue.

Expired CHTC authentication is an access transition, not a campaign blocker and
never a satisfactory terminal status while remote work remains. When a queue
inspection, packaging step, fetch, or submission is due and no authenticated
session survives, the author's standing instruction is to run the trusted
preflight and automatically initiate one new SSH login and one Duo push; the
author will accept the notification. The supervisor must not ask another
readiness or authorization question before initiating that login. It must not
report merely that
``CHTC authentication remains unavailable``, claim that no action is needed, or
silently defer the remote frontier.

Keep the authenticated SSH PTY open and reuse it across heartbeats, with a
roughly 12-hour reuse target; never reconnect once per heartbeat. Verify the
session before reconnecting because its actual server lifetime is not
guaranteed. This standing instruction authorizes initiating the push, not
accepting it: the author accepts Duo. Issue at most one push per login attempt
and do not automatically retry a denied or timed-out push. While a push is
pending, continue any safe local ledger, report, validation, or packaging work
that does not require CHTC access.

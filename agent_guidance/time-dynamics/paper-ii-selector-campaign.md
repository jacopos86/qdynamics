# Paper II selector campaign

Updated: 2026-09-09. **Original local pilot: 24/24 completed; 6/6 singleton
parity pairs passed. Incremental/rung revision: 24/24 low/high-budget reruns
completed; 6/6 singleton pairs passed. Natural fixed-UCCSD seed: 24/24 local
trajectories completed; 12/12 singleton pairs passed.** The separate three-drive scientific campaign still has
**72 full trajectories submitted** (`10024165`–`10024176`, six jobs each). The six currently running
recovery jobs are separate from that matrix; the full campaign remains active
work, not a completed or cancelled objective.

Compare search policy, gain evaluation, insertion batching, and work budget
across three drive regimes. Full trajectories are the primary comparison.
Keep singleton joint/factorized pairs as end-to-end parity checks; record any
verified shared route before considering a reduction in the run matrix.

Implementation baseline: `206d3a204edcedbaaf0274d9396483e8837b7434`.
Pin the actual tested revision and executor package hash in the submission
manifest. Existing small trajectory tests are software validation, not results
from this campaign.

Before campaign execution: the user requested many small local debug runs.
The [current selector audit](paper-ii-selector-audit-20260908.md) precedes
those runs and records confirmed proxy redundancy, permission-budget coverage
effects, and the distinction between the new searches and the old frontier
path. The subsequent authorized correction removes redundant singleton proxy
work and rejected-set evaluation charges. The user deferred the legacy
frontier-policy change; its utility and generic defaults remain intact.
The local pilot below is complete. The expanded scientific grid remains pending.

## Current campaign scope (2026-09-09, author clarification)

The full comparison remains **3 searches × 2 gain modes × 2 admission settings
× 2 work budgets × 3 drives = 72 trajectories**. Use t=20 on 501 points,
ascending tiers, and the already selected Paper II state-motion settings for
fastweak, slowstrong and strongfast. The matrix uses work budgets **10,000 and 50,000**, following the author's
direction to proceed with submission on 9 September. Earlier 256/24,576
values describe local pilots, not a new adopted matrix configuration.

The six running recovery jobs `10024096.0–5` use fastweak, W10000, A1 and the
15% ansatz-relative deletion bound. They vary only search and gain. They are
neither the strict A1/D1 arm nor the A4/D15% batching arm, so they do not fill six
of the 72 cells. Preserve and report these useful recovery results separately.
The completed local W50000 A1/D15% run is likewise separate.

The earlier overnight 18-attempt limit describes that bounded execution phase;
it is not a continuing restriction on the author's already requested matrix.
The heartbeat must track the larger campaign through preparation, qualification,
submission, collection and validation. Completion of the six recovery jobs alone
does not complete the campaign. Preserve existing running jobs and source locks.

Routine automation runs directly in the root task. Subagents are authorized for
bounded performance investigations, implementation and independent review.
The Fable second pass is completed in
`output/paper_ii_selector_performance_audit_20260909/CODEX_SECOND_PASS.md`;
use its disposition ledger rather than repeating the audit. Performance work
proceeds alongside the running jobs and is not an automatic submission hold.

The strict singleton arm remains A1/D1. The batched arm is A4 with deletion size
bounded by the existing 15% rule and complete-set permissions. Lower sections
saying drives, horizon or singleton interpretation are pending are historical;
these choices are resolved above. All 72 variant-aware action identities and effective configurations are now
committed at source `7eeb05193dd8529616747a57cadd0e25bcec142c` and submitted.
Runner/worker bytes equal the qualified `339a1d05` source; only locks differ.
No further general permission to use CHTC is required.

The full matrix is **submitted and queue-verified, with no holds at submission**.
Twelve six-action packages passed local validation, all 72 commands passed the
exact worker-container source/argument checks, and all Condor dry runs passed.
The initial queue snapshot had 72 queued jobs; later states belong in receipts.

Submission and source receipts:
`/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-full-matrix-20260909/output/paper_ii_full_matrix_build_20260909/submission_receipt.json`
and `remote_qualification_receipt.json` beside it. The source/argument proof
covers 855 distinct archived files. Pre-execution calibration reports zero
errors; new trajectory accuracy and continuity are still post-execution checks.
Keep all 72 matrix jobs and the six separate recovery jobs in the monitor.

## Corrected regression diagnosis and current work (2026-09-09 UTC)

This section supersedes the older next-step and pending-approval statements below. **Zero new-selector t20 trajectories completed among the seven timeout attempts. Historical Paper II t20 results did complete.** The user has explicitly authorized the implementation/execution correction and subsequent CHTC work; the older W64-package approval request is historical and is not the current blocker.

The actual a43 fast-weak state-motion run reached t=20 on the 501-point grid. Its archived `run_lock.numerics` equals the preserved 07bc W256-prefix numerical contract, including Euler, ridge `1e-6`, rcond `1e-10`, zero damping, state-motion bound `5e-5`, insertion cut `3e-6`, and the repair settings. Parent-observed `condor_history` reports cluster `9720203.3`, exit0, and **6,610 seconds** (1h50m10s), with9 CPUs and12,288 MB requested. Search and execution settings differ:

| Setting/evidence | Successful a43 fast-weak state-motion | Failed new-selector runs / preserved prefix |
|---|---|---|
| Search work budget | 50,000 | 256 remotely;64 locally |
| Maximum insertion block | 1 | 4 |
| Scoring / compile workers | 4 / 8 | 1 / 4 |
| First accepted historical word | `yyeexx` | Absent from the observed W256 evaluated prefix |
| Search route | Historical `family_frontiers` | New ascending searches |

These are concrete discrepancies to diagnose; they do not assign the entire runtime regression to one cause. The independently verified adjacent-duplicate ridge gain remains a separate observation. **It is not a demonstrated introduced regression, and resolving it is not a prerequisite for this correction or subsequent CHTC work.** Ridge and scientific acceptance settings remain unchanged.

Current diagnostic status, bound to the receipts below:

- The current-source `family_frontiers` case was stopped as an **invalid historical control**. Its rejected-deletion scan is unbounded in a way the actual a43 route was not. The stop receipt preserves this failure; parent observed guard session74839 exit241. This finding is scoped to the current family path, separate from the new searches' bounded discovery.
- The actual a43 source control is **prepared, not run** (809 source files verified).
- The current greedy/joint case started at12:48:52 UTC with W50,000, maximum insertion block1, scoring4/compile8, ascending scheduling, and t=0..0.08 on three points. Its 12GB-guard launch verifies842 frozen source files. Completion is not asserted here. Maximum deletion block remains unset under the new route's existing permission/cardinality policy; this is not a claim of identical search semantics.

The next work is the bounded differential, the smallest justified coverage/execution fix, and focused validation, followed by the **existing Paper II operational workflow**: lock, source-bound package, validate, qualify, submit, collect by receipt, and score. The protocol was located at commit `7c9394a7`; this record does not create a second packager. Preserve all seven timeout archives, source locks and historical successes. The 72-action matrix remains unlaunched; paper adoption remains separate. The unimplemented checkpoint proposal is not a new blocking prerequisite.

[Historical comparison and exact settings](../../output/paper_ii_protocol_regression_audit_20260909_671b4963/independent_comparison_01.md) · [Current-family regression](../../output/paper_ii_successful_protocol_differential_20260909_671b4963/family_rejection_regression_01/finding.md) · [Stopped control receipt](../../output/paper_ii_successful_protocol_differential_20260909_671b4963/family_frontiers_joint_batch_a1_w50000_compile8_score4_t008/debug_stop_receipt.json) · [Actual a43 preparation](../../output/paper_ii_successful_protocol_historical_a43_20260909/preparation_receipt.json) · [Greedy launch receipt](../../output/paper_ii_successful_protocol_differential_20260909_671b4963/greedy_joint_batch_a1_w50000_compile8_score4_t008/launch_receipt.json) · [Preserved operational protocol](../../output/paper_ii_protocol_regression_audit_20260909_671b4963/state_update_01/operational_run_protocol_7c9394a7.md)

## Final local overnight attempt: timeout (2026-09-09 UTC)

The separate macOS greedy/joint W64 attempt also timed out after21,600.632 seconds, at10:42:27 UTC. Last completed progress was checkpoint59, t=2.36, with663 parameters; checkpoint60 at t=2.40 had started. The case receipt is `failed`, exit124, reason `case_deadline_21600s`. Outer driver exit0 only records successful preservation of that failure.

The inner guard sampled4,952 MB and the outer guard4,980 MB, below their12,000/13,000 MB limits. No owned trajectory/driver processes remained at reconciliation; its13GB reservation is released. Independent verification passed all842 frozen source hashes and exact membership, plan/case/authorization bindings and the four recorded artifact hashes. There is no `run.json`, `measurement.json` or resumable checkpoint, so no success reporter or post-run physics/resource calculation was invoked.

**All seven new-selector t20 attempts are now terminal timeouts; zero new-selector t20 results completed.** The remote W256 and local W64 attempts differ in platform and budget; timeout progress does not establish a controlled speed ratio or a winning selector. No further scientific attempt, threshold change, acceptance-floor change or remote upload occurred.

The next-step and approval assessment recorded at this timeout was superseded by the corrected regression section above. Its original text is preserved in the [pre-update campaign snapshot](../../output/paper_ii_protocol_regression_audit_20260909_671b4963/state_update_01/paper-ii-selector-campaign.before.md). Compiler worker reuse is qualified locally, and the future output-remap correction passed19 checks. The [checkpoint callback proposal](../../output/paper_ii_t20_exploratory_20260909_07bc223e/returned_w256_attempt01/checkpoint_persistence_proposal_01.md) remains unimplemented; interruption/resume parity is required only before relying on that proposed recovery capability.

[Verified local timeout receipt](../../output/paper_ii_t20_local_w64_20260909_07bc223e/terminal_timeout_reconciliation_01.json) · [Current overnight status](../../output/paper_ii_t20_exploratory_20260909_07bc223e/status.md)

## First overnight t20 tranche: six trajectory timeouts (2026-09-09 UTC)

All six weak-fast W256 attempts in cluster `9887737.0–5` ended normally with exit code1 at the worker deadline, about21,540 seconds. Verified `failure.json` records classify every attempt as `TIMED_OUT` in `trajectory`; this was neither a completed t20 result nor a reporting-stage failure.

| Search | Gain | Last completed log time | Parameters | Outcome |
|---|---|---:|---:|---|
| greedy | joint_batch | 0.84 | 272 | trajectory timeout |
| greedy | singleton_factorized | 0.96 | 310 | trajectory timeout |
| block_exchange | joint_batch | 0.96 | 310 | trajectory timeout |
| block_exchange | singleton_factorized | 0.92 | 298 | trajectory timeout |
| bounded_exhaustive | joint_batch | 1.04 | 336 | trajectory timeout |
| bounded_exhaustive | singleton_factorized | 0.96 | 310 | trajectory timeout |

The six archives and receipts plus18 scheduler files were recovered and downloaded. Archive hashes, manifest/attempt identities, inventories and settings passed ingestion; the independent audit passed342 checks. Scheduler sampled memory was1,169–1,390 MB against12,000 MB requested, with no memory-kill, hold or eviction evidence. Only progress and failure records survived: there is no trajectory JSON, resumable checkpoint, complete work ledger, physical-error report or terminal compiled cost. Progress is not a substitute for those artifacts.

An output-path defect returned the archive/receipt basenames into the submit directory instead of the configured staging directory. The files were recovered intact. The descriptor uses `archives/...` on the left of `transfer_output_remaps`, although explicitly transferred output paths are flattened to basenames by default. This location defect did not cause the trajectory timeouts. A separate, local-only future-helper correction passed19 checks and independent review; all112 protected files and the pending W64 upload archive are unchanged. Actual endpoint transport verification is still pending. [Transport correction and validation](../../output/paper_ii_t20_exploratory_20260909_07bc223e/preparation_worker_transport_fix_01/README.md).

At 10:27 UTC, the separate macOS W64 attempt remained active at last completed log time t=2.32, 650 parameters, entering t=2.36; its deadline is approximately10:42 UTC. Across the seven actual attempts, zero t20 results have completed. There are no additional remote submissions.

The short-prefix compiler-dominance and narrow generator-coverage findings remain separate evidence. These timeouts do not provide complete phase timing or establish how much ansatz growth the single verified duplicate-ridge mechanism caused. Do not rank the search methods using their timeout progress or silently change scientific acceptance to obtain a completion.

[Verified aggregate](../../output/paper_ii_t20_exploratory_20260909_07bc223e/returned_w256_attempt01/comparison_01.json) · [Independent findings](../../output/paper_ii_t20_exploratory_20260909_07bc223e/returned_w256_attempt01/independent_audit_01/findings.md)


### Recovery implementation and launch update (2026-09-09)

Deletion discovery is bounded in commit `2c00f5a4` (99 focused tests). Independent execution windows are committed in `2f5d8271` (174 focused cases; exact HH prefix science/work parity). The observed 260.12-to-245.18-second prefix comparison is a single timing observation, not a broad speedup claim. Ridge, thresholds, conditioning and certification were preserved.

A full local t20/501 greedy joint A1/D15% baseline remains active at immutable source `2c00f5a4`, W50000, scoring4/compile8. At this update it completed t=0.2 with38coordinates and started t=0.24. The budget bound is active at some decisions. No t20 completion is yet available.

The separate W10000 t=.16 diagnostic completed successfully in752.85seconds. All74 saved checks passed: first four committed patches, compiled word hashes and marginal costs, gains/scores, and early theta/integration agree with the W50000 reference. Later trajectory choices can differ when the budget binds. Evidence: `output/paper_ii_execution_window_w10000_t016_20260909_8a76ccac/comparison_01.json`.

The user explicitly approved W10000 for the first CHTC tranche. The clean packaging branch `codex/paper-ii-t20-package-20260909` now binds fresh locks at source `339a1d05`; runner files remain byte-identical to reviewed `2f5d8271`. Both the worker-smoke and six-actiont20 packages pass local manifest/source/all-action/lifecycle/shell checks. Their receipts and next commands are in that checkout's `output/paper_ii_w10000_package_build_20260909/READY.md`. W50000 packages are preserved. Both W10000 packages are uploaded and verified. Single worker smoke `10024094.0` is submitted. The author subsequently directed concurrent execution: all six science jobs `10024096.0–5` are submitted while the performance audit proceeds.

Ordinary automation subagent work stopped; the author subsequently permitted
subagents for performance improvement only. The existing15-minute monitor was updated for direct execution by this task, the approved W10000 settings, single-worker smoke, six-action submission, receipt-addressed collection and exact-reference scoring. Its authority is the linked recovery continuation state.

## Staged CHTC comparison and longer horizons (2026-09-08)

The user selected a two-stage campaign: run the complete CHTC matrix first,
then rerun the most promising variants over longer time horizons. Complete the
remaining local exchange validation and worker-package smoke before the matrix.

1. **Initial CHTC matrix:** execute all 72 cells below at a common initial
   horizon and time grid within each drive. Retain all searches, gain modes,
   singleton/batched settings and low/high budgets, including singleton parity
   pairs. Choose a horizon that exercises adaptation; its value remains pending.
2. **Shortlist:** compare validated trajectory accuracy, compiled resources,
   total estimator acquisition (selection, validation and evolution), classical
   selection/compilation work, and reliability across all three drives. Define
   the comparison criteria before inspecting matrix results. Keep close
   contenders and distinct accuracy/resource tradeoffs; a favorable result on
   one drive alone does not establish the strongest overall variant. The
   checkpoint patch score is not an end-to-end trajectory performance metric.
3. **Longer horizons:** rerun the selected configurations from the same initial
   seeds across the same three drives at shared longer horizons. Preserve the
   timestep/checkpoint spacing and adaptive integration policy, increasing the
   number of points with the horizon. Preserve each selected configuration's
   per-decision budgets, thresholds, compiler and safeguards; total work can
   grow as more decisions occur. Record any additional change as a separate
   experimental change rather than attributing it to horizon alone.

The shortlist and longer-horizon run count depend on the initial matrix;
these extra trajectories are outside the 72-cell count. Short-horizon ordering
is provisional: the longer runs test whether accuracy, resource savings and
stability persist. Preserve the full first-stage results and record the
shortlist rationale before launching its follow-up. This staging does not
change scientific defaults or adopt results into Paper II.

## Overnight t=20 priority (2026-09-08 evening)

The user explicitly requested completed t=20 APM results overnight. After the
remaining local and exact-worker gates, prioritize a small matched t=20
tranche on the new route: three searches x two gain modes at one shared
batched setting and modest work budget on weak-fast. Extend the same tranche
to the other two drives only after its execution envelope is sound. That overnight phase originally covered at most18science actions. The current
campaign-scope clarification above governs the continued 72-cell objective. The earlier t2 prototype remains an
execution-disabled example, not the adopted overnight horizon.

Use the chosen Paper II state-motion settings below. If measured selection or
compilation work causes abnormal runtime, the user authorizes a lower common
work budget for fresh, clearly labeled matched attempts. Preserve the
original attempts and distinguish search cost from queue delay and numerical
integration. Physical parameters, thresholds, controller tolerances, compiler
settings, conditioning and certification remain fixed.

Continuation is recorded in
`output/paper_ii_hh_ascending_local_20260908_07bc223e/continuation_state.json`.
The active heartbeat is `overnight-paper-ii-t20-apm-validation`. Source-frozen
local diagnostics completed at `07bc223e`. Worker preparation continues without
modifying the frozen scientific source.

## Overnight local gate and worker transport (2026-09-09 UTC)

All 30 frozen HH diagnostics completed at source `07bc223e`: 24 ascending
cases and six interleaved controls. The final independent raw audit
`audit_all30_03.json` passed 625,323 checks with no failures; the report
`report_all30_02/comparison.json` passed all six singleton joint/factorized
parity comparisons. Both are under the continuation root above. The earlier
failed audit is retained: its 23 batch-identity flags were false positives in
the scratch checker, corrected with five focused regression tests and 25
existing contract checks. Scientific source and raw trajectories did not change.

Ascending stayed in tier one at both 256 and 1024. Interleaved exhaustive and
block controls evaluated and accepted insertion batches; exhaustive controls
also accepted complete delete/add exchanges. Report this coverage explicitly:
permission to batch does not establish that a particular budget reached batching.

The calibration receipt `calibration_all30_01.json` reports INCOMPLETE with zero
ERRORs. Benchmark-point sameness and response-normalized matched accuracy are
not checked by that receipt; the independent diagnostic audit separately checks
shared configuration and source/seed binding. These are software validation
results, without an equal-accuracy resource claim.

The first exact-container transport smoke, CHTC `9887726.0`, failed before any
trajectory because the descriptor was hash-checked but not transferred. Its
archive was retrieved and verified. Corrected transport smoke `9887730.0`
completed successfully in 252 scheduler wall seconds. The full streaming reporter
then passed its exact-SIF test against this returned13-point trajectory.

The smoke exposed a strict-zero score sensitivity across platforms: D22 scored
zero on the Mac and4.44e-16 on Linux at checkpoint1, changing its deletion time
and subsequent integration subdivisions. Final parameter/energy differences were
5.59e-5/3.31e-6, so1e-10 trajectory parity fails. Source, seed, settings, compiled
marginals and population normalization match. The evidence is preserved under
`worker_smoke_9887730_parity/`; no scoring policy was changed. This limits numerical
reproducibility claims, while the exploratory comparison uses one frozen worker
container.

The six initial weak-fast t=20 actions are now **submitted as9887737.0–5**:
greedy, block exchange and bounded exhaustive, each joint/factorized, ascending,
A4/DNone intersected with the15% deletion bound, shared W256/compile256. They use
501points(dt0.04), Euler/ridge1e-6, PaperII state-motion5e-5, and the registered
12-round/12-finalist/2-per-deletion-branch search profile. The12-round setting
currently permits up to13 selector calls per checkpoint. Each envelope is5CPU,
12GBRAM,12GBdisk and6hours including reporting. These six count toward the bounded
18-attempt overnight scope; they are exploratory, not a promoted72-cell campaign.

Source-package digest and per-action identities are frozen in
`output/paper_ii_t20_exploratory_20260909_07bc223e/package_fastweak_w256_attempt01/`.
Retrieve archive/receipt pairs from the owned staging root named in that package,
verify them before ingestion, and retain every failure/timeout. The worker writes
raw run/resume JSON only at trajectory completion; ON_EXIT archives do not supply
eviction recovery or preserve an unfinished in-memory trajectory.

## Measured budget reduction and local t20 run (2026-09-09 UTC)

The matched local timing diagnostic used the scientific state-motion settings,
three intervals at dt0.04, and W256 versus W64. Both completed: wall113.36s versus
57.66s; compilation101.52s versus47.72s; integration0.670s versus0.675s. This
supports a classical-work reduction. It does not establish equal accuracy or a
measurement saving: logical selector estimator work increased about4.4%. Both
runs exhausted their budgets on tier one and used27 selector calls. Evidence:
`output/paper_ii_t20_budget_prefix_local_20260909_07bc223e/comparison.json` and its
human-readable companion.

Six fresh remote W64 actions were prepared with all non-budget scientific
settings fixed, but **were not uploaded or submitted**. Automatic approval
review twice rejected the archive transfer to the existing ap2001 campaign
directory as unauthorized export. A specific user approval question is pending;
do not bypass this restriction through another transport or remote reconstruction.
The original six W256 jobs continue.

One independent **local W64 t20/501 greedy-joint trajectory** started at
04:42:26UTC, using an extracted immutable842-file source snapshot and the existing
runtime profiler. Its platform differs from the frozenLinux workers and must be
reported separately. Root:
`output/paper_ii_t20_local_w64_20260909_07bc223e/`. It reserves13GB aggregate,
12GB per case and a21600s deadline; do not start another heavy local run outside
the18GB aggregate ceiling. This brings actual overnight t20 attempts to **seven**
(six remote plus one local), not twelve. If the pending six remote attempts are
later approved, refresh their authorization ledger to include the local attempt;
the total would become thirteen, within the eighteen-attempt ceiling.

For returned remote artifacts, use the tested ingestion helper at
`output/paper_ii_t20_exploratory_20260909_07bc223e/preparation_ingestion/launcher_failure_v2/`.
Its19 tests include the actual minimal launcher-failure format. It verifies
archive/inventory/manifest and scientific fields; its all-call score consistency
check is count/minimum/maximum, not individual-commit arithmetic.

## Within-rung generator coverage diagnostic (2026-09-09 UTC)

The seven existing t20 attempts continue; none is complete at this update. See
[the current overnight status](../../output/paper_ii_t20_exploratory_20260909_07bc223e/status.md).
No additional long trajectory was launched after the W16 diagnostic.

A source-bound audit of the saved greedy prefixes confirms that each generator's
placement classes are exhausted before the next generator. The retained last-call
populations reach4/3/3 words at W64 and8/7/6 at W256, out of125. All27 calls in each
prefix acquire full-pool positioned geometry first. This is a finite-budget
traversal limitation, distinct from a pool cap or pure-insertion-score frontier.
The full candidate identities survive only for the last call per checkpoint;
do not extrapolate these exact word counts to every call or the live long run.

The fresh W16/compile16 prefix completed t0.12/4points in25.14s, with19.26s
compilation and0.60s integration. Its retained populations cover2/2/1 words;
the last adaptation L2 remains149times the threshold. It overlapped the local
W64 run, so no controlled speed-ratio claim is made. All842 source files remained
unchanged, and the whole process tree peaked at1.04GB under its4GB guard.
Do not turn this into an automatic full W16 trajectory or continue lowering
budgets without considering coverage.

The smallest proposed policy change is discussion-only: interleave singleton
placement classes across words within each existing branch/tier, preserving
complete patches and the compiled ranker. Budgets below125 still cannot ensure
all words are reached; larger insertion tuples need an explicit ordering decision.
No enumeration, mathematical permission, threshold, or production source changed.

Evidence: [coverage and current source references](../../output/paper_ii_t20_budget_prefix_local_20260909_07bc223e/generator_coverage_audit_01/findings.md)
and [the W16 diagnostic](../../output/paper_ii_t20_budget16_prefix_local_20260909_07bc223e/report.md).

## Compiler-pool reuse and accepted tiny gains (2026-09-09 UTC)

At the 06:01 UTC status observation, the six existing CHTC W256 jobs were
running (JobStatus=2), with no holds. The separate local W64 trajectory had
completed t1.32 and started t1.36 with372 parameters. **No full t20 trajectory
was complete.** Session20597 finished its compiler reuse diagnostic and released
its4GB reservation; the local t20 run still reserves13GB. The prepared remote
W64 transfer remains blocked pending explicit approval; no transfer or new
submission was made.

The two sequential W64 prefixes compared ordinary compiler-pool lifetime with
reuse of the worker processes. Total time including final cleanup was
**65.494s ordinary versus40.913s reuse** (runner time65.255s versus40.503s).
Real pool creations fell from27 to1. All3242 completed compiled word/resource
results match byte-for-byte, as do scientific values,176 retained population
rows across three populations, candidate membership/order/K/score,26 commits,
logical work and trajectory parameters. All842 frozen source files are unchanged.

This is qualified scientific parity, not raw-JSON identity: three compile-duration
values differ, and six backend `operation_names` audit lists have different
order. Each list has the same eleven names and multiplicities; only these
unordered capability lists were canonicalized. No candidate ordering was
normalized. The original strict `comparison.json` remains preserved beside
[the qualified comparison](../../output/paper_ii_compiler_pool_reuse_diagnostic_20260909_07bc223e/comparison_02.md)
and its JSON receipt. Eight bounded coordinator checks and source review passed.
Both arms ran while the longer W64 trajectory was active, so these preliminary
timings do not establish a controlled repeatable speed ratio. That diagnostic
introduced no production implementation, cost-cache reuse, placement warming
or new full trajectory. Parallel placement compilation remains a separate proposal with
unresolved failure/timeout-accounting details.

The [read-only tiny-gain audit](../../output/paper_ii_t20_budget_prefix_local_20260909_07bc223e/tiny_gain_audit_01/findings.md)
examines every recorded commit in the completed t0..0.12 prefixes:

| Budget | Commits | Realized normalized delta-q range | delta-q <=1e-10 | delta-q <=1e-12 |
|---|---:|---|---:|---:|
| W16 | 15 | 2.55e-14 to2.38e-8 | 14 | 12 |
| W64 | 26 | 1.50e-13 to6.45e-10 | 13 | 6 |
| W256 | 26 | 5.93e-13 to4.34e-6 | 13 | 2 |

Every commit has positive delta-q and score. Authoritative route merit equals
delta-q exactly; the large-miss score S=delta-q/K agrees numerically, with tau=0.
W16 has fifteen pure insertions; W64 and W256 each have24 pure insertions and
two exchanges. Among W16's twelve sub-1e-12 gains, four increase a compiled
resource and eight have zero resource marginal; none saves a resource. Their
normalized increments sum descriptively to2.10e-12 and positive resource
marginals to(8,5,18). These sums are neither fixed-state telescoping gains nor
final resource changes. W64's corresponding six commits comprise four
resource-increasing and two zero-marginal edits; both W256 commits increase
resources. Their summed marginals are(44,34,57) and(19,15,20), respectively.

The realized gain uses a ridge1e-6 safeguarded solve with cutoff1e-10.
`delta_gamma_normalized` instead subtracts two full-geometry ideal captures
with cutoff1e-12; it is not an independent direct-Schur or high-precision
reference. Gamma differences are negative on6/15,9/26 and9/26 commits for
W16/W64/W256. The pure-insertion monotonicity tolerance is1e-9; exchanges do
not receive that pure-insertion test. These different ordinates should not
be equated.

The smallest W16 realized gain spans230 representable spacings at q~0.933,
and its median spans1660. Calling every gain a one-ULP subtraction accident
would overstate the evidence; these counts also do not bound propagated
geometry/inverse error. Existing records cannot distinguish genuine tiny
regularized improvements from solve/cancellation error. Strict positive
ranking and strict lower post-certification L2 currently have no minimum
improvement margin. No tolerance was proposed or adopted by this audit.
The last adaptation L2 remains149.2607,149.2194 and148.8915 times the common
3e-6 cut. Numerical attribution and the cause of long-run support growth
remain unresolved; no high-precision or direct-Schur recomputation was done.

### Actual duplicate insertion arithmetic (2026-09-09 UTC)

A later [reconstruction and 80-digit calculation](../../output/paper_ii_t20_budget16_prefix_local_20260909_07bc223e/ridge_duplicate_diagnostic_01/findings.md),
with [independent review](../../output/paper_ii_t20_budget16_prefix_local_20260909_07bc223e/ridge_duplicate_diagnostic_01/independent_review.json),
resolves the arithmetic attribution for **one** earlier W16 pure insertion.
At checkpoint 1, t=0.04, selector call 3, the accepted coefficient -0.5
`eeeexy` insertion at cut 28 duplicates the existing coordinate 28.
The identical rotations are adjacent; their projected tangents coincide
algebraically and are bit-identical in the reconstruction. The **exact tangent
span gain is zero**.

Nevertheless, the actual double Gram/force lifted exactly into Decimal80 gives
positive normalized realized gain **2.1433523850240010e-13**. Consistent Decimal80
dot products from the stored double vectors give **2.1433523849943650e-13**;
an independent mapped-velocity gain identity agrees at high precision.
Splitting one physical coefficient across two identical coordinates reduces
its coordinate-based ridge penalty, permitting a positive raw-Q improvement
without adding span. Floating-point subtraction alone does not explain this case.

The reconstructed actual double solve differs from the archived gain by
**one q ULP (1.1102230246251565e-16)**; original unsaved Gram bytes were not
recovered. The exact-duplicate lift reproduces the archived gain exactly.
All source ridge modes survive: **31/31 before and 32/32 after**, with minimum
ridge eigenvalue approximately 1e-6 above the respective 2.74e-10 and 2.99e-10
cutoffs. Thus the Decimal ridge solve respects the full source mode selection
for this case. Stored state/tangent vectors remain double precision; this is
not arbitrary-precision state preparation or evidence of general lambda-squared
scaling.

This result supersedes the earlier unresolved arithmetic attribution **only
for this insertion**. It does not explain all tiny gains or long-run support
growth, and redundancy against current support cannot justify excluding an
exchange conditioned on retained support. No policy, threshold, regularization
or production source was changed. As of 07:23 UTC, all seven existing t20
attempts remained unchanged and none had completed; the remote W64 upload and
operator-interleaving questions still awaited replies.

## Production compiler-session validation (2026-09-09 UTC)

The production implementation now passes the identical W64 t0..0.12 prefix
against the preserved ordinary arm. This validation uses production lifecycle
code directly; its observer only journals `CompileWarmPool.__init__` and
`compile_jobs`, without supplying a lifecycle monkeypatch or facade.

All **3,242 ordered compiled-word/resource journal rows**, **176 retained
population rows**, **26 all-call commits**, logical-work ledgers and raw
scientific values match exactly. One real pool is created. All **842 source
hashes** pass, including the three approved source edits; the new snapshot
identity is separately verified. Comparison exceptions remain limited to actual
compile-duration fields and narrowly proven unordered backend `operation_names`
lists with identical membership and multiplicity. Scientific ordering, scores,
compiled costs and safeguards are not excluded or normalized.

Total production time was **40.581 seconds**, versus **65.494 seconds** for the
saved ordinary arm. This remains preliminary timing under possible changing
CPU contention, not a controlled speed ratio. Session 92540 exited successfully;
its whole process tree peaked at **1,066 MB** under the 4 GB guard, and that
reservation is released. The 14 comparison-helper checks pass. The source test
logs preserve the initial failed check, the subsequent 103 passing checks, and
five passing targeted cleanup checks including the final regression.

Evidence: [production comparison](../../output/paper_ii_compiler_session_production_validation_20260909_07bc223e/comparison.md),
[full comparison receipt](../../output/paper_ii_compiler_session_production_validation_20260909_07bc223e/comparison.json),
and [copied test-log and ownership manifest](../../output/paper_ii_compiler_session_production_validation_20260909_07bc223e/evidence/evidence_manifest.json).
The original logs and earlier diagnostic evidence remain preserved. The
receipt records the exact source snapshot and patch for the validated compiler
session. It introduces no cost-cache sharing, placement warming or new
scientific control.

The seven existing t20 attempts remain unchanged. The latest local observation
at 06:21 UTC completed t1.44 with 411 parameters; **no full t20 trajectory is
complete**. The local run still reserves 13 GB. The asynchronous question about
rotating operator placement classes within each ascending tier is pending:
**do not implement that selection-policy change without the reply**. Remote W64
egress approval also remains pending; no transfer or new submission was made.

GitNexus index session 19258 was deliberately interrupted with exit 130 because
the initial command omitted the existing PDG mode. Its 4 GB reservation is
released. Subsequent indexing must retain PDG; the interrupted indexing
attempt changed no scientific source. Its receipt is retained in the
continuation record.

## Ascending trial and state-motion campaign settings (2026-09-08)

The user approved an **optional ascending scheduler trial**. Define the tier
of a complete patch by

\[
t(D,A)=\max\{1,|D|,|A|\}.
\]

Tier one contains stay, pure singleton insertions/deletions, and D1/A1
exchanges. Finish all currently runnable tier-one work before tier two,
which contains patches with up to two deletions and two additions. Continue
in ascending tiers subject to the existing domain and work limits. Within
the active tier, retain interleaving between patch types and deletion
branches. Newly exposed lower-tier work after a greedy/block move or a
settled-branch wake regains priority. Logical requests remain complete before
the search takes its next move; tier order is a priority, not a claim that
no lower-tier candidate can ever be exposed later.

Declare `--patch-rung-schedule ascending` for this trial; preserve explicit
`interleaved` controls and the existing default. Ascending affects discovery
and evaluation ordering, so it can change candidate coverage, population
normalization snapshots, and trajectories. It does not change the score,
conditioning, joint/factorized definitions, permission, compilation, or
certification. Every completed evaluation remains available when a budget
ends partway through a tier. Report tiers never reached; batched permission
does not imply batched evaluation at a small budget.

Next local validation: 24 ascending HH trajectories (three searches, two
gain modes, A1/D1 versus A4 with the 15% deletion bound, budgets 256/1024),
plus six interleaved batch controls at budget 256. Reuse the previous HH
probe's seed, physical/compiler/safeguard settings, weak-fast drive and
t=0..0.5, 13 points. Use a fresh source/plan lock. These are **30 diagnostic
runs**, separate from the 72 scientific cells. Run sequentially with a
12,000 MB per-case guard rather than inheriting the old 3,000 MB limit.

For the later scientific matrix, the user selected Paper II's **state-motion
controller**. Use the existing reported controller settings and insertion
activation thresholds as starting values, fixed across all selector variants
within each drive:

| Drive | (amplitude, frequency) | Insertion L2 threshold | Controller |
|---|---|---:|---|
| weak-fast / `fastweak` | (0.6, 3) | 3e-6 | `state_motion_5e-5` |
| strong-slow / `slowstrong` | (1.2, 1) | 5.080526342527e-12 | `state_motion_5e-5` |
| strong-fast / `strongfast` | (2.4, 3) | 3e-6 | `state_motion_5e-5_sub15` |

Numerics: Euler, ridge 1e-6, pseudoinverse cutoff 1e-10, zero damping;
state-motion cap 5e-5, parameter kink guard 5e-3, subdivision limit 11
except strong-fast's 15. Retain pre-permission ray bound 2e-3, normalized
reverse-Schur loss threshold 1e-3, post-refit ray bound 2e-3, velocity
continuity threshold 1e-3 and maintenance allowance 1e-3. Retain the HH seed
`chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json` (SHA256
`7a36a2da06bba447c6d65d95692c8a45bc3aa64933d02f1f641e2178f4ea8f91`),
complete 125-word pool, FakeMarrakesh opt1/seed7/structural-angle1 and compiled
resource weights (0.30, 0.30, 0.25). Do not infer faster execution from the
controller choice; measure it, especially at the tight strong-slow cut.

Threshold provenance: `pipelines/time_dynamics/campaigns/`
`paper_ii_corrected_exchange_cells_v1.json`, SHA256
`4705aaf2c36e9e0c8176f7b6c6688a04420898eb22933a71c44b350ff5c892b1`,
and `paper_ii_corrected_exchange_strongfast_state_motion_sub15_cells_v1.json`,
SHA256 `fbafd400623d776813c49b28f469416a3757960e2d889fc6fc4312248136cad9`.
These agree with the active manuscript's reported settings. They are distinct
from the local RK4 diagnostic settings and its 1e-3 insertion cut.

Reuse existing execution, receipt, calibration and reporting components with
a variant-aware manifest. The older staged campaign's `CellKey` omits the
search/gain/batching/budget axes and its automatic per-cell threshold tuning
would violate this comparison. Do not inherit that identity, tuning or the
legacy frontier route. Scientific horizon, two workload-calibrated budgets,
final scheduler choice, exact worker envelope and source-bound submission
manifest still require completion of the local and worker validation gates.

## Ansatz-relative deletion discovery (2026-09-08)

The next local validation and campaign use the user-approved maximum deletion
batch fraction **0.15** in all three experimental searches. At each new frozen
search, n is the current runtime coordinate count, including protected
coordinates; m counts structurally deletable coordinates before whole-set
ray-displacement and reverse-Schur permission. Resolve

\[
k_{\max}=\min\{m,\lfloor0.15n\rfloor,n-n_{\min},
k_{\mathrm{explicit}}\text{ when set}\},\qquad
W_D=\min\left\{W_{\max},\sum_{k=1}^{k_{\max}}\binom{m}{k}\right\}.
\]

Existing support feasibility still applies. The singleton arm retains its
explicit D1/A1 limits; the batch arm intersects its deletion domain with the
fraction bound. Rounding is down, with no minimum-one adjustment: n<7 permits
no nonempty deletion batch. Wmax is the existing configured
`max_joint_patch_evaluations` value, used as an **independent discovery
allowance**; patch evaluation has its own counter with that same ceiling.
With Wmax=None, discovery is bounded only by the finite declared population.

Count each distinct nonempty D at first inspection, including permission
failures. Cached reuse on the same frozen search is free; D=() has no discovery
charge. Permission failures consume no patch-evaluation or compilation slots.
Budget and permission caches reset for each new frozen search. Every complete
D still receives its own permission: singleton Schur failures do not filter
the batch universe.

The original deletion-discovery revision kept the interleaved schedule; the
optional ascending trial above changes ordering explicitly. Discovery exhaustion stops
only new deletion batches, including partway through any rung. Existing pure
insertion and approved deletion/exchange branches retain their queues and
completed evidence, subject to the remaining evaluation and compilation
allowances. Final population ranking includes all completed evaluated patches.
Receipts expose the fraction, resolved sizes/population, discovery allowance,
inspection count, and truncation. Truncated discovery cannot claim exhaustive
completion. The binary insertion gate and all permission/certification
mathematics remain unchanged.

This changes candidate coverage and may change trajectories; the prior local
results above retain their original source locks. Revalidate the local HH
exchange matrix before CHTC. The deferred `family_frontiers` route is unchanged.

## Incremental admission and rung scheduling (2026-09-08)

The user authorized this revision after reviewing the pilot's low-budget block
refusal. It changes finite-budget candidate coverage and can change trajectories;
the recorded 24-cell pilot remains evidence for its original implementation.

The three experimental searches share `incremental_complete_patch_v1` admission
and `cardinality_rung_round_robin_v1` scheduling, recorded in every search receipt.
The full candidate pool, cardinality bounds, whole-D ray-displacement and
reverse-Schur permission, gain modes, compiled-Q population score, placement
pricing and finalist certification retain their existing definitions.

Interface: `PatchRequest(label, patches_by_insertion_size)` describes one
logical extension set or neighborhood. `schedule_rungs` advances its positioned
patch iterators without exhausting one cardinality before reaching another.
Each complete evaluated realization is cached immediately. Budget exhaustion
ends the search with its completed prefix available to final ranking; it does
not establish neighborhood exhaustion or complete placement coverage.

Deletion-cardinality discovery is progressive and lazy. Each active cardinality
gets at most one new whole-D permission visit per sweep, including refused
visits. Each live (|D|,|A|) rung gets one complete patch turn and rotates among
its deletion branches. Unused turns pass to other work without reserving slots.
Pure insertion/deletion faces and unbalanced exchanges remain reachable within
the configured search domain and active scoring regime. No fixed thirds, pool
cap, beam width or parent-score admission filter is introduced.

Greedy completes its extension set before choosing a move; block completes its
neighborhood, then revisits settled branches when cumulative evidence grows.
Exhaustive retains every permitted block and realization in its search domain.
All move comparisons use one current cumulative-population ranking snapshot.
Interleaving may change those snapshots even when the work cap is nonbinding.
Factorized gain ranking remains distinct from later joint validation.

Implemented at `a2a43a83fbccbc34ec56d43d9e12f62ad9953f77`. **310 focused tests
pass; 12/12 matched low-budget full trajectories complete; 3/3 singleton
joint/factorized parity pairs pass.** Read the
[revision findings](../../output/paper_ii_rung_schedule_local_20260908_a2a43a83/findings.md)
and [frozen plan](../../output/paper_ii_rung_schedule_local_20260908_a2a43a83/plan.json).
The [independent audit](../../output/paper_ii_rung_schedule_local_20260908_a2a43a83/audit.json)
passes 7,466 checks across all twelve runs with zero failures, including all
802 source hashes and the selection/validation work accounting.

At the active insertion checkpoint, block evaluates 128 A1 and 127 A2
realizations, instead of refusing the entire neighborhood. Exhaustive reaches
A1/A2/A3/A4 with counts 64/64/64/63; greedy evaluates 214 A1 and 41 A2.
All fit after the stay patch under budget256. The selected batch sizes differ
between gain modes; the factorized A4 finalist's six validation pair entries
are charged. Every trajectory ledger remains exact. No D>0 passes permission
in this physical window; synthetic exchange coverage and the small evolving
trajectory tests remain distinct evidence. High-budget reruns are recorded below;
the three-drive scientific campaign remains pending. No CHTC submission or evidence promotion
is part of this implementation.

## Completed local continuation and natural fixed-VQE seed

All runs in this continuation used revision `970eb4800f6a4556179942f5d2127519d1dfed0d`
and the same 802 Python-source hashes as the low-budget revision. Production
code and scientific safeguards were unchanged. TETRIS-style candidate ordering
remains deferred.

The [high-budget continuation](../../output/paper_ii_rung_high_budget_local_20260908_970eb480/findings.md)
completed all twelve original-pilot cells at budget 24,576. The raw-receipt audit
passed 7,680 checks and all three singleton joint/factorized pairs passed.
Every batch arm retained 24,576 evaluated patches at budget termination, including
the stay patch. Committed insertion sizes, joint/factorized, were greedy 3/3,
exhaustive 2/4 and block 3/4. Factorized selection acquired no candidate-pair
cross terms; its finalist validation charged 3/6/6 pair entries respectively.
These runs still permitted no nonempty deletion set.

The user requested a naturally overcomplete Hubbard seed from fixed UCCSD with
hopping. The [prepared seed](../../output/paper_ii_uccsd_hopping_seed_local_20260908_970eb480/findings.md)
uses two complete layers of alpha singles, beta singles, doubles and Hubbard
hopping. All eight logical angles were optimized; the lowest-energy result of
four deterministic SLSQP restarts was selected, independently of deletability.
Its static energy error is 9.52e-8. It has 32 prepared Pauli coordinates, versus
16 in the original seed; the existing per-Pauli dynamics mode and four protected
drive coordinates give 36 initial runtime coordinates. Both coordinate-mode
reconstructions reproduce the prepared state to 1.04e-15. The Hamiltonian,
Hartree-Fock reference and complete ordered 22-operator dynamic pool match the
original Hubbard pilot. No dummy gates or post-optimization padding were used.

Eighteen matched A1/D1 trajectories tested the three searches and two gain modes
under the registered fastweak, strongfast and slowstrong drives, t=0..2 with 49
points, budget 256. All completed and all nine singleton pairs passed. Each
evaluated only 48 stay patches; no insertion gate opened and no D passed
permission. Independent audits passed 11,461 checks for the
[fastweak group](../../output/paper_ii_uccsd_hopping_trajectories_local_20260908_970eb480/independent_audit_final.json)
and 22,934 for the
[two additional drive groups](../../output/paper_ii_uccsd_drive_comparison_local_20260908_970eb480/independent_audit_final.json).

The strongfast trajectory showed an actual zero crossing between checkpoints:
one effective angle moved from -0.00332943 at t=1.58333 to +0.00200064 at t=1.625,
missing the unchanged 0.002 permission window on the coarse grid. A separately
frozen [97-point strongfast comparison](../../output/paper_ii_uccsd_strongfast_fine_local_20260908_970eb480/findings.md)
halved the common checkpoint spacing to 1/48. All six trajectories completed and
all three singleton pairs passed. Its independent audit passed 23,042 checks,
including scalar recomputation of 12 compiled cost populations and 36 candidate
records, with certification and commit bindings. Every variant evaluated six pure deletions
and certified D={23} at t=1.6041667, reducing support 36→35. The compiled patch
marginal was N2q=-8, D2q=-8, Dc=-34. The first variant's mean infidelity was
6.69e-10 and mean absolute energy error 4.74e-5. Refinement changes numerical and
controller sampling; these are within-grid comparisons, not a timing ablation.

This establishes a natural physical pruning fixture under the original
ray-displacement, reverse-Schur and certification safeguards. The insertion
gate remained closed, so these UCCSD runs do not establish exchange-search
performance. The separate existing-Hubbard-Holstein
[diagnostic probe](../../output/paper_ii_hh_exchange_probe_local_20260908_970eb480/summary.json)
evaluated 700 D1/A1 realizations and certified one pure deletion. Extending the
original Hubbard seed to t=2 still produced no permitted deletion; that
[negative probe](../../output/paper_ii_exchange_window_local_20260908_970eb480/summary.json)
is preserved as well.

The fine audit verifies the authoritative compiled `route_score` (0.00025 for
the committed deletion). The generic `score` still records older utility;
that field-name cleanup remains outstanding and must not be mistaken for the
quantity that selected the patch.

All reports retain exact-reference isolation and exact logical-work receipts.
Paper calibration reports INCOMPLETE with no ERROR because these diagnostic
artifacts lack the paper benchmark-row fields; direct audits bind actual seed,
settings and source evidence. No paper qualification or promotion is implied.
High-budget raw candidate arrays remain intact; bounded-memory reporting
projections name the two omitted duplicate audit arrays and bind both files by
hash. The fine-grid plan's inherited drive-lineage metadata is clarified by a
[hash-bound sidecar](../../output/paper_ii_uccsd_strongfast_fine_local_20260908_970eb480/lineage_clarification.json);
its verified effective-argument change is only `num_times`: 49→97.

## Corrected implementation and campaign entry point

- `family_robust_v1` now uses its exact singleton identity (cost factor one)
  before raw estimation. Real population-proxy and raw-legacy consumers retain
  their numerical behavior. This is decision-preserving cleanup.
- Complete-D permission precedes patch-budget charging and branch-pool
  construction. Rejected sets consume no evaluation slots; existing receipts
  count their permission work separately. This earlier correction retained
  whole-request admission; the revision above supersedes it for the three
  experimental searches.
  This changes finite-budget coverage, including on the retained family path.
- Build new comparison commands with
  `paper_ii_runs.build_selector_comparison_run(search_policy=...,
  gain_evaluation=..., insertion_cardinality=..., deletion_cardinality=...,
  **run_settings)`.
  The helper pins compiled-Q and validates effective runner arguments, including
  late overrides. Search, gain, and both cardinalities are required and included
  in run identity.
  Named profile budgets remain explicit; the generic builder is unchanged.
- The legacy frontier's ordering is deferred, as requested. Existing v1
  revisions and evidence remain unchanged; current accounting corrections
  should not be represented as a reproduction of v1 execution.

Local numerical validation now includes six tight-budget singleton trajectories
(three searches times two gain modes) and six batched trajectory smokes. The
tight-budget test first reproduced four evaluated patches instead of six;
after correction both permitted exchanges are reached and joint/factorized
singleton trajectories agree through the full evolution. These are small
software checks, not the 72 planned scientific trajectories.

Permission scans can remain combinatorial: the patch-evaluation budget does
not cap rejected-set inspections. Record that classical work separately when
choosing campaign runtime resources; no new arbitrary pool or permission cap
was introduced.

Validation of this correction: **332 tests passed** across four disjoint
groups, each under a 3 GB RAM guard. Independent source review found no
additional issue.

| Test group | Passed | Wall time |
|---|---:|---:|
| Permission, search, family admission and selector | 154 | 0.80 s |
| Proxy elimination and legacy numerical parity | 10 | 9.71 s |
| Tight-budget parity plus batched evolving trajectories | 9 (12 trajectories) | 32.13 s |
| Adapter, gain modes, frontier preservation, configuration and run commands | 159 | 23.32 s |

The two targeted regressions were observed failing before their respective
fixes: the redundant estimator call and four-versus-six evaluated patches.
The largest sampled resident footprint in the passing groups was 906 MB;
short test groups can finish before the guard obtains a meaningful RSS sample.

## Campaign matrix

| Axis | Values | Count |
|---|---|---:|
| Search policy | `greedy`, `bounded_exhaustive`, `block_exchange` | 3 |
| Gain evaluation | `joint_batch`, `singleton_factorized` | 2 |
| Patch admission | Singleton insertion/deletion; batching allowed | 2 |
| Work budget | 10,000; 50,000 | 2 |
| Drive regime | fastweak, slowstrong, strongfast | 3 |

\[
N_{\mathrm{runs}}=3\times2\times2\times2\times3=72.
\]

This is **8 runs per search per drive**, **24 per search across three drives**,
and **24 per drive across three searches**. Each cell starts from the same
declared initial state; trajectories may diverge after different selections.
Extra controller, seed, or repeat axes are outside this count.

Working admission settings are `max_insertion_batch_size=1` and `4`.
Four allows the block and exhaustive searches to differ: with maximum two
and nonbinding budgets, the radius-two block search already evaluates every
allowed insertion block from its empty starting block. The block-search move
radius remains two in both admission settings.

The user clarified singleton-only as \(|A|\leq1\) and \(|D|\leq1\) **per patch**,
while keeping complete candidate pools. The explicit optional deletion cap is
`max_deletion_batch_size`; `None` preserves unrestricted deletion cardinality
subject to complete-set permissions. The batching arm uses A4 with the existing 15% deletion bound.
This is an enumeration policy change; the ray-displacement bound and
reverse-Schur loss permission remain separate, unchanged tests on each D.

## Local diagnostic pilot

Completed at `7284dbd6a9d8811895611027e280cd2e8fc6ebd0`; all 24 runs have exact
estimator accounting, no execution failures, and matching singleton parity.
Read the [pilot findings](../../output/paper_ii_selector_local_pilot_20260908_fbebbabc/pilot_24/findings.md)
and [frozen plan](../../output/paper_ii_selector_local_pilot_20260908_fbebbabc/pilot_24/plan.json).
These runs are distinct from the 72 planned scientific trajectories.

The primary observed restriction is low-budget block search's whole-neighborhood
admission: it commits no insertion, while greedy/exhaustive admit singleton
insertions. At the high budget all three admit A2. Factorized ranking reduces
acquisition work but barely changes compilation-dominated wall time. No D>0
passes permission, and no A3/A4 request is evaluated; those cases still require
a suitable diagnostic before expanding the campaign.

Before extending the campaign, execute one budget-calibration trajectory and
24 matched local trajectories (three searches, two gains, A1/D1 versus A4/D-all,
two budgets). The user authorized local execution on 2026-09-08. Record the
chosen budgets from calibration before freezing the 24-cell plan.

Use the existing complete-pool Hubbard SNAKE seed, existing `fastweak` drive,
RK4 ridge-1e-7 numerics, and state-motion safeguards. The initial t=0..0.125
calibration never opened the insertion gate. The full registered smoke grid,
t=0..0.5 with 13 reporting points, did: 214 singleton placements at t=0.4583
and one accepted insertion under budget256. Use that full smoke grid for the
24-cell pilot. `rounds_per_checkpoint=1` is common to every cell. The current
runtime loop permits up to two selector calls under this setting; preserve that
behavior for this pilot and report all calls.
The seed resolves to 22 deduplicated operators without fallback or truncation;
the immutable plan binds its hash and all effective arguments.

Pilot budgets are 256 and 24,576 for both structural evaluations and compiled
patches. The full-smoke calibration evaluated 214 singleton realizations.
Classical enumeration of its next A2 request found exactly 23,503 complete
paired-word realizations; both exhaustive and block search require 23,718
cumulative structural slots to admit that stage. These counts concern the
observed checkpoint, not a guarantee about later stages. Validate the high
budget locally before launching the matched matrix.

`diagnostics.selector_local_pilot` builds commands through the shared comparison
helper. Its execution helper gives each case an independent 3 GB RAM guard,
600-second wall limit, four compiler workers, one scoring worker and one BLAS
thread per process. Timeouts and partial
attempts are retained. No CHTC work is submitted by this pilot.

Search receipts count evaluated circuit realizations by (|D|,|A|) and visited
permitted/refused deletion sets by |D|. Unvisited sets are unknown, not refused.
The round ledger retains every selector call's compact coverage, guards and
committed patch; estimator totals include every pass and its validation/evolution
work. Exact-reference propagation and terminal-resource reporting occur only
after execution. These local diagnostics are not promoted scientific results.

## Gain modes and singleton parity

Follow the user's `paper_ii_selector_evaluation_handoff.md` contract:

- Joint evaluation uses the complete patched safeguarded solve.
- Factorized evaluation sums singleton increments against the same retained
  support \(R=C\setminus D\), subtracting the deletion contribution once.
  It is a ranking approximation; it acquires no candidate–candidate terms
  during ranking.
- Both modes price the complete compiled patch. Factorized gain does not
  imply additive compiled cost. Finalist joint validation, certification,
  refitting and subsequent evolution remain authoritative and fully charged.

**Verified shared implementation at the baseline revision:** for zero or one
insertion, `PatchEvaluator.q_of` calls `_joint_q_of` under both gain flags.
The stabilized solve, Gamma calculation, conditioning/history operations,
compiler, population ranker and certification are shared. The factorized
formula is entered only for more than one insertion. Parallel factorized
warming has a separate branch, but yields the same singleton keys and order.

The flags still exercise configuration validation, parallel preparation and
provenance. Keep all **36 singleton slots** in the tracker as **18 paired
parity checks**. They are not independent evidence for different singleton
gain algorithms. There is no automatic reduction to 54 runs.

For each parity pair, compare candidate coverage, selected patches, commit and
rejection decisions, state/observable trajectories, compiled resources, and
estimator acquisition. Ignore intentional gain-mode labels and run IDs when
comparing numerical outputs; compare runtime as a measurement, not an equality
requirement. Record the shared implementation and the parity outcome alongside
any later decision to omit duplicate execution.

Sources: [evaluator](../../pipelines/time_dynamics/ap_mclachlan/patch_evaluation.py),
[selector](../../pipelines/time_dynamics/ap_mclachlan/exchange_selector.py),
[configuration](../../pipelines/time_dynamics/ap_mclachlan/exchange_config.py),
[evaluator tests](../../test/test_ap_mclachlan_patch_evaluation.py).

## Shared settings and open decisions

Within each drive, vary only the three declared experimental axes: search,
gain mode, and admission setting, crossed with the two budget levels. Bind
all remaining settings explicitly in one machine-readable manifest.

| Setting | Campaign decision |
|---|---|
| Three drives, including amplitudes and frequencies | Pending selection; fill D1–D3 below |
| Hamiltonian, physical cutoff, initial seed and seed hash | Pending binding; shared across all variants of each drive |
| Initial matrix horizon and time grid | Pending; shared across variants within each drive; longer horizons follow the shortlist with spacing preserved |
| Integrator, inverse policy, controller and activation thresholds | Pending binding to one common configuration per drive |
| Candidate pool | Complete eligible pool; no operator-pool truncation or pure-insertion frontier |
| Compiled-resource route | `paper_ii_exchange_q_compiled_v1`; same compiler settings, placement policy and compiler seed |
| Conditioning, history, deletion permission and certification | Same settings for all variants; retain scientific safeguards |
| Low/high budget definitions | Pending exact limits and an explicit list of fields changed between levels |
| Parallel workers, cache policy and runtime measurement | Same declared execution settings; independent run caches |
| Diagnostic references and observables | Shared per drive, with verified data hashes; reporting only |

Define low/high as the same two budget profiles across searches, gain modes
and admission settings. Name each capped quantity: structural patch requests,
compiled patches, or another explicitly chosen work measure. Keep numerical
tolerances separate from computational budgets. Record actual expenditure;
an identical cap does not imply identical runtime or estimator acquisition.

Existing registry profiles provide settings to inspect, not an adopted launch
configuration. Compose commands through
[paper_ii_runs.py](../../pipelines/time_dynamics/paper_ii_runs.py) and audit
their normalized settings. The old 11-cell/66-run proposal does not define this
three-drive campaign.

## Run tracker

Each table cell is one complete trajectory. `planned` means no submission.
Replace it with a status and a link to the run record as work proceeds:
`queued`, `running`, `completed`, `validated`, or `failed`.
The linked record should hold the manifest/run ID, scheduler ID, output path,
and validation outcome. Mark retries against the same planned slot rather
than counting them as new experimental conditions.

| Drive slot | Drive ID | Amplitude | Frequency | Seed/reference bindings |
|---|---|---|---|---|
| D1 | Pending | Pending | Pending | Pending |
| D2 | Pending | Pending | Pending | Pending |
| D3 | Pending | Pending | Pending | Pending |

### D1

| Admission | Gain | Budget | Greedy | Exhaustive | Block |
|---|---|---|---|---|---|
| Singleton | Joint | Low | planned | planned | planned |
| Singleton | Joint | High | planned | planned | planned |
| Singleton | Factorized | Low | planned | planned | planned |
| Singleton | Factorized | High | planned | planned | planned |
| Batched | Joint | Low | planned | planned | planned |
| Batched | Joint | High | planned | planned | planned |
| Batched | Factorized | Low | planned | planned | planned |
| Batched | Factorized | High | planned | planned | planned |

### D2

| Admission | Gain | Budget | Greedy | Exhaustive | Block |
|---|---|---|---|---|---|
| Singleton | Joint | Low | planned | planned | planned |
| Singleton | Joint | High | planned | planned | planned |
| Singleton | Factorized | Low | planned | planned | planned |
| Singleton | Factorized | High | planned | planned | planned |
| Batched | Joint | Low | planned | planned | planned |
| Batched | Joint | High | planned | planned | planned |
| Batched | Factorized | Low | planned | planned | planned |
| Batched | Factorized | High | planned | planned | planned |

### D3

| Admission | Gain | Budget | Greedy | Exhaustive | Block |
|---|---|---|---|---|---|
| Singleton | Joint | Low | planned | planned | planned |
| Singleton | Joint | High | planned | planned | planned |
| Singleton | Factorized | Low | planned | planned | planned |
| Singleton | Factorized | High | planned | planned | planned |
| Batched | Joint | Low | planned | planned | planned |
| Batched | Joint | High | planned | planned | planned |
| Batched | Factorized | Low | planned | planned | planned |
| Batched | Factorized | High | planned | planned | planned |

## Preparation and reporting

- [x] Implement three search policies with both gain modes and shared safeguards.
- [x] Verify small evolving trajectories and staged acquisition accounting.
- [x] Inspect shared singleton code; retain paired end-to-end checks.
- [ ] Resolve the open settings and singleton/deletion interpretation above.
- [ ] Record coverage by deletion/insertion cardinality and the limiting budget.
  Distinguish no permitted deletions from a budget exhausted before exchanges.
  Discuss any change to enumeration order or budget allocation before coding it.
- [ ] Generate 72 uniquely identified manifest rows. Verify that only declared
  experimental axes and administrative output IDs differ within matched cells.
- [ ] Pin implementation/package, seed and reference hashes; complete the
  applicable [run workflow](../skills/paper-ii-run/SKILL.md) checks.
- [ ] Submit to CHTC and attach submission receipts to this tracker.
- [ ] Validate all completed trajectories and singleton parity pairs.
- [ ] Compare trajectory error, compiled resources, support size, successful
  exchanges/batch sizes, classical runtime and cumulative estimator acquisition.
  Separate selection, validation and evolution costs, including failed attempts.
  Identify budget-limited searches; call exhaustive search complete only when
  its entire declared domain was evaluated.

Report the full matrix, including failures and parity mismatches. Choose no
winner from local gain alone. Maintain links to raw run records and aggregate
results here; manuscript adoption is a later decision.

Files to edit: this tracker as decisions and results arrive. Campaign-generation
and coverage-instrumentation paths will be recorded when their implementation
scope is defined.

# Paper II selector campaign

Updated: 2026-09-08. **Original local pilot: 24/24 completed; 6/6 singleton
parity pairs passed. Incremental/rung revision: 12/12 low-budget reruns
completed; 3/3 singleton pairs passed.** The separate three-drive scientific campaign still has
**72 full trajectories planned; 0 submitted**; its settings and submission
manifest remain pending.

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
trajectory tests remain distinct evidence. High-budget reruns and the
three-drive campaign remain pending. No CHTC submission or evidence promotion
is part of this implementation.

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
| Work budget | Low; high — numerical limits to be chosen | 2 |
| Drive regime | D1, D2, D3 — physical drives to be chosen | 3 |

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
subject to complete-set permissions. The batching arm uses A4/D-unrestricted.
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
| Full horizon and reporting grid | Pending; shared across variants |
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

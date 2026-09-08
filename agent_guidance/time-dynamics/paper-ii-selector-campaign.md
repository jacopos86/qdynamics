# Paper II selector campaign

Updated: 2026-09-08. Status: design recorded; run settings and submission
manifest pending. **72 full trajectories planned; 0 submitted for this campaign.**

Compare search policy, gain evaluation, insertion batching, and work budget
across three drive regimes. Full trajectories are the primary comparison.
Keep singleton joint/factorized pairs as end-to-end parity checks; record any
verified shared route before considering a reduction in the run matrix.

Implementation baseline: `206d3a204edcedbaaf0274d9396483e8837b7434`.
Pin the actual tested revision and executor package hash in the submission
manifest. Existing small trajectory tests are software validation, not results
from this campaign.

## Campaign matrix

| Axis | Values | Count |
|---|---|---:|
| Search policy | `greedy`, `bounded_exhaustive`, `block_exchange` | 3 |
| Gain evaluation | `joint_batch`, `singleton_factorized` | 2 |
| Insertion admission | Singleton only; batching allowed | 2 |
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

The current interpretation of singleton-only is \(|A|\leq1\) **per patch**.
The existing deletion-set permissions and repeated checkpoint rounds remain
shared. Confirm before launch whether “no batching” should also restrict
\(|D|\); that would require an explicit deletion-cardinality policy.

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

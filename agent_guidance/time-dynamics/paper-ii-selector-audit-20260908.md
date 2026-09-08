# Paper II selector audit — 2026-09-08

Status: findings and proposed corrections; no production behavior changed.
Audited worktree: `/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907`.
HEAD: `02b9f8fb`; scientific implementation baseline:
`206d3a204edcedbaaf0274d9396483e8837b7434`.
Pre-existing equation-tag/viewer edits were left untouched.
This audit does not reinterpret v1 results or claim numerical revalidation.

Follow-up: the user authorized correction of redundant proxy work and rejected
deletion-set budget charges, while deferring the legacy frontier-policy change.
The [campaign tracker](paper-ii-selector-campaign.md#corrected-implementation-and-campaign-entry-point)
records the implemented follow-up and local trajectory checks. Findings and
source positions below describe the pre-correction baseline named above.

## Verdict

| Concern | Three experimental searches with compiled-Q | Existing `family_frontiers` path |
|---|---|---|
| Graphical proxy work and old utility | Still computed; not used to choose patches. Removable decision-dead work, with output-schema cleanup. | Still computed; utility actively controls acquisition priorities. Historical and repaired-proxy routes also use it in final ranking. |
| Permission and budget | Correct whole-set permission; refused boundaries still consume structural slots and can block permitted exchanges. Retained accounting policy, not a permission error. | Correct whole-set permission when reached; raw rung sizing can refuse a rung before permission and reserve slots for forbidden variants. Confirmed coverage effect. |
| Pure-insertion frontier | Resolved by the new enumeration design: no sealed pure-insertion frontier. Greedy/block use the shared compiled-Q population ranker for their choices. | Still utility-ranked; under default singleton normalization positive pure-insertion ordering reduces to realized gain alone. A search-policy mismatch with the intended cost-aware rule when the frontier actually restricts coverage. |

The ideal-Gamma objective route is also still selectable. Removing that route
and removing its unused scoring machinery are justified, but the shared
Gamma-based numerical checks currently used by compiled-Q have separate
consumers and must not disappear accidentally.

## Entry points and effective configuration

The CLI and controller still default to the historical bundle
`paper_ii_generalized_exchange_v2`, `family_frontiers`,
`joint_batch`, insertion cardinality one, structural budget 50,000,
`exchange_insertion_frontier=None`, and
`cost_normalization_mode=family_robust_v1`.
The default positioned exchange frontier is therefore unrestricted; a finite
width is an explicit setting. Named campaign profiles may supply such a width.
Sources: [adaptive_trajectory.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/adaptive_trajectory.py:185),
[CLI](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py:1947).

The planned experiments explicitly select
`paper_ii_exchange_q_compiled_v1` and one of `greedy`,
`bounded_exhaustive`, `block_exchange`. The six search/gain configurations
resolve to the same realized-capture objective and compiled population cost.
Experimental policies reject frontier options. Joint mode still accepts the
ideal-Gamma bundle; factorized mode rejects it.
Sources: [exchange_config.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_config.py:189) and
[executed configuration probe](/tmp/paper-ii-curation-20260907/current-audit/config_audit.json).
That probe builds the real CLI parser and resolves real controller objects;
it does not start a trajectory.

The sealed compiler is FakeMarrakesh, optimization level 1, seed 7,
nonzero structural angle 1.0, and resource coordinates
(two-qubit count, two-qubit depth, total depth) with weights
(0.30, 0.30, 0.25). See [exchange_config.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_config.py:46).
Zero-angle tangent acquisition and nonzero structural resource compilation
remain distinct operations.

## 1. Proxy computation and consumers

The adapter unconditionally supplies insertion and deletion proxy callbacks
to the shared evaluator:
[exchange_integration.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_integration.py:1144).
The evaluator calls insertion cost for every record with nonempty A and
deletion cost for every record with nonempty D, before applying the
compiled-resource population rule:
[patch_evaluation.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/patch_evaluation.py:341).

`_paper_i_proxy_denominator` passes exactly one raw record to
`append_cost_telemetry_for_family`:
[exchange_integration.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_integration.py:148).
With `family_robust_v1`, each coordinate equals its singleton median,
so every centered coordinate is zero and the denominator is exactly one.
This is true for each insertion block or deletion set treated as that
one record; it is not population-relative comparison across competing patches.

There are distinct live consumers:

- The three new searches use cumulative population rankings to choose
  extensions and finalists; old candidate utility does not enter those orders.
- `family_frontiers` reads the old utility for both positioned pure-insertion
  screening and nested block priorities. Deletion history and conditioning
  utility can influence the latter.
- Historical final ordering and the repaired population-proxy route retain
  utility consumers. `raw_legacy_v1` can also produce nonunit denominators.
- Attempt and committed JSON fields called `score` still carry old utility;
  authoritative compiled rank information is recorded separately.
  See [exchange_selector.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_selector.py:49) and
  [exchange_integration.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_integration.py:1794).
  Current trajectory acceptance/installation follows certification and does
  not read this old committed score.

Thus removing old utility from the new compiled-Q core preserves its
mathematical decisions, subject to verification of output consumers and
failure behavior. Removing it from the legacy frontier scheduler changes
candidate coverage. Nonzero conditioning/history policies cannot be deleted
merely because one profile sets their weights to zero; distinguish their
legacy preference role from the independent permission and certification
policies that must remain.

Observed counts from the real two-qubit adapter, with batch bound two:

| Case | Insertion estimates | Deletion estimates | Singleton normalizations | Per-atom graphical calls |
|---|---:|---:|---:|---:|
| Greedy, either gain mode | 10 | 8 | 18 | 22 |
| Block, either gain mode | 12 | 10 | 22 | 28 |
| Exhaustive, either gain mode | 12 | 10 | 22 | 28 |
| Compiled-Q family mode | 6 | 6 | 12 | 12 |
| Historical bundle control | 6 | 6 | 12 | 12 |
| Repaired population-proxy control | 12 | 10 | 12 | 22 |

All singleton-normalized results were exactly 1.0. Two scoring workers
reproduced exhaustive serial counts. The repaired-proxy control additionally
performed two genuine population normalizations with nonunit outputs;
the raw-legacy control produced 1.025, 1.075 and 1.125. Those controls use
permissive test-fixture gates, not every raw controller default.

The six compiled-Q cases spent **1.94–2.56 ms** inside the summed disjoint
raw-estimation and singleton-normalization bodies, against **0.762–1.503 s**
adapter wall time. Each is one instrumented toy call. These measurements
establish small local overhead, not campaign or QPU speedup. The per-atom
timer is nested inside estimation and is not added again.
[Compiled counts](/tmp/paper-ii-curation-20260907/current-audit/proxy_audit_compiled_counts.json),
[controls](/tmp/paper-ii-curation-20260907/current-audit/proxy_audit.json),
[probe](/tmp/paper-ii-curation-20260907/current-audit/proxy_audit.py).

The narrowest decision-preserving optimization in isolation would substitute
the proved constant-one result before constructing raw estimates, while
retaining genuine population-proxy inputs for their live consumers. The user's
requested utility deletion should instead remove those consumers coherently;
the constant identity is evidence of redundancy, not a proposed permanent
dummy scoring layer.

## 2. Deletion permission and accounting

An eligible coordinate is d; a proposed deletion set is D. The active sequence is:

1. Target/origin policy, drive protection, support constraints and cooldown
   nominate coordinates d:
   [exchange_integration.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_integration.py:915).
2. Lazy cardinality enumeration constructs complete D, respecting retained
   support size:
   [deletion_family.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/deletion_family.py:61).
3. For each D, permission applies the **ray-displacement bound and reverse-Schur
   loss permission**. Failure of the first bound skips Schur work. The scalar
   second threshold is the normalized exact captured loss, not the Schur
   matrix:
   [deletion_permission.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/deletion_permission.py:156).
4. Refusing D excludes both its pure deletion and its exchanges. The new
   searches check this before asking for that branch's insertion pool.
   Singleton permission never substitutes for permission of a larger D.
5. Whole-request admission precedes actual patch evaluation. Finalist
   materialization, refitting, and post-refit continuity/conditioning
   certification occur later.

Real-array probes test both a combined-angle failure and a redundant-direction
Schur failure: D={0} and D={1} pass separately while D={0,1} fails. Across
family mode and all six experimental variants, no refused-D proposal is
solved or priced. With a nonbinding budget, this fixture gives:

| Count | Family mode | Each new search |
|---|---:|---:|
| Completed permitted patch records | 12 | 12 |
| Pricing callbacks | 9 | 9 |
| Structural slots charged | 15 | 13 |
| Refused-D proposal solves / pricing | 0 / 0 | 0 / 0 |

The pricing callbacks here are observed stand-ins, not Qiskit executions.
Joint mode performs 12 unique array solves; factorized mode performs 9.
The latter's completed batch records reuse singleton gain inputs.

Family mode counts all three raw singleton variants of the refused D.
New searches charge one refused boundary:
[search_methods/session.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/search_methods/session.py:129).
These are not counts of actual patch solves. The work policy deliberately
bounds enumeration too, but mixes two kinds of work under one setting.

### Demonstrated loss of permitted coverage

With the same two active coordinates and two insertion choices, insertion
cardinality one and structural budget six, D={0} is forbidden and D={1}
is permitted:

- Family mode completes only three D=empty records. Its raw next rung costs
  six against three remaining slots, so it rejects before either permission
  test and evaluates no exchanges.
- Each new search completes four records and charges five slots: three
  D=empty records, one refused boundary, and permitted pure deletion D={1}.
  Its two permitted exchanges are then refused as a whole because only one
  slot remains.
- Removing only forbidden sets from this diagnostic's enumeration leaves the
  permitted mathematical universe unchanged. All four policies then complete
  six records, including both exchanges, under the same budget.

This is a concrete accounting-induced coverage effect, not a failure of the
ray-displacement or reverse-Schur mathematics.
[Test](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/test/test_ap_mclachlan_permission_budget_audit.py:243).

### Reservations, whole-rung admission and bounded construction

Whole-request refusal is transactional: no slots are reserved and no fraction
of the request is evaluated. Already admitted reservations are not released
when an unexpected backend failure interrupts evaluation; a probe retained
two charged slots and one completed evidence record. This is conservative
attempt accounting, not proof of two completed solves/compiles.
[search_methods/session.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/search_methods/session.py:143).

Zero structural budget still performs the shared baseline solve; it performs
no subsequent set-permission, proposal-solve or pricing work.

Family-mode raw-rung refusal is separate from the ordering of permission and
charging. It can stop the entire cardinality rung before inspecting any D:
[exchange_structural.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_structural.py:349). New searches admit branch boundaries
and their search neighborhoods as separate whole requests.

One additional behavior-preserving optimization is justified in this scope:
family-mode batch construction tests only structural remaining work while
building plans, then checks compiled remaining work after construction.
With unbounded structural budget and the compiled budget already exhausted,
the probe constructs all 12 joint words before refusing; no extra solve or
pricing follows. Stop plan construction at the first proven compiled-budget
excess while preserving whole-family refusal.
[exchange_structural.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_structural.py:466).

## 3. Frontier ranking versus complete-patch ranking

Use delta here for the code's normalized realized captured-drift change,
to distinguish it from the separate ideal-projection Gamma objective:

\[
\delta_a =
\frac{Q(C+a)-Q(C)}{\|b\|^2+\epsilon},\qquad
Q=2f^\top v-v^\top Gv.
\]

The sealed positioned exchange frontier uses:

\[
s(\varnothing,a)
=\frac{[\delta_a]_+}{c_{\rm proxy}(a)^{\alpha_{\rm ins}}}
+w_\delta\delta_a.
\]

It reads `candidate.score`, not the debt-order key and not ideal Gamma.
For positive gains and default singleton-normalized cost,
\(s=(1+w_\delta)\delta_a\), with default \(w_\delta=1\).
Sources: [patch_evaluation.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/patch_evaluation.py:341),
[exchange_structural.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_structural.py:240),
[exchange_structural.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_structural.py:330).

Nested block frontiers have a further distinction: child priority is its
maximum old utility over all acquired singleton deletion branches, so deletion
utility can influence those priorities:
[exchange_structural.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_structural.py:291) and
[exchange_structural.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_structural.py:426).

Final compiled-Q ranking uses:

\[
S(P)=\frac{\delta_P+\tau}{K(P)}-\tau.
\]

Tau is zero in large-miss operation; maintenance uses the declared deletion
loss tolerance. The normalization population consists of valid, acquired,
regime-active complete patches before merit-sign filtering. Large miss admits
insertion-containing competitors; maintenance admits pure deletion
competitors. Stay is a fixed reference rather than a cost observation.
For family mode this population is the acquired certification level. For the
new searches it is cumulative acquired evidence at each ranking step.
A sealed finalist population is not renormalized after a certification failure.
See [exchange_integration.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_integration.py:564).

For each resource coordinate j, with complete-patch marginal x:

\[
z_j(P)=\frac{2}{\pi}\arctan
\frac{x_j(P)-\operatorname{median}_{\mathcal P}x_j}
{\max(\operatorname{MAD}_{\mathcal P}x_j,10^{-12})},
\quad
\bar z(P)=\frac{\sum_j w_jz_j(P)}{\sum_jw_j},
\quad
K(P)=\frac{2}{2-\bar z(P)}.
\]

Source: [population_scoring.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/population_scoring.py:396).
Different searches can acquire different populations and consequently assign
different K to the same patch. They share the normalization rule; they do not
necessarily share a normalization population.

The reference resource triple is the unchanged checkpoint circuit at its
current parameters. Commutation-equivalent placement trials compile complete
circuits containing the prospective insertion at the structural angle.
Their placement choice is cached against this undeleted base. Complete
deletion/insertion patches are then compiled with remapped chosen placements,
and their resource triples are differenced against the same unchanged base.
Placement is not reoptimized after D.
[exchange_integration.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_integration.py:310).

Singleton complete-patch costs are already acquired during scoring before
the old frontier is sealed; that frontier simply does not use them. A focused
counterexample with raw gains .8/.6 and **synthetic** complete marginals
100/0 gives K=4/3 and 4/5: the frontier retains the higher-gain insertion,
while final compiled-Q prefers the lower-gain exchange. Real-Qiskit adapter
tests separately confirm call timing, singleton normalization, and placement
reuse. [compiled_patch_cost.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/compiled_patch_cost.py:746) and
[audit tests](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/test/test_ap_mclachlan_frontier_audit.py:77).

All three new searches acquire both branch singleton alternatives in this
counterexample. They do not exclude an insertion from exchange using its
pure-insertion score. Remaining cardinality, neighborhood and work-budget
restrictions are explicit search policies.

## Original correction proposals

These proposals preceded the user's follow-up scope decision. Redundant
singleton proxy computation and permission-budget accounting have since been
corrected. Frontier/utility retirement remains deferred; the new campaign
entry point instead pins the intended search and compiled-Q objective.

1. **Remove the ideal-Gamma objective route** from the executable bundle
   registry, CLI, live run builders and objective switch. Retired route IDs
   should fail clearly; do not silently reinterpret them as compiled-Q.
   Keep shared numerical Gamma checks used by compiled-Q until a separate
   mathematical change is proposed. Current shared uses are visible at
   [patch_evaluation.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/patch_evaluation.py:414) and
   [exchange_integration.py](/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-scoring-curation-20260907/pipelines/time_dynamics/ap_mclachlan/exchange_integration.py:605).
2. **Remove old utility and graphical proxy work from the compiled-Q
   evaluator**, replacing ambiguous output score fields with authoritative
   population ranking evidence. Preserve both provisional ranking gain and
   later realized joint-validation gain for factorized batches; they are
   different quantities. Put conditioning permission/certification policies
   outside the deleted legacy preference calculation.
3. **Retire the live legacy utility consumers together**, rather than leave
   dummy scores. These include historical/proxy objective routes and the
   utility-based frontier scheduler. The agreed three-search campaign does
   not require that scheduler. If a frontier policy is retained for future
   experiments, specify and review its compiled-Q ranking/population as a
   separate search policy before implementing it.
4. **Separate permission-work accounting from patch-evaluation admission.**
   Evaluate permission on complete D; rejected sets should not enlarge an
   admitted patch-evaluation request. Record permission attempts separately
   and bound their classical work explicitly if needed. Keep whole-request
   admission and conditioning/certification safeguards. This changes coverage
   under finite limits and requires local trajectory revalidation.
5. **If family mode survives the route deletion**, add the early compiled-budget
   construction check described above. This preserves its selected population
   and refusal semantics, avoiding needless plan materialization.

Items 1–2 preserve compiled-Q choices if shared safeguards, evidence and
ranking inputs are retained. Item 1 removes an intentionally distinct
objective, not an equivalent implementation. Item 2 changes the reporting
schema. Retiring/replacing the default family policy and item 4 change candidate
coverage, normalization populations and potentially trajectories; test them
as such. Do not remove conditioning/history policies solely because default
weights are zero.

v1 is preserved by its existing code revision and unchanged saved artifacts.
No live compatibility route is needed solely to keep that evidence available.
Do not rewrite old receipts or published results under the new score schema.

## Validation and limitations

GitNexus was bound to this exact worktree. Its index advertises a two-commit
lag (stamp `ff198414`); direct current-source verification and executed
call-path probes establish the findings. Impact inspection identifies nine
direct PatchEvaluator dependents and the population-ranker integration caller.
The process inventory is bounded, so an empty process result does not prove
absence of dynamic consumers. No production symbol was edited.

Focused audit tests and instrumentation cover real array geometry, real
permission, the actual adapter, actual Qiskit compilation where stated,
and deterministic synthetic budget/cost counterexamples. They do not measure
campaign-scale wall-time savings, QPU latency, or full-trajectory numerical
effects. Existing v1 trajectory evidence was not re-audited in this task.

Combined focused verification: **53 passed in 1.59 s**, under the 3 GB guard:

```sh
/Users/lordskyler/Holstein_test/.venv/bin/python pipelines/shell/ram_guard.py --limit-mb 3000 -- /Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q test/test_ap_mclachlan_frontier_audit.py test/test_ap_mclachlan_permission_budget_audit.py
```

Run from the audited worktree. The separate configuration probe passed all
six compiled-Q combinations and checked that ideal-Gamma remains allowed
for joint mode. The proxy probe completed eight compiled cases and three
controls. Its first invocation failed only while reporting a historical
`compile=None` receipt after the eight successful compiled cases; that
reporting code was corrected and controls were collected successfully.
No production error is inferred from that instrumentation failure.

Audit tests intentionally expose current shortcomings; adapt those assertions
to the corrected contract when implementing a fix instead of preserving the
shortcoming as required behavior.

Files to edit:
- No production files in this audit. Proposed production surfaces for the next
  reviewed increment are `exchange_config.py`, `patch_evaluation.py`,
  `exchange_integration.py`, `exchange_selector.py`,
  `exchange_structural.py`, `search_methods/session.py`,
  `adaptive_trajectory.py`, their live runner/run-builder consumers, and
  focused tests.

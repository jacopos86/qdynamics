# Claude Code handoff: independent AVQE estimator/cache audit

## Objective

Independently determine whether the current Paper-I audit is correct about
AVQE Powell caching, estimator-query accounting, the claimed need to rerun
the synthetic-noise AVQE comparator, and the disputed identity of the actual
Paper-I HH RA route.  Do not inherit the Codex conclusion.  Trace the frozen
source and executed receipts yourself, then record a clearly labeled Claude
Code review in the single audit ledger.

The disputed claims are:

1. Paper I should charge every Powell objective callback as a Hamiltonian
   expectation, including repeated callbacks at identical parameters.
2. AVQE memoizes exact parameter-byte keys, records only cache misses, and
   serializes that distinct-acquisition count as `optimizer.nfev`, thereby
   losing SciPy's callback count.
3. Frozen replay recovered weak-HH callback counts
   `3351/2566/2565`, compared with stored Powell counts
   `3122/2386/2374`, and therefore gives total estimator counts
   `13779/11098/12045` after the proposed outer-energy correction.
4. In the synthetic-noise comparator, cache hits skip both an acquisition and
   a noise draw, so the existing trajectory supposedly requires a no-cache
   rerun rather than an accounting correction.
5. The earlier audit claimed that no available Paper-I HH campaign executed
   the flat global-singleton route.  The author disputes this and says the
   audit found an old compatibility route rather than the route actually used.

## Decisions already made

- This is an evidence audit, not authorization to edit Paper I, regenerate
  figures, submit jobs, or launch reruns.
- Distinguish optimizer callback requests from physical/logical quantum
  acquisitions.  Do not assume they are the same resource.
- A real QPU does not by itself prohibit retaining and reusing an expectation
  already acquired at an identical parameter point.  Determine whether the
  manuscript defines requested callbacks or executed acquisitions.
- Preserve repeated Powell evaluations at genuinely different parameter
  points.  The only disputed case is an exact repeated point within the same
  optimizer execution.
- Treat the existing Codex audit prose as a hypothesis, not evidence.

## Evidence locations

- Single audit ledger, especially `EST-04`, `EST-07`, `EST-08`, `QPU-03`,
  `QPU-04`, and `RES-04`:
  `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/paper-i-factual-audit-20260822/docs/reports/PAPER_I_FACTUAL_MATHEMATICAL_QPU_AUDIT_20260822.md`
- Earlier estimator findings, which may contradict the later interpretation:
  `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/ra-refactor-stage2/chtc/paper_i_ra_adapt_repair_20260727/FINDINGS_ESTIMATOR_LEDGER_DUPLICATION_20260821.md`
- Paper-I source:
  `/Users/jakestrobel/local_repos/ADAPT---Paper-I/Paper_I_overleaf_20260818/Paper_I.tex`
- Source-locked package whose source archive has SHA-256
  `1f949b0cc8b61dca63911832e8dc8bb32614174755ac476827956bb0812accee`:
  `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/paper-i-factual-audit-20260822/chtc/paper_i_ra_adapt_repair_20260727/stationary_core_full48_r50_20260728_v5_chtc/`
- Current source and run artifacts:
  `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3`
  and
  `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/ra-refactor-stage2`

## Required audit

1. Identify the exact frozen AVQE objective wrapper.  State whether its cache
   increments `objective_calls` on every SciPy callback or only on cache misses,
   and what is serialized as `history[].optimizer.nfev`.  Cite paths and exact
   symbols or line numbers.
2. Independently reproduce—or reject as unreproducible—the three pairs of
   callback/cache-miss counts and the derived estimator totals.  Name every
   script, receipt, or artifact used.  Do not accept numbers merely because
   they appear in the audit ledger.
3. Read the manuscript's estimator definition and decide whether its resource
   is requested objective callbacks, unique logical scalar evaluations, or
   actual acquisitions.  Flag any place where the Codex audit silently changed
   that definition.
   Whichever convention you recommend, apply it symmetrically to AVQE and RA.
   If exact-point reuse is treated as mathematically free even when one runtime
   recomputed it, determine whether RA Powell repetitions must also be removed;
   do not preserve AVQE's implementation cache while charging RA's equivalent
   repeated points merely because their wrappers differ.
4. Trace the fixed-realization noise oracle.  Determine whether a cache hit
   advances its evaluation ordinal or RNG state, whether disabling the cache
   would change the trajectory, and whether that makes the existing run false
   as reported or merely a different valid executed convention.
5. Give each disputed claim one verdict: `CONFIRMED`, `PARTLY CONFIRMED`,
   `CONTRADICTED`, or `UNVERIFIED`.
6. Independently bind the real Paper-I HH RA route from executed receipts,
   not from stale-looking protocol field names.  Compare at least:
   - cluster 9664325, archive
     `output/local_runs/ra_adapt_costexp_depthfix_sixregime_20260820_v2/transfer/sixregime__b_depth_append_ra__weak_weak__nph3__9664325__12.tar.gz`;
   - cluster 9674089, archive
     `output/local_runs/paper_i_hh_b3_measured_residual_20260821/transfer/b3mr__mr_append_ra__weak_weak__nph3__9674089__0.tar.gz`;
   - if relevant, cluster 9674592/B6.
   In each checkpoint, inspect the Phase-0 input population and the executed
   `projected_phase3_population_receipt`, especially `split_parent_count`,
   `split_child_count`, `unsplit_singleton_count`, `split_parent`,
   `child_atom_count`, adapter identity, and Phase-I/II supply/exposure.
   Do not infer execution from `phase3_runtime_split_mode` alone.

## Codex reproduction results to check independently

These are observations for cross-checking, not premises for the Claude review.

- Frozen AVQE replay reproduced all three weak-HH prefixes:
  - W--W, k=11: 3351 Powell callbacks, 3122 stored cache misses, gap 229,
    proposed callback-count total 13779.
  - I--W, k=9: 2566 callbacks, 2386 misses, gap 180, proposed total 11098.
  - S--W, k=10: 2565 callbacks, 2374 misses, gap 191, proposed total 12045.
  The bounded replay script is `/tmp/replay_aavqe_callbacks.py`; it runs from
  `/tmp/paper_i_aavqe_source_audit.08a5oX` and must be inspected rather than
  trusted blindly.
- The executed round-1 receipt in cluster 9664325 reports a Phase-0 input of
  948 global guarded singletons, retains 24, then reports
  `split_parent_count=0`, `split_child_count=0`,
  `unsplit_singleton_count=24`, all `split_parent=false`, all
  `child_atom_count=0`, and identity exposure into Phase II.  Its semantic
  contract says `global_guarded_singleton_pool_v1`,
  `compatibility_resolution_active=false`, and
  `adapt_child_pool_expansion_mode=off`.
- The executed round-1 receipt in cluster 9674089 instead reports eight split
  parents producing 29 children (`split_parent_count=8`,
  `split_child_count=29`, `unsplit_singleton_count=0`).  Cluster 9674592/B6
  shows the same 8-to-29 split in the inspected W--W Append archive.
- This evidence currently indicates that 9664325 is a real global-singleton
  Paper-I route and that 9674089/B5 is a distinct legacy-hierarchy rerun.  The
  current manuscript figure provenance was later switched from Bundle 3 to
  B5, so Claude should determine which campaign the author actually designated
  as the scientific result rather than treating B5's contents as the method
  definition.

## Authority and constraints

You may run focused read-only inspections and bounded local reproductions that
do not launch a scientific campaign.  Stream large JSON; do not load giant
checkpoints into memory.  Do not modify implementation code or Paper I.  Update
only the single audit ledger by appending a subsection titled
`Claude Code independent review — AVQE Powell cache and noise`, including the
verdict table, source citations, and the minimal correction to the existing
audit.  Preserve the prior text so disagreements remain visible.

## Definition of done

- The cache/noise claims and the route-identity claim have independent verdicts
  backed by source/artifact evidence.
- The callback-versus-acquisition convention is resolved explicitly.
- The ledger states whether any estimator recount or noise rerun is actually
  necessary, optional, or unsupported.
- No manuscript, figure, runtime, or campaign file is changed.

## True stop conditions

Stop only if a necessary frozen source or artifact is irrecoverably absent, or
if a bounded reproduction would require launching a new scientific run.  In
that case, record exactly what is missing and leave the affected claim
`UNVERIFIED`; do not infer a rerun requirement from absence alone.

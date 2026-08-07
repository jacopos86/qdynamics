# Oracle Plan

# Summary

Run117 is most likely slow because the exact support-patch selector is repeatedly doing dense LAPACK eigensolve/rank/inverse work for candidate patches, especially through prune and exchange scoring, while recomputing checkpoint-invariant `K_before` quantities for every candidate. The slowdown appears CPU-bound, not I/O-bound. Treat this first as a targeted performance/cache-miss problem in `adaptive_trajectory.py` and `support_patch.py`, not as a reason to change Paper-II selection policy.

# Current-state analysis

Based on the provided code evidence, the relevant flow is:

1. `adaptive_trajectory.py:_select_unified_support_patch`
   - Builds a stay finalist.
   - Scores append candidates when residual exceeds threshold.
   - Scores prune candidates through the active prune ladder.
   - Scores exchange pairs when `exchange_enabled` and both append/prune batches exist.
   - Chooses one finalist.

2. Append path:
   - `_score_append_atom_set`
   - Appends candidate atoms.
   - Evaluates full McLachlan geometry for the appended state.
   - Builds `SupportPatchGeometry`.
   - Calls `support_patch.py:score_support_patch`.

3. Prune path:
   - `_score_prune_ladder_batch`
   - Scores rung candidates via `_score_prune_atom_set`.
   - `_score_prune_atom_set` calls:
     - `score_support_patch(delete patch)`
     - `prune_conditioning_diagnostics(delete patch)`

4. Exchange path:
   - `_score_exchange_candidate_only`
   - For every append/delete pair:
     - Appends candidate atoms.
     - Evaluates full McLachlan geometry.
     - Calls `score_support_patch` three times:
       - true exchange
       - delete-only control
       - append-only control

5. `support_patch.py:score_support_patch`
   - Builds after geometry.
   - Optionally computes Schur novelty.
   - Solves on `K_before`.
   - Solves on `K_after`.
   - Computes rank of ridged `K_before`.
   - Computes rank of ridged `K_after`.

6. `support_patch.py:prune_conditioning_diagnostics`
   - Recomputes `_shared_supported_inverse(K_before)` per delete candidate.
   - Computes `_shared_supported_inverse(K_after)`.
   - Computes removed-vs-retained novelty.

The key blocking issue is that `K_before` and `f_before` are checkpoint-invariant, but their solve/rank/inverse products are recomputed per candidate and per exchange sub-score. The macOS sample showing `numpy.linalg.eigh` / `eigvalsh` / `dsyevd` strongly matches repeated dense linear algebra, especially `_shared_supported_inverse`, Schur novelty, rank, or related conditioning diagnostics.

# Design

## 1. Most likely root cause

The most likely root cause is the multiplicative combination of:

1. **Repeated checkpoint-invariant LAPACK work**
   - `score_support_patch` recomputes before-solve and before-rank for every candidate.
   - `prune_conditioning_diagnostics` recomputes before-inverse for every delete candidate.
   - These should be checkpoint-scoped cached values.

2. **Exchange amplification**
   - Each exchange pair performs three `score_support_patch` calls.
   - Append-only and delete-only controls are likely identical to already-scored append/prune candidates or repeated across many pairs.
   - This turns a small candidate set into many repeated eigensolve/rank calls.

3. **Possible BLAS oversubscription**
   - `_ordered_parallel_map` uses `ThreadPoolExecutor`.
   - Each worker calls NumPy/SciPy LAPACK, which may itself use multiple BLAS threads.
   - On macOS this can cause severe oversubscription: Python workers × BLAS threads.

4. **Silent no-edit checkpoints still do full scoring**
   - The log only shows edits/rejections.
   - The long gap between `k=8` and `k=48` likely still contains expensive no-edit support-patch scoring.

## 2. Pure performance bugs/cache misses vs intended semantics

| Area | Classification | Notes |
|---|---:|---|
| Recomputing `K_before` solve/rank in `score_support_patch` | Pure performance bug | Same checkpoint input, same result. Cache safely. |
| Recomputing `_shared_supported_inverse(K_before)` in `prune_conditioning_diagnostics` | Pure performance bug | Checkpoint-invariant. Cache safely. |
| Recomputing append-only/delete-only exchange control scores per pair | Likely performance bug | Reuse only after validating exact semantic equivalence with append/prune batch scores. |
| Re-evaluating appended-state geometry for same append atom set in append and exchange | Likely performance bug | Cache per append candidate within checkpoint. |
| ThreadPool + multithreaded BLAS oversubscription | Performance/config bug | Tune worker/thread layering without changing algorithm. |
| Scoring append candidates with full McLachlan geometry | Intended expensive semantics | Do not replace with proxy scoring yet. |
| Prune ladder rung-by-rung scoring | Intended Paper-II semantics | Do not flatten/skip rungs yet. |
| Exchange true-pair scoring | Intended expensive semantics | True exchange score should still be evaluated exactly. |
| Prune persistence wait | Unknown | If wait is deterministic cooldown, scoring before rejecting may be waste. If it accumulates persistence evidence, scoring is semantic. Instrument first. |
| Rank/conditioning/novelty diagnostics | Intended semantics | Cache exact results, but do not approximate/remove. |

# Minimal instrumentation before policy changes

Add profiling behind an explicit flag, e.g. `AP_MCLACHLAN_PROFILE_SUPPORT_PATCH=1`, and write to the existing run log because interrupted runs may not produce JSON.

## In `adaptive_trajectory.py`

Instrument:

- `_select_unified_support_patch`
  - `k`, `rho`, `params`
  - append candidate count
  - prune candidates by rung
  - exchange pair count
  - selected finalist kind
  - rejection reason, e.g. `prune_persistence_wait`
  - phase timings:
    - stay
    - append batch
    - prune ladder
    - exchange batch
    - finalist selection

- `_ordered_parallel_map`
  - item count
  - worker count
  - wall time
  - per-item min/p50/max if cheap
  - BLAS/thread environment:
    - `OMP_NUM_THREADS`
    - `OPENBLAS_NUM_THREADS`
    - `MKL_NUM_THREADS`
    - `VECLIB_MAXIMUM_THREADS`

- `_score_append_atom_set`
  - appended atom-set key
  - full geometry evaluation time
  - `score_support_patch` time
  - matrix dimensions

- `_score_prune_ladder_batch`
  - rung number
  - candidate count
  - accepted/rejected reason
  - whether persistence gate was known before scoring

- `_score_prune_atom_set`
  - `score_support_patch` time
  - `prune_conditioning_diagnostics` time

- `_score_exchange_candidate_only`
  - append key
  - delete key
  - geometry evaluation time
  - true-exchange score time
  - delete-only control score time
  - append-only control score time

## In `support_patch.py`

Instrument:

- `score_support_patch`
  - total time
  - `build_after_geometry` time
  - Schur novelty time
  - before solve time
  - after solve time
  - before rank time
  - after rank time
  - matrix shapes

- `prune_conditioning_diagnostics`
  - before inverse time
  - after inverse time
  - novelty time
  - matrix shapes

Useful summary log shape:

```text
PROFILE_SUPPORT k=50 params=74 rho=1.09e-4 workers=...
counts append=12 prune_by_rung={3:2} exchange_pairs=...
time append=... prune=... exchange=...
la_calls before_solve=... before_rank=... before_inverse=...
la_time before_solve=... before_rank=... before_inverse=...
```

# Semantic-preserving speedups to prioritize

## Priority 1: Add checkpoint-scoped before-cache

In `support_patch.py`, introduce an internal checkpoint-scoped cache for invariant before-side linear algebra.

Conceptual shape:

```text
SupportPatchBeforeCache
- K_before shape/fingerprint
- f_before shape/fingerprint
- ridged_K_before
- before_solution
- before_rank
- before_shared_inverse optional
```

Modify these functions additively:

```text
score_support_patch(..., before_cache=None, profile=None)

prune_conditioning_diagnostics(..., before_cache=None, profile=None)
```

Rules:

- If `before_cache is None`, behavior remains exactly current.
- If provided, use cached before solve/rank/inverse.
- Cache must be checkpoint-local.
- Do not reuse across checkpoints.
- Build cache before parallel candidate scoring to avoid thread races.

## Priority 2: Reuse append/prune control scores in exchange

In `adaptive_trajectory.py:_select_unified_support_patch`:

- Keep maps from canonical atom-set keys to scored append/prune results.
- In `_score_exchange_candidate_only`, compute only the true exchange score.
- Reuse:
  - append-only score from append batch
  - delete-only score from prune batch

Only do this after validating that the reused score has identical inputs to the current control score path.

## Priority 3: Cache appended-state geometry per append candidate

The appended-state full McLachlan geometry is reused by:

- append scoring
- true exchange scoring for every pair using that append candidate
- append-only exchange control, if still needed

Keep this cache checkpoint-local and release it after finalist selection.

## Priority 4: Tune Python workers vs BLAS threads

Before changing algorithmic policy, benchmark:

- `support_patch_scoring_workers=1`, BLAS default
- multiple Python workers, BLAS threads forced to 1
- current setting

If adding code support, make BLAS thread limiting opt-in/configurable first. This preserves semantics but may slightly change floating-point low bits depending on LAPACK backend.

## Priority 5: Deduplicate canonical candidate atom sets

Before calling `_ordered_parallel_map`, canonicalize append/delete atom-set keys and score each unique candidate once. Map duplicate keys back to original candidate records.

# What not to change yet

Do **not** change these until profiling proves exact scoring remains too slow after caching:

- Do not reduce append candidate count.
- Do not skip prune ladder rungs.
- Do not disable exchange.
- Do not restrict exchange pairs heuristically.
- Do not change residual thresholds.
- Do not change prune persistence semantics.
- Do not remove rank, conditioning, or Schur novelty diagnostics.
- Do not replace full McLachlan geometry with proxy geometry.
- Do not downsample `n=601`.
- Do not change ridge/tolerance definitions.
- Do not accept/reject delete patches earlier unless instrumentation proves the persistence gate is independent of scored candidate identity.

# File-by-file impact

## `adaptive_trajectory.py`

Changes:

- Add optional support-patch profiling in:
  - `_select_unified_support_patch`
  - `_ordered_parallel_map`
  - `_score_append_atom_set`
  - `_score_prune_ladder_batch`
  - `_score_prune_atom_set`
  - `_score_exchange_candidate_only`

- Add propagation of `before_cache` into scoring calls.

- Later, add checkpoint-local maps:
  - append candidate key → append score/result
  - append candidate key → appended geometry
  - delete candidate key → prune score/result

Why:

- This file owns candidate orchestration and can see duplicate/reused work across append/prune/exchange phases.

Dependencies:

- Requires additive `before_cache` support in `support_patch.py`.
- Exchange memoization depends on validating control-score equivalence.

## `support_patch.py`

Changes:

- Add optional profiling around dense linear algebra sections.
- Add checkpoint-scoped before-cache type/helper.
- Modify:
  - `score_support_patch`
  - `prune_conditioning_diagnostics`
- Reuse cached:
  - before solve
  - before rank
  - before shared inverse

Why:

- This file owns the repeated LAPACK work shown in the sample.

Dependencies:

- Must remain backward-compatible for existing callers by making new parameters optional/keyword-only.

# Risks and migration

- No persistence schema change is needed.
- No JSON schema change is needed for the first instrumentation pass; log-only profiling is preferable because run117 produced no JSON.
- Main correctness risk is stale cache reuse. Mitigate by making caches checkpoint-local and asserting matrix shape/config compatibility.
- Main concurrency risk is mutable shared cache access from worker threads. Mitigate by precomputing cache contents before submitting parallel work and treating cache as immutable.
- Main reproducibility risk is BLAS thread tuning. Keep it configurable and compare outputs against a short baseline.

# Implementation order

1. **Add log-only profiling first**
   - No algorithm changes.
   - Confirm whether time is dominated by before inverse/rank/solve, exchange controls, geometry evaluation, or thread oversubscription.

2. **Run a short reproduction**
   - Same run settings, limited checkpoint range if possible.
   - Capture per-checkpoint candidate counts and LAPACK call counts.

3. **Add `SupportPatchBeforeCache`**
   - Wire into `score_support_patch` and `prune_conditioning_diagnostics`.
   - Verify outputs match uncached path on representative append/delete/exchange candidates.

4. **Pass before-cache from `_select_unified_support_patch`**
   - Precompute once per checkpoint.
   - Re-run short profile.

5. **Memoize exchange append/delete controls**
   - Reuse existing append/prune batch scores only after equivalence validation.
   - Keep true exchange scoring exact.

6. **Cache appended geometry**
   - Reuse per append atom set across append and exchange phases.

7. **Benchmark worker/thread settings**
   - Choose a documented default or run-specific recommendation after measuring.

8. **Only then consider policy changes**
   - If exact cached semantics remain too slow, separately evaluate Paper-II policy changes such as exchange pruning or candidate-count reduction.
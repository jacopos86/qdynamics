# Manuscript–implementation math handoff: batch, beam, ablation

**For:** the Paper I manuscript agent.
**Purpose:** the appendix descriptions of the three optional controls are not
yet isomorphic to what the code executes. This document states, for each
control, (a) what the manuscript currently claims, (b) the exact code that
runs, (c) my reading of the mathematics that code implements, and (d) the
specific divergences, each with the options for resolving it.

**This document does not decide anything.** Several divergences can be closed
either by changing the prose or by changing the code, and those are different
scientific claims with different rerun costs. Discuss with the author before
editing the manuscript. Where I am uncertain I say so explicitly rather than
guessing — treat any statement not backed by a quoted line as provisional.

---

## 0. Anchors

| | |
|---|---|
| Code repo | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820` |
| Branch | `appendix-demos-20260820` |
| Commit | `e80989373c994197d30fb1994e3720256979b90f` |
| Manuscript | `/Users/jakestrobel/local_repos/ADAPT---Paper-I/Paper_I_overleaf_20260818/Paper_I.tex` |
| Appendices | `app:batching`, `app:ablation`, `app:beam` |
| Campaign | `ra_adapt_appendix_demos_l3_20260820_v9` |
| Figure numbers | `chtc/ra_adapt_appendix_demos_20260820/results_l3_v9/figures/ra_appendix_l3_v9_provenance.json` |

Physics for every arm: half-filled `L=3` Hubbard chain, open boundary, `t=1`,
`U/t=1.25`, 6 qubits, HF reference, `E_ED = -2.1414935642714075`. Pool is
`macro_generator_v1`, 24 sector-guarded intact generators. Insertion is
append-only, no warm start, horizon 15, Powell refit at `maxiter=200`,
`xtol=1e-5`, `ftol=1e-12`.

---

## 1. The shared cost object

All three controls are supposed to divide a predicted gain by a *relative*
resource-cost factor. The manuscript defines it once, in
`subsec:cost_weighted_record_ranking`, Eq. `eq:hardware_cost_sum`:

$$K_t(r) = \frac{2}{2 - \bar C_\lambda(r;t)}$$

with `C̄` the record's compiled burden, median-centred, MAD-scaled, squashed
through `arctan` into `(-1, 1)`. Therefore:

- `C̄ → -1` (cheapest) gives `K → 2/3`
- `C̄ = 0` (median) gives `K = 1` **exactly**
- `C̄ → +1` (dearest) gives `K → 2`

**`K = 1` is the median-neutral point of a relative factor, not a disabled
cost term.** This trips people up, including me: observing `K = 1` on an
accepted record means that record sat at the median of its population, not
that cost was ignored. Do not "fix" a `K = 1` observation.

The weights are `(w_2q, w_D2q, w_D) = (0.30, 0.30, 0.25)` with the one-qubit
and size weights zero, and the burden is a FakeMarrakesh transpile at
optimization level 1, seed 7, of the **full circuit including reference-state
preparation**, no initial layout. This is the same backend and the same
weights the paper reports its `(N2q, D2q, Dc)` tuples on — verified by
recompiling the reported prefixes and matching the stored digests.

The beam and the ablation each apply this construction at a **different
population level** than Phase III does. That is the single most important
thing for the manuscript to get right, and it is developed per control below.

---

## 2. Batch admission (`app:batching`)

### 2.1 What the manuscript claims

$$\mathfrak B_k \subseteq 2^{\mathcal R^{(3)}} \setminus \{\varnothing\}$$

containing exactly the batches `B` with `|B| <= b_max` for which **every
distinct pair** `r, r' ∈ B` has **disjoint generator support** and satisfies
**`[A_r, A_r'] = 0`**. Admission is

$$B_k^\star \in \arg\max_{B \in \mathfrak B_k} \frac{\Delta E_3(B)}{K_3(B)}$$

### 2.2 What the code runs

Contract, `pipelines/static_adapt/sr_snake/contracts.py`:

```python
class CombinatorialBatchAdmission(SerializableContract):
    maximum_size: int = 3
    search_window_size: int | FullCombinatorialSearchWindow | None = None
    kind: ClassVar[str] = "combinatorial_batch"

    def __post_init__(self) -> None:
        ...
        if self.search_window_size is None:
            object.__setattr__(
                self, "search_window_size", min(2 * maximum_size, 10)
            )
```

The appendix arm uses `maximum_size=3`, so `search_window_size = 6`.

Enumeration, `pipelines/scaffold/hh_continuation_scoring.py:17041`
(`combinatorial_reduced_plane_batch_proposals`):

```python
    search_population, rank_prefilter = _rank_feasible_child_phase2_population(
        shell, cfg=cfg, selected_ops=selected_ops, theta=theta,
        psi_state=psi_state, h_compiled=h_compiled,
    )
    if _canonical_ranked_child_batch_search(cfg):
        ...
        search_pool_size = (
            int(len(search_population))
            if requested_search_pool_size == 0
            else int(min(len(search_population), requested_search_pool_size))
        )
        enumeration_size_cap = int(cap_count)
    ...
    search_pool = [dict(rec) for rec in search_population[:search_pool_size]]
    ...
    for size in range(1, int(enumeration_size_cap) + 1):
        for combo in itertools.combinations(search_pool, int(size)):
```

### 2.3 Mathematical reading

The executed object is

$$\mathfrak B_k^{\rm exec} = \{ B \subseteq \mathcal W_k : 1 \le |B| \le b_{\max},\ \text{feasible}(B) \}$$

where `W_k` is the **top-`W` prefix of a feasibility-ranked child population**,
itself derived from the Phase-II/III shell rather than from `R^(3)` directly.
With `b_max = 3`, `W = 6` this is at most
`C(6,1) + C(6,2) + C(6,3) = 6 + 15 + 20 = 41` subsets scored per adaptive step.

Within the window the search **is exhaustive** — `itertools.combinations` over
all sizes up to the cap. It is not a greedy construction. But it is also not
the unrestricted ideal over `R^(3)`.

### 2.4 Divergences

**(B1) The search set is windowed, and the window is not in the manuscript.**
The prose now mentions `W = 6` and `b_max = 3` (the author added this), which
is good. What still needs care is that `W_k` is a prefix of a *feasibility-
ranked child* population, not simply "the top `W` Phase-III records". If the
manuscript says the latter, it is close but not exact.

**(B2) "Greedy is the practical approximation" is not what runs.** The
executed arm is exhaustive-within-window. Greedy exists in the codebase
(`greedy_reduced_plane_batch_proposals`) but the appendix arm binds
`combinatorial_reduced_plane_batch_proposals`. Any sentence implying the
reported batch numbers came from a greedy admission is wrong.

**(B3) The pairwise condition may not be disjoint-support-and-commuting.**
This is the one I could not close. The manuscript defines `B_k` by an explicit
pairwise algebraic condition. The code rejects a subset when
`summary["feasible"]` is false, under
`batch_search_feasibility_policy = BATCH_SEARCH_FEASIBILITY_JOINT_SUBSET_GATE_V1`,
and the surrounding diagnostics are about `active_metric_support_rank` and
`joint_metric_support_rank` — i.e. a **rank condition on the joint metric**,
not obviously the stated commutator condition. There is also an explicit
`duplicate_exact_child_identity` rejection.

I did not trace the joint subset gate to the point where I could state its
mathematical content, so **do not write either version as settled**. This needs
either (i) someone to read `_evaluate_ordered_reduced_plane_batch_proposal` and
the joint subset gate to the bottom, or (ii) an instrumented run recording, per
rejected subset, whether the pair was non-commuting, support-overlapping, or
rank-deficient. Option (ii) is a small change and would settle it empirically.

---

## 3. Beam continuation (`app:beam`)

### 3.1 What the manuscript claims

Branch-local charging, removal of Pareto-dominated branches, and ranking by

$$\Phi_b = \frac{E_{\rm root} - E_b}{K_b}$$

with the top `B_live` retained.

### 3.2 What the code runs

Shape for the appendix arm: `ForkLocalBeam(live_parent_branches=4,
admission_children_per_parent=3, maximum_admission_children_per_round=12,
s_alg_weight=0.0, comparison="branch_compiled_cost_v1")`.

Pricing settings are read off the route contract, so the beam cannot drift
onto a different backend than Phase III and the reported tuples:

`pipelines/static_adapt/sr_snake/_controller.py:1268`

```python
def _branch_compiled_cost_contract(runtime: Any) -> dict[str, Any]:
    """Read the beam's pricing settings off the route contract.

    The branch price has to be the same quantity Phase III scores records by
    and the same quantity the paper reports, so the backend, optimization
    level, transpiler seed and the three weights are taken from the route
    contract rather than restated here -- otherwise a contract change would
    silently leave the beam optimizing the old cost.
    """

    context = getattr(runtime, "context", None)
    contract = getattr(context, "route_contract", None)
    execution = (
        contract.get("execution_settings")
        if isinstance(contract, Mapping)
        else None
    )
    if not isinstance(execution, Mapping):
        raise RuntimeError(
            "Branch-compiled-cost beam requires the route contract's "
            "execution settings."
        )
    backend = str(execution.get("phase3_backend_name") or "")
    if not backend:
        raise RuntimeError(
            "Branch-compiled-cost beam requires a declared compile backend."
        )
    return {
        "backend": backend,
        "optimization_level": int(
            execution["phase3_backend_optimization_level"]
        ),
        "seed_transpiler": int(execution["phase3_backend_transpile_seed"]),
        "weights": {
            "N2q": float(execution["phase3_backend_w_2q"]),
            "D2q": float(execution["phase3_backend_w_depth_2q"]),
            "Dc": float(execution["phase3_backend_w_depth"]),
        },
    }
```

The branch price itself:

`pipelines/static_adapt/sr_snake/_controller.py:1309`

```python
def _branch_compiled_cost(branch: Any) -> float:
    """Total weighted FakeMarrakesh cost of one branch's accepted prefix.

    This is a branch-versus-branch quantity, not the Phase-III per-record
    factor: it prices the whole compiled prefix a branch has built, so a beam
    that buys the same energy with a cheaper circuit is preferred.
    """

    runtime = branch.runtime
    cursor = getattr(runtime, "cursor", None)
    operators = list(getattr(cursor, "selected_ops", ()) or ())
    if not operators:
        return 0.0
    settings = _branch_compiled_cost_contract(runtime)
    key = (
        settings["backend"],
        str(settings["optimization_level"]),
        str(settings["seed_transpiler"]),
        *(str(op.label) for op in operators),
    )
    cached = _BRANCH_COMPILED_COST_CACHE.get(key)
    if cached is not None:
        return cached
    from pipelines.hardcoded.adapt_circuit_execution import (
        build_structural_ansatz_circuit,
    )
    from pipelines.qiskit_backend_tools import (
        compile_circuit_for_backend,
        compiled_gate_stats,
        resolve_backend_targets,
        safe_circuit_depth,
    )

    context = getattr(runtime, "context", None)
    reference_state = np.asarray(
        getattr(context, "reference_state", None), dtype=complex
    ).reshape(-1)
    num_qubits = int(round(math.log2(reference_state.size)))
    targets, _audit = resolve_backend_targets(
        requested_names=(settings["backend"],),
        preferred_fake_backends=(settings["backend"],),
        allow_preferred_fallback=False,
        fallback_mode="single",
        allow_runtime_lookup=False,
    )
    _layout, circuit = build_structural_ansatz_circuit(
        operators,
        nq=num_qubits,
        ref_state=reference_state,
        structure_theta_value=1.0,
    )
    compiled = compile_circuit_for_backend(
        circuit,
        targets[0].backend_obj,
        seed_transpiler=settings["seed_transpiler"],
        optimization_level=settings["optimization_level"],
        initial_layout=None,
    )["compiled"]
    stats = compiled_gate_stats(compiled)
    weights = settings["weights"]
    total = (
        weights["N2q"] * float(stats["compiled_count_2q"])
        + weights["D2q"] * float(stats["compiled_depth_2q"])
        + weights["Dc"] * float(safe_circuit_depth(compiled))
    )
    _BRANCH_COMPILED_COST_CACHE[key] = total
    return total
```

The relative factor across siblings:

`pipelines/static_adapt/sr_snake/_controller.py:1378`

```python
def _branch_relative_cost_factors(costs: list[float]) -> list[float]:
    """Paper-I K construction applied across sibling branches.

    Median-centred, MAD-scaled, squashed through arctan into [-1, 1], then
    K = 2 / (2 - Cbar): unity at the median branch, 2/3 for the cheapest and
    2 for the dearest, so the factor discounts cheap branches and penalises
    expensive ones exactly as the phase scores do for records.
    """

    if not costs:
        return []
    ordered = sorted(costs)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    deviations = sorted(abs(value - median) for value in costs)
    mid = len(deviations) // 2
    scale = (
        deviations[mid]
        if len(deviations) % 2
        else 0.5 * (deviations[mid - 1] + deviations[mid])
    )
    if not math.isfinite(scale) or scale <= 0.0:
        # Siblings tie often -- many generators compile to the same cost --
        # and a majority tie drives the median absolute deviation to zero.
        # Falling straight through to "no discrimination" would silently
        # switch the whole cost comparison off for that round, so fall back
        # to the mean absolute deviation, which only vanishes when every
        # branch really does cost the same.
        scale = sum(deviations) / len(deviations)
    if not math.isfinite(scale) or scale <= 0.0:
        return [1.0] * len(costs)
    factors = []
    for value in costs:
        c_bar = (2.0 / math.pi) * math.atan((value - median) / scale)
        factors.append(2.0 / (2.0 - c_bar))
    return factors
```

### 3.3 Mathematical reading

For a branch `b` with accepted prefix `P_b`:

$$W_b = w_{2q} N_{2q}(P_b) + w_{D_{2q}} D_{2q}(P_b) + w_D D_c(P_b)$$

Let `S` be the set of sibling branches live in the round, `m = median{W_b}`
and `s = MAD{W_b}` (falling back to the **mean** absolute deviation when the
MAD vanishes — see (M2)). Then

$$\bar C_b = \frac{2}{\pi} \arctan\!\left( \frac{W_b - m}{s} \right), \qquad K_b = \frac{2}{2 - \bar C_b}$$

and the branch is ranked by `argmin` of `-(G_b / K_b)` where

$$G_b = E_{\rm init} - E_b$$

is the descent from the **run's initial state**, not from the branch's own
parent.

Three points the manuscript should carry:

1. **The population is the sibling set, not the record set.** Phase III ranks
   records against records; the beam ranks whole branches against whole
   branches, using the same `K` construction one level up. `W_b` is the cost of
   the entire accepted prefix, not a marginal.
2. **Descent is measured from the common root.** Siblings need not share a
   parent (up to 4 parents are live), so a per-parent gain would flatter the
   children of the weakest lineages. The manuscript's `E_root` is correct; the
   code uses the run initial state, which is the common ancestor of all
   branches.
3. **Pareto removal is on `(E_b, W_b)`** and happens before ranking.

### 3.4 Divergences

**(M1) The manuscript's denominator was `1 + ΔK_b`, an *added* burden.** The
code uses `K_b` from the *total* prefix burden. These are different objects:
one is marginal, one is cumulative, and the normalisation differs. The author
has already changed the prose toward `Φ_b = (E_root − E_b)/K_b`; confirm the
final wording says total-prefix burden over the sibling population.

**(M2) MAD fallback is undocumented.** Generators frequently compile to
identical costs, so a tied majority drives `MAD = 0`. The code then falls back
to the mean absolute deviation, and only if *that* also vanishes does it return
`K = 1` for every branch. Without this the cost ranking silently switches off
for the round. This is a real algorithmic detail and belongs in the text.

**(M3) Pareto filtering is far more aggressive than the prose implies.**
Measured over 15 rounds: 49 of 75 executed children were dominated, and the
live frontier collapsed to 1–3 branches rather than the configured 4. Effect
on the delivered ansatz is ~2% (111/81/330 filtered vs 109/77/319 unfiltered);
effect on the search is 2.2x (75 branches / 52,196 all-branch estimator queries
vs 168 / 115,082). **The reported `E` column is branch-local and therefore
excludes exactly what the filter removes**, so the saving is invisible in the
tuples. If the manuscript claims the filter saves work, it must say where that
saving is visible, because it is not in the reported column.

**(M4) The sibling-diversity constraint is absent from the manuscript.** The
route contract carries `beam_sibling_diversity_policy` and each child carries a
`sibling_diversity_receipt` constraining which pool indices siblings may take,
so a parent cannot spend its children re-exploring one direction. This is a
real restriction on the search and is currently undescribed.

**(M5) Three widths, not one.** The prose says a parent "can spawn `N`
children" from a top-`N` list, conflating the proposal-list length with
children-per-parent. The code has `live_parent_branches=4`,
`admission_children_per_parent=3`, `maximum_admission_children_per_round=12` as
independent knobs.

---

## 4. Generator ablation (`app:ablation`)

### 4.1 What the manuscript claims

Nominate `d* ∈ argmin_{d ∈ D_k} Λ(d)/K(d)` subject to `Λ(d*)/K(d*) <= ε`,
with `K(d)` "the same normalized circuit/shot-cost denominator used in the RA
candidate scores, evaluated over the current ansatz-entry population", and the
ratio "clipped to `[0,1]` before nomination". Then delete, refit, and retain iff

$$\Delta E(d^\star) \le \min\{\epsilon,\ \zeta [E(U_{k-1}) - E(U_k)]\}$$

### 4.2 What the code runs

Eligibility and the amplitude screen,
`pipelines/static_adapt/adapt_pipeline.py`:

```python
    eligible_indices = [
        index
        for index, admission_round in enumerate(
            numerical_runtime.operator_admission_rounds
        )
        if int(current_round - int(admission_round))
        >= int(config.protect_steps)
    ]
    ...
    if config.small_theta_abs is not None:
        threshold = float(config.small_theta_abs)
        screened = [
            index
            for index in eligible_indices
            if abs(float(logical_theta[index])) < threshold
        ]
        if len(screened) != len(eligible_indices):
            eligible_indices = screened
```

The surrogate and the nomination:

```python
        metric_rows = build_metric_regularized_prune_surrogate_scores(
            theta=np.asarray(logical_theta, dtype=float),
            labels=list(operator_ids),
            hessian=np.asarray(scaffold.H_window_hessian, dtype=float),
            metric=np.asarray(scaffold.Q_window, dtype=float),
            gradient=gradient,
            local_window_size=0,
            ridge=1.0e-6,
            metric_mu=float(config.metric_mu),
            solve_mode=str(config.metric_solve_mode),
            monotonicity_tol=1.0e-10,
            recovery_trust_radius=float(trust_before.radius),
            entry_cost_denominators=(
                _prune_entry_cost_denominators
                if _prune_entry_cost_denominators is not None
                else np.ones(logical_count, dtype=float)
            ),
        )
        ...
        metric_nominee = min(
            eligible_metric_rows,
            key=lambda row: (
                float(row["score"]),
                -float(row.get("entry_cost_denominator", 1.0)),
                int(row["index"]),
            ),
        )
```

The surrogate itself:

`pipelines/scaffold/hh_continuation_pruning.py:1773`

```python
def build_metric_regularized_prune_surrogate_scores(
    *,
    theta: Sequence[float] | np.ndarray,
    labels: Sequence[str],
    hessian: Sequence[Sequence[float]] | np.ndarray,
    metric: Sequence[Sequence[float]] | np.ndarray,
    gradient: Sequence[float] | np.ndarray | None = None,
    local_window_size: int = 4,
    ridge: float = 1e-6,
    metric_mu: float = 1e-6,
    solve_mode: str = PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
    monotonicity_tol: float = 1e-10,
    recovery_trust_radius: float = 0.0,
    entry_cost_denominators: Sequence[float] | np.ndarray | None = None,
) -> dict[int, dict[str, Any]]:
    """Build scalar cost-weighted metric-Schur prune nomination rows."""

    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    labels_list = [str(x) for x in labels]
    n = int(theta_vec.size)
    if len(labels_list) != n:
        raise ValueError("labels must match theta")
    H = np.asarray(hessian, dtype=float)
    G = np.asarray(metric, dtype=float)
    if H.shape != (n, n):
        raise ValueError("hessian shape must match theta")
    if G.shape != (n, n):
        raise ValueError("metric shape must match theta")
    gradient_vec: np.ndarray | None = None
    if gradient is not None:
        gradient_vec = np.asarray(gradient, dtype=float).reshape(-1)
        if int(gradient_vec.size) != n:
            raise ValueError("gradient must match theta")
    denominators = _sanitize_entry_cost_denominators(entry_cost_denominators, n=n)
    out: dict[int, dict[str, Any]] = {}
    for idx in range(n):
        survivor = [j for j in range(n) if int(j) != int(idx)]
        local_window_raw = int(local_window_size)
        if local_window_raw <= 0:
            local = list(survivor)
        else:
            omega = int(max(0, min(local_window_raw, len(survivor))))
            if omega <= 0:
                local = []
            else:
                start = max(0, min(int(idx) - (omega - 1) // 2, len(survivor) - omega))
                local = survivor[start : start + omega]
        windows = [[], list(local), survivor]
        ladder = metric_regularized_prune_schur_surrogate_ladder(
            theta=theta_vec,
            hessian=H,
            metric=G,
            block_indices=[int(idx)],
            windows=windows,
            ridge=float(ridge),
            metric_mu=float(metric_mu),
            solve_mode=str(solve_mode),
            monotonicity_tol=float(monotonicity_tol),
            recovery_trust_radius=float(recovery_trust_radius),
            gradient=gradient_vec,
        )
        values = [float(x) for x in ladder.get("values", [])]
        bounded_values = [float(x) for x in ladder.get("bounded_values", [])]
        bounded_active = bool(ladder.get("bounded_recovery_active", False))
        schur_score = float(min(values)) if values else float("inf")
        bounded_score = float(min(bounded_values)) if bounded_values else float(schur_score)
        unweighted_score = float(bounded_score if bounded_active else schur_score)
        denominator = float(denominators[int(idx)])
        score = float(unweighted_score / max(1.0, denominator))
        out[int(idx)] = {
            "index": int(idx),
            "label": str(labels_list[int(idx)]),
            "score": float(score),
            "unweighted_score": float(unweighted_score),
            "schur_min": float(schur_score),
            "bounded_score": float(bounded_score),
            "entry_cost_denominator": float(denominator),
            "cost_weighting": PRUNE_METRIC_COST_WEIGHT_ANSATZ_ENTRY_DENOMINATOR_V1,
            "bounded_recovery_active": bool(bounded_active),
            "recovery_trust_radius": float(ladder.get("recovery_trust_radius", 0.0)),
            "metric_mu": float(ladder.get("metric_mu", metric_mu)),
            "metric_schur_solve_mode": str(ladder.get("metric_schur_solve_mode", solve_mode)),
            "schur_rows": list(ladder.get("rows", [])),
            "schur_health": str(ladder.get("health", "unavailable")),
            "schur_monotone": bool(ladder.get("monotone", False)),
            "schur_model": PRUNE_SCHUR_ROUTE_METRIC_REGULARIZED_V1,
            "surrogate_authority": "metric_regularized_rank_window_diag_only",
            "used_for_acceptance": False,
        }
    return out
```

The denominator sanitizer — note the `max(1.0, val)` floor, which drives (P2):

`pipelines/scaffold/hh_continuation_pruning.py:1476`

```python
def _sanitize_entry_cost_denominators(
    values: Sequence[float] | np.ndarray | None,
    *,
    n: int,
) -> list[float]:
    if values is None:
        return [1.0 for _ in range(int(n))]
    out: list[float] = []
    for raw in list(values)[: int(n)]:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = 1.0
        if not math.isfinite(val) or val <= 0.0:
            val = 1.0
        out.append(float(max(1.0, val)))
    while len(out) < int(n):
        out.append(1.0)
    return out
```

The per-entry cost, added at commit `f56f8b58`:

`pipelines/static_adapt/adapt_pipeline.py:8392`

```python
def _ansatz_prefix_compiled_cost(
    operators: Sequence[Any],
    *,
    reference_state: np.ndarray,
    settings: Mapping[str, Any],
) -> float:
    """Three-term compiled cost of one ansatz prefix on the cost backend."""

    if not operators:
        return 0.0
    key = (
        str(settings["backend"]),
        str(settings["optimization_level"]),
        str(settings["seed_transpiler"]),
        *(str(op.label) for op in operators),
    )
    cached = _ANSATZ_ENTRY_COST_CACHE.get(key)
    if cached is not None:
        return cached
    from pipelines.hardcoded.adapt_circuit_execution import (
        build_structural_ansatz_circuit,
    )
    from pipelines.qiskit_backend_tools import (
        compile_circuit_for_backend,
        compiled_gate_stats,
        resolve_backend_targets,
        safe_circuit_depth,
    )

    state = np.asarray(reference_state, dtype=complex).reshape(-1)
    targets, _audit = resolve_backend_targets(
        requested_names=(settings["backend"],),
        preferred_fake_backends=(settings["backend"],),
        allow_preferred_fallback=False,
        fallback_mode="single",
        allow_runtime_lookup=False,
    )
    _layout, circuit = build_structural_ansatz_circuit(
        list(operators),
        nq=int(round(math.log2(state.size))),
        ref_state=state,
        structure_theta_value=1.0,
    )
    compiled = compile_circuit_for_backend(
        circuit,
        targets[0].backend_obj,
        seed_transpiler=int(settings["seed_transpiler"]),
        optimization_level=int(settings["optimization_level"]),
        initial_layout=None,
    )["compiled"]
    stats = compiled_gate_stats(compiled)
    weights = settings["weights"]
    total = (
        weights["N2q"] * float(stats["compiled_count_2q"])
        + weights["D2q"] * float(stats["compiled_depth_2q"])
        + weights["Dc"] * float(safe_circuit_depth(compiled))
    )
    _ANSATZ_ENTRY_COST_CACHE[key] = total
    return total
```

`pipelines/static_adapt/adapt_pipeline.py:8453`

```python
def _ansatz_entry_cost_denominators(
    operators: Sequence[Any],
    *,
    reference_state: np.ndarray,
    settings: Mapping[str, Any] | None,
) -> list[float] | None:
    """Relative resource-cost factor per active ansatz entry.

    An entry is priced by the compiled burden its deletion would relieve --
    the three-term cost of the full prefix minus the cost of the prefix
    without it -- and the median and scale are taken over the whole active
    ansatz, so the factor ranks each entry against the population it sits in.
    Returns ``None`` when the route declares no compile-cost backend, or when
    fewer than two entries are active, leaving nomination on the unweighted
    surrogate.
    """

    if settings is None or len(operators) < 2:
        return None
    ordered = list(operators)
    full = _ansatz_prefix_compiled_cost(
        ordered, reference_state=reference_state, settings=settings
    )
    relieved = [
        full
        - _ansatz_prefix_compiled_cost(
            ordered[:index] + ordered[index + 1 :],
            reference_state=reference_state,
            settings=settings,
        )
        for index in range(len(ordered))
    ]
    from pipelines.static_adapt.sr_snake._controller import (
        _branch_relative_cost_factors,
    )

    factors = _branch_relative_cost_factors(relieved)
    # K lies strictly inside (2/3, 2); the surrogate's denominator sanitizer
    # floors at one, which would erase every below-median entry.  Scaling by
    # the constant 3/2 lifts the whole range into (1, 3) without touching the
    # ordering, so argmin over Lambda/K is unchanged.
    return [1.5 * float(value) for value in factors]
```

`pipelines/static_adapt/adapt_pipeline.py:56470`

```python
def _prune_entry_cost_settings(
    typed_contract: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Lift the route's compile-cost identity for prune nomination."""

    execution = (
        typed_contract.get("execution_settings")
        if isinstance(typed_contract, Mapping)
        else None
    )
    if not isinstance(execution, Mapping):
        return None
    backend = execution.get("phase3_backend_name")
    if not backend:
        return None
    return {
        "backend": str(backend),
        "optimization_level": int(
            execution["phase3_backend_optimization_level"]
        ),
        "seed_transpiler": int(execution["phase3_backend_transpile_seed"]),
        "weights": {
            "N2q": float(execution["phase3_backend_w_2q"]),
            "D2q": float(execution["phase3_backend_w_depth_2q"]),
            "Dc": float(execution["phase3_backend_w_depth"]),
        },
    }
```

### 4.3 Mathematical reading

For active entry `p` at position `p` in the ansatz, `Λ(d)` is the
metric-regularised Schur deletion surrogate evaluated at `β = −θ*_p`, minimised
over a ladder of recovery windows:

$$\Lambda(d) = \min_{\text{windows}} \left[ g(d)\beta + \frac{\beta^2}{2} \left( \mathcal Q_{\beta\beta} - \mathcal Q_{\beta\theta} \mathcal Q_{\theta\theta}^{+} \mathcal Q_{\theta\beta} \right) \right]$$

This is the predicted energy **rise** from removing `d` after the surviving
coordinates relax to compensate. `Λ = 0` means the deletion is predicted free —
the generator is redundant — which is exactly the generator you want to delete,
hence `argmin`.

The cost side: entry `p` is priced by the burden its deletion would relieve,

$$\Delta W_p = W(P) - W(P \setminus p)$$

with `W` the same three-term compiled cost as the beam, and the statistics
(median, scale) taken over the **whole active ansatz**. Then `K_p` follows the
same `2/(2 - C̄)` construction, and the executed denominator is `1.5 K_p`
(see P2). Nomination is `argmin` over the screened eligible set of
`Λ(d) / (1.5 K_d)`, ties broken by larger relieved burden, then by index.

Eligible set:

$$\mathcal D_k = \{p : k - k_{\rm adm}(p) \ge n_{\rm protect}\ \text{and}\ |\theta^\star_p| < \theta_{\rm screen}\}$$

with the reported arm at `n_protect = 1`, `θ_screen = 1e-3` (strict `<`).

### 4.4 Divergences

**(P1) Until commit `f56f8b58`, `K(d) ≡ 1`.** The call passed
`entry_cost_denominators=np.ones(logical_count)`, so nomination ranked by
`Λ(d)` alone. **Every prune number in v8 and in the earlier v9 cells was
produced this way.** The current code computes the cost properly. This is
*not* the `K_t = 1` situation of §1 — there the factor is computed and 1 is its
median-neutral value; here the cost term was never computed at all.

**(P2) The denominator is scaled by 3/2 and the manuscript does not say so.**
`_sanitize_entry_cost_denominators` floors every value at 1.0. Since
`K ∈ (2/3, 2)`, passing `K` directly would clamp the entire below-median half
to 1 and erase it. The code therefore passes `1.5 K ∈ (1, 3)`. Because this is
a single positive constant it leaves the `argmin` unchanged, but it means the
recorded `predicted_energy_change` is `Λ/(1.5K)`, i.e. **2/3 of the manuscript's
`Λ/K`**. Either document the scaling or divide it back out before recording.

**(P3) The `[0,1]` clip is claimed but I could not find it.** The manuscript
says `Λ(d)/K(d)` is clipped to `[0,1]` before nomination. I did not locate any
such clip. Either it does not exist (delete the sentence) or it lives somewhere
I did not read (find it and cite the line). **Do not assume either way.**

**(P4) The cost term is wired but not demonstrably deciding anything.** On this
problem the accepted deletions all occur at `Λ = 0` exactly — the depth-8
saturation means the admitted generators from round 9 onward carry angles
`~1e-9` and are genuinely free to remove. But `Λ/K = 0` for **every** `K`, so
the ratio cannot rank them; the cost only enters through the tie-break I added.
And the amplitude screen may be leaving a single eligible entry, in which case
even the tie-break never fires.

Measured: enabling the cost changed nothing (112/93/377, 3 accepted, 0 rejected,
identical deleted labels with and without the tie-break).

**The defensible claim today is "nomination screened by amplitude, ranked by
predicted deletion loss, ties broken on relieved compiled burden" — not
"cost-weighted deletion".** The eligible-set size is not recorded in the
receipt, so this cannot currently be settled from artifacts. Recording it plus
the per-entry denominators would settle it on one run.

**(P5) The reported arm no longer shows the ungated comparison.** Both screen
settings were measured: ungated (`n_protect=2`, no angle screen) produced
**8 nominations that were measured and rejected** against 5 accepted; gated
produced **0 rejected** and 6 accepted, with the first deletion a round earlier.
The gated arm strictly dominates here. Only the gated arm is plotted now, so if
the prose motivates the screen it should cite these numbers rather than argue
from first principles.

**(P6) Ablation does not improve the accuracy round.** In v9 the ablation arm
and RA-append cross `1e-6` at the same round with **identical** resources there
(104, 88, 363, 374, 202, 4124), because the first accepted deletion lands at
round 10, two rounds after the crossing. The benefit is entirely
post-convergence circuit reduction (112/93/378 vs 127/109/441 at k=15). Any
sentence implying earlier convergence is false.

---

## 5. Divergence summary

| id | Control | Divergence | Resolution options |
|---|---|---|---|
| B1 | batch | window is a feasibility-ranked child prefix, not top-`W` of `R^(3)` | prose |
| B2 | batch | "greedy is the practical route" — executed arm is exhaustive-in-window | prose |
| B3 | batch | pairwise condition may be a joint-metric rank gate, not disjoint+commuting | **investigate first** |
| M1 | beam | `1 + ΔK_b` (marginal) vs `K_b` (total prefix) | prose (in progress) |
| M2 | beam | MAD→mean-abs-dev fallback undocumented | prose |
| M3 | beam | Pareto saving invisible in the branch-local `E` column | prose |
| M4 | beam | sibling-diversity policy undocumented | prose |
| M5 | beam | three independent widths described as one | prose |
| P1 | ablation | `K ≡ 1` before `f56f8b58`; older numbers are uncost-weighted | prose + note which numbers |
| P2 | ablation | denominator scaled by 3/2 to clear the sanitizer floor | prose or code |
| P3 | ablation | `[0,1]` clip claimed, not found | **investigate first** |
| P4 | ablation | cost wired but not demonstrably decisive | weaken claim, or instrument |
| P5 | ablation | ungated comparison measured but no longer plotted | prose |
| P6 | ablation | no accuracy-round benefit | **prose is currently wrong** |

Three of these (B3, P3, P4) are open questions about what the code does, not
stylistic choices. They should be closed before the corresponding sentences are
written, not after.

---

## 6. Numbers of record

From `ra_appendix_l3_v9_provenance.json`, campaign
`ra_adapt_appendix_demos_l3_20260820_v9`, horizon 15.

At each route's first prefix with `|E_k − E_ED| <= 1e-6`:

| curve | round | N2q | D2q | Dc | N1q | W1q | E |
|---|---|---|---|---|---|---|---|
| AAVQE | 8 | 216 | 204 | 837 | 741 | 382 | 6510 |
| RA-append | 8 | 104 | 88 | 363 | 374 | 202 | 4124 |
| + ablation (gated) | 8 | 104 | 88 | 363 | 374 | 202 | 4124 |
| + beam (Pareto) | 8 | 88 | 71 | 298 | 331 | 202 | 4508 |
| + batch | 3 | 200 | 188 | 776 | 690 | 382 | 2683 |

At the horizon `k=15`:

| curve | N2q | D2q | Dc | E | err |
|---|---|---|---|---|---|
| AAVQE | 508 | 493 | 2024 | 8522 | 1.64e-14 |
| RA-append | 127 | 109 | 441 | 9547 | 7.11e-15 |
| + ablation (gated) | 112 | 93 | 378 | 10658 | 8.88e-16 |
| + beam (Pareto) | 111 | 81 | 330 | 10185 | 2.22e-15 |
| + batch | 229 | 206 | 853 | 11823 | 8.44e-15 |

`E` is branch-local throughout: on the beam it counts the winning lineage only.
Accepted deletions for the ablation arm occur at rounds 10–15.

---

## 7. What I would not write without checking

- Any statement that batch admission is greedy.
- Any statement that ablation converges earlier than RA-append.
- Any statement that the prune nomination is cost-weighted in a way that
  changed which generator was deleted.
- Any statement that the `[0,1]` clip or the disjoint-support/commuting pair
  condition is implemented, until B3 and P3 are closed.
- Any statement that the Pareto filter's estimator saving appears in the
  reported `E` column.

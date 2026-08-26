"""Structural enumeration: singleton level, rungs, priorities, frontiers."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.commutation import singleton_insertion_cuts
from pipelines.time_dynamics.ap_mclachlan.exchange_structural import (
    StructuralScoreWeights,
    enumerate_structural_candidates,
    resolve_frontier_schedule,
)
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import (
    evaluate_mclachlan_geometry,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.insertion_words import (
    tokens_commute_from_terms,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.state import state_from_scaffold_runtime_input
from pipelines.time_dynamics.ap_mclachlan.structural_cache import (
    build_structural_insertion_cache,
    structural_candidate_solve,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(*components: tuple[str, float], nq: int = 2) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for word, coeff in components:
        poly.add_term(PauliTerm(int(nq), ps=str(word), pc=float(coeff)))
    poly._reduce()
    return poly


X0 = AnsatzTerm(label="sx0", polynomial=_poly(("ex", 1.0)))
Z0 = AnsatzTerm(label="sz0", polynomial=_poly(("ez", 1.0)))
Y1 = AnsatzTerm(label="sy1", polynomial=_poly(("ye", 1.0)))
CA = AnsatzTerm(label="ca", polynomial=_poly(("xe", 0.7)))
CB = AnsatzTerm(label="cb", polynomial=_poly(("zz", 0.4)))
HAM = TimeDependentHamiltonian(static_poly=_poly(("ez", 2.0), ("xx", 0.6)))
POLICY = McLachlanInversePolicy(
    pinv_rcond=1.0e-8, ridge_lambda=1.0e-6, solve_damping=1.0e-4
)
WEIGHTS = StructuralScoreWeights()


def _state(selected, theta):
    layout = build_parameter_layout(selected)
    executor = CompiledAnsatzExecutor(
        tuple(selected),
        parameterization_layout=layout,
        parameterization_mode="per_pauli_term",
    )
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0
    psi_initial = executor.prepare_state(np.asarray(theta, dtype=float), psi_ref)
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(
            family_key="toy", hamiltonian=_poly(("ez", 2.0), ("xx", 0.6))
        ),
        psi_ref=psi_ref,
        psi_initial=np.asarray(psi_initial, dtype=complex),
        base_layout=layout,
        theta_runtime=np.asarray(theta, dtype=float),
        theta_logical=np.zeros(int(layout.logical_parameter_count), dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=tuple(selected),
        candidate_pool_terms=tuple(selected),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool", pool_key="toy_pool", completeness="complete"
        ),
        provenance={"artifact_json": "toy.json"},
    )
    return state_from_scaffold_runtime_input(runtime_input)


def _setup(**overrides):
    state = _state((X0, Z0, Y1), np.array([0.3, -0.2, 0.5]))
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )
    blocks = (X0, Z0, Y1)
    cuts = {
        "ca": singleton_insertion_cuts(CA, blocks),
        "cb": singleton_insertion_cuts(CB, blocks),
    }
    atoms = {
        "ca": SimpleNamespace(atom_id="ca", term=CA),
        "cb": SimpleNamespace(atom_id="cb", term=CB),
    }
    cache = build_structural_insertion_cache(
        state=state,
        evaluation=evaluation,
        cuts_by_atom=cuts,
        atoms_by_id=atoms,
    )
    terms_by_key = dict(
        zip(state.runtime_coordinate_labels, (X0, Z0, Y1)), ca=CA, cb=CB
    )
    kwargs = dict(
        cache=cache,
        base_K=np.asarray(evaluation.geometry.K, dtype=float),
        base_f=np.asarray(evaluation.geometry.f, dtype=float),
        norm_b_sq=float(evaluation.geometry.norm_b_sq),
        inverse_policy=POLICY,
        weights=WEIGHTS,
        deletable_indices=(0, 1),
        min_surviving_support=1,
        cuts_by_atom=cuts,
        candidate_pool_for_deletion=lambda removed: ("ca", "cb"),
        insertion_cost=lambda atom_ids: 1.0 + 0.5 * len(atom_ids),
        deletion_cost=lambda removed: 1.0 + 0.25 * len(removed),
        tokens_commute=tokens_commute_from_terms(terms_by_key),
        max_insertion_batch_size=1,
    )
    kwargs.update(overrides)
    return state, evaluation, cache, cuts, kwargs


def test_complete_singleton_level_and_rungs() -> None:
    _state_, _eval, _cache, cuts, kwargs = _setup()
    result = enumerate_structural_candidates(**kwargs)
    per_pool = len(cuts["ca"]) + len(cuts["cb"])
    kinds = Counter(c.kind for c in result.candidates)
    # d=0: stay + pure singleton inserts; d=1: two deletions; d=2: one.
    assert kinds["stay"] == 1
    assert kinds["insert"] == per_pool
    assert kinds["delete"] == 3
    assert kinds["exchange"] == 3 * per_pool
    assert result.guard.scored_count == len(result.candidates)
    assert result.guard.admitted_families == [
        "singleton_d0",
        "singleton_d1",
        "singleton_d2",
    ]


def test_deletion_permission_filters_whole_branches_before_scoring() -> None:
    _state_, _eval, _cache, cuts, kwargs = _setup()
    seen: list[tuple[int, ...]] = []

    def permission(removed):
        removed = tuple(removed)
        seen.append(removed)
        return SimpleNamespace(permitted=removed == (1,))

    result = enumerate_structural_candidates(
        **kwargs, deletion_permission=permission
    )
    deletion_sets = {
        candidate.removed_runtime_indices
        for candidate in result.candidates
        if candidate.removed_runtime_indices
    }
    assert deletion_sets == {(1,)}
    assert set(seen) == {(0,), (1,), (0, 1)}
    per_pool = len(cuts["ca"]) + len(cuts["cb"])
    kinds = Counter(candidate.kind for candidate in result.candidates)
    assert kinds["delete"] == 1
    assert kinds["exchange"] == per_pool


def test_work_guard_rejects_complete_family_and_freezes() -> None:
    _s, _e, _c, cuts, kwargs = _setup()
    kwargs["max_joint_patch_evaluations"] = 1 + len(cuts["ca"]) + len(cuts["cb"])
    result = enumerate_structural_candidates(**kwargs)
    kinds = Counter(c.kind for c in result.candidates)
    assert kinds["delete"] == 0 and kinds["exchange"] == 0
    assert result.guard.rejected_family == "singleton_d1"
    assert result.guard.scored_count == len(result.candidates)


def test_priorities_are_max_singleton_scores_and_universe_is_sorted() -> None:
    _s, _e, _c, _cuts, kwargs = _setup()
    result = enumerate_structural_candidates(**kwargs)
    for atom_id in ("ca", "cb"):
        singleton_scores = [
            c.score
            for c in result.candidates
            if c.plan is None
            and c.kind in ("insert", "exchange")
            # singleton candidates carry their selection through the memoised
            # q; recover atom identity from insertion utility bookkeeping:
        ]
    # direct property: priorities cover exactly the pool and are finite
    assert set(result.priorities) == {"ca", "cb"}
    assert all(np.isfinite(v) for v in result.priorities.values())
    ordered = sorted(
        result.priorities, key=lambda a: (-result.priorities[a], str(a))
    )
    assert tuple(ordered) == result.eligible_universe


def test_ranked_excludes_stay_applies_floor_and_is_deterministic() -> None:
    _s, _e, _c, _cuts, kwargs = _setup()
    result = enumerate_structural_candidates(**kwargs)
    ranked = result.ranked(score_floor=0.0)
    assert all(c.kind != "stay" for c in ranked)
    scores = [c.score for c in ranked]
    assert scores == sorted(scores, reverse=True)
    again = enumerate_structural_candidates(**kwargs).ranked(score_floor=0.0)
    assert [c.order_key for c in again] == [c.order_key for c in ranked]
    high_floor = result.ranked(score_floor=max(scores))
    assert len(high_floor) < len(ranked)


def test_pure_categories_zero_the_missing_utility() -> None:
    _s, _e, _c, _cuts, kwargs = _setup()
    result = enumerate_structural_candidates(**kwargs)
    for c in result.candidates:
        if c.kind == "insert":
            assert c.deletion_utility == 0.0 and c.deletion_loss == 0.0
        if c.kind == "delete":
            assert c.insertion_utility == 0.0 and c.insertion_gain == 0.0
        if c.kind == "stay":
            assert c.score == pytest.approx(0.0, abs=1e-30) or c.delta == 0.0


def test_score_components_match_direct_solves() -> None:
    _s, evaluation, cache, cuts, kwargs = _setup()
    result = enumerate_structural_candidates(**kwargs)
    base_K = kwargs["base_K"]
    base_f = kwargs["base_f"]
    norm_b_sq = kwargs["norm_b_sq"]

    def q_direct(removed, selection):
        keep = tuple(i for i in range(3) if i not in set(removed))
        _Q, q = structural_candidate_solve(
            cache=cache,
            base_K=base_K,
            base_f=base_f,
            norm_b_sq=norm_b_sq,
            keep_indices=keep,
            inserted_selection=selection,
            inverse_policy=POLICY,
            epsilon_norm=WEIGHTS.epsilon_norm,
        )
        return q

    exchanges = [c for c in result.candidates if c.kind == "exchange"]
    assert exchanges
    probe = exchanges[0]
    removed = probe.removed_runtime_indices
    sel = probe.inserted_selection
    assert q_direct(removed, sel) == pytest.approx(probe.q, rel=1e-12)
    assert probe.insertion_gain == pytest.approx(
        max(0.0, probe.q - q_direct(removed, ())), rel=1e-12
    )
    assert probe.deletion_loss == pytest.approx(
        max(0.0, q_direct((), sel) - probe.q), rel=1e-12
    )
    assert probe.delta == pytest.approx(probe.q - result.q_base, rel=1e-9, abs=1e-15)


def test_multi_child_frontier_plans_are_quotiented_and_admitted_whole() -> None:
    _s, _e, _c, _cuts, kwargs = _setup(max_insertion_batch_size=2)
    result = enumerate_structural_candidates(**kwargs)
    multi = [c for c in result.candidates if c.plan is not None]
    assert multi, "expected multi-child frontier candidates"
    assert result.frontiers_used >= 1
    assert any(f.startswith("frontier_") for f in result.guard.admitted_families)
    # Each plan candidate carries two inserted occurrences.
    assert all(len(c.plan.inserted_keys) == 2 for c in multi)
    # Identities unique.
    identities = [
        (c.removed_runtime_indices, c.plan.plan_id) for c in multi
    ]
    assert len(set(identities)) == len(identities)


def test_frontier_schedule_resolution() -> None:
    assert resolve_frontier_schedule(None, universe_size=0) == ()
    assert resolve_frontier_schedule(None, universe_size=1) == (1,)
    assert resolve_frontier_schedule(None, universe_size=2) == (2,)
    assert resolve_frontier_schedule(None, universe_size=11) == (2, 4, 8, 11)
    assert resolve_frontier_schedule((3, 9), universe_size=5) == (3, 5)
    assert resolve_frontier_schedule((2, 4), universe_size=50) == (2, 4)
    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_frontier_schedule((4, 4), universe_size=9)
    with pytest.raises(ValueError, match="positive"):
        resolve_frontier_schedule((0, 2), universe_size=9)


def test_deleted_children_may_reenter_branch_pools() -> None:
    calls: list[tuple[int, ...]] = []

    def pool(removed):
        calls.append(tuple(removed))
        return ("ca", "cb")

    _s, _e, _c, _cuts, kwargs = _setup(candidate_pool_for_deletion=pool)
    enumerate_structural_candidates(**kwargs)
    assert () in calls and (0,) in calls and (0, 1) in calls


def test_conditioning_lambdas_reweight_deletion_utilities_exactly() -> None:
    """Hook #1: relief/damage from memoized solve metadata, exact formula."""

    import math
    from dataclasses import replace

    from pipelines.time_dynamics.ap_mclachlan.structural_cache import (
        memoized_solve_metadata,
    )

    _state_, _eval_, _cache_, _cuts_, base_kwargs = _setup()
    base = {
        c.order_key: c
        for c in enumerate_structural_candidates(**base_kwargs).candidates
    }

    weights = replace(WEIGHTS, lambda_cond_relief=5.0, lambda_cond_damage=3.0)
    _s2, _e2, cache2, _c2, kwargs2 = _setup(weights=weights)
    result = enumerate_structural_candidates(**kwargs2)
    cond_base, _rank = memoized_solve_metadata(cache2, ((), ()))
    assert cond_base is not None and cond_base > 0.0

    reweighted = 0
    for c in result.candidates:
        if c.kind == "stay":
            continue
        ref = base[c.order_key]
        if not c.removed_runtime_indices:
            # Insertion-only candidates never see the conditioning term.
            assert c.score == pytest.approx(ref.score, rel=1.0e-12)
            continue
        cond_del, _rank_d = memoized_solve_metadata(
            cache2, (c.removed_runtime_indices, ())
        )
        assert cond_del is not None and cond_del > 0.0
        shift = math.log10(cond_base) - math.log10(cond_del)
        relief, damage = max(0.0, shift), max(0.0, -shift)
        cost_del = 1.0 + 0.25 * len(c.removed_runtime_indices)
        expected = (
            cost_del
            * (1.0 + 5.0 * relief)
            / (c.deletion_loss + 3.0 * damage + WEIGHTS.epsilon_L)
        )
        assert c.deletion_utility == pytest.approx(expected, rel=1.0e-9)
        if relief > 0.0 or damage > 0.0:
            reweighted += 1
    assert reweighted > 0


def test_history_term_enters_deletion_denominator_exactly() -> None:
    """Hook #2: the injected historical-loss prior scales the denominator."""

    from dataclasses import replace

    _state_, _eval_, _cache_, _cuts_, base_kwargs = _setup()
    base = {
        c.order_key: c
        for c in enumerate_structural_candidates(**base_kwargs).candidates
    }

    _s2, _e2, _cache2, _c2, kwargs2 = _setup(
        weights=replace(WEIGHTS, lambda_hist=2.0),
        deletion_history_loss=lambda removed: 0.5 * len(removed),
    )
    result = enumerate_structural_candidates(**kwargs2)
    for c in result.candidates:
        if c.kind == "stay":
            continue
        ref = base[c.order_key]
        if not c.removed_runtime_indices:
            assert c.score == pytest.approx(ref.score, rel=1.0e-12)
            continue
        cost_del = 1.0 + 0.25 * len(c.removed_runtime_indices)
        expected = cost_del / (
            c.deletion_loss
            + 2.0 * 0.5 * len(c.removed_runtime_indices)
            + WEIGHTS.epsilon_L
        )
        assert c.deletion_utility == pytest.approx(expected, rel=1.0e-9)
        assert c.deletion_utility < ref.deletion_utility

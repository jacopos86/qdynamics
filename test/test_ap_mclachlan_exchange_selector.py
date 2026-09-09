"""Selection loop: level-by-level certification, atomic commit, stay fallback."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.commutation import singleton_insertion_cuts
from pipelines.time_dynamics.ap_mclachlan.exchange_certification import (
    CertificationGates,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_selector import (
    AttemptRecord,
    RankInfo,
    RankedLevel,
    candidate_insertions,
    select_exchange_patch,
)
from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import (
    StructuralCandidate,
    StructuralScoreWeights,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import solve_fixed_mclachlan_step
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import (
    evaluate_mclachlan_geometry,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.insertion_words import (
    inserted_tokens,
    quotient_insertion_plans,
    survivor_tokens,
    tokens_commute_from_terms,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.objective_ordinates import DEFAULT_GAMMA_PINV_RCOND
from pipelines.time_dynamics.ap_mclachlan.state import state_from_scaffold_runtime_input
from pipelines.time_dynamics.ap_mclachlan.structural_cache import (
    build_structural_insertion_cache,
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
POLICY = McLachlanInversePolicy(pinv_rcond=1.0e-10, ridge_lambda=1.0e-7)


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


def _selection_kwargs(theta=(0.05, -0.04, 0.06), **structural_overrides):
    state = _state((X0, Z0, Y1), np.array(theta))
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )
    step = solve_fixed_mclachlan_step(evaluation.geometry, inverse_policy=POLICY)
    blocks = (X0, Z0, Y1)
    cuts = {
        "ca": singleton_insertion_cuts(CA, blocks),
        "cb": singleton_insertion_cuts(CB, blocks),
    }
    atoms = {
        "ca": SimpleNamespace(atom_id="ca", term=CA, atom_label="ca"),
        "cb": SimpleNamespace(atom_id="cb", term=CB, atom_label="cb"),
    }
    cache = build_structural_insertion_cache(
        state=state,
        evaluation=evaluation,
        cuts_by_atom=structural_overrides.get("cuts_by_atom", cuts),
        atoms_by_id=atoms,
    )
    terms_by_key = dict(
        zip(state.runtime_coordinate_labels, (X0, Z0, Y1)), ca=CA, cb=CB
    )
    structural = dict(
        cache=cache,
        base_K=np.asarray(evaluation.geometry.K, dtype=float),
        base_f=np.asarray(evaluation.geometry.f, dtype=float),
        norm_b_sq=float(evaluation.geometry.norm_b_sq),
        inverse_policy=POLICY,
        weights=StructuralScoreWeights(),
        deletable_indices=(0, 1),
        min_surviving_support=1,
        cuts_by_atom=cuts,
        candidate_pool_for_deletion=lambda removed: ("ca", "cb"),
        insertion_cost=lambda atom_ids: 1.0 + 0.5 * len(atom_ids),
        deletion_cost=lambda removed: 1.0 + 0.25 * len(removed),
        tokens_commute=tokens_commute_from_terms(terms_by_key),
        max_insertion_batch_size=1,
    )
    structural.update(structural_overrides)

    pauli = {"ca": "xe", "cb": "zz"}

    def occurrence_label(atom, cut, ordinal):
        return f"{atom.atom_id}c{cut}o{ordinal}::r0::{pauli[atom.atom_id]}"

    return dict(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        base_evaluation=evaluation,
        base_step=step,
        inverse_policy=POLICY,
        atoms_by_id=atoms,
        occurrence_label=occurrence_label,
        structural_kwargs=structural,
    )


def test_commit_is_highest_certifiable_candidate_and_atomic() -> None:
    kwargs = _selection_kwargs()
    selection = select_exchange_patch(
        gates=CertificationGates(ray_distance_max=1.0, smoothness_eta_max=1e6),
        **kwargs,
    )
    assert selection.kind != "stay"
    assert selection.stop_reason == "committed"
    assert selection.certification is not None and selection.certification.certified
    # The committed candidate is the first attempt that certified; with loose
    # gates that is the very first ranked candidate.
    assert selection.attempts[-1].reason == "certified"
    assert selection.committed.score == max(a.score for a in selection.attempts)
    # Route objects are carried for the trajectory loop.
    assert selection.certification.state is not None
    assert selection.certification.step is not None
    assert selection.certification_work
    certified_work = selection.certification_work[-1]
    assert certified_work["reason"] == "certified"
    assert certified_work["schema"] == (
        "paper_ii_exchange_certification_estimator_work_v1"
    )
    assert certified_work["logical_full_geometry_entry_count"] > 0
    assert certified_work["complete"] is True


def test_batch_candidate_materializes_the_word_that_compiled_cost_priced() -> None:
    plans = quotient_insertion_plans(
        survivors=survivor_tokens(("s0", "s1")),
        inserted=inserted_tokens(("ca", "cb")),
        removed_keys=(),
        tokens_commute=lambda _left, _right: False,
    )
    candidate = StructuralCandidate(
        removed_runtime_indices=(),
        inserted_selection=(("ca", 0), ("cb", 2)),
        plan=plans[0],
        family="frontier_test",
        q=1.0,
        insertion_gain=1.0,
        deletion_loss=0.0,
        delta=1.0,
        insertion_utility=1.0,
        deletion_utility=0.0,
        score=1.0,
        placement_cuts=(("ca", 2), ("cb", 0)),
        patch_word_sha256="priced-word",
    )
    atoms = {
        "ca": SimpleNamespace(atom_id="ca", term=CA),
        "cb": SimpleNamespace(atom_id="cb", term=CB),
    }
    entries = candidate_insertions(
        candidate,
        atoms_by_id=atoms,
        occurrence_label=lambda atom, cut, ordinal: f"{atom.atom_id}:{cut}:{ordinal}",
    )
    assert [(cut, label) for cut, _term, label in entries] == [
        (2, "ca:2:0"),
        (0, "cb:0:1"),
    ]


def test_all_gates_failing_returns_stay_with_full_attempt_record() -> None:
    # Deletion-only pool: every candidate moves the ray (nonzero angles), so a
    # tight ray gate rejects them all and the selector returns stay.
    kwargs = _selection_kwargs(
        candidate_pool_for_deletion=lambda removed: (),
        cuts_by_atom={},
    )
    selection = select_exchange_patch(
        gates=CertificationGates(
            ray_distance_max=1.0e-15, smoothness_eta_max=1.0e-15
        ),
        **kwargs,
    )
    assert selection.kind == "stay"
    assert selection.committed is None
    assert selection.attempts, "every ranked candidate must have been attempted"
    assert all(a.reason == "ray_distance_above_max" for a in selection.attempts)
    assert {a.kind for a in selection.attempts} == {"delete"}
    assert selection.telemetry is not None
    assert selection.stop_reason == "level_exhausted"
    assert len(selection.certification_work) == len(selection.attempts)
    assert all(
        row["reason"] == "ray_distance_above_max"
        for row in selection.certification_work
    )


def test_failed_higher_score_advances_to_next_candidate() -> None:
    # Deletion-only pool with per-branch costs that rank the big-angle
    # deletion first; the ray gate sits between the two deletions' ray
    # displacements, so the top-ranked fails and the tiny-angle one passes.
    def deletion_cost(removed):
        return 1.0e15 if 0 in removed else 1.0

    kwargs = _selection_kwargs(
        theta=(0.3, 1.0e-6, 0.06),
        candidate_pool_for_deletion=lambda removed: (),
        cuts_by_atom={},
        deletion_cost=deletion_cost,
    )
    selection = select_exchange_patch(
        gates=CertificationGates(ray_distance_max=1.0e-3, smoothness_eta_max=1e6),
        **kwargs,
    )
    assert selection.kind == "delete"
    assert selection.committed.removed_runtime_indices == (1,)
    reasons = [a.reason for a in selection.attempts]
    assert reasons[0] == "ray_distance_above_max"
    assert reasons[-1] == "certified"
    failed_scores = [a.score for a in selection.attempts if a.reason != "certified"]
    assert max(failed_scores) > selection.committed.score


def test_realized_authority_rejection_advances_without_resealing_population() -> None:
    ranker_calls = 0

    def ranker(candidates):
        nonlocal ranker_calls
        ranker_calls += 1
        ordered = tuple(c for c in candidates if c.kind != "stay")[:2]
        return RankedLevel(
            ordered=ordered,
            rank_info={
                c.order_key: RankInfo(K=1.0, score=float(2 - index), merit=c.delta)
                for index, c in enumerate(ordered)
            },
            telemetry={"population_sha256": "sealed-once"},
        )

    authority_calls = 0

    def authority_gate(_result):
        nonlocal authority_calls
        authority_calls += 1
        return authority_calls > 1

    selection = select_exchange_patch(
        gates=CertificationGates(ray_distance_max=1.0, smoothness_eta_max=1e6),
        population_ranker=ranker,
        realized_l2_gate=authority_gate,
        realized_l2_gate_reason="realized_q_loss_budget_exceeded",
        **_selection_kwargs(),
    )
    assert selection.committed is not None
    assert [attempt.reason for attempt in selection.attempts[:2]] == [
        "realized_q_loss_budget_exceeded",
        "certified",
    ]
    assert ranker_calls == 1
    assert selection.level_telemetry == ({"population_sha256": "sealed-once"},)


def test_escalation_predicate_false_stops_after_first_level() -> None:
    kwargs = _selection_kwargs(
        candidate_pool_for_deletion=lambda removed: (),
        cuts_by_atom={},
    )
    selection = select_exchange_patch(
        gates=CertificationGates(
            ray_distance_max=1.0e-15, smoothness_eta_max=1.0e-15
        ),
        escalate=lambda: False,
        **kwargs,
    )
    assert selection.kind == "stay"
    assert selection.stop_reason == "escalation_predicate_false"
    # Every deletion rung belongs to the first certification level (deletions
    # always compete on score), so pure-deletion attempts exist even with the
    # escalation predicate false — it gates only insertion frontiers.
    assert all(a.kind == "delete" for a in selection.attempts)
    assert selection.attempts, "deletion rungs must be scored and attempted"
    assert selection.telemetry is not None


def test_score_floor_excludes_low_candidates_from_certification() -> None:
    kwargs = _selection_kwargs(
        candidate_pool_for_deletion=lambda removed: (),
        cuts_by_atom={},
    )
    gates = CertificationGates(ray_distance_max=1.0e-15, smoothness_eta_max=1.0e-15)
    loose = select_exchange_patch(gates=gates, **kwargs)
    floored = select_exchange_patch(
        gates=gates,
        score_floor=max(a.score for a in loose.attempts),
        **kwargs,
    )
    assert len(floored.attempts) < len(loose.attempts)


def test_selection_is_deterministic() -> None:
    kwargs_a = _selection_kwargs()
    kwargs_b = _selection_kwargs()
    gates = CertificationGates(ray_distance_max=1.0, smoothness_eta_max=1e6)
    a = select_exchange_patch(gates=gates, **kwargs_a)
    b = select_exchange_patch(gates=gates, **kwargs_b)
    assert [x.to_json_dict() for x in a.attempts] == [
        x.to_json_dict() for x in b.attempts
    ]
    assert a.committed.order_key == b.committed.order_key


def test_selection_parallel_matches_serial_attempts_and_commit() -> None:
    # Bitwise contract: scoring workers must not change any attempt, score,
    # order, or commit; floats compare with ==, never approx.
    scenarios = (
        dict(gates=CertificationGates(ray_distance_max=1.0, smoothness_eta_max=1e6)),
        dict(
            gates=CertificationGates(
                ray_distance_max=1.0e-15, smoothness_eta_max=1.0e-15
            ),
            structural=dict(
                candidate_pool_for_deletion=lambda removed: (),
                cuts_by_atom={},
            ),
        ),
    )
    for scenario in scenarios:
        results = []
        for workers in (1, 4):
            overrides = dict(scenario.get("structural", {}))
            overrides["scoring_workers"] = workers
            kwargs = _selection_kwargs(**overrides)
            selection = select_exchange_patch(
                gates=scenario["gates"], **kwargs
            )
            results.append(selection)
        serial, parallel = results
        assert [a.to_json_dict() for a in parallel.attempts] == [
            a.to_json_dict() for a in serial.attempts
        ]
        assert parallel.stop_reason == serial.stop_reason
        assert parallel.structural_scored_count == serial.structural_scored_count
        assert parallel.certification_work == serial.certification_work
        if serial.committed is None:
            assert parallel.committed is None
        else:
            assert parallel.committed.order_key == serial.committed.order_key
            assert parallel.committed.score == serial.committed.score


@pytest.mark.parametrize("stop_at_first_level", ["commit", "escalation_stop"])
def test_stopping_singleton_level_does_not_evaluate_joint_patches(
    monkeypatch, stop_at_first_level
) -> None:
    from pipelines.time_dynamics.ap_mclachlan import patch_evaluation
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import resolve_route_bundle
    from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import CandidateCostInputs

    solved_insertions = []
    priced_patches = []
    ranked_populations = []
    original_solve = patch_evaluation.structural_candidate_solve

    def observe_solve(**kwargs):
        solved_insertions.append(tuple(kwargs["inserted_selection"]))
        return original_solve(**kwargs)

    monkeypatch.setattr(patch_evaluation, "structural_candidate_solve", observe_solve)

    def patch_key(removed, insertions, plan):
        return (tuple(removed), tuple(insertions), None if plan is None else plan.plan_id)

    def price(removed, insertions, plan):
        priced_patches.append(patch_key(removed, insertions, plan))
        return CandidateCostInputs(cost_marginal=(0, 0, 0), compile_calls=1)

    def rank(candidates):
        population = {
            patch_key(c.removed_runtime_indices, c.inserted_selection, c.plan)
            for c in candidates
        }
        ranked_populations.append(population)
        # Pricing is observable work: no evaluated family may be held back
        # from ranking while an earlier family is allowed to commit.
        assert set(priced_patches) <= population
        if stop_at_first_level == "escalation_stop":
            return RankedLevel(ordered=(), rank_info={}, telemetry={})
        insertions = tuple(
            c for c in candidates if c.kind == "insert" and len(c.inserted_selection) == 1
        )
        assert insertions
        return RankedLevel(ordered=insertions, rank_info={}, telemetry={})

    kwargs = _selection_kwargs(
        deletable_indices=(),
        max_insertion_batch_size=2,
        interaction_frontier_widths=(2,),
        route_bundle=resolve_route_bundle("paper_ii_exchange_q_compiled_v1"),
        candidate_cost_inputs=price,
    )
    selection = select_exchange_patch(
        gates=CertificationGates(
            ray_distance_max=1.0, smoothness_eta_max=1e6, condition_number_max=None
        ),
        population_ranker=rank,
        escalate=lambda: False,
        **kwargs,
    )
    assert len(ranked_populations) == 1
    assert priced_patches
    assert all(len(insertions) <= 1 for insertions in solved_insertions)
    assert all(len(key[1]) == 1 for key in priced_patches)
    if stop_at_first_level == "commit":
        assert selection.committed is not None
        assert len(selection.committed.inserted_selection) == 1
    else:
        assert selection.committed is None
        assert selection.stop_reason == "escalation_predicate_false"


@pytest.mark.parametrize("required_deletion", [(), (0,)])
def test_escalated_joint_patches_are_ranked_including_deletion_only_pool_members(
    required_deletion,
) -> None:
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import resolve_route_bundle
    from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import CandidateCostInputs

    priced_patches = []
    ranked_populations = []
    consumed_prices = 0
    escalation_calls = []

    def patch_key(removed, insertions, plan):
        return (tuple(removed), tuple(insertions), None if plan is None else plan.plan_id)

    def price(removed, insertions, plan):
        priced_patches.append(patch_key(removed, insertions, plan))
        return CandidateCostInputs(cost_marginal=(0, 0, 0), compile_calls=1)

    def rank(candidates):
        nonlocal consumed_prices
        population = {
            patch_key(c.removed_runtime_indices, c.inserted_selection, c.plan)
            for c in candidates
        }
        ranked_populations.append(population)
        assert set(priced_patches[consumed_prices:]) <= population
        consumed_prices = len(priced_patches)
        if len(ranked_populations) == 1:
            # A route ranker may find no acceptable singleton. Only after
            # escalation should joint candidates consume work and compete.
            assert all(len(c.inserted_selection) <= 1 for c in candidates)
            return RankedLevel(ordered=(), rank_info={}, telemetry={})
        joint = tuple(
            c for c in candidates
            if c.removed_runtime_indices == required_deletion
            and {atom for atom, _cut in c.inserted_selection} == {"ca", "cb"}
            and len(c.inserted_selection) == 2
        )
        assert joint, "the evaluated next level must offer its joint candidates"
        return RankedLevel(ordered=joint, rank_info={}, telemetry={})

    def escalate():
        escalation_calls.append(True)
        return True

    kwargs = _selection_kwargs(
        deletable_indices=required_deletion,
        # With D=(0,), neither candidate is available on the undeleted face.
        # Their deletion-conditioned availability must feed the joint level.
        candidate_pool_for_deletion=lambda removed: (
            ("ca", "cb") if tuple(removed) == required_deletion else ()
        ),
        max_insertion_batch_size=2,
        interaction_frontier_widths=(2,),
        route_bundle=resolve_route_bundle("paper_ii_exchange_q_compiled_v1"),
        candidate_cost_inputs=price,
    )
    selection = select_exchange_patch(
        gates=CertificationGates(
            ray_distance_max=1.0, smoothness_eta_max=1e6, condition_number_max=None
        ),
        population_ranker=rank,
        escalate=escalate,
        **kwargs,
    )
    assert escalation_calls
    assert len(ranked_populations) == 2
    assert selection.committed is not None
    assert selection.committed.removed_runtime_indices == required_deletion
    assert len(selection.committed.inserted_selection) == 2
    assert any(len(key[1]) == 2 for key in priced_patches)
    assert set(priced_patches) <= set().union(*ranked_populations)


def test_factorized_finalists_validate_joint_geometry_without_reranking(monkeypatch):
    from dataclasses import replace
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import resolve_route_bundle
    from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import (
        CandidateCostInputs, PatchEvaluator,
    )

    ranked_populations, validated, certified = [], [], []

    def rank(candidates):
        ordered = tuple(c for c in candidates if len(c.inserted_selection) == 2)
        ranked_populations.append(tuple(c.order_key for c in ordered))
        return RankedLevel(
            ordered=ordered,
            rank_info={c.order_key: RankInfo(K=1, score=100-i, merit=c.delta)
                       for i, c in enumerate(ordered)},
            telemetry={},
        )

    original = PatchEvaluator.authoritative_joint_candidate

    def validate(self, candidate):
        validated.append(candidate.order_key)
        result = original(self, candidate)
        # First provisional finalist fails a joint-only invariant. Its
        # acquisition is still paid; the next keeps its original rank.
        return replace(result, ordinate_error="joint_test_failure") if len(validated) == 1 else result

    from pipelines.time_dynamics.ap_mclachlan import exchange_selector
    original_certify = exchange_selector.certify_finalist

    def certify(**kwargs):
        certified.append(len(validated))
        return original_certify(**kwargs)

    monkeypatch.setattr(PatchEvaluator, "authoritative_joint_candidate", validate)
    monkeypatch.setattr(exchange_selector, "certify_finalist", certify)
    kwargs = _selection_kwargs(
        deletable_indices=(), max_insertion_batch_size=2,
        gamma_rcond=DEFAULT_GAMMA_PINV_RCOND,
        route_bundle=resolve_route_bundle("paper_ii_exchange_q_compiled_v1"),
        candidate_cost_inputs=lambda *_args: CandidateCostInputs(cost_marginal=(0, 0, 0)),
    )
    selection = select_exchange_patch(
        gates=CertificationGates(ray_distance_max=1, smoothness_eta_max=1e6, condition_number_max=None),
        population_ranker=rank, search_policy="bounded_exhaustive",
        gain_evaluation="singleton_factorized", **kwargs,
    )
    assert len(ranked_populations) == 1
    assert len(ranked_populations[0]) >= 2
    assert validated == list(ranked_populations[0][:len(validated)])
    assert certified and certified[0] == 2
    assert selection.attempts[0].reason == "joint_validation_failed:joint_test_failure"
    failed = selection.certification_work[0]
    assert failed["complete"] is True
    assert failed["logical_total_estimator_entries"] == 0  # frozen work belongs to cache receipt
    assert selection.committed is not None
    assert not selection.committed.joint_validation_required
    for index, attempt in enumerate(selection.attempts):
        reported = attempt.to_json_dict(compiled_score=True)
        assert reported["score"] == 100 - index
        assert reported["score"] != attempt.score
        assert "route_score" not in reported


@pytest.mark.parametrize("failure", ["resolution", "word_identity"])
def test_unvalidated_factorized_failure_does_not_write_deletion_history(monkeypatch, failure):
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import resolve_route_bundle
    from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import CandidateCostInputs, PatchEvaluator
    from pipelines.time_dynamics.ap_mclachlan import exchange_selector
    from pipelines.time_dynamics.ap_mclachlan.search_methods import runtime as search_runtime

    # This three-coordinate fixture isolates failure reporting. The live 15%
    # bound would close its deletion face before the failure under test.
    original_problem = search_runtime.PatchProblem
    monkeypatch.setattr(
        search_runtime, "PatchProblem",
        lambda **fields: original_problem(**{**fields, "max_deletion_batch_fraction": None}),
    )

    def rank(candidates):
        ordered = tuple(c for c in candidates if c.removed_runtime_indices and len(c.inserted_selection) == 2)
        return RankedLevel(
            ordered=ordered,
            rank_info={c.order_key: RankInfo(K=0.8, score=0.25, merit=0.2) for c in ordered},
            telemetry={},
        )

    def unexpected_validation(*_args):
        pytest.fail("invalid word must be rejected before joint acquisition")

    monkeypatch.setattr(PatchEvaluator, "authoritative_joint_candidate", unexpected_validation)
    if failure == "resolution":
        def fail_resolution(*_args, **_kwargs):
            raise ValueError("missing atom")
        monkeypatch.setattr(exchange_selector, "candidate_insertions", fail_resolution)
    kwargs = _selection_kwargs(
        max_insertion_batch_size=2,
        gamma_rcond=DEFAULT_GAMMA_PINV_RCOND,
        route_bundle=resolve_route_bundle("paper_ii_exchange_q_compiled_v1"),
        candidate_cost_inputs=lambda *_args: CandidateCostInputs(cost_marginal=(0, 0, 0)),
    )
    selection = select_exchange_patch(
        gates=CertificationGates(), population_ranker=rank,
        search_policy="bounded_exhaustive", gain_evaluation="singleton_factorized",
        candidate_word_validator=lambda *_args: False, **kwargs,
    )
    assert selection.attempts
    assert all(attempt.deletion_loss is None for attempt in selection.attempts)
    assert selection.certification_work == ()
    for attempt in selection.attempts:
        reported = attempt.to_json_dict(compiled_score=True)
        assert reported["score"] == 0.25
        assert reported["cost_factor_k"] == 0.8
        assert reported["route_merit"] == 0.2
        assert "route_score" not in reported


@pytest.mark.parametrize("sealed_score", [None, 0.0, 0.00025])
def test_compiled_attempt_serialization_never_substitutes_legacy_utility(sealed_score):
    attempt = AttemptRecord(
        family="bounded_exhaustive", kind="delete", score=3.1486e13,
        removed_runtime_indices=(23,), inserted_selection=(), reason="recorded",
        route_score=sealed_score,
    )
    historical = attempt.to_json_dict()
    assert historical["score"] == 3.1486e13
    assert historical["route_score"] == sealed_score
    compiled = attempt.to_json_dict(compiled_score=True)
    assert compiled["score"] == sealed_score
    assert "route_score" not in compiled
    assert "legacy_utility" not in compiled
    assert {k: v for k, v in historical.items() if k not in {"score", "route_score"}} == {
        k: v for k, v in compiled.items() if k != "score"
    }

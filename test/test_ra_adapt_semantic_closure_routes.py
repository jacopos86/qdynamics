from __future__ import annotations

from dataclasses import replace
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.scaffold.hh_continuation_scoring import SimpleScoreConfig
from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.ra_adapt import engine as ra_engine
from pipelines.static_adapt.ra_adapt import runtime as ra_runtime
from pipelines.static_adapt.ra_adapt import semantic_closure_routes as semantic_routes
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_STATIONARY,
    RAAdaptOperationalControls,
    RAAdaptRequest,
    RESOURCE_WEIGHTING_ALL_PHASE,
    canonical_sha256,
    ra_adapt_request_from_mapping,
    resolved_ra_adapt_protocol_from_mapping,
)
from pipelines.static_adapt.ra_adapt.adapters import (
    GlobalSingletonGradientPhase0CandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.engine import (
    RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
    _repaired_route_contract,
)
from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    PAPER_I_RA_PHASE0_GRADIENT_FIXED24,
    PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
    PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
    PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
    PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1,
    PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
    PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
    PAPER_I_RA_PHASE0_PROXY_ADAPTIVE,
    PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
    PAPER_I_RA_CANONICAL_REGIME_IDS,
    PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
    PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW,
    PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS,
    PAPER_I_RA_PHASE3_NATURAL_TERMINAL_ROUTE_VARIANTS,
    PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION,
    PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_V2,
    PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
    build_paper_i_ra_strong_weak_nph3_problem,
    build_paper_i_ra_all_phase_adaptive_request,
    build_paper_i_ra_all_phase_adaptive_natural_terminal_request,
    build_paper_i_ra_all_phase_position_adaptive_request,
    build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request,
    build_paper_i_ra_hh_regime_problem,
    materialize_paper_i_ra_strong_weak_always_k5_protocol,
    materialize_paper_i_ra_semantic_protocol,
    build_paper_i_ra_strong_weak_always_k5_request,
    build_paper_i_ra_strong_weak_plateau_k5_request,
    preflight_paper_i_ra_strong_weak_always_k5,
    preflight_paper_i_ra_strong_weak_plateau_k5,
    preflight_paper_i_ra_semantic,
    project_approved_phase0_ablation,
    execute_semantic_phase0_runtime,
    build_semantic_gradient_adaptive_phase0_receipt,
    build_semantic_position_phase0_receipt,
    filter_semantic_phase0_position_domain,
    validate_semantic_position_phase0_receipt,
    validate_semantic_phase0_runtime_binding,
    semantic_closure_source_implementation_inventory,
    semantic_closure_route_identity,
    validate_semantic_final_selector_accounting,
    validate_semantic_phase3_no_positive_terminal_receipt,
    validate_semantic_phase3_natural_terminal_route_contract,
)
from pipelines.static_adapt.adaptive_phase_contracts import (
    ADAPTIVE_HORIZON_POLICY_EXACT_TARGET_V1,
    ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1,
    ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_RAISE_V1,
    ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1,
)
from pipelines.static_adapt.ra_adapt.adaptive_phase_shortlist import (
    ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1,
    ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1,
    AdaptivePhaseCandidateScore,
    select_adaptive_phase_shortlist,
)
from pipelines.static_adapt.hh_backend_compile_oracle import (
    BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
)
from pipelines.static_adapt.selector_measurement_proxy import (
    ControllerMeasurementWorkAccumulator,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AlwaysCommutationReducedInsertion,
    AcceptedStateResume,
    AppendOnlyInsertion,
    BeamOff,
    FreshStart,
    PruningOff,
    PlateauCommutationInsertion,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRObservationPolicy,
    SRStopPolicy,
    SingletonAdmission,
)
from pipelines.static_adapt.sr_snake import _controller as sr_controller
from pipelines.scaffold import hh_continuation_scoring as continuation_scoring
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm
from pipelines.static_adapt.sr_snake._selection import (
    _SRControllerState,
    _SelectionWorkspace,
)
from pipelines.static_adapt.sr_snake._transition import (
    _AcceptedStateSnapshot,
)


def test_gradient_and_proxy_fixed24_are_a_single_declared_phase0_ablation() -> None:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    gradient = preflight_paper_i_ra_strong_weak_always_k5(
        problem,
        build_paper_i_ra_strong_weak_always_k5_request(
            PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2
        ),
    )
    proxy = preflight_paper_i_ra_strong_weak_always_k5(
        problem,
        build_paper_i_ra_strong_weak_always_k5_request(
            PAPER_I_RA_PHASE0_PROXY_FIXED24_V2
        ),
    )

    gradient_route = gradient.route_contract
    proxy_route = proxy.route_contract
    assert gradient_route is not None
    assert proxy_route is not None
    assert project_approved_phase0_ablation(gradient_route) == (
        project_approved_phase0_ablation(proxy_route)
    )

    gradient_semantics = gradient_route["native_semantic_contract"]
    proxy_semantics = proxy_route["native_semantic_contract"]
    assert gradient_semantics["phase0_policy"] == {
        "adaptive_shadow_receipt": False,
        "benefit": "absolute_append_endpoint_generator_gradient_v1",
        "fubini_study_metric": "off",
        "graph_proxy_cost": "off",
        "population": "same_ordered_append_endpoint_generator_population_v1",
        "qiskit_compile": "off",
        "score": "absolute_append_endpoint_generator_gradient_v1",
        "shortlist": "fixed_top_24_v1",
    }
    assert proxy_semantics["phase0_policy"] == {
        "adaptive_shadow_receipt": False,
        "benefit": "absolute_append_endpoint_generator_gradient_v1",
        "fubini_study_metric": "off",
        "graph_proxy_cost": "paper_i_structural_graph_proxy_transform_v1",
        "population": "same_ordered_append_endpoint_generator_population_v1",
        "qiskit_compile": "off",
        "score": "absolute_append_gradient_over_graph_proxy_cost_v1",
        "shortlist": "fixed_top_24_v1",
    }


def test_v2_phase0_matrix_separates_score_and_cardinality_axes() -> None:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    expected = {
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2: (
            "absolute_append_endpoint_generator_gradient_v1",
            "off",
            "fixed_top_24_v1",
        ),
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2: (
            "absolute_append_endpoint_generator_gradient_v1",
            "off",
            "phase0_active_score_effective_competition_shortlist_v2",
        ),
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2: (
            "absolute_append_gradient_over_graph_proxy_cost_v1",
            "paper_i_structural_graph_proxy_transform_v1",
            "fixed_top_24_v1",
        ),
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2: (
            "absolute_append_gradient_over_graph_proxy_cost_v1",
            "paper_i_structural_graph_proxy_transform_v1",
            "phase0_active_score_effective_competition_shortlist_v2",
        ),
    }
    projected = []
    for route_variant, (score, graph_proxy, shortlist) in expected.items():
        protocol = preflight_paper_i_ra_strong_weak_always_k5(
            problem,
            build_paper_i_ra_strong_weak_always_k5_request(route_variant),
        )
        assert protocol.route_contract is not None
        route = protocol.route_contract
        native = route["native_semantic_contract"]
        assert native["semantic_implementation_version"] == (
            PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_V2
        )
        assert native["phase0_policy"]["score"] == score
        assert native["phase0_policy"]["graph_proxy_cost"] == graph_proxy
        assert native["phase0_policy"]["shortlist"] == shortlist
        assert native["qiskit_active_phases"] == [
            "phase_i",
            "phase_ii",
            "phase_iii",
        ]
        projected.append(project_approved_phase0_ablation(route))

    assert all(value == projected[0] for value in projected[1:])


def test_position_phase0_matrix_adds_only_placement_score_and_cardinality() -> None:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    expected = {
        PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1: (
            "absolute_position_record_gradient_v1",
            "off",
            "fixed_top_24_v1",
        ),
        PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1: (
            "absolute_position_record_gradient_v1",
            "off",
            "phase0_active_score_effective_competition_shortlist_v2",
        ),
        PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1: (
            "absolute_position_gradient_over_graph_proxy_cost_v1",
            "paper_i_structural_graph_proxy_transform_v1",
            "fixed_top_24_v1",
        ),
        PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1: (
            "absolute_position_gradient_over_graph_proxy_cost_v1",
            "paper_i_structural_graph_proxy_transform_v1",
            "phase0_active_score_effective_competition_shortlist_v2",
        ),
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1: (
            "absolute_position_record_gradient_v1",
            "off",
            "phase0_active_score_effective_competition_shortlist_v2",
        ),
    }
    assert PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS == frozenset(expected)
    for route_variant, (score, graph_proxy, shortlist) in expected.items():
        protocol = preflight_paper_i_ra_strong_weak_plateau_k5(
            problem,
            build_paper_i_ra_strong_weak_plateau_k5_request(route_variant),
        )
        assert protocol.route_contract is not None
        phase0 = protocol.route_contract["native_semantic_contract"][
            "phase0_policy"
        ]
        expected_phase0 = {
            "adaptive_shadow_receipt": False,
            "benefit": "absolute_position_record_gradient_v1",
            "fubini_study_metric": "off",
            "graph_proxy_cost": graph_proxy,
            "population": (
                "current_commutation_reduced_candidate_position_records_v1"
            ),
            "qiskit_compile": "off",
            "score": score,
            "shortlist": shortlist,
        }
        if route_variant == PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1:
            expected_phase0.update(
                {
                    "placement_activation": (
                        "append_record_when_closed_full_commutation_reduced_"
                        "records_when_open_v1"
                    ),
                    "generator_level_reexpansion_after_phase0": False,
                }
            )
        assert phase0 == expected_phase0


def test_strong_weak_plateau_k5_builder_is_exact_and_non_authorized() -> None:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    request = build_paper_i_ra_strong_weak_plateau_k5_request(
        PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1
    )
    assert isinstance(request.method.insertion, PlateauCommutationInsertion)
    assert request.execution.stop.maximum_controller_rounds == 5
    assert isinstance(request.execution.resume, FreshStart)

    protocol = preflight_paper_i_ra_strong_weak_plateau_k5(problem, request)
    assert protocol.execution_authorized is False
    assert protocol.request.method.insertion.kind == "plateau_commutation"


@pytest.mark.parametrize(
    "insertion",
    [
        AppendOnlyInsertion(),
        PlateauCommutationInsertion(),
        AlwaysCommutationReducedInsertion(),
    ],
    ids=["append-only", "plateau", "always-open"],
)
def test_position_phase0_public_seam_materializes_all_typed_insertion_policies(
    insertion: object,
) -> None:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    seed = build_paper_i_ra_strong_weak_plateau_k5_request(
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1
    )
    request = replace(seed, method=replace(seed.method, insertion=insertion))

    preflight = preflight_paper_i_ra_semantic(problem, request)
    materialized = materialize_paper_i_ra_semantic_protocol(problem, request)

    assert preflight.execution_authorized is False
    assert materialized.execution_authorized is False
    assert materialized.request.method.insertion.kind == insertion.kind
    assert materialized.route_contract is not None
    assert materialized.route_contract["native_semantic_contract"][
        "insertion_policy"
    ] == insertion.kind


def test_append_only_position_population_reproduces_generator_first_scores_within_128_ulp() -> None:
    rows = [
        {
            "domain_record_id": f"g{index}@3",
            "generator_id": f"g{index}",
            "pool_index": index,
            "pool_label": f"G{index}",
            "insertion_position": 3,
            "position_class": "append",
            "gradient_signed": gradient,
            "graph_proxy_denominator": denominator,
        }
        for index, (gradient, denominator) in enumerate(
            ((-0.75, 1.5), (0.25, 0.5), (1.0, 4.0))
        )
    ]
    events = [f"append-only:{index}" for index in range(len(rows))]
    for fixed, adaptive, proxy in (
        (
            PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1,
            PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
            False,
        ),
        (
            PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
            PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
            True,
        ),
    ):
        fixed_receipt = build_semantic_position_phase0_receipt(
            rows,
            estimator_event_ids=events,
            route_variant=fixed,
        )
        adaptive_receipt = build_semantic_position_phase0_receipt(
            rows,
            estimator_event_ids=events,
            route_variant=adaptive,
        )
        expected = {
            int(row["pool_index"]): (
                abs(float(row["gradient_signed"]))
                / float(row["graph_proxy_denominator"])
                if proxy
                else abs(float(row["gradient_signed"]))
            )
            for row in rows
        }
        expected_order = sorted(expected, key=lambda index: (-expected[index], index))
        for receipt in (fixed_receipt, adaptive_receipt):
            assert receipt["input_candidate_count"] == len(rows)
            assert all(
                row["position_class"] == "append"
                for row in receipt["population"]
            )
            assert [row["pool_index"] for row in receipt["ranking"]] == expected_order
            for row in receipt["ranking"]:
                target = expected[int(row["pool_index"])]
                assert abs(float(row["active_score"]) - target) <= 128 * math.ulp(target)

    generator_first = build_semantic_gradient_adaptive_phase0_receipt(
        available_indices=tuple(range(len(rows))),
        gradients=tuple(float(row["gradient_signed"]) for row in rows),
        pool_labels=tuple(str(row["pool_label"]) for row in rows),
        estimator_event_ids=events,
        route_variant=PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
    )
    position_aware = build_semantic_position_phase0_receipt(
        rows,
        estimator_event_ids=events,
        route_variant=PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
    )
    assert [row["pool_index"] for row in position_aware["ranking"]] == (
        generator_first["ranked_pool_indices"]
    )
    assert [row["pool_index"] for row in position_aware["retained_records"]] == (
        generator_first["retained_pool_indices"]
    )
    generator_scores = {
        int(row["pool_index"]): float(row["active_score"])
        for row in generator_first["ranking"]
    }
    for row in position_aware["ranking"]:
        target = generator_scores[int(row["pool_index"])]
        assert abs(float(row["active_score"]) - target) <= 128 * math.ulp(target)


@pytest.mark.parametrize(
    "insertion_policy",
    ["append_only", "plateau_commutation"],
    ids=["append-only", "plateau-closed"],
)
def test_live_append_endpoint_position_route_matches_generator_first_within_128_ulp(
    insertion_policy: str,
) -> None:
    def term(label: str, word: str) -> AnsatzTerm:
        return AnsatzTerm(
            label=label,
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(len(word), ps=word, pc=1.0)],
            ),
        )

    problem = build_paper_i_ra_strong_weak_nph3_problem()
    generator_protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_adaptive_request(
            insertion_policy=insertion_policy,
            maximum_controller_rounds=5,
        ),
    )
    position_protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_request(
            insertion_policy=insertion_policy,
            maximum_controller_rounds=5,
        ),
    )
    assert generator_protocol.request.method.insertion.kind == insertion_policy
    assert position_protocol.request.method.insertion.kind == insertion_policy

    rng = np.random.default_rng(20260817)
    reference = np.asarray(
        rng.normal(size=4) + 1.0j * rng.normal(size=4),
        dtype=complex,
    )
    reference /= np.linalg.norm(reference)
    selected = [term("selected-x", "xe"), term("selected-yz", "yz")]
    pool = [
        term("candidate-zx", "zx"),
        term("candidate-zz", "zz"),
        term("candidate-ex", "ex"),
    ]
    theta = np.asarray([0.17, -0.23], dtype=float)
    state = CompiledAnsatzExecutor(selected).prepare_state(theta, reference)
    compiled_hamiltonian = compile_polynomial_action(
        PauliPolynomial(
            "JW",
            [
                PauliTerm(2, ps="ze", pc=0.7),
                PauliTerm(2, ps="ex", pc=-0.4),
                PauliTerm(2, ps="xx", pc=0.3),
            ],
        ),
        pauli_action_cache={},
    )
    h_state = apply_compiled_polynomial(state, compiled_hamiltonian)
    compiled_pool = tuple(
        compile_polynomial_action(row.polynomial, pauli_action_cache={})
        for row in pool
    )

    class Cursor:
        selected_ops = selected

        @staticmethod
        def selection_available_indices() -> set[int]:
            return set(range(len(pool)))

    class Estimator:
        calls = 0

        @classmethod
        def _record_gradient_surface_primitives(
            cls,
            **_kwargs: object,
        ) -> None:
            cls.calls += 1

    pending = SimpleNamespace(
        psi_current=state,
        hpsi_current=h_state,
        theta_logical_current=theta,
        gradients=np.zeros(len(pool), dtype=float),
        grad_magnitudes=np.zeros(len(pool), dtype=float),
        gradient_parallel_info={},
        gradient_eval_elapsed_s=0.0,
        deferred_observations=[],
        depth=len(selected),
    )
    live_session = SimpleNamespace(
        context=SimpleNamespace(
            transition_services=SimpleNamespace(
                controller_noise_runtime=None
            ),
            compiled_pool=compiled_pool,
            pool=pool,
            parallel_gradient_workers=1,
        ),
        cursor=Cursor(),
        estimator_service=Estimator(),
    )
    adapt_pipeline._DefaultNoPruneNumericalSession._evaluate_default_candidate_gradient_surface(
        live_session,
        pending,
        consumer_scope="live_append_endpoint_parity_test",
    )
    assert Estimator.calls == 1

    append_position = len(selected)
    position_context = (
        continuation_scoring._prepare_exact_insertion_first_order_context(
            selected_ops=selected,
            theta=theta,
            psi_ref=reference,
            psi_state=state,
            hpsi_state=h_state,
            pauli_action_cache={},
            state_consistency_tolerance=1.0e-10,
        )
    )
    position_gradients = []
    domain = []
    rows = []
    for pool_index, candidate in enumerate(pool):
        geometry = (
            continuation_scoring._exact_insertion_first_order_candidate_geometry(
                context=position_context,
                candidate_term=candidate,
                position_id=append_position,
                candidate_compiled=compiled_pool[pool_index],
            )
        )
        position_gradient = float(geometry["energy_gradient"])
        generator_gradient = float(pending.gradients[pool_index])
        scale = max(abs(position_gradient), abs(generator_gradient))
        tolerance = 128 * math.ulp(scale)
        assert abs(position_gradient - generator_gradient) <= tolerance
        position_gradients.append(position_gradient)
        record = adapt_pipeline._CandidatePositionRecord(
            domain_record_id=f"g{pool_index}@{append_position}",
            generator_id=f"g{pool_index}",
            parent_generator_id=None,
            pool_index=pool_index,
            pool_label=str(candidate.label),
            insertion_position=append_position,
            symmetry_identity=f"sym-{pool_index}",
            lineage_identity=(f"g{pool_index}",),
        )
        domain.append(record)
        rows.append(
            {
                "domain_record_id": record.domain_record_id,
                "generator_id": record.generator_id,
                "pool_index": record.pool_index,
                "pool_label": record.pool_label,
                "insertion_position": record.insertion_position,
                "position_class": "append",
                "gradient_signed": position_gradient,
                "graph_proxy_denominator": 1.0,
            }
        )

    event_ids = [f"live-gradient:{index}" for index in range(len(pool))]
    generator_receipt = build_semantic_gradient_adaptive_phase0_receipt(
        available_indices=tuple(range(len(pool))),
        gradients=tuple(float(value) for value in pending.gradients),
        pool_labels=tuple(str(row.label) for row in pool),
        estimator_event_ids=event_ids,
        route_variant=PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
    )
    position_receipt = build_semantic_position_phase0_receipt(
        rows,
        estimator_event_ids=event_ids,
        route_variant=PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
    )
    assert [row["pool_index"] for row in position_receipt["ranking"]] == (
        generator_receipt["ranked_pool_indices"]
    )
    assert [row["pool_index"] for row in position_receipt["retained_records"]] == (
        generator_receipt["retained_pool_indices"]
    )
    generator_phase_i = filter_semantic_phase0_position_domain(
        tuple(domain),
        ranked_pool_indices=generator_receipt["ranked_pool_indices"],
        retained_pool_indices=generator_receipt["retained_pool_indices"],
    )
    position_by_id = {row.domain_record_id: row for row in domain}
    position_phase_i = tuple(
        position_by_id[str(row["domain_record_id"])]
        for row in position_receipt["retained_records"]
    )
    assert [row.domain_record_id for row in position_phase_i] == [
        row.domain_record_id for row in generator_phase_i
    ]
    assert position_phase_i[0].domain_record_id == (
        generator_phase_i[0].domain_record_id
    )
    assert position_receipt["qiskit_compile_cost_policy"] == "off"
    assert position_receipt["metric_policy"] == "off"


@pytest.mark.parametrize(
    "route_variant",
    [
        PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1,
        PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
        PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
        PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
    ],
)
def test_position_routes_dispatch_to_native_position_record_phase0(
    route_variant: str,
) -> None:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    protocol = preflight_paper_i_ra_strong_weak_plateau_k5(
        problem,
        build_paper_i_ra_strong_weak_plateau_k5_request(route_variant),
    )
    assert protocol.route_contract is not None
    calls: list[dict[str, object]] = []

    class PositionTransaction:
        context = SimpleNamespace(
            candidate_adapter=protocol.request.adapter,
            route_contract=protocol.route_contract,
        )

        def run_position_record_phase0(self, **kwargs: object) -> str:
            calls.append(dict(kwargs))
            return "position-phase0"

    domain = (SimpleNamespace(domain_record_id="g0@0"),)
    assert execute_semantic_phase0_runtime(
        PositionTransaction(),
        admissible_domain=domain,
    ) == "position-phase0"
    assert calls == [
        {
            "admissible_domain": domain,
            "route_variant": route_variant,
            "shortlist_size": 24,
        }
    ]


def test_generator_first_all_phase_v1_remains_distinct_and_cross_binding_fails_closed() -> None:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    old_protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_adaptive_request(
            insertion_policy="plateau_commutation",
            maximum_controller_rounds=5,
        ),
    )
    new_protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_request(
            insertion_policy="plateau_commutation",
            maximum_controller_rounds=5,
        ),
    )

    assert old_protocol.algorithm_id != new_protocol.algorithm_id
    assert old_protocol.request.adapter.route_variant == (
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1
    )
    assert new_protocol.request.adapter.route_variant == (
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1
    )
    with pytest.raises(RuntimeError, match="route and runtime adapter|binding drifted"):
        validate_semantic_phase0_runtime_binding(
            old_protocol.request.adapter,
            new_protocol.route_contract,
        )


def test_position_fixed_and_adaptive_share_exact_active_score_ranking() -> None:
    rows = [
        {
            "domain_record_id": "g0@0",
            "generator_id": "g0",
            "pool_index": 0,
            "pool_label": "G0",
            "insertion_position": 0,
            "position_class": "interior",
            "gradient_signed": 3.0,
            "graph_proxy_denominator": 3.0,
        },
        {
            "domain_record_id": "g0@2",
            "generator_id": "g0",
            "pool_index": 0,
            "pool_label": "G0",
            "insertion_position": 2,
            "position_class": "append",
            "gradient_signed": 2.0,
            "graph_proxy_denominator": 1.0,
        },
        {
            "domain_record_id": "g1@1",
            "generator_id": "g1::pool[1]",
            "pool_index": 1,
            "pool_label": "G1",
            "insertion_position": 1,
            "position_class": "interior",
            "gradient_signed": 1.0,
            "graph_proxy_denominator": 1.0,
        },
    ]
    event_ids = ["estimator:1:a", "estimator:2:b", "estimator:3:c"]
    for fixed, adaptive, expected in (
        (
            PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1,
            PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
            ["g0@0", "g0@2", "g1@1"],
        ),
        (
            PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
            PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
            ["g0@2", "g0@0", "g1@1"],
        ),
    ):
        fixed_receipt = build_semantic_position_phase0_receipt(
            rows,
            estimator_event_ids=event_ids,
            route_variant=fixed,
            cap=24,
        )
        adaptive_receipt = build_semantic_position_phase0_receipt(
            rows,
            estimator_event_ids=event_ids,
            route_variant=adaptive,
            cap=24,
        )
        assert [row["domain_record_id"] for row in fixed_receipt["ranking"]] == expected
        assert [row["domain_record_id"] for row in adaptive_receipt["ranking"]] == expected
        assert [row["active_score"] for row in fixed_receipt["ranking"]] == [
            row["active_score"] for row in adaptive_receipt["ranking"]
        ]
        assert validate_semantic_position_phase0_receipt(fixed_receipt) == fixed_receipt
        assert validate_semantic_position_phase0_receipt(adaptive_receipt) == adaptive_receipt


def test_position_phase0_receipt_cross_binds_exact_scored_records() -> None:
    rows = [
        {
            "domain_record_id": "g0@0",
            "generator_id": "g0",
            "pool_index": 0,
            "pool_label": "G0",
            "insertion_position": 0,
            "position_class": "interior",
            "gradient_signed": 4.0,
            "graph_proxy_denominator": 1.0,
        },
        {
            "domain_record_id": "g0@2",
            "generator_id": "g0",
            "pool_index": 0,
            "pool_label": "G0",
            "insertion_position": 2,
            "position_class": "append",
            "gradient_signed": 0.1,
            "graph_proxy_denominator": 1.0,
        },
    ]
    receipt = build_semantic_position_phase0_receipt(
        rows,
        estimator_event_ids=["gradient:0", "gradient:1"],
        route_variant=PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
    )

    def project(row: object) -> dict[str, object]:
        assert isinstance(row, dict)
        return {
            key: row[key]
            for key in (
                "domain_record_id",
                "generator_id",
                "pool_index",
                "pool_label",
                "insertion_position",
                "position_class",
            )
        }

    population = [project(row) for row in receipt["population"]]
    shortlist = [project(row) for row in receipt["retained_records"]]
    screen = {
        "schema": "paper_i_scored_gradient_phase0_population_v1",
        "population_count": len(population),
        "population": population,
        "ordered_population_sha256": canonical_sha256(population),
        "shortlist_count": len(shortlist),
        "shortlist": shortlist,
        "ordered_shortlist_sha256": canonical_sha256(shortlist),
    }
    scored = {"phase0_gradient_screen": screen}

    assert validate_semantic_position_phase0_receipt(
        receipt,
        scored_population=scored,
    ) == receipt

    detached = copy.deepcopy(scored)
    detached_screen = detached["phase0_gradient_screen"]
    detached_screen["shortlist"] = [population[1]]
    detached_screen["shortlist_count"] = 1
    detached_screen["ordered_shortlist_sha256"] = canonical_sha256(
        detached_screen["shortlist"]
    )
    with pytest.raises(RuntimeError, match="retained position-record domain drifted"):
        validate_semantic_position_phase0_receipt(
            receipt,
            scored_population=detached,
        )


@pytest.mark.parametrize(
    ("route_variant", "insertion_contract"),
    [
        (PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1, "always"),
        (PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1, "always"),
        (PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1, "plateau_open"),
    ],
    ids=["position-v1-always", "all-phase-always", "all-phase-plateau-open"],
)
def test_native_position_phase0_evaluates_each_record_once_and_filters_plans(
    monkeypatch: pytest.MonkeyPatch,
    route_variant: str,
    insertion_contract: str,
) -> None:
    session, pending, proxy, transaction, domain = (
        _semantic_phase0_runtime_fixture(route_variant)
    )
    if insertion_contract == "plateau_open":
        request = build_paper_i_ra_all_phase_position_adaptive_request(
            insertion_policy="plateau_commutation",
            maximum_controller_rounds=5,
        )
        protocol = preflight_paper_i_ra_semantic(
            build_paper_i_ra_strong_weak_nph3_problem(),
            request,
        )
        session.context.candidate_adapter = request.adapter
        session.context.route_contract = protocol.route_contract
    session.context.reference_state = np.asarray([1.0], dtype=complex)
    session.context.compiled_pool = tuple(object() for _ in range(3))
    session.cursor.selected_ops = []
    session.cursor.pauli_action_cache = {}
    pending.psi_current = np.asarray([1.0], dtype=complex)
    pending.hpsi_current = np.asarray([1.0], dtype=complex)
    pending.phase2_score_cfg_round = SimpleNamespace(
        batch_state_consistency_tolerance=1.0e-10
    )
    pending.insertion_mode = "full_commutation_reduced"
    pending.candidate_position_plans = {
        0: {
            "schema": "test-plan-v1",
            "requested_positions": [0, 2],
            "representative_positions": [0, 2],
            "representative_by_position": {0: 0, 2: 2},
            "members_by_representative": {0: [0], 2: [2]},
            "commuting_crossings": [],
            "collapsed_position_count": 0,
        },
        1: {
            "schema": "test-plan-v1",
            "requested_positions": [0, 2],
            "representative_positions": [0, 2],
            "representative_by_position": {0: 0, 2: 2},
            "members_by_representative": {0: [0], 2: [2]},
            "commuting_crossings": [],
            "collapsed_position_count": 0,
        },
        2: {
            "schema": "test-plan-v1",
            "requested_positions": [1],
            "representative_positions": [1],
            "representative_by_position": {1: 1},
            "members_by_representative": {1: [1]},
            "commuting_crossings": [],
            "collapsed_position_count": 0,
        },
    }
    gradients = {
        (0, 0): 4.0,
        (0, 2): 0.10,
        (1, 0): 0.09,
        (1, 2): 0.08,
        (2, 1): 0.07,
    }
    monkeypatch.setattr(
        adapt_pipeline,
        "_prepare_exact_insertion_first_order_context",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "_exact_insertion_first_order_candidate_geometry",
        lambda *, candidate_term, position_id, **_kwargs: {
            "energy_gradient": gradients[
                (int(str(candidate_term.label)[1:]), int(position_id))
            ],
            "fubini_study_metric": 1.0,
        },
    )

    class PositionEstimator:
        def _candidate_physical_tangent(self, *_args: object, **kwargs: object) -> tuple[str, int]:
            return ("candidate", int(kwargs["insertion_position"]))

        def _record_estimator_primitive(self, **kwargs: object) -> object:
            index = len(session.occurrences) + 1
            session.occurrences.append(
                {
                    "component": kwargs["component"],
                    "consumer_scope": kwargs["consumer_scope"],
                    "sequence": index,
                    "primitive_id": f"position-gradient-{index}",
                }
            )
            return SimpleNamespace(primitive_id=f"position-gradient-{index}")

        def _record_candidate_self_metric_primitive(
            self, **_kwargs: object
        ) -> None:
            return None

    session.estimator_service = PositionEstimator()
    phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
        session=session,
        transaction=transaction,
        admissible_domain=domain,
    )
    assert phase0 is not None

    assert len(session.occurrences) == len(domain)
    assert len({row["primitive_id"] for row in session.occurrences}) == len(domain)
    assert [record.domain_record_id for record in phase0.shortlist] == ["g0@0"]
    assert pending.shortlist == [0]
    assert pending.candidate_position_plans[0]["representative_positions"] == [0]
    assert set(pending.candidate_position_plans) == {0}
    assert proxy.calls == []
    assert transaction.context.phase3_backend_compile_oracle.calls == []
    receipt = pending.phase0_gradient_shortlist_receipt
    assert receipt["position_aware_gradient_surface"] is True
    assert receipt["generator_level_reexpansion_after_phase0"] is False
    assert receipt["qiskit_compile_cost_policy"] == "off"
    assert receipt["metric_policy"] == "off"
    assert pending.gradients.tolist() == pytest.approx([0.10, 0.08, 0.0])
    assert pending.grad_magnitudes.tolist() == pytest.approx(
        [0.10, 0.08, 0.0]
    )
    assert pending.max_grad == pytest.approx(4.0)

    phase1_window = SimpleNamespace(
        old_pre_indices=(),
        active_post_indices=(),
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "_predict_nested_refit_window_for_position",
        lambda **_kwargs: phase1_window,
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "_resolve_phase3_response_window_for_position",
        lambda **_kwargs: phase1_window,
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "build_nested_window_accounting",
        lambda *_args, **_kwargs: SimpleNamespace(
            compile_proxy_refit_count=0
        ),
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "measurement_group_specs_for_term",
        lambda _candidate: (),
    )
    observed_phase1_gradients: list[float] = []

    def build_phase1_feature(**kwargs: object) -> SimpleNamespace:
        gradient_signed = float(kwargs["gradient_signed"])
        observed_phase1_gradients.append(gradient_signed)
        score = abs(gradient_signed)
        return SimpleNamespace(
            g_abs=score,
            simple_score=score,
            phase1_score_mode="fixture_abs_gradient_v1",
            phase1_legacy_simple_score=score,
            phase1_trust_region_gain=None,
            phase1_trust_region_score=None,
            phase1_rho=None,
        )

    monkeypatch.setattr(
        adapt_pipeline,
        "build_candidate_features",
        build_phase1_feature,
    )
    session.context.compiled_pool = tuple(
        SimpleNamespace(terms=(object(),)) for _ in range(3)
    )
    session.context.backend_compile_scope = "fixture_no_backend_compile"
    session.context.backend_compile_oracle = None
    session.context.pool_family_ids = tuple(
        f"fixture-family-{index}" for index in range(3)
    )
    session.context.phase3_geometry_window_size = 0
    session.context.phase3_response_coordinate_scope = (
        "legacy_reopt_coupled_v1"
    )
    session.context.phase3_enabled = False
    session.context.phase1_residual_indices = set()
    session.context.pool_symmetry_specs = ({}, {}, {})
    session.context.phase3_symmetry_mitigation_mode = "off"
    session.context.max_depth = 5
    session.context.phase3_lifetime_cost_mode = "off"
    session.context.selector_candidate_metadata = SimpleNamespace(
        geometry_policy_key=lambda: "fixture_geometry_v1",
        response_geometry_policy_key=lambda: "fixture_response_v1",
        attach_selector_metadata=(
            lambda **kwargs: (kwargs["feat_obj"], {})
        ),
    )
    session.cursor.phase1_measure_cache = SimpleNamespace(
        estimate=lambda _groups: object()
    )
    session.cursor.pool_generator_registry = {}
    session._base_metric_for_candidate = lambda **_kwargs: 1.0
    pending.backend_compile_snapshot = None
    pending.candidate_metric_cache = {0: 1.0}
    pending.family_repeat_cache = {"fixture-family-0": 0.0}
    pending.phase3_sigma_by_label = {}
    pending.stage_name = "selection"
    pending.phase1_score_cfg_round = SimpleScoreConfig()
    pending.depth = 0
    pending.controller_pre_snapshot_dict = {}
    pending.insertion_round_policy = None
    pending.domain_by_pool_position = {
        (int(record.pool_index), int(record.insertion_position)): record
        for record in phase0.shortlist
    }
    monkeypatch.setattr(
        adapt_pipeline,
        "rescore_hardware_cost_family",
        lambda records, _cfg: records,
    )

    phase1 = (
        adapt_pipeline._DefaultNoPruneNumericalSession
        ._evaluate_default_phase1_positions(
            session,
            pending,
        )
    )
    assert observed_phase1_gradients == pytest.approx([4.0])
    assert len(phase1["records"]) == 1
    assert phase1["records"][0]["feature"].g_abs == pytest.approx(4.0)
    assert phase1["records"][0]["phase1_active_score"] == pytest.approx(
        4.0
    )


@pytest.mark.parametrize(
    "route_variant",
    [
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24,
        PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW,
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE,
    ],
)
def test_v1_phase0_routes_are_provenance_only_and_fail_canonical_preflight(
    route_variant: str,
) -> None:
    identity = semantic_closure_route_identity(route_variant)
    assert identity.route_variant == route_variant
    request = build_paper_i_ra_strong_weak_always_k5_request(route_variant)

    with pytest.raises(ValueError, match="retired.*v2"):
        preflight_paper_i_ra_strong_weak_always_k5(
            build_paper_i_ra_strong_weak_nph3_problem(),
            request,
        )
    with pytest.raises(ValueError, match="retired.*v2"):
        materialize_paper_i_ra_semantic_protocol(
            build_paper_i_ra_strong_weak_nph3_problem(),
            request,
        )


def test_v1_phase0_route_contract_fails_closed_at_runtime_binding() -> None:
    request = build_paper_i_ra_strong_weak_always_k5_request(
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE
    )
    identity = semantic_closure_route_identity(request.adapter.route_variant)
    _, _, route, _ = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        algorithm_id=identity.algorithm_id,
        problem=build_paper_i_ra_strong_weak_nph3_problem(),
    )

    with pytest.raises(RuntimeError, match="retired.*v2"):
        semantic_routes.validate_semantic_phase0_runtime_binding(
            request.adapter,
            route,
        )


def test_ordinary_accepted_state_replay_remains_reachable() -> None:
    receipt = ra_runtime._accepted_state(
        history=[{"energy_after_opt": -1.25}],
        checkpoint={
            "ordered_active_operator_labels": ["G0"],
            "ordered_active_operators": [
                {"generator_id": "generator:0", "label": "G0"}
            ],
            "signed_unwrapped_logical_parameters": [0.125],
            "signed_unwrapped_runtime_parameters": [0.25],
            "projective_state_fingerprint": "state:1",
        },
        round_index=1,
        insertion_positions=(0,),
    )

    assert receipt.controller_round == 1
    assert receipt.operators == ("G0",)
    assert receipt.generator_ids == ("generator:0",)
    assert receipt.logical_parameters == (0.125,)
    assert receipt.runtime_parameters == (0.25,)
    assert receipt.energy == -1.25


def test_proxy_adaptive_route_binds_the_effective_competition_policy() -> None:
    identity = semantic_closure_route_identity(
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
    )
    request = build_paper_i_ra_strong_weak_always_k5_request(
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
    )
    protocol = preflight_paper_i_ra_strong_weak_always_k5(
        build_paper_i_ra_strong_weak_nph3_problem(),
        request,
    )

    assert protocol.algorithm_id == identity.algorithm_id
    assert protocol.route_contract is not None
    native = protocol.route_contract["native_semantic_contract"]
    assert native["semantic_implementation_version"] == (
        PAPER_I_RA_SEMANTIC_IMPLEMENTATION_VERSION_V2
    )
    assert native["phase0_policy"] == {
        "adaptive_shadow_receipt": False,
        "benefit": "absolute_append_endpoint_generator_gradient_v1",
        "fubini_study_metric": "off",
        "graph_proxy_cost": "paper_i_structural_graph_proxy_transform_v1",
        "population": "same_ordered_append_endpoint_generator_population_v1",
        "qiskit_compile": "off",
        "score": "absolute_append_gradient_over_graph_proxy_cost_v1",
        "shortlist": (
            "phase0_active_score_effective_competition_shortlist_v2"
        ),
    }
    assert native["phase0_adaptive_cap"] == 24


def test_exact_strong_weak_always_k5_request_round_trips_and_preflights() -> None:
    request = build_paper_i_ra_strong_weak_always_k5_request(
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2
    )
    restored = ra_adapt_request_from_mapping(request.to_dict())

    assert isinstance(
        restored.adapter,
        PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
    )
    assert restored.to_dict() == request.to_dict()
    assert isinstance(request.method.admission, SingletonAdmission)
    assert isinstance(request.method.insertion, AlwaysCommutationReducedInsertion)
    assert isinstance(request.method.pruning, PruningOff)
    assert isinstance(request.method.beam, BeamOff)
    assert isinstance(request.execution.resume, FreshStart)
    assert request.execution.stop.maximum_controller_rounds == 5
    assert request.execution.stop.exact_ed_target is None

    protocol = preflight_paper_i_ra_strong_weak_always_k5(
        build_paper_i_ra_strong_weak_nph3_problem(),
        restored,
    )
    assert protocol.problem.num_sites == 2
    assert protocol.problem.u == 8.0
    assert protocol.problem.n_ph_max == 3
    assert protocol.horizon == 5
    assert protocol.optimizer == "powell"
    assert protocol.optimizer_maxiter == 200
    assert protocol.seeds == {"adapt": 7, "transpiler": 7}
    assert protocol.execution_authorized is False
    assert protocol.estimator_accounting_convention == (
        "s_alg_equals_n_h_outer_plus_n_h_refit_plus_n_grad_plus_n_metric_v1"
    )
    assert protocol.route_contract is not None
    native = protocol.route_contract["native_semantic_contract"]
    assert native["qiskit_active_phases"] == [
        "phase_i",
        "phase_ii",
        "phase_iii",
    ]
    assert native["qiskit_full_trial_compile_semantics"] == (
        "full_base_and_trial_ansatz_at_recorded_insertion_position_v1"
    )
    assert native["hardware_cost_normalization"] == (
        "zero_centered_signed_arctan_v1"
    )
    assert native["negative_compile_delta_policy"] == (
        "negative_delta_is_reward_v1"
    )
    assert native["compile_work_in_s_alg"] is False


def test_semantic_route_rejects_adapter_version_and_request_drift() -> None:
    with pytest.raises(ValueError, match="semantic implementation version"):
        PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
            route_variant=PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
            semantic_implementation_version="stale_v0",
        )

    request = build_paper_i_ra_strong_weak_always_k5_request(
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2
    )
    drifted = replace(
        request,
        execution=replace(
            request.execution,
            stop=replace(
                request.execution.stop,
                maximum_controller_rounds=6,
            ),
        ),
    )
    with pytest.raises(ValueError, match="strong--weak always-open k=5"):
        preflight_paper_i_ra_strong_weak_always_k5(
            build_paper_i_ra_strong_weak_nph3_problem(),
            drifted,
        )


@pytest.mark.parametrize(
    ("route_variant", "route_sha256"),
    [
        (
            PAPER_I_RA_PHASE0_GRADIENT_FIXED24,
            "39afd3a2dc71c26978c0a7131c555e187f9c10f18947f77947ad611c19588e8f",
        ),
        (
            PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW,
            "7733bbd295622b068e17103c9d474e3fe3364a5178fbc7e5f1144fcd9aab79a0",
        ),
        (
            PAPER_I_RA_PHASE0_PROXY_ADAPTIVE,
            "475594d56335d32c2dd38ca11058e80a39e5e7866d3ecbbf7dcbeb392a9d16a8",
        ),
    ],
)
def test_retired_v1_route_contract_digests_remain_literal_regression_pins(
    route_variant: str,
    route_sha256: str,
) -> None:
    request = build_paper_i_ra_strong_weak_always_k5_request(route_variant)
    identity = semantic_closure_route_identity(route_variant)
    _, _, _, observed_sha256 = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        algorithm_id=identity.algorithm_id,
        problem=build_paper_i_ra_strong_weak_nph3_problem(),
    )

    assert observed_sha256 == route_sha256


def test_exact_preflight_binds_phase123_qiskit_and_native_engine_settings() -> None:
    protocol = preflight_paper_i_ra_strong_weak_always_k5(
        build_paper_i_ra_strong_weak_nph3_problem(),
        build_paper_i_ra_strong_weak_always_k5_request(
            PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
        ),
    )
    assert protocol.route_contract is not None
    execution = protocol.route_contract["execution_settings"]
    invariants = protocol.route_contract["semantic_invariants"]
    assert execution["phase3_backend_cost_scope"] == (
        BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1
    )
    assert execution["phase3_hardware_cost_normalization_mode"] == (
        "zero_centered_signed_arctan_v1"
    )
    assert execution["phase3_backend_cost_mode"] == "marrakesh_graph_span_v1"
    assert execution["phase3_backend_name"] == "FakeMarrakesh"
    assert execution["phase3_backend_optimization_level"] == 1
    assert execution["phase3_backend_transpile_seed"] == 7
    assert execution["adapt_parallel_gradient_workers"] == 4
    assert execution["static_lane_route"] == "global_single_population"
    assert "physical_lane_shortlist_aggressiveness" not in execution
    assert invariants["phase_i_compile_cost_source"] == "backend_transpile_v1"
    assert invariants["phase_ii_compile_cost_source"] == "backend_transpile_v1"
    assert invariants["phase_iii_compile_cost_source"] == "backend_transpile_v1"
    assert invariants[
        "phase_i_phase_ii_phase_iii_qiskit_negative_delta_reward_enabled"
    ] is True
    assert invariants["qiskit_compile_work_excluded_from_s_alg"] is True


def test_semantic_route_rejects_adapter_algorithm_mismatch() -> None:
    request = build_paper_i_ra_strong_weak_always_k5_request(
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2
    )
    mismatched_algorithm = semantic_closure_route_identity(
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
    ).algorithm_id

    with pytest.raises(ValueError, match="adapter and algorithm identity"):
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=mismatched_algorithm,
            problem=build_paper_i_ra_strong_weak_nph3_problem(),
        )


def test_historical_gradient_phase0_route_digest_is_unchanged() -> None:
    request = RAAdaptRequest(
        adapter=GlobalSingletonGradientPhase0CandidateAdapter(),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=PlateauCommutationInsertion(),
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=5),
            resume=FreshStart(),
        ),
        observation=SRObservationPolicy(),
    )
    _, profile, _, digest = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        algorithm_id=(
            RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID
        ),
        problem=build_paper_i_ra_strong_weak_nph3_problem(),
    )

    assert profile == (
        "paper_i_ra_adapt__single_pauli_word_v1__insertion_commutation_"
        "plateau_v2__global_guarded_singleton_phase_i__identity_phase_ii__"
        "stationary_source_response_v1__all_phase_resource_weighting_v1__"
        "global_singleton_abs_gradient_phase0_then_singleton_phase1_then_"
        "qiskit_phase2_phase3_no_lanes_v1"
    )
    assert digest == (
        "9811652b332b592bee048a8e5f3048972256abae186921ed7efea52bfd5f3dd8"
    )


def test_semantic_route_builders_are_exported_from_public_package() -> None:
    import pipelines.static_adapt.ra_adapt as public_ra

    assert public_ra.PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2 == (
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2
    )
    assert public_ra.PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2 == (
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2
    )
    assert public_ra.PAPER_I_RA_PHASE0_PROXY_FIXED24_V2 == (
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2
    )
    assert public_ra.PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2 == (
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
    )
    assert public_ra.PAPER_I_RA_PHASE0_V2_ROUTE_VARIANTS == {
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    }
    assert public_ra.PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1 == (
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1
    )
    assert (
        public_ra.PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
        == PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
    )
    assert public_ra.PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1 == (
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1
    )
    assert (
        public_ra.PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
        == PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
    )
    assert (
        public_ra.validate_semantic_phase3_natural_terminal_route_contract
        is validate_semantic_phase3_natural_terminal_route_contract
    )
    assert public_ra.build_paper_i_ra_all_phase_adaptive_request is (
        build_paper_i_ra_all_phase_adaptive_request
    )
    assert public_ra.build_paper_i_ra_all_phase_adaptive_natural_terminal_request is (
        build_paper_i_ra_all_phase_adaptive_natural_terminal_request
    )
    assert public_ra.build_paper_i_ra_all_phase_position_adaptive_request is (
        build_paper_i_ra_all_phase_position_adaptive_request
    )
    assert (
        public_ra.build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request
        is build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request
    )
    assert public_ra.build_paper_i_ra_hh_regime_problem is (
        build_paper_i_ra_hh_regime_problem
    )
    assert public_ra.PaperIRASemanticClosureGlobalSingletonCandidateAdapter is (
        PaperIRASemanticClosureGlobalSingletonCandidateAdapter
    )
    assert callable(public_ra.build_paper_i_ra_strong_weak_always_k5_request)
    assert callable(public_ra.preflight_paper_i_ra_strong_weak_always_k5)


def test_all_phase_v2_routes_are_the_only_natural_terminal_routes() -> None:
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    v1 = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_request(
            insertion_policy="append_only",
            maximum_controller_rounds=3,
        ),
    )
    v2 = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
            insertion_policy="append_only",
            maximum_controller_rounds=3,
        ),
    )

    assert PAPER_I_RA_PHASE3_NATURAL_TERMINAL_ROUTE_VARIANTS == {
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    }
    assert v1.algorithm_id != v2.algorithm_id
    assert v1.route_contract["route_id"] != v2.route_contract["route_id"]
    assert v1.route_contract["route_profile"] != v2.route_contract["route_profile"]
    assert v1.bundle_id != v2.bundle_id

    v1_route = v1.route_contract
    assert v1_route["native_semantic_contract"]["route_variant"] == (
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1
    )
    for surface, policy_key, horizon_key in (
        (
            v1_route["native_semantic_contract"],
            "phase3_no_positive_policy",
            "controller_horizon_policy",
        ),
        (
            v1_route["execution_settings"],
            "ra_phase3_no_positive_policy",
            "ra_controller_horizon_policy",
        ),
        (
            v1_route["semantic_invariants"],
            "phase3_no_positive_policy",
            "controller_horizon_policy",
        ),
    ):
        assert policy_key not in surface
        assert horizon_key not in surface

    v2_route = v2.route_contract
    assert v2_route["native_semantic_contract"]["route_variant"] == (
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
    )
    for surface, policy_key, horizon_key in (
        (
            v2_route["native_semantic_contract"],
            "phase3_no_positive_policy",
            "controller_horizon_policy",
        ),
        (
            v2_route["execution_settings"],
            "ra_phase3_no_positive_policy",
            "ra_controller_horizon_policy",
        ),
        (
            v2_route["semantic_invariants"],
            "phase3_no_positive_policy",
            "controller_horizon_policy",
        ),
    ):
        assert surface[policy_key] == (
            ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1
        )
        assert surface[horizon_key] == ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1


def test_position_all_phase_v1_route_contract_bytes_remain_compatible() -> None:
    protocol = materialize_paper_i_ra_semantic_protocol(
        build_paper_i_ra_hh_regime_problem("weak_weak"),
        build_paper_i_ra_all_phase_position_adaptive_request(
            insertion_policy="append_only",
            maximum_controller_rounds=50,
        ),
    )

    assert protocol.route_contract["sha256"] == (
        "143da003a995e2dda690e557314c7cf31a2fd30bdf97dd5ec1a38bf21ed30e09"
    )


def test_natural_terminal_route_authentication_rejects_v1_and_policy_tamper(
) -> None:
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    v1 = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_request(
            insertion_policy="append_only",
            maximum_controller_rounds=3,
        ),
    )
    v2 = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
            insertion_policy="append_only",
            maximum_controller_rounds=3,
        ),
    )

    v2_route = copy.deepcopy(dict(v2.route_contract))
    v2_sha256 = v2_route.pop("sha256")
    assert validate_semantic_phase3_natural_terminal_route_contract(
        v2_route,
        expected_route_contract_sha256=v2_sha256,
    ) == v2_route

    v1_route = copy.deepcopy(dict(v1.route_contract))
    v1_sha256 = v1_route.pop("sha256")
    with pytest.raises(ValueError, match="V2 natural-terminal route"):
        validate_semantic_phase3_natural_terminal_route_contract(
            v1_route,
            expected_route_contract_sha256=v1_sha256,
        )

    tampered = copy.deepcopy(v2_route)
    tampered["execution_settings"]["ra_controller_horizon_policy"] = (
        ADAPTIVE_HORIZON_POLICY_EXACT_TARGET_V1
    )
    with pytest.raises(ValueError, match="contract digest"):
        validate_semantic_phase3_natural_terminal_route_contract(
            tampered,
            expected_route_contract_sha256=v2_sha256,
        )


class _RuntimeCursor:
    def __init__(self) -> None:
        self.available_indices = {0, 1, 2}

    def selection_available_indices(self) -> set[int]:
        return set(self.available_indices)


class _RuntimeSession:
    def __init__(
        self,
        *,
        context: object,
        cursor: _RuntimeCursor,
        gradients: tuple[float, float, float] = (3.0, 2.99, 0.5),
    ) -> None:
        self.context = context
        self.cursor = cursor
        self.gradients = np.asarray(gradients, dtype=float)
        self.gradient_calls = 0
        self.occurrences: list[dict[str, object]] = []

    def _evaluate_default_candidate_gradient_surface(
        self,
        pending: object,
        *,
        consumer_scope: str,
    ) -> None:
        self.gradient_calls += 1
        pending.gradients[:] = self.gradients
        pending.grad_magnitudes[:] = np.abs(pending.gradients)
        self.occurrences.extend(
            {
                "component": "N_grad",
                "consumer_scope": consumer_scope,
                "sequence": index + 1,
                "primitive_id": f"gradient-{index}",
            }
            for index in range(3)
        )

    def _refresh_default_candidate_gradient_summaries(
        self,
        _pending: object,
    ) -> None:
        return None


class _GraphProxyOracle:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def estimate(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        pool_index = int(str(kwargs["candidate_term"].label)[1:])
        # Candidate 0 has the largest |g| but a much larger graph burden.
        cost = (100.0, 1.0, 1.0)[pool_index]
        return SimpleNamespace(
            c_hat_2q=cost,
            c_hat_d=cost,
            c_hat_1q=cost,
            c_hat_theta=1.0,
            hardware_cost_source="proxy_logical_ladder_span_v1",
            source_mode="proxy",
        )


class _ForbiddenQiskitOracle:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def estimate_insertion(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        raise AssertionError("Phase 0 must not invoke Qiskit.")


def _semantic_phase0_runtime_fixture(
    route_variant: str,
    *,
    gradients: tuple[float, float, float] = (3.0, 2.99, 0.5),
):
    request = build_paper_i_ra_strong_weak_always_k5_request(route_variant)
    protocol = preflight_paper_i_ra_strong_weak_always_k5(
        build_paper_i_ra_strong_weak_nph3_problem(),
        request,
    )
    assert protocol.route_contract is not None
    proxy = _GraphProxyOracle()
    qiskit = _ForbiddenQiskitOracle()
    context = SimpleNamespace(
        candidate_adapter=request.adapter,
        route_contract=protocol.route_contract,
        pool=tuple(SimpleNamespace(label=f"G{index}") for index in range(3)),
        compiled_pool=tuple(
            SimpleNamespace(terms=(object(),)) for _ in range(3)
        ),
        phase1_compile_oracle=proxy,
        phase3_backend_compile_oracle=qiskit,
        reoptimization_policy="windowed",
        reoptimization_window_size=2,
        reoptimization_window_topk=2,
        transition_services=SimpleNamespace(controller_noise_runtime=None),
    )
    cursor = _RuntimeCursor()
    session = _RuntimeSession(
        context=context,
        cursor=cursor,
        gradients=gradients,
    )
    cursor.estimator_call_ledger = SimpleNamespace(
        to_payload=lambda: {"occurrences": list(session.occurrences)}
    )
    pending = SimpleNamespace(
        phase0_gradient_shortlist_receipt=None,
        append_position=2,
        theta_logical_current=np.asarray([0.1, 0.2]),
        gradients=np.zeros(3, dtype=float),
        grad_magnitudes=np.zeros(3, dtype=float),
        available_sorted=[0, 1, 2],
        shortlist=[0, 1, 2],
        phase2_score_cfg_round=SimpleScoreConfig(),
    )
    transaction = adapt_pipeline._DefaultNoPruneSelectionTransaction(
        session=session,
        pending=pending,
    )
    domain = tuple(
        adapt_pipeline._CandidatePositionRecord(
            domain_record_id=f"g{pool_index}@{position}",
            generator_id=f"g{pool_index}",
            parent_generator_id=None,
            pool_index=pool_index,
            pool_label=f"G{pool_index}",
            insertion_position=position,
            symmetry_identity=f"sym-{pool_index}",
            lineage_identity=(f"g{pool_index}",),
        )
        for pool_index, position in (
            (0, 0),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 1),
        )
    )
    return session, pending, proxy, transaction, domain


@pytest.mark.parametrize(
    ("route_variant", "expected_shortlist", "expected_proxy_calls"),
    [
        (PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2, [0, 1, 2], 0),
        (PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2, [0, 1], 0),
        (PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1, [0, 1], 0),
        (PAPER_I_RA_PHASE0_PROXY_FIXED24_V2, [1, 0, 2], 3),
        (PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2, [1, 0], 3),
    ],
)
def test_v2_phase0_runtime_keeps_score_and_cardinality_axes_independent(
    route_variant: str,
    expected_shortlist: list[int],
    expected_proxy_calls: int,
) -> None:
    session, pending, proxy, transaction, domain = (
        _semantic_phase0_runtime_fixture(route_variant)
    )

    phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
        session=session,
        transaction=transaction,
        admissible_domain=domain,
    )

    assert phase0 is not None
    assert pending.shortlist == expected_shortlist
    assert session.gradient_calls == 1
    assert len(session.occurrences) == 3
    assert len(proxy.calls) == expected_proxy_calls
    assert transaction.context.phase3_backend_compile_oracle.calls == []
    receipt = pending.phase0_gradient_shortlist_receipt
    assert receipt.get(
        "qiskit_compile_cost_policy",
        receipt.get("compile_cost_policy"),
    ) == "off"
    assert receipt["metric_policy"] == "off"


def test_v2_fixed_and_adaptive_arms_share_each_score_ranking_exactly() -> None:
    observed: dict[str, list[int]] = {}
    full_rankings: dict[str, list[int]] = {}
    for route_variant in (
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    ):
        session, pending, _, transaction, domain = (
            _semantic_phase0_runtime_fixture(route_variant)
        )
        phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
            session=session,
            transaction=transaction,
            admissible_domain=domain,
        )
        assert phase0 is not None
        observed[route_variant] = [
            row.pool_index for row in phase0.shortlist_ranking
        ]
        receipt = pending.phase0_gradient_shortlist_receipt
        full_rankings[route_variant] = (
            list(receipt["ranked_pool_indices"])
            if "ranked_pool_indices" in receipt
            else [int(row["pool_index"]) for row in receipt["ranking"]]
        )

    assert observed[PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2] == [0, 0, 1, 1, 2]
    assert observed[PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2] == [0, 0, 1, 1]
    assert observed[PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1] == [0, 0, 1, 1]
    assert observed[PAPER_I_RA_PHASE0_PROXY_FIXED24_V2] == [1, 1, 0, 0, 2]
    assert observed[PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2] == [1, 1, 0, 0]
    assert full_rankings[PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2] == (
        full_rankings[PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2]
    )
    assert full_rankings[PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1] == (
        full_rankings[PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2]
    )
    assert full_rankings[PAPER_I_RA_PHASE0_PROXY_FIXED24_V2] == (
        full_rankings[PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2]
    )


@pytest.mark.parametrize(
    "route_variant",
    [
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    ],
)
def test_accepted_round_validator_dispatches_all_v2_phase0_receipts(
    route_variant: str,
) -> None:
    session, pending, _, transaction, domain = (
        _semantic_phase0_runtime_fixture(route_variant)
    )
    phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
        session=session,
        transaction=transaction,
        admissible_domain=domain,
    )
    assert phase0 is not None

    def project(record: object) -> dict[str, object]:
        return {
            "domain_record_id": record.domain_record_id,
            "generator_id": record.generator_id,
            "pool_index": record.pool_index,
            "pool_label": record.pool_label,
            "insertion_position": record.insertion_position,
            "position_class": (
                "interior"
                if record.insertion_position < pending.append_position
                else "append"
            ),
        }

    population = [project(row) for row in phase0.population]
    shortlist = [project(row) for row in phase0.shortlist]
    scored = {
        "phase0_gradient_screen": {
            "schema": "paper_i_scored_gradient_phase0_population_v1",
            "population_count": len(population),
            "population": population,
            "ordered_population_sha256": canonical_sha256(population),
            "shortlist_count": len(shortlist),
            "shortlist": shortlist,
            "ordered_shortlist_sha256": canonical_sha256(shortlist),
        }
    }
    identity = semantic_closure_route_identity(route_variant)

    assert ra_engine._validated_gradient_phase0_round_receipt(
        {
            "ra_gradient_phase0_shortlist": (
                pending.phase0_gradient_shortlist_receipt
            )
        },
        scored_population=scored,
        algorithm_id=identity.algorithm_id,
    ) == pending.phase0_gradient_shortlist_receipt


@pytest.mark.parametrize(
    ("route_variant", "adaptive_role"),
    [
        (PAPER_I_RA_PHASE0_PROXY_FIXED24_V2, "off"),
        (PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2, "active"),
    ],
)
def test_native_proxy_phase0_runs_the_actual_transaction_once_at_append_endpoint(
    route_variant: str,
    adaptive_role: str,
) -> None:
    session, pending, proxy, transaction, domain = (
        _semantic_phase0_runtime_fixture(route_variant)
    )

    phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
        session=session,
        transaction=transaction,
        admissible_domain=domain,
    )

    assert phase0 is not None
    assert session.gradient_calls == 1
    assert len(session.occurrences) == 3
    assert [call["position_id"] for call in proxy.calls] == [2, 2, 2]
    assert [call["append_position"] for call in proxy.calls] == [2, 2, 2]
    receipt = pending.phase0_gradient_shortlist_receipt
    assert receipt["route_variant"] == route_variant
    assert receipt["adaptive_decision_role"] == adaptive_role
    assert receipt["qiskit_compile_cost_policy"] == "off"
    assert receipt["metric_policy"] == "off"
    assert receipt["estimator_accounting"]["N_grad"] == 3
    assert receipt["adaptive_shadow_accounting"]["S_alg"] == 0
    assert receipt["score"] == (
        "absolute_append_gradient_over_graph_proxy_denominator_v1"
    )
    assert receipt["ranking_order"] == (
        "descending_absolute_gradient_over_graph_proxy_then_pool_index_v1"
    )
    if route_variant == PAPER_I_RA_PHASE0_PROXY_FIXED24_V2:
        assert receipt["adaptive_decision"] is None
    else:
        assert receipt["adaptive_decision"]["score"] == (
            "absolute_append_gradient_over_graph_proxy_cost_v1"
        )
    retained = set(receipt["retained_pool_indices"])
    assert {
        (row.pool_index, row.insertion_position) for row in phase0.shortlist
    } == {
        (row.pool_index, row.insertion_position)
        for row in domain
        if row.pool_index in retained
    }


def test_fixed24_proxy_uses_absolute_benefit_and_labels_c_shadow_separately() -> None:
    decision = semantic_routes.select_semantic_proxy_phase0_rows(
        [
            {
                "pool_index": 0,
                "append_gradient_signed": 2.0,
                "graph_proxy_denominator": 1.0,
            },
            {
                "pool_index": 1,
                "append_gradient_signed": 10.0,
                "graph_proxy_denominator": 10.0,
            },
        ],
        route_variant=PAPER_I_RA_PHASE0_PROXY_FIXED24_SHADOW,
        cap=2,
    )

    assert decision["active_score"] == (
        "absolute_append_gradient_over_graph_proxy_cost_v1"
    )
    assert decision["ranked_pool_indices"] == [0, 1]
    assert decision["adaptive_decision_role"] == "shadow"
    assert decision["adaptive_decision"]["score"] == (
        "squared_append_gradient_over_graph_proxy_cost_v1"
    )
    assert decision["adaptive_decision"]["ranked_generator_indices"] == [1, 0]


def test_proxy_phase0_selector_rejects_the_gradient_all_phase_route() -> None:
    rows = [
        {
            "pool_index": 0,
            "append_gradient_signed": 2.0,
            "graph_proxy_denominator": 1.0,
        },
        {
            "pool_index": 1,
            "append_gradient_signed": 10.0,
            "graph_proxy_denominator": 10.0,
        },
    ]
    with pytest.raises(ValueError, match="not a proxy"):
        semantic_routes.select_semantic_proxy_phase0_rows(
            rows,
            route_variant=PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
            cap=24,
        )


def test_all_phase_route_executes_native_gradient_phase0_without_cost_work() -> None:
    expected_runtime = _semantic_phase0_runtime_fixture(
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2
    )
    actual_runtime = _semantic_phase0_runtime_fixture(
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1
    )
    (
        expected_session,
        expected_pending,
        expected_proxy,
        expected_transaction,
        expected_domain,
    ) = expected_runtime
    (
        actual_session,
        actual_pending,
        actual_proxy,
        actual_transaction,
        actual_domain,
    ) = actual_runtime

    expected_phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
        session=expected_session,
        transaction=expected_transaction,
        admissible_domain=expected_domain,
    )
    actual_phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
        session=actual_session,
        transaction=actual_transaction,
        admissible_domain=actual_domain,
    )

    assert expected_phase0 is not None and actual_phase0 is not None
    assert actual_session.gradient_calls == 1
    assert len(actual_session.occurrences) == 3
    assert actual_proxy.calls == expected_proxy.calls == []
    assert actual_transaction.context.phase3_backend_compile_oracle.calls == []
    expected_receipt = expected_pending.phase0_gradient_shortlist_receipt
    actual_receipt = actual_pending.phase0_gradient_shortlist_receipt
    for key in (
        "score",
        "ranking_order",
        "adaptive_law",
        "ranked_pool_indices",
        "retained_pool_indices",
        "ranking",
        "adaptive_decision",
        "estimator_accounting",
        "graph_proxy_cost_policy",
        "qiskit_compile_cost_policy",
        "metric_policy",
    ):
        assert actual_receipt[key] == expected_receipt[key]
    assert actual_receipt["score"] == (
        "absolute_append_endpoint_generator_gradient_v1"
    )
    assert actual_receipt["graph_proxy_cost_policy"] == "off"
    assert actual_receipt["qiskit_compile_cost_policy"] == "off"
    assert actual_receipt["metric_policy"] == "off"


def test_native_adaptive_phase0_zero_surface_returns_clean_stationary_receipt() -> None:
    session, pending, proxy, transaction, domain = (
        _semantic_phase0_runtime_fixture(
            PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
            gradients=(0.0, 0.0, 0.0),
        )
    )

    phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
        session=session,
        transaction=transaction,
        admissible_domain=domain,
    )

    assert phase0 is not None
    assert phase0.phase == "phase0"
    assert phase0.population == domain
    assert phase0.shortlist == ()
    assert phase0.shortlist_ranking == ()
    assert phase0.terminal_outcome == (
        "phase0_stationary_no_competitive_candidate_v1"
    )
    assert pending.shortlist == []
    assert pending.phase0_gradient_shortlist_receipt["status"] == "stationary"
    assert pending.phase0_gradient_shortlist_receipt[
        "retained_candidate_count"
    ] == 0
    assert pending.phase0_gradient_shortlist_receipt[
        "terminal_controller_outcome"
    ] == phase0.terminal_outcome
    assert session.gradient_calls == 1
    assert len(session.occurrences) == 3
    assert len(proxy.calls) == 3
    assert transaction.context.phase3_backend_compile_oracle.calls == []


@pytest.mark.parametrize(
    "phase0_population",
    ["generator_first", "position_aware"],
)
def test_actual_selection_kernel_and_controller_stop_before_phase_i(
    monkeypatch: pytest.MonkeyPatch,
    phase0_population: str,
) -> None:
    if phase0_population == "generator_first":
        session, _, _, transaction, domain = (
            _semantic_phase0_runtime_fixture(
                PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
                gradients=(0.0, 0.0, 0.0),
            )
        )
    else:
        session, pending, _, transaction, domain = (
            _semantic_phase0_runtime_fixture(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1
            )
        )
        session.context.reference_state = np.asarray([1.0], dtype=complex)
        session.cursor.selected_ops = []
        session.cursor.pauli_action_cache = {}
        pending.psi_current = np.asarray([1.0], dtype=complex)
        pending.hpsi_current = np.asarray([1.0], dtype=complex)
        pending.phase2_score_cfg_round = SimpleNamespace(
            batch_state_consistency_tolerance=1.0e-10
        )
        monkeypatch.setattr(
            adapt_pipeline,
            "_prepare_exact_insertion_first_order_context",
            lambda **_kwargs: object(),
        )
        monkeypatch.setattr(
            adapt_pipeline,
            "_exact_insertion_first_order_candidate_geometry",
            lambda **_kwargs: {"energy_gradient": 0.0},
        )

        class ZeroPositionEstimator:
            def _candidate_physical_tangent(
                self,
                *_args: object,
                **kwargs: object,
            ) -> tuple[str, int]:
                return ("candidate", int(kwargs["insertion_position"]))

            def _record_estimator_primitive(
                self,
                **kwargs: object,
            ) -> object:
                index = len(session.occurrences) + 1
                primitive_id = f"zero-position-gradient-{index}"
                session.occurrences.append(
                    {
                        "component": kwargs["component"],
                        "consumer_scope": kwargs["consumer_scope"],
                        "sequence": index,
                        "primitive_id": primitive_id,
                    }
                )
                return SimpleNamespace(primitive_id=primitive_id)

        session.estimator_service = ZeroPositionEstimator()
    initial = _AcceptedStateSnapshot(
        controller_round=0,
        accepted_operator_ids=(),
        accepted_insertion_positions=(),
        logical_parameter_ids=(),
        logical_parameter_values=(),
        runtime_parameter_ids=(),
        runtime_parameter_values=(),
        accepted_energy=-0.5,
        accepted_state_fingerprint="state:initial",
        available_generator_ids=("g0", "g1", "g2"),
        selection_counts=(("g0", 0), ("g1", 0), ("g2", 0)),
        trust_state_identity="trust:initial",
        optimizer_memory_identity="optimizer:initial",
        estimator_prefix_identity="ledger:initial",
    )
    controller_state = _SRControllerState(
        controller_round=0,
        accepted_operator_ids=(),
        accepted_insertion_positions=(),
        logical_parameter_ids=(),
        logical_parameter_values=(),
        runtime_parameter_ids=(),
        runtime_parameter_values=(),
        accepted_energy=-0.5,
        accepted_state_fingerprint="state:initial",
        available_generator_ids=("g0", "g1", "g2"),
        selection_counts=(("g0", 0), ("g1", 0), ("g2", 0)),
        phase_live=(True, True, True),
        trust_state_identity="trust:initial",
        optimizer_memory_identity="optimizer:initial",
        estimator_prefix_identity="ledger:initial",
        admissible_domain_record_ids=tuple(
            row.domain_record_id for row in domain
        ),
    )

    def forbidden_phase(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Phase I was entered after stationary Phase0")

    kernel = adapt_pipeline._DefaultSingletonSelectionKernel(
        phases=adapt_pipeline._DefaultSelectionPhaseRunners(
            gradient_surface=forbidden_phase,
            phase_i=forbidden_phase,
            phase_ii=forbidden_phase,
            projected_phase_iii=forbidden_phase,
            supported_response=forbidden_phase,
            record_phase3_work=forbidden_phase,
            phase0=lambda rows: adapt_pipeline._run_global_singleton_gradient_phase0(
                session=session,
                transaction=transaction,
                admissible_domain=rows,
            ),
        ),
        receipts=adapt_pipeline._DefaultSelectionReceiptAdapters(
            record_from_live=forbidden_phase,
            shortlist_ranks=forbidden_phase,
            ledger_occurrences=lambda: list(session.occurrences),
            restore_singleton_coordinates=forbidden_phase,
            phase1_score_key="phase1",
            phase3_score_key="phase3",
            phase3_tie_break_score_key="phase3_tie",
            coordinate_solve_policy="fixture",
        ),
        runtime=adapt_pipeline._DefaultSelectionRuntime(
            expected_domain=domain,
            accepted_state_snapshotter=lambda: initial,
            sidecar={},
        ),
    )

    class Runtime:
        initial_accepted_state = initial
        closed = False

        def prepare_selection(self, _state: object) -> object:
            return sr_controller._PreparedSelection(
                controller_state=controller_state,
                workspace=_SelectionWorkspace(
                    admissible_records=domain,
                    kernel=kernel,
                ),
            )

        def finalize_stationary_phase0(self, **kwargs: object) -> object:
            assert kwargs["final_state"] is initial
            assert kwargs["transitions"] == ()
            return sr_controller._DefaultControllerFinalization.from_mapping(
                {
                    "success": True,
                    "route_family": "ra_adapt",
                    "route_profile": "semantic-fixture",
                    "sr_route_profile_contract": {},
                    "sr_route_profile_contract_sha256": "fixture",
                    "history": [],
                    "estimator_call_accounting": {},
                    "continuation": {},
                    "terminal_controller_outcome": (
                        "phase0_stationary_no_competitive_candidate_v1"
                    ),
                    "terminal_phase0_selection_receipt": dict(
                        transaction.pending.phase0_gradient_shortlist_receipt
                    ),
                }
            )

        def close(self) -> None:
            self.closed = True

    runtime = Runtime()
    outcome = sr_controller._run_default_singleton_controller(
        runtime,
        SRStopPolicy(maximum_controller_rounds=5),
    )

    assert outcome.accepted_states == ()
    assert outcome.transitions == ()
    assert outcome.stop.primary_reason == "phase0_stationary"
    terminal_receipt = outcome.finalization.to_serialization_mapping()[
        "terminal_phase0_selection_receipt"
    ]
    assert terminal_receipt["status"] == "stationary"
    if phase0_population == "position_aware":
        assert terminal_receipt["position_aware_gradient_surface"] is True
        assert session.gradient_calls == 0
        assert len(session.occurrences) == len(domain)
    assert runtime.closed is True


def test_native_gradient_phase0_uses_same_transaction_without_graph_or_qiskit() -> None:
    session, pending, proxy, transaction, domain = (
        _semantic_phase0_runtime_fixture(PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2)
    )

    phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
        session=session,
        transaction=transaction,
        admissible_domain=domain,
    )

    assert phase0 is not None
    assert session.gradient_calls == 1
    assert proxy.calls == []
    assert pending.phase0_gradient_shortlist_receipt["compile_cost_policy"] == "off"
    assert pending.phase0_gradient_shortlist_receipt["metric_policy"] == "off"
    assert pending.shortlist == [0, 1, 2]


def test_native_fixed24_ablation_keeps_one_ordered_population_but_changes_rank() -> None:
    gradient_runtime = _semantic_phase0_runtime_fixture(
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2
    )
    proxy_runtime = _semantic_phase0_runtime_fixture(
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2
    )
    gradient_session, gradient_pending, _, gradient_transaction, gradient_domain = (
        gradient_runtime
    )
    proxy_session, proxy_pending, _, proxy_transaction, proxy_domain = proxy_runtime

    gradient_phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
        session=gradient_session,
        transaction=gradient_transaction,
        admissible_domain=gradient_domain,
    )
    proxy_phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
        session=proxy_session,
        transaction=proxy_transaction,
        admissible_domain=proxy_domain,
    )
    assert gradient_phase0 is not None and proxy_phase0 is not None

    project = lambda record: (  # noqa: E731 - compact identity projection
        record.domain_record_id,
        record.generator_id,
        record.pool_index,
        record.pool_label,
        record.insertion_position,
    )
    assert [project(row) for row in gradient_phase0.population] == [
        project(row) for row in proxy_phase0.population
    ]
    gradient_by_pool = {
        int(row["pool_index"]): float(row["gradient_signed"])
        for row in gradient_pending.phase0_gradient_shortlist_receipt["ranking"]
    }
    proxy_by_pool = {
        int(row["pool_index"]): float(row["append_gradient_signed"])
        for row in proxy_pending.phase0_gradient_shortlist_receipt["ranking"]
    }
    assert gradient_by_pool == proxy_by_pool
    gradient_rank = [
        row.pool_index for row in gradient_phase0.shortlist_ranking
    ]
    proxy_rank = [row.pool_index for row in proxy_phase0.shortlist_ranking]
    assert gradient_rank[:2] == [0, 0]
    assert proxy_rank[:2] == [1, 1]
    assert (
        gradient_transaction.context.phase3_backend_compile_oracle.calls
        == []
    )
    assert proxy_transaction.context.phase3_backend_compile_oracle.calls == []


def test_fixed24_ablation_keeps_the_same_zero_and_tiny_gradient_domain() -> None:
    gradients = (0.0, 1.0e-150, 0.0)
    gradient_runtime = _semantic_phase0_runtime_fixture(
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
        gradients=gradients,
    )
    proxy_runtime = _semantic_phase0_runtime_fixture(
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
        gradients=gradients,
    )
    gradient_session, _, _, gradient_transaction, gradient_domain = (
        gradient_runtime
    )
    proxy_session, proxy_pending, _, proxy_transaction, proxy_domain = (
        proxy_runtime
    )

    gradient_phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
        session=gradient_session,
        transaction=gradient_transaction,
        admissible_domain=gradient_domain,
    )
    proxy_phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
        session=proxy_session,
        transaction=proxy_transaction,
        admissible_domain=proxy_domain,
    )

    assert gradient_phase0 is not None and proxy_phase0 is not None
    domain_identity = lambda record: (  # noqa: E731
        record.domain_record_id,
        record.pool_index,
        record.insertion_position,
    )
    expected_domain = {domain_identity(record) for record in gradient_domain}
    assert len(gradient_phase0.shortlist) == len(gradient_domain)
    assert len(proxy_phase0.shortlist) == len(proxy_domain)
    assert {domain_identity(record) for record in gradient_phase0.shortlist} == (
        expected_domain
    )
    assert {domain_identity(record) for record in proxy_phase0.shortlist} == (
        expected_domain
    )
    assert proxy_pending.phase0_gradient_shortlist_receipt[
        "retained_candidate_count"
    ] == 3


def test_semantic_native_materialization_binds_source_inventory_and_authority() -> None:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    request = build_paper_i_ra_strong_weak_always_k5_request(
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
    )

    protocol = materialize_paper_i_ra_strong_weak_always_k5_protocol(
        problem,
        request,
    )

    inventory = semantic_closure_source_implementation_inventory(
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
    )
    assert protocol.execution_authorized is False
    assert protocol.bundle_id == (
        "paper_i_ra_phase0_placement_score_cardinality_matrix_phase123_"
        "qiskit_native_v1"
    )
    assert protocol.bundle_id == (
        semantic_routes.PAPER_I_RA_SEMANTIC_NATIVE_EIGHT_ARM_BUNDLE_ID_V1
    )
    assert semantic_routes.semantic_closure_native_bundle_manifest(
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24
    )["bundle_id"] == semantic_routes.PAPER_I_RA_SEMANTIC_NATIVE_BUNDLE_ID
    assert protocol.bundle_materialization is not None
    assert protocol.bundle_materialization.algorithm_id == protocol.algorithm_id
    assert protocol.source_locks[
        "implementation_source_inventory_sha256"
    ] == inventory["sha256"]
    assert protocol._materialization_authority is not None
    assert protocol._materialization_authority.protocol_sha256 == protocol.sha256

    restored = resolved_ra_adapt_protocol_from_mapping(protocol.to_dict())
    assert restored.to_dict() == protocol.to_dict()
    assert restored._materialization_authority is None
    with pytest.raises(ValueError, match="loaded through ra_adapt.bundles"):
        ra_engine.run_ra_adapt(problem, restored)


def test_all_phase_adaptive_route_materializes_new_deep_shortlist_identity() -> None:
    assert PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1 == (
        "gradient_only_adaptive_shortlist_phase123_adaptive_v1"
    )
    assert (
        "structural_proxy_cost_adaptive_shortlist_phase123_adaptive_v1"
        not in semantic_routes.PAPER_I_RA_SEMANTIC_ROUTE_VARIANTS
    )
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    request = build_paper_i_ra_strong_weak_plateau_k5_request(
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1
    )

    protocol = materialize_paper_i_ra_semantic_protocol(problem, request)
    native = protocol.route_contract["native_semantic_contract"]
    execution = protocol.route_contract["execution_settings"]

    assert protocol.execution_authorized is False
    assert protocol.route_contract["semantic_implementation_version"] == (
        "paper_i_ra_all_phase_adaptive_gradient_phase0_phase123_qiskit_"
        "semantic_closure_v1"
    )
    assert native["phase0_policy"]["score"] == (
        "absolute_append_endpoint_generator_gradient_v1"
    )
    assert native["phase0_policy"]["graph_proxy_cost"] == "off"
    assert native["phase0_policy"]["shortlist"] == (
        "phase0_active_score_effective_competition_shortlist_v2"
    )
    assert native["phase123_shortlist_policy"] == (
        ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1
    )
    assert native["phase_shortlist_maxima"] == {
        "phase_i": 24,
        "phase_ii": 12,
        "phase_iii": 12,
    }
    assert native["phase_frontier_ratio_role"] == "eligibility_only"
    assert native["phase_frontier_ratios"] == {
        "phase_i": 0.9,
        "phase_ii": 0.9,
        "phase_iii": 0.9,
    }
    assert execution["ra_phase123_shortlist_policy"] == (
        ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1
    )


def test_all_phase_position_adaptive_route_materializes_distinct_placement_identity() -> None:
    assert PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1 == (
        "position_records_gradient_only_adaptive_shortlist_phase123_"
        "adaptive_v1"
    )
    assert PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1 != (
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1
    )
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    request = build_paper_i_ra_all_phase_position_adaptive_request(
        insertion_policy="plateau_commutation",
        maximum_controller_rounds=5,
    )

    protocol = materialize_paper_i_ra_semantic_protocol(problem, request)
    native = protocol.route_contract["native_semantic_contract"]
    phase0 = native["phase0_policy"]

    assert protocol.execution_authorized is False
    assert request.adapter.route_variant == (
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1
    )
    assert protocol.bundle_id == (
        "paper_i_ra_all_phase_adaptive_position_gradient_phase0_phase123_"
        "qiskit_native_v1"
    )
    assert phase0 == {
        "population": (
            "current_commutation_reduced_candidate_position_records_v1"
        ),
        "benefit": "absolute_position_record_gradient_v1",
        "fubini_study_metric": "off",
        "qiskit_compile": "off",
        "graph_proxy_cost": "off",
        "score": "absolute_position_record_gradient_v1",
        "shortlist": "phase0_active_score_effective_competition_shortlist_v2",
        "adaptive_shadow_receipt": False,
        "placement_activation": (
            "append_record_when_closed_full_commutation_reduced_records_when_open_v1"
        ),
        "generator_level_reexpansion_after_phase0": False,
    }
    assert native["phase123_shortlist_policy"] == (
        ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1
    )
    assert protocol.route_contract["execution_settings"][
        "ra_phase123_shortlist_policy"
    ] == ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1
    assert native["qiskit_active_phases"] == [
        "phase_i",
        "phase_ii",
        "phase_iii",
    ]


def test_all_phase_adaptive_six_regime_public_builders_are_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_settings = {
        "weak_weak": (0.25, math.sqrt(0.125), 3),
        "intermediate_weak": (1.25, math.sqrt(0.125), 3),
        "strong_weak_u8": (8.0, math.sqrt(0.125), 3),
        "weak_strong": (0.25, math.sqrt(0.625), 7),
        "intermediate_strong": (1.25, math.sqrt(0.625), 7),
        "strong_strong_u8": (8.0, math.sqrt(0.625), 7),
    }

    def _capture(request: ProblemRequest) -> SimpleNamespace:
        return SimpleNamespace(request=request)

    monkeypatch.setattr(semantic_routes, "resolve_problem_context", _capture)
    for regime_id in PAPER_I_RA_CANONICAL_REGIME_IDS:
        monkeypatch.setattr(
            semantic_routes,
            "_canonical_paper_i_hh_regime_id",
            lambda _problem, value=regime_id: value,
        )
        problem = build_paper_i_ra_hh_regime_problem(regime_id)
        expected_u, expected_g, expected_nph = expected_settings[regime_id]
        assert problem.request.u == expected_u
        assert problem.request.g_ep == expected_g
        assert problem.request.n_ph_max == expected_nph

    for insertion_policy, expected_kind in (
        ("append_only", "append_only"),
        ("plateau_commutation", "plateau_commutation"),
    ):
        request = build_paper_i_ra_all_phase_adaptive_request(
            insertion_policy=insertion_policy,
            maximum_controller_rounds=15,
        )
        assert request.method.insertion.kind == expected_kind
        assert request.execution.stop.maximum_controller_rounds == 15


def test_semantic_materialization_digests_are_cross_process_deterministic() -> None:
    script = r"""
import json

from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
    PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
    PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
    PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
    PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1,
    PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
    PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
    PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
    build_paper_i_ra_strong_weak_always_k5_request,
    build_paper_i_ra_strong_weak_plateau_k5_request,
    build_paper_i_ra_strong_weak_nph3_problem,
    materialize_paper_i_ra_semantic_protocol,
    semantic_closure_materialization_contract,
)

problem = build_paper_i_ra_strong_weak_nph3_problem()
rows = {}
for variant in (
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
    PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
    PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1,
    PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
    PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
    PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
    PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
    PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
):
    request = build_paper_i_ra_strong_weak_plateau_k5_request(variant)
    contract = semantic_closure_materialization_contract(problem, request)
    protocol = materialize_paper_i_ra_semantic_protocol(problem, request)
    rows[variant] = {
        "bundle_id": protocol.bundle_id,
        "problem_scientific_content_sha256": contract[
            "problem_scientific_content_sha256"
        ],
        "source_implementation_inventory_sha256": contract[
            "source_implementation_inventory_sha256"
        ],
        "bundle_manifest_sha256": contract["bundle_manifest_sha256"],
        "materialization_contract_sha256": contract["sha256"],
        "route_sha256": protocol.route_contract["sha256"],
        "protocol_sha256": protocol.sha256,
        "materialization_receipt_sha256": protocol.bundle_materialization.sha256,
    }
print(json.dumps(rows, sort_keys=True))
"""
    run = lambda: json.loads(  # noqa: E731 - one exact fresh-process probe
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )

    first = run()
    second = run()

    _superseded_four_arm_expected = {
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2: {
            "bundle_id": (
                "paper_i_ra_phase0_score_cardinality_matrix_phase123_"
                "qiskit_native_v2"
            ),
            "problem_scientific_content_sha256": (
                "b54c65e40eddfd22c6674cdb3fc7817e889519a27ecec548ef02c600cf7ff944"
            ),
            "source_implementation_inventory_sha256": (
                "21c4de2fd8adc07c9e5325879236927f361dc2d447622849f10a6ed75f91448d"
            ),
            "bundle_manifest_sha256": (
                "4e856dcffe8d68efc449c7e5d4c790178aeec4b8d17f804b58f40c38989aacab"
            ),
            "materialization_contract_sha256": (
                "87a375c90c06f28a509c271c10211bc9640753fa12f3ded666a9e7e9420740ef"
            ),
            "route_sha256": (
                "44547b1c940fbbe4c4be290fcf7071295a5f1d3f2c43c908219472a9adec39b0"
            ),
            "protocol_sha256": (
                "46e852877383ff24c2f1aa64e42aea13dcd31ac5a177445eca64e80ecf200f3b"
            ),
            "materialization_receipt_sha256": (
                "593bc8bf97ecfcfb177be80ac55794e5c8484f86aaaced20597929fdf065e0c9"
            ),
        },
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2: {
            "bundle_id": (
                "paper_i_ra_phase0_score_cardinality_matrix_phase123_"
                "qiskit_native_v2"
            ),
            "problem_scientific_content_sha256": (
                "b54c65e40eddfd22c6674cdb3fc7817e889519a27ecec548ef02c600cf7ff944"
            ),
            "source_implementation_inventory_sha256": (
                "21c4de2fd8adc07c9e5325879236927f361dc2d447622849f10a6ed75f91448d"
            ),
            "bundle_manifest_sha256": (
                "7b381995731e835d5a804de5302da31d510b11b9a753284e82e870259d36a4a1"
            ),
            "materialization_contract_sha256": (
                "79f8b551c487123ed3436ad15b67a0546149afd8f354b7ef6572b8399352d58a"
            ),
            "route_sha256": (
                "37211fe39461d75e988ddbc6bce29d8c54645fea47d3f30ff393c9c3258738c8"
            ),
            "protocol_sha256": (
                "8b460ac39ce5f81d5f2b8cf30d185934364e0d4eeac4a052e0c5d740f876c922"
            ),
            "materialization_receipt_sha256": (
                "970b3785da33bd2393a837490048c3188dfd02691d5ac01ce186a9bbd8c96c3e"
            ),
        },
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2: {
            "bundle_id": (
                "paper_i_ra_phase0_score_cardinality_matrix_phase123_"
                "qiskit_native_v2"
            ),
            "problem_scientific_content_sha256": (
                "b54c65e40eddfd22c6674cdb3fc7817e889519a27ecec548ef02c600cf7ff944"
            ),
            "source_implementation_inventory_sha256": (
                "21c4de2fd8adc07c9e5325879236927f361dc2d447622849f10a6ed75f91448d"
            ),
            "bundle_manifest_sha256": (
                "c682c7adb347d9f307f3906fc8d3a3c774033207708a3e3ea253703be80b6ad6"
            ),
            "materialization_contract_sha256": (
                "7428bbff1416ae4204949207105b56e66e72d7bfb9ad028efefd1b518f8ac0a9"
            ),
            "route_sha256": (
                "e5d62f074b6f1acdf47194bae9a5cfe976c6a645ffba8683337c52192e0019a6"
            ),
            "protocol_sha256": (
                "a9913dd5c6daa85b40492300ca3cf885da696779cd4fba8a41651db6928d4b04"
            ),
            "materialization_receipt_sha256": (
                "15f27291423b5c4291e0b3d2eadf591e620d2192a55365df4e2c62c0fdd9ce9f"
            ),
        },
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2: {
            "bundle_id": (
                "paper_i_ra_phase0_score_cardinality_matrix_phase123_"
                "qiskit_native_v2"
            ),
            "problem_scientific_content_sha256": (
                "b54c65e40eddfd22c6674cdb3fc7817e889519a27ecec548ef02c600cf7ff944"
            ),
            "source_implementation_inventory_sha256": (
                "21c4de2fd8adc07c9e5325879236927f361dc2d447622849f10a6ed75f91448d"
            ),
            "bundle_manifest_sha256": (
                "6a8459e2e80597ecaee483d44213f3c8db988764769c6436869af17fb6410187"
            ),
            "materialization_contract_sha256": (
                "05c81f9f5577bbc45a3ee036aa0e617bef45b25421cd92e1b47e6b36c62548bd"
            ),
            "route_sha256": (
                "5be47cfd77cf2f73a36cb02b9ddc3e9023dad897eb147b9c9e9728602843ba6f"
            ),
            "protocol_sha256": (
                "236e9ddc2467761d700a8a40abc11273640ec871dfb2cf542539f3dcda4d2caa"
            ),
            "materialization_receipt_sha256": (
                "89205b6ab4c0fa1405207144317784b19ac61620d54668eb340bfc40e6356341"
            ),
        },
    }
    common = {
        "bundle_id": (
            "paper_i_ra_phase0_placement_score_cardinality_matrix_phase123_"
            "qiskit_native_v1"
        ),
        "problem_scientific_content_sha256": (
            "b54c65e40eddfd22c6674cdb3fc7817e889519a27ecec548ef02c600cf7ff944"
        ),
        "source_implementation_inventory_sha256": (
            "fd6fa1549ed19bd2de55bce697e5ef5f3adc6be573fef3b7c784634baeaeee62"
        ),
    }
    expected = {
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1: {
            **common,
            "bundle_id": (
                "paper_i_ra_all_phase_adaptive_gradient_phase0_phase123_"
                "qiskit_native_v1"
            ),
            "bundle_manifest_sha256": "6b8e9aae2ed5ee97ad40bb9392f8daf9bc05627053de24893655f146d5d797ee",
            "materialization_contract_sha256": "05d9ed7aac4e5fdffbe9bee02e8fe0a3f19d7ceb5e16ad055dab0910af4bb814",
            "route_sha256": "22216475b33fbca88cb4e4f2371d6f77eb2ebfad37c7b84738943dcea8d4dfec",
            "protocol_sha256": "93823333e74ee3643e74ce9fb092ce4cc94f0f82b0288be681552e829584dffe",
            "materialization_receipt_sha256": "81d91055d675361b5b674d4b1808d53e91fb7aebd6511124570642833df2f8fd",
        },
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1: {
            **common,
            "bundle_id": (
                "paper_i_ra_all_phase_adaptive_position_gradient_phase0_"
                "phase123_qiskit_native_v1"
            ),
            "bundle_manifest_sha256": "bd300469b6d0c84d0f6f25b6b3794c9f0170adc8c24161291f8992f7883073b0",
            "materialization_contract_sha256": "c503bc0d62859a9929418cde90081a7add8b2e706cfd05c0cdbb6d08c0be0892",
            "route_sha256": "23e1c4e3c1ccdc84ed6f81836876047357715a35e3c93de896b31e1d9d7b44fc",
            "protocol_sha256": "e2a820650311243acc5c8e0184b4133ce3c559de83e5bbb69ef045085a8a7e6c",
            "materialization_receipt_sha256": "3d83bbc195249b02e9720171514e169a30788a75239c3e4fff74a10532c70377",
        },
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2: {
            **common,
            "bundle_manifest_sha256": "45f14341e085076d6a4c1bab94f17db5cb2a84808f57c8b2fe931a518f55dca8",
            "materialization_contract_sha256": "73d8d35cc0e340730f274a086573bb28e5ac9e462d09ff0adae6c1757c68da63",
            "route_sha256": "bff7d4768674a159f428e9f280f6132051f67a64877c1ff49e9ac06991aff6a0",
            "protocol_sha256": "ef59ce901afda55ac609dd9b36dc9f3e391e35ee6f8669c7b6388b421f7eebcc",
            "materialization_receipt_sha256": "dfe0ded706c64e919eb11e1d369555df55b0cb3b6d5d182ce1a6b57782c1c007",
        },
        PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1: {
            **common,
            "bundle_manifest_sha256": "92f9f0751d8b15013e3f9518355bb4e074f280dee05f77aedd5780bdddb69fcb",
            "materialization_contract_sha256": "642d35aad8b32127289afe23c1392cd80ce2647f4708101b30608bf9ea1402d4",
            "route_sha256": "bb674cb929be8a9fe501571e79c3115fa7eb8e18c63fc70c99497565a22fd953",
            "protocol_sha256": "4a2338178c776b328acb735252b1c104e0f675bc44fc7fcc145aa26315882432",
            "materialization_receipt_sha256": "5c7556100bd6e044523d7b6687de9a5aef4155d3578d2ea141cc9e6276e106e3",
        },
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2: {
            **common,
            "bundle_manifest_sha256": "d8d11c0c775e41a9a80492597928f7780c2c9a247f1c4fc73e808d59a1e16f82",
            "materialization_contract_sha256": "f596370051fb57e43836d663bcbaa9eb59fccd10f5f78a009a7ee8970c46b8f2",
            "route_sha256": "76473bf1904dad75f08de934b8e429acef9192090165d1b058f4ea4b8f97b9b2",
            "protocol_sha256": "599b55e54a6e1333586f0aa7048f896078d7311455da25c270f0e6c47b6be57f",
            "materialization_receipt_sha256": "f24df2732c441ea8416ce9c9d3f502e4ba7497549607ed53992af1150adfe916",
        },
        PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1: {
            **common,
            "bundle_manifest_sha256": "85fe9d30761ac294617fc1d82b0fc3b89a0c8395d8b829b7dc929f43dbe1741a",
            "materialization_contract_sha256": "7f0c22ab1460eef403c16aec9272e5b544576c5c2ef7bcafb1eedaa80be2743c",
            "route_sha256": "1d9c1d7ec26eed627503949799b29ce7a11acdaf90204a1158ff02a6d0a419e7",
            "protocol_sha256": "013eebbe18a99334ddc7f6e6697f3205ca538661eae4b8560074cac6d003f061",
            "materialization_receipt_sha256": "2579954e30c24e53b017720a61fbcbd9b481b8004f6dc8f0a8e41184a2729e00",
        },
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2: {
            **common,
            "bundle_manifest_sha256": "5c3ef98cb76ec574ac939d993f1123b609f39a171658ddbe0556f936e27c569e",
            "materialization_contract_sha256": "025c346e5b974ed9146fe71d59d60abeb8f6b512006f66498756e445a799c2bf",
            "route_sha256": "c83bb7c064e7d5bfedf31c70691a0da141e1617de22bb61117b41a9775cf224b",
            "protocol_sha256": "eedc92c5a4c4c07f2a77b029324563e0100cf075e692ef4c9febf0db1b02b523",
            "materialization_receipt_sha256": "d44edc2597f655c9d0a7fa48f912842e948b928d82acfa418ddf6c5d258242ea",
        },
        PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1: {
            **common,
            "bundle_manifest_sha256": "3ccab1754371f592d494a5f0350f780d73fc49a3969104b2f17f44e60ca71b83",
            "materialization_contract_sha256": "88156af75a121bbfddb042085f8a049bc918b3f2d4ca1cec05d0b0cff5b75fd6",
            "route_sha256": "051e6d69e2823fa47c85b9bf4f83067fe1aaf43ee0ab3db51041ef0517736c99",
            "protocol_sha256": "9c8c256df0629cfa6ecd20301ad59647ec12688b06a429e67d496b5c5d8660a8",
            "materialization_receipt_sha256": "578c592c6790ec017eda84abd8fa09c418af5186825e749b6a9990ba9542157a",
        },
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2: {
            **common,
            "bundle_manifest_sha256": "94cbf5764d7eb0a8f43bdd2ad0c6357c18cf838b5186184a4805d03b0978016c",
            "materialization_contract_sha256": "d81dd0f2d67c7485150e19f6506eaf6b7db8c206b8f1668653ef2dc54b5e68db",
            "route_sha256": "119f4b3c6c2a4bbb848e9e1169a47fa24d908a7eabc1058480afe7a95af907a3",
            "protocol_sha256": "914bce4c486cd6c4d06a6a4640c25e4dc37357c1c2c2dd001fa1998d9596ba73",
            "materialization_receipt_sha256": "465790b0352afec3dcaf2625157f741b4744f8e975ee9263a6abe5b2cf8bc8fc",
        },
        PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1: {
            **common,
            "bundle_manifest_sha256": "e2d9b927ab299859746ecbfdf035bae6be326cd8ea2e35f3b70969a87572c960",
            "materialization_contract_sha256": "7e2d9ad3ca082c9513d6599b5814616655f341e45747e7f76f19187ab33addf9",
            "route_sha256": "12e08e8f82fd861623494fa827f036c276d9e479f94fe00b322eb3ccb3123758",
            "protocol_sha256": "d6a903ef84259fbf6fa65457fba72cf59939a6e505575a6c15b76548f871ae78",
            "materialization_receipt_sha256": "0ae55dce70289e79fd859406ac3dce67d1f49081904bdb2ae28bd04b18b98997",
        },
    }
    expected_variants = {
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_MIN_FLOORS_V1,
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
        PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1,
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
        PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
        PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
        PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
    }
    assert first == second
    assert set(first) == expected_variants
    assert {
        row["source_implementation_inventory_sha256"]
        for row in first.values()
    } == {"456d7a3521ef0f2092106030834fc0a686af9e9dfe3b12187a3e1cca041e7571"}
    assert canonical_sha256(first) == (
        "07262da34484ddb78ff0067d04751c5d0e45c5110402f2204ee4ba6680665c7b"
    )


def test_semantic_materialization_ignores_supplied_callbacks_without_science() -> None:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    request = build_paper_i_ra_strong_weak_always_k5_request(
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
    )
    baseline = materialize_paper_i_ra_semantic_protocol(problem, request)
    callback_calls = {"reference": 0, "exact": 0, "fallback": 0}

    def forbidden_reference() -> np.ndarray:
        callback_calls["reference"] += 1
        raise AssertionError("materialization executed reference science")

    altered_reference = replace(
        problem,
        reference_state=replace(
            problem.reference_state,
            build_state=forbidden_reference,
        ),
    )

    reference_protocol = materialize_paper_i_ra_semantic_protocol(
        altered_reference,
        request,
    )
    assert reference_protocol.to_dict() == baseline.to_dict()
    canonical_reference = semantic_routes.canonical_semantic_execution_problem(
        altered_reference
    )
    assert canonical_reference.reference_state.build_state is not (
        forbidden_reference
    )
    assert callback_calls["reference"] == 0

    def forbidden_exact(*_args: object, **_kwargs: object) -> float:
        callback_calls["exact"] += 1
        raise AssertionError("materialization executed exact-target science")

    altered_exact_target = resolve_problem_context(
        problem.request,
        exact_energy_impl=forbidden_exact,
    )
    exact_protocol = materialize_paper_i_ra_semantic_protocol(
        altered_exact_target,
        request,
    )
    assert exact_protocol.to_dict() == baseline.to_dict()
    canonical_exact = semantic_routes.canonical_semantic_execution_problem(
        altered_exact_target
    )
    assert canonical_exact.exact_target.resolve_energy is not (
        altered_exact_target.exact_target.resolve_energy
    )
    assert callback_calls["exact"] == 0

    def forbidden_fallback() -> np.ndarray:
        callback_calls["fallback"] += 1
        raise AssertionError("materialization executed fallback science")

    altered_fallback = replace(
        problem,
        exact_target=replace(
            problem.exact_target,
            build_fallback_anchor_state=forbidden_fallback,
        ),
    )
    fallback_protocol = materialize_paper_i_ra_semantic_protocol(
        altered_fallback,
        request,
    )
    assert fallback_protocol.to_dict() == baseline.to_dict()
    canonical_fallback = semantic_routes.canonical_semantic_execution_problem(
        altered_fallback
    )
    assert canonical_fallback.exact_target.build_fallback_anchor_state is not (
        forbidden_fallback
    )
    assert callback_calls == {"reference": 0, "exact": 0, "fallback": 0}


def test_semantic_materializer_rejects_any_nonexact_k5_request() -> None:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    request = build_paper_i_ra_strong_weak_always_k5_request(
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
    )
    drifted = replace(
        request,
        execution=replace(
            request.execution,
            stop=replace(
                request.execution.stop,
                maximum_controller_rounds=6,
            ),
        ),
    )

    with pytest.raises(ValueError, match="strong--weak always-open k=5"):
        materialize_paper_i_ra_strong_weak_always_k5_protocol(
            problem,
            drifted,
        )


def test_general_semantic_materializer_binds_another_canonical_matrix_cell() -> None:
    from pipelines.static_adapt import ra_adapt as public_ra_adapt

    assert (
        public_ra_adapt.materialize_paper_i_ra_semantic_protocol
        is materialize_paper_i_ra_semantic_protocol
    )
    problem = resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.25,
            dv=0.0,
            omega0=1.0,
            g_ep=math.sqrt(0.625),
            n_ph_max=7,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            include_zero_point=True,
            v_nn=0.0,
            t_prime=0.0,
            n_fermions=None,
        )
    )
    seed = build_paper_i_ra_strong_weak_always_k5_request(
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
    )
    request = replace(
        seed,
        method=replace(
            seed.method,
            insertion=PlateauCommutationInsertion(),
        ),
        execution=replace(
            seed.execution,
            stop=replace(
                seed.execution.stop,
                maximum_controller_rounds=17,
            ),
        ),
    )

    protocol = materialize_paper_i_ra_semantic_protocol(problem, request)

    assert protocol.execution_authorized is False
    assert protocol._materialization_authority is not None
    assert "weak_strong" in protocol.bundle_materialization.cell_id
    assert protocol.bundle_materialization.cell_id.endswith(
        "__plateau_commutation__k17"
    )

    outside_matrix = resolve_problem_context(
        replace(problem.request, u=3.0)
    )
    with pytest.raises(ValueError, match="canonical Paper-I L=2"):
        materialize_paper_i_ra_semantic_protocol(outside_matrix, request)


def test_semantic_authority_revalidates_live_source_inventory_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    protocol = materialize_paper_i_ra_strong_weak_always_k5_protocol(
        problem,
        build_paper_i_ra_strong_weak_always_k5_request(
            PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
        ),
    )
    inventory = semantic_closure_source_implementation_inventory(
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
    )
    assert inventory["coverage"] == "conservative_production_python_tree_v1"
    target_suffix = "pipelines/static_adapt/phase_shortlists.py"
    inventory_paths = {
        str(row["path"]) for row in inventory["sources"]
    }
    assert target_suffix in inventory_paths
    assert "pipelines/static_adapt/engine_support.py" in inventory_paths
    assert "pipelines/reporting/paper_i_run_summary.py" in inventory_paths
    assert (
        "pipelines/exact_bench/table_i_qiskit_resource_compile.py"
        in inventory_paths
    )
    original_read_bytes = Path.read_bytes

    def _drift_one_bound_dependency(path: Path) -> bytes:
        payload = original_read_bytes(path)
        if path.as_posix().endswith(target_suffix):
            return payload + b"\n# simulated-post-mint-drift"
        return payload

    monkeypatch.setattr(Path, "read_bytes", _drift_one_bound_dependency)
    monkeypatch.setattr(
        ra_engine,
        "_resolve_execution_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("source drift reached runtime resolution")
        ),
    )

    with pytest.raises(ValueError, match="source-bound native contract"):
        ra_engine.run_ra_adapt(problem, protocol)


class _NoScienceRuntimeReached(RuntimeError):
    pass


def test_authorized_run_reaches_native_phase0_dispatch_without_science(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_problem = build_paper_i_ra_strong_weak_nph3_problem()
    callback_calls = {"reference": 0, "exact": 0, "fallback": 0}

    def forbidden_reference() -> np.ndarray:
        callback_calls["reference"] += 1
        raise AssertionError("runtime executed supplied reference callback")

    def forbidden_exact(*_args: object, **_kwargs: object) -> float:
        callback_calls["exact"] += 1
        raise AssertionError("runtime executed supplied exact callback")

    def forbidden_fallback() -> np.ndarray:
        callback_calls["fallback"] += 1
        raise AssertionError("runtime executed supplied fallback callback")

    altered_exact = resolve_problem_context(
        canonical_problem.request,
        exact_energy_impl=forbidden_exact,
    )
    problem = replace(
        altered_exact,
        reference_state=replace(
            altered_exact.reference_state,
            build_state=forbidden_reference,
        ),
        exact_target=replace(
            altered_exact.exact_target,
            build_fallback_anchor_state=forbidden_fallback,
        ),
    )
    protocol = materialize_paper_i_ra_strong_weak_always_k5_protocol(
        problem,
        build_paper_i_ra_strong_weak_always_k5_request(
            PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
        ),
    )
    observed: dict[str, object] = {}

    def execute_phase0_only(
        transaction: object,
        *,
        admissible_domain: object,
    ) -> object:
        context = getattr(transaction, "context")
        observed["resolved_adapter"] = getattr(
            context,
            "candidate_adapter",
        )
        observed["phase0_route_variant"] = getattr(
            observed["resolved_adapter"],
            "route_variant",
        )
        observed["domain_count"] = len(admissible_domain)  # type: ignore[arg-type]
        observed["runtime_horizon"] = getattr(context, "max_depth")
        observed["bound_horizon"] = context.route_contract[
            "native_semantic_contract"
        ]["horizon"]
        canonical_runtime_problem = context.resolved_problem
        observed["canonical_reference"] = (
            canonical_runtime_problem.reference_state.build_state
            is not forbidden_reference
        )
        observed["canonical_exact"] = (
            canonical_runtime_problem.exact_target.resolve_energy
            is not altered_exact.exact_target.resolve_energy
        )
        observed["canonical_fallback"] = (
            canonical_runtime_problem.exact_target.build_fallback_anchor_state
            is not forbidden_fallback
        )
        raise _NoScienceRuntimeReached

    monkeypatch.setattr(
        semantic_routes,
        "execute_semantic_phase0_runtime",
        execute_phase0_only,
    )

    with pytest.raises(_NoScienceRuntimeReached):
        ra_engine.run_ra_adapt(
            problem,
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=4
            ),
        )

    assert isinstance(
        observed["resolved_adapter"],
        PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
    )
    assert observed["phase0_route_variant"] == (
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
    )
    assert int(observed["domain_count"]) > 0
    assert observed["runtime_horizon"] == 4
    assert observed["bound_horizon"] == 5
    assert observed["canonical_reference"] is True
    assert observed["canonical_exact"] is True
    assert observed["canonical_fallback"] is True
    assert callback_calls == {"reference": 0, "exact": 0, "fallback": 0}


def test_all_phase_adaptive_run_seam_binds_runtime_policy_before_science(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    request = build_paper_i_ra_all_phase_adaptive_request(
        insertion_policy="append_only",
        maximum_controller_rounds=1,
    )
    protocol = materialize_paper_i_ra_semantic_protocol(problem, request)
    observed: dict[str, object] = {}

    def stop_at_phase0(
        transaction: object,
        *,
        admissible_domain: object,
    ) -> object:
        context = getattr(transaction, "context")
        observed["policy"] = context.phase123_shortlist_policy
        observed["domain_count"] = len(admissible_domain)  # type: ignore[arg-type]
        raise _NoScienceRuntimeReached

    monkeypatch.setattr(
        semantic_routes,
        "execute_semantic_phase0_runtime",
        stop_at_phase0,
    )
    with pytest.raises(_NoScienceRuntimeReached):
        ra_engine.run_ra_adapt(problem, protocol)

    assert observed["policy"] == ADAPTIVE_PHASE123_SHORTLIST_POLICY_V1
    assert int(observed["domain_count"]) > 0


def test_all_phase_position_adaptive_one_round_closes_native_transition() -> None:
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_request(
            insertion_policy="append_only",
            maximum_controller_rounds=1,
        ),
    )

    result = ra_engine.run_ra_adapt(problem, protocol)

    accepted = result.scientific_receipts["accepted_round_receipts"]
    assert len(accepted) == 1
    round_receipt = accepted[0]
    assert round_receipt["accepted_round_ordinal"] == 1
    phase0_receipt = round_receipt["ra_gradient_phase0_shortlist"]
    assert phase0_receipt[
        "position_aware_gradient_surface"
    ] is True
    phase0_retained = phase0_receipt["retained_records"]
    assert phase0_retained
    assert all("::pool[" in row["generator_id"] for row in phase0_retained)

    scored = round_receipt["scored_insertion_position_population"]
    phase_i_records = scored["phases"][0]["records"]
    assert phase_i_records
    assert all("::pool[" not in row["generator_id"] for row in phase_i_records)

    projected = round_receipt["projected_phase3_population_receipt"]
    phase_i_qiskit_rows = projected[
        "phase123_qiskit_population_normalization_receipts"
    ]["phase_i"]["rows"]
    assert phase_i_qiskit_rows
    for row in phase_i_qiskit_rows:
        assert "::pool[" not in row["generator_id"]
        assert row["compile_cache_generator_id"] == (
            f"{row['generator_id']}::pool[{row['candidate_pool_index']}]"
        )
        assert row["compile_cache_identity"]["generator_id"] == row[
            "compile_cache_generator_id"
        ]


@pytest.mark.parametrize(
    "insertion_policy",
    ["append_only", "plateau_commutation"],
    ids=["append-only", "plateau-closed"],
)
def test_all_phase_position_adaptive_public_run_matches_generator_first_when_closed(
    insertion_policy: str,
) -> None:
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    generator_result = ra_engine.run_ra_adapt(
        problem,
        materialize_paper_i_ra_semantic_protocol(
            problem,
            build_paper_i_ra_all_phase_adaptive_request(
                insertion_policy=insertion_policy,
                maximum_controller_rounds=2,
            ),
        ),
    )
    position_result = ra_engine.run_ra_adapt(
        problem,
        materialize_paper_i_ra_semantic_protocol(
            problem,
            build_paper_i_ra_all_phase_position_adaptive_request(
                insertion_policy=insertion_policy,
                maximum_controller_rounds=2,
            ),
        ),
    )

    generator_states = generator_result.run.accepted_trajectory
    position_states = position_result.run.accepted_trajectory
    assert len(generator_states) == len(position_states) == 2

    def assert_within_128_ulp(actual: float, expected: float) -> None:
        scale = max(abs(float(actual)), abs(float(expected)))
        assert abs(float(actual) - float(expected)) <= 128 * math.ulp(scale)

    for generator_state, position_state in zip(
        generator_states,
        position_states,
        strict=True,
    ):
        assert position_state.controller_round == generator_state.controller_round
        assert position_state.operators == generator_state.operators
        assert position_state.insertion_positions == generator_state.insertion_positions
        assert position_state.generator_ids == generator_state.generator_ids
        assert_within_128_ulp(position_state.energy, generator_state.energy)
        for position_value, generator_value in zip(
            position_state.logical_parameters,
            generator_state.logical_parameters,
            strict=True,
        ):
            assert_within_128_ulp(position_value, generator_value)
        for position_value, generator_value in zip(
            position_state.runtime_parameters,
            generator_state.runtime_parameters,
            strict=True,
        ):
            assert_within_128_ulp(position_value, generator_value)

    generator_receipts = generator_result.scientific_receipts[
        "accepted_round_receipts"
    ]
    position_receipts = position_result.scientific_receipts[
        "accepted_round_receipts"
    ]
    assert len(generator_receipts) == len(position_receipts) == 2
    for generator_round, position_round in zip(
        generator_receipts,
        position_receipts,
        strict=True,
    ):
        generator_phase0 = generator_round["ra_gradient_phase0_shortlist"]
        position_phase0 = position_round["ra_gradient_phase0_shortlist"]
        assert [row["pool_index"] for row in position_phase0["ranking"]] == (
            generator_phase0["ranked_pool_indices"]
        )
        assert [
            row["pool_index"] for row in position_phase0["retained_records"]
        ] == generator_phase0["retained_pool_indices"]
        generator_scores = {
            int(row["pool_index"]): float(row["active_score"])
            for row in generator_phase0["ranking"]
        }
        for row in position_phase0["ranking"]:
            assert_within_128_ulp(
                float(row["active_score"]),
                generator_scores[int(row["pool_index"])],
            )

        generator_scored = generator_round[
            "scored_insertion_position_population"
        ]
        position_scored = position_round[
            "scored_insertion_position_population"
        ]
        assert generator_scored["interior_scored_count"] == 0
        assert position_scored["interior_scored_count"] == 0

        def phase_i_coordinates(receipt: dict[str, object]) -> list[tuple[object, ...]]:
            return [
                (
                    row["domain_record_id"],
                    row["generator_id"],
                    row["pool_index"],
                    row["pool_label"],
                    row["insertion_position"],
                    row["position_class"],
                )
                for row in receipt["phases"][0]["records"]  # type: ignore[index]
            ]

        assert phase_i_coordinates(position_scored) == phase_i_coordinates(
            generator_scored
        )
        if insertion_policy == "plateau_commutation":
            assert generator_round["insertion_commutation_plateau"][
                "domain_open"
            ] is False
            assert position_round["insertion_commutation_plateau"][
                "domain_open"
            ] is False

    assert [
        receipt.s_alg
        for receipt in position_result.run.canonical_reporting.accepted_prefix_work
    ] == [
        receipt.s_alg
        for receipt in generator_result.run.canonical_reporting.accepted_prefix_work
    ]


def test_all_phase_position_adaptive_two_round_always_open_passes_records_directly() -> None:
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    seed = build_paper_i_ra_all_phase_position_adaptive_request(
        insertion_policy="append_only",
        maximum_controller_rounds=2,
    )
    request = replace(
        seed,
        method=replace(
            seed.method,
            insertion=AlwaysCommutationReducedInsertion(),
        ),
    )
    protocol = materialize_paper_i_ra_semantic_protocol(problem, request)

    result = ra_engine.run_ra_adapt(problem, protocol)

    accepted = result.scientific_receipts["accepted_round_receipts"]
    assert len(accepted) == 2
    for round_receipt in accepted:
        phase0 = round_receipt["ra_gradient_phase0_shortlist"]
        retained = phase0["retained_records"]
        phase_i = round_receipt["scored_insertion_position_population"][
            "phases"
        ][0]["records"]
        assert phase0["generator_level_reexpansion_after_phase0"] is False
        assert len(phase_i) == len(retained)

        def coordinate(row: dict[str, object]) -> tuple[str, int, int, str]:
            return (
                str(row["domain_record_id"]),
                int(row["pool_index"]),
                int(row["insertion_position"]),
                str(row["position_class"]),
            )

        assert {coordinate(row) for row in phase_i} == {
            coordinate(row) for row in retained
        }
        phase_i_qiskit = round_receipt[
            "projected_phase3_population_receipt"
        ]["phase123_qiskit_population_normalization_receipts"]["phase_i"][
            "rows"
        ]
        assert {
            (
                str(row["generator_id"]),
                int(row["candidate_pool_index"]),
                int(row["position_id"]),
                str(row["candidate_label"]),
            )
            for row in phase_i_qiskit
        } == {
            (
                str(row["generator_id"]),
                int(row["pool_index"]),
                int(row["insertion_position"]),
                str(row["pool_label"]),
            )
            for row in phase_i
        }


def test_semantic_operational_resume_preserves_bound_route_before_hydration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    protocol = materialize_paper_i_ra_strong_weak_always_k5_protocol(
        problem,
        build_paper_i_ra_strong_weak_always_k5_request(
            PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2
        ),
    )
    observed: dict[str, object] = {}

    def stop_before_resume_hydration(
        _problem: object,
        sr_request: object,
        *,
        route_override: object,
        candidate_adapter: object,
    ) -> object:
        observed["request"] = sr_request
        observed["route_override"] = route_override
        observed["adapter"] = candidate_adapter
        raise _NoScienceRuntimeReached

    monkeypatch.setattr(
        ra_engine,
        "_resolve_execution_context",
        stop_before_resume_hydration,
    )
    with pytest.raises(_NoScienceRuntimeReached):
        ra_engine.run_ra_adapt(
            problem,
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=4,
                resume=AcceptedStateResume(
                    checkpoint_path=tmp_path / "not-read.current.json",
                    checkpoint_sha256="0" * 64,
                ),
            ),
        )

    sr_request = observed["request"]
    assert sr_request.execution.stop.maximum_controller_rounds == 4
    assert isinstance(sr_request.execution.resume, AcceptedStateResume)
    route_override = observed["route_override"]
    assert route_override[2]["native_semantic_contract"]["horizon"] == 5
    assert isinstance(
        observed["adapter"],
        PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
    )


def _semantic_final_accounting_fixture(
    route_variant: str = PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    str,
    dict[str, object],
]:
    protocol = preflight_paper_i_ra_strong_weak_always_k5(
        build_paper_i_ra_strong_weak_nph3_problem(),
        build_paper_i_ra_strong_weak_always_k5_request(route_variant),
    )
    assert protocol.route_contract is not None
    session, pending, _, transaction, domain = (
        _semantic_phase0_runtime_fixture(route_variant)
    )
    if route_variant in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS:
        position_rows = [
            {
                "domain_record_id": row.domain_record_id,
                "generator_id": (
                    f"{row.generator_id}::pool[{row.pool_index}]"
                ),
                "pool_index": row.pool_index,
                "pool_label": row.pool_label,
                "insertion_position": row.insertion_position,
                "position_class": (
                    "interior"
                    if row.insertion_position < pending.append_position
                    else "append"
                ),
                "gradient_signed": (
                    4.0
                    if row.generator_id == "g1" and row.insertion_position == 2
                    else 0.01
                ),
                "graph_proxy_denominator": 1.0,
            }
            for row in domain
        ]
        position_receipt = build_semantic_position_phase0_receipt(
            position_rows,
            estimator_event_ids=[
                f"position-gradient:{index}"
                for index in range(len(position_rows))
            ],
            route_variant=route_variant,
        )
        retained_ids = {
            str(row["domain_record_id"])
            for row in position_receipt["retained_records"]
        }
        pending.phase0_gradient_shortlist_receipt = position_receipt
        phase0 = SimpleNamespace(
            population=domain,
            shortlist=tuple(
                row for row in domain if row.domain_record_id in retained_ids
            ),
        )
    else:
        phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
            session=session,
            transaction=transaction,
            admissible_domain=domain,
        )
    assert phase0 is not None

    def project(record: object) -> dict[str, object]:
        return {
            "domain_record_id": record.domain_record_id,
            "generator_id": record.generator_id,
            "pool_index": record.pool_index,
            "pool_label": record.pool_label,
            "insertion_position": record.insertion_position,
            "position_class": (
                "interior"
                if record.insertion_position < pending.append_position
                else "append"
            ),
        }

    if route_variant in PAPER_I_RA_PHASE0_POSITION_ROUTE_VARIANTS:
        position_receipt = pending.phase0_gradient_shortlist_receipt
        assert isinstance(position_receipt, dict)
        population = [
            {
                key: row[key]
                for key in (
                    "domain_record_id",
                    "generator_id",
                    "pool_index",
                    "pool_label",
                    "insertion_position",
                    "position_class",
                )
            }
            for row in position_receipt["population"]
        ]
        shortlist = [
            {
                key: row[key]
                for key in (
                    "domain_record_id",
                    "generator_id",
                    "pool_index",
                    "pool_label",
                    "insertion_position",
                    "position_class",
                )
            }
            for row in position_receipt["retained_records"]
        ]
    else:
        population = [project(row) for row in phase0.population]
        shortlist = [project(row) for row in phase0.shortlist]
    scored = {
        "phase0_gradient_screen": {
            "schema": "paper_i_scored_gradient_phase0_population_v1",
            "population_count": len(population),
            "population": population,
            "ordered_population_sha256": canonical_sha256(population),
            "shortlist_count": len(shortlist),
            "shortlist": shortlist,
            "ordered_shortlist_sha256": canonical_sha256(shortlist),
        }
    }
    accepted_rounds = [
        {
            "ra_gradient_phase0_shortlist": copy.deepcopy(
                pending.phase0_gradient_shortlist_receipt
            ),
            "scored_insertion_position_population": scored,
        }
    ]

    phase_receipts: dict[str, dict[str, object]] = {}
    for offset, phase in enumerate(("phase_i", "phase_ii", "phase_iii")):
        population_hash = f"{offset + 1:064x}"
        compile_cache_identity = {
            "schema": "phase123_qiskit_candidate_position_compile_cache_v1",
            "scope": BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
            "candidate_label": "G1",
            "generator_id": "g1::pool[1]",
            "position_id": 2,
            "base_structure_key": "a" * 64,
            "trial_structure_key": "b" * 64,
        }
        rows = [
            {
                "candidate_label": "G1",
                "candidate_pool_index": 1,
                "generator_id": "g1",
                "compile_cache_generator_id": "g1::pool[1]",
                "position_id": 2,
                "base_structure_key": "a" * 64,
                "trial_structure_key": "b" * 64,
                "compile_cache_identity": compile_cache_identity,
                "compile_cache_identity_sha256": canonical_sha256(
                    compile_cache_identity
                ),
                "raw_delta_compiled_count_2q": -1.0,
                "raw_delta_compiled_depth_2q": 2.0,
                "raw_delta_compiled_count_1q": -3.0,
                "hardware_cost_signed_index": -0.25,
                "hardware_cost_score_factor": 1.25,
                "hardware_cost_population_hash": population_hash,
            }
        ]
        phase_receipts[phase] = {
            "schema": "paper_i_phase123_qiskit_population_normalization_v1",
            "scope": BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
            "phase": phase,
            "population_count": 1,
            "normalization_count": 1,
            "normalization_policy": "zero_centered_signed_arctan_v1",
            "population_hash": population_hash,
            "negative_delta_reward_enabled": True,
            "full_base_trial_at_recorded_insertion": True,
            "excluded_from_s_alg": True,
            "rows": rows,
            "rows_sha256": canonical_sha256(rows),
        }
    executed_route = dict(protocol.route_contract)
    executed_route_sha256 = executed_route.pop("sha256")
    finalization = {
        "sr_route_profile_contract": executed_route,
        "sr_route_profile_contract_sha256": executed_route_sha256,
        "history": [
            {
                "projected_phase3_population_receipt": {
                    "schema": "paper_i_projected_phase3_population_receipt_v2",
                    "phase123_qiskit_population_normalization_receipts": (
                        phase_receipts
                    ),
                    "phase3_qiskit_selector_cost_receipt": {
                        "phase123_population_normalization_receipt": (
                            phase_receipts["phase_iii"]
                        )
                    },
                }
            }
        ]
    }
    accepted_rounds[0]["projected_phase3_population_receipt"] = copy.deepcopy(
        finalization["history"][0]["projected_phase3_population_receipt"]
    )
    selector_accounting = {
        "schema": "paper_i_selector_compile_cost_accounting_v1",
        "scope": BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
        "phase_i_phase_ii": None,
        "phase_iii": {
            "role": "phase_i_phase_ii_phase_iii",
            "mode": "transpile_single_v1",
            "optimization_level": 1,
            "seed_transpiler": 7,
            "structure_theta_value": 1.0,
            "negative_delta_reward_enabled": True,
            "one_qubit_coordinate_policy": "compiled_positive_delta_v1",
            "targets": [
                {
                    "resolved_name": "FakeMarrakesh",
                    "resolution_kind": "fake_exact",
                }
            ],
        },
        "phase_i_cost_source": "backend_transpile_v1",
        "qiskit_applied_phases": ["phase_i", "phase_ii", "phase_iii"],
        "phase_iii_reuses_phase_i_phase_ii_oracle": False,
        "excluded_from_s_alg": True,
    }
    return (
        dict(protocol.route_contract),
        finalization,
        accepted_rounds,
        protocol.algorithm_id,
        selector_accounting,
    )


@pytest.mark.parametrize(
    "route_variant",
    [
        PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
        PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    ],
)
def test_semantic_final_accounting_closes_phase0_and_phase123_receipts(
    route_variant: str,
) -> None:
    route, finalization, accepted, algorithm_id, accounting = (
        _semantic_final_accounting_fixture(route_variant)
    )

    closure = validate_semantic_final_selector_accounting(
        algorithm_id=algorithm_id,
        route_contract=route,
        selector_compile_cost_accounting=accounting,
        finalization=finalization,
        accepted_round_receipts=accepted,
    )

    assert closure["schema"] == (
        "paper_i_ra_semantic_final_selector_accounting_closure_v1"
    )
    assert closure["validated_round_count"] == 1


def _attach_adaptive_phase123_fixture(
    accepted_round: dict[str, object],
) -> None:
    scored = accepted_round["scored_insertion_position_population"]
    assert isinstance(scored, dict)
    phase0 = accepted_round["ra_gradient_phase0_shortlist"]
    assert isinstance(phase0, dict)
    retained_position_rows = phase0.get("retained_records")
    if isinstance(retained_position_rows, list) and retained_position_rows:
        retained_position = retained_position_rows[0]
        assert isinstance(retained_position, dict)
        domain_record_id = str(retained_position["domain_record_id"])
        pool_index = int(retained_position["pool_index"])
        controller_generator_id = str(retained_position["generator_id"])
        pool_suffix = f"::pool[{pool_index}]"
        assert controller_generator_id.endswith(pool_suffix)
        generator_id = controller_generator_id[: -len(pool_suffix)]
        pool_label = str(retained_position["pool_label"])
        insertion_position = int(retained_position["insertion_position"])
        position_class = str(retained_position["position_class"])
    else:
        domain_record_id = "d1"
        generator_id = "g1"
        pool_index = 1
        pool_label = "G1"
        insertion_position = 2
        position_class = "append"
    phase_rows: list[dict[str, object]] = []
    for phase, score_key, cap in (
        ("phase_i", "phase1_active_score", 24),
        ("phase_ii", "phase2_raw_score", 12),
        ("phase_iii", "full_v2_score", 12),
    ):
        record_id = (
            f"{generator_id}|pool:{pool_index}|position:{insertion_position}"
        )
        decision = select_adaptive_phase_shortlist(
            (
                AdaptivePhaseCandidateScore(
                    record_id=record_id,
                    pool_index=pool_index,
                    insertion_position=insertion_position,
                    active_score=1.0,
                    tie_break_score=1.0,
                ),
            ),
            phase=phase,
            score_key=score_key,
            hard_cap=cap,
            threshold=0.0,
            frontier_ratio=0.9,
        )
        records = [
            {
                "domain_record_id": domain_record_id,
                "generator_id": generator_id,
                "pool_index": pool_index,
                "pool_label": pool_label,
                "insertion_position": insertion_position,
                "adaptive_record_id": record_id,
                "position_class": position_class,
            }
        ]
        score_rows = [
            score.to_dict() for score in decision.receipt.input_scores
        ]
        phase_rows.append(
            {
                "phase": phase,
                "population_count": 1,
                "records": records,
                "shortlist_count": 1,
                "shortlist_records": copy.deepcopy(records),
                "adaptive_shortlist": decision.receipt.to_dict(),
                "adaptive_population_scores": score_rows,
                "ordered_adaptive_population_scores_sha256": (
                    canonical_sha256(score_rows)
                ),
                "final_admission_record_id": (
                    record_id if phase == "phase_iii" else None
                ),
                "estimator_event_ids": [],
                "ordered_population_sha256": canonical_sha256(records),
            }
        )
    scored["phases"] = phase_rows


def test_all_phase_adaptive_final_closure_cross_binds_live_scores_and_winner() -> None:
    route, finalization, accepted, algorithm_id, accounting = (
        _semantic_final_accounting_fixture(
            PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1
        )
    )
    _attach_adaptive_phase123_fixture(accepted[0])

    closure = validate_semantic_final_selector_accounting(
        algorithm_id=algorithm_id,
        route_contract=route,
        selector_compile_cost_accounting=accounting,
        finalization=finalization,
        accepted_round_receipts=accepted,
    )

    assert set(
        closure["rounds"][0][
            "phase123_adaptive_shortlist_receipt_sha256"
        ]
    ) == {"phase_i", "phase_ii", "phase_iii"}

    detached = copy.deepcopy(accepted)
    detached_scores = detached[0]["scored_insertion_position_population"][
        "phases"
    ][0]["adaptive_population_scores"]
    detached_scores[0]["active_score"] = 0.5
    detached[0]["scored_insertion_position_population"]["phases"][0][
        "ordered_adaptive_population_scores_sha256"
    ] = canonical_sha256(detached_scores)
    with pytest.raises(RuntimeError, match="selector accounting is invalid"):
        validate_semantic_final_selector_accounting(
            algorithm_id=algorithm_id,
            route_contract=route,
            selector_compile_cost_accounting=accounting,
            finalization=finalization,
            accepted_round_receipts=detached,
        )

    wrong_winner = copy.deepcopy(accepted)
    wrong_winner[0]["scored_insertion_position_population"]["phases"][2][
        "final_admission_record_id"
    ] = "detached-winner"
    with pytest.raises(RuntimeError, match="selector accounting is invalid"):
        validate_semantic_final_selector_accounting(
            algorithm_id=algorithm_id,
            route_contract=route,
            selector_compile_cost_accounting=accounting,
            finalization=finalization,
            accepted_round_receipts=wrong_winner,
        )

    detached_qiskit_finalization = copy.deepcopy(finalization)
    detached_qiskit_accepted = copy.deepcopy(accepted)
    phase_i_qiskit = detached_qiskit_finalization["history"][0][
        "projected_phase3_population_receipt"
    ]["phase123_qiskit_population_normalization_receipts"]["phase_i"]
    phase_i_qiskit["rows"][0]["candidate_pool_index"] = 99
    phase_i_qiskit["rows_sha256"] = canonical_sha256(
        phase_i_qiskit["rows"]
    )
    detached_qiskit_accepted[0][
        "projected_phase3_population_receipt"
    ] = copy.deepcopy(
        detached_qiskit_finalization["history"][0][
            "projected_phase3_population_receipt"
        ]
    )
    with pytest.raises(RuntimeError, match="selector accounting is invalid"):
        validate_semantic_final_selector_accounting(
            algorithm_id=algorithm_id,
            route_contract=route,
            selector_compile_cost_accounting=accounting,
            finalization=detached_qiskit_finalization,
            accepted_round_receipts=detached_qiskit_accepted,
        )

    wrong_label_finalization = copy.deepcopy(finalization)
    wrong_label_phase_i = wrong_label_finalization["history"][0][
        "projected_phase3_population_receipt"
    ]["phase123_qiskit_population_normalization_receipts"]["phase_i"]
    wrong_label_row = wrong_label_phase_i["rows"][0]
    wrong_label_row["candidate_label"] = "WRONG_LABEL"
    wrong_label_row["compile_cache_identity"][
        "candidate_label"
    ] = "WRONG_LABEL"
    wrong_label_row["compile_cache_identity_sha256"] = canonical_sha256(
        wrong_label_row["compile_cache_identity"]
    )
    wrong_label_phase_i["rows_sha256"] = canonical_sha256(
        wrong_label_phase_i["rows"]
    )
    wrong_label_accepted = copy.deepcopy(accepted)
    wrong_label_accepted[0][
        "projected_phase3_population_receipt"
    ] = copy.deepcopy(
        wrong_label_finalization["history"][0][
            "projected_phase3_population_receipt"
        ]
    )
    with pytest.raises(RuntimeError, match="selector accounting is invalid"):
        validate_semantic_final_selector_accounting(
            algorithm_id=algorithm_id,
            route_contract=route,
            selector_compile_cost_accounting=accounting,
            finalization=wrong_label_finalization,
            accepted_round_receipts=wrong_label_accepted,
        )
    assert closure["qiskit_phases"] == ["phase_i", "phase_ii", "phase_iii"]
    assert closure["qiskit_compile_work_excluded_from_s_alg"] is True
    assert closure["sha256"] == canonical_sha256(
        {key: value for key, value in closure.items() if key != "sha256"}
    )


def test_all_phase_adaptive_final_closure_authenticates_no_positive_terminal() -> None:
    route, finalization, accepted, algorithm_id, accounting = (
        _semantic_final_accounting_fixture(
            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
        )
    )
    append_protocol = preflight_paper_i_ra_semantic(
        build_paper_i_ra_strong_weak_nph3_problem(),
        build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
            insertion_policy="append_only",
            maximum_controller_rounds=5,
        ),
    )
    assert append_protocol.route_contract is not None
    route = copy.deepcopy(dict(append_protocol.route_contract))
    executed_route = copy.deepcopy(route)
    executed_route_sha256 = executed_route.pop("sha256")
    finalization["sr_route_profile_contract"] = executed_route
    finalization["sr_route_profile_contract_sha256"] = executed_route_sha256
    _attach_adaptive_phase123_fixture(accepted[0])
    scored = copy.deepcopy(
        accepted[0]["scored_insertion_position_population"]
    )
    phase3 = scored["phases"][2]
    phase3_scores = copy.deepcopy(phase3["adaptive_population_scores"])
    phase3_scores[0]["active_score"] = 0.0
    phase3_decision = select_adaptive_phase_shortlist(
        tuple(
            AdaptivePhaseCandidateScore(
                record_id=str(row["record_id"]),
                pool_index=int(row["pool_index"]),
                insertion_position=int(row["insertion_position"]),
                active_score=float(row["active_score"]),
                tie_break_score=float(row["tie_break_score"]),
            )
            for row in phase3_scores
        ),
        phase="phase_iii",
        score_key="full_v2_score",
        hard_cap=12,
        threshold=0.0,
        frontier_ratio=0.9,
    ).receipt
    assert phase3_decision.status == "no_positive_population"
    phase3.update(
        {
            "shortlist_count": 0,
            "shortlist_records": [],
            "adaptive_shortlist": phase3_decision.to_dict(),
            "adaptive_population_scores": phase3_scores,
            "ordered_adaptive_population_scores_sha256": canonical_sha256(
                phase3_scores
            ),
            "final_admission_record_id": None,
            "terminal_outcome": (
                ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
            ),
        }
    )
    all_scored_records = [
        record
        for phase_row in scored["phases"]
        for record in phase_row["records"]
    ]
    scored.update(
        {
            "schema": "paper_i_scored_insertion_position_population_v1",
            "coordinate_chart": "exact_ordered_insertion_zero_angle_v1",
            "append_position": max(
                int(record["insertion_position"])
                for record in all_scored_records
            ),
            "phase_order": ["phase_i", "phase_ii", "phase_iii"],
            "scored_record_count": len(all_scored_records),
            "interior_scored_count": sum(
                record["position_class"] == "interior"
                for record in all_scored_records
            ),
            "append_scored_count": sum(
                record["position_class"] == "append"
                for record in all_scored_records
            ),
        }
    )
    scored["sha256"] = canonical_sha256(
        {key: value for key, value in scored.items() if key != "sha256"}
    )
    projected = copy.deepcopy(
        finalization["history"][0][
            "projected_phase3_population_receipt"
        ]
    )
    terminal_checkpoint = {
        "checkpoint_kind": "terminal_phase3_no_positive",
        "outer_iteration": 1,
        "active_ansatz_depth": 1,
        "projective_state_fingerprint": "projective_state_v1:" + "a" * 64,
    }
    terminal_checkpoint["checkpoint_sha256"] = canonical_sha256(
        terminal_checkpoint
    )
    finalization["history"][0]["active_prefix_checkpoint"] = {
        "checkpoint_kind": "post_admission_prune",
        "outer_iteration": 1,
        "active_ansatz_depth": 1,
        "projective_state_fingerprint": terminal_checkpoint[
            "projective_state_fingerprint"
        ],
    }
    terminal_occurrences = [
        {
            "sequence": 1,
            "primitive_id": "outer",
            "component": "N_H_outer",
            "consumer_scope": "outer_state_refresh",
            "branch_id": None,
            "charged": True,
        },
        {
            "sequence": 2,
            "primitive_id": "gradient",
            "component": "N_grad",
            "consumer_scope": "phase_i",
            "branch_id": None,
            "charged": True,
        },
        {
            "sequence": 3,
            "primitive_id": "metric",
            "component": "N_metric",
            "consumer_scope": "phase_iii",
            "branch_id": None,
            "charged": True,
        },
    ]
    terminal_components = {
        "N_H_outer": 1,
        "N_H_refit": 0,
        "N_grad": 1,
        "N_metric": 1,
    }
    terminal_prefix = {
        "schema": "paper_i_active_prefix_estimator_ledger_receipt_v2",
        "enabled": True,
        "status": "complete",
        "checkpoint_sequence": 1,
        "outer_iteration": 1,
        "checkpoint_kind": "terminal_phase3_no_positive",
        "branch_id": None,
        "parent_branch_id": None,
        "occurrence_sequence_start_exclusive": 0,
        "occurrence_sequence_end_inclusive": 3,
        "raw_occurrence_delta": {
            "components": terminal_components,
            "total": 3,
        },
        "executed_query_delta": {
            "components": terminal_components,
            "S_alg": 3,
        },
        "unique_primitive_delta": {
            "components": terminal_components,
            "S_unique": 3,
        },
        "cumulative_raw_occurrences": {
            "components": terminal_components,
            "total": 3,
        },
        "cumulative_executed_queries": {
            "components": terminal_components,
            "S_alg": 3,
            "unit": "executed_logical_scalar_estimator_invocation",
        },
        "cumulative_unique_primitives": {
            "components": terminal_components,
            "S_unique": 3,
        },
        "runtime_estimator_occurrence_contract": (
            "all_instrumented_logical_scalar_estimator_calls_v1"
        ),
        "physical_identity_collapse_is_diagnostic_only": True,
        "raw_occurrences_preserved": True,
    }
    terminal_event_ids = [
        "estimator:2:gradient",
        "estimator:3:metric",
    ]
    controller_work = ControllerMeasurementWorkAccumulator()
    for phase in ("phase1", "phase2", "phase3"):
        controller_work.record_event(
            phase=phase,
            event_kind=f"terminal_fixture_{phase}",
            group_keys=[],
            records_evaluated=1,
            candidate_count=1,
            evaluated_count=1,
            pre_shortlist_count=1,
            shortlist_size=1,
            retained_count=1,
            rejected_count=0,
            probe_role="metric",
            actual_operator_probe_count=1,
        )
    terminal = {
        "schema": "paper_i_ra_phase3_no_positive_selection_terminal_v1",
        "terminal_controller_outcome": (
            ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        ),
        "accepted_controller_round": 1,
        "attempted_controller_round": 2,
        "accepted_state_fingerprint": terminal_checkpoint[
            "projective_state_fingerprint"
        ],
        "accepted_operator_count": 1,
        "accepted_state_unchanged": True,
        "final_admission_record_id": None,
        "phase0_gradient_shortlist": copy.deepcopy(
            accepted[0]["ra_gradient_phase0_shortlist"]
        ),
        "insertion_mode": "append_only",
        "insertion_commutation_plateau": None,
        "insertion_commutation_reduced": None,
        "phase3_population_activation": {
            "schema": "ra_phase3_population_activation_receipt_v1",
            "policy": "all_controller_rounds_v1",
            "competitive_population_live": True,
            "activation_source": "route_default_all_rounds_v1",
            "preplateau_admission_authority": None,
            "winner_materialization_policy": None,
            "insertion_plateau_domain_open": None,
            "independent_latch_active": False,
            "hysteresis_active": False,
        },
        "controller_measurement_work_proxy": controller_work.summary_since(
            0,
            include_events=False,
        ),
        "scored_insertion_position_population": scored,
        "projected_phase3_population_receipt": projected,
        "phase123_qiskit_population_normalization_receipts": copy.deepcopy(
            projected["phase123_qiskit_population_normalization_receipts"]
        ),
        "estimator_event_ids": terminal_event_ids,
        "estimator_event_count": 2,
        "estimator_event_ids_sha256": canonical_sha256(terminal_event_ids),
        "terminal_active_prefix_checkpoint_sha256": canonical_sha256(
            terminal_checkpoint
        ),
        "terminal_estimator_prefix_receipt": terminal_prefix,
        "terminal_estimator_prefix_receipt_sha256": canonical_sha256(
            terminal_prefix
        ),
    }
    terminal["sha256"] = canonical_sha256(terminal)
    finalization.update(
        {
            "terminal_controller_outcome": (
                ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
            ),
            "terminal_phase3_selection_receipt": terminal,
            "terminal_active_prefix_checkpoint": terminal_checkpoint,
            "continuation": {
                "all_active_prefix_estimator_ledger_receipts": [
                    terminal_prefix
                ],
                "terminal_phase3_selection_receipt": terminal,
            },
            "estimator_call_accounting": {
                "full_ledger": {
                    "schema": "estimator_call_ledger_v1",
                    "component_contract": [
                        "N_H_outer",
                        "N_H_refit",
                        "N_grad",
                        "N_metric",
                    ],
                    "occurrences": terminal_occurrences,
                }
            },
        }
    )

    closure = validate_semantic_final_selector_accounting(
        algorithm_id=algorithm_id,
        route_contract=route,
        selector_compile_cost_accounting=accounting,
        finalization=finalization,
        accepted_round_receipts=accepted,
    )

    assert closure["validated_round_count"] == 1
    assert closure["terminal_attempted_controller_round"] == 2
    assert closure["terminal_phase3_selection_receipt_sha256"] == (
        terminal["sha256"]
    )

    v1_protocol = preflight_paper_i_ra_semantic(
        build_paper_i_ra_strong_weak_nph3_problem(),
        build_paper_i_ra_all_phase_position_adaptive_request(
            insertion_policy="append_only",
            maximum_controller_rounds=5,
        ),
    )
    assert v1_protocol.route_contract is not None
    with pytest.raises(ValueError, match="V2 natural-terminal route"):
        validate_semantic_phase3_no_positive_terminal_receipt(
            terminal,
            route_variant=(
                PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1
            ),
            route_contract=v1_protocol.route_contract,
            expected_route_contract_sha256=(
                v1_protocol.route_contract["sha256"]
            ),
            accepted_round_count=1,
            terminal_active_prefix_checkpoint=terminal_checkpoint,
            finalization=finalization,
        )

    tampered = copy.deepcopy(finalization)
    tampered["terminal_phase3_selection_receipt"][
        "accepted_state_unchanged"
    ] = False
    tampered["terminal_phase3_selection_receipt"]["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered[
                "terminal_phase3_selection_receipt"
            ].items()
            if key != "sha256"
        }
    )
    with pytest.raises(RuntimeError, match="selector accounting is invalid"):
        validate_semantic_final_selector_accounting(
            algorithm_id=algorithm_id,
            route_contract=route,
            selector_compile_cost_accounting=accounting,
            finalization=tampered,
            accepted_round_receipts=accepted,
        )

    detached_prefix = copy.deepcopy(finalization)
    detached_prefix["continuation"][
        "all_active_prefix_estimator_ledger_receipts"
    ][-1] = copy.deepcopy(
        detached_prefix["continuation"][
            "all_active_prefix_estimator_ledger_receipts"
        ][-1]
    )
    detached_prefix["continuation"][
        "all_active_prefix_estimator_ledger_receipts"
    ][-1]["raw_occurrence_delta"]["total"] = 999
    with pytest.raises(RuntimeError, match="selector accounting is invalid"):
        validate_semantic_final_selector_accounting(
            algorithm_id=algorithm_id,
            route_contract=route,
            selector_compile_cost_accounting=accounting,
            finalization=detached_prefix,
            accepted_round_receipts=accepted,
        )


def test_all_phase_position_adaptive_final_closure_binds_phase0_records_to_phase_i() -> None:
    route, finalization, accepted, algorithm_id, accounting = (
        _semantic_final_accounting_fixture(
            PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1
        )
    )
    _attach_adaptive_phase123_fixture(accepted[0])

    closure = validate_semantic_final_selector_accounting(
        algorithm_id=algorithm_id,
        route_contract=route,
        selector_compile_cost_accounting=accounting,
        finalization=finalization,
        accepted_round_receipts=accepted,
    )

    assert canonical_sha256(
        {
            "phase0_retained_domain_coordinates": [
                ["g1@2", 1, 2, "append"]
            ],
            "phase_i_population_domain_coordinates": [
                ["g1@2", 1, 2, "append"]
            ],
            "phase0_retained_controller_generator_ids": ["g1::pool[1]"],
            "phase_i_population_physical_generator_ids": ["g1"],
            "phase0_retained_pool_labels": ["G1"],
            "phase_i_population_pool_labels": ["G1"],
        }
    ) == closure["rounds"][0][
        "phase0_phase_i_direct_population_link_sha256"
    ]

    detached_identity = copy.deepcopy(accepted)
    original_phase0 = detached_identity[0][
        "ra_gradient_phase0_shortlist"
    ]
    retained_domain_id = str(
        original_phase0["retained_records"][0]["domain_record_id"]
    )
    detached_rows = copy.deepcopy(original_phase0["population"])
    for row in detached_rows:
        if str(row["domain_record_id"]) == retained_domain_id:
            row["generator_id"] = (
                f"wrong-physical-id::pool[{int(row['pool_index'])}]"
            )
        row.pop("gradient_abs", None)
        row.pop("active_score", None)
    detached_receipt = build_semantic_position_phase0_receipt(
        detached_rows,
        estimator_event_ids=original_phase0["estimator_event_ids"],
        route_variant=PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
    )
    detached_identity[0][
        "ra_gradient_phase0_shortlist"
    ] = detached_receipt
    identity_fields = (
        "domain_record_id",
        "generator_id",
        "pool_index",
        "pool_label",
        "insertion_position",
        "position_class",
    )
    identity_screen = detached_identity[0][
        "scored_insertion_position_population"
    ]["phase0_gradient_screen"]
    identity_population = [
        {field: row[field] for field in identity_fields}
        for row in detached_receipt["population"]
    ]
    identity_shortlist = [
        {field: row[field] for field in identity_fields}
        for row in detached_receipt["retained_records"]
    ]
    identity_screen.update(
        {
            "population_count": len(identity_population),
            "population": identity_population,
            "ordered_population_sha256": canonical_sha256(
                identity_population
            ),
            "shortlist_count": len(identity_shortlist),
            "shortlist": identity_shortlist,
            "ordered_shortlist_sha256": canonical_sha256(
                identity_shortlist
            ),
        }
    )
    with pytest.raises(RuntimeError, match="selector accounting is invalid"):
        validate_semantic_final_selector_accounting(
            algorithm_id=algorithm_id,
            route_contract=route,
            selector_compile_cost_accounting=accounting,
            finalization=finalization,
            accepted_round_receipts=detached_identity,
        )

    detached = copy.deepcopy(accepted)
    original_phase0 = detached[0]["ra_gradient_phase0_shortlist"]
    replacement_rows = copy.deepcopy(original_phase0["population"])
    for row in replacement_rows:
        row["gradient_signed"] = (
            5.0 if row["domain_record_id"] == "g0@0" else 0.01
        )
        row.pop("gradient_abs", None)
        row.pop("active_score", None)
    replacement = build_semantic_position_phase0_receipt(
        replacement_rows,
        estimator_event_ids=original_phase0["estimator_event_ids"],
        route_variant=PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
    )
    detached[0]["ra_gradient_phase0_shortlist"] = replacement
    projection_fields = (
        "domain_record_id",
        "generator_id",
        "pool_index",
        "pool_label",
        "insertion_position",
        "position_class",
    )

    def project(row: dict[str, object]) -> dict[str, object]:
        return {field: row[field] for field in projection_fields}

    population = [project(row) for row in replacement["population"]]
    shortlist = [project(row) for row in replacement["retained_records"]]
    screen = detached[0]["scored_insertion_position_population"][
        "phase0_gradient_screen"
    ]
    screen.update(
        {
            "population_count": len(population),
            "population": population,
            "ordered_population_sha256": canonical_sha256(population),
            "shortlist_count": len(shortlist),
            "shortlist": shortlist,
            "ordered_shortlist_sha256": canonical_sha256(shortlist),
        }
    )
    with pytest.raises(RuntimeError, match="selector accounting is invalid"):
        validate_semantic_final_selector_accounting(
            algorithm_id=algorithm_id,
            route_contract=route,
            selector_compile_cost_accounting=accounting,
            finalization=finalization,
            accepted_round_receipts=detached,
        )


@pytest.mark.parametrize(
    "route_variant",
    [
        PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
        PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    ],
)
def test_semantic_final_accounting_closes_round_zero_stationary_phase0(
    route_variant: str,
) -> None:
    protocol = preflight_paper_i_ra_strong_weak_always_k5(
        build_paper_i_ra_strong_weak_nph3_problem(),
        build_paper_i_ra_strong_weak_always_k5_request(
            route_variant
        ),
    )
    assert protocol.route_contract is not None
    session, pending, _, transaction, domain = (
        _semantic_phase0_runtime_fixture(
            route_variant,
            gradients=(0.0, 0.0, 0.0),
        )
    )
    phase0 = adapt_pipeline._run_global_singleton_gradient_phase0(
        session=session,
        transaction=transaction,
        admissible_domain=domain,
    )
    assert phase0 is not None and phase0.shortlist == ()

    _, _, _, _, accounting = _semantic_final_accounting_fixture(route_variant)
    executed_route = dict(protocol.route_contract)
    executed_route_sha256 = executed_route.pop("sha256")
    finalization = {
        "sr_route_profile_contract": executed_route,
        "sr_route_profile_contract_sha256": executed_route_sha256,
        "history": [],
        "terminal_controller_outcome": phase0.terminal_outcome,
        "terminal_phase0_selection_receipt": copy.deepcopy(
            pending.phase0_gradient_shortlist_receipt
        ),
    }

    closure = validate_semantic_final_selector_accounting(
        algorithm_id=protocol.algorithm_id,
        route_contract=protocol.route_contract,
        selector_compile_cost_accounting=accounting,
        finalization=finalization,
        accepted_round_receipts=[],
    )

    assert closure["validated_round_count"] == 0
    assert closure["terminal_controller_outcome"] == (
        "phase0_stationary_no_competitive_candidate_v1"
    )
    assert closure["terminal_phase0_receipt_sha256"] == (
        pending.phase0_gradient_shortlist_receipt["sha256"]
    )
    assert closure["phase_i_entered_after_terminal_phase0"] is False


def test_accepted_round_projection_preserves_semantic_phase0_for_final_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route, finalization, seed_accepted, algorithm_id, accounting = (
        _semantic_final_accounting_fixture()
    )
    phase0 = copy.deepcopy(seed_accepted[0]["ra_gradient_phase0_shortlist"])
    phase0_screen = copy.deepcopy(
        seed_accepted[0]["scored_insertion_position_population"][
            "phase0_gradient_screen"
        ]
    )
    phase_records = [
        {
            "pool_index": 1,
            "generator_id": "g1",
            "pool_label": "G1",
            "insertion_position": 2,
            "position_class": "append",
        }
    ]
    phases = [
        {
            "phase": phase,
            "population_count": len(phase_records),
            "records": copy.deepcopy(phase_records),
            "ordered_population_sha256": canonical_sha256(phase_records),
        }
        for phase in ("phase_i", "phase_ii", "phase_iii")
    ]
    scored = {
        "schema": "paper_i_scored_insertion_position_population_v1",
        "coordinate_chart": ra_engine.EXACT_ORDERED_INSERTION_CHART,
        "phase_order": ["phase_i", "phase_ii", "phase_iii"],
        "append_position": 2,
        "scored_record_count": 3,
        "interior_scored_count": 0,
        "append_scored_count": 3,
        "phases": phases,
        "phase0_gradient_screen": phase0_screen,
    }
    scored["sha256"] = canonical_sha256(scored)
    history_row = finalization["history"][0]
    history_row.update(
        {
            "route_a_trust_region_update": {
                "source_metric_trust_transaction": {"fixture": True}
            },
            "ra_gradient_phase0_shortlist": phase0,
            "scored_insertion_position_population": scored,
        }
    )

    fake_trust = SimpleNamespace(as_dict=lambda: {"fixture": "trust"})
    fake_support = SimpleNamespace(as_dict=lambda: {"fixture": "support"})
    fake_chart = SimpleNamespace(
        as_dict=lambda: {"fixture": "chart"},
        sha256="a" * 64,
    )
    fake_lineage = SimpleNamespace(
        to_dict=lambda: {
            "generator_id": "g1",
            "pool_index": 1,
            "insertion_position": 2,
        }
    )
    monkeypatch.setattr(
        ra_engine,
        "source_gram_no_overlap_trust_receipt_from_mapping",
        lambda *_args, **_kwargs: fake_trust,
    )
    monkeypatch.setattr(
        ra_engine,
        "_required_retained_support",
        lambda *_args, **_kwargs: fake_support,
    )
    monkeypatch.setattr(
        ra_engine,
        "_required_phase3_stabilization",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        ra_engine,
        "_accepted_candidate_lineage_receipts",
        lambda *_args, **_kwargs: [fake_lineage],
    )
    monkeypatch.setattr(
        ra_engine,
        "_required_accepted_refit_fixed_chart",
        lambda *_args, **_kwargs: fake_chart,
    )

    projected = ra_engine._accepted_round_scientific_receipts(
        finalization,
        adapter_id="semantic-fixture",
        candidate_representation="single_pauli_word_v1",
        executable_inventory=SimpleNamespace(),
        algorithm_id=algorithm_id,
    )
    assert projected[0]["ra_gradient_phase0_shortlist"] == phase0
    assert projected[0]["projected_phase3_population_receipt"] == (
        history_row["projected_phase3_population_receipt"]
    )

    closure = validate_semantic_final_selector_accounting(
        algorithm_id=algorithm_id,
        route_contract=route,
        selector_compile_cost_accounting=accounting,
        finalization=finalization,
        accepted_round_receipts=projected,
    )
    assert closure["validated_round_count"] == 1


@pytest.mark.parametrize(
    "tamper",
    [
        "phase0_missing",
        "phase_ii_double_normalized",
        "compile_cache_identity",
        "s_alg_included",
    ],
)
def test_semantic_final_accounting_rejects_tampered_closure(tamper: str) -> None:
    route, finalization, accepted, algorithm_id, accounting = (
        _semantic_final_accounting_fixture()
    )
    if tamper == "phase0_missing":
        accepted[0].pop("ra_gradient_phase0_shortlist")
    elif tamper == "phase_ii_double_normalized":
        receipts = finalization["history"][0][
            "projected_phase3_population_receipt"
        ]["phase123_qiskit_population_normalization_receipts"]
        receipts["phase_ii"]["normalization_count"] = 2
    elif tamper == "compile_cache_identity":
        receipt = finalization["history"][0][
            "projected_phase3_population_receipt"
        ]["phase123_qiskit_population_normalization_receipts"]["phase_ii"]
        receipt["rows"][0]["compile_cache_identity"]["position_id"] = 1
        receipt["rows_sha256"] = canonical_sha256(receipt["rows"])
    else:
        accounting["excluded_from_s_alg"] = False

    with pytest.raises(RuntimeError, match="semantic final selector accounting"):
        validate_semantic_final_selector_accounting(
            algorithm_id=algorithm_id,
            route_contract=route,
            selector_compile_cost_accounting=accounting,
            finalization=finalization,
            accepted_round_receipts=accepted,
        )


def test_position_natural_terminal_builder_accepts_always_open() -> None:
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
            insertion_policy="always_commutation_reduced",
            maximum_controller_rounds=3,
        ),
    )
    route = protocol.route_contract
    native = route["native_semantic_contract"]
    execution = route["execution_settings"]
    assert native["route_variant"] == (
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
    )
    assert execution["adapt_insertion_mode"] == "full_commutation_reduced"
    assert native["phase0_policy"]["population"] == (
        "current_commutation_reduced_candidate_position_records_v1"
    )
    for surface, policy_key, horizon_key in (
        (native, "phase3_no_positive_policy", "controller_horizon_policy"),
        (
            execution,
            "ra_phase3_no_positive_policy",
            "ra_controller_horizon_policy",
        ),
        (
            route["semantic_invariants"],
            "phase3_no_positive_policy",
            "controller_horizon_policy",
        ),
    ):
        assert surface[policy_key] == (
            ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1
        )
        assert surface[horizon_key] == ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1

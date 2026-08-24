from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.contracts.problem import ProblemRequest
from pipelines.scaffold.hh_continuation_scoring import (
    FullScoreConfig,
    HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1,
    HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1,
    HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1,
    rescore_hardware_cost_family,
)
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt.hh_backend_compile_oracle import (
    BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1,
    BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1,
    MARRAKESH_GRAPH_SPAN_MODE,
    ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1,
)
from pipelines.static_adapt.ra_adapt.adapters import (
    GlobalSinglePauliWordCandidateAdapter,
    GlobalSingletonGradientPhase0CandidateAdapter,
    MacroCandidateAdapter,
    MacroGradientPhase0CandidateAdapter,
    MacroGradientPhase0ThenSingletonCandidateAdapter,
    MacroThenSingletonPhaseICandidateAdapter,
)
from pipelines.static_adapt.ra_adapt import bundles as bundle_module
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    ACTIVE_GRADIENT_STATIONARY,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    RAAdaptRequest,
    RESOURCE_WEIGHTING_ALL_PHASE,
    RESOURCE_WEIGHTING_LATE,
)
from pipelines.static_adapt.ra_adapt.engine import (
    RA_ADAPT_ALGORITHM_ID,
    RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
    RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ROUTE_SUFFIX,
    RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID,
    RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID,
    RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
    RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ROUTE_SUFFIX,
    RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
    RA_ADAPT_MACRO_GRADIENT_PHASE0_PHASE23_QISKIT_ROUTE_SUFFIX,
    RA_ADAPT_MACRO_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
    RA_ADAPT_PHASE23_QISKIT_COST_POLICY,
    RA_ADAPT_PHASE23_QISKIT_COST_ROUTE_SUFFIX,
    RA_ADAPT_PHASE3_QISKIT_DENOMINATOR_POLICY,
    RA_ADAPT_PHASE3_QISKIT_COST_PHASE_REUSE,
    RA_ADAPT_PHASE3_QISKIT_COST_POLICY,
    RA_ADAPT_PHASE3_QISKIT_COST_ROUTE_SUFFIX,
    _repaired_route_contract,
    _validate_executed_insertion_contract,
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.lane_routes import (
    STATIC_LANE_ROUTE_GLOBAL_SINGLE_POPULATION,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AppendOnlyInsertion,
    PlateauCommutationInsertion,
    SRMethodPolicy,
)


SOURCE_ROUTE_SHA256 = (
    "3f4ebed3d48ca972abb2867e8300032f6816a57ba3c5845f28fe1cf37cbbdcdd"
)
SOURCE_ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__"
    "global_guarded_singleton_phase_i__identity_phase_ii__"
    "stationary_source_response_v1__all_phase_resource_weighting_v1"
)


def test_macro_then_singleton_phase123_route_binds_qiskit_to_phase23_only(
) -> None:
    request = _request(adapter=MacroThenSingletonPhaseICandidateAdapter())
    _request_profile, profile, contract, digest = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        algorithm_id=(
            RA_ADAPT_MACRO_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID
        ),
    )

    assert profile.endswith(RA_ADAPT_PHASE23_QISKIT_COST_ROUTE_SUFFIX)
    assert len(digest) == 64
    execution = contract["execution_settings"]
    invariants = contract["semantic_invariants"]
    assert execution["phase3_backend_cost_scope"] == (
        BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1
    )
    assert execution["static_lane_route"] == (
        STATIC_LANE_ROUTE_GLOBAL_SINGLE_POPULATION
    )
    assert invariants["selector_compile_cost_policy"] == (
        RA_ADAPT_PHASE23_QISKIT_COST_POLICY
    )
    assert invariants["phase_i_compile_cost_source"] == (
        "structural_proxy_v1"
    )
    assert invariants["phase_ii_compile_cost_source"] == (
        "backend_transpile_v1"
    )
    assert invariants["phase_iii_compile_cost_source"] == (
        "backend_transpile_v1"
    )
    assert invariants[
        "phase_ii_phase_iii_qiskit_negative_delta_reward_enabled"
    ] is True
    assert invariants["candidate_funnel_order"] == (
        "macro_phase1_shortlist_then_guarded_singleton_phase1_shortlist_"
        "then_singleton_phase2_then_singleton_phase3_v1"
    )


def test_macro_only_gradient_phase0_route_binds_proxy_then_qiskit_without_singleton_exposure(
) -> None:
    request = _request(adapter=MacroGradientPhase0CandidateAdapter())
    _request_profile, profile, contract, digest = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        algorithm_id=(
            RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID
        ),
    )

    assert profile.endswith(
        RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ROUTE_SUFFIX
    )
    assert len(digest) == 64
    execution = contract["execution_settings"]
    invariants = contract["semantic_invariants"]
    assert execution["phase3_backend_cost_scope"] == (
        BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1
    )
    assert execution["static_lane_route"] == (
        STATIC_LANE_ROUTE_GLOBAL_SINGLE_POPULATION
    )
    assert execution["ra_phase0_gradient_shortlist_size"] == 24
    assert invariants["phase0_compile_cost_active"] is False
    assert invariants["phase_i_compile_cost_source"] == "structural_proxy_v1"
    assert invariants["phase_ii_compile_cost_source"] == "backend_transpile_v1"
    assert invariants["phase_iii_compile_cost_source"] == "backend_transpile_v1"
    assert invariants["selector_qiskit_compile_cost_active"] is True
    assert invariants["macro_generator_identity_preserved_all_phases"] is True
    assert invariants["singleton_child_exposure_active"] is False
    assert invariants["candidate_funnel_order"] == (
        "macro_gradient_phase0_shortlist_then_macro_phase1_then_identity_"
        "macro_phase2_then_macro_phase3_v1"
    )
    assert invariants["plateau_prior_mean_decrease_ratio_threshold"] == 1.0e-4


@pytest.mark.parametrize(
    ("adapter", "algorithm_id", "funnel_prefix", "route_suffix"),
    [
        (
            MacroGradientPhase0ThenSingletonCandidateAdapter(),
            RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
            "macro_gradient_phase0_shortlist",
            RA_ADAPT_MACRO_GRADIENT_PHASE0_PHASE23_QISKIT_ROUTE_SUFFIX,
        ),
        (
            GlobalSingletonGradientPhase0CandidateAdapter(),
            RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
            "global_singleton_gradient_phase0_shortlist",
            RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ROUTE_SUFFIX,
        ),
    ],
)
def test_gradient_phase0_routes_bind_abs_gradient_without_metric_or_cost(
    adapter: object,
    algorithm_id: str,
    funnel_prefix: str,
    route_suffix: str,
) -> None:
    request = _request(adapter=adapter)
    _request_profile, profile, contract, digest = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        algorithm_id=algorithm_id,
    )

    assert profile.endswith(route_suffix)
    assert len(digest) == 64
    execution = contract["execution_settings"]
    invariants = contract["semantic_invariants"]
    assert execution["ra_phase0_gradient_shortlist_size"] == 24
    assert execution["static_lane_route"] == (
        STATIC_LANE_ROUTE_GLOBAL_SINGLE_POPULATION
    )
    assert invariants["phase0_active"] is True
    assert invariants["phase0_score"] == (
        "standard_adapt_absolute_gradient_v1"
    )
    assert invariants["phase0_fubini_metric_active"] is False
    assert invariants["phase0_resource_cost_active"] is False
    assert invariants["phase0_compile_cost_active"] is False
    assert invariants["phase0_estimator_components"] == ["N_grad"]
    assert invariants["candidate_funnel_order"].startswith(funnel_prefix)
    assert invariants["selector_compile_cost_scope"] == (
        BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1
    )


@pytest.mark.parametrize(
    "adapter",
    [
        MacroGradientPhase0ThenSingletonCandidateAdapter(),
        GlobalSingletonGradientPhase0CandidateAdapter(),
    ],
)
def test_gradient_phase0_adapters_reject_unrelated_algorithm_identity(
    adapter: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="gradient-Phase-0 adapter and algorithm identity",
    ):
        _repaired_route_contract(
            _request(adapter=adapter),
            active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=RA_ADAPT_ALGORITHM_ID,
        )


def test_phase23_qiskit_scope_selects_proxy_then_qiskit_oracles() -> None:
    proxy_oracle = object()
    qiskit_oracle = object()
    context = SimpleNamespace(
        backend_compile_scope=(
            BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1
        ),
        backend_compile_oracle=proxy_oracle,
        phase3_backend_compile_oracle=qiskit_oracle,
    )
    pending = SimpleNamespace(
        backend_compile_snapshot="proxy-snapshot",
        phase3_backend_compile_snapshot="qiskit-snapshot",
    )

    assert adapt_pipeline._default_no_prune_compile_oracle_for_stage(
        context=context,
        pending=pending,
        evaluation_stage="phase1",
    ) == (None, None, False)
    for stage in ("phase2", "phase3"):
        assert adapt_pipeline._default_no_prune_compile_oracle_for_stage(
            context=context,
            pending=pending,
            evaluation_stage=stage,
        ) == (qiskit_oracle, "qiskit-snapshot", True)


def test_phase23_qiskit_signed_cost_rewards_true_compiled_cancellation() -> None:
    negative = _compiled_feature(
        label="cancels",
        pool_index=0,
        c2q=0.0,
        depth2q=0.0,
        c1q=0.0,
    )
    positive = _compiled_feature(
        label="adds",
        pool_index=1,
        c2q=3.0,
        depth2q=2.0,
        c1q=1.0,
    )
    negative.compiled_position_cost_backend.update(
        {
            "raw_delta_compiled_count_2q": -3.0,
            "raw_delta_compiled_depth_2q": -2.0,
            "raw_delta_compiled_count_1q": -1.0,
            "negative_delta_reward_enabled": True,
        }
    )
    positive.compiled_position_cost_backend.update(
        {"negative_delta_reward_enabled": True}
    )
    records = rescore_hardware_cost_family(
        [
            {"feature": negative},
            {"feature": positive},
        ],
        FullScoreConfig(
            hardware_cost_normalization_mode=(
                HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
            )
        ),
    )

    cancel_feature = records[0]["feature"]
    add_feature = records[1]["feature"]
    assert cancel_feature.hardware_cost_score_factor > 1.0
    assert add_feature.hardware_cost_score_factor < 1.0
    assert cancel_feature.hardware_cost_signed_index < 0.0
    assert add_feature.hardware_cost_signed_index > 0.0


def _request(
    *,
    adapter=None,
    insertion=None,
) -> RAAdaptRequest:
    return RAAdaptRequest(
        adapter=(
            GlobalSinglePauliWordCandidateAdapter()
            if adapter is None
            else adapter
        ),
        method=SRMethodPolicy(
            insertion=(
                PlateauCommutationInsertion()
                if insertion is None
                else insertion
            )
        ),
    )


def _compiled_feature(
    *,
    label: str,
    pool_index: int,
    c2q: float,
    depth2q: float,
    c1q: float,
) -> CandidateFeatures:
    return CandidateFeatures(
        stage_name="phase3",
        candidate_label=label,
        candidate_family="fixture",
        candidate_pool_index=pool_index,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        g_signed=0.8,
        g_abs=0.8,
        g_lcb=0.8,
        sigma_hat=0.0,
        F=1.0,
        novelty=1.0,
        curvature_mode="fixture",
        novelty_mode="fixture",
        refit_window_indices=[],
        compiled_position_cost_proxy={},
        measurement_cache_stats={},
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        simple_score=None,
        score_version="fixture_v1",
        c_hat_2q=c2q,
        c_hat_d=depth2q,
        c_hat_1q=c1q,
        hardware_cost_source="backend_transpile_v1",
        compile_cost_source="backend_transpile_v1",
        compile_gate_open=True,
        generator_id=f"generator::{label}",
        compiled_position_cost_backend={
            "selected_backend_name": "FakeMarrakesh",
            "selected_resolution_kind": "fake_exact",
            "raw_delta_compiled_count_2q": c2q,
            "delta_compiled_count_2q": c2q,
            "raw_delta_compiled_depth_2q": depth2q,
            "delta_compiled_depth_2q": depth2q,
            "raw_delta_compiled_count_1q": c1q,
            "delta_compiled_count_1q": c1q,
            "base_structure_key": "1" * 64,
            "trial_structure_key": "2" * 64,
            "base_initial_layout": None,
            "trial_initial_layout": None,
            "base_logical_to_physical": [0, 1],
            "trial_logical_to_physical": [0, 1],
            "base_trial_layout_coupling_policy": (
                "independent_unconstrained_full_transpiles_v1"
            ),
        },
    )


def test_phase3_only_route_is_exactly_derived_from_page7_source() -> None:
    _request_profile, profile, contract, digest = _repaired_route_contract(
        _request(),
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        algorithm_id=RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID,
    )

    assert profile == (
        SOURCE_ROUTE_PROFILE
        + "__"
        + RA_ADAPT_PHASE3_QISKIT_COST_ROUTE_SUFFIX
    )
    assert len(digest) == 64
    lineage = contract["lineage_authority"]
    assert lineage["parent_route_profile"] == SOURCE_ROUTE_PROFILE
    assert lineage["parent_contract_sha256"] == SOURCE_ROUTE_SHA256
    assert lineage["only_intended_scientific_changes"][-1] == (
        "phase3_selector_cost_graph_span_to_qiskit_positive_clipped_"
        "marginal_transpile"
    )

    execution = contract["execution_settings"]
    assert execution["phase3_backend_cost_mode"] == MARRAKESH_GRAPH_SPAN_MODE
    assert execution["phase3_backend_cost_scope"] == (
        BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
    )
    assert execution["phase3_backend_name"] == "FakeMarrakesh"
    assert execution["phase3_backend_optimization_level"] == 1
    assert execution["phase3_backend_transpile_seed"] == 7

    invariants = contract["semantic_invariants"]
    assert invariants["selector_compile_cost_policy"] == (
        RA_ADAPT_PHASE3_QISKIT_COST_POLICY
    )
    assert invariants["selector_compile_cost_phase_reuse"] == (
        RA_ADAPT_PHASE3_QISKIT_COST_PHASE_REUSE
    )
    assert invariants["selector_compile_cost_scope"] == (
        BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
    )
    assert invariants["phase_iii_qiskit_one_qubit_coordinate_policy"] == (
        ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1
    )
    assert invariants["phase_iii_qiskit_selector_circuit_coordinates"] == [
        "positive_clip_delta_N2q",
        "positive_clip_delta_D2q",
        "positive_clip_delta_N1q",
    ]
    assert invariants["phase_iii_qiskit_structure_theta_value"] == 1.0
    assert invariants["phase_iii_qiskit_population_rescore_policy"] == (
        "complete_evaluated_phase3_population_before_ranking_v1"
    )
    assert invariants[
        "phase_iii_qiskit_population_normalization_policy"
    ] == "family_robust_symmetric_arctan_v1"
    assert invariants["phase_iii_qiskit_failure_policy"] == "abort_run_v1"


def test_phase3_only_resolved_protocol_binds_route_without_full_response_semantics(
) -> None:
    problem = resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        )
    )
    request = _request()
    cell = bundle_module.BundleCellSpec(
        cell_id="phase3_qiskit_protocol_fixture",
        stage="validation",
        regime_id="fixture",
        nph=1,
        route_id="global_singleton_plateau_phase3_qiskit",
        algorithm_id=RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID,
        selector_family="ra_adapt",
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        horizon=1,
        source_lock_id="fixture_lock",
    )
    source_lock_refs = {
        "source_locks_manifest_sha256": "1" * 64,
        "implementation_source_inventory_sha256": "2" * 64,
        "cell_source_lock_id": "fixture_lock",
        "cell_source_lock_sha256": "3" * 64,
        "visible_provenance_sha256": "4" * 64,
        "provenance_tracker_sha256": "5" * 64,
        "ed_cutoff_reference_sha256": "6" * 64,
        "resolver_script_sha256": "7" * 64,
    }
    authority = bundle_module._bundle_protocol_materialization_authority(
        cell=cell,
        bundle_id="phase3_qiskit_protocol_fixture_bundle",
        bundle_manifest_sha256="8" * 64,
        source_locks_sha256="1" * 64,
        source_lock_refs=source_lock_refs,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )

    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=authority,
    )
    _profile_request, _profile, _contract, route_sha256 = (
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=(
                RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
            ),
            problem=problem,
        )
    )

    assert "algorithm_semantics" not in protocol.lineage_authority
    assert protocol.route_contract is not None
    assert protocol.route_contract["sha256"] == route_sha256


def test_phase3_qiskit_denominator_no_lanes_protocol_is_literal_and_tau1em6(
) -> None:
    problem = resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        )
    )
    request = _request()
    algorithm_id = (
        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID
    )
    cell = bundle_module.BundleCellSpec(
        cell_id="phase3_qiskit_denominator_no_lanes_fixture",
        stage="validation",
        regime_id="fixture",
        nph=1,
        route_id="global_singleton_plateau_phase3_qiskit_denominator_no_lanes",
        algorithm_id=algorithm_id,
        selector_family="ra_adapt",
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        horizon=50,
        source_lock_id="fixture_lock",
    )
    source_lock_refs = {
        "source_locks_manifest_sha256": "1" * 64,
        "implementation_source_inventory_sha256": "2" * 64,
        "cell_source_lock_id": "fixture_lock",
        "cell_source_lock_sha256": "3" * 64,
        "visible_provenance_sha256": "4" * 64,
        "provenance_tracker_sha256": "5" * 64,
        "ed_cutoff_reference_sha256": "6" * 64,
        "resolver_script_sha256": "7" * 64,
    }
    authority = bundle_module._bundle_protocol_materialization_authority(
        cell=cell,
        bundle_id="phase3_qiskit_denominator_no_lanes_fixture_bundle",
        bundle_manifest_sha256="8" * 64,
        source_locks_sha256="1" * 64,
        source_lock_refs=source_lock_refs,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )

    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=authority,
    )

    assert protocol.algorithm_id == algorithm_id
    assert protocol.horizon == 50
    assert protocol.route_contract is not None
    execution = protocol.route_contract["execution_settings"]
    invariants = protocol.route_contract["semantic_invariants"]
    assert execution["phase3_hardware_cost_normalization_mode"] == (
        HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1
    )
    assert execution["static_lane_route"] == (
        STATIC_LANE_ROUTE_GLOBAL_SINGLE_POPULATION
    )
    assert "physical_lane_shortlist_aggressiveness" not in execution
    assert invariants["selector_compile_cost_policy"] == (
        RA_ADAPT_PHASE3_QISKIT_DENOMINATOR_POLICY
    )
    assert invariants["phase_iii_qiskit_population_normalization_policy"] == (
        HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1
    )
    assert invariants["phase_iii_score_formula"] == (
        "B3/(1+lambda_2q*cbar_2q+lambda_d*cbar_d+"
        "lambda_1q*cbar_1q)"
    )
    assert invariants["phase_iii_qiskit_theta_and_shot_lambdas"] == {
        "theta": 0.0,
        "shot": 0.0,
    }
    assert invariants["physical_operator_lanes_active"] is False
    assert invariants["shortlist_population_policy"] == (
        "single_global_population_v1"
    )
    assert invariants["plateau_prior_mean_decrease_ratio_threshold"] == (
        1.0e-6
    )


def test_phase3_qiskit_denominator_receipt_carries_literal_cost_burden() -> None:
    oracle = SimpleNamespace(estimate_count=0)
    features = [
        _compiled_feature(
            label="expensive",
            pool_index=0,
            c2q=8.0,
            depth2q=7.0,
            c1q=6.0,
        ),
        _compiled_feature(
            label="cheap",
            pool_index=1,
            c2q=1.0,
            depth2q=1.0,
            c1q=1.0,
        ),
    ]
    retained = [
        {
            "feature": feature,
            "candidate_label": feature.candidate_label,
            "candidate_pool_index": feature.candidate_pool_index,
            "position_id": feature.position_id,
        }
        for feature in features
    ]
    factories = {}
    for retained_row, feature in zip(retained, features, strict=True):
        key = adapt_pipeline._batch_admission_record_key(retained_row)

        def _factory(feature=feature, retained_row=retained_row):
            oracle.estimate_count += 1
            return ([{**retained_row, "feature": feature}], {})

        factories[key] = _factory

    records, projected, _shortlisted = (
        adapt_pipeline._default_no_prune_projected_phase3_population(
            phase2_shortlisted_records=retained,
            archival_phase3_factory_by_parent_key=factories,
            archival_phase2_parent_expansions=[],
            phase2_full_records_evaluated=retained,
            controller_snapshot=None,
            phase3_live=False,
            pool_generator_registry={},
            phase2_score_cfg_round=FullScoreConfig(
                hardware_cost_normalization_mode=(
                    HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1
                )
            ),
            phase3_runtime_split_summary={},
            phase_shortlist_runtime=None,
            phase3_shortlist_size=None,
            phase3_backend_compile_oracle=oracle,
            backend_compile_scope=(
                BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
            ),
        )
    )

    receipt = projected["phase3_qiskit_selector_cost_receipt"]
    assert receipt["population_normalization"] == (
        HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1
    )
    by_label = {row["candidate_label"]: row for row in receipt["rows"]}
    assert by_label["cheap"]["hardware_cost_denominator"] == 1.0
    assert by_label["expensive"]["hardware_cost_denominator"] > 1.0
    assert by_label["expensive"]["hardware_cost_policy"] == (
        HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1
    )
    assert records[0]["candidate_label"] == "cheap"


@pytest.mark.parametrize(
    ("case_request", "gradient_policy", "resource_scope"),
    [
        (_request(adapter=MacroCandidateAdapter()), ACTIVE_GRADIENT_STATIONARY, RESOURCE_WEIGHTING_ALL_PHASE),
        (_request(insertion=AppendOnlyInsertion()), ACTIVE_GRADIENT_STATIONARY, RESOURCE_WEIGHTING_ALL_PHASE),
        (_request(), ACTIVE_GRADIENT_MEASURED, RESOURCE_WEIGHTING_ALL_PHASE),
        (_request(), ACTIVE_GRADIENT_STATIONARY, RESOURCE_WEIGHTING_LATE),
    ],
)
def test_phase3_only_route_rejects_source_identity_drift(
    case_request: RAAdaptRequest,
    gradient_policy: str,
    resource_scope: str,
) -> None:
    with pytest.raises(ValueError, match="source-locked"):
        _repaired_route_contract(
            case_request,
            active_gradient_policy=gradient_policy,
            resource_weighting_scope=resource_scope,
            algorithm_id=(
                RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
            ),
        )


@pytest.mark.parametrize(
    ("surface", "field", "tampered"),
    [
        ("execution_settings", "phase3_backend_cost_mode", "proxy"),
        ("execution_settings", "phase3_backend_cost_scope", "wrong"),
        ("execution_settings", "phase3_backend_name", "FakeFez"),
        ("execution_settings", "phase3_backend_optimization_level", 0),
        ("execution_settings", "phase3_backend_transpile_seed", 8),
        ("semantic_invariants", "selector_compile_cost_policy", "wrong"),
        ("semantic_invariants", "phase_iii_qiskit_failure_policy", "skip"),
        (
            "semantic_invariants",
            "phase_iii_qiskit_one_qubit_coordinate_policy",
            "proxy_baseline_v1",
        ),
        (
            "semantic_invariants",
            "phase_iii_qiskit_population_rescore_policy",
            "per_candidate_v1",
        ),
        (
            "semantic_invariants",
            "phase_iii_qiskit_population_normalization_policy",
            "raw_legacy_v1",
        ),
    ],
)
def test_executed_route_semantics_reject_rehashed_contract_tampering(
    surface: str,
    field: str,
    tampered: object,
) -> None:
    request = _request()
    _request_profile, _profile, contract, _digest = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        algorithm_id=RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID,
    )
    _validate_executed_insertion_contract(
        request,
        contract,
        algorithm_id=RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID,
    )
    altered = copy.deepcopy(contract)
    altered[surface][field] = tampered

    with pytest.raises(RuntimeError, match="source-locked"):
        _validate_executed_insertion_contract(
            request,
            altered,
            algorithm_id=(
                RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
            ),
        )


def test_compile_oracle_stage_routing_preserves_existing_scopes() -> None:
    graph_oracle = object()
    graph_snapshot = object()
    qiskit_oracle = object()
    qiskit_snapshot = object()
    context = SimpleNamespace(
        backend_compile_oracle=graph_oracle,
        phase3_backend_compile_oracle=qiskit_oracle,
    )
    pending = SimpleNamespace(
        backend_compile_snapshot=graph_snapshot,
        phase3_backend_compile_snapshot=qiskit_snapshot,
    )

    for stage in ("phase1", "phase2", "full"):
        assert adapt_pipeline._default_no_prune_compile_oracle_for_stage(
            context=context,
            pending=pending,
            evaluation_stage=stage,
        ) == (graph_oracle, graph_snapshot, False)
    assert adapt_pipeline._default_no_prune_compile_oracle_for_stage(
        context=context,
        pending=pending,
        evaluation_stage="phase3",
    ) == (qiskit_oracle, qiskit_snapshot, True)

    existing_all_phase_context = SimpleNamespace(
        backend_compile_oracle=qiskit_oracle,
        phase3_backend_compile_oracle=None,
    )
    for stage in ("phase1", "phase2", "phase3", "full"):
        assert adapt_pipeline._default_no_prune_compile_oracle_for_stage(
            context=existing_all_phase_context,
            pending=pending,
            evaluation_stage=stage,
        ) == (qiskit_oracle, graph_snapshot, False)


@pytest.mark.parametrize("increment_per_factory", [0, 2])
def test_phase3_qiskit_estimate_count_must_close_exact_population(
    increment_per_factory: int,
) -> None:
    oracle = SimpleNamespace(estimate_count=0)
    features = [
        _compiled_feature(
            label="a",
            pool_index=0,
            c2q=5.0,
            depth2q=4.0,
            c1q=3.0,
        ),
        _compiled_feature(
            label="b",
            pool_index=1,
            c2q=1.0,
            depth2q=1.0,
            c1q=1.0,
        ),
    ]
    retained = [
        {
            "feature": feature,
            "candidate_label": feature.candidate_label,
            "candidate_pool_index": feature.candidate_pool_index,
            "position_id": feature.position_id,
        }
        for feature in features
    ]
    factories = {}
    for retained_row, feature in zip(retained, features, strict=True):
        key = adapt_pipeline._batch_admission_record_key(retained_row)

        def _factory(feature=feature):
            oracle.estimate_count += increment_per_factory
            return (
                [
                    {
                        "feature": feature,
                        "candidate_label": feature.candidate_label,
                        "candidate_pool_index": feature.candidate_pool_index,
                        "position_id": feature.position_id,
                    }
                ],
                {},
            )

        factories[key] = _factory

    with pytest.raises(RuntimeError, match="did not close"):
        adapt_pipeline._default_no_prune_projected_phase3_population(
            phase2_shortlisted_records=retained,
            archival_phase3_factory_by_parent_key=factories,
            archival_phase2_parent_expansions=[],
            phase2_full_records_evaluated=retained,
            controller_snapshot=None,
            phase3_live=False,
            pool_generator_registry={},
            phase2_score_cfg_round=FullScoreConfig(
                hardware_cost_normalization_mode=(
                    HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
                )
            ),
            phase3_runtime_split_summary={},
            phase_shortlist_runtime=None,
            phase3_shortlist_size=None,
            phase3_backend_compile_oracle=oracle,
            backend_compile_scope=(
                BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
            ),
        )


def test_phase3_qiskit_receipt_closes_and_uses_one_population_normalization(
) -> None:
    oracle = SimpleNamespace(estimate_count=0)
    features = [
        _compiled_feature(
            label="expensive",
            pool_index=0,
            c2q=8.0,
            depth2q=7.0,
            c1q=6.0,
        ),
        _compiled_feature(
            label="cheap",
            pool_index=1,
            c2q=1.0,
            depth2q=1.0,
            c1q=1.0,
        ),
    ]
    retained = [
        {
            "feature": feature,
            "candidate_label": feature.candidate_label,
            "candidate_pool_index": feature.candidate_pool_index,
            "position_id": feature.position_id,
        }
        for feature in features
    ]
    factories = {}
    for retained_row, feature in zip(retained, features, strict=True):
        key = adapt_pipeline._batch_admission_record_key(retained_row)

        def _factory(feature=feature):
            oracle.estimate_count += 1
            return (
                [
                    {
                        "feature": feature,
                        "candidate_label": feature.candidate_label,
                        "candidate_pool_index": feature.candidate_pool_index,
                        "position_id": feature.position_id,
                    }
                ],
                {},
            )

        factories[key] = _factory

    records, projected, shortlisted = (
        adapt_pipeline._default_no_prune_projected_phase3_population(
            phase2_shortlisted_records=retained,
            archival_phase3_factory_by_parent_key=factories,
            archival_phase2_parent_expansions=[],
            phase2_full_records_evaluated=retained,
            controller_snapshot=None,
            phase3_live=False,
            pool_generator_registry={},
            phase2_score_cfg_round=FullScoreConfig(
                hardware_cost_normalization_mode=(
                    HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
                )
            ),
            phase3_runtime_split_summary={},
            phase_shortlist_runtime=None,
            phase3_shortlist_size=None,
            phase3_backend_compile_oracle=oracle,
            backend_compile_scope=(
                BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
            ),
        )
    )

    assert shortlisted == []
    assert len(records) == 2
    receipt = projected["phase3_qiskit_selector_cost_receipt"]
    assert projected["phase3_evaluated_candidate_count"] == 2
    assert receipt["phase3_evaluated_candidate_count"] == 2
    assert receipt["phase3_qiskit_estimate_count_delta"] == 2
    assert len(receipt["rows"]) == 2
    assert receipt["rows_sha256"] == (
        adapt_pipeline._candidate_record_payload_digest(receipt["rows"])
    )
    population_identities = projected[
        "phase3_evaluated_population_identities"
    ]
    assert projected[
        "phase3_evaluated_population_identities_sha256"
    ] == adapt_pipeline._candidate_record_payload_digest(
        population_identities
    )
    assert population_identities == [
        {
            key: row[key]
            for key in (
                "candidate_label",
                "candidate_pool_index",
                "generator_id",
                "position_id",
            )
        }
        for row in receipt["rows"]
    ]
    assert all(
        row["base_structure_key"] != row["trial_structure_key"]
        and row["base_initial_layout"] is None
        and row["trial_initial_layout"] is None
        and row["base_logical_to_physical"] == [0, 1]
        and row["trial_logical_to_physical"] == [0, 1]
        and row["base_trial_layout_coupling_policy"]
        == "independent_unconstrained_full_transpiles_v1"
        for row in receipt["rows"]
    )
    population_hashes = {
        row["hardware_cost_population_hash"] for row in receipt["rows"]
    }
    assert len(population_hashes) == 1
    assert None not in population_hashes
    assert records[0]["candidate_label"] == "cheap"

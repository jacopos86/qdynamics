from __future__ import annotations

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.static_adapt.ra_adapt.adapters import (
    MACRO_GRADIENT_PHASE0_THEN_SINGLETON_ADAPTER_ID,
    MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY,
    POST_EXPOSURE_PHASE_I_RETAINED_PARENT_SINGLETONS,
    MacroGradientPhase0ThenSingletonCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    RAAdaptRequest,
    ra_adapt_request_from_mapping,
)


def test_macro_gradient_phase0_adapter_identity_round_trips_exactly() -> None:
    adapter = MacroGradientPhase0ThenSingletonCandidateAdapter()
    request = RAAdaptRequest(adapter=adapter)

    restored = ra_adapt_request_from_mapping(request.to_dict())

    assert isinstance(
        restored.adapter,
        MacroGradientPhase0ThenSingletonCandidateAdapter,
    )
    assert restored.to_dict() == request.to_dict()
    assert adapter.adapter_id == (
        MACRO_GRADIENT_PHASE0_THEN_SINGLETON_ADAPTER_ID
    )
    assert adapter.candidate_representation_id == (
        CANDIDATE_REPRESENTATION_SINGLE_PAULI
    )
    assert adapter.macro_phase0_policy_id == (
        MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY
    )
    assert adapter.post_exposure_phase_i_shortlist_id == (
        POST_EXPOSURE_PHASE_I_RETAINED_PARENT_SINGLETONS
    )


def test_macro_phase0_parent_context_is_gradient_only_and_position_exact(
) -> None:
    feature = adapt_pipeline._macro_gradient_phase0_parent_context_feature(
        stage_name="residual",
        candidate_label="parent::macro",
        candidate_family="fermion_hop",
        candidate_pool_index=7,
        position_id=2,
        append_position=4,
        positions_considered=[0, 2, 4],
        gradient_signed=-0.75,
        refit_window_indices=[0, 1],
        phase3_geometry_window_indices=[0, 1, 3],
        phase3_geometry_active_post_indices=[0, 1, 3, 4],
        generator_metadata={"generator_id": "macro::7"},
        symmetry_spec={"sector": "fixture"},
        controller_snapshot={"step_index": 5},
    )

    assert feature.position_id == 2
    assert feature.positions_considered == [0, 2, 4]
    assert feature.g_signed == -0.75
    assert feature.g_abs == 0.75
    assert feature.phase0_raw_gradient_abs == 0.75
    assert feature.phase0_cost_enabled is False
    assert feature.metric_proxy == 1.0
    assert feature.compile_cost_source == "not_acquired_macro_phase0_context_v1"
    assert feature.measurement_cache_stats == {}

from __future__ import annotations

import numpy as np
import pytest

from pipelines.scaffold.hh_continuation_scoring import (
    FullScoreConfig,
    SimpleScoreConfig,
    phase1_score_payload,
    phase2_raw_geometry_score,
    phase3_canonical_score_components,
)
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    PolicyEchoReceipt,
    RESOURCE_WEIGHTING_ALL_PHASE,
    RESOURCE_WEIGHTING_LATE,
)


def _feature() -> CandidateFeatures:
    return CandidateFeatures(
        stage_name="phase1",
        candidate_label="fixture",
        candidate_family="fixture",
        candidate_pool_index=0,
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
        c_bar_2q=3.0,
    )


def test_resource_scope_changes_only_phase_one_weighting_and_is_receipted(
) -> None:
    feature = _feature()
    cost_fields = {
        "lambda_2q": 1.0,
        "lambda_d": 0.0,
        "lambda_1q": 0.0,
        "lambda_theta": 0.0,
        "lambda_shot": 0.0,
    }
    payloads: dict[str, tuple[dict[str, object], ...]] = {}
    for scope in (
        RESOURCE_WEIGHTING_LATE,
        RESOURCE_WEIGHTING_ALL_PHASE,
    ):
        phase1 = phase1_score_payload(
            feature,
            SimpleScoreConfig(
                **cost_fields,
                resource_weighting_scope=scope,
            ),
        )
        full_cfg = FullScoreConfig(
            **cost_fields,
            resource_weighting_scope=scope,
            z_alpha=0.0,
            rho=0.25,
        )
        phase2 = phase2_raw_geometry_score(
            feature,
            F_raw=1.0,
            h_raw=0.0,
            q_window=[],
            Q_window=np.zeros((0, 0), dtype=float),
            cfg=full_cfg,
        )
        phase3 = phase3_canonical_score_components(feature, full_cfg)
        policy = PolicyEchoReceipt(
            active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
            resource_weighting_scope=scope,
        ).to_dict()
        assert phase1["resource_weighting_scope"] == scope
        assert policy["resource_weighting_scope"] == scope
        payloads[scope] = (phase1, phase2, phase3, policy)

    late1, late2, late3, _late_policy = payloads[
        RESOURCE_WEIGHTING_LATE
    ]
    all1, all2, all3, _all_policy = payloads[
        RESOURCE_WEIGHTING_ALL_PHASE
    ]

    assert late1["phase1_resource_weighting_active"] is False
    assert late1["phase1_effective_cost_factor"] == pytest.approx(1.0)
    assert late1["phase1_effective_burden"] == pytest.approx(1.0)
    assert all1["phase1_resource_weighting_active"] is True
    assert all1["phase1_effective_burden"] == pytest.approx(4.0)
    assert all1["trust_region_score"] == pytest.approx(
        float(all1["trust_region_gain"]) / 4.0
    )

    assert late2["phase2_burden_total"] == pytest.approx(4.0)
    assert late2["phase2_raw_score"] == pytest.approx(
        float(late2["phase2_raw_trust_gain"]) / 4.0
    )
    assert late2 == all2

    assert late3["phase3_denominator_1_plus_K3"] == pytest.approx(4.0)
    assert late3["phase3_primary_score"] == pytest.approx(
        float(late3["DeltaE_TR"]) / 4.0
    )
    assert late3 == all3

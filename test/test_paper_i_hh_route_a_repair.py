from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
import pipelines.static_adapt.runtime_split as runtime_split_module
from pipelines.contracts.static_provenance import HH_FULL_META_CLASSIFIER_VERSION
from pipelines.static_adapt.cli_config import (
    _build_run_hardcoded_adapt_vqe_kwargs,
)
from pipelines.static_adapt.route_a_child_padding import (
    ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1,
    RouteAChildPaddingConfig,
)
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_holstein_hamiltonian,
)


@pytest.fixture(autouse=True)
def _disable_adapt_caches_and_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)


@pytest.fixture(scope="module")
def _hh_nph2_hamiltonian():
    return build_hubbard_holstein_hamiltonian(
        dims=2,
        J=1.0,
        U=0.5,
        omega0=1.0,
        g=0.2,
        n_ph_max=2,
        boson_encoding="binary",
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_zero_point=True,
    )


def _write_two_singlet_pool_filter(tmp_path: Path) -> Path:
    path = tmp_path / "hh_full_meta_uccsd_sing_only.json"
    path.write_text(
        json.dumps(
            {
                "keep_classes": ["uccsd_sing"],
                "classifier_version": HH_FULL_META_CLASSIFIER_VERSION,
                "source_pool": "full_meta",
                "source_problem": "hh",
                "source_num_sites": 2,
                "source_n_ph_max": 2,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_initial_outer_measurement_reuse_requires_exact_state_identity() -> None:
    state = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
    assert adapt_pipeline._initial_outer_measurement_state_matches(
        state, state.copy()
    )

    nearby = state.copy()
    nearby[1] = 1.0e-13 + 0.0j
    assert not adapt_pipeline._initial_outer_measurement_state_matches(
        state, nearby
    )


def test_sr_escape_phase1_population_keeps_zero_score_record_outside_ordinary_cap() -> None:
    ordinary = {
        "candidate_label": "ordinary",
        "candidate_pool_index": 1,
        "position_id": 0,
        "simple_score": 1.0,
    }
    zero_gradient = {
        "candidate_label": "zero-gradient-saddle-candidate",
        "candidate_pool_index": 2,
        "position_id": 0,
        "simple_score": 0.0,
    }

    records, summary = adapt_pipeline._sr_escape_phase1_evaluation_population(
        [ordinary, zero_gradient],
        [ordinary],
        active=True,
    )

    assert [record["candidate_label"] for record in records] == [
        "ordinary",
        "zero-gradient-saddle-candidate",
    ]
    assert records[0]["sr_escape_ordinary_phase1_eligible"] is True
    assert records[1]["sr_escape_ordinary_phase1_eligible"] is False
    assert records[1]["sr_escape_phase1_reachable"] is True
    assert summary["service_policy"] == (
        "complete_finite_eager_no_gradient_gate_v1"
    )
    assert summary["zero_gradient_gate_applied"] is False
    assert summary["population_complete"] is True


def test_sr_escape_phase3_population_rejects_duplicate_reachable_ids() -> None:
    record = {
        "candidate_label": "duplicate",
        "candidate_pool_index": 3,
        "position_id": 1,
    }
    with pytest.raises(RuntimeError, match="duplicate record ids"):
        adapt_pipeline._sr_escape_ordered_phase3_population([record, record])


def test_sr_escape_fixed_phi_keeps_every_valid_child_and_not_parent() -> None:
    parent = {
        "candidate_label": "parent",
        "candidate_pool_index": 7,
        "position_id": 2,
        "phase2_raw_score": 10.0,
    }
    child_a = {
        "candidate_label": "child-a",
        "candidate_pool_index": 7,
        "position_id": 2,
        "phase2_raw_score": 4.0,
    }
    child_b_zero_ordinary = {
        "candidate_label": "child-b",
        "candidate_pool_index": 7,
        "position_id": 2,
        "phase2_raw_score": 0.0,
    }

    successors = adapt_pipeline._sr_escape_runtime_split_successor_population(
        parent_record=parent,
        split_child_set_entries=[
            {"candidate_label": "child-b", "record": child_b_zero_ordinary},
            {"candidate_label": "child-a", "record": child_a},
        ],
        active=True,
        runtime_split_required=True,
    )

    assert [record["candidate_label"] for record in successors] == [
        "child-a",
        "child-b",
    ]
    assert all(record["candidate_label"] != "parent" for record in successors)
    assert successors[1]["phase2_raw_score"] == 0.0

    ordinary_off, escape_off = (
        adapt_pipeline._sr_escape_runtime_split_record_domains(
            parent_record=parent,
            ordered_ordinary_variants=[child_a, child_b_zero_ordinary],
            split_child_set_entries=[
                {"candidate_label": "child-a", "record": child_a},
                {
                    "candidate_label": "child-b",
                    "record": child_b_zero_ordinary,
                },
            ],
            escape_active=False,
            runtime_split_required=True,
            ordinary_runtime_split_required=True,
        )
    )
    ordinary_on, escape_on = (
        adapt_pipeline._sr_escape_runtime_split_record_domains(
            parent_record=parent,
            ordered_ordinary_variants=[child_a, child_b_zero_ordinary],
            split_child_set_entries=[
                {"candidate_label": "child-a", "record": child_a},
                {
                    "candidate_label": "child-b",
                    "record": child_b_zero_ordinary,
                },
            ],
            escape_active=True,
            runtime_split_required=True,
            ordinary_runtime_split_required=True,
        )
    )
    assert [record["candidate_label"] for record in ordinary_off] == [
        "child-a"
    ]
    assert ordinary_on == ordinary_off
    assert escape_off == []
    assert [record["candidate_label"] for record in escape_on] == [
        "child-a",
        "child-b",
    ]


def test_sr_escape_no_split_parent_has_total_empty_successor_ledger() -> None:
    parent = {
        "candidate_label": "parent",
        "candidate_pool_index": 7,
        "position_id": 2,
    }

    ordinary_off, escape_off = adapt_pipeline._sr_escape_runtime_split_record_domains(
        parent_record=parent,
        ordered_ordinary_variants=[parent],
        split_child_set_entries=[],
        escape_active=False,
        runtime_split_required=False,
        ordinary_runtime_split_required=False,
    )
    ordinary_on, escape_on = adapt_pipeline._sr_escape_runtime_split_record_domains(
        parent_record=parent,
        ordered_ordinary_variants=[parent],
        split_child_set_entries=[],
        escape_active=True,
        runtime_split_required=False,
        ordinary_runtime_split_required=False,
    )

    assert ordinary_off == [parent]
    assert ordinary_on == ordinary_off
    assert escape_off == []
    assert escape_on == [parent]


def test_sr_escape_fixed_phi_fails_closed_on_unevaluated_valid_child() -> None:
    parent = {
        "candidate_label": "parent",
        "candidate_pool_index": 7,
        "position_id": 2,
    }
    with pytest.raises(RuntimeError, match="lacks its full Phase-II evaluation"):
        adapt_pipeline._sr_escape_runtime_split_successor_population(
            parent_record=parent,
            split_child_set_entries=[{"candidate_label": "child-a", "record": None}],
            active=True,
            runtime_split_required=True,
        )


def test_sr_escape_cost_normalization_cannot_contaminate_ordinary_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def _family_rescore(
        records: list[dict[str, Any]], _cfg: object
    ) -> list[dict[str, Any]]:
        rows = [dict(record) for record in records]
        calls.append([str(record["candidate_label"]) for record in rows])
        scale = max(float(record["cost"]) for record in rows)
        return [
            {
                **record,
                "normalized_score": float(record["cost"]) / scale,
            }
            for record in rows
        ]

    monkeypatch.setattr(
        adapt_pipeline,
        "rescore_hardware_cost_family",
        _family_rescore,
    )
    ordinary = [
        {
            "candidate_label": "ord-a",
            "cost": 1.0,
            "sr_escape_ordinary_phase1_eligible": True,
        },
        {
            "candidate_label": "ord-b",
            "cost": 2.0,
            "sr_escape_ordinary_phase1_eligible": True,
        },
    ]
    outlier = {
        "candidate_label": "escape-outlier",
        "cost": 1000.0,
        "sr_escape_ordinary_phase1_eligible": False,
    }

    disabled_ordinary, disabled_escape = (
        adapt_pipeline._sr_escape_rescore_phase2_domains(
            ordinary,
            [],
            cfg=object(),
            active=False,
        )
    )
    saddle_ordinary, saddle_escape = (
        adapt_pipeline._sr_escape_rescore_phase2_domains(
            [*ordinary, outlier],
            [outlier],
            cfg=object(),
            active=True,
        )
    )

    assert saddle_ordinary == disabled_ordinary
    assert disabled_escape == []
    assert [record["candidate_label"] for record in saddle_escape] == [
        "escape-outlier"
    ]
    assert calls == [
        ["ord-a", "ord-b"],
        ["ord-a", "ord-b"],
        ["escape-outlier"],
    ]


def _depth_one_hh_kwargs(
    *,
    h_poly: Any,
    pool_filter_json: Path,
    padding_policy: str,
    source_lock_sequence: str | None = None,
) -> dict[str, Any]:
    if padding_policy == ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1:
        padding_config = RouteAChildPaddingConfig(
            policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
            problem_key="hh",
            num_sites=2,
            n_ph_max=2,
            boson_encoding="binary",
            total_register_width=8,
        )
    else:
        padding_config = RouteAChildPaddingConfig(
            policy=ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1,
        )
    return {
        "h_poly": h_poly,
        "num_sites": 2,
        "ordering": "blocked",
        "problem": "hh",
        "adapt_pool": "full_meta",
        "adapt_pool_class_filter_json": pool_filter_json,
        "t": 1.0,
        "u": 0.5,
        "dv": 0.0,
        "boundary": "open",
        "omega0": 1.0,
        "g_ep": 0.2,
        "n_ph_max": 2,
        "boson_encoding": "binary",
        "max_depth": 1,
        "eps_grad": 1e-12,
        "eps_energy": 0.0,
        "maxiter": 2,
        "seed": 7,
        "adapt_inner_optimizer": "POWELL",
        "allow_repeats": True,
        "finite_angle_fallback": False,
        "finite_angle": 0.1,
        "finite_angle_min_improvement": 1e-12,
        "adapt_state_backend": "compiled",
        "adapt_reopt_policy": "windowed",
        "adapt_window_size": 3,
        "adapt_full_refit_every": 0,
        "adapt_final_full_refit": False,
        "adapt_estimator_call_ledger_enabled": True,
        "adapt_drop_floor": -1.0,
        "adapt_grad_floor": -1.0,
        "adapt_continuation_mode": "phase3_v1",
        # This smoke isolates the repaired archival mechanism. Route identity is
        # tested elsewhere against the full Paper-I controller/prune contract.
        "static_route_id": "unspecified",
        "static_meta_feature_profile": "paper_i_production_v1",
        "static_lane_route": "physical_operator_type",
        "phase0_pilot_enabled": True,
        "phase0_pilot_max_operators": 256,
        "phase1_shortlist_size": 256,
        "phase1_probe_max_positions": 8,
        "phase2_shortlist_fraction": 1.0,
        "phase2_shortlist_size": 256,
        "phase3_shortlist_size": 256,
        "phase3_runtime_split_mode": "shortlist_pauli_children_v1",
        "phase3_runtime_split_selection_mode": "archival_child_set_forward_v1",
        "phase3_runtime_split_subset_sizes": "1",
        "phase3_runtime_split_child_set_symmetry_policy": "hard_guard",
        "phase3_source_lock_preferred_sequence": source_lock_sequence,
        "phase3_lifetime_cost_mode": "phase3_v1",
        "phase3_symmetry_mitigation_mode": "verify_only",
        "phase3_enable_rescue": False,
        "phase3_backend_cost_mode": "proxy",
        "route_a_child_padding_config": padding_config,
    }


def _historical_paper_i_budget_kwargs() -> dict[str, Any]:
    return {
        "problem_key": "hh",
        "static_route_id_key": "route_a",
        "static_meta_feature_profile": "paper_i_production_v1",
        "static_lane_route_key": "physical_operator_type",
        "route_a_funnel_active": False,
        "adapt_pool": "full_meta",
        "adapt_continuation_mode": "phase3_v1",
        "phase2_enable_batching": False,
        "phase3_enable_batching": False,
        "phase3_runtime_split_mode": "shortlist_pauli_children_v1",
        "phase3_runtime_split_selection_mode": (
            "archival_child_set_forward_v1"
        ),
        "phase3_runtime_split_max_subset_size": 1,
        "phase3_runtime_split_subset_sizes": "1",
        "physical_lane_shortlist_factor": 3,
        "phase1_shortlist_size_base": 24,
        "phase2_shortlist_size_base": 12,
        "phase2_shortlist_fraction_base": 0.25,
    }


@pytest.mark.parametrize(
    ("regime", "u", "g_ep"),
    [
        ("weak-weak", 0.5, 0.2),
        ("intermediate-weak", 2.0, 0.2),
        ("strong-weak-u8", 8.0, 0.2),
        ("weak-strong", 0.5, 2.0),
        ("intermediate-strong", 2.0, 2.0),
        ("strong-strong-u8", 8.0, 2.0),
    ],
)
def test_historical_paper_i_six_regimes_resolve_july8_shortlist_budget(
    regime: str,
    u: float,
    g_ep: float,
) -> None:
    assert regime
    assert u >= 0.0
    assert g_ep >= 0.0
    contract = adapt_pipeline._resolve_physical_lane_shortlist_budget_contract(
        **_historical_paper_i_budget_kwargs()
    )

    assert contract["historical_paper_i_contract_active"] is True
    assert contract["historical_route_compatibility_id"] == (
        "paper_i_july8_physical_singleton_route_v1"
    )
    assert contract["phase1_shortlist_size_effective"] == 8
    assert contract["phase2_shortlist_size_effective"] == 4
    assert contract["phase2_shortlist_fraction_effective"] == pytest.approx(
        1.0 / 12.0
    )


@pytest.mark.parametrize(
    "override",
    [
        {"static_route_id_key": "route_b"},
        {"static_meta_feature_profile": "safe_core_v1"},
        {"static_lane_route_key": "algebraic"},
        {"route_a_funnel_active": True},
        {"adapt_pool": "hva"},
        {"adapt_continuation_mode": "phase2_v1"},
        {"phase2_enable_batching": True},
        {"phase3_enable_batching": True},
        {"phase3_runtime_split_selection_mode": "joint_response_v1"},
        {"phase3_runtime_split_max_subset_size": 2},
        {"phase3_runtime_split_subset_sizes": "1,2"},
        {"physical_lane_shortlist_factor": 2},
        {"phase1_shortlist_size_base": 32},
        {"phase2_shortlist_size_base": 24},
        {"phase2_shortlist_fraction_base": 0.5},
    ],
)
def test_historical_paper_i_compatibility_does_not_leak_to_other_routes(
    override: dict[str, Any],
) -> None:
    kwargs = _historical_paper_i_budget_kwargs()
    kwargs.update(override)
    contract = adapt_pipeline._resolve_physical_lane_shortlist_budget_contract(
        **kwargs
    )

    assert contract["historical_paper_i_contract_active"] is False
    assert contract["historical_route_compatibility_id"] is None
    assert contract["phase1_shortlist_size_effective"] == int(
        kwargs["phase1_shortlist_size_base"]
    )
    assert contract["phase2_shortlist_size_effective"] == int(
        kwargs["phase2_shortlist_size_base"]
    )
    assert contract["phase2_shortlist_fraction_effective"] == pytest.approx(
        float(kwargs["phase2_shortlist_fraction_base"])
    )


def test_historical_beam_phase1_record_omits_phase2_cheap_score_config() -> None:
    phase2_cfg = object()

    assert (
        adapt_pipeline._phase1_full_record_cheap_score_cfg(
            historical_paper_i_route_compat=True,
            phase3_enabled=True,
            phase2_score_cfg=phase2_cfg,
        )
        is None
    )
    assert (
        adapt_pipeline._phase1_full_record_cheap_score_cfg(
            historical_paper_i_route_compat=False,
            phase3_enabled=True,
            phase2_score_cfg=phase2_cfg,
        )
        is phase2_cfg
    )


def test_cli_propagates_exact_projected_child_padding_and_ledger_enablement(
    tmp_path: Path,
) -> None:
    ledger_json = tmp_path / "estimator_ledger.json"
    args = adapt_pipeline.parse_args(
        [
            "--problem",
            "hh",
            "--L",
            "2",
            "--n-ph-max",
            "2",
            "--boson-encoding",
            "binary",
            "--allow-archival-phase3-runtime-split",
            "--phase3-runtime-split-mode",
            "shortlist_pauli_children_v1",
            "--phase3-runtime-split-selection-mode",
            "archival_child_set_forward_v1",
            "--phase3-runtime-split-subset-sizes",
            "1",
            "--phase3-runtime-split-child-set-symmetry-policy",
            "hard_guard",
            "--phase3-runtime-split-child-padding-policy",
            "exact_projected_grouped_v1",
            "--adapt-estimator-call-ledger-json",
            str(ledger_json),
        ]
    )
    context = SimpleNamespace(layout=SimpleNamespace(total_qubits=8))

    kwargs = _build_run_hardcoded_adapt_vqe_kwargs(
        args,
        h_poly=None,
        resolved_problem_context=context,
        cli_adapt_continuation_mode="phase3_v1",
        adapt_ref_base_depth=0,
        psi_ref_override=None,
        psi_ref_source=None,
        psi_ref_handoff_state_kind=None,
        exact_gs_override=0.0,
        phase3_oracle_gradient_config=None,
        final_noise_audit_config=None,
    )

    padding = kwargs["route_a_child_padding_config"]
    assert isinstance(padding, RouteAChildPaddingConfig)
    assert padding.as_dict() == {
        "policy": ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
        "problem_key": "hh",
        "num_sites": 2,
        "n_ph_max": 2,
        "boson_encoding": "binary",
        "total_register_width": 8,
    }
    assert kwargs["adapt_estimator_call_ledger_enabled"] is True


@pytest.mark.parametrize(
    ("coordinate_policy", "trust_policy", "whitening", "adaptive"),
    [
        ("archival_reduced_scalar_v1", "fixed", False, False),
        (
            "supported_metric_whitened_eigh_v1",
            "fixed",
            True,
            False,
        ),
        (
            "archival_reduced_scalar_v1",
            "displacement_calibrated_unbounded_v2",
            False,
            True,
        ),
        (
            "supported_metric_whitened_eigh_v1",
            "displacement_calibrated_unbounded_v2",
            True,
            True,
        ),
    ],
)
def test_cli_exposes_historical_singleton_controls_as_orthogonal_fields(
    coordinate_policy: str,
    trust_policy: str,
    whitening: bool,
    adaptive: bool,
) -> None:
    args = adapt_pipeline.parse_args(
        [
            "--historical-singleton-coordinate-solve-policy",
            coordinate_policy,
            "--historical-singleton-trust-region-update-policy",
            trust_policy,
        ]
    )
    kwargs = _build_run_hardcoded_adapt_vqe_kwargs(
        args,
        h_poly=None,
        resolved_problem_context=None,
        cli_adapt_continuation_mode="phase3_v1",
        adapt_ref_base_depth=0,
        psi_ref_override=None,
        psi_ref_source=None,
        psi_ref_handoff_state_kind=None,
        exact_gs_override=0.0,
        phase3_oracle_gradient_config=None,
        final_noise_audit_config=None,
    )

    assert kwargs["historical_singleton_coordinate_solve_policy"] == (
        coordinate_policy
    )
    assert kwargs["historical_singleton_trust_region_update_policy"] == (
        trust_policy
    )
    assert (
        kwargs["historical_singleton_coordinate_solve_policy"]
        == "supported_metric_whitened_eigh_v1"
    ) is whitening
    assert (
        kwargs["historical_singleton_trust_region_update_policy"]
        == "displacement_calibrated_unbounded_v2"
    ) is adaptive
    assert kwargs["historical_singleton_coordinate_solve_scope"] == (
        "phase3_only_v1"
    )


def test_cli_exposes_opt_in_phase2_and_phase3_whitening_scope() -> None:
    args = adapt_pipeline.parse_args(
        [
            "--historical-singleton-coordinate-solve-policy",
            "supported_metric_whitened_eigh_v1",
            "--historical-singleton-coordinate-solve-scope",
            "phase2_and_phase3_v1",
            "--historical-singleton-trust-region-update-policy",
            "displacement_calibrated_unbounded_v2",
        ]
    )
    kwargs = _build_run_hardcoded_adapt_vqe_kwargs(
        args,
        h_poly=None,
        resolved_problem_context=None,
        cli_adapt_continuation_mode="phase3_v1",
        adapt_ref_base_depth=0,
        psi_ref_override=None,
        psi_ref_source=None,
        psi_ref_handoff_state_kind=None,
        exact_gs_override=0.0,
        phase3_oracle_gradient_config=None,
        final_noise_audit_config=None,
    )

    assert kwargs["historical_singleton_coordinate_solve_policy"] == (
        "supported_metric_whitened_eigh_v1"
    )
    assert kwargs["historical_singleton_coordinate_solve_scope"] == (
        "phase2_and_phase3_v1"
    )
    assert kwargs["historical_singleton_trust_region_update_policy"] == (
        "displacement_calibrated_unbounded_v2"
    )


def test_phase2_whitening_manifest_delta_is_only_coordinate_scope() -> None:
    common_argv = [
        "--historical-singleton-coordinate-solve-policy",
        "supported_metric_whitened_eigh_v1",
        "--historical-singleton-trust-region-update-policy",
        "displacement_calibrated_unbounded_v2",
    ]

    def _kwargs(argv: list[str]) -> dict[str, Any]:
        return _build_run_hardcoded_adapt_vqe_kwargs(
            adapt_pipeline.parse_args(argv),
            h_poly=None,
            resolved_problem_context=None,
            cli_adapt_continuation_mode="phase3_v1",
            adapt_ref_base_depth=0,
            psi_ref_override=None,
            psi_ref_source=None,
            psi_ref_handoff_state_kind=None,
            exact_gs_override=0.0,
            phase3_oracle_gradient_config=None,
            final_noise_audit_config=None,
        )

    current = _kwargs(common_argv)
    phase2_whitened = _kwargs(
        [
            *common_argv,
            "--historical-singleton-coordinate-solve-scope",
            "phase2_and_phase3_v1",
        ]
    )
    differing_keys = {
        key
        for key in set(current) | set(phase2_whitened)
        if current.get(key) != phase2_whitened.get(key)
    }
    assert differing_keys == {
        "historical_singleton_coordinate_solve_scope"
    }


def test_cli_active_child_padding_fails_without_resolved_layout() -> None:
    args = adapt_pipeline.parse_args(
        [
            "--problem",
            "hh",
            "--L",
            "2",
            "--n-ph-max",
            "2",
            "--allow-archival-phase3-runtime-split",
            "--phase3-runtime-split-mode",
            "shortlist_pauli_children_v1",
            "--phase3-runtime-split-selection-mode",
            "archival_child_set_forward_v1",
            "--phase3-runtime-split-subset-sizes",
            "1",
            "--phase3-runtime-split-child-set-symmetry-policy",
            "hard_guard",
            "--phase3-runtime-split-child-padding-policy",
            "exact_projected_grouped_v1",
        ]
    )

    with pytest.raises(ValueError, match="requires a resolved problem context"):
        _build_run_hardcoded_adapt_vqe_kwargs(
            args,
            h_poly=None,
            resolved_problem_context=None,
            cli_adapt_continuation_mode="phase3_v1",
            adapt_ref_base_depth=0,
            psi_ref_override=None,
            psi_ref_source=None,
            psi_ref_handoff_state_kind=None,
            exact_gs_override=0.0,
            phase3_oracle_gradient_config=None,
            final_noise_audit_config=None,
        )




def test_sr_no_singleton_log_fields_are_compact_and_attributed() -> None:
    payload = adapt_pipeline._sr_escape_no_singleton_log_fields(
        {
            "record_id": "decision-record",
            "certificate_record_id": "certificate-record",
            "actionable": False,
            "consumes_singleton": False,
            "reachable_population_complete": False,
            "reachable_record_ids": ["a", "b", "c"],
            "certificate_kind_counts": {
                "RedundantCertificate": 2,
                "UnresolvedCertificate": 1,
            },
            "unresolved_certificate_reason_counts": {
                "physical_active_image_subspace_rotation_unresolved": 1,
            },
            "ordinary_model_live_record_ids": ["a"],
            "contradicted_ordinary_record_ids": ["b", "c"],
        }
    )

    assert payload == {
        "record_id": "decision-record",
        "certificate_record_id": "certificate-record",
        "actionable": False,
        "consumes_singleton": False,
        "reachable_population_complete": False,
        "reachable_population_digest": None,
        "state_stationarity_certified": False,
        "state_stationarity_blocker": None,
        "reachable_record_count": 3,
        "certificate_kind_counts": {
            "RedundantCertificate": 2,
            "UnresolvedCertificate": 1,
        },
        "unresolved_certificate_reason_counts": {
            "physical_active_image_subspace_rotation_unresolved": 1,
        },
        "ordinary_model_live_record_count": 1,
        "contradicted_ordinary_record_count": 2,
    }
    assert "reachable_record_ids" not in payload





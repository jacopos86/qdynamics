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
    beam_live_branches: int,
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
        "phase1_prune_enabled": False,
        "phase2_shortlist_fraction": 1.0,
        "phase2_shortlist_size": 256,
        "phase3_shortlist_size": 256,
        "phase2_enable_batching": False,
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
        "adapt_beam_live_branches": int(beam_live_branches),
        "adapt_beam_children_per_parent": 1,
        "adapt_beam_terminated_keep": 2,
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


@pytest.mark.parametrize("beam_live_branches", [1, 2])
def test_depth_one_hh_archival_repair_enforces_guard_and_projection(
    beam_live_branches: int,
    tmp_path: Path,
    _hh_nph2_hamiltonian: Any,
) -> None:
    payload, _psi = adapt_pipeline._run_hardcoded_adapt_vqe(
        **_depth_one_hh_kwargs(
            h_poly=_hh_nph2_hamiltonian,
            pool_filter_json=_write_two_singlet_pool_filter(tmp_path),
            beam_live_branches=beam_live_branches,
            padding_policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
        )
    )

    assert payload["success"] is True
    assert "route_family" not in payload["static_route_identity"]
    assert payload["ansatz_depth"] == 1
    assert payload["stop_reason"] == "max_depth"
    assert payload["operators"][0].endswith("::legal_projected")
    split = payload["continuation"]["runtime_split_summary"]
    assert split["selection_mode"] == "archival_child_set_forward_v1"
    assert split["child_set_symmetry_policy"] == "hard_guard"
    assert split["child_padding_projection_active"] is True
    assert split["requested_subset_sizes"] == [1]
    assert split["probed_parent_count"] == 2
    assert split["admissible_child_set_count"] > 0
    assert split["projected_child_count_padding"] > 0
    assert split["selected_child_set_count"] == 1

    selected = payload["history"][0]["selected_feature_rows"][0]
    assert selected["runtime_split_chosen_representation"] == "child_set"
    metadata = selected["generator_metadata"]
    gate = metadata["compile_metadata"]["runtime_split"]["symmetry_gate"]
    assert gate["checked"] is True
    assert gate["passed"] is True
    assert gate.get("skipped_reason") is None
    assert gate["gate_scope"] == "fixed_count_sector_invariance_v1"
    assert gate["fixed_count_sector"]["fixed_num_particles"] == {
        "n_up": 1,
        "n_dn": 1,
    }
    projection = metadata["compile_metadata"][
        "route_a_child_padding_projection"
    ]
    assert projection["active"] is True
    assert projection["policy"] == ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1
    checkpoint = payload["active_prefix_checkpoints"][0]
    assert checkpoint["outer_iteration"] == 1
    assert checkpoint["active_ansatz_depth"] == 1
    assert checkpoint["ordered_active_operator_labels"] == payload["operators"]
    assert checkpoint["fixed_spin_sector_illegal_probability"] == pytest.approx(
        0.0, abs=1e-14
    )
    assert checkpoint["boson_illegal_codeword_probability"] == pytest.approx(
        0.0, abs=1e-14
    )
    assert checkpoint["ordered_active_operators"][0][
        "serialized_terms_exyz_in_execution_order"
    ]
    checkpoint_ledger = checkpoint["estimator_ledger_receipt"]
    assert checkpoint_ledger["enabled"] is True
    assert checkpoint_ledger["status"] == "complete"
    assert checkpoint_ledger["raw_occurrences_preserved"] is True
    assert (
        checkpoint_ledger[
            "physical_identity_collapse_is_diagnostic_only"
        ]
        is True
    )
    assert checkpoint_ledger["unique_primitive_delta"]["S_unique"] == sum(
        checkpoint_ledger["unique_primitive_delta"]["components"].values()
    )
    accounting = payload["estimator_call_accounting"]
    prefix_closure = payload["continuation"][
        "active_prefix_estimator_ledger_closure"
    ]
    assert prefix_closure["passed"] is True
    assert prefix_closure["summed_unique_primitives"] == (
        prefix_closure["terminal_unique_primitives"]
    )
    assert prefix_closure["summed_raw_occurrences"] == (
        prefix_closure["terminal_raw_occurrences"]
    )
    assert accounting["complete"] is True
    assert accounting["status"] == (
        "resolved_from_live_state_keyed_instrumentation"
    )
    winning = accounting["winning_lineage"]
    assert winning["S_alg"] == sum(
        winning[key]
        for key in ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
    )
    assert winning["N_H_outer"] >= 1
    assert winning["N_grad"] >= 1
    assert winning["N_metric"] >= 1
    executed = accounting["executed_occurrence_accounting"]["all_execution"]
    scope_counts = executed["occurrence_count_by_consumer_scope"]
    if beam_live_branches == 1:
        assert int(scope_counts.get("full_candidate_gradient_recompute", 0)) == 0
        assert int(
            scope_counts.get("full_candidate_self_metric_recompute", 0)
        ) == 0
        assert int(scope_counts.get("phase2_scaffold_geometry", 0)) == 0
        assert int(scope_counts.get("phase2_candidate_self_hessian", 0)) == int(
            split["probed_parent_count"]
        )
    child_gradient_occurrences = int(
        scope_counts.get(
            "runtime_split_child_gradient", 0
        )
    )
    assert child_gradient_occurrences == int(
        split["admissible_child_set_count"]
    )
    child_gradient_unique = int(
        accounting["all_branch_unique_primitive_diagnostic"][
            "unique_primitive_count_by_consumer_scope"
        ].get("runtime_split_child_gradient", 0)
    )
    assert 0 < child_gradient_unique <= child_gradient_occurrences
    emitted_occurrences = [
        row
        for row in accounting["full_ledger"]["occurrences"]
        if row["consumer_scope"] == "runtime_split_child_gradient"
    ]
    assert len(emitted_occurrences) == child_gradient_occurrences


def test_nonbeam_projected_route_shortlists_parents_before_child_measurement(
    tmp_path: Path,
    _hh_nph2_hamiltonian: Any,
) -> None:
    kwargs = _depth_one_hh_kwargs(
        h_poly=_hh_nph2_hamiltonian,
        pool_filter_json=_write_two_singlet_pool_filter(tmp_path),
        beam_live_branches=1,
        padding_policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    )
    kwargs["phase2_shortlist_size"] = 1
    kwargs["phase2_shortlist_fraction"] = 1.0

    payload, _psi = adapt_pipeline._run_hardcoded_adapt_vqe(**kwargs)

    row = payload["history"][0]
    receipt = row["projected_phase3_population_receipt"]
    assert receipt["phase2_input_parent_count"] == 2
    assert receipt["phase2_retained_parent_count"] == 1
    assert receipt["split_parent_count"] == 1
    assert receipt["unsplit_singleton_count"] == 0
    assert len(receipt["split_children_by_parent"]) == 1
    assert len(receipt["parent_expansions"]) == 1
    assert receipt["split_child_count"] == (
        receipt["phase3_evaluated_candidate_count"]
    )
    assert receipt["split_child_count"] > 1
    assert receipt["child_primitive_reuse_count"] == 0
    assert receipt["cross_outer_iteration_reuse_count"] == 0

    parent_surface = row["scored_surface_records"]
    assert len(parent_surface) == 2
    assert all(
        record["runtime_split_mode"] == "off"
        for record in parent_surface
    )
    expanded_parent_labels = {
        str(expansion["parent_label"])
        for expansion in receipt["parent_expansions"]
    }
    surface_parent_labels = {
        str(record["candidate_label"])
        for record in parent_surface
    }
    assert expanded_parent_labels < surface_parent_labels

    controller = row["controller_measurement_work_proxy"]
    phase2 = controller["by_phase"]["phase2"]
    phase3 = controller["by_phase"]["phase3"]
    assert phase2["method_input_candidate_count_total"] == 2
    assert phase2["method_retained_candidate_count_total"] == 1
    assert phase3["method_input_candidate_count_total"] == (
        receipt["phase3_evaluated_candidate_count"]
    )
    assert phase3["actual_evaluated_candidate_count_total"] == (
        receipt["phase3_evaluated_candidate_count"]
    )

    scopes = payload["estimator_call_accounting"][
        "executed_occurrence_accounting"
    ]["all_execution"]["occurrence_count_by_consumer_scope"]
    assert scopes["phase2_candidate_self_hessian"] == 2
    assert scopes["runtime_split_child_gradient"] == (
        receipt["phase3_evaluated_candidate_count"]
    )
    assert scopes["runtime_split_child_self_metric"] == (
        receipt["phase3_evaluated_candidate_count"]
    )
    assert scopes["phase2_phase3_candidate_geometry"] == (
        receipt["phase3_evaluated_candidate_count"]
    )
    assert int(scopes.get("phase2_scaffold_geometry", 0)) == 0


def test_nonbeam_projected_phase3_acquires_fresh_geometry_once_at_n_positive(
    tmp_path: Path,
    _hh_nph2_hamiltonian: Any,
) -> None:
    kwargs = _depth_one_hh_kwargs(
        h_poly=_hh_nph2_hamiltonian,
        pool_filter_json=_write_two_singlet_pool_filter(tmp_path),
        beam_live_branches=1,
        padding_policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    )
    kwargs.update(
        {
            "max_depth": 2,
            "phase2_shortlist_size": 1,
            "phase2_shortlist_fraction": 1.0,
        }
    )

    payload, _psi = adapt_pipeline._run_hardcoded_adapt_vqe(**kwargs)

    assert payload["ansatz_depth"] == 2
    receipts = [
        row["projected_phase3_population_receipt"]
        for row in payload["history"]
    ]
    assert [receipt["phase2_input_parent_count"] for receipt in receipts] == [
        2,
        2,
    ]
    assert [
        receipt["phase2_retained_parent_count"] for receipt in receipts
    ] == [1, 1]
    r3_counts = [
        int(receipt["phase3_evaluated_candidate_count"])
        for receipt in receipts
    ]
    assert r3_counts == [2, 2]

    scopes = payload["estimator_call_accounting"][
        "executed_occurrence_accounting"
    ]["all_execution"]["occurrence_count_by_consumer_scope"]
    assert scopes["phase2_candidate_self_hessian"] == 4
    assert scopes["runtime_split_child_gradient"] == sum(r3_counts)
    assert scopes["runtime_split_child_self_metric"] == sum(r3_counts)
    assert int(scopes.get("phase2_candidate_geometry", 0)) == 0
    assert int(scopes.get("phase2_scaffold_geometry", 0)) == 0
    assert scopes["phase3_scaffold_geometry"] == sum(
        n_active * (n_active + 1) for n_active in range(2)
    )
    assert scopes["phase2_phase3_candidate_geometry"] == sum(
        r3 * (2 * n_active + 1)
        for n_active, r3 in enumerate(r3_counts)
    )


def test_depth_one_hh_sr_saddle_profile_smoke(
    tmp_path: Path,
    _hh_nph2_hamiltonian: Any,
) -> None:
    kwargs = _depth_one_hh_kwargs(
        h_poly=_hh_nph2_hamiltonian,
        pool_filter_json=_write_two_singlet_pool_filter(tmp_path),
        beam_live_branches=2,
        padding_policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    )
    kwargs.update(
        {
            "static_route_id": "route_a",
            "phase0_pilot_enabled": False,
            "phase0_pilot_max_operators": 0,
            "phase1_shortlist_size": 24,
            "phase2_shortlist_size": 12,
            "phase2_shortlist_fraction": 0.25,
            "phase3_shortlist_size": 12,
            "phase3_runtime_split_max_subset_size": 1,
            "phase3_selector_policy": "algebraic_nested_v1",
            "phase1_prune_enabled": True,
            "phase1_prune_policy": "recoverability_ladder_v1",
            "phase1_prune_mode": "both",
            "historical_singleton_coordinate_solve_policy": (
                "supported_metric_global_trust_eigh_v2"
            ),
            "historical_singleton_trust_region_update_policy": (
                "displacement_calibrated_unbounded_v2"
            ),
            "sr_escape_mode": "saddle_only",
        }
    )

    payload, _psi = adapt_pipeline._run_hardcoded_adapt_vqe(**kwargs)

    route_identity = payload["static_route_identity"]
    assert route_identity["route_family"] == "singleton_response_snake"
    assert route_identity["route_profile"] == (
        "supported_whitened_adaptive_trust_saddle_escape_v2"
    )
    assert route_identity["sr_route_family"] == "singleton_response_snake"
    assert route_identity["sr_escape_mode"] == "saddle_only"
    assert route_identity["sr_route_profile"] == (
        "supported_whitened_adaptive_trust_saddle_escape_v2"
    )
    overlay = payload["continuation"][
        "historical_singleton_coordinate_trust_overlay"
    ]
    assert overlay["sr_escape_active"] is True
    assert overlay["coordinate_solve_policy"] == (
        "supported_metric_global_trust_eigh_v2"
    )
    round_overlay = payload["continuation"]["runtime_split_summary"][
        "historical_singleton_coordinate_overlay_last_round"
    ]
    assert round_overlay["active"] is True
    assert round_overlay["rescore"]["sr_escape_controller"]["mode"] == (
        "saddle_only"
    )
    phase1_population = round_overlay["sr_escape_phase1_population"]
    assert phase1_population["active"] is True
    assert phase1_population["service_policy"] == (
        "complete_finite_eager_no_gradient_gate_v1"
    )
    assert phase1_population["zero_gradient_gate_applied"] is False
    assert phase1_population["escape_evaluation_count"] == (
        phase1_population["raw_record_count"]
    )
    assert round_overlay["sr_escape_reachable_record_ids"]
    assert set(round_overlay["sr_escape_ordinary_record_ids"]).issubset(
        set(round_overlay["sr_escape_reachable_record_ids"])
    )
    assert all(
        record["joint_linear_solve_policy_effective"]
        == "supported_metric_global_trust_eigh_v2"
        for record in round_overlay["records"]
    )


def test_depth_one_hh_sr_phase2_and_phase3_whitening_profile_smoke(
    tmp_path: Path,
    _hh_nph2_hamiltonian: Any,
) -> None:
    kwargs = _depth_one_hh_kwargs(
        h_poly=_hh_nph2_hamiltonian,
        pool_filter_json=_write_two_singlet_pool_filter(tmp_path),
        beam_live_branches=2,
        padding_policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    )
    kwargs.update(
        {
            "static_route_id": "route_a",
            "phase0_pilot_enabled": False,
            "phase0_pilot_max_operators": 0,
            "phase1_shortlist_size": 24,
            "phase2_shortlist_size": 12,
            "phase2_shortlist_fraction": 0.25,
            "phase3_shortlist_size": 12,
            "phase3_runtime_split_max_subset_size": 1,
            "phase3_selector_policy": "algebraic_nested_v1",
            "phase1_prune_enabled": True,
            "phase1_prune_policy": "recoverability_ladder_v1",
            "phase1_prune_mode": "both",
            "historical_singleton_coordinate_solve_policy": (
                "supported_metric_whitened_eigh_v1"
            ),
            "historical_singleton_coordinate_solve_scope": (
                "phase2_and_phase3_v1"
            ),
            "historical_singleton_trust_region_update_policy": (
                "displacement_calibrated_unbounded_v2"
            ),
            "sr_escape_mode": "disabled",
        }
    )

    payload, _psi = adapt_pipeline._run_hardcoded_adapt_vqe(**kwargs)

    route_identity = payload["static_route_identity"]
    assert route_identity["route_family"] == "singleton_response_snake"
    assert route_identity["route_profile"] == (
        "supported_phase2_phase3_whitened_adaptive_trust_v2"
    )
    assert route_identity["coordinate_solve_scope"] == (
        "phase2_and_phase3_v1"
    )
    assert route_identity["phase2_whitening_active"] is True
    assert route_identity["phase3_whitening_active"] is True
    overlay = payload["continuation"][
        "historical_singleton_coordinate_trust_overlay"
    ]
    assert overlay["phase2_coordinate_solve_policy"] == (
        "supported_metric_whitened_eigh_v1"
    )
    assert overlay["phase3_coordinate_solve_policy"] == (
        "supported_metric_whitened_eigh_v1"
    )
    assert overlay["phase2_whitening_active"] is True
    assert overlay["phase2_batching_enabled"] is False
    runtime_summary = payload["continuation"]["runtime_split_summary"]
    phase2_overlay = runtime_summary[
        "historical_singleton_phase2_coordinate_overlay_last_round"
    ]
    assert phase2_overlay["active"] is True
    assert phase2_overlay["membership_preserved"] is True
    assert phase2_overlay["order_preserved"] is True
    assert phase2_overlay["novelty_preserved"] is True
    assert phase2_overlay["cost_denominator_preserved"] is True
    assert phase2_overlay["batching_applied"] is False
    assert phase2_overlay["candidate_pair_measurement_count"] == 0
    assert all(
        record["trust_radius_sq"]
        == pytest.approx(float(phase2_overlay["live_radius"]) ** 2)
        for record in phase2_overlay["records"]
    )
    assert payload["route_a_trust_region_state"]["update_count"] == 1
    checkpoint = payload["active_prefix_checkpoints"][0]
    assert checkpoint["fixed_spin_sector_illegal_probability"] == pytest.approx(
        0.0, abs=1e-14
    )
    assert checkpoint["boson_illegal_codeword_probability"] == pytest.approx(
        0.0, abs=1e-14
    )
    checkpoint_ledger = checkpoint["estimator_ledger_receipt"]
    assert checkpoint_ledger["enabled"] is True
    assert checkpoint_ledger["status"] == "complete"
    assert checkpoint_ledger["unique_primitive_delta"]["S_unique"] == sum(
        checkpoint_ledger["unique_primitive_delta"]["components"].values()
    )
    accounting = payload["estimator_call_accounting"]
    prefix_closure = payload["continuation"][
        "active_prefix_estimator_ledger_closure"
    ]
    assert prefix_closure["passed"] is True
    assert prefix_closure["summed_unique_primitives"] == (
        prefix_closure["terminal_unique_primitives"]
    )
    assert accounting["complete"] is True
    assert accounting["status"] == (
        "resolved_from_live_state_keyed_instrumentation"
    )
    winning = accounting["winning_lineage"]
    assert winning["S_alg"] == sum(
        winning[key]
        for key in ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
    )


@pytest.mark.parametrize(
    ("coordinate_policy", "trust_policy", "blocker"),
    [
        (
            "archival_reduced_scalar_v1",
            "displacement_calibrated_unbounded_v2",
            "supported_metric_whitened_eigh_v1_inactive",
        ),
        (
            "supported_metric_whitened_eigh_v1",
            "fixed",
            "adaptive_trust_inactive",
        ),
    ],
)
def test_phase2_whitening_scope_fails_closed_outside_locked_base_profile(
    tmp_path: Path,
    _hh_nph2_hamiltonian: Any,
    coordinate_policy: str,
    trust_policy: str,
    blocker: str,
) -> None:
    kwargs = _depth_one_hh_kwargs(
        h_poly=_hh_nph2_hamiltonian,
        pool_filter_json=_write_two_singlet_pool_filter(tmp_path),
        beam_live_branches=2,
        padding_policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    )
    kwargs.update(
        {
            "historical_singleton_coordinate_solve_policy": coordinate_policy,
            "historical_singleton_coordinate_solve_scope": (
                "phase2_and_phase3_v1"
            ),
            "historical_singleton_trust_region_update_policy": trust_policy,
        }
    )

    with pytest.raises(ValueError, match=blocker):
        adapt_pipeline._run_hardcoded_adapt_vqe(**kwargs)


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


@pytest.mark.parametrize(
    (
        "decision_kind",
        "receipt_schema",
        "selection_mode",
        "force_no_progress",
        "force_backtracking_refinement",
    ),
    [
        (
            "active_only_correction",
            "sr_active_only_correction_transaction_v1",
            "sr_snake_active_only_correction_v1",
            False,
            False,
        ),
        (
            "active_stationarity_correction",
            "sr_active_stationarity_correction_transaction_v1",
            "sr_snake_active_stationarity_correction_v1",
            True,
            False,
        ),
        (
            "active_stationarity_correction",
            "sr_active_stationarity_correction_transaction_v1",
            "sr_snake_active_stationarity_correction_v1",
            False,
            True,
        ),
    ],
)
def test_sr_active_only_correction_refits_without_singleton_or_depth_consumption(
    decision_kind: str,
    receipt_schema: str,
    selection_mode: str,
    force_no_progress: bool,
    force_backtracking_refinement: bool,
    tmp_path: Path,
    _hh_nph2_hamiltonian: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_rescore = (
        adapt_pipeline.rescore_historical_phase3_records_with_coordinate_models
    )
    call_count = 0

    def _force_second_round_active_only(*args: Any, **kwargs: Any):
        nonlocal call_count
        call_count += 1
        records, telemetry = original_rescore(*args, **kwargs)
        if call_count < 2 or (
            force_backtracking_refinement and call_count > 2
        ):
            return records, telemetry
        assert records
        target = dict(records[0])
        feature = target["feature"]
        coordinate_summary = dict(feature.phase2_joint_geometry_reuse)
        active_count = int(
            coordinate_summary.get("active_coordinate_count", 1)
        )
        assert active_count == 1
        active_restriction = {
            "schema": "sr_v2_shared_support_active_restriction_v1",
            "valid": True,
            "reason": "synthetic_active_only_integration_certificate",
            "trust_global_optimality_certified": True,
            "active_coordinate_count": 1,
            "batch_coordinate_count": 1,
            "joint_step": [0.05, 0.0],
            "batch_coordinate_step": [0.0],
            "predicted_reduction": 0.01,
            "joint_fubini_study_displacement_sq": 0.0025,
            "active_restriction_batch_zero_tolerance": 1e-12,
            "active_restriction_source": (
                "full_v2_supported_metric_whitened_coordinate_restriction_v1"
            ),
            "active_restriction_provenance_id": "synthetic-active-proof",
            "hard_case_sign_candidates_joint": [],
            "hard_case_sign_candidate_predicted_reductions": [],
            "hard_case_sign_candidate_point_estimate_roles": [],
            "restricted_coordinate_trust_solve": {
                "trust_regularization_applied": False,
                "trust_clipped": False,
                "trust_radius_binding": False,
            },
        }
        live_radius = float(kwargs["cfg"].rho)
        coordinate_summary["trust_radius_sq"] = live_radius**2
        coordinate_summary["trust_radius_binding_tolerance_sq"] = 1e-12
        coordinate_summary["active_restriction_solve"] = active_restriction
        target["feature"] = replace(
            feature,
            phase2_joint_geometry_reuse=coordinate_summary,
            selector_score=float("-inf"),
            phase3_reduced_trust_gain=0.0,
        )
        certificate_id = adapt_pipeline._sr_escape_record_id_for_mapping(target)
        updated_records = []
        for index, record in enumerate(records):
            updated = dict(record)
            updated["sr_escape_admission_eligible"] = False
            updated["sr_escape_decision_kind"] = str(decision_kind)
            updated["selector_score"] = float("-inf")
            if index == 0:
                updated.update(target)
            updated_records.append(updated)
        updated_telemetry = dict(telemetry)
        controller = dict(updated_telemetry["sr_escape_controller"])
        controller.update(
            {
                "decision_kind": str(decision_kind),
                "reason": "synthetic_active_only_integration_certificate",
                "record_id": None,
                "certificate_record_id": certificate_id,
                "consumes_singleton": False,
                "actionable": True,
                "stage_b_eligible": False,
                "admission_eligible_record_ids": [],
            }
        )
        updated_telemetry["sr_escape_controller"] = controller
        return updated_records, updated_telemetry

    monkeypatch.setattr(
        adapt_pipeline,
        "rescore_historical_phase3_records_with_coordinate_models",
        _force_second_round_active_only,
    )
    if force_no_progress or force_backtracking_refinement:

        def _reject_active_seed(*args: Any, **kwargs: Any):
            chart = kwargs["chart"]
            assert kwargs["retained_joint_step_candidates"] == [[0.05, 0.0]]
            if force_backtracking_refinement:
                return (
                    np.asarray(chart.x0, dtype=float).copy(),
                    {
                        "status": "rejected",
                        "reason": (
                            "active_only_nonlinear_backtracking_exhausted"
                        ),
                        "transaction_failure_kind": (
                            "finite_nonlinear_model_disagreement"
                        ),
                        "nonlinear_backtracking_exhausted": True,
                        "all_backtracking_candidates_finite": True,
                        "trust_action": "contract_branch_radius",
                        "no_state_transition": True,
                    },
                    0,
                )
            return (
                np.asarray(chart.x0, dtype=float).copy(),
                {
                    "status": "rejected",
                    "reason": "synthetic_stationarity_no_progress",
                    "transaction_failure_kind": "certificate_non_downhill",
                },
                0,
            )

        monkeypatch.setattr(
            adapt_pipeline,
            "_guard_sr_active_only_step",
            _reject_active_seed,
        )
    ai_events: list[dict[str, Any]] = []

    def _capture_ai_log(event: str, **payload: Any) -> None:
        ai_events.append({"event": str(event), **dict(payload)})

    monkeypatch.setattr(adapt_pipeline, "_ai_log", _capture_ai_log)
    kwargs = _depth_one_hh_kwargs(
        h_poly=_hh_nph2_hamiltonian,
        pool_filter_json=_write_two_singlet_pool_filter(tmp_path),
        beam_live_branches=2,
        padding_policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    )
    kwargs.update(
        {
            "max_depth": 3 if force_backtracking_refinement else 2,
            "static_route_id": "route_a",
            "phase0_pilot_enabled": False,
            "phase0_pilot_max_operators": 0,
            "phase1_shortlist_size": 24,
            "phase2_shortlist_size": 12,
            "phase2_shortlist_fraction": 0.25,
            "phase3_shortlist_size": 12,
            "phase3_runtime_split_max_subset_size": 1,
            "phase3_selector_policy": "algebraic_nested_v1",
            "phase1_prune_enabled": True,
            "phase1_prune_policy": "recoverability_ladder_v1",
            "phase1_prune_mode": "both",
            "historical_singleton_coordinate_solve_policy": (
                "supported_metric_global_trust_eigh_v2"
            ),
            "historical_singleton_trust_region_update_policy": (
                "displacement_calibrated_unbounded_v2"
            ),
            "sr_escape_mode": "saddle_only",
            "phase3_selector_debug_topk": 1,
            "phase3_selector_debug_max_depth": (
                3 if force_backtracking_refinement else 2
            ),
        }
    )

    payload, _psi = adapt_pipeline._run_hardcoded_adapt_vqe(**kwargs)

    assert call_count >= 2
    if force_backtracking_refinement:
        assert call_count >= 3
        assert len(payload["operators"]) >= 1
        assert len(payload["history"]) >= 1
    else:
        assert len(payload["operators"]) == 1
        assert len(payload["history"]) == 1
    receipts = [
        row
        for row in payload["continuation"]["rescue_history"]
        if row.get("schema") == receipt_schema
    ]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["decision_kind"] == decision_kind
    assert receipt["singleton_consumed"] is False
    assert receipt["selected_labels_admitted"] == []
    assert receipt["ansatz_depth_before"] == receipt["ansatz_depth_after"] == 1
    assert receipt["controller_depth_before"] == receipt["controller_depth_after"]
    assert receipt["selection_counts_unchanged"] is True
    assert receipt["available_indices_unchanged"] is True
    assert receipt["admission_history_unchanged"] is True
    assert receipt["energy_nonworsening"] is True
    assert receipt["safe_refit_outcome"]["nonworsening_certified"] is True
    assert receipt["trust_radius_update"]["sr_active_only_correction"] is True
    assert receipt["progress_certified"] is False
    assert receipt["terminal_no_progress_guard"] is force_no_progress
    if force_no_progress:
        assert payload["stop_reason"] == (
            "sr_active_stationarity_correction_no_progress"
        )
    if force_backtracking_refinement:
        assert receipt[
            "nonlinear_backtracking_refinement_scheduled"
        ] is True
        assert receipt["transaction_outcome"] == (
            "branch_trust_radius_contracted_no_state_transition"
        )
        trust_update = receipt["trust_radius_update"]
        assert trust_update["radius_after"] == pytest.approx(
            0.5 * trust_update["radius_before"]
        )
        assert payload["stop_reason"] != (
            "sr_active_stationarity_correction_no_progress"
        )
    selector_debug = [
        event
        for event in ai_events
        if event.get("event") == "hardcoded_adapt_phase3_selector_debug"
        and int(event.get("depth", -1)) == 2
    ]
    assert selector_debug
    assert all(
        event["selection_mode"] == selection_mode
        for event in selector_debug
    )


@pytest.mark.parametrize("beam_live_branches", [1, 2])
def test_required_hard_guard_rejects_missing_admission_gate(
    beam_live_branches: int,
    tmp_path: Path,
    _hh_nph2_hamiltonian: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runtime_split_module.build_runtime_split_child_sets

    def _strip_executed_gate(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        stripped: list[dict[str, Any]] = []
        for source in original(*args, **kwargs):
            row = dict(source)
            metadata = dict(row["candidate_generator_metadata"])
            compile_metadata = dict(metadata.get("compile_metadata", {}))
            runtime_split = dict(compile_metadata.get("runtime_split", {}))
            runtime_split.pop("symmetry_gate", None)
            compile_metadata.pop("symmetry_gate", None)
            compile_metadata["runtime_split"] = runtime_split
            metadata["compile_metadata"] = compile_metadata
            row["candidate_generator_metadata"] = metadata
            stripped.append(row)
        return stripped

    monkeypatch.setattr(
        adapt_pipeline,
        "build_runtime_split_child_sets",
        _strip_executed_gate,
    )
    monkeypatch.setattr(
        runtime_split_module,
        "build_runtime_split_child_sets",
        _strip_executed_gate,
    )

    with pytest.raises(
        RuntimeError,
        match="Required runtime-split hard guard has no executed gate payload",
    ):
        adapt_pipeline._run_hardcoded_adapt_vqe(
            **_depth_one_hh_kwargs(
                h_poly=_hh_nph2_hamiltonian,
                pool_filter_json=_write_two_singlet_pool_filter(tmp_path),
                beam_live_branches=beam_live_branches,
                padding_policy=ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1,
            )
        )


@pytest.mark.parametrize("beam_live_branches", [1, 2])
def test_source_lock_cannot_rebuild_without_required_hard_guard(
    beam_live_branches: int,
    tmp_path: Path,
    _hh_nph2_hamiltonian: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runtime_split_module.build_runtime_split_child_sets
    construction_contracts: list[dict[str, Any]] = []

    def _record_contract(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        construction_contracts.append(
            {
                "symmetry_spec": kwargs.get("symmetry_spec"),
                "fixed_num_particles": kwargs.get("fixed_num_particles"),
                "hard_guard_required": kwargs.get("hard_guard_required", False),
            }
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        adapt_pipeline,
        "build_runtime_split_child_sets",
        _record_contract,
    )
    monkeypatch.setattr(
        runtime_split_module,
        "build_runtime_split_child_sets",
        _record_contract,
    )
    payload, _psi = adapt_pipeline._run_hardcoded_adapt_vqe(
        **_depth_one_hh_kwargs(
            h_poly=_hh_nph2_hamiltonian,
            pool_filter_json=_write_two_singlet_pool_filter(tmp_path),
            beam_live_branches=beam_live_branches,
            padding_policy=ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1,
            source_lock_sequence='["not_a_current_guarded_child_set"]',
        )
    )

    assert construction_contracts
    assert all(
        contract["hard_guard_required"] is True
        and contract["symmetry_spec"] is not None
        and tuple(contract["fixed_num_particles"]) == (1, 1)
        for contract in construction_contracts
    )
    split = payload["continuation"]["runtime_split_summary"]
    assert int(split.get("source_lock_forced_child_set_count", 0)) == 0
    source_lock = payload["continuation"]["source_lock_preferred_sequence"]
    assert source_lock["enabled"] is True
    assert all(
        "rebuilt_without_current_child_set_hard_guard"
        not in str(event.get("reason", ""))
        for event in source_lock["events"]
    )

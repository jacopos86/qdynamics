from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import pipelines.static_adapt.ra_adapt.support as support_module
import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.static_adapt.accepted_refit import (
    ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1,
    ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1,
    AcceptedRefitConfig,
    AcceptedRefitFixedChartReceipt,
    build_supported_fs_powell_chart,
)
from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
from pipelines.static_adapt.joint_linear_solve import (
    JointLinearSolveConfig,
    factor_supported_metric,
)
from pipelines.static_adapt.route_a_child_padding import (
    ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    RouteAChildPaddingConfig,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
)
from pipelines.contracts.static_provenance import HH_FULL_META_CLASSIFIER_VERSION
from src.quantum.ansatz_parameterization import project_runtime_theta_block_mean
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    compile_polynomial_action,
    energy_via_one_apply,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm
from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_holstein_hamiltonian,
)


def _normalized_state(nq: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=1 << nq) + 1.0j * rng.normal(size=1 << nq)
    return np.asarray(state / np.linalg.norm(state), dtype=complex)


def _problem():
    terms = [
        AnsatzTerm(
            label="multi",
            polynomial=PauliPolynomial(
                "JW",
                [
                    PauliTerm(2, ps="xx", pc=0.75),
                    PauliTerm(2, ps="zz", pc=-0.4),
                ],
            ),
        ),
        AnsatzTerm(
            label="single",
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(2, ps="ey", pc=0.6)],
            ),
        ),
    ]
    executor = CompiledAnsatzExecutor(
        terms,
        coefficient_tolerance=1.0e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode="logical_shared",
    )
    psi_ref = _normalized_state(2, seed=8143)
    hamiltonian = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="ze", pc=0.7),
            PauliTerm(2, ps="ex", pc=-0.25),
            PauliTerm(2, ps="yy", pc=0.31),
        ],
    )
    h_compiled = compile_polynomial_action(hamiltonian, tol=1.0e-14)
    theta_runtime = np.asarray([0.08, 0.08, 0.11], dtype=float)

    def objective(runtime_theta: np.ndarray) -> float:
        logical = project_runtime_theta_block_mean(
            np.asarray(runtime_theta, dtype=float), executor.layout
        )
        state = executor.prepare_state(logical, psi_ref)
        energy, _ = energy_via_one_apply(state, h_compiled)
        return float(energy)

    return executor, psi_ref, h_compiled, theta_runtime, objective


def _config(base_chart_policy: str) -> AcceptedRefitConfig:
    return AcceptedRefitConfig(
        scope=ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1,
        coordinate_chart=ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1,
        base_chart_policy=base_chart_policy,
        supported_metric=JointLinearSolveConfig(
            rank_relative_tolerance=1.0e-8,
            metric_regularization=1.0e-9,
        ),
    )


def _chart(base_chart_policy: str):
    executor, psi_ref, h_compiled, theta_runtime, objective = _problem()
    chart = build_supported_fs_powell_chart(
        executor=executor,
        layout=executor.layout,
        theta_runtime=theta_runtime,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        runtime_objective=objective,
        config=_config(base_chart_policy),
        manifold_id=f"accepted_refit_test:{base_chart_policy}",
    )
    return chart, executor, psi_ref, theta_runtime, objective


def test_full_accepted_refit_scope_is_independent_of_selector_window() -> None:
    config = _config(SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1)
    selector_indices = [1]

    accepted = config.resolve_logical_indices(
        selector_active_indices=selector_indices,
        logical_parameter_count=4,
    )

    assert accepted == (0, 1, 2, 3)
    assert selector_indices == [1]
    assert config.as_dict()["base_chart_policy"] == (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    )


def test_accepted_refit_refactors_full_gram_through_canonical_support_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    original = support_module.factor_retained_support

    def _recording_factorization(
        gram: np.ndarray,
        **kwargs: object,
    ) -> object:
        calls.append(
            {
                "gram": np.asarray(gram, dtype=float).copy(),
                **kwargs,
            }
        )
        return original(gram, **kwargs)

    monkeypatch.setattr(
        support_module,
        "factor_retained_support",
        _recording_factorization,
    )
    chart, executor, _psi_ref, _theta_runtime, _objective = _chart(
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    )

    assert len(calls) == 1
    assert np.asarray(calls[0]["gram"]).shape == (
        executor.layout.logical_parameter_count,
        executor.layout.logical_parameter_count,
    )
    receipt = chart.base_telemetry["retained_support_receipt"]
    assert receipt["schema"] == "ra_adapt_retained_support_receipt_v1"
    assert receipt["source_provenance_id"].startswith(
        "accepted_refit_full_post_admission_gram:"
    )


def test_accepted_refit_fixed_chart_receipt_binds_scope_maps_and_origin() -> None:
    chart, _executor, _psi_ref, _theta_runtime, _objective = _chart(
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    )
    raw = chart.base_telemetry[
        "accepted_refit_fixed_chart_receipt"
    ]
    receipt = AcceptedRefitFixedChartReceipt(
        scope=raw["scope"],
        coordinate_chart=raw["coordinate_chart"],
        base_chart_policy=raw["base_chart_policy"],
        manifold_id=raw["manifold_id"],
        construction_hashes=raw["construction_hashes"],
        support_factorization_provenance_id=raw[
            "support_factorization_provenance_id"
        ],
        support_receipt_provenance_id=raw[
            "support_receipt_provenance_id"
        ],
        external_gram_receipt_id=raw["external_gram_receipt_id"],
        sha256=raw["sha256"],
    )

    assert receipt.sha256 == chart.base_telemetry[
        "accepted_refit_fixed_chart_sha256"
    ]
    assert receipt.scope == ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1
    assert receipt.coordinate_chart == (
        ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1
    )
    assert receipt.base_chart_policy == (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    )
    tampered_hashes = dict(receipt.construction_hashes)
    tampered_hashes["raw_base_metric_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(receipt, construction_hashes=tampered_hashes)


def test_cli_exposes_primary_and_expanded_whitened_base_charts() -> None:
    parser = _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-8)
    defaults = parser.parse_args([])
    expanded = parser.parse_args(
        [
            "--adapt-accepted-refit-scope",
            "full_ansatz_v1",
            "--adapt-accepted-refit-coordinate-chart",
            "supported_fs_whitened_fixed_v1",
            "--adapt-accepted-refit-base-chart-policy",
            "expanded_runtime_projected_logical_v1",
        ]
    )

    assert defaults.adapt_accepted_refit_base_chart_policy == (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    )
    assert expanded.adapt_accepted_refit_scope == "full_ansatz_v1"
    assert expanded.adapt_accepted_refit_coordinate_chart == (
        "supported_fs_whitened_fixed_v1"
    )
    assert expanded.adapt_accepted_refit_base_chart_policy == (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    )


def test_resume_boundary_refit_uses_accepted_refit_chart_builder() -> None:
    source = inspect.getsource(adapt_pipeline._run_hardcoded_adapt_vqe)
    start = source.index("def _run_resume_boundary_refit_if_needed()")
    stop = source.index("\n        _run_resume_boundary_refit_if_needed()", start)
    boundary = source[start:stop]

    assert "_make_accepted_refit_optimizer_chart(" in boundary
    assert "resume_optimizer_chart = _make_selected_optimizer_chart(" not in boundary
    assert '"accepted_refit_invocation"' in boundary
    assert "SupportedFSPowellChart" in boundary
    assert "resume_optimizer_chart.result_telemetry(" in boundary


@pytest.mark.parametrize(
    "base_chart_policy",
    [
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    ],
)
def test_zero_displacement_preserves_origin_and_final_mapping(
    base_chart_policy: str,
) -> None:
    chart, executor, psi_ref, theta_runtime, objective = _chart(base_chart_policy)

    zero_runtime = chart.lift_to_runtime(np.zeros_like(chart.x0))
    np.testing.assert_allclose(zero_runtime, theta_runtime, atol=2.0e-12, rtol=0.0)
    origin_state = executor.prepare_state(
        project_runtime_theta_block_mean(zero_runtime, executor.layout), psi_ref
    )
    fidelity = abs(np.vdot(origin_state, chart.origin_state)) ** 2
    assert fidelity == pytest.approx(1.0, abs=2.0e-13)

    trial = np.linspace(0.01, 0.02, chart.x0.size, dtype=float)
    trial_runtime = chart.lift_to_runtime(trial)
    trial_energy = chart.objective(trial)
    receipt = chart.result_telemetry(
        optimizer_x=trial,
        final_runtime_theta=trial_runtime,
        final_energy=trial_energy,
    )
    assert trial_energy == pytest.approx(objective(trial_runtime), abs=1.0e-13)
    assert receipt["final_energy"] == pytest.approx(trial_energy, abs=0.0)
    assert receipt["origin_kind"] == "inherited_zero_growth_state_v1"
    json.dumps(receipt, allow_nan=False, sort_keys=True)


def test_logical_chart_is_raw_fs_orthonormal() -> None:
    chart, *_ = _chart(SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1)
    telemetry = chart.base_telemetry

    raw_gradient = np.asarray(
        telemetry["raw_logical_energy_gradient"], dtype=float
    )
    assert raw_gradient.shape == (
        int(telemetry["logical_parameter_count"]),
    )
    assert np.all(np.isfinite(raw_gradient))

    np.testing.assert_allclose(
        telemetry["raw_metric_in_powell_chart"],
        np.eye(int(telemetry["supported_rank"])),
        atol=2.0e-12,
        rtol=0.0,
    )
    assert telemetry["raw_metric_identity_residual"] < 1.0e-10
    assert telemetry["base_coordinate_kind"] == "logical_shared_reduced"
    assert telemetry["classical_factorization_quantum_query_charge"] == 0


def test_expanded_runtime_chart_removes_redundant_block_direction() -> None:
    chart, *_ = _chart(
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    )
    telemetry = chart.base_telemetry

    assert telemetry["base_parameter_count"] == 3
    assert telemetry["supported_rank"] == 2
    assert telemetry["metric_retained_mask"].count(False) == 1
    assert telemetry["base_coordinate_kind"] == (
        "expanded_runtime_projected_logical"
    )
    np.testing.assert_allclose(
        telemetry["raw_metric_in_powell_chart"],
        np.eye(2),
        atol=2.0e-12,
        rtol=0.0,
    )


def test_logical_and_expanded_whitened_charts_span_same_physical_space() -> None:
    logical, *_ = _chart(
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    )
    expanded, *_ = _chart(
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    )
    q_logical, _ = np.linalg.qr(logical.whitened_to_logical_map)
    q_expanded, _ = np.linalg.qr(expanded.whitened_to_logical_map)

    np.testing.assert_allclose(
        q_logical @ q_logical.T,
        q_expanded @ q_expanded.T,
        atol=2.0e-11,
        rtol=0.0,
    )


def test_one_dimensional_whitening_is_invariant_to_parameter_rescaling() -> None:
    raw_metric = np.asarray([[4.0]], dtype=float)
    scale = 7.0
    scaled_metric = np.asarray([[4.0 / scale**2]], dtype=float)
    native = factor_supported_metric(
        raw_metric, rank_relative_tolerance=1.0e-12, metric_regularization=0.0
    )
    scaled = factor_supported_metric(
        scaled_metric, rank_relative_tolerance=1.0e-12, metric_regularization=0.0
    )
    y = np.asarray([0.3], dtype=float)
    physical_native = native.raw_orthonormalizer @ y
    physical_scaled = (scaled.raw_orthonormalizer @ y) / scale

    np.testing.assert_allclose(
        physical_scaled, physical_native, atol=1.0e-14, rtol=0.0
    )


def test_nonuniform_runtime_alias_cannot_define_fixed_logical_origin() -> None:
    executor, psi_ref, h_compiled, _, objective = _problem()
    with pytest.raises(ValueError, match="not a uniform logical alias"):
        build_supported_fs_powell_chart(
            executor=executor,
            layout=executor.layout,
            theta_runtime=np.asarray([0.08, 0.09, 0.11], dtype=float),
            psi_ref=psi_ref,
            h_compiled=h_compiled,
            runtime_objective=objective,
            config=_config(
                SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
            ),
            manifold_id="accepted_refit_nonuniform_alias",
        )


def _write_small_full_meta_filter(tmp_path: Path) -> Path:
    path = tmp_path / "accepted_refit_full_meta_filter.json"
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


@pytest.mark.parametrize(
    "base_chart_policy",
    [
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    ],
)
def test_nonbeam_sr_phase3_empty_shortlist_falls_back_to_full_response_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_chart_policy: str,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_a, **_k: None)
    coordinate_evaluator_calls: list[int] = []
    trust_update_calls: list[int] = []
    coordinate_evaluator = (
        adapt_pipeline.evaluate_historical_singleton_coordinate_models
    )
    trust_updater = adapt_pipeline.update_trust_region_state

    def _coordinate_evaluator_spy(*args, **kwargs):
        records = args[0] if args else kwargs["records"]
        coordinate_evaluator_calls.append(len(records))
        return coordinate_evaluator(*args, **kwargs)

    def _trust_update_spy(*args, **kwargs):
        trust_update_calls.append(1)
        return trust_updater(*args, **kwargs)

    monkeypatch.setattr(
        adapt_pipeline,
        "evaluate_historical_singleton_coordinate_models",
        _coordinate_evaluator_spy,
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "update_trust_region_state",
        _trust_update_spy,
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "_phase_shortlist_with_legacy_hook",
        lambda *_args, **_kwargs: [],
    )
    h_poly = build_hubbard_holstein_hamiltonian(
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
    qpb = boson_qubits_per_site(2, "binary")

    payload, _ = adapt_pipeline._run_hardcoded_adapt_vqe(
        h_poly=h_poly,
        num_sites=2,
        ordering="blocked",
        problem="hh",
        adapt_pool="full_meta",
        adapt_pool_class_filter_json=_write_small_full_meta_filter(tmp_path),
        t=1.0,
        u=0.5,
        dv=0.0,
        boundary="open",
        omega0=1.0,
        g_ep=0.2,
        n_ph_max=2,
        boson_encoding="binary",
        max_depth=2,
        eps_grad=0.0,
        eps_energy=0.0,
        maxiter=2,
        seed=7,
        adapt_inner_optimizer="POWELL",
        allow_repeats=True,
        finite_angle_fallback=False,
        finite_angle=0.1,
        finite_angle_min_improvement=1.0e-12,
        adapt_reopt_policy="windowed",
        adapt_window_size=1,
        adapt_window_topk=0,
        adapt_final_full_refit=False,
        adapt_insertion_mode="append_only",
        phase0_pilot_enabled=False,
        static_route_id="route_a",
        static_meta_feature_profile="paper_i_production_v1",
        static_lane_route="physical_operator_type",
        physical_lane_shortlist_aggressiveness=3,
        phase1_shortlist_size=24,
        phase2_shortlist_size=12,
        phase2_shortlist_fraction=0.25,
        phase3_selector_policy="algebraic_nested_v1",
        phase3_runtime_split_mode="shortlist_pauli_children_v1",
        phase3_runtime_split_selection_mode="archival_child_set_forward_v1",
        phase3_runtime_split_max_subset_size=1,
        phase3_runtime_split_subset_sizes="1",
        phase3_runtime_split_child_set_symmetry_policy="hard_guard",
        phase3_response_coordinate_scope="full_active_plus_singleton_v1",
        route_a_child_padding_config=RouteAChildPaddingConfig(
            policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
            problem_key="hh",
            num_sites=2,
            n_ph_max=2,
            boson_encoding="binary",
            total_register_width=int(4 + 2 * qpb),
        ),
        historical_singleton_coordinate_solve_policy=(
            "supported_metric_whitened_eigh_v1"
        ),
        historical_singleton_coordinate_solve_scope="phase3_only_v1",
        historical_singleton_trust_region_update_policy=(
            "displacement_calibrated_unbounded_v2"
        ),
        adapt_accepted_refit_scope="full_ansatz_v1",
        adapt_accepted_refit_coordinate_chart=(
            "supported_fs_whitened_fixed_v1"
        ),
        adapt_accepted_refit_base_chart_policy=base_chart_policy,
        adapt_estimator_call_ledger_enabled=True,
    )

    assert payload["success"] is True
    assert len(payload["history"]) == 2
    assert all(int(row["batch_size"]) == 1 for row in payload["history"])
    assert len(coordinate_evaluator_calls) == len(payload["history"])
    assert all(count > 0 for count in coordinate_evaluator_calls)
    assert len(trust_update_calls) == len(payload["history"])
    for active_count, row in enumerate(payload["history"]):
        assert row["phase3_response_coordinate_scope"] == (
            "full_active_plus_singleton_v1"
        )
        assert row["phase3_active_logical_coordinate_count"] == active_count
        assert row["phase3_response_pre_support_count"] == active_count + 1
        assert row["phase3_response_coordinate_indices"] == list(
            range(active_count + 1)
        )
        assert 0 < int(row["phase3_response_supported_rank"]) <= active_count + 1
        assert row["selected_feature_rows"][0][
            "phase2_joint_geometry_reuse"
        ]["schema"] == "historical_singleton_coordinate_model_v1"
        trust_receipt = row["route_a_trust_region_update"]
        assert trust_receipt["policy"] == "displacement_calibrated_unbounded_v2"
        assert trust_receipt["context_mode"] == "full_ansatz_v1"
        assert trust_receipt["full_coordinate_refit"] is True
        assert trust_receipt["update_reason"] != "context_mode_not_supported"

    trust_state = payload["route_a_trust_region_state"]
    assert trust_state["update_count"] == len(payload["history"])
    accounting = payload["estimator_call_accounting"]
    ledger_payload = accounting["full_ledger"]
    scope_counts = accounting["executed_occurrence_accounting"][
        "all_execution"
    ]["occurrence_count_by_consumer_scope"]
    assert int(
        scope_counts.get("adaptive_trust_endpoint_overlap", 0)
    ) == len(payload["history"])
    assert int(
        scope_counts.get(
            "historical_singleton_whitening_active_gradient", 0
        )
    ) == sum(range(len(payload["history"])))
    active_gradient_ids = {
        str(occurrence["primitive_id"])
        for occurrence in ledger_payload["occurrences"]
        if occurrence["consumer_scope"]
        == "historical_singleton_whitening_active_gradient"
    }
    active_gradient_entries = [
        entry
        for entry in ledger_payload["entries"]
        if str(entry["primitive_id"]) in active_gradient_ids
    ]
    assert len(active_gradient_entries) == len(active_gradient_ids) == 1
    assert all(
        entry["identity"]["schema"] == "estimator_call_key_v2"
        and entry["identity"]["primitive_kind"] == "coordinate_gradient"
        and str(entry["identity"]["operand_identity"]).startswith(
            "physical_tangent_operand_v2:"
        )
        for entry in active_gradient_entries
    )
    for row in payload["history"]:
        overlap_accounting = row["route_a_trust_region_update"][
            "endpoint_overlap_query_accounting"
        ]
        assert overlap_accounting["status"] == "complete"
        assert overlap_accounting["formal_query_category"] == "N_cross"
    overlay = payload["continuation"]["runtime_split_summary"][
        "historical_singleton_coordinate_overlay_last_round"
    ]
    assert overlay["full_response_evaluated_count"] == coordinate_evaluator_calls[-1]
    assert overlay["retained_count_after"] == 1
    empty_shortlist_fallback = overlay["admission_domain"][
        "phase3_shortlist_empty_fallback"
    ]
    assert empty_shortlist_fallback == {
        "schema": "full_response_phase3_shortlist_empty_fallback_v1",
        "authority": "already_evaluated_full_response_population_argmax_v1",
        "selected_count": 1,
    }

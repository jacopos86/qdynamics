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



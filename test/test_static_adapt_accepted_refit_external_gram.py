from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pytest

import pipelines.static_adapt.accepted_refit as accepted_refit
from pipelines.static_adapt.accepted_refit import (
    ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1,
    ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1,
    AcceptedRefitConfig,
    ExternalLogicalFSGramReceipt,
    build_supported_fs_powell_chart,
)
from pipelines.static_adapt.exact_geometry_backend import (
    build_compiled_exact_manifold_adapter,
)
from pipelines.static_adapt.joint_linear_solve import JointLinearSolveConfig
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
)
from src.quantum.ansatz_parameterization import (
    project_runtime_theta_block_mean,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    compile_polynomial_action,
    energy_via_one_apply,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _horizontal_projective_fubini_study_gram(
    state: np.ndarray,
    tangents: np.ndarray,
) -> np.ndarray:
    """Independently form the real Gram matrix from horizontal tangents."""

    psi = np.asarray(state, dtype=complex).reshape(-1)
    tangent_matrix = np.asarray(tangents, dtype=complex)
    norm_sq = float(np.real(np.vdot(psi, psi)))
    overlaps = np.conjugate(psi) @ tangent_matrix
    horizontal_tangents = tangent_matrix - np.outer(
        psi,
        overlaps / norm_sq,
    )
    gram = np.asarray(
        np.real(
            np.conjugate(horizontal_tangents).T
            @ horizontal_tangents
            / norm_sq
        ),
        dtype=float,
    )
    return 0.5 * (gram + gram.T)


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
                "JW", [PauliTerm(2, ps="ey", pc=0.6)]
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
    rng = np.random.default_rng(9127)
    psi_ref = rng.normal(size=4) + 1.0j * rng.normal(size=4)
    psi_ref = np.asarray(psi_ref / np.linalg.norm(psi_ref), dtype=complex)
    h_compiled = compile_polynomial_action(
        PauliPolynomial(
            "JW",
            [
                PauliTerm(2, ps="ze", pc=0.7),
                PauliTerm(2, ps="ex", pc=-0.25),
                PauliTerm(2, ps="yy", pc=0.31),
            ],
        ),
        tol=1.0e-14,
    )
    theta_runtime = np.asarray([0.08, 0.08, 0.11], dtype=float)

    def objective(runtime_theta: np.ndarray) -> float:
        logical = project_runtime_theta_block_mean(
            np.asarray(runtime_theta, dtype=float), executor.layout
        )
        state = executor.prepare_state(logical, psi_ref)
        energy, _ = energy_via_one_apply(state, h_compiled)
        return float(energy)

    config = AcceptedRefitConfig(
        scope=ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1,
        coordinate_chart=ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1,
        base_chart_policy=(
            SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        ),
        supported_metric=JointLinearSolveConfig(
            rank_relative_tolerance=1.0e-8,
            metric_regularization=1.0e-9,
        ),
    )
    return executor, psi_ref, h_compiled, theta_runtime, objective, config


def _receipt_inputs():
    executor, psi_ref, h_compiled, theta_runtime, objective, config = _problem()
    adapter = build_compiled_exact_manifold_adapter(
        executor=executor,
        layout=executor.layout,
        theta_runtime=theta_runtime,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        manifold_id="external_gram_receipt_test",
    )
    evaluation = adapter.backend.evaluate(adapter.x0)
    tangents = np.asarray(evaluation.tangents, dtype=complex)
    gram = _horizontal_projective_fubini_study_gram(
        np.asarray(evaluation.statevector, dtype=complex),
        tangents,
    )
    summary = adapter.summary
    kwargs = {
        "logical_gram": gram,
        "origin_state": evaluation.statevector,
        "origin_energy": evaluation.energy,
        "origin_gradient": evaluation.gradient,
        "origin_logical_theta": adapter.x0,
        "origin_runtime_theta": theta_runtime,
        "coordinate_registry": tuple(adapter.coordinate_registry),
        "layout_fingerprint_sha256": summary["layout_sha256"],
        "coordinate_registry_fingerprint_sha256": summary[
            "coordinate_registry_sha256"
        ],
        "hamiltonian_fingerprint_sha256": summary[
            "hamiltonian_fingerprint"
        ],
        "ordered_scaffold_fingerprint_sha256": summary[
            "ordered_scaffold_fingerprint"
        ],
        "provenance_schema": "accepted_refit_external_test_source_v1",
        "provenance_id": "external-test-source-1",
        "provenance_payload": {
            "metric_status": "reused",
            "metric_tensor_convention": (
                "horizontal_projective_fubini_study_v1"
            ),
            "source": "independent_horizontal_projection_fixture",
        },
    }
    return (
        executor,
        psi_ref,
        h_compiled,
        theta_runtime,
        objective,
        config,
        adapter,
        kwargs,
    )


def _build(*, receipt: ExternalLogicalFSGramReceipt | None):
    executor, psi_ref, h_compiled, theta_runtime, objective, config = _problem()
    return build_supported_fs_powell_chart(
        executor=executor,
        layout=executor.layout,
        theta_runtime=theta_runtime,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        runtime_objective=objective,
        config=config,
        manifold_id="external_gram_receipt_test",
        external_logical_fs_gram_receipt=receipt,
    )


def test_fubini_study_metric_removes_pure_global_phase_tangent() -> None:
    state = np.asarray(
        [1.0 + 2.0j, -0.5 + 0.25j, 0.75 - 1.25j],
        dtype=complex,
    )
    state /= np.linalg.norm(state)
    pure_phase_tangent = np.asarray(1.0j * state, dtype=complex).reshape(-1, 1)

    gram = accepted_refit._fubini_study_gram(
        state,
        pure_phase_tangent,
    )

    np.testing.assert_allclose(
        gram,
        np.zeros((1, 1), dtype=float),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_external_exact_receipt_has_bitwise_chart_parity() -> None:
    *_, kwargs = _receipt_inputs()
    receipt = ExternalLogicalFSGramReceipt(**kwargs)
    acquired = _build(receipt=None)
    reused = _build(receipt=receipt)

    np.testing.assert_array_equal(reused.x0, acquired.x0)
    np.testing.assert_array_equal(
        reused.whitened_to_logical_map, acquired.whitened_to_logical_map
    )
    np.testing.assert_array_equal(
        reused.logical_to_whitened_map, acquired.logical_to_whitened_map
    )
    np.testing.assert_array_equal(reused.origin_state, acquired.origin_state)
    np.testing.assert_array_equal(
        reused.origin_logical_theta, acquired.origin_logical_theta
    )
    np.testing.assert_array_equal(
        reused.origin_runtime_theta, acquired.origin_runtime_theta
    )
    assert reused.base_telemetry["supported_metric_whitening_provenance_id"] == (
        acquired.base_telemetry["supported_metric_whitening_provenance_id"]
    )
    assert reused.base_telemetry["metric_input_status"] == "reused"
    assert acquired.base_telemetry["metric_input_status"] == "acquired"
    assert reused.base_telemetry["metric_backend_evaluation_performed"] is False
    assert acquired.base_telemetry["metric_backend_evaluation_performed"] is True
    assert reused.base_telemetry["metric_element_count_acquired_for_chart"] == 0
    assert reused.base_telemetry["metric_element_count_reused_for_chart"] == 3
    assert reused.base_telemetry["external_logical_fs_gram_receipt"][
        "receipt_id"
    ] == receipt.receipt_id


def test_external_receipt_preserves_uniform_wide_runtime_aliases_bitwise() -> None:
    pauli_words = [
        "".join(letters)
        for letters in itertools.product(("e", "x", "y", "z"), repeat=3)
        if letters != ("e", "e", "e")
    ][:24]
    executor = CompiledAnsatzExecutor(
        [
            AnsatzTerm(
                label="wide-macro",
                polynomial=PauliPolynomial(
                    "JW",
                    [
                        PauliTerm(3, ps=word, pc=1.0 / (index + 1))
                        for index, word in enumerate(pauli_words)
                    ],
                ),
            )
        ],
        coefficient_tolerance=1.0e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode="logical_shared",
    )
    assert executor.layout.runtime_parameter_count == 24
    runtime_value = float.fromhex("-0x1.5f4dcc508b69ap-2")
    theta_runtime = np.full(
        executor.layout.runtime_parameter_count,
        runtime_value,
        dtype=float,
    )
    psi_ref = np.zeros(8, dtype=complex)
    psi_ref[0] = 1.0
    h_compiled = compile_polynomial_action(
        PauliPolynomial(
            "JW",
            [
                PauliTerm(3, ps="zee", pc=0.7),
                PauliTerm(3, ps="exe", pc=-0.25),
            ],
        ),
        tol=1.0e-14,
    )

    def objective(runtime_theta: np.ndarray) -> float:
        logical = project_runtime_theta_block_mean(
            np.asarray(runtime_theta, dtype=float), executor.layout
        )
        state = executor.prepare_state(logical, psi_ref)
        energy, _ = energy_via_one_apply(state, h_compiled)
        return float(energy)

    config = AcceptedRefitConfig(
        scope=ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1,
        coordinate_chart=ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1,
        base_chart_policy=(
            SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        ),
        supported_metric=JointLinearSolveConfig(
            rank_relative_tolerance=1.0e-8,
            metric_regularization=1.0e-9,
        ),
    )
    adapter = build_compiled_exact_manifold_adapter(
        executor=executor,
        layout=executor.layout,
        theta_runtime=theta_runtime,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        manifold_id="external_gram_wide_uniform_alias_test",
    )
    evaluation = adapter.backend.evaluate(adapter.x0)
    gram = _horizontal_projective_fubini_study_gram(
        np.asarray(evaluation.statevector, dtype=complex),
        np.asarray(evaluation.tangents, dtype=complex),
    )
    summary = adapter.summary
    receipt = ExternalLogicalFSGramReceipt(
        logical_gram=gram,
        origin_state=evaluation.statevector,
        origin_energy=evaluation.energy,
        origin_gradient=evaluation.gradient,
        origin_logical_theta=adapter.x0,
        origin_runtime_theta=theta_runtime,
        coordinate_registry=tuple(adapter.coordinate_registry),
        layout_fingerprint_sha256=summary["layout_sha256"],
        coordinate_registry_fingerprint_sha256=summary[
            "coordinate_registry_sha256"
        ],
        hamiltonian_fingerprint_sha256=summary[
            "hamiltonian_fingerprint"
        ],
        ordered_scaffold_fingerprint_sha256=summary[
            "ordered_scaffold_fingerprint"
        ],
        provenance_schema="accepted_refit_external_test_source_v1",
        provenance_id="external-wide-uniform-test-source",
        provenance_payload={
            "metric_status": "reused",
            "metric_tensor_convention": (
                "horizontal_projective_fubini_study_v1"
            ),
            "source": "wide_uniform_runtime_alias_fixture",
        },
    )

    chart = build_supported_fs_powell_chart(
        executor=executor,
        layout=executor.layout,
        theta_runtime=theta_runtime,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        runtime_objective=objective,
        config=config,
        manifold_id="external_gram_wide_uniform_alias_test",
        external_logical_fs_gram_receipt=receipt,
    )

    np.testing.assert_array_equal(
        chart.origin_runtime_theta,
        theta_runtime,
    )
    np.testing.assert_array_equal(
        adapter.lift_to_runtime(chart.origin_logical_theta),
        theta_runtime,
    )


def test_external_receipt_skips_backend_evaluate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        executor,
        psi_ref,
        h_compiled,
        theta_runtime,
        objective,
        config,
        adapter,
        kwargs,
    ) = _receipt_inputs()
    receipt = ExternalLogicalFSGramReceipt(**kwargs)

    def _forbidden(_theta: np.ndarray):
        raise AssertionError("full backend evaluate was invoked")

    adapter.backend._evaluate_fn = _forbidden
    monkeypatch.setattr(
        accepted_refit,
        "build_compiled_exact_manifold_adapter",
        lambda **_kwargs: adapter,
    )
    chart = build_supported_fs_powell_chart(
        executor=executor,
        layout=executor.layout,
        theta_runtime=theta_runtime,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        runtime_objective=objective,
        config=config,
        manifold_id="external_gram_receipt_test",
        external_logical_fs_gram_receipt=receipt,
    )
    assert chart.base_telemetry["metric_backend_evaluation_performed"] is False

    with pytest.raises(AssertionError, match="full backend evaluate was invoked"):
        build_supported_fs_powell_chart(
            executor=executor,
            layout=executor.layout,
            theta_runtime=theta_runtime,
            psi_ref=psi_ref,
            h_compiled=h_compiled,
            runtime_objective=objective,
            config=config,
            manifold_id="external_gram_receipt_test",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda data: {
                **data,
                "layout_fingerprint_sha256": "0" * 64,
            },
            "layout fingerprint mismatch",
        ),
        (
            lambda data: {
                **data,
                "coordinate_registry": tuple(reversed(data["coordinate_registry"])),
                "coordinate_registry_fingerprint_sha256": _json_sha256(
                    list(reversed(data["coordinate_registry"]))
                ),
            },
            "coordinate registry mismatch",
        ),
        (
            lambda data: {
                **data,
                "origin_logical_theta": np.asarray(
                    data["origin_logical_theta"], dtype=float
                )
                + 1.0e-5,
            },
            "logical-theta fingerprint mismatch",
        ),
    ],
)
def test_external_receipt_tampering_fails_closed(mutation, match: str) -> None:
    *_, kwargs = _receipt_inputs()
    receipt = ExternalLogicalFSGramReceipt(**mutation(kwargs))
    with pytest.raises(ValueError, match=match):
        _build(receipt=receipt)


def test_external_receipt_dimension_mismatch_fails_at_construction() -> None:
    *_, kwargs = _receipt_inputs()
    kwargs = {
        **kwargs,
        "logical_gram": np.eye(3, dtype=float),
    }
    with pytest.raises(ValueError, match="logical_gram shape"):
        ExternalLogicalFSGramReceipt(**kwargs)

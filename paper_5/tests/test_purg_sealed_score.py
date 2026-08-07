from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.purg_sealed_score import (
    BlindAuditConstructionStop,
    BlindAuditSettings,
    FrozenScoreConfig,
    ObservablePath,
    PropagationFamily,
    ResolutionEvaluation,
    _array_fingerprint,
    _exponential_action,
    block_path_metrics,
    build_blind_residual_audit_basis,
    contract_analytic_observable_path,
    evaluate_resolution_and_science,
    propagate_state_midpoint,
    run_sealed_score,
)


def _path(
    *,
    output_value: float = 0.0,
    derivative_value: float = 0.0,
    norm_drift: float = 0.0,
    count: int = 5,
) -> ObservablePath:
    times = np.linspace(0.0, 0.01 * (count - 1), count)
    outputs = np.zeros((count, 31), dtype=float)
    derivatives = np.zeros_like(outputs)
    outputs[:, 0] = output_value
    derivatives[:, 0] = derivative_value
    return ObservablePath(
        times=times,
        outputs=outputs,
        derivatives=derivatives,
        maximum_norm_drift=norm_drift,
        method="synthetic",
    )


def _family(
    *,
    fine: ObservablePath | None = None,
    repeat: ObservablePath | None = None,
    coarse: ObservablePath | None = None,
    coarse_repeat: ObservablePath | None = None,
    dop853: ObservablePath | None = None,
) -> PropagationFamily:
    fine_path = fine or _path()
    coarse_path = coarse or ObservablePath(
        times=fine_path.times[::2],
        outputs=fine_path.outputs[::2],
        derivatives=fine_path.derivatives[::2],
        maximum_norm_drift=fine_path.maximum_norm_drift,
        method="synthetic_coarse",
    )
    return PropagationFamily(
        fine_primary=fine_path,
        fine_repeat=repeat or fine_path,
        coarse_primary=coarse_path,
        coarse_repeat=coarse_repeat or coarse_path,
        dop853=dop853 or fine_path,
    )


def _evaluation(*, numerical: bool, scientific: bool | None) -> ResolutionEvaluation:
    zero = {
        f"{quantity}.{block}.{statistic}": 0.0
        for quantity in ("output", "derivative")
        for block in ("rho", "B", "N", "A", "C")
        for statistic in ("rms", "max")
    }
    components = {
        "method": zero.copy(),
        "step": zero.copy(),
        "tolerance": zero.copy(),
    }
    return ResolutionEvaluation(
        numerical_passed=numerical,
        numerical_failures=() if numerical else ("synthetic numerical failure",),
        model_errors={128: zero.copy(), 160: zero.copy()},
        model_resolution={128: zero.copy(), 160: zero.copy()},
        model_resolution_components={128: components, 160: components},
        rank_difference=zero.copy(),
        rank_resolution=zero.copy(),
        rank_resolution_components=components,
        tolerance_repeat=zero.copy(),
        norm_drifts={},
        scientific_passed=scientific,
        scientific_failures=(
            () if scientific is not False else ("synthetic scientific failure",)
        ),
    )


def test_explicit_tolerance_exponential_action_matches_dense_exponential() -> None:
    hamiltonian = np.array(
        [[0.7, 0.2 - 0.1j], [0.2 + 0.1j, -0.3]], dtype=complex
    )
    state = np.array([0.3 + 0.4j, -0.2 + 0.5j], dtype=complex)
    state /= np.linalg.norm(state)
    step = 0.037

    actual = _exponential_action(
        -1j * hamiltonian,
        state,
        step=step,
        relative_tolerance=1.0e-13,
    )
    expected = expm(-1j * step * hamiltonian) @ state
    np.testing.assert_allclose(actual, expected, atol=2.0e-14, rtol=2.0e-14)


def test_midpoint_constant_hamiltonian_is_unitary_without_renormalization() -> None:
    hamiltonian = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    drive = np.zeros_like(hamiltonian)
    initial = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)
    parameters = DimerParameters(drive_amplitude=0.0)

    times, states, drift = propagate_state_midpoint(
        hamiltonian,
        drive,
        initial,
        parameters,
        final_time=0.04,
        step=0.01,
        exponential_action_tolerance=1.0e-13,
    )
    expected = np.asarray(
        [expm(-1j * time * hamiltonian) @ initial for time in times]
    )
    np.testing.assert_allclose(states, expected, atol=4.0e-14, rtol=4.0e-14)
    assert drift < 4.0e-14


def test_blind_greedy_pivots_use_earliest_relative_tie_and_preserve_nesting() -> None:
    base = np.eye(7, dtype=complex)[:, :2]
    residuals = np.zeros((7, 5), dtype=complex)
    residuals[2, 0] = 1.0j
    residuals[3, 1] = 1.0 + 5.0e-15
    residuals[4, 2] = 0.7 - 0.2j
    residuals[5, 3] = 0.3
    residuals[6, 4] = 0.2
    settings = BlindAuditSettings(
        base_rank=2,
        appended_directions=3,
        final_time=0.04,
        step=0.01,
    )

    result = build_blind_residual_audit_basis(
        base,
        residuals,
        np.arange(5, dtype=float) * 0.01,
        settings=settings,
    )

    assert result.pivot_indices == (0, 1, 2)
    assert result.basis.shape == (7, 5)
    assert result.orthogonality_residual < 1.0e-14
    assert result.nesting_residual < 1.0e-14


def test_blind_greedy_deflation_is_a_construction_stop() -> None:
    base = np.eye(5, dtype=complex)[:, :2]
    residuals = np.zeros((5, 2), dtype=complex)
    residuals[2, 0] = 1.0
    residuals[3, 1] = 1.0e-13
    settings = BlindAuditSettings(
        base_rank=2,
        appended_directions=2,
        final_time=0.01,
        step=0.01,
    )

    with pytest.raises(BlindAuditConstructionStop, match="fewer than 32"):
        build_blind_residual_audit_basis(
            base,
            residuals,
            np.array([0.0, 0.01]),
            settings=settings,
        )


def test_analytic_centered_derivative_matches_wavefunction_difference() -> None:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    observables = [np.zeros((2, 2), dtype=complex) for _ in range(29)]
    observables[0] = sigma_z
    state = np.array([0.8, 0.6j], dtype=complex)
    state /= np.linalg.norm(state)
    parameters = DimerParameters(drive_amplitude=0.0)
    path = contract_analytic_observable_path(
        times=np.array([0.0, 0.01]),
        states=np.vstack((state, expm(-0.01j * sigma_x) @ state)),
        static_hamiltonian=sigma_x,
        drive_hamiltonian=np.zeros_like(sigma_x),
        raw_observables=observables,
        parameters=parameters,
        method="analytic_test",
    )

    step = 1.0e-7
    minus = expm(1j * step * sigma_x) @ state
    plus = expm(-1j * step * sigma_x) @ state
    finite_difference = (
        contract_analytic_observable_path(
            times=np.array([0.0, 0.01]),
            states=np.vstack((plus, plus)),
            static_hamiltonian=sigma_x,
            drive_hamiltonian=np.zeros_like(sigma_x),
            raw_observables=observables,
            parameters=parameters,
            method="plus",
        ).outputs[0]
        - contract_analytic_observable_path(
            times=np.array([0.0, 0.01]),
            states=np.vstack((minus, minus)),
            static_hamiltonian=sigma_x,
            drive_hamiltonian=np.zeros_like(sigma_x),
            raw_observables=observables,
            parameters=parameters,
            method="minus",
        ).outputs[0]
    ) / (2.0 * step)
    np.testing.assert_allclose(
        path.derivatives[0], finite_difference, atol=2.0e-9, rtol=2.0e-9
    )


def test_block_metrics_use_equal_weight_discrete_rms_and_14_real_c_pack() -> None:
    output = np.zeros((2, 31), dtype=float)
    derivative = np.zeros_like(output)
    output[:, 0] = (3.0, 4.0)
    derivative[0, 17:31] = 1.0
    metrics = block_path_metrics(output, derivative)

    assert metrics["output.rho.rms"] == pytest.approx(np.sqrt(12.5))
    assert metrics["output.rho.max"] == 4.0
    assert metrics["derivative.C.rms"] == pytest.approx(np.sqrt(7.0))
    assert metrics["derivative.C.max"] == pytest.approx(np.sqrt(14.0))


def test_numerical_resolution_adds_full_and_reduced_errors_before_gating() -> None:
    fine = _path()
    shifted = _path(output_value=1.0e-5)
    full = _family(fine=shifted, repeat=shifted, dop853=fine)
    reduced = _family(fine=shifted, repeat=shifted, dop853=fine)
    rank_160 = _family(fine=shifted, repeat=shifted, dop853=fine)

    evaluation = evaluate_resolution_and_science(
        full=full,
        rank_128=reduced,
        rank_160=rank_160,
    )

    assert evaluation.model_errors[128]["output.rho.rms"] == 0.0
    assert evaluation.model_resolution[128]["output.rho.rms"] == pytest.approx(
        2.0e-5
    )
    assert not evaluation.numerical_passed
    assert evaluation.scientific_passed is None


def test_rank_160_cannot_rescue_a_scientifically_failing_rank_128() -> None:
    exact = _family()
    rank_128 = _family(fine=_path(output_value=2.0e-4))
    rank_160 = _family()

    evaluation = evaluate_resolution_and_science(
        full=exact,
        rank_128=rank_128,
        rank_160=rank_160,
    )

    assert evaluation.numerical_passed
    assert evaluation.scientific_passed is False
    assert any("rank_128.scientific.output.rho" in value for value in evaluation.scientific_failures)


def test_array_fingerprint_is_content_and_shape_sensitive() -> None:
    first = np.arange(6, dtype=np.float64).reshape(2, 3)
    same = first.copy()
    changed = first.copy()
    changed[0, 0] = -1.0

    assert _array_fingerprint(first) == _array_fingerprint(same)
    assert _array_fingerprint(first) != _array_fingerprint(changed)
    assert _array_fingerprint(first) != _array_fingerprint(first.reshape(3, 2))


def test_observable_path_rejects_nonfinite_data_and_norm_drift() -> None:
    outputs = np.zeros((2, 31), dtype=float)
    derivatives = np.zeros_like(outputs)
    outputs[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        ObservablePath(
            times=np.array([0.0, 0.01]),
            outputs=outputs,
            derivatives=derivatives,
            maximum_norm_drift=0.0,
            method="nonfinite",
        )
    with pytest.raises(ValueError, match="norm_drift must be finite"):
        ObservablePath(
            times=np.array([0.0, 0.01]),
            outputs=np.zeros((2, 31)),
            derivatives=derivatives,
            maximum_norm_drift=float("nan"),
            method="nonfinite_drift",
        )


def test_sealed_score_uses_exactly_one_fallback_and_serializes_no_model(
    tmp_path, monkeypatch
) -> None:
    import paper5.stability.purg_sealed_score as module

    prepared = SimpleNamespace(manifest_sha256="a" * 64)
    build_calls: list[bool] = []

    def fake_build(*args, fallback: bool, **kwargs):
        build_calls.append(fallback)
        return {"full": object(), "rank_128": object(), "rank_160": object()}

    evaluations = iter(
        (
            _evaluation(numerical=False, scientific=None),
            _evaluation(numerical=True, scientific=False),
        )
    )
    monkeypatch.setattr(module, "load_prepared_score", lambda *a, **k: prepared)
    monkeypatch.setattr(module, "_build_all_families", fake_build)
    monkeypatch.setattr(
        module, "evaluate_resolution_and_science", lambda **kwargs: next(evaluations)
    )
    monkeypatch.setattr(
        module, "_authoritative_arrays", lambda families: {"sentinel": np.array([1])}
    )

    prepared_directory = tmp_path / "prepared"
    prepared_directory.mkdir()
    output = tmp_path / "score"
    summary = run_sealed_score(
        prepared_directory,
        output,
        repo_root=tmp_path,
        config=FrozenScoreConfig(),
    )

    assert build_calls == [False, True]
    assert summary["fallback_count"] == 1
    assert summary["status"] == "scientific_hard_stop"
    assert summary["serialized_model"] is None
    assert not (output / "rank_128_model.npz").exists()
    assert (prepared_directory / "score_consumption_receipt.json").is_file()

    with pytest.raises(FileExistsError, match="consumption"):
        run_sealed_score(
            prepared_directory,
            tmp_path / "second_score",
            repo_root=tmp_path,
            config=FrozenScoreConfig(),
        )


def test_second_numerical_failure_is_an_indeterminate_stop(
    tmp_path, monkeypatch
) -> None:
    import paper5.stability.purg_sealed_score as module

    prepared = SimpleNamespace(manifest_sha256="b" * 64)
    prepared_directory = tmp_path / "prepared"
    prepared_directory.mkdir()
    build_calls: list[bool] = []

    def fake_build(*args, fallback: bool, **kwargs):
        build_calls.append(fallback)
        return {"full": object(), "rank_128": object(), "rank_160": object()}

    evaluations = iter(
        (
            _evaluation(numerical=False, scientific=None),
            _evaluation(numerical=False, scientific=None),
        )
    )
    monkeypatch.setattr(module, "load_prepared_score", lambda *a, **k: prepared)
    monkeypatch.setattr(module, "_build_all_families", fake_build)
    monkeypatch.setattr(
        module, "evaluate_resolution_and_science", lambda **kwargs: next(evaluations)
    )
    monkeypatch.setattr(
        module, "_authoritative_arrays", lambda families: {"sentinel": np.array([1])}
    )

    output = tmp_path / "score"
    summary = run_sealed_score(
        prepared_directory,
        output,
        repo_root=tmp_path,
        config=FrozenScoreConfig(),
    )

    assert build_calls == [False, True]
    assert summary["status"] == "indeterminate_numerical_stop"
    assert summary["final_attempt"]["scientific"] is None
    assert summary["serialized_model"] is None
    assert not (output / "rank_128_model.npz").exists()

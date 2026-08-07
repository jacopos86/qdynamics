from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from paper5.stability import (
    ELECTRONIC_CONDITIONED_GAUSSIAN_CLOSURE,
    FOURTH_CUMULANT_HIERARCHY,
    FOURTH_CUMULANT_MOMENT_KEYS,
    ZERO_CUMULANT_CLOSURE,
    DimerParameters,
    MomentKey,
    fourth_cumulant_rhs,
    pack_fourth_cumulant_state,
)
from paper5.stability.moment_hierarchy import PAULI_LABELS
from paper5.stability.conditional_closure_analysis import (
    run_conditional_closure_analysis,
)

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _gaussian_raw_moment(
    x_power: int,
    p_power: int,
    mean_x: float,
    mean_p: float,
    covariance_xx: float,
    covariance_xp: float,
    covariance_pp: float,
) -> float:
    @lru_cache(maxsize=None)
    def moment(a: int, b: int) -> float:
        if a == 0 and b == 0:
            return 1.0
        if a > 0:
            value = mean_x * moment(a - 1, b)
            if a >= 2:
                value += (a - 1) * covariance_xx * moment(a - 2, b)
            if b > 0:
                value += b * covariance_xp * moment(a - 1, b - 1)
            return value
        value = mean_p * moment(0, b - 1)
        if b >= 2:
            value += (b - 1) * covariance_pp * moment(0, b - 2)
        return value

    return moment(x_power, p_power)


def _conditional_gaussian_matrix(x_power: int, p_power: int) -> np.ndarray:
    weights = np.array([0.18, 0.31, 0.31, 0.20])
    mean_x = np.array([-1.2, -0.25, -0.25, 0.95])
    mean_p = np.array([0.13, -0.07, -0.07, 0.21])
    covariance_xx = np.array([0.62, 0.48, 0.48, 0.71])
    covariance_xp = np.array([0.04, -0.03, -0.03, 0.02])
    covariance_pp = np.array([0.55, 0.66, 0.66, 0.59])
    diagonal = [
        weights[index]
        * _gaussian_raw_moment(
            x_power,
            p_power,
            mean_x[index],
            mean_p[index],
            covariance_xx[index],
            covariance_xp[index],
            covariance_pp[index],
        )
        for index in range(4)
    ]
    return np.diag(diagonal).astype(complex)


def _symmetric_pauli_expectation(
    matrix: np.ndarray,
    left: str,
    right: str,
) -> float:
    oriented = np.kron(_PAULI[left], _PAULI[right])
    value = np.trace(oriented @ matrix)
    if left != right:
        swapped = np.kron(_PAULI[right], _PAULI[left])
        value = 0.5 * (value + np.trace(swapped @ matrix))
    assert abs(value.imag) < 1e-14
    return float(value.real)


def _conditional_gaussian_moments() -> dict[MomentKey, float]:
    return {
        key: _symmetric_pauli_expectation(
            _conditional_gaussian_matrix(key.x_power, key.p_power),
            key.spin_up,
            key.spin_down,
        )
        for key in FOURTH_CUMULANT_MOMENT_KEYS
    }


def test_electronic_conditioned_gaussian_recovers_branchwise_wick_rule() -> None:
    moments = _conditional_gaussian_moments()
    resolver = ELECTRONIC_CONDITIONED_GAUSSIAN_CLOSURE.prepare(moments, 4)

    assert resolver.diagnostics["support_rank"] == 4
    assert resolver.diagnostics["maximum_jordan_relative_residual"] < 2e-14
    for x_power in range(4, -1, -1):
        p_power = 4 - x_power
        for pauli in PAULI_LABELS[1:]:
            key = MomentKey("I", pauli, x_power, p_power)
            expected = _symmetric_pauli_expectation(
                _conditional_gaussian_matrix(x_power, p_power),
                key.spin_up,
                key.spin_down,
            )
            assert abs(resolver.moment(key) - expected) < 2e-12

    for x_power in range(3, -1, -1):
        p_power = 3 - x_power
        for left_index, left in enumerate(PAULI_LABELS[1:], start=1):
            for right in PAULI_LABELS[left_index:]:
                key = MomentKey(left, right, x_power, p_power)
                expected = _symmetric_pauli_expectation(
                    _conditional_gaussian_matrix(x_power, p_power),
                    key.spin_up,
                    key.spin_down,
                )
                assert abs(resolver.moment(key) - expected) < 2e-12


def test_default_fourth_rhs_remains_the_zero_cumulant_adapter() -> None:
    moments = _conditional_gaussian_moments()
    state = pack_fourth_cumulant_state(-0.2 + 0.1j, moments)
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)

    np.testing.assert_array_equal(
        fourth_cumulant_rhs(0.3, state, parameters),
        fourth_cumulant_rhs(
            0.3,
            state,
            parameters,
            closure=ZERO_CUMULANT_CLOSURE,
        ),
    )
    adapted = fourth_cumulant_rhs(
        0.3,
        state,
        parameters,
        closure=ELECTRONIC_CONDITIONED_GAUSSIAN_CLOSURE,
    )
    assert adapted.shape == (FOURTH_CUMULANT_HIERARCHY.coordinate_count,)
    assert np.all(np.isfinite(adapted))


def test_conditional_closure_analysis_writes_retrievable_gate(
    tmp_path: Path,
) -> None:
    summary = run_conditional_closure_analysis(
        tmp_path,
        parameters=DimerParameters(
            lambda_ep=0.0,
            gamma=0.5,
            drive_amplitude=1.0,
        ),
        final_time=0.1,
        sample_step=0.1,
        phonon_cutoff=3,
        convergence_cutoffs=(3,),
        convergence_time=0.1,
        maximum_step=0.02,
    )

    assert summary["validation_gate"]["adapted_closure_passed"]
    assert summary["decoupled_control_maximum_component_defect"] < 1e-12
    prefix = "electronic_conditioned_gaussian_closure_gate"
    assert (tmp_path / f"{prefix}.npz").is_file()
    assert (tmp_path / f"{prefix}.png").is_file()
    manifest = json.loads(
        (tmp_path / "runtime_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert "summary.json" in manifest["artifact_hashes"]

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.matrix_reference import closed_scalar_rhs
from paper5.stability.quantum_target_form import (
    ClosedPolynomialForm,
    carleman_block_derivative,
    carleman_dimensions,
    carleman_lift,
    carleman_truncation_remainder_norms,
    extract_closed_polynomial_form,
    extract_cubic_polynomial,
    integrate_carleman_rk4,
    linear_part_spectrum,
)


@pytest.fixture(scope="module")
def form() -> ClosedPolynomialForm:
    return extract_closed_polynomial_form(DimerParameters())


def _symmetric(tensor: np.ndarray) -> np.ndarray:
    axes = tensor.ndim - 1
    if axes == 2:
        return 0.5 * (tensor + tensor.transpose(0, 2, 1))
    from itertools import permutations

    total = np.zeros_like(tensor)
    perms = list(permutations(range(1, axes + 1)))
    for perm in perms:
        total += tensor.transpose((0, *perm))
    return total / len(perms)


def _synthetic_form(n: int, seed: int) -> ClosedPolynomialForm:
    rng = np.random.default_rng(seed)
    return ClosedPolynomialForm(
        parameters=DimerParameters(),
        constant=rng.normal(size=n),
        linear_free=rng.normal(size=(n, n)),
        linear_drive=rng.normal(size=(n, n)),
        quadratic=_symmetric(rng.normal(size=(n, n, n))),
        cubic=_symmetric(rng.normal(size=(n, n, n, n))),
    )


def test_cubic_interpolation_recovers_a_synthetic_cubic_exactly():
    reference = _synthetic_form(5, 7)
    extracted = extract_cubic_polynomial(
        lambda x: reference.evaluate(0.0, x), 5
    )
    assert extracted.evaluations == 1 + 3 * 5 + 3 * 10 + 10
    np.testing.assert_allclose(extracted.constant, reference.constant, atol=1e-13)
    np.testing.assert_allclose(extracted.linear, reference.linear_free, atol=1e-12)
    np.testing.assert_allclose(extracted.quadratic, reference.quadratic, atol=1e-12)
    np.testing.assert_allclose(extracted.cubic, reference.cubic, atol=1e-12)


def test_polynomial_form_reproduces_committed_rhs(form):
    parameters = form.parameters
    rng = np.random.default_rng(11)
    for scale in (0.3, 2.0):
        for _ in range(40):
            x = scale * rng.normal(size=31)
            t = float(rng.uniform(0.0, 6.0))
            reference = closed_scalar_rhs(t, x, parameters)
            error = np.max(np.abs(form.evaluate(t, x) - reference))
            assert error <= 1e-13 * max(1.0, float(np.max(np.abs(reference))))


def test_drive_enters_only_through_a_skew_linear_term(form):
    for row in form.drive_audit.values():
        assert row["constant_drive_max_abs"] <= 1e-14
        assert row["quadratic_drive_max_abs"] <= 1e-14
        assert row["cubic_drive_max_abs"] <= 1e-14
        assert row["linear_drive_disagreement_max_abs"] <= 1e-14
    np.testing.assert_allclose(
        form.linear_drive, -form.linear_drive.T, atol=1e-14
    )
    undriven = replace(form.parameters, drive_amplitude=0.0)
    x = np.random.default_rng(3).normal(size=31)
    np.testing.assert_allclose(
        form.evaluate(0.0, x), closed_scalar_rhs(0.0, x, undriven), atol=1e-13
    )


def test_archive_rhs_has_no_cubic_part_and_is_divergence_free(form):
    # Measured: the Eq. (14d) normal particle and hole sources combine into
    # -i N[q,q'] [g_q'^T, rho], so their rho g rho products cancel.
    assert np.max(np.abs(form.cubic)) <= 1e-14
    assert abs(np.trace(form.linear_free)) <= 1e-14
    assert abs(np.trace(form.linear_drive)) <= 1e-14
    assert np.max(np.abs(np.einsum("iij->j", form.quadratic))) <= 1e-14


def test_drive_free_linear_part_has_imaginary_spectrum(form):
    spectrum = linear_part_spectrum(form.linear_free)
    assert spectrum["max_abs_real_part"] <= 1e-12
    assert spectrum["symmetric_part_max"] > 1.0  # not Euclidean-skew


def test_carleman_dimensions():
    assert carleman_dimensions(31, 2) == {"tensor": 992, "symmetric": 527}
    assert carleman_dimensions(31, 4) == {"tensor": 954304, "symmetric": 52359}


@pytest.mark.parametrize("order", [1, 2, 3])
def test_truncation_drops_exactly_the_reported_terms(order):
    form = _synthetic_form(4, 19)
    x = np.random.default_rng(5).normal(size=4)
    time = 0.6
    blocks = carleman_lift(x, order)
    truncated = carleman_block_derivative(form, time, blocks, order)
    velocity = form.evaluate(time, x)
    dropped = []
    for k in range(1, order + 1):
        lifted = np.zeros((4,) * k)
        for position in range(k):
            factors = [x] * k
            factors[position] = velocity
            term = factors[0]
            for factor in factors[1:]:
                term = np.multiply.outer(term, factor)
            lifted = lifted + term
        dropped.append(np.linalg.norm(lifted - truncated[k - 1]))
    report = carleman_truncation_remainder_norms(form, time, x, order)
    np.testing.assert_allclose(
        report["dropped_by_block"], np.asarray(dropped), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("order", [2, 3, 4])
def test_carleman_of_riccati_equation_is_the_taylor_partial_sum(order):
    # x' = x^2: the order-K truncation gives sum_{j<K} x0^{j+1} t^j exactly.
    form = ClosedPolynomialForm(
        parameters=DimerParameters(drive_amplitude=0.0),
        constant=np.zeros(1),
        linear_free=np.zeros((1, 1)),
        linear_drive=np.zeros((1, 1)),
        quadratic=np.ones((1, 1, 1)),
        cubic=np.zeros((1, 1, 1, 1)),
    )
    times = np.linspace(0.0, 1.0, 41)
    solution = integrate_carleman_rk4(form, np.array([0.5]), order, sample_times=times)
    expected = sum(0.5 ** (j + 1) * times**j for j in range(order))
    np.testing.assert_allclose(solution.first_block[:, 0], expected, atol=1e-13)

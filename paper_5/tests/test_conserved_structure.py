from __future__ import annotations

import numpy as np
import pytest

from paper5.stability.conserved_structure import (
    B0I,
    B0R,
    SHARED_TRACE_REAL,
    centre_mode_basis,
    common_metric_obstruction,
    linear_first_integral,
    neutral_mode_report,
    order_two_resonances,
    reduce_at_centre_mode_fixed_point,
    signed_sector_report,
    skew_adjoint_metric_dimension,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.initial_conditions import exact_ground_closed_scalar_coordinates
from paper5.stability.quantum_target_form import extract_closed_polynomial_form


@pytest.fixture(scope="module")
def form():
    return extract_closed_polynomial_form(DimerParameters())


@pytest.fixture(scope="module")
def reduced(form):
    x0 = exact_ground_closed_scalar_coordinates(form.parameters, phonon_cutoff=16)
    return reduce_at_centre_mode_fixed_point(form, x0)


def test_centre_mode_coherent_field_decouples_exactly(form):
    s, s_inv = centre_mode_basis()
    np.testing.assert_allclose(s @ s_inv, np.eye(31), atol=1e-15)
    plus = [B0R, B0I]
    others = [i for i in range(31) if i not in plus]
    a1z = s_inv @ form.linear_free @ s
    assert np.max(np.abs(a1z[np.ix_(plus, others)])) == 0.0
    assert np.max(np.abs((s_inv @ form.linear_drive @ s)[plus])) == 0.0
    a2z = np.einsum("ma,abc,bj,ck->mjk", s_inv, form.quadratic, s, s)
    assert np.max(np.abs(a2z[plus])) == 0.0
    # dB_+/dt = -i omega B_+ - 2 i g
    omega, g = form.parameters.omega_ph, form.parameters.coupling
    np.testing.assert_allclose(a1z[np.ix_(plus, plus)], [[0.0, omega], [-omega, 0.0]], atol=1e-15)
    np.testing.assert_allclose((s_inv @ form.constant)[plus], [0.0, -2.0 * g], atol=1e-15)


def test_linear_first_integral(form):
    ell, singular = linear_first_integral(form)
    g = form.parameters.coupling
    expected = np.zeros(31)
    expected[SHARED_TRACE_REAL] = 1.0
    expected[7] = expected[8] = g / 6.0
    expected[9] = g / 3.0
    np.testing.assert_allclose(ell, expected, atol=1e-13)
    assert np.sort(singular)[0] <= 1e-13 and np.sort(singular)[1] >= 0.1


def test_neutral_modes_receive_the_quadratic_term(form, reduced):
    full = neutral_mode_report(form.constant, form.linear_free, form.quadratic)
    assert (full["rank_linear"], full["rank_forcing_linear"], full["rank_forcing_linear_quadratic"]) == (26, 26, 30)
    assert full["forcing_projection_norm"] <= 1e-12
    assert sum(v > 0.5 for v in full["quadratic_projection_singular_values"]) == 4  # the fifth is ell
    red = neutral_mode_report(reduced.constant, reduced.linear, reduced.quadratic)
    assert (red["rank_linear"], red["rank_forcing_linear"], red["rank_forcing_linear_quadratic"]) == (24, 24, 28)
    assert red["forcing_projection_norm"] <= 1e-12
    assert red["left_null_dimension"] == 4
    assert min(red["quadratic_projection_singular_values"]) >= 0.5  # all four zero modes
    assert abs(red["trace_linear"]) <= 1e-12 and red["max_abs_diagonal_linear"] <= 1e-12
    assert red["max_abs_real_eigenvalue"] <= 1e-12


def test_order_two_resonances_carry_quadratic_coupling(form, reduced):
    for linear, quadratic in ((form.linear_free, form.quadratic), (reduced.linear, reduced.quadratic)):
        report = order_two_resonances(linear, quadratic)
        assert report["resonant_ordered_pairs"] == 79
        assert report["with_nonzero_quadratic_coupling"] == 33
        assert report["nonzero_with_a_zero_input"] == 29
        assert report["smallest_nonzero_coupling"] >= 0.01
        assert report["largest_vanishing_coupling"] <= 1e-12


def test_no_common_skew_adjoint_metric_for_linear_part_and_drive(form, reduced):
    assert skew_adjoint_metric_dimension([form.linear_free]) == 46
    assert skew_adjoint_metric_dimension([form.linear_free, form.linear_drive]) == 8
    assert skew_adjoint_metric_dimension([reduced.linear, reduced.drive]) == 4
    for matrices in ([form.linear_free, form.linear_drive], [reduced.linear, reduced.drive]):
        report = common_metric_obstruction(matrices)
        assert report["algebra_dimension"] == 91
        assert report["top_element_square_frobenius"] <= 1e-10  # nilpotent of index 2
        assert report["top_element_exp_norms"]["1000"] >= 100.0  # unbounded group


def test_signed_sector_partition_fails(form, reduced):
    assert signed_sector_report(form.linear_free, form.quadratic)["modes_fitting_neither_sector"] == 5
    red = signed_sector_report(reduced.linear, reduced.quadratic)
    assert red["modes_fitting_neither_sector"] == 4
    assert red["quadratic_share_outside_allowed_blocks"] >= 0.5

from __future__ import annotations

import numpy as np

from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.mixed_operator_commutator import (
    FIRST_AUXILIARY_OPERATOR_LABELS,
    mixed_operator_commutator_audit,
)


def test_first_mixed_liouvillian_layer_matches_implemented_hamiltonian() -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    audit = mixed_operator_commutator_audit(
        parameters,
        time=1.37,
        relative_dimension=9,
        drive_protocol=GaussianSineDrive(
            amplitude=1.0,
            pulse_width=1.0,
            delays=(0.0, 8.0),
        ),
    )

    assert len(FIRST_AUXILIARY_OPERATOR_LABELS) == 9
    assert np.max(audit.cutoff_corrected_relative_residual) < 2e-15
    assert np.min(audit.infinite_boson_relative_residual) > 0.1
    np.testing.assert_allclose(
        audit.infinite_boson_relative_residual,
        audit.cutoff_boundary_relative_norm,
        atol=2e-15,
        rtol=2e-15,
    )


def test_commutator_audit_requires_nontrivial_relative_space() -> None:
    parameters = DimerParameters()
    try:
        mixed_operator_commutator_audit(
            parameters,
            time=0.0,
            relative_dimension=1,
        )
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("expected a relative-dimension validation error")

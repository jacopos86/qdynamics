from __future__ import annotations

import numpy as np

from pipelines.exact_bench.audit_h2o_cross_code_reference import (
    _comparison,
    _fixed_h2o_geometry_angstrom,
)


def test_fixed_h2o_geometry_has_requested_internal_coordinates() -> None:
    atoms = _fixed_h2o_geometry_angstrom(
        bond_length_angstrom=0.9578,
        bond_angle_degrees=104.49,
    )
    oxygen = np.asarray(atoms[0][1], dtype=float)
    hydrogen_1 = np.asarray(atoms[1][1], dtype=float)
    hydrogen_2 = np.asarray(atoms[2][1], dtype=float)
    vector_1 = hydrogen_1 - oxygen
    vector_2 = hydrogen_2 - oxygen

    np.testing.assert_allclose(
        np.linalg.norm(vector_1),
        0.9578,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(np.linalg.norm(vector_2), 0.9578, atol=1.0e-12)
    cosine = float(np.dot(vector_1, vector_2) / (0.9578**2))
    angle = float(np.degrees(np.arccos(cosine)))
    np.testing.assert_allclose(angle, 104.49, atol=1.0e-12)


def test_comparison_reports_signed_and_maximum_absolute_deltas() -> None:
    observed = {
        "scf_energy_hartree": -2.1,
        "frequencies_cm1": [10.0, 20.5, 29.0],
    }
    reference = {
        "scf_energy_hartree": -2.0,
        "frequencies_cm1": [10.5, 20.0, 30.0],
    }

    result = _comparison(observed, reference)

    np.testing.assert_allclose(result["energy_delta_hartree"], -0.1)
    np.testing.assert_allclose(result["energy_abs_delta_hartree"], 0.1)
    np.testing.assert_allclose(result["frequency_deltas_cm1"], [-0.5, 0.5, -1.0])
    np.testing.assert_allclose(result["frequency_max_abs_delta_cm1"], 1.0)

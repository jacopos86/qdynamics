from __future__ import annotations

import numpy as np
import pytest

from pipelines.scaffold.ground_space_fidelity import (
    GroundSpaceFidelityError,
    GroundSpaceTolerance,
    evaluate_ground_space_fidelity,
    projector_fidelity,
)


def _all_basis(dimension: int) -> tuple[int, ...]:
    return tuple(range(int(dimension)))


def _evaluate(
    hamiltonian: np.ndarray,
    state: np.ndarray,
    *,
    fixed: tuple[int, ...] | None = None,
    legal: tuple[int, ...] | None = None,
    working_cutoff: int = 3,
    reference_cutoff: int = 3,
) -> dict:
    dimension = int(np.asarray(hamiltonian).shape[0])
    return evaluate_ground_space_fidelity(
        hamiltonian=hamiltonian,
        variational_state=state,
        working_cutoff=working_cutoff,
        reference_cutoff=reference_cutoff,
        fixed_sector_basis_indices=(
            _all_basis(dimension) if fixed is None else fixed
        ),
        legal_binary_basis_indices=(
            _all_basis(dimension) if legal is None else legal
        ),
        fixed_sector_label="unit_test_fixed_sector",
        legal_binary_basis_label="unit_test_legal_binary_codewords",
    )


def test_unique_ground_state_records_gap_and_reporting_only_contract() -> None:
    result = _evaluate(
        np.diag([0.0, 1.25]).astype(complex),
        np.array([1.0, 0.0], dtype=complex),
    )

    assert result["status"] == "ok"
    assert result["reference_convention"] == "unique_ground_state_vector"
    assert result["ground_space_multiplicity"] == 1
    assert result["ground_space_unique_proved"] is True
    assert result["ground_space_gap"] == pytest.approx(1.25)
    assert result["fidelity"] == pytest.approx(1.0)
    assert result["infidelity"] == pytest.approx(0.0)
    assert result["usage_scope"] == "post_run_reporting_only"
    assert result["controller_decision_eligible"] is False
    assert result["optimizer_input_eligible"] is False
    assert result["stopping_input_eligible"] is False
    assert result["s_alg_charged"] is False
    assert len(result["physical_basis_sha256"]) == 64
    assert len(result["projector_sha256"]) == 64


def test_degenerate_ground_space_uses_projector_fidelity() -> None:
    state = np.array([1.0, 1.0j, 0.0], dtype=complex) / np.sqrt(2.0)
    result = _evaluate(np.diag([0.0, 0.0, 2.0]).astype(complex), state)

    assert result["reference_convention"] == "degenerate_ground_space_projector"
    assert result["ground_space_multiplicity"] == 2
    assert result["ground_space_unique_proved"] is False
    assert result["ground_space_gap"] == pytest.approx(2.0)
    assert result["fidelity"] == pytest.approx(1.0)


def test_projector_fidelity_is_invariant_to_ground_basis_rotation_and_global_phase() -> None:
    canonical = np.eye(3, dtype=complex)[:, :2]
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=complex,
    )
    rotated = canonical @ rotation
    state = np.exp(0.61j) * np.array([0.5, 0.5j, np.sqrt(0.5)], dtype=complex)

    expected = 0.5
    assert projector_fidelity(state, canonical) == pytest.approx(expected)
    assert projector_fidelity(np.exp(-1.7j) * state, rotated) == pytest.approx(
        expected
    )
    assert projector_fidelity(4.3 * state, canonical) == pytest.approx(expected)


def test_physical_basis_excludes_lower_unphysical_state() -> None:
    # The full-register minimum is index 3, but index 3 is outside the legal
    # binary basis.  The physical fixed-sector minimum is index 1.
    hamiltonian = np.diag([3.0, 0.0, 2.0, -5.0]).astype(complex)
    state = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    result = _evaluate(
        hamiltonian,
        state,
        fixed=(1, 3),
        legal=(0, 1, 2),
    )

    assert result["physical_basis_count"] == 1
    assert result["ground_energy"] == pytest.approx(0.0)
    assert result["fidelity"] == pytest.approx(1.0)


def test_near_degenerate_cluster_uses_explicit_tolerance() -> None:
    tolerance = GroundSpaceTolerance(absolute=1.0e-8, relative=0.0)
    result = evaluate_ground_space_fidelity(
        hamiltonian=np.diag([0.0, 5.0e-9, 1.0]).astype(complex),
        variational_state=np.array([0.0, 1.0, 0.0], dtype=complex),
        working_cutoff=7,
        reference_cutoff=7,
        fixed_sector_basis_indices=(0, 1, 2),
        legal_binary_basis_indices=(0, 1, 2),
        fixed_sector_label="fixed",
        legal_binary_basis_label="legal",
        tolerance=tolerance,
    )

    assert result["ground_space_multiplicity"] == 2
    assert result["fidelity"] == pytest.approx(1.0)
    assert result["degeneracy_tolerance"]["resolved_threshold"] == pytest.approx(
        1.0e-8
    )


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"working_cutoff": 3, "reference_cutoff": 7}, "cutoff_mismatch"),
        (
            {"fixed_sector_basis_indices": ()},
            "empty_fixed_sector_basis_indices",
        ),
        (
            {"legal_binary_basis_indices": ()},
            "empty_legal_binary_basis_indices",
        ),
        ({"fixed_sector_label": ""}, "missing_fixed_sector_label"),
        (
            {"legal_binary_basis_label": ""},
            "missing_legal_binary_basis_label",
        ),
    ],
)
def test_required_physical_metadata_fails_closed(kwargs: dict, code: str) -> None:
    inputs = {
        "hamiltonian": np.diag([0.0, 1.0]).astype(complex),
        "variational_state": np.array([1.0, 0.0], dtype=complex),
        "working_cutoff": 3,
        "reference_cutoff": 3,
        "fixed_sector_basis_indices": (0, 1),
        "legal_binary_basis_indices": (0, 1),
        "fixed_sector_label": "fixed",
        "legal_binary_basis_label": "legal",
    }
    inputs.update(kwargs)

    with pytest.raises(GroundSpaceFidelityError) as excinfo:
        evaluate_ground_space_fidelity(**inputs)

    assert excinfo.value.code == code


def test_variational_sector_or_padding_leakage_fails_closed() -> None:
    with pytest.raises(GroundSpaceFidelityError) as excinfo:
        _evaluate(
            np.diag([0.0, 1.0]).astype(complex),
            np.array([np.sqrt(0.9), np.sqrt(0.1)], dtype=complex),
            fixed=(0,),
            legal=(0, 1),
        )

    assert excinfo.value.code == "variational_state_outside_physical_basis"


def test_tolerated_leakage_is_not_renormalized_out_of_reported_fidelity() -> None:
    leakage = 5.0e-11
    result = evaluate_ground_space_fidelity(
        hamiltonian=np.diag([0.0, 1.0]).astype(complex),
        variational_state=np.array(
            [np.sqrt(1.0 - leakage), np.sqrt(leakage)], dtype=complex
        ),
        working_cutoff=3,
        reference_cutoff=3,
        fixed_sector_basis_indices=(0,),
        legal_binary_basis_indices=(0, 1),
        fixed_sector_label="fixed",
        legal_binary_basis_label="legal",
        state_leakage_tolerance=1.0e-10,
    )

    assert result["variational_state_leakage_probability"] == pytest.approx(
        leakage, abs=1.0e-14
    )
    assert result["fidelity"] == pytest.approx(1.0 - leakage, abs=1.0e-14)


def test_nonhermitian_hamiltonian_fails_closed() -> None:
    with pytest.raises(GroundSpaceFidelityError) as excinfo:
        _evaluate(
            np.array([[0.0, 1.0], [0.0, 1.0]], dtype=complex),
            np.array([1.0, 0.0], dtype=complex),
        )

    assert excinfo.value.code == "hamiltonian_not_hermitian"


def test_hamiltonian_physical_subspace_coupling_fails_closed() -> None:
    with pytest.raises(GroundSpaceFidelityError) as excinfo:
        _evaluate(
            np.array([[0.0, 0.2], [0.2, 1.0]], dtype=complex),
            np.array([1.0, 0.0], dtype=complex),
            fixed=(0,),
            legal=(0, 1),
        )

    assert (
        excinfo.value.code
        == "hamiltonian_does_not_preserve_physical_basis"
    )

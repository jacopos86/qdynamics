from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.excited_dynamics.__main__ import main
from pipelines.excited_dynamics.schemas import (
    EXCITED_STATE_SEED_SCHEMA_VERSION,
    QSE_RESULT_SCHEMA_VERSION,
    ValidationError,
    build_excited_state_seed_manifest,
    validate_excited_state_seed_manifest,
    validate_qse_result_manifest,
)


def _coeffs(n: int) -> list[dict[str, float | int]]:
    return [
        {"basis_index": i, "re": 1.0 if i == 0 else 0.0, "im": 0.0}
        for i in range(n)
    ]


def _minimal_qse_manifest(*, basis_size: int = 2) -> dict:
    return {
        "schema_version": QSE_RESULT_SCHEMA_VERSION,
        "pipeline": "qse_spectra",
        "generated_utc": "2026-05-04T00:00:00+00:00",
        "backend": "ideal_statevector",
        "uses_qiskit": False,
        "settings": {"problem": "hh", "n_ph_max": 1},
        "operator_basis": [
            {"basis_index": i, "name": f"b{i}", "kind": "pauli_string", "pauli_exyz": "e" if i == 0 else "x"}
            for i in range(basis_size)
        ],
        "diagnostics": {
            "num_qubits": 1,
            "hilbert_dim": 2,
            "basis_size": basis_size,
            "retained_rank": basis_size,
            "discarded_rank": 0,
            "overlap_condition_estimate": 1.0,
            "overlap_pruning_threshold": 1.0e-12,
        },
        "overlap_spectrum": {"raw_eigenvalues_desc": [1.0] * basis_size},
        "eigenvalues": [
            {
                "state_index": 0,
                "energy": -1.0,
                "energy_relative_to_lowest_qse": 0.0,
                "generalized_residual_norm": 0.0,
                "basis_coefficients": _coeffs(basis_size),
            },
            {
                "state_index": 1,
                "energy": 1.0,
                "energy_relative_to_lowest_qse": 2.0,
                "generalized_residual_norm": 1.0e-14,
                "basis_coefficients": _coeffs(basis_size),
            },
        ],
        "matrices": {"included": False},
    }


def test_validate_qse_manifest_accepts_minimal_qse_result() -> None:
    summary = validate_qse_result_manifest(_minimal_qse_manifest())

    assert summary.schema_version == QSE_RESULT_SCHEMA_VERSION
    assert summary.pipeline == "qse_spectra"
    assert summary.backend == "ideal_statevector"
    assert summary.uses_qiskit is False
    assert summary.num_qubits == 1
    assert summary.basis_size == 2
    assert summary.retained_rank == 2
    assert summary.eigenvalue_count == 2


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda p: p.update({"schema_version": "bad"}), "schema_version"),
        (lambda p: p.update({"backend": "exact_ed"}), "backend"),
        (lambda p: p.update({"uses_qiskit": True}), "qiskit"),
        (lambda p: p["diagnostics"].update({"retained_rank": 3}), "retained_rank"),
        (lambda p: p["operator_basis"].pop(), "operator_basis length"),
        (lambda p: p["eigenvalues"][1].update({"energy": float("nan")}), "finite"),
        (lambda p: p["eigenvalues"][1]["basis_coefficients"].pop(), "basis_coefficients length"),
    ],
)
def test_validate_qse_manifest_rejects_bad_payloads(mutation, match: str) -> None:
    payload = _minimal_qse_manifest()
    mutation(payload)

    with pytest.raises(ValidationError, match=match):
        validate_qse_result_manifest(payload)


def test_build_excited_state_seed_rejects_ground_state_by_default() -> None:
    with pytest.raises(ValidationError, match="state_index=0"):
        build_excited_state_seed_manifest(_minimal_qse_manifest(), state_index=0)


def test_q0_projected_root_zero_is_accepted_as_lowest_excited_candidate() -> None:
    payload = _minimal_qse_manifest()
    policy = {
        "reference_projection": "q0",
        "basis_vector_normalization": "raw_projected",
        "sector_projection": "identity",
        "sector_label": "unit_test_sector",
    }
    payload["settings"]["basis_vector_policy"] = dict(policy)
    payload["diagnostics"]["basis_vector_policy"] = dict(policy)
    payload["eigenvalues"][0]["energy_relative_to_reference"] = 0.75

    seed = build_excited_state_seed_manifest(payload, state_index=0)

    assert seed["qse_ritz"]["root_role"] == "lowest_orthogonal_ritz_root"
    assert seed["qse_ritz"]["excitation_energy"] == pytest.approx(0.75)
    assert seed["qse_ritz"]["excitation_energy_reference"] == "prepared_reference_state"
    assert "q0_root_zero_is_lowest_orthogonal_ritz_root_not_ground_state" in seed["warnings"]


def test_build_and_validate_excited_state_seed_manifest() -> None:
    seed = build_excited_state_seed_manifest(
        _minimal_qse_manifest(),
        state_index=1,
        source_qse_path="qse.json",
        source_qse_sha256="abc123",
    )

    summary = validate_excited_state_seed_manifest(seed)

    assert seed["schema_version"] == EXCITED_STATE_SEED_SCHEMA_VERSION
    assert seed["state_preparation_mode"] == "qse_ritz_statevector_diagnostic"
    assert seed["qpu_faithful_preparation"] is False
    assert seed["diagnostic_exact_assisted"] is True
    assert seed["controller_exact_input_mode"] == "off"
    assert seed["controller_boundary"]["controller_usable"] is False
    assert seed["controller_boundary"]["requires_scaffold_fit"] is True
    assert seed["controller_boundary"]["feeds_controller_decisions"] is False
    assert seed["basis"]["basis_vector_normalization"] == "qse_core_normalized_B_i_psi"
    assert "basis_vector_policy" not in seed["basis"]
    assert seed["qse_ritz"]["state_index"] == 1
    assert seed["qse_ritz"]["energy"] == pytest.approx(1.0)
    assert seed["qse_ritz"]["energy_relative_to_lowest_qse"] == pytest.approx(2.0)
    assert len(seed["qse_ritz"]["basis_coefficients"]) == 2
    assert summary.state_index == 1
    assert summary.controller_usable is False
    assert summary.qpu_faithful_preparation is False
    assert summary.promotion_status == "diagnostic"


def test_excited_state_seed_accepts_paper_iii_policy_metadata_as_diagnostic_only() -> None:
    payload = _minimal_qse_manifest()
    policy = {
        "reference_projection": "q0",
        "basis_vector_normalization": "raw_projected",
        "sector_projection": "identity",
        "sector_label": "unit_test_sector",
    }
    payload["settings"]["basis_vector_policy"] = dict(policy)
    payload["diagnostics"]["basis_vector_policy"] = dict(policy)
    payload["diagnostics"]["basis_vector_diagnostics"] = [
        {
            "basis_index": 0,
            "raw_action_norm": 1.0,
            "projected_norm": 0.0,
            "matrix_vector_norm": 0.0,
            "projected_out_by_q0": True,
        }
    ]
    payload["static_record_selection"] = {
        "schema_version": "qse_static_record_selection_v1",
        "controller_boundary": {"feeds_controller_decisions": False},
        "summary": {"input_basis_size": 3, "selected_basis_size": 2},
        "selected_mapping": [{"original_basis_index": 1, "selected_basis_index": 0}],
    }
    payload["transition_observables"] = [
        {
            "name": "dipole",
            "kind": "pauli_string",
            "transition_strengths": [1.0, 0.0],
        }
    ]
    payload["spectral_functions"] = {
        "schema_version": "qse_spectral_functions_v1",
        "controller_boundary": {"feeds_controller_decisions": False},
        "grid": {"values": [0.0, 1.0], "omega_min": 0.0, "omega_max": 1.0, "num_points": 2},
        "observables": [{"name": "dipole", "values": [0.0, 1.0]}],
    }
    payload["spectral_window_metrics"] = {
        "schema_version": "qse_spectral_window_metrics_v1",
        "controller_boundary": {"feeds_controller_decisions": False},
        "observables": [{"name": "dipole", "window_metrics": []}],
    }
    payload["cutoff_boundary_diagnostics"] = {
        "schema_version": "qse_cutoff_boundary_diagnostics_v1",
        "controller_boundary": {"feeds_controller_decisions": False},
        "roots": [{"state_index": 1, "ell_cut": 0.0}],
    }

    summary = validate_qse_result_manifest(payload)
    seed = build_excited_state_seed_manifest(payload, state_index=1)
    seed_summary = validate_excited_state_seed_manifest(seed)

    assert summary.basis_size == 2
    assert seed["basis"]["basis_vector_normalization"] == "raw_projected"
    assert seed["basis"]["basis_vector_policy"] == policy
    assert seed["qpu_faithful_preparation"] is False
    assert seed["diagnostic_exact_assisted"] is True
    assert seed["controller_boundary"]["controller_usable"] is False
    assert seed["controller_boundary"]["feeds_controller_decisions"] is False
    assert seed_summary.controller_usable is False
    visibility = seed["visibility"]
    diagnostic_refs = [
        "static_record_selection",
        "transition_observables",
        "spectral_functions",
        "spectral_window_metrics",
        "cutoff_boundary_diagnostics",
    ]
    for ref in diagnostic_refs:
        assert ref not in seed
        assert ref not in seed.get("controller_boundary", {})
        assert ref not in visibility["controller_visible_payload_refs"]
        assert ref in visibility["diagnostic_only_payload_refs"]
        assert ref in visibility["forbidden_to_controller_refs"]


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda p: p.update({"qpu_faithful_preparation": True}), "qpu_faithful"),
        (lambda p: p.update({"diagnostic_exact_assisted": False}), "diagnostic_exact_assisted"),
        (lambda p: p.update({"controller_exact_input_mode": "benchmark_exact"}), "controller_exact_input_mode"),
        (lambda p: p["controller_boundary"].update({"controller_usable": True}), "controller_usable"),
        (lambda p: p["controller_boundary"].update({"decision_path_allowed": True}), "decision_path_allowed"),
        (lambda p: p["controller_boundary"].update({"feeds_controller_decisions": True}), "feeds_controller_decisions"),
    ],
)
def test_validate_excited_state_seed_rejects_overclaiming(mutation, match: str) -> None:
    seed = build_excited_state_seed_manifest(_minimal_qse_manifest(), state_index=1)
    mutation(seed)

    with pytest.raises(ValidationError, match=match):
        validate_excited_state_seed_manifest(seed)


def test_cli_writes_excited_state_seed_from_qse_manifest(tmp_path: Path) -> None:
    qse_path = tmp_path / "qse.json"
    out_path = tmp_path / "nested" / "seed.json"
    qse_path.write_text(json.dumps(_minimal_qse_manifest()), encoding="utf-8")

    assert main(["--qse-result-json", str(qse_path), "--state-index", "1", "--output-json", str(out_path)]) == 0

    seed = json.loads(out_path.read_text(encoding="utf-8"))
    assert seed["schema_version"] == EXCITED_STATE_SEED_SCHEMA_VERSION
    assert seed["source"]["source_qse_path"] == str(qse_path)
    assert seed["source"]["source_qse_sha256"]
    assert seed["qse_ritz"]["state_index"] == 1
    assert validate_excited_state_seed_manifest(seed).state_index == 1

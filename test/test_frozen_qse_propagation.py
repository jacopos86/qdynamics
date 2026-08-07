from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.excited_dynamics.frozen_qse_propagation import (
    FROZEN_QSE_PROPAGATION_SCHEMA_VERSION,
    FrozenQSEPropagationConfig,
    FrozenQSEPropagationError,
    main,
    run_frozen_qse_propagation,
)
from pipelines.excited_dynamics.schemas import QSE_RESULT_SCHEMA_VERSION


def _cj(value: complex | float) -> dict[str, float]:
    z = complex(value)
    return {"re": float(z.real), "im": float(z.imag)}


def _matrix_json(matrix: np.ndarray) -> list[list[dict[str, float]]]:
    array = np.asarray(matrix, dtype=complex)
    return [[_cj(array[row, col]) for col in range(array.shape[1])] for row in range(array.shape[0])]


def _coeffs(vector: np.ndarray) -> list[dict[str, float | int]]:
    flat = np.asarray(vector, dtype=complex).reshape(-1)
    return [{"basis_index": int(idx), **_cj(value)} for idx, value in enumerate(flat)]


def _minimal_qse_manifest(
    *,
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    initial_coefficients: np.ndarray,
    matrices_included: bool = True,
) -> dict:
    overlap = np.asarray(overlap, dtype=complex)
    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    basis_size = int(overlap.shape[0])
    matrices = {"included": bool(matrices_included)}
    if matrices_included:
        matrices.update(
            {
                "overlap": _matrix_json(overlap),
                "hamiltonian": _matrix_json(hamiltonian),
            }
        )
    return {
        "schema_version": QSE_RESULT_SCHEMA_VERSION,
        "pipeline": "qse_spectra",
        "generated_utc": "2026-05-16T00:00:00Z",
        "backend": "ideal_statevector",
        "uses_qiskit": False,
        "settings": {
            "overlap_negative_absolute_tolerance": 1.0e-12,
            "overlap_negative_relative_tolerance": 1.0e-9,
            "hermitian_absolute_tolerance": 1.0e-10,
            "hermitian_relative_tolerance": 1.0e-8,
        },
        "operator_basis": [
            {"basis_index": idx, "name": f"b{idx}", "kind": "pauli_string", "pauli_exyz": "e"}
            for idx in range(basis_size)
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
        "overlap_spectrum": [
            {"index": idx, "raw_value": 1.0, "clamped_value": 1.0, "retained": True}
            for idx in range(basis_size)
        ],
        "eigenvalues": [
            {
                "state_index": 0,
                "energy": 0.0,
                "energy_relative_to_lowest_qse": 0.0,
                "generalized_residual_norm": 0.0,
                "basis_coefficients": _coeffs(initial_coefficients),
            }
        ],
        "matrices": matrices,
    }


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "qse.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _coeff_vector(row: dict, key: str) -> np.ndarray:
    return np.asarray([complex(item["re"], item["im"]) for item in row[key]], dtype=complex)


def test_one_dimensional_phase_and_qse_norm_preservation(tmp_path: Path) -> None:
    omega = 1.75
    qse_path = _write_manifest(
        tmp_path,
        _minimal_qse_manifest(
            overlap=np.asarray([[1.0]], dtype=complex),
            hamiltonian=np.asarray([[omega]], dtype=complex),
            initial_coefficients=np.asarray([1.0 + 0.0j]),
        ),
    )

    artifact = run_frozen_qse_propagation(
        FrozenQSEPropagationConfig(qse_manifest_json=qse_path, initial_root_index=0, t_final=0.4, num_steps=4)
    )

    assert artifact["schema_version"] == FROZEN_QSE_PROPAGATION_SCHEMA_VERSION
    assert artifact["metrics"]["trajectory_rows"] == 5
    assert artifact["metrics"]["max_qse_norm_error"] == pytest.approx(0.0, abs=1.0e-12)
    final_c = _coeff_vector(artifact["trajectory"][-1], "qse_basis_coefficients")
    assert final_c[0].real == pytest.approx(math.cos(-omega * 0.4), abs=1.0e-12)
    assert final_c[0].imag == pytest.approx(math.sin(-omega * 0.4), abs=1.0e-12)
    assert artifact["trajectory"][-1]["qse_norm"] == pytest.approx(1.0, abs=1.0e-12)


def test_two_level_static_hamiltonian_rabi_populations(tmp_path: Path) -> None:
    t_final = 0.3
    qse_path = _write_manifest(
        tmp_path,
        _minimal_qse_manifest(
            overlap=np.eye(2, dtype=complex),
            hamiltonian=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
            initial_coefficients=np.asarray([1.0 + 0.0j, 0.0 + 0.0j]),
        ),
    )

    artifact = run_frozen_qse_propagation(
        FrozenQSEPropagationConfig(qse_manifest_json=qse_path, initial_root_index=0, t_final=t_final, num_steps=6)
    )

    final_c = _coeff_vector(artifact["trajectory"][-1], "qse_basis_coefficients")
    populations = np.abs(final_c) ** 2
    assert populations[0] == pytest.approx(math.cos(t_final) ** 2, abs=1.0e-12)
    assert populations[1] == pytest.approx(math.sin(t_final) ** 2, abs=1.0e-12)
    assert float(np.sum(populations)) == pytest.approx(1.0, abs=1.0e-12)
    assert artifact["metrics"]["max_qse_norm_error"] == pytest.approx(0.0, abs=1.0e-12)


def test_nonorthogonal_positive_overlap_retained_support_preserves_norm(tmp_path: Path) -> None:
    overlap = np.asarray([[1.0, 0.2], [0.2, 1.0]], dtype=complex)
    hamiltonian = np.asarray([[0.1, 0.05], [0.05, 0.7]], dtype=complex)
    qse_path = _write_manifest(
        tmp_path,
        _minimal_qse_manifest(
            overlap=overlap,
            hamiltonian=hamiltonian,
            initial_coefficients=np.asarray([1.0 + 0.0j, 0.0 + 0.0j]),
        ),
    )

    artifact = run_frozen_qse_propagation(
        FrozenQSEPropagationConfig(qse_manifest_json=qse_path, initial_root_index=0, t_final=0.5, num_steps=5)
    )

    assert artifact["qse_support"]["retained_rank"] == 2
    assert artifact["qse_support"]["overlap_condition_estimate"] == pytest.approx(1.5, abs=1.0e-12)
    assert artifact["initial_condition"]["qse_norm_after_retained_projection"] == pytest.approx(1.0, abs=1.0e-12)
    assert artifact["metrics"]["max_qse_norm_error"] == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda p: p.update({"matrices": {"included": False}}), "must include matrices"),
        (
            lambda p: p["matrices"].update(
                {"hamiltonian": _matrix_json(np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex))}
            ),
            "non-Hermitian",
        ),
        (
            lambda p: p["matrices"].update(
                {"overlap": _matrix_json(np.asarray([[1.0, 0.0], [0.0, -1.0e-4]], dtype=complex))}
            ),
            "negative eigenvalue",
        ),
    ],
)
def test_invalid_qse_matrix_inputs_fail_closed(tmp_path: Path, mutation, match: str) -> None:
    payload = _minimal_qse_manifest(
        overlap=np.eye(2, dtype=complex),
        hamiltonian=np.eye(2, dtype=complex),
        initial_coefficients=np.asarray([1.0 + 0.0j, 0.0 + 0.0j]),
    )
    mutation(payload)
    qse_path = _write_manifest(tmp_path, payload)

    with pytest.raises(FrozenQSEPropagationError, match=match):
        run_frozen_qse_propagation(
            FrozenQSEPropagationConfig(qse_manifest_json=qse_path, initial_root_index=0, t_final=0.1, num_steps=1)
        )


def test_zero_retained_overlap_support_fails_closed(tmp_path: Path) -> None:
    qse_path = _write_manifest(
        tmp_path,
        _minimal_qse_manifest(
            overlap=np.eye(2, dtype=complex),
            hamiltonian=np.eye(2, dtype=complex),
            initial_coefficients=np.asarray([1.0 + 0.0j, 0.0 + 0.0j]),
        ),
    )

    with pytest.raises(FrozenQSEPropagationError, match="retained rank is zero"):
        run_frozen_qse_propagation(
            FrozenQSEPropagationConfig(
                qse_manifest_json=qse_path,
                initial_root_index=0,
                t_final=0.1,
                num_steps=1,
                support_cutoff=2.0,
            )
        )


def test_output_boundary_flags_and_no_raw_physical_vectors(tmp_path: Path) -> None:
    qse_path = _write_manifest(
        tmp_path,
        _minimal_qse_manifest(
            overlap=np.eye(1, dtype=complex),
            hamiltonian=np.asarray([[0.0]], dtype=complex),
            initial_coefficients=np.asarray([1.0 + 0.0j]),
        ),
    )

    artifact = run_frozen_qse_propagation(
        FrozenQSEPropagationConfig(qse_manifest_json=qse_path, initial_root_index=0, t_final=0.0, num_steps=1)
    )
    boundary = artifact["controller_boundary"]

    assert artifact["controller_usable"] is False
    assert artifact["feeds_controller_decisions"] is False
    assert artifact["exact_or_ed_reference_used"] is False
    assert boundary["controller_usable"] is False
    assert boundary["feeds_controller_decisions"] is False
    assert boundary["decision_path_allowed"] is False
    assert artifact["visibility"]["controller_visible_payload_refs"] == []

    payload_text = json.dumps(artifact, sort_keys=True)
    for forbidden in ("amplitudes_qn_to_q0", "statevector", "basis_matrix_vectors", "raw_physical_state"):
        assert forbidden not in payload_text


def test_cli_writes_frozen_qse_propagation_json(tmp_path: Path) -> None:
    qse_path = _write_manifest(
        tmp_path,
        _minimal_qse_manifest(
            overlap=np.eye(1, dtype=complex),
            hamiltonian=np.asarray([[0.25]], dtype=complex),
            initial_coefficients=np.asarray([1.0 + 0.0j]),
        ),
    )
    out_path = tmp_path / "nested" / "frozen.json"

    assert (
        main(
            [
                "--qse-manifest-json",
                str(qse_path),
                "--initial-root-index",
                "0",
                "--t-final",
                "0.2",
                "--num-steps",
                "2",
                "--output-json",
                str(out_path),
            ]
        )
        == 0
    )

    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == FROZEN_QSE_PROPAGATION_SCHEMA_VERSION
    assert artifact["source"]["source_qse_sha256"]
    assert artifact["metrics"]["trajectory_rows"] == 3
    assert artifact["metrics"]["retained_rank"] == 1


def test_frozen_qse_module_has_no_forbidden_imports() -> None:
    module = REPO_ROOT / "pipelines" / "excited_dynamics" / "frozen_qse_propagation.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if int(node.level) > 0:
                targets.add("<relative>")
            if node.module is not None:
                targets.add(node.module)

    forbidden_roots = {
        "qiskit",
        "pipelines.time_dynamics",
        "pipelines.hardcoded",
        "pipelines.shell",
        "pipelines.exact_bench",
    }
    forbidden_fragments = ("realtime", "controller", "chtc", "runner")
    offenders: list[str] = []
    for target in targets:
        if target == "<relative>":
            offenders.append(target)
        for root in forbidden_roots:
            if target == root or target.startswith(root + "."):
                offenders.append(target)
        if any(fragment in target for fragment in forbidden_fragments):
            offenders.append(target)

    assert offenders == []

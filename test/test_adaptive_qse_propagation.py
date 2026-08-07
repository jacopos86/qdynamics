from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.excited_dynamics.adaptive_qse_propagation import (
    ADAPTIVE_QSE_PROPAGATION_SCHEMA_VERSION,
    AdaptiveQSEPropagationConfig,
    AdaptiveQSEPropagationError,
    main,
    run_adaptive_qse_propagation,
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


def _coupled_manifest(*, size: int = 3, couplings: tuple[float, ...] = (0.4, 0.1)) -> dict:
    hamiltonian = np.zeros((size, size), dtype=complex)
    for offset, coupling in enumerate(couplings, start=1):
        if offset >= size:
            break
        hamiltonian[0, offset] = coupling
        hamiltonian[offset, 0] = coupling
    return _minimal_qse_manifest(
        overlap=np.eye(size, dtype=complex),
        hamiltonian=hamiltonian,
        initial_coefficients=np.asarray([1.0 + 0.0j, *([0.0 + 0.0j] * (size - 1))]),
    )


def _run_config(
    qse_path: Path,
    *,
    initial_active_indices=(0,),
    escape_threshold: float = 0.05,
    max_add_per_checkpoint: int = 1,
    max_active_records: int = 3,
    t_final: float = 0.2,
    num_steps: int = 2,
    checkpoint_every_steps: int = 1,
    support_cutoff: float = 1.0e-12,
) -> AdaptiveQSEPropagationConfig:
    return AdaptiveQSEPropagationConfig(
        qse_manifest_json=qse_path,
        initial_active_indices=initial_active_indices,
        initial_root_index=0,
        t_final=t_final,
        num_steps=num_steps,
        checkpoint_every_steps=checkpoint_every_steps,
        support_cutoff=support_cutoff,
        escape_threshold=escape_threshold,
        max_add_per_checkpoint=max_add_per_checkpoint,
        max_active_records=max_active_records,
    )


def test_candidate_score_selects_coupled_inactive_basis_vector(tmp_path: Path) -> None:
    qse_path = _write_manifest(tmp_path, _coupled_manifest(size=3, couplings=(0.4, 0.1)))

    artifact = run_adaptive_qse_propagation(_run_config(qse_path))

    assert artifact["schema_version"] == ADAPTIVE_QSE_PROPAGATION_SCHEMA_VERSION
    assert artifact["metrics"]["trajectory_rows"] == 3
    assert artifact["metrics"]["adaptation_event_count"] == 1
    event = artifact["adaptation_events"][0]
    assert event["added_indices"] == [1]
    assert event["candidate_score_summary"][0]["basis_index"] == 1
    assert event["candidate_score_summary"][0]["score"] == pytest.approx(0.4, abs=1.0e-12)
    assert artifact["metrics"]["final_active_record_count"] == 2


def test_within_tolerance_hermitian_noise_uses_canonical_scoring_matrices(tmp_path: Path) -> None:
    overlap = np.eye(3, dtype=complex)
    overlap[1, 1] = 1.0 + 2.0e-9j
    hamiltonian = np.asarray(
        [[0.0, 0.4 + 2.0e-9j, 0.1], [0.4, 0.0, 0.0], [0.1, 0.0, 0.0]],
        dtype=complex,
    )
    qse_path = _write_manifest(
        tmp_path,
        _minimal_qse_manifest(
            overlap=overlap,
            hamiltonian=hamiltonian,
            initial_coefficients=np.asarray([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        ),
    )

    artifact = run_adaptive_qse_propagation(_run_config(qse_path))

    assert artifact["metrics"]["adaptation_event_count"] == 1
    assert artifact["adaptation_events"][0]["added_indices"] == [1]
    assert artifact["metrics"]["max_qse_norm_error"] == pytest.approx(0.0, abs=1.0e-12)


def test_no_adaptation_when_escape_threshold_exceeds_scores(tmp_path: Path) -> None:
    qse_path = _write_manifest(tmp_path, _coupled_manifest(size=3, couplings=(0.4, 0.1)))

    artifact = run_adaptive_qse_propagation(_run_config(qse_path, escape_threshold=1.0))

    assert artifact["adaptation_events"] == []
    assert artifact["metrics"]["adaptation_event_count"] == 0
    assert artifact["metrics"]["final_active_record_count"] == 1
    assert artifact["metrics"]["max_escape_score"] == pytest.approx(0.4, abs=1.0e-12)


def test_growth_caps_are_enforced_deterministically(tmp_path: Path) -> None:
    qse_path = _write_manifest(tmp_path, _coupled_manifest(size=4, couplings=(0.5, 0.4, 0.3)))

    artifact = run_adaptive_qse_propagation(
        _run_config(
            qse_path,
            max_add_per_checkpoint=2,
            max_active_records=2,
            t_final=0.4,
            num_steps=4,
        )
    )

    assert artifact["metrics"]["adaptation_event_count"] == 1
    event = artifact["adaptation_events"][0]
    assert event["remaining_capacity_before"] == 1
    assert event["added_indices"] == [1]
    assert event["active_indices_after"] == [0, 1]
    assert artifact["metrics"]["final_active_record_count"] == 2


def test_qse_norm_preserved_across_basis_growth(tmp_path: Path) -> None:
    qse_path = _write_manifest(tmp_path, _coupled_manifest(size=3, couplings=(0.4, 0.2)))

    artifact = run_adaptive_qse_propagation(
        _run_config(qse_path, t_final=0.4, num_steps=4, max_active_records=3)
    )

    assert artifact["metrics"]["adaptation_event_count"] >= 1
    assert artifact["metrics"]["max_qse_norm_error"] == pytest.approx(0.0, abs=1.0e-12)
    for row in artifact["trajectory"]:
        assert row["qse_norm"] == pytest.approx(1.0, abs=1.0e-12)
        assert row["retained_support_norm"] == pytest.approx(1.0, abs=1.0e-12)
    for event in artifact["adaptation_events"]:
        assert event["remap"]["qse_norm_after_rescale"] == pytest.approx(event["remap"]["target_qse_norm"], abs=1.0e-12)


def test_zero_retained_active_overlap_support_fails_closed(tmp_path: Path) -> None:
    qse_path = _write_manifest(
        tmp_path,
        _minimal_qse_manifest(
            overlap=np.asarray([[1.0e-16, 0.0], [0.0, 1.0]], dtype=complex),
            hamiltonian=np.eye(2, dtype=complex),
            initial_coefficients=np.asarray([1.0 + 0.0j, 0.0 + 0.0j]),
        ),
    )

    with pytest.raises(AdaptiveQSEPropagationError, match="retained rank is zero"):
        run_adaptive_qse_propagation(_run_config(qse_path, max_active_records=2))


def test_missing_matrices_fail_closed(tmp_path: Path) -> None:
    qse_path = _write_manifest(
        tmp_path,
        _minimal_qse_manifest(
            overlap=np.eye(2, dtype=complex),
            hamiltonian=np.eye(2, dtype=complex),
            initial_coefficients=np.asarray([1.0 + 0.0j, 0.0 + 0.0j]),
            matrices_included=False,
        ),
    )

    with pytest.raises(AdaptiveQSEPropagationError, match="must include matrices"):
        run_adaptive_qse_propagation(_run_config(qse_path, max_active_records=2))


@pytest.mark.parametrize(
    "initial_active_indices, max_active_records, match",
    [
        ([], 3, "non-empty"),
        ([0, 0], 3, "duplicate"),
        ([3], 3, "out-of-range"),
        ([0, 1], 1, "max_active_records"),
    ],
)
def test_invalid_active_indices_fail_closed(
    tmp_path: Path,
    initial_active_indices,
    max_active_records: int,
    match: str,
) -> None:
    qse_path = _write_manifest(tmp_path, _coupled_manifest(size=3, couplings=(0.4, 0.1)))

    with pytest.raises(AdaptiveQSEPropagationError, match=match):
        run_adaptive_qse_propagation(
            _run_config(
                qse_path,
                initial_active_indices=initial_active_indices,
                max_active_records=max_active_records,
            )
        )


def test_output_diagnostic_only_flags_and_no_raw_physical_vectors(tmp_path: Path) -> None:
    qse_path = _write_manifest(tmp_path, _coupled_manifest(size=3, couplings=(0.4, 0.1)))

    artifact = run_adaptive_qse_propagation(_run_config(qse_path))
    boundary = artifact["controller_boundary"]

    assert artifact["controller_usable"] is False
    assert artifact["feeds_controller_decisions"] is False
    assert artifact["exact_or_ed_reference_used"] is False
    assert artifact["raw_physical_statevectors_emitted"] is False
    assert artifact["uses_qiskit"] is False
    assert boundary["controller_usable"] is False
    assert boundary["feeds_controller_decisions"] is False
    assert boundary["decision_path_allowed"] is False
    assert boundary["realtime_route_integrated"] is False
    assert artifact["visibility"]["controller_visible_payload_refs"] == []

    payload_text = json.dumps(artifact, sort_keys=True)
    for forbidden in (
        "amplitudes_qn_to_q0",
        "basis_matrix_vectors",
        "ideal_statevector",
        "raw_physical_state_payload",
        "exact_step_forecast",
        "decision_backend",
        "state_at(",
    ):
        assert forbidden not in payload_text


def test_cli_writes_adaptive_qse_propagation_json(tmp_path: Path) -> None:
    qse_path = _write_manifest(tmp_path, _coupled_manifest(size=3, couplings=(0.4, 0.1)))
    out_path = tmp_path / "nested" / "adaptive.json"

    assert (
        main(
            [
                "--qse-manifest-json",
                str(qse_path),
                "--initial-active-indices",
                "0",
                "--initial-root-index",
                "0",
                "--t-final",
                "0.2",
                "--num-steps",
                "2",
                "--checkpoint-every-steps",
                "1",
                "--support-cutoff",
                "1e-12",
                "--escape-threshold",
                "0.05",
                "--max-add-per-checkpoint",
                "1",
                "--max-active-records",
                "3",
                "--output-json",
                str(out_path),
            ]
        )
        == 0
    )

    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == ADAPTIVE_QSE_PROPAGATION_SCHEMA_VERSION
    assert artifact["source"]["source_qse_sha256"]
    assert artifact["metrics"]["trajectory_rows"] == 3
    assert artifact["metrics"]["adaptation_event_count"] == 1


def test_adaptive_qse_module_has_no_forbidden_imports() -> None:
    module = REPO_ROOT / "pipelines" / "excited_dynamics" / "adaptive_qse_propagation.py"
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

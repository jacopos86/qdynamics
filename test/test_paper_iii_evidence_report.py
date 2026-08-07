from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.excited_dynamics.io import sha256_file
from pipelines.excited_dynamics.paper_iii_evidence_report import (
    PAPER_III_EVIDENCE_REPORT_SCHEMA_VERSION,
    PaperIIIEvidenceReportConfig,
    PaperIIIEvidenceReportError,
    main,
    run_paper_iii_evidence_report,
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


def _minimal_qse_manifest(*, overlap: np.ndarray, hamiltonian: np.ndarray, initial_coefficients: np.ndarray) -> dict:
    overlap = np.asarray(overlap, dtype=complex)
    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    basis_size = int(overlap.shape[0])
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
        "matrices": {
            "included": True,
            "overlap": _matrix_json(overlap),
            "hamiltonian": _matrix_json(hamiltonian),
        },
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _boundary(*, usable: bool = False) -> dict:
    return {
        "controller_usable": usable,
        "feeds_controller_decisions": False,
        "decision_path_allowed": False,
        "post_run_diagnostic_only": True,
        "requires_scaffold_fit": True,
        "controller_exact_input_mode": "off",
        "diagnostic_exact_reference_mode": "off",
        "realtime_route_integrated": False,
    }


def _full_coeff_row(step_index: int, time: float, vector: np.ndarray) -> dict:
    return {
        "step_index": step_index,
        "time": time,
        "qse_norm": 1.0,
        "qse_energy_expectation": 0.0,
        "qse_basis_coefficients": _coeffs(vector),
    }


def _active_coeff_row(step_index: int, time: float, records: list[tuple[int, complex]]) -> dict:
    return {
        "step_index": step_index,
        "time": time,
        "active_indices": [basis_index for basis_index, _value in records],
        "active_record_count": len(records),
        "qse_norm": 1.0,
        "qse_energy_expectation": 0.0,
        "qse_basis_coefficients": [
            {"active_index": active_index, "basis_index": basis_index, **_cj(value)}
            for active_index, (basis_index, value) in enumerate(records)
        ],
    }


def _minimal_p6_output() -> dict:
    decision_row = {
        "append_attempted": False,
        "controller_exact_input_mode": "off",
        "decision_backend": "ideal_observable",
        "decision_data_flow": "ideal_observable_estimator",
        "decision_noise_mode": "ideal",
        "diagnostic_exact_reference_mode": "off",
        "prune_attempted": False,
        "strict_measurement_oracle_certified": True,
        "structure_edit_attempted": False,
        "uses_future_exact_forecast_for_decision": False,
        "uses_reference_for_decision": False,
    }
    return {
        "schema_version": "qse_promoted_mclachlan_run_v1",
        "pipeline": "qse_promoted_mclachlan_smoke",
        "generated_utc": "2026-05-16T00:00:00Z",
        "uses_qiskit": False,
        "source": {
            "loader_boundary": "runtime_payload_only",
            "controller_visible_payload_refs_used": ["runtime_payload"],
            "promoted_artifact_json": "promoted.json",
            "promoted_artifact_sha256": "abc",
            "runtime_payload_sha256": "def",
        },
        "runtime_contract": {
            "loader_boundary": "runtime_payload_only",
            "input_runtime_contract_status": "validated",
            "structure_locked": True,
            "can_structural_edit": False,
            "reference_energy_absent": True,
            "problem_key": "spin_boson",
            "runtime_parameter_count": 3,
            "logical_operator_count": 2,
            "selected_term_count": 2,
        },
        "controller_boundary": {
            "controller_usable": True,
            "source_payload_loaded": "runtime_payload_only",
            "runtime_payload_feeds_controller_decisions": True,
            "top_level_diagnostic_metadata_feeds_controller_decisions": False,
            "qse_diagnostics_forbidden_to_controller": True,
            "structural_editing_allowed": False,
            "append_allowed": False,
            "prune_allowed": False,
            "exact_target_inputs_allowed": False,
            "matches_scaffold_runtime_contract": True,
        },
        "decision_data_flow": {
            "controller_exact_input_mode": "off",
            "decision_backend": "ideal_observable",
            "decision_data_flow": "ideal_observable_estimator",
            "decision_noise_mode": "ideal",
            "diagnostic_exact_reference_mode": "off",
            "strict_measurement_oracle_certified": True,
            "uses_future_exact_forecast_for_decision": False,
            "uses_reference_for_decision": False,
        },
        "forbidden_marker_audit": {"passed": True, "hit_count": 0, "hits": []},
        "strict_decision_contract_audit": {
            "passed": True,
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "violation_count": 0,
            "violations": [],
        },
        "summary": {
            "scope_label": "P6a contract/plumbing smoke only",
            "paper_iii_science_benchmark": False,
            "strict_decision_contract_passed": True,
            "trajectory_row_count": 2,
            "step_count": 1,
            "append_count": 0,
            "prune_count": 0,
            "structure_edit_count": 0,
            "max_rhs_residual_ratio": 0.1,
            "max_state_norm_error": 0.0,
        },
        "trajectory": [dict(decision_row, time=0.0), dict(decision_row, time=0.1)],
        "mclachlan_steps": [dict(decision_row, interval_index=0)],
    }


def _source_bundle(tmp_path: Path) -> dict[str, Path]:
    qse = _minimal_qse_manifest(
        overlap=np.eye(2, dtype=complex),
        hamiltonian=np.zeros((2, 2), dtype=complex),
        initial_coefficients=np.asarray([1.0 + 0.0j, 0.0 + 0.0j]),
    )
    frozen_qse_path = _write_json(tmp_path / "p3" / "tiny_qse_manifest.json", qse)
    adaptive_qse_path = _write_json(tmp_path / "p4" / "tiny_full_pool_qse_manifest.json", qse)

    # Global phase should be removed by the overlap-aware comparison.
    phased_reference = np.asarray([0.0 + 1.0j, 0.0 + 0.0j])
    frozen_output = {
        "schema_version": "frozen_qse_propagation_v1",
        "pipeline": "frozen_qse_propagation",
        "generated_utc": "2026-05-16T00:00:00Z",
        "controller_usable": False,
        "feeds_controller_decisions": False,
        "exact_or_ed_reference_used": False,
        "uses_qiskit": False,
        "controller_boundary": _boundary(),
        "source": {
            "source_qse_path": str(frozen_qse_path),
            "source_qse_sha256": sha256_file(frozen_qse_path),
            "initial_root_index": 0,
            "qse_basis_size": 2,
        },
        "trajectory": [_full_coeff_row(0, 0.0, phased_reference), _full_coeff_row(1, 0.1, phased_reference)],
        "metrics": {"trajectory_rows": 2, "retained_rank": 2, "max_qse_norm_error": 0.0, "max_energy_drift_abs": 0.0},
    }
    frozen_output_path = _write_json(tmp_path / "p3" / "frozen_qse_propagation.json", frozen_output)
    frozen_run_manifest = {
        "schema_version": "agent_run_manifest_v1",
        "slice": "paper_iii_p3a_frozen_qse_propagation",
        "artifacts": {
            "input_qse_manifest_json": str(frozen_qse_path),
            "output_json": str(frozen_output_path),
            "command_log_md": str(tmp_path / "p3" / "command_log.md"),
        },
        "controller_boundary": _boundary(),
        "output_summary": {"feeds_controller_decisions": False},
    }
    frozen_run_manifest_path = _write_json(tmp_path / "p3" / "run_manifest.json", frozen_run_manifest)

    adaptive_output = {
        "schema_version": "adaptive_qse_propagation_v1",
        "pipeline": "adaptive_qse_propagation",
        "generated_utc": "2026-05-16T00:00:00Z",
        "controller_usable": False,
        "feeds_controller_decisions": False,
        "exact_or_ed_reference_used": False,
        "uses_qiskit": False,
        "controller_boundary": _boundary(),
        "source": {
            "source_qse_path": str(adaptive_qse_path),
            "source_qse_sha256": sha256_file(adaptive_qse_path),
            "initial_root_index": 0,
            "qse_basis_size": 2,
        },
        "trajectory": [
            _active_coeff_row(0, 0.0, [(1, 0.0 + 0.0j), (0, 0.0 + 1.0j)]),
            _active_coeff_row(1, 0.1, [(1, 0.0 + 0.0j), (0, 0.0 + 1.0j)]),
        ],
        "active_support_history": [
            {"stage": "initial", "active_indices": [1, 0], "active_record_count": 2, "step_index": 0, "time": 0.0}
        ],
        "adaptation_events": [
            {"event_index": 0, "active_indices_before": [0], "active_indices_after": [0, 1], "added_indices": [1]}
        ],
        "metrics": {
            "trajectory_rows": 2,
            "initial_active_record_count": 1,
            "final_active_record_count": 2,
            "adaptation_event_count": 1,
            "max_escape_score": 0.0,
            "max_qse_norm_error": 0.0,
            "raw_physical_statevectors_emitted": False,
            "uses_qiskit": False,
        },
    }
    adaptive_output_path = _write_json(tmp_path / "p4" / "adaptive_qse_propagation.json", adaptive_output)
    adaptive_run_manifest = {
        "schema_version": "agent_run_manifest_v1",
        "slice": "paper_iii_p4a_adaptive_qse_propagation",
        "artifacts": {
            "input_qse_manifest_json": str(adaptive_qse_path),
            "output_json": str(adaptive_output_path),
            "command_log_md": str(tmp_path / "p4" / "command_log.md"),
        },
        "controller_boundary": _boundary(),
        "output_summary": {"feeds_controller_decisions": False},
    }
    adaptive_run_manifest_path = _write_json(tmp_path / "p4" / "run_manifest.json", adaptive_run_manifest)

    p6_output_path = _write_json(tmp_path / "p6" / "qse_promoted_mclachlan_run.json", _minimal_p6_output())
    p6_run_manifest = {
        "schema_version": "paper_iii_p6a_promoted_mclachlan_smoke_v1",
        "output_json": str(p6_output_path),
        "loader_boundary": "runtime_payload_only",
        "loader_structure_locked": True,
        "loader_can_structural_edit": False,
        "forbidden_marker_hit_count": 0,
        "paper_iii_science_benchmark": False,
    }
    p6_run_manifest_path = _write_json(tmp_path / "p6" / "run_manifest.json", p6_run_manifest)

    return {
        "frozen_run_manifest": frozen_run_manifest_path,
        "adaptive_run_manifest": adaptive_run_manifest_path,
        "promoted_run_manifest": p6_run_manifest_path,
        "frozen_output": frozen_output_path,
        "adaptive_output": adaptive_output_path,
        "promoted_output": p6_output_path,
    }


def _config(tmp_path: Path, bundle: dict[str, Path]) -> PaperIIIEvidenceReportConfig:
    return PaperIIIEvidenceReportConfig(
        frozen_run_manifest=bundle["frozen_run_manifest"],
        adaptive_run_manifest=bundle["adaptive_run_manifest"],
        promoted_mclachlan_run_manifest=bundle["promoted_run_manifest"],
        output_json=tmp_path / "p7" / "paper_iii_evidence_report.json",
        output_md=tmp_path / "p7" / "paper_iii_evidence_report.md",
    )


def test_report_builds_boundary_flags_phase_distances_and_p6_classification(tmp_path: Path) -> None:
    bundle = _source_bundle(tmp_path)

    report = run_paper_iii_evidence_report(_config(tmp_path, bundle), command=["python", "-m", "test"])

    assert report["schema_version"] == PAPER_III_EVIDENCE_REPORT_SCHEMA_VERSION
    assert report["controller_usable"] is False
    assert report["feeds_controller_decisions"] is False
    assert report["reference_comparisons_feed_controller_decisions"] is False
    assert report["raw_physical_vectors_emitted"] is False
    assert report["source_artifacts_modified"] is False

    frozen = report["reference_comparisons"]["frozen_qse"]
    adaptive = report["reference_comparisons"]["adaptive_qse"]
    assert frozen["feeds_controller_decisions"] is False
    assert adaptive["feeds_controller_decisions"] is False
    assert frozen["summary"]["max_overlap_phase_distance"] == pytest.approx(0.0, abs=1.0e-12)
    assert adaptive["summary"]["max_overlap_phase_distance"] == pytest.approx(0.0, abs=1.0e-12)
    assert all(row["feeds_controller_decisions"] is False for row in frozen["rows"])
    assert all(row["feeds_controller_decisions"] is False for row in adaptive["rows"])

    assert adaptive["active_support_summary"]["final_active_record_count"] == 2
    assert adaptive["active_support_summary"]["adaptation_event_count"] == 1
    promoted = report["source_artifacts"]["promoted_mclachlan"]
    assert promoted["evidence_classification"] == "contract_plumbing"
    assert promoted["paper_iii_science_benchmark"] is False
    assert promoted["source_payload_loaded"] == "runtime_payload_only"
    assert promoted["strict_decision_contract_passed"] is True
    assert promoted["forbidden_marker_hit_count"] == 0


def test_cli_writes_json_markdown_command_log_run_manifest_and_preserves_sources(tmp_path: Path) -> None:
    bundle = _source_bundle(tmp_path)
    before_hashes = {name: sha256_file(path) for name, path in bundle.items() if name.endswith("output")}
    output_json = tmp_path / "p7" / "paper_iii_evidence_report.json"
    output_md = tmp_path / "p7" / "paper_iii_evidence_report.md"

    assert main(
        [
            "--frozen-run-manifest",
            str(bundle["frozen_run_manifest"]),
            "--adaptive-run-manifest",
            str(bundle["adaptive_run_manifest"]),
            "--promoted-mclachlan-run-manifest",
            str(bundle["promoted_run_manifest"]),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    ) == 0

    assert output_json.exists()
    assert output_md.exists()
    command_log = output_json.parent / "command_log.md"
    run_manifest = output_json.parent / "run_manifest.json"
    assert command_log.exists()
    assert run_manifest.exists()
    after_hashes = {name: sha256_file(path) for name, path in bundle.items() if name.endswith("output")}
    assert after_hashes == before_hashes

    report = json.loads(output_json.read_text(encoding="utf-8"))
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    assert manifest["output_summary"]["controller_usable"] is False
    assert manifest["output_summary"]["feeds_controller_decisions"] is False
    assert manifest["output_summary"]["reference_comparisons_feed_controller_decisions"] is False
    assert manifest["output_summary"]["promoted_mclachlan_evidence_classification"] == "contract_plumbing"
    assert "controller_usable: `false`" in output_md.read_text(encoding="utf-8")
    assert "Exit code: `0`" in command_log.read_text(encoding="utf-8")

    for text in (
        json.dumps(report, sort_keys=True),
        output_md.read_text(encoding="utf-8"),
        command_log.read_text(encoding="utf-8"),
        json.dumps(manifest, sort_keys=True),
    ):
        assert "amplitudes_qn_to_q0" not in text
        assert "raw_physical_state" not in text
        assert "statevector" not in text


def test_fail_closed_if_source_or_manifest_feeds_controller_decisions(tmp_path: Path) -> None:
    bundle = _source_bundle(tmp_path)
    frozen = json.loads(bundle["frozen_output"].read_text(encoding="utf-8"))
    frozen["feeds_controller_decisions"] = True
    _write_json(bundle["frozen_output"], frozen)

    with pytest.raises(PaperIIIEvidenceReportError, match="feeds_controller_decisions"):
        run_paper_iii_evidence_report(_config(tmp_path, bundle))


def test_fail_closed_if_active_basis_indices_are_inconsistent(tmp_path: Path) -> None:
    bundle = _source_bundle(tmp_path)
    adaptive = json.loads(bundle["adaptive_output"].read_text(encoding="utf-8"))
    adaptive["trajectory"][0]["qse_basis_coefficients"][0]["basis_index"] = 3
    _write_json(bundle["adaptive_output"], adaptive)

    with pytest.raises(PaperIIIEvidenceReportError, match="basis_index"):
        run_paper_iii_evidence_report(_config(tmp_path, bundle))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p["source"].__setitem__("loader_boundary", "whole_artifact"), "loader_boundary"),
        (lambda p: p["runtime_contract"].__setitem__("structure_locked", False), "structure_locked"),
        (lambda p: p["forbidden_marker_audit"].update({"passed": False, "hit_count": 1, "hits": ["exact"]}), "passed"),
        (lambda p: p["trajectory"][0].__setitem__("uses_reference_for_decision", True), "uses_reference_for_decision"),
        (lambda p: p["summary"].__setitem__("paper_iii_science_benchmark", True), "paper_iii_science_benchmark"),
    ],
)
def test_fail_closed_for_p6a_contract_violations(
    tmp_path: Path,
    mutate: Callable[[dict], None],
    match: str,
) -> None:
    bundle = _source_bundle(tmp_path)
    p6 = json.loads(bundle["promoted_output"].read_text(encoding="utf-8"))
    mutate(p6)
    _write_json(bundle["promoted_output"], p6)

    with pytest.raises(PaperIIIEvidenceReportError, match=match):
        run_paper_iii_evidence_report(_config(tmp_path, bundle))


def test_fail_closed_when_qse_matrices_are_missing(tmp_path: Path) -> None:
    bundle = _source_bundle(tmp_path)
    run_manifest = json.loads(bundle["frozen_run_manifest"].read_text(encoding="utf-8"))
    qse_path = Path(run_manifest["artifacts"]["input_qse_manifest_json"])
    qse = json.loads(qse_path.read_text(encoding="utf-8"))
    qse["matrices"] = {"included": False}
    _write_json(qse_path, qse)
    frozen = json.loads(bundle["frozen_output"].read_text(encoding="utf-8"))
    frozen["source"]["source_qse_sha256"] = sha256_file(qse_path)
    _write_json(bundle["frozen_output"], frozen)

    with pytest.raises(PaperIIIEvidenceReportError, match="must include matrices"):
        run_paper_iii_evidence_report(_config(tmp_path, bundle))

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.source_maps import (
    PAPER_III_QSE_SOURCE_MAP_AUDIT_SCHEMA_VERSION,
    PAPER_III_QSE_SOURCE_MAP_SCHEMA_VERSION,
    PaperIIIQSESourceMapError,
    PaperIIIQSESourceSpec,
    audit_paper_iii_qse_source_map,
    build_paper_iii_qse_source_map,
    sha256_file,
    validate_paper_iii_qse_source_map,
    write_paper_iii_qse_source_map,
)
from pipelines.reporting.paper_iii_qse_audit import run_paper_iii_qse_source_map_audit


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _controller_boundary() -> dict[str, Any]:
    return {
        "feeds_controller_decisions": False,
        "uses_exact_reference_for_decision": False,
        "uses_future_exact_forecast_for_decision": False,
        "exact_reference_role": "diagnostic_reporting_only",
        "promotion_requires_user_approval": True,
    }


def _conductivity_payload() -> dict[str, Any]:
    return {
        "schema_version": "qse_conductivity_response_v1",
        "policy": "diagnostic_only_current_response_postprocessing",
        "response_kind": "conductivity_current",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "observables": [{"label": "hh_J[positive_chain]"}, {"label": "hh_K[positive_chain]"}],
        "channels": [
            {
                "current_label": "hh_J[positive_chain]",
                "contact_label": "hh_K[positive_chain]",
                "channel_kind": "longitudinal_charge",
                "current_source": {"status": "evaluated", "zero_current_source": False},
                "contact_term": {"status": "evaluated"},
                "drude_weight": {"status": "not_evaluated"},
            }
        ],
    }


def _green_function_payload() -> dict[str, Any]:
    return {
        "schema_version": "qse_green_function_v1",
        "policy": "diagnostic_only_single_particle_green_function_postprocessing",
        "response_kind": "single_particle_retarded_green_function_diagonal",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "mode_domain": {"fermion_mode_count": 4},
        "summary": {"mode_count": 2, "sector_count": 4, "solved_sector_count": 3, "zero_source_sector_count": 1},
        "modes": [
            {"label": "up0", "mode_index": 0, "diagonal_sum_rule_diagnostics": {"status": "evaluated"}},
            {"label": "dn0", "mode_index": 1, "diagonal_sum_rule_diagnostics": {"status": "evaluated"}},
        ],
    }


def _qse_manifest() -> dict[str, Any]:
    return {
        "schema_version": "qse_spectra_v1",
        "pipeline": "qse_spectra",
        "generated_utc": "2026-06-07T00:00:00Z",
        "run_tag": "hh_weak_case",
        "input": {
            "hamiltonian": {"source_schema": "terms", "term_count_input": 3, "term_count_output": 2},
            "operator_basis": {"source_schema": "artifact_basis_source:full_meta_filtered", "basis_size": 2},
        },
        "diagnostics": {
            "num_qubits": 1,
            "basis_size": 2,
            "retained_rank": 2,
            "discarded_rank": 0,
            "overlap_condition_estimate": 4.0,
        },
        "operator_basis": [
            {"basis_index": 0, "name": "e", "kind": "pauli_string", "pauli_exyz": "e"},
            {"basis_index": 1, "name": "x", "kind": "pauli_string", "pauli_exyz": "x"},
        ],
        "transition_observables": [{"name": "density", "transition_strengths": [0.0, 1.0]}],
        "eigenvalues": [
            {"state_index": 0, "energy": -1.0, "generalized_residual_norm": 1.0e-10},
            {"state_index": 1, "energy": 0.2, "generalized_residual_norm": 2.0e-9},
        ],
        "static_record_selection": {
            "selection_config": {"mode": "geometry_selected", "geometry_target_roots": 6},
            "controller_boundary": {
                "controller_usable": False,
                "feeds_controller_decisions": False,
                "post_run_diagnostic_only": True,
            },
        },
        "paper_iii_contract": {
            "schema_version": "paper_iii_qse_production_contract_v1",
            "run_class": "candidate",
            "visible_target": "tab:qse_static_claims",
            "compatibility_tier": "n_ph_2_compatibility",
            "approval_status": "user_review_required",
            "controller_boundary": _controller_boundary(),
        },
        "qse_response_functions_v1": {
            "schema_version": "qse_response_functions_v1",
            "policy": "diagnostic_only_neutral_response_postprocessing",
            "response_kind": "neutral",
            "controller_boundary": {
                "feeds_controller_decisions": False,
                "controller_usable": False,
                "post_run_diagnostic_only": True,
            },
            "observables": [{"label": "density"}],
            "channels": [
                {
                    "A_label": "density",
                    "B_label": "density",
                    "channel_kind": "nn",
                    "sum_rule_deficits": {"status": "evaluated", "m0": {"deficit_abs": 0.0}},
                }
            ],
        },
        "qse_conductivity_response_v1": _conductivity_payload(),
        "qse_green_function_v1": _green_function_payload(),
    }


def _aggregate() -> dict[str, Any]:
    return {
        "schema_version": "paper_iii_qse_table_aggregate_v1",
        "pipeline": "qse_table_aggregate",
        "generated_utc": "2026-06-07T00:01:00Z",
        "rows": [
            {
                "row_id": "hh_weak_case",
                "method_id": "qse_selection::geometry_selected",
                "run_class": "candidate",
                "compatibility_tier": "n_ph_2_compatibility",
                "approval_status": "user_review_required",
                "controller_boundary_passed": True,
            }
        ],
        "summary": {
            "row_count": 1,
            "method_ids": ["qse_selection::geometry_selected"],
            "run_classes": ["candidate"],
            "compatibility_tiers": ["n_ph_2_compatibility"],
            "all_rows_controller_boundary_passed": True,
        },
    }


def _optuna_summary() -> dict[str, Any]:
    return {
        "schema_version": "paper_iii_static_qse_geometry_optuna_v1",
        "pipeline": "paper_iii_static_qse_geometry_optuna",
        "study_kind": "production_settings_search",
        "settings_candidate_kind": "optuna_geometry_selection_settings_candidate",
        "settings_candidate_status": "settings_candidate_not_promoted_user_review_required",
        "settings_promoted": False,
        "run_class": "candidate",
        "visible_target": "tab:qse_static_claims",
        "compatibility_tier": "n_ph_2_compatibility",
        "approval_status": "user_review_required",
        "regime_label": "hh_weak_case",
        "generated_utc": "2026-06-07T00:02:00Z",
        "completed_utc": "2026-06-07T00:03:00Z",
        "objective_contract": {
            "mode": "qse_native",
            "uses_diagnostic_reference_supervision": False,
            "exact_reference_role": "not_used_by_objective",
            "controller_decision_input": False,
        },
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
            "exact_reference_role": "not_used_by_objective",
        },
    }


def _compatibility_manifest() -> dict[str, Any]:
    return {
        "schema_version": "paper_iii_compatibility_matrix_manifest_v1",
        "records_schema_version": "paper_iii_compatibility_matrix_records_v1",
        "profile": "paper_iii_nph2_compatibility_v1",
        "matrix_batch_id": "paper_iii_nph2_compatibility_v1_local_audit",
        "generated_utc": "2026-06-07T00:04:00Z",
        "n_ph_max": 2,
        "target_excited_roots": 6,
        "exact_ed_qse_diagnostics_role": "report_only_never_controller_decision_input",
        "strict_policy": {
            "controller_exact_input_mode": "off",
            "diagnostic_exact_reference_mode": "benchmark_exact",
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "exact_decision_checkpoints": 0,
            "strict_measurement_oracle_certified": True,
            "qpu_faithful_decisions_passed": True,
        },
    }


def _evidence_report() -> dict[str, Any]:
    return {
        "schema_version": "paper_iii_evidence_report_v1",
        "pipeline": "paper_iii_evidence_report",
        "generated_utc": "2026-06-07T00:05:00Z",
        "report_kind": "paper_iii_p7a_local_evidence_report",
        "controller_usable": False,
        "feeds_controller_decisions": False,
        "reference_comparisons_feed_controller_decisions": False,
        "exact_or_ed_reference_values_feed_controller_decisions": False,
        "raw_physical_vectors_emitted": False,
        "comparison_method": {
            "coefficient_only": True,
            "post_run_only": True,
            "feeds_controller_decisions": False,
        },
        "reference_comparisons": {"post_run_only": True, "feeds_controller_decisions": False},
        "controller_boundary": {
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "reference_comparisons_feed_controller_decisions": False,
            "decision_path_allowed": False,
            "post_run_diagnostic_only": True,
            "controller_exact_input_mode": "off",
            "diagnostic_exact_reference_mode": "benchmark_post_run_only",
        },
        "metrics": {
            "source_artifact_count": 3,
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "reference_comparisons_feed_controller_decisions": False,
            "raw_physical_vectors_emitted": False,
        },
        "scope_guardrails": {
            "report_only": True,
            "source_artifacts_modified": False,
            "reference_comparisons_feed_controller_decisions": False,
        },
    }


def _report_input() -> dict[str, Any]:
    return {
        "schema_version": "paper_iii_qse_report_input_v1",
        "pipeline": "paper_iii_qse_results_report_input",
        "method_id": "qse_selection::geometry_selected",
        "run_class": "candidate",
        "compatibility_tier": "n_ph_2_compatibility",
        "regime": "hh_weak_case",
        "approval_status": "user_review_required",
        "generated_utc": "2026-06-07T00:06:00Z",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "exact_reference_role": "diagnostic_reporting_only",
        },
    }


def _source_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "qse_manifest": _write_json(tmp_path / "qse.json", _qse_manifest()),
        "aggregate": _write_json(tmp_path / "aggregate.json", _aggregate()),
        "optuna_study": _write_json(tmp_path / "optuna_summary.json", _optuna_summary()),
        "compatibility_matrix": _write_json(tmp_path / "compatibility_manifest.json", _compatibility_manifest()),
        "evidence_report": _write_json(tmp_path / "evidence_report.json", _evidence_report()),
        "report_input": _write_json(tmp_path / "report_input.json", _report_input()),
    }


def _source_map(tmp_path: Path) -> dict[str, Any]:
    paths = _source_paths(tmp_path)
    return build_paper_iii_qse_source_map(
        [PaperIIIQSESourceSpec(role=role, path=path) for role, path in paths.items()],
        source_root=tmp_path,
        map_id="tmp_item8_source_map",
    )


def test_builds_and_validates_paper_iii_qse_source_map_for_all_roles(tmp_path: Path) -> None:
    source_map = _source_map(tmp_path)
    output_json = tmp_path / "source_map.json"
    write_paper_iii_qse_source_map(source_map, output_json)

    assert source_map["schema_version"] == PAPER_III_QSE_SOURCE_MAP_SCHEMA_VERSION
    assert source_map["source_count"] == 6
    assert source_map["summary"]["all_sources_have_approval_status"] is True
    assert source_map["summary"]["all_sources_controller_boundary_passed"] is True
    roles = {record["source_map_role"] for record in source_map["sources"]}
    assert roles == {
        "qse_manifest",
        "optuna_study",
        "aggregate",
        "compatibility_matrix",
        "evidence_report",
        "report_input",
    }
    qse_record = next(record for record in source_map["sources"] if record["source_map_role"] == "qse_manifest")
    assert qse_record["source_path"] == "qse.json"
    assert qse_record["source_sha256"] == sha256_file(tmp_path / "qse.json")
    assert qse_record["schema_version"] == "qse_spectra_v1"
    assert qse_record["pipeline"] == "qse_spectra"
    assert qse_record["run_class"] == "candidate"
    assert qse_record["compatibility_tier"] == "n_ph_2_compatibility"
    assert qse_record["method_id"] == "qse_selection::geometry_selected"
    assert qse_record["regime_or_case"] == "hh_weak_case"
    assert qse_record["approval_status"] == "user_review_required"
    assert qse_record["controller_boundary"]["passed"] is True
    assert "qse_conductivity_response_v1.controller_boundary" in qse_record["controller_boundary"]["checked_sections"]
    assert "qse_green_function_v1.controller_boundary" in qse_record["controller_boundary"]["checked_sections"]

    report = validate_paper_iii_qse_source_map(source_map, base_dir=tmp_path)
    assert report["schema_version"] == PAPER_III_QSE_SOURCE_MAP_AUDIT_SCHEMA_VERSION
    assert report["ok"] is True

    audit_output = tmp_path / "audit.json"
    wrapper_report = run_paper_iii_qse_source_map_audit(output_json, base_dir=tmp_path, output_json=audit_output)
    assert wrapper_report["ok"] is True
    assert audit_output.exists()


def test_audit_fails_on_missing_source_file(tmp_path: Path) -> None:
    source_map = _source_map(tmp_path)
    (tmp_path / "qse.json").unlink()

    with pytest.raises(PaperIIIQSESourceMapError, match="missing_source_file"):
        validate_paper_iii_qse_source_map(source_map, base_dir=tmp_path)


def test_audit_fails_on_sha_mismatch(tmp_path: Path) -> None:
    source_map = _source_map(tmp_path)
    qse_path = tmp_path / "qse.json"
    payload = json.loads(qse_path.read_text(encoding="utf-8"))
    payload["run_tag"] = "changed_after_source_map"
    _write_json(qse_path, payload)

    with pytest.raises(PaperIIIQSESourceMapError, match="source_sha256_mismatch"):
        validate_paper_iii_qse_source_map(source_map, base_dir=tmp_path)


def test_audit_fails_on_schema_mismatch(tmp_path: Path) -> None:
    source_map = _source_map(tmp_path)
    source_map["sources"][0]["expected_schema_version"] = "wrong_schema_v1"

    with pytest.raises(PaperIIIQSESourceMapError, match="source_schema_mismatch"):
        validate_paper_iii_qse_source_map(source_map, base_dir=tmp_path)


def test_audit_fails_on_duplicate_source_ids(tmp_path: Path) -> None:
    source_map = _source_map(tmp_path)
    source_map["sources"].append(dict(source_map["sources"][0]))

    with pytest.raises(PaperIIIQSESourceMapError, match="duplicate_source_id"):
        validate_paper_iii_qse_source_map(source_map, base_dir=tmp_path)


def test_audit_fails_closed_on_aggregate_only_blocked_gate(tmp_path: Path) -> None:
    aggregate = _aggregate()
    aggregate["rows"][0].update(
        {
            "production_gate_present": True,
            "production_gate_ok": False,
            "production_gate_production_ready": False,
        }
    )
    aggregate["summary"]["all_rows_controller_boundary_passed"] = True
    aggregate_path = _write_json(tmp_path / "aggregate_blocked.json", aggregate)
    source_map = build_paper_iii_qse_source_map(
        [PaperIIIQSESourceSpec(role="aggregate", path=aggregate_path)],
        source_root=tmp_path,
    )

    with pytest.raises(PaperIIIQSESourceMapError, match="production_gate_blocked"):
        validate_paper_iii_qse_source_map(source_map, base_dir=tmp_path)


def test_audit_fails_on_missing_approval_status(tmp_path: Path) -> None:
    source_map = _source_map(tmp_path)
    source_map["sources"][0]["approval_status"] = ""

    report = audit_paper_iii_qse_source_map(source_map, base_dir=tmp_path)
    assert report["ok"] is False
    assert any(failure["code"] == "missing_approval_status" for failure in report["failures"])
    with pytest.raises(PaperIIIQSESourceMapError, match="missing_approval_status"):
        validate_paper_iii_qse_source_map(source_map, base_dir=tmp_path)


@pytest.mark.parametrize("payload_key", ["qse_conductivity_response_v1", "qse_green_function_v1"])
def test_audit_fails_on_current_green_missing_controller_boundary(tmp_path: Path, payload_key: str) -> None:
    paths = _source_paths(tmp_path)
    qse = _qse_manifest()
    del qse[payload_key]["controller_boundary"]
    _write_json(paths["qse_manifest"], qse)
    source_map = build_paper_iii_qse_source_map(
        [PaperIIIQSESourceSpec(role=role, path=path) for role, path in paths.items()],
        source_root=tmp_path,
    )

    with pytest.raises(PaperIIIQSESourceMapError, match="controller_boundary_missing"):
        validate_paper_iii_qse_source_map(source_map, base_dir=tmp_path)


@pytest.mark.parametrize("payload_key", ["qse_conductivity_response_v1", "qse_green_function_v1"])
def test_audit_fails_on_current_green_controller_boundary_violation(tmp_path: Path, payload_key: str) -> None:
    paths = _source_paths(tmp_path)
    qse = _qse_manifest()
    qse[payload_key]["controller_boundary"]["feeds_controller_decisions"] = True
    _write_json(paths["qse_manifest"], qse)
    source_map = build_paper_iii_qse_source_map(
        [PaperIIIQSESourceSpec(role=role, path=path) for role, path in paths.items()],
        source_root=tmp_path,
    )

    with pytest.raises(PaperIIIQSESourceMapError, match="controller_boundary_violation"):
        validate_paper_iii_qse_source_map(source_map, base_dir=tmp_path)


def test_audit_fails_closed_on_blocked_candidate_production_gate(tmp_path: Path) -> None:
    paths = _source_paths(tmp_path)
    qse = _qse_manifest()
    qse["run_tag"] = "hh_strong_weak_nph2_candidate_blocked_gate"
    qse["paper_iii_production_gate"] = {
        "schema_version": "paper_iii_qse_production_gate_v1",
        "ok": False,
        "first_pass_ready": True,
        "production_ready": False,
        "n_ph2_production_readiness": "blocked",
        "exact_reference_boundary_status": {"status": "pass", "violation_count": 0},
    }
    _write_json(paths["qse_manifest"], qse)
    source_map = build_paper_iii_qse_source_map(
        [PaperIIIQSESourceSpec(role=role, path=path) for role, path in paths.items()],
        source_root=tmp_path,
    )

    with pytest.raises(PaperIIIQSESourceMapError, match="production_gate_blocked"):
        validate_paper_iii_qse_source_map(source_map, base_dir=tmp_path)


def test_audit_fails_on_controller_boundary_violation(tmp_path: Path) -> None:
    paths = _source_paths(tmp_path)
    qse = _qse_manifest()
    qse["paper_iii_contract"]["controller_boundary"]["uses_exact_reference_for_decision"] = True
    _write_json(paths["qse_manifest"], qse)
    source_map = build_paper_iii_qse_source_map(
        [PaperIIIQSESourceSpec(role=role, path=path) for role, path in paths.items()],
        source_root=tmp_path,
    )

    with pytest.raises(PaperIIIQSESourceMapError, match="controller_boundary_violation"):
        validate_paper_iii_qse_source_map(source_map, base_dir=tmp_path)

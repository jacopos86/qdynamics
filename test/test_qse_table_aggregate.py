from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.table_aggregate import (
    QSE_TABLE_AGGREGATE_SCHEMA_VERSION,
    QSETableAggregateConfig,
    build_qse_table_aggregate,
    main,
    summarize_qse_manifest,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest(*, selection_mode: str | None = None) -> dict:
    payload = {
        "schema_version": "qse_spectra_v1",
        "pipeline": "qse_spectra",
        "input": {
            "hamiltonian": {"source_schema": "terms", "term_count_input": 3, "term_count_output": 2},
            "operator_basis": {"source_schema": "operator_basis", "basis_size": 2},
        },
        "operator_basis": [
            {"basis_index": 0, "name": "x", "kind": "pauli_string", "pauli_exyz": "x"},
            {
                "basis_index": 1,
                "name": "poly",
                "kind": "pauli_polynomial",
                "terms": [
                    {"pauli_exyz": "x", "coeff": {"re": 1.0, "im": 0.0}, "nq": 1},
                    {"pauli_exyz": "z", "coeff": {"re": 0.5, "im": 0.0}, "nq": 1},
                ],
            },
        ],
        "transition_observables": [{"name": "dipole", "transition_strengths": [1.0]}],
        "diagnostics": {
            "num_qubits": 1,
            "basis_size": 2,
            "retained_rank": 2,
            "discarded_rank": 0,
            "overlap_condition_estimate": 3.0,
        },
        "eigenvalues": [
            {"state_index": 0, "energy": -1.0, "generalized_residual_norm": 1.0e-12},
            {"state_index": 1, "energy": 1.0, "generalized_residual_norm": 2.0e-12},
        ],
        "spectral_window_metrics": {
            "observables": [
                {
                    "name": "dipole",
                    "window_metrics": [
                        {
                            "window_name": "gap",
                            "reference_comparison": {
                                "l1_error": 0.2,
                                "l2_error": 0.1,
                                "max_abs_error": 0.3,
                                "normalized_l1_error": 0.05,
                            },
                        }
                    ],
                }
            ]
        },
    }
    if selection_mode is not None:
        payload["static_record_selection"] = {
            "selection_config": {"mode": selection_mode},
            "summary": {"selected_basis_size": 2},
        }
    return payload


def _conductivity_payload() -> dict[str, object]:
    return {
        "schema_version": "qse_conductivity_response_v1",
        "policy": "diagnostic_only_current_response_postprocessing",
        "response_kind": "conductivity_current",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "contact_policy": {"name": "contact_record_only_no_drude_delta"},
        "peierls_policy": {"name": "standard_hh_1d_charge_peierls"},
        "regular_conductivity_policy": {"name": "positive_frequency_paramagnetic_sjj_over_omega"},
        "frequency_grid": {"num_points": 3, "values": [0.0, 1.0, 2.0]},
        "observables": [{"label": "hh_J[positive_chain]"}, {"label": "hh_K[positive_chain]"}],
        "channels": [
            {
                "current_label": "hh_J[positive_chain]",
                "contact_label": "hh_K[positive_chain]",
                "channel_kind": "longitudinal_charge",
                "current_source": {"status": "evaluated", "zero_current_source": False},
                "contact_term": {"status": "evaluated", "expectation_value": [0.5, 0.0]},
                "drude_weight": {"status": "not_evaluated"},
                "regular_conductivity": {"values": [[0.0, 0.0], [0.2, 0.0], [0.1, 0.0]]},
            }
        ],
    }


def _green_function_payload() -> dict[str, object]:
    return {
        "schema_version": "qse_green_function_v1",
        "policy": "diagnostic_only_single_particle_green_function_postprocessing",
        "response_kind": "single_particle_retarded_green_function_diagonal",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "frequency_grid": {"num_points": 3, "values": [0.0, 1.0, 2.0]},
        "mode_domain": {"fermion_mode_count": 4},
        "summary": {"mode_count": 2, "sector_count": 4, "solved_sector_count": 3, "zero_source_sector_count": 1},
        "modes": [
            {
                "label": "up0",
                "mode_index": 0,
                "diagonal_sum_rule_diagnostics": {
                    "status": "evaluated",
                    "source_norm_canonical_deficit_abs": 0.01,
                    "residue_canonical_deficit_abs": 0.02,
                },
            },
            {
                "label": "dn0",
                "mode_index": 1,
                "diagonal_sum_rule_diagnostics": {
                    "status": "evaluated",
                    "source_norm_canonical_deficit_abs": 0.03,
                    "residue_canonical_deficit_abs": 0.04,
                },
            },
        ],
    }


def test_summarize_qse_manifest_extracts_table_row_metrics(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "qse.json", _manifest(selection_mode="geometry_selected"))

    row = summarize_qse_manifest(path)

    assert row["method_id"] == "qse_selection::geometry_selected"
    assert row["basis_size"] == 2
    assert row["retained_rank"] == 2
    assert row["overlap_condition_estimate"] == pytest.approx(3.0)
    assert row["eigenvalue_count"] == 2
    assert row["condition_number"] == pytest.approx(3.0)
    assert row["target_root_count"] is None
    assert row["lowest_energy"] == pytest.approx(-1.0)
    assert row["max_generalized_residual_norm"] == pytest.approx(2.0e-12)
    assert row["matrix_measurement_proxy"] == {
        "basis_pairs_upper_triangle": 3,
        "hamiltonian_term_count": 2,
        "basis_term_proxy": 3,
        "overlap_entries": 3,
        "hamiltonian_entries": 6,
        "transition_entries": 2,
        "total": 11,
    }
    assert row["spectral_reference_l2_error_max"] == pytest.approx(0.1)
    assert row["spectral_reference_max_abs_error_max"] == pytest.approx(0.3)
    assert row["run_class"] is None
    assert row["approval_status"] is None
    assert row["compatibility_tier"] is None
    assert row["n_ph_max"] is None
    assert row["response_channel_count"] is None
    assert row["conductivity_response_present"] is False
    assert row["conductivity_channel_count"] is None
    assert row["green_function_present"] is False
    assert row["green_function_mode_count"] is None
    assert row["moment_deficit_status"] == "missing"
    assert row["controller_boundary_status"] == "missing"
    assert row["controller_boundary_passed"] is None
    assert row["spectral_window_metric_count"] == 1


def test_summarize_qse_manifest_extracts_paper_iii_production_fields(tmp_path: Path) -> None:
    payload = _manifest(selection_mode="geometry_selected")
    payload["static_record_selection"]["controller_boundary"] = {
        "controller_usable": False,
        "feeds_controller_decisions": False,
        "post_run_diagnostic_only": True,
    }
    payload["static_record_selection"]["selection_config"]["geometry_target_roots"] = 6
    payload["paper_iii_contract"] = {
        "schema_version": "paper_iii_qse_production_contract_v1",
        "run_class": "candidate",
        "visible_target": "tab:qse_static_claims",
        "compatibility_tier": "n_ph_2_compatibility",
        "approval_status": "user_review_required",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "uses_exact_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "exact_reference_role": "diagnostic_reporting_only",
            "promotion_requires_user_approval": True,
        },
        "hh_full_meta_provenance": {
            "layout": {
                "problem_key": "hh",
                "num_sites": 2,
                "n_ph_max": 2,
                "boson_encoding": "binary",
                "ordering": "blocked",
            }
        },
    }
    payload["qse_response_functions_v1"] = {
        "schema_version": "qse_response_functions_v1",
        "policy": "diagnostic_only_neutral_response_postprocessing",
        "response_kind": "neutral",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "moment_orders": [0, 1],
        "observables": [{"label": "density"}, {"label": "displacement"}],
        "channels": [
            {
                "A_label": "density",
                "B_label": "density",
                "channel_kind": "nn",
                "sum_rule_deficits": {
                    "status": "evaluated",
                    "m0": {"deficit_abs": 0.02},
                    "m1": {"deficit_abs": 0.03},
                },
            },
            {
                "A_label": "density",
                "B_label": "displacement",
                "channel_kind": "nX",
                "sum_rule_deficits": {
                    "status": "not_evaluated",
                    "reason": "direct_state_hamiltonian_or_evaluation_not_supplied",
                },
            },
        ],
    }
    payload["spectral_functions"] = {
        "schema_version": "qse_spectral_functions_v1",
        "controller_boundary": {"feeds_controller_decisions": False, "controller_usable": False},
        "observables": [{"name": "density"}, {"name": "displacement"}],
    }
    payload["spectral_window_metrics"]["schema_version"] = "qse_spectral_window_metrics_v1"
    payload["spectral_window_metrics"]["controller_boundary"] = {
        "feeds_controller_decisions": False,
        "controller_usable": False,
    }
    payload["spectral_window_metrics"]["windows"] = [{"name": "gap", "omega_min": 0.0, "omega_max": 4.0}]
    payload["cutoff_boundary_diagnostics"] = {
        "schema_version": "qse_cutoff_boundary_diagnostics_v1",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "layout": {
            "num_sites": 2,
            "n_ph_max": 2,
            "boson_encoding": "binary",
            "fermion_qubits": 4,
            "qubits_per_boson_site": 2,
            "total_qubits": 8,
        },
        "roots": [{"state_index": 0, "ell_cut": 0.12, "illegal_probability_max": 0.0}],
    }
    payload["qse_conductivity_response_v1"] = _conductivity_payload()
    payload["qse_green_function_v1"] = _green_function_payload()
    payload["paper_iii_production_gate"] = {
        "schema_version": "paper_iii_qse_production_gate_v1",
        "ok": True,
        "first_pass_ready": True,
        "production_ready": True,
        "n_ph1_first_pass_status": "ready",
        "n_ph2_production_readiness": "ready",
        "required_target_excited_roots": 6,
        "exact_reference_boundary_status": {"status": "pass", "violation_count": 0},
    }
    path = _write_json(tmp_path / "production_qse.json", payload)

    row = summarize_qse_manifest(path)

    assert row["run_class"] == "candidate"
    assert row["approval_status"] == "user_review_required"
    assert row["compatibility_tier"] == "n_ph_2_compatibility"
    assert row["visible_target"] == "tab:qse_static_claims"
    assert row["n_ph_max"] == 2
    assert row["n_ph_source"] == "paper_iii_contract.hh_full_meta_provenance.layout.n_ph_max"
    assert row["target_root_count"] == 6
    assert row["target_root_count_source"] == "static_record_selection.selection_config.geometry_target_roots"
    assert row["cutoff_n_ph_max"] == 2
    assert row["cutoff_num_sites"] == 2
    assert row["cutoff_boson_encoding"] == "binary"
    assert row["response_channel_count"] == 2
    assert row["response_observable_count"] == 2
    assert row["moment_deficit_summary"]["status_counts"] == {"evaluated": 1, "not_evaluated": 1}
    assert row["moment_deficit_evaluated_channel_count"] == 1
    assert row["moment_deficit_not_evaluated_channel_count"] == 1
    assert row["moment_deficit_m0_abs_max"] == pytest.approx(0.02)
    assert row["moment_deficit_m1_abs_max"] == pytest.approx(0.03)
    assert row["conductivity_response_present"] is True
    assert row["conductivity_schema_version"] == "qse_conductivity_response_v1"
    assert row["conductivity_channel_count"] == 1
    assert row["conductivity_observable_count"] == 2
    assert row["conductivity_contact_supplied_channel_count"] == 1
    assert row["conductivity_zero_current_source_count"] == 0
    assert row["conductivity_summary"]["contact_status_counts"] == {"evaluated": 1}
    assert row["conductivity_summary"]["drude_status_counts"] == {"not_evaluated": 1}
    assert row["green_function_present"] is True
    assert row["green_function_schema_version"] == "qse_green_function_v1"
    assert row["green_function_mode_count"] == 2
    assert row["green_function_sector_count"] == 4
    assert row["green_function_solved_sector_count"] == 3
    assert row["green_function_zero_source_sector_count"] == 1
    assert row["green_function_source_norm_canonical_deficit_abs_max"] == pytest.approx(0.03)
    assert row["green_function_residue_canonical_deficit_abs_max"] == pytest.approx(0.04)
    assert row["spectral_function_observable_count"] == 2
    assert row["spectral_window_count"] == 1
    assert row["spectral_window_metric_count"] == 1
    assert row["cutoff_root_count"] == 1
    assert row["cutoff_ell_cut_max"] == pytest.approx(0.12)
    assert row["production_gate_ok"] is True
    assert row["production_gate_production_ready"] is True
    assert row["production_gate_required_target_excited_roots"] == 6
    assert row["production_gate_exact_reference_boundary_status"] == "pass"
    assert row["controller_boundary_status"] == "pass"
    assert row["controller_boundary_passed"] is True
    assert "qse_response_functions_v1" in row["controller_boundary"]["checked_sections"]
    assert "qse_conductivity_response_v1" in row["controller_boundary"]["checked_sections"]
    assert "qse_green_function_v1" in row["controller_boundary"]["checked_sections"]


def test_summarize_qse_manifest_reports_controller_boundary_failure_and_missing_response(
    tmp_path: Path,
) -> None:
    payload = _manifest()
    payload["paper_iii_contract"] = {
        "schema_version": "paper_iii_qse_production_contract_v1",
        "run_class": "candidate",
        "visible_target": "tab:qse_static_claims",
        "compatibility_tier": "not_evaluated",
        "approval_status": "user_review_required",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "uses_exact_reference_for_decision": True,
            "exact_reference_role": "diagnostic_reporting_only",
        },
    }
    path = _write_json(tmp_path / "boundary_fail.json", payload)

    row = summarize_qse_manifest(path)

    assert row["response_channel_count"] is None
    assert row["moment_deficit_status"] == "missing"
    assert row["moment_deficit_m0_abs_max"] is None
    assert row["controller_boundary_status"] == "fail"
    assert row["controller_boundary_passed"] is False
    assert any("uses_exact_reference_for_decision" in item for item in row["controller_boundary"]["failures"])


def test_build_qse_table_aggregate_and_cli_write_outputs(tmp_path: Path) -> None:
    first = _write_json(tmp_path / "first.json", _manifest(selection_mode="geometry_selected"))
    second = _write_json(tmp_path / "second.json", _manifest(selection_mode=None))
    output_json = tmp_path / "aggregate" / "rows.json"
    output_tsv = tmp_path / "aggregate" / "rows.tsv"
    output_md = tmp_path / "aggregate" / "rows.md"

    aggregate = build_qse_table_aggregate(
        QSETableAggregateConfig(qse_manifest_paths=(first, second), output_json=output_json)
    )

    assert aggregate["schema_version"] == QSE_TABLE_AGGREGATE_SCHEMA_VERSION
    assert aggregate["summary"]["row_count"] == 2
    assert aggregate["summary"]["method_ids"] == ["qse_basis::operator_basis", "qse_selection::geometry_selected"]
    assert aggregate["summary"]["rows_with_conductivity_payload"] == 0
    assert aggregate["summary"]["rows_with_green_function_payload"] == 0

    assert main(
        [
            "--qse-manifest",
            str(first),
            "--qse-manifest",
            str(second),
            "--output-json",
            str(output_json),
            "--output-tsv",
            str(output_tsv),
            "--output-md",
            str(output_md),
        ]
    ) == 0
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["summary"]["row_count"] == 2
    assert "matrix_measurement_proxy_total" in output_tsv.read_text(encoding="utf-8")
    assert "Paper III QSE table aggregate" in output_md.read_text(encoding="utf-8")

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.__main__ import main
from pipelines.qse_spectra.io import (
    basis_elements_from_artifact_source,
    load_polynomial_json,
    polynomial_from_serialized_terms,
    statevector_from_manifest,
)
from pipelines.scaffold.handoff_state_bundle import build_statevector_manifest


def test_statevector_manifest_round_trip_uses_qn_to_q0_ordering() -> None:
    psi = np.asarray([0.0, 1.0j], dtype=complex)
    manifest = build_statevector_manifest(
        psi_state=psi,
        source="unit_test",
        handoff_state_kind="prepared_state",
    )

    loaded, provenance = statevector_from_manifest(manifest, expected_nq=1)

    assert provenance["source_schema"] == "top_level_state_manifest"
    assert provenance["selected_state_key"] == "top_level"
    assert provenance["nq_total"] == 1
    assert np.allclose(loaded, np.asarray([0.0, 1.0j], dtype=complex))


def test_statevector_manifest_can_select_explicit_artifact_state_key() -> None:
    initial = build_statevector_manifest(psi_state=np.asarray([1.0, 0.0]), source="initial")
    ansatz = build_statevector_manifest(psi_state=np.asarray([0.0, 1.0]), source="ansatz")
    payload = {"initial_state": initial, "ansatz_input_state": ansatz}

    loaded, provenance = statevector_from_manifest(payload, expected_nq=1, state_key="ansatz_input_state")

    assert provenance["source_schema"] == "artifact_state_block"
    assert provenance["selected_state_key"] == "ansatz_input_state"
    assert sorted(provenance["available_state_keys"]) == ["ansatz_input_state", "initial_state"]
    assert np.allclose(loaded, np.asarray([0.0, 1.0], dtype=complex))


def test_statevector_manifest_rejects_bad_bitstrings() -> None:
    payload = {
        "nq_total": 2,
        "amplitudes_qn_to_q0": {
            "02": {"re": 1.0, "im": 0.0},
        },
    }

    with pytest.raises(ValueError, match="0/1"):
        statevector_from_manifest(payload)


def test_statevector_manifest_rejects_noninteger_nq_total() -> None:
    payload = {
        "nq_total": 1.9,
        "amplitudes_qn_to_q0": {
            "0": {"re": 1.0, "im": 0.0},
        },
    }

    with pytest.raises(ValueError, match="integer"):
        statevector_from_manifest(payload)


def test_serialized_hamiltonian_terms_accumulate_duplicates_and_normalize_uppercase() -> None:
    poly = polynomial_from_serialized_terms(
        [
            {"pauli_exyz": "X", "coeff_re": 1.0, "coeff_im": 0.0},
            {"label_exyz": "x", "coeff": {"re": 2.0, "im": 0.0}},
            {"pauli_exyz": "I", "coeff_re": 0.5, "coeff_im": 0.0},
        ],
        require_real_coefficients=True,
    )

    terms = poly.return_polynomial()
    assert [term.pw2strng() for term in terms] == ["x", "e"]
    assert [complex(term.p_coeff).real for term in terms] == pytest.approx([3.0, 0.5])


def test_load_polynomial_json_accepts_hamiltonian_coefficients_exyz(tmp_path: Path) -> None:
    ham_path = tmp_path / "ham.json"
    ham_path.write_text(
        json.dumps(
            {
                "hamiltonian": {
                    "coefficients_exyz": [
                        {"label_exyz": "Z", "coeff": {"re": 1.0, "im": 0.0}},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    poly, provenance = load_polynomial_json(ham_path)

    assert provenance["source_schema"] == "hamiltonian.coefficients_exyz"
    assert poly.return_polynomial()[0].pw2strng() == "z"


def test_load_polynomial_json_can_rebuild_hh_hamiltonian_from_artifact_settings(tmp_path: Path) -> None:
    artifact_path = tmp_path / "hh_settings_artifact.json"
    artifact_path.write_text(
        json.dumps(
            {
                "settings": {
                    "problem": "hh",
                    "L": 1,
                    "t": 1.0,
                    "u": 2.0,
                    "omega0": 1.0,
                    "g_ep": 0.25,
                    "dv": 0.0,
                    "n_ph_max": 1,
                    "boson_encoding": "binary",
                    "ordering": "blocked",
                    "boundary": "open",
                }
            }
        ),
        encoding="utf-8",
    )

    poly, provenance = load_polynomial_json(artifact_path)

    assert provenance["source_schema"] == "artifact_settings.build_problem_hamiltonian"
    assert provenance["problem_key"] == "hh"
    assert provenance["num_sites"] == 1
    assert provenance["n_ph_max"] == 1
    assert provenance["term_count_output"] == poly.count_number_terms()
    assert poly.count_number_terms() > 0


def test_cli_writes_qse_manifest_for_tiny_problem(tmp_path: Path) -> None:
    ham_path = tmp_path / "one_qubit_ham.json"
    out_path = tmp_path / "nested" / "qse.json"
    ham_path.write_text(
        json.dumps(
            {
                "terms": [
                    {"pauli_exyz": "z", "coeff_re": 1.0, "coeff_im": 0.0},
                    {"pauli_exyz": "x", "coeff_re": 0.25, "coeff_im": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "--hamiltonian-json",
            str(ham_path),
            "--state-bitstring",
            "0",
            "--operator-basis-label",
            "I",
            "--operator-basis-label",
            "X",
            "--output-json",
            str(out_path),
        ]
    )

    assert rc == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "qse_spectra_v1"
    assert data["pipeline"] == "qse_spectra"
    assert data["backend"] == "ideal_statevector"
    assert data["uses_qiskit"] is False
    assert data["diagnostics"]["num_qubits"] == 1
    assert data["diagnostics"]["basis_size"] == 2
    assert data["diagnostics"]["retained_rank"] == 2
    assert len(data["eigenvalues"]) == 2
    assert data["eigenvalues"][0]["energy"] < data["eigenvalues"][1]["energy"]
    assert data["eigenvalues"][0]["energy_relative_to_lowest_qse"] == pytest.approx(0.0)
    assert data["operator_basis"][0]["pauli_exyz"] == "e"
    assert data["operator_basis"][1]["pauli_exyz"] == "x"
    assert data["matrices"]["included"] is True
    assert set(data["matrices"]["overlap"][0][0]) == {"re", "im"}
    assert data["settings"]["basis_vector_policy"]["reference_projection"] == "none"
    assert data["settings"]["basis_vector_policy"]["basis_vector_normalization"] == "normalized"
    assert data["diagnostics"]["basis_vector_policy"] == data["settings"]["basis_vector_policy"]
    assert "transition_observables" not in data
    assert "static_record_selection" not in data
    assert "spectral_functions" not in data
    assert "spectral_window_metrics" not in data
    assert "cutoff_boundary_diagnostics" not in data


def test_cli_paper_iii_mode_emits_projection_diagnostics_and_transition_observable(tmp_path: Path) -> None:
    ham_path = tmp_path / "paper_iii_ham.json"
    out_path = tmp_path / "paper_iii_qse.json"
    ham_path.write_text(
        json.dumps({"terms": [{"pauli_exyz": "z", "coeff_re": 1.0, "coeff_im": 0.0}]}),
        encoding="utf-8",
    )

    assert main(
        [
            "--hamiltonian-json",
            str(ham_path),
            "--state-bitstring",
            "0",
            "--operator-basis-label",
            "I",
            "--operator-basis-label",
            "Z",
            "--operator-basis-label",
            "X",
            "--paper-iii-static-qse-mode",
            "--sector-label",
            "unit_test_sector",
            "--transition-observable-label",
            "dipole=X",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "qse_spectra_v1"
    policy = data["settings"]["basis_vector_policy"]
    assert policy["reference_projection"] == "q0"
    assert policy["basis_vector_normalization"] == "raw_projected"
    assert policy["sector_projection"] == "identity"
    assert policy["sector_label"] == "unit_test_sector"
    assert data["settings"]["paper_iii_static_qse_mode"] is True
    assert data["diagnostics"]["basis_action_norms"] == pytest.approx([1.0, 1.0, 1.0])
    assert data["diagnostics"]["basis_projected_norms"] == pytest.approx([0.0, 0.0, 1.0])
    assert data["diagnostics"]["basis_matrix_vector_norms"] == pytest.approx([0.0, 0.0, 1.0])
    diagnostics = data["diagnostics"]["basis_vector_diagnostics"]
    assert [row["projected_out_by_q0"] for row in diagnostics] == [True, True, False]
    assert [row["zero_vector"] for row in diagnostics] == [True, True, False]
    assert all(row["sector_label"] == "unit_test_sector" for row in diagnostics)
    assert diagnostics[0]["reference_overlap_before_projection"]["re"] == pytest.approx(1.0)
    assert diagnostics[0]["reference_overlap_after_projection_abs"] == pytest.approx(0.0, abs=1e-12)
    assert len(data["transition_observables"]) == 1
    transition = data["transition_observables"][0]
    assert transition["name"] == "dipole"
    assert transition["operator"]["pauli_exyz"] == "x"
    assert transition["transition_vector"][2]["re"] == pytest.approx(1.0)
    assert transition["transition_amplitudes"][0]["strength"] == pytest.approx(1.0)
    assert data["input"]["transition_observables"][0]["source_schema"] == "cli_transition_observable_label"
    assert "static_record_selection" not in data
    assert "spectral_functions" not in data
    assert "spectral_window_metrics" not in data
    assert "cutoff_boundary_diagnostics" not in data


def test_cli_opt_in_writes_spectral_window_and_cutoff_diagnostics(tmp_path: Path) -> None:
    ham_path = tmp_path / "minus_z_ham.json"
    out_path = tmp_path / "paper_iii_p2_qse.json"
    ham_path.write_text(
        json.dumps({"terms": [{"pauli_exyz": "z", "coeff_re": -1.0, "coeff_im": 0.0}]}),
        encoding="utf-8",
    )

    assert main(
        [
            "--hamiltonian-json",
            str(ham_path),
            "--state-bitstring",
            "0",
            "--operator-basis-label",
            "X",
            "--paper-iii-static-qse-mode",
            "--transition-observable-label",
            "dipole=X",
            "--spectral-grid-min",
            "0",
            "--spectral-grid-max",
            "4",
            "--spectral-grid-num",
            "81",
            "--spectral-eta",
            "0.05",
            "--spectral-kernel",
            "lorentzian",
            "--spectral-window",
            "gap:1.5:2.5",
            "--cutoff-boundary-diagnostics",
            "--cutoff-num-sites",
            "1",
            "--cutoff-n-ph-max",
            "1",
            "--cutoff-boson-encoding",
            "binary",
            "--cutoff-fermion-qubits",
            "0",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "qse_spectra_v1"
    assert data["matrices"] == {"included": False}
    spectral = data["spectral_functions"]
    assert spectral["schema_version"] == "qse_spectral_functions_v1"
    assert spectral["controller_boundary"]["feeds_controller_decisions"] is False
    assert spectral["kernel"]["name"] == "lorentzian"
    assert spectral["grid"]["num_points"] == 81
    assert spectral["reference_energy"] == pytest.approx(-1.0)
    observable = spectral["observables"][0]
    assert observable["name"] == "dipole"
    assert observable["roots"][0]["omega"] == pytest.approx(2.0)
    assert observable["peak_omega"] == pytest.approx(2.0)
    assert observable["roots"][0]["transition_strength"] == pytest.approx(1.0)

    windows = data["spectral_window_metrics"]
    assert windows["schema_version"] == "qse_spectral_window_metrics_v1"
    assert windows["controller_boundary"]["feeds_controller_decisions"] is False
    window_metric = windows["observables"][0]["window_metrics"][0]
    assert window_metric["window_name"] == "gap"
    assert window_metric["centroid"] == pytest.approx(2.0, abs=1.0e-12)
    assert window_metric["integrated_weight"] > 0.9

    cutoff = data["cutoff_boundary_diagnostics"]
    assert cutoff["schema_version"] == "qse_cutoff_boundary_diagnostics_v1"
    assert cutoff["controller_boundary"]["feeds_controller_decisions"] is False
    assert cutoff["layout"]["boson_encoding"] == "binary"
    assert cutoff["layout"]["qubits_per_boson_site"] == 1
    assert cutoff["roots"][0]["ell_cut"] == pytest.approx(1.0)
    assert "static_record_selection" not in data


def test_cli_opt_in_static_record_selection_writes_sidecar_payload(tmp_path: Path) -> None:
    ham_path = tmp_path / "two_qubit_ham.json"
    basis_path = tmp_path / "basis_candidates.json"
    out_path = tmp_path / "qse_selected.json"
    ham_path.write_text(
        json.dumps(
            {
                "terms": [
                    {"pauli_exyz": "ze", "coeff_re": 1.0, "coeff_im": 0.0},
                    {"pauli_exyz": "ez", "coeff_re": 0.5, "coeff_im": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    basis_path.write_text(
        json.dumps(
            [
                {
                    "name": "expensive_poly",
                    "kind": "pauli_polynomial",
                    "terms": [
                        {"pauli_exyz": "xx", "coeff_re": 1.0, "coeff_im": 0.0},
                        {"pauli_exyz": "zz", "coeff_re": 1.0, "coeff_im": 0.0},
                        {"pauli_exyz": "ez", "coeff_re": 1.0, "coeff_im": 0.0},
                    ],
                },
                {
                    "name": "q0_flip",
                    "kind": "pauli_string",
                    "pauli_exyz": "ex",
                    "metadata": {"source": "unit_test"},
                },
                {"name": "q1_flip", "kind": "pauli_string", "pauli_exyz": "xe"},
                {"name": "pair_flip", "kind": "pauli_string", "pauli_exyz": "xx"},
            ]
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--hamiltonian-json",
            str(ham_path),
            "--state-bitstring",
            "00",
            "--operator-basis-json",
            str(basis_path),
            "--paper-iii-static-qse-mode",
            "--static-record-selection-mode",
            "cost_proxy",
            "--static-record-selection-max-records",
            "2",
            "--static-record-selection-min-retained-rank",
            "2",
            "--static-record-selection-max-overlap-condition",
            "10",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["diagnostics"]["basis_size"] == 2
    assert data["operator_basis"][0]["name"] == "q0_flip"
    assert data["operator_basis"][1]["name"] == "q1_flip"
    assert data["input"]["operator_basis"]["static_record_selection_enabled"] is True
    assert data["input"]["operator_basis"]["candidate_basis_size"] == 4
    assert data["input"]["operator_basis"]["selected_basis_size"] == 2
    assert data["input"]["operator_basis"]["selected_original_basis_indices"] == [1, 2]

    selection = data["static_record_selection"]
    assert selection["schema_version"] == "qse_static_record_selection_v1"
    assert selection["controller_boundary"]["feeds_controller_decisions"] is False
    assert selection["selection_config"]["mode"] == "cost_proxy"
    assert selection["summary"]["input_basis_size"] == 4
    assert selection["summary"]["selected_basis_size"] == 2
    assert selection["candidates"][0]["features"]["term_count"] == 3
    assert selection["selected_mapping"] == [
        {"original_basis_index": 1, "selected_basis_index": 0},
        {"original_basis_index": 2, "selected_basis_index": 1},
    ]
    assert selection["selected_records"][0]["name"] == "q0_flip"
    assert selection["selected_records"][1]["name"] == "q1_flip"
    post = selection["post_qse_diagnostics"]
    assert post["retained_rank"] == 2
    assert post["basis_vector_zero_count"] == 0
    assert post["guards"]["min_retained_rank"]["passed"] is True
    assert post["guards"]["max_overlap_condition"]["passed"] is True
    assert post["guards"]["all_configured_guards_passed"] is True
    assert "spectral_functions" not in data
    assert "spectral_window_metrics" not in data
    assert "cutoff_boundary_diagnostics" not in data


def test_cli_geometry_selected_static_record_selection_uses_qse_telemetry(tmp_path: Path) -> None:
    ham_path = tmp_path / "geom_ham.json"
    basis_path = tmp_path / "geom_basis.json"
    out_path = tmp_path / "geom_qse.json"
    ham_path.write_text(
        json.dumps(
            {
                "terms": [
                    {"pauli_exyz": "z", "coeff_re": 1.0, "coeff_im": 0.0},
                    {"pauli_exyz": "x", "coeff_re": 0.5, "coeff_im": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    basis_path.write_text(
        json.dumps(
            [
                {"name": "identity", "kind": "pauli_string", "pauli_exyz": "I"},
                {"name": "parallel_z", "kind": "pauli_string", "pauli_exyz": "Z"},
                {"name": "residual_flip", "kind": "pauli_string", "pauli_exyz": "X"},
            ]
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--hamiltonian-json",
            str(ham_path),
            "--state-bitstring",
            "0",
            "--operator-basis-json",
            str(basis_path),
            "--paper-iii-static-qse-mode",
            "--static-record-selection-mode",
            "geometry_selected",
            "--static-record-selection-max-records",
            "1",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["operator_basis"][0]["name"] == "residual_flip"
    selection = data["static_record_selection"]
    assert selection["selection_config"]["mode"] == "geometry_selected"
    assert selection["selected_original_basis_indices"] == [2]
    assert selection["selected_records"][0]["selection_score"] > 0.0
    candidate = selection["candidates"][2]
    assert candidate["geometry"]["metric_novelty_fraction"] == pytest.approx(1.0)
    assert candidate["geometry"]["residual_capture"] > 0.0


def test_cli_rejects_static_record_selection_partial_flags(tmp_path: Path) -> None:
    ham_path = tmp_path / "ham.json"
    ham_path.write_text(
        json.dumps({"terms": [{"pauli_exyz": "z", "coeff_re": 1.0, "coeff_im": 0.0}]}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--hamiltonian-json",
                str(ham_path),
                "--state-bitstring",
                "0",
                "--static-record-selection-max-records",
                "1",
            ]
        )
    assert exc_info.value.code == 2

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--hamiltonian-json",
                str(ham_path),
                "--state-bitstring",
                "0",
                "--static-record-selection-mode",
                "input_order",
            ]
        )
    assert exc_info.value.code == 2


def test_cli_rejects_spectral_mode_without_transition_observable(tmp_path: Path) -> None:
    ham_path = tmp_path / "ham.json"
    ham_path.write_text(
        json.dumps({"terms": [{"pauli_exyz": "z", "coeff_re": -1.0, "coeff_im": 0.0}]}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--hamiltonian-json",
                str(ham_path),
                "--state-bitstring",
                "0",
                "--operator-basis-label",
                "X",
                "--paper-iii-static-qse-mode",
                "--spectral-grid-min",
                "0",
                "--spectral-grid-max",
                "4",
                "--spectral-grid-num",
                "81",
                "--spectral-eta",
                "0.05",
            ]
        )

    assert exc_info.value.code == 2


def test_cli_rejects_cutoff_diagnostics_without_explicit_layout(tmp_path: Path) -> None:
    ham_path = tmp_path / "ham.json"
    ham_path.write_text(
        json.dumps({"terms": [{"pauli_exyz": "z", "coeff_re": -1.0, "coeff_im": 0.0}]}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--hamiltonian-json",
                str(ham_path),
                "--state-bitstring",
                "0",
                "--cutoff-boundary-diagnostics",
                "--cutoff-num-sites",
                "1",
                "--cutoff-n-ph-max",
                "1",
            ]
        )

    assert exc_info.value.code == 2


def test_cli_rejects_paper_iii_mode_with_conflicting_policy_override(tmp_path: Path) -> None:
    ham_path = tmp_path / "ham.json"
    ham_path.write_text(
        json.dumps({"terms": [{"pauli_exyz": "z", "coeff_re": 1.0, "coeff_im": 0.0}]}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--hamiltonian-json",
                str(ham_path),
                "--state-bitstring",
                "0",
                "--operator-basis-label",
                "X",
                "--paper-iii-static-qse-mode",
                "--reference-projection",
                "none",
            ]
        )

    assert exc_info.value.code == 2


def test_cli_rejects_q0_projection_without_explicit_basis(tmp_path: Path) -> None:
    ham_path = tmp_path / "ham.json"
    ham_path.write_text(
        json.dumps({"terms": [{"pauli_exyz": "z", "coeff_re": 1.0, "coeff_im": 0.0}]}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--hamiltonian-json",
                str(ham_path),
                "--state-bitstring",
                "0",
                "--reference-projection",
                "q0",
            ]
        )

    assert exc_info.value.code == 2


def test_cli_accepts_operator_basis_json_with_polynomial_record(tmp_path: Path) -> None:
    ham_path = tmp_path / "ham.json"
    basis_path = tmp_path / "basis.json"
    out_path = tmp_path / "qse_basis_json.json"
    ham_path.write_text(
        json.dumps({"terms": [{"pauli_exyz": "z", "coeff_re": 1.0, "coeff_im": 0.0}]}),
        encoding="utf-8",
    )
    basis_path.write_text(
        json.dumps(
            [
                {"name": "identity", "kind": "pauli_string", "pauli_exyz": "I"},
                {
                    "name": "x_poly",
                    "kind": "pauli_polynomial",
                    "terms": [{"pauli_exyz": "X", "coeff_re": 1.0, "coeff_im": 0.0}],
                },
            ]
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--hamiltonian-json",
            str(ham_path),
            "--state-bitstring",
            "0",
            "--operator-basis-json",
            str(basis_path),
            "--output-json",
            str(out_path),
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["diagnostics"]["retained_rank"] == 2
    assert data["operator_basis"][0]["pauli_exyz"] == "e"
    assert data["operator_basis"][1]["terms"][0]["pauli_exyz"] == "x"
    assert [row["energy"] for row in data["eigenvalues"]] == pytest.approx([-1.0, 1.0])


def test_cli_can_build_basis_from_selected_adapt_blocks(tmp_path: Path) -> None:
    artifact_path = tmp_path / "adapt_artifact.json"
    out_path = tmp_path / "selected_basis_qse.json"
    artifact_path.write_text(
        json.dumps(
            {
                "hamiltonian": {
                    "num_qubits": 1,
                    "coefficients_exyz": [
                        {"label_exyz": "z", "coeff": {"re": 1.0, "im": 0.0}},
                    ],
                },
                "initial_state": {
                    "nq_total": 1,
                    "amplitudes_qn_to_q0": {"0": {"re": 1.0, "im": 0.0}},
                },
                "adapt_vqe": {
                    "parameterization": {
                        "blocks": [
                            {
                                "candidate_label": "unit_test_flip",
                                "runtime_terms_exyz": [
                                    {"pauli_exyz": "x", "coeff_re": 1.0, "coeff_im": 0.0, "nq": 1}
                                ],
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--hamiltonian-json",
            str(artifact_path),
            "--state-json",
            str(artifact_path),
            "--operator-basis-source",
            "selected_adapt_blocks",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["input"]["operator_basis"]["source_schema"] == "artifact_basis_source:selected_adapt_blocks"
    assert data["diagnostics"]["basis_size"] == 2
    assert data["operator_basis"][0]["pauli_exyz"] == "e"
    assert data["operator_basis"][1]["name"] == "adapt::unit_test_flip"
    assert [row["energy"] for row in data["eigenvalues"]] == pytest.approx([-1.0, 1.0])


def test_cli_can_build_basis_from_legacy_adapt_operator_labels(tmp_path: Path) -> None:
    artifact_path = tmp_path / "legacy_adapt_artifact.json"
    out_path = tmp_path / "legacy_selected_basis_qse.json"
    artifact_path.write_text(
        json.dumps(
            {
                "hamiltonian": {
                    "num_qubits": 1,
                    "coefficients_exyz": [
                        {"label_exyz": "z", "coeff": {"re": 1.0, "im": 0.0}},
                    ],
                },
                "initial_state": {
                    "nq_total": 1,
                    "amplitudes_qn_to_q0": {"0": {"re": 1.0, "im": 0.0}},
                },
                "adapt_vqe": {
                    "operators": [
                        "unit_test_flip(X)",
                        "duplicate_unit_test_flip(x)",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--hamiltonian-json",
            str(artifact_path),
            "--state-json",
            str(artifact_path),
            "--operator-basis-source",
            "selected_adapt_blocks",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["input"]["operator_basis"]["source_schema"] == "artifact_basis_source:selected_adapt_blocks"
    assert data["input"]["operator_basis"]["adapt_operator_count"] == 2
    assert data["input"]["operator_basis"]["adapt_operator_records_emitted"] == 1
    assert data["diagnostics"]["basis_size"] == 2
    assert data["operator_basis"][0]["pauli_exyz"] == "e"
    assert data["operator_basis"][1]["pauli_exyz"] == "x"
    assert data["operator_basis"][1]["name"] == "adapt::unit_test_flip(X)"
    assert [row["energy"] for row in data["eigenvalues"]] == pytest.approx([-1.0, 1.0])


def test_artifact_full_meta_basis_source_builds_logical_polynomial_elements(tmp_path: Path) -> None:
    artifact_path = tmp_path / "hh_artifact.json"
    artifact_path.write_text(
        json.dumps(
            {
                "settings": {
                    "L": 2,
                    "t": 1.0,
                    "u": 4.0,
                    "omega0": 1.0,
                    "g_ep": 0.5,
                    "dv": 0.0,
                    "n_ph_max": 1,
                    "boson_encoding": "binary",
                    "ordering": "blocked",
                    "boundary": "open",
                    "paop_r": 1,
                    "paop_split_paulis": False,
                    "paop_prune_eps": 0.0,
                    "paop_normalization": "none",
                },
                "hamiltonian": {
                    "num_qubits": 6,
                    "coefficients_exyz": [
                        {"label_exyz": "eeeeee", "coeff": {"re": 4.0, "im": 0.0}},
                        {"label_exyz": "eeeexx", "coeff": {"re": -0.5, "im": 0.0}},
                    ],
                },
                "adapt_vqe": {"num_particles": {"n_up": 1, "n_dn": 1}},
            }
        ),
        encoding="utf-8",
    )
    hamiltonian, _ = load_polynomial_json(artifact_path)

    basis, provenance = basis_elements_from_artifact_source(
        artifact_path,
        nq=6,
        hamiltonian=hamiltonian,
        source="full_meta_filtered",
        full_meta_keep_classes="uccsd_sing,uccsd_dbl",
    )

    assert provenance["source_schema"] == "artifact_basis_source:full_meta_filtered"
    assert provenance["full_meta_keep_classes"] == ["uccsd_sing", "uccsd_dbl"]
    assert basis[0].kind == "pauli_string"
    assert basis[0].pauli_label_exyz == "eeeeee"
    assert any(element.kind == "pauli_polynomial" for element in basis[1:])
    assert provenance["basis_size"] == len(basis)


def test_cli_can_build_hamiltonian_term_basis_source(tmp_path: Path) -> None:
    artifact_path = tmp_path / "ham_terms_artifact.json"
    out_path = tmp_path / "ham_terms_qse.json"
    artifact_path.write_text(
        json.dumps(
            {
                "hamiltonian": {
                    "num_qubits": 2,
                    "coefficients_exyz": [
                        {"label_exyz": "ee", "coeff": {"re": 0.25, "im": 0.0}},
                        {"label_exyz": "ze", "coeff": {"re": -1.0, "im": 0.0}},
                        {"label_exyz": "xx", "coeff": {"re": 0.1, "im": 0.0}},
                    ],
                },
                "initial_state": {
                    "nq_total": 2,
                    "amplitudes_qn_to_q0": {"00": {"re": 1.0, "im": 0.0}},
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--hamiltonian-json",
            str(artifact_path),
            "--state-json",
            str(artifact_path),
            "--operator-basis-source",
            "hamiltonian_terms",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["input"]["operator_basis"]["source_schema"] == "artifact_basis_source:hamiltonian_terms"
    assert data["input"]["operator_basis"]["hamiltonian_term_basis_records_emitted"] == 2
    assert data["operator_basis"][0]["pauli_exyz"] == "ee"
    assert {row["pauli_exyz"] for row in data["operator_basis"][1:]} == {"ze", "xx"}


def test_cli_can_omit_matrices(tmp_path: Path) -> None:
    ham_path = tmp_path / "ham.json"
    out_path = tmp_path / "qse_no_matrices.json"
    ham_path.write_text(
        json.dumps({"terms": [{"pauli_exyz": "z", "coeff_re": 1.0, "coeff_im": 0.0}]}),
        encoding="utf-8",
    )

    assert main(
        [
            "--hamiltonian-json",
            str(ham_path),
            "--state-bitstring",
            "0",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["matrices"] == {"included": False}
    assert data["schema_version"] == "qse_spectra_v1"
    assert data["settings"]["basis_vector_policy"]["reference_projection"] == "none"
    assert data["settings"]["basis_vector_policy"]["basis_vector_normalization"] == "normalized"
    assert "transition_observables" not in data
    assert "static_record_selection" not in data
    assert "spectral_functions" not in data
    assert "spectral_window_metrics" not in data
    assert "cutoff_boundary_diagnostics" not in data

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.__main__ import main as qse_main
from pipelines.qse_spectra.exact_reference import main as exact_reference_main
from pipelines.qse_spectra.spectral_functions import load_spectral_references_json
from pipelines.qse_spectra.table_aggregate import summarize_qse_manifest


def test_exact_reference_cli_writes_loadable_spectral_reference(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    reference_path = tmp_path / "exact_reference.json"
    artifact_path.write_text(
        json.dumps(
            {
                "hamiltonian": {
                    "coefficients_exyz": [
                        {"pauli_exyz": "z", "coeff_re": -1.0, "coeff_im": 0.0},
                    ]
                },
                "initial_state": {
                    "nq_total": 1,
                    "amplitudes_qn_to_q0": {"0": {"re": 1.0, "im": 0.0}},
                },
            }
        ),
        encoding="utf-8",
    )

    assert exact_reference_main(
        [
            "--hamiltonian-json",
            str(artifact_path),
            "--state-json",
            str(artifact_path),
            "--transition-observable-label",
            "probe=X",
            "--spectral-grid-min",
            "0",
            "--spectral-grid-max",
            "4",
            "--spectral-grid-num",
            "201",
            "--spectral-eta",
            "0.05",
            "--output-json",
            str(reference_path),
        ]
    ) == 0

    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "qse_exact_spectral_reference_v1"
    assert payload["controller_boundary"]["feeds_controller_decisions"] is False
    assert payload["diagnostics"]["num_qubits"] == 1
    assert payload["diagnostics"]["reference_energy"] == pytest.approx(-1.0)
    assert payload["references"][0]["observable_name"] == "probe"
    refs = load_spectral_references_json(reference_path)
    assert len(refs) == 1
    peak_idx = max(range(len(refs[0].values)), key=lambda idx: refs[0].values[idx])
    assert refs[0].grid[peak_idx] == pytest.approx(2.0)


def test_qse_spectral_reference_flows_into_table_aggregate(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    reference_path = tmp_path / "exact_reference.json"
    qse_path = tmp_path / "qse.json"
    artifact_path.write_text(
        json.dumps(
            {
                "hamiltonian": {
                    "coefficients_exyz": [
                        {"pauli_exyz": "z", "coeff_re": -1.0, "coeff_im": 0.0},
                    ]
                },
                "initial_state": {
                    "nq_total": 1,
                    "amplitudes_qn_to_q0": {"0": {"re": 1.0, "im": 0.0}},
                },
            }
        ),
        encoding="utf-8",
    )
    assert exact_reference_main(
        [
            "--hamiltonian-json",
            str(artifact_path),
            "--state-json",
            str(artifact_path),
            "--transition-observable-label",
            "probe=X",
            "--spectral-grid-min",
            "0",
            "--spectral-grid-max",
            "4",
            "--spectral-grid-num",
            "201",
            "--spectral-eta",
            "0.05",
            "--output-json",
            str(reference_path),
        ]
    ) == 0

    assert qse_main(
        [
            "--hamiltonian-json",
            str(artifact_path),
            "--state-json",
            str(artifact_path),
            "--operator-basis-label",
            "X",
            "--paper-iii-static-qse-mode",
            "--transition-observable-label",
            "probe=X",
            "--spectral-grid-min",
            "0",
            "--spectral-grid-max",
            "4",
            "--spectral-grid-num",
            "201",
            "--spectral-eta",
            "0.05",
            "--spectral-window",
            "main:0:4",
            "--spectral-reference-json",
            str(reference_path),
            "--output-json",
            str(qse_path),
            "--omit-matrices",
        ]
    ) == 0

    row = summarize_qse_manifest(qse_path)
    assert row["spectral_reference_l2_error_max"] == pytest.approx(0.0, abs=1.0e-12)
    assert row["spectral_reference_l1_error_max"] == pytest.approx(0.0, abs=1.0e-12)

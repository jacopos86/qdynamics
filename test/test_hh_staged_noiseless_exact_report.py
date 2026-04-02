from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_staged_noiseless_exact_report import (
    load_noiseless_series,
    write_noiseless_exact_report,
)


def _write_payload(path: Path) -> None:
    payload = {
        "pipeline": "hh_staged_noiseless",
        "settings": {
            "physics": {
                "L": 2,
                "t": 1.0,
                "u": 2.0,
                "dv": 0.0,
                "omega0": 1.0,
                "g_ep": 1.0,
                "n_ph_max": 1,
                "boundary": "periodic",
                "ordering": "blocked",
            },
            "dynamics": {
                "enable_drive": True,
                "methods": ["cfqm4"],
                "t_final": 1.0,
                "num_times": 3,
                "trotter_steps": 8,
                "exact_steps_multiplier": 2,
                "drive_pattern": "staggered",
                "drive_A": 0.6,
                "drive_omega": 1.0,
                "drive_tbar": 1.0,
                "drive_phi": 0.0,
                "drive_t0": 0.0,
            },
        },
        "dynamics_noiseless": {
            "profiles": {
                "drive": {
                    "times": [0.0, 0.5, 1.0],
                    "methods": {
                        "cfqm4": {
                            "trajectory": [
                                {
                                    "time": 0.0,
                                    "fidelity": 1.0,
                                    "energy_total_exact": 0.80,
                                    "energy_total_trotter": 0.80,
                                    "staggered_exact": 0.30,
                                    "staggered_trotter": 0.30,
                                    "doublon_exact": 0.50,
                                    "doublon_trotter": 0.50,
                                },
                                {
                                    "time": 0.5,
                                    "fidelity": 0.999,
                                    "energy_total_exact": 0.90,
                                    "energy_total_trotter": 0.89,
                                    "staggered_exact": 0.05,
                                    "staggered_trotter": 0.04,
                                    "doublon_exact": 0.45,
                                    "doublon_trotter": 0.44,
                                },
                                {
                                    "time": 1.0,
                                    "fidelity": 0.997,
                                    "energy_total_exact": 0.85,
                                    "energy_total_trotter": 0.84,
                                    "staggered_exact": -0.10,
                                    "staggered_trotter": -0.11,
                                    "doublon_exact": 0.40,
                                    "doublon_trotter": 0.41,
                                },
                            ]
                        }
                    },
                }
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_noiseless_series_and_write_pdf(tmp_path: Path) -> None:
    input_json = tmp_path / "workflow.json"
    output_pdf = tmp_path / "workflow.pdf"
    _write_payload(input_json)

    series = load_noiseless_series(json.loads(input_json.read_text()), profile="drive", method="cfqm4")
    assert series.times.tolist() == [0.0, 0.5, 1.0]
    assert series.fidelity.tolist() == [1.0, 0.999, 0.997]
    assert series.abs_energy_total_error.tolist() == pytest.approx([0.0, 0.01, 0.01])

    result = write_noiseless_exact_report(
        input_json=input_json,
        output_pdf=output_pdf,
        profile="drive",
        method="cfqm4",
        run_command="python report.py --input-json workflow.json",
    )

    assert result == output_pdf
    assert output_pdf.exists()
    assert output_pdf.read_bytes().startswith(b"%PDF")
    assert output_pdf.stat().st_size > 0

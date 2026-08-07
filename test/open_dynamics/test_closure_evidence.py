from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from paper5.stability import DimerParameters
from paper5.stability.electron_phonon_analysis import analyze_matched_case
from pipelines.open_dynamics.closure_evidence import (
    EVIDENCE_SCHEMA_VERSION,
    EvidenceSourceError,
    analyze_closure_evidence,
    write_closure_evidence,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@pytest.fixture()
def verified_source_run(tmp_path: Path) -> Path:
    run_directory = tmp_path / "source"
    run_directory.mkdir()
    parameters = DimerParameters(
        lambda_ep=0.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    case = analyze_matched_case(
        parameters,
        final_time=0.1,
        time_step=0.05,
        phonon_cutoff=2,
        activation_margin=1.0e-5,
        exact_relative_tolerance=1.0e-10,
        exact_absolute_tolerance=1.0e-12,
        exact_maximum_step=0.05,
    )
    key = "lambda_0p5__gamma_0p5__drive_1"
    arrays_path = run_directory / "parameter_grid_trajectories.npz"
    np.savez_compressed(
        arrays_path,
        **{
            f"{key}__times": case.raw.times,
            f"{key}__exact": case.exact_coordinates,
            f"{key}__raw": case.raw.coordinates,
            f"{key}__corrected": case.corrected.coordinates,
        },
    )
    plan_path = run_directory / "plan.json"
    _write_json(
        plan_path,
        {
            "run_id": "synthetic_verified_source",
            "baseline_parameters": {
                "hopping": 1.0,
                "pulse_width": 1.0,
            },
            "integration": {
                "exact_relative_tolerance": 1.0e-10,
                "exact_absolute_tolerance": 1.0e-12,
                "exact_maximum_step": 0.05,
            },
            "parameter_grid": {
                "lambda_ep": [0.5],
                "gamma": [0.5],
                "drive_amplitude": [1.0],
                "time_step": 0.05,
            },
        },
    )
    summary_path = run_directory / "summary.json"
    _write_json(
        summary_path,
        {
            "status": "complete",
            "scientific_question": "synthetic test source",
        },
    )
    _write_json(
        run_directory / "runtime_manifest.json",
        {
            "status": "complete",
            "run_id": "synthetic_verified_source",
            "evidence_status": "exploratory_local_not_promoted",
            "exact_reference_usage": "never used by autonomous decisions",
            "artifact_hashes": {
                plan_path.name: _sha256(plan_path),
                summary_path.name: _sha256(summary_path),
                arrays_path.name: _sha256(arrays_path),
            },
        },
    )
    return run_directory


def test_analyzer_adds_only_available_coherent_fields(
    verified_source_run: Path,
) -> None:
    result = analyze_closure_evidence(verified_source_run)

    assert result.summary["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert result.summary["aggregate"]["case_count"] == 1
    case = result.summary["cases"][0]
    assert set(case["methods"]) == {
        "coherent_only",
        "raw_closure",
        "gram_corrected_closure",
    }
    key = case["case_id"]
    assert f"{key}__coherent_electron_1rdm" in result.arrays
    assert f"{key}__coherent_coherent_phonon" in result.arrays
    assert f"{key}__coherent_joint_gram_minimum" not in result.arrays
    assert np.isfinite(
        case["comparisons"]["raw_to_coherent_rho_rms_ratio"]
    )
    assert (
        case["methods"]["coherent_only"]["step_refinement"][
            "maximum_electron_1rdm_frobenius_difference"
        ]
        < 1.0e-8
    )


def test_analyzer_rejects_a_tampered_source_hash(
    verified_source_run: Path,
) -> None:
    summary_path = verified_source_run / "summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(EvidenceSourceError, match="source hash mismatch"):
        analyze_closure_evidence(verified_source_run)


def test_writer_records_artifact_hashes_and_refuses_overwrite(
    verified_source_run: Path,
    tmp_path: Path,
) -> None:
    result = analyze_closure_evidence(verified_source_run)
    destination = tmp_path / "evidence"
    write_closure_evidence(result, destination)

    for name in (
        "summary.json",
        "matched_trajectories.npz",
        "source_point_trajectory.pdf",
        "grid_method_comparison.pdf",
        "gram_severity_vs_error.pdf",
        "runtime_manifest.json",
    ):
        assert (destination / name).is_file()
    runtime = json.loads(
        (destination / "runtime_manifest.json").read_text(encoding="utf-8")
    )
    assert runtime["status"] == "complete"
    assert runtime["artifact_hashes"]["summary.json"] == _sha256(
        destination / "summary.json"
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_closure_evidence(result, destination)

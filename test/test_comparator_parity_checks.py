#!/usr/bin/env python3
"""Tests for read-only artifact-level comparator parity checks."""

from __future__ import annotations

import json
from pathlib import Path

from pipelines.exact_bench.comparator_parity_checks import (
    build_static_row_parity_sidecar,
    load_row_artifact,
    main,
)


def _append_row(*, energy: float, selected: list[str], count_2q: int = 4) -> dict[str, object]:
    return {
        "algorithm_id": "static_full_meta_append_adapt_vqe",
        "energy": energy,
        "selected_operators": list(selected),
        "compiled_count_2q_total": count_2q,
        "compiled_depth_total": 12,
        "state_fidelity": 0.999999999,
    }


def _qiskit_row(*, energy: float, selected: list[str], count_2q: int = 4) -> dict[str, object]:
    return {
        "algorithm_id": "static_qiskit_adapt_vqe",
        "energy": energy,
        "qiskit_selected_operator_labels": list(selected),
        "compiled_count_2q_total": count_2q,
        "compiled_depth_total": 12,
        "state_fidelity": 0.999999999,
    }


def test_static_row_parity_sidecar_passes_when_common_quantities_match() -> None:
    payload = build_static_row_parity_sidecar(
        algorithm_id="static_full_meta_append_adapt_vqe",
        subject_row=_append_row(energy=-1.0, selected=["a", "b"]),
        reference_row=_qiskit_row(energy=-1.0 + 1.0e-10, selected=["a", "b"]),
        parity_reference_algorithm_id="static_qiskit_adapt_vqe",
        energy_tolerance=1.0e-8,
    )

    assert payload["parity_status"] == "passed"
    assert 0.0 < payload["parity_energy_abs_delta"] < 1.0e-8
    assert payload["parity_selected_generators_match"] is True
    assert payload["parity_compiled_cost_match"] is True
    assert payload["extra"]["failed_quantities"] == []


def test_static_row_parity_sidecar_fails_on_sequence_or_resource_mismatch() -> None:
    payload = build_static_row_parity_sidecar(
        algorithm_id="static_full_meta_append_adapt_vqe",
        subject_row=_append_row(energy=-1.0, selected=["a", "b"], count_2q=4),
        reference_row=_qiskit_row(energy=-1.0, selected=["a", "c"], count_2q=5),
        parity_reference_algorithm_id="static_qiskit_adapt_vqe",
    )

    assert payload["parity_status"] == "failed"
    assert payload["parity_selected_generators_match"] is False
    assert payload["parity_compiled_cost_match"] is False
    assert payload["extra"]["failed_quantities"] == ["selected_sequence", "compiled_cost"]


def test_static_row_parity_sidecar_records_missing_quantities_as_partial() -> None:
    payload = build_static_row_parity_sidecar(
        algorithm_id="static_family_informed_vqe",
        subject_row={"algorithm_id": "static_family_informed_vqe", "energy": -2.0},
        reference_row={"algorithm_id": "qiskit_fixed_ansatz_evaluator", "energy": -2.0},
        parity_reference_algorithm_id="qiskit_fixed_ansatz_evaluator",
    )

    assert payload["parity_status"] == "partial_common_quantities_passed"
    assert payload["parity_energy_abs_delta"] == 0.0
    assert set(payload["extra"]["missing_quantities"]) == {
        "selected_sequence",
        "compiled_cost",
        "state_infidelity",
    }


def test_load_row_artifact_accepts_result_or_rows_payload(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    rows_path = tmp_path / "rows.json"
    result_path.write_text(json.dumps({"result": {"energy": -1.0}}), encoding="utf-8")
    rows_path.write_text(json.dumps({"rows": [{"energy": -2.0}]}), encoding="utf-8")

    assert load_row_artifact(result_path)["energy"] == -1.0
    assert load_row_artifact(rows_path)["energy"] == -2.0


def test_parity_check_cli_writes_sidecar_only(tmp_path: Path, capsys) -> None:
    subject_path = tmp_path / "subject.json"
    reference_path = tmp_path / "reference.json"
    out_dir = tmp_path / "parity"
    subject_path.write_text(json.dumps({"result": _append_row(energy=-1.0, selected=["a"])}), encoding="utf-8")
    reference_path.write_text(json.dumps({"result": _qiskit_row(energy=-1.0, selected=["a"])}), encoding="utf-8")

    rc = main(
        [
            "--subject",
            str(subject_path),
            "--reference",
            str(reference_path),
            "--algorithm-id",
            "static_full_meta_append_adapt_vqe",
            "--parity-reference-algorithm-id",
            "static_qiskit_adapt_vqe",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    sidecar = out_dir / "comparator_parity_sidecar.json"
    assert sidecar.exists()
    loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    assert loaded["parity_status"] == "passed"
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["algorithm_id"] == "static_full_meta_append_adapt_vqe"

"""Qiskit-free terminal logical-resource artifact contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.diagnostics.terminal_logical_graph_resources import (
    TERMINAL_LOGICAL_GRAPH_RESOURCE_FILENAME,
    build_terminal_logical_graph_resource_record,
    load_terminal_resources,
    write_terminal_logical_graph_resource_record,
)


def _write_run(path: Path, *, logical_shared: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if logical_shared:
        coordinates = [
            {
                "label": "macro-0",
                "runtime_label": "macro-0",
                "execution_mode": "termwise_product",
                "rotations": [
                    {"pauli_exyz": "xeee", "coeff_real": 1.0, "nq": 4},
                    {"pauli_exyz": "eexx", "coeff_real": -0.5, "nq": 4},
                ],
            }
        ]
        resume = {
            "schema": "paper_ii_resume_state_v2",
            "time": 5.0,
            "problem_family": "hubbard",
            "parameterization_mode": "logical_shared",
            "layout": {
                "term_order": "native",
                "ignore_identity": True,
                "coefficient_tolerance": 1.0e-12,
            },
            "coordinates": coordinates,
            "theta_runtime": [0.25],
            "theta_logical": [0.25],
        }
        selected = ["macro-0"]
        initial_state = {
            "parameterization_mode": "logical_shared",
            "runtime_parameter_count": 1,
            "runtime_coordinate_labels": ["macro-0"],
        }
    else:
        coordinates = [
            {
                "label": "seed::r0::xeee",
                "pauli_exyz": "xeee",
                "coeff_real": 1.0,
                "nq": 4,
            },
            {
                "label": "insert::r0::eexx",
                "pauli_exyz": "eexx",
                "coeff_real": -0.5,
                "nq": 4,
            },
        ]
        resume = {
            "schema": "paper_ii_resume_state_v1",
            "time": 5.0,
            "parameterization_mode": "per_pauli_term",
            "coordinates": coordinates,
            "theta_runtime": [0.25, -0.1],
        }
        selected = [coordinate["label"] for coordinate in coordinates]
        initial_state = {
            "parameterization_mode": "per_pauli_term",
            "runtime_parameter_count": 1,
            "runtime_coordinate_labels": ["seed::r0::xeee"],
        }
    path.write_text(
        json.dumps(
            {
                "run_lock": {"physics": {"family_key": "hubbard"}},
                "resume_state": resume,
                "initial_state": initial_state,
                "final_state": {
                    "parameterization_mode": resume["parameterization_mode"],
                    "runtime_parameter_count": len(coordinates),
                    "selected_term_labels": selected,
                },
                "summary": {"runtime_parameter_count_final": len(coordinates)},
                "plot_rows": [{"time": 0.0}, {"time": 5.0}],
            }
        ),
        encoding="utf-8",
    )


def _write_legacy(run_path: Path, *, terminal_theta: int = 2) -> Path:
    path = run_path.with_name("terminal_qiskit_cost.json")
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "graph_proxy_terminal_c_hat_2q": 102,
                        "graph_proxy_terminal_c_hat_d": 82,
                        "graph_proxy_terminal_c_hat_1q": 122,
                        "graph_proxy_terminal_theta_count": terminal_theta,
                        "graph_proxy_delta_c_hat_2q": 12,
                        "graph_proxy_delta_c_hat_d": 9,
                        "graph_proxy_delta_c_hat_1q": 14,
                        "graph_proxy_delta_theta_count": 1,
                        "terminal_reconstruction_parity": {"passed": True},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_builder_prices_serialized_coordinates_without_compilation(tmp_path) -> None:
    run_path = tmp_path / "run.json"
    _write_run(run_path)

    record = build_terminal_logical_graph_resource_record(run_path)

    assert record["terminal_resource_tuple"] == {
        "C_hat_2q": 2,
        "C_hat_d": 2,
        "C_hat_1q": 8,
        "N_theta": 2,
    }
    assert record["initial_resource_tuple"] is None
    assert record["terminal_minus_initial_resource_tuple"] is None
    assert record["qiskit_compilation"]["required"] is False
    assert record["qiskit_compilation"]["performed"] is False


def test_logical_shared_n_theta_counts_macros_not_pauli_rotations(tmp_path) -> None:
    run_path = tmp_path / "run.json"
    _write_run(run_path, logical_shared=True)

    record = build_terminal_logical_graph_resource_record(run_path)

    assert record["terminal_resource_tuple"]["N_theta"] == 1
    assert record["terminal_graph_span_ledger"][
        "nonidentity_pauli_rotation_count"
    ] == 2
    assert record["terminal_minus_initial_resource_tuple"] is None


def test_loader_prefers_canonical_and_binds_it_to_the_run(tmp_path) -> None:
    run_path = tmp_path / "run.json"
    _write_run(run_path)
    canonical = run_path.with_name(TERMINAL_LOGICAL_GRAPH_RESOURCE_FILENAME)
    write_terminal_logical_graph_resource_record(
        canonical, build_terminal_logical_graph_resource_record(run_path)
    )
    _write_legacy(run_path)

    loaded = load_terminal_resources(run_path)

    assert loaded.source_kind == "canonical_logical_graph_resource"
    assert loaded.artifact_path == canonical
    assert loaded.n_theta == 2
    assert loaded.graph == (2, 2, 8)
    assert loaded.delta_graph is None
    assert loaded.delta_n_theta is None


def test_present_but_corrupt_canonical_never_falls_back(tmp_path) -> None:
    run_path = tmp_path / "run.json"
    _write_run(run_path)
    canonical = run_path.with_name(TERMINAL_LOGICAL_GRAPH_RESOURCE_FILENAME)
    record = build_terminal_logical_graph_resource_record(run_path)
    record["terminal_resource_tuple"]["N_theta"] = 99
    canonical.write_text(json.dumps(record), encoding="utf-8")
    _write_legacy(run_path)

    with pytest.raises(ValueError, match="digest mismatch"):
        load_terminal_resources(run_path)


def test_legacy_fallback_is_explicit_and_normalized(tmp_path) -> None:
    run_path = tmp_path / "run.json"
    _write_run(run_path)
    legacy = _write_legacy(run_path)

    loaded = load_terminal_resources(run_path)

    assert loaded.source_kind == "legacy_terminal_qiskit_cost_fallback"
    assert loaded.artifact_path == legacy
    assert loaded.n_theta == 2
    assert loaded.graph == (102, 82, 122)
    assert loaded.delta_graph == (12, 9, 14)
    with pytest.raises(FileNotFoundError, match="missing canonical"):
        load_terminal_resources(run_path, allow_legacy_qiskit_fallback=False)


def test_immutable_writer_rejects_replacement(tmp_path) -> None:
    run_path = tmp_path / "run.json"
    _write_run(run_path)
    destination = tmp_path / TERMINAL_LOGICAL_GRAPH_RESOURCE_FILENAME
    record = build_terminal_logical_graph_resource_record(run_path)
    write_terminal_logical_graph_resource_record(destination, record)
    write_terminal_logical_graph_resource_record(destination, record)

    altered = dict(record)
    altered["resource_sha256"] = "0" * 64
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_terminal_logical_graph_resource_record(destination, altered)

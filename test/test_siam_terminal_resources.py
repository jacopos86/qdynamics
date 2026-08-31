"""Canonical SIAM terminal resources do not depend on Qiskit compilation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from pipelines.time_dynamics.ap_mclachlan.state import runtime_coordinate_labels
from pipelines.time_dynamics.diagnostics.siam_terminal_resources import (
    FAKE_MARRAKESH_UNDIRECTED_EDGES_SHA256_V1,
    FAKE_MARRAKESH_UNDIRECTED_EDGES_V1,
    SIAM_TERMINAL_RESOURCE_V1,
    build_siam_terminal_resource_record,
    write_siam_terminal_resource_record,
)
from pipelines.time_dynamics.resume import RESUME_SCHEMA_V2
from pipelines.time_dynamics.siam_screening import FIXTURE_NB2
from pipelines.time_dynamics.siam_seed import (
    load_siam_runtime_input,
    write_siam_seed_manifest,
)
from src.quantum.ansatz_parameterization import (
    build_parameter_layout,
    iter_runtime_rotation_terms,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trajectory_fixture(tmp_path: Path) -> tuple[Path, Path]:
    seed = tmp_path / "siam_seed.json"
    write_siam_seed_manifest(seed, FIXTURE_NB2)
    base = load_siam_runtime_input(seed)
    source_term = base.candidate_pool_terms[0]
    terminal_term = replace(source_term, label=f"{source_term.label}::insr0c0o0")
    layout = build_parameter_layout(
        (terminal_term,),
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    rotations = [
        {
            "pauli_exyz": str(spec.pauli_exyz),
            "coeff_real": float(spec.coeff_real),
            "nq": int(spec.nq),
        }
        for spec in iter_runtime_rotation_terms(
            terminal_term.polynomial,
            ignore_identity=True,
            coefficient_tolerance=1.0e-12,
            sort_terms=True,
        )
    ]
    labels = runtime_coordinate_labels(
        layout,
        parameterization_mode="logical_shared",
    )
    resume = {
        "schema": RESUME_SCHEMA_V2,
        "time": 0.5,
        "seed_artifact_json": str(seed),
        "problem_family": "siam",
        "parameterization_mode": "logical_shared",
        "layout": {
            "term_order": "sorted",
            "ignore_identity": True,
            "coefficient_tolerance": 1.0e-12,
        },
        "coordinates": [
            {
                "label": terminal_term.label,
                "runtime_label": labels[0],
                "execution_mode": terminal_term.execution_mode,
                "rotations": rotations,
            }
        ],
        "theta_runtime": [0.2],
        "theta_logical": [0.2],
    }
    payload = {
        "source_artifact_json": str(seed),
        "summary": {"accepted_internal_substep_count": 17},
        "run_lock": {
            "physics": {
                "family_key": "siam",
                "seed_sha256": _sha256(seed),
            }
        },
        "plot_rows": [
            {"time": 0.5, "integration_local_substep_count": 17}
        ],
        "final_state": {
            "parameterization_mode": "logical_shared",
            "selected_term_labels": [terminal_term.label],
            "active_parameter_count": 1,
            "logical_parameter_count": 1,
            "runtime_parameter_count": 1,
            "runtime_pauli_parameter_count": len(rotations),
        },
        "resume_state": resume,
    }
    trajectory = tmp_path / "run.json"
    trajectory.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return trajectory, seed


def test_direct_siam_proxy_counts_independent_logical_coordinates(
    tmp_path: Path,
) -> None:
    trajectory, seed = _trajectory_fixture(tmp_path)

    result = build_siam_terminal_resource_record(
        trajectory,
        seed_path=seed,
    )

    assert result["schema"] == SIAM_TERMINAL_RESOURCE_V1
    assert result["canonical_tuple_order"] == [
        "C_hat_2q",
        "C_hat_d",
        "C_hat_1q",
        "N_theta",
    ]
    assert result["terminal_resource_tuple"]["N_theta"] == 1
    assert result["terminal_logical_macro_count"] == 1
    assert result["accepted_integration_steps"] == 17
    assert result["terminal_pauli_rotation_count"] > 1
    assert result["graph_span_comparison"]["no_transpile"] is True
    assert result["graph_span_comparison"]["terminal"][
        "independent_parameter_count"
    ] == 1
    assert result["graph_span_comparison"]["terminal"][
        "coupling_edge_count"
    ] == 176
    assert result["qiskit_compilation"] == {
        "required": False,
        "performed": False,
        "role": "optional_diagnostic_sidecar_only",
    }
    assert len(result["resource_sha256"]) == 64


def test_frozen_graph_snapshot_has_declared_digest() -> None:
    payload = json.dumps(
        [list(edge) for edge in FAKE_MARRAKESH_UNDIRECTED_EDGES_V1],
        separators=(",", ":"),
    ).encode("utf-8")
    assert len(FAKE_MARRAKESH_UNDIRECTED_EDGES_V1) == 176
    assert hashlib.sha256(payload).hexdigest() == (
        FAKE_MARRAKESH_UNDIRECTED_EDGES_SHA256_V1
    )


def test_direct_resource_builder_runs_when_every_qiskit_import_is_blocked(
    tmp_path: Path,
) -> None:
    trajectory, seed = _trajectory_fixture(tmp_path)
    script = """
import builtins
import json
import sys
original_import = builtins.__import__
def deny_qiskit(name, *args, **kwargs):
    if name == 'qiskit' or name.startswith('qiskit.') or name.startswith('qiskit_'):
        raise ImportError('qiskit intentionally blocked')
    return original_import(name, *args, **kwargs)
builtins.__import__ = deny_qiskit
from pipelines.time_dynamics.diagnostics.siam_terminal_resources import build_siam_terminal_resource_record
record = build_siam_terminal_resource_record(sys.argv[1], seed_path=sys.argv[2])
print(json.dumps(record['terminal_resource_tuple'], sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(trajectory), str(seed)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["N_theta"] == 1


def test_direct_resource_builder_rejects_pauli_count_as_n_theta(
    tmp_path: Path,
) -> None:
    trajectory, seed = _trajectory_fixture(tmp_path)
    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    payload["final_state"]["runtime_parameter_count"] = payload[
        "final_state"
    ]["runtime_pauli_parameter_count"]
    trajectory.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runtime_parameter_count differs"):
        build_siam_terminal_resource_record(trajectory, seed_path=seed)


def test_direct_resource_builder_requires_accepted_step_count(
    tmp_path: Path,
) -> None:
    trajectory, seed = _trajectory_fixture(tmp_path)
    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    payload["summary"].pop("accepted_internal_substep_count")
    trajectory.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="accepted-integration-step count"):
        build_siam_terminal_resource_record(trajectory, seed_path=seed)


def test_direct_resource_builder_reconciles_checkpoint_step_total(
    tmp_path: Path,
) -> None:
    trajectory, seed = _trajectory_fixture(tmp_path)
    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    payload["plot_rows"][0]["integration_local_substep_count"] = 16
    trajectory.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differs from checkpoint totals"):
        build_siam_terminal_resource_record(trajectory, seed_path=seed)


def test_direct_resource_builder_requires_step_count_on_every_nonterminal_row(
    tmp_path: Path,
) -> None:
    trajectory, seed = _trajectory_fixture(tmp_path)
    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    payload["plot_rows"] = [
        {"time": 0.25, "integration_local_substep_count": None},
        {"time": 0.5, "integration_local_substep_count": 17},
    ]
    trajectory.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nonterminal row 0"):
        build_siam_terminal_resource_record(trajectory, seed_path=seed)


def test_direct_resource_builder_permits_null_only_on_terminal_row(
    tmp_path: Path,
) -> None:
    trajectory, seed = _trajectory_fixture(tmp_path)
    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    payload["plot_rows"] = [
        {"time": 0.25, "integration_local_substep_count": 17},
        {"time": 0.5, "integration_local_substep_count": None},
    ]
    trajectory.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = build_siam_terminal_resource_record(trajectory, seed_path=seed)

    assert result["accepted_integration_steps"] == 17


def test_direct_resource_builder_uses_parameter_control_step_precedence(
    tmp_path: Path,
) -> None:
    trajectory, seed = _trajectory_fixture(tmp_path)
    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    payload["plot_rows"][0]["parameter_control_substep_count"] = 17
    payload["plot_rows"][0]["integration_local_substep_count"] = 3
    trajectory.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = build_siam_terminal_resource_record(trajectory, seed_path=seed)

    assert result["accepted_integration_steps"] == 17


def test_direct_resource_builder_rejects_invalid_parameter_control_owner(
    tmp_path: Path,
) -> None:
    trajectory, seed = _trajectory_fixture(tmp_path)
    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    payload["plot_rows"][0]["parameter_control_substep_count"] = -1
    trajectory.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="accepted-substep owner counts"):
        build_siam_terminal_resource_record(trajectory, seed_path=seed)


def test_resource_digest_binds_bytes_from_same_snapshot_as_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory, seed = _trajectory_fixture(tmp_path)
    original_bytes = trajectory.read_bytes()
    mutated_payload = json.loads(original_bytes.decode("utf-8"))
    mutated_payload["post_read_mutation"] = True
    mutated_bytes = (
        json.dumps(mutated_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    mutated = False

    def mutate_after_read_text(path: Path, *args: object, **kwargs: object) -> str:
        nonlocal mutated
        value = original_read_text(path, *args, **kwargs)
        if path == trajectory and not mutated:
            mutated = True
            trajectory.write_bytes(mutated_bytes)
        return value

    def mutate_after_read_bytes(path: Path) -> bytes:
        nonlocal mutated
        value = original_read_bytes(path)
        if path == trajectory and not mutated:
            mutated = True
            trajectory.write_bytes(mutated_bytes)
        return value

    monkeypatch.setattr(Path, "read_text", mutate_after_read_text)
    monkeypatch.setattr(Path, "read_bytes", mutate_after_read_bytes)

    result = build_siam_terminal_resource_record(trajectory, seed_path=seed)

    assert mutated is True
    assert result["trajectory_json_sha256"] == hashlib.sha256(
        original_bytes
    ).hexdigest()


def test_resource_record_write_is_idempotent_and_immutable(tmp_path: Path) -> None:
    trajectory, seed = _trajectory_fixture(tmp_path)
    record = build_siam_terminal_resource_record(trajectory, seed_path=seed)
    destination = tmp_path / "terminal_resources.json"

    write_siam_terminal_resource_record(destination, record)
    first = destination.read_bytes()
    write_siam_terminal_resource_record(destination, record)
    assert destination.read_bytes() == first

    altered = dict(record)
    altered["terminal_time"] = 1.0
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        write_siam_terminal_resource_record(destination, altered)

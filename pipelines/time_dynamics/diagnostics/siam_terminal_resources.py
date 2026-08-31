#!/usr/bin/env python3
"""Qiskit-free terminal resource accounting for SIAM trajectories.

The canonical SIAM resource record is the deterministic Paper-I graph-span
proxy plus the number of independent logical coordinates.  It never builds or
transpiles a circuit.  Qiskit compilation may be produced as a separate
diagnostic sidecar, but it is neither an input nor a prerequisite here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.resource_estimation.pauli_graph_span import (
    compare_pauli_graph_span_ansatzes,
    marrakesh_logical_embedding_v1,
)
from pipelines.time_dynamics.resume import (
    RESUME_SCHEMA_V2,
    runtime_input_from_resume_state,
)
from pipelines.time_dynamics.siam_seed import load_siam_runtime_input


SIAM_TERMINAL_RESOURCE_V1 = "siam_terminal_logical_graph_proxy_v1"
FAKE_MARRAKESH_GRAPH_SNAPSHOT_V1 = "fake_marrakesh_undirected_graph_v1"
FAKE_MARRAKESH_BACKEND_NAME = "FakeMarrakesh"
FAKE_MARRAKESH_NUM_QUBITS = 156

# Frozen from qiskit-ibm-runtime's FakeMarrakesh conf_marrakesh.json.  Only the
# undirected connectivity is retained because the graph-span estimator is
# routing-independent and does not transpile.  Keeping the snapshot here makes
# canonical SIAM scoring independent of both qiskit and qiskit-ibm-runtime.
FAKE_MARRAKESH_UNDIRECTED_EDGES_V1: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4), (3, 16), (4, 5), (5, 6), (6, 7),
    (7, 8), (7, 17), (8, 9), (9, 10), (10, 11), (11, 12), (11, 18), (12, 13),
    (13, 14), (14, 15), (15, 19), (16, 23), (17, 27), (18, 31), (19, 35),
    (20, 21), (21, 22), (21, 36), (22, 23), (23, 24), (24, 25), (25, 26),
    (25, 37), (26, 27), (27, 28), (28, 29), (29, 30), (29, 38), (30, 31),
    (31, 32), (32, 33), (33, 34), (33, 39), (34, 35), (36, 41), (37, 45),
    (38, 49), (39, 53), (40, 41), (41, 42), (42, 43), (43, 44), (43, 56),
    (44, 45), (45, 46), (46, 47), (47, 48), (47, 57), (48, 49), (49, 50),
    (50, 51), (51, 52), (51, 58), (52, 53), (53, 54), (54, 55), (55, 59),
    (56, 63), (57, 67), (58, 71), (59, 75), (60, 61), (61, 62), (61, 76),
    (62, 63), (63, 64), (64, 65), (65, 66), (65, 77), (66, 67), (67, 68),
    (68, 69), (69, 70), (69, 78), (70, 71), (71, 72), (72, 73), (73, 74),
    (73, 79), (74, 75), (76, 81), (77, 85), (78, 89), (79, 93), (80, 81),
    (81, 82), (82, 83), (83, 84), (83, 96), (84, 85), (85, 86), (86, 87),
    (87, 88), (87, 97), (88, 89), (89, 90), (90, 91), (91, 92), (91, 98),
    (92, 93), (93, 94), (94, 95), (95, 99), (96, 103), (97, 107), (98, 111),
    (99, 115), (100, 101), (101, 102), (101, 116), (102, 103), (103, 104),
    (104, 105), (105, 106), (105, 117), (106, 107), (107, 108), (108, 109),
    (109, 110), (109, 118), (110, 111), (111, 112), (112, 113), (113, 114),
    (113, 119), (114, 115), (116, 121), (117, 125), (118, 129), (119, 133),
    (120, 121), (121, 122), (122, 123), (123, 124), (123, 136), (124, 125),
    (125, 126), (126, 127), (127, 128), (127, 137), (128, 129), (129, 130),
    (130, 131), (131, 132), (131, 138), (132, 133), (133, 134), (134, 135),
    (135, 139), (136, 143), (137, 147), (138, 151), (139, 155), (140, 141),
    (141, 142), (142, 143), (143, 144), (144, 145), (145, 146), (146, 147),
    (147, 148), (148, 149), (149, 150), (150, 151), (151, 152), (152, 153),
    (153, 154), (154, 155),
)
FAKE_MARRAKESH_UNDIRECTED_EDGES_SHA256_V1 = (
    "234fb1b6afe2d36a595dcb245317b542350e1bdb88f428cad98a3a60328c798b"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _document_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(dict(value))).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _load_trajectory_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    """Parse and hash one immutable byte snapshot of a trajectory file."""

    try:
        content = path.read_bytes()
        value = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"SIAM trajectory is not readable JSON: {path}") from exc
    payload = dict(_mapping(value, name="SIAM trajectory"))
    return payload, hashlib.sha256(content).hexdigest()


def _graph_snapshot() -> dict[str, Any]:
    observed = hashlib.sha256(
        json.dumps(
            [list(edge) for edge in FAKE_MARRAKESH_UNDIRECTED_EDGES_V1],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if observed != FAKE_MARRAKESH_UNDIRECTED_EDGES_SHA256_V1:
        raise RuntimeError("Frozen FakeMarrakesh graph snapshot digest mismatch.")
    return {
        "schema": FAKE_MARRAKESH_GRAPH_SNAPSHOT_V1,
        "backend_name": FAKE_MARRAKESH_BACKEND_NAME,
        "num_qubits": FAKE_MARRAKESH_NUM_QUBITS,
        "undirected_coupling_edge_count": len(
            FAKE_MARRAKESH_UNDIRECTED_EDGES_V1
        ),
        "undirected_coupling_edges_sha256": observed,
        "source": "frozen_fake_marrakesh_conf_marrakesh_v1",
    }


def _resource_tuple(ledger: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        "C_hat_2q": float(ledger["absolute_c_hat_2q"]),
        "C_hat_d": float(ledger["absolute_c_hat_d"]),
        "C_hat_1q": float(ledger["absolute_c_hat_1q"]),
        "N_theta": int(round(float(ledger["absolute_theta_count"]))),
    }


def build_siam_terminal_resource_record(
    trajectory_path: str | Path,
    *,
    seed_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the canonical terminal SIAM resource record without Qiskit.

    ``N_theta`` is the number of independent logical-shared coordinates.  The
    three graph costs still expand every Pauli rotation within every ordered
    logical macro, including repeated positioned occurrences.
    """

    trajectory_source = Path(trajectory_path)
    payload, trajectory_sha256 = _load_trajectory_snapshot(trajectory_source)
    recorded_seed = str(payload.get("source_artifact_json") or "").strip()
    if seed_path is None and not recorded_seed:
        raise ValueError("SIAM trajectory does not identify its seed artifact.")
    seed_source = Path(seed_path if seed_path is not None else recorded_seed)
    seed_sha256 = _file_sha256(seed_source)

    run_lock = _mapping(payload.get("run_lock"), name="SIAM run lock")
    physics = _mapping(run_lock.get("physics"), name="SIAM run-lock physics")
    if str(physics.get("family_key", "")).strip().lower() != "siam":
        raise ValueError("Terminal SIAM resources require family_key='siam'.")
    if str(physics.get("seed_sha256", "")) != seed_sha256:
        raise ValueError("SIAM resource seed digest differs from its run lock.")

    resume = _mapping(payload.get("resume_state"), name="SIAM resume state")
    if str(resume.get("schema")) != RESUME_SCHEMA_V2:
        raise ValueError(
            f"Terminal SIAM resources require resume schema {RESUME_SCHEMA_V2!r}."
        )
    if str(resume.get("problem_family", "")).strip().lower() != "siam":
        raise ValueError("SIAM resource resume payload has the wrong family.")
    if str(resume.get("parameterization_mode")) != "logical_shared":
        raise ValueError("SIAM canonical N_theta requires logical_shared mode.")

    base = load_siam_runtime_input(seed_source)
    terminal = runtime_input_from_resume_state(
        resume,
        base_runtime_input=base,
    )
    initial_terms = tuple(base.selected_terms)
    terminal_terms = tuple(terminal.selected_terms)
    theta_logical = np.asarray(terminal.theta_logical, dtype=float).reshape(-1)
    if not np.all(np.isfinite(theta_logical)):
        raise ValueError("SIAM terminal logical coordinates must be finite.")
    n_theta = int(theta_logical.size)
    if n_theta != len(terminal_terms):
        raise ValueError("SIAM logical coordinate count differs from macro count.")

    final_state = _mapping(payload.get("final_state"), name="SIAM final state")
    if str(final_state.get("parameterization_mode")) != "logical_shared":
        raise ValueError("SIAM final state does not preserve logical_shared mode.")
    selected_labels = tuple(str(value) for value in final_state.get(
        "selected_term_labels", ()
    ))
    rebuilt_labels = tuple(str(getattr(term, "label", "")) for term in terminal_terms)
    if selected_labels != rebuilt_labels:
        raise ValueError("SIAM terminal support labels fail resume parity.")
    for field in (
        "active_parameter_count",
        "logical_parameter_count",
        "runtime_parameter_count",
    ):
        if int(final_state.get(field, -1)) != n_theta:
            raise ValueError(f"SIAM final-state {field} differs from N_theta.")
    runtime_pauli_count = int(terminal.base_layout.runtime_parameter_count)
    if int(final_state.get("runtime_pauli_parameter_count", -1)) != (
        runtime_pauli_count
    ):
        raise ValueError("SIAM terminal Pauli-rotation count fails resume parity.")

    rows = tuple(payload.get("plot_rows") or ())
    if not rows or not isinstance(rows[-1], Mapping):
        raise ValueError("SIAM trajectory lacks a terminal checkpoint row.")
    terminal_time = float(rows[-1]["time"])
    if not math.isfinite(terminal_time) or not math.isclose(
        float(resume.get("time", float("nan"))),
        terminal_time,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("SIAM resume time differs from the terminal checkpoint.")

    summary = _mapping(payload.get("summary"), name="SIAM trajectory summary")
    accepted_steps_value = summary.get("accepted_internal_substep_count")
    if isinstance(accepted_steps_value, bool) or not isinstance(
        accepted_steps_value, int
    ):
        raise ValueError(
            "SIAM trajectory summary lacks an integer accepted-integration-step count."
        )
    accepted_integration_steps = int(accepted_steps_value)
    if accepted_integration_steps < 0:
        raise ValueError("SIAM accepted integration steps cannot be negative.")
    row_step_counts: list[int] = []
    final_row_index = len(rows) - 1
    for index, row_value in enumerate(rows):
        row = _mapping(row_value, name=f"SIAM checkpoint row {index}")
        # Mirror the runner's accepted_internal_substep_count definition:
        # state/parameter-control substeps own an interval when present;
        # otherwise the integration-local count owns it.
        parameter_value = row.get("parameter_control_substep_count")
        value = (
            parameter_value
            if parameter_value is not None
            else row.get("integration_local_substep_count")
        )
        if value is None:
            if index != final_row_index:
                raise ValueError(
                    "SIAM checkpoint accepted-substep owner must be a nonnegative "
                    f"integer on nonterminal row {index}."
                )
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "SIAM checkpoint accepted-substep owner counts must be "
                f"nonnegative integers; row {index} is invalid."
            )
        row_step_counts.append(int(value))
    if sum(row_step_counts) != accepted_integration_steps:
        raise ValueError(
            "SIAM accepted-integration-step summary differs from checkpoint totals."
        )

    n_qubits = int(base.resolved_problem.n_qubits)
    graph_snapshot = _graph_snapshot()
    comparison = compare_pauli_graph_span_ansatzes(
        initial_terms=initial_terms,
        terminal_terms=terminal_terms,
        num_qubits=n_qubits,
        embedding=marrakesh_logical_embedding_v1(n_qubits),
        coupling_edges=FAKE_MARRAKESH_UNDIRECTED_EDGES_V1,
        backend_name=FAKE_MARRAKESH_BACKEND_NAME,
        graph_snapshot=graph_snapshot,
        initial_parameter_count=int(base.base_layout.logical_parameter_count),
        terminal_parameter_count=n_theta,
    )
    initial_resource = _resource_tuple(comparison["initial"])
    terminal_resource = _resource_tuple(comparison["terminal"])
    delta_resource = _resource_tuple(comparison["terminal_minus_initial"])
    if terminal_resource["N_theta"] != n_theta:
        raise RuntimeError("SIAM graph record changed the independent N_theta count.")

    core = {
        "schema": SIAM_TERMINAL_RESOURCE_V1,
        "trajectory_json": str(trajectory_source.resolve()),
        "trajectory_json_sha256": trajectory_sha256,
        "seed_json": str(seed_source.resolve()),
        "seed_json_sha256": seed_sha256,
        "problem_family": "siam",
        "parameterization_mode": "logical_shared",
        "terminal_time": terminal_time,
        "accepted_integration_steps": accepted_integration_steps,
        "terminal_logical_macro_count": len(terminal_terms),
        "terminal_pauli_rotation_count": runtime_pauli_count,
        "canonical_tuple_order": ["C_hat_2q", "C_hat_d", "C_hat_1q", "N_theta"],
        "initial_resource_tuple": initial_resource,
        "terminal_resource_tuple": terminal_resource,
        "terminal_minus_initial_resource_tuple": delta_resource,
        "graph_span_comparison": comparison,
        "qiskit_compilation": {
            "required": False,
            "performed": False,
            "role": "optional_diagnostic_sidecar_only",
        },
    }
    return {**core, "resource_sha256": _document_sha256(core)}


def write_siam_terminal_resource_record(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    """Write one digest-bound record idempotently without replacing evidence."""

    destination = Path(path)
    content = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == content:
            return
        raise FileExistsError(
            f"Refusing to replace immutable SIAM resource record: {destination}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise FileExistsError(
                    f"Refusing to replace immutable SIAM resource record: {destination}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--seed-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_siam_terminal_resource_record(
        args.trajectory_json,
        seed_path=args.seed_json,
    )
    write_siam_terminal_resource_record(args.output_json, payload)
    print(json.dumps({
        "output_json": str(args.output_json),
        "resource_sha256": payload["resource_sha256"],
        "terminal_resource_tuple": payload["terminal_resource_tuple"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FAKE_MARRAKESH_GRAPH_SNAPSHOT_V1",
    "FAKE_MARRAKESH_UNDIRECTED_EDGES_SHA256_V1",
    "FAKE_MARRAKESH_UNDIRECTED_EDGES_V1",
    "SIAM_TERMINAL_RESOURCE_V1",
    "build_siam_terminal_resource_record",
    "write_siam_terminal_resource_record",
]

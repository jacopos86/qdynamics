#!/usr/bin/env python3
"""Qiskit-free terminal logical-coordinate and graph-span resources.

The canonical artifact is reconstructed directly from the trajectory's
serialized resume coordinates.  Its ``N_theta`` is therefore the number of
independent runtime coordinates, while the graph tuple prices every ordered
Pauli rotation on the frozen FakeMarrakesh connectivity without building or
transpiling a circuit.

Existing Paper-II evidence predates this artifact.  ``load_terminal_resources``
therefore has one explicit compatibility adapter for
``terminal_qiskit_cost.json``.  A present canonical artifact is always
authoritative: malformed canonical evidence is rejected and never hidden by
falling back to the legacy sidecar.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.resource_estimation.pauli_graph_span import (
    evaluate_pauli_graph_span_ansatz,
    marrakesh_logical_embedding_v1,
)
from pipelines.time_dynamics.diagnostics.siam_terminal_resources import (
    FAKE_MARRAKESH_BACKEND_NAME,
    FAKE_MARRAKESH_GRAPH_SNAPSHOT_V1,
    FAKE_MARRAKESH_NUM_QUBITS,
    FAKE_MARRAKESH_UNDIRECTED_EDGES_SHA256_V1,
    FAKE_MARRAKESH_UNDIRECTED_EDGES_V1,
)
from pipelines.time_dynamics.resume import RESUME_SCHEMA_V1, RESUME_SCHEMA_V2
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


TERMINAL_LOGICAL_GRAPH_RESOURCE_FILENAME = (
    "terminal_logical_graph_resources.json"
)
LEGACY_TERMINAL_QISKIT_COST_FILENAME = "terminal_qiskit_cost.json"
TERMINAL_LOGICAL_GRAPH_RESOURCE_V1 = "terminal_logical_graph_resources_v1"
CANONICAL_TUPLE_ORDER = ("C_hat_2q", "C_hat_d", "C_hat_1q", "N_theta")


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


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return dict(value)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _finite_integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be a finite integer")
    result = int(numeric)
    if result < int(minimum):
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _resource_tuple(ledger: Mapping[str, Any]) -> dict[str, int]:
    return {
        "C_hat_2q": _finite_integer(
            ledger["absolute_c_hat_2q"], label="C_hat_2q"
        ),
        "C_hat_d": _finite_integer(
            ledger["absolute_c_hat_d"], label="C_hat_d"
        ),
        "C_hat_1q": _finite_integer(
            ledger["absolute_c_hat_1q"], label="C_hat_1q"
        ),
        "N_theta": _finite_integer(
            ledger["absolute_theta_count"], label="N_theta"
        ),
    }


def _graph_snapshot() -> dict[str, Any]:
    observed = hashlib.sha256(
        _canonical_bytes(
            [list(edge) for edge in FAKE_MARRAKESH_UNDIRECTED_EDGES_V1]
        )
    ).hexdigest()
    if observed != FAKE_MARRAKESH_UNDIRECTED_EDGES_SHA256_V1:
        raise RuntimeError("frozen FakeMarrakesh graph snapshot changed")
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


def _resume_terms(
    resume: Mapping[str, Any],
) -> tuple[tuple[AnsatzTerm, ...], int, int, str]:
    schema = str(resume.get("schema"))
    mode = str(resume.get("parameterization_mode", "per_pauli_term"))
    coordinates = resume.get("coordinates")
    if not isinstance(coordinates, Sequence) or isinstance(
        coordinates, (str, bytes)
    ):
        raise ValueError("terminal resource resume coordinates must be a sequence")
    if schema not in {RESUME_SCHEMA_V1, RESUME_SCHEMA_V2}:
        raise ValueError(f"unsupported terminal resource resume schema: {schema!r}")
    if not coordinates:
        raise ValueError("terminal resource resume coordinates cannot be empty")

    terms: list[AnsatzTerm] = []
    widths: set[int] = set()
    if schema == RESUME_SCHEMA_V1:
        if mode == "logical_shared":
            raise ValueError("resume v1 cannot establish logical-shared coordinates")
        for index, raw_coordinate in enumerate(coordinates):
            coordinate = _mapping(
                raw_coordinate, label=f"terminal coordinate {index}"
            )
            nq = _finite_integer(
                coordinate.get("nq"), label=f"terminal coordinate {index} nq", minimum=1
            )
            pauli = str(coordinate.get("pauli_exyz", "")).strip().lower()
            if len(pauli) != nq or any(char not in "exyz" for char in pauli):
                raise ValueError(
                    f"terminal coordinate {index} has an invalid Pauli word"
                )
            coefficient = float(coordinate.get("coeff_real"))
            if not math.isfinite(coefficient) or coefficient == 0.0:
                raise ValueError(
                    f"terminal coordinate {index} coefficient must be finite and nonzero"
                )
            widths.add(nq)
            terms.append(
                AnsatzTerm(
                    label=str(coordinate.get("label", "")),
                    polynomial=PauliPolynomial(
                        "exyz", [PauliTerm(nq, ps=pauli, pc=coefficient)]
                    ),
                    execution_mode="termwise_product",
                )
            )
    else:
        if mode != "logical_shared":
            raise ValueError("resume v2 requires logical_shared parameterization")
        for index, raw_coordinate in enumerate(coordinates):
            coordinate = _mapping(
                raw_coordinate, label=f"terminal logical coordinate {index}"
            )
            rotations = coordinate.get("rotations")
            if not isinstance(rotations, Sequence) or isinstance(
                rotations, (str, bytes)
            ) or not rotations:
                raise ValueError(
                    f"terminal logical coordinate {index} needs ordered rotations"
                )
            polynomial = PauliPolynomial("exyz")
            for rotation_index, raw_rotation in enumerate(rotations):
                rotation = _mapping(
                    raw_rotation,
                    label=(
                        f"terminal logical coordinate {index} rotation "
                        f"{rotation_index}"
                    ),
                )
                nq = _finite_integer(
                    rotation.get("nq"),
                    label=(
                        f"terminal logical coordinate {index} rotation "
                        f"{rotation_index} nq"
                    ),
                    minimum=1,
                )
                pauli = str(rotation.get("pauli_exyz", "")).strip().lower()
                if len(pauli) != nq or any(char not in "exyz" for char in pauli):
                    raise ValueError(
                        f"terminal logical coordinate {index} has an invalid Pauli word"
                    )
                coefficient = float(rotation.get("coeff_real"))
                if not math.isfinite(coefficient) or coefficient == 0.0:
                    raise ValueError(
                        f"terminal logical coordinate {index} has an invalid coefficient"
                    )
                widths.add(nq)
                # add_term intentionally preserves repeated positioned rotations.
                polynomial.add_term(
                    PauliTerm(nq, ps=pauli, pc=coefficient)
                )
            terms.append(
                AnsatzTerm(
                    label=str(coordinate.get("label", "")),
                    polynomial=polynomial,
                    execution_mode=str(
                        coordinate.get("execution_mode", "termwise_product")
                        or "termwise_product"
                    ),
                )
            )

    if len(widths) != 1:
        raise ValueError(f"terminal coordinates disagree on qubit count: {sorted(widths)}")
    n_theta = len(terms)
    theta_runtime = np.asarray(resume.get("theta_runtime"), dtype=float).reshape(-1)
    if theta_runtime.size != n_theta or not np.all(np.isfinite(theta_runtime)):
        raise ValueError("terminal theta_runtime does not match independent coordinates")
    if schema == RESUME_SCHEMA_V2:
        theta_logical = np.asarray(resume.get("theta_logical"), dtype=float).reshape(-1)
        if (
            theta_logical.size != n_theta
            or not np.all(np.isfinite(theta_logical))
            or not np.array_equal(theta_runtime, theta_logical)
        ):
            raise ValueError("terminal logical angles do not match runtime angles")
    return tuple(terms), n_theta, next(iter(widths)), mode


def build_terminal_logical_graph_resource_record(
    trajectory_path: str | Path,
) -> dict[str, Any]:
    """Build one digest-bound terminal logical graph record without Qiskit."""

    trajectory = Path(trajectory_path)
    payload = _read_mapping(trajectory, label="trajectory JSON")
    trajectory_sha256 = _file_sha256(trajectory)
    resume = _mapping(payload.get("resume_state"), label="trajectory resume state")
    terms, n_theta, num_qubits, mode = _resume_terms(resume)

    final_state = _mapping(payload.get("final_state"), label="trajectory final state")
    selected_labels = tuple(str(value) for value in final_state.get(
        "selected_term_labels", ()
    ))
    rebuilt_labels = tuple(str(term.label) for term in terms)
    if selected_labels != rebuilt_labels:
        raise ValueError("terminal resume labels fail final-state parity")
    if str(final_state.get("parameterization_mode")) != mode:
        raise ValueError("terminal resume mode fails final-state parity")
    for label, raw_value in (
        ("final runtime parameter count", final_state.get("runtime_parameter_count")),
        (
            "summary runtime parameter count",
            _mapping(payload.get("summary"), label="trajectory summary").get(
                "runtime_parameter_count_final"
            ),
        ),
    ):
        if _finite_integer(raw_value, label=label) != n_theta:
            raise ValueError(f"{label} differs from terminal N_theta")

    rows = payload.get("plot_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("trajectory lacks terminal plot rows")
    terminal_row = _mapping(rows[-1], label="terminal plot row")
    terminal_time = float(terminal_row.get("time"))
    if not math.isfinite(terminal_time) or not math.isclose(
        terminal_time,
        float(resume.get("time")),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("terminal plot time differs from resume time")

    snapshot = _graph_snapshot()
    terminal_ledger = evaluate_pauli_graph_span_ansatz(
        terms,
        num_qubits=num_qubits,
        embedding=marrakesh_logical_embedding_v1(num_qubits),
        coupling_edges=FAKE_MARRAKESH_UNDIRECTED_EDGES_V1,
        backend_name=FAKE_MARRAKESH_BACKEND_NAME,
        graph_snapshot=snapshot,
        independent_parameter_count=n_theta,
    ).to_json_dict()
    terminal_resource = _resource_tuple(terminal_ledger)
    if terminal_resource["N_theta"] != n_theta:
        raise RuntimeError("graph-span evaluator changed terminal N_theta")

    family = str(
        resume.get("problem_family")
        or dict(payload.get("run_lock") or {}).get("physics", {}).get(
            "family_key", ""
        )
        or "unknown"
    ).strip().lower()
    core = {
        "schema": TERMINAL_LOGICAL_GRAPH_RESOURCE_V1,
        "trajectory_json": str(trajectory.resolve()),
        "trajectory_json_sha256": trajectory_sha256,
        "problem_family": family,
        "parameterization_mode": mode,
        "terminal_time": terminal_time,
        "canonical_tuple_order": list(CANONICAL_TUPLE_ORDER),
        # The trajectory resume state authoritatively serializes the terminal
        # support.  It does not, for a continued leg, authoritatively identify
        # the original t=0 seed support.  Do not mislabel the leg-start support
        # as a campaign-seed delta.
        "initial_resource_tuple": None,
        "terminal_resource_tuple": terminal_resource,
        "terminal_minus_initial_resource_tuple": None,
        "terminal_graph_span_ledger": terminal_ledger,
        "terminal_support_parity": {
            "schema": "terminal_logical_support_parity_v1",
            "passed": True,
            "resume_coordinate_count": n_theta,
            "final_state_coordinate_count": n_theta,
            "summary_coordinate_count": n_theta,
            "selected_term_labels_match": True,
        },
        "qiskit_compilation": {
            "required": False,
            "performed": False,
            "role": "not_part_of_canonical_resource_accounting",
        },
    }
    return {**core, "resource_sha256": _document_sha256(core)}


def write_terminal_logical_graph_resource_record(
    path: str | Path, payload: Mapping[str, Any]
) -> None:
    """Write an immutable canonical resource record through one part path."""

    destination = Path(path)
    content = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == content:
            return
        raise FileExistsError(
            f"refusing to replace terminal resource evidence: {destination}"
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
                    f"refusing to replace terminal resource evidence: {destination}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class TerminalLogicalGraphResources:
    """Normalized reporting view over canonical and legacy resource evidence."""

    artifact_path: Path
    artifact_sha256: str
    source_kind: str
    n_theta: int
    graph: tuple[int, int, int]
    delta_n_theta: int | None
    delta_graph: tuple[int, int, int] | None


def _normalized_tuple(
    value: Any, *, label: str, allow_negative: bool = False
) -> tuple[int, int, int, int]:
    payload = _mapping(value, label=label)
    if set(payload) != set(CANONICAL_TUPLE_ORDER):
        raise ValueError(f"{label} does not have the canonical tuple fields")
    minimum = -(1 << 62) if allow_negative else 0
    return tuple(
        _finite_integer(payload[name], label=f"{label} {name}", minimum=minimum)
        for name in CANONICAL_TUPLE_ORDER
    )  # type: ignore[return-value]


def _load_canonical(
    run_path: Path, artifact_path: Path
) -> TerminalLogicalGraphResources:
    payload = _read_mapping(artifact_path, label="terminal logical resource")
    if str(payload.get("schema")) != TERMINAL_LOGICAL_GRAPH_RESOURCE_V1:
        raise ValueError(f"unsupported terminal logical resource schema: {artifact_path}")
    if tuple(payload.get("canonical_tuple_order") or ()) != CANONICAL_TUPLE_ORDER:
        raise ValueError(f"terminal logical resource tuple order changed: {artifact_path}")
    recorded_digest = str(payload.get("resource_sha256", ""))
    core = dict(payload)
    core.pop("resource_sha256", None)
    if recorded_digest != _document_sha256(core):
        raise ValueError(f"terminal logical resource digest mismatch: {artifact_path}")
    if str(payload.get("trajectory_json_sha256")) != _file_sha256(run_path):
        raise ValueError(f"terminal logical resource binds another run: {artifact_path}")
    compilation = _mapping(
        payload.get("qiskit_compilation"), label="compilation exclusion"
    )
    if compilation.get("required") is not False or compilation.get("performed") is not False:
        raise ValueError("canonical logical resource cannot depend on compilation")
    parity = _mapping(
        payload.get("terminal_support_parity"), label="terminal support parity"
    )
    if parity.get("passed") is not True:
        raise ValueError(f"terminal logical support parity failed: {artifact_path}")
    terminal = _normalized_tuple(
        payload.get("terminal_resource_tuple"), label="terminal resource tuple"
    )
    run = _read_mapping(run_path, label="trajectory JSON")
    summary = _mapping(run.get("summary"), label="trajectory summary")
    if terminal[3] != _finite_integer(
        summary.get("runtime_parameter_count_final"),
        label="trajectory terminal N_theta",
    ):
        raise ValueError("terminal logical resource N_theta differs from trajectory")
    delta_value = payload.get("terminal_minus_initial_resource_tuple")
    delta = (
        None
        if delta_value is None
        else _normalized_tuple(
            delta_value,
            label="terminal-minus-initial resource tuple",
            allow_negative=True,
        )
    )
    return TerminalLogicalGraphResources(
        artifact_path=artifact_path,
        artifact_sha256=_file_sha256(artifact_path),
        source_kind="canonical_logical_graph_resource",
        n_theta=terminal[3],
        graph=terminal[:3],
        delta_n_theta=None if delta is None else delta[3],
        delta_graph=None if delta is None else delta[:3],
    )


def _load_legacy(
    run_path: Path, artifact_path: Path
) -> TerminalLogicalGraphResources:
    payload = _read_mapping(artifact_path, label="legacy terminal cost")
    rows = payload.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != 1:
        raise ValueError(f"legacy terminal cost must contain one row: {artifact_path}")
    row = _mapping(rows[0], label="legacy terminal cost row")
    parity = _mapping(
        row.get("terminal_reconstruction_parity"),
        label="legacy terminal reconstruction parity",
    )
    if parity.get("passed") is not True:
        raise ValueError(f"legacy terminal reconstruction parity failed: {artifact_path}")
    terminal = tuple(
        _finite_integer(row[name], label=f"legacy {name}")
        for name in (
            "graph_proxy_terminal_c_hat_2q",
            "graph_proxy_terminal_c_hat_d",
            "graph_proxy_terminal_c_hat_1q",
            "graph_proxy_terminal_theta_count",
        )
    )
    delta = tuple(
        _finite_integer(row[name], label=f"legacy {name}", minimum=-(1 << 62))
        for name in (
            "graph_proxy_delta_c_hat_2q",
            "graph_proxy_delta_c_hat_d",
            "graph_proxy_delta_c_hat_1q",
            "graph_proxy_delta_theta_count",
        )
    )
    run = _read_mapping(run_path, label="trajectory JSON")
    summary = _mapping(run.get("summary"), label="trajectory summary")
    if terminal[3] != _finite_integer(
        summary.get("runtime_parameter_count_final"),
        label="trajectory terminal N_theta",
    ):
        raise ValueError("legacy terminal N_theta differs from trajectory")
    return TerminalLogicalGraphResources(
        artifact_path=artifact_path,
        artifact_sha256=_file_sha256(artifact_path),
        source_kind="legacy_terminal_qiskit_cost_fallback",
        n_theta=terminal[3],
        graph=terminal[:3],
        delta_n_theta=delta[3],
        delta_graph=delta[:3],
    )


def load_terminal_resources(
    run_path: str | Path,
    *,
    allow_legacy_qiskit_fallback: bool = True,
) -> TerminalLogicalGraphResources:
    """Load canonical logical resources, or an explicitly allowed legacy row."""

    trajectory = Path(run_path)
    canonical = trajectory.with_name(TERMINAL_LOGICAL_GRAPH_RESOURCE_FILENAME)
    if canonical.exists():
        return _load_canonical(trajectory, canonical)
    legacy = trajectory.with_name(LEGACY_TERMINAL_QISKIT_COST_FILENAME)
    if allow_legacy_qiskit_fallback and legacy.exists():
        return _load_legacy(trajectory, legacy)
    raise FileNotFoundError(
        "missing canonical terminal logical graph resource beside "
        f"{trajectory}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        help=(
            "Defaults to terminal_logical_graph_resources.json beside the "
            "trajectory."
        ),
    )
    args = parser.parse_args(argv)
    output = args.output_json or args.trajectory_json.with_name(
        TERMINAL_LOGICAL_GRAPH_RESOURCE_FILENAME
    )
    payload = build_terminal_logical_graph_resource_record(args.trajectory_json)
    write_terminal_logical_graph_resource_record(output, payload)
    print(
        json.dumps(
            {
                "output_json": str(output),
                "resource_sha256": payload["resource_sha256"],
                "terminal_resource_tuple": payload["terminal_resource_tuple"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANONICAL_TUPLE_ORDER",
    "LEGACY_TERMINAL_QISKIT_COST_FILENAME",
    "TERMINAL_LOGICAL_GRAPH_RESOURCE_FILENAME",
    "TERMINAL_LOGICAL_GRAPH_RESOURCE_V1",
    "TerminalLogicalGraphResources",
    "build_terminal_logical_graph_resource_record",
    "load_terminal_resources",
    "write_terminal_logical_graph_resource_record",
]

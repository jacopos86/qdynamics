#!/usr/bin/env python3
"""Shared Qiskit compilation helpers for Paper-I Table-I resources.

The fixed-accuracy Table-I resource columns are hardware/circuit evidence, not
selector proxies.  This module is the single local convention for compiling an
ansatz circuit before reporting ``N_2q``, ``D_2q`` and ``D_circ``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
from typing import Any, Mapping, Sequence

import numpy as np

_QISKIT_HELPER_IMPORT_ERROR: str | None = None
if os.environ.get("HOLSTEIN_SKIP_QISKIT_IMPORT") == "1":
    _QISKIT_HELPER_IMPORT_ERROR = "HOLSTEIN_SKIP_QISKIT_IMPORT=1"
    append_pauli_rotation_exyz = None
    append_reference_state = None
    build_structural_ansatz_circuit = None
else:
    try:
        from pipelines.hardcoded.adapt_circuit_execution import (
            append_pauli_rotation_exyz,
            append_reference_state,
            build_structural_ansatz_circuit,
        )
    except Exception as exc:  # pragma: no cover - optional dependency varies
        _QISKIT_HELPER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        append_pauli_rotation_exyz = None
        append_reference_state = None
        build_structural_ansatz_circuit = None
from src.quantum.ansatz_parameterization import iter_runtime_rotation_terms
from src.quantum.vqe_latex_python_pairs import AnsatzTerm

TABLE_I_QISKIT_COMPILE_CONVENTION = "table_i_basis_gate_transpile_v1"
TABLE_I_STRUCTURAL_ANGLE_CONVENTION = "structural_nonzero_placeholder_angles_v1"
TABLE_I_GROUPED_EXACT_SYNTHESIS_ID = "commuting_pauli_or_active_support_unitary_exact_v1"
TABLE_I_QISKIT_BASIS_WORK_SCHEMA = "qiskit_pretranspile_pauli_basis_work_v1"
TABLE_I_COMPILED_BASIS_GATES: tuple[str, ...] = (
    "id",
    "x",
    "sx",
    "rx",
    "ry",
    "rz",
    "h",
    "s",
    "sdg",
    "cx",
    "cz",
)


@dataclass(frozen=True)
class TableIQiskitCompileConfig:
    basis_gates: tuple[str, ...] = TABLE_I_COMPILED_BASIS_GATES
    optimization_level: int = 0
    seed_transpiler: int | None = 7
    structure_theta_value: float = 1.0
    include_reference_state: bool = True
    compile_convention: str = TABLE_I_QISKIT_COMPILE_CONVENTION
    coefficient_tolerance: float = 1.0e-12
    # n_ph_max=4 Paper-I HH cloud generators act on three binary-boson
    # qubits plus two fermion qubits.  Five is therefore the smallest limit
    # that covers the canonical six-regime parent-generator matrix.
    grouped_exact_max_active_qubits: int = 5


class TableICompileUnavailable(RuntimeError):
    """Expected fail-closed absence of Qiskit Table-I compile evidence."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = str(status)
        self.reason = str(reason)


def _require_qiskit_circuit_helpers() -> None:
    if (
        append_pauli_rotation_exyz is None
        or append_reference_state is None
        or build_structural_ansatz_circuit is None
    ):
        reason = _QISKIT_HELPER_IMPORT_ERROR or "Qiskit circuit helpers are unavailable"
        raise TableICompileUnavailable("qiskit_circuit_helpers_unavailable", reason)


def _jsonable_op_counts(raw: Mapping[str, Any] | None) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value)
        except Exception:
            continue
    return out


def _empty_qiskit_basis_work_counts() -> dict[str, int]:
    return {"h": 0, "s": 0, "sdg": 0, "rz": 0}


def _add_qiskit_basis_work_counts(
    total: dict[str, int],
    increment: Mapping[str, Any],
) -> None:
    for name in tuple(total):
        total[name] += int(increment.get(name, 0))


def _append_pauli_rotation_with_qiskit_basis_work(
    circuit: Any,
    *,
    label_exyz: str,
    angle: Any,
) -> dict[str, int]:
    """Append one Pauli rotation and count the Qiskit instructions it emits."""

    start = len(circuit.data)
    append_pauli_rotation_exyz(
        circuit,
        label_exyz=str(label_exyz),
        angle=angle,
    )
    counts = _empty_qiskit_basis_work_counts()
    for instruction in circuit.data[start:]:
        operation = getattr(instruction, "operation", None)
        if operation is None:
            try:
                operation = instruction[0]
            except Exception:
                continue
        name = str(getattr(operation, "name", "")).lower()
        if name in counts:
            counts[name] += 1
    return counts


def _qiskit_basis_work_payload(
    counts: Mapping[str, Any],
    *,
    attributable: bool,
    non_attributable_operator_count: int = 0,
) -> dict[str, Any]:
    components = {
        name: int(counts.get(name, 0))
        for name in ("h", "s", "sdg", "rz")
    }
    basis_change_total = int(
        components["h"] + components["s"] + components["sdg"]
    )
    pauli_1q_work_total = int(basis_change_total + components["rz"])
    return {
        "qiskit_basis_work_schema": TABLE_I_QISKIT_BASIS_WORK_SCHEMA,
        "qiskit_basis_work_status": (
            "ok"
            if attributable
            else "unavailable_noncommuting_grouped_exact_synthesis"
        ),
        "qiskit_pretranspile_basis_change_1q_total": (
            basis_change_total if attributable else None
        ),
        "qiskit_pretranspile_pauli_rotation_rz_total": (
            components["rz"] if attributable else None
        ),
        "qiskit_pretranspile_pauli_1q_work_total": (
            pauli_1q_work_total if attributable else None
        ),
        "qiskit_pretranspile_pauli_1q_work_components": components,
        "qiskit_basis_work_non_attributable_operator_count": int(
            non_attributable_operator_count
        ),
        "qiskit_basis_work_semantics": (
            "Qiskit instructions emitted by the Pauli-rotation synthesis "
            "before transpilation and excluding reference-state preparation; "
            "basis-change total counts h, s, and sdg, while Pauli one-qubit "
            "work additionally counts the central rz"
        ),
    }


def _qiskit_version() -> str | None:
    for package in ("qiskit", "qiskit-terra"):
        try:
            return str(metadata.version(package))
        except metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return None


def _compiled_stats_payload(compiled: Any) -> dict[str, Any]:
    try:
        from pipelines.qiskit_backend_tools import (
            compiled_gate_stats,
            safe_circuit_depth,
            safe_two_qubit_depth,
        )
    except Exception as exc:  # pragma: no cover - import failure depends on env
        raise TableICompileUnavailable("qiskit_backend_tools_unavailable", str(exc)) from exc

    try:
        gate_stats = dict(compiled_gate_stats(compiled))
        depth_total = int(safe_circuit_depth(compiled))
        depth_2q = int(safe_two_qubit_depth(compiled))
    except Exception as exc:
        raise TableICompileUnavailable("qiskit_compiled_stats_failed", str(exc)) from exc
    count_1q = gate_stats.get("compiled_count_1q")
    if count_1q is None:
        raise TableICompileUnavailable("qiskit_compiled_count_1q_missing", "compiled_gate_stats did not return compiled_count_1q")
    count_2q = gate_stats.get("compiled_count_2q")
    if count_2q is None:
        raise TableICompileUnavailable("qiskit_compiled_count_2q_missing", "compiled_gate_stats did not return compiled_count_2q")
    if int(depth_total) < int(depth_2q):
        raise TableICompileUnavailable(
            "qiskit_compiled_depth_order_invalid",
            "compiled total depth is smaller than compiled two-qubit depth",
        )
    return {
        "compiled_count_1q_total": int(count_1q),
        "compiled_count_1q_semantics": str(gate_stats.get("compiled_count_1q_semantics") or "post_transpile_one_qubit_quantum_ops"),
        "compiled_count_2q_total": int(count_2q),
        "compiled_depth_2q_total": int(depth_2q),
        "compiled_depth_total": int(depth_total),
        "compiled_op_counts": _jsonable_op_counts(gate_stats.get("compiled_op_counts")),
    }


def _transpile_table_i_circuit(circuit: Any, *, config: TableIQiskitCompileConfig) -> Any:
    try:
        from qiskit import transpile
    except Exception as exc:  # pragma: no cover - optional dependency varies
        raise TableICompileUnavailable("qiskit_transpile_unavailable", str(exc)) from exc
    try:
        try:
            decomposed = circuit.decompose(reps=10)
        except Exception:
            decomposed = circuit
        return transpile(
            decomposed,
            basis_gates=list(config.basis_gates),
            optimization_level=int(config.optimization_level),
            seed_transpiler=config.seed_transpiler,
        )
    except Exception as exc:
        raise TableICompileUnavailable("qiskit_transpile_failed", str(exc)) from exc


def _base_resource_payload(
    *,
    stats: Mapping[str, Any],
    source_kind: str,
    config: TableIQiskitCompileConfig,
    num_qubits: int,
    logical_operator_count: int,
    runtime_rotation_count: int | None,
    reference_state_included: bool,
) -> dict[str, Any]:
    source_kind_text = str(source_kind)
    first_hit = "first_hit" in source_kind_text
    return {
        "compiled_circuit_stats_status": "ok",
        "first_hit_cost_source_kind": source_kind_text,
        "compiled_resource_source_kind": source_kind_text,
        "compiled_resource_qiskit_validated": True,
        "qiskit_first_hit_cost_validated": bool(first_hit),
        "compiled_basis_gates": list(config.basis_gates),
        "compile_convention": str(config.compile_convention),
        "qiskit_version": _qiskit_version(),
        "qiskit_transpile_optimization_level": int(config.optimization_level),
        "qiskit_transpile_seed": config.seed_transpiler,
        "grouped_exact_coefficient_tolerance": float(config.coefficient_tolerance),
        "grouped_exact_max_active_qubits": int(config.grouped_exact_max_active_qubits),
        "angle_convention": TABLE_I_STRUCTURAL_ANGLE_CONVENTION,
        "compiled_depth_2q_semantics": "qiskit_compiled_two_qubit_layer_depth_ansatz_circuit",
        "depth_2q_semantics": "qiskit_compiled_two_qubit_layer_depth_ansatz_circuit",
        "compiled_circuit_scope": "ansatz_circuit_including_reference_state" if reference_state_included else "ansatz_circuit_no_reference_state",
        "num_qubits": int(num_qubits),
        "logical_operator_count": int(logical_operator_count),
        "runtime_rotation_count": None if runtime_rotation_count is None else int(runtime_rotation_count),
        **dict(stats),
    }


def pauli_label_groups_from_ansatz_terms(ops: Sequence[AnsatzTerm]) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for op in ops:
        try:
            terms = iter_runtime_rotation_terms(
                getattr(op, "polynomial"),
                ignore_identity=True,
                coefficient_tolerance=1e-12,
                sort_terms=True,
            )
        except Exception as exc:
            raise TableICompileUnavailable("ansatz_term_pauli_labels_unavailable", str(exc)) from exc
        groups.append(tuple(str(term.pauli_exyz).lower() for term in terms))
    return tuple(groups)


def compile_table_i_pauli_label_groups(
    *,
    pauli_label_groups: Sequence[Sequence[str]],
    num_qubits: int,
    reference_state: np.ndarray | Sequence[complex] | None,
    source_kind: str,
    config: TableIQiskitCompileConfig | None = None,
) -> dict[str, Any]:
    """Compile explicit Pauli-label groups as a Table-I ansatz circuit."""

    _require_qiskit_circuit_helpers()
    cfg = config or TableIQiskitCompileConfig()
    nq = int(num_qubits)
    if nq <= 0:
        raise TableICompileUnavailable("invalid_num_qubits", f"num_qubits={num_qubits!r}")
    try:
        from qiskit import QuantumCircuit
    except Exception as exc:  # pragma: no cover - optional dependency varies
        raise TableICompileUnavailable("qiskit_quantum_circuit_unavailable", str(exc)) from exc
    qc = QuantumCircuit(nq)
    ref_included = bool(cfg.include_reference_state and reference_state is not None)
    if ref_included:
        try:
            append_reference_state(qc, np.asarray(reference_state, dtype=complex).reshape(-1))
        except Exception as exc:
            raise TableICompileUnavailable("reference_state_preparation_failed", str(exc)) from exc
    runtime_rotation_count = 0
    basis_work_counts = _empty_qiskit_basis_work_counts()
    normalized_groups: list[tuple[str, ...]] = []
    for group in pauli_label_groups:
        labels = tuple(str(label).strip().lower() for label in group if str(label).strip())
        normalized_groups.append(labels)
        for label in labels:
            if len(label) != nq:
                raise TableICompileUnavailable(
                    "pauli_label_width_mismatch",
                    f"label {label!r} has width {len(label)}, expected {nq}",
                )
            if label == "e" * nq:
                continue
            try:
                emitted = _append_pauli_rotation_with_qiskit_basis_work(
                    qc,
                    label_exyz=label,
                    angle=float(cfg.structure_theta_value),
                )
            except Exception as exc:
                raise TableICompileUnavailable("pauli_rotation_append_failed", str(exc)) from exc
            _add_qiskit_basis_work_counts(basis_work_counts, emitted)
            runtime_rotation_count += 1
    compiled = _transpile_table_i_circuit(qc, config=cfg)
    stats = _compiled_stats_payload(compiled)
    payload = _base_resource_payload(
        stats=stats,
        source_kind=source_kind,
        config=cfg,
        num_qubits=nq,
        logical_operator_count=len(normalized_groups),
        runtime_rotation_count=runtime_rotation_count,
        reference_state_included=ref_included,
    )
    payload.update(
        _qiskit_basis_work_payload(
            basis_work_counts,
            attributable=True,
        )
    )
    return payload


def _canonical_polynomial_terms(
    polynomial: Any,
    *,
    coefficient_tolerance: float,
) -> tuple[tuple[str, complex], ...]:
    """Return coefficient-bearing Pauli terms in deterministic repo order."""

    combined: dict[str, complex] = {}
    try:
        raw_terms = tuple(polynomial.return_polynomial())
    except Exception as exc:
        raise TableICompileUnavailable("ansatz_term_pauli_coefficients_unavailable", str(exc)) from exc
    for term in raw_terms:
        try:
            label = str(term.pw2strng()).strip().lower()
            coefficient = complex(term.p_coeff)
        except Exception as exc:
            raise TableICompileUnavailable("ansatz_term_pauli_coefficients_unavailable", str(exc)) from exc
        if not label or abs(coefficient) <= float(coefficient_tolerance):
            continue
        combined[label] = combined.get(label, 0.0 + 0.0j) + coefficient
    return tuple(
        (label, coefficient)
        for label, coefficient in sorted(combined.items())
        if abs(coefficient) > float(coefficient_tolerance)
        and set(label) != {"e"}
    )


def _coefficient_payload(terms: Sequence[tuple[str, complex]]) -> list[dict[str, Any]]:
    return [
        {
            "pauli_exyz": str(label),
            "coeff_re": float(complex(coefficient).real),
            "coeff_im": float(complex(coefficient).imag),
        }
        for label, coefficient in terms
    ]


def _coefficient_sha256(terms: Sequence[tuple[str, complex]]) -> str:
    return hashlib.sha256(
        json.dumps(_coefficient_payload(terms), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _pauli_labels_commute(left: str, right: str) -> bool:
    anti_count = 0
    for a, b in zip(str(left), str(right), strict=True):
        if a == "e" or b == "e" or a == b:
            continue
        anti_count += 1
    return anti_count % 2 == 0


def _all_pauli_terms_commute(terms: Sequence[tuple[str, complex]]) -> bool:
    labels = [label for label, _coefficient in terms]
    return all(
        _pauli_labels_commute(labels[i], labels[j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    )


def _active_qubits_for_terms(terms: Sequence[tuple[str, complex]], *, num_qubits: int) -> tuple[int, ...]:
    active: set[int] = set()
    for label, _coefficient in terms:
        if len(label) != int(num_qubits):
            raise TableICompileUnavailable(
                "pauli_label_width_mismatch",
                f"label {label!r} has width {len(label)}, expected {num_qubits}",
            )
        for index, letter in enumerate(label):
            if letter != "e":
                active.add(int(num_qubits) - 1 - int(index))
    return tuple(sorted(active))


def _local_label(label: str, active_qubits: Sequence[int], *, num_qubits: int) -> str:
    # Qiskit Pauli labels are q_(m-1)...q_0 while the circuit arguments below
    # are supplied in ascending local-qubit order.
    return "".join(
        str(label)[int(num_qubits) - 1 - int(qubit)].upper().replace("E", "I")
        for qubit in reversed(tuple(active_qubits))
    )


def _append_exact_active_support_unitary(
    circuit: Any,
    *,
    terms: Sequence[tuple[str, complex]],
    theta: float,
    num_qubits: int,
    max_active_qubits: int,
    coefficient_tolerance: float,
) -> dict[str, Any]:
    try:
        from qiskit.circuit.library import UnitaryGate
        from qiskit.quantum_info import SparsePauliOp
    except Exception as exc:  # pragma: no cover - optional dependency varies
        raise TableICompileUnavailable("qiskit_grouped_exact_synthesis_unavailable", str(exc)) from exc

    active_qubits = _active_qubits_for_terms(terms, num_qubits=int(num_qubits))
    if not active_qubits:
        return {
            "synthesis": "identity_skipped",
            "active_qubits": [],
            "active_support_width": 0,
            "hermitian_error": 0.0,
        }
    if len(active_qubits) > int(max_active_qubits):
        raise TableICompileUnavailable(
            "grouped_exact_active_support_too_wide",
            f"active support width {len(active_qubits)} exceeds exact-synthesis limit {max_active_qubits}",
        )
    qop = SparsePauliOp.from_list(
        [
            (
                _local_label(label, active_qubits, num_qubits=int(num_qubits)),
                complex(coefficient),
            )
            for label, coefficient in terms
        ]
    ).simplify(atol=float(coefficient_tolerance))
    generator = np.asarray(qop.to_matrix(), dtype=complex)
    hermitian_error = float(np.max(np.abs(generator - generator.conj().T))) if generator.size else 0.0
    if hermitian_error > max(1.0e-10, 10.0 * float(coefficient_tolerance)):
        raise TableICompileUnavailable(
            "grouped_exact_generator_not_hermitian",
            f"active-support generator Hermitian error={hermitian_error:.3e}",
        )
    eigvals, eigvecs = np.linalg.eigh(generator)
    unitary = (eigvecs * np.exp(-1.0j * float(theta) * eigvals)) @ eigvecs.conj().T
    circuit.append(UnitaryGate(unitary, label="grouped_exact"), list(active_qubits))
    return {
        "synthesis": "active_support_unitary_exact",
        "active_qubits": [int(qubit) for qubit in active_qubits],
        "active_support_width": int(len(active_qubits)),
        "hermitian_error": hermitian_error,
    }


def build_table_i_execution_aware_circuit(
    *,
    ops: Sequence[Any],
    num_qubits: int,
    reference_state: np.ndarray | Sequence[complex] | None,
    config: TableIQiskitCompileConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build a structural circuit that honors each generator execution mode.

    ``termwise_product`` keeps the established coefficient-scaled Pauli
    rotations.  A commuting ``grouped_exact`` generator uses the same rotations
    exactly.  A noncommuting grouped generator is exponentiated on its active
    support and inserted as an exact unitary block before Qiskit transpilation.
    """

    _require_qiskit_circuit_helpers()
    cfg = config or TableIQiskitCompileConfig()
    nq = int(num_qubits)
    if nq <= 0:
        raise TableICompileUnavailable("invalid_num_qubits", f"num_qubits={num_qubits!r}")
    try:
        from qiskit import QuantumCircuit
    except Exception as exc:  # pragma: no cover - optional dependency varies
        raise TableICompileUnavailable("qiskit_quantum_circuit_unavailable", str(exc)) from exc

    circuit = QuantumCircuit(nq)
    ref_included = bool(cfg.include_reference_state and reference_state is not None)
    if ref_included:
        try:
            append_reference_state(circuit, np.asarray(reference_state, dtype=complex).reshape(-1))
        except Exception as exc:
            raise TableICompileUnavailable("reference_state_preparation_failed", str(exc)) from exc

    operator_rows: list[dict[str, Any]] = []
    runtime_rotation_count = 0
    basis_work_counts = _empty_qiskit_basis_work_counts()
    non_attributable_operator_count = 0
    for index, op in enumerate(tuple(ops)):
        label = str(getattr(op, "label", f"operator_{index}"))
        mode = str(getattr(op, "execution_mode", "termwise_product") or "termwise_product").strip().lower()
        if mode not in {"termwise_product", "grouped_exact"}:
            raise TableICompileUnavailable(
                "unsupported_generator_execution_mode",
                f"operator {label!r} uses unsupported execution_mode={mode!r}",
            )
        terms = _canonical_polynomial_terms(
            getattr(op, "polynomial", None),
            coefficient_tolerance=float(cfg.coefficient_tolerance),
        )
        if any(abs(complex(coefficient).imag) > float(cfg.coefficient_tolerance) for _pauli, coefficient in terms):
            raise TableICompileUnavailable(
                "complex_generator_coefficients_unsupported",
                f"operator {label!r} has a non-real Pauli coefficient",
            )
        coefficient_sha = _coefficient_sha256(terms)
        row: dict[str, Any] = {
            "operator_index": int(index),
            "label": label,
            "execution_mode": mode,
            "pauli_term_count": int(len(terms)),
            "coefficient_sha256": coefficient_sha,
        }
        if not terms:
            row.update(synthesis="identity_skipped", active_support_width=0)
            operator_rows.append(row)
            continue
        commuting = _all_pauli_terms_commute(terms)
        if mode == "termwise_product" or commuting:
            operator_basis_work_counts = _empty_qiskit_basis_work_counts()
            for pauli_label, coefficient in terms:
                emitted = _append_pauli_rotation_with_qiskit_basis_work(
                    circuit,
                    label_exyz=pauli_label,
                    angle=2.0 * float(cfg.structure_theta_value) * float(complex(coefficient).real),
                )
                _add_qiskit_basis_work_counts(
                    operator_basis_work_counts,
                    emitted,
                )
                _add_qiskit_basis_work_counts(basis_work_counts, emitted)
                runtime_rotation_count += 1
            row.update(
                synthesis=(
                    "termwise_product_pauli_rotations"
                    if mode == "termwise_product"
                    else "commuting_pauli_rotations_exact"
                ),
                active_support_width=len(_active_qubits_for_terms(terms, num_qubits=nq)),
                commuting=True,
                qiskit_basis_work=_qiskit_basis_work_payload(
                    operator_basis_work_counts,
                    attributable=True,
                ),
            )
        else:
            exact_meta = _append_exact_active_support_unitary(
                circuit,
                terms=terms,
                theta=float(cfg.structure_theta_value),
                num_qubits=nq,
                max_active_qubits=int(cfg.grouped_exact_max_active_qubits),
                coefficient_tolerance=float(cfg.coefficient_tolerance),
            )
            runtime_rotation_count += 1
            non_attributable_operator_count += 1
            row.update(
                exact_meta,
                commuting=False,
                qiskit_basis_work=_qiskit_basis_work_payload(
                    _empty_qiskit_basis_work_counts(),
                    attributable=False,
                    non_attributable_operator_count=1,
                ),
            )
        operator_rows.append(row)

    coefficient_manifest = [
        {
            "label": row["label"],
            "execution_mode": row["execution_mode"],
            "coefficient_sha256": row["coefficient_sha256"],
        }
        for row in operator_rows
    ]
    return circuit, {
        "reference_state_included": ref_included,
        "logical_operator_count": int(len(tuple(ops))),
        "runtime_rotation_count": int(runtime_rotation_count),
        "grouped_exact_synthesis_id": TABLE_I_GROUPED_EXACT_SYNTHESIS_ID,
        "generator_coefficients_sha256": hashlib.sha256(
            json.dumps(coefficient_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "operator_synthesis": operator_rows,
        "qiskit_basis_work": _qiskit_basis_work_payload(
            basis_work_counts,
            attributable=non_attributable_operator_count == 0,
            non_attributable_operator_count=non_attributable_operator_count,
        ),
    }


def compile_table_i_ansatz_terms(
    *,
    ops: Sequence[AnsatzTerm],
    num_qubits: int,
    reference_state: np.ndarray | Sequence[complex] | None,
    source_kind: str,
    config: TableIQiskitCompileConfig | None = None,
) -> dict[str, Any]:
    """Compile coefficient-bearing terms with execution-aware synthesis."""

    cfg = config or TableIQiskitCompileConfig()
    try:
        circuit, synthesis = build_table_i_execution_aware_circuit(
            ops=tuple(ops),
            num_qubits=int(num_qubits),
            reference_state=reference_state,
            config=cfg,
        )
    except TableICompileUnavailable:
        raise
    except Exception as exc:
        raise TableICompileUnavailable("structural_ansatz_circuit_failed", str(exc)) from exc
    compiled = _transpile_table_i_circuit(circuit, config=cfg)
    stats = _compiled_stats_payload(compiled)
    payload = _base_resource_payload(
        stats=stats,
        source_kind=source_kind,
        config=cfg,
        num_qubits=int(num_qubits),
        logical_operator_count=len(tuple(ops)),
        runtime_rotation_count=int(synthesis["runtime_rotation_count"]),
        reference_state_included=bool(synthesis["reference_state_included"]),
    )
    payload.update(
        grouped_exact_synthesis_id=synthesis["grouped_exact_synthesis_id"],
        generator_coefficients_sha256=synthesis["generator_coefficients_sha256"],
        operator_synthesis=synthesis["operator_synthesis"],
        **synthesis["qiskit_basis_work"],
    )
    return payload


__all__ = [
    "TABLE_I_COMPILED_BASIS_GATES",
    "TABLE_I_GROUPED_EXACT_SYNTHESIS_ID",
    "TABLE_I_QISKIT_BASIS_WORK_SCHEMA",
    "TABLE_I_QISKIT_COMPILE_CONVENTION",
    "TABLE_I_STRUCTURAL_ANGLE_CONVENTION",
    "TableICompileUnavailable",
    "TableIQiskitCompileConfig",
    "build_table_i_execution_aware_circuit",
    "compile_table_i_ansatz_terms",
    "compile_table_i_pauli_label_groups",
    "pauli_label_groups_from_ansatz_terms",
]

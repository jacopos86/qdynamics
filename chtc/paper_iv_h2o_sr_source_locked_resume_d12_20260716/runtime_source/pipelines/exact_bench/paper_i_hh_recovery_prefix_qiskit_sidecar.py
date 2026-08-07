#!/usr/bin/env python3
"""Compile a preserved Paper-I HH active prefix under two locked conventions.

The July-2026 displayed HH SNAKE resource cells were compiled with the
backend-free Table-I basis-gate convention.  Current Joint-Response (JR)
SNAKE sidecars instead bind the saved runtime parameters and transpile the
resulting circuit to ``FakeMarrakesh``.  These are different compilation
experiments.  This postprocessor emits both results without conflating them.

Only an explicit ``paper_i_signed_active_prefix_checkpoint_v1`` is accepted.
In particular, this tool never reconstructs an active prefix by replaying the
admission history: that would silently reinsert operators removed by pruning.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.table_i_qiskit_resource_compile import (  # noqa: E402
    TABLE_I_COMPILED_BASIS_GATES,
    TABLE_I_QISKIT_COMPILE_CONVENTION,
    TABLE_I_STRUCTURAL_ANGLE_CONVENTION,
    TableIQiskitCompileConfig,
    compile_table_i_pauli_label_groups,
)
from pipelines.hardcoded.adapt_circuit_execution import (  # noqa: E402
    build_ansatz_circuit,
)
from pipelines.scaffold.runtime_loader import (  # noqa: E402
    load_scaffold_runtime_input_from_payload,
)
from pipelines.static_adapt.estimator_call_ledger import (  # noqa: E402
    projective_state_fingerprint,
)
from pipelines.static_adapt.statevector_runtime import (  # noqa: E402
    _apply_compiled_polynomial,
    _compile_polynomial_action,
)
from pipelines.qiskit_backend_tools import (  # noqa: E402
    backend_coupling_graph_snapshot,
    compile_circuit_for_backend,
    compiled_gate_stats,
    load_local_fake_backend,
    safe_circuit_depth,
    snapshot_backend_target,
)
from src.quantum.ansatz_parameterization import (  # noqa: E402
    AnsatzParameterLayout,
    deserialize_layout,
    serialize_layout,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor  # noqa: E402


SIDECAR_SCHEMA = "paper_i_hh_recovery_prune_aware_qiskit_sidecar_v1"
CHECKPOINT_SCHEMA = "paper_i_signed_active_prefix_checkpoint_v1"
CHECKPOINT_ORDER_REPAIR_SCHEMA = "paper_i_checkpoint_execution_order_repair_v1"
HISTORICAL_DISPLAYED_CONVENTION = TABLE_I_QISKIT_COMPILE_CONVENTION
CURRENT_JR_CONVENTION = "jr_signed_runtime_fake_marrakesh_transpile_v1"
CURRENT_JR_BACKEND = "FakeMarrakesh"
CURRENT_JR_OPTIMIZATION_LEVEL = 1
LOCKED_SEED_TRANSPILER = 7


@dataclass(frozen=True)
class CheckpointResolution:
    checkpoint: dict[str, Any]
    locations: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedCheckpoint:
    checkpoint: dict[str, Any]
    checkpoint_sha256: str
    layout: AnsatzParameterLayout
    runtime_parameters: np.ndarray
    logical_parameters: np.ndarray
    pauli_label_groups: tuple[tuple[str, ...], ...]
    num_qubits: int


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _mapping_at(payload: Mapping[str, Any], keys: Sequence[str]) -> Mapping[str, Any] | None:
    node: Any = payload
    for key in keys:
        if not isinstance(node, Mapping):
            return None
        node = node.get(str(key))
    return node if isinstance(node, Mapping) else None


def _checkpoint_containers(payload: Mapping[str, Any]) -> list[tuple[str, Any]]:
    containers: list[tuple[str, Any]] = []
    parents: tuple[tuple[str, ...], ...] = (
        (),
        ("adapt_vqe",),
        ("continuation",),
        ("adapt_vqe", "continuation"),
    )
    for parent_keys in parents:
        parent = payload if not parent_keys else _mapping_at(payload, parent_keys)
        if not isinstance(parent, Mapping):
            continue
        stem = ".".join(parent_keys)
        prefix = f"{stem}." if stem else ""
        checkpoints = parent.get("active_prefix_checkpoints")
        if isinstance(checkpoints, Sequence) and not isinstance(
            checkpoints, (str, bytes, bytearray)
        ):
            for index, checkpoint in enumerate(checkpoints):
                if isinstance(checkpoint, Mapping):
                    containers.append(
                        (f"{prefix}active_prefix_checkpoints[{index}]", checkpoint)
                    )
        terminal = parent.get("terminal_active_prefix_checkpoint")
        if isinstance(terminal, Mapping):
            containers.append((f"{prefix}terminal_active_prefix_checkpoint", terminal))
    return containers


def resolve_active_prefix_checkpoint(
    payload: Mapping[str, Any],
    *,
    outer_iteration: int,
    checkpoint_kind: str | None = None,
) -> CheckpointResolution:
    """Resolve one exact active-prefix checkpoint and fail on ambiguity."""

    matches: list[tuple[str, dict[str, Any]]] = []
    for location, raw_checkpoint in _checkpoint_containers(payload):
        try:
            observed_iteration = int(raw_checkpoint.get("outer_iteration"))
        except (TypeError, ValueError):
            continue
        if observed_iteration != int(outer_iteration):
            continue
        if checkpoint_kind is not None and str(raw_checkpoint.get("checkpoint_kind")) != str(
            checkpoint_kind
        ):
            continue
        matches.append((str(location), dict(raw_checkpoint)))
    if not matches:
        kind_note = "" if checkpoint_kind is None else f" and kind={checkpoint_kind!r}"
        raise ValueError(
            f"No active-prefix checkpoint has outer_iteration={int(outer_iteration)}{kind_note}."
        )

    by_identity: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for location, checkpoint in matches:
        identity = str(checkpoint.get("checkpoint_sha256") or _canonical_sha256(checkpoint))
        by_identity.setdefault(identity, []).append((location, checkpoint))
    if len(by_identity) != 1:
        detail = {
            identity: [location for location, _checkpoint in rows]
            for identity, rows in by_identity.items()
        }
        raise ValueError(
            "Multiple nonidentical active-prefix checkpoints match the requested outer "
            f"iteration; pass --checkpoint-kind or repair the result: {detail!r}."
        )
    rows = next(iter(by_identity.values()))
    return CheckpointResolution(
        checkpoint=dict(rows[0][1]),
        locations=tuple(location for location, _checkpoint in rows),
    )


def _checkpoint_hash(checkpoint: Mapping[str, Any]) -> tuple[str, str]:
    expected = str(checkpoint.get("checkpoint_sha256") or "").strip()
    if not expected:
        raise ValueError("Active-prefix checkpoint is missing checkpoint_sha256.")
    hash_input = dict(checkpoint)
    hash_input.pop("checkpoint_sha256", None)
    observed = _canonical_sha256(hash_input)
    if observed != expected:
        raise ValueError(
            "Active-prefix checkpoint SHA-256 mismatch: "
            f"embedded={expected}, recomputed={observed}."
        )
    return expected, observed


def _normalized_term(raw: Mapping[str, Any]) -> dict[str, Any]:
    label = str(raw.get("pauli_exyz") or "").strip().lower()
    if not label:
        raise ValueError("Serialized runtime term is missing pauli_exyz.")
    nq = int(raw.get("nq", len(label)))
    if len(label) != nq:
        raise ValueError(
            f"Serialized runtime term {label!r} has width {len(label)}, expected nq={nq}."
        )
    coeff_re = float(raw.get("coeff_re", 0.0))
    coeff_im = float(raw.get("coeff_im", 0.0))
    if not np.isfinite(coeff_re) or not np.isfinite(coeff_im):
        raise ValueError(f"Serialized runtime term {label!r} has a nonfinite coefficient.")
    return {
        "pauli_exyz": label,
        "coeff_re": coeff_re,
        "coeff_im": coeff_im,
        "nq": nq,
    }


def validate_active_prefix_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_outer_iteration: int,
) -> ValidatedCheckpoint:
    """Validate the signed, ordered, post-prune checkpoint contract."""

    data = dict(checkpoint)
    if str(data.get("schema")) != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"Unsupported checkpoint schema {data.get('schema')!r}; expected {CHECKPOINT_SCHEMA!r}."
        )
    if int(data.get("outer_iteration", -1)) != int(expected_outer_iteration):
        raise ValueError("Checkpoint outer_iteration does not match the requested iteration.")
    checkpoint_sha256, _ = _checkpoint_hash(data)

    labels_raw = data.get("ordered_active_operator_labels")
    operators_raw = data.get("ordered_active_operators")
    if not isinstance(labels_raw, Sequence) or isinstance(labels_raw, (str, bytes, bytearray)):
        raise ValueError("Checkpoint ordered_active_operator_labels must be a sequence.")
    if not isinstance(operators_raw, Sequence) or isinstance(
        operators_raw, (str, bytes, bytearray)
    ):
        raise ValueError("Checkpoint ordered_active_operators must be a sequence.")
    labels = tuple(str(value) for value in labels_raw)
    operators = tuple(value for value in operators_raw if isinstance(value, Mapping))
    active_depth = int(data.get("active_ansatz_depth", -1))
    if len(operators) != len(tuple(operators_raw)):
        raise ValueError("Every ordered_active_operators row must be a mapping.")
    if active_depth != len(labels) or active_depth != len(operators):
        raise ValueError(
            "Checkpoint active depth disagrees with ordered labels/operators: "
            f"depth={active_depth}, labels={len(labels)}, operators={len(operators)}."
        )
    operator_labels = tuple(str(row.get("label")) for row in operators)
    if labels != operator_labels:
        raise ValueError("Checkpoint ordered label list disagrees with ordered operator rows.")

    parameterization = data.get("parameterization")
    if not isinstance(parameterization, Mapping):
        raise ValueError("Checkpoint is missing a serialized parameterization mapping.")
    layout = deserialize_layout(parameterization)
    if int(layout.logical_parameter_count) != active_depth:
        raise ValueError(
            "Serialized parameterization logical depth does not match active_ansatz_depth."
        )
    layout_labels = tuple(str(block.candidate_label) for block in layout.blocks)
    if layout_labels != labels:
        raise ValueError("Serialized parameterization block order disagrees with active operator order.")

    runtime_parameters = np.asarray(
        data.get("signed_unwrapped_runtime_parameters", []), dtype=float
    ).reshape(-1)
    logical_parameters = np.asarray(
        data.get("signed_unwrapped_logical_parameters", []), dtype=float
    ).reshape(-1)
    if not np.all(np.isfinite(runtime_parameters)) or not np.all(
        np.isfinite(logical_parameters)
    ):
        raise ValueError("Checkpoint contains nonfinite signed parameters.")
    if int(runtime_parameters.size) != int(layout.runtime_parameter_count):
        raise ValueError(
            "Signed runtime parameter count disagrees with serialized parameterization: "
            f"theta={runtime_parameters.size}, layout={layout.runtime_parameter_count}."
        )
    if int(logical_parameters.size) != int(layout.logical_parameter_count):
        raise ValueError(
            "Signed logical parameter count disagrees with serialized parameterization: "
            f"theta={logical_parameters.size}, layout={layout.logical_parameter_count}."
        )

    pauli_label_groups: list[tuple[str, ...]] = []
    widths: set[int] = set()
    for operator_index, (operator, block) in enumerate(zip(operators, layout.blocks, strict=True)):
        raw_terms = operator.get("serialized_terms_exyz_in_execution_order")
        if not isinstance(raw_terms, Sequence) or isinstance(
            raw_terms, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"Active operator {operator_index} lacks serialized execution-order terms."
            )
        observed_terms = tuple(
            _normalized_term(raw) for raw in raw_terms if isinstance(raw, Mapping)
        )
        if len(observed_terms) != len(tuple(raw_terms)):
            raise ValueError(f"Active operator {operator_index} has a non-mapping runtime term.")
        expected_terms = tuple(
            {
                "pauli_exyz": str(term.pauli_exyz).lower(),
                "coeff_re": float(term.coeff_real),
                "coeff_im": 0.0,
                "nq": int(term.nq),
            }
            for term in block.terms
        )
        if observed_terms != expected_terms:
            raise ValueError(
                f"Active operator {operator_index} execution terms disagree with parameterization."
            )
        labels_now = tuple(str(row["pauli_exyz"]) for row in observed_terms)
        if not labels_now:
            raise ValueError(f"Active operator {operator_index} has no nonidentity runtime terms.")
        widths.update(int(row["nq"]) for row in observed_terms)
        pauli_label_groups.append(labels_now)
    if len(widths) != 1:
        raise ValueError(f"Active prefix has inconsistent Pauli widths: {sorted(widths)!r}.")
    num_qubits = int(next(iter(widths)))
    return ValidatedCheckpoint(
        checkpoint=data,
        checkpoint_sha256=checkpoint_sha256,
        layout=layout,
        runtime_parameters=runtime_parameters,
        logical_parameters=logical_parameters,
        pauli_label_groups=tuple(pauli_label_groups),
        num_qubits=num_qubits,
    )


def derive_execution_order_repaired_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_outer_iteration: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Repair only permutation-order drift against the runtime parameterization.

    Historical recovery checkpoints copied native-order compile metadata into a
    field explicitly named ``serialized_terms_exyz_in_execution_order`` even
    though the executor used the sorted terms in ``parameterization``.  This
    derivation remains fail-closed on any coefficient, width, label, or term-set
    disagreement and preserves the original checkpoint hash in its repair
    record.
    """

    source = copy.deepcopy(dict(checkpoint))
    source_sha256, _ = _checkpoint_hash(source)
    if str(source.get("schema")) != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"Unsupported checkpoint schema {source.get('schema')!r}; expected {CHECKPOINT_SCHEMA!r}."
        )
    if int(source.get("outer_iteration", -1)) != int(expected_outer_iteration):
        raise ValueError("Checkpoint outer_iteration does not match the requested iteration.")
    parameterization = source.get("parameterization")
    if not isinstance(parameterization, Mapping):
        raise ValueError("Checkpoint is missing a serialized parameterization mapping.")
    layout = deserialize_layout(parameterization)
    operators_raw = source.get("ordered_active_operators")
    if not isinstance(operators_raw, Sequence) or isinstance(
        operators_raw, (str, bytes, bytearray)
    ):
        raise ValueError("Checkpoint ordered_active_operators must be a sequence.")
    operators = [row for row in operators_raw if isinstance(row, Mapping)]
    if len(operators) != len(tuple(operators_raw)) or len(operators) != len(layout.blocks):
        raise ValueError("Checkpoint operators cannot be aligned with parameterization blocks.")

    repaired_indices: list[int] = []
    repaired_labels: list[str] = []
    repaired_operators: list[dict[str, Any]] = []
    for operator_index, (operator_raw, block) in enumerate(
        zip(operators, layout.blocks, strict=True)
    ):
        operator = copy.deepcopy(dict(operator_raw))
        if str(operator.get("label")) != str(block.candidate_label):
            raise ValueError(
                f"Active operator {operator_index} label disagrees with parameterization."
            )
        raw_terms = operator.get("serialized_terms_exyz_in_execution_order")
        if not isinstance(raw_terms, Sequence) or isinstance(
            raw_terms, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"Active operator {operator_index} lacks serialized execution-order terms."
            )
        observed_terms = tuple(
            _normalized_term(raw) for raw in raw_terms if isinstance(raw, Mapping)
        )
        if len(observed_terms) != len(tuple(raw_terms)):
            raise ValueError(f"Active operator {operator_index} has a non-mapping runtime term.")
        expected_terms = tuple(
            {
                "pauli_exyz": str(term.pauli_exyz).lower(),
                "coeff_re": float(term.coeff_real),
                "coeff_im": 0.0,
                "nq": int(term.nq),
            }
            for term in block.terms
        )
        if observed_terms != expected_terms:
            observed_multiset = sorted(
                (
                    str(row["pauli_exyz"]),
                    float(row["coeff_re"]),
                    float(row["coeff_im"]),
                    int(row["nq"]),
                )
                for row in observed_terms
            )
            expected_multiset = sorted(
                (
                    str(row["pauli_exyz"]),
                    float(row["coeff_re"]),
                    float(row["coeff_im"]),
                    int(row["nq"]),
                )
                for row in expected_terms
            )
            if observed_multiset != expected_multiset:
                raise ValueError(
                    f"Active operator {operator_index} terms differ substantively from parameterization."
                )
            operator["serialized_terms_exyz_in_execution_order"] = [
                dict(row) for row in expected_terms
            ]
            repaired_indices.append(int(operator_index))
            repaired_labels.append(str(block.candidate_label))
        repaired_operators.append(operator)

    repaired = copy.deepcopy(source)
    repaired["ordered_active_operators"] = repaired_operators
    repaired.pop("checkpoint_sha256", None)
    repaired_sha256 = _canonical_sha256(repaired)
    repaired["checkpoint_sha256"] = repaired_sha256
    validate_active_prefix_checkpoint(
        repaired,
        expected_outer_iteration=int(expected_outer_iteration),
    )
    return repaired, {
        "status": "repaired_permutation_only" if repaired_indices else "already_consistent",
        "authority": "checkpoint.parameterization.blocks[].runtime_terms_exyz",
        "authority_reason": (
            "The selected-state executor and signed runtime parameter vector use this sorted layout."
        ),
        "source_checkpoint_sha256": source_sha256,
        "repaired_checkpoint_sha256": repaired_sha256,
        "repaired_operator_count": int(len(repaired_indices)),
        "repaired_operator_indices": repaired_indices,
        "repaired_operator_labels": repaired_labels,
        "substantive_term_changes": False,
    }


def build_checkpoint_order_repair_record(
    *,
    result_json: Path,
    outer_iteration: int,
    checkpoint_kind: str | None = None,
    expected_result_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    source_path = Path(result_json).resolve()
    source_sha256 = _sha256_path(source_path)
    if expected_result_sha256 is not None and source_sha256 != str(
        expected_result_sha256
    ).strip():
        raise ValueError(
            f"Result SHA-256 mismatch: expected={expected_result_sha256}, observed={source_sha256}."
        )
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Result JSON root must be a mapping.")
    resolution = resolve_active_prefix_checkpoint(
        payload,
        outer_iteration=int(outer_iteration),
        checkpoint_kind=checkpoint_kind,
    )
    source_checkpoint_sha256, _ = _checkpoint_hash(resolution.checkpoint)
    if expected_checkpoint_sha256 is not None and source_checkpoint_sha256 != str(
        expected_checkpoint_sha256
    ).strip():
        raise ValueError(
            "Checkpoint SHA-256 mismatch against caller lock: "
            f"expected={expected_checkpoint_sha256}, observed={source_checkpoint_sha256}."
        )
    repaired, repair = derive_execution_order_repaired_checkpoint(
        resolution.checkpoint,
        expected_outer_iteration=int(outer_iteration),
    )
    return {
        "schema": CHECKPOINT_ORDER_REPAIR_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "result_json": str(source_path),
            "result_sha256": source_sha256,
            "checkpoint_locations": list(resolution.locations),
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "checkpoint_sha256": source_checkpoint_sha256,
            "checkpoint_hash_verified": True,
            "outer_iteration": int(outer_iteration),
            "checkpoint_kind": str(resolution.checkpoint.get("checkpoint_kind")),
        },
        "repair": repair,
        "repaired_checkpoint": repaired,
    }


def _complex_amplitude(raw: Any) -> complex:
    if isinstance(raw, Mapping):
        return complex(
            float(raw.get("re", raw.get("real", 0.0))),
            float(raw.get("im", raw.get("imag", 0.0))),
        )
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        if len(raw) != 2:
            raise ValueError("Statevector amplitude sequences must have length two.")
        return complex(float(raw[0]), float(raw[1]))
    return complex(float(raw), 0.0)


def _reference_state_payload(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    candidates: tuple[tuple[str, ...], ...] = (
        ("ansatz_input_state",),
        ("adapt_vqe", "ansatz_input_state"),
        ("continuation", "ansatz_input_state"),
        ("adapt_vqe", "continuation", "ansatz_input_state"),
    )
    for keys in candidates:
        state = _mapping_at(payload, keys)
        if isinstance(state, Mapping):
            return state, ".".join(keys)
    raise ValueError(
        "Result has no preserved ansatz_input_state; fixed-prefix compilation fails closed."
    )


def reconstruct_reference_state(
    payload: Mapping[str, Any], *, num_qubits: int
) -> tuple[np.ndarray, dict[str, Any]]:
    state_payload, location = _reference_state_payload(payload)
    nq = int(
        state_payload.get(
            "nq_total", state_payload.get("num_qubits", state_payload.get("nq", 0))
        )
        or 0
    )
    if nq != int(num_qubits):
        raise ValueError(
            f"Reference-state width nq={nq} disagrees with active-prefix width {num_qubits}."
        )
    amplitudes = state_payload.get("amplitudes_qn_to_q0")
    if not isinstance(amplitudes, Mapping) or not amplitudes:
        raise ValueError("Reference state is missing amplitudes_qn_to_q0.")
    state = np.zeros(1 << nq, dtype=complex)
    for bitstring_raw, raw_amplitude in amplitudes.items():
        bitstring = str(bitstring_raw).strip()
        if len(bitstring) != nq or set(bitstring) - {"0", "1"}:
            raise ValueError(f"Invalid q_(n-1)..q_0 reference bitstring {bitstring!r}.")
        state[int(bitstring, 2)] = _complex_amplitude(raw_amplitude)
    norm_before = float(np.linalg.norm(state))
    if not np.isfinite(norm_before) or norm_before <= 0.0:
        raise ValueError("Preserved reference state has zero or nonfinite norm.")
    state = np.asarray(state / norm_before, dtype=complex).reshape(-1)
    return state, {
        "source_location": location,
        "source": state_payload.get("source"),
        "handoff_state_kind": state_payload.get("handoff_state_kind"),
        "num_qubits": nq,
        "amplitude_count": int(len(amplitudes)),
        "input_norm": norm_before,
        "normalized_for_circuit": True,
        "bitstring_order": "q_(n-1)...q_0; q0 is rightmost",
    }


def replay_fixed_prefix(
    payload: Mapping[str, Any],
    *,
    result_json: Path,
    validated: ValidatedCheckpoint,
    energy_tolerance_abs: float = 1.0e-12,
) -> dict[str, Any]:
    """Rebuild the preserved ordered prefix and replay its exact-state energy."""

    runtime_input = load_scaffold_runtime_input_from_payload(
        payload,
        artifact_json=Path(result_json),
    )
    checkpoint_parameterization = validated.checkpoint.get("parameterization")
    runtime_parameterization = serialize_layout(runtime_input.base_layout)
    if checkpoint_parameterization != runtime_parameterization:
        raise ValueError(
            "Runtime-loader parameterization disagrees with the validated checkpoint."
        )
    checkpoint_labels = tuple(
        str(value) for value in validated.checkpoint["ordered_active_operator_labels"]
    )
    runtime_labels = tuple(str(term.label) for term in runtime_input.selected_terms)
    if checkpoint_labels != runtime_labels:
        raise ValueError("Runtime-loader prefix order disagrees with the validated checkpoint.")
    parameterization_mode = str(
        validated.checkpoint.get("parameterization_mode") or ""
    ).strip()
    if parameterization_mode == "logical_shared":
        executor_mode = "logical_shared"
        theta = np.asarray(validated.logical_parameters, dtype=float)
    elif parameterization_mode in {"per_pauli_term", "per_pauli_term_v1"}:
        executor_mode = "per_pauli_term"
        theta = np.asarray(validated.runtime_parameters, dtype=float)
    else:
        raise ValueError(
            f"Unsupported fixed-prefix replay parameterization mode {parameterization_mode!r}."
        )
    executor = CompiledAnsatzExecutor(
        runtime_input.selected_terms,
        parameterization_mode=executor_mode,
        parameterization_layout=runtime_input.base_layout,
    )
    state = executor.prepare_state(
        theta,
        np.asarray(runtime_input.psi_ref, dtype=complex),
    )
    compiled_hamiltonian = _compile_polynomial_action(runtime_input.h_poly)
    applied = _apply_compiled_polynomial(state, compiled_hamiltonian)
    replayed_energy = float(np.real(np.vdot(state, applied)))
    adapt_vqe = payload.get("adapt_vqe")
    if not isinstance(adapt_vqe, Mapping) or adapt_vqe.get("energy") is None:
        raise ValueError("Result is missing adapt_vqe.energy for fixed-prefix replay.")
    reported_energy = float(adapt_vqe["energy"])
    energy_abs_discrepancy = float(abs(replayed_energy - reported_energy))
    observed_fingerprint = projective_state_fingerprint(state)
    expected_fingerprint = str(
        validated.checkpoint.get("projective_state_fingerprint") or ""
    )
    fingerprint_matches = bool(
        expected_fingerprint and observed_fingerprint == expected_fingerprint
    )
    passed = bool(
        np.isfinite(replayed_energy)
        and energy_abs_discrepancy <= float(energy_tolerance_abs)
        and fingerprint_matches
    )
    if not passed:
        raise ValueError(
            "Fixed-prefix replay failed energy/fingerprint validation: "
            f"abs_delta={energy_abs_discrepancy}, tolerance={energy_tolerance_abs}, "
            f"fingerprint_matches={fingerprint_matches}."
        )
    return {
        "status": "pass",
        "prefix_reconstructed": True,
        "prefix_order_matches_checkpoint": True,
        "parameterization_matches_checkpoint": True,
        "parameterization_mode": parameterization_mode,
        "active_ansatz_depth": int(validated.layout.logical_parameter_count),
        "runtime_parameter_count": int(validated.layout.runtime_parameter_count),
        "reported_energy": reported_energy,
        "replayed_energy": replayed_energy,
        "energy_abs_discrepancy": energy_abs_discrepancy,
        "energy_tolerance_abs": float(energy_tolerance_abs),
        "projective_state_fingerprint_expected": expected_fingerprint,
        "projective_state_fingerprint_replayed": observed_fingerprint,
        "projective_state_fingerprint_matches": fingerprint_matches,
        "state_norm": float(np.linalg.norm(state)),
    }


def _package_version(package: str) -> str | None:
    try:
        return str(importlib_metadata.version(package))
    except Exception:
        return None


def _source_file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256_path(resolved),
    }


def _metric_triplet(
    payload: Mapping[str, Any],
    *,
    count_key: str,
    depth_2q_key: str,
    depth_key: str,
) -> dict[str, int]:
    return {
        "N2q": int(payload[count_key]),
        "D2q": int(payload[depth_2q_key]),
        "Dc": int(payload[depth_key]),
    }


def compile_historical_displayed_convention(
    validated: ValidatedCheckpoint,
    *,
    reference_state: np.ndarray,
) -> dict[str, Any]:
    config = TableIQiskitCompileConfig(
        basis_gates=tuple(TABLE_I_COMPILED_BASIS_GATES),
        optimization_level=0,
        seed_transpiler=LOCKED_SEED_TRANSPILER,
        structure_theta_value=1.0,
        include_reference_state=True,
        compile_convention=TABLE_I_QISKIT_COMPILE_CONVENTION,
    )
    compiled = compile_table_i_pauli_label_groups(
        pauli_label_groups=validated.pauli_label_groups,
        num_qubits=int(validated.num_qubits),
        reference_state=reference_state,
        source_kind="paper_i_hh_recovery_active_post_prune_prefix",
        config=config,
    )
    return {
        "identity": HISTORICAL_DISPLAYED_CONVENTION,
        "status": "ok",
        "prune_aware": True,
        "backend": None,
        "coupling_map": None,
        "basis_gates": list(TABLE_I_COMPILED_BASIS_GATES),
        "optimization_level": 0,
        "seed_transpiler": LOCKED_SEED_TRANSPILER,
        "angle_convention": TABLE_I_STRUCTURAL_ANGLE_CONVENTION,
        "parameter_source": "structural_nonzero_placeholder; saved optimized theta not bound",
        "reference_state_included": True,
        "metrics": _metric_triplet(
            compiled,
            count_key="compiled_count_2q_total",
            depth_2q_key="compiled_depth_2q_total",
            depth_key="compiled_depth_total",
        ),
        "raw_compile_payload": dict(compiled),
    }


def compile_current_jr_convention(
    validated: ValidatedCheckpoint,
    *,
    reference_state: np.ndarray,
) -> dict[str, Any]:
    circuit = build_ansatz_circuit(
        validated.layout,
        np.asarray(validated.runtime_parameters, dtype=float),
        int(validated.num_qubits),
        ref_state=np.asarray(reference_state, dtype=complex),
    )
    backend, resolved_name = load_local_fake_backend(CURRENT_JR_BACKEND)
    compiled_info = compile_circuit_for_backend(
        circuit,
        backend,
        seed_transpiler=LOCKED_SEED_TRANSPILER,
        optimization_level=CURRENT_JR_OPTIMIZATION_LEVEL,
    )
    compiled_circuit = compiled_info["compiled"]
    stats = dict(compiled_gate_stats(compiled_circuit))
    stats["compiled_depth"] = int(safe_circuit_depth(compiled_circuit))
    graph_snapshot = dict(backend_coupling_graph_snapshot(backend))
    graph_sha256 = _canonical_sha256(graph_snapshot)
    return {
        "identity": CURRENT_JR_CONVENTION,
        "status": "ok",
        "prune_aware": True,
        "pipeline_identity": "adapt_circuit_compile_scout",
        "requested_backend": CURRENT_JR_BACKEND,
        "resolved_backend": str(resolved_name),
        "backend_class": {
            "module": str(type(backend).__module__),
            "name": str(type(backend).__name__),
        },
        "backend_target": dict(snapshot_backend_target(backend)),
        "coupling_graph": graph_snapshot,
        "coupling_graph_sha256": graph_sha256,
        "optimization_level": CURRENT_JR_OPTIMIZATION_LEVEL,
        "seed_transpiler": LOCKED_SEED_TRANSPILER,
        "angle_convention": "angle = 2 * signed_runtime_theta[i] * coeff_real",
        "parameter_source": "checkpoint.signed_unwrapped_runtime_parameters",
        "reference_state_included": True,
        "logical_circuit": {
            "num_qubits": int(circuit.num_qubits),
            "size": int(circuit.size()),
            "depth": int(safe_circuit_depth(circuit)),
            "logical_operator_count": int(validated.layout.logical_parameter_count),
            "runtime_parameter_count": int(validated.layout.runtime_parameter_count),
        },
        "metrics": _metric_triplet(
            stats,
            count_key="compiled_count_2q",
            depth_2q_key="compiled_depth_2q",
            depth_key="compiled_depth",
        ),
        "compiled_stats": stats,
        "compiled_num_qubits": int(compiled_info["compiled_num_qubits"]),
        "logical_to_physical_qubits": [
            int(value) for value in compiled_info["logical_to_physical"]
        ],
    }


def build_sidecar(
    *,
    result_json: Path,
    outer_iteration: int,
    checkpoint_kind: str | None = None,
    expected_result_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    checkpoint_order_repair_record: Mapping[str, Any] | None = None,
    require_fixed_prefix_replay: bool = False,
) -> dict[str, Any]:
    source_path = Path(result_json).resolve()
    source_sha256 = _sha256_path(source_path)
    if expected_result_sha256 is not None and source_sha256 != str(
        expected_result_sha256
    ).strip():
        raise ValueError(
            f"Result SHA-256 mismatch: expected={expected_result_sha256}, observed={source_sha256}."
        )
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Result JSON root must be a mapping.")
    resolution = resolve_active_prefix_checkpoint(
        payload,
        outer_iteration=int(outer_iteration),
        checkpoint_kind=checkpoint_kind,
    )
    source_checkpoint_sha256, _ = _checkpoint_hash(resolution.checkpoint)
    if expected_checkpoint_sha256 is not None and source_checkpoint_sha256 != str(
        expected_checkpoint_sha256
    ).strip():
        raise ValueError(
            "Checkpoint SHA-256 mismatch against caller lock: "
            f"expected={expected_checkpoint_sha256}, observed={source_checkpoint_sha256}."
        )
    checkpoint_for_validation: Mapping[str, Any] = resolution.checkpoint
    order_repair_summary: dict[str, Any] | None = None
    if checkpoint_order_repair_record is not None:
        repair_record = dict(checkpoint_order_repair_record)
        if str(repair_record.get("schema")) != CHECKPOINT_ORDER_REPAIR_SCHEMA:
            raise ValueError("Unsupported checkpoint execution-order repair schema.")
        repair_source = repair_record.get("source")
        repair_summary = repair_record.get("repair")
        repaired_checkpoint = repair_record.get("repaired_checkpoint")
        if not isinstance(repair_source, Mapping) or not isinstance(
            repair_summary, Mapping
        ) or not isinstance(repaired_checkpoint, Mapping):
            raise ValueError("Checkpoint execution-order repair record is incomplete.")
        if str(repair_source.get("result_sha256")) != source_sha256:
            raise ValueError("Checkpoint repair result hash disagrees with the requested result.")
        if str(repair_source.get("checkpoint_sha256")) != source_checkpoint_sha256:
            raise ValueError("Checkpoint repair source hash disagrees with the resolved checkpoint.")
        repaired_checkpoint_sha256, _ = _checkpoint_hash(repaired_checkpoint)
        if str(repair_summary.get("repaired_checkpoint_sha256")) != repaired_checkpoint_sha256:
            raise ValueError("Checkpoint repair record disagrees with the repaired checkpoint hash.")
        checkpoint_for_validation = repaired_checkpoint
        order_repair_summary = dict(repair_summary)
    validated = validate_active_prefix_checkpoint(
        checkpoint_for_validation,
        expected_outer_iteration=int(outer_iteration),
    )
    reference_state, reference_meta = reconstruct_reference_state(
        payload, num_qubits=int(validated.num_qubits)
    )
    try:
        fixed_prefix_replay = replay_fixed_prefix(
            payload,
            result_json=source_path,
            validated=validated,
        )
    except Exception as exc:
        if bool(require_fixed_prefix_replay):
            raise
        fixed_prefix_replay = {
            "status": "unavailable",
            "prefix_reconstructed": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    historical = compile_historical_displayed_convention(
        validated, reference_state=reference_state
    )
    current_jr = compile_current_jr_convention(
        validated, reference_state=reference_state
    )

    table_compiler_path = Path(
        sys.modules[
            "pipelines.exact_bench.table_i_qiskit_resource_compile"
        ].__file__
    )
    backend_tools_path = Path(sys.modules["pipelines.qiskit_backend_tools"].__file__)
    execution_path = Path(
        sys.modules["pipelines.hardcoded.adapt_circuit_execution"].__file__
    )
    return {
        "schema": SIDECAR_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "source": {
            "result_json": str(source_path),
            "result_sha256": source_sha256,
            "checkpoint_locations": list(resolution.locations),
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "checkpoint_sha256": validated.checkpoint_sha256,
            "checkpoint_hash_verified": True,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "checkpoint_was_derived": bool(order_repair_summary is not None),
            "checkpoint_execution_order_repair": order_repair_summary,
            "outer_iteration": int(outer_iteration),
            "checkpoint_kind": str(validated.checkpoint.get("checkpoint_kind")),
        },
        "prefix": {
            "prune_aware": True,
            "active_prefix_source": "explicit preserved checkpoint; admission history not replayed",
            "active_ansatz_depth": int(validated.layout.logical_parameter_count),
            "runtime_parameter_count": int(validated.layout.runtime_parameter_count),
            "ordered_active_operator_labels": list(
                validated.checkpoint["ordered_active_operator_labels"]
            ),
            "signed_parameter_source": "preserved full-precision checkpoint",
            "execution_order_source": (
                "derived checkpoint.parameterization.blocks[].runtime_terms_exyz"
                if order_repair_summary is not None
                else "checkpoint.ordered_active_operators[].serialized_terms_exyz_in_execution_order"
            ),
            "projective_state_fingerprint": validated.checkpoint.get(
                "projective_state_fingerprint"
            ),
            "boson_legal_codeword_probability": validated.checkpoint.get(
                "boson_legal_codeword_probability"
            ),
            "fixed_spin_sector_probability": validated.checkpoint.get(
                "fixed_spin_sector_probability"
            ),
        },
        "reference_state": reference_meta,
        "fixed_prefix_replay": fixed_prefix_replay,
        "historical_displayed_convention": historical,
        "current_jr_fake_marrakesh_convention": current_jr,
        "convention_comparison": {
            "same_convention": False,
            "direct_metric_equality_implies_method_equivalence": False,
            "differences": [
                "historical displayed compiler has no backend target or coupling map",
                "historical displayed compiler uses a broad explicit basis gate set",
                "historical displayed compiler uses optimization_level=0",
                "historical displayed compiler uses structural placeholder angles",
                "current JR compiler binds the preserved signed runtime parameters",
                "current JR compiler targets FakeMarrakesh at optimization_level=1",
            ],
        },
        "software": {
            "python": sys.version.split()[0],
            "qiskit": _package_version("qiskit"),
            "qiskit_terra": _package_version("qiskit-terra"),
            "qiskit_ibm_runtime": _package_version("qiskit-ibm-runtime"),
        },
        "implementation_sources": {
            "postprocessor": _source_file_record(Path(__file__)),
            "historical_table_i_compiler": _source_file_record(table_compiler_path),
            "ansatz_circuit_builder": _source_file_record(execution_path),
            "backend_transpile_tools": _source_file_record(backend_tools_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile one preserved, post-prune Paper-I HH active-prefix checkpoint under "
            "the historical displayed and current JR FakeMarrakesh conventions."
        )
    )
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--outer-iteration", type=int, required=True)
    parser.add_argument("--checkpoint-kind", default=None)
    parser.add_argument("--expected-result-sha256", default=None)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument(
        "--repair-permutation-only-execution-order",
        action="store_true",
        help=(
            "Derive a separately preserved checkpoint whose execution-order terms follow "
            "the runtime parameterization; fail on any non-permutation term difference."
        ),
    )
    parser.add_argument("--repaired-checkpoint-json", type=Path, default=None)
    parser.add_argument("--require-fixed-prefix-replay", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    repair_record = None
    if bool(args.repair_permutation_only_execution_order):
        if args.repaired_checkpoint_json is None:
            raise ValueError(
                "--repaired-checkpoint-json is required when execution-order repair is requested."
            )
        repair_record = build_checkpoint_order_repair_record(
            result_json=Path(args.result_json),
            outer_iteration=int(args.outer_iteration),
            checkpoint_kind=args.checkpoint_kind,
            expected_result_sha256=args.expected_result_sha256,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        )
        repaired_path = Path(args.repaired_checkpoint_json)
        repaired_path.parent.mkdir(parents=True, exist_ok=True)
        repaired_path.write_text(
            json.dumps(repair_record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    sidecar = build_sidecar(
        result_json=Path(args.result_json),
        outer_iteration=int(args.outer_iteration),
        checkpoint_kind=args.checkpoint_kind,
        expected_result_sha256=args.expected_result_sha256,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        checkpoint_order_repair_record=repair_record,
        require_fixed_prefix_replay=bool(args.require_fixed_prefix_replay),
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.repaired_checkpoint_json is not None:
        print(Path(args.repaired_checkpoint_json))
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

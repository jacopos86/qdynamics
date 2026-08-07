#!/usr/bin/env python3
"""Audit Paper-I visible rows with same-cutoff ground-space fidelity.

This is a post-run reporting tool.  It never refits an ansatz, changes a
controller decision, or imports an exact state into a scientific run.  A row
is evaluated only when its saved terminal state (or an exact signed prefix
checkpoint) is replayable.  Missing prefix parameters fail closed instead of
being reconstructed by truncating terminal parameters or by running Powell.

The reported quantity is projector fidelity with the complete physical-sector
ground space.  The shared implementation lives in
``pipelines.scaffold.ground_space_fidelity`` and records the degeneracy
tolerance, gap, multiplicity, physical-basis hash, and projector hash.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.scaffold.ground_space_fidelity import (  # noqa: E402
    GroundSpaceFidelityError,
    GroundSpaceTolerance,
    evaluate_ground_space_fidelity,
)


AUDIT_SCHEMA = "paper_i_main_ground_space_fidelity_audit_v1"
ROW_SCHEMA = "paper_i_main_ground_space_fidelity_row_v1"


class FidelityBlocked(RuntimeError):
    """Expected fail-closed audit blocker with a stable status code."""

    def __init__(self, status: str, message: str | None = None) -> None:
        super().__init__(message or status)
        self.status = str(status)


def _resolve(path: str | Path) -> Path:
    value = Path(str(path)).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise FidelityBlocked("source_json_root_not_mapping")
    return dict(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _finite_float(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and value == ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    number = _finite_float(value)
    return None if number is None else int(round(number))


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _family_key(value: Any) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "hubbard_family": "hubbard",
        "hubbard_holstein_family": "hubbard_holstein",
        "hh_family": "hubbard_holstein",
        "spin_boson_family": "spin_boson",
    }
    return aliases.get(key, key.removesuffix("_family"))


def _regime_from_case(case_id: Any) -> str:
    text = str(case_id or "").strip().lower()
    for suffix in (
        "strong_strong_u8",
        "strong_weak_u8",
        "intermediate_strong",
        "intermediate_weak",
        "weak_strong",
        "strong_strong",
        "strong_weak",
        "weak_weak",
        "intermediate",
        "strong",
        "weak",
    ):
        if text.endswith(suffix):
            return suffix
    return text.rsplit("_", 1)[-1] if text else "unknown"


def _tables_i_ii_specs(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_map_path = Path(path)
    payload = _read_json(source_map_path)
    specs: list[dict[str, Any]] = []
    rows = payload.get("source_rows", [])
    if not _sequence(rows):
        raise FidelityBlocked("tables_i_ii_source_rows_invalid")
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            continue
        family = _family_key(raw_row.get("family"))
        cases = raw_row.get("cases", [])
        if not _sequence(cases):
            continue
        for raw_case in cases:
            if not isinstance(raw_case, Mapping):
                continue
            case_id = str(raw_case.get("case_id") or "").strip()
            source_json = raw_case.get("generic_static_single_json")
            if source_json is None:
                source_json = raw_case.get("source_json")
            specs.append(
                {
                    "table_label": raw_row.get("table_label"),
                    "table_surface": "tables_i_ii",
                    "family": family,
                    "case_id": case_id,
                    "regime": str(raw_case.get("regime") or _regime_from_case(case_id)),
                    "method": raw_row.get("method_label", raw_row.get("method")),
                    "algorithm_id": raw_row.get("method_id", raw_row.get("algorithm_id")),
                    "source_json": None if source_json is None else str(source_json),
                    "source_sha256": raw_case.get(
                        "generic_static_single_sha256", raw_case.get("source_sha256")
                    ),
                    "visible_one_minus_F": raw_case.get(
                        "one_minus_F_display", raw_case.get("visible_one_minus_F", "--")
                    ),
                    "prefix_operator_count": None,
                    "source_map_json": str(source_map_path),
                }
            )
    return specs, payload


def _source_map_rows(path: str | Path | None) -> list[Mapping[str, Any]]:
    if path is None:
        return []
    payload = _read_json(Path(path))
    rows = payload.get("rows", payload.get("source_rows", []))
    if not _sequence(rows):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _hh_source_lookup(
    rows: Sequence[Mapping[str, Any]], *, regime: str, method: str
) -> Mapping[str, Any] | None:
    regime_key = str(regime).strip().lower().replace("-", "_")
    method_key = str(method).strip().lower()
    matches = [
        row
        for row in rows
        if str(row.get("regime") or "").strip().lower().replace("-", "_")
        == regime_key
        and str(row.get("method") or row.get("method_label") or "").strip().lower()
        == method_key
    ]
    if len(matches) > 1:
        raise FidelityBlocked("hh_source_map_ambiguous")
    return matches[0] if matches else None


def _hh_table_iii_specs(
    path: str | Path,
    source_map_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_path = Path(path)
    payload = _read_json(audit_path)
    source_rows = _source_map_rows(source_map_path)
    rows = payload.get("rows", [])
    if not _sequence(rows):
        raise FidelityBlocked("hh_table_iii_rows_invalid")
    specs: list[dict[str, Any]] = []
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            continue
        regime = str(raw_row.get("regime") or "unknown").strip().lower().replace(
            "-", "_"
        )
        method = str(raw_row.get("method") or raw_row.get("method_label") or "")
        joined = _hh_source_lookup(source_rows, regime=regime, method=method)
        source_json = raw_row.get("source_json")
        if source_json is None and isinstance(joined, Mapping):
            source_json = joined.get("source_json")
        source_sha = raw_row.get("source_sha256")
        if source_sha is None and isinstance(joined, Mapping):
            source_sha = joined.get("source_sha256")
        n_ph = _int_or_none(raw_row.get("n_ph_work", raw_row.get("n_ph")))
        case_id = raw_row.get("case_id")
        if case_id is None and n_ph is not None:
            case_id = f"hh_L2_nph{n_ph}_three_model_sym_{regime}"
        specs.append(
            {
                "table_label": raw_row.get(
                    "table_label", "tab:hh_first_plateau_prefix_costs"
                ),
                "table_surface": "table_iii_hubbard_holstein",
                "family": "hubbard_holstein",
                "case_id": None if case_id is None else str(case_id),
                "regime": regime,
                "method": method,
                "algorithm_id": raw_row.get("algorithm_id"),
                "source_json": None if source_json is None else str(source_json),
                "source_sha256": source_sha,
                "visible_one_minus_F": raw_row.get("one_minus_F_display", "--"),
                "prefix_operator_count": _int_or_none(
                    raw_row.get(
                        "accepted_operator_groups", raw_row.get("prefix_operator_count")
                    )
                ),
                "plateau_iteration": _int_or_none(raw_row.get("plateau_iteration")),
                "n_ph_work": n_ph,
                "source_map_json": str(audit_path),
            }
        )
    return specs, payload


def _supplemental_specs(
    path: str | Path | None, *, surface: str, default_family: str
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return [], None
    source_path = Path(path)
    payload = _read_json(source_path)
    rows = payload.get("rows", payload.get("source_rows", []))
    if not _sequence(rows):
        raise FidelityBlocked(f"{surface}_rows_invalid")
    specs: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        case_id = str(raw.get("case_id") or raw.get("benchmark_id") or "")
        specs.append(
            {
                "table_label": raw.get("table_label"),
                "table_surface": surface,
                "family": _family_key(raw.get("family", default_family)),
                "case_id": case_id,
                "regime": str(raw.get("regime") or _regime_from_case(case_id)),
                "method": raw.get("method", raw.get("method_label", "SNAKE")),
                "algorithm_id": raw.get("algorithm_id"),
                "source_json": raw.get("source_json", raw.get("source_result_json")),
                "source_sha256": raw.get("source_sha256", raw.get("source_result_sha256")),
                "visible_one_minus_F": raw.get("one_minus_F_display", "--"),
                "prefix_operator_count": _int_or_none(raw.get("prefix_operator_count")),
                "source_map_json": str(source_path),
            }
        )
    return specs, payload


def format_one_minus_fidelity(value: float | None) -> str:
    """Paper-I display convention without granting manuscript-transfer authority."""

    if value is None:
        return "--"
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return "--"
    if number < 1.0e-5:
        return "0"
    return f"{number:.3g}"


def _result_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result")
    if isinstance(result, Mapping):
        return result
    adapt = payload.get("adapt_vqe")
    return adapt if isinstance(adapt, Mapping) else payload


def _labels_and_theta(row: Mapping[str, Any]) -> tuple[list[str], list[float]]:
    labels_raw: Any = None
    for key in (
        "selected_operators",
        "operators",
        "selected_ops",
        "ordered_active_operator_labels",
        "active_operator_order",
    ):
        if _sequence(row.get(key)):
            labels_raw = row.get(key)
            break
    theta_raw: Any = None
    for key in (
        "theta",
        "optimal_point",
        "signed_unwrapped_runtime_parameters",
        "runtime_theta",
    ):
        if _sequence(row.get(key)):
            theta_raw = row.get(key)
            break
    labels = [] if labels_raw is None else [str(value) for value in labels_raw]
    try:
        theta = [] if theta_raw is None else [float(value) for value in theta_raw]
    except (TypeError, ValueError) as exc:
        raise FidelityBlocked("not_reconstructable_nonfinite_parameters") from exc
    if not all(math.isfinite(value) for value in theta):
        raise FidelityBlocked("not_reconstructable_nonfinite_parameters")
    return labels, theta


def _runtime_parameter_count(row: Mapping[str, Any], *, logical_count: int) -> int:
    parameterization = row.get("parameterization")
    if not isinstance(parameterization, Mapping):
        parameterization = row.get("parameterization_layout")
    if isinstance(parameterization, Mapping):
        count = _int_or_none(parameterization.get("runtime_parameter_count"))
        if count is None or count < 0:
            raise FidelityBlocked("not_reconstructable_parameterization_invalid")
        logical = _int_or_none(parameterization.get("logical_operator_count"))
        if logical is not None and logical != int(logical_count):
            raise FidelityBlocked("not_reconstructable_parameter_count_mismatch")
        return int(count)
    return int(logical_count)


def _checkpoint_candidates(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    parents: list[Mapping[str, Any]] = [row]
    for key in ("adapt_vqe", "continuation"):
        child = row.get(key)
        if isinstance(child, Mapping):
            parents.append(child)
    adapt = row.get("adapt_vqe")
    if isinstance(adapt, Mapping) and isinstance(adapt.get("continuation"), Mapping):
        parents.append(adapt["continuation"])
    for parent in parents:
        checkpoints = parent.get("active_prefix_checkpoints")
        if _sequence(checkpoints):
            out.extend(value for value in checkpoints if isinstance(value, Mapping))
        terminal = parent.get("terminal_active_prefix_checkpoint")
        if isinstance(terminal, Mapping):
            out.append(terminal)
        history = parent.get("history", parent.get("adapt_history"))
        if _sequence(history):
            for history_row in history:
                if not isinstance(history_row, Mapping):
                    continue
                checkpoint = history_row.get("active_prefix_checkpoint")
                if isinstance(checkpoint, Mapping):
                    out.append(checkpoint)
    return out


def _checkpoint_identity(checkpoint: Mapping[str, Any]) -> str:
    embedded = str(checkpoint.get("checkpoint_sha256") or "").strip()
    if embedded:
        return embedded
    return hashlib.sha256(
        json.dumps(
            dict(checkpoint), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


_CHECKPOINT_COPY_ENVELOPE_FIELDS = frozenset(
    {
        "checkpoint_sha256",
        "checkpoint_kind",
        "estimator_ledger_receipt",
    }
)


def _checkpoint_canonical_copy_identity(checkpoint: Mapping[str, Any]) -> str:
    """Hash replay semantics while ignoring only checkpoint-copy provenance.

    The terminal writer may make a second signed copy of the already-saved
    post-admission state after the final reporting boundary.  That copy has a
    different ``checkpoint_kind`` and estimator-ledger receipt, so its complete
    checkpoint SHA-256 is intentionally different even when the ordered
    operators, signed parameters, parameterization, route contract, and state
    fingerprint are byte-for-byte identical.  Fidelity replay may collapse
    those copies, but no scientific/replay field is permitted to differ.

    Any embedded full-payload checksum is verified before the copy-only fields
    are removed.  The returned canonical digest therefore does not let a
    malformed or genuinely conflicting checkpoint hide behind another copy.
    """

    payload = dict(checkpoint)
    embedded = str(payload.pop("checkpoint_sha256", "") or "").strip().lower()
    try:
        observed = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise FidelityBlocked(
            "not_reconstructable_checkpoint_canonicalization_failed"
        ) from exc
    if embedded and (not _is_sha256(embedded) or embedded != observed):
        raise FidelityBlocked("not_reconstructable_checkpoint_sha256_mismatch")

    canonical = dict(checkpoint)
    for field in _CHECKPOINT_COPY_ENVELOPE_FIELDS:
        canonical.pop(field, None)
    try:
        return hashlib.sha256(
            json.dumps(
                canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise FidelityBlocked(
            "not_reconstructable_checkpoint_canonicalization_failed"
        ) from exc


def _checkpoint_iteration(checkpoint: Mapping[str, Any]) -> int | None:
    outer = _int_or_none(checkpoint.get("outer_iteration"))
    if outer is not None:
        return outer
    raw = _int_or_none(checkpoint.get("iteration"))
    if raw is None:
        return None
    if str(checkpoint.get("schema")) == "paper_i_comparator_active_prefix_checkpoint_v1":
        return int(raw) + 1
    return raw


def _checkpoint_active_depth(checkpoint: Mapping[str, Any]) -> int | None:
    depth = _int_or_none(checkpoint.get("active_ansatz_depth"))
    if depth is not None:
        return depth
    return _int_or_none(checkpoint.get("active_logical_coordinate_count"))


def _terminal_checkpoint(
    result: Mapping[str, Any], *, labels: Sequence[str], theta: Sequence[float]
) -> Mapping[str, Any] | None:
    """Return one checkpoint exactly matching the saved terminal identity."""

    matches: list[Mapping[str, Any]] = []
    theta_array = np.asarray(theta, dtype=float).reshape(-1)
    for checkpoint in _checkpoint_candidates(result):
        checkpoint_labels, checkpoint_theta = _labels_and_theta(checkpoint)
        if checkpoint_labels != [str(value) for value in labels]:
            continue
        checkpoint_theta_options: list[np.ndarray] = [
            np.asarray(checkpoint_theta, dtype=float).reshape(-1)
        ]
        for key in (
            "logical_theta",
            "signed_unwrapped_logical_parameters",
            "runtime_theta",
            "signed_unwrapped_runtime_parameters",
        ):
            raw = checkpoint.get(key)
            if _sequence(raw):
                checkpoint_theta_options.append(
                    np.asarray(raw, dtype=float).reshape(-1)
                )
        if not any(
            option.shape == theta_array.shape
            and np.allclose(option, theta_array, rtol=0.0, atol=1.0e-13)
            for option in checkpoint_theta_options
        ):
            continue
        matches.append(checkpoint)
    if not matches:
        return None
    identities = {
        _checkpoint_canonical_copy_identity(checkpoint) for checkpoint in matches
    }
    if len(identities) != 1:
        raise FidelityBlocked("not_reconstructable_ambiguous_terminal_checkpoint")
    terminal_copies = [
        checkpoint
        for checkpoint in matches
        if str(checkpoint.get("checkpoint_kind") or "").startswith("terminal_")
    ]
    return terminal_copies[-1] if terminal_copies else matches[-1]


def _prefix_static_row(
    result: Mapping[str, Any], *, prefix_count: int | None
) -> dict[str, Any]:
    """Resolve a terminal row or an exact saved prefix; never truncate/refit."""

    if not isinstance(result, Mapping):
        raise FidelityBlocked("not_reconstructable_result_not_mapping")
    labels, theta = _labels_and_theta(result)
    if labels or theta:
        expected_runtime = _runtime_parameter_count(result, logical_count=len(labels))
        if len(theta) != expected_runtime:
            raise FidelityBlocked("not_reconstructable_parameter_count_mismatch")
    if prefix_count is None:
        if not labels or not theta:
            raise FidelityBlocked("not_reconstructable_terminal_parameters_missing")
        out = dict(result)
        out["state_replay_source"] = "terminal_saved_parameters"
        terminal_checkpoint = _terminal_checkpoint(
            result, labels=labels, theta=theta
        )
        if terminal_checkpoint is not None:
            out["_active_prefix_checkpoint"] = dict(terminal_checkpoint)
            out["state_replay_source"] = "exact_terminal_active_prefix_checkpoint"
        return out
    requested = int(prefix_count)
    if requested < 0:
        raise FidelityBlocked("not_reconstructable_invalid_prefix_count")
    if labels and requested == len(labels):
        out = dict(result)
        out["state_replay_source"] = "terminal_saved_parameters_exact_prefix"
        terminal_checkpoint = _terminal_checkpoint(
            result, labels=labels, theta=theta
        )
        if terminal_checkpoint is not None:
            out["_active_prefix_checkpoint"] = dict(terminal_checkpoint)
            out["state_replay_source"] = "exact_terminal_active_prefix_checkpoint"
        return out
    if labels and requested > len(labels):
        raise FidelityBlocked("not_reconstructable_prefix_exceeds_terminal_depth")

    checkpoints = _checkpoint_candidates(result)
    by_depth = [
        checkpoint
        for checkpoint in checkpoints
        if _checkpoint_active_depth(checkpoint) == requested
    ]
    by_round = [
        checkpoint
        for checkpoint in checkpoints
        if _checkpoint_iteration(checkpoint) == requested
    ]
    matches = by_depth or by_round
    if not matches:
        raise FidelityBlocked("not_reconstructable_missing_exact_prefix_checkpoint")
    identities = {_checkpoint_identity(checkpoint) for checkpoint in matches}
    if len(identities) != 1:
        raise FidelityBlocked("not_reconstructable_ambiguous_prefix_checkpoint")
    checkpoint = dict(matches[0])
    checkpoint_labels, checkpoint_theta = _labels_and_theta(checkpoint)
    if not checkpoint_labels:
        raise FidelityBlocked("not_reconstructable_checkpoint_operators_missing")
    expected_runtime = _runtime_parameter_count(
        checkpoint, logical_count=len(checkpoint_labels)
    )
    if len(checkpoint_theta) != expected_runtime:
        raise FidelityBlocked("not_reconstructable_parameter_count_mismatch")
    checkpoint["selected_operators"] = list(checkpoint_labels)
    checkpoint["theta"] = list(checkpoint_theta)
    checkpoint["state_replay_source"] = "exact_signed_active_prefix_checkpoint"
    checkpoint["_active_prefix_checkpoint"] = dict(matches[0])
    return checkpoint


def _explicit_fidelity_inputs(
    payload: Mapping[str, Any], static_row: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    for parent in (static_row, _result_mapping(payload), payload):
        for key in (
            "ground_space_fidelity_inputs",
            "fidelity_audit_inputs",
            "reporting_fidelity_inputs",
        ):
            value = parent.get(key) if isinstance(parent, Mapping) else None
            if isinstance(value, Mapping):
                return value
    return None


def _complex_array(value: Any, *, field: str) -> np.ndarray:
    if isinstance(value, Mapping):
        real = value.get("real", value.get("re"))
        imag = value.get("imag", value.get("im"))
        if real is None:
            raise FidelityBlocked(f"fidelity_inputs_missing_{field}_real")
        real_array = np.asarray(real, dtype=float)
        imag_array = np.zeros_like(real_array) if imag is None else np.asarray(imag, dtype=float)
        if real_array.shape != imag_array.shape:
            raise FidelityBlocked(f"fidelity_inputs_{field}_shape_mismatch")
        out = np.asarray(real_array + 1j * imag_array, dtype=complex)
    else:
        try:
            out = np.asarray(value, dtype=complex)
        except (TypeError, ValueError) as exc:
            raise FidelityBlocked(f"fidelity_inputs_invalid_{field}") from exc
    if out.size == 0 or not np.all(np.isfinite(out.real)) or not np.all(
        np.isfinite(out.imag)
    ):
        raise FidelityBlocked(f"fidelity_inputs_invalid_{field}")
    return out


def _compute_explicit_fidelity(inputs: Mapping[str, Any]) -> dict[str, Any]:
    hamiltonian_raw = inputs.get("hamiltonian", inputs.get("hamiltonian_matrix"))
    state_raw = inputs.get("variational_state", inputs.get("statevector"))
    if hamiltonian_raw is None:
        raise FidelityBlocked("fidelity_inputs_missing_hamiltonian")
    if state_raw is None:
        raise FidelityBlocked("fidelity_inputs_missing_variational_state")
    for required_key in (
        "working_cutoff",
        "reference_cutoff",
        "fixed_sector_basis_indices",
        "legal_binary_basis_indices",
        "fixed_sector_label",
        "legal_binary_basis_label",
    ):
        if required_key not in inputs:
            raise FidelityBlocked(f"fidelity_inputs_missing_{required_key}")
    tolerance_raw = inputs.get("degeneracy_tolerance", {})
    tolerance_raw = tolerance_raw if isinstance(tolerance_raw, Mapping) else {}
    return evaluate_ground_space_fidelity(
        hamiltonian=_complex_array(hamiltonian_raw, field="hamiltonian"),
        variational_state=_complex_array(state_raw, field="variational_state").reshape(-1),
        working_cutoff=int(inputs["working_cutoff"]),
        reference_cutoff=int(inputs["reference_cutoff"]),
        fixed_sector_basis_indices=inputs["fixed_sector_basis_indices"],
        legal_binary_basis_indices=inputs["legal_binary_basis_indices"],
        fixed_sector_label=str(inputs["fixed_sector_label"]),
        legal_binary_basis_label=str(inputs["legal_binary_basis_label"]),
        tolerance=GroundSpaceTolerance(
            absolute=float(tolerance_raw.get("absolute", 1.0e-10)),
            relative=float(tolerance_raw.get("relative", 1.0e-10)),
        ),
        state_leakage_tolerance=float(inputs.get("state_leakage_tolerance", 1.0e-10)),
        hermiticity_tolerance=float(inputs.get("hermiticity_tolerance", 1.0e-10)),
        subspace_invariance_tolerance=float(
            inputs.get("subspace_invariance_tolerance", 1.0e-10)
        ),
    )


def _dense_hamiltonian(polynomial: Any, *, dimension: int) -> np.ndarray:
    from pipelines.static_adapt.statevector_runtime import (
        _apply_compiled_polynomial,
        _compile_polynomial_action,
    )

    compiled = _compile_polynomial_action(polynomial)
    matrix = np.empty((int(dimension), int(dimension)), dtype=complex)
    for column in range(int(dimension)):
        basis = np.zeros(int(dimension), dtype=complex)
        basis[column] = 1.0
        matrix[:, column] = _apply_compiled_polynomial(basis, compiled)
    return matrix


def _fixed_and_legal_bases(runtime_input: Any) -> tuple[tuple[int, ...], tuple[int, ...], str, str]:
    context = runtime_input.resolved_problem
    family = str(context.family_key).strip().lower()
    request = context.request
    nq = int(context.layout.total_qubits)
    dimension = 1 << nq
    legal_label = "full_computational_register"
    legal: tuple[int, ...] = tuple(range(dimension))
    if family in {"hh", "spin_boson", "bose_hubbard", "harmonic_kerr_chain"}:
        from pipelines.static_adapt.builders.legal_subspace_filter import (
            legal_subspace_basis_for_problem,
        )

        legal_info = legal_subspace_basis_for_problem(
            problem_key=family,
            num_sites=int(request.num_sites),
            n_ph_max=int(request.n_ph_max),
            boson_encoding=str(request.boson_encoding),
            total_register_width=nq,
        )
        legal = tuple(int(value) for value in legal_info["legal_indices"])
        legal_label = str(legal_info["legal_subspace_scope"])

    if family in {"hubbard", "extended_hubbard", "ttprime_hubbard", "hh"}:
        particles = context.sector.num_particles
        if particles is None or len(particles) != 2:
            raise FidelityBlocked("fixed_sector_particles_missing")
        from pipelines.static_adapt.builders.problem_setup import (
            _spinful_sector_basis_indices,
        )

        fixed = tuple(
            _spinful_sector_basis_indices(
                n_qubits=nq,
                num_sites=int(request.num_sites),
                indexing=str(request.ordering),
                n_alpha=int(particles[0]),
                n_beta=int(particles[1]),
            )
        )
    elif family == "spinless_tv":
        target = int(context.default_num_particles[0])
        fixed = tuple(index for index in range(dimension) if int(index).bit_count() == target)
    elif family == "spin_boson":
        # The two least-significant emitter qubits are a one-hot g/e register.
        fixed = tuple(
            index for index in range(dimension) if (int(index) & 0b11).bit_count() == 1
        )
    elif family in {"bose_hubbard", "harmonic_kerr_chain"}:
        fixed = legal
    else:
        raise FidelityBlocked(f"fixed_sector_basis_unsupported_family:{family}")
    return fixed, legal, str(context.sector.label), legal_label


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _assert_checkpoint_hash(checkpoint: Mapping[str, Any]) -> str:
    expected = str(checkpoint.get("checkpoint_sha256") or "").strip().lower()
    if not _is_sha256(expected):
        raise FidelityBlocked("not_reconstructable_checkpoint_sha256_missing")
    payload = dict(checkpoint)
    payload.pop("checkpoint_sha256", None)
    if _canonical_sha256(payload) != expected:
        raise FidelityBlocked("not_reconstructable_checkpoint_sha256_mismatch")
    return expected


def _normalized_pauli_terms(raw_terms: Any, *, label: str) -> tuple[dict[str, Any], ...]:
    if not _sequence(raw_terms):
        raise FidelityBlocked(f"not_reconstructable_{label}_pauli_terms_missing")
    out: list[dict[str, Any]] = []
    for raw in raw_terms:
        if not isinstance(raw, Mapping):
            raise FidelityBlocked(f"not_reconstructable_{label}_pauli_term_invalid")
        pauli = str(raw.get("pauli_exyz") or "").strip().lower()
        if not pauli or set(pauli) - set("exyz"):
            raise FidelityBlocked(f"not_reconstructable_{label}_pauli_word_invalid")
        coeff_re = _finite_float(raw.get("coeff_re", 0.0))
        coeff_im = _finite_float(raw.get("coeff_im", 0.0))
        if coeff_re is None or coeff_im is None:
            raise FidelityBlocked(f"not_reconstructable_{label}_coefficient_nonfinite")
        nq = int(raw.get("nq", len(pauli)))
        if nq != len(pauli):
            raise FidelityBlocked(f"not_reconstructable_{label}_pauli_width_mismatch")
        out.append(
            {
                "pauli_exyz": pauli,
                "coeff_re": float(coeff_re),
                "coeff_im": float(coeff_im),
                "nq": nq,
            }
        )
    if not out:
        raise FidelityBlocked(f"not_reconstructable_{label}_pauli_terms_empty")
    return tuple(out)


def _replay_generic_checkpoint(
    runtime_input: Any, checkpoint: Mapping[str, Any]
) -> np.ndarray:
    """Validate and replay the generic comparator checkpoint schema exactly."""

    if str(checkpoint.get("schema")) != "paper_i_comparator_active_prefix_checkpoint_v1":
        raise FidelityBlocked("not_reconstructable_generic_checkpoint_schema")
    _assert_checkpoint_hash(checkpoint)
    iteration = _int_or_none(checkpoint.get("iteration"))
    if iteration is None or iteration < 0:
        raise FidelityBlocked("not_reconstructable_generic_checkpoint_iteration")
    outer_iteration = _int_or_none(checkpoint.get("outer_iteration"))
    if outer_iteration is not None and outer_iteration != int(iteration) + 1:
        raise FidelityBlocked("not_reconstructable_generic_checkpoint_outer_iteration")
    labels_raw = checkpoint.get("active_operator_order")
    operators_raw = checkpoint.get("active_operators")
    if not _sequence(labels_raw) or not _sequence(operators_raw):
        raise FidelityBlocked("not_reconstructable_generic_checkpoint_operator_order")
    labels = tuple(str(value) for value in labels_raw)
    operators = tuple(value for value in operators_raw if isinstance(value, Mapping))
    if len(operators) != len(tuple(operators_raw)) or len(labels) != len(operators):
        raise FidelityBlocked("not_reconstructable_generic_checkpoint_operator_count")
    active_count = _int_or_none(checkpoint.get("active_logical_coordinate_count"))
    if active_count != len(labels):
        raise FidelityBlocked("not_reconstructable_generic_checkpoint_active_count")
    active_depth = _int_or_none(checkpoint.get("active_ansatz_depth"))
    if active_depth is not None and active_depth != len(labels):
        raise FidelityBlocked("not_reconstructable_generic_checkpoint_active_depth")
    if tuple(str(operator.get("label")) for operator in operators) != labels:
        raise FidelityBlocked("not_reconstructable_generic_checkpoint_label_order")

    parameterization = checkpoint.get("parameterization_layout")
    if not isinstance(parameterization, Mapping):
        raise FidelityBlocked("not_reconstructable_generic_parameterization_missing")
    from src.quantum.ansatz_parameterization import deserialize_layout

    try:
        layout = deserialize_layout(parameterization)
    except Exception as exc:
        raise FidelityBlocked("not_reconstructable_generic_parameterization_invalid") from exc
    if int(layout.logical_parameter_count) != len(labels):
        raise FidelityBlocked("not_reconstructable_generic_parameterization_logical_count")
    if tuple(str(block.candidate_label) for block in layout.blocks) != labels:
        raise FidelityBlocked("not_reconstructable_generic_parameterization_label_order")
    if _int_or_none(checkpoint.get("logical_parameter_count")) != int(
        layout.logical_parameter_count
    ) or _int_or_none(checkpoint.get("runtime_parameter_count")) != int(
        layout.runtime_parameter_count
    ):
        raise FidelityBlocked("not_reconstructable_generic_parameterization_count")

    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.qubitization_module import PauliTerm
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm

    selected_terms: list[AnsatzTerm] = []
    widths: set[int] = set()
    for index, (operator, block) in enumerate(zip(operators, layout.blocks, strict=True)):
        observed = _normalized_pauli_terms(
            operator.get("pauli_terms"), label=f"generic_operator_{index}"
        )
        expected = tuple(
            {
                "pauli_exyz": str(term.pauli_exyz).lower(),
                "coeff_re": float(term.coeff_real),
                "coeff_im": 0.0,
                "nq": int(term.nq),
            }
            for term in block.terms
        )
        observed_sorted = sorted(
            (row["pauli_exyz"], row["coeff_re"], row["coeff_im"], row["nq"])
            for row in observed
        )
        expected_sorted = sorted(
            (row["pauli_exyz"], row["coeff_re"], row["coeff_im"], row["nq"])
            for row in expected
        )
        if observed_sorted != expected_sorted:
            raise FidelityBlocked("not_reconstructable_generic_operator_layout_mismatch")
        widths.update(int(row["nq"]) for row in observed)
        selected_terms.append(
            AnsatzTerm(
                label=str(operator.get("label")),
                polynomial=PauliPolynomial(
                    "JW",
                    [
                        PauliTerm(
                            int(row["nq"]),
                            ps=str(row["pauli_exyz"]),
                            pc=complex(row["coeff_re"], row["coeff_im"]),
                        )
                        for row in observed
                    ],
                ),
                execution_mode=str(operator.get("execution_mode") or "termwise_product"),
            )
        )
    if len(widths) != 1 or (1 << next(iter(widths))) != int(
        np.asarray(runtime_input.psi_ref).size
    ):
        raise FidelityBlocked("not_reconstructable_generic_checkpoint_width")

    from pipelines.static_adapt.estimator_call_ledger import projective_state_fingerprint

    input_fingerprint = str(
        checkpoint.get("ansatz_input_state_projective_fingerprint") or ""
    )
    if not input_fingerprint or projective_state_fingerprint(
        np.asarray(runtime_input.psi_ref, dtype=complex).reshape(-1)
    ) != input_fingerprint:
        raise FidelityBlocked("not_reconstructable_generic_input_fingerprint_mismatch")
    mode = str(checkpoint.get("parameterization_mode") or "").strip().lower()
    if mode.startswith("per_pauli"):
        theta = np.asarray(checkpoint.get("runtime_theta", []), dtype=float).reshape(-1)
        expected_count = int(layout.runtime_parameter_count)
        executor_mode = "per_pauli_term"
    elif mode == "logical_shared":
        theta = np.asarray(checkpoint.get("logical_theta", []), dtype=float).reshape(-1)
        expected_count = int(layout.logical_parameter_count)
        executor_mode = "logical_shared"
    else:
        raise FidelityBlocked("not_reconstructable_generic_parameterization_mode")
    if theta.size != expected_count or not np.all(np.isfinite(theta)):
        raise FidelityBlocked("not_reconstructable_generic_theta_count")
    from src.quantum.compiled_ansatz import CompiledAnsatzExecutor

    executor = CompiledAnsatzExecutor(
        selected_terms,
        parameterization_mode=executor_mode,
        parameterization_layout=layout,
    )
    state = np.asarray(
        executor.prepare_state(theta, np.asarray(runtime_input.psi_ref, dtype=complex)),
        dtype=complex,
    ).reshape(-1)
    expected_fingerprint = str(
        checkpoint.get("prepared_state_projective_fingerprint") or ""
    )
    if not expected_fingerprint or projective_state_fingerprint(state) != expected_fingerprint:
        raise FidelityBlocked("not_reconstructable_generic_prepared_fingerprint_mismatch")
    return state


def _replay_terminal_saved_ansatz(
    runtime_input: Any, static_row: Mapping[str, Any]
) -> np.ndarray:
    """Replay saved terminal terms/theta and require a projective receipt."""

    selected_terms = tuple(runtime_input.selected_terms)
    labels, _ = _labels_and_theta(static_row)
    if tuple(str(term.label) for term in selected_terms) != tuple(labels):
        raise FidelityBlocked("not_reconstructable_terminal_operator_identity_mismatch")
    layout = runtime_input.base_layout
    mode = str(static_row.get("parameterization_mode") or "").strip().lower()
    if not mode and isinstance(static_row.get("parameterization"), Mapping):
        mode = str(static_row["parameterization"].get("mode") or "").strip().lower()
    if mode.startswith("per_pauli"):
        theta = np.asarray(runtime_input.theta_runtime, dtype=float).reshape(-1)
        executor_mode = "per_pauli_term"
    elif mode == "logical_shared":
        logical = runtime_input.theta_logical
        if logical is None:
            raise FidelityBlocked("not_reconstructable_terminal_logical_theta_missing")
        theta = np.asarray(logical, dtype=float).reshape(-1)
        executor_mode = "logical_shared"
    else:
        raise FidelityBlocked("not_reconstructable_terminal_parameterization_mode")
    from src.quantum.compiled_ansatz import CompiledAnsatzExecutor

    executor = CompiledAnsatzExecutor(
        selected_terms,
        parameterization_mode=executor_mode,
        parameterization_layout=layout,
    )
    state = np.asarray(
        executor.prepare_state(theta, np.asarray(runtime_input.psi_ref, dtype=complex)),
        dtype=complex,
    ).reshape(-1)
    expected = str(
        static_row.get("prepared_state_projective_fingerprint")
        or static_row.get("projective_state_fingerprint")
        or ""
    )
    if not expected:
        raise FidelityBlocked("not_reconstructable_terminal_projective_fingerprint_missing")
    from pipelines.static_adapt.estimator_call_ledger import projective_state_fingerprint

    if projective_state_fingerprint(state) != expected:
        raise FidelityBlocked("not_reconstructable_terminal_projective_fingerprint_mismatch")
    return state


def _runtime_replayed_state(
    payload: Mapping[str, Any], *, source_path: Path, static_row: Mapping[str, Any]
) -> tuple[Any, np.ndarray, str]:
    from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload

    runtime_input = load_scaffold_runtime_input_from_payload(
        payload, artifact_json=source_path
    )
    checkpoint = static_row.get("_active_prefix_checkpoint")
    if not isinstance(checkpoint, Mapping):
        state = _replay_terminal_saved_ansatz(runtime_input, static_row)
        return runtime_input, state, "terminal_saved_ansatz_fingerprint_replay"

    if str(checkpoint.get("schema")) == "paper_i_comparator_active_prefix_checkpoint_v1":
        state = _replay_generic_checkpoint(runtime_input, checkpoint)
        return runtime_input, state, "exact_generic_comparator_prefix_checkpoint_replay"

    from pipelines.exact_bench.paper_i_hh_recovery_prefix_qiskit_sidecar import (
        derive_execution_order_repaired_checkpoint,
        reconstruct_reference_state,
        validate_active_prefix_checkpoint,
    )
    from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.qubitization_module import PauliTerm
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm

    outer_iteration = int(checkpoint.get("outer_iteration"))
    checkpoint_order_repaired = False
    try:
        validated = validate_active_prefix_checkpoint(
            checkpoint, expected_outer_iteration=outer_iteration
        )
    except ValueError:
        repaired_checkpoint, repair = derive_execution_order_repaired_checkpoint(
            checkpoint,
            expected_outer_iteration=outer_iteration,
        )
        if repair.get("status") != "repaired_permutation_only":
            raise
        validated = validate_active_prefix_checkpoint(
            repaired_checkpoint,
            expected_outer_iteration=outer_iteration,
        )
        checkpoint_order_repaired = True
    selected_terms: list[AnsatzTerm] = []
    for block in validated.layout.blocks:
        terms = [
            PauliTerm(
                int(term.nq),
                ps=str(term.pauli_exyz),
                pc=complex(float(term.coeff_real), 0.0),
            )
            for term in block.terms
        ]
        selected_terms.append(
            AnsatzTerm(
                label=str(block.candidate_label),
                polynomial=PauliPolynomial("JW", terms),
                execution_mode=str(block.execution_mode),
            )
        )
    state_ref, _ = reconstruct_reference_state(
        payload, num_qubits=int(validated.num_qubits)
    )
    mode_raw = str(checkpoint.get("parameterization_mode") or "").strip().lower()
    if mode_raw == "logical_shared":
        executor_mode = "logical_shared"
        theta = validated.logical_parameters
    elif mode_raw in {"per_pauli_term", "per_pauli_term_v1"}:
        executor_mode = "per_pauli_term"
        theta = validated.runtime_parameters
    else:
        raise FidelityBlocked("not_reconstructable_parameterization_mode_unsupported")
    executor = CompiledAnsatzExecutor(
        selected_terms,
        parameterization_mode=executor_mode,
        parameterization_layout=validated.layout,
    )
    state = np.asarray(executor.prepare_state(theta, state_ref), dtype=complex).reshape(-1)
    expected_fingerprint = str(checkpoint.get("projective_state_fingerprint") or "")
    if not expected_fingerprint:
        raise FidelityBlocked("not_reconstructable_projective_fingerprint_missing")
    from pipelines.static_adapt.estimator_call_ledger import projective_state_fingerprint

    if projective_state_fingerprint(state) != expected_fingerprint:
        raise FidelityBlocked("not_reconstructable_projective_fingerprint_mismatch")
    replay_source = (
        "exact_signed_prefix_checkpoint_permutation_repaired_replay"
        if checkpoint_order_repaired
        else "exact_signed_prefix_checkpoint_replay"
    )
    return runtime_input, state, replay_source


def _compute_runtime_fidelity(
    payload: Mapping[str, Any], *, source_path: Path, static_row: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    runtime_input, state, replay_source = _runtime_replayed_state(
        payload, source_path=source_path, static_row=static_row
    )
    dimension = int(state.size)
    if dimension <= 0 or dimension & (dimension - 1):
        raise FidelityBlocked("replayed_state_dimension_not_power_of_two")
    hamiltonian = _dense_hamiltonian(runtime_input.h_poly, dimension=dimension)
    fixed, legal, fixed_label, legal_label = _fixed_and_legal_bases(runtime_input)
    working_cutoff = int(runtime_input.resolved_problem.request.n_ph_max)
    result = evaluate_ground_space_fidelity(
        hamiltonian=hamiltonian,
        variational_state=state,
        working_cutoff=working_cutoff,
        reference_cutoff=working_cutoff,
        fixed_sector_basis_indices=fixed,
        legal_binary_basis_indices=legal,
        fixed_sector_label=fixed_label,
        legal_binary_basis_label=legal_label,
    )
    return result, replay_source


def compute_row_fidelity(row: Mapping[str, Any]) -> dict[str, Any]:
    source_raw = row.get("source_json")
    if source_raw is None or (isinstance(source_raw, str) and source_raw == ""):
        raise FidelityBlocked("source_json_missing")
    source_path = _resolve(str(source_raw))
    if not source_path.is_file():
        raise FidelityBlocked("source_json_missing")
    expected_sha = row.get("source_sha256")
    if _is_sha256(expected_sha) and _sha256(source_path) != str(expected_sha).lower():
        raise FidelityBlocked("source_json_sha256_mismatch")
    payload = _read_json(source_path)
    result = _result_mapping(payload)
    static_row = _prefix_static_row(
        result,
        prefix_count=_int_or_none(row.get("prefix_operator_count")),
    )
    explicit = _explicit_fidelity_inputs(payload, static_row)
    if explicit is not None:
        ground = _compute_explicit_fidelity(explicit)
        replay_source = "explicit_locked_reporting_inputs"
    else:
        ground, replay_source = _compute_runtime_fidelity(
            payload, source_path=source_path, static_row=static_row
        )
    return {
        "one_minus_fidelity": float(ground["infidelity"]),
        "fidelity": float(ground["fidelity"]),
        "infidelity_source_key": "ground_space_projector_infidelity_same_cutoff",
        "reference_kind": "same_cutoff_physical_sector_ground_space_projector",
        "metric_statuses": {
            "ground_space_projector_infidelity_same_cutoff": "ok"
        },
        "state_replay_source": replay_source,
        "ground_space_fidelity": ground,
    }


def _audit_one(spec: Mapping[str, Any]) -> dict[str, Any]:
    row = {"schema": ROW_SCHEMA, **dict(spec)}
    source_raw = row.get("source_json")
    if (
        source_raw is None
        or (isinstance(source_raw, str) and source_raw == "")
        or not _resolve(str(source_raw)).is_file()
    ):
        row.update(
            {
                "status": "blocked",
                "blocker": "source_json_missing",
                "one_minus_fidelity": None,
                "fidelity": None,
                "one_minus_F_display": "--",
                "safe_for_manuscript_transfer": False,
            }
        )
        return row
    try:
        computed = compute_row_fidelity(row)
    except FidelityBlocked as exc:
        row.update(
            {
                "status": "blocked",
                "blocker": str(exc.status),
                "blocker_detail": str(exc),
                "one_minus_fidelity": None,
                "fidelity": None,
                "one_minus_F_display": "--",
                "safe_for_manuscript_transfer": False,
            }
        )
        return row
    except GroundSpaceFidelityError as exc:
        row.update(
            {
                "status": "blocked",
                "blocker": f"ground_space_fidelity:{exc.code}",
                "blocker_detail": str(exc),
                "one_minus_fidelity": None,
                "fidelity": None,
                "one_minus_F_display": "--",
                "safe_for_manuscript_transfer": False,
            }
        )
        return row
    except Exception as exc:  # fail closed at the reporting boundary
        row.update(
            {
                "status": "blocked",
                "blocker": "fidelity_computation_failed",
                "blocker_exception_type": type(exc).__name__,
                "blocker_detail": str(exc),
                "one_minus_fidelity": None,
                "fidelity": None,
                "one_minus_F_display": "--",
                "safe_for_manuscript_transfer": False,
            }
        )
        return row
    row.update(dict(computed))
    row["status"] = "computed"
    row["blocker"] = None
    row["one_minus_F_display"] = format_one_minus_fidelity(
        _finite_float(row.get("one_minus_fidelity"))
    )
    # This audit intentionally does not grant promotion/manuscript authority.
    row["safe_for_manuscript_transfer"] = False
    return row


def build_audit(
    *,
    tables_i_ii_promotion: str | Path,
    hh_table_iii_prefix_audit: str | Path,
    hh_table_iii_source_map: str | Path | None,
    hubbard_snake_audit: str | Path | None,
    spin_boson_snake_audit: str | Path | None,
) -> dict[str, Any]:
    table_specs, tables_payload = _tables_i_ii_specs(tables_i_ii_promotion)
    hh_specs, hh_payload = _hh_table_iii_specs(
        hh_table_iii_prefix_audit, hh_table_iii_source_map
    )
    hubbard_specs, hubbard_payload = _supplemental_specs(
        hubbard_snake_audit, surface="table_i_hubbard_snake", default_family="hubbard"
    )
    spin_specs, spin_payload = _supplemental_specs(
        spin_boson_snake_audit,
        surface="table_ii_spin_boson_snake",
        default_family="spin_boson",
    )
    specs = [*table_specs, *hh_specs, *hubbard_specs, *spin_specs]
    rows = [_audit_one(spec) for spec in specs]
    counts = Counter(str(row.get("status") or "unknown") for row in rows)
    inputs: dict[str, Any] = {
        "tables_i_ii_promotion": {
            "path": str(tables_i_ii_promotion),
            "schema": tables_payload.get("schema"),
        },
        "hh_table_iii_prefix_audit": {
            "path": str(hh_table_iii_prefix_audit),
            "schema": hh_payload.get("schema"),
        },
    }
    if hh_table_iii_source_map is not None:
        inputs["hh_table_iii_source_map"] = {"path": str(hh_table_iii_source_map)}
    if hubbard_payload is not None:
        inputs["hubbard_snake_audit"] = {
            "path": str(hubbard_snake_audit),
            "schema": hubbard_payload.get("schema"),
        }
    if spin_payload is not None:
        inputs["spin_boson_snake_audit"] = {
            "path": str(spin_boson_snake_audit),
            "schema": spin_payload.get("schema"),
        }
    return {
        "schema": AUDIT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "usage_scope": "post_run_reporting_only",
        "controller_decision_eligible": False,
        "optimizer_input_eligible": False,
        "stopping_input_eligible": False,
        "s_alg_charged": False,
        "manuscript_edited": False,
        "manuscript_transfer_authorized": False,
        "reference_policy": "same_cutoff_physical_sector_ground_space_projector",
        "prefix_policy": "saved_terminal_or_exact_signed_checkpoint_only_no_refit",
        "inputs": inputs,
        "status_counts": dict(sorted(counts.items())),
        "rows": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-i-ii-promotion", required=True)
    parser.add_argument("--hh-table-iii-prefix-audit", required=True)
    parser.add_argument("--hh-table-iii-source-map")
    parser.add_argument("--hubbard-snake-audit")
    parser.add_argument("--spin-boson-snake-audit")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_audit(
        tables_i_ii_promotion=args.tables_i_ii_promotion,
        hh_table_iii_prefix_audit=args.hh_table_iii_prefix_audit,
        hh_table_iii_source_map=args.hh_table_iii_source_map,
        hubbard_snake_audit=args.hubbard_snake_audit,
        spin_boson_snake_audit=args.spin_boson_snake_audit,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

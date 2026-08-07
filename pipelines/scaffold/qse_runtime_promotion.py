"""Offline runtime-contract promotion for QSE root-refit ansatz artifacts.

P5b is intentionally an offline promotion step.  It sanitizes the ansatz payload
from a ``qse_root_refit_v1`` artifact into a QPU-preparable circuit/state payload
and only marks it controller-usable when the existing scaffold runtime loader
validates the exact emitted runtime payload.  QSE Ritz diagnostics, target states,
fit diagnostics, spectra, and exact/benchmark data remain diagnostic-only and are
never copied into controller-visible payloads.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.io import (
    polynomial_from_serialized_terms,
    statevector_from_manifest,
    write_manifest_json,
)
from pipelines.scaffold.handoff_state_bundle import build_statevector_manifest
from pipelines.scaffold.qse_root_refit import (
    QSE_ROOT_REFIT_SCHEMA_VERSION,
    QSERootRefitError,
    reconstruct_ansatz_state_from_payload,
)
from src.quantum.ansatz_parameterization import (
    deserialize_layout,
    project_runtime_theta_block_mean,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


QSE_RUNTIME_PROMOTED_SCHEMA_VERSION = "qse_runtime_promoted_ansatz_v1"
QSE_RUNTIME_PROMOTION_PIPELINE = "qse_runtime_promotion"
_RUNTIME_PAYLOAD_PIPELINE = "promoted_ansatz_runtime_payload_v1"
_RUNTIME_VALIDATED_STATUSES = {"validated"}

_FORBIDDEN_SOURCE_CONTROLLER_REFS = (
    "qse_root_refit.qse_ritz_diagnostics",
    "qse_root_refit.qse_ritz_diagnostics.energy",
    "qse_root_refit.qse_ritz_diagnostics.energy_relative_to_lowest_qse",
    "qse_root_refit.qse_ritz_diagnostics.generalized_residual_norm",
    "qse_root_refit.qse_ritz_diagnostics.basis_coefficients",
    "qse_root_refit.target_state_diagnostics",
    "qse_root_refit.target_state_diagnostics.amplitudes_qn_to_q0",
    "qse_root_refit.fit_summary",
    "qse_root_refit.spectral_functions",
    "qse_root_refit.spectral_window_metrics",
    "qse_root_refit.cutoff_boundary_diagnostics",
    "source.source_state_index",
    "sanitization.source_to_sanitized_operator_labels.source_label",
)

_SAFE_RUNTIME_SETTING_KEYS = {
    "problem",
    "L",
    "num_sites",
    "t",
    "u",
    "U",
    "dv",
    "omega0",
    "g_ep",
    "n_ph_max",
    "boson_encoding",
    "ordering",
    "boundary",
    "include_zero_point",
    "molecular_problem_json",
    "v_nn",
    "t_prime",
    "n_fermions",
    "sector_n_up",
    "sector_n_dn",
    "adapt_pool",
}

_FORBIDDEN_RUNTIME_PAYLOAD_MARKERS = (
    "basis_coefficients",
    "qse_ritz_diagnostics",
    "target_state_diagnostics",
    "fit_summary",
    "poison_if_copied",
    "exact",
    "ground_state",
    "exact_gs_energy",
)


class QSERuntimePromotionError(ValueError):
    """Raised when QSE runtime promotion cannot safely proceed."""


@dataclass(frozen=True)
class QSERuntimePromotionConfig:
    qse_root_refit_json: Path
    output_json: Path
    runtime_template_json: Path | None = None
    require_runtime_contract: bool = False
    max_reconstruction_error: float = 1.0e-10
    amplitude_cutoff: float = 1.0e-12


@dataclass(frozen=True)
class RuntimeContractPromotionResult:
    status: str
    validation_attempted: bool
    reconstruction_error: float | None
    failure_reason: str | None
    loader_mode: str | None = None
    problem_key: str | None = None
    logical_operator_count: int | None = None
    runtime_parameter_count: int | None = None
    selected_term_count: int | None = None


@dataclass(frozen=True)
class _ValidatedSource:
    payload: Mapping[str, Any]
    ansatz_payload: Mapping[str, Any]
    layout: Any
    theta_runtime: np.ndarray
    theta_logical: np.ndarray
    source_terms: tuple[AnsatzTerm, ...]


@dataclass(frozen=True)
class _SanitizedBuild:
    sanitized_ansatz: dict[str, Any]
    sanitization: dict[str, Any]
    prepared_state_replay_error: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QSERuntimePromotionError(f"{name} must be a mapping.")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise QSERuntimePromotionError(f"{name} must be a sequence.")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QSERuntimePromotionError(f"{name} must be a finite number.")
    out = float(value)
    if not math.isfinite(out):
        raise QSERuntimePromotionError(f"{name} must be a finite number.")
    return out


def _strict_int(value: Any, *, name: str, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QSERuntimePromotionError(f"{name} must be an integer.")
    out = int(value)
    if min_value is not None and out < int(min_value):
        raise QSERuntimePromotionError(f"{name} must be >= {min_value}.")
    return out


def _finite_float_array(value: Any, *, name: str) -> np.ndarray:
    raw = _sequence(value, name=name)
    out = np.asarray([_finite_float(x, name=f"{name}[{idx}]") for idx, x in enumerate(raw)], dtype=float)
    return out.reshape(-1)


def _validate_config(config: QSERuntimePromotionConfig) -> None:
    max_error = float(config.max_reconstruction_error)
    if not math.isfinite(max_error) or max_error < 0.0:
        raise QSERuntimePromotionError("max_reconstruction_error must be finite and non-negative.")
    cutoff = float(config.amplitude_cutoff)
    if not math.isfinite(cutoff) or cutoff < 0.0:
        raise QSERuntimePromotionError("amplitude_cutoff must be finite and non-negative.")


def _terms_from_ansatz_payload(ansatz_payload: Mapping[str, Any]) -> tuple[AnsatzTerm, ...]:
    records = _sequence(ansatz_payload.get("generator_terms"), name="ansatz_payload.generator_terms")
    terms: list[AnsatzTerm] = []
    for idx, raw in enumerate(records):
        record = _mapping(raw, name=f"ansatz_payload.generator_terms[{idx}]")
        term_records = _sequence(record.get("terms"), name=f"ansatz_payload.generator_terms[{idx}].terms")
        try:
            poly = polynomial_from_serialized_terms(
                term_records,
                require_real_coefficients=True,
                allow_empty_after_pruning=False,
            )
        except Exception as exc:  # pragma: no cover - exact message depends on IO helper.
            raise QSERuntimePromotionError(
                f"Invalid ansatz_payload.generator_terms[{idx}].terms: {exc}"
            ) from exc
        terms.append(
            AnsatzTerm(
                label=str(record.get("label", f"term_{idx}")),
                polynomial=poly,
                execution_mode=str(record.get("execution_mode", "termwise_product")),
            )
        )
    return tuple(terms)


def _validate_source(payload: Mapping[str, Any]) -> _ValidatedSource:
    root = _mapping(payload, name="qse_root_refit")
    if root.get("schema_version") != QSE_ROOT_REFIT_SCHEMA_VERSION:
        raise QSERuntimePromotionError(f"schema_version must be {QSE_ROOT_REFIT_SCHEMA_VERSION!r}.")
    if root.get("pipeline") != "qse_root_refit":
        raise QSERuntimePromotionError("pipeline must be 'qse_root_refit'.")
    if root.get("uses_qiskit") is not False:
        raise QSERuntimePromotionError("qse_root_refit source must have uses_qiskit=false.")

    boundary = _mapping(root.get("controller_boundary"), name="controller_boundary")
    if boundary.get("controller_usable") is not False:
        raise QSERuntimePromotionError("qse_root_refit source must not already be controller_usable.")
    if boundary.get("ansatz_payload_potentially_promotable") is not True:
        raise QSERuntimePromotionError("controller_boundary.ansatz_payload_potentially_promotable must be true.")

    fit_summary = _mapping(root.get("fit_summary"), name="fit_summary")
    passes = _mapping(fit_summary.get("passes"), name="fit_summary.passes")
    if passes.get("all_thresholds") is not True:
        raise QSERuntimePromotionError("fit_summary.passes.all_thresholds must be true for promotion.")

    visibility = _mapping(root.get("visibility"), name="visibility")
    controller_refs = list(
        _sequence(
            visibility.get("controller_visible_payload_refs", []),
            name="visibility.controller_visible_payload_refs",
        )
    )
    if controller_refs:
        raise QSERuntimePromotionError(
            "qse_root_refit source must not expose controller-visible refs before promotion."
        )

    ansatz_payload = _mapping(root.get("ansatz_payload"), name="ansatz_payload")
    if ansatz_payload.get("ansatz_schema") != "pauli_rotation_ansatz_v1":
        raise QSERuntimePromotionError("ansatz_payload.ansatz_schema must be 'pauli_rotation_ansatz_v1'.")
    if ansatz_payload.get("qpu_preparable_in_principle") is not True:
        raise QSERuntimePromotionError("ansatz_payload.qpu_preparable_in_principle must be true.")

    try:
        layout = deserialize_layout(_mapping(ansatz_payload.get("parameterization"), name="ansatz_payload.parameterization"))
    except Exception as exc:
        raise QSERuntimePromotionError(f"Invalid ansatz_payload.parameterization: {exc}") from exc
    if int(layout.runtime_parameter_count) <= 0:
        raise QSERuntimePromotionError("ansatz_payload parameterization has zero runtime parameters.")

    theta_runtime = _finite_float_array(ansatz_payload.get("theta_runtime"), name="ansatz_payload.theta_runtime")
    if int(theta_runtime.size) != int(layout.runtime_parameter_count):
        raise QSERuntimePromotionError(
            "ansatz_payload.theta_runtime length "
            f"{theta_runtime.size} does not match runtime_parameter_count {layout.runtime_parameter_count}."
        )

    raw_theta_logical = ansatz_payload.get("theta_logical", None)
    if raw_theta_logical is None:
        theta_logical = project_runtime_theta_block_mean(theta_runtime, layout)
    else:
        theta_logical = _finite_float_array(raw_theta_logical, name="ansatz_payload.theta_logical")
        if int(theta_logical.size) != int(layout.logical_parameter_count):
            theta_logical = project_runtime_theta_block_mean(theta_runtime, layout)

    generator_records = _sequence(ansatz_payload.get("generator_terms"), name="ansatz_payload.generator_terms")
    if int(len(generator_records)) != int(layout.logical_parameter_count):
        raise QSERuntimePromotionError(
            "ansatz_payload.generator_terms length "
            f"{len(generator_records)} does not match logical_operator_count {layout.logical_parameter_count}."
        )
    source_terms = _terms_from_ansatz_payload(ansatz_payload)
    if int(len(source_terms)) != int(layout.logical_parameter_count):
        raise QSERuntimePromotionError("ansatz_payload.generator_terms did not deserialize to the expected term count.")

    try:
        reconstruct_ansatz_state_from_payload({"ansatz_payload": dict(ansatz_payload)})
    except (QSERootRefitError, Exception) as exc:
        raise QSERuntimePromotionError(f"ansatz_payload replay failed: {exc}") from exc

    return _ValidatedSource(
        payload=root,
        ansatz_payload=ansatz_payload,
        layout=layout,
        theta_runtime=np.asarray(theta_runtime, dtype=float).reshape(-1),
        theta_logical=np.asarray(theta_logical, dtype=float).reshape(-1),
        source_terms=source_terms,
    )


def _complex_coeff_from_term_record(record: Mapping[str, Any], *, name: str) -> complex:
    if "coeff_re" in record or "coeff_im" in record:
        return complex(
            _finite_float(record.get("coeff_re", 0.0), name=f"{name}.coeff_re"),
            _finite_float(record.get("coeff_im", 0.0), name=f"{name}.coeff_im"),
        )
    for key in ("coeff", "coefficient", "value"):
        if key not in record:
            continue
        coeff = record[key]
        if isinstance(coeff, Mapping):
            return complex(
                _finite_float(coeff.get("re", 0.0), name=f"{name}.{key}.re"),
                _finite_float(coeff.get("im", 0.0), name=f"{name}.{key}.im"),
            )
        return complex(_finite_float(coeff, name=f"{name}.{key}"), 0.0)
    raise QSERuntimePromotionError(f"{name} is missing coefficient fields.")


def _pauli_label_from_term_record(record: Mapping[str, Any], *, name: str) -> str:
    for key in ("pauli_exyz", "label_exyz", "pauli_label_exyz", "label"):
        raw = record.get(key)
        if raw not in {None, ""}:
            return str(raw)
    raise QSERuntimePromotionError(f"{name} is missing a Pauli label.")


def _sanitize_pauli_term_record(raw: Any, *, name: str) -> dict[str, Any]:
    record = _mapping(raw, name=name)
    pauli_exyz = _pauli_label_from_term_record(record, name=name)
    nq = _strict_int(record.get("nq", len(pauli_exyz)), name=f"{name}.nq", min_value=0)
    if len(pauli_exyz) != int(nq):
        raise QSERuntimePromotionError(f"{name}.pauli_exyz length does not match nq={nq}.")
    coeff = _complex_coeff_from_term_record(record, name=name)
    if abs(float(coeff.imag)) > 1.0e-12:
        raise QSERuntimePromotionError(f"{name} has non-real coefficient {coeff!r}.")
    return {
        "pauli_exyz": str(pauli_exyz),
        "coeff_re": float(coeff.real),
        "coeff_im": 0.0,
        "nq": int(nq),
    }


def _sanitize_parameterization(parameterization: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    blocks_raw = _sequence(parameterization.get("blocks"), name="ansatz_payload.parameterization.blocks")
    label_map: list[dict[str, Any]] = []
    blocks_out: list[dict[str, Any]] = []
    runtime_start_expected = 0
    for idx, raw_block in enumerate(blocks_raw):
        block = _mapping(raw_block, name=f"ansatz_payload.parameterization.blocks[{idx}]")
        source_label = str(block.get("candidate_label", f"term_{idx}"))
        sanitized_label = f"promoted_generator_{idx}"
        logical_index = _strict_int(block.get("logical_index", idx), name=f"parameterization.blocks[{idx}].logical_index", min_value=0)
        if int(logical_index) != int(idx):
            raise QSERuntimePromotionError(
                f"parameterization.blocks[{idx}].logical_index must equal {idx}."
            )
        runtime_start = _strict_int(
            block.get("runtime_start", runtime_start_expected),
            name=f"parameterization.blocks[{idx}].runtime_start",
            min_value=0,
        )
        if int(runtime_start) != int(runtime_start_expected):
            raise QSERuntimePromotionError(
                f"parameterization.blocks[{idx}].runtime_start must equal {runtime_start_expected}."
            )
        terms_out = [
            _sanitize_pauli_term_record(term, name=f"parameterization.blocks[{idx}].runtime_terms_exyz[{term_idx}]")
            for term_idx, term in enumerate(
                _sequence(
                    block.get("runtime_terms_exyz"),
                    name=f"parameterization.blocks[{idx}].runtime_terms_exyz",
                )
            )
        ]
        runtime_count = _strict_int(
            block.get("runtime_count", len(terms_out)),
            name=f"parameterization.blocks[{idx}].runtime_count",
            min_value=0,
        )
        if int(runtime_count) != int(len(terms_out)):
            raise QSERuntimePromotionError(
                f"parameterization.blocks[{idx}].runtime_count must equal sanitized term count {len(terms_out)}."
            )
        blocks_out.append(
            {
                "candidate_label": sanitized_label,
                "logical_index": int(logical_index),
                "runtime_start": int(runtime_start),
                "runtime_count": int(runtime_count),
                "runtime_terms_exyz": terms_out,
            }
        )
        label_map.append(
            {
                "logical_index": int(logical_index),
                "source_label": source_label,
                "sanitized_label": sanitized_label,
            }
        )
        runtime_start_expected += int(runtime_count)
    sanitized = {
        "mode": str(parameterization.get("mode", "per_pauli_term_v1")),
        "term_order": str(parameterization.get("term_order", "sorted")),
        "ignore_identity": bool(parameterization.get("ignore_identity", True)),
        "coefficient_tolerance": _finite_float(
            parameterization.get("coefficient_tolerance", 1.0e-12),
            name="parameterization.coefficient_tolerance",
        ),
        "logical_operator_count": int(len(blocks_out)),
        "runtime_parameter_count": int(runtime_start_expected),
        "blocks": blocks_out,
    }
    return sanitized, label_map


def _sanitize_generator_terms(
    records: Sequence[Any],
    label_map: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(records) != len(label_map):
        raise QSERuntimePromotionError("generator_terms and sanitized label map length mismatch.")
    for idx, raw in enumerate(records):
        record = _mapping(raw, name=f"ansatz_payload.generator_terms[{idx}]")
        term_records = _sequence(record.get("terms"), name=f"ansatz_payload.generator_terms[{idx}].terms")
        out.append(
            {
                "logical_index": int(label_map[idx]["logical_index"]),
                "label": str(label_map[idx]["sanitized_label"]),
                "execution_mode": str(record.get("execution_mode", "termwise_product")),
                "terms": [
                    _sanitize_pauli_term_record(
                        term,
                        name=f"ansatz_payload.generator_terms[{idx}].terms[{term_idx}]",
                    )
                    for term_idx, term in enumerate(term_records)
                ],
            }
        )
    return out


def _runtime_pauli_labels(parameterization: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    blocks = _sequence(parameterization.get("blocks"), name="parameterization.blocks")
    for block_idx, raw_block in enumerate(blocks):
        block = _mapping(raw_block, name=f"parameterization.blocks[{block_idx}]")
        for term_idx, raw_term in enumerate(
            _sequence(block.get("runtime_terms_exyz"), name=f"parameterization.blocks[{block_idx}].runtime_terms_exyz")
        ):
            term = _mapping(raw_term, name=f"parameterization.blocks[{block_idx}].runtime_terms_exyz[{term_idx}]")
            labels.append(str(term.get("pauli_exyz")))
    return labels


def reconstruct_promoted_ansatz_state_from_payload(payload: Mapping[str, Any]) -> np.ndarray:
    """Replay a promoted artifact's sanitized ansatz payload."""

    root = _mapping(payload, name="payload")
    if isinstance(root.get("sanitized_ansatz"), Mapping):
        ansatz_payload = _mapping(root["sanitized_ansatz"], name="sanitized_ansatz")
    elif isinstance(root.get("ansatz_payload"), Mapping):
        ansatz_payload = _mapping(root["ansatz_payload"], name="ansatz_payload")
    else:
        ansatz_payload = root
    try:
        return reconstruct_ansatz_state_from_payload({"ansatz_payload": ansatz_payload})
    except (QSERootRefitError, Exception) as exc:
        raise QSERuntimePromotionError(f"promoted ansatz replay failed: {exc}") from exc


def _build_sanitized_ansatz(
    validated: _ValidatedSource,
    *,
    amplitude_cutoff: float,
) -> _SanitizedBuild:
    ansatz = validated.ansatz_payload
    parameterization, label_map = _sanitize_parameterization(
        _mapping(ansatz.get("parameterization"), name="ansatz_payload.parameterization")
    )
    generator_terms = _sanitize_generator_terms(
        _sequence(ansatz.get("generator_terms"), name="ansatz_payload.generator_terms"),
        label_map,
    )
    theta_runtime = [float(x) for x in validated.theta_runtime.tolist()]
    theta_logical = [float(x) for x in validated.theta_logical.tolist()]

    reference_state, _reference_provenance = statevector_from_manifest(
        _mapping(ansatz.get("reference_state"), name="ansatz_payload.reference_state"),
        state_key="auto",
    )
    prepared_state = reconstruct_ansatz_state_from_payload({"ansatz_payload": dict(ansatz)})

    reference_manifest = build_statevector_manifest(
        psi_state=reference_state,
        source="promoted_ansatz_reference_state",
        handoff_state_kind="reference_state",
        amplitude_cutoff=float(amplitude_cutoff),
    )
    initial_manifest = build_statevector_manifest(
        psi_state=prepared_state,
        source="promoted_ansatz_prepared_state",
        handoff_state_kind="prepared_state",
        amplitude_cutoff=float(amplitude_cutoff),
    )

    operators = [str(item["sanitized_label"]) for item in label_map]
    sanitized_ansatz: dict[str, Any] = {
        "ansatz_schema": "pauli_rotation_ansatz_v1",
        "parameterization_mode": "per_pauli_term",
        "operator_label_policy": "promoted_generator_index",
        "operators": list(operators),
        "selected_operator_labels": list(operators),
        "runtime_pauli_labels_exyz": _runtime_pauli_labels(parameterization),
        "generator_terms": generator_terms,
        "parameterization": parameterization,
        "theta_runtime": theta_runtime,
        "theta_logical": theta_logical,
        "reference_state": reference_manifest,
        "ansatz_input_state": reference_manifest,
        "initial_state": initial_manifest,
        "qpu_preparable_in_principle": True,
        "matches_scaffold_runtime_contract": False,
        "runtime_parameter_count": int(validated.layout.runtime_parameter_count),
        "logical_operator_count": int(validated.layout.logical_parameter_count),
    }

    replayed = reconstruct_promoted_ansatz_state_from_payload({"sanitized_ansatz": sanitized_ansatz})
    replay_error = float(
        np.linalg.norm(
            np.asarray(replayed, dtype=complex).reshape(-1)
            - np.asarray(prepared_state, dtype=complex).reshape(-1)
        )
    )
    sanitized_ansatz["prepared_state_replay_error"] = float(replay_error)

    sanitization = {
        "ansatz_source": "qse_root_refit.ansatz_payload",
        "controller_visible_data_policy": "only_sanitized_circuit_parameters_and_state_manifests_after_runtime_validation",
        "operator_label_policy": "promoted_generator_index",
        "source_to_sanitized_operator_labels": [dict(item) for item in label_map],
        "qse_diagnostics_copied_to_controller_payload": False,
        "target_amplitudes_copied_to_controller_payload": False,
        "fit_diagnostics_copied_to_controller_payload": False,
    }
    return _SanitizedBuild(
        sanitized_ansatz=sanitized_ansatz,
        sanitization=sanitization,
        prepared_state_replay_error=float(replay_error),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QSERuntimePromotionError("Cannot serialize non-finite runtime setting.")
        return float(value)
    return value


def _reject_forbidden_runtime_payload_markers(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            for marker in _FORBIDDEN_RUNTIME_PAYLOAD_MARKERS:
                if marker in key_lower:
                    raise QSERuntimePromotionError(
                        f"runtime_payload contains forbidden marker {marker!r} at {path}.{key_text}."
                    )
            _reject_forbidden_runtime_payload_markers(item, path=f"{path}.{key_text}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for idx, item in enumerate(value):
            _reject_forbidden_runtime_payload_markers(item, path=f"{path}[{idx}]")
        return
    if isinstance(value, str):
        value_lower = value.lower()
        for marker in _FORBIDDEN_RUNTIME_PAYLOAD_MARKERS:
            if marker in value_lower:
                raise QSERuntimePromotionError(
                    f"runtime_payload contains forbidden marker {marker!r} at {path}."
                )


def _safe_runtime_settings(template_payload: Mapping[str, Any]) -> dict[str, Any]:
    settings = _mapping(template_payload.get("settings"), name="runtime_template.settings")
    out: dict[str, Any] = {}
    for key in sorted(_SAFE_RUNTIME_SETTING_KEYS):
        if key in settings:
            out[key] = _json_safe(settings[key])
    if "problem" not in out:
        raise QSERuntimePromotionError("runtime template settings must include a supported problem key.")
    return out


def _safe_pool_type(template_payload: Mapping[str, Any], settings: Mapping[str, Any]) -> str | None:
    adapt = template_payload.get("adapt_vqe")
    candidates: list[Any] = []
    if isinstance(adapt, Mapping):
        candidates.append(adapt.get("pool_type"))
    candidates.append(settings.get("adapt_pool"))
    for raw in candidates:
        if raw not in {None, ""}:
            text = str(raw).strip()
            if text and text != "fixed_scaffold_locked":
                return text
    return None


def _build_runtime_payload(
    *,
    sanitized_ansatz: Mapping[str, Any],
    template_payload: Mapping[str, Any],
) -> dict[str, Any]:
    settings = _safe_runtime_settings(template_payload)
    pool_type = _safe_pool_type(template_payload, settings)
    operators = [str(x) for x in _sequence(sanitized_ansatz.get("operators"), name="sanitized_ansatz.operators")]
    theta_runtime = [float(x) for x in _finite_float_array(sanitized_ansatz.get("theta_runtime"), name="sanitized_ansatz.theta_runtime")]
    theta_logical = [float(x) for x in _finite_float_array(sanitized_ansatz.get("theta_logical"), name="sanitized_ansatz.theta_logical")]
    parameterization = copy.deepcopy(dict(_mapping(sanitized_ansatz.get("parameterization"), name="sanitized_ansatz.parameterization")))
    layout = deserialize_layout(parameterization)
    problem_key = str(settings.get("problem", "")).strip().lower()
    hh_fixed_scaffold = problem_key == "hh"
    runtime_pauli_labels: list[str] = []

    if hh_fixed_scaffold:
        # The HH loader preserves a locked imported scaffold only through its
        # explicit fixed-scaffold route.  A bare structure_locked=true field is
        # otherwise normalized as replay_family and loses the lock boundary.
        settings["adapt_pool"] = "fixed_scaffold_locked"
        runtime_pauli_labels = _runtime_pauli_labels(parameterization)

    adapt_vqe: dict[str, Any] = {
        "success": True,
        "method": "offline_qse_runtime_promotion",
        "operators": list(operators),
        "optimal_point": list(theta_runtime),
        "logical_optimal_point": list(theta_logical),
        "parameterization": parameterization,
        "num_parameters": int(layout.runtime_parameter_count),
        "logical_num_parameters": int(layout.logical_parameter_count),
        "ansatz_depth": int(layout.logical_parameter_count),
        "structure_locked": True,
        "fixed_scaffold_kind": "promoted_ansatz_locked_v1",
    }
    if hh_fixed_scaffold:
        adapt_vqe["pool_type"] = "fixed_scaffold_locked"
        adapt_vqe["fixed_scaffold_metadata"] = {
            "schema_version": 1,
            "route_family": "locked_imported_scaffold_v1",
            "subject_kind": "qse_excited_state_refit_v1",
            "structure_locked": True,
            "operator_count": int(layout.logical_parameter_count),
            "runtime_term_count": int(layout.runtime_parameter_count),
            "term_order_id": "promoted_runtime_parameterization",
            "term_order_basis": str(parameterization.get("term_order", "serialized")),
            "source_order_runtime_indices": list(range(int(layout.runtime_parameter_count))),
            "runtime_term_labels_exyz": list(runtime_pauli_labels),
            "source_order_runtime_term_labels_exyz": list(runtime_pauli_labels),
            "source_pool_type": pool_type,
        }
    elif pool_type is not None:
        adapt_vqe["pool_type"] = str(pool_type)

    runtime_payload = {
        "pipeline": _RUNTIME_PAYLOAD_PIPELINE,
        "generated_utc": _utc_now(),
        "settings": settings,
        "adapt_vqe": adapt_vqe,
        "ansatz_input_state": copy.deepcopy(dict(_mapping(sanitized_ansatz.get("ansatz_input_state"), name="sanitized_ansatz.ansatz_input_state"))),
        "initial_state": copy.deepcopy(dict(_mapping(sanitized_ansatz.get("initial_state"), name="sanitized_ansatz.initial_state"))),
    }
    _reject_forbidden_runtime_payload_markers(runtime_payload, path="runtime_payload")
    return runtime_payload


def _replay_runtime_input_state(runtime_input: Any) -> np.ndarray:
    layout = runtime_input.base_layout
    executor = CompiledAnsatzExecutor(
        list(runtime_input.selected_terms),
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )
    return np.asarray(
        executor.prepare_state(
            np.asarray(runtime_input.theta_runtime, dtype=float).reshape(-1),
            np.asarray(runtime_input.psi_ref, dtype=complex).reshape(-1),
        ),
        dtype=complex,
    ).reshape(-1)


def _layout_signature(layout: Any) -> tuple[Any, ...]:
    return tuple(
        (
            str(block.candidate_label),
            int(block.logical_index),
            int(block.runtime_start),
            tuple(
                (
                    str(spec.pauli_exyz),
                    round(float(spec.coeff_real), 15),
                    int(spec.nq),
                )
                for spec in block.terms
            ),
        )
        for block in layout.blocks
    )


def _term_signature(term: AnsatzTerm) -> tuple[tuple[str, float, float, int], ...]:
    records: list[tuple[str, float, float, int]] = []
    for raw_term in term.polynomial.return_polynomial():
        coeff = complex(raw_term.p_coeff)
        records.append(
            (
                str(raw_term.pw2strng()),
                round(float(coeff.real), 15),
                round(float(coeff.imag), 15),
                int(raw_term.nqubit()),
            )
        )
    return tuple(sorted(records))


def _terms_signature(terms: Sequence[AnsatzTerm]) -> tuple[tuple[str, tuple[tuple[str, float, float, int], ...]], ...]:
    return tuple((str(term.label), _term_signature(term)) for term in terms)


def _validate_runtime_payload(
    *,
    runtime_payload: Mapping[str, Any],
    output_json: Path,
    expected_layout: Any,
    expected_theta_runtime: np.ndarray,
    expected_operator_labels: Sequence[str],
    expected_terms: Sequence[AnsatzTerm],
    max_reconstruction_error: float,
) -> RuntimeContractPromotionResult:
    try:
        from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload

        runtime_input = load_scaffold_runtime_input_from_payload(
            runtime_payload,
            artifact_json=Path(output_json),
        )
        loader_mode = str(runtime_input.provenance.get("loader_mode", "")) or None
        problem_key = str(getattr(runtime_input.resolved_problem, "family_key", "")) or None

        runtime_count = int(runtime_input.base_layout.runtime_parameter_count)
        logical_count = int(runtime_input.base_layout.logical_parameter_count)
        expected_runtime_count = int(expected_layout.runtime_parameter_count)
        expected_logical_count = int(expected_layout.logical_parameter_count)
        if runtime_count != expected_runtime_count:
            raise QSERuntimePromotionError(
                f"loader runtime_parameter_count={runtime_count} does not match emitted count {expected_runtime_count}."
            )
        if logical_count != expected_logical_count:
            raise QSERuntimePromotionError(
                f"loader logical_operator_count={logical_count} does not match emitted count {expected_logical_count}."
            )
        theta = np.asarray(runtime_input.theta_runtime, dtype=float).reshape(-1)
        expected_theta = np.asarray(expected_theta_runtime, dtype=float).reshape(-1)
        if theta.shape != expected_theta.shape or not np.allclose(theta, expected_theta, atol=1.0e-12, rtol=1.0e-12):
            raise QSERuntimePromotionError("loader theta_runtime does not match emitted theta_runtime.")
        if int(len(runtime_input.selected_terms)) != expected_logical_count:
            raise QSERuntimePromotionError(
                f"loader selected_term_count={len(runtime_input.selected_terms)} does not match logical count {expected_logical_count}."
            )
        selected_labels = [str(term.label) for term in runtime_input.selected_terms]
        if selected_labels != [str(label) for label in expected_operator_labels]:
            raise QSERuntimePromotionError(
                f"loader selected operator labels {selected_labels!r} do not match emitted labels {list(expected_operator_labels)!r}."
            )
        if _layout_signature(runtime_input.base_layout) != _layout_signature(expected_layout):
            raise QSERuntimePromotionError("loader base_layout does not exactly match emitted parameterization.")
        if _terms_signature(runtime_input.selected_terms) != _terms_signature(expected_terms):
            raise QSERuntimePromotionError("loader selected_terms do not exactly match emitted generator_terms.")
        if runtime_input.structure_locked is not True:
            raise QSERuntimePromotionError("loader did not preserve structure_locked=true.")
        if bool(runtime_input.can_structural_edit):
            raise QSERuntimePromotionError("loader returned can_structural_edit=true for promoted locked payload.")
        if runtime_input.exact_energy is not None:
            raise QSERuntimePromotionError("loader exposed exact_energy for promoted runtime payload.")

        emitted_initial, _ = statevector_from_manifest(
            _mapping(runtime_payload.get("initial_state"), name="runtime_payload.initial_state"),
            state_key="auto",
        )
        replayed = _replay_runtime_input_state(runtime_input)
        replay_error = float(
            np.linalg.norm(
                np.asarray(replayed, dtype=complex).reshape(-1)
                - np.asarray(emitted_initial, dtype=complex).reshape(-1)
            )
        )
        loader_initial_error = float(
            np.linalg.norm(
                np.asarray(runtime_input.psi_initial, dtype=complex).reshape(-1)
                - np.asarray(emitted_initial, dtype=complex).reshape(-1)
            )
        )
        reconstruction_error = float(max(replay_error, loader_initial_error))
        if reconstruction_error > float(max_reconstruction_error):
            raise QSERuntimePromotionError(
                "loader replay reconstruction error "
                f"{reconstruction_error:.3e} exceeds tolerance {float(max_reconstruction_error):.3e}."
            )
        return RuntimeContractPromotionResult(
            status="validated",
            validation_attempted=True,
            reconstruction_error=float(reconstruction_error),
            failure_reason=None,
            loader_mode=loader_mode,
            problem_key=problem_key,
            logical_operator_count=logical_count,
            runtime_parameter_count=runtime_count,
            selected_term_count=int(len(runtime_input.selected_terms)),
        )
    except Exception as exc:
        return RuntimeContractPromotionResult(
            status="failed",
            validation_attempted=True,
            reconstruction_error=None,
            failure_reason=str(exc),
            loader_mode=None,
            problem_key=None,
            logical_operator_count=None,
            runtime_parameter_count=None,
            selected_term_count=None,
        )


def _runtime_contract_payload(
    result: RuntimeContractPromotionResult,
    *,
    max_reconstruction_error: float,
    emitted_runtime_parameter_count: int,
    emitted_logical_operator_count: int,
) -> dict[str, Any]:
    return {
        "status": str(result.status),
        "controller_usable": str(result.status) == "validated",
        "validation_attempted": bool(result.validation_attempted),
        "loader_mode": result.loader_mode,
        "problem_key": result.problem_key,
        "prepared_state_reconstruction_error": result.reconstruction_error,
        "max_reconstruction_error": float(max_reconstruction_error),
        "failure_reason": result.failure_reason,
        "runtime_parameter_count": result.runtime_parameter_count,
        "logical_operator_count": result.logical_operator_count,
        "selected_term_count": result.selected_term_count,
        "emitted_runtime_parameter_count": int(emitted_runtime_parameter_count),
        "emitted_logical_operator_count": int(emitted_logical_operator_count),
    }


def _controller_boundary(validated: bool) -> dict[str, Any]:
    return {
        "controller_usable": bool(validated),
        "matches_scaffold_runtime_contract": bool(validated),
        "feeds_controller_decisions": False,
        "decision_path_allowed": bool(validated),
        "realtime_wiring": False,
        "live_route_executed": False,
        "qse_diagnostics_forbidden_to_controller": True,
        "promotion_requires_runtime_contract_validation": not bool(validated),
        "runtime_payload_controller_visible": bool(validated),
    }


def _visibility(validated: bool, *, runtime_payload_present: bool) -> dict[str, Any]:
    diagnostic_refs = [
        "source",
        "sanitization",
        "runtime_contract",
        "warnings",
    ]
    if not validated:
        diagnostic_refs.append("sanitized_ansatz")
    if runtime_payload_present and not validated:
        diagnostic_refs.append("runtime_payload")
    return {
        "controller_visible_payload_refs": ["runtime_payload"] if validated else [],
        "potentially_promotable_payload_refs": ["sanitized_ansatz"],
        "diagnostic_only_payload_refs": diagnostic_refs,
        "forbidden_to_controller_refs": list(_FORBIDDEN_SOURCE_CONTROLLER_REFS),
    }


def promote_qse_root_refit(config: QSERuntimePromotionConfig) -> dict[str, Any]:
    """Promote a ``qse_root_refit_v1`` artifact into a sanitized runtime artifact."""

    _validate_config(config)
    source_path = Path(config.qse_root_refit_json)
    output_path = Path(config.output_json)
    source_payload = _mapping(_read_json(source_path), name="qse_root_refit")
    validated_source = _validate_source(source_payload)
    sanitized_build = _build_sanitized_ansatz(
        validated_source,
        amplitude_cutoff=float(config.amplitude_cutoff),
    )
    sanitized_ansatz = sanitized_build.sanitized_ansatz

    runtime_payload: dict[str, Any] | None = None
    if config.runtime_template_json is None:
        runtime_result = RuntimeContractPromotionResult(
            status="not_representable",
            validation_attempted=False,
            reconstruction_error=None,
            failure_reason=(
                "runtime_template_json not provided; existing scaffold runtime loader requires "
                "supported runtime settings before controller use."
            ),
            logical_operator_count=None,
            runtime_parameter_count=None,
            selected_term_count=None,
        )
    else:
        try:
            template_payload = _mapping(_read_json(Path(config.runtime_template_json)), name="runtime_template")
            runtime_payload = _build_runtime_payload(
                sanitized_ansatz=sanitized_ansatz,
                template_payload=template_payload,
            )
            runtime_result = _validate_runtime_payload(
                runtime_payload=runtime_payload,
                output_json=output_path,
                expected_layout=deserialize_layout(
                    _mapping(sanitized_ansatz.get("parameterization"), name="sanitized_ansatz.parameterization")
                ),
                expected_theta_runtime=np.asarray(sanitized_ansatz.get("theta_runtime"), dtype=float).reshape(-1),
                expected_operator_labels=[
                    str(label)
                    for label in _sequence(
                        sanitized_ansatz.get("operators"),
                        name="sanitized_ansatz.operators",
                    )
                ],
                expected_terms=_terms_from_ansatz_payload(sanitized_ansatz),
                max_reconstruction_error=float(config.max_reconstruction_error),
            )
        except Exception as exc:
            runtime_result = RuntimeContractPromotionResult(
                status="failed",
                validation_attempted=True,
                reconstruction_error=None,
                failure_reason=str(exc),
            )

    if bool(config.require_runtime_contract) and runtime_result.status not in _RUNTIME_VALIDATED_STATUSES:
        raise QSERuntimePromotionError(
            "runtime contract validation required but status is "
            f"{runtime_result.status!r}: {runtime_result.failure_reason}"
        )

    controller_usable = runtime_result.status == "validated"
    sanitized_ansatz["matches_scaffold_runtime_contract"] = bool(controller_usable)

    source_qse_diag = validated_source.payload.get("qse_ritz_diagnostics")
    source_state_index = (
        source_qse_diag.get("state_index")
        if isinstance(source_qse_diag, Mapping)
        else None
    )
    runtime_contract = _runtime_contract_payload(
        runtime_result,
        max_reconstruction_error=float(config.max_reconstruction_error),
        emitted_runtime_parameter_count=int(sanitized_ansatz["runtime_parameter_count"]),
        emitted_logical_operator_count=int(sanitized_ansatz["logical_operator_count"]),
    )
    artifact = {
        "schema_version": QSE_RUNTIME_PROMOTED_SCHEMA_VERSION,
        "pipeline": QSE_RUNTIME_PROMOTION_PIPELINE,
        "generated_utc": _utc_now(),
        "backend": "offline_statevector_promotion",
        "uses_qiskit": False,
        "controller_boundary": _controller_boundary(controller_usable),
        "source": {
            "qse_root_refit_json": str(source_path),
            "qse_root_refit_sha256": _sha256_file(source_path),
            "source_schema_version": validated_source.payload.get("schema_version"),
            "source_pipeline": validated_source.payload.get("pipeline"),
            "source_backend": validated_source.payload.get("backend"),
            "source_ansatz_payload_sha256": _sha256_json(validated_source.ansatz_payload),
            "source_state_index": source_state_index,
            "source_state_index_visibility": "diagnostic_only_forbidden_to_controller",
        },
        "sanitization": sanitized_build.sanitization,
        "sanitized_ansatz": sanitized_ansatz,
        "runtime_payload": runtime_payload,
        "runtime_contract": runtime_contract,
        "visibility": _visibility(controller_usable, runtime_payload_present=runtime_payload is not None),
        "warnings": [
            "offline_runtime_contract_promotion_only",
            "no_realtime_or_controller_route_executed",
            "qse_ritz_diagnostics_fit_diagnostics_and_target_amplitudes_forbidden_to_controller",
            *([] if controller_usable else ["promoted_artifact_not_controller_usable_without_validated_runtime_contract"]),
        ],
    }
    write_manifest_json(output_path, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline promotion of qse_root_refit_v1 into qse_runtime_promoted_ansatz_v1."
    )
    parser.add_argument("--qse-root-refit-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--runtime-template-json", type=Path, default=None)
    parser.add_argument("--require-runtime-contract", action="store_true")
    parser.add_argument("--max-reconstruction-error", type=float, default=1.0e-10)
    parser.add_argument("--amplitude-cutoff", type=float, default=1.0e-12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = QSERuntimePromotionConfig(
        qse_root_refit_json=Path(args.qse_root_refit_json),
        output_json=Path(args.output_json),
        runtime_template_json=args.runtime_template_json,
        require_runtime_contract=bool(args.require_runtime_contract),
        max_reconstruction_error=float(args.max_reconstruction_error),
        amplitude_cutoff=float(args.amplitude_cutoff),
    )
    try:
        artifact = promote_qse_root_refit(config)
    except QSERuntimePromotionError as exc:
        parser.error(str(exc))

    runtime_contract = artifact["runtime_contract"]
    print(f"output_json: {args.output_json}")
    print(f"runtime_contract_status: {runtime_contract['status']}")
    print(f"controller_usable: {str(artifact['controller_boundary']['controller_usable']).lower()}")
    print(f"runtime_parameter_count: {artifact['sanitized_ansatz']['runtime_parameter_count']}")
    print(
        "prepared_state_reconstruction_error: "
        f"{artifact['sanitized_ansatz']['prepared_state_replay_error']}"
    )
    return 0


__all__ = [
    "QSE_RUNTIME_PROMOTED_SCHEMA_VERSION",
    "QSE_RUNTIME_PROMOTION_PIPELINE",
    "QSERuntimePromotionConfig",
    "QSERuntimePromotionError",
    "RuntimeContractPromotionResult",
    "build_parser",
    "main",
    "promote_qse_root_refit",
    "reconstruct_promoted_ansatz_state_from_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())

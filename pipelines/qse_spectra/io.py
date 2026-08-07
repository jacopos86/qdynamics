"""JSON IO and manifest helpers for the isolated QSE spectra sidecar."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import (
    QSEBasisElement,
    QSEBasisVectorDiagnostics,
    QSEBasisVectorPolicy,
    QSEObservable,
    QSEPruningConfig,
    QSEResult,
    QSETransitionObservableResult,
    computational_basis_state,
    normalize_statevector,
    pauli_string_basis_element,
    pauli_string_observable,
    polynomial_basis_element,
    polynomial_observable,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


_LABEL_KEYS = ("pauli_exyz", "label_exyz", "pauli_label_exyz", "label")
_METADATA_KEYS = (
    "source",
    "alphabet_family",
    "sector_label",
    "record_label",
    "full_meta_class",
    "candidate_label",
)
_COMPLEX_COEFF_KEYS = ("coeff", "coefficient", "value")
_STATE_KEYS = ("initial_state", "ansatz_input_state")
_ARTIFACT_BASIS_SOURCES = ("selected_adapt_blocks", "full_meta", "full_meta_filtered", "hamiltonian_terms")
_DEFAULT_FULL_META_FILTER_CLASSES = (
    "uccsd_sing",
    "uccsd_dbl",
    "hh_hamiltonian_block",
    "hh_fermionic_reusable",
    "paop_cloud_p",
    "paop_cloud_x",
    "paop_disp",
    "paop_dbl",
    "paop_hopdrag",
    "paop_dbl_p",
    "paop_dbl_x",
    "paop_curdrag",
    "paop_hop2",
)
_TRANSLATION = {
    "e": "e",
    "E": "e",
    "i": "e",
    "I": "e",
    "x": "x",
    "X": "x",
    "y": "y",
    "Y": "y",
    "z": "z",
    "Z": "z",
}


def _normalize_label(label: str, *, nq: int | None = None) -> str:
    raw = str(label)
    try:
        out = "".join(_TRANSLATION[ch] for ch in raw)
    except KeyError as exc:
        raise ValueError(f"Unsupported Pauli symbol {exc.args[0]!r} in label {raw!r}.") from exc
    if nq is not None and len(out) != int(nq):
        raise ValueError(f"Pauli label {raw!r} has length {len(out)}; expected {int(nq)}.")
    return out


def _finite_float(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric; got {value!r}.") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite; got {out!r}.")
    return out


def _strict_int(value: Any, *, name: str, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer; got {value!r}.")
    out = int(value)
    if min_value is not None and out < int(min_value):
        raise ValueError(f"{name} must be >= {int(min_value)}; got {out}.")
    return out


def _complex_from_payload(value: Any, *, name: str) -> complex:
    if isinstance(value, Mapping):
        re = _finite_float(value.get("re", 0.0), name=f"{name}.re")
        im = _finite_float(value.get("im", 0.0), name=f"{name}.im")
        return complex(re, im)
    if isinstance(value, (int, float)):
        return complex(_finite_float(value, name=name), 0.0)
    raise ValueError(f"{name} must be a complex mapping or finite number; got {value!r}.")


def _complex_to_json(value: complex) -> dict[str, float]:
    value_c = complex(value)
    re = float(value_c.real)
    im = float(value_c.imag)
    if not math.isfinite(re) or not math.isfinite(im):
        raise ValueError(f"Cannot serialize non-finite complex value {value_c!r}.")
    return {"re": re, "im": im}


def _json_safe(value: Any) -> Any:
    if isinstance(value, complex):
        return _complex_to_json(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (int, float, str, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Cannot serialize non-finite float {value!r}.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _metadata_from_record(record: Mapping[str, Any], *, extra: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    raw_meta = record.get("metadata")
    if isinstance(raw_meta, Mapping):
        metadata.update(_json_safe(raw_meta))
    for key in _METADATA_KEYS:
        if key in record and key not in metadata:
            metadata[key] = _json_safe(record[key])
    if extra is not None:
        metadata.update(_json_safe(extra))
    return metadata or None


def _matrix_to_json(matrix: np.ndarray) -> list[list[dict[str, float]]]:
    arr = np.asarray(matrix, dtype=complex)
    if arr.ndim != 2:
        raise ValueError("matrix serialization expects a 2D array.")
    return [[_complex_to_json(arr[i, j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]


def _vector_to_json(vector: np.ndarray) -> list[dict[str, float]]:
    arr = np.asarray(vector, dtype=complex).reshape(-1)
    return [_complex_to_json(value) for value in arr]


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _polynomial_to_term_records(poly: PauliPolynomial) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        records.append(
            {
                "pauli_exyz": str(term.pw2strng()),
                "coeff_re": float(coeff.real),
                "coeff_im": float(coeff.imag),
                "nq": int(term.nqubit()),
            }
        )
    return records


def _polynomial_signature(poly: PauliPolynomial) -> tuple[tuple[str, float, float], ...]:
    signature: list[tuple[str, float, float]] = []
    for record in _polynomial_to_term_records(poly):
        signature.append(
            (
                str(record["pauli_exyz"]),
                round(float(record["coeff_re"]), 15),
                round(float(record["coeff_im"]), 15),
            )
        )
    return tuple(signature)


def _available_state_keys(payload: Mapping[str, Any]) -> list[str]:
    return [
        key
        for key, value in payload.items()
        if isinstance(value, Mapping) and isinstance(value.get("amplitudes_qn_to_q0"), Mapping)
    ]


def _select_state_payload(
    payload: Mapping[str, Any],
    *,
    state_key: str,
) -> tuple[Mapping[str, Any], str, list[str], str]:
    if isinstance(payload.get("amplitudes_qn_to_q0"), Mapping):
        return payload, "top_level", _available_state_keys(payload), "top_level_state_manifest"

    available = _available_state_keys(payload)
    key = str(state_key or "auto")
    if key == "auto":
        key = "initial_state"
    if key not in payload or not isinstance(payload.get(key), Mapping):
        raise ValueError(f"State key {key!r} not found in payload; available state keys: {available!r}.")
    block = payload[key]
    if not isinstance(block.get("amplitudes_qn_to_q0"), Mapping):
        raise ValueError(f"State key {key!r} does not contain amplitudes_qn_to_q0.")
    return block, key, available, "artifact_state_block"


def statevector_from_manifest(
    payload: Mapping[str, Any],
    *,
    expected_nq: int | None = None,
    state_key: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a dense normalized statevector from a sparse state manifest."""

    if not isinstance(payload, Mapping):
        raise TypeError("state manifest payload must be a mapping.")
    block, selected_key, available_keys, schema = _select_state_payload(payload, state_key=state_key)
    raw_nq = block.get("nq_total", expected_nq)
    if raw_nq is None:
        raise ValueError("State manifest requires nq_total when expected_nq is not supplied.")
    nq = _strict_int(raw_nq, name="nq_total", min_value=0)
    if expected_nq is not None and int(expected_nq) != nq:
        raise ValueError(f"State manifest nq_total={nq} does not match expected_nq={expected_nq}.")

    amps = block.get("amplitudes_qn_to_q0")
    if not isinstance(amps, Mapping):
        raise ValueError("State manifest requires amplitudes_qn_to_q0 mapping.")
    psi = np.zeros(1 << int(nq), dtype=complex)
    for bitstring, amp_payload in amps.items():
        bits = str(bitstring)
        if len(bits) != int(nq):
            raise ValueError(f"State bitstring {bits!r} has length {len(bits)}; expected {nq}.")
        if set(bits) - {"0", "1"}:
            raise ValueError(f"State bitstring {bits!r} must contain only 0/1 symbols.")
        idx = int(bits, 2) if bits else 0
        psi[idx] = _complex_from_payload(amp_payload, name=f"amplitudes_qn_to_q0[{bits!r}]")

    psi_normed, norm_before, nq_inferred = normalize_statevector(psi)
    if int(nq_inferred) != int(nq):
        raise ValueError(f"Loaded state inferred nq={nq_inferred}; manifest nq_total={nq}.")
    provenance = {
        "source_schema": schema,
        "selected_state_key": selected_key,
        "available_state_keys": available_keys,
        "nq_total": int(nq),
        "stored_norm": block.get("norm"),
        "norm_before_normalization": float(norm_before),
        "nonzero_amplitude_count": int(len(amps)),
    }
    if "source" in block:
        provenance["state_source"] = str(block.get("source"))
    return psi_normed, provenance


def load_state_json(
    path: Path,
    *,
    expected_nq: int | None = None,
    state_key: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = _load_json(path)
    psi, provenance = statevector_from_manifest(payload, expected_nq=expected_nq, state_key=state_key)
    provenance["path"] = str(Path(path))
    return psi, provenance


def _extract_label(record: Mapping[str, Any], *, index: int) -> str:
    for key in _LABEL_KEYS:
        if key in record:
            return str(record[key])
    raise ValueError(f"Term record {index} is missing a Pauli label key from {_LABEL_KEYS!r}.")


def _extract_coeff(record: Mapping[str, Any], *, index: int) -> complex:
    if "coeff_re" in record or "coeff_im" in record:
        return complex(
            _finite_float(record.get("coeff_re", 0.0), name=f"terms[{index}].coeff_re"),
            _finite_float(record.get("coeff_im", 0.0), name=f"terms[{index}].coeff_im"),
        )
    for key in _COMPLEX_COEFF_KEYS:
        if key in record:
            return _complex_from_payload(record[key], name=f"terms[{index}].{key}")
    raise ValueError(f"Term record {index} is missing coefficient fields.")


def polynomial_from_serialized_terms(
    records: Sequence[Mapping[str, Any]],
    *,
    repr_mode: str = "JW",
    drop_abs_tol: float = 1.0e-15,
    require_real_coefficients: bool = False,
    coeff_imag_abs_tol: float = 1.0e-12,
    allow_empty_after_pruning: bool = False,
) -> PauliPolynomial:
    """Build a PauliPolynomial from serialized exyz/IXYZ term records."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of term mappings.")
    order: list[str] = []
    coeff_by_label: dict[str, complex] = {}
    nq: int | None = None
    for idx, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"Term record {idx} must be a mapping.")
        label_raw = _extract_label(raw_record, index=idx)
        label_nq = raw_record.get("nq")
        if label_nq is not None:
            label = _normalize_label(label_raw, nq=_strict_int(label_nq, name=f"terms[{idx}].nq", min_value=0))
        else:
            label = _normalize_label(label_raw)
        if nq is None:
            nq = len(label)
        if len(label) != int(nq):
            raise ValueError(f"Term record {idx} label length {len(label)} does not match nq={nq}.")
        if label not in coeff_by_label:
            order.append(label)
            coeff_by_label[label] = 0.0 + 0.0j
        coeff_by_label[label] += _extract_coeff(raw_record, index=idx)

    if nq is None:
        raise ValueError("At least one serialized Pauli term is required.")

    poly = PauliPolynomial(str(repr_mode))
    retained = 0
    for label in order:
        coeff = complex(coeff_by_label[label])
        if abs(coeff) <= float(drop_abs_tol):
            continue
        if require_real_coefficients:
            if abs(float(coeff.imag)) > float(coeff_imag_abs_tol):
                raise ValueError(
                    f"Hamiltonian coefficient for {label!r} has imaginary part {coeff.imag}, "
                    f"exceeding tolerance {coeff_imag_abs_tol}."
                )
            coeff = float(coeff.real) + 0.0j
        poly.add_term(PauliTerm(int(nq), ps=label, pc=coeff))
        retained += 1
    if retained == 0:
        if not bool(allow_empty_after_pruning):
            raise ValueError("No serialized Pauli terms remain after coefficient pruning.")
        poly.add_term(PauliTerm(int(nq), ps="e" * int(nq), pc=0.0))
    return poly


def _term_records_from_payload(payload: Any) -> tuple[Sequence[Mapping[str, Any]], str]:
    if isinstance(payload, list):
        return payload, "top_level_list"
    if not isinstance(payload, Mapping):
        raise TypeError("Hamiltonian JSON must be a list or mapping.")
    if isinstance(payload.get("terms"), list):
        return payload["terms"], "terms"
    hamiltonian = payload.get("hamiltonian")
    if isinstance(hamiltonian, Mapping):
        if isinstance(hamiltonian.get("terms"), list):
            return hamiltonian["terms"], "hamiltonian.terms"
        if isinstance(hamiltonian.get("coefficients_exyz"), list):
            return hamiltonian["coefficients_exyz"], "hamiltonian.coefficients_exyz"
    if isinstance(payload.get("coefficients_exyz"), list):
        return payload["coefficients_exyz"], "coefficients_exyz"
    raise ValueError("Could not locate Hamiltonian term records in JSON payload.")


def _settings_from_payload_for_hamiltonian(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    settings = payload.get("settings")
    if isinstance(settings, Mapping):
        return settings
    if any(key in payload for key in ("problem", "problem_key", "family")):
        return payload
    return None


def _settings_get(settings: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in settings and settings[key] is not None:
            return settings[key]
    return default


def _settings_bool(settings: Mapping[str, Any], *keys: str, default: bool) -> bool:
    value = _settings_get(settings, *keys, default=default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _problem_key_from_settings(settings: Mapping[str, Any]) -> str | None:
    raw = _settings_get(settings, "problem", "problem_key", "family", default=None)
    if raw is None:
        return None
    key = str(raw).strip().lower()
    aliases = {
        "hubbard_holstein": "hh",
        "hubbard-holstein": "hh",
        "holstein": "hh",
    }
    return aliases.get(key, key)


def _polynomial_from_artifact_settings(
    payload: Mapping[str, Any],
    *,
    drop_abs_tol: float,
    require_real_coefficients: bool,
    coeff_imag_abs_tol: float,
) -> tuple[PauliPolynomial, dict[str, Any]] | None:
    """Build a Hamiltonian from artifact settings when no serialized terms exist."""

    settings = _settings_from_payload_for_hamiltonian(payload)
    if settings is None:
        return None
    problem_key = _problem_key_from_settings(settings)
    if problem_key is None:
        return None

    from pipelines.qse_spectra.static_adapt_adapter import build_artifact_problem_hamiltonian

    num_sites = int(_settings_get(settings, "L", "num_sites", default=2))
    n_ph_max = int(_settings_get(settings, "n_ph_max", "nph", "n_ph", default=0))
    built = build_artifact_problem_hamiltonian(
        problem_key=str(problem_key),
        num_sites=int(num_sites),
        t=float(_settings_get(settings, "t", "J", default=1.0)),
        u=float(_settings_get(settings, "u", "U", default=4.0)),
        dv=float(_settings_get(settings, "dv", "delta_v", default=0.0)),
        omega0=float(_settings_get(settings, "omega0", default=1.0)),
        g_ep=float(_settings_get(settings, "g_ep", "g", default=0.0)),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(_settings_get(settings, "boson_encoding", default="binary")),
        ordering=str(_settings_get(settings, "ordering", "indexing", default="blocked")),
        boundary=str(_settings_get(settings, "boundary", default="open")),
        include_zero_point=_settings_bool(settings, "include_zero_point", default=True),
        v_nn=float(_settings_get(settings, "v_nn", "V", default=0.0)),
        t_prime=float(_settings_get(settings, "t_prime", "tprime", default=0.0)),
    )
    records = _polynomial_to_term_records(built)
    poly = polynomial_from_serialized_terms(
        records,
        drop_abs_tol=float(drop_abs_tol),
        require_real_coefficients=bool(require_real_coefficients),
        coeff_imag_abs_tol=float(coeff_imag_abs_tol),
    )
    provenance = {
        "source_schema": "artifact_settings.build_problem_hamiltonian",
        "problem_key": str(problem_key),
        "num_sites": int(num_sites),
        "n_ph_max": int(n_ph_max),
        "boson_encoding": str(_settings_get(settings, "boson_encoding", default="binary")),
        "term_count_input": int(len(records)),
        "term_count_output": int(poly.count_number_terms()),
    }
    return poly, provenance


def load_polynomial_json(
    path: Path,
    *,
    drop_abs_tol: float = 1.0e-15,
    require_real_coefficients: bool = True,
    coeff_imag_abs_tol: float = 1.0e-12,
) -> tuple[PauliPolynomial, dict[str, Any]]:
    payload = _load_json(path)
    try:
        records, schema = _term_records_from_payload(payload)
    except ValueError as exc:
        if isinstance(payload, Mapping):
            try:
                fallback = _polynomial_from_artifact_settings(
                    payload,
                    drop_abs_tol=float(drop_abs_tol),
                    require_real_coefficients=bool(require_real_coefficients),
                    coeff_imag_abs_tol=float(coeff_imag_abs_tol),
                )
            except Exception as fallback_exc:
                raise ValueError(
                    "Could not locate Hamiltonian term records in JSON payload, "
                    f"and artifact-settings Hamiltonian rebuild failed: {fallback_exc}"
                ) from fallback_exc
            if fallback is not None:
                poly, provenance = fallback
                provenance["path"] = str(Path(path))
                return poly, provenance
        raise exc
    poly = polynomial_from_serialized_terms(
        records,
        drop_abs_tol=float(drop_abs_tol),
        require_real_coefficients=bool(require_real_coefficients),
        coeff_imag_abs_tol=float(coeff_imag_abs_tol),
    )
    provenance = {
        "path": str(Path(path)),
        "source_schema": schema,
        "term_count_input": int(len(records)),
        "term_count_output": int(poly.count_number_terms()),
    }
    return poly, provenance


def basis_elements_from_labels(labels: Sequence[str], *, nq: int) -> tuple[QSEBasisElement, ...]:
    return tuple(pauli_string_basis_element(label, nq=int(nq)) for label in labels)


def _identity_basis_element(*, nq: int) -> QSEBasisElement:
    return pauli_string_basis_element("e" * int(nq), nq=int(nq), name="identity")


def _basis_from_serialized_operator_records(
    records: Sequence[Mapping[str, Any]],
    *,
    nq: int,
    name_prefix: str,
) -> tuple[QSEBasisElement, ...]:
    basis: list[QSEBasisElement] = []
    for idx, record in enumerate(records):
        name = str(record.get("name", f"{name_prefix}_{idx}"))
        kind = str(record.get("kind", "pauli_polynomial"))
        metadata = _metadata_from_record(record)
        if kind == "pauli_string":
            label = _extract_label(record, index=idx)
            basis.append(pauli_string_basis_element(label, nq=int(nq), name=name, metadata=metadata))
            continue
        if kind != "pauli_polynomial":
            raise ValueError(f"Unsupported generated operator kind {kind!r} at index {idx}.")
        terms = record.get("terms")
        if not isinstance(terms, list):
            raise ValueError(f"Generated polynomial basis record {idx} requires a terms list.")
        poly = polynomial_from_serialized_terms(
            terms,
            require_real_coefficients=False,
            allow_empty_after_pruning=True,
        )
        if poly.return_polynomial()[0].nqubit() != int(nq):
            raise ValueError(
                f"Generated polynomial basis record {idx} has nq={poly.return_polynomial()[0].nqubit()}; "
                f"expected {nq}."
            )
        basis.append(polynomial_basis_element(poly, name=name, metadata=metadata))
    return tuple(basis)


def _adapt_operator_label(raw: Any) -> str:
    text = str(raw).strip()
    if "(" in text and text.endswith(")"):
        inner = text[text.rfind("(") + 1 : -1].strip()
        if inner:
            return inner
    return text


def _hamiltonian_term_basis_records(hamiltonian: PauliPolynomial) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for term in hamiltonian.return_polynomial():
        label = str(term.pw2strng())
        if set(label) <= {"e"}:
            continue
        if label in seen:
            continue
        seen.add(label)
        records.append(
            {
                "kind": "pauli_string",
                "name": f"ham::{label}",
                "pauli_exyz": label,
                "metadata": {"source": "hamiltonian_terms"},
            }
        )
    return records


def _selected_adapt_block_records(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adapt_vqe = payload.get("adapt_vqe")
    if not isinstance(adapt_vqe, Mapping):
        raise ValueError("Artifact basis source selected_adapt_blocks requires an adapt_vqe mapping.")
    parameterization = adapt_vqe.get("parameterization")
    blocks = parameterization.get("blocks") if isinstance(parameterization, Mapping) else None

    if not isinstance(blocks, list):
        operators = adapt_vqe.get("operators")
        if not isinstance(operators, list):
            raise ValueError(
                "Artifact basis source selected_adapt_blocks requires either "
                "adapt_vqe.parameterization.blocks or adapt_vqe.operators."
            )
        records: list[dict[str, Any]] = []
        seen_labels: set[str] = set()
        skipped = 0
        for idx, operator in enumerate(operators):
            label_raw = _adapt_operator_label(operator)
            try:
                label = _normalize_label(label_raw)
            except ValueError:
                skipped += 1
                continue
            if label in seen_labels:
                continue
            seen_labels.add(label)
            records.append(
                {
                    "kind": "pauli_string",
                    "name": f"adapt::{operator}",
                    "pauli_exyz": label,
                    "metadata": {
                        "source": "selected_adapt_operators",
                        "candidate_label": str(operator),
                    },
                }
            )
        if not records:
            raise ValueError("Artifact adapt_vqe.operators did not yield any valid Pauli labels.")
        meta = {
            "adapt_block_count": 0,
            "adapt_block_records_emitted": 0,
            "adapt_block_records_skipped": 0,
            "adapt_operator_count": int(len(operators)),
            "adapt_operator_records_emitted": int(len(records)),
            "adapt_operator_records_skipped": int(skipped),
        }
        return records, meta

    records: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, float, float], ...]] = set()
    skipped = 0
    for idx, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            raise TypeError(f"ADAPT parameterization block {idx} must be a mapping.")
        terms = block.get("runtime_terms_exyz")
        if not isinstance(terms, list):
            skipped += 1
            continue
        normalized_terms = []
        for term_idx, term in enumerate(terms):
            if not isinstance(term, Mapping):
                raise TypeError(f"ADAPT block {idx} term {term_idx} must be a mapping.")
            label = _extract_label(term, index=term_idx)
            coeff = _extract_coeff(term, index=term_idx)
            nq = term.get("nq")
            normalized_terms.append(
                {
                    "pauli_exyz": label,
                    "coeff_re": float(complex(coeff).real),
                    "coeff_im": float(complex(coeff).imag),
                    **({} if nq is None else {"nq": _strict_int(nq, name=f"blocks[{idx}].terms[{term_idx}].nq", min_value=0)}),
                }
            )
        poly = polynomial_from_serialized_terms(
            normalized_terms,
            require_real_coefficients=False,
            allow_empty_after_pruning=True,
        )
        signature = _polynomial_signature(poly)
        if signature in seen:
            continue
        seen.add(signature)
        records.append(
            {
                "kind": "pauli_polynomial",
                "name": f"adapt::{block.get('candidate_label', f'block_{idx}')}",
                "terms": normalized_terms,
                "metadata": {
                    "source": "selected_adapt_blocks",
                    "candidate_label": str(block.get("candidate_label", f"block_{idx}")),
                },
            }
        )

    meta = {
        "adapt_block_count": int(len(blocks)),
        "adapt_block_records_emitted": int(len(records)),
        "adapt_block_records_skipped": int(skipped),
    }
    return records, meta


def _settings_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = payload.get("settings")
    if isinstance(settings, Mapping):
        return settings
    return payload


def _artifact_int(settings: Mapping[str, Any], *keys: str, default: int | None = None) -> int:
    for key in keys:
        if key in settings:
            value = settings[key]
            if isinstance(value, bool):
                raise ValueError(f"Artifact setting {key!r} must be an integer, got {value!r}.")
            return int(value)
    if default is None:
        raise ValueError(f"Artifact settings missing required integer field from {keys!r}.")
    return int(default)


def _artifact_float(settings: Mapping[str, Any], *keys: str, default: float | None = None) -> float:
    for key in keys:
        if key in settings:
            value = float(settings[key])
            if not math.isfinite(value):
                raise ValueError(f"Artifact setting {key!r} must be finite, got {value!r}.")
            return value
    if default is None:
        raise ValueError(f"Artifact settings missing required float field from {keys!r}.")
    return float(default)


def _artifact_str(settings: Mapping[str, Any], *keys: str, default: str) -> str:
    for key in keys:
        if key in settings:
            return str(settings[key])
    return str(default)


def _artifact_bool(settings: Mapping[str, Any], *keys: str, default: bool) -> bool:
    for key in keys:
        if key in settings:
            value = settings[key]
            if isinstance(value, bool):
                return bool(value)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
    return bool(default)


def _artifact_num_particles(payload: Mapping[str, Any]) -> tuple[int, int]:
    adapt_vqe = payload.get("adapt_vqe")
    if isinstance(adapt_vqe, Mapping):
        num_particles = adapt_vqe.get("num_particles")
        if isinstance(num_particles, Mapping):
            return (
                _strict_int(num_particles.get("n_up"), name="adapt_vqe.num_particles.n_up", min_value=0),
                _strict_int(num_particles.get("n_dn"), name="adapt_vqe.num_particles.n_dn", min_value=0),
            )
    settings = _settings_mapping(payload)
    if "num_particles" in settings and isinstance(settings["num_particles"], Sequence):
        values = list(settings["num_particles"])
        if len(values) != 2:
            raise ValueError("settings.num_particles must contain exactly two values.")
        return (int(values[0]), int(values[1]))
    return (1, 1)


def _normalize_keep_classes(raw: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        items = raw.split(",")
    else:
        items = list(raw)
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item).strip()
        if value == "" or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _full_meta_records(
    payload: Mapping[str, Any],
    *,
    hamiltonian: PauliPolynomial,
    keep_classes: Sequence[str] | str | None,
    canonical_production: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from pipelines.contracts.static_provenance import (
        HHFullMetaClassFilterSpec,
        HH_FULL_META_CLASSIFIER_VERSION,
        HH_MATH_MD_FULL_META_POOL_KEY,
        classify_hh_full_meta_label,
        normalize_hh_full_meta_keep_classes,
        summarize_hh_full_meta_pool_classes,
    )
    from pipelines.qse_spectra.static_adapt_adapter import (
        build_canonical_hh_full_meta_pool_for_qse,
        build_hh_full_meta_pool_for_qse,
    )

    settings = _settings_mapping(payload)
    num_sites = _artifact_int(settings, "L", "num_sites")
    n_ph_max = _artifact_int(settings, "n_ph_max", "nph", "n_ph")
    keep_classes_t = _normalize_keep_classes(keep_classes)
    if keep_classes_t is not None:
        keep_classes_t = normalize_hh_full_meta_keep_classes(keep_classes_t)
    t_value = _artifact_float(settings, "t", "J", default=1.0)
    u_value = _artifact_float(settings, "u", "U", default=4.0)
    omega0_value = _artifact_float(settings, "omega0", default=1.0)
    g_ep_value = _artifact_float(settings, "g_ep", "g", default=0.0)
    dv_value = _artifact_float(settings, "dv", "delta_v", default=0.0)
    boson_encoding_value = _artifact_str(settings, "boson_encoding", default="binary")
    ordering_value = _artifact_str(settings, "ordering", "indexing", default="blocked")
    boundary_value = _artifact_str(settings, "boundary", default="open")
    paop_r_value = _artifact_int(settings, "paop_r", default=0)
    paop_split_value = _artifact_bool(settings, "paop_split_paulis", default=False)
    paop_prune_value = _artifact_float(settings, "paop_prune_eps", default=0.0)
    paop_norm_value = _artifact_str(settings, "paop_normalization", default="none")
    class_filter_meta = None
    legal_filter_meta = None
    if not bool(canonical_production):
        pool, meta = build_hh_full_meta_pool_for_qse(
            h_poly=hamiltonian,
            num_sites=int(num_sites),
            t=t_value,
            u=u_value,
            omega0=omega0_value,
            g_ep=g_ep_value,
            dv=dv_value,
            n_ph_max=int(n_ph_max),
            boson_encoding=boson_encoding_value,
            ordering=ordering_value,
            boundary=boundary_value,
            paop_r=paop_r_value,
            paop_split_paulis=paop_split_value,
            paop_prune_eps=paop_prune_value,
            paop_normalization=paop_norm_value,
            num_particles=_artifact_num_particles(payload),
        )
        class_counts_before = summarize_hh_full_meta_pool_classes(pool)
        if keep_classes_t is not None:
            keep_set = set(keep_classes_t)
            pool = [term for term in pool if classify_hh_full_meta_label(str(term.label)) in keep_set]
            if not pool:
                raise ValueError(f"full_meta keep_classes={sorted(keep_set)!r} removed every operator.")
        class_counts_after = summarize_hh_full_meta_pool_classes(pool)
        builder_meta = dict(meta)
        method_name = "_build_hh_full_meta_pool"
    else:
        builder_log_payloads: list[dict[str, Any]] = []

        def _capture_builder_log(event: str, **fields: Any) -> None:
            if str(event) == "hardcoded_adapt_full_meta_pool_built":
                builder_log_payloads.append(_json_safe(fields))

        full_meta_class_filter_spec = None
        if keep_classes_t is not None:
            full_meta_class_filter_spec = HHFullMetaClassFilterSpec(
                keep_classes=tuple(keep_classes_t),
                source_pool="full_meta",
                source_problem="hh",
                source_num_sites=int(num_sites),
                source_n_ph_max=int(n_ph_max),
                source_json=None,
            )
        pool_result = build_canonical_hh_full_meta_pool_for_qse(
            pool_key_hh="full_meta",
            h_poly=hamiltonian,
            num_sites=int(num_sites),
            t=t_value,
            u=u_value,
            omega0=omega0_value,
            g_ep=g_ep_value,
            dv=dv_value,
            n_ph_max=int(n_ph_max),
            boson_encoding=boson_encoding_value,
            ordering=ordering_value,
            boundary=boundary_value,
            paop_r=paop_r_value,
            paop_split_paulis=paop_split_value,
            paop_prune_eps=paop_prune_value,
            paop_normalization=paop_norm_value,
            num_particles=_artifact_num_particles(payload),
            full_meta_class_filter_spec=full_meta_class_filter_spec,
            ai_log=_capture_builder_log,
        )
        pool, method_name, class_filter_meta, _label_filter_meta, legal_filter_meta = pool_result
        if builder_log_payloads:
            builder_meta = builder_log_payloads[-1]
        else:
            builder_meta = {
                "pool_surface_key": HH_MATH_MD_FULL_META_POOL_KEY,
                "pool_display_name": (
                    legal_filter_meta.get("pool_display_name", "Math.md Full Meta")
                    if isinstance(legal_filter_meta, Mapping)
                    else "Math.md Full Meta"
                ),
                "pool_surface_source": "canonical_hh_full_meta_pool_for_qse",
                "raw_total": (
                    int(class_filter_meta.get("dedup_total_before"))
                    if isinstance(class_filter_meta, Mapping) and class_filter_meta.get("dedup_total_before") is not None
                    else int(legal_filter_meta.get("original_pool_size", len(pool)))
                    if isinstance(legal_filter_meta, Mapping)
                    else int(len(pool))
                ),
                "method_name": str(method_name),
            }
        class_counts_before = (
            dict(class_filter_meta.get("class_counts_before", {}))
            if isinstance(class_filter_meta, Mapping)
            else summarize_hh_full_meta_pool_classes(pool)
        )
        class_counts_after = summarize_hh_full_meta_pool_classes(pool)

    records: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, float, float], ...]] = set()
    for term in pool:
        signature = _polynomial_signature(term.polynomial)
        if signature in seen:
            continue
        seen.add(signature)
        full_meta_class = classify_hh_full_meta_label(str(term.label))
        records.append(
            {
                "kind": "pauli_polynomial",
                "name": str(term.label),
                "terms": _polynomial_to_term_records(term.polynomial),
                "metadata": {
                    "source": "full_meta",
                    "record_label": str(term.label),
                    "full_meta_class": full_meta_class,
                    "full_meta_classifier_version": HH_FULL_META_CLASSIFIER_VERSION,
                    "full_meta_pool_surface_key": HH_MATH_MD_FULL_META_POOL_KEY,
                },
            }
        )
    meta_out = {
        "full_meta_raw_total": int(builder_meta.get("raw_total", len(pool))),
        "full_meta_dedup_total": int(len(pool)),
        "full_meta_records_emitted": int(len(records)),
        "full_meta_keep_classes": None if keep_classes_t is None else list(keep_classes_t),
        "full_meta_class_counts_before": class_counts_before,
        "full_meta_class_counts_after": class_counts_after,
        "full_meta_builder_meta": dict(builder_meta),
    }
    if bool(canonical_production):
        meta_out.update(
            {
                "full_meta_builder_method_name": str(method_name),
                "full_meta_class_filter_meta": None if class_filter_meta is None else dict(class_filter_meta),
                "full_meta_legal_subspace_filter_meta": None if legal_filter_meta is None else dict(legal_filter_meta),
            }
        )
    return records, meta_out


def basis_elements_from_artifact_source(
    path: Path,
    *,
    nq: int,
    hamiltonian: PauliPolynomial,
    source: str,
    full_meta_keep_classes: Sequence[str] | str | None = None,
    include_hamiltonian_terms: bool = False,
    canonical_hh_full_meta: bool = False,
) -> tuple[tuple[QSEBasisElement, ...], dict[str, Any]]:
    """Build a QSE basis from an ADAPT/HH artifact without touching main pipelines."""

    source_s = str(source).strip()
    if source_s not in _ARTIFACT_BASIS_SOURCES:
        raise ValueError(f"Unsupported artifact basis source {source_s!r}; expected one of {_ARTIFACT_BASIS_SOURCES!r}.")
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise TypeError("Artifact basis source JSON must be a mapping.")

    if source_s == "selected_adapt_blocks":
        records, meta = _selected_adapt_block_records(payload)
    elif source_s == "hamiltonian_terms":
        records = _hamiltonian_term_basis_records(hamiltonian)
        meta = {
            "hamiltonian_term_basis_records_emitted": int(len(records)),
        }
    else:
        keep = full_meta_keep_classes
        if source_s == "full_meta_filtered" and keep is None:
            keep = _DEFAULT_FULL_META_FILTER_CLASSES
        records, meta = _full_meta_records(
            payload,
            hamiltonian=hamiltonian,
            keep_classes=keep,
            canonical_production=bool(canonical_hh_full_meta),
        )

    if include_hamiltonian_terms:
        records = list(records) + _hamiltonian_term_basis_records(hamiltonian)

    basis = [_identity_basis_element(nq=int(nq))]
    generated = _basis_from_serialized_operator_records(records, nq=int(nq), name_prefix=source_s)
    seen: set[tuple[str, tuple[tuple[str, float, float], ...] | str]] = {("pauli_string", "e" * int(nq))}
    dedup_generated: list[QSEBasisElement] = []
    duplicate_count = 0
    for element in generated:
        if element.kind == "pauli_string":
            sig: tuple[str, tuple[tuple[str, float, float], ...] | str] = ("pauli_string", str(element.pauli_label_exyz))
        elif element.polynomial is not None:
            sig = ("pauli_polynomial", _polynomial_signature(element.polynomial))
        else:
            raise ValueError("Generated basis element is missing polynomial data.")
        if sig in seen:
            duplicate_count += 1
            continue
        seen.add(sig)
        dedup_generated.append(element)
    basis.extend(dedup_generated)

    provenance = {
        "path": str(Path(path)),
        "source_schema": f"artifact_basis_source:{source_s}",
        "basis_size": int(len(basis)),
        "generated_record_count": int(len(records)),
        "duplicate_records_removed": int(duplicate_count),
        "include_hamiltonian_terms": bool(include_hamiltonian_terms),
        **meta,
    }
    return tuple(basis), provenance


def load_operator_basis_json(path: Path, *, nq: int) -> tuple[tuple[QSEBasisElement, ...], dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, list):
        records = payload
        schema = "top_level_list"
    elif isinstance(payload, Mapping) and isinstance(payload.get("basis"), list):
        records = payload["basis"]
        schema = "basis"
    elif isinstance(payload, Mapping) and isinstance(payload.get("operator_basis"), list):
        records = payload["operator_basis"]
        schema = "operator_basis"
    else:
        raise ValueError("Operator basis JSON must be a list or contain basis/operator_basis list.")

    basis: list[QSEBasisElement] = []
    for idx, record in enumerate(records):
        if isinstance(record, str):
            basis.append(pauli_string_basis_element(record, nq=int(nq)))
            continue
        if not isinstance(record, Mapping):
            raise TypeError(f"Operator basis record {idx} must be a string or mapping.")
        kind = str(record.get("kind", "pauli_string"))
        name = str(record.get("name", f"basis_{idx}"))
        metadata = _metadata_from_record(record)
        if kind == "pauli_string":
            label = _extract_label(record, index=idx)
            basis.append(pauli_string_basis_element(label, nq=int(nq), name=name, metadata=metadata))
        elif kind == "pauli_polynomial":
            terms = record.get("terms")
            if not isinstance(terms, list):
                raise ValueError(f"Polynomial basis record {idx} requires a terms list.")
            poly = polynomial_from_serialized_terms(
                terms,
                require_real_coefficients=False,
                allow_empty_after_pruning=True,
            )
            # Validate against the state/Hamiltonian qubit count by constructing through core helper.
            if poly.return_polynomial()[0].nqubit() != int(nq):
                raise ValueError(f"Polynomial basis record {idx} has nq={poly.return_polynomial()[0].nqubit()}; expected {nq}.")
            basis.append(polynomial_basis_element(poly, name=name, metadata=metadata))
        else:
            raise ValueError(f"Unsupported operator basis kind {kind!r} at index {idx}.")

    provenance = {
        "path": str(Path(path)),
        "source_schema": schema,
        "basis_size": int(len(basis)),
    }
    return tuple(basis), provenance


def transition_observables_from_labels(labels: Sequence[str], *, nq: int) -> tuple[QSEObservable, ...]:
    observables: list[QSEObservable] = []
    for raw in labels:
        text = str(raw).strip()
        if text == "":
            raise ValueError("transition observable labels must be non-empty.")
        if "=" in text:
            name, label = text.split("=", 1)
            name = name.strip()
            label = label.strip()
            if name == "" or label == "":
                raise ValueError(f"Invalid transition observable label specification {text!r}.")
        else:
            label = text
            name = text
        observables.append(pauli_string_observable(label, nq=int(nq), name=name, metadata={"source": "cli_label"}))
    return tuple(observables)


def _observable_from_record(record: Any, *, index: int, nq: int) -> QSEObservable:
    if isinstance(record, str):
        return transition_observables_from_labels([record], nq=int(nq))[0]
    if not isinstance(record, Mapping):
        raise TypeError(f"Transition observable record {index} must be a string or mapping.")
    kind = str(record.get("kind", "pauli_string"))
    name = str(record.get("name", f"observable_{index}"))
    metadata = _metadata_from_record(record)
    if kind == "pauli_string":
        label = _extract_label(record, index=index)
        return pauli_string_observable(label, nq=int(nq), name=name, metadata=metadata)
    if kind == "pauli_polynomial":
        terms = record.get("terms")
        if not isinstance(terms, list):
            raise ValueError(f"Transition polynomial observable record {index} requires a terms list.")
        poly = polynomial_from_serialized_terms(
            terms,
            require_real_coefficients=False,
            allow_empty_after_pruning=True,
        )
        if poly.return_polynomial()[0].nqubit() != int(nq):
            raise ValueError(f"Transition observable record {index} has nq={poly.return_polynomial()[0].nqubit()}; expected {nq}.")
        return polynomial_observable(poly, name=name, metadata=metadata)
    raise ValueError(f"Unsupported transition observable kind {kind!r} at index {index}.")


def load_transition_observables_json(path: Path, *, nq: int) -> tuple[tuple[QSEObservable, ...], dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, list):
        records = payload
        schema = "top_level_list"
    elif isinstance(payload, Mapping) and isinstance(payload.get("transition_observables"), list):
        records = payload["transition_observables"]
        schema = "transition_observables"
    elif isinstance(payload, Mapping) and isinstance(payload.get("observables"), list):
        records = payload["observables"]
        schema = "observables"
    else:
        raise ValueError("Transition observable JSON must be a list or contain transition_observables/observables list.")
    observables = tuple(_observable_from_record(record, index=idx, nq=int(nq)) for idx, record in enumerate(records))
    provenance = {
        "path": str(Path(path)),
        "source_schema": schema,
        "observable_count": int(len(observables)),
    }
    return observables, provenance


def _basis_element_to_manifest(element: QSEBasisElement, *, index: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "basis_index": int(index),
        "name": str(element.name),
        "kind": str(element.kind),
    }
    if element.kind == "pauli_string":
        out["pauli_exyz"] = str(element.pauli_label_exyz)
    elif element.polynomial is not None:
        terms = []
        for term in element.polynomial.return_polynomial():
            terms.append(
                {
                    "pauli_exyz": str(term.pw2strng()),
                    "coeff": _complex_to_json(complex(term.p_coeff)),
                    "nq": int(term.nqubit()),
                }
            )
        out["terms"] = terms
    if element.metadata is not None:
        out["metadata"] = _json_safe(element.metadata)
    return out


def _observable_to_manifest(observable: QSEObservable) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": str(observable.name),
        "kind": str(observable.kind),
    }
    if observable.kind == "pauli_string":
        out["pauli_exyz"] = str(observable.pauli_label_exyz)
    elif observable.polynomial is not None:
        out["terms"] = [
            {
                "pauli_exyz": str(term.pw2strng()),
                "coeff": _complex_to_json(complex(term.p_coeff)),
                "nq": int(term.nqubit()),
            }
            for term in observable.polynomial.return_polynomial()
        ]
    if observable.metadata is not None:
        out["metadata"] = _json_safe(observable.metadata)
    return out


def _basis_vector_policy_to_manifest(policy: QSEBasisVectorPolicy) -> dict[str, Any]:
    return {
        "reference_projection": str(policy.reference_projection),
        "basis_vector_normalization": str(policy.basis_vector_normalization),
        "sector_projection": str(policy.sector_projection),
        "sector_label": None if policy.sector_label is None else str(policy.sector_label),
    }


def _basis_vector_diagnostic_to_manifest(diagnostic: QSEBasisVectorDiagnostics) -> dict[str, Any]:
    return {
        "basis_index": int(diagnostic.basis_index),
        "name": str(diagnostic.name),
        "kind": str(diagnostic.kind),
        "reference_projection": str(diagnostic.reference_projection),
        "basis_vector_normalization": str(diagnostic.basis_vector_normalization),
        "sector_projection": str(diagnostic.sector_projection),
        "sector_label": diagnostic.sector_label,
        "raw_action_norm": float(diagnostic.raw_action_norm),
        "projected_norm": float(diagnostic.projected_norm),
        "matrix_vector_norm": float(diagnostic.matrix_vector_norm),
        "reference_overlap_before_projection": _complex_to_json(diagnostic.reference_overlap_before_projection),
        "reference_overlap_after_projection": _complex_to_json(diagnostic.reference_overlap_after_projection),
        "reference_overlap_before_projection_abs": float(diagnostic.reference_overlap_before_projection_abs),
        "reference_overlap_after_projection_abs": float(diagnostic.reference_overlap_after_projection_abs),
        "normalized_for_matrices": bool(diagnostic.normalized_for_matrices),
        "zero_vector": bool(diagnostic.zero_vector),
        "projected_out_by_q0": bool(diagnostic.projected_out_by_q0),
        "metadata": _json_safe(diagnostic.metadata),
    }


def _transition_observable_result_to_manifest(result: QSETransitionObservableResult) -> dict[str, Any]:
    return {
        "name": str(result.observable.name),
        "kind": str(result.observable.kind),
        "operator": _observable_to_manifest(result.observable),
        "observable_matrix": _matrix_to_json(result.observable_matrix),
        "transition_vector": [
            {"basis_index": int(idx), **_complex_to_json(value)}
            for idx, value in enumerate(np.asarray(result.transition_vector, dtype=complex).reshape(-1))
        ],
        "transition_amplitudes": [
            {
                "state_index": int(idx),
                "amplitude": _complex_to_json(value),
                "strength": float(result.transition_strengths[idx]),
            }
            for idx, value in enumerate(np.asarray(result.transition_amplitudes, dtype=complex).reshape(-1))
        ],
        "transition_strengths": [float(x) for x in np.asarray(result.transition_strengths, dtype=float).reshape(-1)],
        "diagnostics": {
            "observable_matrix_hermitian_residual_max_abs": result.observable_matrix_hermitian_residual_max_abs,
        },
    }


def _config_to_manifest(settings: Mapping[str, Any] | QSEPruningConfig) -> dict[str, Any]:
    if isinstance(settings, QSEPruningConfig):
        return {key: float(value) for key, value in asdict(settings).items()}
    return dict(settings)


def qse_result_to_manifest(
    result: QSEResult,
    *,
    input_provenance: Mapping[str, Any],
    settings_provenance: Mapping[str, Any] | QSEPruningConfig,
    include_matrices: bool = True,
    static_record_selection_payload: Mapping[str, Any] | None = None,
    spectral_functions_payload: Mapping[str, Any] | None = None,
    spectral_window_metrics_payload: Mapping[str, Any] | None = None,
    cutoff_boundary_diagnostics_payload: Mapping[str, Any] | None = None,
    qse_response_functions_payload: Mapping[str, Any] | None = None,
    qse_conductivity_response_payload: Mapping[str, Any] | None = None,
    qse_green_function_payload: Mapping[str, Any] | None = None,
    paper_iii_contract_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a QSE result to the sidecar's normalized JSON manifest."""

    matrices = result.matrices
    lowest_qse = float(result.eigenvalues[0]) if result.eigenvalues.size else None
    retained = set(result.retained_overlap_indices)
    eigen_records: list[dict[str, Any]] = []
    for state_idx, energy in enumerate(result.eigenvalues):
        coeffs = result.eigenvectors_basis[:, state_idx]
        eigen_records.append(
            {
                "state_index": int(state_idx),
                "energy": float(energy),
                "energy_relative_to_reference": float(float(energy) - float(matrices.reference_energy)),
                "energy_relative_to_lowest_qse": None if lowest_qse is None else float(float(energy) - lowest_qse),
                "generalized_residual_norm": float(result.generalized_residual_norms[state_idx]),
                "basis_coefficients": [
                    {
                        "basis_index": int(basis_idx),
                        **_complex_to_json(coeff),
                    }
                    for basis_idx, coeff in enumerate(coeffs)
                ],
            }
        )

    settings_manifest = _config_to_manifest(settings_provenance)
    settings_manifest["basis_vector_policy"] = _basis_vector_policy_to_manifest(matrices.basis_vector_policy)
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest: dict[str, Any] = {
        "schema_version": "qse_spectra_v1",
        "pipeline": "qse_spectra",
        "generated_utc": generated_utc,
        "backend": "ideal_statevector",
        "uses_qiskit": False,
        "input": dict(input_provenance),
        "settings": settings_manifest,
        "operator_basis": [
            _basis_element_to_manifest(element, index=idx)
            for idx, element in enumerate(matrices.basis_elements)
        ],
        "diagnostics": {
            "num_qubits": int(matrices.nq),
            "hilbert_dim": int(matrices.hilbert_dim),
            "basis_size": int(len(matrices.basis_elements)),
            "retained_rank": int(result.retained_rank),
            "discarded_rank": int(result.discarded_rank),
            "reference_energy": float(matrices.reference_energy),
            "reference_energy_imag_abs": float(matrices.reference_energy_imag_abs),
            "basis_vector_norms": [float(x) for x in matrices.basis_vector_norms],
            "basis_vector_policy": _basis_vector_policy_to_manifest(matrices.basis_vector_policy),
            "basis_action_norms": [float(x) for x in matrices.basis_action_norms],
            "basis_projected_norms": [float(x) for x in matrices.basis_projected_norms],
            "basis_matrix_vector_norms": [float(x) for x in matrices.basis_matrix_vector_norms],
            "basis_vector_diagnostics": [
                _basis_vector_diagnostic_to_manifest(diagnostic)
                for diagnostic in matrices.basis_vector_diagnostics
            ],
            "overlap_pruning_threshold": float(result.overlap_pruning_threshold),
            "overlap_condition_estimate": result.overlap_condition_estimate,
            "overlap_min_eigenvalue_raw": float(result.overlap_min_eigenvalue_raw),
            "overlap_max_eigenvalue_raw": float(result.overlap_max_eigenvalue_raw),
            "overlap_hermitian_residual_max_abs_raw": float(matrices.overlap_hermitian_residual_max_abs_raw),
            "hamiltonian_hermitian_residual_max_abs_raw": float(matrices.hamiltonian_hermitian_residual_max_abs_raw),
            "hamiltonian_coeff_imag_max_abs": float(matrices.hamiltonian_coeff_imag_max_abs),
            "solver_status": str(result.solver_status),
        },
        "overlap_spectrum": [
            {
                "index": int(idx),
                "raw_value": float(raw),
                "clamped_value": float(result.overlap_eigenvalues_clamped[idx]),
                "retained": bool(idx in retained),
            }
            for idx, raw in enumerate(result.overlap_eigenvalues_raw)
        ],
        "eigenvalues": eigen_records,
        "matrices": {
            "included": bool(include_matrices),
        },
    }
    if include_matrices:
        manifest["matrices"].update(
            {
                "overlap": _matrix_to_json(matrices.overlap),
                "hamiltonian": _matrix_to_json(matrices.hamiltonian),
            }
        )
    if result.transition_observables:
        manifest["transition_observables"] = [
            _transition_observable_result_to_manifest(item)
            for item in result.transition_observables
        ]
    if static_record_selection_payload is not None:
        manifest["static_record_selection"] = _json_safe(static_record_selection_payload)
    if spectral_functions_payload is not None:
        manifest["spectral_functions"] = _json_safe(spectral_functions_payload)
    if spectral_window_metrics_payload is not None:
        manifest["spectral_window_metrics"] = _json_safe(spectral_window_metrics_payload)
    if cutoff_boundary_diagnostics_payload is not None:
        manifest["cutoff_boundary_diagnostics"] = _json_safe(cutoff_boundary_diagnostics_payload)
    if qse_response_functions_payload is not None:
        manifest["qse_response_functions_v1"] = _json_safe(qse_response_functions_payload)
    if qse_conductivity_response_payload is not None:
        manifest["qse_conductivity_response_v1"] = _json_safe(qse_conductivity_response_payload)
    if qse_green_function_payload is not None:
        manifest["qse_green_function_v1"] = _json_safe(qse_green_function_payload)
    if paper_iii_contract_payload is not None:
        paper_iii_contract = _json_safe(paper_iii_contract_payload)
        if not isinstance(paper_iii_contract, dict):
            raise ValueError("paper_iii_contract_payload must serialize to a JSON object.")
        paper_iii_contract.setdefault("generated_utc", generated_utc)
        manifest["paper_iii_contract"] = paper_iii_contract
    return manifest


def write_manifest_json(path: Path, manifest: Mapping[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")


__all__ = [
    "statevector_from_manifest",
    "load_state_json",
    "polynomial_from_serialized_terms",
    "load_polynomial_json",
    "basis_elements_from_labels",
    "basis_elements_from_artifact_source",
    "load_operator_basis_json",
    "transition_observables_from_labels",
    "load_transition_observables_json",
    "qse_result_to_manifest",
    "write_manifest_json",
    "computational_basis_state",
]

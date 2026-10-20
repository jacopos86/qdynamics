"""Opt-in snapshot validation for real closed-shell molecular diagnostics.

This module does not change ordinary intake, select a pool or qualify a seed.
Receipts describe consumed snapshots; downstream mutable objects are not locked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np

from pipelines.contracts.problem import ProblemRequest, ResolvedProblemContext
from pipelines.static_adapt.builders.problem_registry import (
    _resolve_molecular_problem_context_from_problem,
    get_problem_family_spec,
)
from src.quantum.chemistry import molecular_hamiltonian
from src.quantum.chemistry.psi4_adapter import (
    _unwrap_restricted_closed_shell_problem_payload,
    restricted_closed_shell_problem_from_jsonable,
)


_SCHEMA = "molecular_diagnostic_intake_v1"
_ENCODING_VERSION = "length_delimited_le_float64_v1"
_BUILDER_TOL = 1.0e-12  # Existing builder policy, independent of diagnostic atol.


@dataclass(frozen=True)
class MolecularDiagnosticDeclaration:
    energy_units: str
    eri_convention: str
    orbital_basis: str
    occupied_order: str
    scalar_semantics: str
    nuclear_scalar_hartree: float
    frozen_core_scalar_hartree: float
    one_body_core_policy: str


def _real(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number without coercion.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} is not representable as float64.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    if isinstance(value, Integral) and result != int(value):
        raise ValueError(f"{label} loses integer precision in float64.")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer without coercion.")
    return int(value)


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        objects = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must have shape {shape}.") from exc
    if objects.shape != shape:
        raise ValueError(f"{label} must have shape {shape}; got {objects.shape}.")
    values = [_real(item, label) for item in objects.flat]
    return np.asarray(values, dtype=np.float64).reshape(shape)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}.")


def _payload_location(document: Any) -> str:
    if not isinstance(document, dict):
        raise ValueError("Molecular JSON must be an object.")
    candidates = [("$", document), ("$.problem", document.get("problem"))]
    chemistry = document.get("chemistry")
    candidates.append(("$.chemistry", chemistry))
    if isinstance(chemistry, dict):
        candidates.append(("$.chemistry.problem", chemistry.get("problem")))
    locations = [
        path for path, value in candidates
        if isinstance(value, dict) and "n_spatial_orbitals" in value
    ]
    if len(locations) != 1:
        raise ValueError("Require exactly one supported molecular payload location.")
    return locations[0]


def _float_bytes(value: float) -> bytes:
    # Canonicalize either signed zero; preserve all other exact float64 values.
    return struct.pack("<d", 0.0 if value == 0.0 else value)


def _tensor_bytes(array: np.ndarray) -> bytes:
    canonical = np.array(array, dtype="<f8", order="C", copy=True)
    canonical[canonical == 0.0] = 0.0
    return canonical.tobytes(order="C")


def _digest(fields: list[tuple[str, bytes]]) -> str:
    """Hash ordered UTF-8 names and values, each prefixed by uint64-BE length."""
    digest = hashlib.sha256()
    for name, value in [("encoding", _ENCODING_VERSION.encode()), *fields]:
        key = name.encode("utf-8")
        for part in (key, value):
            digest.update(struct.pack(">Q", len(part)))
            digest.update(part)
    return digest.hexdigest()


def _operator_receipt(operator: Any, nq: int, atol: float, builder: dict) -> tuple[str, float]:
    if getattr(operator, "_repr_mode", None) != "JW":
        raise ValueError("Mapped operator must use the JW representation.")
    terms = list(operator.return_polynomial())
    if not terms:
        raise ValueError("Mapped empty polynomial has no bound operator width.")
    records: list[tuple[str, int, complex]] = []
    max_imaginary = 0.0
    for position, term in enumerate(terms):
        label = str(term.pw2strng())
        if _integer(term.nqubit(), "mapped operator width") != nq or len(label) != nq or set(label) - set("exyz"):
            raise ValueError("Mapped operator width or Pauli alphabet is invalid.")
        coefficient = complex(term.p_coeff)
        if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
            raise ValueError("Mapped operator coefficients must be finite.")
        max_imaginary = max(max_imaginary, abs(coefficient.imag))
        records.append((label, position, coefficient))
    if max_imaginary > atol:
        raise ValueError("Mapped Pauli coefficients fail real Hermiticity.")
    fields = [
        ("schema", _SCHEMA.encode()),
        ("width", str(nq).encode()),
        ("word_convention", b"exyz_left_q_n_minus_1_right_q_0"),
        ("builder_name", builder["name"].encode()),
        ("builder_source_file_sha256", builder["source_file_sha256"].encode()),
        ("mapping", b"JW_blocked"),
        ("fixed_pruning_tolerance", _float_bytes(_BUILDER_TOL)),
        ("pruning_comparison", builder["pruning_comparison"].encode()),
    ]
    # Bind the actual operator's order; do not rebuild, merge or prune it.
    for label, _, coefficient in records:
        fields.append(("term", label.encode() + _float_bytes(coefficient.real) + _float_bytes(coefficient.imag)))
    return _digest(fields), max_imaginary


def resolve_molecular_diagnostic_intake(
    request: ProblemRequest,
    *,
    expected_raw_sha256: str,
    declaration: MolecularDiagnosticDeclaration,
    atol: float = 1e-12,
) -> ResolvedProblemContext:
    """Validate one consumed byte snapshot and bind the actual mapped operator.

    The HF check is an input-tensor determinant-energy witness. It establishes
    neither SCF stationarity nor mapped parity near the fixed builder cutoff.
    """
    tolerance = _real(atol, "diagnostic atol")
    if tolerance < 0:
        raise ValueError("diagnostic atol must be nonnegative.")
    if not isinstance(expected_raw_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected_raw_sha256) is None:
        raise ValueError("expected_raw_sha256 must be a 64-character SHA256 hex digest.")
    spec = get_problem_family_spec(request.problem_key)
    if spec.family_key != "molecular_restricted_closed_shell":
        raise ValueError("Diagnostic intake supports molecular_restricted_closed_shell only.")
    if request.ordering != "blocked":
        raise ValueError("Diagnostic intake requires ordering='blocked'.")
    if not request.molecular_problem_json:
        raise ValueError("Diagnostic intake requires molecular_problem_json.")
    raw = Path(request.molecular_problem_json).read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_sha256 != expected_raw_sha256.lower():
        raise ValueError("Consumed molecular JSON SHA256 mismatch.")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid UTF-8 molecular JSON.") from exc
    location = _payload_location(document)
    payload = dict(_unwrap_restricted_closed_shell_problem_payload(document))
    try:
        m = _integer(payload["n_spatial_orbitals"], "n_spatial_orbitals")
        alpha = _integer(payload["n_alpha"], "n_alpha")
        beta = _integer(payload["n_beta"], "n_beta")
        multiplicity = _integer(payload["multiplicity"], "multiplicity")
        scalar = _real(payload["nuclear_repulsion_energy"], "nuclear_repulsion_energy")
        hf = _real(payload["hf_energy"], "hf_energy")
        one = payload["one_body_integrals_mo"]
        two = payload["two_body_integrals_mo"]
    except KeyError as exc:
        raise ValueError(f"Missing strict molecular field: {exc.args[0]}.") from exc
    if m <= 0 or 2 * m > 12:
        raise ValueError("Diagnostic molecular register requires 1 <= 2m <= 12.")
    if not 0 <= alpha == beta <= m or multiplicity != 1 or str(payload.get("reference", "")).lower() != "rhf":
        raise ValueError("Require RHF, multiplicity one, and 0 <= n_alpha == n_beta <= m.")
    if _integer(request.num_sites, "request.num_sites") != m:
        raise ValueError("request.num_sites must equal n_spatial_orbitals.")
    if "n_spin_orbitals" in payload and _integer(payload["n_spin_orbitals"], "n_spin_orbitals") != 2 * m:
        raise ValueError("n_spin_orbitals must equal 2m.")
    if "charge" in payload:
        _integer(payload["charge"], "charge")
    if not isinstance(declaration, MolecularDiagnosticDeclaration):
        raise ValueError("Require a MolecularDiagnosticDeclaration.")
    declared = asdict(declaration)
    conventions = dict(energy_units="hartree", eri_convention="real_spatial_chemists", orbital_basis="orthonormal_spatial_mo", occupied_order="occupied_prefix")
    if any(declared[name] != value for name, value in conventions.items()):
        raise ValueError("Unsupported diagnostic convention declaration.")
    nuclear = _real(declaration.nuclear_scalar_hartree, "declared nuclear scalar")
    core = _real(declaration.frozen_core_scalar_hartree, "declared frozen-core scalar")
    if declaration.scalar_semantics == "nuclear_only":
        valid_core = core == 0.0 and declaration.one_body_core_policy == "none"
    elif declaration.scalar_semantics == "nuclear_plus_frozen_core":
        valid_core = declaration.one_body_core_policy == "already_contracted"
    else:
        valid_core = False
    if not valid_core or not math.isfinite(nuclear + core) or abs(nuclear + core - scalar) > tolerance:
        raise ValueError("Inconsistent scalar/core declaration; scalar is inserted once.")
    h = _array(one, (m, m), "one_body_integrals_mo")
    g = _array(two, (m, m, m, m), "two_body_integrals_mo")
    checks = {
        "h_hermitian_max_abs": float(np.max(np.abs(h - h.T))),
        "eri_first_pair_max_abs": float(np.max(np.abs(g - g.transpose(1, 0, 2, 3)))),
        "eri_second_pair_max_abs": float(np.max(np.abs(g - g.transpose(0, 1, 3, 2)))),
        "eri_pair_exchange_max_abs": float(np.max(np.abs(g - g.transpose(2, 3, 0, 1)))),
    }
    if any(value > tolerance for value in checks.values()):
        raise ValueError("Tensor Hermiticity or real chemists' ERI symmetry failed.")
    try:
        determinant = math.fsum([scalar, *(2.0 * h[i, i] for i in range(alpha)), *(2.0 * g[i, i, j, j] - g[i, j, j, i] for i in range(alpha) for j in range(alpha))])
    except (OverflowError, ValueError) as exc:
        raise ValueError("Nonfinite input-tensor determinant-energy witness.") from exc
    if not math.isfinite(determinant) or abs(determinant - hf) > tolerance:
        raise ValueError("Input-tensor determinant-energy witness disagrees with hf_energy.")
    checks.update(input_tensor_determinant_energy=determinant, hf_witness_abs_defect=abs(determinant - hf))
    # Parse the validated object; never invoke the path loader a second time.
    problem = restricted_closed_shell_problem_from_jsonable(payload)
    context = _resolve_molecular_problem_context_from_problem(spec, request, problem)
    builder = {
        "name": "build_restricted_closed_shell_molecular_hamiltonian",
        "repr_mode": "JW", "ordering": "blocked", "fixed_pruning_tolerance": _BUILDER_TOL,
        "pruning_comparison": "abs_coefficient_lte_tolerance_before_and_after_aggregation",
        "source_file_sha256": hashlib.sha256(Path(molecular_hamiltonian.__file__).read_bytes()).hexdigest(),
    }
    operator_sha256, imaginary_defect = _operator_receipt(context.hamiltonian, 2 * m, tolerance, builder)
    checks["mapped_coeff_imag_max_abs"] = imaginary_defect
    tensor_sha256 = _digest([
        ("h_shape", json.dumps(h.shape).encode()), ("h_float64", _tensor_bytes(h)),
        ("g_shape", json.dumps(g.shape).encode()), ("g_float64", _tensor_bytes(g)),
    ])
    semantic_fields = [("schema", _SCHEMA.encode()), ("tensor_sha256", tensor_sha256.encode())]
    for name, value in sorted({"m": m, "n_alpha": alpha, "n_beta": beta, "multiplicity": multiplicity, "charge": payload.get("charge", 0)}.items()):
        semantic_fields.append((name, str(value).encode()))
    for name, value in sorted({"ordering": request.ordering, "reference": problem.reference, "geometry_spec": problem.geometry_spec, "basis": problem.basis}.items()):
        semantic_fields.append((name, value.encode("utf-8")))
    for name, value in sorted(declared.items()):
        semantic_fields.append((f"declaration.{name}", _float_bytes(_real(value, name)) if name.endswith("_hartree") else value.encode("utf-8")))
    semantic_fields.extend([("scalar", _float_bytes(scalar)), ("hf_energy", _float_bytes(hf)), ("diagnostic_atol", _float_bytes(tolerance)), ("builder_name", builder["name"].encode()), ("builder_source_file_sha256", builder["source_file_sha256"].encode()), ("fixed_pruning_tolerance", _float_bytes(_BUILDER_TOL)), ("pruning_comparison", builder["pruning_comparison"].encode())])
    receipt = {
        "schema": _SCHEMA, "expected_raw_sha256": expected_raw_sha256.lower(), "raw_sha256": raw_sha256,
        "payload_location": location, "tensor_sha256": tensor_sha256,
        "semantic_sha256": _digest(semantic_fields), "operator_sha256": operator_sha256,
        "diagnostic_atol": tolerance, "builder": builder, "declarations": declared, "checks": checks,
        "encoding": {"version": _ENCODING_VERSION, "framing": "uint64_big_endian_length_for_each_UTF8_name_and_value", "tensor_dtype": "little_endian_float64", "tensor_order": "C", "signed_zero": "canonical_positive_zero", "operator_order": "actual_return_polynomial_order", "word_convention": "exyz_left_q_n_minus_1_right_q_0", "operator_width": 2 * m},
        "qualification": {"hf_check": "input_tensor_determinant_energy_witness", "assumptions": "declarations_not_physically_verified", "binding": "snapshot_only_no_downstream_enforcement", "builder_source_identity": "observed_source_file_hash"},
    }
    return replace(context, runtime_data={**(context.runtime_data or {}), "molecular_diagnostic_intake_receipt": receipt})

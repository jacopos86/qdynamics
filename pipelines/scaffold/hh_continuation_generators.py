#!/usr/bin/env python3
"""Generator metadata helpers for HH continuation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import itertools
import json
from numbers import Integral
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from pipelines.scaffold.hh_continuation_types import GeneratorMetadata, GeneratorSplitEvent
from src.quantum.hubbard_latex_python_pairs import SPIN_DN, SPIN_UP, mode_index
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


_GENERATOR_REGISTRY_CACHE_SCHEMA = "hh_generator_registry_cache_v1"
_GENERATOR_REGISTRY_CACHE_CODE_VERSION = "hh_continuation_generators_20260626_sector_guard_v1"
_GENERATOR_REGISTRY_CACHE_ENV = "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE"
_GENERATOR_REGISTRY_CACHE_DIR_ENV = "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR"
_GENERATOR_REGISTRY_CACHE_DISABLED_VALUES = {"0", "false", "no", "off", "disabled"}
_GENERATOR_REGISTRY_CACHE_MEMORY_ONLY_VALUES = {"memory", "mem", "memory_only", "in_memory"}
_GENERATOR_REGISTRY_CACHE_BYTES: dict[str, bytes] = {}


def _generator_registry_cache_mode() -> str:
    raw = os.environ.get(
        _GENERATOR_REGISTRY_CACHE_ENV,
        os.environ.get("STATIC_ADAPT_HH_POOL_CACHE", "disk"),
    )
    value = str(raw).strip().lower()
    if value in _GENERATOR_REGISTRY_CACHE_DISABLED_VALUES:
        return "off"
    if value in _GENERATOR_REGISTRY_CACHE_MEMORY_ONLY_VALUES:
        return "memory"
    return "disk"


def _generator_registry_cache_dir() -> Path:
    raw = os.environ.get(_GENERATOR_REGISTRY_CACHE_DIR_ENV)
    if raw is not None and str(raw).strip() != "":
        return Path(str(raw)).expanduser()
    return Path("raw_outputs") / "cache" / _GENERATOR_REGISTRY_CACHE_SCHEMA


def _generator_registry_cache_normalize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {
            str(k): _generator_registry_cache_normalize(value[k])
            for k in sorted(value.keys(), key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_generator_registry_cache_normalize(item) for item in value]
    if isinstance(value, set):
        return [_generator_registry_cache_normalize(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return format(float(value), ".17g")
    return str(value)


def _generator_registry_cache_key_payload(
    *,
    terms: Sequence[Any],
    family_ids: Sequence[str],
    num_sites: int,
    ordering: str,
    qpb: int,
    symmetry_specs: Sequence[Mapping[str, Any] | None],
    split_policy: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx, term in enumerate(terms):
        polynomial = term.polynomial
        rows.append(
            {
                "label": str(term.label),
                "family_id": str(family_ids[idx] if idx < len(family_ids) else "unknown"),
                "raw_term_count": int(len(list(polynomial.return_polynomial()))),
                "support_qubits": [int(q) for q in _support_qubits(polynomial)],
                "signature": _generator_registry_cache_normalize(_polynomial_signature(polynomial)),
                "serialized_terms_exyz": _generator_registry_cache_normalize(
                    _serialize_polynomial_terms(polynomial)
                ),
                "symmetry_spec": _generator_registry_cache_normalize(
                    symmetry_specs[idx] if idx < len(symmetry_specs) else None
                ),
            }
        )
    return {
        "schema": _GENERATOR_REGISTRY_CACHE_SCHEMA,
        "code_version": _GENERATOR_REGISTRY_CACHE_CODE_VERSION,
        "python_pickle_abi": f"{sys.version_info.major}.{sys.version_info.minor}",
        "num_sites": int(num_sites),
        "ordering": str(ordering),
        "qpb": int(qpb),
        "split_policy": str(split_policy),
        "terms": rows,
    }


def _generator_registry_cache_digest(key_payload: Mapping[str, Any]) -> str:
    key_json = json.dumps(key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(key_json.encode("utf-8")).hexdigest()


def _generator_registry_cache_path(digest: str) -> Path:
    return _generator_registry_cache_dir() / f"{digest}.pickle"


def _generator_registry_cache_load(
    *,
    key_payload: Mapping[str, Any],
    digest: str,
    ai_log: Callable[..., None] | None,
) -> dict[str, dict[str, Any]] | None:
    mode = _generator_registry_cache_mode()
    if mode == "off":
        return None
    cache_level = "memory"
    blob = _GENERATOR_REGISTRY_CACHE_BYTES.get(str(digest))
    cache_path: Path | None = None
    if blob is None and mode == "disk":
        cache_path = _generator_registry_cache_path(str(digest))
        try:
            if cache_path.is_file():
                blob = cache_path.read_bytes()
                _GENERATOR_REGISTRY_CACHE_BYTES[str(digest)] = blob
                cache_level = "disk"
        except Exception as exc:  # pragma: no cover - defensive cache fallback
            if callable(ai_log):
                ai_log(
                    "hardcoded_adapt_generator_registry_cache_load_failed",
                    cache_key=str(digest),
                    cache_path=str(cache_path),
                    error=str(exc),
                )
            return None
    if blob is None:
        return None
    try:
        payload = pickle.loads(blob)
    except Exception as exc:  # pragma: no cover - defensive cache fallback
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_generator_registry_cache_decode_failed",
                cache_key=str(digest),
                cache_path=(str(cache_path) if cache_path is not None else None),
                error=str(exc),
            )
        _GENERATOR_REGISTRY_CACHE_BYTES.pop(str(digest), None)
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != _GENERATOR_REGISTRY_CACHE_SCHEMA
        or payload.get("key") != dict(key_payload)
        or not isinstance(payload.get("registry"), dict)
    ):
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_generator_registry_cache_ignored",
                cache_key=str(digest),
                reason="schema_or_key_mismatch",
            )
        return None
    registry = {
        str(label): dict(meta)
        for label, meta in payload.get("registry", {}).items()
        if isinstance(meta, Mapping)
    }
    if callable(ai_log):
        ai_log(
            "hardcoded_adapt_generator_registry_cache_hit",
            cache_key=str(digest),
            cache_level=str(cache_level),
            cache_path=(
                str(cache_path)
                if cache_path is not None
                else str(_generator_registry_cache_path(str(digest))) if mode == "disk" else None
            ),
            registry_size=int(len(registry)),
        )
    return registry


def _generator_registry_cache_store(
    *,
    key_payload: Mapping[str, Any],
    digest: str,
    registry: Mapping[str, Mapping[str, Any]],
    ai_log: Callable[..., None] | None,
) -> None:
    mode = _generator_registry_cache_mode()
    if mode == "off":
        return
    payload = {
        "schema": _GENERATOR_REGISTRY_CACHE_SCHEMA,
        "key": dict(key_payload),
        "cache_key": str(digest),
        "registry": {str(label): dict(meta) for label, meta in registry.items()},
        "created_unix_s": time.time(),
    }
    try:
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        _GENERATOR_REGISTRY_CACHE_BYTES[str(digest)] = blob
    except Exception as exc:  # pragma: no cover - defensive cache fallback
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_generator_registry_cache_encode_failed",
                cache_key=str(digest),
                error=str(exc),
            )
        return
    if mode != "disk":
        return
    cache_dir = _generator_registry_cache_dir()
    cache_path = cache_dir / f"{digest}.pickle"
    tmp_path: Path | None = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=cache_dir, prefix=f".{digest}.", suffix=".tmp") as fh:
            tmp_path = Path(fh.name)
            fh.write(blob)
        os.replace(tmp_path, cache_path)
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_generator_registry_cache_stored",
                cache_key=str(digest),
                cache_path=str(cache_path),
                registry_size=int(len(registry)),
                bytes=int(len(blob)),
            )
    except Exception as exc:  # pragma: no cover - defensive cache fallback
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_generator_registry_cache_store_failed",
                cache_key=str(digest),
                cache_path=str(cache_path),
                error=str(exc),
            )


def clear_pool_generator_registry_cache_memory() -> None:
    """Clear only the in-process HH generator-registry cache layer."""
    _GENERATOR_REGISTRY_CACHE_BYTES.clear()


def _polynomial_signature(poly: Any, *, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    items: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in generator polynomial: {coeff}")
        items.append((str(term.pw2strng()), float(round(coeff.real, 12))))
    items.sort()
    return tuple(items)


def _support_qubits(poly: Any) -> list[int]:
    support: set[int] = set()
    for term in poly.return_polynomial():
        word = str(term.pw2strng())
        nq = int(term.nqubit())
        for idx, ch in enumerate(word):
            if ch == "e":
                continue
            support.add(int(nq - 1 - idx))
    return sorted(int(q) for q in support)


def _signature_from_serialized_terms(
    serialized_terms: Sequence[Mapping[str, Any]],
    *,
    tol: float = 1e-12,
) -> tuple[tuple[str, float], ...]:
    acc: dict[str, complex] = {}
    for raw in serialized_terms:
        if not isinstance(raw, Mapping):
            continue
        label = str(raw.get("pauli_exyz", ""))
        if not label:
            continue
        coeff = complex(float(raw.get("coeff_re", 0.0)), float(raw.get("coeff_im", 0.0)))
        if abs(coeff) <= float(tol):
            continue
        acc[label] = acc.get(label, 0.0 + 0.0j) + coeff
    items: list[tuple[str, float]] = []
    for label, coeff in acc.items():
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in serialized generator polynomial: {coeff}")
        items.append((str(label), float(round(coeff.real, 12))))
    items.sort()
    return tuple(items)


def _support_qubits_from_serialized_terms(serialized_terms: Sequence[Mapping[str, Any]]) -> list[int]:
    support: set[int] = set()
    for raw in serialized_terms:
        if not isinstance(raw, Mapping):
            continue
        word = str(raw.get("pauli_exyz", ""))
        nq = int(raw.get("nq", len(word)))
        if not word or nq <= 0:
            continue
        for idx, ch in enumerate(word):
            if ch == "e":
                continue
            support.add(int(nq - 1 - idx))
    return sorted(int(q) for q in support)


def _qubit_to_site(
    qubit: int,
    *,
    num_sites: int,
    ordering: str,
    qpb: int,
) -> int:
    q = int(qubit)
    n_sites = int(num_sites)
    fermion_qubits = 2 * n_sites
    if q < fermion_qubits:
        ordering_key = str(ordering).strip().lower()
        if ordering_key == "interleaved":
            return int(q // 2)
        return int(q % n_sites)
    return int((q - fermion_qubits) // int(max(1, qpb)))


def _support_sites(
    support_qubits: Sequence[int],
    *,
    num_sites: int,
    ordering: str,
    qpb: int,
) -> list[int]:
    out = {
        _qubit_to_site(int(q), num_sites=int(num_sites), ordering=str(ordering), qpb=int(qpb))
        for q in support_qubits
    }
    return sorted(int(x) for x in out)


def _relative_site_offsets(sites: Sequence[int]) -> list[int]:
    if not sites:
        return []
    site_min = min(int(x) for x in sites)
    return [int(int(x) - site_min) for x in sites]


def _template_id(
    *,
    family_id: str,
    support_site_offsets: Sequence[int],
    n_poly_terms: int,
    has_boson_support: bool,
    has_fermion_support: bool,
    is_macro_generator: bool,
) -> str:
    parts = [
        str(family_id),
        "macro" if bool(is_macro_generator) else "atomic",
        f"terms{int(n_poly_terms)}",
        f"sites{','.join(str(int(x)) for x in support_site_offsets)}",
        f"bos{int(bool(has_boson_support))}",
        f"ferm{int(bool(has_fermion_support))}",
    ]
    return "|".join(parts)


def _serialize_polynomial_terms(poly: Any, *, tol: float = 1e-12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        out.append(
            {
                "pauli_exyz": str(term.pw2strng()),
                "coeff_re": float(coeff.real),
                "coeff_im": float(coeff.imag),
                "nq": int(term.nqubit()),
            }
        )
    return out


def serialize_polynomial_terms_exyz(poly: Any, *, tol: float = 1e-12) -> list[dict[str, Any]]:
    """Public serializer for pool-contract fingerprints."""
    return _serialize_polynomial_terms(poly, tol=float(tol))


def _build_number_operator(*, nq: int, qubit: int) -> PauliPolynomial:
    z_word = ["e"] * int(nq)
    z_word[int(nq - 1 - int(qubit))] = "z"
    return PauliPolynomial(
        "JW",
        [
            PauliTerm(int(nq), ps=("e" * int(nq)), pc=0.5),
            PauliTerm(int(nq), ps=("".join(z_word)), pc=-0.5),
        ],
    )


def _fermion_number_operators(
    *,
    nq: int,
    num_sites: int,
    ordering: str,
) -> tuple[PauliPolynomial, PauliPolynomial]:
    n_up = PauliPolynomial("JW", [])
    n_dn = PauliPolynomial("JW", [])
    for site in range(int(num_sites)):
        n_up += _build_number_operator(
            nq=int(nq),
            qubit=int(mode_index(int(site), SPIN_UP, indexing=str(ordering), n_sites=int(num_sites))),
        )
        n_dn += _build_number_operator(
            nq=int(nq),
            qubit=int(mode_index(int(site), SPIN_DN, indexing=str(ordering), n_sites=int(num_sites))),
        )
    return n_up, n_dn


_PAULI_PRODUCT_TABLE: dict[tuple[str, str], tuple[str, complex]] = {
    ("e", "e"): ("e", 1.0 + 0.0j),
    ("e", "x"): ("x", 1.0 + 0.0j),
    ("e", "y"): ("y", 1.0 + 0.0j),
    ("e", "z"): ("z", 1.0 + 0.0j),
    ("x", "e"): ("x", 1.0 + 0.0j),
    ("y", "e"): ("y", 1.0 + 0.0j),
    ("z", "e"): ("z", 1.0 + 0.0j),
    ("x", "x"): ("e", 1.0 + 0.0j),
    ("y", "y"): ("e", 1.0 + 0.0j),
    ("z", "z"): ("e", 1.0 + 0.0j),
    ("x", "y"): ("z", 0.0 + 1.0j),
    ("y", "z"): ("x", 0.0 + 1.0j),
    ("z", "x"): ("y", 0.0 + 1.0j),
    ("y", "x"): ("z", 0.0 - 1.0j),
    ("z", "y"): ("x", 0.0 - 1.0j),
    ("x", "z"): ("y", 0.0 - 1.0j),
}


def _multiply_pauli_word_strings(lhs_word: str, rhs_word: str) -> tuple[str, complex]:
    lhs = str(lhs_word).lower()
    rhs = str(rhs_word).lower()
    if len(lhs) != len(rhs):
        raise ValueError("Pauli words must have identical lengths for commutator metadata.")
    out: list[str] = []
    phase = 1.0 + 0.0j
    for left, right in zip(lhs, rhs):
        try:
            char, local_phase = _PAULI_PRODUCT_TABLE[(left, right)]
        except KeyError as exc:
            raise ValueError(f"Unsupported Pauli symbols in metadata commutator: {left!r}, {right!r}") from exc
        out.append(char)
        phase *= local_phase
    return "".join(out), complex(phase)


def _commutator_l1_norm(lhs: PauliPolynomial, rhs: PauliPolynomial) -> float:
    """Return ||[lhs, rhs]||_1 without constructing a reduced PauliPolynomial."""

    acc: dict[str, complex] = {}
    lhs_terms = list(lhs.return_polynomial())
    rhs_terms = list(rhs.return_polynomial())
    for lhs_term in lhs_terms:
        lhs_word = str(lhs_term.pw2strng())
        lhs_coeff = complex(lhs_term.p_coeff)
        if abs(lhs_coeff) <= 0.0:
            continue
        for rhs_term in rhs_terms:
            rhs_word = str(rhs_term.pw2strng())
            rhs_coeff = complex(rhs_term.p_coeff)
            if abs(rhs_coeff) <= 0.0:
                continue
            lr_word, lr_phase = _multiply_pauli_word_strings(lhs_word, rhs_word)
            rl_word, rl_phase = _multiply_pauli_word_strings(rhs_word, lhs_word)
            acc[lr_word] = acc.get(lr_word, 0.0 + 0.0j) + lhs_coeff * rhs_coeff * lr_phase
            acc[rl_word] = acc.get(rl_word, 0.0 + 0.0j) - rhs_coeff * lhs_coeff * rl_phase
    return float(sum(abs(coeff) for coeff in acc.values()))


def _fixed_count_sector_basis(
    *,
    num_sites: int,
    ordering: str,
    fixed_num_particles: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    counts = tuple(int(value) for value in fixed_num_particles)
    if len(counts) != 2:
        raise ValueError("fixed_num_particles must contain (n_up, n_dn).")
    n_up_target, n_dn_target = counts
    if not (0 <= n_up_target <= int(num_sites)) or not (
        0 <= n_dn_target <= int(num_sites)
    ):
        raise ValueError(
            "fixed_num_particles must lie within the available spin orbitals."
        )
    up_qubits = tuple(
        int(mode_index(site, SPIN_UP, indexing=str(ordering), n_sites=int(num_sites)))
        for site in range(int(num_sites))
    )
    dn_qubits = tuple(
        int(mode_index(site, SPIN_DN, indexing=str(ordering), n_sites=int(num_sites)))
        for site in range(int(num_sites))
    )
    basis: list[int] = []
    for occupied_up in itertools.combinations(up_qubits, n_up_target):
        for occupied_dn in itertools.combinations(dn_qubits, n_dn_target):
            basis_index = 0
            for qubit in (*occupied_up, *occupied_dn):
                basis_index |= 1 << int(qubit)
            basis.append(int(basis_index))
    return tuple(basis), up_qubits, dn_qubits


def _apply_pauli_word_to_fermion_basis(
    word: str,
    basis_index: int,
    *,
    fermion_qubit_count: int,
) -> tuple[int, complex]:
    out_index = int(basis_index)
    phase = 1.0 + 0.0j
    for qubit in range(int(fermion_qubit_count)):
        symbol = str(word)[-1 - int(qubit)].lower()
        occupied = (int(basis_index) >> int(qubit)) & 1
        if symbol == "x":
            out_index ^= 1 << int(qubit)
        elif symbol == "y":
            out_index ^= 1 << int(qubit)
            phase *= 1.0j if int(occupied) == 0 else -1.0j
        elif symbol == "z":
            if int(occupied) == 1:
                phase *= -1.0
        elif symbol != "e":
            raise ValueError(f"Unsupported Pauli symbol {symbol!r} in sector gate.")
    return int(out_index), complex(phase)


def _fixed_count_sector_invariance_gate(
    *,
    polynomial: Any,
    num_sites: int,
    ordering: str,
    fixed_num_particles: Sequence[int],
    tol: float,
) -> dict[str, Any]:
    """Check Q O P = 0 for the requested fixed-spin-count sector."""

    basis, up_qubits, dn_qubits = _fixed_count_sector_basis(
        num_sites=int(num_sites),
        ordering=str(ordering),
        fixed_num_particles=fixed_num_particles,
    )
    n_up_target, n_dn_target = (int(value) for value in fixed_num_particles)
    fermion_qubit_count = int(2 * int(num_sites))
    particle_leak_l1 = 0.0
    spin_leak_l1 = 0.0
    leak_max_abs = 0.0
    for basis_index in basis:
        amplitudes: dict[tuple[str, int], complex] = {}
        for term in polynomial.return_polynomial():
            word = str(term.pw2strng()).lower()
            if len(word) < int(fermion_qubit_count):
                raise ValueError(
                    "Pauli word is shorter than the fermion register in sector gate."
                )
            output_index, phase = _apply_pauli_word_to_fermion_basis(
                word,
                int(basis_index),
                fermion_qubit_count=int(fermion_qubit_count),
            )
            boson_word = word[: len(word) - int(fermion_qubit_count)]
            key = (str(boson_word), int(output_index))
            amplitudes[key] = amplitudes.get(key, 0.0 + 0.0j) + (
                complex(term.p_coeff) * complex(phase)
            )
        for (_boson_word, output_index), amplitude in amplitudes.items():
            magnitude = float(abs(amplitude))
            if magnitude <= float(tol):
                continue
            output_up = sum(
                (int(output_index) >> int(qubit)) & 1 for qubit in up_qubits
            )
            output_dn = sum(
                (int(output_index) >> int(qubit)) & 1 for qubit in dn_qubits
            )
            if int(output_up + output_dn) != int(n_up_target + n_dn_target):
                particle_leak_l1 += magnitude
                leak_max_abs = max(leak_max_abs, magnitude)
            if int(output_up) != int(n_up_target) or int(output_dn) != int(
                n_dn_target
            ):
                spin_leak_l1 += magnitude
                leak_max_abs = max(leak_max_abs, magnitude)
    return {
        "gate_scope": "fixed_count_sector_invariance_v1",
        "fixed_num_particles": {
            "n_up": int(n_up_target),
            "n_dn": int(n_dn_target),
        },
        "sector_basis_dimension": int(len(basis)),
        "particle_sector_invariant": bool(particle_leak_l1 <= float(tol)),
        "spin_sector_invariant": bool(spin_leak_l1 <= float(tol)),
        "particle_sector_leakage_l1": float(particle_leak_l1),
        "spin_sector_leakage_l1": float(spin_leak_l1),
        "sector_leakage_max_abs": float(leak_max_abs),
    }


def _operator_symmetry_gate(
    *,
    polynomial: Any,
    num_sites: int,
    ordering: str,
    symmetry_spec: Mapping[str, Any] | None,
    fixed_num_particles: Sequence[int] | None = None,
    tol: float = 1e-10,
) -> dict[str, Any]:
    terms = list(polynomial.return_polynomial())
    nq = int(terms[0].nqubit()) if terms else 0
    if nq <= 0 or int(num_sites) <= 0:
        return {
            "checked": False,
            "passed": True,
            "particle_number_preserving": True,
            "spin_sector_preserving": True,
            "commutator_l1_total": 0.0,
            "commutator_l1_up": 0.0,
            "commutator_l1_dn": 0.0,
        }
    require_particle = bool(
        not isinstance(symmetry_spec, Mapping)
        or str(symmetry_spec.get("particle_number_mode", "preserving")) == "preserving"
    )
    require_spin = bool(
        not isinstance(symmetry_spec, Mapping)
        or str(symmetry_spec.get("spin_sector_mode", "preserving")) == "preserving"
    )
    n_up, n_dn = _fermion_number_operators(
        nq=int(nq),
        num_sites=int(num_sites),
        ordering=str(ordering),
    )
    comm_up = _commutator_l1_norm(n_up, polynomial)
    comm_dn = _commutator_l1_norm(n_dn, polynomial)
    comm_total = _commutator_l1_norm(n_up + n_dn, polynomial)
    global_particle_ok = bool(comm_total <= float(tol))
    global_spin_ok = bool(comm_up <= float(tol) and comm_dn <= float(tol))
    sector_gate = None
    if fixed_num_particles is not None:
        sector_gate = _fixed_count_sector_invariance_gate(
            polynomial=polynomial,
            num_sites=int(num_sites),
            ordering=str(ordering),
            fixed_num_particles=fixed_num_particles,
            tol=float(tol),
        )
    particle_preserving = bool(
        sector_gate.get("particle_sector_invariant", False)
        if isinstance(sector_gate, Mapping)
        else global_particle_ok
    )
    spin_preserving = bool(
        sector_gate.get("spin_sector_invariant", False)
        if isinstance(sector_gate, Mapping)
        else global_spin_ok
    )
    particle_ok = bool((not require_particle) or particle_preserving)
    spin_ok = bool((not require_spin) or spin_preserving)
    return {
        "checked": True,
        "passed": bool(particle_ok and spin_ok),
        "particle_number_preserving": bool(particle_preserving),
        "spin_sector_preserving": bool(spin_preserving),
        "commutator_l1_total": float(comm_total),
        "commutator_l1_up": float(comm_up),
        "commutator_l1_dn": float(comm_dn),
        "globally_particle_number_commuting": bool(global_particle_ok),
        "globally_spin_sector_commuting": bool(global_spin_ok),
        "gate_scope": (
            str(sector_gate.get("gate_scope"))
            if isinstance(sector_gate, Mapping)
            else "global_commutator_v1"
        ),
        "fixed_count_sector": (
            dict(sector_gate) if isinstance(sector_gate, Mapping) else None
        ),
        "required_particle_number": bool(require_particle),
        "required_spin_sector": bool(require_spin),
    }


def _runtime_split_requires_hard_symmetry_gate(symmetry_spec: Mapping[str, Any] | None) -> bool:
    return bool(isinstance(symmetry_spec, Mapping) and bool(symmetry_spec.get("hard_guard", False)))


def _runtime_split_should_check_symmetry_gate(symmetry_spec: Mapping[str, Any] | None) -> bool:
    if not isinstance(symmetry_spec, Mapping):
        return False
    if bool(symmetry_spec.get("hard_guard", False)):
        return True
    particle_mode = str(symmetry_spec.get("particle_number_mode", "preserving")).strip().lower()
    spin_mode = str(symmetry_spec.get("spin_sector_mode", "preserving")).strip().lower()
    return bool(particle_mode == "preserving" or spin_mode == "preserving")


def _runtime_split_skipped_symmetry_gate(symmetry_spec: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "checked": False,
        "passed": True,
        "particle_number_preserving": True,
        "spin_sector_preserving": True,
        "commutator_l1_total": 0.0,
        "commutator_l1_up": 0.0,
        "commutator_l1_dn": 0.0,
        "required_particle_number": False,
        "required_spin_sector": False,
        "skipped_reason": (
            "runtime_split_symmetry_hard_guard_off"
            if isinstance(symmetry_spec, Mapping)
            else "runtime_split_symmetry_spec_missing"
        ),
    }


def _runtime_split_required_guard_rejection(
    *,
    reason: str,
    symmetry_spec: Mapping[str, Any] | None,
    fixed_num_particles: Sequence[int] | None,
) -> dict[str, Any]:
    fixed_counts = None
    if (
        isinstance(fixed_num_particles, Sequence)
        and not isinstance(fixed_num_particles, (str, bytes))
    ):
        fixed_counts = [
            int(value) if isinstance(value, Integral) and not isinstance(value, bool) else None
            for value in fixed_num_particles
        ]
    return {
        "checked": False,
        "passed": False,
        "rejected": True,
        "rejection_reason": str(reason),
        "particle_number_preserving": False,
        "spin_sector_preserving": False,
        "commutator_l1_total": 0.0,
        "commutator_l1_up": 0.0,
        "commutator_l1_dn": 0.0,
        "commutator_evaluated": False,
        "required_particle_number": True,
        "required_spin_sector": True,
        "hard_guard_required": True,
        "hard_guard_present": bool(
            isinstance(symmetry_spec, Mapping)
            and symmetry_spec.get("hard_guard") is True
        ),
        "gate_scope": "fixed_count_sector_invariance_v1",
        "fixed_num_particles": fixed_counts,
    }


def _runtime_split_required_guard_contract_rejection_reason(
    *,
    symmetry_spec: Mapping[str, Any] | None,
    fixed_num_particles: Sequence[int] | None,
    num_sites: int,
) -> str | None:
    if symmetry_spec is None:
        return "runtime_split_required_symmetry_spec_missing"
    if not isinstance(symmetry_spec, Mapping):
        return "runtime_split_required_symmetry_spec_malformed"
    if symmetry_spec.get("hard_guard") is not True:
        return "runtime_split_required_symmetry_spec_malformed"
    particle_mode = str(symmetry_spec.get("particle_number_mode", "")).strip().lower()
    spin_mode = str(symmetry_spec.get("spin_sector_mode", "")).strip().lower()
    if particle_mode != "preserving" or spin_mode != "preserving":
        return "runtime_split_required_symmetry_spec_malformed"
    if fixed_num_particles is None:
        return "runtime_split_required_fixed_num_particles_missing"
    if not isinstance(fixed_num_particles, Sequence) or isinstance(
        fixed_num_particles, (str, bytes)
    ):
        return "runtime_split_required_fixed_num_particles_malformed"
    if len(fixed_num_particles) != 2:
        return "runtime_split_required_fixed_num_particles_malformed"
    if int(num_sites) <= 0:
        return "runtime_split_required_fixed_num_particles_malformed"
    for value in fixed_num_particles:
        if not isinstance(value, Integral) or isinstance(value, bool):
            return "runtime_split_required_fixed_num_particles_malformed"
        if not 0 <= int(value) <= int(num_sites):
            return "runtime_split_required_fixed_num_particles_malformed"
    return None


def _runtime_split_symmetry_gate(
    *,
    polynomial: Any,
    num_sites: int,
    ordering: str,
    symmetry_spec: Mapping[str, Any] | None,
    fixed_num_particles: Sequence[int] | None = None,
    hard_guard_required: bool = False,
    tol: float = 1e-10,
) -> dict[str, Any]:
    if bool(hard_guard_required):
        rejection_reason = _runtime_split_required_guard_contract_rejection_reason(
            symmetry_spec=symmetry_spec,
            fixed_num_particles=fixed_num_particles,
            num_sites=int(num_sites),
        )
        if rejection_reason is not None:
            return _runtime_split_required_guard_rejection(
                reason=str(rejection_reason),
                symmetry_spec=symmetry_spec,
                fixed_num_particles=fixed_num_particles,
            )
    if not _runtime_split_should_check_symmetry_gate(symmetry_spec):
        return _runtime_split_skipped_symmetry_gate(symmetry_spec)
    gate = _operator_symmetry_gate(
        polynomial=polynomial,
        num_sites=int(num_sites),
        ordering=str(ordering),
        symmetry_spec=symmetry_spec,
        fixed_num_particles=fixed_num_particles,
        tol=float(tol),
    )
    if not bool(hard_guard_required):
        return gate
    if not bool(gate.get("checked", False)):
        return _runtime_split_required_guard_rejection(
            reason="runtime_split_required_symmetry_check_not_executed",
            symmetry_spec=symmetry_spec,
            fixed_num_particles=fixed_num_particles,
        )
    out = dict(gate)
    out["hard_guard_required"] = True
    out["hard_guard_present"] = True
    out["rejected"] = bool(not bool(out.get("passed", False)))
    out["rejection_reason"] = (
        "runtime_split_fixed_count_sector_violation"
        if bool(out["rejected"])
        else None
    )
    return out


def _symmetry_spec_with_gate(
    *,
    base_spec: Mapping[str, Any] | None,
    gate: Mapping[str, Any],
    checked_tag: str,
    rejected_tag: str,
) -> dict[str, Any] | None:
    if not isinstance(base_spec, Mapping):
        return None
    out = dict(base_spec)
    raw_tags = out.get("tags", [])
    tags = (
        [str(tag) for tag in raw_tags]
        if isinstance(raw_tags, Sequence) and not isinstance(raw_tags, (str, bytes))
        else []
    )
    if str(checked_tag) not in tags:
        tags.append(str(checked_tag))
    particle_ok = bool(gate.get("particle_number_preserving", True))
    spin_ok = bool(gate.get("spin_sector_preserving", True))
    out["particle_number_mode"] = "preserving" if particle_ok else "violating"
    out["spin_sector_mode"] = "preserving" if spin_ok else "violating"
    if not bool(gate.get("passed", True)):
        out["leakage_risk"] = float(max(float(out.get("leakage_risk", 0.0)), 1.0))
        out["hard_guard"] = True
        if str(rejected_tag) not in tags:
            tags.append(str(rejected_tag))
    else:
        out["leakage_risk"] = 0.0
    out["tags"] = tags
    return out


def _symmetry_spec_with_runtime_gate(
    *,
    base_spec: Mapping[str, Any] | None,
    gate: Mapping[str, Any],
) -> dict[str, Any] | None:
    return _symmetry_spec_with_gate(
        base_spec=base_spec,
        gate=gate,
        checked_tag="runtime_split_checked",
        rejected_tag="runtime_split_rejected",
    )


def rebuild_polynomial_from_serialized_terms(
    serialized_terms: Sequence[Mapping[str, Any]],
    *,
    drop_abs_tol: float = 1.0e-7,
) -> PauliPolynomial:
    nq_expected: int | None = None
    coeffs_by_label: dict[str, complex] = {}
    label_order: list[str] = []
    for raw in serialized_terms:
        if not isinstance(raw, Mapping):
            continue
        nq = int(raw.get("nq", 0))
        label = str(raw.get("pauli_exyz", ""))
        coeff = complex(float(raw.get("coeff_re", 0.0)), float(raw.get("coeff_im", 0.0)))
        if nq <= 0 or label == "":
            continue
        if nq_expected is None:
            nq_expected = int(nq)
        elif int(nq) != int(nq_expected):
            raise ValueError("Serialized runtime-split terms use inconsistent nq values.")
        if label not in coeffs_by_label:
            label_order.append(label)
            coeffs_by_label[label] = complex(0.0)
        coeffs_by_label[label] += coeff
    if nq_expected is None or not label_order:
        raise ValueError("Serialized runtime-split terms are missing or invalid.")

    poly = PauliPolynomial("JW")
    for label in label_order:
        coeff = complex(coeffs_by_label[label])
        if abs(coeff) < float(drop_abs_tol):
            continue
        poly.add_term(PauliTerm(int(nq_expected), ps=label, pc=coeff))
    if int(poly.count_number_terms()) <= 0:
        raise ValueError("Serialized runtime-split terms cancel below tolerance.")
    return poly


def build_generator_metadata(
    *,
    label: str,
    polynomial: Any,
    family_id: str,
    num_sites: int,
    ordering: str,
    qpb: int,
    split_policy: str = "preserve",
    parent_generator_id: str | None = None,
    symmetry_spec: Mapping[str, Any] | None = None,
    fixed_num_particles: Sequence[int] | None = None,
    serialized_terms: Sequence[Mapping[str, Any]] | None = None,
    signature: tuple[tuple[str, float], ...] | None = None,
) -> GeneratorMetadata:
    serialized_terms_list = (
        [dict(term) for term in serialized_terms if isinstance(term, Mapping)]
        if serialized_terms is not None
        else None
    )
    signature = (
        tuple(signature)
        if signature is not None
        else (
            _signature_from_serialized_terms(serialized_terms_list)
            if serialized_terms_list is not None
            else _polynomial_signature(polynomial)
        )
    )
    support_qubits = (
        _support_qubits_from_serialized_terms(serialized_terms_list)
        if serialized_terms_list is not None
        else _support_qubits(polynomial)
    )
    support_sites = _support_sites(
        support_qubits,
        num_sites=int(num_sites),
        ordering=str(ordering),
        qpb=int(qpb),
    )
    support_site_offsets = _relative_site_offsets(support_sites)
    has_fermion_support = any(int(q) < 2 * int(num_sites) for q in support_qubits)
    has_boson_support = any(int(q) >= 2 * int(num_sites) for q in support_qubits)
    n_poly_terms = int(len(serialized_terms_list)) if serialized_terms_list is not None else int(len(list(polynomial.return_polynomial())))
    is_macro = bool(
        n_poly_terms > 1
        and str(split_policy)
        not in {"deliberate_split", "runtime_split_projected_child"}
    )
    template_id = _template_id(
        family_id=str(family_id),
        support_site_offsets=support_site_offsets,
        n_poly_terms=int(n_poly_terms),
        has_boson_support=bool(has_boson_support),
        has_fermion_support=bool(has_fermion_support),
        is_macro_generator=bool(is_macro),
    )
    digest = hashlib.sha1(
        (
            f"{family_id}|{template_id}|{signature}|{split_policy}|{parent_generator_id or ''}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    compile_metadata: dict[str, Any] = {
        "num_polynomial_terms": int(n_poly_terms),
        "signature_size": int(len(signature)),
        "has_boson_support": bool(has_boson_support),
        "has_fermion_support": bool(has_fermion_support),
        "support_size": int(len(support_qubits)),
        "serialized_terms_exyz": (
            [dict(term) for term in serialized_terms_list]
            if serialized_terms_list is not None
            else _serialize_polynomial_terms(polynomial)
        ),
    }
    symmetry_spec_out = (dict(symmetry_spec) if isinstance(symmetry_spec, Mapping) else None)
    if symmetry_spec_out is not None:
        symmetry_gate = _operator_symmetry_gate(
            polynomial=polynomial,
            num_sites=int(num_sites),
            ordering=str(ordering),
            symmetry_spec=symmetry_spec_out,
            fixed_num_particles=fixed_num_particles,
        )
        compile_metadata["symmetry_intent"] = dict(symmetry_spec_out)
        compile_metadata["symmetry_gate"] = dict(symmetry_gate)
        symmetry_spec_out = _symmetry_spec_with_gate(
            base_spec=symmetry_spec_out,
            gate=symmetry_gate,
            checked_tag="operator_symmetry_checked",
            rejected_tag="operator_symmetry_rejected",
        )
    return GeneratorMetadata(
        generator_id=f"gen:{digest}",
        family_id=str(family_id),
        template_id=str(template_id),
        candidate_label=str(label),
        support_qubits=[int(x) for x in support_qubits],
        support_sites=[int(x) for x in support_sites],
        support_site_offsets=[int(x) for x in support_site_offsets],
        is_macro_generator=bool(is_macro),
        split_policy=str(split_policy),
        parent_generator_id=(str(parent_generator_id) if parent_generator_id is not None else None),
        symmetry_spec=symmetry_spec_out,
        compile_metadata=compile_metadata,
    )


def build_pool_generator_registry(
    *,
    terms: Sequence[Any],
    family_ids: Sequence[str],
    num_sites: int,
    ordering: str,
    qpb: int,
    symmetry_specs: Sequence[Mapping[str, Any] | None] | None = None,
    split_policy: str = "preserve",
    ai_log: Callable[..., None] | None = None,
) -> dict[str, dict[str, Any]]:
    terms_list = list(terms)
    family_ids_list = [str(x) for x in list(family_ids)]
    sym_specs = list(symmetry_specs) if symmetry_specs is not None else [None] * len(terms_list)

    cache_key_payload: dict[str, Any] | None = None
    cache_digest: str | None = None
    if _generator_registry_cache_mode() != "off":
        cache_key_payload = _generator_registry_cache_key_payload(
            terms=terms_list,
            family_ids=family_ids_list,
            num_sites=int(num_sites),
            ordering=str(ordering),
            qpb=int(qpb),
            symmetry_specs=sym_specs,
            split_policy=str(split_policy),
        )
        cache_digest = _generator_registry_cache_digest(cache_key_payload)
        cached_registry = _generator_registry_cache_load(
            key_payload=cache_key_payload,
            digest=cache_digest,
            ai_log=ai_log,
        )
        if cached_registry is not None:
            return dict(cached_registry)

    registry: dict[str, dict[str, Any]] = {}
    for idx, term in enumerate(terms_list):
        family_id = str(family_ids_list[idx] if idx < len(family_ids_list) else "unknown")
        symmetry_spec = sym_specs[idx] if idx < len(sym_specs) else None
        meta = build_generator_metadata(
            label=str(term.label),
            polynomial=term.polynomial,
            family_id=str(family_id),
            num_sites=int(num_sites),
            ordering=str(ordering),
            qpb=int(qpb),
            split_policy=str(split_policy),
            symmetry_spec=symmetry_spec,
        )
        registry[str(term.label)] = asdict(meta)

    if cache_key_payload is not None and cache_digest is not None:
        _generator_registry_cache_store(
            key_payload=cache_key_payload,
            digest=cache_digest,
            registry=registry,
            ai_log=ai_log,
        )
    return registry


def build_runtime_split_children(
    *,
    parent_label: str,
    polynomial: Any,
    family_id: str,
    num_sites: int,
    ordering: str,
    qpb: int,
    split_mode: str,
    parent_generator_metadata: Mapping[str, Any] | None = None,
    symmetry_spec: Mapping[str, Any] | None = None,
    fixed_num_particles: Sequence[int] | None = None,
    hard_guard_required: bool = False,
    include_unsplit_singleton: bool = False,
    max_children: int | None = None,
    tol: float = 1e-12,
) -> list[dict[str, Any]]:
    serialized = _serialize_polynomial_terms(polynomial, tol=float(tol))
    total_children = int(len(serialized))
    if total_children <= 0 or (
        total_children == 1 and not bool(include_unsplit_singleton)
    ):
        return []
    out: list[dict[str, Any]] = []
    parent_generator_id = None
    if isinstance(parent_generator_metadata, Mapping) and parent_generator_metadata.get("generator_id") is not None:
        parent_generator_id = str(parent_generator_metadata.get("generator_id"))
    child_limit = total_children if max_children is None or int(max_children) <= 0 else min(total_children, int(max_children))
    for child_index, term_info in enumerate(serialized[:child_limit]):
        child_poly = rebuild_polynomial_from_serialized_terms(
            [term_info],
            drop_abs_tol=float(tol),
        )
        symmetry_gate = _runtime_split_symmetry_gate(
            polynomial=child_poly,
            num_sites=int(num_sites),
            ordering=str(ordering),
            symmetry_spec=symmetry_spec,
            fixed_num_particles=fixed_num_particles,
            hard_guard_required=bool(hard_guard_required),
        )
        child_label = (
            f"{str(parent_label)}::split[{int(child_index)}]::{str(term_info.get('pauli_exyz', ''))}"
        )
        child_symmetry_spec = _symmetry_spec_with_runtime_gate(
            base_spec=symmetry_spec,
            gate=symmetry_gate,
        )
        child_meta = asdict(
            build_generator_metadata(
                label=str(child_label),
                polynomial=child_poly,
                family_id=str(family_id),
                num_sites=int(num_sites),
                ordering=str(ordering),
                qpb=int(qpb),
                split_policy="deliberate_split",
                parent_generator_id=parent_generator_id,
                symmetry_spec=(child_symmetry_spec if bool(symmetry_gate.get("checked", False)) else None),
                fixed_num_particles=fixed_num_particles,
                serialized_terms=[term_info],
            )
        )
        if child_symmetry_spec is not None and not bool(symmetry_gate.get("checked", False)):
            child_meta["symmetry_spec"] = dict(child_symmetry_spec)
        compile_metadata = dict(child_meta.get("compile_metadata", {}))
        compile_metadata["runtime_split"] = {
            "mode": str(split_mode),
            "parent_label": str(parent_label),
            "child_index": int(child_index),
            "child_count": int(total_children),
            "representation": "child_atom",
            "symmetry_gate": dict(symmetry_gate),
        }
        compile_metadata["serialized_terms_exyz"] = [dict(term_info)]
        child_meta["compile_metadata"] = compile_metadata
        out.append(
            {
                "child_label": str(child_label),
                "child_polynomial": child_poly,
                "child_generator_metadata": dict(child_meta),
                "child_index": int(child_index),
                "child_count": int(total_children),
                "parent_label": str(parent_label),
                "symmetry_gate": dict(symmetry_gate),
            }
        )
    return out


def normalize_runtime_split_subset_sizes(
    subset_sizes: Sequence[int] | str | int | None,
    *,
    legacy_max_subset_size: int | None = None,
) -> tuple[int, ...]:
    """Resolve exact Pauli-word subset cardinalities with a legacy max bridge."""

    if subset_sizes is None:
        if legacy_max_subset_size is None:
            return (1,)
        cap = int(legacy_max_subset_size)
        if cap < 1:
            raise ValueError("legacy max subset size must be >= 1.")
        return tuple(range(1, cap + 1))
    if isinstance(subset_sizes, str):
        tokens = [token.strip() for token in subset_sizes.split(",") if token.strip()]
        if not tokens:
            raise ValueError("subset_sizes must contain at least one positive integer.")
        raw_sizes = [int(token) for token in tokens]
    elif isinstance(subset_sizes, int):
        raw_sizes = [int(subset_sizes)]
    else:
        raw_sizes = [int(value) for value in subset_sizes]
    if not raw_sizes or any(size < 1 for size in raw_sizes):
        raise ValueError("subset_sizes must contain only positive integers.")
    return tuple(sorted(set(raw_sizes)))


def build_runtime_split_child_sets(
    *,
    parent_label: str,
    family_id: str,
    num_sites: int,
    ordering: str,
    qpb: int,
    split_mode: str,
    children: Sequence[Mapping[str, Any]],
    parent_generator_metadata: Mapping[str, Any] | None = None,
    symmetry_spec: Mapping[str, Any] | None = None,
    fixed_num_particles: Sequence[int] | None = None,
    hard_guard_required: bool = False,
    subset_sizes: Sequence[int] | str | int | None = None,
    max_subset_size: int | None = None,
    tol: float = 1e-12,
) -> list[dict[str, Any]]:
    parent_generator_id = None
    if isinstance(parent_generator_metadata, Mapping) and parent_generator_metadata.get("generator_id") is not None:
        parent_generator_id = str(parent_generator_metadata.get("generator_id"))
    parent_signature = None
    if isinstance(parent_generator_metadata, Mapping):
        compile_meta = parent_generator_metadata.get("compile_metadata")
        if isinstance(compile_meta, Mapping):
            serialized_parent = compile_meta.get("serialized_terms_exyz")
            if isinstance(serialized_parent, Sequence):
                try:
                    parent_signature = _polynomial_signature(
                        rebuild_polynomial_from_serialized_terms(
                            serialized_parent,
                            drop_abs_tol=float(tol),
                        ),
                        tol=float(tol),
                    )
                except Exception:
                    parent_signature = None
    child_rows = [dict(row) for row in children if isinstance(row, Mapping)]
    if len(child_rows) <= 1:
        return []
    if bool(hard_guard_required):
        child_rows = [
            row
            for row in child_rows
            if isinstance(row.get("symmetry_gate"), Mapping)
            and bool(row["symmetry_gate"].get("checked", False))
            and bool(row["symmetry_gate"].get("passed", False))
        ]
    if not child_rows:
        return []
    requested_subset_sizes = normalize_runtime_split_subset_sizes(
        subset_sizes,
        legacy_max_subset_size=max_subset_size,
    )
    out: list[dict[str, Any]] = []
    seen_signatures: set[tuple[tuple[str, float], ...]] = set()
    for subset_size in requested_subset_sizes:
        if int(subset_size) > len(child_rows):
            continue
        for subset in itertools.combinations(child_rows, subset_size):
            serialized_subset: list[dict[str, Any]] = []
            child_labels: list[str] = []
            child_indices: list[int] = []
            child_generator_ids: list[str] = []
            for child in subset:
                child_labels.append(str(child.get("child_label")))
                if child.get("child_index") is not None:
                    child_indices.append(int(child.get("child_index")))
                child_meta = child.get("child_generator_metadata")
                if isinstance(child_meta, Mapping) and child_meta.get("generator_id") is not None:
                    child_generator_ids.append(str(child_meta.get("generator_id")))
                compile_meta = child_meta.get("compile_metadata", {}) if isinstance(child_meta, Mapping) else {}
                serialized_terms = compile_meta.get("serialized_terms_exyz", []) if isinstance(compile_meta, Mapping) else []
                for term_info in serialized_terms:
                    if isinstance(term_info, Mapping):
                        serialized_subset.append(dict(term_info))
            if not serialized_subset:
                continue
            subset_poly = rebuild_polynomial_from_serialized_terms(
                serialized_subset,
                drop_abs_tol=float(tol),
            )
            subset_signature = _polynomial_signature(subset_poly, tol=float(tol))
            if parent_signature is not None and subset_signature == parent_signature:
                continue
            if subset_signature in seen_signatures:
                continue
            symmetry_gate = _runtime_split_symmetry_gate(
                polynomial=subset_poly,
                num_sites=int(num_sites),
                ordering=str(ordering),
                symmetry_spec=symmetry_spec,
                fixed_num_particles=fixed_num_particles,
                hard_guard_required=bool(hard_guard_required),
            )
            if not bool(symmetry_gate.get("passed", True)):
                continue
            child_component_gates: list[dict[str, Any] | None] = []
            for child in subset:
                raw_gate = child.get("symmetry_gate")
                child_component_gates.append(dict(raw_gate) if isinstance(raw_gate, Mapping) else None)
            termwise_child_gates_all_passed = bool(
                child_component_gates
                and all(
                    isinstance(gate, Mapping)
                    and bool(gate.get("checked", False))
                    and bool(gate.get("passed", True))
                    for gate in child_component_gates
                )
            )
            recommended_execution_mode = (
                "termwise_product" if termwise_child_gates_all_passed else "grouped_exact"
            )
            child_index_tag = ",".join(str(int(idx)) for idx in child_indices)
            subset_label = f"{str(parent_label)}::child_set[{child_index_tag}]"
            subset_symmetry_spec = _symmetry_spec_with_runtime_gate(
                base_spec=symmetry_spec,
                gate=symmetry_gate,
            )
            subset_meta = asdict(
                build_generator_metadata(
                    label=str(subset_label),
                    polynomial=subset_poly,
                    family_id=str(family_id),
                    num_sites=int(num_sites),
                    ordering=str(ordering),
                    qpb=int(qpb),
                    split_policy="runtime_split_child_set",
                    parent_generator_id=parent_generator_id,
                    symmetry_spec=(subset_symmetry_spec if bool(symmetry_gate.get("checked", False)) else None),
                    fixed_num_particles=fixed_num_particles,
                    serialized_terms=serialized_subset,
                    signature=subset_signature,
                )
            )
            if subset_symmetry_spec is not None and not bool(symmetry_gate.get("checked", False)):
                subset_meta["symmetry_spec"] = dict(subset_symmetry_spec)
            compile_metadata = dict(subset_meta.get("compile_metadata", {}))
            compile_metadata["runtime_split"] = {
                "mode": str(split_mode),
                "parent_label": str(parent_label),
                "child_indices": [int(idx) for idx in child_indices],
                "child_labels": [str(label) for label in child_labels],
                "child_generator_ids": [str(x) for x in child_generator_ids],
                "child_count": int(len(child_rows)),
                "representation": "child_set",
                "subset_cardinality": int(subset_size),
                "requested_subset_sizes": [int(size) for size in requested_subset_sizes],
                "symmetry_gate": dict(symmetry_gate),
                "termwise_child_gates_all_passed": bool(termwise_child_gates_all_passed),
                "recommended_execution_mode": str(recommended_execution_mode),
            }
            compile_metadata["serialized_terms_exyz"] = [dict(term) for term in serialized_subset]
            subset_meta["compile_metadata"] = compile_metadata
            out.append(
                {
                    "candidate_label": str(subset_label),
                    "candidate_polynomial": subset_poly,
                    "candidate_generator_metadata": dict(subset_meta),
                    "child_indices": [int(idx) for idx in child_indices],
                    "child_labels": [str(label) for label in child_labels],
                    "child_generator_ids": [str(x) for x in child_generator_ids],
                    "subset_cardinality": int(subset_size),
                    "requested_subset_sizes": [int(size) for size in requested_subset_sizes],
                    "symmetry_gate": dict(symmetry_gate),
                    "recommended_execution_mode": str(recommended_execution_mode),
                }
            )
            seen_signatures.add(subset_signature)
    return out


def selected_generator_metadata_for_labels(
    labels: Sequence[str],
    registry: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in labels:
        meta = registry.get(str(label))
        if isinstance(meta, Mapping):
            out.append(dict(meta))
    return out


def build_split_event(
    *,
    parent_generator_id: str,
    child_generator_ids: Sequence[str],
    reason: str,
    split_mode: str,
    probe_trigger: str | None = None,
    choice_reason: str | None = None,
    parent_score: float | None = None,
    child_scores: Mapping[str, float] | None = None,
    admissible_child_subsets: Sequence[Sequence[str]] | None = None,
    chosen_representation: str = "parent",
    chosen_child_ids: Sequence[str] | None = None,
    split_margin: float | None = None,
    symmetry_gate_results: Mapping[str, Any] | None = None,
    parent_collapse_diagnostic: Mapping[str, Any] | None = None,
    compiled_cost_parent: float | None = None,
    compiled_cost_children: float | None = None,
    insertion_positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    event = GeneratorSplitEvent(
        parent_generator_id=str(parent_generator_id),
        child_generator_ids=[str(x) for x in child_generator_ids],
        reason=str(reason),
        split_mode=str(split_mode),
        probe_trigger=(str(probe_trigger) if probe_trigger is not None else None),
        choice_reason=(str(choice_reason) if choice_reason is not None else None),
        parent_score=(float(parent_score) if parent_score is not None else None),
        child_scores=(
            {str(key): float(val) for key, val in child_scores.items()}
            if isinstance(child_scores, Mapping)
            else {}
        ),
        admissible_child_subsets=(
            [[str(x) for x in subset] for subset in admissible_child_subsets]
            if admissible_child_subsets is not None
            else []
        ),
        chosen_representation=str(chosen_representation),
        chosen_child_ids=([str(x) for x in chosen_child_ids] if chosen_child_ids is not None else []),
        split_margin=(float(split_margin) if split_margin is not None else None),
        symmetry_gate_results=(
            dict(symmetry_gate_results) if isinstance(symmetry_gate_results, Mapping) else {}
        ),
        parent_collapse_diagnostic=(
            dict(parent_collapse_diagnostic)
            if isinstance(parent_collapse_diagnostic, Mapping)
            else {}
        ),
        compiled_cost_parent=(float(compiled_cost_parent) if compiled_cost_parent is not None else None),
        compiled_cost_children=(float(compiled_cost_children) if compiled_cost_children is not None else None),
        insertion_positions=([int(x) for x in insertion_positions] if insertion_positions is not None else []),
    )
    return asdict(event)

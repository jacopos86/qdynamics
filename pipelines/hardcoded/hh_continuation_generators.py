#!/usr/bin/env python3
"""Generator metadata helpers for HH continuation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any, Mapping, Sequence

from pipelines.hardcoded.hh_continuation_types import GeneratorMetadata, GeneratorSplitEvent
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


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


def rebuild_polynomial_from_serialized_terms(
    serialized_terms: Sequence[Mapping[str, Any]],
) -> PauliPolynomial:
    pauli_terms: list[PauliTerm] = []
    nq_expected: int | None = None
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
        pauli_terms.append(PauliTerm(int(nq), ps=label, pc=coeff))
    if nq_expected is None or not pauli_terms:
        raise ValueError("Serialized runtime-split terms are missing or invalid.")
    return PauliPolynomial("JW", list(pauli_terms))


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
) -> GeneratorMetadata:
    signature = _polynomial_signature(polynomial)
    support_qubits = _support_qubits(polynomial)
    support_sites = _support_sites(
        support_qubits,
        num_sites=int(num_sites),
        ordering=str(ordering),
        qpb=int(qpb),
    )
    support_site_offsets = _relative_site_offsets(support_sites)
    has_fermion_support = any(int(q) < 2 * int(num_sites) for q in support_qubits)
    has_boson_support = any(int(q) >= 2 * int(num_sites) for q in support_qubits)
    n_poly_terms = int(len(list(polynomial.return_polynomial())))
    is_macro = bool(n_poly_terms > 1 and str(split_policy) != "deliberate_split")
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
        symmetry_spec=(dict(symmetry_spec) if isinstance(symmetry_spec, Mapping) else None),
        compile_metadata={
            "num_polynomial_terms": int(n_poly_terms),
            "signature_size": int(len(signature)),
            "has_boson_support": bool(has_boson_support),
            "has_fermion_support": bool(has_fermion_support),
            "support_size": int(len(support_qubits)),
            "serialized_terms_exyz": _serialize_polynomial_terms(polynomial),
        },
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
) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    sym_specs = list(symmetry_specs) if symmetry_specs is not None else [None] * len(list(terms))
    for idx, term in enumerate(terms):
        family_id = str(family_ids[idx] if idx < len(family_ids) else "unknown")
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
    max_children: int | None = None,
    tol: float = 1e-12,
) -> list[dict[str, Any]]:
    serialized = _serialize_polynomial_terms(polynomial, tol=float(tol))
    total_children = int(len(serialized))
    if total_children <= 1:
        return []
    out: list[dict[str, Any]] = []
    parent_generator_id = None
    if isinstance(parent_generator_metadata, Mapping) and parent_generator_metadata.get("generator_id") is not None:
        parent_generator_id = str(parent_generator_metadata.get("generator_id"))
    child_limit = total_children if max_children is None or int(max_children) <= 0 else min(total_children, int(max_children))
    for child_index, term_info in enumerate(serialized[:child_limit]):
        child_poly = rebuild_polynomial_from_serialized_terms([term_info])
        child_label = (
            f"{str(parent_label)}::split[{int(child_index)}]::{str(term_info.get('pauli_exyz', ''))}"
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
                symmetry_spec=symmetry_spec,
            )
        )
        compile_metadata = dict(child_meta.get("compile_metadata", {}))
        compile_metadata["runtime_split"] = {
            "mode": str(split_mode),
            "parent_label": str(parent_label),
            "child_index": int(child_index),
            "child_count": int(total_children),
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
            }
        )
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
) -> dict[str, Any]:
    event = GeneratorSplitEvent(
        parent_generator_id=str(parent_generator_id),
        child_generator_ids=[str(x) for x in child_generator_ids],
        reason=str(reason),
        split_mode=str(split_mode),
    )
    return asdict(event)

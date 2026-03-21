#!/usr/bin/env python3
"""Generator metadata helpers for HH continuation."""

from __future__ import annotations

from dataclasses import asdict
import itertools
import hashlib
from typing import Any, Mapping, Sequence

from pipelines.hardcoded.hh_continuation_types import GeneratorMetadata, GeneratorSplitEvent
from src.quantum.hubbard_latex_python_pairs import SPIN_DN, SPIN_UP, mode_index
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


def _commutator_l1_norm(lhs: PauliPolynomial, rhs: PauliPolynomial) -> float:
    comm = lhs * rhs - rhs * lhs
    return float(sum(abs(complex(term.p_coeff)) for term in comm.return_polynomial()))


def _operator_symmetry_gate(
    *,
    polynomial: Any,
    num_sites: int,
    ordering: str,
    symmetry_spec: Mapping[str, Any] | None,
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
    particle_ok = bool((not require_particle) or comm_total <= float(tol))
    spin_ok = bool((not require_spin) or (comm_up <= float(tol) and comm_dn <= float(tol)))
    return {
        "checked": True,
        "passed": bool(particle_ok and spin_ok),
        "particle_number_preserving": bool(comm_total <= float(tol)),
        "spin_sector_preserving": bool(comm_up <= float(tol) and comm_dn <= float(tol)),
        "commutator_l1_total": float(comm_total),
        "commutator_l1_up": float(comm_up),
        "commutator_l1_dn": float(comm_dn),
        "required_particle_number": bool(require_particle),
        "required_spin_sector": bool(require_spin),
    }


def _runtime_split_symmetry_gate(
    *,
    polynomial: Any,
    num_sites: int,
    ordering: str,
    symmetry_spec: Mapping[str, Any] | None,
    tol: float = 1e-10,
) -> dict[str, Any]:
    return _operator_symmetry_gate(
        polynomial=polynomial,
        num_sites=int(num_sites),
        ordering=str(ordering),
        symmetry_spec=symmetry_spec,
        tol=float(tol),
    )


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
    compile_metadata: dict[str, Any] = {
        "num_polynomial_terms": int(n_poly_terms),
        "signature_size": int(len(signature)),
        "has_boson_support": bool(has_boson_support),
        "has_fermion_support": bool(has_fermion_support),
        "support_size": int(len(support_qubits)),
        "serialized_terms_exyz": _serialize_polynomial_terms(polynomial),
    }
    symmetry_spec_out = (dict(symmetry_spec) if isinstance(symmetry_spec, Mapping) else None)
    if symmetry_spec_out is not None:
        symmetry_gate = _operator_symmetry_gate(
            polynomial=polynomial,
            num_sites=int(num_sites),
            ordering=str(ordering),
            symmetry_spec=symmetry_spec_out,
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
        symmetry_gate = _runtime_split_symmetry_gate(
            polynomial=child_poly,
            num_sites=int(num_sites),
            ordering=str(ordering),
            symmetry_spec=symmetry_spec,
        )
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
                symmetry_spec=_symmetry_spec_with_runtime_gate(
                    base_spec=symmetry_spec,
                    gate=symmetry_gate,
                ),
            )
        )
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
    max_subset_size: int = 3,
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
                        rebuild_polynomial_from_serialized_terms(serialized_parent),
                        tol=float(tol),
                    )
                except Exception:
                    parent_signature = None
    child_rows = [dict(row) for row in children if isinstance(row, Mapping)]
    if len(child_rows) <= 1:
        return []
    subset_cap = int(max(1, min(int(max_subset_size), len(child_rows))))
    out: list[dict[str, Any]] = []
    seen_signatures: set[tuple[tuple[str, float], ...]] = set()
    for subset_size in range(1, subset_cap + 1):
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
            subset_poly = rebuild_polynomial_from_serialized_terms(serialized_subset)
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
            )
            if not bool(symmetry_gate.get("passed", True)):
                continue
            child_index_tag = ",".join(str(int(idx)) for idx in child_indices)
            subset_label = f"{str(parent_label)}::child_set[{child_index_tag}]"
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
                    symmetry_spec=_symmetry_spec_with_runtime_gate(
                        base_spec=symmetry_spec,
                        gate=symmetry_gate,
                    ),
                )
            )
            compile_metadata = dict(subset_meta.get("compile_metadata", {}))
            compile_metadata["runtime_split"] = {
                "mode": str(split_mode),
                "parent_label": str(parent_label),
                "child_indices": [int(idx) for idx in child_indices],
                "child_labels": [str(label) for label in child_labels],
                "child_generator_ids": [str(x) for x in child_generator_ids],
                "child_count": int(len(child_rows)),
                "representation": "child_set",
                "symmetry_gate": dict(symmetry_gate),
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
                    "symmetry_gate": dict(symmetry_gate),
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
        compiled_cost_parent=(float(compiled_cost_parent) if compiled_cost_parent is not None else None),
        compiled_cost_children=(float(compiled_cost_children) if compiled_cost_children is not None else None),
        insertion_positions=([int(x) for x in insertion_positions] if insertion_positions is not None else []),
    )
    return asdict(event)

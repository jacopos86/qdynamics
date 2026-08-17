"""Support-atom adapters for AP-McLachlan support patch controllers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.state import (
    AP_PARAMETERIZATION_LOGICAL_SHARED,
    AP_PARAMETERIZATION_PER_PAULI_TERM,
    APMcLachlanState,
    RuntimeCoordinateRecord,
    normalize_parameterization_mode,
    runtime_coordinate_records,
    state_with_appended_runtime_coordinates,
    state_with_runtime_coordinate_patch,
    state_without_runtime_indices,
)
from src.quantum.ansatz_parameterization import iter_runtime_rotation_terms
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT = "unique_support"
APPEND_OCCURRENCE_POLICY_LAYER_REUSE = "layer_reuse"
APPEND_OCCURRENCE_POLICIES = (
    APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT,
    APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
)
AP_APPEND_OCCURRENCE_LEDGER_KEY = "ap_append_occurrence_ledger_v1"
AP_APPEND_ORIGIN_LEDGER_KEY = "ap_append_origin_ledger_v1"
_APPEND_OCCURRENCE_SUFFIX_RE = re.compile(
    r"^(?P<base>.+)::ap_occ(?P<index>[2-9][0-9]*)$"
)
_PAULI_CHILD_LABEL_RE = re.compile(
    r"^(?P<parent>.+)::r(?P<local_index>[0-9]+)::(?P<pauli>.+)$"
)
_PAULI_CHILD_OCCURRENCE_RE = re.compile(
    r"^(?P<parent>.+)::ap_occ(?P<index>[2-9][0-9]*)"
    r"::r(?P<local_index>[0-9]+)::(?P<pauli>.+)$"
)


@dataclass(frozen=True)
class SupportAtom:
    """One candidate support unit for append/prune/exchange decisions."""

    atom_id: str
    atom_label: str
    parent_label: str
    term: Any
    parameterization_mode: str
    runtime_count: int
    origin_kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActiveSupportAtom(SupportAtom):
    """A support atom already present in the active AP runtime state."""

    runtime_indices: tuple[int, ...] = ()
    logical_index: int | None = None
    theta_values: tuple[float, ...] = ()


def active_support_atoms(
    state: APMcLachlanState,
    theta_runtime: Sequence[float] | np.ndarray | None = None,
) -> tuple[ActiveSupportAtom, ...]:
    """Return active support atoms in the state's configured granularity."""

    records = runtime_coordinate_records(state, theta_runtime)
    mode = normalize_parameterization_mode(state.parameterization_mode)
    if mode == AP_PARAMETERIZATION_LOGICAL_SHARED:
        return tuple(_logical_shared_active_atom(record) for record in records)
    return tuple(_per_pauli_active_atom(record) for record in records)


def candidate_append_atoms(
    state: APMcLachlanState,
    *,
    max_atoms: int | None = None,
    allow_incomplete_candidate_pool: bool = False,
    occurrence_policy: str = APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT,
) -> tuple[SupportAtom, ...]:
    """Return candidate append atoms at the active parameterization granularity."""

    if not bool(state.can_structural_edit) and not bool(allow_incomplete_candidate_pool):
        raise ValueError(
            "Cannot enumerate AP support atoms from an incomplete candidate pool; "
            "set allow_incomplete_candidate_pool=True only for diagnostics."
        )
    mode = normalize_parameterization_mode(state.parameterization_mode)
    policy = normalize_append_occurrence_policy(occurrence_policy)
    active = active_support_atoms(state)
    active_base_labels = {
        _split_append_occurrence_label(str(atom.atom_label))[0] for atom in active
    }
    occurrence_counts = _append_occurrence_counts(state, active_atoms=active)
    no_pauli_split_parents = _no_pauli_split_parent_labels(state)
    atoms: list[SupportAtom] = []
    seen: set[str] = set()
    for candidate_index, term in enumerate(tuple(state.candidate_pool_terms or ())):
        parent_label = str(getattr(term, "label", f"candidate_{candidate_index}"))
        if mode == AP_PARAMETERIZATION_LOGICAL_SHARED:
            base_atom_label = parent_label
            if (
                policy == APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT
                and base_atom_label in active_base_labels
            ):
                continue
            atom_label, occurrence_index = _next_append_occurrence_label(
                base_atom_label,
                parameterization_mode=mode,
                occurrence_counts=occurrence_counts,
                occurrence_policy=policy,
            )
            atom = SupportAtom(
                atom_id=f"logical:{atom_label}",
                atom_label=atom_label,
                parent_label=atom_label,
                term=term,
                parameterization_mode=mode,
                runtime_count=1,
                origin_kind="candidate_pool",
                metadata={
                    "candidate_index": int(candidate_index),
                    "runtime_child_count": int(
                        len(_rotation_specs_for_term(term, state=state))
                    ),
                    "base_atom_id": f"logical:{base_atom_label}",
                    "base_atom_label": base_atom_label,
                    "base_parent_label": parent_label,
                    "occurrence_index": int(occurrence_index),
                    "append_occurrence_policy": policy,
                },
            )
            _append_unique_atom(atoms, seen, atom)
            if max_atoms is not None and int(len(atoms)) >= int(max_atoms):
                return tuple(atoms)
            continue

        if parent_label in no_pauli_split_parents:
            continue
        for local_index, spec in enumerate(_rotation_specs_for_term(term, state=state)):
            base_atom_label = f"{parent_label}::r{int(local_index)}::{spec.pauli_exyz}"
            if (
                policy == APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT
                and base_atom_label in active_base_labels
            ):
                continue
            atom_label, occurrence_index = _next_append_occurrence_label(
                base_atom_label,
                parameterization_mode=mode,
                occurrence_counts=occurrence_counts,
                occurrence_policy=policy,
            )
            atom = SupportAtom(
                atom_id=f"pauli:{atom_label}",
                atom_label=atom_label,
                parent_label=parent_label,
                term=_single_child_term(
                    label=atom_label,
                    pauli_exyz=str(spec.pauli_exyz),
                    coeff_real=float(spec.coeff_real),
                    nq=int(spec.nq),
                    repr_mode=_polynomial_repr_mode(getattr(term, "polynomial")),
                ),
                parameterization_mode=mode,
                runtime_count=1,
                origin_kind="candidate_pool",
                metadata={
                    "candidate_index": int(candidate_index),
                    "local_child_index": int(local_index),
                    "pauli_exyz": str(spec.pauli_exyz),
                    "coeff_real": float(spec.coeff_real),
                    "nq": int(spec.nq),
                    "base_atom_id": f"pauli:{base_atom_label}",
                    "base_atom_label": base_atom_label,
                    "base_parent_label": parent_label,
                    "occurrence_index": int(occurrence_index),
                    "append_occurrence_policy": policy,
                },
            )
            _append_unique_atom(atoms, seen, atom)
            if max_atoms is not None and int(len(atoms)) >= int(max_atoms):
                return tuple(atoms)
    if max_atoms is None:
        return tuple(atoms)
    return tuple(atoms[: int(max_atoms)])


def state_with_appended_atoms(
    state: APMcLachlanState,
    atoms: Sequence[SupportAtom],
    *,
    theta_runtime: Sequence[float] | np.ndarray,
) -> tuple[APMcLachlanState, np.ndarray]:
    """Append support atoms as zero-angle runtime coordinates."""

    prior_counts = _append_occurrence_counts(state)
    _validate_unique_base_atoms_within_batch(atoms)
    next_state, theta = state_with_appended_runtime_coordinates(
        state,
        tuple(atom.term for atom in atoms),
        coordinate_labels=_runtime_coordinate_labels_for_atoms(state, atoms),
        theta_runtime=theta_runtime,
        metadata={"context": "state_with_appended_atoms"},
    )
    return (
        _state_with_recorded_append_occurrences(
            next_state,
            atoms,
            prior_counts=prior_counts,
        ),
        theta,
    )


# NOTE: ``state_with_inserted_atoms`` used to live here as a compatibility
# alias that silently forwarded to :func:`state_with_appended_atoms`.  Removed
# 2026-08-15: the name promised positional insertion this lane has never
# implemented, and it is reserved for the deletion-conditioned exchange
# selector's real insertion-at-cut materialization.


def state_without_active_atoms(
    state: APMcLachlanState,
    atoms: Sequence[ActiveSupportAtom],
    *,
    theta_runtime: Sequence[float] | np.ndarray,
) -> tuple[APMcLachlanState, np.ndarray]:
    """Delete active support atoms by their exact runtime indices."""

    prior_counts = _append_occurrence_counts(state)
    removed = tuple(sorted({idx for atom in atoms for idx in atom.runtime_indices}))
    next_state, theta = state_without_runtime_indices(
        state,
        removed,
        theta_runtime=theta_runtime,
    )
    return (
        _state_with_recorded_append_occurrences(
            next_state,
            (),
            prior_counts=prior_counts,
        ),
        theta,
    )


def state_with_support_patch_atoms(
    state: APMcLachlanState,
    *,
    removed_runtime_indices: Sequence[int],
    inserted_atoms: Sequence[SupportAtom],
    theta_runtime: Sequence[float] | np.ndarray,
) -> tuple[APMcLachlanState, np.ndarray]:
    """Apply a delete-then-insert support-atom patch without mutating input state."""

    prior_counts = _append_occurrence_counts(state)
    _validate_unique_base_atoms_within_batch(inserted_atoms)
    next_state, theta = state_with_runtime_coordinate_patch(
        state,
        removed_runtime_indices=removed_runtime_indices,
        inserted_coordinate_terms=tuple(atom.term for atom in inserted_atoms),
        inserted_coordinate_labels=_runtime_coordinate_labels_for_atoms(
            state, inserted_atoms
        ),
        theta_runtime=theta_runtime,
        metadata={"context": "state_with_support_patch_atoms"},
    )
    return (
        _state_with_recorded_append_occurrences(
            next_state,
            inserted_atoms,
            prior_counts=prior_counts,
        ),
        theta,
    )


def _runtime_coordinate_labels_for_atoms(
    state: APMcLachlanState,
    atoms: Sequence[SupportAtom],
) -> tuple[str, ...]:
    mode = normalize_parameterization_mode(state.parameterization_mode)
    if mode == AP_PARAMETERIZATION_LOGICAL_SHARED:
        return tuple(
            f"{str(atom.parent_label)}::logical::generator" for atom in atoms
        )
    return tuple(str(atom.atom_label) for atom in atoms)


def _per_pauli_active_atom(record: RuntimeCoordinateRecord) -> ActiveSupportAtom:
    base_atom_label, occurrence_index = _split_append_occurrence_label(
        str(record.runtime_label)
    )
    return ActiveSupportAtom(
        atom_id=f"pauli:{record.runtime_label}",
        atom_label=str(record.runtime_label),
        parent_label=str(record.parent_label),
        term=_single_child_term(
            label=str(record.runtime_label),
            pauli_exyz=str(record.metadata["pauli_exyz"]),
            coeff_real=float(record.metadata["coeff_real"]),
            nq=int(record.metadata["nq"]),
            repr_mode=_polynomial_repr_mode(getattr(record.term, "polynomial")),
        ),
        parameterization_mode=AP_PARAMETERIZATION_PER_PAULI_TERM,
        runtime_count=1,
        origin_kind="active_support",
        metadata={
            **dict(record.metadata),
            "base_atom_id": f"pauli:{base_atom_label}",
            "base_atom_label": base_atom_label,
            "occurrence_index": int(occurrence_index),
        },
        runtime_indices=(int(record.runtime_index),),
        logical_index=record.logical_index,
        theta_values=(float(record.theta_value),),
    )


def _logical_shared_active_atom(record: RuntimeCoordinateRecord) -> ActiveSupportAtom:
    base_atom_label, occurrence_index = _split_append_occurrence_label(
        str(record.parent_label)
    )
    return ActiveSupportAtom(
        atom_id=f"logical:{record.parent_label}",
        atom_label=str(record.parent_label),
        parent_label=str(record.parent_label),
        term=record.term,
        parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
        runtime_count=1,
        origin_kind="active_support",
        metadata={
            **dict(record.metadata),
            "base_atom_id": f"logical:{base_atom_label}",
            "base_atom_label": base_atom_label,
            "occurrence_index": int(occurrence_index),
        },
        runtime_indices=(int(record.runtime_index),),
        logical_index=record.logical_index,
        theta_values=(float(record.theta_value),),
    )


def _rotation_specs_for_term(term: Any, *, state: APMcLachlanState) -> tuple[Any, ...]:
    return iter_runtime_rotation_terms(
        getattr(term, "polynomial"),
        ignore_identity=bool(state.layout.ignore_identity),
        coefficient_tolerance=float(state.layout.coefficient_tolerance),
        sort_terms=(str(state.layout.term_order).strip().lower() == "sorted"),
    )


def _polynomial_repr_mode(poly: Any) -> str:
    return str(getattr(poly, "_repr_mode", "JW") or "JW")


def _single_child_term(
    *,
    label: str,
    pauli_exyz: str,
    coeff_real: float,
    nq: int,
    repr_mode: str,
) -> AnsatzTerm:
    return AnsatzTerm(
        label=str(label),
        polynomial=PauliPolynomial(
            str(repr_mode),
            [PauliTerm(int(nq), ps=str(pauli_exyz), pc=float(coeff_real))],
        ),
        execution_mode="termwise_product",
    )


def _no_pauli_split_parent_labels(state: APMcLachlanState) -> set[str]:
    payload: Any = None
    source = getattr(state, "candidate_pool_source", None)
    filter_payload = getattr(source, "filter_payload", None)
    if isinstance(filter_payload, Mapping):
        payload = filter_payload.get("legal_subspace_append_guard", None)
    if not isinstance(payload, Mapping):
        extensions = getattr(state, "extensions", {}) or {}
        if isinstance(extensions, Mapping):
            payload = extensions.get("legal_subspace_append_guard", None)
    if not isinstance(payload, Mapping):
        return set()
    labels = payload.get("no_pauli_split_parent_labels", None)
    if labels is None:
        labels = payload.get("no_pauli_split_parent_labels_sample", ())
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        return set()
    return {str(label) for label in labels if str(label)}


def no_pauli_split_parent_labels(state: APMcLachlanState) -> set[str]:
    return _no_pauli_split_parent_labels(state)


def normalize_append_occurrence_policy(value: str | None) -> str:
    policy = str(value or APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT).strip().lower()
    if policy not in set(APPEND_OCCURRENCE_POLICIES):
        raise ValueError(
            "append occurrence policy must be one of "
            f"{APPEND_OCCURRENCE_POLICIES!r}; got {value!r}."
        )
    return policy


def append_occurrence_base_label(label: str) -> str:
    """Return the reusable base-support label for one ANZATS occurrence."""

    return _split_append_occurrence_label(str(label))[0]


def _next_append_occurrence_label(
    base_atom_label: str,
    *,
    parameterization_mode: str,
    occurrence_counts: Mapping[str, int],
    occurrence_policy: str,
) -> tuple[str, int]:
    policy = normalize_append_occurrence_policy(occurrence_policy)
    if policy == APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT:
        return str(base_atom_label), 1
    prefix = (
        "logical"
        if normalize_parameterization_mode(parameterization_mode)
        == AP_PARAMETERIZATION_LOGICAL_SHARED
        else "pauli"
    )
    base_atom_id = f"{prefix}:{base_atom_label}"
    occurrence_index = int(occurrence_counts.get(base_atom_id, 0)) + 1
    if occurrence_index <= 1:
        return str(base_atom_label), 1
    if prefix == "pauli":
        match = _PAULI_CHILD_LABEL_RE.fullmatch(str(base_atom_label))
        if match is None:
            raise ValueError(
                "per_pauli_term append atom label does not have child-coordinate "
                f"form: {base_atom_label!r}."
            )
        return (
            f"{match.group('parent')}::ap_occ{occurrence_index}"
            f"::r{match.group('local_index')}::{match.group('pauli')}",
            occurrence_index,
        )
    return f"{base_atom_label}::ap_occ{occurrence_index}", occurrence_index


def _split_append_occurrence_label(label: str) -> tuple[str, int]:
    pauli_match = _PAULI_CHILD_OCCURRENCE_RE.fullmatch(str(label))
    if pauli_match is not None:
        return (
            f"{pauli_match.group('parent')}::r{pauli_match.group('local_index')}"
            f"::{pauli_match.group('pauli')}",
            int(pauli_match.group("index")),
        )
    match = _APPEND_OCCURRENCE_SUFFIX_RE.fullmatch(str(label))
    if match is None:
        return str(label), 1
    return str(match.group("base")), int(match.group("index"))


def _append_occurrence_counts(
    state: APMcLachlanState,
    *,
    active_atoms: Sequence[ActiveSupportAtom] | None = None,
) -> dict[str, int]:
    extensions = dict(state.extensions or {})
    payload = extensions.get(AP_APPEND_OCCURRENCE_LEDGER_KEY, {})
    raw_counts = payload.get("counts", {}) if isinstance(payload, Mapping) else {}
    counts = {
        str(key): int(value)
        for key, value in dict(raw_counts or {}).items()
        if int(value) >= 0
    }
    atoms = tuple(active_atoms) if active_atoms is not None else active_support_atoms(state)
    for atom in atoms:
        metadata = dict(atom.metadata or {})
        base_atom_label, parsed_index = _split_append_occurrence_label(atom.atom_label)
        prefix = (
            "logical"
            if atom.parameterization_mode == AP_PARAMETERIZATION_LOGICAL_SHARED
            else "pauli"
        )
        base_atom_id = str(metadata.get("base_atom_id", f"{prefix}:{base_atom_label}"))
        occurrence_index = int(metadata.get("occurrence_index", parsed_index))
        counts[base_atom_id] = max(int(counts.get(base_atom_id, 0)), occurrence_index)
    return counts


def _support_atom_base_id(atom: SupportAtom) -> str:
    metadata = dict(atom.metadata or {})
    base_atom_id = metadata.get("base_atom_id")
    if base_atom_id is not None and str(base_atom_id):
        return str(base_atom_id)
    base_atom_label, _ = _split_append_occurrence_label(atom.atom_label)
    prefix = (
        "logical"
        if atom.parameterization_mode == AP_PARAMETERIZATION_LOGICAL_SHARED
        else "pauli"
    )
    return f"{prefix}:{base_atom_label}"


def _validate_unique_base_atoms_within_batch(atoms: Sequence[SupportAtom]) -> None:
    base_atom_ids = tuple(_support_atom_base_id(atom) for atom in tuple(atoms))
    if len(set(base_atom_ids)) != len(base_atom_ids):
        raise ValueError(
            "An append batch may contain each base support atom at most once; "
            f"got {base_atom_ids!r}."
        )


def _state_with_recorded_append_occurrences(
    state: APMcLachlanState,
    atoms: Sequence[SupportAtom],
    *,
    prior_counts: Mapping[str, int] | None = None,
) -> APMcLachlanState:
    atom_tuple = tuple(atoms)
    if not atom_tuple and not prior_counts:
        return state
    counts = _append_occurrence_counts(state)
    for base_atom_id, occurrence_index in dict(prior_counts or {}).items():
        counts[str(base_atom_id)] = max(
            int(counts.get(str(base_atom_id), 0)),
            int(occurrence_index),
        )
    for atom in atom_tuple:
        metadata = dict(atom.metadata or {})
        _, parsed_index = _split_append_occurrence_label(atom.atom_label)
        occurrence_index = int(metadata.get("occurrence_index", parsed_index))
        base_atom_id = _support_atom_base_id(atom)
        counts[base_atom_id] = max(int(counts.get(base_atom_id, 0)), occurrence_index)
    extensions = dict(state.extensions or {})
    extensions[AP_APPEND_OCCURRENCE_LEDGER_KEY] = {
        "schema": AP_APPEND_OCCURRENCE_LEDGER_KEY,
        "counts": dict(sorted(counts.items())),
    }
    prior_origin_payload = extensions.get(AP_APPEND_ORIGIN_LEDGER_KEY, {})
    prior_origin_labels = (
        prior_origin_payload.get("atom_labels", ())
        if isinstance(prior_origin_payload, Mapping)
        else ()
    )
    origin_labels = {
        str(label) for label in tuple(prior_origin_labels or ()) if str(label)
    }
    origin_labels.update(
        str(atom.atom_label) for atom in atom_tuple if str(atom.atom_label)
    )
    extensions[AP_APPEND_ORIGIN_LEDGER_KEY] = {
        "schema": AP_APPEND_ORIGIN_LEDGER_KEY,
        "atom_labels": sorted(origin_labels),
    }
    return replace(state, extensions=extensions)


def appended_origin_atom_labels(state: APMcLachlanState) -> frozenset[str]:
    """Return support labels introduced after the serialized seed was loaded."""

    extensions = dict(state.extensions or {})
    payload = extensions.get(AP_APPEND_ORIGIN_LEDGER_KEY, {})
    labels = payload.get("atom_labels", ()) if isinstance(payload, Mapping) else ()
    return frozenset(str(label) for label in tuple(labels or ()) if str(label))


def _append_unique_atom(
    atoms: list[SupportAtom],
    seen: set[str],
    atom: SupportAtom,
) -> None:
    if atom.atom_label in seen:
        raise ValueError(f"Duplicate support atom label: {atom.atom_label!r}.")
    seen.add(str(atom.atom_label))
    atoms.append(atom)


__all__ = [
    "APPEND_OCCURRENCE_POLICIES",
    "APPEND_OCCURRENCE_POLICY_LAYER_REUSE",
    "APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT",
    "AP_APPEND_OCCURRENCE_LEDGER_KEY",
    "AP_APPEND_ORIGIN_LEDGER_KEY",
    "ActiveSupportAtom",
    "SupportAtom",
    "active_support_atoms",
    "appended_origin_atom_labels",
    "append_occurrence_base_label",
    "candidate_append_atoms",
    "no_pauli_split_parent_labels",
    "normalize_append_occurrence_policy",
    "state_with_appended_atoms",
    "state_with_support_patch_atoms",
    "state_without_active_atoms",
]

"""Support-atom adapters for AP-McLachlan support patch controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
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
) -> tuple[SupportAtom, ...]:
    """Return candidate append atoms at the active parameterization granularity."""

    if not bool(state.can_structural_edit) and not bool(allow_incomplete_candidate_pool):
        raise ValueError(
            "Cannot enumerate AP support atoms from an incomplete candidate pool; "
            "set allow_incomplete_candidate_pool=True only for diagnostics."
        )
    mode = normalize_parameterization_mode(state.parameterization_mode)
    active = active_support_atoms(state)
    active_labels = {str(atom.atom_label) for atom in active}
    active_parents = {str(atom.parent_label) for atom in active}
    no_pauli_split_parents = _no_pauli_split_parent_labels(state)
    atoms: list[SupportAtom] = []
    seen: set[str] = set()
    for candidate_index, term in enumerate(tuple(state.candidate_pool_terms or ())):
        parent_label = str(getattr(term, "label", f"candidate_{candidate_index}"))
        if mode == AP_PARAMETERIZATION_LOGICAL_SHARED:
            atom_label = parent_label
            if parent_label in active_parents:
                continue
            atom = SupportAtom(
                atom_id=f"logical:{atom_label}",
                atom_label=atom_label,
                parent_label=parent_label,
                term=term,
                parameterization_mode=mode,
                runtime_count=1,
                origin_kind="candidate_pool",
                metadata={
                    "candidate_index": int(candidate_index),
                    "runtime_child_count": int(
                        len(_rotation_specs_for_term(term, state=state))
                    ),
                },
            )
            _append_unique_atom(atoms, seen, atom)
            if max_atoms is not None and int(len(atoms)) >= int(max_atoms):
                return tuple(atoms)
            continue

        if parent_label in no_pauli_split_parents:
            continue
        for local_index, spec in enumerate(_rotation_specs_for_term(term, state=state)):
            atom_label = f"{parent_label}::r{int(local_index)}::{spec.pauli_exyz}"
            if atom_label in active_labels:
                continue
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

    return state_with_appended_runtime_coordinates(
        state,
        tuple(atom.term for atom in atoms),
        coordinate_labels=_runtime_coordinate_labels_for_atoms(state, atoms),
        theta_runtime=theta_runtime,
        metadata={"context": "state_with_appended_atoms"},
    )


def state_with_inserted_atoms(
    state: APMcLachlanState,
    atoms: Sequence[SupportAtom],
    *,
    theta_runtime: Sequence[float] | np.ndarray,
) -> tuple[APMcLachlanState, np.ndarray]:
    """Compatibility wrapper for old code; active terminology is append."""

    return state_with_appended_atoms(
        state,
        atoms,
        theta_runtime=theta_runtime,
    )


def state_without_active_atoms(
    state: APMcLachlanState,
    atoms: Sequence[ActiveSupportAtom],
    *,
    theta_runtime: Sequence[float] | np.ndarray,
) -> tuple[APMcLachlanState, np.ndarray]:
    """Delete active support atoms by their exact runtime indices."""

    removed = tuple(sorted({idx for atom in atoms for idx in atom.runtime_indices}))
    return state_without_runtime_indices(
        state,
        removed,
        theta_runtime=theta_runtime,
    )


def state_with_support_patch_atoms(
    state: APMcLachlanState,
    *,
    removed_runtime_indices: Sequence[int],
    inserted_atoms: Sequence[SupportAtom],
    theta_runtime: Sequence[float] | np.ndarray,
) -> tuple[APMcLachlanState, np.ndarray]:
    """Apply a delete-then-insert support-atom patch without mutating input state."""

    return state_with_runtime_coordinate_patch(
        state,
        removed_runtime_indices=removed_runtime_indices,
        inserted_coordinate_terms=tuple(atom.term for atom in inserted_atoms),
        inserted_coordinate_labels=_runtime_coordinate_labels_for_atoms(
            state, inserted_atoms
        ),
        theta_runtime=theta_runtime,
        metadata={"context": "state_with_support_patch_atoms"},
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
        metadata=dict(record.metadata),
        runtime_indices=(int(record.runtime_index),),
        logical_index=record.logical_index,
        theta_values=(float(record.theta_value),),
    )


def _logical_shared_active_atom(record: RuntimeCoordinateRecord) -> ActiveSupportAtom:
    return ActiveSupportAtom(
        atom_id=f"logical:{record.parent_label}",
        atom_label=str(record.parent_label),
        parent_label=str(record.parent_label),
        term=record.term,
        parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
        runtime_count=1,
        origin_kind="active_support",
        metadata=dict(record.metadata),
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
    "ActiveSupportAtom",
    "SupportAtom",
    "active_support_atoms",
    "candidate_append_atoms",
    "no_pauli_split_parent_labels",
    "state_with_appended_atoms",
    "state_with_inserted_atoms",
    "state_with_support_patch_atoms",
    "state_without_active_atoms",
]

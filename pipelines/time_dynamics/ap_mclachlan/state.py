"""Neutral scaffold-state adapter for AP-McLachlan.

This module is the AP-facing boundary for static ADAPT/SNAKE/Geo seed
artifacts.  It consumes the shared ``ScaffoldRuntimeInput`` contract and exposes
only the runtime manifold pieces needed by AP-McLachlan propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
from src.quantum.ansatz_parameterization import AnsatzParameterLayout, build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


AP_MCLACHLAN_STATE_SCHEMA_V1 = "ap_mclachlan_state_v1"
AP_PARAMETERIZATION_PER_PAULI_TERM = "per_pauli_term"
AP_PARAMETERIZATION_LOGICAL_SHARED = "logical_shared"
AP_SUPPORTED_PARAMETERIZATION_MODES = (
    AP_PARAMETERIZATION_PER_PAULI_TERM,
    AP_PARAMETERIZATION_LOGICAL_SHARED,
)
AP_PREPARED_STATE_PARITY_ATOL = 1.0e-10


class APMcLachlanStateParityError(RuntimeError):
    """Raised when a mutated AP state no longer has a valid runtime layout."""


@dataclass(frozen=True)
class RuntimeCoordinateRecord:
    """One editable runtime coordinate in the active AP ansatz."""

    runtime_index: int
    runtime_label: str
    parent_label: str
    logical_index: int | None
    local_child_index: int | None
    parameterization_mode: str
    theta_value: float
    term: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class APMcLachlanState:
    """Runtime view of a fixed variational scaffold and its mutable parameters."""

    terms: tuple[Any, ...]
    layout: AnsatzParameterLayout
    theta_runtime: np.ndarray | Sequence[float]
    # psi_ref is before the selected ansatz acts; psi_initial is the
    # static-route handoff state U(theta_runtime)|psi_ref>.
    psi_ref: np.ndarray | Sequence[complex]
    psi_initial: np.ndarray | Sequence[complex]
    executor: CompiledAnsatzExecutor
    static_hamiltonian: Any
    resolved_problem: Any
    parameterization_mode: str = AP_PARAMETERIZATION_PER_PAULI_TERM
    exact_energy: float | None = None
    candidate_pool_terms: tuple[Any, ...] = ()
    candidate_pool_source: Any | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = normalize_parameterization_mode(self.parameterization_mode)
        theta = np.asarray(self.theta_runtime, dtype=float).reshape(-1)
        psi_ref = _normalize_state(self.psi_ref, name="psi_ref")
        psi_initial = _normalize_state(self.psi_initial, name="psi_initial")
        expected = int(parameter_count_for_layout(self.layout, mode))
        if int(theta.size) != expected:
            raise ValueError(
                f"theta_runtime length mismatch for AP-McLachlan state: "
                f"got {theta.size}, expected {expected} for parameterization_mode={mode!r}."
            )
        if int(psi_ref.size) != int(psi_initial.size):
            raise ValueError("psi_ref and psi_initial must have the same Hilbert dimension.")
        if int(len(self.terms)) != int(self.layout.logical_parameter_count):
            raise ValueError(
                "selected term count must match layout logical_parameter_count: "
                f"got {len(self.terms)}, expected {self.layout.logical_parameter_count}."
            )
        if str(getattr(self.executor, "parameterization_mode", "")) != mode:
            raise ValueError(
                "executor parameterization_mode does not match AP state: "
                f"{getattr(self.executor, 'parameterization_mode', None)!r} vs {mode!r}."
            )
        object.__setattr__(self, "theta_runtime", theta)
        object.__setattr__(self, "psi_ref", psi_ref)
        object.__setattr__(self, "psi_initial", psi_initial)
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "parameterization_mode", mode)
        object.__setattr__(
            self,
            "exact_energy",
            None if self.exact_energy is None else float(self.exact_energy),
        )
        object.__setattr__(self, "candidate_pool_terms", tuple(self.candidate_pool_terms))
        object.__setattr__(self, "provenance", dict(self.provenance or {}))
        object.__setattr__(self, "extensions", dict(self.extensions or {}))

    @property
    def runtime_parameter_count(self) -> int:
        return int(np.asarray(self.theta_runtime, dtype=float).reshape(-1).size)

    @property
    def active_parameter_count(self) -> int:
        return int(self.runtime_parameter_count)

    @property
    def runtime_pauli_parameter_count(self) -> int:
        return int(self.layout.runtime_parameter_count)

    @property
    def logical_parameter_count(self) -> int:
        return int(self.layout.logical_parameter_count)

    @property
    def parameterization_label(self) -> str:
        return parameterization_mode_label(self.parameterization_mode)

    @property
    def runtime_coordinate_labels(self) -> tuple[str, ...]:
        return runtime_coordinate_labels(
            self.layout,
            parameterization_mode=self.parameterization_mode,
        )

    @property
    def runtime_pauli_coordinate_labels(self) -> tuple[str, ...]:
        return runtime_coordinate_labels(
            self.layout,
            parameterization_mode=AP_PARAMETERIZATION_PER_PAULI_TERM,
        )

    @property
    def can_structural_edit(self) -> bool:
        source = self.candidate_pool_source
        return bool(getattr(source, "candidate_pool_complete", False))

    def prepare_state(self, theta_runtime: np.ndarray | Sequence[float] | None = None) -> np.ndarray:
        theta = (
            np.asarray(self.theta_runtime, dtype=float).reshape(-1)
            if theta_runtime is None
            else np.asarray(theta_runtime, dtype=float).reshape(-1)
        )
        if int(theta.size) != int(self.runtime_parameter_count):
            raise ValueError(
                f"theta_runtime length mismatch: got {theta.size}, expected {self.runtime_parameter_count}."
            )
        return _normalize_state(
            self.executor.prepare_state(theta, np.asarray(self.psi_ref, dtype=complex).reshape(-1)),
            name="prepared_state",
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": AP_MCLACHLAN_STATE_SCHEMA_V1,
            "parameterization_mode": str(self.parameterization_mode),
            "parameterization_label": str(self.parameterization_label),
            "active_parameter_count": int(self.active_parameter_count),
            "logical_parameter_count": int(self.logical_parameter_count),
            "runtime_parameter_count": int(self.runtime_parameter_count),
            "runtime_pauli_parameter_count": int(self.runtime_pauli_parameter_count),
            "runtime_coordinate_labels": [str(label) for label in self.runtime_coordinate_labels],
            "runtime_pauli_coordinate_labels": [
                str(label) for label in self.runtime_pauli_coordinate_labels
            ],
            "selected_term_labels": [str(getattr(term, "label", "")) for term in self.terms],
            "candidate_pool_term_count": int(len(self.candidate_pool_terms)),
            "candidate_pool_complete": bool(self.can_structural_edit),
            "candidate_pool_source": _candidate_pool_source_json(
                self.candidate_pool_source
            ),
            "exact_energy": None if self.exact_energy is None else float(self.exact_energy),
            "provenance": _json_safe(dict(self.provenance or {})),
            "extensions": _json_safe(dict(self.extensions or {})),
        }


def normalize_parameterization_mode(mode: str | None) -> str:
    value = str(mode or AP_PARAMETERIZATION_PER_PAULI_TERM).strip()
    if value not in set(AP_SUPPORTED_PARAMETERIZATION_MODES):
        raise ValueError(
            "Unsupported AP-McLachlan parameterization_mode: "
            f"{value!r}. Expected one of {AP_SUPPORTED_PARAMETERIZATION_MODES}."
        )
    return value


def parameterization_mode_label(mode: str | None) -> str:
    value = normalize_parameterization_mode(mode)
    if value == AP_PARAMETERIZATION_LOGICAL_SHARED:
        return "per logical / macro generator"
    return "per Pauli / polynomial term"


def parameter_count_for_layout(layout: AnsatzParameterLayout, mode: str | None) -> int:
    value = normalize_parameterization_mode(mode)
    if value == AP_PARAMETERIZATION_LOGICAL_SHARED:
        return int(layout.logical_parameter_count)
    return int(layout.runtime_parameter_count)


def _normalize_state(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=complex).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a vector.")
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have positive finite norm.")
    return np.asarray(arr / norm, dtype=complex)


def _candidate_pool_source_json(source: Any | None) -> dict[str, Any] | None:
    if source is None:
        return None
    return {
        "source_kind": str(getattr(source, "source_kind", "")),
        "pool_key": (
            None
            if getattr(source, "pool_key", None) is None
            else str(getattr(source, "pool_key"))
        ),
        "completeness": str(getattr(source, "completeness", "")),
        "candidate_pool_complete": bool(
            getattr(source, "candidate_pool_complete", False)
        ),
        "pool_build_kwargs": _json_safe(
            dict(getattr(source, "pool_build_kwargs", {}) or {})
        ),
        "filter_payload": _json_safe(
            dict(getattr(source, "filter_payload", {}) or {})
        ),
    }


def _executor_for_terms(
    terms: Sequence[Any],
    layout: AnsatzParameterLayout,
    *,
    parameterization_mode: str = AP_PARAMETERIZATION_PER_PAULI_TERM,
) -> CompiledAnsatzExecutor:
    mode = normalize_parameterization_mode(parameterization_mode)
    return CompiledAnsatzExecutor(
        tuple(terms),
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_mode=mode,
        parameterization_layout=layout,
    )


def runtime_coordinate_labels(
    layout: AnsatzParameterLayout,
    *,
    parameterization_mode: str = AP_PARAMETERIZATION_PER_PAULI_TERM,
) -> tuple[str, ...]:
    mode = normalize_parameterization_mode(parameterization_mode)
    if mode == AP_PARAMETERIZATION_LOGICAL_SHARED:
        return tuple(
            f"{str(block.candidate_label)}::logical::generator"
            for block in layout.blocks
        )
    labels: list[str] = []
    for block in layout.blocks:
        block_label = str(block.candidate_label)
        for local_index, spec in enumerate(block.terms):
            if (
                int(block.runtime_count) == 1
                and _is_stable_child_coordinate_label(block_label, str(spec.pauli_exyz))
            ):
                labels.append(block_label)
                continue
            labels.append(f"{block_label}::r{int(local_index)}::{spec.pauli_exyz}")
    return tuple(labels)


def runtime_indices_by_block_label(layout: AnsatzParameterLayout) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for block in layout.blocks:
        out[str(block.candidate_label)] = tuple(
            range(int(block.runtime_start), int(block.runtime_stop))
        )
    return out


def runtime_coordinate_records(
    state: APMcLachlanState,
    theta_runtime: Sequence[float] | np.ndarray | None = None,
) -> tuple[RuntimeCoordinateRecord, ...]:
    """Return ordered records for the currently active runtime coordinates."""

    theta = (
        np.asarray(state.theta_runtime, dtype=float).reshape(-1)
        if theta_runtime is None
        else np.asarray(theta_runtime, dtype=float).reshape(-1)
    )
    if int(theta.size) != int(state.runtime_parameter_count):
        raise ValueError(
            "theta_runtime length mismatch for runtime coordinate records: "
            f"got {theta.size}, expected {state.runtime_parameter_count}."
        )
    labels = state.runtime_coordinate_labels
    if int(len(labels)) != int(state.runtime_parameter_count):
        raise APMcLachlanStateParityError(
            "runtime coordinate label count does not match runtime parameter count: "
            f"{len(labels)} vs {state.runtime_parameter_count}."
        )

    records: list[RuntimeCoordinateRecord] = []
    mode = normalize_parameterization_mode(state.parameterization_mode)
    if mode == AP_PARAMETERIZATION_LOGICAL_SHARED:
        for runtime_index, (term, block) in enumerate(zip(state.terms, state.layout.blocks)):
            records.append(
                RuntimeCoordinateRecord(
                    runtime_index=int(runtime_index),
                    runtime_label=str(labels[runtime_index]),
                    parent_label=str(block.candidate_label),
                    logical_index=int(block.logical_index),
                    local_child_index=None,
                    parameterization_mode=mode,
                    theta_value=float(theta[runtime_index]),
                    term=term,
                    metadata={
                        "runtime_start": int(block.runtime_start),
                        "runtime_stop": int(block.runtime_stop),
                        "runtime_child_count": int(block.runtime_count),
                    },
                )
            )
        return tuple(records)

    for logical_index, (term, block) in enumerate(zip(state.terms, state.layout.blocks)):
        for local_child_index, spec in enumerate(block.terms):
            runtime_index = int(block.runtime_start + local_child_index)
            records.append(
                RuntimeCoordinateRecord(
                    runtime_index=int(runtime_index),
                    runtime_label=str(labels[runtime_index]),
                    parent_label=str(block.candidate_label),
                    logical_index=int(logical_index),
                    local_child_index=int(local_child_index),
                    parameterization_mode=mode,
                    theta_value=float(theta[runtime_index]),
                    term=term,
                    metadata={
                        "pauli_exyz": str(spec.pauli_exyz),
                        "coeff_real": float(spec.coeff_real),
                        "nq": int(spec.nq),
                        "runtime_start": int(block.runtime_start),
                        "runtime_stop": int(block.runtime_stop),
                    },
                )
            )
    return tuple(records)


def validate_runtime_state_parity(
    state: APMcLachlanState,
    theta_runtime: Sequence[float] | np.ndarray,
    *,
    context: str,
) -> None:
    """Fail loudly if a runtime state/theta pair cannot define AP propagation."""

    theta = np.asarray(theta_runtime, dtype=float).reshape(-1)
    labels = state.runtime_coordinate_labels
    if int(theta.size) != int(state.runtime_parameter_count):
        raise APMcLachlanStateParityError(
            f"{context}: theta_runtime length mismatch: got {theta.size}, "
            f"expected {state.runtime_parameter_count}."
        )
    if int(len(labels)) != int(state.runtime_parameter_count):
        raise APMcLachlanStateParityError(
            f"{context}: runtime label count mismatch: got {len(labels)}, "
            f"expected {state.runtime_parameter_count}."
        )
    if int(len(set(labels))) != int(len(labels)):
        raise APMcLachlanStateParityError(
            f"{context}: duplicate runtime coordinate labels are not allowed."
        )
    executor_count = int(getattr(state.executor, "num_parameters", -1))
    if executor_count != int(state.runtime_parameter_count):
        raise APMcLachlanStateParityError(
            f"{context}: executor tangent count mismatch: got {executor_count}, "
            f"expected {state.runtime_parameter_count}."
        )
    if str(getattr(state.executor, "parameterization_mode", "")) != state.parameterization_mode:
        raise APMcLachlanStateParityError(
            f"{context}: executor parameterization_mode mismatch: "
            f"{getattr(state.executor, 'parameterization_mode', None)!r} vs "
            f"{state.parameterization_mode!r}."
        )
    prepared = state.prepare_state(theta)
    if int(prepared.size) != int(np.asarray(state.psi_ref, dtype=complex).reshape(-1).size):
        raise APMcLachlanStateParityError(
            f"{context}: prepared-state Hilbert dimension mismatch."
        )


def state_with_appended_runtime_coordinates(
    state: APMcLachlanState,
    coordinate_terms: Sequence[Any],
    *,
    coordinate_labels: Sequence[str],
    theta_runtime: np.ndarray | Sequence[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[APMcLachlanState, np.ndarray]:
    """Return a new AP state with zero-angle runtime coordinates appended."""

    labels = tuple(str(label) for label in coordinate_labels)
    terms_in = tuple(coordinate_terms)
    if int(len(labels)) != int(len(terms_in)):
        raise ValueError(
            "coordinate_terms and coordinate_labels must have the same length: "
            f"{len(terms_in)} vs {len(labels)}."
        )
    if not terms_in:
        theta = (
            np.asarray(state.theta_runtime, dtype=float).reshape(-1)
            if theta_runtime is None
            else np.asarray(theta_runtime, dtype=float).reshape(-1)
        )
        validate_runtime_state_parity(state, theta, context="append_runtime_coordinates(noop)")
        return state, theta

    appended = tuple(
        _term_with_label(term, label=_term_label_for_appended_coordinate(state, label))
        for term, label in zip(terms_in, labels)
    )
    if state.parameterization_mode == AP_PARAMETERIZATION_PER_PAULI_TERM:
        appended_layout = build_parameter_layout(
            appended,
            ignore_identity=bool(state.layout.ignore_identity),
            coefficient_tolerance=float(state.layout.coefficient_tolerance),
            sort_terms=(str(state.layout.term_order).strip().lower() == "sorted"),
        )
        multi = [
            str(block.candidate_label)
            for block in appended_layout.blocks
            if int(block.runtime_count) != 1
        ]
        if multi:
            raise ValueError(
                "per_pauli_term runtime-coordinate append requires one Pauli child "
                f"per appended term; invalid labels: {multi}."
            )

    next_state = state_with_appended_terms(
        state,
        appended,
        theta_runtime=theta_runtime,
    )
    theta = np.asarray(next_state.theta_runtime, dtype=float).reshape(-1)
    validate_runtime_state_parity(
        next_state,
        theta,
        context=str((metadata or {}).get("context", "append_runtime_coordinates")),
    )
    appended_tail = next_state.runtime_coordinate_labels[-len(labels):]
    if tuple(appended_tail) != labels:
        raise APMcLachlanStateParityError(
            "Appended runtime labels do not match requested coordinate labels: "
            f"{tuple(appended_tail)!r} vs {labels!r}."
        )
    return next_state, theta


# NOTE: a compatibility alias named ``state_with_inserted_runtime_coordinates``
# used to live here and silently forwarded to the *append* function above.  It
# had zero callers and its name promised positional insertion that this module
# has never implemented.  It was removed 2026-08-15 so the name stays free for
# the real insertion-at-cut materialization introduced by the
# deletion-conditioned exchange selector (see
# prompt-exports/paper_ii_noiseless_conditional_exchange_implementation_spec.md).


def _term_label_for_appended_coordinate(state: APMcLachlanState, label: str) -> str:
    raw = str(label)
    if state.parameterization_mode == AP_PARAMETERIZATION_LOGICAL_SHARED:
        suffix = "::logical::generator"
        if raw.endswith(suffix):
            return raw[: -len(suffix)]
    return raw


def state_without_runtime_indices(
    state: APMcLachlanState,
    removed_runtime_indices: Sequence[int],
    *,
    theta_runtime: np.ndarray | Sequence[float] | None = None,
) -> tuple[APMcLachlanState, np.ndarray]:
    """Return a new AP state after deleting exact runtime coordinates."""

    theta_base = (
        np.asarray(state.theta_runtime, dtype=float).reshape(-1)
        if theta_runtime is None
        else np.asarray(theta_runtime, dtype=float).reshape(-1)
    )
    validate_runtime_state_parity(state, theta_base, context="delete_runtime_indices(input)")
    removed = tuple(sorted(set(int(index) for index in removed_runtime_indices)))
    if not removed:
        return state, theta_base
    for index in removed:
        if index < 0 or index >= int(state.runtime_parameter_count):
            raise ValueError(
                f"runtime index {index} out of bounds for runtime_parameter_count="
                f"{state.runtime_parameter_count}."
            )

    keep_indices = tuple(
        index for index in range(int(state.runtime_parameter_count)) if index not in set(removed)
    )
    old_labels = state.runtime_coordinate_labels
    old_keep_labels = tuple(str(old_labels[index]) for index in keep_indices)
    theta = np.asarray(theta_base[list(keep_indices)], dtype=float).reshape(-1)

    if state.parameterization_mode == AP_PARAMETERIZATION_LOGICAL_SHARED:
        terms = tuple(
            term for index, term in enumerate(state.terms) if index not in set(removed)
        )
    else:
        terms = _terms_after_per_pauli_runtime_deletion(state, removed=set(removed))

    layout = build_parameter_layout(
        terms,
        ignore_identity=bool(state.layout.ignore_identity),
        coefficient_tolerance=float(state.layout.coefficient_tolerance),
        sort_terms=(str(state.layout.term_order).strip().lower() == "sorted"),
    )
    expected = int(parameter_count_for_layout(layout, state.parameterization_mode))
    if int(theta.size) != expected:
        raise APMcLachlanStateParityError(
            "Deleted theta length does not match rebuilt layout: "
            f"got {theta.size}, expected {expected}."
        )
    new_labels = runtime_coordinate_labels(layout, parameterization_mode=state.parameterization_mode)
    if tuple(new_labels) != old_keep_labels:
        raise APMcLachlanStateParityError(
            "Runtime-coordinate deletion did not preserve retained labels: "
            f"{tuple(new_labels)!r} vs {old_keep_labels!r}."
        )

    executor = _executor_for_terms(
        terms,
        layout,
        parameterization_mode=state.parameterization_mode,
    )
    psi_initial = _normalize_state(
        executor.prepare_state(theta, np.asarray(state.psi_ref, dtype=complex).reshape(-1)),
        name="psi_initial",
    )
    next_state = APMcLachlanState(
        terms=terms,
        layout=layout,
        theta_runtime=theta,
        psi_ref=state.psi_ref,
        psi_initial=psi_initial,
        executor=executor,
        static_hamiltonian=state.static_hamiltonian,
        resolved_problem=state.resolved_problem,
        parameterization_mode=state.parameterization_mode,
        exact_energy=state.exact_energy,
        candidate_pool_terms=state.candidate_pool_terms,
        candidate_pool_source=state.candidate_pool_source,
        provenance=state.provenance,
        extensions=state.extensions,
    )
    validate_runtime_state_parity(
        next_state,
        theta,
        context="delete_runtime_indices(output)",
    )
    return next_state, theta


def state_with_runtime_coordinate_patch(
    state: APMcLachlanState,
    *,
    removed_runtime_indices: Sequence[int],
    inserted_coordinate_terms: Sequence[Any],
    inserted_coordinate_labels: Sequence[str],
    theta_runtime: np.ndarray | Sequence[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[APMcLachlanState, np.ndarray]:
    """Return a new AP state after deleting and then appending coordinates."""

    pruned_state, pruned_theta = state_without_runtime_indices(
        state,
        removed_runtime_indices,
        theta_runtime=theta_runtime,
    )
    return state_with_appended_runtime_coordinates(
        pruned_state,
        inserted_coordinate_terms,
        coordinate_labels=inserted_coordinate_labels,
        theta_runtime=pruned_theta,
        metadata=metadata,
    )


def _is_stable_child_coordinate_label(label: str, pauli_exyz: str) -> bool:
    return "::r" in str(label) and str(label).endswith(f"::{pauli_exyz}")


def _polynomial_repr_mode(poly: Any) -> str:
    return str(getattr(poly, "_repr_mode", "JW") or "JW")


def _single_child_polynomial(*, repr_mode: str, pauli_exyz: str, coeff_real: float, nq: int) -> PauliPolynomial:
    return PauliPolynomial(
        repr_mode,
        [PauliTerm(int(nq), ps=str(pauli_exyz), pc=float(coeff_real))],
    )


def _term_with_label(term: Any, *, label: str) -> AnsatzTerm:
    return AnsatzTerm(
        label=str(label),
        polynomial=getattr(term, "polynomial"),
        execution_mode=str(getattr(term, "execution_mode", "termwise_product") or "termwise_product"),
    )


def _terms_for_ap_parameterization(
    terms: Sequence[Any],
    *,
    parameterization_mode: str,
) -> tuple[tuple[Any, ...], tuple[Mapping[str, Any], ...]]:
    """Adapt serialized logical execution hints to AP runtime coordinates."""

    if parameterization_mode != AP_PARAMETERIZATION_PER_PAULI_TERM:
        return tuple(terms), ()
    resolved: list[Any] = []
    adaptations: list[Mapping[str, Any]] = []
    for logical_index, term in enumerate(terms):
        execution_mode = str(
            getattr(term, "execution_mode", "termwise_product") or "termwise_product"
        ).strip().lower()
        if execution_mode != "grouped_exact":
            resolved.append(term)
            continue
        resolved.append(
            AnsatzTerm(
                label=str(getattr(term, "label", f"term_{logical_index}")),
                polynomial=getattr(term, "polynomial"),
                execution_mode="termwise_product",
            )
        )
        adaptations.append(
            {
                "logical_index": int(logical_index),
                "label": str(getattr(term, "label", f"term_{logical_index}")),
                "source_execution_mode": "grouped_exact",
                "ap_execution_mode": "termwise_product",
            }
        )
    return tuple(resolved), tuple(adaptations)


def _single_child_term_from_spec(
    *,
    label: str,
    spec: Any,
    repr_mode: str,
) -> AnsatzTerm:
    return AnsatzTerm(
        label=str(label),
        polynomial=_single_child_polynomial(
            repr_mode=repr_mode,
            pauli_exyz=str(spec.pauli_exyz),
            coeff_real=float(spec.coeff_real),
            nq=int(spec.nq),
        ),
        execution_mode="termwise_product",
    )


def _terms_after_per_pauli_runtime_deletion(
    state: APMcLachlanState,
    *,
    removed: set[int],
) -> tuple[Any, ...]:
    labels = state.runtime_coordinate_labels
    terms: list[Any] = []
    for term, block in zip(state.terms, state.layout.blocks):
        block_indices = tuple(range(int(block.runtime_start), int(block.runtime_stop)))
        kept_block_indices = tuple(index for index in block_indices if index not in removed)
        if not kept_block_indices:
            continue
        if len(kept_block_indices) == len(block_indices):
            terms.append(term)
            continue
        repr_mode = _polynomial_repr_mode(getattr(term, "polynomial"))
        for runtime_index in kept_block_indices:
            local_child_index = int(runtime_index - int(block.runtime_start))
            spec = block.terms[local_child_index]
            terms.append(
                _single_child_term_from_spec(
                    label=str(labels[runtime_index]),
                    spec=spec,
                    repr_mode=repr_mode,
                )
            )
    return tuple(terms)


def _theta_for_parameterization(
    runtime_input: Any,
    *,
    layout: AnsatzParameterLayout,
    parameterization_mode: str,
) -> np.ndarray:
    mode = normalize_parameterization_mode(parameterization_mode)
    if mode == AP_PARAMETERIZATION_PER_PAULI_TERM:
        return np.asarray(getattr(runtime_input, "theta_runtime"), dtype=float).reshape(-1)
    theta_logical = getattr(runtime_input, "theta_logical", None)
    if theta_logical is None:
        raise ValueError(
            "parameterization_mode='logical_shared' requires theta_logical from the seed "
            "artifact; refusing to infer it from per-Pauli theta_runtime."
        )
    theta = np.asarray(theta_logical, dtype=float).reshape(-1)
    if int(theta.size) != int(layout.logical_parameter_count):
        raise ValueError(
            "theta_logical length mismatch for parameterization_mode='logical_shared': "
            f"got {theta.size}, expected {layout.logical_parameter_count}."
        )
    return theta


def _validate_prepared_state_for_parameterization(state: APMcLachlanState) -> None:
    reconstructed = state.prepare_state(state.theta_runtime)
    err = float(
        np.linalg.norm(
            np.asarray(reconstructed, dtype=complex).reshape(-1)
            - np.asarray(state.psi_initial, dtype=complex).reshape(-1)
        )
    )
    if not np.isfinite(err) or err > AP_PREPARED_STATE_PARITY_ATOL:
        raise ValueError(
            "Prepared-state parity check failed for AP-McLachlan "
            f"parameterization_mode={state.parameterization_mode!r}: "
            f"||U(theta)psi_ref - psi_initial||_2 = {err:.6e}. "
            "Use a seed whose logical generator parameters reproduce the handoff "
            "state, or run with parameterization_mode='per_pauli_term'."
        )


def state_from_scaffold_runtime_input(
    runtime_input: Any,
    *,
    parameterization_mode: str = AP_PARAMETERIZATION_PER_PAULI_TERM,
) -> APMcLachlanState:
    mode = normalize_parameterization_mode(parameterization_mode)
    source_terms = tuple(getattr(runtime_input, "selected_terms", ()) or ())
    source_layout = getattr(runtime_input, "base_layout", None)
    if source_layout is None:
        raise ValueError("ScaffoldRuntimeInput is missing base_layout.")
    theta_runtime = _theta_for_parameterization(
        runtime_input,
        layout=source_layout,
        parameterization_mode=mode,
    )
    terms, relabeling = _disambiguate_repeated_selected_term_labels(source_terms)
    terms, execution_adaptations = _terms_for_ap_parameterization(
        terms,
        parameterization_mode=mode,
    )
    if relabeling or execution_adaptations:
        layout = build_parameter_layout(
            terms,
            ignore_identity=bool(source_layout.ignore_identity),
            coefficient_tolerance=float(source_layout.coefficient_tolerance),
            sort_terms=(str(source_layout.term_order).strip().lower() == "sorted"),
        )
        if int(parameter_count_for_layout(layout, mode)) != int(theta_runtime.size):
            raise APMcLachlanStateParityError(
                "Selected-support adaptation changed the AP runtime parameter count."
            )
    else:
        layout = source_layout
    extensions = dict(getattr(runtime_input, "extensions", {}) or {})
    if relabeling:
        extensions["selected_term_label_disambiguation"] = {
            "schema": "ap_selected_term_label_disambiguation_v1",
            "reason": "repeated_static_seed_generator_labels",
            "relabeling": [dict(item) for item in relabeling],
        }
    if execution_adaptations:
        extensions["selected_term_execution_mode_adaptation"] = {
            "schema": "ap_selected_term_execution_mode_adaptation_v1",
            "reason": "per_pauli_runtime_coordinates_from_serialized_seed_layout",
            "adaptations": [dict(item) for item in execution_adaptations],
        }
    executor = _executor_for_terms(terms, layout, parameterization_mode=mode)
    state = APMcLachlanState(
        terms=terms,
        layout=layout,
        theta_runtime=theta_runtime,
        psi_ref=np.asarray(getattr(runtime_input, "psi_ref"), dtype=complex).reshape(-1),
        psi_initial=np.asarray(getattr(runtime_input, "psi_initial"), dtype=complex).reshape(-1),
        executor=executor,
        static_hamiltonian=getattr(runtime_input, "h_poly"),
        resolved_problem=getattr(runtime_input, "resolved_problem", None),
        parameterization_mode=mode,
        exact_energy=getattr(runtime_input, "exact_energy", None),
        candidate_pool_terms=tuple(getattr(runtime_input, "candidate_pool_terms", ()) or ()),
        candidate_pool_source=getattr(runtime_input, "candidate_pool_source", None),
        provenance=dict(getattr(runtime_input, "provenance", {}) or {}),
        extensions=extensions,
    )
    _validate_prepared_state_for_parameterization(state)
    return state


def _disambiguate_repeated_selected_term_labels(
    terms: Sequence[Any],
) -> tuple[tuple[Any, ...], tuple[Mapping[str, Any], ...]]:
    """Give repeated static-seed blocks stable, nonphysical occurrence labels."""

    totals: dict[str, int] = {}
    for term in terms:
        label = str(getattr(term, "label", ""))
        totals[label] = int(totals.get(label, 0)) + 1
    seen: dict[str, int] = {}
    used: set[str] = set()
    resolved: list[Any] = []
    relabeling: list[Mapping[str, Any]] = []
    for logical_index, term in enumerate(terms):
        source_label = str(getattr(term, "label", ""))
        occurrence = int(seen.get(source_label, 0)) + 1
        seen[source_label] = occurrence
        resolved_label = source_label
        if int(totals[source_label]) > 1 and occurrence > 1:
            resolved_label = f"{source_label}::ap_occurrence[{occurrence}]"
            while resolved_label in used:
                occurrence += 1
                resolved_label = f"{source_label}::ap_occurrence[{occurrence}]"
        used.add(resolved_label)
        if resolved_label == source_label:
            resolved.append(term)
            continue
        resolved.append(_term_with_label(term, label=resolved_label))
        relabeling.append(
            {
                "logical_index": int(logical_index),
                "source_label": source_label,
                "resolved_label": resolved_label,
                "occurrence": int(occurrence),
            }
        )
    return tuple(resolved), tuple(relabeling)


def state_with_appended_terms(
    state: APMcLachlanState,
    appended_terms: Sequence[Any],
    *,
    theta_runtime: np.ndarray | Sequence[float] | None = None,
) -> APMcLachlanState:
    """Return a new AP state with appended generator blocks at zero angles."""

    appended = tuple(appended_terms)
    if not appended:
        return state
    theta_base = (
        np.asarray(state.theta_runtime, dtype=float).reshape(-1)
        if theta_runtime is None
        else np.asarray(theta_runtime, dtype=float).reshape(-1)
    )
    if int(theta_base.size) != int(state.runtime_parameter_count):
        raise ValueError(
            f"theta_runtime length mismatch for append: got {theta_base.size}, "
            f"expected {state.runtime_parameter_count}."
        )
    terms = tuple(state.terms) + appended
    layout = build_parameter_layout(
        terms,
        ignore_identity=bool(state.layout.ignore_identity),
        coefficient_tolerance=float(state.layout.coefficient_tolerance),
        sort_terms=(str(state.layout.term_order).strip().lower() == "sorted"),
    )
    prefix_parameter_count = (
        0
        if not state.terms
        else (
            int(layout.blocks[len(state.terms) - 1].runtime_stop)
            if state.parameterization_mode == AP_PARAMETERIZATION_PER_PAULI_TERM
            else int(len(state.terms))
        )
    )
    if prefix_parameter_count != int(state.runtime_parameter_count):
        raise ValueError("Rebuilt append layout does not preserve existing active support.")
    theta = np.zeros(
        int(parameter_count_for_layout(layout, state.parameterization_mode)),
        dtype=float,
    )
    theta[: int(theta_base.size)] = theta_base
    executor = _executor_for_terms(
        terms,
        layout,
        parameterization_mode=state.parameterization_mode,
    )
    return APMcLachlanState(
        terms=terms,
        layout=layout,
        theta_runtime=theta,
        psi_ref=state.psi_ref,
        psi_initial=state.prepare_state(theta_base),
        executor=executor,
        static_hamiltonian=state.static_hamiltonian,
        resolved_problem=state.resolved_problem,
        parameterization_mode=state.parameterization_mode,
        exact_energy=state.exact_energy,
        candidate_pool_terms=state.candidate_pool_terms,
        candidate_pool_source=state.candidate_pool_source,
        provenance=state.provenance,
        extensions=state.extensions,
    )


def state_without_term_labels(
    state: APMcLachlanState,
    removed_labels: Sequence[str],
    *,
    theta_runtime: np.ndarray | Sequence[float] | None = None,
) -> APMcLachlanState:
    """Return a new APM state after deleting complete logical generator blocks."""

    labels_to_remove = {str(label) for label in removed_labels if str(label) != ""}
    if not labels_to_remove:
        return state
    theta_base = (
        np.asarray(state.theta_runtime, dtype=float).reshape(-1)
        if theta_runtime is None
        else np.asarray(theta_runtime, dtype=float).reshape(-1)
    )
    if int(theta_base.size) != int(state.runtime_parameter_count):
        raise ValueError(
            f"theta_runtime length mismatch for prune: got {theta_base.size}, "
            f"expected {state.runtime_parameter_count}."
        )

    old_labels = tuple(str(getattr(term, "label", "")) for term in state.terms)
    missing = sorted(label for label in labels_to_remove if label not in set(old_labels))
    if missing:
        raise ValueError(f"Cannot prune missing generator labels: {missing}.")

    terms = tuple(
        term
        for term in state.terms
        if str(getattr(term, "label", "")) not in labels_to_remove
    )
    layout = build_parameter_layout(
        terms,
        ignore_identity=bool(state.layout.ignore_identity),
        coefficient_tolerance=float(state.layout.coefficient_tolerance),
        sort_terms=(str(state.layout.term_order).strip().lower() == "sorted"),
    )
    removed_indices = _runtime_indices_for_labels(state, labels_to_remove)
    keep_indices = tuple(
        index for index in range(int(state.runtime_parameter_count)) if index not in removed_indices
    )
    theta = np.asarray(theta_base[list(keep_indices)], dtype=float).reshape(-1)
    expected = int(parameter_count_for_layout(layout, state.parameterization_mode))
    if int(theta.size) != expected:
        raise ValueError(
            "Pruned theta length does not match rebuilt layout: "
            f"got {theta.size}, expected {expected}."
        )
    old_keep_labels = tuple(state.runtime_coordinate_labels[index] for index in keep_indices)
    new_labels = runtime_coordinate_labels(layout, parameterization_mode=state.parameterization_mode)
    if old_keep_labels != new_labels:
        raise ValueError("Pruned layout does not preserve remaining runtime coordinates.")

    executor = _executor_for_terms(
        terms,
        layout,
        parameterization_mode=state.parameterization_mode,
    )
    psi_initial = _normalize_state(
        executor.prepare_state(theta, np.asarray(state.psi_ref, dtype=complex).reshape(-1)),
        name="psi_initial",
    )
    return APMcLachlanState(
        terms=terms,
        layout=layout,
        theta_runtime=theta,
        psi_ref=state.psi_ref,
        psi_initial=psi_initial,
        executor=executor,
        static_hamiltonian=state.static_hamiltonian,
        resolved_problem=state.resolved_problem,
        parameterization_mode=state.parameterization_mode,
        exact_energy=state.exact_energy,
        candidate_pool_terms=state.candidate_pool_terms,
        candidate_pool_source=state.candidate_pool_source,
        provenance=state.provenance,
        extensions=state.extensions,
    )


def _runtime_indices_for_labels(state: APMcLachlanState, labels: set[str]) -> tuple[int, ...]:
    if state.parameterization_mode == AP_PARAMETERIZATION_LOGICAL_SHARED:
        out = tuple(
            index
            for index, term in enumerate(state.terms)
            if str(getattr(term, "label", "")) in labels
        )
        return tuple(sorted(out))
    by_label = runtime_indices_by_block_label(state.layout)
    out: list[int] = []
    for label in labels:
        out.extend(int(index) for index in by_label.get(str(label), ()))
    return tuple(sorted(set(out)))


def load_ap_mclachlan_state(
    artifact_json: str | Path,
    *,
    loader_mode: str | None = None,
    tag: str | None = None,
    generator_family: str = "match_adapt",
    fallback_family: str = "full_meta",
    parameterization_mode: str = AP_PARAMETERIZATION_PER_PAULI_TERM,
) -> APMcLachlanState:
    runtime_input = load_scaffold_runtime_input(
        artifact_json,
        loader_mode=loader_mode,
        tag=tag,
        generator_family=generator_family,
        fallback_family=fallback_family,
    )
    return state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode=parameterization_mode,
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    return out if np.isfinite(out) else None


__all__ = [
    "AP_MCLACHLAN_STATE_SCHEMA_V1",
    "AP_PARAMETERIZATION_LOGICAL_SHARED",
    "AP_PARAMETERIZATION_PER_PAULI_TERM",
    "AP_SUPPORTED_PARAMETERIZATION_MODES",
    "APMcLachlanStateParityError",
    "APMcLachlanState",
    "RuntimeCoordinateRecord",
    "load_ap_mclachlan_state",
    "normalize_parameterization_mode",
    "parameterization_mode_label",
    "runtime_coordinate_records",
    "runtime_coordinate_labels",
    "runtime_indices_by_block_label",
    "state_from_scaffold_runtime_input",
    "state_with_appended_runtime_coordinates",
    "state_with_appended_terms",
    "state_with_runtime_coordinate_patch",
    "state_without_runtime_indices",
    "state_without_term_labels",
    "validate_runtime_state_parity",
]

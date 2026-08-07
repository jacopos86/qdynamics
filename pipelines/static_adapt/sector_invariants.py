"""Executable conserved-sector invariants for static adaptive ansatz runs.

The problem registry already declares the comparison sector.  This module
turns the fixed-count part of that declaration into runtime checks without
branching on a Hamiltonian family name.  In particular, it distinguishes a
logical generator from its Pauli implementation coordinates: a grouped
generator may commute with a conserved count even though its individual Pauli
components do not.  Such a generator must never receive independent component
angles.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pipelines.contracts.problem import (
    FixedCountConstraint,
    ResolvedProblemContext,
)
from src.quantum.hubbard_latex_python_pairs import SPIN_DN, SPIN_UP, mode_index
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


@dataclass(frozen=True)
class FixedCountQubitGroup:
    """One registry fixed-count constraint resolved to physical qubits."""

    quantity: str
    target: int
    qubits: tuple[int, ...]
    scope: str


def _fermion_block(resolved_problem: ResolvedProblemContext) -> Any | None:
    named = resolved_problem.layout.block("fermion")
    if named is not None:
        return named
    fermion_blocks = [
        block
        for block in resolved_problem.layout.blocks
        if str(block.kind).strip().lower() == "fermion"
    ]
    return fermion_blocks[0] if len(fermion_blocks) == 1 else None


def resolve_fixed_count_qubit_groups(
    resolved_problem: ResolvedProblemContext,
) -> tuple[tuple[FixedCountQubitGroup, ...], tuple[str, ...]]:
    """Resolve declared fixed counts to qubit groups.

    Unknown fixed-count quantities are reported instead of silently treated as
    verified.  Truncation constraints are intentionally outside this contract:
    they describe the finite computational representation, not a conserved
    phonon-number law.
    """

    block = _fermion_block(resolved_problem)
    groups: list[FixedCountQubitGroup] = []
    unsupported: list[str] = []
    for constraint in resolved_problem.sector.constraints:
        if not isinstance(constraint, FixedCountConstraint):
            continue
        quantity = str(constraint.quantity).strip().lower()
        if block is None:
            unsupported.append(quantity)
            continue
        start = int(block.start_qubit)
        stop = int(block.stop_qubit)
        if quantity in {"n_f", "n_fermion", "fermion_number", "particle_number"}:
            qubits = tuple(range(start, stop))
        elif quantity in {"n_up", "n_alpha"}:
            qubits = tuple(
                start
                + int(
                    mode_index(
                        site,
                        SPIN_UP,
                        indexing=str(resolved_problem.layout.ordering),
                        n_sites=int(resolved_problem.request.num_sites),
                    )
                )
                for site in range(int(resolved_problem.request.num_sites))
            )
        elif quantity in {"n_dn", "n_down", "n_beta"}:
            qubits = tuple(
                start
                + int(
                    mode_index(
                        site,
                        SPIN_DN,
                        indexing=str(resolved_problem.layout.ordering),
                        n_sites=int(resolved_problem.request.num_sites),
                    )
                )
                for site in range(int(resolved_problem.request.num_sites))
            )
        else:
            unsupported.append(quantity)
            continue
        if not qubits or min(qubits) < start or max(qubits) >= stop:
            unsupported.append(quantity)
            continue
        groups.append(
            FixedCountQubitGroup(
                quantity=quantity,
                target=int(constraint.value),
                qubits=qubits,
                scope=str(constraint.scope),
            )
        )
    return tuple(groups), tuple(sorted(set(unsupported)))


def _count_operator(*, nq: int, qubits: Sequence[int]) -> PauliPolynomial:
    terms: list[PauliTerm] = []
    for qubit in qubits:
        q = int(qubit)
        if q < 0 or q >= int(nq):
            raise ValueError(f"count-operator qubit {q} is outside nq={nq}.")
        word = ["e"] * int(nq)
        word[int(nq) - 1 - q] = "z"
        # The identity part of n_q=(I-Z_q)/2 commutes and can be omitted.
        terms.append(PauliTerm(int(nq), ps="".join(word), pc=-0.5))
    return PauliPolynomial("JW", terms)


_PAULI_PRODUCT: dict[tuple[str, str], tuple[str, complex]] = {
    ("x", "x"): ("e", 1.0 + 0.0j),
    ("y", "y"): ("e", 1.0 + 0.0j),
    ("z", "z"): ("e", 1.0 + 0.0j),
    ("x", "y"): ("z", 1.0j),
    ("y", "x"): ("z", -1.0j),
    ("y", "z"): ("x", 1.0j),
    ("z", "y"): ("x", -1.0j),
    ("z", "x"): ("y", 1.0j),
    ("x", "z"): ("y", -1.0j),
}


def _canonical_pauli_word(word: str) -> str:
    normalized = str(word).strip().lower().replace("i", "e")
    unsupported = sorted(set(normalized) - {"e", "x", "y", "z"})
    if unsupported:
        raise ValueError(
            f"Unsupported Pauli symbols {unsupported!r} in word {word!r}."
        )
    return normalized


def _multiply_pauli_words(left: str, right: str) -> tuple[str, complex]:
    lhs = _canonical_pauli_word(left)
    rhs = _canonical_pauli_word(right)
    if len(lhs) != len(rhs):
        raise ValueError(
            f"Cannot multiply Pauli words with different lengths: {len(lhs)} != {len(rhs)}."
        )
    out: list[str] = []
    phase = 1.0 + 0.0j
    for left_symbol, right_symbol in zip(lhs, rhs):
        if left_symbol == "e":
            out.append(right_symbol)
        elif right_symbol == "e":
            out.append(left_symbol)
        else:
            symbol, local_phase = _PAULI_PRODUCT[(left_symbol, right_symbol)]
            out.append(symbol)
            phase *= local_phase
    return "".join(out), complex(phase)


def _commutator_l1_norm(left: PauliPolynomial, right: PauliPolynomial) -> float:
    """Return a cancellation-stable coefficient L1 norm of ``[left, right]``."""

    contributions: dict[str, list[complex]] = defaultdict(list)
    left_terms = tuple(left.return_polynomial())
    right_terms = tuple(right.return_polynomial())
    for left_term in left_terms:
        left_word = _canonical_pauli_word(str(left_term.pw2strng()))
        left_coeff = complex(left_term.p_coeff)
        for right_term in right_terms:
            right_word = _canonical_pauli_word(str(right_term.pw2strng()))
            if int(left_term.nqubit()) != int(right_term.nqubit()):
                raise ValueError(
                    "Cannot commute Pauli terms with different qubit counts: "
                    f"{left_term.nqubit()} != {right_term.nqubit()}."
                )
            right_coeff = complex(right_term.p_coeff)
            lr_word, lr_phase = _multiply_pauli_words(left_word, right_word)
            rl_word, rl_phase = _multiply_pauli_words(right_word, left_word)
            contributions[lr_word].append(left_coeff * right_coeff * lr_phase)
            contributions[rl_word].append(-right_coeff * left_coeff * rl_phase)

    norm = 0.0
    for terms in contributions.values():
        coefficient = complex(
            math.fsum(float(value.real) for value in terms),
            math.fsum(float(value.imag) for value in terms),
        )
        norm += abs(coefficient)
    return float(norm)


def _pauli_words_commute(left: str, right: str) -> bool:
    lhs = _canonical_pauli_word(left)
    rhs = _canonical_pauli_word(right)
    if len(lhs) != len(rhs):
        raise ValueError(
            f"Cannot compare Pauli words with different lengths: {len(lhs)} != {len(rhs)}."
        )
    anticommuting_positions = sum(
        1
        for left_symbol, right_symbol in zip(lhs, rhs)
        if left_symbol != "e"
        and right_symbol != "e"
        and left_symbol != right_symbol
    )
    return bool(anticommuting_positions % 2 == 0)


def audit_generator_sector_contract(
    term: Any,
    *,
    groups: Sequence[FixedCountQubitGroup],
    total_qubits: int,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Audit one logical generator and its independently addressable factors."""

    polynomial = getattr(term, "polynomial")
    components = list(polynomial.return_polynomial())
    if components and any(int(component.nqubit()) != int(total_qubits) for component in components):
        raise ValueError(
            f"Generator {getattr(term, 'label', '<unlabeled>')!r} has a qubit-count mismatch."
        )
    grouped_norms: dict[str, float] = {}
    component_max_norms: dict[str, float] = {}
    grouped_preserves = True
    componentwise_preserves = True
    for group in groups:
        count_operator = _count_operator(nq=int(total_qubits), qubits=group.qubits)
        grouped_norm = _commutator_l1_norm(count_operator, polynomial)
        component_norms = [
            _commutator_l1_norm(
                count_operator,
                PauliPolynomial(
                    "JW",
                    [
                        PauliTerm(
                            int(component.nqubit()),
                            ps=str(component.pw2strng()),
                            pc=complex(component.p_coeff),
                        )
                    ],
                ),
            )
            for component in components
        ]
        component_max = float(max(component_norms, default=0.0))
        grouped_norms[str(group.quantity)] = float(grouped_norm)
        component_max_norms[str(group.quantity)] = float(component_max)
        grouped_preserves = bool(grouped_preserves and grouped_norm <= float(tolerance))
        componentwise_preserves = bool(
            componentwise_preserves and component_max <= float(tolerance)
        )
    execution_mode = str(
        getattr(term, "execution_mode", "termwise_product") or "termwise_product"
    ).strip().lower()
    nonzero_component_indices = [
        int(index)
        for index, component in enumerate(components)
        if abs(complex(component.p_coeff)) > 0.0
    ]
    pairwise_component_pair_count = (
        len(nonzero_component_indices) * (len(nonzero_component_indices) - 1) // 2
    )
    noncommuting_component_pair_count = 0
    noncommuting_component_pairs_sample: list[list[int]] = []
    for left_offset, left_index in enumerate(nonzero_component_indices):
        left_word = str(components[left_index].pw2strng())
        for right_index in nonzero_component_indices[left_offset + 1 :]:
            right_word = str(components[right_index].pw2strng())
            if not _pauli_words_commute(left_word, right_word):
                noncommuting_component_pair_count += 1
                if len(noncommuting_component_pairs_sample) < 64:
                    noncommuting_component_pairs_sample.append(
                        [int(left_index), int(right_index)]
                    )
    all_nonzero_components_mutually_commute = bool(
        noncommuting_component_pair_count == 0
    )
    if execution_mode == "grouped_exact":
        execution_preserves = bool(grouped_preserves)
    elif execution_mode == "termwise_product":
        execution_preserves = bool(
            componentwise_preserves
            or (
                grouped_preserves
                and all_nonzero_components_mutually_commute
            )
        )
    else:
        # The runtime currently has only two declared execution modes.  An
        # unrecognized mode must not inherit a sector-safe classification.
        execution_preserves = False
    requires_logical_shared = bool(
        execution_mode == "grouped_exact"
        or (grouped_preserves and not componentwise_preserves)
    )
    return {
        "label": str(getattr(term, "label", "")),
        "execution_mode": execution_mode,
        "component_count": int(len(components)),
        "grouped_commutator_l1": grouped_norms,
        "max_component_commutator_l1": component_max_norms,
        "grouped_preserves_fixed_counts": bool(grouped_preserves),
        "components_individually_preserve_fixed_counts": bool(componentwise_preserves),
        "nonzero_component_count": int(len(nonzero_component_indices)),
        "all_nonzero_components_mutually_commute": bool(
            all_nonzero_components_mutually_commute
        ),
        "pairwise_component_pair_count": int(pairwise_component_pair_count),
        "pairwise_commuting_component_pair_count": int(
            pairwise_component_pair_count - noncommuting_component_pair_count
        ),
        "noncommuting_component_pair_count": int(
            noncommuting_component_pair_count
        ),
        "noncommuting_component_pairs_sample": noncommuting_component_pairs_sample,
        "execution_preserves_fixed_counts": bool(execution_preserves),
        "requires_logical_shared_parameterization": bool(requires_logical_shared),
    }


def audit_candidate_pool_sector_contract(
    terms: Sequence[Any],
    *,
    resolved_problem: ResolvedProblemContext,
    tolerance: float = 1e-10,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Audit the pool against the problem registry's fixed-count contract."""

    groups, unsupported = resolve_fixed_count_qubit_groups(resolved_problem)
    if not groups:
        has_unsupported = bool(unsupported)
        return {
            "schema": "static_adapt_generator_sector_contract_v1",
            "checked": False,
            "passed": bool(not has_unsupported),
            "skip_reason": (
                "unsupported_fixed_count_quantities"
                if has_unsupported
                else "no_fixed_count_constraints"
            ),
            "fixed_count_support_complete": bool(not has_unsupported),
            "unsupported_fixed_count_quantities": list(unsupported),
            "requires_logical_shared_parameterization": False,
            "generator_count": int(len(terms)),
            "grouped_violation_count": 0,
            "grouped_violation_indices": [],
            "execution_checked": False,
            "execution_passed": bool(not has_unsupported),
            "execution_violation_count": 0,
            "execution_violation_indices": [],
            "execution_violation_labels": [],
            "execution_violation_sample": [],
            "logical_shared_required_count": 0,
        }
    rows = []
    for pool_index, term in enumerate(terms):
        row = audit_generator_sector_contract(
            term,
            groups=groups,
            total_qubits=int(resolved_problem.layout.total_qubits),
            tolerance=float(tolerance),
        )
        row["pool_index"] = int(pool_index)
        rows.append(row)
    grouped_offenders = [
        row for row in rows if not bool(row["grouped_preserves_fixed_counts"])
    ]
    execution_offenders = [
        row for row in rows if not bool(row["execution_preserves_fixed_counts"])
    ]
    logical_shared_rows = [
        row for row in rows if bool(row["requires_logical_shared_parameterization"])
    ]
    return {
        "schema": "static_adapt_generator_sector_contract_v1",
        "checked": True,
        # ``passed`` retains the algebraic grouped-generator contract because
        # this audit API is also used for the runtime Hamiltonian, which is not
        # executed as a termwise ansatz product.  Candidate execution safety is
        # the separate, fail-closed ``execution_passed`` contract below.
        "passed": bool(not unsupported and not grouped_offenders),
        "skip_reason": None,
        "fixed_count_support_complete": bool(not unsupported),
        "fixed_count_constraints": [
            {
                "quantity": group.quantity,
                "target": int(group.target),
                "qubits": [int(qubit) for qubit in group.qubits],
                "scope": group.scope,
            }
            for group in groups
        ],
        "unsupported_fixed_count_quantities": list(unsupported),
        "generator_count": int(len(rows)),
        "grouped_violation_count": int(len(grouped_offenders)),
        "grouped_violation_indices": [
            int(row["pool_index"]) for row in grouped_offenders
        ],
        "grouped_violation_labels": [
            str(row["label"]) for row in grouped_offenders
        ],
        "grouped_violation_sample": grouped_offenders[: int(sample_limit)],
        "execution_checked": True,
        "execution_passed": bool(not unsupported and not execution_offenders),
        "execution_violation_count": int(len(execution_offenders)),
        "execution_violation_indices": [
            int(row["pool_index"]) for row in execution_offenders
        ],
        "execution_violation_labels": [
            str(row["label"]) for row in execution_offenders
        ],
        "execution_violation_sample": execution_offenders[: int(sample_limit)],
        "logical_shared_required_count": int(len(logical_shared_rows)),
        "logical_shared_required_sample": logical_shared_rows[: int(sample_limit)],
        "requires_logical_shared_parameterization": bool(logical_shared_rows),
        "tolerance": float(tolerance),
    }


class FixedCountSectorStateAuditor:
    """Vectorized norm and fixed-count-sector audit for prepared states."""

    def __init__(
        self,
        resolved_problem: ResolvedProblemContext,
        *,
        norm_tolerance: float = 1e-9,
        sector_tolerance: float = 1e-9,
    ) -> None:
        self.resolved_problem = resolved_problem
        self.norm_tolerance = float(norm_tolerance)
        self.sector_tolerance = float(sector_tolerance)
        self.groups, self.unsupported = resolve_fixed_count_qubit_groups(resolved_problem)
        self.dimension = 1 << int(resolved_problem.layout.total_qubits)
        basis = np.arange(self.dimension, dtype=np.uint64)
        self._eigenvalues = tuple(
            sum(
                ((basis >> np.uint64(qubit)) & np.uint64(1)).astype(np.int16)
                for qubit in group.qubits
            )
            for group in self.groups
        )
        joint_mask = np.ones(self.dimension, dtype=bool)
        for group, eigenvalues in zip(self.groups, self._eigenvalues):
            joint_mask &= eigenvalues == int(group.target)
        self._joint_target_mask = joint_mask
        self._joint_target_indices = np.flatnonzero(joint_mask)
        self._joint_target_indices.flags.writeable = False

    @property
    def joint_target_indices(self) -> np.ndarray:
        """Return the immutable basis indices for the declared fixed-count sector."""

        return self._joint_target_indices

    def _coerce_state(self, state: np.ndarray, *, source: str) -> np.ndarray:
        psi = np.asarray(state, dtype=complex).reshape(-1)
        if int(psi.size) != int(self.dimension):
            raise ValueError(
                f"State dimension mismatch at {source}: got {psi.size}, expected {self.dimension}."
            )
        return psi

    def _lightweight_metrics(
        self,
        state: np.ndarray,
        *,
        source: str,
    ) -> tuple[float, float, float | None]:
        psi = self._coerce_state(state, source=source)
        norm_sq = float(np.vdot(psi, psi).real)
        norm_error = float(abs(norm_sq - 1.0))
        if not self.groups:
            return norm_sq, norm_error, None
        denominator = norm_sq if norm_sq > 0.0 else 1.0
        target_state = psi[self._joint_target_indices]
        target_weight = float(np.vdot(target_state, target_state).real)
        joint_probability = float(target_weight / denominator)
        return norm_sq, norm_error, joint_probability

    def assert_valid_fast(self, state: np.ndarray, *, source: str) -> None:
        """Assert norm and joint-sector membership without detailed moments.

        The joint target mask is compiled once in ``__init__``.  This method is
        suitable for optimizer state-preparation hot paths; call ``audit`` or
        ``assert_valid`` at checkpoints when per-constraint moments are needed.
        """

        if self.unsupported:
            raise RuntimeError(
                "Cannot validate the declared problem-sector contract because "
                "fixed-count quantities are unsupported: "
                f"{list(self.unsupported)!r} at {source}."
            )
        norm_sq, norm_error, joint_probability = self._lightweight_metrics(
            state,
            source=source,
        )
        passed = bool(
            norm_error <= self.norm_tolerance
            and (
                joint_probability is None
                or joint_probability >= 1.0 - self.sector_tolerance
            )
        )
        if not passed:
            raise RuntimeError(
                "Prepared state violated the declared problem-sector contract at "
                f"{source}: norm_squared={norm_sq:.16g}, "
                f"norm_error={norm_error:.3e}, "
                f"joint_target_sector_probability={joint_probability!r}, "
                f"norm_tolerance={self.norm_tolerance:.3e}, "
                f"sector_tolerance={self.sector_tolerance:.3e}."
            )

    def audit(self, state: np.ndarray, *, source: str) -> dict[str, Any]:
        psi = self._coerce_state(state, source=source)
        probabilities = np.abs(psi) ** 2
        norm_sq = float(np.sum(probabilities))
        norm_error = float(abs(norm_sq - 1.0))
        denominator = norm_sq if norm_sq > 0.0 else 1.0
        rows: list[dict[str, Any]] = []
        for group, eigenvalues in zip(self.groups, self._eigenvalues):
            expectation = float(np.dot(probabilities, eigenvalues) / denominator)
            variance = float(
                np.dot(probabilities, (eigenvalues - expectation) ** 2) / denominator
            )
            target_mask = eigenvalues == int(group.target)
            target_probability = float(np.sum(probabilities[target_mask]) / denominator)
            rows.append(
                {
                    "quantity": group.quantity,
                    "target": int(group.target),
                    "expectation": expectation,
                    "variance": variance,
                    "target_probability": target_probability,
                }
            )
        joint_probability = (
            float(np.sum(probabilities[self._joint_target_mask]) / denominator)
            if self.groups
            else None
        )
        passed = bool(
            not self.unsupported
            and norm_error <= self.norm_tolerance
            and (
                joint_probability is None
                or joint_probability >= 1.0 - self.sector_tolerance
            )
        )
        return {
            "schema": "static_adapt_state_sector_contract_v1",
            "source": str(source),
            "checked": bool(self.groups),
            "passed": passed,
            "fixed_count_support_complete": bool(not self.unsupported),
            "state_norm": float(np.sqrt(max(norm_sq, 0.0))),
            "state_norm_squared": norm_sq,
            "norm_error": norm_error,
            "fixed_count_constraints": rows,
            "joint_target_sector_probability": joint_probability,
            "joint_target_sector_illegal_probability": (
                None
                if joint_probability is None
                else float(max(0.0, 1.0 - joint_probability))
            ),
            "unsupported_fixed_count_quantities": list(self.unsupported),
            "norm_tolerance": self.norm_tolerance,
            "sector_tolerance": self.sector_tolerance,
        }

    def assert_valid(self, state: np.ndarray, *, source: str) -> dict[str, Any]:
        audit = self.audit(state, source=source)
        if not bool(audit["passed"]):
            raise RuntimeError(
                "Prepared state violated the declared problem-sector contract at "
                f"{source}: {audit!r}"
            )
        return audit


def audit_strict_state_replay(
    expected_state: np.ndarray,
    replayed_state: np.ndarray,
    *,
    source: str,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Compare two statevectors up to a physically irrelevant global phase."""

    expected = np.asarray(expected_state, dtype=complex).reshape(-1)
    replayed = np.asarray(replayed_state, dtype=complex).reshape(-1)
    if expected.shape != replayed.shape:
        raise ValueError(
            f"Strict replay shape mismatch at {source}: {expected.shape} != {replayed.shape}."
        )
    overlap = complex(np.vdot(expected, replayed))
    fidelity = float(abs(overlap) ** 2)
    phase = np.exp(-1.0j * np.angle(overlap)) if abs(overlap) > 0.0 else 1.0 + 0.0j
    phase_aligned_l2 = float(np.linalg.norm(expected - phase * replayed))
    passed = bool(phase_aligned_l2 <= float(tolerance))
    return {
        "schema": "static_adapt_strict_state_replay_v1",
        "source": str(source),
        "passed": passed,
        "fidelity": fidelity,
        "phase_aligned_l2": phase_aligned_l2,
        "tolerance": float(tolerance),
    }


__all__ = [
    "FixedCountQubitGroup",
    "FixedCountSectorStateAuditor",
    "audit_candidate_pool_sector_contract",
    "audit_generator_sector_contract",
    "audit_strict_state_replay",
    "resolve_fixed_count_qubit_groups",
]

"""Pure-Hubbard fixed-structure VQE construction for the Paper-II L4 lane.

This module intentionally does not route the pure-Hubbard problem through the
Hubbard--Holstein model wrapper.  The latter assigns at least one boson qubit
per site even for ``n_ph_max=0`` and therefore is not the frozen eight-qubit
Hubbard-L4 problem.

The ansatz is generator-derived in the narrow, reproducible sense used here:
its fixed ordered generators are the repository's deduplicated pure-Hubbard
``uccsd_qeb_hva_blocks`` pool, restricted to generators that preserve the
frozen ``(N_up, N_down)`` sector.  The repository pool order is retained and
the complete filtered pool is repeated as a layer.  No adaptive selection or
Qiskit compilation occurs in this construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

import numpy as np

from pipelines.static_adapt.builders.primitive_pools import (
    _build_hubbard_uccsd_qeb_hva_blocks_pool,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
from src.quantum.hubbard_latex_python_pairs import build_hubbard_hamiltonian
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    VQEResult,
    exact_ground_energy_sector,
    vqe_minimize,
)


PURE_HUBBARD_GENERATOR_POOL = "uccsd_qeb_hva_blocks"
PURE_HUBBARD_GENERATOR_FILTER = "fixed_spin_sector_logical_generators_v1"
PURE_HUBBARD_GENERATOR_ORDER = "repository_deduplicated_pool_order_v1"
PURE_HUBBARD_GENERATOR_EXECUTION = "grouped_exact_logical_shared_v1"

_QEB_PAIR_RE = re.compile(r"^qeb_pair\((\d+),(\d+)\)$")
_QEB_DOUBLE_RE = re.compile(
    r"^qeb_double\((\d+),(\d+)->(\d+),(\d+)\)$"
)


@dataclass(frozen=True)
class PureHubbardFixedVQEConfig:
    """Typed inputs for an independent pure-Hubbard fixed-structure seed."""

    num_sites: int = 4
    t: float = 1.0
    u: float = 4.0
    n_up: int = 2
    n_down: int = 2
    ordering: str = "blocked"
    boundary: str = "open"
    reps: int = 2
    restarts: int = 4
    optimizer_seed: int = 1_293_143_593
    optimizer_method: str = "SLSQP"
    maxiter: int = 1800
    initial_point_stddev: float = 0.3
    energy_gate: float = 1.0e-6

    def __post_init__(self) -> None:
        if int(self.num_sites) <= 0:
            raise ValueError("pure-Hubbard fixed VQE requires num_sites >= 1")
        if str(self.ordering) not in {"blocked", "interleaved"}:
            raise ValueError("ordering must be 'blocked' or 'interleaved'")
        if str(self.boundary) not in {"open", "periodic"}:
            raise ValueError("boundary must be 'open' or 'periodic'")
        if int(self.n_up) < 0 or int(self.n_down) < 0:
            raise ValueError("particle counts must be non-negative")
        if int(self.n_up) > int(self.num_sites) or int(self.n_down) > int(self.num_sites):
            raise ValueError("each spin-sector count must be <= num_sites")
        if int(self.reps) <= 0:
            raise ValueError("reps must be positive")
        if int(self.restarts) <= 0:
            raise ValueError("restarts must be positive")
        if int(self.maxiter) <= 0:
            raise ValueError("maxiter must be positive")
        for name in ("t", "u", "initial_point_stddev", "energy_gate"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if float(self.initial_point_stddev) < 0.0:
            raise ValueError("initial_point_stddev must be non-negative")
        if float(self.energy_gate) <= 0.0:
            raise ValueError("energy_gate must be positive")

    @property
    def total_qubits(self) -> int:
        """Return the exact pure-fermion register size (no boson qubits)."""

        return 2 * int(self.num_sites)

    @property
    def num_particles(self) -> tuple[int, int]:
        return (int(self.n_up), int(self.n_down))

    @property
    def pbc(self) -> bool:
        return str(self.boundary) == "periodic"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "problem": "hubbard",
            "L": int(self.num_sites),
            "t": float(self.t),
            "u": float(self.u),
            "v": 0.0,
            "ordering": str(self.ordering),
            "boundary": str(self.boundary),
            "sector_n_up": int(self.n_up),
            "sector_n_down": int(self.n_down),
            "total_qubits": int(self.total_qubits),
            "initializer": "generator_derived_fixed_structure_vqe_v1",
            "generator_pool": PURE_HUBBARD_GENERATOR_POOL,
            "generator_filter": PURE_HUBBARD_GENERATOR_FILTER,
            "generator_order": PURE_HUBBARD_GENERATOR_ORDER,
            "generator_execution": PURE_HUBBARD_GENERATOR_EXECUTION,
            "reps": int(self.reps),
            "restarts": int(self.restarts),
            "optimizer_seed": int(self.optimizer_seed),
            "optimizer_method": str(self.optimizer_method),
            "maxiter": int(self.maxiter),
            "initial_point_stddev": float(self.initial_point_stddev),
            "energy_gate": float(self.energy_gate),
            "qiskit_compilation": False,
        }


@dataclass(frozen=True)
class PureHubbardFixedVQEProblem:
    """Resolved pure-Hubbard Hamiltonian, sector, reference, and ansatz."""

    config: PureHubbardFixedVQEConfig
    hamiltonian: PauliPolynomial
    ansatz: CompiledAnsatzExecutor
    generator_terms: tuple[AnsatzTerm, ...]
    unique_generator_count: int
    generator_pool_sha256: str
    sector_dimension: int
    reference_state: np.ndarray
    exact_energy: float


@dataclass(frozen=True)
class PureHubbardFixedVQEResult:
    """Optimizer result plus the frozen seed-energy qualification decision."""

    problem: PureHubbardFixedVQEProblem
    optimizer_result: VQEResult
    absolute_energy_error: float
    energy_gate: float
    qualified: bool


def _spin_block(qubit: int, *, num_sites: int) -> int:
    qubit_i = int(qubit)
    if qubit_i < 0 or qubit_i >= 2 * int(num_sites):
        raise ValueError("generator label contains an out-of-range qubit")
    return 0 if qubit_i < int(num_sites) else 1


def _preserves_fixed_spin_sector(label: str, *, num_sites: int) -> bool:
    """Classify the repository's typed Hubbard generator labels.

    UCCSD labels are constructed with explicit alpha/beta excitation channels;
    Hubbard HVA blocks commute with both spin-number operators.  QEB labels are
    retained only when their sources and targets carry identical spin-block
    multiplicities.  Unknown labels fail closed.
    """

    value = str(label)
    if value.startswith(("uccsd_sing(alpha:", "uccsd_sing(beta:")):
        return True
    if value.startswith(("uccsd_dbl(aa:", "uccsd_dbl(bb:", "uccsd_dbl(ab:")):
        return True
    if value in {"hva_block::hop_layer", "hva_block::onsite_layer"}:
        return True
    pair = _QEB_PAIR_RE.fullmatch(value)
    if pair is not None:
        left, right = (int(x) for x in pair.groups())
        return _spin_block(left, num_sites=num_sites) == _spin_block(
            right, num_sites=num_sites
        )
    double = _QEB_DOUBLE_RE.fullmatch(value)
    if double is not None:
        source_a, source_b, target_a, target_b = (int(x) for x in double.groups())
        source_spins = sorted(
            (
                _spin_block(source_a, num_sites=num_sites),
                _spin_block(source_b, num_sites=num_sites),
            )
        )
        target_spins = sorted(
            (
                _spin_block(target_a, num_sites=num_sites),
                _spin_block(target_b, num_sites=num_sites),
            )
        )
        return source_spins == target_spins
    return False


def _polynomial_payload(polynomial: PauliPolynomial) -> list[dict[str, Any]]:
    rows = [
        {
            "pauli_exyz": str(term.pw2strng()),
            "coefficient_real": float(complex(term.p_coeff).real),
            "coefficient_imag": float(complex(term.p_coeff).imag),
        }
        for term in polynomial.return_polynomial()
    ]
    return sorted(rows, key=lambda row: str(row["pauli_exyz"]))


def _generator_pool_digest(terms: tuple[AnsatzTerm, ...]) -> str:
    payload = [
        {
            "label": str(term.label),
            "execution_mode": str(term.execution_mode),
            "polynomial": _polynomial_payload(term.polynomial),
        }
        for term in terms
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_spin_sector_preserving_hubbard_generator_pool(
    config: PureHubbardFixedVQEConfig,
) -> tuple[AnsatzTerm, ...]:
    """Return the canonical ordered pool after the hard fixed-spin filter.

    Every retained logical generator is executed as one exact exponential.
    This matters because individual Pauli factors of an excitation generator
    need not preserve ``(N_up, N_down)`` even though their grouped polynomial
    does.  The eight-qubit Hubbard register is below the repository executor's
    grouped-exact dimension limit.
    """

    raw_pool = _build_hubbard_uccsd_qeb_hva_blocks_pool(
        num_sites=int(config.num_sites),
        num_particles=config.num_particles,
        ordering=str(config.ordering),
        t=float(config.t),
        u=float(config.u),
        dv=0.0,
        boundary=str(config.boundary),
    )
    retained: list[AnsatzTerm] = []
    for term in raw_pool:
        if not _preserves_fixed_spin_sector(
            str(term.label), num_sites=int(config.num_sites)
        ):
            continue
        retained.append(
            AnsatzTerm(
                label=str(term.label),
                polynomial=term.polynomial,
                execution_mode="grouped_exact",
            )
        )
    if not retained:
        raise ValueError("pure-Hubbard fixed VQE found no sector-preserving generators")
    return tuple(retained)


def build_pure_hubbard_fixed_vqe_problem(
    config: PureHubbardFixedVQEConfig,
) -> PureHubbardFixedVQEProblem:
    """Resolve the pure-Hubbard fixed-structure VQE problem exactly once."""

    hamiltonian = build_hubbard_hamiltonian(
        dims=int(config.num_sites),
        t=float(config.t),
        U=float(config.u),
        v=0.0,
        repr_mode="JW",
        indexing=str(config.ordering),
        pbc=bool(config.pbc),
    )
    unique_generators = build_spin_sector_preserving_hubbard_generator_pool(config)
    layered_generators = tuple(
        AnsatzTerm(
            label=f"layer::{layer_index:02d}::{term.label}",
            polynomial=term.polynomial,
            execution_mode="grouped_exact",
        )
        for layer_index in range(int(config.reps))
        for term in unique_generators
    )
    ansatz = CompiledAnsatzExecutor(
        layered_generators,
        parameterization_mode="logical_shared",
    )
    reference_state = hartree_fock_statevector(
        int(config.num_sites),
        config.num_particles,
        indexing=str(config.ordering),
    )
    exact_energy = exact_ground_energy_sector(
        hamiltonian,
        num_sites=int(config.num_sites),
        num_particles=config.num_particles,
        indexing=str(config.ordering),
    )

    expected_dimension = 1 << int(config.total_qubits)
    if int(reference_state.size) != expected_dimension:
        raise ValueError(
            "pure-Hubbard reference-state dimension mismatch: "
            f"expected {expected_dimension}, got {int(reference_state.size)}"
        )
    for term in hamiltonian.return_polynomial():
        if int(term.nqubit()) != int(config.total_qubits):
            raise ValueError("pure-Hubbard Hamiltonian contains a wrong-width Pauli term")
    for generator in layered_generators:
        for term in generator.polynomial.return_polynomial():
            if int(term.nqubit()) != int(config.total_qubits):
                raise ValueError("pure-Hubbard ansatz contains a wrong-width Pauli term")

    return PureHubbardFixedVQEProblem(
        config=config,
        hamiltonian=hamiltonian,
        ansatz=ansatz,
        generator_terms=layered_generators,
        unique_generator_count=int(len(unique_generators)),
        generator_pool_sha256=_generator_pool_digest(unique_generators),
        sector_dimension=int(
            math.comb(int(config.num_sites), int(config.n_up))
            * math.comb(int(config.num_sites), int(config.n_down))
        ),
        reference_state=np.asarray(reference_state, dtype=complex),
        exact_energy=float(exact_energy),
    )


def qualify_pure_hubbard_fixed_vqe_result(
    problem: PureHubbardFixedVQEProblem,
    optimizer_result: VQEResult,
) -> PureHubbardFixedVQEResult:
    """Apply the source-locked absolute seed-energy gate."""

    error = abs(float(optimizer_result.energy) - float(problem.exact_energy))
    gate = float(problem.config.energy_gate)
    return PureHubbardFixedVQEResult(
        problem=problem,
        optimizer_result=optimizer_result,
        absolute_energy_error=float(error),
        energy_gate=gate,
        qualified=bool(np.isfinite(error) and error <= gate),
    )


def run_pure_hubbard_fixed_vqe(
    config: PureHubbardFixedVQEConfig,
) -> PureHubbardFixedVQEResult:
    """Run the generic repository VQE driver on the pure-Hubbard problem."""

    problem = build_pure_hubbard_fixed_vqe_problem(config)
    optimizer_result = vqe_minimize(
        problem.hamiltonian,
        problem.ansatz,
        problem.reference_state,
        restarts=int(config.restarts),
        seed=int(config.optimizer_seed),
        initial_point_stddev=float(config.initial_point_stddev),
        method=str(config.optimizer_method),
        maxiter=int(config.maxiter),
    )
    return qualify_pure_hubbard_fixed_vqe_result(problem, optimizer_result)


__all__ = [
    "PureHubbardFixedVQEConfig",
    "PureHubbardFixedVQEProblem",
    "PureHubbardFixedVQEResult",
    "build_spin_sector_preserving_hubbard_generator_pool",
    "build_pure_hubbard_fixed_vqe_problem",
    "qualify_pure_hubbard_fixed_vqe_result",
    "run_pure_hubbard_fixed_vqe",
]

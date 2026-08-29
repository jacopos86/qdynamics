#!/usr/bin/env python3
"""Paper-III problem adapter for the Peierls--Hubbard dimer.

This promotes the already-tested ``peierls_v1`` pilot Hamiltonian and pool
into the same problem interface consumed by the locked selected-QSE growth
pipeline and the direct-Davidson comparator.  Exact diagonalization is used
only to prepare the benchmark ground state and to score the completed routes;
no exact excited-state information enters selection, exchange, or stopping.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pipelines.exact_bench.paper_iii_qse_peierls_pilot import (
    PEIERLS_REGIMES,
    _DN_MODES,
    _N_PH_MAX,
    _NQ,
    _UP_MODES,
    _dense_hamiltonian,
    build_peierls_hamiltonian,
    build_peierls_pool,
)

PEIERLS_STATIC_ADAPTER_SCHEMA_VERSION = "paper_iii_peierls_static_adapter_v1"
PEIERLS_MODEL_FAMILY = "peierls_hubbard_dimer_bond_phonon"
PEIERLS_POOL_POLICY_ID = "peierls_v1_macro_records"
PEIERLS_FIXED_CLASS_POLICY_ID = "peierls_linear_response_v1"


class PeierlsStaticAdapterError(ValueError):
    """Raised when a Peierls case or support policy is undeclared."""


@dataclass(frozen=True)
class PeierlsStaticKey:
    regime: str
    u: float
    g_ep: float
    n_ph_max: int = _N_PH_MAX
    n_qubits: int = _NQ

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": PEIERLS_STATIC_ADAPTER_SCHEMA_VERSION,
            "model_family": PEIERLS_MODEL_FAMILY,
            "regime": str(self.regime),
            "u": float(self.u),
            "g_ep": float(self.g_ep),
            "n_ph_max": int(self.n_ph_max),
            "n_qubits": int(self.n_qubits),
            "sector": {"n_up": 1, "n_down": 1},
            "jw_order": "spin_blocked_fermions_then_binary_bond_phonon",
        }

    def digest(self) -> str:
        blob = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def peierls_static_key(regime: str) -> PeierlsStaticKey:
    for case_name, u, g_ep in PEIERLS_REGIMES:
        if str(case_name) == str(regime):
            return PeierlsStaticKey(regime=str(case_name), u=float(u), g_ep=float(g_ep))
    raise PeierlsStaticAdapterError(
        f"unknown Peierls regime {regime!r}; expected one of "
        f"{[row[0] for row in PEIERLS_REGIMES]!r}."
    )


def _ordered_pool_digest(basis: Sequence[Any]) -> str:
    blob = json.dumps([str(element.name) for element in basis], separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def peierls_sector_eigensystem(
    key: PeierlsStaticKey,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all exact sector eigenvalues and full-register eigenvectors."""

    hamiltonian = build_peierls_hamiltonian(u=float(key.u), g_ep=float(key.g_ep))
    dense = _dense_hamiltonian(hamiltonian, 1 << int(key.n_qubits))
    dim = int(dense.shape[0])
    occ_up = np.array(
        [sum((index >> qubit) & 1 for qubit in _UP_MODES) for index in range(dim)]
    )
    occ_down = np.array(
        [sum((index >> qubit) & 1 for qubit in _DN_MODES) for index in range(dim)]
    )
    sector = np.where((occ_up == 1) & (occ_down == 1))[0]
    restricted = dense[np.ix_(sector, sector)]
    energies, vectors = np.linalg.eigh(0.5 * (restricted + restricted.conj().T))
    embedded = np.zeros((dim, len(sector)), dtype=complex)
    embedded[sector, :] = vectors
    return np.asarray(energies, dtype=float), embedded


@dataclass(frozen=True)
class PeierlsStaticProblem:
    key: PeierlsStaticKey
    hamiltonian: Any
    basis: tuple[Any, ...]
    costs: tuple[float, ...]
    resources: tuple[dict[str, float], ...]
    ground: np.ndarray
    spectrum: tuple[float, ...]
    pool_digest: str
    pool_policy_id: str = PEIERLS_POOL_POLICY_ID

    @property
    def num_qubits(self) -> int:
        return int(self.key.n_qubits)

    @property
    def n_ph_max(self) -> int:
        return int(self.key.n_ph_max)

    @property
    def references(self) -> tuple[float, ...]:
        return tuple(self.spectrum[1:])

    @property
    def ground_energy(self) -> float:
        return float(self.spectrum[0])

    def resource_triple(self, indices: Sequence[int]) -> dict[str, float]:
        n2q = float(sum(self.resources[int(i)]["c_hat_2q"] for i in indices))
        d2q = float(sum(self.resources[int(i)]["c_hat_d"] for i in indices))
        one = float(sum(self.resources[int(i)]["c_hat_1q"] for i in indices))
        return {"n2q": n2q, "d2q": d2q, "dc": d2q + one}

    def estimator_cost(self, indices: Sequence[int]) -> dict[str, int]:
        from pipelines.exact_bench.paper_iii_qse_measurement_cost import (
            _element_words,
            _polynomial_words,
            _qwc_cover,
            _word_product,
        )

        ordered = [int(index) for index in indices]
        ham_words = _polynomial_words(self.hamiltonian)
        words = {
            index: _element_words(self.basis[index], nq=self.num_qubits)
            for index in ordered
        }
        overlap: set[str] = set()
        hamiltonian_words: set[str] = set()
        naive = 0
        for position, left in enumerate(ordered):
            for right in ordered[position:]:
                products = {
                    _word_product(a, b) for a in words[left] for b in words[right]
                }
                overlap |= products
                naive += len(products)
                h_products = {
                    _word_product(product, h)
                    for product in products
                    for h in ham_words
                }
                hamiltonian_words |= h_products
                naive += len(h_products)
        all_words = overlap | hamiltonian_words
        return {
            "pair_count": len(ordered) * (len(ordered) + 1) // 2,
            "naive_terms": int(naive),
            "distinct_words": len(all_words),
            "qwc_groups": _qwc_cover(sorted(all_words), nq=self.num_qubits),
        }

    def arm_receipt(self) -> dict[str, Any]:
        return {
            "model_family": PEIERLS_MODEL_FAMILY,
            "case_id": str(self.key.regime),
            "physics_digest": self.key.digest(),
            "pool_digest": str(self.pool_digest),
            "pool_policy_id": str(self.pool_policy_id),
            "pool_size": len(self.basis),
            "num_qubits": self.num_qubits,
            "ground_source": "exact_sector_benchmark_seed",
            "diagnostic_only": False,
            "paper_facing": True,
            "benchmark_scope": "isolates_static_qse",
            "uses_exact_excited_state_information_online": False,
            "granularity": "macro_records_uniform",
        }


def load_peierls_problem(
    regime: str, *, target_roots: int = 6
) -> PeierlsStaticProblem:
    from pipelines.qse_spectra.compiled_costs import (
        ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        annotate_basis_with_compiled_costs,
        resolve_cost_weights_preset,
    )
    from pipelines.qse_spectra.paper_iii_problem import assert_uniform_granularity

    key = peierls_static_key(regime)
    hamiltonian = build_peierls_hamiltonian(u=float(key.u), g_ep=float(key.g_ep))
    basis = tuple(build_peierls_pool())
    assert_uniform_granularity(basis)
    rows = annotate_basis_with_compiled_costs(
        basis,
        num_qubits=int(key.n_qubits),
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )
    resources = tuple(
        {
            "c_hat_2q": float(row.estimate.c_hat_2q),
            "c_hat_d": float(row.estimate.c_hat_d),
            "c_hat_1q": float(row.estimate.c_hat_1q),
        }
        for row in rows
    )
    energies, vectors = peierls_sector_eigensystem(key)
    count = int(target_roots) + 1
    if count > len(energies):
        raise PeierlsStaticAdapterError(
            f"requested {count} sector roots but only {len(energies)} exist."
        )
    return PeierlsStaticProblem(
        key=key,
        hamiltonian=hamiltonian,
        basis=basis,
        costs=tuple(float(row.scalarized_canonical_cost) for row in rows),
        resources=resources,
        ground=np.asarray(vectors[:, 0], dtype=complex),
        spectrum=tuple(float(value) for value in energies[:count]),
        pool_digest=_ordered_pool_digest(basis),
    )


__all__ = [
    "PEIERLS_FIXED_CLASS_POLICY_ID",
    "PEIERLS_MODEL_FAMILY",
    "PEIERLS_POOL_POLICY_ID",
    "PEIERLS_STATIC_ADAPTER_SCHEMA_VERSION",
    "PeierlsStaticAdapterError",
    "PeierlsStaticKey",
    "PeierlsStaticProblem",
    "load_peierls_problem",
    "peierls_sector_eigensystem",
    "peierls_static_key",
]

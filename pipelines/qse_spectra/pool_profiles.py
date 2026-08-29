"""Versioned Hubbard--Holstein record pools for Paper-III experiments.

The shared problem provider remains authoritative for the Hamiltonian,
prepared state, exact audit reference, and base macro alphabet. This module
adds declared, deterministic pool transformations without mutating or
silently reinterpreting that provider.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any

from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    annotate_basis_with_compiled_costs,
    resolve_cost_weights_preset,
)
from pipelines.qse_spectra.core import pauli_string_basis_element
from pipelines.qse_spectra.paper_iii_problem import PaperIIIProblem


HH_POOL_BASE_MACRO_V1 = "hh_full_meta_macro_v1"
HH_POOL_ENCODED_BOSON_COMPLETION_V1 = (
    "hh_full_meta_plus_encoded_boson_pauli_completion_v1"
)
DECLARED_HH_POOL_POLICIES = (
    HH_POOL_BASE_MACRO_V1,
    HH_POOL_ENCODED_BOSON_COMPLETION_V1,
)
_FERMION_QUBIT_COUNT = 4
_PROFILE_CACHE: dict[tuple[str, str, str], PaperIIIProblem] = {}


def _ordered_pool_digest(basis: tuple[Any, ...]) -> str:
    names = [str(element.name) for element in basis]
    blob = json.dumps(names, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def encoded_boson_completion_elements(*, num_qubits: int) -> tuple[Any, ...]:
    """Return the declared sector-preserving encoded-boson completion.

    The first ``num_qubits - 4`` label positions are the encoded boson
    register in the Hubbard--Holstein layout. For every nonidentity boson
    Pauli word, the completion contains its tensor product with fermion
    identity and with each one-body fermion ``Z_j``. It therefore never
    flips a fermion occupation sector.
    """

    nq = int(num_qubits)
    boson_qubits = nq - _FERMION_QUBIT_COUNT
    if boson_qubits < 1:
        raise ValueError(
            "encoded-boson completion requires at least one boson qubit "
            f"besides the {_FERMION_QUBIT_COUNT} fermion qubits."
        )
    fermion_factors = ["e" * _FERMION_QUBIT_COUNT]
    for index in range(_FERMION_QUBIT_COUNT):
        chars = ["e"] * _FERMION_QUBIT_COUNT
        chars[index] = "z"
        fermion_factors.append("".join(chars))

    records: list[Any] = []
    identity_boson = "e" * boson_qubits
    for symbols in itertools.product("exyz", repeat=boson_qubits):
        boson_word = "".join(symbols)
        if boson_word == identity_boson:
            continue
        for factor_index, fermion_word in enumerate(fermion_factors):
            fermion_label = "I" if factor_index == 0 else f"Z{factor_index - 1}"
            records.append(
                pauli_string_basis_element(
                    boson_word + fermion_word,
                    nq=nq,
                    name=(
                        "encoded_boson_completion"
                        f"(pb={boson_word},fermion={fermion_label})"
                    ),
                    metadata={
                        "record_family": "encoded_boson_completion",
                        "pool_policy_id": HH_POOL_ENCODED_BOSON_COMPLETION_V1,
                        "boson_word": boson_word,
                        "fermion_factor": fermion_label,
                    },
                )
            )
    return tuple(records)


def apply_hh_pool_policy(
    problem: PaperIIIProblem,
    pool_policy_id: str,
) -> PaperIIIProblem:
    """Return ``problem`` under one declared fixed-pool policy."""

    policy = str(pool_policy_id)
    if policy == HH_POOL_BASE_MACRO_V1:
        return problem
    if policy != HH_POOL_ENCODED_BOSON_COMPLETION_V1:
        raise ValueError(
            f"unknown HH pool policy {policy!r}; expected one of "
            f"{list(DECLARED_HH_POOL_POLICIES)!r}."
        )

    key = (str(problem.reference_key), str(problem.pool_digest), policy)
    cached = _PROFILE_CACHE.get(key)
    if cached is not None:
        return cached

    completion = encoded_boson_completion_elements(num_qubits=problem.num_qubits)
    rows = annotate_basis_with_compiled_costs(
        completion,
        num_qubits=int(problem.num_qubits),
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )
    completion_costs = tuple(float(row.scalarized_canonical_cost) for row in rows)
    completion_resources = tuple(
        {
            "c_hat_2q": float(row.estimate.c_hat_2q),
            "c_hat_d": float(row.estimate.c_hat_d),
            "c_hat_1q": float(row.estimate.c_hat_1q),
        }
        for row in rows
    )
    basis = tuple(problem.basis) + completion
    profiled = PaperIIIProblem(
        regime=str(problem.regime),
        u=float(problem.u),
        g_ep=float(problem.g_ep),
        n_ph_max=int(problem.n_ph_max),
        num_qubits=int(problem.num_qubits),
        hamiltonian=problem.hamiltonian,
        basis=basis,
        costs=tuple(problem.costs) + completion_costs,
        resources=tuple(problem.resources) + completion_resources,
        ground=problem.ground,
        spectrum=tuple(problem.spectrum),
        pool_digest=_ordered_pool_digest(basis),
        reference_key=str(problem.reference_key),
    )
    _PROFILE_CACHE[key] = profiled
    return profiled


__all__ = [
    "DECLARED_HH_POOL_POLICIES",
    "HH_POOL_BASE_MACRO_V1",
    "HH_POOL_ENCODED_BOSON_COMPLETION_V1",
    "apply_hh_pool_policy",
    "encoded_boson_completion_elements",
]

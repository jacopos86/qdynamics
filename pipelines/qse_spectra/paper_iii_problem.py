#!/usr/bin/env python3
"""Single source of the Paper III problem: Hamiltonian, pool, costs, reference.

Every Paper III route must obtain its physics through this provider rather
than building its own. Three properties it exists to guarantee:

1. **One Hamiltonian and one operator alphabet per regime.** Drivers used to
   call the pool builder independently (the campaign called it twice per
   regime), so "identical alphabet" was an assumption. Here the pool is built
   once per process and handed out by reference.
2. **Provable alphabet identity.** Every problem carries ``pool_digest``, an
   ordered digest of the record names. Equal pool *sizes* do not prove equal
   pools; an arm receipt must carry this digest so a shared-alphabet claim is
   checkable after the fact.
3. **One cached exact reference.** The exact sector-restricted reference is
   content-addressed on disk, verified on read, and never recomputed per arm.

The provider also asserts **granularity homogeneity**: the selection route
never decomposes a macro record into Pauli children between phases, so a pool
mixing macro records with pre-split children would silently change what a
"record" costs. ``assert_uniform_granularity`` fails closed on that.

Reporting/diagnostic use only; never feeds controller decisions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_STORE = REPO_ROOT / "output/reference_store/paper_iii_exact_sector"

_PROBLEM_CACHE: dict[tuple[str, float, float, int, int], "PaperIIIProblem"] = {}


@dataclass(frozen=True)
class PaperIIIProblem:
    """One regime's physics, shared by every arm."""

    regime: str
    u: float
    g_ep: float
    n_ph_max: int
    num_qubits: int
    hamiltonian: Any
    basis: tuple[Any, ...]
    costs: tuple[float, ...]
    resources: tuple[dict[str, float], ...]
    ground: np.ndarray
    spectrum: tuple[float, ...]
    pool_digest: str
    reference_key: str

    @property
    def references(self) -> tuple[float, ...]:
        """Excitation energies E_1..E_R (the target window)."""

        return tuple(self.spectrum[1:])

    @property
    def ground_energy(self) -> float:
        return float(self.spectrum[0])

    def resource_triple(self, indices: Sequence[int]) -> dict[str, float]:
        """Deterministic graph-span proxy N2q / D2q / Dc for a support."""

        n2q = float(sum(self.resources[int(i)]["c_hat_2q"] for i in indices))
        d2q = float(sum(self.resources[int(i)]["c_hat_d"] for i in indices))
        one = float(sum(self.resources[int(i)]["c_hat_1q"] for i in indices))
        return {"n2q": n2q, "d2q": d2q, "dc": d2q + one}

    def arm_receipt(self) -> dict[str, Any]:
        """Provenance every arm must record, so alphabet identity is checkable."""

        return {
            "pool_digest": self.pool_digest,
            "pool_size": len(self.basis),
            "reference_key": self.reference_key,
            "num_qubits": int(self.num_qubits),
            "granularity": "macro_records_uniform",
        }


def _ordered_pool_digest(basis: Sequence[Any]) -> str:
    names = [str(element.name) for element in basis]
    blob = json.dumps(names, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def assert_uniform_granularity(basis: Sequence[Any]) -> None:
    """Fail closed if the pool mixes macro records with pre-split children.

    The selection route never decomposes a record between phases, so a mixed
    pool would make "one record" mean two different resource quantities.
    """

    child_like = [
        str(e.name) for e in basis
        if str(e.kind) == "pauli_string" and "::child" in str(e.name)
    ]
    if child_like:
        raise ValueError(
            "pool mixes pre-split Pauli children with macro records; the Paper III "
            f"route requires uniform granularity. Offenders: {child_like[:5]}"
        )


def _reference_identity(regime: str, u: float, g_ep: float, n_ph_max: int, count: int) -> dict[str, Any]:
    return {
        "family": "hh_l2_half_filled_11_sector",
        "regime": str(regime),
        "u": float(u),
        "g_ep": float(g_ep),
        "n_ph_max": int(n_ph_max),
        "count": int(count),
        "schema": "paper_iii_exact_sector_reference_v1",
    }


def _reference_key(identity: dict[str, Any]) -> str:
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def load_problem(
    *,
    regime: str,
    u: float,
    g_ep: float,
    n_ph_max: int,
    target_roots: int = 6,
) -> PaperIIIProblem:
    """Build (or return the cached) problem for one regime."""

    from pipelines.exact_bench.paper_iii_qse_paper_i_convention_sweep import (
        _build_regime_pool,
        _num_qubits,
    )
    from pipelines.exact_bench.paper_iii_qse_regime_frontier_sweep import (
        _dense_hamiltonian,
        _sector_spectrum,
    )
    from pipelines.qse_spectra.compiled_costs import (
        ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        annotate_basis_with_compiled_costs,
        resolve_cost_weights_preset,
    )

    cache_key = (str(regime), float(u), float(g_ep), int(n_ph_max), int(target_roots))
    hit = _PROBLEM_CACHE.get(cache_key)
    if hit is not None:
        return hit

    nq = _num_qubits(n_ph_max)
    hamiltonian, basis, _meta = _build_regime_pool(u=u, g_ep=g_ep, n_ph_max=n_ph_max)
    basis = tuple(basis)
    assert_uniform_granularity(basis)

    rows = annotate_basis_with_compiled_costs(
        basis,
        num_qubits=nq,
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )
    costs = tuple(float(r.scalarized_canonical_cost) for r in rows)
    resources = tuple(
        {
            "c_hat_2q": float(r.estimate.c_hat_2q),
            "c_hat_d": float(r.estimate.c_hat_d),
            "c_hat_1q": float(r.estimate.c_hat_1q),
        }
        for r in rows
    )

    count = int(target_roots) + 1
    identity = _reference_identity(regime, u, g_ep, n_ph_max, count)
    key = _reference_key(identity)
    path = REFERENCE_STORE / f"{key}.npz"
    if path.is_file():
        payload = np.load(path, allow_pickle=False)
        stored = json.loads(str(payload["identity_json"]))
        if stored != identity:
            raise RuntimeError(
                f"reference-store identity mismatch at {path}: {stored} != {identity}"
            )
        ground = np.asarray(payload["ground"], dtype=complex)
        spectrum = tuple(float(x) for x in payload["energies"])
    else:
        dense = _dense_hamiltonian(hamiltonian, 1 << nq)
        ground_arr, energies = _sector_spectrum(dense, count=count)
        del dense
        ground = np.asarray(ground_arr, dtype=complex)
        spectrum = tuple(float(x) for x in energies)
        REFERENCE_STORE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            ground=ground,
            energies=np.asarray(spectrum, dtype=float),
            identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
        )
        tmp.rename(path)

    problem = PaperIIIProblem(
        regime=str(regime),
        u=float(u),
        g_ep=float(g_ep),
        n_ph_max=int(n_ph_max),
        num_qubits=int(nq),
        hamiltonian=hamiltonian,
        basis=basis,
        costs=costs,
        resources=resources,
        ground=ground,
        spectrum=spectrum,
        pool_digest=_ordered_pool_digest(basis),
        reference_key=key,
    )
    _PROBLEM_CACHE[cache_key] = problem
    return problem


__all__ = [
    "PaperIIIProblem",
    "REFERENCE_STORE",
    "assert_uniform_granularity",
    "load_problem",
]

"""Deterministic non-oracular starts for the direct-Davidson comparator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.adaptive_qse_benchmark import hamiltonian_diagonal
from pipelines.qse_spectra.static_campaign import CONSTRUCTION_DIRECT_DAVIDSON


DAVIDSON_START_SCHEMA_VERSION = "paper_iii_direct_davidson_start_v1"
ORIGIN_FIXED_RECORD = "fixed_linear_response_record"
ORIGIN_COMPUTATIONAL_PROBE = "target_sector_computational_probe"


@dataclass(frozen=True)
class DavidsonSeedDirection:
    """One declared initial direction and the preparation cost it carries."""

    payload: Any
    label: str
    origin: str
    resources: Mapping[str, float]


@dataclass(frozen=True)
class DavidsonStart:
    """A complete direct-Davidson initialization block and audit receipt."""

    seed_directions: tuple[DavidsonSeedDirection, ...]
    carried_resources: Mapping[str, float]
    receipt: Mapping[str, Any]


def target_sector_indices(problem: Any) -> tuple[int, ...]:
    """Return computational indices in the problem's declared target sector."""

    key = getattr(problem, "key", None)
    if key is not None and hasattr(key, "n_bath"):
        from src.quantum.siam_model import sector_indices

        return tuple(int(index) for index in sector_indices(int(key.n_bath)))

    nq = int(getattr(problem, "num_qubits"))
    if not hasattr(problem, "n_ph_max") or nq < 4:
        raise ValueError(
            "direct-Davidson initialization does not know the declared sector "
            "for this problem family."
        )
    out = []
    for index in range(1 << nq):
        n_up = sum((index >> qubit) & 1 for qubit in (0, 1))
        n_down = sum((index >> qubit) & 1 for qubit in (2, 3))
        if n_up == 1 and n_down == 1:
            out.append(int(index))
    if not out:
        raise ValueError("declared HH (N_up,N_down)=(1,1) sector is empty.")
    return tuple(out)


def _record_resources(problem: Any, index: int) -> dict[str, float]:
    resources: Sequence[Mapping[str, Any]] = tuple(getattr(problem, "resources"))
    row = resources[int(index)]
    return {
        "n2q": float(row["c_hat_2q"]),
        "d2q": float(row["c_hat_d"]),
        "dc": float(row["c_hat_d"]) + float(row["c_hat_1q"]),
    }


def build_direct_davidson_start(
    problem: Any,
    *,
    target_root_count: int,
    fixed_record_indices: Sequence[int],
    computational_probe_count: int | None = None,
) -> DavidsonStart:
    """Build fixed-LR plus lowest-diagonal target-sector probes.

    Probe ordering is deterministic ``(H_ii, basis_index)``.  It uses neither
    exact roots nor eigenvectors.  The default block contains ``R+1`` probes,
    providing one oversampling direction beyond the requested excitation
    window while retaining the fixed linear-response warm start for HH.
    """

    roots = int(target_root_count)
    if roots <= 0:
        raise ValueError("target_root_count must be positive.")
    requested_probes = roots + 1 if computational_probe_count is None else int(
        computational_probe_count
    )
    if requested_probes < 0:
        raise ValueError("computational_probe_count must be non-negative.")

    nq = int(problem.num_qubits)
    dim = 1 << nq
    diagonal = hamiltonian_diagonal(problem.hamiltonian, num_qubits=nq)
    sector = target_sector_indices(problem)
    if any(index < 0 or index >= dim for index in sector):
        raise ValueError("target-sector index lies outside the problem Hilbert space.")
    ordered_sector = sorted(sector, key=lambda index: (float(diagonal[index]), int(index)))
    probe_indices = ordered_sector[: min(requested_probes, len(ordered_sector))]

    directions: list[DavidsonSeedDirection] = []
    record_indices = tuple(int(index) for index in fixed_record_indices)
    for index in record_indices:
        directions.append(
            DavidsonSeedDirection(
                payload=problem.basis[index],
                label=f"fixed_record::{index}",
                origin=ORIGIN_FIXED_RECORD,
                resources=_record_resources(problem, index),
            )
        )
    for index in probe_indices:
        vector = np.zeros(dim, dtype=complex)
        vector[int(index)] = 1.0
        directions.append(
            DavidsonSeedDirection(
                payload=vector,
                label=f"computational_probe::{int(index)}",
                origin=ORIGIN_COMPUTATIONAL_PROBE,
                resources={
                    "n2q": 0.0,
                    "d2q": 0.0,
                    "dc": float(int(index).bit_count()),
                },
            )
        )

    return DavidsonStart(
        seed_directions=tuple(directions),
        carried_resources={"n2q": 0.0, "d2q": 0.0, "dc": 0.0},
        receipt={
            "schema_version": DAVIDSON_START_SCHEMA_VERSION,
            "construction_policy_id": CONSTRUCTION_DIRECT_DAVIDSON,
            "policy": "fixed_lr_plus_lowest_diagonal_target_sector_probes",
            "ordering": "ascending_hamiltonian_diagonal_then_basis_index",
            "target_root_count": roots,
            "requested_computational_probe_count": requested_probes,
            "fixed_record_indices": list(record_indices),
            "computational_probe_indices": [int(index) for index in probe_indices],
            "declared_sector_size": len(sector),
            "uses_exact_reference": False,
        },
    )


__all__ = [
    "DAVIDSON_START_SCHEMA_VERSION",
    "DavidsonSeedDirection",
    "DavidsonStart",
    "ORIGIN_COMPUTATIONAL_PROBE",
    "ORIGIN_FIXED_RECORD",
    "build_direct_davidson_start",
    "target_sector_indices",
]

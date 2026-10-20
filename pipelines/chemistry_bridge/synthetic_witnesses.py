"""Synthetic fixtures for the bridge, labelled synthetic everywhere they appear.

None of these is a molecule, a benchmark, or Doron's output.  They exist so the
converter, the backend and the method runners can be exercised and verified
against analytic or independent witnesses before any real export arrives.

* ``contract_four_qubit_payload`` reproduces the strict-intake contract's
  section-5.4 fixture (m=2, (1,1), analytic 4x4 sector block with roots
  ``0.1 -/+ sqrt(785)/20``, ``-0.15``, ``-0.05``).
* ``random_symmetric_tensors`` builds real spatial tensors with the full
  eightfold chemists' symmetry from a seeded generator (a sum of symmetric
  pair-products, so the symmetry holds by construction).
* ``qdynamics_mapping_from_spatial`` interleaves spatial tensors exactly as the
  pinned producer's ``PySCFDriver.get_matrix_elements`` does and attaches the
  producer's metadata keys, giving a ``load_h5``-shaped mapping.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pipelines.chemistry_bridge.qdynamics_export import (
    EXPECTED_H1_CONVENTION,
    EXPECTED_H2_CONVENTION,
    closed_shell_determinant_energy,
)


CONTRACT_FOUR_QUBIT_ROOTS = (0.1 - np.sqrt(785.0) / 20.0, -0.15, -0.05, 0.1 + np.sqrt(785.0) / 20.0)


def contract_four_qubit_tensors() -> tuple[np.ndarray, np.ndarray, float, float]:
    """(h, g, scalar, hf_energy) of the contract's synthetic m=2 fixture."""

    eri = np.zeros((2, 2, 2, 2))
    for indices, value in [
        ((0, 0, 0, 0), 0.4), ((1, 1, 1, 1), 0.2),
        ((0, 0, 1, 1), 0.1), ((1, 1, 0, 0), 0.1),
        ((0, 1, 0, 1), 0.05), ((1, 0, 0, 1), 0.05),
        ((0, 1, 1, 0), 0.05), ((1, 0, 1, 0), 0.05),
    ]:
        eri[indices] = value
    h = np.array([[-1.0, 0.0], [0.0, 0.5]])
    return h, eri, 0.3, -1.3


def contract_four_qubit_payload() -> dict[str, Any]:
    h, g, scalar, hf = contract_four_qubit_tensors()
    return {
        "geometry_spec": "synthetic mathematical fixture",
        "basis": "synthetic orthonormal spatial basis",
        "charge": 0, "multiplicity": 1, "reference": "rhf",
        "n_spatial_orbitals": 2, "n_spin_orbitals": 4,
        "n_alpha": 1, "n_beta": 1,
        "hf_energy": hf, "nuclear_repulsion_energy": scalar,
        "one_body_integrals_mo": h.tolist(),
        "two_body_integrals_mo": g.tolist(),
    }


def random_symmetric_tensors(n_orbitals: int, *, seed: int, scale: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Real h (symmetric) and g with the eightfold symmetry, by construction."""

    rng = np.random.default_rng(int(seed))
    n = int(n_orbitals)
    h = rng.standard_normal((n, n))
    h = 0.5 * (h + h.T)
    g = np.zeros((n, n, n, n))
    for _ in range(max(2, n)):
        w = rng.standard_normal((n, n))
        w = 0.5 * (w + w.T)
        g += float(scale) * np.einsum("pq,rs->pqrs", w, w)
    g += float(scale) * np.einsum("pq,rs->pqrs", np.eye(n), np.eye(n))
    return h, g


def qdynamics_mapping_from_spatial(
    h_mo: np.ndarray,
    eri_mo: np.ndarray,
    *,
    nuclear_repulsion: float,
    n_occupied: int,
    e_scf: float | None = None,
    geometry: str = "synthetic",
    basis: str = "synthetic",
    extra_metadata: dict[str, Any] | None = None,
    include_mo_occ: bool = True,
) -> dict[str, Any]:
    """Interleave spatial tensors exactly like ``PySCFDriver.get_matrix_elements``.

    ``h1[0::2, 0::2] = h1[1::2, 1::2] = h_mo``;
    ``h2[s::2, s::2, s'::2, s'::2] = eri_mo`` for ``s, s'`` in ``{0, 1}``.
    ``e_scf`` defaults to the closed-shell determinant energy so the witness
    holds exactly; pass another value to build a negative control.
    """

    h = np.asarray(h_mo, dtype=np.float64)
    g = np.asarray(eri_mo, dtype=np.float64)
    n = int(h.shape[0])
    h1 = np.zeros((2 * n, 2 * n))
    h1[0::2, 0::2] = h
    h1[1::2, 1::2] = h
    h2 = np.zeros((2 * n, 2 * n, 2 * n, 2 * n))
    for spin_ij in range(2):
        for spin_kl in range(2):
            h2[spin_ij::2, spin_ij::2, spin_kl::2, spin_kl::2] = g
    witness = closed_shell_determinant_energy(float(nuclear_repulsion), h, g, int(n_occupied))
    metadata = {
        "mol_str": str(geometry),
        "basis": str(basis),
        "method": "RHF",
        "charge": 0,
        "spin": 0,
        "unit": "Angstrom",
        "xc": "",
        "e_scf": float(witness if e_scf is None else e_scf),
        "converged": True,
        "nelec": int(2 * n_occupied),
        "nmo": int(n),
        "ms2": 0,
        "e_converg": 1.0e-12,
        "d_converg": 1.0e-8,
        "max_iter": 100,
        "isotope_masses": "None",
        "h1_convention": EXPECTED_H1_CONVENTION,
        "h2_convention": EXPECTED_H2_CONVENTION,
        "h2_antisymmetrized": False,
    }
    metadata.update(dict(extra_metadata or {}))
    mapping: dict[str, Any] = {
        "metadata": metadata,
        "h1": h1,
        "h2": h2,
        "nuclear_repulsion": float(nuclear_repulsion),
    }
    if include_mo_occ:
        mapping["mo_occ"] = np.concatenate([np.full(int(n_occupied), 2.0), np.zeros(n - int(n_occupied))])
        mapping["mo_energy"] = np.sort(np.diag(h))
    return mapping


def synthetic_qdynamics_mapping(kind: str, *, seed: int = 20261019) -> dict[str, Any]:
    """Named synthetic mappings in the producer's schema (never real molecules)."""

    key = str(kind).strip().lower()
    if key == "contract4":
        h, g, scalar, _ = contract_four_qubit_tensors()
        return qdynamics_mapping_from_spatial(h, g, nuclear_repulsion=scalar, n_occupied=1, geometry="synthetic mathematical fixture", basis="synthetic orthonormal spatial basis")
    if key == "random6":
        h, g = random_symmetric_tensors(3, seed=seed)
        return qdynamics_mapping_from_spatial(h, g, nuclear_repulsion=0.25, n_occupied=1, geometry="synthetic random m=3", basis="synthetic")
    if key == "random6_two_occupied":
        h, g = random_symmetric_tensors(3, seed=seed + 1)
        return qdynamics_mapping_from_spatial(h, g, nuclear_repulsion=0.75, n_occupied=2, geometry="synthetic random m=3 (2,2)", basis="synthetic")
    if key == "random8":
        h, g = random_symmetric_tensors(4, seed=seed + 2)
        return qdynamics_mapping_from_spatial(h, g, nuclear_repulsion=1.0, n_occupied=2, geometry="synthetic random m=4 (2,2)", basis="synthetic")
    raise ValueError(f"unknown synthetic kind {kind!r}; use contract4, random6, random6_two_occupied, random8")


SYNTHETIC_KINDS = ("contract4", "random6", "random6_two_occupied", "random8")

__all__ = [
    "CONTRACT_FOUR_QUBIT_ROOTS",
    "SYNTHETIC_KINDS",
    "contract_four_qubit_payload",
    "contract_four_qubit_tensors",
    "qdynamics_mapping_from_spatial",
    "random_symmetric_tensors",
    "synthetic_qdynamics_mapping",
]

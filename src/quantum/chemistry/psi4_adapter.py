from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class RestrictedClosedShellPsi4Snapshot:
    problem: "RestrictedClosedShellMolecularProblem"
    coeff_alpha_mo: np.ndarray
    basis_set: Any


@dataclass(frozen=True)
class RestrictedClosedShellMolecularProblem:
    geometry_spec: str
    basis: str
    charge: int
    multiplicity: int
    reference: str
    n_spatial_orbitals: int
    n_alpha: int
    n_beta: int
    hf_energy: float
    nuclear_repulsion_energy: float
    one_body_integrals_mo: np.ndarray
    two_body_integrals_mo: np.ndarray

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * int(self.n_spatial_orbitals)

    @property
    def num_particles(self) -> tuple[int, int]:
        return int(self.n_alpha), int(self.n_beta)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "geometry_spec": str(self.geometry_spec),
            "basis": str(self.basis),
            "charge": int(self.charge),
            "multiplicity": int(self.multiplicity),
            "reference": str(self.reference),
            "n_spatial_orbitals": int(self.n_spatial_orbitals),
            "n_spin_orbitals": int(self.n_spin_orbitals),
            "n_alpha": int(self.n_alpha),
            "n_beta": int(self.n_beta),
            "hf_energy": float(self.hf_energy),
            "nuclear_repulsion_energy": float(self.nuclear_repulsion_energy),
            "one_body_integrals_mo": np.asarray(self.one_body_integrals_mo, dtype=float).tolist(),
            "two_body_integrals_mo": np.asarray(self.two_body_integrals_mo, dtype=float).tolist(),
        }


def _matrix_to_ndarray(obj: Any) -> np.ndarray:
    if hasattr(obj, "np"):
        return np.asarray(obj.np, dtype=float)
    if hasattr(obj, "to_array"):
        return np.asarray(obj.to_array(), dtype=float)
    return np.asarray(obj, dtype=float)


def build_h2_geometry(bond_length_angstrom: float = 0.7414) -> str:
    bond = float(bond_length_angstrom)
    if bond <= 0.0:
        raise ValueError("bond_length_angstrom must be > 0.")
    return f"H 0.0 0.0 0.0\nH 0.0 0.0 {bond:.12f}"


def _normalized_geometry_block(
    *,
    geometry_spec: str,
    charge: int,
    multiplicity: int,
    units: str,
) -> str:
    geom = str(geometry_spec).strip()
    if not geom:
        raise ValueError("geometry_spec must be non-empty.")
    units_key = str(units).strip().lower()
    if units_key not in {"angstrom", "bohr"}:
        raise ValueError("units must be one of {'angstrom','bohr'}.")
    return (
        f"{int(charge)} {int(multiplicity)}\n"
        f"{geom}\n"
        "symmetry c1\n"
        f"units {units_key}\n"
        "no_reorient\n"
        "no_com\n"
    )


def load_restricted_closed_shell_snapshot_from_psi4(
    *,
    geometry_spec: str,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    units: str = "angstrom",
    reference: str = "rhf",
    scf_type: str = "pk",
    memory: str | None = None,
    output_file: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> RestrictedClosedShellPsi4Snapshot:
    try:
        import psi4
    except Exception as exc:  # pragma: no cover - depends on local Psi4 install
        raise ImportError(
            "Psi4 is required for the chemistry prototype. Install psi4 to run this path."
        ) from exc

    reference_key = str(reference).strip().lower()
    if reference_key != "rhf":
        raise ValueError("Prototype supports reference='rhf' only.")
    if int(multiplicity) != 1:
        raise ValueError("Prototype supports multiplicity=1 only.")

    if output_file not in {None, ""}:
        psi4.core.set_output_file(str(output_file), False)
    if memory not in {None, ""}:
        psi4.set_memory(str(memory))

    reserved_option_keys = {
        "basis",
        "reference",
        "scf_type",
        "e_convergence",
        "d_convergence",
    }
    user_options = {str(k): v for k, v in dict(options or {}).items()}
    overlap = reserved_option_keys.intersection(user_options)
    if overlap:
        blocked = ", ".join(sorted(overlap))
        raise ValueError(f"options may not override reserved prototype Psi4 keys: {blocked}")

    geom_block = _normalized_geometry_block(
        geometry_spec=str(geometry_spec),
        charge=int(charge),
        multiplicity=int(multiplicity),
        units=str(units),
    )
    mol = psi4.geometry(geom_block)
    mol.update_geometry()

    psi4.set_options(
        {
            "basis": str(basis),
            "reference": str(reference_key),
            "scf_type": str(scf_type),
            "e_convergence": 1e-10,
            "d_convergence": 1e-10,
            **user_options,
        }
    )

    hf_energy, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    n_alpha = int(wfn.nalpha())
    n_beta = int(wfn.nbeta())
    if n_alpha != n_beta:
        raise ValueError(
            "Prototype currently supports restricted closed-shell systems only (n_alpha == n_beta)."
        )

    mints = psi4.core.MintsHelper(wfn.basisset())
    coeff_alpha = _matrix_to_ndarray(wfn.Ca())
    h_ao = _matrix_to_ndarray(mints.ao_kinetic()) + _matrix_to_ndarray(mints.ao_potential())
    h_mo = coeff_alpha.T @ h_ao @ coeff_alpha
    eri_mo = _matrix_to_ndarray(mints.mo_eri(wfn.Ca(), wfn.Ca(), wfn.Ca(), wfn.Ca()))

    n_spatial = int(h_mo.shape[0])
    if h_mo.shape != (n_spatial, n_spatial):
        raise ValueError(f"Unexpected one-body integral shape: {h_mo.shape}")
    if eri_mo.shape != (n_spatial, n_spatial, n_spatial, n_spatial):
        raise ValueError(f"Unexpected two-body integral shape: {eri_mo.shape}")

    problem = RestrictedClosedShellMolecularProblem(
        geometry_spec=str(geometry_spec).strip(),
        basis=str(basis),
        charge=int(charge),
        multiplicity=int(multiplicity),
        reference=str(reference_key),
        n_spatial_orbitals=n_spatial,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hf_energy=float(hf_energy),
        nuclear_repulsion_energy=float(mol.nuclear_repulsion_energy()),
        one_body_integrals_mo=np.asarray(h_mo, dtype=float),
        two_body_integrals_mo=np.asarray(eri_mo, dtype=float),
    )
    return RestrictedClosedShellPsi4Snapshot(
        problem=problem,
        coeff_alpha_mo=np.asarray(coeff_alpha, dtype=float),
        basis_set=wfn.basisset(),
    )


def load_restricted_closed_shell_problem_from_psi4(
    *,
    geometry_spec: str,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    units: str = "angstrom",
    reference: str = "rhf",
    scf_type: str = "pk",
    memory: str | None = None,
    output_file: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> RestrictedClosedShellMolecularProblem:
    return load_restricted_closed_shell_snapshot_from_psi4(
        geometry_spec=geometry_spec,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        units=units,
        reference=reference,
        scf_type=scf_type,
        memory=memory,
        output_file=output_file,
        options=options,
    ).problem


def build_h2_snapshot_from_psi4(
    *,
    bond_length_angstrom: float = 0.7414,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    memory: str | None = None,
    output_file: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> RestrictedClosedShellPsi4Snapshot:
    return load_restricted_closed_shell_snapshot_from_psi4(
        geometry_spec=build_h2_geometry(float(bond_length_angstrom)),
        basis=str(basis),
        charge=int(charge),
        multiplicity=int(multiplicity),
        units="angstrom",
        reference="rhf",
        scf_type="pk",
        memory=memory,
        output_file=output_file,
        options=options,
    )


def build_h2_problem_from_psi4(
    *,
    bond_length_angstrom: float = 0.7414,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    memory: str | None = None,
    output_file: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> RestrictedClosedShellMolecularProblem:
    return build_h2_snapshot_from_psi4(
        bond_length_angstrom=bond_length_angstrom,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        memory=memory,
        output_file=output_file,
        options=options,
    ).problem

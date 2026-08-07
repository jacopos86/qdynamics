from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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


def restricted_closed_shell_problem_from_jsonable(
    payload: Mapping[str, Any],
) -> RestrictedClosedShellMolecularProblem:
    if not isinstance(payload, Mapping):
        raise ValueError("Molecular problem payload must be a mapping.")
    try:
        n_spatial_orbitals = int(payload["n_spatial_orbitals"])
        n_alpha = int(payload["n_alpha"])
        n_beta = int(payload["n_beta"])
        one_body_integrals_mo = np.asarray(payload["one_body_integrals_mo"], dtype=float)
        two_body_integrals_mo = np.asarray(payload["two_body_integrals_mo"], dtype=float)
    except KeyError as exc:
        raise ValueError(f"Missing molecular problem field: {exc.args[0]}") from exc
    except Exception as exc:
        raise ValueError(f"Invalid molecular problem payload: {exc}") from exc

    if int(n_spatial_orbitals) <= 0:
        raise ValueError("n_spatial_orbitals must be positive.")
    reference = str(payload.get("reference", "rhf")).strip().lower()
    if reference != "rhf":
        raise ValueError("Restricted closed-shell JSON payload requires reference='rhf'.")
    multiplicity = int(payload.get("multiplicity", 1))
    if int(multiplicity) != 1:
        raise ValueError("Restricted closed-shell JSON payload requires multiplicity=1.")
    if int(n_alpha) != int(n_beta):
        raise ValueError(
            "Restricted closed-shell JSON payload requires n_alpha == n_beta."
        )
    if one_body_integrals_mo.shape != (int(n_spatial_orbitals), int(n_spatial_orbitals)):
        raise ValueError(
            "one_body_integrals_mo shape must equal "
            f"({int(n_spatial_orbitals)}, {int(n_spatial_orbitals)}); "
            f"got {tuple(one_body_integrals_mo.shape)}."
        )
    if two_body_integrals_mo.shape != (
        int(n_spatial_orbitals),
        int(n_spatial_orbitals),
        int(n_spatial_orbitals),
        int(n_spatial_orbitals),
    ):
        raise ValueError(
            "two_body_integrals_mo shape must equal "
            f"({int(n_spatial_orbitals)}, {int(n_spatial_orbitals)}, "
            f"{int(n_spatial_orbitals)}, {int(n_spatial_orbitals)}); "
            f"got {tuple(two_body_integrals_mo.shape)}."
        )
    n_spin_orbitals_raw = payload.get("n_spin_orbitals")
    if n_spin_orbitals_raw is not None and int(n_spin_orbitals_raw) != 2 * int(n_spatial_orbitals):
        raise ValueError(
            "n_spin_orbitals must equal 2 * n_spatial_orbitals for restricted closed-shell problems."
        )
    if int(n_alpha) < 0 or int(n_beta) < 0:
        raise ValueError("n_alpha and n_beta must be non-negative.")

    return RestrictedClosedShellMolecularProblem(
        geometry_spec=str(payload.get("geometry_spec", "")),
        basis=str(payload.get("basis", "unknown")),
        charge=int(payload.get("charge", 0)),
        multiplicity=int(multiplicity),
        reference=str(reference),
        n_spatial_orbitals=int(n_spatial_orbitals),
        n_alpha=int(n_alpha),
        n_beta=int(n_beta),
        hf_energy=float(payload.get("hf_energy", 0.0)),
        nuclear_repulsion_energy=float(payload.get("nuclear_repulsion_energy", 0.0)),
        one_body_integrals_mo=np.asarray(one_body_integrals_mo, dtype=float),
        two_body_integrals_mo=np.asarray(two_body_integrals_mo, dtype=float),
    )


def _unwrap_restricted_closed_shell_problem_payload(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Molecular JSON payload must be an object.")
    if "n_spatial_orbitals" in payload:
        return payload
    nested_problem = payload.get("problem")
    if isinstance(nested_problem, Mapping) and "n_spatial_orbitals" in nested_problem:
        return nested_problem
    chemistry_payload = payload.get("chemistry")
    if isinstance(chemistry_payload, Mapping):
        if "n_spatial_orbitals" in chemistry_payload:
            return chemistry_payload
        nested_problem = chemistry_payload.get("problem")
        if isinstance(nested_problem, Mapping) and "n_spatial_orbitals" in nested_problem:
            return nested_problem
    raise ValueError(
        "Could not find a restricted closed-shell molecular problem object in the provided JSON payload."
    )


def load_restricted_closed_shell_problem_from_json(
    json_path: str | Path,
) -> RestrictedClosedShellMolecularProblem:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Molecular problem JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return restricted_closed_shell_problem_from_jsonable(
        _unwrap_restricted_closed_shell_problem_payload(payload)
    )


def _matrix_to_ndarray(obj: Any) -> np.ndarray:
    if hasattr(obj, "np"):
        return np.asarray(obj.np, dtype=float)
    if hasattr(obj, "to_array"):
        return np.asarray(obj.to_array(), dtype=float)
    return np.asarray(obj, dtype=float)


def align_restricted_closed_shell_snapshot_to_center_mo(
    snapshot: RestrictedClosedShellPsi4Snapshot,
    *,
    center_snapshot: RestrictedClosedShellPsi4Snapshot,
) -> RestrictedClosedShellMolecularProblem:
    """Rotate a displaced Psi4 snapshot into the center-geometry MO gauge.

    The vibronic H2 finite-difference route differentiates Hamiltonian
    coefficients, so displaced integrals must be compared in a common MO frame.
    This helper performs the existing orthogonal-Procrustes alignment against
    ``center_snapshot`` and returns a problem with aligned one-/two-body MO
    integrals. Energies and nuclear repulsion are copied from the displaced
    snapshot and are not recomputed.
    """

    if int(snapshot.problem.n_spatial_orbitals) != int(center_snapshot.problem.n_spatial_orbitals):
        raise ValueError("Snapshot orbital counts do not match center geometry.")
    try:
        import psi4
    except Exception as exc:  # pragma: no cover - depends on local Psi4 install
        raise ImportError("Psi4 is required for center-MO overlap alignment.") from exc

    mints = psi4.core.MintsHelper(center_snapshot.basis_set)
    s_cross = _matrix_to_ndarray(mints.ao_overlap(center_snapshot.basis_set, snapshot.basis_set))
    overlap_mo = np.asarray(center_snapshot.coeff_alpha_mo, dtype=float).T @ s_cross @ np.asarray(snapshot.coeff_alpha_mo, dtype=float)
    u, _singular_values, vh = np.linalg.svd(overlap_mo, full_matrices=False)
    rotation = np.asarray(vh.T @ u.T, dtype=float)

    h_old = np.asarray(snapshot.problem.one_body_integrals_mo, dtype=float)
    eri_old = np.asarray(snapshot.problem.two_body_integrals_mo, dtype=float)
    h_aligned = rotation.T @ h_old @ rotation
    eri_aligned = np.einsum("ap,bq,cr,ds,abcd->pqrs", rotation, rotation, rotation, rotation, eri_old, optimize=True)

    return RestrictedClosedShellMolecularProblem(
        geometry_spec=str(snapshot.problem.geometry_spec),
        basis=str(snapshot.problem.basis),
        charge=int(snapshot.problem.charge),
        multiplicity=int(snapshot.problem.multiplicity),
        reference=str(snapshot.problem.reference),
        n_spatial_orbitals=int(snapshot.problem.n_spatial_orbitals),
        n_alpha=int(snapshot.problem.n_alpha),
        n_beta=int(snapshot.problem.n_beta),
        hf_energy=float(snapshot.problem.hf_energy),
        nuclear_repulsion_energy=float(snapshot.problem.nuclear_repulsion_energy),
        one_body_integrals_mo=np.asarray(h_aligned, dtype=float),
        two_body_integrals_mo=np.asarray(eri_aligned, dtype=float),
    )


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


def compute_total_energy_from_psi4(
    *,
    geometry_spec: str,
    method: str,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    units: str = "angstrom",
    reference: str = "rhf",
    scf_type: str = "pk",
    memory: str | None = None,
    output_file: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> float:
    """Compute a total molecular energy with Psi4 for optional references.

    This is used by fixture generators for H2 FCI/reference surfaces. It imports
    Psi4 lazily and returns the total energy including nuclear repulsion, matching
    Psi4's normal energy convention.
    """

    try:
        import psi4
    except Exception as exc:  # pragma: no cover - depends on local Psi4 install
        raise ImportError("Psi4 is required for optional chemistry energy references.") from exc

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
            "reference": str(reference).strip().lower(),
            "scf_type": str(scf_type),
            "e_convergence": 1e-10,
            "d_convergence": 1e-10,
            **user_options,
        }
    )
    return float(psi4.energy(str(method), molecule=mol))


def compute_h2_total_energy_from_psi4(
    *,
    bond_length_angstrom: float = 0.7414,
    method: str = "fci",
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    memory: str | None = None,
    output_file: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> float:
    return compute_total_energy_from_psi4(
        geometry_spec=build_h2_geometry(float(bond_length_angstrom)),
        method=str(method),
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

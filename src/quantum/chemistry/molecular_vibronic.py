"""Molecule-agnostic linear vibronic qubit Hamiltonian builder.

For a closed-shell molecule at a geometry R0 the model is

    H = H_el(R0) + sum_nu omega_nu (b_nu^dag b_nu + 1/2) + sum_nu Q_nu C_nu,
    Q_nu = (b_nu + b_nu^dag) / sqrt(2 omega_nu),

with H_el the active-space electronic Hamiltonian

    H_el = E_scalar + sum_{pq,s} h_pq a^dag_{ps} a_{qs}
         + 1/2 sum_{pqrs,st} (pq|rs) a^dag_{ps} a^dag_{rt} a_{st} a_{qs}

in chemists' notation, omega_nu the oscillator frequency of mode nu (a choice,
``harmonic_curvature`` below; the mode vectors come from the mass-weighted RHF
Hessian), and C_nu the linear coupling operator. Two coupling
models are implemented, both from the same displaced self-consistent-field
calculations with the displaced molecular orbitals rotated into the R0 gauge:

    full_linear     C_nu = dH_el/dQ_nu: derivative of the scalar, one-body and
                    two-body active-space tensors (the Paper IV H2O model).
    mean_field_fd   C_nu = sum_{pq,s} (dF/dQ_nu)_pq a^dag a + 2 sum_{c in core}
                    (dF/dQ_nu)_cc: derivative of the Fock matrix only, the
                    mean-field electron-phonon vertex, frozen-core contracted.

A third route, ``mean_field_pyscf_eph``, reproduces the PySCF ``eph`` vertex
(derivative of the nuclear-attraction plus Hartree-Fock potential with the
atomic-orbital basis held fixed) for comparison with external producers.

Spin orbitals are mapped by Jordan-Wigner in blocked order (all alpha, then all
beta), each mode is truncated at occupation ``n_ph_max`` and binary (or unary)
encoded on qubits above the fermion register. The electronic structure backend
is PySCF (restricted Hartree-Fock); it is imported lazily so the module loads
without it. Everything after the backend snapshot is backend independent.

The oscillator frequency is a choice (``harmonic_curvature``). The default,
``pes_consistent``, sets omega_nu^2 = K_nu^PES + chi_nu: the curvature of the
frozen-core (active-space) FCI potential energy surface along the mode plus the
model's own correlated response to Q_nu C_nu, so that the exactly solved model
reproduces the diagonal of the frozen-core FCI Hessian in the RHF normal-mode
basis (second-order perturbation theory). The off-diagonal is not reproduced:
between two modes of the same irreducible representation the model's adiabatic
Hessian holds -chi_{mu nu} (its cross response) where the FCI surface holds
K^PES_{mu nu}; for H2O/STO-3G that leaves the model's adiabatic bend and
symmetric-stretch frequencies at 1949 / 4010 cm^-1 against the surface's
1975 / 3998. ``rhf_hessian`` (the Paper IV H2O definition) takes omega_nu from
the RHF Hessian, which already holds the orbital relaxation the coupling
regenerates; ``frozen_reference`` is the Hartree-Fock analogue of the default.
Displaced orbitals are aligned to the R0 gauge block by block (frozen core,
active, external) so the frozen core stays the canonical core of each displaced
SCF; the Paper IV full-set alignment is kept as ``alignment="full"``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.chemistry.molecular_hamiltonian import (
    build_restricted_closed_shell_molecular_hamiltonian,
)
from src.quantum.chemistry.psi4_adapter import RestrictedClosedShellMolecularProblem
from src.quantum.chemistry.vibronic_h2 import (
    _boson_momentum_operator,
    _clean_real_polynomial,
    _lift_fermion_polynomial,
    pauli_polynomial_from_jsonable,
    pauli_polynomial_to_jsonable,
)
from src.quantum.chemistry.vibronic_h2o_linear_fd import BosonModeRegister, RegisterLayout
from src.quantum.hartree_fock_reference_state import hartree_fock_occupied_qubits
from src.quantum.hubbard_latex_python_pairs import boson_operator, boson_qubits_per_site
from src.quantum.pauli_polynomial_class import PauliPolynomial

MODEL_SCHEMA = "molecular_vibronic_qubit_hamiltonian_v1"
HARTREE_TO_CM_INV = 219474.6313632
AMU_TO_ELECTRON_MASS = 1822.888486209
BOHR_TO_ANGSTROM = 0.529177210903
COUPLING_MODELS = ("full_linear", "mean_field_fd")
HARMONIC_CURVATURES = ("pes_consistent", "rhf_hessian", "frozen_reference")
ALIGNMENTS = ("block_diagonal", "full")
LADDER_CONVENTION = "Q=(2*omega)^(-1/2)*(b_dag+b)"

# Most-abundant-isotope masses in atomic mass units (the Psi4 convention the
# Paper IV H2O fixture used). Elements outside this table fall back to the
# backend's standard atomic weight.
MOST_ABUNDANT_ISOTOPE_MASS_AMU: dict[str, float] = {
    "H": 1.00782503223, "He": 4.00260325413, "Li": 7.0160034366, "Be": 9.012183065,
    "B": 11.00930536, "C": 12.0, "N": 14.00307400443, "O": 15.99491461957,
    "F": 18.99840316273, "Ne": 19.9924401762, "Na": 22.989769282, "Mg": 23.985041697,
    "Al": 26.98153853, "Si": 27.97692653465, "P": 30.97376199842, "S": 31.9720711744,
    "Cl": 34.968852682, "Ar": 39.9623831237,
}
# Doubly occupied core orbitals per element for the ``frozen_core`` policy.
CORE_ORBITALS_BY_ELEMENT: dict[str, int] = {
    "H": 0, "He": 0, "Li": 1, "Be": 1, "B": 1, "C": 1, "N": 1, "O": 1, "F": 1, "Ne": 1,
    "Na": 5, "Mg": 5, "Al": 5, "Si": 5, "P": 5, "S": 5, "Cl": 5, "Ar": 5,
}


# ------------------------------------------------------------------ records


@dataclass(frozen=True)
class MolecularGeometry:
    symbols: tuple[str, ...]
    coordinates_bohr: np.ndarray  # (n_atoms, 3)
    charge: int = 0
    multiplicity: int = 1

    def __post_init__(self) -> None:
        coords = np.array(self.coordinates_bohr, dtype=float, copy=True)
        if coords.shape != (len(self.symbols), 3):
            raise ValueError(f"coordinates must have shape ({len(self.symbols)}, 3); got {coords.shape}.")
        if not np.all(np.isfinite(coords)):
            raise ValueError("coordinates must be finite.")
        coords.setflags(write=False)
        object.__setattr__(self, "coordinates_bohr", coords)
        object.__setattr__(self, "symbols", tuple(str(s) for s in self.symbols))
        if int(self.multiplicity) != 1:
            raise ValueError("only closed-shell (multiplicity 1) molecules are supported.")

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    def displaced(self, delta_bohr: np.ndarray) -> "MolecularGeometry":
        return MolecularGeometry(self.symbols, self.coordinates_bohr + np.asarray(delta_bohr, dtype=float),
                                 self.charge, self.multiplicity)

    def to_jsonable(self) -> dict[str, Any]:
        return {"symbols": list(self.symbols), "coordinates_bohr": self.coordinates_bohr.tolist(),
                "charge": int(self.charge), "multiplicity": int(self.multiplicity)}


def parse_geometry(spec: str, *, units: str = "angstrom", charge: int = 0, multiplicity: int = 1) -> MolecularGeometry:
    """Parse ``'O 0 0 0; H 0 0.76 0.59'``, newline-separated lines, or an .xyz block
    (atom count, comment line, atoms) into Bohr coordinates."""
    scale = {"angstrom": 1.0 / BOHR_TO_ANGSTROM, "bohr": 1.0}[str(units).strip().lower()]
    lines = [line for line in str(spec).replace(";", "\n").splitlines() if line.strip()]
    declared: int | None = None
    if lines and lines[0].strip().isdigit():
        declared = int(lines[0].strip())
        lines = lines[2:]   # .xyz: the count line, then one free-text comment line
    symbols: list[str] = []
    rows: list[list[float]] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"geometry line must be 'Sym x y z'; got {line!r}.")
        symbols.append(parts[0].capitalize())
        rows.append([float(v) * scale for v in parts[1:]])
    if not symbols:
        raise ValueError("geometry is empty.")
    if declared is not None and declared != len(symbols):
        raise ValueError(f".xyz header declares {declared} atoms but {len(symbols)} were given.")
    return MolecularGeometry(tuple(symbols), np.asarray(rows, dtype=float), int(charge), int(multiplicity))


@dataclass(frozen=True)
class ElectronicSnapshot:
    """Converged restricted Hartree-Fock data at one geometry, in its own MO basis."""

    geometry: MolecularGeometry
    basis: str
    hf_energy: float
    nuclear_repulsion: float
    mo_coeff: np.ndarray       # (n_ao, n_mo)
    mo_energy: np.ndarray      # (n_mo,)
    h_mo: np.ndarray           # (n_mo, n_mo) one-body integrals
    eri_mo: np.ndarray         # (n_mo,)*4 chemists' (pq|rs)
    fock_mo: np.ndarray        # (n_mo, n_mo) converged Fock matrix
    n_alpha: int
    n_beta: int
    backend: str
    backend_version: str
    backend_handle: Any = field(repr=False, compare=False, default=None)

    @property
    def n_spatial_orbitals(self) -> int:
        return int(self.h_mo.shape[0])


@dataclass(frozen=True)
class NormalMode:
    mode_index: int
    frequency_hartree: float
    mass_weighted_vector: np.ndarray  # (n_atoms, 3), unit norm
    label: str = ""

    @property
    def frequency_cm1(self) -> float:
        return float(self.frequency_hartree) * HARTREE_TO_CM_INV

    def to_jsonable(self) -> dict[str, Any]:
        return {"mode_index": int(self.mode_index), "label": str(self.label),
                "frequency_hartree": float(self.frequency_hartree), "frequency_cm1": self.frequency_cm1,
                "mass_weighted_vector": np.asarray(self.mass_weighted_vector, dtype=float).tolist(),
                "coordinate_convention": "mass_weighted_normal_au", "ladder_convention": LADDER_CONVENTION}


@dataclass(frozen=True)
class ActiveSpace:
    frozen_core: tuple[int, ...]
    active: tuple[int, ...]
    external: tuple[int, ...]
    n_alpha: int   # active alpha electrons
    n_beta: int
    kind: str

    @property
    def n_active_orbitals(self) -> int:
        return len(self.active)

    def to_jsonable(self) -> dict[str, Any]:
        return {"kind": self.kind, "frozen_core": list(self.frozen_core), "active": list(self.active),
                "external": list(self.external), "num_particles": [int(self.n_alpha), int(self.n_beta)]}


@dataclass(frozen=True)
class ActiveTensors:
    scalar: float
    one_body: np.ndarray
    two_body: np.ndarray


@dataclass(frozen=True)
class LinearCoupling:
    """dX/dQ_nu for one mode, as active-space tensors; ``two_body`` is None for one-body models."""

    mode_index: int
    model: str
    scalar: float
    one_body: np.ndarray
    two_body: np.ndarray | None
    q_step_au: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VibronicQubitModel:
    geometry: MolecularGeometry
    basis: str
    active_space: ActiveSpace
    modes: tuple[NormalMode, ...]
    cutoffs: tuple[int, ...]
    encoding: str
    coupling_model: str
    layout: RegisterLayout
    h_electronic: PauliPolynomial
    h_vibronic: PauliPolynomial
    coupling_operators: tuple[PauliPolynomial, ...]   # C_nu on the fermion register
    q_operators: tuple[PauliPolynomial, ...]
    p_operators: tuple[PauliPolynomial, ...]
    n_operators: tuple[PauliPolynomial, ...]
    active_tensors: ActiveTensors
    couplings: tuple[LinearCoupling, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_qubits(self) -> int:
        return int(self.layout.n_total_qubits)


# ----------------------------------------------------------- PySCF backend


def _pyscf_molecule(geometry: MolecularGeometry, basis: str) -> Any:
    from pyscf import gto  # type: ignore

    atoms = [(sym, tuple(float(v) for v in xyz)) for sym, xyz in zip(geometry.symbols, geometry.coordinates_bohr)]
    return gto.M(atom=atoms, unit="Bohr", basis=str(basis), charge=int(geometry.charge),
                 spin=int(geometry.multiplicity) - 1, verbose=0)


def pyscf_rhf_snapshot(
    geometry: MolecularGeometry,
    *,
    basis: str,
    conv_tol: float = 1.0e-12,
    conv_tol_grad: float = 1.0e-9,
    max_cycle: int = 200,
    initial_density: np.ndarray | None = None,
) -> ElectronicSnapshot:
    """Converged RHF at ``geometry`` with MO-basis integrals (chemists' ERIs)."""
    import pyscf  # type: ignore
    from pyscf import ao2mo, scf  # type: ignore

    mol = _pyscf_molecule(geometry, basis)
    mf = scf.RHF(mol)
    mf.conv_tol = float(conv_tol)
    mf.conv_tol_grad = float(conv_tol_grad)
    mf.max_cycle = int(max_cycle)
    energy = mf.kernel(dm0=initial_density) if initial_density is not None else mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge.")
    coeff = np.asarray(mf.mo_coeff, dtype=float)
    n_mo = coeff.shape[1]
    h_mo = coeff.T @ np.asarray(mf.get_hcore(), dtype=float) @ coeff
    fock_mo = coeff.T @ np.asarray(mf.get_fock(), dtype=float) @ coeff
    eri_mo = np.asarray(ao2mo.full(mol, coeff, compact=False), dtype=float).reshape(n_mo, n_mo, n_mo, n_mo)
    n_alpha, n_beta = (int(v) for v in mol.nelec)
    if n_alpha != n_beta:
        raise ValueError("RHF snapshot requires n_alpha == n_beta.")
    return ElectronicSnapshot(
        geometry=geometry, basis=str(basis), hf_energy=float(energy),
        nuclear_repulsion=float(mol.energy_nuc()), mo_coeff=coeff,
        mo_energy=np.asarray(mf.mo_energy, dtype=float), h_mo=h_mo, eri_mo=eri_mo, fock_mo=fock_mo,
        n_alpha=n_alpha, n_beta=n_beta, backend="pyscf", backend_version=str(pyscf.__version__),
        backend_handle={"mol": mol, "mf": mf},
    )


def pyscf_hessian(snapshot: ElectronicSnapshot) -> np.ndarray:
    """Analytic RHF nuclear Hessian, (3N, 3N) in Hartree/Bohr^2."""
    mf = snapshot.backend_handle["mf"]
    hess = np.asarray(mf.Hessian().kernel(), dtype=float)   # (N, N, 3, 3)
    n_atoms = hess.shape[0]
    return hess.transpose(0, 2, 1, 3).reshape(3 * n_atoms, 3 * n_atoms)


def pyscf_ao_cross_overlap(center: ElectronicSnapshot, other: ElectronicSnapshot) -> np.ndarray:
    from pyscf import gto  # type: ignore

    return np.asarray(gto.intor_cross("int1e_ovlp", center.backend_handle["mol"], other.backend_handle["mol"]),
                      dtype=float)


def pyscf_mean_field_eph_vertex(
    center: ElectronicSnapshot,
    modes: Sequence[NormalMode],
    masses_me: np.ndarray,
) -> np.ndarray:
    """PySCF's analytic electron-phonon vertex projected on ``modes``: (n_modes, n_mo, n_mo).

    This is g_nu,pq = <p| dV/dQ_nu |q> / sqrt(2 omega_nu) with V the nuclear
    attraction plus the Hartree-Fock potential, the derivative taken with the
    atomic-orbital basis fixed (no kinetic or basis-motion terms), in the R0 MO
    basis. The PySCF kernel's own mode analysis (integer mass numbers times the
    proton-electron mass ratio) is bypassed: the Cartesian vertex is contracted
    with the caller's mass-weighted modes and masses instead.
    """
    from pyscf.eph import rhf as eph_rhf  # type: ignore

    mf = center.backend_handle["mf"]
    ephobj = eph_rhf.EPH(mf)
    mo_coeff, mo_occ, mo_energy = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    h1ao = ephobj.make_h1(mo_coeff, mo_occ)
    mo1, _mo_e1 = ephobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao)
    # Cartesian vertex <mu| dV/dR_{A,alpha} |nu> per atom, as in eph_rhf.get_eph.
    mol = center.backend_handle["mol"]
    vnuc_deriv = ephobj.vnuc_generator(mol)
    vind = eph_rhf.rhf_deriv_generator(mf, mo_coeff, mo_occ)
    mocc = mo_coeff[:, mo_occ > 0]
    dm0 = np.dot(mocc, mocc.T) * 2
    aoslices = mol.aoslice_by_atom()
    from pyscf.hessian import rhf as hess_rhf  # type: ignore

    vcore = []
    for ia in range(mol.natm):
        h1 = vnuc_deriv(ia)
        v1 = vind(mo1[ia])
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas) * 3
        vj1, vk1 = hess_rhf._get_jk(mol, "int2e_ip1", 3, "s2kl",
                                    ["ji->s2kl", -dm0[:, p0:p1], "li->s1kj", -dm0[:, p0:p1]],
                                    shls_slice=shls_slice)
        vhf = vj1 - vk1 * 0.5
        vcore.append(h1 + v1 + vhf + vhf.transpose(0, 2, 1))
    vcore = np.asarray(vcore).reshape(-1, mol.nao_nr(), mol.nao_nr())   # (3N, nao, nao)
    masses = np.asarray(masses_me, dtype=float).reshape(-1)
    out = []
    for mode in modes:
        cart = (np.asarray(mode.mass_weighted_vector, dtype=float) / np.sqrt(masses)[:, None]).reshape(-1)
        v_mode = np.einsum("x,xuv->uv", cart, vcore) / math.sqrt(2.0 * float(mode.frequency_hartree))
        out.append(center.mo_coeff.T @ v_mode @ center.mo_coeff)
    return np.asarray(out, dtype=float)


# ------------------------------------------------------- masses and modes


def isotope_masses_me(symbols: Sequence[str], overrides_amu: Mapping[str, float] | None = None) -> np.ndarray:
    """Most-abundant-isotope masses in electron-mass units.

    Elements beyond the module table use PySCF's ``COMMON_ISOTOPE_MASSES``
    (same convention); ``overrides_amu`` keys are element symbols.
    """
    overrides = {str(k).capitalize(): float(v) for k, v in dict(overrides_amu or {}).items()}
    out = []
    for sym in symbols:
        key = str(sym).capitalize()
        if key in overrides:
            amu = overrides[key]
        elif key in MOST_ABUNDANT_ISOTOPE_MASS_AMU:
            amu = MOST_ABUNDANT_ISOTOPE_MASS_AMU[key]
        else:
            from pyscf.data import elements  # type: ignore

            amu = float(elements.COMMON_ISOTOPE_MASSES[elements.charge(key)])
        if not amu > 0.0:
            raise ValueError(f"mass for {key} must be positive; got {amu}.")
        out.append(amu * AMU_TO_ELECTRON_MASS)
    return np.asarray(out, dtype=float)


LINEAR_MOLECULE_RELATIVE_TOLERANCE = 1.0e-6


def translation_rotation_basis(coordinates_bohr: np.ndarray, masses_me: np.ndarray) -> np.ndarray:
    """Orthonormal mass-weighted translation and rotation vectors, (3N, 5 or 6).

    The six raw vectors are rank-revealed by SVD with a relative tolerance, so
    a molecule that is linear to within rounding (where the rotation about the
    axis is a nearly null vector) keeps five external directions, not six.
    """
    coords = np.asarray(coordinates_bohr, dtype=float)
    masses = np.asarray(masses_me, dtype=float).reshape(-1)
    sqrt_m = np.sqrt(masses)
    com = np.sum(masses[:, None] * coords, axis=0) / float(np.sum(masses))
    shifted = coords - com[None, :]
    raw: list[np.ndarray] = []
    for axis in range(3):
        vec = np.zeros_like(coords)
        vec[:, axis] = sqrt_m
        raw.append(vec.reshape(-1))
    for axis in range(3):
        cross = np.cross(np.eye(3)[axis], shifted)
        raw.append((sqrt_m[:, None] * cross).reshape(-1))
    stacked = np.column_stack(raw)
    u, singular, _vh = np.linalg.svd(stacked, full_matrices=False)
    keep = singular > LINEAR_MOLECULE_RELATIVE_TOLERANCE * float(singular[0])
    return u[:, keep]


def vibrational_modes_from_hessian(
    hessian: np.ndarray,
    *,
    coordinates_bohr: np.ndarray,
    masses_me: np.ndarray,
    imaginary_tolerance_hartree: float = 1.0e-6,
) -> tuple[NormalMode, ...]:
    """Mass-weight, project out translations and rotations, diagonalize.

    Returns the 3N-6 (3N-5 for linear molecules) vibrational modes in ascending
    frequency, with mass-weighted eigenvectors e_nu normalized to
    sum_{A,alpha} e_nu,A,alpha e_mu,A,alpha = delta_mu,nu and the sign fixed so
    the first (by index) of the largest-magnitude components is positive. Q_nu
    is the coordinate along e_nu in mass-weighted atomic units: the Cartesian
    displacement is Q e / sqrt(M).
    """
    hess = np.asarray(hessian, dtype=float)
    masses = np.asarray(masses_me, dtype=float).reshape(-1)
    n_atoms = len(masses)
    if hess.shape != (3 * n_atoms, 3 * n_atoms):
        raise ValueError(f"hessian must be ({3 * n_atoms}, {3 * n_atoms}); got {hess.shape}.")
    inv_sqrt = 1.0 / np.sqrt(np.repeat(masses, 3))
    weighted = 0.5 * (hess + hess.T) * np.outer(inv_sqrt, inv_sqrt)
    basis = translation_rotation_basis(coordinates_bohr, masses)
    n_external = int(basis.shape[1])
    if n_external not in (5, 6) and n_atoms > 1:
        raise ValueError(f"expected 5 or 6 external directions; found {n_external}.")
    projector = np.eye(3 * n_atoms) - basis @ basis.T
    weighted = projector @ weighted @ projector
    weighted = 0.5 * (weighted + weighted.T)
    evals, evecs = np.linalg.eigh(weighted)
    # The projected-out subspace contributes n_external exact zeros; drop the
    # n_external smallest-magnitude eigenvalues and keep the rest in ascending order.
    keep = np.sort(np.argsort(np.abs(evals))[n_external:])
    evals, evecs = evals[keep], evecs[:, keep]
    if len(evals) != 3 * n_atoms - n_external:
        raise ValueError("vibrational mode count does not match 3N minus the external directions.")
    modes: list[NormalMode] = []
    for k, value in enumerate(evals):
        if value < 0.0:
            if math.sqrt(-value) > imaginary_tolerance_hartree:
                raise ValueError(
                    f"imaginary vibrational frequency {math.sqrt(-value) * HARTREE_TO_CM_INV:.2f} cm^-1: "
                    "the geometry is not a minimum."
                )
            value = 0.0
        vector = evecs[:, k].reshape(n_atoms, 3)
        flat = vector.reshape(-1)
        largest = float(np.max(np.abs(flat)))
        # Tie-tolerant orientation: symmetry-equivalent atoms can carry equal and
        # opposite largest components, where argmax would follow rounding noise.
        pivot = int(np.flatnonzero(np.abs(flat) >= largest * (1.0 - 1.0e-8))[0])
        if flat[pivot] < 0.0:
            vector = -vector
        vector = vector / float(np.linalg.norm(vector))
        modes.append(NormalMode(mode_index=k, frequency_hartree=float(math.sqrt(value)),
                                mass_weighted_vector=np.asarray(vector, dtype=float), label=f"mode_{k}"))
    return tuple(modes)


def displace_along_mode(geometry: MolecularGeometry, mode: NormalMode, masses_me: np.ndarray, q_au: float) -> MolecularGeometry:
    """R = R0 + Q e_nu / sqrt(M), with Q in mass-weighted atomic units."""
    delta = float(q_au) * np.asarray(mode.mass_weighted_vector, dtype=float) / np.sqrt(np.asarray(masses_me, dtype=float))[:, None]
    return geometry.displaced(delta)


# ----------------------------------------------------- active space, frozen core


def select_active_space(
    n_spatial_orbitals: int,
    n_alpha: int,
    n_beta: int,
    *,
    policy: str | Mapping[str, Any] = "frozen_core",
    symbols: Sequence[str] | None = None,
) -> ActiveSpace:
    """Policies: ``full``; ``frozen_core`` (element core counts); ``frozen_core:<n>``;
    ``cas(<n_electrons>,<n_orbitals>)`` around the Fermi level; or an explicit
    mapping ``{"frozen_core": [...], "active": [...]}`` (external orbitals are
    the remainder and are discarded)."""
    n = int(n_spatial_orbitals)
    if int(n_alpha) != int(n_beta):
        raise ValueError("closed-shell active spaces require n_alpha == n_beta.")
    n_occ = int(n_alpha)
    if isinstance(policy, Mapping):
        # Sorted, so the first n_act_el active orbitals are the occupied ones that
        # closed_shell_determinant_energy, reference_force and hartree_fock_state assume.
        frozen = tuple(sorted({int(v) for v in policy.get("frozen_core", ())}))
        active = tuple(sorted({int(v) for v in policy["active"]}))
        if len(active) != len(tuple(policy["active"])) or len(frozen) != len(tuple(policy.get("frozen_core", ()))):
            raise ValueError("frozen_core and active orbital lists must not repeat an index.")
        kind = str(policy.get("kind", "explicit"))
    else:
        key = str(policy).strip().lower()
        if key == "full":
            frozen, active, kind = (), tuple(range(n)), "full"
        elif key == "frozen_core" or key.startswith("frozen_core:"):
            if ":" in key:
                n_core = int(key.split(":", 1)[1])
            else:
                if symbols is None:
                    raise ValueError("frozen_core policy needs the element symbols.")
                unknown = sorted({str(s).capitalize() for s in symbols} - set(CORE_ORBITALS_BY_ELEMENT))
                if unknown:
                    raise ValueError(f"no core-orbital count for {unknown}; use frozen_core:<n> or cas(<ne>,<no>).")
                n_core = sum(CORE_ORBITALS_BY_ELEMENT[str(s).capitalize()] for s in symbols)
            frozen, active, kind = tuple(range(n_core)), tuple(range(n_core, n)), f"frozen_core:{n_core}"
        elif key.startswith("cas(") and key.endswith(")"):
            n_el, n_orb = (int(v) for v in key[4:-1].split(","))
            if n_el % 2:
                raise ValueError("cas(n_el, n_orb) needs an even electron count for a closed shell.")
            n_core = n_occ - n_el // 2
            if n_core < 0 or n_core + n_orb > n:
                raise ValueError(f"cas({n_el},{n_orb}) does not fit {n} orbitals with {2 * n_occ} electrons.")
            frozen, active, kind = tuple(range(n_core)), tuple(range(n_core, n_core + n_orb)), f"cas({n_el},{n_orb})"
        else:
            raise ValueError(f"unknown active-space policy {policy!r}.")
    if set(frozen) & set(active) or any(i >= n or i < 0 for i in frozen + active):
        raise ValueError("frozen and active orbital sets must be disjoint and in range.")
    if any(i >= n_occ for i in frozen):
        raise ValueError("frozen core orbitals must be doubly occupied in the reference.")
    external = tuple(i for i in range(n) if i not in set(frozen) | set(active))
    n_act_el = n_occ - len(frozen)
    if n_act_el < 0 or n_act_el > len(active):
        raise ValueError("active electron count is outside the active orbital capacity.")
    if any(i < n_occ for i in external):
        raise ValueError("an occupied orbital cannot be discarded as external.")
    return ActiveSpace(frozen, active, external, n_act_el, n_act_el, kind)


def frozen_core_contract(h: np.ndarray, g: np.ndarray, scalar: float, space: ActiveSpace) -> ActiveTensors:
    """Closed-shell frozen-core contraction in chemists' notation.

        E_core = scalar + 2 sum_c h_cc + sum_{cd} [2 (cc|dd) - (cd|dc)]
        h'_pq  = h_pq + sum_c [2 (pq|cc) - (pc|cq)]      p, q active
        g'     = g restricted to the active orbitals
    """
    h = np.asarray(h, dtype=float)
    g = np.asarray(g, dtype=float)
    core = np.asarray(space.frozen_core, dtype=int)
    act = np.asarray(space.active, dtype=int)
    e_core = float(scalar)
    h_act = np.array(h[np.ix_(act, act)], dtype=float, copy=True)
    if core.size:
        e_core += 2.0 * float(np.trace(h[np.ix_(core, core)]))
        g_cc = g[np.ix_(core, core, core, core)]
        e_core += float(2.0 * np.einsum("ccdd->", g_cc) - np.einsum("cddc->", g_cc))
        h_act += 2.0 * np.einsum("pqcc->pq", g[np.ix_(act, act, core, core)]) - np.einsum(
            "pccq->pq", g[np.ix_(act, core, core, act)])
    return ActiveTensors(e_core, h_act, np.array(g[np.ix_(act, act, act, act)], dtype=float, copy=True))


def frozen_core_contract_one_body(matrix: np.ndarray, space: ActiveSpace) -> tuple[float, np.ndarray]:
    """Contract a one-body operator sum_pq m_pq a^dag a: scalar 2 sum_c m_cc plus the active block."""
    m = np.asarray(matrix, dtype=float)
    core = np.asarray(space.frozen_core, dtype=int)
    act = np.asarray(space.active, dtype=int)
    scalar = 2.0 * float(np.trace(m[np.ix_(core, core)])) if core.size else 0.0
    return scalar, np.array(m[np.ix_(act, act)], dtype=float, copy=True)


def closed_shell_determinant_energy(t: ActiveTensors, n_occ: int) -> float:
    """<HF| H_el |HF> = E + 2 sum_i h_ii + sum_ij [2 (ii|jj) - (ij|ji)] over the active occupied orbitals."""
    occ = np.arange(int(n_occ))
    g = t.two_body[np.ix_(occ, occ, occ, occ)]
    return float(t.scalar + 2.0 * np.trace(t.one_body[np.ix_(occ, occ)]) + 2.0 * np.einsum("iijj->", g) - np.einsum("ijji->", g))


def active_space_fci_energy(t: ActiveTensors, num_particles: tuple[int, int], *, tol: float = 1.0e-12, max_cycle: int = 300) -> float:
    """Exact (full CI) ground energy of the active-space Hamiltonian, scalar included (PySCF ``direct_spin1``)."""
    from pyscf import fci  # type: ignore

    n = int(t.one_body.shape[0])
    nelec = (int(num_particles[0]), int(num_particles[1]))
    energy, _ = fci.direct_spin1.kernel(np.asarray(t.one_body, dtype=float), np.asarray(t.two_body, dtype=float), n, nelec,
                                        ecore=float(t.scalar), tol=float(tol), max_cycle=int(max_cycle))
    return float(energy)


def active_response_curvature(
    t0: ActiveTensors,
    coupling: LinearCoupling,
    num_particles: tuple[int, int],
    *,
    q_step_au: float,
    energy: Any = None,
) -> float:
    """chi_nu = -d2/dQ2 E[H_el + Q C_nu] at Q = 0, the response of the model's own
    electrons to the linear coupling, by the central second difference at ``q_step_au``.

    ``energy(tensors, num_particles)`` is the ground energy of an active-space
    Hamiltonian (default: FCI through PySCF). A scalar in C_nu, such as the
    reference-force counter-term, shifts E linearly and drops out. For any
    variational ground state chi_nu >= 0: it is the second-order lowering
    2 sum_k |<k|C_nu|0>|^2 / (E_k - E_0) when the step is small.
    """
    if energy is None:
        energy = active_space_fci_energy
    q = float(q_step_au)
    if not (math.isfinite(q) and q > 0.0):
        raise ValueError(f"q_step_au must be a positive finite number; got {q_step_au!r}.")
    one = np.asarray(coupling.one_body, dtype=float)
    two = None if coupling.two_body is None else np.asarray(coupling.two_body, dtype=float)

    def displaced(sign: int) -> ActiveTensors:
        g = t0.two_body if two is None else t0.two_body + sign * q * two
        return ActiveTensors(float(t0.scalar) + sign * q * float(coupling.scalar), t0.one_body + sign * q * one, g)

    e_plus = float(energy(displaced(+1), num_particles))
    e_zero = float(energy(t0, num_particles))
    e_minus = float(energy(displaced(-1), num_particles))
    return -(e_plus - 2.0 * e_zero + e_minus) / (q * q)


def _oscillator_frequency(curvature: float, *, mode: NormalMode, source: str) -> float:
    """sqrt of a curvature in Hartree per Q^2; refuses a non-positive one (no oscillator)."""
    value = float(curvature)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(
            f"{source} curvature along mode {mode.mode_index} ({mode.label}) is {value:.3e} Hartree per Q^2; "
            "an oscillator needs a positive curvature, so the geometry is not a minimum of that surface along this mode.")
    return math.sqrt(value)


# ---------------------------------------------------------------- alignment


def align_to_center(
    snapshot: ElectronicSnapshot,
    center: ElectronicSnapshot,
    *,
    cross_overlap_ao: np.ndarray,
    blocks: Sequence[Sequence[int]] | None = None,
    partition: Sequence[Sequence[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Rotate a displaced snapshot's MO tensors into the center MO gauge.

    With O = C0^T S_cross C (the MO overlap between the center and displaced
    orbitals), the orthogonal Procrustes rotation R = V W^T from O = W Sigma V^T
    maximizes the overlap of C R with C0. Returns (h', g', F') with
    h' = R^T h R, g'_pqrs = sum R_ap R_bq R_cr R_ds g_abcd, F' = R^T F R.

    ``blocks``, a partition of the orbital indices (frozen core, active,
    external), makes R block diagonal: each block of displaced orbitals is
    aligned to the same block of center orbitals and no orbital crosses a
    block boundary, so a frozen core stays the canonical core of the displaced
    SCF. A full-set rotation (``blocks=None``, the Paper IV H2O convention)
    instead transports the R0 core rigidly; frozen, that core differs from the
    displaced canonical core by a part odd in the displacement, which cancels
    in a second difference, and an even part that raises the energy at second
    order, which the second difference then counts as curvature.

    ``partition`` (default: ``blocks``) adds one diagnostic, the smallest
    singular value of O inside each of its blocks. It stays near 1 for a small
    displacement and drops far below it when an orbital has crossed a block
    boundary between the two geometries, which neither the full-set singular
    values (gauge invariant) nor the occupied-block one can see.
    """
    overlap_mo = center.mo_coeff.T @ np.asarray(cross_overlap_ao, dtype=float) @ snapshot.mo_coeff
    n = int(overlap_mo.shape[0])
    # The full-overlap singular values are gauge invariant and cannot tell a
    # displaced SCF that landed on another determinant; the occupied block can.
    singular = np.linalg.svd(overlap_mo, compute_uv=False)

    def as_partition(spec: Sequence[Sequence[int]], what: str) -> tuple[tuple[int, ...], ...]:
        out = tuple(tuple(int(i) for i in block) for block in spec)
        if sorted(i for block in out for i in block) != list(range(n)):
            raise ValueError(f"alignment {what} must partition the orbital indices, each index exactly once.")
        return out

    rotation_blocks = (tuple(range(n)),) if blocks is None else as_partition(blocks, "blocks")
    rotation = np.zeros((n, n))
    for block in rotation_blocks:
        if not block:
            continue
        idx = np.asarray(block, dtype=int)
        w, _, vh = np.linalg.svd(overlap_mo[np.ix_(idx, idx)], full_matrices=False)
        rotation[np.ix_(idx, idx)] = vh.T @ w.T
    h = rotation.T @ snapshot.h_mo @ rotation
    f = rotation.T @ snapshot.fock_mo @ rotation
    g = np.einsum("ap,bq,cr,ds,abcd->pqrs", rotation, rotation, rotation, rotation, snapshot.eri_mo, optimize=True)
    n_occ = int(center.n_alpha)
    occupied_singular = np.linalg.svd(overlap_mo[:n_occ, :n_occ], compute_uv=False)
    report = blocks if partition is None else partition
    block_singular = None
    if report is not None:
        block_singular = []
        for block in as_partition(report, "partition"):
            idx = np.asarray(block, dtype=int)
            block_singular.append(float(np.min(np.linalg.svd(overlap_mo[np.ix_(idx, idx)], compute_uv=False))) if idx.size else 1.0)
    diag = {
        "min_singular_value": float(np.min(singular)),
        "occupied_block_min_singular_value": float(np.min(occupied_singular)),
        # For a block-diagonal rotation this residual also holds the overlap
        # between different blocks that the rotation leaves alone, by design.
        "alignment_residual_fro": float(np.linalg.norm(overlap_mo @ rotation - np.eye(n))),
        "block_diagonal": blocks is not None,
        "block_min_singular_values": block_singular,
    }
    return h, g, f, diag


# ------------------------------------------------------------ couplings


@dataclass(frozen=True)
class CouplingComputation:
    center: ElectronicSnapshot
    active_space: ActiveSpace
    active_tensors: ActiveTensors
    modes: tuple[NormalMode, ...]
    masses_me: np.ndarray
    couplings: dict[str, tuple[LinearCoupling, ...]]   # model -> per-mode couplings
    hf_force_by_mode: dict[str, tuple[float, ...]]     # model -> <HF|C_nu|HF>
    frozen_reference_curvature_by_mode: tuple[float, ...]   # d2/dQ2 <HF0|H_el(Q)|HF0>, Hartree per Q^2
    timing_seconds: dict[str, float]
    alignment: str = "block_diagonal"
    # d2/dQ2 E_FCI(Q) of the frozen-core active space with the canonical core of each
    # displaced SCF, Hartree per Q^2; empty unless the FCI surface was requested.
    fci_surface_curvature_by_mode: tuple[float, ...] = ()
    center_fci_energy: float | None = None


def _central_difference(plus: np.ndarray | float, minus: np.ndarray | float, step: float) -> Any:
    return (np.asarray(plus, dtype=float) - np.asarray(minus, dtype=float)) / (2.0 * float(step))


def _difference_diagnostics(primary: Any, alternative: Any) -> dict[str, float]:
    """Absolute and relative Frobenius distance; the relative one is floored so a
    derivative that vanishes by symmetry does not report noise over noise."""
    a = np.asarray(primary, dtype=float).reshape(-1)
    b = np.asarray(alternative, dtype=float).reshape(-1)
    absolute = float(np.linalg.norm(a - b))
    return {"abs": absolute, "rel": absolute / max(float(np.linalg.norm(b)), 1.0e-8)}


def compute_linear_couplings(
    center: ElectronicSnapshot,
    modes: Sequence[NormalMode],
    *,
    masses_me: np.ndarray,
    active_space: ActiveSpace,
    q_step_au: float = 0.1,
    q_step_alt_au: float | None = 0.05,
    models: Sequence[str] = COUPLING_MODELS,
    snapshot_factory: Any = None,
    cross_overlap_factory: Any = None,
    alignment: str = "block_diagonal",
    fci_surface: bool = False,
    fci_energy: Any = None,
    block_overlap_tolerance: float = 0.5,
) -> CouplingComputation:
    """Central finite differences of the aligned active-space tensors along each mode.

    For each mode nu and step q the SCF is reconverged at R0 +/- q e_nu/sqrt(M),
    the displaced MOs are rotated into the R0 gauge (``alignment``:
    ``block_diagonal`` keeps the frozen core, active and external blocks apart,
    ``full`` rotates the whole set as in Paper IV), the core is frozen and the
    active tensors sliced; then dX/dQ = [X(+q) - X(-q)] / (2q). The optional
    alternative step reports the finite-difference drift ||dX(q) - dX(q_alt)||,
    absolute and relative to ||dX(q_alt)||, per tensor as a step-size diagnostic.

    With ``fci_surface`` the frozen-core FCI energy of each displaced SCF (its
    canonical core and orbitals, no alignment needed: the energy is invariant
    to rotations inside the blocks) is evaluated as well, and its central second
    difference K_nu^PES = d2 E_FCI/dQ2 is recorded per mode; ``fci_energy``
    (``(tensors, num_particles) -> float``) defaults to PySCF full CI.

    Both the block-diagonal alignment and the FCI surface take the displaced
    SCF's frozen-core / active / external partition by canonical index, which
    is only right while no orbital crosses a block boundary between R0 and the
    displaced geometry. The smallest overlap singular value inside each block
    is recorded per coupling, and a value below ``block_overlap_tolerance``
    raises (an orbital crossing makes it drop towards zero).
    """
    if snapshot_factory is None:
        def snapshot_factory(geometry: MolecularGeometry) -> ElectronicSnapshot:
            dm0 = None
            mf = center.backend_handle["mf"] if center.backend_handle else None
            if mf is not None:
                dm0 = mf.make_rdm1()
            return pyscf_rhf_snapshot(geometry, basis=center.basis, initial_density=dm0)
    if cross_overlap_factory is None:
        cross_overlap_factory = pyscf_ao_cross_overlap
    for model in models:
        if model not in COUPLING_MODELS:
            raise ValueError(f"unknown coupling model {model!r}; choose from {COUPLING_MODELS}.")
    for name, step in (("q_step_au", q_step_au), ("q_step_alt_au", q_step_alt_au)):
        if step is not None and not (math.isfinite(float(step)) and float(step) > 0.0):
            raise ValueError(f"{name} must be a positive finite number; got {step!r}.")
    if alignment not in ALIGNMENTS:
        raise ValueError(f"alignment must be one of {ALIGNMENTS}; got {alignment!r}.")
    partition = (active_space.frozen_core, active_space.active, active_space.external)
    blocks = None if alignment == "full" else partition
    if fci_surface and fci_energy is None:
        fci_energy = active_space_fci_energy
    t0 = time.time()
    center_tensors = frozen_core_contract(center.h_mo, center.eri_mo, center.nuclear_repulsion, active_space)
    n_occ = int(active_space.n_alpha)
    particles = (int(active_space.n_alpha), int(active_space.n_beta))
    center_determinant = closed_shell_determinant_energy(center_tensors, n_occ)
    center_fci = float(fci_energy(center_tensors, particles)) if fci_surface else None
    steps = [("primary", float(q_step_au))]
    if q_step_alt_au is not None:
        steps.append(("alt", float(q_step_alt_au)))
    per_model: dict[str, list[LinearCoupling]] = {m: [] for m in models}
    forces: dict[str, list[float]] = {m: [] for m in models}
    curvatures: list[float] = []
    fci_curvatures: list[float] = []
    for mode in modes:
        derivs: dict[str, dict[str, Any]] = {}
        for step_kind, q in steps:
            side: dict[int, dict[str, Any]] = {}
            for sign in (1, -1):
                geometry = displace_along_mode(center.geometry, mode, masses_me, sign * q)
                snap = snapshot_factory(geometry)
                h, g, f, diag = align_to_center(snap, center, cross_overlap_ao=cross_overlap_factory(center, snap),
                                                blocks=blocks, partition=partition)
                if min(diag["block_min_singular_values"]) < float(block_overlap_tolerance):
                    raise ValueError(
                        f"an orbital crossed a frozen-core/active/external boundary between R0 and Q = {sign * q:+.3f} "
                        f"along mode {mode.mode_index} ({mode.label}): smallest overlap singular value per block "
                        f"{diag['block_min_singular_values']}. The canonical partition of the displaced SCF no longer "
                        "matches R0, so neither the couplings nor the FCI surface along this mode are defined by index; "
                        "reduce q_step_au or change the active space.")
                full = frozen_core_contract(h, g, snap.nuclear_repulsion, active_space)
                fock_scalar, fock_active = frozen_core_contract_one_body(f, active_space)
                side[sign] = {"full": full, "fock_scalar": fock_scalar, "fock": fock_active,
                              "hf_energy": snap.hf_energy, "alignment": diag}
                if fci_surface:
                    # The displaced SCF's own canonical core and orbitals: the frozen-core
                    # FCI energy does not depend on the alignment, only on the partition.
                    canonical = frozen_core_contract(snap.h_mo, snap.eri_mo, snap.nuclear_repulsion, active_space)
                    side[sign]["fci_energy"] = float(fci_energy(canonical, particles))
            # Frozen-reference curvature: the R0 determinant evaluated with the
            # displaced, aligned integrals. Its second difference is the bare
            # curvature K_nu = d2/dQ2 <HF0|H_el(Q)|HF0>; the RHF Hessian's
            # omega_nu^2 is K_nu minus the HF orbital relaxation.
            frozen_plus = closed_shell_determinant_energy(side[1]["full"], n_occ)
            frozen_minus = closed_shell_determinant_energy(side[-1]["full"], n_occ)
            derivs[step_kind] = {
                "scalar": _central_difference(side[1]["full"].scalar, side[-1]["full"].scalar, q),
                "one_body": _central_difference(side[1]["full"].one_body, side[-1]["full"].one_body, q),
                "two_body": _central_difference(side[1]["full"].two_body, side[-1]["full"].two_body, q),
                "fock_scalar": _central_difference(side[1]["fock_scalar"], side[-1]["fock_scalar"], q),
                "fock": _central_difference(side[1]["fock"], side[-1]["fock"], q),
                "hf_energy_derivative": _central_difference(side[1]["hf_energy"], side[-1]["hf_energy"], q),
                "frozen_reference_curvature": (frozen_plus - 2.0 * center_determinant + frozen_minus) / (q * q),
                "alignment": {s: side[s]["alignment"] for s in (1, -1)},
            }
            if fci_surface:
                derivs[step_kind]["fci_surface_curvature"] = (
                    side[1]["fci_energy"] - 2.0 * center_fci + side[-1]["fci_energy"]) / (q * q)
        primary = derivs["primary"]
        drift: dict[str, dict[str, float]] = {}
        if "alt" in derivs:
            for key in ("scalar", "one_body", "two_body", "fock", "frozen_reference_curvature",
                        *(("fci_surface_curvature",) if fci_surface else ())):
                drift[key] = _difference_diagnostics(primary[key], derivs["alt"][key])
        omega_sq = float(mode.frequency_hartree) ** 2
        curvature = float(primary["frozen_reference_curvature"])
        curvatures.append(curvature)
        base_diag = {
            "finite_difference_drift": drift,
            "hf_energy_derivative_hartree_per_q": float(primary["hf_energy_derivative"]),
            "alignment": alignment,
            "alignment_min_singular_value": min(float(primary["alignment"][s]["min_singular_value"]) for s in (1, -1)),
            "alignment_occupied_block_min_singular_value": min(
                float(primary["alignment"][s]["occupied_block_min_singular_value"]) for s in (1, -1)),
            # Smallest overlap singular value inside the frozen-core, active and external
            # blocks over both displacements: near 1 unless an orbital crossed a boundary.
            "alignment_block_min_singular_values": [
                min(float(primary["alignment"][s]["block_min_singular_values"][b]) for s in (1, -1))
                for b in range(len(partition))],
            "rhf_hessian_curvature_hartree_per_q2": omega_sq,
            "frozen_reference_curvature_hartree_per_q2": curvature,
            "frozen_reference_frequency_cm1": math.sqrt(max(curvature, 0.0)) * HARTREE_TO_CM_INV,
            "hf_orbital_relaxation_hartree_per_q2": omega_sq - curvature,
        }
        if fci_surface:
            k_pes = float(primary["fci_surface_curvature"])
            fci_curvatures.append(k_pes)
            base_diag["fci_surface_curvature_hartree_per_q2"] = k_pes
            base_diag["fci_surface_frequency_cm1"] = math.sqrt(max(k_pes, 0.0)) * HARTREE_TO_CM_INV
        if "full_linear" in per_model:
            coupling = LinearCoupling(mode.mode_index, "full_linear", float(primary["scalar"]), primary["one_body"],
                                      primary["two_body"], float(q_step_au), base_diag)
            per_model["full_linear"].append(coupling)
            forces["full_linear"].append(closed_shell_determinant_energy(
                ActiveTensors(coupling.scalar, coupling.one_body, coupling.two_body), n_occ))
        if "mean_field_fd" in per_model:
            coupling = LinearCoupling(mode.mode_index, "mean_field_fd", float(primary["fock_scalar"]), primary["fock"],
                                      None, float(q_step_au), base_diag)
            per_model["mean_field_fd"].append(coupling)
            forces["mean_field_fd"].append(coupling.scalar + 2.0 * float(np.trace(coupling.one_body[:n_occ, :n_occ])))
    return CouplingComputation(
        center=center, active_space=active_space, active_tensors=center_tensors, modes=tuple(modes),
        masses_me=np.asarray(masses_me, dtype=float),
        couplings={m: tuple(v) for m, v in per_model.items()},
        hf_force_by_mode={m: tuple(v) for m, v in forces.items()},
        frozen_reference_curvature_by_mode=tuple(curvatures),
        timing_seconds={"couplings": time.time() - t0},
        alignment=str(alignment),
        fci_surface_curvature_by_mode=tuple(fci_curvatures),
        center_fci_energy=center_fci,
    )


def mean_field_pyscf_eph_couplings(computation: CouplingComputation) -> tuple[LinearCoupling, ...]:
    """Doron-style vertex (PySCF ``eph``) as frozen-core-contracted one-body couplings, per mode."""
    vertex = pyscf_mean_field_eph_vertex(computation.center, computation.modes, computation.masses_me)
    out = []
    for mode, g in zip(computation.modes, vertex):
        # The eph vertex already carries 1/sqrt(2 omega); undo it so the coupling is per unit Q.
        per_q = g * math.sqrt(2.0 * float(mode.frequency_hartree))
        scalar, active = frozen_core_contract_one_body(per_q, computation.active_space)
        out.append(LinearCoupling(mode.mode_index, "mean_field_pyscf_eph", scalar, active, None, 0.0,
                                  {"source": "pyscf.eph analytic vertex, AO basis fixed"}))
    return tuple(out)


# --------------------------------------------------------------- assembly


def reference_force(coupling: LinearCoupling, n_occ: int) -> float:
    """<HF| C_nu |HF>: the linear force on mode nu in the closed-shell reference determinant."""
    if coupling.two_body is None:
        return float(coupling.scalar + 2.0 * np.trace(np.asarray(coupling.one_body)[:int(n_occ), :int(n_occ)]))
    return closed_shell_determinant_energy(ActiveTensors(coupling.scalar, coupling.one_body, coupling.two_body), int(n_occ))


def subtract_reference_force(couplings: Sequence[LinearCoupling], n_occ: int) -> tuple[LinearCoupling, ...]:
    """C_nu -> C_nu - <HF|C_nu|HF>, so the oscillators stay centred on the reference geometry.

    The harmonic term omega^2 Q^2 / 2 already describes the total energy around
    its minimum, so a linear force that survives in the reference state would
    displace every oscillator and double count the geometry relaxation. For the
    full linear model that force is the residual energy gradient (about zero at
    an optimized geometry); for a mean-field vertex it is 2 sum_i g_ii, which is
    large, so this counter-term is what makes a mean-field model usable.
    """
    out = []
    for c in couplings:
        force = reference_force(c, n_occ)
        diag = {**dict(c.diagnostics), "reference_force_subtracted_hartree_per_q": force}
        out.append(LinearCoupling(c.mode_index, c.model, float(c.scalar) - force, c.one_body, c.two_body, c.q_step_au, diag))
    return tuple(out)


def _active_problem(t: ActiveTensors, num_particles: tuple[int, int], *, basis: str = "") -> RestrictedClosedShellMolecularProblem:
    n = int(t.one_body.shape[0])
    return RestrictedClosedShellMolecularProblem(
        geometry_spec="active-space tensors", basis=str(basis), charge=0, multiplicity=1, reference="rhf",
        n_spatial_orbitals=n, n_alpha=int(num_particles[0]), n_beta=int(num_particles[1]),
        hf_energy=float(t.scalar), nuclear_repulsion_energy=float(t.scalar),
        one_body_integrals_mo=np.asarray(t.one_body, dtype=float),
        two_body_integrals_mo=np.asarray(t.two_body, dtype=float),
    )


def fermion_polynomial_from_tensors(
    scalar: float, one_body: np.ndarray, two_body: np.ndarray | None, num_particles: tuple[int, int], *, tol: float = 1.0e-12,
) -> PauliPolynomial:
    """Jordan-Wigner (blocked spin order) polynomial of E + sum h a^dag a + 1/2 sum g a^dag a^dag a a."""
    n = int(np.asarray(one_body).shape[0])
    g = np.zeros((n, n, n, n)) if two_body is None else np.asarray(two_body, dtype=float)
    poly = build_restricted_closed_shell_molecular_hamiltonian(
        _active_problem(ActiveTensors(float(scalar), np.asarray(one_body, dtype=float), g), num_particles), tol=tol)
    return _clean_real_polynomial(poly, tol=tol)


def build_register_layout(n_spatial_orbitals: int, modes: Sequence[NormalMode], cutoffs: Sequence[int], encoding: str) -> RegisterLayout:
    n_fermion = 2 * int(n_spatial_orbitals)
    start = n_fermion
    blocks = []
    for mode, cutoff in zip(modes, cutoffs):
        if int(cutoff) < 1:
            raise ValueError("each mode cutoff n_ph_max must be >= 1.")
        qpb = boson_qubits_per_site(int(cutoff), encoding=str(encoding))
        blocks.append(BosonModeRegister(int(mode.mode_index), str(mode.label), int(start), int(qpb), int(cutoff), str(encoding)))
        start += int(qpb)
    return RegisterLayout(n_fermion, tuple(range(n_fermion)), tuple(blocks), "blocked")


def assemble_vibronic_model(
    *,
    geometry: MolecularGeometry,
    basis: str,
    active_space: ActiveSpace,
    active_tensors: ActiveTensors,
    modes: Sequence[NormalMode],
    couplings: Sequence[LinearCoupling],
    cutoffs: Sequence[int] | int,
    encoding: str = "binary",
    coupling_model: str | None = None,
    tol: float = 1.0e-12,
    provenance: Mapping[str, Any] | None = None,
) -> VibronicQubitModel:
    """H = H_el + sum omega (n + 1/2) + sum C_nu Q_nu on the fermion+boson register."""
    modes = tuple(modes)
    cut = (int(cutoffs),) * len(modes) if isinstance(cutoffs, int) else tuple(int(v) for v in cutoffs)
    if len(cut) != len(modes) or len(couplings) != len(modes):
        raise ValueError("one cutoff and one coupling per mode are required.")
    model_name = coupling_model or (couplings[0].model if couplings else "none")
    if any(c.model != model_name for c in couplings):
        raise ValueError("all couplings must come from the same coupling model.")
    for mode, coupling in zip(modes, couplings):
        if int(coupling.mode_index) != int(mode.mode_index):
            raise ValueError(f"coupling for mode {coupling.mode_index} paired with mode {mode.mode_index}; keep them in the same order.")
    particles = (int(active_space.n_alpha), int(active_space.n_beta))
    layout = build_register_layout(active_tensors.one_body.shape[0], modes, cut, encoding)
    nq = int(layout.n_total_qubits)
    n_boson = int(layout.n_boson_qubits)
    h_el = fermion_polynomial_from_tensors(active_tensors.scalar, active_tensors.one_body, active_tensors.two_body, particles, tol=tol)
    h_vib = _lift_fermion_polynomial(h_el, boson_qubits=n_boson)
    c_ops, q_ops, p_ops, n_ops = [], [], [], []
    for mode, block, coupling in zip(modes, layout.boson_modes, couplings):
        omega = float(mode.frequency_hartree)
        c_poly = fermion_polynomial_from_tensors(coupling.scalar, coupling.one_body, coupling.two_body, particles, tol=tol)
        q_op = _clean_real_polynomial((1.0 / math.sqrt(2.0 * omega)) * boson_operator(
            "JW", nq, block.qubits, which="x", n_ph_max=int(block.n_ph_max), encoding=str(encoding)), tol=tol)
        p_op = _clean_real_polynomial(math.sqrt(omega / 2.0) * _boson_momentum_operator(
            nq_total=nq, boson_qubits=block.qubits, n_ph_max=int(block.n_ph_max), boson_encoding=str(encoding)), tol=tol)
        n_op = _clean_real_polynomial(boson_operator(
            "JW", nq, block.qubits, which="n", n_ph_max=int(block.n_ph_max), encoding=str(encoding)), tol=tol)
        h_vib = h_vib + (omega * (n_op + 0.5)) + (_lift_fermion_polynomial(c_poly, boson_qubits=n_boson) * q_op)
        c_ops.append(c_poly)
        q_ops.append(q_op)
        p_ops.append(p_op)
        n_ops.append(n_op)
    h_vib = _clean_real_polynomial(h_vib, tol=tol)
    return VibronicQubitModel(
        geometry=geometry, basis=str(basis), active_space=active_space, modes=modes, cutoffs=cut,
        encoding=str(encoding), coupling_model=str(model_name), layout=layout, h_electronic=h_el, h_vibronic=h_vib,
        coupling_operators=tuple(c_ops), q_operators=tuple(q_ops), p_operators=tuple(p_ops), n_operators=tuple(n_ops),
        active_tensors=active_tensors, couplings=tuple(couplings), provenance=dict(provenance or {}),
    )


# ------------------------------------------------------- exact references


def physical_sector_basis(layout: RegisterLayout, *, n_spatial_orbitals: int, num_particles: tuple[int, int]) -> np.ndarray:
    """Sorted computational-basis indices with N_alpha, N_beta electrons and n_nu <= n_ph_max."""
    n = int(n_spatial_orbitals)
    n_alpha, n_beta = (int(v) for v in num_particles)
    if int(layout.n_total_qubits) > 62:
        raise ValueError(f"the sector basis uses int64 bit masks; {layout.n_total_qubits} qubits exceed 62.")
    alpha = np.array([sum(1 << q for q in occ) for occ in itertools.combinations(range(n), n_alpha)], dtype=np.int64)
    beta = np.array([sum(1 << (n + q) for q in occ) for occ in itertools.combinations(range(n), n_beta)], dtype=np.int64)
    fermion = (alpha[:, None] | beta[None, :]).reshape(-1)
    boson = np.zeros(1, dtype=np.int64)
    for block in layout.boson_modes:
        if str(block.encoding) == "binary":
            levels = np.arange(int(block.n_ph_max) + 1, dtype=np.int64) << int(block.qubit_start)
        elif str(block.encoding) == "unary":
            levels = np.int64(1) << (int(block.qubit_start) + np.arange(int(block.n_ph_max) + 1, dtype=np.int64))
        else:
            raise ValueError(f"unknown boson encoding {block.encoding!r}.")
        boson = (boson[:, None] | levels[None, :]).reshape(-1)
    return np.sort((fermion[:, None] | boson[None, :]).reshape(-1))


def sector_sparse_matrix(poly: PauliPolynomial, basis: np.ndarray, *, coeff_tol: float = 1.0e-12) -> Any:
    """P H P on the sector basis as a CSR matrix (rows and columns in basis order).

    Each Pauli word acts on every basis index at once: X and Y flip bits, Z and
    Y contribute (-1)^bit, and each Y contributes i(-1)^bit. Words whose image
    leaves the sector are dropped, which is exact for a projected Hamiltonian.
    """
    from scipy.sparse import coo_matrix  # type: ignore

    basis = np.asarray(basis, dtype=np.int64)
    if basis.ndim != 1 or (basis.size > 1 and not np.all(np.diff(basis) > 0)):
        raise ValueError("basis must be a strictly increasing 1-D array of computational-basis indices.")
    dim = int(basis.size)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(coeff_tol):
            continue
        word = str(term.pw2strng()).lower().replace("i", "e")
        nq = len(word)
        flip = 0
        zmask = 0
        n_y = 0
        for q in range(nq):
            op = word[nq - 1 - q]
            if op in "xy":
                flip |= 1 << q
            if op in "zy":
                zmask |= 1 << q
            if op == "y":
                n_y += 1
        images = basis ^ np.int64(flip)
        parity = np.bitwise_count(basis & np.int64(zmask)) & 1
        phase = coeff * (1j ** n_y) * np.where(parity == 1, -1.0, 1.0)
        position = np.searchsorted(basis, images)
        inside = (position < dim)
        inside[inside] &= basis[position[inside]] == images[inside]
        rows.append(position[inside])
        cols.append(np.nonzero(inside)[0])
        vals.append(phase[inside])
    if not rows:
        return coo_matrix((dim, dim), dtype=complex).tocsr()
    matrix = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim), dtype=complex).tocsr()
    matrix.sum_duplicates()
    return matrix


def exact_sector_energies(model: VibronicQubitModel, *, n_low: int = 4, tol: float = 1.0e-10, return_state: bool = False) -> tuple[tuple[float, ...], np.ndarray | None]:
    """Lowest eigenvalues of H_vibronic in the physical sector (and the ground state, full register, if asked)."""
    from scipy.sparse.linalg import eigsh  # type: ignore

    particles = (int(model.active_space.n_alpha), int(model.active_space.n_beta))
    basis = physical_sector_basis(model.layout, n_spatial_orbitals=model.active_tensors.one_body.shape[0], num_particles=particles)
    matrix = sector_sparse_matrix(model.h_vibronic, basis)
    dim = int(matrix.shape[0])
    k = max(1, int(n_low))
    if dim <= k + 1:
        evals, evecs = np.linalg.eigh(matrix.toarray())
    else:
        evals, evecs = eigsh(matrix, k=min(k, dim - 1), which="SA", tol=float(tol))
    order = np.argsort(np.real(evals))
    energies = tuple(float(np.real(evals[i])) for i in order[:k])
    if not return_state:
        return energies, None
    state = np.zeros(1 << model.n_qubits, dtype=complex)
    state[basis] = evecs[:, order[0]]
    state /= np.linalg.norm(state)
    return energies, state


def hartree_fock_state(model: VibronicQubitModel) -> np.ndarray:
    """|HF> x |vacuum>_bosons on the full register (binary vacuum 0...0; unary vacuum one-hot on level 0)."""
    occupied = hartree_fock_occupied_qubits(int(model.active_tensors.one_body.shape[0]),
                                            (int(model.active_space.n_alpha), int(model.active_space.n_beta)), indexing="blocked")
    index = sum(1 << int(q) for q in occupied)
    for block in model.layout.boson_modes:
        if str(block.encoding) == "unary":
            index |= 1 << int(block.qubit_start)
    state = np.zeros(1 << model.n_qubits, dtype=complex)
    state[index] = 1.0
    return state


# ------------------------------------------------------------ end to end


def build_molecular_vibronic_model(
    geometry: MolecularGeometry,
    *,
    basis: str = "sto-3g",
    active_space: str | Mapping[str, Any] = "frozen_core",
    cutoffs: Sequence[int] | int = 3,
    encoding: str = "binary",
    coupling_model: str = "full_linear",
    q_step_au: float = 0.1,
    q_step_alt_au: float | None = 0.05,
    masses_amu: Mapping[str, float] | None = None,
    mode_indices: Sequence[int] | None = None,
    reference_force_counter_term: bool | None = None,
    harmonic_curvature: str = "pes_consistent",
    alignment: str = "block_diagonal",
    degenerate_boundary_tolerance_hartree: float = 1.0e-6,
    stationary_gradient_tolerance: float = 1.0e-3,
    tol: float = 1.0e-12,
) -> tuple[VibronicQubitModel, CouplingComputation]:
    """Geometry in, qubit Hamiltonian out: RHF, Hessian, modes, displaced SCFs, assembly.

    ``mode_indices`` restricts the boson register to a subset of the modes
    (ascending-frequency indices); the other modes are dropped from the model.
    ``reference_force_counter_term`` subtracts <HF|C_nu|HF> from each coupling
    (see ``subtract_reference_force``); by default it is applied to the
    mean-field models and not to ``full_linear``.

    ``harmonic_curvature`` chooses the oscillator frequency omega_nu:

    * ``pes_consistent`` (default): omega_nu^2 = K_nu^PES + chi_nu, the
      curvature of the frozen-core FCI potential energy surface along the mode
      (central second difference of the active-space FCI energy of the
      displaced SCFs, each with its own canonical core) plus the model's own
      correlated response chi_nu = -d2/dQ2 E_FCI[H_el + Q C_nu] to the coupling
      actually assembled. By second-order perturbation theory the exactly
      solved model then has adiabatic curvature omega_nu^2 - chi_nu = K_nu^PES:
      the phonon is screened by everything except the electrons the model
      treats explicitly. Costs the FCI of the active space at the center, at
      every displaced geometry and at +/- q_step along each coupling.
      The off-diagonal of the FCI Hessian between modes of one irreducible
      representation is not reproduced (the model holds its cross response
      -chi_{mu nu} there); see the module docstring.
    * ``rhf_hessian``: omega_nu from the RHF Hessian (the Paper IV H2O
      definition). The Hessian already holds the Hartree-Fock orbital
      relaxation, omega_nu^2 = K_nu - chi_nu^HF, and the coupling regenerates
      that relaxation inside the model, so the model's surface has curvature
      omega_nu^2 - chi_nu: softer than the RHF surface, and off the FCI
      surface by omega_nu^2 - K_nu^PES - chi_nu, of either sign (H2O/STO-3G:
      +2.6e-6 on the bend, -2.3e-4 and -1.3e-4 Hartree per Q^2 on the
      stretches).
    * ``frozen_reference``: omega_nu^2 = K_nu = d2/dQ2 <HF0|H_el(Q)|HF0>, the
      R0 determinant transported rigidly to the displaced geometry: the
      Hartree-Fock analogue of ``pes_consistent``. With the block-diagonal
      alignment K_nu - chi_nu^HF is the RHF Hessian for any frozen-core /
      active / external partition, provided the canonical partition is the
      same at the displaced geometries (the builder raises otherwise).

    ``alignment`` is ``block_diagonal`` (frozen core, active and external
    orbitals aligned to R0 separately, so the frozen core is the canonical core
    of each displaced SCF) or ``full`` (one rotation of the whole MO set, the
    Paper IV H2O convention). With a frozen core the full rotation transports
    the R0 core rigidly and adds a spurious curvature, about the size of the
    H2O/STO-3G force constants, to the frozen-reference curvature (K^PES does
    not depend on the alignment), and it changes the ``full_linear`` couplings
    at the 1e-4 Hartree per Q level (0.5-5 % of their norm; the mean-field
    couplings are unchanged), which moves the pes_consistent frequencies by a
    few cm^-1 through chi and the H2O cutoff-3 exact ground energy by 2e-4
    Hartree. Reproducing the Paper IV H2O model therefore needs both
    ``harmonic_curvature="rhf_hessian"`` and ``alignment="full"``.
    The geometry is used as given (never optimized): the residual gradient is
    recorded and a warning is issued above ``stationary_gradient_tolerance``
    Hartree/Bohr.
    """
    import warnings

    t0 = time.time()
    if harmonic_curvature not in HARMONIC_CURVATURES:
        raise ValueError(f"harmonic_curvature must be one of {HARMONIC_CURVATURES}; got {harmonic_curvature!r}.")
    if alignment not in ALIGNMENTS:
        raise ValueError(f"alignment must be one of {ALIGNMENTS}; got {alignment!r}.")
    center = pyscf_rhf_snapshot(geometry, basis=basis)
    space = select_active_space(center.n_spatial_orbitals, center.n_alpha, center.n_beta, policy=active_space, symbols=geometry.symbols)
    _check_degenerate_orbital_boundaries(center.mo_energy, space, tolerance=degenerate_boundary_tolerance_hartree)
    masses = isotope_masses_me(geometry.symbols, masses_amu)
    gradient = np.asarray(center.backend_handle["mf"].nuc_grad_method().kernel(), dtype=float)
    gradient_max = float(np.max(np.abs(gradient)))
    if gradient_max > float(stationary_gradient_tolerance):
        warnings.warn(f"geometry is not a stationary point: max |dE/dR| = {gradient_max:.2e} Hartree/Bohr; "
                      "the harmonic model and the reference force assume an optimized geometry.", stacklevel=2)
    hessian = pyscf_hessian(center)
    all_modes = vibrational_modes_from_hessian(hessian, coordinates_bohr=geometry.coordinates_bohr, masses_me=masses)
    if mode_indices is None:
        modes = all_modes
    else:
        chosen = [int(i) for i in mode_indices]
        if len(set(chosen)) != len(chosen) or any(i < 0 or i >= len(all_modes) for i in chosen):
            raise ValueError(f"mode_indices must be unique and within 0..{len(all_modes) - 1}; got {chosen}.")
        modes = tuple(all_modes[i] for i in chosen)
    n_cutoffs = 1 if isinstance(cutoffs, int) else len(tuple(cutoffs))
    if n_cutoffs not in (1, len(modes)):
        raise ValueError(f"{n_cutoffs} cutoffs given for {len(modes)} modes; give one value or one per mode.")
    # Both finite-difference models come from the same displaced SCFs, so both
    # are always kept on the computation; only the assembled model picks one.
    computation = compute_linear_couplings(center, modes, masses_me=masses, active_space=space, q_step_au=q_step_au,
                                           q_step_alt_au=q_step_alt_au, models=COUPLING_MODELS, alignment=alignment,
                                           fci_surface=(harmonic_curvature == "pes_consistent"))
    if coupling_model == "mean_field_pyscf_eph":
        couplings = mean_field_pyscf_eph_couplings(computation)
    else:
        couplings = computation.couplings[coupling_model]
    if reference_force_counter_term is None:
        reference_force_counter_term = coupling_model != "full_linear"
    if reference_force_counter_term:
        couplings = subtract_reference_force(couplings, int(space.n_alpha))
    particles = (int(space.n_alpha), int(space.n_beta))
    model_modes = modes
    curvature_record: dict[str, Any] = {}
    if harmonic_curvature == "frozen_reference":
        model_modes = tuple(
            NormalMode(m.mode_index, _oscillator_frequency(k, mode=m, source="frozen-reference"), m.mass_weighted_vector, m.label)
            for m, k in zip(modes, computation.frozen_reference_curvature_by_mode))
    elif harmonic_curvature == "pes_consistent":
        # chi_nu for the coupling that goes into the model (model dependent: a mean-field
        # vertex has a different response than the full linear coupling).
        responses = [active_response_curvature(computation.active_tensors, c, particles, q_step_au=q_step_au) for c in couplings]
        response_drift = None
        if q_step_alt_au is not None:
            alt = [active_response_curvature(computation.active_tensors, c, particles, q_step_au=q_step_alt_au) for c in couplings]
            response_drift = [_difference_diagnostics(a, b) for a, b in zip(responses, alt)]
        omega_sq = [k + chi for k, chi in zip(computation.fci_surface_curvature_by_mode, responses)]
        model_modes = tuple(
            NormalMode(m.mode_index, _oscillator_frequency(w2, mode=m, source="K_PES + chi"), m.mass_weighted_vector, m.label)
            for m, w2 in zip(modes, omega_sq))
        curvature_record = {
            "fci_surface_curvature_hartree_per_q2": list(computation.fci_surface_curvature_by_mode),
            "active_response_hartree_per_q2": [float(v) for v in responses],
            "active_response_drift": response_drift,
            "pes_consistent_frequencies_cm1": [m.frequency_cm1 for m in model_modes],
            "center_fci_energy_hartree": computation.center_fci_energy,
        }
    provenance = {
        "producer": "src.quantum.chemistry.molecular_vibronic", "schema": MODEL_SCHEMA,
        "reference_force_counter_term": bool(reference_force_counter_term),
        "harmonic_curvature": str(harmonic_curvature), "alignment": str(alignment),
        "rhf_hessian_frequencies_cm1": [m.frequency_cm1 for m in modes],
        "frozen_reference_frequencies_cm1": [math.sqrt(max(k, 0.0)) * HARTREE_TO_CM_INV
                                             for k in computation.frozen_reference_curvature_by_mode],
        **curvature_record,
        "backend": center.backend, "backend_version": center.backend_version, "hf_energy_hartree": center.hf_energy,
        "scf": {"reference": "rhf", "conv_tol": 1.0e-12, "conv_tol_grad": 1.0e-9},
        "geometry_optimized_here": False, "center_gradient_max_abs_hartree_per_bohr": gradient_max,
        "nuclear_repulsion_hartree": center.nuclear_repulsion, "n_spatial_orbitals_full": center.n_spatial_orbitals,
        "mo_energies_hartree": center.mo_energy.tolist(),
        "masses_me": masses.tolist(), "q_step_au": float(q_step_au), "q_step_alt_au": q_step_alt_au,
        "mode_indices": None if mode_indices is None else [int(i) for i in mode_indices],
        "all_mode_frequencies_cm1": [m.frequency_cm1 for m in all_modes],
        "hf_force_by_mode": {k: list(v) for k, v in computation.hf_force_by_mode.items()},
    }
    model = assemble_vibronic_model(geometry=geometry, basis=basis, active_space=space, active_tensors=computation.active_tensors,
                                    modes=model_modes, couplings=couplings, cutoffs=cutoffs, encoding=encoding,
                                    coupling_model=coupling_model, tol=tol, provenance=provenance)
    model.provenance["wall_seconds"] = time.time() - t0
    return model, computation


def _check_degenerate_orbital_boundaries(mo_energy: np.ndarray, space: ActiveSpace, *, tolerance: float) -> None:
    """Refuse an active space whose frozen/active or active/external boundary
    splits orbitals that are degenerate to within ``tolerance``: the SCF then
    picks an arbitrary orientation inside the degenerate pair, and the sliced
    tensors (and every displaced-geometry derivative) depend on it."""
    energies = np.asarray(mo_energy, dtype=float)
    boundaries = []
    if space.frozen_core and space.active:
        boundaries.append((max(space.frozen_core), min(space.active), "frozen core / active"))
    if space.active and space.external:
        boundaries.append((max(space.active), min(space.external), "active / external"))
    for low, high, label in boundaries:
        if abs(float(energies[high]) - float(energies[low])) < float(tolerance):
            raise ValueError(
                f"the {label} boundary splits degenerate orbitals {low} and {high} "
                f"(epsilon {energies[low]:.8f} and {energies[high]:.8f} Hartree); enlarge or shrink the active space.")


# ------------------------------------------------------------- serialization


def vibronic_model_to_jsonable(model: VibronicQubitModel) -> dict[str, Any]:
    layout = model.layout
    return {
        "schema": MODEL_SCHEMA,
        "energy_units": "hartree",
        "fermion_mapping": "jordan_wigner",
        "spin_orbital_ordering": "blocked",
        "pauli_word_order": "q_(n-1)...q_0",
        "ladder_convention": LADDER_CONVENTION,
        "geometry": model.geometry.to_jsonable(),
        "basis": model.basis,
        "active_space": model.active_space.to_jsonable(),
        "active_tensors": {"scalar_hartree": model.active_tensors.scalar,
                           "one_body": model.active_tensors.one_body.tolist(),
                           "two_body_convention": "chemist_eri_pqrs",
                           "two_body": model.active_tensors.two_body.tolist()},
        "modes": [m.to_jsonable() for m in model.modes],
        "cutoffs": list(model.cutoffs),
        "boson_encoding": model.encoding,
        "coupling_model": model.coupling_model,
        "couplings": [{"mode_index": c.mode_index, "model": c.model, "scalar_hartree_per_q": c.scalar,
                       "one_body": np.asarray(c.one_body).tolist(),
                       "two_body": None if c.two_body is None else np.asarray(c.two_body).tolist(),
                       "q_step_au": c.q_step_au, "diagnostics": dict(c.diagnostics)} for c in model.couplings],
        "register_layout": {"n_fermion_qubits": layout.n_fermion_qubits, "n_total_qubits": layout.n_total_qubits,
                            "boson_modes": [{"mode_index": b.mode_index, "label": b.mode_label, "qubit_start": b.qubit_start,
                                             "n_qubits": b.n_qubits, "n_ph_max": b.n_ph_max, "encoding": b.encoding}
                                            for b in layout.boson_modes]},
        "operators": {"h_electronic": pauli_polynomial_to_jsonable(model.h_electronic),
                      "h_vibronic": pauli_polynomial_to_jsonable(model.h_vibronic),
                      "coupling_by_mode": [pauli_polynomial_to_jsonable(p) for p in model.coupling_operators],
                      "q_by_mode": [pauli_polynomial_to_jsonable(p) for p in model.q_operators],
                      "p_by_mode": [pauli_polynomial_to_jsonable(p) for p in model.p_operators],
                      "n_by_mode": [pauli_polynomial_to_jsonable(p) for p in model.n_operators]},
        "provenance": dict(model.provenance),
    }


def write_vibronic_model(path: str | Path, model: VibronicQubitModel) -> None:
    Path(path).write_text(json.dumps(vibronic_model_to_jsonable(model), indent=1, sort_keys=True, allow_nan=False) + "\n",
                          encoding="utf-8")


def load_vibronic_hamiltonian(path: str | Path) -> tuple[PauliPolynomial, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != MODEL_SCHEMA:
        raise ValueError(f"unsupported schema {payload.get('schema')!r}.")
    return pauli_polynomial_from_jsonable(payload["operators"]["h_vibronic"]), payload


# ---------------------------------------------------------------------- CLI


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a linear vibronic qubit Hamiltonian for a closed-shell molecule.")
    parser.add_argument("--geometry", required=True, help="'Sym x y z; Sym x y z; ...' or a path to an .xyz-style block")
    parser.add_argument("--units", default="angstrom", choices=["angstrom", "bohr"])
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--active-space", default="frozen_core", help="full | frozen_core | frozen_core:<n> | cas(<ne>,<no>)")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[3], help="n_ph_max, one value or one per mode")
    parser.add_argument("--encoding", default="binary", choices=["binary", "unary"])
    parser.add_argument("--coupling", default="full_linear", choices=[*COUPLING_MODELS, "mean_field_pyscf_eph"])
    parser.add_argument("--reference-force-counter-term", choices=["auto", "on", "off"], default="auto",
                        help="subtract <HF|C|HF> from each coupling (auto: on for mean-field models only)")
    parser.add_argument("--harmonic-curvature", choices=list(HARMONIC_CURVATURES), default="pes_consistent",
                        help="oscillator frequency: omega^2 = K_PES + chi (FCI surface curvature plus the model's own "
                             "response; default), the RHF Hessian (Paper IV H2O), or the bare frozen-reference curvature")
    parser.add_argument("--alignment", choices=list(ALIGNMENTS), default="block_diagonal",
                        help="displaced-orbital alignment: block_diagonal keeps the canonical frozen core (default); "
                             "full rotates the whole MO set (Paper IV H2O)")
    parser.add_argument("--q-step", type=float, default=0.1)
    parser.add_argument("--q-step-alt", type=float, default=0.05)
    parser.add_argument("--modes", type=int, nargs="*", default=None, help="subset of mode indices (ascending frequency)")
    parser.add_argument("--exact", type=int, default=0, help="also report this many lowest exact sector energies")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    spec = str(args.geometry)
    if ";" not in spec and "\n" not in spec:
        try:
            if Path(spec).is_file():
                spec = Path(spec).read_text(encoding="utf-8")
        except OSError:
            pass   # an inline geometry too long for a path name
    geometry = parse_geometry(spec, units=args.units, charge=args.charge)
    cutoffs = args.cutoffs[0] if len(args.cutoffs) == 1 else args.cutoffs
    model, computation = build_molecular_vibronic_model(
        geometry, basis=args.basis, active_space=args.active_space, cutoffs=cutoffs, encoding=args.encoding,
        coupling_model=args.coupling, q_step_au=args.q_step, q_step_alt_au=args.q_step_alt, mode_indices=args.modes,
        reference_force_counter_term={"auto": None, "on": True, "off": False}[args.reference_force_counter_term],
        harmonic_curvature=args.harmonic_curvature, alignment=args.alignment)
    report: dict[str, Any] = {"n_qubits": model.n_qubits, "n_terms": len(model.h_vibronic.return_polynomial()),
                              "frequencies_cm1": [m.frequency_cm1 for m in model.modes]}
    if int(args.exact) > 0:
        energies, _ = exact_sector_energies(model, n_low=int(args.exact))
        report["exact_sector_energies_hartree"] = list(energies)
    model = VibronicQubitModel(**{**model.__dict__, "provenance": {**model.provenance, "report": report}})
    write_vibronic_model(args.output, model)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

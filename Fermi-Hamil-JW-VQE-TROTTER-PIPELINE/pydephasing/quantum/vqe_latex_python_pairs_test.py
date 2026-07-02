# Auto-generated from vqe_latex_python_pairs_test.ipynb (benchmark cell removed)
# Do not edit by hand unless you also update the notebook conversion logic.

# pydephasing/quantum/vqe_latex_python_pairs.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable when notebook is run from nested CWDs.
_cwd = Path.cwd().resolve()
for _candidate in (_cwd, *_cwd.parents):
    if (_candidate / 'pydephasing' / 'quantum' / 'pauli_polynomial_class.py').exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import inspect
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from IPython.display import Markdown, Math as IPyMath, display
except Exception:  # pragma: no cover
    Markdown = None
    IPyMath = None
    display = None

try:
    from pydephasing.utilities.log import log
except Exception:  # pragma: no cover
    class _FallbackLog:
        @staticmethod
        def error(msg: str):
            raise RuntimeError(msg)

        @staticmethod
        def info(msg: str):
            print(msg)

    log = _FallbackLog()

try:
    from pydephasing.quantum.pauli_polynomial_class import (
        PauliPolynomial,
        fermion_minus_operator,
        fermion_plus_operator,
    )
    from pydephasing.quantum.pauli_words import PauliTerm
except Exception as _dep_exc:  # pragma: no cover
    PauliPolynomial = Any  # type: ignore[assignment]

    def _missing_dep(*_args, **_kwargs):
        raise ImportError(
            "pydephasing quantum dependencies are unavailable in this environment"
        ) from _dep_exc

    fermion_minus_operator = _missing_dep  # type: ignore[assignment]
    fermion_plus_operator = _missing_dep  # type: ignore[assignment]
    PauliTerm = _missing_dep  # type: ignore[assignment]

try:
    # Reuse your canonical Hubbard lattice helpers for consistent indexing/edges.
    from pydephasing.quantum.hubbard_latex_python_pairs import (
        Dims,
        SPIN_DN,
        SPIN_UP,
        Spin,
        bravais_nearest_neighbor_edges,
        mode_index,
        n_sites_from_dims,
    )
except Exception:  # pragma: no cover
    Dims = Union[int, Tuple[int, ...]]
    Spin = int
    SPIN_UP = 0
    SPIN_DN = 1

    def n_sites_from_dims(dims: Dims) -> int:
        if isinstance(dims, int):
            return int(dims)
        out = 1
        for L in dims:
            out *= int(L)
        return out

    def bravais_nearest_neighbor_edges(dims: Dims, pbc: Union[bool, Sequence[bool]] = True):
        raise ImportError("bravais_nearest_neighbor_edges unavailable (import Hubbard helpers)")

    def mode_index(site: int, spin: Spin, indexing: str = "interleaved", n_sites: Optional[int] = None) -> int:
        raise ImportError("mode_index unavailable (import Hubbard helpers)")


LATEX_TERMS: Dict[str, Dict[str, str]] = {
    "hamiltonian_sum": {
        "title": "Hamiltonian (Pauli Expansion)",
        "latex": (
            r"\hat H := \sum_{j=1}^{m} h_j\,\hat P_j,"
            r"\qquad"
            r"\hat P_j := \bigotimes_{i=0}^{N-1} \sigma_i^{(j)}"
        ),
    },
    "energy_expectation": {
        "title": "VQE Energy",
        "latex": (
            r"E(\vec\theta_k)"
            r":="
            r"\langle \psi(\vec\theta_k)|\hat H|\psi(\vec\theta_k)\rangle"
            r"="
            r"\sum_{j=1}^{m} h_j\,\langle \psi(\vec\theta_k)|\hat P_j|\psi(\vec\theta_k)\rangle"
        ),
    },
    "ansatz_state": {
        "title": "Parameterized State (Ansatz)",
        "latex": (
            r"|\psi(\vec\theta)\rangle"
            r":="
            r"\hat U(\vec\theta)\,|\psi_{\mathrm{ref}}\rangle"
            r":="
            r"\hat U_{p}(\theta_{p})\cdots \hat U_{2}(\theta_{2})\hat U_{1}(\theta_{1})|0\rangle^{\otimes N}"
        ),
    },
    "pauli_rotation": {
        "title": "Pauli Rotation Primitive",
        "latex": (
            r"R_{P}(\varphi)"
            r":="
            r"\exp\!\left(-i\frac{\varphi}{2}P\right),"
            r"\qquad P^2=I"
        ),
    },
    "trotter_step": {
        "title": "First-Order (Lie) Product Formula",
        "latex": (
            r"\exp\!\left(-i\,\theta\sum_{j=1}^{m}h_j P_j\right)"
            r"\approx"
            r"\prod_{j=1}^{m}\exp\!\left(-i\,\theta\,h_j P_j\right)"
        ),
    },
}

def _normalize_pauli_string(pauli: str) -> str:
    """
    Accept e/x/y/z (repo default) or I/X/Y/Z, return lower-case with 'e' as identity.
    Pauli-word order: left-to-right is q_(n-1) ... q_0.
    """
    if not isinstance(pauli, str):
        log.error("pauli string must be a str")
    p = pauli.strip()
    trans = {"I": "e", "X": "x", "Y": "y", "Z": "z"}
    out = []
    for ch in p:
        out.append(trans[ch] if ch in trans else ch.lower())
    return "".join(out)


def basis_state(nq: int, bitstring: Optional[str] = None) -> np.ndarray:
    r"""
    |b\rangle in C^(2^nq), with bitstring ordered q_(n-1)...q_0.
    """
    nq_i = int(nq)
    if nq_i <= 0:
        log.error("nq must be positive")
    dim = 1 << nq_i
    psi = np.zeros(dim, dtype=complex)
    idx = 0 if bitstring is None else int(bitstring, 2)
    if bitstring is not None and len(bitstring) != nq_i:
        log.error("bitstring length must equal nq")
    psi[idx] = 1.0 + 0.0j
    return psi


def apply_pauli_string(state: np.ndarray, pauli: str) -> np.ndarray:
    r"""
    Apply P in {I,X,Y,Z}^{\otimes n} to |psi>.
    Pauli string is ordered q_(n-1)...q_0 (left-to-right).
    """
    ps = _normalize_pauli_string(pauli)
    nq = len(ps)
    dim = int(state.size)
    if dim != (1 << nq):
        log.error("state length must be 2^n with n=len(pauli)")
    out = np.empty_like(state)

    for idx in range(dim):
        amp = state[idx]
        phase = 1.0 + 0.0j
        j = idx
        for q in range(nq):
            op = ps[nq - 1 - q]  # op on qubit q
            bit = (idx >> q) & 1
            if op == "e":
                continue
            if op == "z":
                if bit:
                    phase = -phase
                continue
            if op == "x":
                j ^= (1 << q)
                continue
            if op == "y":
                j ^= (1 << q)
                phase *= (1j if bit == 0 else -1j)
                continue
            log.error(f"invalid Pauli symbol '{op}' in string '{pauli}'")
        out[j] = phase * amp
    return out


def expval_pauli_string(state: np.ndarray, pauli: str) -> complex:
    r"""<psi|P|psi>."""
    return np.vdot(state, apply_pauli_string(state, pauli))


def expval_pauli_polynomial(state: np.ndarray, H: PauliPolynomial, tol: float = 1e-12) -> float:
    r"""<psi|H|psi>, with H a PauliPolynomial."""
    terms = H.return_polynomial()
    if not terms:
        return 0.0

    nq = int(terms[0].nqubit())
    id_str = "e" * nq

    acc = 0.0 + 0.0j
    for term in terms:
        ps = term.pw2strng()
        coeff = complex(term.p_coeff)
        if abs(coeff) < tol:
            continue
        if ps == id_str:
            acc += coeff
        else:
            acc += coeff * expval_pauli_string(state, ps)

    if abs(acc.imag) > 1e-8:
        log.error(f"Non-negligible imaginary energy residual: {acc}")
    return float(acc.real)

def apply_pauli_rotation(state: np.ndarray, pauli: str, angle: float) -> np.ndarray:
    r"""
    R_P(angle) := exp(-i angle/2 * P).
    Uses P^2=I, so exp(-i a P) = cos(a)I - i sin(a)P with a=angle/2.
    """
    Ppsi = apply_pauli_string(state, pauli)
    c = math.cos(0.5 * float(angle))
    s = math.sin(0.5 * float(angle))
    return c * state - 1j * s * Ppsi

def apply_exp_pauli_polynomial(
    state: np.ndarray,
    H: PauliPolynomial,
    theta: float,
    *,
    ignore_identity: bool = True,
    coefficient_tolerance: float = 1e-12,
    sort_terms: bool = True,
) -> np.ndarray:
    r"""
    Approximate exp(-i theta * H)|psi> with first-order product over Pauli terms.
    angle_j = 2 * theta * h_j for each Pauli term h_j P_j.
    """
    terms = H.return_polynomial()
    if not terms:
        return np.array(state, copy=True)

    nq = int(terms[0].nqubit())
    id_str = "e" * nq

    ordered = list(terms)
    if sort_terms:
        ordered.sort(key=lambda t: t.pw2strng())

    psi = np.array(state, copy=True)
    for term in ordered:
        ps = term.pw2strng()
        coeff = complex(term.p_coeff)
        if abs(coeff) < coefficient_tolerance:
            continue
        if ignore_identity and ps == id_str:
            continue
        if abs(coeff.imag) > coefficient_tolerance:
            log.error(f"non-negligible imaginary coefficient in term {ps}: {coeff}")
        angle = 2.0 * float(theta) * float(coeff.real)
        psi = apply_pauli_rotation(psi, ps, angle)
    return psi

def half_filled_num_particles(num_sites: int) -> Tuple[int, int]:
    L = int(num_sites)
    if L <= 0:
        log.error("num_sites must be positive")
    return ((L + 1) // 2, L // 2)


# Canonical HF helper imported from module to avoid logic drift.
try:
    from hartree_fock_reference_state import hartree_fock_bitstring
except Exception:
    from pydephasing.quantum.hartree_fock_reference_state import hartree_fock_bitstring

def jw_number_operator(repr_mode: str, nq: int, p_mode: int) -> PauliPolynomial:
    if repr_mode != "JW":
        log.error("jw_number_operator supports repr_mode='JW' only")
    nq_i = int(nq)
    p_i = int(p_mode)
    if p_i < 0 or p_i >= nq_i:
        log.error("mode index out of range -> 0 <= p_mode < nq")

    id_str = "e" * nq_i
    z_pos = nq_i - 1 - p_i
    z_str = ("e" * z_pos) + "z" + ("e" * (nq_i - 1 - z_pos))

    return PauliPolynomial(
        repr_mode,
        [
            PauliTerm(nq_i, ps=id_str, pc=0.5),
            PauliTerm(nq_i, ps=z_str, pc=-0.5),
        ],
    )


def hubbard_hop_term(
    nq: int,
    p_mode: int,
    q_mode: int,
    t: float,
    *,
    repr_mode: str = "JW",
) -> PauliPolynomial:
    cd_p = fermion_plus_operator(repr_mode, nq, int(p_mode))
    cm_q = fermion_minus_operator(repr_mode, nq, int(q_mode))
    cd_q = fermion_plus_operator(repr_mode, nq, int(q_mode))
    cm_p = fermion_minus_operator(repr_mode, nq, int(p_mode))
    return (-float(t)) * ((cd_p * cm_q) + (cd_q * cm_p))


def hubbard_onsite_term(
    nq: int,
    p_up: int,
    p_dn: int,
    U: float,
    *,
    repr_mode: str = "JW",
) -> PauliPolynomial:
    n_up = jw_number_operator(repr_mode, nq, int(p_up))
    n_dn = jw_number_operator(repr_mode, nq, int(p_dn))
    return float(U) * (n_up * n_dn)


def hubbard_potential_term(
    nq: int,
    p_mode: int,
    v_i: float,
    *,
    repr_mode: str = "JW",
) -> PauliPolynomial:
    return (-float(v_i)) * jw_number_operator(repr_mode, nq, int(p_mode))


def _parse_site_potential(
    v: Optional[Union[float, Sequence[float], Dict[int, float]]],
    n_sites: int,
) -> List[float]:
    if v is None:
        return [0.0] * int(n_sites)
    if isinstance(v, (int, float, complex)):
        return [float(v)] * int(n_sites)
    if isinstance(v, dict):
        out = [0.0] * int(n_sites)
        for k, val in v.items():
            idx = int(k)
            if idx < 0 or idx >= int(n_sites):
                log.error("site-potential key out of bounds")
            out[idx] = float(val)
        return out
    if len(v) != int(n_sites):
        log.error("site potential v must be scalar, dict, or length n_sites")
    return [float(val) for val in v]

@dataclass(frozen=True)
class AnsatzTerm:
    """One parameterized unitary U_k(theta_k) := exp(-i theta_k * H_k)."""

    label: str
    polynomial: PauliPolynomial


class HubbardTermwiseAnsatz:
    """
    Term-wise Hubbard ansatz (HVA-like), aligned with future Trotter time dynamics.

    Per layer, append:
      (A) all hopping terms  H_{<i,j>,sigma}^{(t)}
      (B) all onsite terms   H_i^{(U)}
      (C) all potential terms H_{i,sigma}^{(v)} (optional, only if v_i != 0)

    Full ansatz: reps repetitions of (A)->(B)->(C).
    """

    def __init__(
        self,
        dims: Dims,
        t: float,
        U: float,
        *,
        v: Optional[Union[float, Sequence[float], Dict[int, float]]] = None,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        edges: Optional[Sequence[Tuple[int, int]]] = None,
        pbc: Union[bool, Sequence[bool]] = True,
        include_potential_terms: bool = True,
    ):
        self.dims = dims
        self.n_sites = n_sites_from_dims(dims)
        self.nq = 2 * int(self.n_sites)

        self.t = float(t)
        self.U = float(U)
        self.repr_mode = repr_mode
        self.indexing = indexing

        self.edges = list(edges) if edges is not None else bravais_nearest_neighbor_edges(dims, pbc=pbc)
        self.v_list = _parse_site_potential(v, n_sites=int(self.n_sites))
        self.reps = int(reps)
        if self.reps <= 0:
            log.error("reps must be positive")

        self.include_potential_terms = bool(include_potential_terms)

        self.base_terms: List[AnsatzTerm] = []
        self._build_base_terms()

        self.num_parameters = self.reps * len(self.base_terms)

    def _build_base_terms(self) -> None:
        nq = int(self.nq)
        n_sites = int(self.n_sites)

        for (i, j) in self.edges:
            for spin in (SPIN_UP, SPIN_DN):
                p_i = mode_index(int(i), int(spin), indexing=self.indexing, n_sites=n_sites)
                p_j = mode_index(int(j), int(spin), indexing=self.indexing, n_sites=n_sites)
                poly = hubbard_hop_term(nq, p_i, p_j, t=self.t, repr_mode=self.repr_mode)
                self.base_terms.append(AnsatzTerm(label=f"hop(i={i},j={j},spin={spin})", polynomial=poly))

        for i in range(n_sites):
            p_up = mode_index(i, SPIN_UP, indexing=self.indexing, n_sites=n_sites)
            p_dn = mode_index(i, SPIN_DN, indexing=self.indexing, n_sites=n_sites)
            poly = hubbard_onsite_term(nq, p_up, p_dn, U=self.U, repr_mode=self.repr_mode)
            self.base_terms.append(AnsatzTerm(label=f"onsite(i={i})", polynomial=poly))

        if self.include_potential_terms:
            for i in range(n_sites):
                vi = float(self.v_list[i])
                if abs(vi) < 1e-15:
                    continue
                for spin in (SPIN_UP, SPIN_DN):
                    p_mode = mode_index(i, spin, indexing=self.indexing, n_sites=n_sites)
                    poly = hubbard_potential_term(nq, p_mode, v_i=vi, repr_mode=self.repr_mode)
                    self.base_terms.append(AnsatzTerm(label=f"pot(i={i},spin={spin})", polynomial=poly))

    def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ) -> np.ndarray:
        if int(theta.size) != int(self.num_parameters):
            log.error("theta has wrong length for this ansatz")
        psi = np.array(psi_ref, copy=True)
        k = 0
        for _ in range(self.reps):
            for term in self.base_terms:
                psi = apply_exp_pauli_polynomial(
                    psi,
                    term.polynomial,
                    float(theta[k]),
                    ignore_identity=ignore_identity,
                    coefficient_tolerance=coefficient_tolerance,
                    sort_terms=sort_terms,
                )
                k += 1
        return psi


class HardcodedUCCSDAnsatz:
    """
    Hardcoded UCCSD-style ansatz built directly from fermionic ladder operators
    mapped through the local JW primitives.

    U(theta) = prod_k exp(-i theta_k G_k),
    where each G_k is Hermitian and corresponds to i(T_k - T_k^dagger)
    for single or double excitations relative to a Hartree-Fock reference sector.
    """

    def __init__(
        self,
        dims: Dims,
        num_particles: Tuple[int, int],
        *,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        include_singles: bool = True,
        include_doubles: bool = True,
    ):
        self.dims = dims
        self.n_sites = n_sites_from_dims(dims)
        self.nq = 2 * int(self.n_sites)

        self.num_particles = (int(num_particles[0]), int(num_particles[1]))
        self.reps = int(reps)
        if self.reps <= 0:
            log.error("reps must be positive")

        self.repr_mode = repr_mode
        self.indexing = indexing
        self.include_singles = bool(include_singles)
        self.include_doubles = bool(include_doubles)

        n_alpha, n_beta = self.num_particles
        if n_alpha < 0 or n_beta < 0:
            log.error("num_particles entries must be non-negative")
        if n_alpha > int(self.n_sites) or n_beta > int(self.n_sites):
            log.error("cannot occupy more than n_sites orbitals per spin")

        self.base_terms: List[AnsatzTerm] = []
        self._build_base_terms()

        self.num_parameters = self.reps * len(self.base_terms)

    def _single_generator(self, p_occ: int, q_virt: int) -> PauliPolynomial:
        cd_q = fermion_plus_operator(self.repr_mode, self.nq, int(q_virt))
        cm_p = fermion_minus_operator(self.repr_mode, self.nq, int(p_occ))
        cd_p = fermion_plus_operator(self.repr_mode, self.nq, int(p_occ))
        cm_q = fermion_minus_operator(self.repr_mode, self.nq, int(q_virt))

        excite = cd_q * cm_p
        deexcite = cd_p * cm_q
        return (1j) * (excite + ((-1.0) * deexcite))

    def _double_generator(self, i_occ: int, j_occ: int, a_virt: int, b_virt: int) -> PauliPolynomial:
        cd_a = fermion_plus_operator(self.repr_mode, self.nq, int(a_virt))
        cd_b = fermion_plus_operator(self.repr_mode, self.nq, int(b_virt))
        cm_j = fermion_minus_operator(self.repr_mode, self.nq, int(j_occ))
        cm_i = fermion_minus_operator(self.repr_mode, self.nq, int(i_occ))

        cd_i = fermion_plus_operator(self.repr_mode, self.nq, int(i_occ))
        cd_j = fermion_plus_operator(self.repr_mode, self.nq, int(j_occ))
        cm_b = fermion_minus_operator(self.repr_mode, self.nq, int(b_virt))
        cm_a = fermion_minus_operator(self.repr_mode, self.nq, int(a_virt))

        excite = (((cd_a * cd_b) * cm_j) * cm_i)
        deexcite = (((cd_i * cd_j) * cm_b) * cm_a)
        return (1j) * (excite + ((-1.0) * deexcite))

    def _build_base_terms(self) -> None:
        n_sites = int(self.n_sites)
        n_alpha, n_beta = self.num_particles

        alpha_all = [mode_index(i, SPIN_UP, indexing=self.indexing, n_sites=n_sites) for i in range(n_sites)]
        beta_all = [mode_index(i, SPIN_DN, indexing=self.indexing, n_sites=n_sites) for i in range(n_sites)]

        alpha_occ = alpha_all[:n_alpha]
        beta_occ = beta_all[:n_beta]

        alpha_virt = alpha_all[n_alpha:]
        beta_virt = beta_all[n_beta:]

        if self.include_singles:
            for i_occ in alpha_occ:
                for a_virt in alpha_virt:
                    gen = self._single_generator(i_occ, a_virt)
                    self.base_terms.append(
                        AnsatzTerm(label=f"uccsd_sing(alpha:{i_occ}->{a_virt})", polynomial=gen)
                    )

            for i_occ in beta_occ:
                for a_virt in beta_virt:
                    gen = self._single_generator(i_occ, a_virt)
                    self.base_terms.append(
                        AnsatzTerm(label=f"uccsd_sing(beta:{i_occ}->{a_virt})", polynomial=gen)
                    )

        if self.include_doubles:
            # alpha-alpha doubles
            for i_pos in range(len(alpha_occ)):
                for j_pos in range(i_pos + 1, len(alpha_occ)):
                    i_occ = alpha_occ[i_pos]
                    j_occ = alpha_occ[j_pos]
                    for a_pos in range(len(alpha_virt)):
                        for b_pos in range(a_pos + 1, len(alpha_virt)):
                            a_virt = alpha_virt[a_pos]
                            b_virt = alpha_virt[b_pos]
                            gen = self._double_generator(i_occ, j_occ, a_virt, b_virt)
                            self.base_terms.append(
                                AnsatzTerm(
                                    label=(
                                        f"uccsd_dbl(aa:{i_occ},{j_occ}->{a_virt},{b_virt})"
                                    ),
                                    polynomial=gen,
                                )
                            )

            # beta-beta doubles
            for i_pos in range(len(beta_occ)):
                for j_pos in range(i_pos + 1, len(beta_occ)):
                    i_occ = beta_occ[i_pos]
                    j_occ = beta_occ[j_pos]
                    for a_pos in range(len(beta_virt)):
                        for b_pos in range(a_pos + 1, len(beta_virt)):
                            a_virt = beta_virt[a_pos]
                            b_virt = beta_virt[b_pos]
                            gen = self._double_generator(i_occ, j_occ, a_virt, b_virt)
                            self.base_terms.append(
                                AnsatzTerm(
                                    label=(
                                        f"uccsd_dbl(bb:{i_occ},{j_occ}->{a_virt},{b_virt})"
                                    ),
                                    polynomial=gen,
                                )
                            )

            # alpha-beta doubles
            for i_occ in alpha_occ:
                for j_occ in beta_occ:
                    for a_virt in alpha_virt:
                        for b_virt in beta_virt:
                            gen = self._double_generator(i_occ, j_occ, a_virt, b_virt)
                            self.base_terms.append(
                                AnsatzTerm(
                                    label=(
                                        f"uccsd_dbl(ab:{i_occ},{j_occ}->{a_virt},{b_virt})"
                                    ),
                                    polynomial=gen,
                                )
                            )

    def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ) -> np.ndarray:
        if int(theta.size) != int(self.num_parameters):
            log.error("theta has wrong length for this ansatz")

        psi = np.array(psi_ref, copy=True)
        k = 0
        for _ in range(self.reps):
            for term in self.base_terms:
                psi = apply_exp_pauli_polynomial(
                    psi,
                    term.polynomial,
                    float(theta[k]),
                    ignore_identity=ignore_identity,
                    coefficient_tolerance=coefficient_tolerance,
                    sort_terms=sort_terms,
                )
                k += 1
        return psi

@dataclass
class VQEResult:
    energy: float
    theta: np.ndarray
    success: bool
    message: str
    nfev: int
    nit: int
    best_restart: int


def _try_import_scipy_minimize():
    try:
        from scipy.optimize import minimize  # type: ignore
    except Exception:
        minimize = None
    return minimize


def vqe_minimize(
    H: PauliPolynomial,
    ansatz: Any,
    psi_ref: np.ndarray,
    *,
    restarts: int = 3,
    seed: int = 7,
    initial_point_stddev: float = 0.3,
    method: str = "SLSQP",
    maxiter: int = 1800,
    bounds: Optional[Tuple[float, float]] = (-math.pi, math.pi),
) -> VQEResult:
    """
    Hardcoded VQE: minimize <psi(theta)|H|psi(theta)> with a statevector backend.
    Uses SciPy if available; otherwise falls back to a tiny coordinate search.
    """
    minimize = _try_import_scipy_minimize()
    rng = np.random.default_rng(int(seed))
    npar = int(ansatz.num_parameters)
    if npar <= 0:
        log.error("ansatz has no parameters")

    def energy_fn(x: np.ndarray) -> float:
        theta = np.asarray(x, dtype=float)
        psi = ansatz.prepare_state(theta, psi_ref)
        return expval_pauli_polynomial(psi, H)

    best_energy = float("inf")
    best_theta = None
    best_restart = -1
    best_nfev = 0
    best_nit = 0
    best_success = False
    best_message = "no run"

    for r in range(int(restarts)):
        x0 = initial_point_stddev * rng.normal(size=npar)

        if minimize is not None:
            bnds = None
            if bounds is not None:
                lo, hi = float(bounds[0]), float(bounds[1])
                bnds = [(lo, hi)] * npar

            res = minimize(
                energy_fn,
                x0,
                method=str(method),
                bounds=bnds,
                options={"maxiter": int(maxiter)},
            )

            energy = float(res.fun)
            theta_opt = np.asarray(res.x, dtype=float)
            nfev = int(getattr(res, "nfev", 0))
            nit = int(getattr(res, "nit", 0))
            success = bool(getattr(res, "success", False))
            message = str(getattr(res, "message", ""))

        else:
            theta_opt = np.array(x0, dtype=float)
            step = 0.2
            nfev = 0
            nit = 0
            energy = energy_fn(theta_opt)
            nfev += 1

            for it in range(int(maxiter)):
                improved = False
                for k in range(npar):
                    for sgn in (+1.0, -1.0):
                        trial = theta_opt.copy()
                        trial[k] += sgn * step
                        e_trial = energy_fn(trial)
                        nfev += 1
                        if e_trial < energy:
                            energy = e_trial
                            theta_opt = trial
                            improved = True
                nit = it + 1
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
            success = True
            message = "fallback coordinate search"

        if energy < best_energy:
            best_energy = energy
            best_theta = theta_opt
            best_restart = r
            best_nfev = nfev
            best_nit = nit
            best_success = success
            best_message = message

    assert best_theta is not None
    return VQEResult(
        energy=float(best_energy),
        theta=np.asarray(best_theta, dtype=float),
        success=bool(best_success),
        message=str(best_message),
        nfev=int(best_nfev),
        nit=int(best_nit),
        best_restart=int(best_restart),
    )

_PAULI_MATS: Dict[str, np.ndarray] = {
    "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def pauli_matrix(pauli: str) -> np.ndarray:
    ps = _normalize_pauli_string(pauli)
    M = np.array([[1.0 + 0.0j]], dtype=complex)
    for ch in ps:  # q_(n-1) ... q_0
        M = np.kron(M, _PAULI_MATS[ch])
    return M


def hamiltonian_matrix(H: PauliPolynomial, tol: float = 1e-12) -> np.ndarray:
    terms = H.return_polynomial()
    if not terms:
        return np.zeros((1, 1), dtype=complex)
    nq = int(terms[0].nqubit())
    dim = 1 << nq
    M = np.zeros((dim, dim), dtype=complex)
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) < tol:
            continue
        M += coeff * pauli_matrix(term.pw2strng())
    return M


def _spin_orbital_index_sets(num_sites: int, ordering: str) -> Tuple[List[int], List[int]]:
    L = int(num_sites)
    if ordering == "blocked":
        return list(range(L)), list(range(L, 2 * L))
    if ordering == "interleaved":
        return list(range(0, 2 * L, 2)), list(range(1, 2 * L, 2))
    log.error("ordering must be 'blocked' or 'interleaved'")
    return [], []


def _sector_basis_indices(
    nq: int,
    alpha_indices: Sequence[int],
    beta_indices: Sequence[int],
    n_alpha: int,
    n_beta: int,
) -> List[int]:
    idxs: List[int] = []
    for idx in range(1 << int(nq)):
        na = sum((idx >> int(q)) & 1 for q in alpha_indices)
        nb = sum((idx >> int(q)) & 1 for q in beta_indices)
        if na == int(n_alpha) and nb == int(n_beta):
            idxs.append(int(idx))
    return idxs


def exact_ground_energy_sector(
    H: PauliPolynomial,
    *,
    num_sites: int,
    num_particles: Tuple[int, int],
    indexing: str = "blocked",
    tol: float = 1e-12,
) -> float:
    M = hamiltonian_matrix(H, tol=tol)
    nq = int(round(math.log2(M.shape[0])))

    alpha_idx, beta_idx = _spin_orbital_index_sets(int(num_sites), ordering=indexing)
    n_alpha, n_beta = int(num_particles[0]), int(num_particles[1])
    basis = _sector_basis_indices(nq, alpha_idx, beta_idx, n_alpha, n_beta)

    sub = M[np.ix_(basis, basis)]
    evals = np.linalg.eigvalsh(sub)
    return float(np.min(np.real(evals)))

def show_latex_and_code(title: str, latex_expr: str, fn) -> None:
    if display is not None and IPyMath is not None:
        if title:
            display(Markdown(f"### {title}"))
        display(IPyMath(latex_expr))
    else:
        if title:
            print(f"### {title}")
        print(latex_expr)
    print(inspect.getsource(fn))


def show_vqe_latex_python_pairs() -> None:
    show_latex_and_code(
        LATEX_TERMS["hamiltonian_sum"]["title"],
        LATEX_TERMS["hamiltonian_sum"]["latex"],
        expval_pauli_polynomial,
    )
    show_latex_and_code(
        LATEX_TERMS["pauli_rotation"]["title"],
        LATEX_TERMS["pauli_rotation"]["latex"],
        apply_pauli_rotation,
    )
    show_latex_and_code(
        LATEX_TERMS["trotter_step"]["title"],
        LATEX_TERMS["trotter_step"]["latex"],
        apply_exp_pauli_polynomial,
    )
    show_latex_and_code(
        LATEX_TERMS["ansatz_state"]["title"],
        LATEX_TERMS["ansatz_state"]["latex"],
        HardcodedUCCSDAnsatz,
    )
    show_latex_and_code(
        LATEX_TERMS["energy_expectation"]["title"],
        LATEX_TERMS["energy_expectation"]["latex"],
        vqe_minimize,
    )


if __name__ == "__main__":
    print(
        "Use this in Jupyter for rendered LaTeX:\n"
        "from pydephasing.quantum.vqe_latex_python_pairs import show_vqe_latex_python_pairs\n"
        "show_vqe_latex_python_pairs()"
    )


# Auto-generated from vqe_latex_python_pairs_test.ipynb (benchmark cell removed)
# Do not edit by hand unless you also update the notebook conversion logic.

# src/quantum/vqe_latex_python_pairs.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable when notebook is run from nested CWDs.
_cwd = Path.cwd().resolve()
for _candidate in (_cwd, *_cwd.parents):
    if (_candidate / 'src' / 'quantum' / 'pauli_polynomial_class.py').exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import inspect
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from src.quantum.spsa_optimizer import SPSAResult, spsa_minimize

try:
    from IPython.display import Markdown, Math as IPyMath, display
except Exception:  # pragma: no cover
    Markdown = None
    IPyMath = None
    display = None

class _FallbackLog:
    @staticmethod
    def error(msg: str):
        raise RuntimeError(msg)

    @staticmethod
    def info(msg: str):
        print(msg)


log = _FallbackLog()

try:
    from src.quantum.pauli_polynomial_class import (
        PauliPolynomial,
        fermion_minus_operator,
        fermion_plus_operator,
    )
    from src.quantum.pauli_words import PauliTerm
except Exception as _dep_exc:  # pragma: no cover
    PauliPolynomial = Any  # type: ignore[assignment]

    def _missing_dep(*_args, **_kwargs):
        raise ImportError(
            "src.quantum dependencies are unavailable in this environment"
        ) from _dep_exc

    fermion_minus_operator = _missing_dep  # type: ignore[assignment]
    fermion_plus_operator = _missing_dep  # type: ignore[assignment]
    PauliTerm = _missing_dep  # type: ignore[assignment]

try:
    # Reuse your canonical Hubbard lattice helpers for consistent indexing/edges.
    from src.quantum.hubbard_latex_python_pairs import (
        Dims,
        SPIN_DN,
        SPIN_UP,
        Spin,
        boson_displacement_operator,
        boson_number_operator,
        boson_qubits_per_site,
        build_holstein_coupling,
        build_holstein_phonon_energy,
        build_hubbard_holstein_drive,
        build_hubbard_kinetic,
        build_hubbard_onsite,
        build_hubbard_potential,
        bravais_nearest_neighbor_edges,
        mode_index,
        n_sites_from_dims,
        phonon_qubit_indices_for_site,
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

    def _missing_hh(*_args, **_kwargs):
        raise ImportError("HH helper unavailable (import Hubbard-Holstein helpers)")

    boson_displacement_operator = _missing_hh
    boson_number_operator = _missing_hh
    boson_qubits_per_site = _missing_hh
    build_holstein_coupling = _missing_hh
    build_holstein_phonon_energy = _missing_hh
    build_hubbard_holstein_drive = _missing_hh
    build_hubbard_kinetic = _missing_hh
    build_hubbard_onsite = _missing_hh
    build_hubbard_potential = _missing_hh
    phonon_qubit_indices_for_site = _missing_hh


__all__ = [
    # Statevector primitives
    "basis_state",
    "apply_pauli_string",
    "expval_pauli_string",
    "expval_pauli_polynomial",
    "expval_pauli_polynomial_one_apply",
    "apply_pauli_rotation",
    "apply_exp_pauli_polynomial",
    # Hamiltonian term builders
    "half_filled_num_particles",
    "jw_number_operator",
    "hubbard_hop_term",
    "hubbard_onsite_term",
    "hubbard_potential_term",
    # Ansatz classes (Hubbard)
    "AnsatzTerm",
    "HubbardTermwiseAnsatz",
    "HubbardLayerwiseAnsatz",
    "HardcodedUCCSDAnsatz",
    "HardcodedUCCSDLayerwiseAnsatz",
    # Ansatz classes (Hubbard-Holstein)
    "HubbardHolsteinTermwiseAnsatz",
    "HubbardHolsteinPhysicalTermwiseAnsatz",
    "HubbardHolsteinLayerwiseAnsatz",
    # VQE driver
    "VQEResult",
    "vqe_minimize",
    # Dense Hamiltonian / exact diag helpers
    "pauli_matrix",
    "hamiltonian_matrix",
    "exact_ground_energy_sector",
    "exact_ground_energy_sector_hh",
    # Display
    "show_latex_and_code",
    "show_vqe_latex_python_pairs",
]

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


def expval_pauli_polynomial_one_apply(
    state: np.ndarray,
    H: PauliPolynomial,
    *,
    tol: float = 1e-12,
    cache: Optional[Dict[str, Any]] = None,
) -> float:
    r"""<psi|H|psi> via one H|psi> apply and one inner product."""
    from src.quantum.compiled_polynomial import compile_polynomial_action, energy_via_one_apply

    cache_dict = cache if cache is not None else {}
    compiled_key = "__compiled_h__"
    h_id_key = "__compiled_h_id__"
    tol_key = "__compiled_h_tol__"
    compiled_h = cache_dict.get(compiled_key)
    cached_h_id = cache_dict.get(h_id_key)
    cached_tol = cache_dict.get(tol_key)
    needs_compile = (
        compiled_h is None
        or cached_h_id != int(id(H))
        or cached_tol != float(tol)
    )

    if needs_compile:
        compiled_h = compile_polynomial_action(
            H,
            tol=float(tol),
            pauli_action_cache=cache_dict,
        )
        cache_dict[compiled_key] = compiled_h
        cache_dict[h_id_key] = int(id(H))
        cache_dict[tol_key] = float(tol)

    energy, _hpsi = energy_via_one_apply(np.asarray(state, dtype=complex), compiled_h)
    return float(energy)

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
    from src.quantum.hartree_fock_reference_state import hartree_fock_bitstring

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


SitePotential = Optional[Union[float, Sequence[float], Dict[int, float]]]
TimePotential = Optional[
    Union[
        float,
        Sequence[float],
        Dict[int, float],
        Callable[[Optional[float]], Union[float, Sequence[float], Dict[int, float]]],
    ]
]


def _single_term_polynomials_sorted(
    poly: PauliPolynomial,
    *,
    repr_mode: str,
    coefficient_tolerance: float = 1e-12,
) -> List[PauliPolynomial]:
    """Split a PauliPolynomial into individual single-term PauliPolynomials, sorted."""
    terms = list(poly.return_polynomial())
    terms.sort(key=lambda t: (t.pw2strng(), float(np.real(t.p_coeff)), float(np.imag(t.p_coeff))))
    out: List[PauliPolynomial] = []
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(coefficient_tolerance):
            continue
        nq = int(term.nqubit())
        out.append(PauliPolynomial(repr_mode, [PauliTerm(nq, ps=term.pw2strng(), pc=coeff)]))
    return out


# ---------------------------------------------------------------------------
# Hubbard-Holstein reference state (thin wrapper around hartree_fock module)
# ---------------------------------------------------------------------------

try:
    from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
except Exception:  # pragma: no cover
    def hubbard_holstein_reference_state(**kwargs) -> np.ndarray:
        raise ImportError("hubbard_holstein_reference_state unavailable")


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


class HubbardLayerwiseAnsatz(HubbardTermwiseAnsatz):
    """
    Layer-wise Hubbard ansatz with shared parameters per physical term group.

    Per layer:
      1) one shared theta over all hopping primitives
      2) one shared theta over all onsite-U primitives
      3) one shared theta over all potential primitives (if present)
    """

    def _build_base_terms(self) -> None:
        nq = int(self.nq)
        n_sites = int(self.n_sites)

        hop_terms: List[AnsatzTerm] = []
        onsite_terms: List[AnsatzTerm] = []
        potential_terms: List[AnsatzTerm] = []

        for (i, j) in self.edges:
            for spin in (SPIN_UP, SPIN_DN):
                p_i = mode_index(int(i), int(spin), indexing=self.indexing, n_sites=n_sites)
                p_j = mode_index(int(j), int(spin), indexing=self.indexing, n_sites=n_sites)
                poly = hubbard_hop_term(nq, p_i, p_j, t=self.t, repr_mode=self.repr_mode)
                hop_terms.append(AnsatzTerm(label=f"hop(i={i},j={j},spin={spin})", polynomial=poly))

        for i in range(n_sites):
            p_up = mode_index(i, SPIN_UP, indexing=self.indexing, n_sites=n_sites)
            p_dn = mode_index(i, SPIN_DN, indexing=self.indexing, n_sites=n_sites)
            poly = hubbard_onsite_term(nq, p_up, p_dn, U=self.U, repr_mode=self.repr_mode)
            onsite_terms.append(AnsatzTerm(label=f"onsite(i={i})", polynomial=poly))

        if self.include_potential_terms:
            for i in range(n_sites):
                vi = float(self.v_list[i])
                if abs(vi) < 1e-15:
                    continue
                for spin in (SPIN_UP, SPIN_DN):
                    p_mode = mode_index(i, spin, indexing=self.indexing, n_sites=n_sites)
                    poly = hubbard_potential_term(nq, p_mode, v_i=vi, repr_mode=self.repr_mode)
                    potential_terms.append(AnsatzTerm(label=f"pot(i={i},spin={spin})", polynomial=poly))

        self.layer_term_groups: List[Tuple[str, List[AnsatzTerm]]] = []
        if hop_terms:
            self.layer_term_groups.append(("hop_layer", hop_terms))
        if onsite_terms:
            self.layer_term_groups.append(("onsite_layer", onsite_terms))
        if potential_terms:
            self.layer_term_groups.append(("potential_layer", potential_terms))

        # Aggregated representative list retained for parameter counting.
        self.base_terms = [
            AnsatzTerm(label=group_name, polynomial=group_terms[0].polynomial)
            for group_name, group_terms in self.layer_term_groups
        ]

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
            for _group_name, group_terms in self.layer_term_groups:
                theta_shared = float(theta[k])
                for term in group_terms:
                    psi = apply_exp_pauli_polynomial(
                        psi,
                        term.polynomial,
                        theta_shared,
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


class HardcodedUCCSDLayerwiseAnsatz(HardcodedUCCSDAnsatz):
    """
    Layer-wise UCCSD ansatz with one shared parameter per excitation group.

    Per layer:
      1) one shared theta across all singles generators (if enabled/present)
      2) one shared theta across all doubles generators (if enabled/present)
    """

    def _build_base_terms(self) -> None:
        n_sites = int(self.n_sites)
        n_alpha, n_beta = self.num_particles

        alpha_all = [mode_index(i, SPIN_UP, indexing=self.indexing, n_sites=n_sites) for i in range(n_sites)]
        beta_all = [mode_index(i, SPIN_DN, indexing=self.indexing, n_sites=n_sites) for i in range(n_sites)]

        alpha_occ = alpha_all[:n_alpha]
        beta_occ = beta_all[:n_beta]

        alpha_virt = alpha_all[n_alpha:]
        beta_virt = beta_all[n_beta:]

        singles_terms: List[AnsatzTerm] = []
        doubles_terms: List[AnsatzTerm] = []

        if self.include_singles:
            for i_occ in alpha_occ:
                for a_virt in alpha_virt:
                    gen = self._single_generator(i_occ, a_virt)
                    singles_terms.append(
                        AnsatzTerm(label=f"uccsd_sing(alpha:{i_occ}->{a_virt})", polynomial=gen)
                    )

            for i_occ in beta_occ:
                for a_virt in beta_virt:
                    gen = self._single_generator(i_occ, a_virt)
                    singles_terms.append(
                        AnsatzTerm(label=f"uccsd_sing(beta:{i_occ}->{a_virt})", polynomial=gen)
                    )

        if self.include_doubles:
            for i_pos in range(len(alpha_occ)):
                for j_pos in range(i_pos + 1, len(alpha_occ)):
                    i_occ = alpha_occ[i_pos]
                    j_occ = alpha_occ[j_pos]
                    for a_pos in range(len(alpha_virt)):
                        for b_pos in range(a_pos + 1, len(alpha_virt)):
                            a_virt = alpha_virt[a_pos]
                            b_virt = alpha_virt[b_pos]
                            gen = self._double_generator(i_occ, j_occ, a_virt, b_virt)
                            doubles_terms.append(
                                AnsatzTerm(
                                    label=(
                                        f"uccsd_dbl(aa:{i_occ},{j_occ}->{a_virt},{b_virt})"
                                    ),
                                    polynomial=gen,
                                )
                            )

            for i_pos in range(len(beta_occ)):
                for j_pos in range(i_pos + 1, len(beta_occ)):
                    i_occ = beta_occ[i_pos]
                    j_occ = beta_occ[j_pos]
                    for a_pos in range(len(beta_virt)):
                        for b_pos in range(a_pos + 1, len(beta_virt)):
                            a_virt = beta_virt[a_pos]
                            b_virt = beta_virt[b_pos]
                            gen = self._double_generator(i_occ, j_occ, a_virt, b_virt)
                            doubles_terms.append(
                                AnsatzTerm(
                                    label=(
                                        f"uccsd_dbl(bb:{i_occ},{j_occ}->{a_virt},{b_virt})"
                                    ),
                                    polynomial=gen,
                                )
                            )

            for i_occ in alpha_occ:
                for j_occ in beta_occ:
                    for a_virt in alpha_virt:
                        for b_virt in beta_virt:
                            gen = self._double_generator(i_occ, j_occ, a_virt, b_virt)
                            doubles_terms.append(
                                AnsatzTerm(
                                    label=(
                                        f"uccsd_dbl(ab:{i_occ},{j_occ}->{a_virt},{b_virt})"
                                    ),
                                    polynomial=gen,
                                )
                            )

        self.layer_term_groups: List[Tuple[str, List[AnsatzTerm]]] = []
        if singles_terms:
            self.layer_term_groups.append(("uccsd_singles_layer", singles_terms))
        if doubles_terms:
            self.layer_term_groups.append(("uccsd_doubles_layer", doubles_terms))

        self.base_terms = [
            AnsatzTerm(label=group_name, polynomial=group_terms[0].polynomial)
            for group_name, group_terms in self.layer_term_groups
        ]

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
            for _group_name, group_terms in self.layer_term_groups:
                theta_shared = float(theta[k])
                for term in group_terms:
                    psi = apply_exp_pauli_polynomial(
                        psi,
                        term.polynomial,
                        theta_shared,
                        ignore_identity=ignore_identity,
                        coefficient_tolerance=coefficient_tolerance,
                        sort_terms=sort_terms,
                    )
                k += 1
        return psi


# ---------------------------------------------------------------------------
# Hubbard-Holstein termwise ansatz  (one θ per Pauli-term exponential)
# ---------------------------------------------------------------------------

class HubbardHolsteinTermwiseAnsatz:
    r"""
    Term-wise Hubbard-Holstein ansatz — one independent θ per Pauli exponential.

    Per layer, apply all individual Pauli-term unitaries in deterministic order:
      1) hopping terms     — each XX/YY pair from H_t gets its own θ
      2) onsite-U terms    — each ZZ/Z/I term from H_U gets its own θ
      3) potential terms    — each Z term from H_v gets its own θ (if present)
      4) phonon terms       — each Z term from H_ph gets its own θ
      5) e-ph coupling      — each term from H_g gets its own θ
      6) drive terms        — each term from H_drive gets its own θ (if present)

    This gives ~16 independent parameters per layer for L=2 n_ph_max=1 (vs ~4
    in the layerwise ansatz), providing the expressivity needed to converge to
    the ground state.

    Compatible with time evolution: each unitary is exp(-i θ_k P_k).
    """

    def __init__(
        self,
        dims: Dims,
        J: float,
        U: float,
        omega0: float,
        g: float,
        n_ph_max: int,
        *,
        boson_encoding: str = "binary",
        v: SitePotential = None,
        v_t: TimePotential = None,
        v0: SitePotential = None,
        t_eval: Optional[float] = None,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        edges: Optional[Sequence[Tuple[int, int]]] = None,
        pbc: Union[bool, Sequence[bool]] = True,
        include_zero_point: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ):
        self.dims = dims
        self.n_sites = int(n_sites_from_dims(dims))
        self.n_ferm = 2 * self.n_sites
        self.n_ph_max = int(n_ph_max)
        self.boson_encoding = str(boson_encoding)
        self.qpb = int(boson_qubits_per_site(self.n_ph_max, self.boson_encoding))
        self.n_total = self.n_ferm + self.n_sites * self.qpb
        self.nq = int(self.n_total)

        self.J = float(J)
        self.U = float(U)
        self.omega0 = float(omega0)
        self.g = float(g)
        self.v_list = _parse_site_potential(v, n_sites=self.n_sites)
        self.v_t = v_t
        self.v0 = v0
        self.t_eval = t_eval
        self.include_zero_point = bool(include_zero_point)

        self.repr_mode = str(repr_mode)
        self.indexing = str(indexing)
        self.edges = list(edges) if edges is not None else bravais_nearest_neighbor_edges(dims, pbc=pbc)
        self.reps = int(reps)
        if self.reps <= 0:
            log.error("reps must be positive")

        self.coefficient_tolerance = float(coefficient_tolerance)
        self.sort_terms = bool(sort_terms)

        self.base_terms: List[AnsatzTerm] = []
        self._build_base_terms()
        self.num_parameters = self.reps * len(self.base_terms)

    def _build_base_terms(self) -> None:
        """Build one AnsatzTerm per individual Pauli exponential."""
        tol = self.coefficient_tolerance
        rm = self.repr_mode

        def _add_terms_from_poly(label_prefix: str, poly: PauliPolynomial) -> None:
            term_polys = _single_term_polynomials_sorted(
                poly, repr_mode=rm, coefficient_tolerance=tol,
            )
            for i, tp in enumerate(term_polys):
                self.base_terms.append(AnsatzTerm(
                    label=f"{label_prefix}_{i}", polynomial=tp,
                ))

        # (1) hopping
        hop_poly = build_hubbard_kinetic(
            dims=self.dims, t=self.J,
            repr_mode=rm, indexing=self.indexing,
            edges=self.edges, pbc=True,
            nq_override=self.n_total,
        )
        _add_terms_from_poly("hop", hop_poly)

        # (2) onsite U
        onsite_poly = build_hubbard_onsite(
            dims=self.dims, U=self.U,
            repr_mode=rm, indexing=self.indexing,
            nq_override=self.n_total,
        )
        _add_terms_from_poly("onsite", onsite_poly)

        # (3) fermion potential (optional)
        potential_poly = build_hubbard_potential(
            dims=self.dims, v=self.v_list,
            repr_mode=rm, indexing=self.indexing,
            nq_override=self.n_total,
        )
        _add_terms_from_poly("pot", potential_poly)

        # (4) phonon energy
        phonon_poly = build_holstein_phonon_energy(
            dims=self.dims, omega0=self.omega0,
            n_ph_max=self.n_ph_max, boson_encoding=self.boson_encoding,
            repr_mode=rm, tol=tol,
            zero_point=self.include_zero_point,
        )
        _add_terms_from_poly("phonon", phonon_poly)

        # (5) electron-phonon coupling
        eph_poly = build_holstein_coupling(
            dims=self.dims, g=self.g,
            n_ph_max=self.n_ph_max, boson_encoding=self.boson_encoding,
            repr_mode=rm, indexing=self.indexing, tol=tol,
        )
        _add_terms_from_poly("eph", eph_poly)

        # (6) drive (optional)
        if self.v_t is not None or self.v0 is not None:
            drive_poly = build_hubbard_holstein_drive(
                dims=self.dims, v_t=self.v_t, v0=self.v0,
                t=self.t_eval, repr_mode=rm,
                indexing=self.indexing, nq_override=self.n_total,
            )
            _add_terms_from_poly("drive", drive_poly)

        if not self.base_terms:
            log.error("HubbardHolsteinTermwiseAnsatz produced no terms")

    def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: Optional[float] = None,
        sort_terms: Optional[bool] = None,
    ) -> np.ndarray:
        if int(theta.size) != int(self.num_parameters):
            log.error("theta has wrong length for this ansatz")
        if int(psi_ref.size) != (1 << int(self.nq)):
            log.error("psi_ref length must be 2^nq for HubbardHolsteinTermwiseAnsatz")

        coeff_tol = self.coefficient_tolerance if coefficient_tolerance is None else float(coefficient_tolerance)
        sort_flag = self.sort_terms if sort_terms is None else bool(sort_terms)

        psi = np.array(psi_ref, copy=True)
        k = 0
        for _ in range(self.reps):
            for term in self.base_terms:
                psi = apply_exp_pauli_polynomial(
                    psi,
                    term.polynomial,
                    float(theta[k]),
                    ignore_identity=ignore_identity,
                    coefficient_tolerance=coeff_tol,
                    sort_terms=sort_flag,
                )
                k += 1
        return psi


# ---------------------------------------------------------------------------
# Hubbard-Holstein physical-termwise ansatz
# ---------------------------------------------------------------------------

class HubbardHolsteinPhysicalTermwiseAnsatz:
    r"""
    Physical-termwise Hubbard-Holstein ansatz (sector-preserving in fermion space).

    Per layer, apply one independent parameter per *physical* HH generator:
      1) hopping terms      H^{(t)}_{ij,\sigma}
      2) onsite-U terms     H^{(U)}_i
      3) static potential   H^{(v)}_{i,\sigma}         (if non-zero)
      4) phonon energy      \omega_0 n_{b,i}
      5) e-ph coupling      g x_i (n_i - I)
      6) drive terms        (v_i(t)-v_{0,i}) n_{i,\sigma}  (if provided)

    Unlike the Pauli-termwise variant, this class does not split each physical
    generator into individual Pauli monomials before exponentiation.
    """

    def __init__(
        self,
        dims: Dims,
        J: float,
        U: float,
        omega0: float,
        g: float,
        n_ph_max: int,
        *,
        boson_encoding: str = "binary",
        v: SitePotential = None,
        v_t: TimePotential = None,
        v0: SitePotential = None,
        t_eval: Optional[float] = None,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        edges: Optional[Sequence[Tuple[int, int]]] = None,
        pbc: Union[bool, Sequence[bool]] = True,
        include_zero_point: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ):
        self.dims = dims
        self.n_sites = int(n_sites_from_dims(dims))
        self.n_ferm = 2 * self.n_sites
        self.n_ph_max = int(n_ph_max)
        self.boson_encoding = str(boson_encoding)
        self.qpb = int(boson_qubits_per_site(self.n_ph_max, self.boson_encoding))
        self.n_total = self.n_ferm + self.n_sites * self.qpb
        self.nq = int(self.n_total)

        self.J = float(J)
        self.U = float(U)
        self.omega0 = float(omega0)
        self.g = float(g)
        self.v_list = _parse_site_potential(v, n_sites=self.n_sites)
        self.v_t = v_t
        self.v0 = v0
        self.t_eval = t_eval
        self.include_zero_point = bool(include_zero_point)

        self.repr_mode = str(repr_mode)
        self.indexing = str(indexing)
        self.edges = list(edges) if edges is not None else bravais_nearest_neighbor_edges(dims, pbc=pbc)
        self.reps = int(reps)
        if self.reps <= 0:
            log.error("reps must be positive")

        self.coefficient_tolerance = float(coefficient_tolerance)
        self.sort_terms = bool(sort_terms)

        self.base_terms: List[AnsatzTerm] = []
        self._build_base_terms()
        self.num_parameters = self.reps * len(self.base_terms)

    def _build_base_terms(self) -> None:
        tol = self.coefficient_tolerance
        nq = int(self.n_total)
        n_sites = int(self.n_sites)
        rm = self.repr_mode
        fermion_qubits = int(self.n_ferm)
        qpb = int(self.qpb)
        identity = PauliPolynomial(rm, [PauliTerm(nq, ps="e" * nq, pc=1.0)])

        # (1) Hopping: one parameter per bond/spin physical hopping generator.
        for (i, j) in self.edges:
            for spin in (SPIN_UP, SPIN_DN):
                p_i = mode_index(int(i), int(spin), indexing=self.indexing, n_sites=n_sites)
                p_j = mode_index(int(j), int(spin), indexing=self.indexing, n_sites=n_sites)
                poly = hubbard_hop_term(nq, p_i, p_j, t=self.J, repr_mode=rm)
                self.base_terms.append(
                    AnsatzTerm(label=f"hh_hop(i={i},j={j},spin={spin})", polynomial=poly)
                )

        # (2) Onsite-U: one parameter per site.
        for i in range(n_sites):
            p_up = mode_index(i, SPIN_UP, indexing=self.indexing, n_sites=n_sites)
            p_dn = mode_index(i, SPIN_DN, indexing=self.indexing, n_sites=n_sites)
            poly = hubbard_onsite_term(nq, p_up, p_dn, U=self.U, repr_mode=rm)
            self.base_terms.append(AnsatzTerm(label=f"hh_onsite(i={i})", polynomial=poly))

        # (3) Static potential (optional): one parameter per site/spin term.
        for i in range(n_sites):
            vi = float(self.v_list[i])
            if abs(vi) <= tol:
                continue
            for spin in (SPIN_UP, SPIN_DN):
                p_mode = mode_index(i, spin, indexing=self.indexing, n_sites=n_sites)
                poly = hubbard_potential_term(nq, p_mode, v_i=vi, repr_mode=rm)
                self.base_terms.append(AnsatzTerm(label=f"hh_pot(i={i},spin={spin})", polynomial=poly))

        # (4) Phonon energy without zero-point identity contribution (global phase).
        if abs(float(self.omega0)) > tol:
            for i in range(n_sites):
                q_i = phonon_qubit_indices_for_site(
                    i,
                    n_sites=n_sites,
                    qpb=qpb,
                    fermion_qubits=fermion_qubits,
                )
                n_b = boson_number_operator(
                    rm,
                    nq,
                    q_i,
                    n_ph_max=self.n_ph_max,
                    encoding=self.boson_encoding,
                    tol=tol,
                )
                self.base_terms.append(
                    AnsatzTerm(label=f"hh_phonon(i={i})", polynomial=float(self.omega0) * n_b)
                )

        # (5) Electron-phonon coupling: one parameter per site.
        if abs(float(self.g)) > tol:
            n_cache: Dict[int, PauliPolynomial] = {}

            def n_op(p_mode: int) -> PauliPolynomial:
                if p_mode not in n_cache:
                    n_cache[p_mode] = jw_number_operator(rm, nq, p_mode)
                return n_cache[p_mode]

            for i in range(n_sites):
                p_up = mode_index(i, SPIN_UP, indexing=self.indexing, n_sites=n_sites)
                p_dn = mode_index(i, SPIN_DN, indexing=self.indexing, n_sites=n_sites)
                n_i = n_op(p_up) + n_op(p_dn)
                q_i = phonon_qubit_indices_for_site(
                    i,
                    n_sites=n_sites,
                    qpb=qpb,
                    fermion_qubits=fermion_qubits,
                )
                x_i = boson_displacement_operator(
                    rm,
                    nq,
                    q_i,
                    n_ph_max=self.n_ph_max,
                    encoding=self.boson_encoding,
                    tol=tol,
                )
                poly = float(self.g) * (x_i * (n_i + ((-1.0) * identity)))
                self.base_terms.append(AnsatzTerm(label=f"hh_eph(i={i})", polynomial=poly))

        # (6) Drive terms (optional): one parameter per site/spin drive primitive.
        if self.v_t is not None or self.v0 is not None:
            if callable(self.v_t):
                if self.t_eval is None:
                    log.error("t_eval must be provided when v_t is callable")
                v_t_list = _parse_site_potential(self.v_t(self.t_eval), n_sites=n_sites)
            else:
                v_t_list = _parse_site_potential(self.v_t, n_sites=n_sites)
            v0_list = _parse_site_potential(self.v0, n_sites=n_sites)
            for i in range(n_sites):
                dv_i = float(v_t_list[i]) - float(v0_list[i])
                if abs(dv_i) <= tol:
                    continue
                for spin in (SPIN_UP, SPIN_DN):
                    p_mode = mode_index(i, spin, indexing=self.indexing, n_sites=n_sites)
                    poly = hubbard_potential_term(nq, p_mode, v_i=(-dv_i), repr_mode=rm)
                    self.base_terms.append(AnsatzTerm(label=f"hh_drive(i={i},spin={spin})", polynomial=poly))

        if not self.base_terms:
            log.error("HubbardHolsteinPhysicalTermwiseAnsatz produced no terms")

    def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: Optional[float] = None,
        sort_terms: Optional[bool] = None,
    ) -> np.ndarray:
        if int(theta.size) != int(self.num_parameters):
            log.error("theta has wrong length for this ansatz")
        if int(psi_ref.size) != (1 << int(self.nq)):
            log.error("psi_ref length must be 2^nq for HubbardHolsteinPhysicalTermwiseAnsatz")

        coeff_tol = self.coefficient_tolerance if coefficient_tolerance is None else float(coefficient_tolerance)
        sort_flag = self.sort_terms if sort_terms is None else bool(sort_terms)

        psi = np.array(psi_ref, copy=True)
        k = 0
        for _ in range(self.reps):
            for term in self.base_terms:
                psi = apply_exp_pauli_polynomial(
                    psi,
                    term.polynomial,
                    float(theta[k]),
                    ignore_identity=ignore_identity,
                    coefficient_tolerance=coeff_tol,
                    sort_terms=sort_flag,
                )
                k += 1
        return psi


# ---------------------------------------------------------------------------
# Hubbard-Holstein layerwise ansatz
# ---------------------------------------------------------------------------

class HubbardHolsteinLayerwiseAnsatz:
    r"""
    Layer-wise Hubbard-Holstein ansatz with shared parameters per physical group.

    Per layer, apply groups in deterministic order:
      1) hopping       — H_t from fermion sector, extended to nq_total qubits
      2) onsite-U      — H_U from fermion sector, extended to nq_total qubits
      3) potential      — H_v from fermion sector (optional, only if v_i ≠ 0)
      4) phonon energy  — H_{ph} = ω₀ Σ_i (n_{b,i} + ½)
      5) e-ph coupling  — H_g  = g  Σ_i x_i (n_i − 𝟙)
      6) drive          — H_{drive} (optional, only if v_t / v0 provided)

    Each group is decomposed into individual Pauli-term exponentials
    (via _single_term_polynomials_sorted), sharing one θ per group per layer.

    Total qubit register: [2L fermion qubits | L · qpb phonon qubits].
    """

    def __init__(
        self,
        dims: Dims,
        J: float,
        U: float,
        omega0: float,
        g: float,
        n_ph_max: int,
        *,
        boson_encoding: str = "binary",
        v: SitePotential = None,
        v_t: TimePotential = None,
        v0: SitePotential = None,
        t_eval: Optional[float] = None,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        edges: Optional[Sequence[Tuple[int, int]]] = None,
        pbc: Union[bool, Sequence[bool]] = True,
        include_zero_point: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ):
        self.dims = dims
        self.n_sites = int(n_sites_from_dims(dims))
        self.n_ferm = 2 * self.n_sites
        self.n_ph_max = int(n_ph_max)
        self.boson_encoding = str(boson_encoding)
        self.qpb = int(boson_qubits_per_site(self.n_ph_max, self.boson_encoding))
        self.n_total = self.n_ferm + self.n_sites * self.qpb
        self.nq = int(self.n_total)

        self.J = float(J)
        self.U = float(U)
        self.omega0 = float(omega0)
        self.g = float(g)
        self.v_list = _parse_site_potential(v, n_sites=self.n_sites)
        self.v_t = v_t
        self.v0 = v0
        self.t_eval = t_eval
        self.include_zero_point = bool(include_zero_point)

        self.repr_mode = str(repr_mode)
        self.indexing = str(indexing)
        self.edges = list(edges) if edges is not None else bravais_nearest_neighbor_edges(dims, pbc=pbc)
        self.reps = int(reps)
        if self.reps <= 0:
            log.error("reps must be positive")

        self.coefficient_tolerance = float(coefficient_tolerance)
        self.sort_terms = bool(sort_terms)

        self.base_terms: List[AnsatzTerm] = []
        self.layer_term_groups: List[Tuple[str, List[AnsatzTerm]]] = []
        self._build_base_terms()
        self.num_parameters = self.reps * len(self.base_terms)

    # -- internal helper: split a polynomial into single-term exponentials --

    def _poly_group(
        self,
        label: str,
        poly: PauliPolynomial,
    ) -> None:
        """Decompose poly into sorted single-term PauliPolynomials and register as a group."""
        term_polys = _single_term_polynomials_sorted(
            poly,
            repr_mode=self.repr_mode,
            coefficient_tolerance=self.coefficient_tolerance,
        )
        if not term_polys:
            return
        group_terms: List[AnsatzTerm] = []
        group_poly: Optional[PauliPolynomial] = None
        for i, term_poly in enumerate(term_polys):
            group_terms.append(AnsatzTerm(label=f"{label}_term_{i}", polynomial=term_poly))
            group_poly = term_poly if group_poly is None else (group_poly + term_poly)
        assert group_poly is not None
        self.layer_term_groups.append((label, group_terms))
        self.base_terms.append(AnsatzTerm(label=label, polynomial=group_poly))

    # -- term construction -------------------------------------------------

    def _build_base_terms(self) -> None:
        # (1) hopping on the full fermion+phonon register
        hop_poly = build_hubbard_kinetic(
            dims=self.dims,
            t=self.J,
            repr_mode=self.repr_mode,
            indexing=self.indexing,
            edges=self.edges,
            pbc=True,
            nq_override=self.n_total,
        )
        self._poly_group("hop_layer", hop_poly)

        # (2) onsite U
        onsite_poly = build_hubbard_onsite(
            dims=self.dims,
            U=self.U,
            repr_mode=self.repr_mode,
            indexing=self.indexing,
            nq_override=self.n_total,
        )
        self._poly_group("onsite_layer", onsite_poly)

        # (3) fermion potential (optional — skipped if all v_i are zero)
        potential_poly = build_hubbard_potential(
            dims=self.dims,
            v=self.v_list,
            repr_mode=self.repr_mode,
            indexing=self.indexing,
            nq_override=self.n_total,
        )
        self._poly_group("potential_layer", potential_poly)

        # (4) phonon energy: H_ph = ω₀ Σ_i (n_{b,i} + ½)
        phonon_poly = build_holstein_phonon_energy(
            dims=self.dims,
            omega0=self.omega0,
            n_ph_max=self.n_ph_max,
            boson_encoding=self.boson_encoding,
            repr_mode=self.repr_mode,
            tol=self.coefficient_tolerance,
            zero_point=self.include_zero_point,
        )
        self._poly_group("phonon_layer", phonon_poly)

        # (5) electron-phonon coupling: H_g = g Σ_i x_i (n_i − 𝟙)
        eph_poly = build_holstein_coupling(
            dims=self.dims,
            g=self.g,
            n_ph_max=self.n_ph_max,
            boson_encoding=self.boson_encoding,
            repr_mode=self.repr_mode,
            indexing=self.indexing,
            tol=self.coefficient_tolerance,
        )
        self._poly_group("eph_layer", eph_poly)

        # (6) drive (optional)
        if self.v_t is not None or self.v0 is not None:
            drive_poly = build_hubbard_holstein_drive(
                dims=self.dims,
                v_t=self.v_t,
                v0=self.v0,
                t=self.t_eval,
                repr_mode=self.repr_mode,
                indexing=self.indexing,
                nq_override=self.n_total,
            )
            self._poly_group("drive_layer", drive_poly)

        if not self.base_terms:
            log.error("HubbardHolsteinLayerwiseAnsatz produced no layer terms")

    # -- state preparation -------------------------------------------------

    def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: Optional[float] = None,
        sort_terms: Optional[bool] = None,
    ) -> np.ndarray:
        if int(theta.size) != int(self.num_parameters):
            log.error("theta has wrong length for this ansatz")
        if not hasattr(self, "layer_term_groups"):
            log.error("HubbardHolsteinLayerwiseAnsatz missing layer term groups")
        if int(psi_ref.size) != (1 << int(self.nq)):
            log.error("psi_ref length must be 2^nq for HubbardHolsteinLayerwiseAnsatz")

        coeff_tol = self.coefficient_tolerance if coefficient_tolerance is None else float(coefficient_tolerance)
        sort_flag = self.sort_terms if sort_terms is None else bool(sort_terms)

        psi = np.array(psi_ref, copy=True)
        k = 0
        for _ in range(self.reps):
            for _label, group_terms in self.layer_term_groups:
                shared_theta = float(theta[k])
                for term in group_terms:
                    psi = apply_exp_pauli_polynomial(
                        psi,
                        term.polynomial,
                        shared_theta,
                        ignore_identity=ignore_identity,
                        coefficient_tolerance=coeff_tol,
                        sort_terms=sort_flag,
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
    progress_history: list[dict[str, Any]] | None = None
    restart_summaries: list[dict[str, Any]] | None = None
    optimizer_memory: dict[str, Any] | None = None


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
    initial_point: Optional[np.ndarray] = None,
    use_initial_point_first_restart: bool = True,
    method: str = "SLSQP",
    maxiter: int = 1800,
    bounds: Optional[Tuple[float, float]] = (-math.pi, math.pi),
    progress_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_every_s: float = 60.0,
    progress_label: str = "vqe_minimize",
    track_history: bool = False,
    emit_theta_in_progress: bool = False,
    return_best_on_keyboard_interrupt: bool = False,
    early_stop_checker: Optional[Callable[[Dict[str, Any]], bool]] = None,
    spsa_a: float = 0.2,
    spsa_c: float = 0.1,
    spsa_alpha: float = 0.602,
    spsa_gamma: float = 0.101,
    spsa_A: float = 10.0,
    spsa_avg_last: int = 0,
    spsa_eval_repeats: int = 1,
    spsa_eval_agg: str = "mean",
    energy_backend: str = "legacy",
    optimizer_memory: Optional[Dict[str, Any]] = None,
    spsa_refresh_every: int = 0,
    spsa_precondition_mode: str = "none",
) -> VQEResult:
    """
    Hardcoded VQE: minimize <psi(theta)|H|psi(theta)> with a statevector backend.
    Uses SciPy if available; otherwise falls back to a tiny coordinate search.
    Optional progress callback can emit restart/heartbeat lifecycle events.
    """
    minimize = _try_import_scipy_minimize()
    rng = np.random.default_rng(int(seed))
    npar = int(ansatz.num_parameters)
    if npar <= 0:
        log.error("ansatz has no parameters")

    initial_point_arr: Optional[np.ndarray] = None
    if initial_point is not None:
        initial_point_arr = np.asarray(initial_point, dtype=float).reshape(-1)
        if int(initial_point_arr.size) != int(npar):
            log.error(
                "initial_point size mismatch: expected "
                f"{int(npar)}, got {int(initial_point_arr.size)}"
            )

    backend_key = str(energy_backend).strip().lower()
    if backend_key not in {"legacy", "one_apply_compiled"}:
        log.error("energy_backend must be 'legacy' or 'one_apply_compiled'")

    energy_cache: Optional[Dict[str, Any]] = None
    if backend_key == "one_apply_compiled":
        from src.quantum.compiled_polynomial import compile_polynomial_action

        energy_cache = {}
        energy_cache["__compiled_h__"] = compile_polynomial_action(
            H,
            tol=1e-12,
            pauli_action_cache=energy_cache,
        )

    def energy_fn(x: np.ndarray) -> float:
        theta = np.asarray(x, dtype=float)
        psi = ansatz.prepare_state(theta, psi_ref)
        if backend_key == "legacy":
            return expval_pauli_polynomial(psi, H)
        return expval_pauli_polynomial_one_apply(psi, H, tol=1e-12, cache=energy_cache)

    best_energy = float("inf")
    best_theta = None
    best_restart = -1
    best_nfev = 0
    best_nit = 0
    best_success = False
    best_message = "no run"
    best_optimizer_memory: Dict[str, Any] | None = None
    total_nfev = 0
    total_nit = 0
    run_t0 = time.perf_counter()
    heartbeat_period = max(0.0, float(progress_every_s))
    emit_heartbeat = progress_logger is not None
    history: List[Dict[str, Any]] = []
    restart_summaries: List[Dict[str, Any]] = []

    def _emit_progress(event: str, **payload: Any) -> None:
        event_payload: Dict[str, Any] = {
            "event": str(event),
            "label": str(progress_label),
            **payload,
        }
        if bool(track_history):
            history.append(dict(event_payload))
        if progress_logger is None:
            return
        try:
            progress_logger(event_payload)
        except Exception:
            # Progress telemetry must never break optimization.
            return

    _emit_progress(
        "run_start",
        restarts_total=int(restarts),
        method=str(method),
        maxiter=int(maxiter),
        npar=int(npar),
        elapsed_s=0.0,
    )

    interrupted = False

    for r in range(int(restarts)):
        if bool(use_initial_point_first_restart) and r == 0 and initial_point_arr is not None:
            x0 = np.array(initial_point_arr, copy=True)
        else:
            x0 = initial_point_stddev * rng.normal(size=npar)
        restart_optimizer_memory: Optional[Dict[str, Any]] = None
        if r == 0 and isinstance(optimizer_memory, dict):
            restart_optimizer_memory = dict(optimizer_memory)
        restart_t0 = time.perf_counter()
        restart_nfev = 0
        restart_best = float("inf")
        restart_last: Optional[float] = None
        restart_first: Optional[float] = None
        restart_last_theta: Optional[np.ndarray] = None
        restart_best_theta: Optional[np.ndarray] = None
        interrupted_by_checker = False
        heartbeat_last_t = restart_t0

        _emit_progress(
            "restart_start",
            restart_index=int(r + 1),
            restarts_total=int(restarts),
            elapsed_s=float(restart_t0 - run_t0),
            elapsed_restart_s=0.0,
            nfev_so_far=int(total_nfev),
            energy_best_global=(None if not np.isfinite(best_energy) else float(best_energy)),
            method=str(method),
            maxiter=int(maxiter),
        )

        def _objective_with_progress(x: np.ndarray) -> float:
            nonlocal restart_nfev, restart_best, restart_last, restart_first, heartbeat_last_t, restart_last_theta, restart_best_theta, interrupted_by_checker
            x_arr = np.asarray(x, dtype=float)
            e_val = float(energy_fn(x))
            restart_nfev += 1
            restart_last = float(e_val)
            restart_last_theta = np.array(x_arr, copy=True)
            if restart_first is None:
                restart_first = float(e_val)
            if e_val < restart_best:
                restart_best = float(e_val)
                restart_best_theta = np.array(x_arr, copy=True)

            now = time.perf_counter()
            energy_best_global = float(min(best_energy, restart_best))

            if early_stop_checker is not None:
                checker_payload: Dict[str, Any] = {
                    "event": "objective_step",
                    "restart_index": int(r + 1),
                    "restarts_total": int(restarts),
                    "elapsed_s": float(now - run_t0),
                    "elapsed_restart_s": float(now - restart_t0),
                    "nfev_restart": int(restart_nfev),
                    "nfev_so_far": int(total_nfev + restart_nfev),
                    "energy_current": float(restart_last),
                    "energy_restart_best": float(restart_best),
                    "energy_best_global": float(energy_best_global),
                }
                stop_now = False
                try:
                    stop_now = bool(early_stop_checker(checker_payload))
                except Exception:
                    stop_now = False
                if stop_now:
                    interrupted_by_checker = True
                    _emit_progress(
                        "early_stop_triggered",
                        restart_index=int(r + 1),
                        restarts_total=int(restarts),
                        elapsed_s=float(now - run_t0),
                        elapsed_restart_s=float(now - restart_t0),
                        nfev_restart=int(restart_nfev),
                        nfev_so_far=int(total_nfev + restart_nfev),
                        energy_current=float(restart_last),
                        energy_restart_best=float(restart_best),
                        energy_best_global=float(energy_best_global),
                        method=str(method),
                        maxiter=int(maxiter),
                    )
                    raise KeyboardInterrupt

            if emit_heartbeat:
                if (heartbeat_period == 0.0) or ((now - heartbeat_last_t) >= heartbeat_period):
                    theta_current_payload: Optional[List[float]] = None
                    theta_best_payload: Optional[List[float]] = None
                    if bool(emit_theta_in_progress):
                        if restart_last_theta is not None:
                            theta_current_payload = [float(v) for v in np.asarray(restart_last_theta, dtype=float).tolist()]
                        if restart_best_theta is not None:
                            theta_best_payload = [float(v) for v in np.asarray(restart_best_theta, dtype=float).tolist()]
                    _emit_progress(
                        "heartbeat",
                        restart_index=int(r + 1),
                        restarts_total=int(restarts),
                        elapsed_s=float(now - run_t0),
                        elapsed_restart_s=float(now - restart_t0),
                        nfev_restart=int(restart_nfev),
                        nfev_so_far=int(total_nfev + restart_nfev),
                        energy_current=float(restart_last),
                        energy_restart_best=float(restart_best),
                        energy_best_global=float(energy_best_global),
                        theta_current=theta_current_payload,
                        theta_restart_best=theta_best_payload,
                        method=str(method),
                        maxiter=int(maxiter),
                    )
                    heartbeat_last_t = now
            return float(e_val)

        method_key = str(method).strip().lower()

        if method_key == "spsa":
            bnds = None
            if bounds is not None:
                lo, hi = float(bounds[0]), float(bounds[1])
                bnds = [(lo, hi)] * npar

            def _spsa_heartbeat(_payload: Dict[str, Any]) -> None:
                nonlocal heartbeat_last_t
                if not emit_heartbeat:
                    return
                now = time.perf_counter()
                if heartbeat_period == 0.0 or (now - heartbeat_last_t) >= heartbeat_period:
                    _emit_progress(
                        "heartbeat",
                        restart_index=int(r + 1),
                        restarts_total=int(restarts),
                        elapsed_s=float(now - run_t0),
                        elapsed_restart_s=float(now - restart_t0),
                        nfev_restart=int(restart_nfev),
                        nfev_so_far=int(total_nfev + restart_nfev),
                        energy_current=float(restart_last if restart_last is not None else np.nan),
                        energy_restart_best=float(restart_best),
                        energy_best_global=float(min(best_energy, restart_best)),
                        theta_current=(
                            [float(v) for v in np.asarray(restart_last_theta, dtype=float).tolist()]
                            if (bool(emit_theta_in_progress) and restart_last_theta is not None)
                            else None
                        ),
                        theta_restart_best=(
                            [float(v) for v in np.asarray(restart_best_theta, dtype=float).tolist()]
                            if (bool(emit_theta_in_progress) and restart_best_theta is not None)
                            else None
                        ),
                        method=str(method),
                        maxiter=int(maxiter),
                    )
                    heartbeat_last_t = now

            try:
                spsa_result: SPSAResult = spsa_minimize(
                    fun=_objective_with_progress,
                    x0=x0,
                    maxiter=int(maxiter),
                    seed=int(seed) + 1000 * int(r),
                    a=float(spsa_a),
                    c=float(spsa_c),
                    alpha=float(spsa_alpha),
                    gamma=float(spsa_gamma),
                    A=float(spsa_A),
                    bounds=bnds,
                    project=("clip" if bounds is not None else "none"),
                    eval_repeats=int(spsa_eval_repeats),
                    eval_agg=str(spsa_eval_agg),
                    avg_last=int(spsa_avg_last),
                    callback=_spsa_heartbeat,
                    callback_every=1,
                    memory=restart_optimizer_memory,
                    refresh_every=int(spsa_refresh_every),
                    precondition_mode=str(spsa_precondition_mode),
                )

                energy = float(spsa_result.fun)
                theta_opt = np.asarray(spsa_result.x, dtype=float)
                nfev = int(spsa_result.nfev)
                nit = int(spsa_result.nit)
                success = bool(spsa_result.success)
                message = str(spsa_result.message)
                restart_optimizer_memory = (
                    dict(spsa_result.optimizer_memory)
                    if isinstance(spsa_result.optimizer_memory, dict)
                    else None
                )
            except KeyboardInterrupt:
                if not bool(return_best_on_keyboard_interrupt):
                    raise
                interrupted = True
                if restart_best_theta is not None and np.isfinite(restart_best):
                    theta_opt = np.asarray(restart_best_theta, dtype=float)
                    energy = float(restart_best)
                elif restart_last_theta is not None and restart_last is not None:
                    theta_opt = np.asarray(restart_last_theta, dtype=float)
                    energy = float(restart_last)
                else:
                    theta_opt = np.asarray(x0, dtype=float)
                    energy = float(energy_fn(theta_opt))
                    restart_nfev += 1
                nfev = int(max(restart_nfev, 0))
                nit = 0
                success = False
                message = (
                    "early_stop_checker_returning_best_restart"
                    if bool(interrupted_by_checker)
                    else "interrupted_keyboard_returning_best_restart"
                )
                restart_optimizer_memory = {
                    "version": "phase2_vqe_missing_optimizer_memory_v1",
                    "optimizer": "SPSA",
                    "parameter_count": int(npar),
                    "available": False,
                    "reason": "keyboard_interrupt_before_spsa_result",
                    "source": "vqe_minimize",
                }

        elif minimize is not None:
            bnds = None
            if bounds is not None:
                lo, hi = float(bounds[0]), float(bounds[1])
                bnds = [(lo, hi)] * npar

            try:
                res = minimize(
                    _objective_with_progress,
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
                restart_optimizer_memory = {
                    "version": "phase2_vqe_missing_optimizer_memory_v1",
                    "optimizer": str(method),
                    "parameter_count": int(npar),
                    "available": False,
                    "reason": "non_spsa_method",
                    "source": "vqe_minimize",
                }
            except KeyboardInterrupt:
                if not bool(return_best_on_keyboard_interrupt):
                    raise
                interrupted = True
                if restart_best_theta is not None and np.isfinite(restart_best):
                    theta_opt = np.asarray(restart_best_theta, dtype=float)
                    energy = float(restart_best)
                elif restart_last_theta is not None and restart_last is not None:
                    theta_opt = np.asarray(restart_last_theta, dtype=float)
                    energy = float(restart_last)
                else:
                    theta_opt = np.asarray(x0, dtype=float)
                    energy = float(energy_fn(theta_opt))
                    restart_nfev += 1
                nfev = int(max(restart_nfev, 0))
                nit = 0
                success = False
                message = (
                    "early_stop_checker_returning_best_restart"
                    if bool(interrupted_by_checker)
                    else "interrupted_keyboard_returning_best_restart"
                )
                restart_optimizer_memory = {
                    "version": "phase2_vqe_missing_optimizer_memory_v1",
                    "optimizer": str(method),
                    "parameter_count": int(npar),
                    "available": False,
                    "reason": "non_spsa_restart_interrupt",
                    "source": "vqe_minimize",
                }

        else:
            theta_opt = np.array(x0, dtype=float)
            step = 0.2
            nfev = int(restart_nfev)
            nit = 0
            energy = _objective_with_progress(theta_opt)
            nfev += 1

            for it in range(int(maxiter)):
                improved = False
                for k in range(npar):
                    for sgn in (+1.0, -1.0):
                        trial = theta_opt.copy()
                        trial[k] += sgn * step
                        e_trial = _objective_with_progress(trial)
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
            restart_optimizer_memory = {
                "version": "phase2_vqe_missing_optimizer_memory_v1",
                "optimizer": str(method),
                "parameter_count": int(npar),
                "available": False,
                "reason": "fallback_coordinate_search",
                "source": "vqe_minimize",
            }

        total_nfev += int(nfev)
        total_nit += int(nit)

        if restart_last is None:
            restart_last = float(energy)
        if not np.isfinite(restart_best):
            restart_best = float(energy)

        _emit_progress(
            "restart_end",
            restart_index=int(r + 1),
            restarts_total=int(restarts),
            elapsed_s=float(time.perf_counter() - run_t0),
            elapsed_restart_s=float(time.perf_counter() - restart_t0),
            nfev_restart=int(nfev),
            nit_restart=int(nit),
            nfev_so_far=int(total_nfev),
            nit_so_far=int(total_nit),
            energy_current=float(restart_last),
            energy_restart_best=float(restart_best),
            energy_best_global=float(min(best_energy, restart_best)),
            theta_restart_best=(
                [float(v) for v in np.asarray(restart_best_theta, dtype=float).tolist()]
                if (bool(emit_theta_in_progress) and restart_best_theta is not None)
                else None
            ),
            improvement_from_start=(None if restart_first is None else float(restart_first - restart_best)),
            success=bool(success),
            message=str(message),
            method=str(method),
            maxiter=int(maxiter),
        )
        restart_summaries.append(
            {
                "restart_index": int(r + 1),
                "energy": float(energy),
                "best_energy": float(restart_best),
                "nfev": int(nfev),
                "nit": int(nit),
                "success": bool(success),
                "message": str(message),
                "optimizer_memory_available": bool(
                    isinstance(restart_optimizer_memory, dict)
                    and restart_optimizer_memory.get("available", False)
                ),
            }
        )

        if energy < best_energy:
            best_energy = energy
            best_theta = theta_opt
            best_restart = r
            best_nfev = nfev
            best_nit = nit
            best_success = success
            best_message = message
            best_optimizer_memory = dict(restart_optimizer_memory) if isinstance(restart_optimizer_memory, dict) else None

        if interrupted:
            _emit_progress(
                "run_interrupted",
                elapsed_s=float(time.perf_counter() - run_t0),
                restarts_total=int(restarts),
                stopped_at_restart=int(r + 1),
                energy_best=(None if not np.isfinite(best_energy) else float(best_energy)),
                nfev_total=int(total_nfev),
                nit_total=int(total_nit),
                message=str(message),
            )
            break

    assert best_theta is not None
    _emit_progress(
        "run_end",
        elapsed_s=float(time.perf_counter() - run_t0),
        restarts_total=int(restarts),
        best_restart=int(best_restart + 1),
        energy_best=float(best_energy),
        nfev_total=int(total_nfev),
        nit_total=int(total_nit),
        success=bool(best_success),
        message=str(best_message),
        history_count=int(len(history)),
    )
    return VQEResult(
        energy=float(best_energy),
        theta=np.asarray(best_theta, dtype=float),
        success=bool(best_success),
        message=str(best_message),
        nfev=int(best_nfev),
        nit=int(best_nit),
        best_restart=int(best_restart),
        progress_history=list(history) if bool(track_history) else [],
        restart_summaries=list(restart_summaries),
        optimizer_memory=(dict(best_optimizer_memory) if isinstance(best_optimizer_memory, dict) else None),
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


def _sector_basis_indices_fermion_only(
    nq_total: int,
    alpha_indices: Sequence[int],
    beta_indices: Sequence[int],
    n_alpha: int,
    n_beta: int,
) -> List[int]:
    """Select basis states by fermion particle-number, phonon qubits unconstrained.

    Like _sector_basis_indices but only checks the specified fermion qubit
    indices; all other (phonon) qubits are free.
    """
    idxs: List[int] = []
    for idx in range(1 << int(nq_total)):
        na = sum((idx >> int(q)) & 1 for q in alpha_indices)
        nb = sum((idx >> int(q)) & 1 for q in beta_indices)
        if na == int(n_alpha) and nb == int(n_beta):
            idxs.append(int(idx))
    return idxs


def exact_ground_energy_sector_hh(
    H: PauliPolynomial,
    *,
    num_sites: int,
    num_particles: Tuple[int, int],
    n_ph_max: int,
    boson_encoding: str = "binary",
    indexing: str = "blocked",
    tol: float = 1e-12,
) -> float:
    r"""Sector-filtered ground energy for a Hubbard-Holstein Hamiltonian.

    Filters on fermion particle-number (alpha/beta spin sectors) while
    treating all phonon qubits as free.  The full Hilbert-space dimension
    is 2^{2L + L·qpb} where qpb = boson_qubits_per_site(n_ph_max).
    """
    M = hamiltonian_matrix(H, tol=tol)
    nq_total = int(round(math.log2(M.shape[0])))

    # Fermion qubit indices — identical to _spin_orbital_index_sets but within
    # the first 2*L qubits of the (2L + L·qpb)-qubit register.
    alpha_idx, beta_idx = _spin_orbital_index_sets(int(num_sites), ordering=indexing)
    n_alpha, n_beta = int(num_particles[0]), int(num_particles[1])

    basis = _sector_basis_indices_fermion_only(
        nq_total, alpha_idx, beta_idx, n_alpha, n_beta,
    )
    if not basis:
        raise ValueError(
            f"No HH basis states for sector (n_up={n_alpha}, n_dn={n_beta}) "
            f"with nq_total={nq_total}, num_sites={num_sites}."
        )

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
        "from src.quantum.vqe_latex_python_pairs import show_vqe_latex_python_pairs\n"
        "show_vqe_latex_python_pairs()"
    )

from __future__ import annotations

import inspect
import itertools
import math
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

try:
    from IPython.display import Markdown, Math, display
except Exception:  # pragma: no cover - fallback when IPython is unavailable
    Markdown = None
    Math = None
    display = None

try:
    from src.quantum.pauli_polynomial_class import (
        PauliPolynomial,
        fermion_minus_operator,
        fermion_plus_operator,
    )
    from src.quantum.qubitization_module import PauliTerm
except Exception as _dep_exc:  # pragma: no cover - allow source-inspection usage without full deps
    PauliPolynomial = Any  # type: ignore[assignment]

    def _missing_dep(*_args, **_kwargs):
        raise ImportError(
            "src.quantum dependencies are unavailable in this environment"
        ) from _dep_exc

    fermion_minus_operator = _missing_dep  # type: ignore[assignment]
    fermion_plus_operator = _missing_dep  # type: ignore[assignment]
    PauliTerm = _missing_dep  # type: ignore[assignment]

    class _FallbackLog:
        @staticmethod
        def error(msg: str):
            raise RuntimeError(msg)

    log = _FallbackLog()

class _FallbackLog:
    @staticmethod
    def error(msg: str):
        raise RuntimeError(msg)


log = _FallbackLog()

Spin = int  # 0 -> up, 1 -> down
Dims = Union[int, Tuple[int, ...]]  # L or (Lx, Ly, ...)
SitePotential = Optional[Union[float, Sequence[float], Dict[int, float]]]
TimePotential = Optional[
    Union[
        float,
        Sequence[float],
        Dict[int, float],
        Callable[[Optional[float]], Union[float, Sequence[float], Dict[int, float]]],
    ]
]

SPIN_UP: Spin = 0
SPIN_DN: Spin = 1

LATEX_TERMS: Dict[str, Dict[str, str]] = {
    "t_term": {
        "title": "T Term",
        "latex": (
            r"H_t = -t\sum_{i,j}\sum_{\sigma\in\{\uparrow,\downarrow\}}"
            r"\left("
            r"\hat{c}_{i\sigma}^{\dagger}\hat{c}_{j\sigma}"
            r"+"
            r"\hat{c}_{j\sigma}^{\dagger}\hat{c}_{i\sigma}"
            r"\right)"
        ),
    },
    "u_term": {
        "title": "U Term",
        "latex": r"H_U = U\sum_i \hat{n}_{i\uparrow}\hat{n}_{i\downarrow}",
    },
    "number_term": {
        "title": "Number Term",
        "latex": (
            r"H_v = -\sum_i\sum_{\sigma\in\{\uparrow,\downarrow\}} v_i\,\hat{n}_{i\sigma},"
            r"\qquad"
            r"\hat{n}_{i\sigma} := \hat{c}_{i\sigma}^{\dagger}\hat{c}_{i\sigma}"
        ),
    },
    "phonon_energy": {
        "title": "Phonon Energy (Holstein)",
        "latex": (
            r"H_{\mathrm{ph}} = \omega_0 \sum_i "
            r"\left( \hat{n}_{b,i} + \tfrac{1}{2} \right)"
        ),
    },
    "electron_phonon_coupling": {
        "title": "Electron-Phonon Coupling (Holstein)",
        "latex": (
            r"H_g = g \sum_i \hat{x}_i "
            r"\left( \hat{n}_i - \mathbb{1} \right),"
            r"\qquad "
            r"\hat{x}_i = \hat{b}_i + \hat{b}_i^{\dagger}"
        ),
    },
}


def mode_index(
    site: int,
    spin: Spin,
    indexing: str = "interleaved",
    n_sites: Optional[int] = None,
) -> int:
    """
    Spin-orbital indexing (0-based site), spin in {0 (up), 1 (down)}:

    - interleaved: p(site, spin) = 2*site + spin
    - blocked:     (up block first) p(site, up)=site, p(site, dn)=n_sites+site
    """
    if spin not in (SPIN_UP, SPIN_DN):
        log.error("spin must be 0 (up) or 1 (down)")
    if site < 0:
        log.error("site must be >= 0")

    if indexing == "interleaved":
        return 2 * int(site) + int(spin)
    if indexing == "blocked":
        if n_sites is None:
            log.error("n_sites is required when indexing='blocked'")
        n_sites_i = int(n_sites)
        if site >= n_sites_i:
            log.error("site index out of range for blocked indexing")
        return int(site) if spin == SPIN_UP else n_sites_i + int(site)

    log.error("indexing must be either 'interleaved' or 'blocked'")
    return -1  # unreachable


def _prod(vals: Sequence[int]) -> int:
    out = 1
    for v in vals:
        out *= int(v)
    return out


def n_sites_from_dims(dims: Dims) -> int:
    """Number of lattice sites for dims = L or dims = (Lx, Ly, ...)."""
    if isinstance(dims, int):
        if dims <= 0:
            log.error("dims must be positive")
        return int(dims)
    if len(dims) == 0:
        log.error("dims must be non-empty")
    for L in dims:
        if int(L) <= 0:
            log.error("all dims entries must be positive")
    return _prod([int(L) for L in dims])


def coord_to_site_index(coord: Sequence[int], dims: Sequence[int]) -> int:
    """Row-major linearization, x fastest: i = x + Lx*(y + Ly*(z + ...))."""
    if len(coord) != len(dims):
        log.error("coord and dims must have same length")
    idx = 0
    stride = 1
    for a in range(len(dims)):
        x = int(coord[a])
        La = int(dims[a])
        if x < 0 or x >= La:
            log.error("coord out of bounds")
        idx += x * stride
        stride *= La
    return idx


def bravais_nearest_neighbor_edges(
    dims: Dims,
    pbc: Union[bool, Sequence[bool]] = True,
) -> List[Tuple[int, int]]:
    """
    Unique undirected nearest-neighbor edges on an orthogonal Bravais lattice.

    dims:
      - int for 1D chain (L)
      - tuple for dD lattice (Lx, Ly, ...)

    pbc:
      - bool (applied on all axes)
      - or per-axis sequence of bools
    """
    if isinstance(dims, int):
        dims_t = (int(dims),)
    else:
        dims_t = tuple(int(L) for L in dims)
    d = len(dims_t)

    if isinstance(pbc, bool):
        pbc_t = (pbc,) * d
    else:
        if len(pbc) != d:
            log.error("pbc must be bool or have same length as dims")
        pbc_t = tuple(bool(b) for b in pbc)

    edges: Set[Tuple[int, int]] = set()

    for coord in itertools.product(*[range(L) for L in dims_t]):
        i = coord_to_site_index(coord, dims_t)
        for axis in range(d):
            nbr = list(coord)
            nbr[axis] += 1
            if nbr[axis] >= dims_t[axis]:
                if not pbc_t[axis]:
                    continue
                nbr[axis] %= dims_t[axis]
            j = coord_to_site_index(nbr, dims_t)
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))

    return sorted(edges)


def jw_number_operator(repr_mode: str, nq: int, p_mode: int) -> PauliPolynomial:
    """
    n_p := c_p^dagger c_p = (I - Z_p)/2 in Jordan-Wigner representation.

    Convention inherited from fermion_(plus|minus)_operator:
      qubit 0 is the rightmost character in the Pauli string.
    """
    if repr_mode != "JW":
        log.error("jw_number_operator supports repr_mode='JW' only")
    if p_mode < 0 or p_mode >= nq:
        log.error("mode index out of range -> 0 <= p_mode < nq")

    id_str = "e" * nq

    # Place 'z' on qubit p_mode, but string index is (nq - 1 - p_mode).
    z_pos = nq - 1 - int(p_mode)
    z_str = ("e" * z_pos) + "z" + ("e" * (nq - 1 - z_pos))

    return PauliPolynomial(
        repr_mode,
        [
            PauliTerm(nq, ps=id_str, pc=0.5),
            PauliTerm(nq, ps=z_str, pc=-0.5),
        ],
    )


def _resolve_hubbard_nq(n_sites: int, nq_override: Optional[int]) -> int:
    """Return total qubit count: 2*n_sites by default, or nq_override if given."""
    nq_min = 2 * int(n_sites)
    if nq_override is None:
        return nq_min
    nq = int(nq_override)
    if nq < nq_min:
        log.error("nq_override must be >= 2*n_sites")
    return nq


# ---------------------------------------------------------------------------
# Boson (phonon) infrastructure — binary encoding of truncated oscillator
# ---------------------------------------------------------------------------

def boson_qubits_per_site(n_ph_max: int, encoding: str = "binary") -> int:
    r"""
    Qubits needed per site for a local truncated phonon Hilbert space.

    d = n\_ph\_max + 1 Fock levels.
      - binary:  ceil(log2(d)) qubits
      - unary:   d = n\_ph\_max + 1 qubits  (one-hot encoding)
    """
    n_ph_i = int(n_ph_max)
    if n_ph_i < 0:
        log.error("n_ph_max must be >= 0")
    d = n_ph_i + 1
    if encoding == "binary":
        return max(1, int(math.ceil(math.log2(d))))
    if encoding == "unary":
        return d
    log.error("unknown boson encoding")
    return max(1, int(math.ceil(math.log2(d))))


def phonon_qubit_indices_for_site(
    site: int,
    *,
    n_sites: int,
    qpb: int,
    fermion_qubits: int,
) -> List[int]:
    """
    Global phonon qubit indices for one site.

    Layout: [fermion qubits | site-0 phonon qubits | site-1 phonon qubits | …]
    """
    site_i = int(site)
    n_sites_i = int(n_sites)
    qpb_i = int(qpb)
    fermion_q_i = int(fermion_qubits)

    if n_sites_i <= 0:
        log.error("n_sites must be positive")
    if site_i < 0 or site_i >= n_sites_i:
        log.error("site index out of bounds")
    if qpb_i <= 0:
        log.error("qpb must be positive")
    if fermion_q_i < 0:
        log.error("fermion_qubits must be >= 0")

    base = fermion_q_i + site_i * qpb_i
    return list(range(base, base + qpb_i))


def _normalize_pauli_symbol(sym: str) -> str:
    """Map I/X/Y/Z (or i/x/y/z) to internal e/x/y/z convention."""
    if not isinstance(sym, str) or len(sym) != 1:
        log.error("Pauli symbol must be a single character")
    trans = {"I": "e", "X": "x", "Y": "y", "Z": "z"}
    if sym in trans:
        return trans[sym]
    out = sym.lower()
    if out not in ("e", "x", "y", "z"):
        log.error(f"invalid Pauli symbol '{sym}'")
    return out


def _identity_pauli_polynomial(repr_mode: str, nq: int) -> PauliPolynomial:
    """Return the identity operator as a single-term PauliPolynomial."""
    return PauliPolynomial(repr_mode, [PauliTerm(int(nq), ps="e" * int(nq), pc=1.0)])


def _pauli_monomial_global(
    repr_mode: str,
    nq: int,
    ops: Dict[int, str],
) -> PauliPolynomial:
    """
    Build a single Pauli-string PauliPolynomial with coefficient 1.

    ops maps *global qubit index* → Pauli letter (I/X/Y/Z or e/x/y/z).
    """
    nq_i = int(nq)
    chars = ["e"] * nq_i
    for qubit, sym in ops.items():
        q = int(qubit)
        if q < 0 or q >= nq_i:
            log.error("global qubit index out of range for Pauli monomial")
        norm = _normalize_pauli_symbol(sym)
        if norm == "e":
            continue
        chars[nq_i - 1 - q] = norm
    return PauliPolynomial(repr_mode, [PauliTerm(nq_i, ps="".join(chars), pc=1.0)])


@lru_cache(maxsize=None)
def _boson_local_matrices(n_ph_max: int) -> Dict[str, np.ndarray]:
    """
    Local truncated oscillator matrices on d = n_ph_max + 1 Fock levels.

    b      – annihilation
    bdag   – creation
    n      – number  (bdag @ b)
    x      – displacement  (b + bdag)
    """
    n_ph_i = int(n_ph_max)
    if n_ph_i < 0:
        log.error("n_ph_max must be >= 0")
    d = n_ph_i + 1
    b = np.zeros((d, d), dtype=np.complex128)
    for n in range(1, d):
        b[n - 1, n] = np.sqrt(float(n))
    bdag = b.conj().T
    n_op = bdag @ b
    x_op = b + bdag
    return {"b": b, "bdag": bdag, "n": n_op, "x": x_op}


def _embed_to_qubit_space(mat_d: np.ndarray, qpb: int) -> np.ndarray:
    """Zero-pad a d×d local operator into a 2^qpb dimensional qubit subspace."""
    qpb_i = int(qpb)
    if qpb_i <= 0:
        log.error("qpb must be positive")
    d = int(mat_d.shape[0])
    if mat_d.shape != (d, d):
        log.error("mat_d must be square")
    dim = 1 << qpb_i
    if d > dim:
        log.error("local operator dimension exceeds qubit encoding dimension")
    out = np.zeros((dim, dim), dtype=np.complex128)
    out[:d, :d] = mat_d
    return out


@lru_cache(maxsize=None)
def _pauli_basis_mats(qpb: int) -> Tuple[List[str], List[np.ndarray]]:
    """All Pauli basis labels/matrices on qpb local qubits."""
    qpb_i = int(qpb)
    if qpb_i <= 0:
        log.error("qpb must be positive")

    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    oneq = {"I": I, "X": X, "Y": Y, "Z": Z}

    labels: List[str] = []
    mats: List[np.ndarray] = []
    for s in itertools.product("IXYZ", repeat=qpb_i):
        lbl = "".join(s)
        mat = oneq[s[0]]
        for k in range(1, qpb_i):
            mat = np.kron(mat, oneq[s[k]])
        labels.append(lbl)
        mats.append(mat)
    return labels, mats


def _matrix_to_pauli_coeffs(
    mat: np.ndarray,
    qpb: int,
    tol: float,
) -> List[Tuple[str, complex]]:
    """Decompose mat in the Pauli basis on qpb qubits."""
    qpb_i = int(qpb)
    labels, mats = _pauli_basis_mats(qpb_i)
    dim = 1 << qpb_i
    out: List[Tuple[str, complex]] = []
    for lbl, P in zip(labels, mats):
        coeff = np.trace(P.conj().T @ mat) / float(dim)
        if abs(coeff) > float(tol):
            out.append((lbl, complex(coeff)))
    return out


@lru_cache(maxsize=None)
def boson_local_operator_pauli_decomp(
    which: str,
    *,
    n_ph_max: int,
    qpb: int,
    tol: float,
) -> List[Tuple[str, complex]]:
    """Cached local Pauli decomposition for boson operators on one site."""
    if which not in ("b", "bdag", "n", "x"):
        log.error("which must be one of: 'b', 'bdag', 'n', 'x'")
    mats = _boson_local_matrices(int(n_ph_max))
    mat_d = mats[which]
    mat_q = _embed_to_qubit_space(mat_d, qpb=int(qpb))
    return _matrix_to_pauli_coeffs(mat_q, qpb=int(qpb), tol=float(tol))


def boson_operator(
    repr_mode: str,
    nq_total: int,
    qubits: Sequence[int],
    *,
    which: str,
    n_ph_max: int,
    encoding: str,
    tol: float = 1e-12,
) -> PauliPolynomial:
    """Bosonic operator embedded on selected global qubits.

    Supports encoding in {"binary", "unary"}.
    which in {"b", "bdag", "n", "x"}.
    """
    qubits_i = [int(q) for q in qubits]
    if len(set(qubits_i)) != len(qubits_i):
        log.error("qubits for boson operator must be unique")
    qpb = len(qubits_i)
    if qpb <= 0:
        log.error("boson operator needs at least one qubit")

    expected_qpb = boson_qubits_per_site(int(n_ph_max), encoding=encoding)
    if qpb != expected_qpb:
        log.error("provided local boson qubit block does not match encoding size")

    # ---- unary dispatch (direct Pauli construction, no matrix decomp) ----
    if encoding == "unary":
        if which == "n":
            return boson_unary_number_operator(
                repr_mode, int(nq_total), qubits_i, n_ph_max=int(n_ph_max)
            )
        if which == "x":
            return boson_unary_displacement_operator(
                repr_mode, int(nq_total), qubits_i, n_ph_max=int(n_ph_max)
            )
        if which == "bdag":
            return boson_unary_bdag_operator(
                repr_mode, int(nq_total), qubits_i, n_ph_max=int(n_ph_max)
            )
        if which == "b":
            return boson_unary_b_operator(
                repr_mode, int(nq_total), qubits_i, n_ph_max=int(n_ph_max)
            )
        log.error(f"unknown boson operator which='{which}' for unary encoding")
        return PauliPolynomial(repr_mode)  # unreachable

    # ---- binary dispatch (matrix → Pauli decomposition) ----
    if encoding != "binary":
        log.error(f"unknown boson encoding '{encoding}'")

    terms = boson_local_operator_pauli_decomp(
        which,
        n_ph_max=int(n_ph_max),
        qpb=qpb,
        tol=float(tol),
    )

    H = PauliPolynomial(repr_mode)
    for lbl, coeff in terms:
        ops: Dict[int, str] = {}
        # Local label ordered left->right as q_(qpb-1)…q_0.
        # Map so rightmost local char acts on qubits_i[0].
        for local_k, P in enumerate(lbl):
            if P == "I":
                continue
            global_q = qubits_i[qpb - 1 - local_k]
            ops[global_q] = P
        H += complex(coeff) * _pauli_monomial_global(repr_mode, int(nq_total), ops)
    return H


def boson_number_operator(
    repr_mode: str,
    nq_total: int,
    qubits: Sequence[int],
    *,
    n_ph_max: int,
    encoding: str = "binary",
    tol: float = 1e-12,
) -> PauliPolynomial:
    """Phonon number operator n_b on the given global qubits."""
    return boson_operator(
        repr_mode,
        nq_total,
        qubits,
        which="n",
        n_ph_max=n_ph_max,
        encoding=encoding,
        tol=tol,
    )


def boson_displacement_operator(
    repr_mode: str,
    nq_total: int,
    qubits: Sequence[int],
    *,
    n_ph_max: int,
    encoding: str = "binary",
    tol: float = 1e-12,
) -> PauliPolynomial:
    """Phonon displacement operator x = b + b† on the given global qubits."""
    return boson_operator(
        repr_mode,
        nq_total,
        qubits,
        which="x",
        n_ph_max=n_ph_max,
        encoding=encoding,
        tol=tol,
    )


# ---------------------------------------------------------------------------
# Unary (one-hot) boson operator helpers
# ---------------------------------------------------------------------------
#
# Unary qubits for site i:
#   q(i,n) = 2L + i*(N_b+1) + n,   n = 0..N_b
#
# Projector on Fock level n:
#   n_{i,n} = |1><1|_{q(i,n)} = (I - Z_{q(i,n)})/2
#
# Number operator on the physical (one-hot) subspace:
#   n_i = Σ_{n=0}^{N_b}  n · n_{i,n}
#
# Raising/lowering in unary:
#   (+) = (X + iY)/2,   (-) = (X - iY)/2
#   b_i† = Σ_{n=0}^{N_b-1} √(n+1) (+)_{q(i,n)} (-)_{q(i,n+1)}
#   b_i  = Σ_{n=0}^{N_b-1} √(n+1) (-)_{q(i,n)} (+)_{q(i,n+1)}
#
# Displacement simplifies:
#   x_i = b_i + b_i† = Σ_{n=0}^{N_b-1} √(n+1) (XX + YY)_{q(i,n),q(i,n+1)} / 2


def _unary_site_qubits_check(qubits: Sequence[int], *, n_ph_max: int) -> None:
    """Validate that ``qubits`` has the correct length for unary encoding."""
    qpb_expected = int(n_ph_max) + 1
    if len(qubits) != qpb_expected:
        log.error(
            f"unary qubits length must be N_b+1={qpb_expected}, got {len(qubits)}"
        )


def boson_unary_number_operator(
    repr_mode: str,
    nq_total: int,
    qubits: Sequence[int],
    *,
    n_ph_max: int,
) -> PauliPolynomial:
    r"""
    Unary number operator on the one-hot subspace:

        n = \sum_{n=0}^{N_b} n \, \hat{n}_n,  \quad  \hat{n}_n = (I - Z_n)/2

    This yields only I and Z terms (single-qubit).
    """
    _unary_site_qubits_check(qubits, n_ph_max=n_ph_max)

    I = _identity_pauli_polynomial(repr_mode, int(nq_total))
    H = PauliPolynomial(repr_mode)

    # n_n = (I - Z_q)/2;  so Σ n·n_n  = (Σ n/2)·I  -  (1/2)·Σ n·Z_q
    N_b = int(n_ph_max)
    const = 0.0
    for n_level in range(N_b + 1):
        const += 0.5 * float(n_level)
        if n_level == 0:
            continue
        # term: -(n_level/2) * Z_q
        q = int(qubits[n_level])
        H += (-(0.5 * float(n_level))) * _pauli_monomial_global(
            repr_mode, int(nq_total), {q: "Z"}
        )

    if abs(const) > 0.0:
        H += float(const) * I

    return H


def boson_unary_displacement_operator(
    repr_mode: str,
    nq_total: int,
    qubits: Sequence[int],
    *,
    n_ph_max: int,
) -> PauliPolynomial:
    r"""
    Displacement operator x = b + b† in unary form:

        x = \sum_{n=0}^{N_b-1} \sqrt{n+1}\, (XX_{n,n+1} + YY_{n,n+1})/2

    Only real XX and YY terms on adjacent one-hot qubits.
    """
    _unary_site_qubits_check(qubits, n_ph_max=n_ph_max)

    N_b = int(n_ph_max)
    x = PauliPolynomial(repr_mode)

    for n_level in range(N_b):
        qn = int(qubits[n_level])
        qp = int(qubits[n_level + 1])
        w = 0.5 * math.sqrt(float(n_level + 1))

        x += float(w) * _pauli_monomial_global(
            repr_mode, int(nq_total), {qn: "X", qp: "X"}
        )
        x += float(w) * _pauli_monomial_global(
            repr_mode, int(nq_total), {qn: "Y", qp: "Y"}
        )

    return x


def boson_unary_bdag_operator(
    repr_mode: str,
    nq_total: int,
    qubits: Sequence[int],
    *,
    n_ph_max: int,
) -> PauliPolynomial:
    r"""
    Creation operator b† in unary form:

        b^\dagger = \sum_{n=0}^{N_b-1} \sqrt{n+1}\, (+)_n (-)_{n+1}

    where (+)=(X+iY)/2, (-)=(X-iY)/2.

    Expand: (+)_a (-)_b = (1/4)[ XX - iXY + iYX + YY ]_{a,b}
    """
    _unary_site_qubits_check(qubits, n_ph_max=n_ph_max)

    N_b = int(n_ph_max)
    bdag = PauliPolynomial(repr_mode)

    for n_level in range(N_b):
        qn = int(qubits[n_level])
        qp = int(qubits[n_level + 1])
        c = 0.25 * math.sqrt(float(n_level + 1))

        bdag += complex(c) * _pauli_monomial_global(
            repr_mode, int(nq_total), {qn: "X", qp: "X"}
        )
        bdag += complex(-1j * c) * _pauli_monomial_global(
            repr_mode, int(nq_total), {qn: "X", qp: "Y"}
        )
        bdag += complex(1j * c) * _pauli_monomial_global(
            repr_mode, int(nq_total), {qn: "Y", qp: "X"}
        )
        bdag += complex(c) * _pauli_monomial_global(
            repr_mode, int(nq_total), {qn: "Y", qp: "Y"}
        )

    return bdag


def boson_unary_b_operator(
    repr_mode: str,
    nq_total: int,
    qubits: Sequence[int],
    *,
    n_ph_max: int,
) -> PauliPolynomial:
    r"""
    Annihilation operator b in unary form:

        b = \sum_{n=0}^{N_b-1} \sqrt{n+1}\, (-)_n (+)_{n+1}

    Expand: (-)_a (+)_b = (1/4)[ XX + iXY - iYX + YY ]_{a,b}
    """
    _unary_site_qubits_check(qubits, n_ph_max=n_ph_max)

    N_b = int(n_ph_max)
    b = PauliPolynomial(repr_mode)

    for n_level in range(N_b):
        qn = int(qubits[n_level])
        qp = int(qubits[n_level + 1])
        c = 0.25 * math.sqrt(float(n_level + 1))

        b += complex(c) * _pauli_monomial_global(
            repr_mode, int(nq_total), {qn: "X", qp: "X"}
        )
        b += complex(1j * c) * _pauli_monomial_global(
            repr_mode, int(nq_total), {qn: "X", qp: "Y"}
        )
        b += complex(-1j * c) * _pauli_monomial_global(
            repr_mode, int(nq_total), {qn: "Y", qp: "X"}
        )
        b += complex(c) * _pauli_monomial_global(
            repr_mode, int(nq_total), {qn: "Y", qp: "Y"}
        )

    return b


def build_hubbard_kinetic(
    dims: Dims,
    t: float,
    *,
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    edges: Optional[Sequence[Tuple[int, int]]] = None,
    pbc: Union[bool, Sequence[bool]] = True,
    nq_override: Optional[int] = None,
) -> PauliPolynomial:
    """Hopping term H_t."""
    n_sites = n_sites_from_dims(dims)
    nq = _resolve_hubbard_nq(n_sites=n_sites, nq_override=nq_override)
    if edges is None:
        edges = bravais_nearest_neighbor_edges(dims, pbc=pbc)

    c_dag: Dict[int, PauliPolynomial] = {}
    c: Dict[int, PauliPolynomial] = {}

    def cd(p_mode: int) -> PauliPolynomial:
        if p_mode not in c_dag:
            c_dag[p_mode] = fermion_plus_operator(repr_mode, nq, p_mode)
        return c_dag[p_mode]

    def cm(p_mode: int) -> PauliPolynomial:
        if p_mode not in c:
            c[p_mode] = fermion_minus_operator(repr_mode, nq, p_mode)
        return c[p_mode]

    Ht = PauliPolynomial(repr_mode)

    for (i, j) in edges:
        for spin in (SPIN_UP, SPIN_DN):
            pi = mode_index(i, spin, indexing=indexing, n_sites=n_sites)
            pj = mode_index(j, spin, indexing=indexing, n_sites=n_sites)
            Ht += (-t) * ((cd(pi) * cm(pj)) + (cd(pj) * cm(pi)))

    return Ht


def build_hubbard_onsite(
    dims: Dims,
    U: float,
    *,
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    nq_override: Optional[int] = None,
) -> PauliPolynomial:
    """Onsite interaction term H_U."""
    n_sites = n_sites_from_dims(dims)
    nq = _resolve_hubbard_nq(n_sites=n_sites, nq_override=nq_override)

    n_cache: Dict[int, PauliPolynomial] = {}

    def n_op(p_mode: int) -> PauliPolynomial:
        if p_mode not in n_cache:
            n_cache[p_mode] = jw_number_operator(repr_mode, nq, p_mode)
        return n_cache[p_mode]

    HU = PauliPolynomial(repr_mode)

    for i in range(n_sites):
        p_up = mode_index(i, SPIN_UP, indexing=indexing, n_sites=n_sites)
        p_dn = mode_index(i, SPIN_DN, indexing=indexing, n_sites=n_sites)
        HU += U * (n_op(p_up) * n_op(p_dn))

    return HU


def _parse_site_potential(
    v: Optional[Union[float, Sequence[float], Dict[int, float]]],
    n_sites: int,
) -> List[float]:
    if v is None:
        return [0.0] * n_sites
    if isinstance(v, (int, float, complex)):
        return [float(v)] * n_sites
    if isinstance(v, dict):
        out = [0.0] * n_sites
        for k, val in v.items():
            idx = int(k)
            if idx < 0 or idx >= n_sites:
                log.error("site-potential key out of bounds")
            out[idx] = float(val)
        return out
    if len(v) != n_sites:
        log.error("site potential v must be scalar, dict, or length n_sites")
    return [float(val) for val in v]


def build_hubbard_potential(
    dims: Dims,
    v: Optional[Union[float, Sequence[float], Dict[int, float]]],
    *,
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    nq_override: Optional[int] = None,
) -> PauliPolynomial:
    """Local potential term H_v."""
    n_sites = n_sites_from_dims(dims)
    nq = _resolve_hubbard_nq(n_sites=n_sites, nq_override=nq_override)
    v_list = _parse_site_potential(v, n_sites=n_sites)

    n_cache: Dict[int, PauliPolynomial] = {}

    def n_op(p_mode: int) -> PauliPolynomial:
        if p_mode not in n_cache:
            n_cache[p_mode] = jw_number_operator(repr_mode, nq, p_mode)
        return n_cache[p_mode]

    Hv = PauliPolynomial(repr_mode)
    for i in range(n_sites):
        vi = v_list[i]
        if abs(vi) < 1e-15:
            continue
        for spin in (SPIN_UP, SPIN_DN):
            p_mode = mode_index(i, spin, indexing=indexing, n_sites=n_sites)
            Hv += (-vi) * n_op(p_mode)

    return Hv


def build_hubbard_hamiltonian(
    dims: Dims,
    t: float,
    U: float,
    *,
    v: Optional[Union[float, Sequence[float], Dict[int, float]]] = None,
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    edges: Optional[Sequence[Tuple[int, int]]] = None,
    pbc: Union[bool, Sequence[bool]] = True,
    nq_override: Optional[int] = None,
) -> PauliPolynomial:
    """Full Hamiltonian H = H_t + H_U + H_v."""
    Ht = build_hubbard_kinetic(
        dims=dims,
        t=t,
        repr_mode=repr_mode,
        indexing=indexing,
        edges=edges,
        pbc=pbc,
        nq_override=nq_override,
    )
    HU = build_hubbard_onsite(
        dims=dims,
        U=U,
        repr_mode=repr_mode,
        indexing=indexing,
        nq_override=nq_override,
    )
    Hv = build_hubbard_potential(
        dims=dims,
        v=v,
        repr_mode=repr_mode,
        indexing=indexing,
        nq_override=nq_override,
    )
    return Ht + HU + Hv


# ---------------------------------------------------------------------------
# Holstein extensions
# ---------------------------------------------------------------------------

def _eval_site_potential_maybe_time_dependent(
    v: TimePotential,
    *,
    t: Optional[float],
    n_sites: int,
) -> List[float]:
    """Evaluate a site potential that may be time-dependent (callable)."""
    if v is None:
        return [0.0] * int(n_sites)
    if callable(v):
        if t is None:
            log.error("t must be provided when v_t is callable")
        return _parse_site_potential(v(t), n_sites=int(n_sites))
    return _parse_site_potential(v, n_sites=int(n_sites))


def build_holstein_phonon_energy(
    dims: Dims,
    omega0: float,
    *,
    n_ph_max: int,
    boson_encoding: str = "binary",
    repr_mode: str = "JW",
    tol: float = 1e-12,
    zero_point: bool = True,
) -> PauliPolynomial:
    r"""
    Phonon energy:
        H_{\rm ph} = \omega_0 \sum_i \bigl( \hat{n}_{b,i} + \tfrac{1}{2} \bigr)

    Each site carries a truncated bosonic mode with up to n_ph_max phonons,
    encoded in binary on ceil(log2(n_ph_max + 1)) qubits appended after the
    fermion register.
    """
    n_sites = n_sites_from_dims(dims)
    fermion_qubits = 2 * n_sites
    qpb = boson_qubits_per_site(int(n_ph_max), boson_encoding)
    nq_total = fermion_qubits + n_sites * qpb

    Hph = PauliPolynomial(repr_mode)
    I = _identity_pauli_polynomial(repr_mode, nq_total)
    for i in range(n_sites):
        q_i = phonon_qubit_indices_for_site(
            i,
            n_sites=n_sites,
            qpb=qpb,
            fermion_qubits=fermion_qubits,
        )
        n_b = boson_number_operator(
            repr_mode,
            nq_total,
            q_i,
            n_ph_max=n_ph_max,
            encoding=boson_encoding,
            tol=tol,
        )
        Hph += float(omega0) * n_b
        if zero_point:
            Hph += (0.5 * float(omega0)) * I
    return Hph


def build_holstein_coupling(
    dims: Dims,
    g: float,
    *,
    n_ph_max: int,
    boson_encoding: str = "binary",
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    tol: float = 1e-12,
) -> PauliPolynomial:
    r"""
    Electron-phonon coupling:
        H_g = g \sum_i \hat{x}_i \bigl( \hat{n}_i - \mathbb{1} \bigr)

    where  n_i = n_{i,\uparrow} + n_{i,\downarrow}  is the electronic
    occupation and  x_i = b_i + b_i^\dagger  is the phonon displacement.
    """
    n_sites = n_sites_from_dims(dims)
    fermion_qubits = 2 * n_sites
    qpb = boson_qubits_per_site(int(n_ph_max), boson_encoding)
    nq_total = fermion_qubits + n_sites * qpb

    I = _identity_pauli_polynomial(repr_mode, nq_total)
    Hg = PauliPolynomial(repr_mode)

    n_cache: Dict[int, PauliPolynomial] = {}

    def n_op(p_mode: int) -> PauliPolynomial:
        if p_mode not in n_cache:
            n_cache[p_mode] = jw_number_operator(repr_mode, nq_total, p_mode)
        return n_cache[p_mode]

    for i in range(n_sites):
        p_up = mode_index(i, SPIN_UP, indexing=indexing, n_sites=n_sites)
        p_dn = mode_index(i, SPIN_DN, indexing=indexing, n_sites=n_sites)
        n_i = n_op(p_up) + n_op(p_dn)

        q_i = phonon_qubit_indices_for_site(
            i,
            n_sites=n_sites,
            qpb=qpb,
            fermion_qubits=fermion_qubits,
        )
        x_i = boson_displacement_operator(
            repr_mode,
            nq_total,
            q_i,
            n_ph_max=n_ph_max,
            encoding=boson_encoding,
            tol=tol,
        )
        Hg += float(g) * (x_i * (n_i + ((-1.0) * I)))
    return Hg


def build_hubbard_holstein_drive(
    dims: Dims,
    *,
    v_t: TimePotential,
    v0: SitePotential,
    t: Optional[float] = None,
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    nq_override: Optional[int] = None,
) -> PauliPolynomial:
    r"""
    Time-dependent drive for Holstein:
        H_{\rm drive} = \sum_{i,\sigma} \bigl(v_i(t) - v_{0,i}\bigr) \hat{n}_{i\sigma}
    """
    n_sites = n_sites_from_dims(dims)
    v_t_list = _eval_site_potential_maybe_time_dependent(v_t, t=t, n_sites=n_sites)
    v0_list = _parse_site_potential(v0, n_sites=n_sites)
    delta = [float(v_t_list[i]) - float(v0_list[i]) for i in range(n_sites)]

    # build_hubbard_potential uses H_v = -sum_i,sigma v_i * n_i,sigma
    v_for_existing = [-dv for dv in delta]
    return build_hubbard_potential(
        dims=dims,
        v=v_for_existing,
        repr_mode=repr_mode,
        indexing=indexing,
        nq_override=nq_override,
    )


def build_hubbard_holstein_hamiltonian(
    dims: Dims,
    *,
    J: float,
    U: float,
    omega0: float,
    g: float,
    n_ph_max: int,
    boson_encoding: str = "binary",
    v_t: TimePotential = None,
    v0: SitePotential = None,
    t_eval: Optional[float] = None,
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    edges: Optional[Sequence[Tuple[int, int]]] = None,
    pbc: Union[bool, Sequence[bool]] = True,
    tol: float = 1e-12,
    include_zero_point: bool = True,
) -> PauliPolynomial:
    r"""
    Full Hubbard-Holstein Hamiltonian on a shared fermion+phonon register:

        H = H_{\rm Hubb} + H_{\rm ph} + H_g + H_{\rm drive}

    Qubit layout: [2·L fermion qubits | L · qpb phonon qubits]
    """
    n_sites = n_sites_from_dims(dims)
    fermion_qubits = 2 * n_sites
    qpb = boson_qubits_per_site(int(n_ph_max), boson_encoding)
    nq_total = fermion_qubits + n_sites * qpb

    H_hubb = build_hubbard_hamiltonian(
        dims=dims,
        t=J,
        U=U,
        v=None,
        repr_mode=repr_mode,
        indexing=indexing,
        edges=edges,
        pbc=pbc,
        nq_override=nq_total,
    )
    H_ph = build_holstein_phonon_energy(
        dims=dims,
        omega0=omega0,
        n_ph_max=n_ph_max,
        boson_encoding=boson_encoding,
        repr_mode=repr_mode,
        tol=tol,
        zero_point=include_zero_point,
    )
    H_g = build_holstein_coupling(
        dims=dims,
        g=g,
        n_ph_max=n_ph_max,
        boson_encoding=boson_encoding,
        repr_mode=repr_mode,
        indexing=indexing,
        tol=tol,
    )
    H_drive = build_hubbard_holstein_drive(
        dims=dims,
        v_t=v_t,
        v0=v0,
        t=t_eval,
        repr_mode=repr_mode,
        indexing=indexing,
        nq_override=nq_total,
    )
    return H_hubb + H_ph + H_g + H_drive


def show_latex_and_code(title: str, latex_expr: str, fn) -> None:
    if display is not None and Math is not None:
        if title:
            display(Markdown(f"### {title}"))
        display(Math(latex_expr))
    else:
        if title:
            print(f"### {title}")
        print(latex_expr)

    print(inspect.getsource(fn))


def show_hubbard_latex_python_pairs() -> None:
    """Render built-in LaTeX terms and print corresponding Python implementations."""
    show_latex_and_code(
        LATEX_TERMS["t_term"]["title"],
        LATEX_TERMS["t_term"]["latex"],
        build_hubbard_kinetic,
    )
    show_latex_and_code(
        LATEX_TERMS["u_term"]["title"],
        LATEX_TERMS["u_term"]["latex"],
        build_hubbard_onsite,
    )
    show_latex_and_code(
        LATEX_TERMS["number_term"]["title"],
        LATEX_TERMS["number_term"]["latex"],
        build_hubbard_potential,
    )
    show_latex_and_code(
        LATEX_TERMS["phonon_energy"]["title"],
        LATEX_TERMS["phonon_energy"]["latex"],
        build_holstein_phonon_energy,
    )
    show_latex_and_code(
        LATEX_TERMS["electron_phonon_coupling"]["title"],
        LATEX_TERMS["electron_phonon_coupling"]["latex"],
        build_holstein_coupling,
    )


if __name__ == "__main__":
    print(
        "Use this in Jupyter for rendered LaTeX:\n"
        "from src.quantum.hubbard_latex_python_pairs import show_hubbard_latex_python_pairs\n"
        "show_hubbard_latex_python_pairs()"
    )

from __future__ import annotations

import inspect
import itertools
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

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
            "pydephasing quantum dependencies are unavailable in this environment"
        ) from _dep_exc

    fermion_minus_operator = _missing_dep  # type: ignore[assignment]
    fermion_plus_operator = _missing_dep  # type: ignore[assignment]
    PauliTerm = _missing_dep  # type: ignore[assignment]

    class _FallbackLog:
        @staticmethod
        def error(msg: str):
            raise RuntimeError(msg)

    log = _FallbackLog()

try:
    from src.utilities.log import log
except Exception:  # pragma: no cover - local fallback when utilities package is absent
    class _FallbackLog:
        @staticmethod
        def error(msg: str):
            raise RuntimeError(msg)

    log = _FallbackLog()

Spin = int  # 0 -> up, 1 -> down
Dims = Union[int, Tuple[int, ...]]  # L or (Lx, Ly, ...)

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


def build_hubbard_kinetic(
    dims: Dims,
    t: float,
    *,
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    edges: Optional[Sequence[Tuple[int, int]]] = None,
    pbc: Union[bool, Sequence[bool]] = True,
) -> PauliPolynomial:
    """Hopping term H_t."""
    n_sites = n_sites_from_dims(dims)
    nq = 2 * n_sites
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
) -> PauliPolynomial:
    """Onsite interaction term H_U."""
    n_sites = n_sites_from_dims(dims)
    nq = 2 * n_sites

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
) -> PauliPolynomial:
    """Local potential term H_v."""
    n_sites = n_sites_from_dims(dims)
    nq = 2 * n_sites
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
) -> PauliPolynomial:
    """Full Hamiltonian H = H_t + H_U + H_v."""
    Ht = build_hubbard_kinetic(
        dims=dims,
        t=t,
        repr_mode=repr_mode,
        indexing=indexing,
        edges=edges,
        pbc=pbc,
    )
    HU = build_hubbard_onsite(
        dims=dims,
        U=U,
        repr_mode=repr_mode,
        indexing=indexing,
    )
    Hv = build_hubbard_potential(
        dims=dims,
        v=v,
        repr_mode=repr_mode,
        indexing=indexing,
    )
    return Ht + HU + Hv


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


if __name__ == "__main__":
    print(
        "Use this in Jupyter for rendered LaTeX:\n"
        "from src.quantum.hubbard_latex_python_pairs import show_hubbard_latex_python_pairs\n"
        "show_hubbard_latex_python_pairs()"
    )

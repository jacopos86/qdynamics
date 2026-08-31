"""CFQM (commutator-free quasi-Magnus) scheme registry.

Public API
----------
get_cfqm_scheme(scheme_id: str) -> dict

Accepted scheme ids (case-insensitive):
- CF4:2: ``"cfqm4"``, ``"cf4:2"``, ``"cf4"``
- CF6:5Opt: ``"cfqm6"``, ``"cf6:5opt"``, ``"cf6"``
"""

from __future__ import annotations

from typing import Any


_DEFAULT_TOL = 1e-12


_SCHEME_DATA: dict[str, dict[str, Any]] = {
    "CF4:2": {
        "name": "CF4:2",
        "order": 4,
        "c": [
            0.2113248654051871,
            0.7886751345948129,
        ],
        "a": [
            [-0.0386751345948129, 0.5386751345948129],
            [0.5386751345948129, -0.0386751345948129],
        ],
        "_expected_row_sums": [
            0.5,
            0.5,
        ],
    },
    "CF6:5Opt": {
        "name": "CF6:5Opt",
        "order": 6,
        "c": [
            0.0694318442029737123880,
            0.3300094782075718675987,
            0.6699905217924281324013,
            0.9305681557970262876120,
        ],
        "a": [
            [-0.0025014052514919785, 0.0086390299226631599, -0.0241007202550846312, 0.1893630955839134498],
            [0.0118125500877527529, -0.0524780954434914029, 0.4364266743440699121, -0.0207973857888688969],
            [-0.0039494320625783987, -0.0424143111368839665, -0.0424143111368839665, -0.0039494320625783987],
            [-0.0207973857888688969, 0.4364266743440699121, -0.0524780954434914029, 0.0118125500877527529],
            [0.1893630955839134498, -0.0241007202550846312, 0.0086390299226631599, -0.0025014052514919785],
        ],
        "_expected_row_sums": [
            0.1714,
            0.37496374319946236513,
            -0.09272748639892473026,
            0.37496374319946236513,
            0.1714,
        ],
        "_expected_col_sums": [
            0.1739274225687269286865,
            0.3260725774312730713135,
            0.3260725774312730713135,
            0.1739274225687269286865,
        ],
    },
}


_SCHEME_ALIASES = {
    "cfqm4": "CF4:2",
    "cf4:2": "CF4:2",
    "cf4": "CF4:2",
    "cfqm6": "CF6:5Opt",
    "cf6:5opt": "CF6:5Opt",
    "cf6": "CF6:5Opt",
}


def _norm_key(scheme_id: str) -> str:
    return str(scheme_id).strip().lower()


def _close(lhs: float, rhs: float, tol: float) -> bool:
    return abs(float(lhs) - float(rhs)) <= float(tol)


def _row_sums(a: list[list[float]]) -> list[float]:
    return [float(sum(row)) for row in a]


def _col_sums(a: list[list[float]]) -> list[float]:
    if not a:
        return []
    m = len(a[0])
    out = [0.0] * m
    for row in a:
        for j, coeff in enumerate(row):
            out[j] += float(coeff)
    return [float(x) for x in out]


def validate_scheme(scheme: dict[str, Any], *, tol: float = _DEFAULT_TOL) -> None:
    """Validate CFQM scheme data and invariants."""
    required = ("name", "order", "c", "a", "s_static")
    for key in required:
        if key not in scheme:
            raise ValueError(f"Scheme missing required field: {key!r}")

    name = str(scheme["name"])
    order = int(scheme["order"])
    c = scheme["c"]
    a = scheme["a"]
    s_static = scheme["s_static"]

    if order <= 0:
        raise ValueError(f"{name}: order must be positive.")
    if not isinstance(c, list) or not c:
        raise ValueError(f"{name}: c must be a non-empty list.")
    if not isinstance(a, list) or not a:
        raise ValueError(f"{name}: a must be a non-empty list of rows.")

    m = len(c)
    s = len(a)
    if not isinstance(s_static, list) or len(s_static) != s:
        raise ValueError(f"{name}: s_static length must equal stage count.")

    for idx, node in enumerate(c):
        node_f = float(node)
        if node_f < -tol or node_f > 1.0 + tol:
            raise ValueError(f"{name}: node c[{idx}]={node_f} not in [0,1].")

    for k, row in enumerate(a):
        if not isinstance(row, list) or len(row) != m:
            raise ValueError(f"{name}: row a[{k}] must have length {m}.")

    total = float(sum(sum(float(x) for x in row) for row in a))
    if not _close(total, 1.0, tol):
        raise ValueError(f"{name}: sum(all a)={total} violates invariant == 1.")

    computed_rows = _row_sums(a)
    for k, (given_row_sum, computed_row_sum) in enumerate(zip(s_static, computed_rows)):
        if not _close(float(given_row_sum), float(computed_row_sum), tol):
            raise ValueError(
                f"{name}: s_static[{k}]={given_row_sum} does not match row-sum {computed_row_sum}."
            )

    expected_rows = scheme.get("_expected_row_sums")
    if expected_rows is not None:
        if len(expected_rows) != len(computed_rows):
            raise ValueError(f"{name}: expected row-sum length mismatch.")
        for k, (expected, computed) in enumerate(zip(expected_rows, computed_rows)):
            if not _close(float(expected), float(computed), tol):
                raise ValueError(
                    f"{name}: row-sum[{k}]={computed} does not match expected {expected}."
                )

    expected_cols = scheme.get("_expected_col_sums")
    if expected_cols is not None:
        computed_cols = _col_sums(a)
        if len(expected_cols) != len(computed_cols):
            raise ValueError(f"{name}: expected column-sum length mismatch.")
        for j, (expected, computed) in enumerate(zip(expected_cols, computed_cols)):
            if not _close(float(expected), float(computed), tol):
                raise ValueError(
                    f"{name}: col-sum[{j}]={computed} does not match expected {expected}."
                )


def get_cfqm_scheme(scheme_id: str) -> dict[str, Any]:
    """Return validated CFQM scheme data for *scheme_id*.

    Returned fields:
    - name: str
    - order: int
    - c: list[float]
    - a: list[list[float]]
    - s_static: list[float]
    """
    key = _norm_key(scheme_id)
    canonical = _SCHEME_ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(sorted(_SCHEME_ALIASES))
        raise ValueError(f"Unknown CFQM scheme_id={scheme_id!r}. Supported ids: {supported}.")

    data = _SCHEME_DATA[canonical]
    c = [float(x) for x in data["c"]]
    a = [[float(v) for v in row] for row in data["a"]]
    out: dict[str, Any] = {
        "name": str(data["name"]),
        "order": int(data["order"]),
        "c": c,
        "a": a,
        "s_static": _row_sums(a),
    }

    if "_expected_row_sums" in data:
        out["_expected_row_sums"] = [float(x) for x in data["_expected_row_sums"]]
    if "_expected_col_sums" in data:
        out["_expected_col_sums"] = [float(x) for x in data["_expected_col_sums"]]

    validate_scheme(out)
    out.pop("_expected_row_sums", None)
    out.pop("_expected_col_sums", None)
    return out


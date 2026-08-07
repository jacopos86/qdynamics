"""Shared Paper-I Qiskit cost-row and tuple presentation."""

from __future__ import annotations

from typing import Any, Callable, Mapping


PAPER_I_QISKIT_COST_TUPLE_FIELDS = (
    "N2q",
    "D2q",
    "Dc",
    "W1q",
    "S_alg",
)
PAPER_I_QISKIT_COST_TUPLE_LATEX = (
    r"(N_{2q},D_{2q},D_c,W_{1q},S_{\rm alg})"
)


def qiskit_cost_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize compiled Paper-I resources and require attributed basis work."""

    metrics = payload.get("metrics")
    source = metrics if isinstance(metrics, Mapping) else payload
    raw_compile_payload = payload.get("raw_compile_payload")
    basis_source = (
        raw_compile_payload
        if isinstance(raw_compile_payload, Mapping)
        else payload
    )
    fields = {
        "N2q": source.get("N2q", source.get("compiled_count_2q_total")),
        "D2q": source.get("D2q", source.get("compiled_depth_2q_total")),
        "Dc": source.get("Dc", source.get("compiled_depth_total")),
    }
    if any(value is None for value in fields.values()):
        raise ValueError(f"compiled Qiskit metric triplet is incomplete: {fields}")
    status = str(
        basis_source.get("qiskit_basis_work_status")
        or source.get("qiskit_basis_work_status")
        or "unavailable_legacy_compile_payload"
    )
    work = basis_source.get("qiskit_pretranspile_pauli_1q_work_total")
    if work is None:
        work = source.get("W1q")
    if status != "ok" or work is None:
        raise ValueError(
            "Paper-I Qiskit one-qubit Pauli work is unavailable: "
            f"status={status!r}, W1q={work!r}"
        )
    basis_change = basis_source.get(
        "qiskit_pretranspile_basis_change_1q_total"
    )
    if basis_change is None:
        basis_change = source.get("B1q")
    return {
        **{key: int(value) for key, value in fields.items()},
        "W1q": int(work),
        "B1q": None if basis_change is None else int(basis_change),
        "qiskit_basis_work_status": status,
        "qiskit_basis_work_schema": basis_source.get(
            "qiskit_basis_work_schema"
        )
        or source.get("qiskit_basis_work_schema"),
    }


def paper_i_cost_tuple_latex(
    row: Mapping[str, Any],
    *,
    marker: str,
    format_s_alg: Callable[[int], str],
) -> str:
    """Format the canonical five-field in-panel Paper-I cost tuple."""

    missing = [
        field for field in PAPER_I_QISKIT_COST_TUPLE_FIELDS if row.get(field) is None
    ]
    if missing:
        raise ValueError(f"Paper-I cost tuple is missing fields: {missing}")
    marker_prefix = rf"{marker}\;" if marker else ""
    return (
        rf"${marker_prefix}("
        rf"{int(row['N2q'])},{int(row['D2q'])},{int(row['Dc'])},"
        rf"{int(row['W1q'])},{format_s_alg(int(row['S_alg']))})$"
    )


__all__ = [
    "PAPER_I_QISKIT_COST_TUPLE_FIELDS",
    "PAPER_I_QISKIT_COST_TUPLE_LATEX",
    "paper_i_cost_tuple_latex",
    "qiskit_cost_fields",
]

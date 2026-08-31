"""Deterministic Pauli-ansatz cost on a declared connectivity graph.

This module owns the routing-independent graph proxy used by Paper I and by
Paper-II terminal reporting.  It does not transpile.  A caller supplies one
logical-to-physical embedding and one undirected coupling graph; every ordered
generator occurrence is expanded into its non-negligible Pauli rotations and
priced on that fixed graph.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from src.quantum.vqe_latex_python_pairs import AnsatzTerm


MARRAKESH_GRAPH_SPAN_MODE = "marrakesh_graph_span_v1"
PAPER_I_MARRAKESH_GRAPH_SPAN_PROFILE_V1 = (
    "paper_i_fake_marrakesh_graph_span_v1"
)

_MARRAKESH_GRAPH_SPAN_EMBEDDINGS_V1: dict[int, tuple[int, ...]] = {
    4: (0, 1, 2, 3),
    6: (0, 1, 2, 3, 4, 16),
    8: (0, 1, 2, 3, 4, 16, 5, 23),
    9: (0, 1, 2, 3, 4, 16, 5, 23, 6),
    10: (0, 1, 2, 3, 4, 16, 5, 23, 6, 22),
    12: (0, 1, 2, 3, 4, 16, 5, 23, 6, 22, 7, 21),
}
_PAULI_CHARS = {"e", "x", "y", "z"}
_ABSOLUTE_COMPONENTS = (
    "absolute_c_hat_2q",
    "absolute_c_hat_d",
    "absolute_c_hat_1q",
    "absolute_theta_count",
)


@dataclass(frozen=True)
class PauliGraphSpanLedger:
    """Absolute analytical resource ledger for one ordered ansatz."""

    profile_id: str
    backend_name: str
    num_qubits: int
    embedding: tuple[int, ...]
    coupling_edge_count: int
    graph_snapshot: Mapping[str, Any]
    coefficient_tolerance: float
    ordered_generator_occurrence_count: int
    nonidentity_pauli_rotation_count: int
    independent_parameter_count: int
    absolute_c_hat_2q: float
    absolute_c_hat_d: float
    absolute_c_hat_1q: float
    ordered_support_sha256: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": "pauli_graph_span_absolute_cost_v1",
            "profile_id": str(self.profile_id),
            "backend_name": str(self.backend_name),
            "no_transpile": True,
            "routing_seed_used": False,
            "num_qubits": int(self.num_qubits),
            "embedding_logical_to_physical": [int(q) for q in self.embedding],
            "coupling_edge_count": int(self.coupling_edge_count),
            "graph_snapshot": dict(self.graph_snapshot),
            "coefficient_tolerance": float(self.coefficient_tolerance),
            "ordered_generator_occurrence_count": int(
                self.ordered_generator_occurrence_count
            ),
            "nonidentity_pauli_rotation_count": int(
                self.nonidentity_pauli_rotation_count
            ),
            "independent_parameter_count": int(self.independent_parameter_count),
            "absolute_c_hat_2q": float(self.absolute_c_hat_2q),
            "absolute_c_hat_d": float(self.absolute_c_hat_d),
            "absolute_c_hat_1q": float(self.absolute_c_hat_1q),
            # Compatibility with the Paper-I final-summary name.  For Paper II
            # this is the explicitly supplied runtime-coordinate count.
            "absolute_theta_count": float(self.independent_parameter_count),
            "ordered_support_sha256": str(self.ordered_support_sha256),
        }


def marrakesh_logical_embedding_v1(num_qubits: int) -> tuple[int, ...]:
    """Return Paper I's fixed logical embedding into FakeMarrakesh."""

    nq = int(num_qubits)
    if nq not in _MARRAKESH_GRAPH_SPAN_EMBEDDINGS_V1:
        raise ValueError(
            f"unsupported_marrakesh_graph_span_embedding_v1_size:{nq}"
        )
    return tuple(int(q) for q in _MARRAKESH_GRAPH_SPAN_EMBEDDINGS_V1[nq])


def _normalize_pauli_label_exyz(raw: Any, *, num_qubits: int) -> str:
    label = str(raw).strip().lower().replace("i", "e")
    if len(label) != int(num_qubits):
        raise ValueError(
            f"pauli_label_length_mismatch:{len(label)}!={int(num_qubits)}:{label}"
        )
    invalid = sorted({ch for ch in label if ch not in _PAULI_CHARS})
    if invalid:
        raise ValueError(f"invalid_pauli_label:{label}:{''.join(invalid)}")
    return label


def marrakesh_pauli_support_from_label(
    label: str,
    *,
    num_qubits: int,
) -> tuple[int, ...]:
    """Translate the repository's big-endian EXYZ word into logical qubits."""

    normalized = _normalize_pauli_label_exyz(label, num_qubits=int(num_qubits))
    nq = int(num_qubits)
    return tuple(
        int(nq - 1 - idx)
        for idx, ch in enumerate(normalized)
        if ch != "e"
    )


def extract_pauli_graph_span_terms(
    candidate_term: AnsatzTerm,
    *,
    num_qubits: int,
    coefficient_tolerance: float = 1.0e-12,
) -> tuple[dict[str, Any], ...]:
    """Return the active Pauli words needed to price one generator occurrence."""

    polynomial = getattr(candidate_term, "polynomial", None)
    if polynomial is None or not hasattr(polynomial, "return_polynomial"):
        raise ValueError("candidate_missing_pauli_polynomial")
    terms = list(polynomial.return_polynomial())
    if not terms:
        raise ValueError("candidate_missing_pauli_polynomial")
    out: list[dict[str, Any]] = []
    for term in terms:
        coeff = complex(getattr(term, "p_coeff"))
        if abs(coeff) <= float(coefficient_tolerance):
            continue
        label = _normalize_pauli_label_exyz(
            term.pw2strng(), num_qubits=int(num_qubits)
        )
        support = marrakesh_pauli_support_from_label(
            label, num_qubits=int(num_qubits)
        )
        out.append(
            {
                "label": str(label),
                "coefficient_abs": float(abs(coeff)),
                "logical_support": [int(q) for q in support],
                "support_size": int(len(support)),
                "is_identity": bool(len(support) == 0),
            }
        )
    return tuple(out)


def _adjacency_from_edges(
    edges: Sequence[Sequence[int]],
) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for raw_u, raw_v in edges:
        u = int(raw_u)
        v = int(raw_v)
        if u == v:
            continue
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    return adjacency


def _bfs_distances(
    adjacency: Mapping[int, set[int]],
    source: int,
) -> dict[int, int]:
    src = int(source)
    distances = {src: 0}
    queue: deque[int] = deque([src])
    while queue:
        node = int(queue.popleft())
        for neighbor in sorted(adjacency.get(node, set())):
            if int(neighbor) in distances:
                continue
            distances[int(neighbor)] = int(distances[node] + 1)
            queue.append(int(neighbor))
    return distances


def marrakesh_graph_span_edges_for_support(
    physical_support: Sequence[int],
    *,
    coupling_edges: Sequence[Sequence[int]],
) -> int:
    """Return the exact minimum connected-subgraph edge count for a support."""

    terminals = tuple(dict.fromkeys(int(q) for q in physical_support))
    if len(terminals) == 0:
        return 0
    adjacency = _adjacency_from_edges(coupling_edges)
    for terminal in terminals:
        if int(terminal) not in adjacency:
            raise ValueError(
                f"marrakesh_graph_span_terminal_not_in_graph:{terminal}"
            )
    if len(terminals) == 1:
        return 0
    if len(terminals) == 2:
        distances = _bfs_distances(adjacency, int(terminals[0]))
        if int(terminals[1]) not in distances:
            raise RuntimeError("marrakesh_graph_span_disconnected_support")
        return int(distances[int(terminals[1])])

    nodes = sorted(adjacency)
    all_mask = (1 << len(terminals)) - 1
    infinity = 10**12
    terminal_distances = {
        int(terminal): _bfs_distances(adjacency, int(terminal))
        for terminal in terminals
    }
    dynamic: dict[int, dict[int, int]] = {
        mask: {node: infinity for node in nodes}
        for mask in range(1, all_mask + 1)
    }
    for terminal_index, terminal in enumerate(terminals):
        mask = 1 << terminal_index
        distances = terminal_distances[int(terminal)]
        for node in nodes:
            if node in distances:
                dynamic[mask][node] = int(distances[node])
    for mask in range(1, all_mask + 1):
        submask = (mask - 1) & mask
        while submask:
            other = mask ^ submask
            if other:
                for node in nodes:
                    merged = int(
                        dynamic[submask][node] + dynamic[other][node]
                    )
                    if merged < dynamic[mask][node]:
                        dynamic[mask][node] = merged
            submask = (submask - 1) & mask
        improved = True
        while improved:
            improved = False
            for node, neighbors in adjacency.items():
                base = dynamic[mask][int(node)]
                if base >= infinity:
                    continue
                for neighbor in neighbors:
                    candidate = int(base + 1)
                    if candidate < dynamic[mask][int(neighbor)]:
                        dynamic[mask][int(neighbor)] = candidate
                        improved = True
    answer = min(dynamic[all_mask].values())
    if answer >= infinity:
        raise RuntimeError("marrakesh_graph_span_disconnected_support")
    return int(answer)


def one_qubit_template_count(pauli_terms: Sequence[Mapping[str, Any]]) -> float:
    """Count basis changes and the central rotation in Paper I's template."""

    total = 0.0
    for row in pauli_terms:
        label = str(row.get("label", ""))
        if int(row.get("support_size", 0)) <= 0:
            continue
        total += float(2 * label.count("x") + 4 * label.count("y") + 1)
    return float(total)


def pauli_graph_span_rows_for_term(
    candidate_term: AnsatzTerm,
    *,
    num_qubits: int,
    embedding: Sequence[int],
    coupling_edges: Sequence[Sequence[int]],
    coefficient_tolerance: float = 1.0e-12,
    span_edges_for_support: Callable[[Sequence[int]], int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return Paper-I proxy rows for every active Pauli word in one term."""

    logical_to_physical = tuple(int(q) for q in embedding)
    if len(logical_to_physical) != int(num_qubits):
        raise ValueError(
            "graph_span_embedding_width_mismatch:"
            f"{len(logical_to_physical)}!={int(num_qubits)}"
        )
    extracted = extract_pauli_graph_span_terms(
        candidate_term,
        num_qubits=int(num_qubits),
        coefficient_tolerance=float(coefficient_tolerance),
    )
    rows: list[dict[str, Any]] = []
    for row in extracted:
        logical_support = [int(q) for q in row.get("logical_support", [])]
        physical_support = [logical_to_physical[q] for q in logical_support]
        support_size = int(len(logical_support))
        span_edges = (
            int(span_edges_for_support(physical_support))
            if span_edges_for_support is not None
            else marrakesh_graph_span_edges_for_support(
                physical_support,
                coupling_edges=coupling_edges,
            )
        )
        rows.append(
            {
                **dict(row),
                "physical_support": [int(q) for q in physical_support],
                "span_edges": int(span_edges),
                "c_hat_2q_term": float(2 * max(0, support_size - 1)),
                "c_hat_d_term": float(2 * int(span_edges)),
            }
        )
    return tuple(rows)


def evaluate_pauli_graph_span_ansatz(
    terms: Sequence[AnsatzTerm],
    *,
    num_qubits: int,
    embedding: Sequence[int],
    coupling_edges: Sequence[Sequence[int]],
    backend_name: str,
    graph_snapshot: Mapping[str, Any],
    independent_parameter_count: int | None,
    profile_id: str = PAPER_I_MARRAKESH_GRAPH_SPAN_PROFILE_V1,
    coefficient_tolerance: float = 1.0e-12,
    span_edges_for_support: Callable[[Sequence[int]], int] | None = None,
) -> PauliGraphSpanLedger:
    """Evaluate one complete ordered ansatz without transpilation."""

    ordered_terms = tuple(terms)
    all_rows: list[dict[str, Any]] = []
    ordered_support: list[dict[str, Any]] = []
    for occurrence_index, term in enumerate(ordered_terms):
        rows = pauli_graph_span_rows_for_term(
            term,
            num_qubits=int(num_qubits),
            embedding=embedding,
            coupling_edges=coupling_edges,
            coefficient_tolerance=float(coefficient_tolerance),
            span_edges_for_support=span_edges_for_support,
        )
        all_rows.extend(rows)
        ordered_support.append(
            {
                "occurrence_index": int(occurrence_index),
                "label": str(getattr(term, "label", "")),
                "pauli_words": [
                    {
                        "label": str(row["label"]),
                        "coefficient_abs": float(row["coefficient_abs"]),
                    }
                    for row in rows
                ],
            }
        )
    support_digest = hashlib.sha256(
        json.dumps(
            ordered_support,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    resolved_parameter_count = (
        int(independent_parameter_count)
        if independent_parameter_count is not None
        else int(
            sum(
                1
                for occurrence in ordered_support
                if bool(occurrence["pauli_words"])
            )
        )
    )
    return PauliGraphSpanLedger(
        profile_id=str(profile_id),
        backend_name=str(backend_name),
        num_qubits=int(num_qubits),
        embedding=tuple(int(q) for q in embedding),
        coupling_edge_count=int(len(tuple(coupling_edges))),
        graph_snapshot=dict(graph_snapshot),
        coefficient_tolerance=float(coefficient_tolerance),
        ordered_generator_occurrence_count=int(len(ordered_terms)),
        nonidentity_pauli_rotation_count=int(
            sum(1 for row in all_rows if int(row["support_size"]) > 0)
        ),
        independent_parameter_count=resolved_parameter_count,
        absolute_c_hat_2q=float(
            sum(float(row["c_hat_2q_term"]) for row in all_rows)
        ),
        absolute_c_hat_d=float(
            sum(float(row["c_hat_d_term"]) for row in all_rows)
        ),
        absolute_c_hat_1q=float(one_qubit_template_count(all_rows)),
        ordered_support_sha256=str(support_digest),
    )


def compare_pauli_graph_span_ansatzes(
    *,
    initial_terms: Sequence[AnsatzTerm],
    terminal_terms: Sequence[AnsatzTerm],
    num_qubits: int,
    embedding: Sequence[int],
    coupling_edges: Sequence[Sequence[int]],
    backend_name: str,
    graph_snapshot: Mapping[str, Any],
    initial_parameter_count: int,
    terminal_parameter_count: int,
    profile_id: str = PAPER_I_MARRAKESH_GRAPH_SPAN_PROFILE_V1,
    coefficient_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Return auditable absolute and seed-relative ledgers for a trajectory."""

    initial = evaluate_pauli_graph_span_ansatz(
        initial_terms,
        num_qubits=int(num_qubits),
        embedding=embedding,
        coupling_edges=coupling_edges,
        backend_name=str(backend_name),
        graph_snapshot=graph_snapshot,
        independent_parameter_count=int(initial_parameter_count),
        profile_id=str(profile_id),
        coefficient_tolerance=float(coefficient_tolerance),
    ).to_json_dict()
    terminal = evaluate_pauli_graph_span_ansatz(
        terminal_terms,
        num_qubits=int(num_qubits),
        embedding=embedding,
        coupling_edges=coupling_edges,
        backend_name=str(backend_name),
        graph_snapshot=graph_snapshot,
        independent_parameter_count=int(terminal_parameter_count),
        profile_id=str(profile_id),
        coefficient_tolerance=float(coefficient_tolerance),
    ).to_json_dict()
    delta = {
        key: float(terminal[key]) - float(initial[key])
        for key in _ABSOLUTE_COMPONENTS
    }
    return {
        "schema": "pauli_graph_span_trajectory_comparison_v1",
        "profile_id": str(profile_id),
        "reporting_role": "routing_independent_structural_cost_primary_v1",
        "no_transpile": True,
        "same_declared_graph_for_initial_and_terminal": True,
        "initial": initial,
        "terminal": terminal,
        "terminal_minus_initial": delta,
    }


__all__ = [
    "MARRAKESH_GRAPH_SPAN_MODE",
    "PAPER_I_MARRAKESH_GRAPH_SPAN_PROFILE_V1",
    "PauliGraphSpanLedger",
    "compare_pauli_graph_span_ansatzes",
    "evaluate_pauli_graph_span_ansatz",
    "extract_pauli_graph_span_terms",
    "marrakesh_graph_span_edges_for_support",
    "marrakesh_logical_embedding_v1",
    "marrakesh_pauli_support_from_label",
    "one_qubit_template_count",
    "pauli_graph_span_rows_for_term",
]

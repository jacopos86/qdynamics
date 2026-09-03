"""Pin the Paper-I FakeMarrakesh graph-span embedding table, including 19q.

The table maps logical qubits to physical FakeMarrakesh qubits for the
analytic ``marrakesh_graph_span_v1`` cost proxy.  Sizes 4..12 are the
historical Paper-I Hubbard--Holstein entries; size 19 is the Paper-IV H2O
(12 fermion + 7 boson qubit) continuation of the same generating rule.  Every
size is checked against the real coupling map: prefix nesting, distinct
in-range qubits, connectivity, and the tree property the historical entries
share.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt import hh_backend_compile_oracle as oracle_module
from pipelines.resource_estimation import pauli_graph_span as span_module
from pipelines.static_adapt.hh_backend_compile_oracle import (
    BackendCompileConfig,
    MARRAKESH_GRAPH_SPAN_MODE,
    MarrakeshGraphSpanCostOracle,
    marrakesh_graph_span_edges_for_support,
    marrakesh_logical_embedding_v1,
)
from pipelines.scaffold.hh_continuation_types import CompileCostEstimate
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


# Historical entries, pinned byte-for-byte.  These back sealed Paper-I
# graph-span costs and must never move.
HISTORICAL_EMBEDDINGS_V1: dict[int, tuple[int, ...]] = {
    4: (0, 1, 2, 3),
    6: (0, 1, 2, 3, 4, 16),
    8: (0, 1, 2, 3, 4, 16, 5, 23),
    9: (0, 1, 2, 3, 4, 16, 5, 23, 6),
    10: (0, 1, 2, 3, 4, 16, 5, 23, 6, 22),
    12: (0, 1, 2, 3, 4, 16, 5, 23, 6, 22, 7, 21),
}
# Paper-IV H2O continuation of the same rule (see the module comment).
EMBEDDING_19Q_V1: tuple[int, ...] = (
    0, 1, 2, 3, 4, 16, 5, 23, 6, 22, 7, 21,
    8, 20, 9, 24, 10, 25, 11,
)
EXPECTED_SIZES = (4, 6, 8, 9, 10, 12, 19)


@pytest.fixture(scope="module")
def marrakesh_graph():
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    backend = FakeMarrakesh()
    edges = sorted(
        {
            tuple(sorted((int(u), int(v))))
            for u, v in backend.coupling_map.get_edges()
        }
    )
    adjacency: dict[int, set[int]] = {}
    for u, v in edges:
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    return int(backend.num_qubits), edges, adjacency


def _induced_edges(edges, qubits):
    member = set(qubits)
    return [(u, v) for u, v in edges if u in member and v in member]


def _connected(adjacency, qubits) -> bool:
    member = set(qubits)
    seen = {qubits[0]}
    stack = [qubits[0]]
    while stack:
        node = stack.pop()
        for neighbour in adjacency.get(node, ()):
            if neighbour in member and neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen == member


def test_table_sizes_are_exactly_the_expected_ones() -> None:
    assert tuple(sorted(oracle_module._MARRAKESH_GRAPH_SPAN_EMBEDDINGS_V1)) == (
        EXPECTED_SIZES
    )


def test_historical_entries_are_byte_for_byte_unchanged() -> None:
    for size, expected in HISTORICAL_EMBEDDINGS_V1.items():
        assert marrakesh_logical_embedding_v1(size) == expected
        assert oracle_module._MARRAKESH_GRAPH_SPAN_EMBEDDINGS_V1[size] == expected


def test_19q_entry_is_the_pinned_tuple() -> None:
    assert marrakesh_logical_embedding_v1(19) == EMBEDDING_19Q_V1
    assert len(EMBEDDING_19Q_V1) == 19


def test_resource_estimation_mirror_table_is_identical() -> None:
    assert (
        span_module._MARRAKESH_GRAPH_SPAN_EMBEDDINGS_V1
        == oracle_module._MARRAKESH_GRAPH_SPAN_EMBEDDINGS_V1
    )
    assert span_module.marrakesh_logical_embedding_v1(19) == EMBEDDING_19Q_V1


def test_generating_rule_reproduces_every_entry() -> None:
    """Logical 0..3 -> row 0; then alternate the top and bottom arms."""

    top_arm = [4, 5, 6, 7, 8, 9, 10, 11]
    bottom_arm = [16, 23, 22, 21, 20, 24, 25]
    sequence = [0, 1, 2, 3]
    top_index = bottom_index = 0
    for logical in range(4, 19):
        if logical % 2 == 0:
            sequence.append(top_arm[top_index])
            top_index += 1
        else:
            sequence.append(bottom_arm[bottom_index])
            bottom_index += 1
    for size in EXPECTED_SIZES:
        assert marrakesh_logical_embedding_v1(size) == tuple(sequence[:size])


@pytest.mark.parametrize("size", EXPECTED_SIZES)
def test_every_size_is_a_nested_connected_tree_on_the_real_coupling_map(
    size: int, marrakesh_graph
) -> None:
    num_qubits, edges, adjacency = marrakesh_graph
    embedding = marrakesh_logical_embedding_v1(size)

    assert len(embedding) == size
    assert len(set(embedding)) == size
    assert all(0 <= qubit < num_qubits for qubit in embedding)
    assert all(qubit in adjacency for qubit in embedding)

    smaller = [other for other in EXPECTED_SIZES if other < size]
    if smaller:
        previous = marrakesh_logical_embedding_v1(max(smaller))
        assert embedding[: len(previous)] == previous

    assert _connected(adjacency, embedding)
    assert len(_induced_edges(edges, embedding)) == size - 1


def test_unsupported_sizes_still_fail_closed() -> None:
    for size in (5, 7, 11, 13, 15, 20):
        with pytest.raises(ValueError, match="unsupported_marrakesh_graph_span_embedding_v1_size"):
            marrakesh_logical_embedding_v1(size)


def _term(label: str, pauli: str) -> AnsatzTerm:
    return AnsatzTerm(
        label=str(label),
        polynomial=PauliPolynomial(
            "JW", [PauliTerm(len(pauli), ps=pauli, pc=1.0)]
        ),
    )


def test_oracle_constructs_at_19q_and_prices_supports_on_the_new_qubits() -> None:
    oracle = MarrakeshGraphSpanCostOracle(
        config=BackendCompileConfig(
            mode=MARRAKESH_GRAPH_SPAN_MODE,
            requested_backend_name="FakeMarrakesh",
            seed_transpiler=7,
            optimization_level=1,
        ),
        num_qubits=19,
    )

    assert oracle.embedding == EMBEDDING_19Q_V1
    assert oracle.targets[0].resolved_name == "FakeMarrakesh"
    assert oracle.targets[0].resolution_kind == "fake_exact"

    # A support spanning exactly the seven qubits added beyond size 12.
    new_physical = [EMBEDDING_19Q_V1[q] for q in range(12, 19)]
    assert new_physical == [8, 20, 9, 24, 10, 25, 11]
    assert marrakesh_graph_span_edges_for_support(
        new_physical, coupling_edges=oracle.coupling_edges
    ) == 13
    assert marrakesh_graph_span_edges_for_support(
        [20, 25], coupling_edges=oracle.coupling_edges
    ) == 5
    # Trivial supports.
    assert marrakesh_graph_span_edges_for_support(
        [EMBEDDING_19Q_V1[18]], coupling_edges=oracle.coupling_edges
    ) == 0

    # Logical support {17, 18} (highest boson qubits) -> physical {25, 11}.
    label = "".join("z" if q in (17, 18) else "e" for q in range(19))[::-1]
    snapshot = oracle.snapshot_base(())
    estimate = oracle.estimate_insertion(
        snapshot, candidate_term=_term("boson_zz", label), position_id=0
    )
    assert isinstance(estimate, CompileCostEstimate)
    assert estimate.source_mode == MARRAKESH_GRAPH_SPAN_MODE
    row = estimate.selected_backend_row["pauli_terms"][0]
    assert sorted(row["physical_support"]) == [11, 25]
    assert row["span_edges"] == marrakesh_graph_span_edges_for_support(
        [11, 25], coupling_edges=oracle.coupling_edges
    )
    assert estimate.c_hat_2q == pytest.approx(2.0)
    assert estimate.c_hat_d == pytest.approx(2.0 * row["span_edges"])

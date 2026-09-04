"""Pin the Paper-I FakeMarrakesh graph-span embedding table, including 19q.

The table maps logical qubits to physical FakeMarrakesh qubits for the
analytic ``marrakesh_graph_span_v1`` cost proxy.  Sizes 4..12 are the
historical Paper-I Hubbard--Holstein entries and are sealed.

Size 19 (Paper-IV H2O, 12 fermion + 7 boson qubits) is a CHOSEN extension,
not a derivation.  The user selected embedding F, which preserves the nested,
tree-valued historical prefix and places the seven added qubits contiguously
on hardware row 0.  Other admissible continuations reproduce all six
historical entries.  The choice moves the reported numbers (see the note above
``_MARRAKESH_GRAPH_SPAN_EMBEDDINGS_V1`` in
``pipelines/static_adapt/hh_backend_compile_oracle.py``).

So these tests separate two kinds of claim:

* PINS -- the chosen tuples themselves, byte-for-byte, and the boson-block
  Steiner span edges they imply on the tracked 19q H2O fixture register.
  These exist so that changing the choice is a visible, deliberate test
  update rather than a silent cost shift.
* PROPERTIES -- what is actually checked against the real coupling map:
  distinct in-range qubits, prefix nesting, and growth by adjacency into a
  tree at every logical index.  A property test constrains the choice; it
  does not make it.
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
    8, 9, 10, 11, 12, 13, 14,
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


def test_every_entry_is_the_chosen_tuple_truncated() -> None:
    """Every table row is a prefix of the one chosen 19-tuple.

    This is a PIN, not a derivation.  The two arm lists below are a
    transcription of the chosen tuple, so agreeing with them proves only
    that the table is internally consistent -- that all seven rows are
    prefixes of a single sequence -- and nothing about why that sequence
    rather than one of the other 429 admissible ones.  The property tests
    below are what actually check the sequence against the hardware.
    """

    sequence = list(EMBEDDING_19Q_V1)
    for size in EXPECTED_SIZES:
        assert marrakesh_logical_embedding_v1(size) == tuple(sequence[:size])


def test_chosen_tuple_grows_by_adjacency_into_a_tree_at_every_index(
    marrakesh_graph,
) -> None:
    """PROPERTY: from the coupling map alone, with nothing hard-coded.

    At every logical index the next physical qubit is new, in range, and
    adjacent to exactly one already-placed qubit.  "Exactly one" is the tree
    condition stated incrementally: a second edge back into the placed set
    would close a cycle.  Index 0 has no predecessor.  This constrains, but
    does not determine, the chosen tuple.
    """

    num_qubits, _edges, adjacency = marrakesh_graph
    embedding = marrakesh_logical_embedding_v1(19)

    placed: list[int] = []
    for logical, physical in enumerate(embedding):
        assert 0 <= physical < num_qubits
        assert physical in adjacency
        assert physical not in placed
        back_edges = sorted(
            other for other in placed if other in adjacency[physical]
        )
        if logical == 0:
            assert back_edges == []
        else:
            assert len(back_edges) == 1, (
                f"logical {logical} -> physical {physical} attaches to "
                f"{back_edges}; a tree extension attaches to exactly one "
                "already-placed qubit"
            )
        placed.append(physical)


@pytest.mark.parametrize("size", EXPECTED_SIZES)
def test_each_table_entry_extends_its_predecessor_by_adjacent_qubits(
    size: int, marrakesh_graph
) -> None:
    """PROPERTY: nesting plus adjacency-only growth between table sizes."""

    _num_qubits, _edges, adjacency = marrakesh_graph
    smaller = [other for other in EXPECTED_SIZES if other < size]
    if not smaller:
        return
    previous = marrakesh_logical_embedding_v1(max(smaller))
    embedding = marrakesh_logical_embedding_v1(size)
    assert embedding[: len(previous)] == previous

    placed = list(previous)
    for physical in embedding[len(previous) :]:
        assert physical not in placed
        assert any(other in adjacency[physical] for other in placed)
        placed.append(physical)


def test_boson_block_steiner_spans_are_pinned_for_the_tracked_fixture() -> None:
    """PIN: the cost the chosen 19-tuple actually imposes on H2O.

    The Paper-IV H2O register puts the three vibrational modes at logical
    qubits 12-13, 14-16 and 17-18.  Under the chosen embedding those land on
    physical (8, 9), (10, 11, 12) and (13, 14), whose Steiner span edge
    counts on FakeMarrakesh are (1, 2, 1).  Alternative admissible
    embeddings give different numbers -- lane gives (10, 8, 8) and B gives
    (8, 7, 6) -- so this triple is the visible consequence of the
    choice, and any future embedding change must update it deliberately.
    """

    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    from src.quantum.chemistry.vibronic_h2o_linear_fd import (
        load_production_vibronic_h2o_fixture,
    )

    fixture_path = (
        REPO_ROOT
        / "test_support"
        / "paper_iv_h2o"
        / "h2o_linear_fd_work373_ref484_fixture_20260826.json"
    )
    assert fixture_path.is_file(), fixture_path
    layout = load_production_vibronic_h2o_fixture(fixture_path).layout

    # The register the pin is stated against.
    assert layout.n_total_qubits == 19
    assert layout.n_fermion_qubits == 12
    boson_blocks = tuple(tuple(block.qubits) for block in layout.boson_modes)
    assert boson_blocks == ((12, 13), (14, 15, 16), (17, 18))

    coupling_edges = sorted(
        {
            tuple(sorted((int(u), int(v))))
            for u, v in FakeMarrakesh().coupling_map.get_edges()
        }
    )
    embedding = marrakesh_logical_embedding_v1(19)
    physical_blocks = tuple(
        tuple(embedding[logical] for logical in block) for block in boson_blocks
    )
    assert physical_blocks == ((8, 9), (10, 11, 12), (13, 14))

    spans = tuple(
        marrakesh_graph_span_edges_for_support(
            list(block), coupling_edges=coupling_edges
        )
        for block in physical_blocks
    )
    assert spans == (1, 2, 1)


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
    assert new_physical == [8, 9, 10, 11, 12, 13, 14]
    assert marrakesh_graph_span_edges_for_support(
        new_physical, coupling_edges=oracle.coupling_edges
    ) == 6
    assert marrakesh_graph_span_edges_for_support(
        [9, 13], coupling_edges=oracle.coupling_edges
    ) == 4
    # Trivial supports.
    assert marrakesh_graph_span_edges_for_support(
        [EMBEDDING_19Q_V1[18]], coupling_edges=oracle.coupling_edges
    ) == 0

    # Logical support {17, 18} (highest boson qubits) -> physical {13, 14}.
    label = "".join("z" if q in (17, 18) else "e" for q in range(19))[::-1]
    snapshot = oracle.snapshot_base(())
    estimate = oracle.estimate_insertion(
        snapshot, candidate_term=_term("boson_zz", label), position_id=0
    )
    assert isinstance(estimate, CompileCostEstimate)
    assert estimate.source_mode == MARRAKESH_GRAPH_SPAN_MODE
    row = estimate.selected_backend_row["pauli_terms"][0]
    assert sorted(row["physical_support"]) == [13, 14]
    assert row["span_edges"] == marrakesh_graph_span_edges_for_support(
        [13, 14], coupling_edges=oracle.coupling_edges
    )
    assert estimate.c_hat_2q == pytest.approx(2.0)
    assert estimate.c_hat_d == pytest.approx(2.0 * row["span_edges"])

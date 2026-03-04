from __future__ import annotations

from pipelines.exact_bench.hh_seq_transition_utils import build_pool_b_strict_union
from src.quantum.qubitization_module import PauliTerm
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _single_term_op(label: str, coeff: float, tag: str) -> AnsatzTerm:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(2, ps=str(label), pc=float(coeff)))
    return AnsatzTerm(label=str(tag), polynomial=poly)


def test_pool_b_strict_union_dedups_and_tracks_presence() -> None:
    uccsd = [
        _single_term_op("xe", 1.0, "u0"),
    ]
    hva = [
        _single_term_op("xe", 1.0, "h_dup_with_u"),
        _single_term_op("ze", 1.0, "h1"),
    ]
    paop = [
        _single_term_op("ze", 1.0, "p_dup_with_h"),
        _single_term_op("ey", 1.0, "p1"),
    ]

    dedup, meta, source_by_sig = build_pool_b_strict_union(
        uccsd_ops=uccsd,
        hva_ops=hva,
        paop_full_ops=paop,
    )

    assert len(dedup) == 3
    assert int(meta["dedup_total"]) == 3
    assert int(meta["raw_sizes"]["uccsd"]) == 1
    assert int(meta["raw_sizes"]["hva"]) == 2
    assert int(meta["raw_sizes"]["paop_full"]) == 2

    presence = meta["dedup_source_presence_counts"]
    assert int(presence["uccsd"]) == 1
    assert int(presence["hva"]) == 2
    assert int(presence["paop_full"]) == 2
    assert int(meta["overlap_count"]) == 2

    # Ensure source provenance map was populated for all dedup signatures.
    assert len(source_by_sig) == 3

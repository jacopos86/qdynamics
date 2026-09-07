"""Complete-patch compiled cost: triples, words, placement, cache, failure."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
import multiprocessing
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

import pipelines.qiskit_backend_tools as backend_tools
from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.hardcoded.adapt_circuit_execution import build_structural_ansatz_circuit
from pipelines.time_dynamics.ap_mclachlan import compiled_patch_cost as cpc
from pipelines.time_dynamics.ap_mclachlan.commutation import singleton_insertion_cuts
from pipelines.time_dynamics.ap_mclachlan.exchange_certification import (
    remap_cuts_after_deletion,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
    COMPILE_COORDINATE_WEIGHTS,
    CompileContract,
)
from pipelines.time_dynamics.ap_mclachlan.state import state_from_scaffold_runtime_input
from src.quantum.ansatz_parameterization import (
    build_parameter_layout,
    iter_runtime_rotation_terms,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(*components: tuple[str, float], nq: int = 2) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for word, coeff in components:
        poly.add_term(PauliTerm(int(nq), ps=str(word), pc=float(coeff)))
    poly._reduce()
    return poly


# Single-qubit blocks (compile to zero two-qubit gates).
X0 = AnsatzTerm(label="sx0", polynomial=_poly(("ex", 1.0)))
Z0 = AnsatzTerm(label="sz0", polynomial=_poly(("ez", 1.0)))
Y1 = AnsatzTerm(label="sy1", polynomial=_poly(("ye", 1.0)))
CA = AnsatzTerm(label="ca", polynomial=_poly(("xe", 0.7)))
CB = AnsatzTerm(label="cb", polynomial=_poly(("zz", 0.4)))
# Two-qubit blocks (each costs entangling gates).
XX = AnsatzTerm(label="xx", polynomial=_poly(("xx", 0.5)))
ZZ = AnsatzTerm(label="zz", polynomial=_poly(("zz", 0.4)))
YY = AnsatzTerm(label="yy", polynomial=_poly(("yy", 0.3)))
XY = AnsatzTerm(label="xy", polynomial=_poly(("xy", 0.6)))

CONTRACT = CompileContract()


def _state(selected, theta):
    layout = build_parameter_layout(selected)
    executor = CompiledAnsatzExecutor(
        tuple(selected),
        parameterization_layout=layout,
        parameterization_mode="per_pauli_term",
    )
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0
    psi_initial = executor.prepare_state(np.asarray(theta, dtype=float), psi_ref)
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(
            family_key="toy", hamiltonian=_poly(("ez", 2.0), ("xx", 0.6))
        ),
        psi_ref=psi_ref,
        psi_initial=np.asarray(psi_initial, dtype=complex),
        base_layout=layout,
        theta_runtime=np.asarray(theta, dtype=float),
        theta_logical=np.zeros(int(layout.logical_parameter_count), dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=tuple(selected),
        candidate_pool_terms=tuple(selected),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool", pool_key="toy_pool", completeness="complete"
        ),
        provenance={"artifact_json": "toy.json"},
    )
    return state_from_scaffold_runtime_input(runtime_input)


def _specs(term) -> tuple[tuple[str, float, int], ...]:
    return tuple(
        (s.pauli_exyz, float(s.coeff_real), int(s.nq))
        for s in iter_runtime_rotation_terms(term.polynomial)
    )


def _weighted(marginal: tuple[int, int, int]) -> float:
    w = COMPILE_COORDINATE_WEIGHTS
    total = w["delta_n2q"] + w["delta_d2q"] + w["delta_dc"]
    return (
        w["delta_n2q"] * marginal[0]
        + w["delta_d2q"] * marginal[1]
        + w["delta_dc"] * marginal[2]
    ) / total


@pytest.fixture(scope="module")
def backend():
    backend_obj, provenance = cpc.resolve_compile_backend(CONTRACT)
    return backend_obj, provenance


# --- resource triple --------------------------------------------------------


def test_resource_triple_matches_extractors_on_hand_built_circuit(backend) -> None:
    from qiskit import QuantumCircuit

    backend_obj, _ = backend
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.3, 1)
    qc.cx(0, 1)
    triple, provenance = cpc.compile_resource_triple(qc, backend_obj, CONTRACT)

    compiled = backend_tools.compile_circuit_for_backend(
        qc, backend_obj, seed_transpiler=7, optimization_level=1
    )["compiled"]
    stats = backend_tools.compiled_gate_stats(compiled)
    assert triple.as_tuple() == (
        int(stats["compiled_count_2q"]),
        int(stats["compiled_depth_2q"]),
        int(backend_tools.safe_circuit_depth(compiled)),
    )
    assert triple.n2q == 2 and triple.d2q == 2 and triple.dc > 0
    assert provenance["optimization_level"] == 1
    assert provenance["transpile_seed"] == 7
    assert provenance["structural_angle"] == 1.0
    assert provenance["logical_num_qubits"] == 2
    assert "compiled_op_counts" in provenance


def test_resource_triple_arithmetic_and_validation() -> None:
    a = cpc.CompiledResourceTriple(n2q=6, d2q=6, dc=32)
    b = cpc.CompiledResourceTriple(n2q=4.0, d2q=4, dc=23)
    assert (a - b).as_tuple() == (2, 2, 9)
    assert b.minus(a).as_delta_dict() == {"delta_n2q": -2, "delta_d2q": -2, "delta_dc": -9}
    with pytest.raises(cpc.CompiledPatchCostError):
        cpc.CompiledResourceTriple(n2q=float("nan"), d2q=0, dc=0)
    with pytest.raises(cpc.CompiledPatchCostError):
        cpc.CompiledResourceTriple(n2q=None, d2q=0, dc=0)
    with pytest.raises(cpc.CompiledPatchCostError):
        cpc.CompiledResourceTriple(n2q=1.5, d2q=0, dc=0)


# --- word identity ------------------------------------------------------------


def test_word_identity_stable_order_angle_nq_sensitive_label_insensitive() -> None:
    sha = cpc.reference_state_sha256(None)
    key = cpc.word_identity((X0, Z0, Y1), 2, sha, 1.0)
    assert key == cpc.word_identity((X0, Z0, Y1), 2, sha, 1.0)
    assert len(key) == 64
    assert key != cpc.word_identity((Z0, X0, Y1), 2, sha, 1.0)
    assert key != cpc.word_identity((X0, Z0, Y1), 2, sha, 0.5)
    assert key != cpc.word_identity((X0, Z0), 2, sha, 1.0)
    relabeled = AnsatzTerm(label="other", polynomial=X0.polynomial)
    assert key == cpc.word_identity((relabeled, Z0, Y1), 2, sha, 1.0)
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1.0
    assert key != cpc.word_identity((X0, Z0, Y1), 2, cpc.reference_state_sha256(psi), 1.0)
    with pytest.raises(cpc.CompiledPatchCostError):
        cpc.word_identity((X0,), 3, sha, 1.0)


def test_word_identity_expansion_matches_builder_layout() -> None:
    multi = AnsatzTerm(label="m", polynomial=_poly(("zz", 0.4), ("xx", 0.5), ("ee", 0.9)))
    terms = (Y1, multi, X0)
    layout, _circuit = build_structural_ansatz_circuit(list(terms), nq=2, ref_state=None)
    builder = [
        [(s.pauli_exyz, float(s.coeff_real), int(s.nq)) for s in block.terms]
        for block in layout.blocks
    ]
    ours = [
        [(s.pauli_exyz, float(s.coeff_real), int(s.nq)) for s in specs]
        for specs in cpc.word_rotation_specs(terms)
    ]
    assert ours == builder
    assert builder[1] == [("xx", 0.5, 2), ("zz", 0.4, 2)]


# --- structural word ------------------------------------------------------------


def test_structural_word_deletes_remaps_and_splices() -> None:
    state = _state((X0, Z0, Y1), np.array([0.3, -0.2, 0.5]))
    labels = state.runtime_coordinate_labels
    base = cpc.base_word(state)
    assert [t.label for t in base] == list(labels)
    assert [_specs(t) for t in base] == [_specs(X0), _specs(Z0), _specs(Y1)]

    # Deletion (index 0) strictly below insertion cut 2 -> cut 2 remaps to 1;
    # cut 0 has no deletion below it and stays 0.
    insertions = ((0, CA, "ca@0"), (2, CB, "cb@2"))
    assert remap_cuts_after_deletion((0, 2), (0,)) == (0, 1)
    word = cpc.structural_word(state, (0,), insertions)
    assert [t.label for t in word] == ["ca@0", labels[1], "cb@2", labels[2]]
    assert [_specs(t) for t in word] == [_specs(CA), _specs(Z0), _specs(CB), _specs(Y1)]

    # Deletion (index 2) after insertion cut 1 -> cut unchanged.
    word = cpc.structural_word(state, (2,), ((1, CA, "ca@1"),))
    assert [t.label for t in word] == [labels[0], "ca@1", labels[1]]

    # Equal remapped cuts keep the given order; tail append at cut n.
    word = cpc.structural_word(state, (), ((3, CB, "first"), (3, CA, "second")))
    assert [t.label for t in word] == list(labels) + ["first", "second"]

    # Deletion-only patch.
    word = cpc.structural_word(state, (1,), ())
    assert [t.label for t in word] == [labels[0], labels[2]]

    with pytest.raises(cpc.CompiledPatchCostError):
        cpc.structural_word(state, (3,), ())
    with pytest.raises(cpc.CompiledPatchCostError):
        cpc.structural_word(state, (), ((4, CA, "bad"),))


# --- commutation classes ----------------------------------------------------------


@pytest.mark.parametrize(
    "term, blocks, expected",
    [
        (CA, (X0, Z0, Y1), {0: (0, 1, 2), 3: (3,)}),
        (CB, (X0, Z0, Y1), {0: (0,), 1: (1, 2), 3: (3,)}),
        (XY, (XX, ZZ, YY), {0: (0,), 1: (1, 2), 3: (3,)}),
        (Z0, (X0, Z0, Y1), {0: (0,), 1: (1, 2, 3)}),
        (Z0, (), {0: (0,)}),
    ],
)
def test_insertion_class_members_keys_and_partition(term, blocks, expected) -> None:
    classes = cpc.insertion_class_members(term, blocks)
    assert classes == expected
    assert tuple(classes) == singleton_insertion_cuts(term, blocks)
    flat = [p for members in classes.values() for p in members]
    assert sorted(flat) == list(range(len(blocks) + 1))
    assert len(flat) == len(set(flat))
    for representative, members in classes.items():
        assert members[0] == representative
        assert list(members) == list(range(members[0], members[-1] + 1))


# --- placement ----------------------------------------------------------------------


def _scripted_cache(script: dict[tuple[str, ...], tuple[int, int, int]]):
    calls: list[tuple[str, ...]] = []

    def compiler(terms, *, nq, ref_state, contract, backend_obj):
        key = tuple(t.label for t in terms)
        calls.append(key)
        return cpc.CompiledResourceTriple(*script[key]), {"scripted": True}

    return cpc.CompiledPatchCostCache(word_compiler=compiler), calls


def test_choose_placement_scripted_min_and_tie_breaks() -> None:
    a = AnsatzTerm(label="a", polynomial=X0.polynomial)
    b = AnsatzTerm(label="b", polynomial=Z0.polynomial)
    c = AnsatzTerm(label="c", polynomial=CA.polynomial)
    common = dict(base_terms=(a, b), nq=2, ref_state=None, contract=CONTRACT, backend_obj=None)

    # Cuts 0 and 1 tie on the weighted marginal with identical raw tuples:
    # the lower cut wins.  Cut 2 is strictly worse.
    cache, calls = _scripted_cache(
        {
            ("a", "b"): (4, 4, 10),
            ("c", "a", "b"): (5, 5, 12),
            ("a", "c", "b"): (5, 5, 12),
            ("a", "b", "c"): (6, 5, 12),
        }
    )
    placement = cpc.choose_placement(c, (0, 1, 2), cache=cache, **common)
    assert placement.chosen_cut == 0
    assert placement.marginal.as_tuple() == (1, 1, 2)
    assert [x.cut for x in placement.candidates_evaluated] == [0, 1, 2]
    assert placement.compile_calls_used == 4
    assert cache.compile_calls == 4 and cache.cache_hits == 0
    assert placement.placement_rule_id == "paper_ii_min_weighted_raw_marginal_placement_v1"
    assert placement.candidates_evaluated[0].weighted_marginal == pytest.approx(
        _weighted((1, 1, 2))
    )
    assert placement.candidates_evaluated[2].weighted_marginal > placement.candidates_evaluated[0].weighted_marginal

    # Equal weighted marginal, different raw tuples: (0,1,0) < (1,0,0), so
    # cut 1 beats the lower cut 0.
    cache, _ = _scripted_cache(
        {("a", "b"): (4, 4, 10), ("c", "a", "b"): (5, 4, 10), ("a", "c", "b"): (4, 5, 10)}
    )
    placement = cpc.choose_placement(c, (0, 1), cache=cache, **common)
    assert placement.candidates_evaluated[0].weighted_marginal == placement.candidates_evaluated[1].weighted_marginal
    assert placement.chosen_cut == 1
    assert placement.marginal.as_tuple() == (0, 1, 0)

    # A strictly smaller weighted marginal wins regardless of cut.
    cache, _ = _scripted_cache(
        {
            ("a", "b"): (4, 4, 10),
            ("c", "a", "b"): (6, 6, 14),
            ("a", "c", "b"): (6, 6, 14),
            ("a", "b", "c"): (4, 4, 10),
        }
    )
    placement = cpc.choose_placement(c, (0, 1, 2), cache=cache, **common)
    assert placement.chosen_cut == 2
    assert placement.marginal.as_tuple() == (0, 0, 0)
    assert placement.to_json_dict()["chosen_cut"] == 2

    with pytest.raises(cpc.CompiledPatchCostError):
        cpc.choose_placement(c, (), cache=cache, **common)
    with pytest.raises(cpc.CompiledPatchCostError):
        cpc.choose_placement(c, (5,), cache=cache, **common)


def test_choose_placement_real_compile_and_cache_reuse(backend) -> None:
    backend_obj, _ = backend
    cache = cpc.CompiledPatchCostCache()
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1.0
    common = dict(nq=2, ref_state=psi, contract=CONTRACT, backend_obj=backend_obj, cache=cache)

    # CA commutes with X0 and Z0: class {0, 1, 2}; every member compiles to the
    # same single-qubit circuit, so all tie and the lowest cut wins.
    classes = cpc.insertion_class_members(CA, (X0, Z0, Y1))
    placement = cpc.choose_placement(CA, classes[0], base_terms=(X0, Z0, Y1), **common)
    assert placement.chosen_cut == 0
    assert placement.marginal.as_tuple() == (0, 0, 0)
    assert placement.candidate_count == 3
    assert placement.compile_calls_used == 4
    assert cache.compile_calls == 4 and cache.cache_hits == 0

    # Repeating the placement is served entirely from the cache.
    again = cpc.choose_placement(CA, classes[0], base_terms=(X0, Z0, Y1), **common)
    assert again.chosen_cut == 0
    assert again.compile_calls_used == 0
    assert cache.compile_calls == 4 and cache.cache_hits == 4
    assert all(c.from_cache for c in again.candidates_evaluated)

    # XY crosses ZZ only: class {1, 2} inside (XX, ZZ, YY).  The rule must
    # reproduce the explicit min over (weighted, raw tuple, cut).
    classes = cpc.insertion_class_members(XY, (XX, ZZ, YY))
    assert classes[1] == (1, 2)
    placement = cpc.choose_placement(XY, classes[1], base_terms=(XX, ZZ, YY), **common)
    rows = [(c.weighted_marginal, c.marginal.as_tuple(), c.cut) for c in placement.candidates_evaluated]
    assert placement.chosen_cut == min(rows)[2]
    assert placement.chosen.marginal == placement.marginal
    assert placement.marginal.n2q > 0
    for c in placement.candidates_evaluated:
        assert c.weighted_marginal == pytest.approx(_weighted(c.marginal.as_tuple()))
    assert cache.compile_calls == 7


# --- complete patch, cache, signed marginals ---------------------------------------


def test_complete_patch_signed_marginals_and_base_compiled_once(backend) -> None:
    backend_obj, _ = backend
    state = _state((XX, ZZ, YY), np.array([0.3, -0.2, 0.5]))
    cache = cpc.CompiledPatchCostCache()
    common = dict(contract=CONTRACT, backend_obj=backend_obj, cache=cache)

    base = cpc.compile_base_word(state, **common)
    assert base.marginal == cpc.ZERO_MARGINAL
    assert base.compile_calls_used == 1 and not base.from_cache
    assert base.triple.n2q > 0
    assert cache.compile_calls == 1

    # The base word is compiled once: a second request is a hit.
    base_again = cpc.compile_base_word(state, **common)
    assert base_again.triple == base.triple and base_again.from_cache
    assert base_again.compile_calls_used == 0
    assert cache.compile_calls == 1 and cache.cache_hits == 1

    # Deletion-only patch: fewer entangling gates -> negative signed marginal.
    deletion = cpc.compile_complete_patch(
        state, removed_runtime_indices=(1,), insertions=(), base_triple=base.triple, **common
    )
    assert deletion.marginal.n2q < 0
    assert deletion.marginal.d2q < 0
    assert deletion.marginal.dc < 0
    assert deletion.triple.minus(base.triple) == deletion.marginal
    assert deletion.word_length == 2
    assert deletion.compile_calls_used == 1 and not deletion.from_cache
    assert deletion.removed_runtime_indices == (1,)

    # Insertion patch: extra entangling term -> positive signed marginal, and
    # the deletion-remap is recorded.
    insertion = cpc.compile_complete_patch(
        state,
        removed_runtime_indices=(),
        insertions=((3, XY, "xy@3"),),
        base_triple=base.triple,
        **common,
    )
    assert insertion.marginal.n2q > 0
    assert insertion.marginal.d2q > 0
    assert insertion.marginal.dc > 0
    assert insertion.word_length == 4
    assert insertion.insertion_cuts_original == (3,)
    assert insertion.insertion_cuts_remapped == (3,)
    assert cache.compile_calls == 3

    # Identical patch again: served from the cache, marginal unchanged.
    repeat = cpc.compile_complete_patch(
        state,
        removed_runtime_indices=(),
        insertions=((3, XY, "xy@3-relabeled"),),
        base_triple=None,
        **common,
    )
    assert repeat.from_cache and repeat.compile_calls_used == 0
    assert repeat.word_sha256 == insertion.word_sha256
    assert repeat.marginal == insertion.marginal
    assert cache.compile_calls == 3 and cache.cache_hits == 3
    assert len(cache) == 3
    assert cache.lookup(insertion.word_sha256) == insertion.triple
    payload = repeat.to_json_dict()
    assert payload["schema"] == "paper_ii_compiled_patch_cost_v1"
    assert payload["marginal"] == insertion.marginal.as_delta_dict()


def test_cache_keys_on_structural_angle(backend) -> None:
    backend_obj, _ = backend
    cache = cpc.CompiledPatchCostCache()
    kwargs = dict(nq=2, ref_state=None, backend_obj=backend_obj)
    first = cache.compile_word((XX, ZZ), contract=CONTRACT, **kwargs)
    assert cache.compile_word((XX, ZZ), contract=CONTRACT, **kwargs) == first
    assert cache.compile_calls == 1 and cache.cache_hits == 1
    other = cache.compile_word((XX, ZZ), contract=CompileContract(structural_angle=0.5), **kwargs)
    assert isinstance(other, cpc.CompiledResourceTriple)
    assert cache.compile_calls == 2 and cache.cache_hits == 1
    assert len(cache) == 2


# --- failure semantics --------------------------------------------------------------


def test_failure_raises_compiled_patch_cost_error_without_fallback(backend, monkeypatch) -> None:
    backend_obj, _ = backend
    state = _state((XX, ZZ, YY), np.array([0.3, -0.2, 0.5]))
    base_triple = cpc.CompiledResourceTriple(n2q=6, d2q=6, dc=32)
    assert issubclass(cpc.CompiledPatchCostError, ValueError)

    def boom(*args, **kwargs):
        raise RuntimeError("transpiler exploded")

    monkeypatch.setattr(backend_tools, "compile_circuit_for_backend", boom)
    cache = cpc.CompiledPatchCostCache()
    with pytest.raises(cpc.CompiledPatchCostError, match="transpiler exploded"):
        cpc.compile_complete_patch(
            state,
            removed_runtime_indices=(1,),
            insertions=(),
            contract=CONTRACT,
            backend_obj=backend_obj,
            cache=cache,
            base_triple=base_triple,
        )
    assert cache.compile_calls == 0 and len(cache) == 0
    with pytest.raises(cpc.CompiledPatchCostError):
        cpc.compile_base_word(state, contract=CONTRACT, backend_obj=backend_obj, cache=cache)
    with pytest.raises(cpc.CompiledPatchCostError):
        cpc.choose_placement(
            XY, (1, 2), base_terms=(XX, ZZ, YY), nq=2, ref_state=None,
            contract=CONTRACT, backend_obj=backend_obj, cache=cache,
        )
    assert cache.compile_calls == 0

    # Non-finite and missing resources are refused too (no transpile: the
    # compile step is stubbed to hand back the logical circuit).
    monkeypatch.setattr(
        backend_tools,
        "compile_circuit_for_backend",
        lambda circuit, backend, **kw: {"compiled": circuit},
    )
    monkeypatch.setattr(
        backend_tools,
        "compiled_gate_stats",
        lambda compiled: {"compiled_count_2q": float("nan"), "compiled_depth_2q": 1},
    )
    with pytest.raises(cpc.CompiledPatchCostError, match="not finite"):
        cpc.compile_base_word(state, contract=CONTRACT, backend_obj=backend_obj, cache=cache)
    monkeypatch.setattr(backend_tools, "compiled_gate_stats", lambda compiled: {})
    with pytest.raises(cpc.CompiledPatchCostError, match="missing"):
        cpc.compile_base_word(state, contract=CONTRACT, backend_obj=backend_obj, cache=cache)
    assert cache.compile_calls == 0 and len(cache) == 0

    # A word that cannot be built is refused before any compile.
    broken = AnsatzTerm(label="broken", polynomial=_poly(("xyz", 0.5), nq=3))
    with pytest.raises(cpc.CompiledPatchCostError):
        cpc.compile_complete_patch(
            state,
            removed_runtime_indices=(),
            insertions=((0, broken, "broken@0"),),
            contract=CONTRACT,
            backend_obj=backend_obj,
            cache=cache,
            base_triple=base_triple,
        )
    assert cache.compile_calls == 0


# --- backend provenance --------------------------------------------------------


def test_resolve_compile_backend_provenance(backend) -> None:
    backend_obj, provenance = backend
    assert backend_obj is not None
    assert "marrakesh" in provenance["backend_name"].lower()
    assert provenance["requested_backend_name"] == "FakeMarrakesh"
    assert provenance["using_fake_backend"] is True
    assert provenance["optimization_level"] == 1
    assert provenance["transpile_seed"] == 7
    assert provenance["structural_angle"] == 1.0
    assert provenance["coordinate_weights"] == dict(COMPILE_COORDINATE_WEIGHTS)
    assert provenance["qiskit_version"] == importlib_metadata.version("qiskit")
    assert "qiskit_ibm_runtime_version" in provenance
    assert provenance["placement_rule_id"] == cpc.PLACEMENT_RULE_ID
    assert provenance["target_snapshot"]["num_qubits"] > 2
    assert provenance["resolution_audit"][0]["success"] is True


def test_resolve_compile_backend_fails_closed_on_unknown_backend() -> None:
    with pytest.raises(cpc.CompiledPatchCostError, match="Could not resolve"):
        cpc.resolve_compile_backend(CompileContract(backend_name="FakeNoSuchDevice"))


def test_process_pool_warm_primes_the_cache_with_identical_triples() -> None:
    # A worker compiles under the same sealed contract, so the primed triple
    # must equal the in-process compile bit for bit, for both job kinds.
    from pipelines.time_dynamics.ap_mclachlan.compiled_patch_cost import (
        CompiledPatchCostCache,
        CompileWarmPool,
        base_word,
        edit_word,
        resolve_compile_backend,
        state_num_qubits,
        warm_word_cache,
        word_key,
    )
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import CompileContract

    x0 = AnsatzTerm(label="sx0", polynomial=_poly(("ex", 1.0)))
    z0 = AnsatzTerm(label="sz0", polynomial=_poly(("ez", 1.0)))
    y1 = AnsatzTerm(label="sy1", polynomial=_poly(("ye", 1.0)))
    ca = AnsatzTerm(label="ca", polynomial=_poly(("xe", 0.7)))
    cb = AnsatzTerm(label="cb", polynomial=_poly(("zz", 0.4)))
    state = _state((x0, z0, y1), np.array([0.3, -0.2, 0.5]))
    contract = CompileContract()
    backend, _prov = resolve_compile_backend(contract)
    nq = state_num_qubits(state)
    ref = np.asarray(state.psi_ref, dtype=complex).reshape(-1)
    base = tuple(base_word(state))
    atoms = {"ca": ca, "cb": cb}

    trial = base[:1] + (ca,) + base[1:]
    patch = edit_word(base, (0,), [(2, cb, "cb@2#0")])
    direct = CompiledPatchCostCache()
    expected_trial = direct.compile_word(trial, nq=nq, ref_state=ref, contract=contract, backend_obj=backend)
    expected_patch = direct.compile_word(patch, nq=nq, ref_state=ref, contract=contract, backend_obj=backend)

    pool = CompileWarmPool(contract=contract, base_terms=base, atom_terms=atoms, nq=nq, ref_state=ref, workers=2)
    try:
        primed_cache = CompiledPatchCostCache()
        jobs = [
            (word_key(trial, nq=nq, ref_state=ref, contract=contract), ("trial", "ca", 1)),
            (word_key(patch, nq=nq, ref_state=ref, contract=contract), ("patch", (0,), ((2, "cb", "cb@2#0"),))),
            (word_key(trial, nq=nq, ref_state=ref, contract=contract), ("trial", "ca", 1)),  # duplicate
        ]
        primed = warm_word_cache(primed_cache, pool, jobs)
    finally:
        pool.close()
    assert primed == 2 and primed_cache.compile_calls == 2
    key_t = word_key(trial, nq=nq, ref_state=ref, contract=contract)
    key_p = word_key(patch, nq=nq, ref_state=ref, contract=contract)
    assert primed_cache.lookup(key_t) == expected_trial
    assert primed_cache.lookup(key_p) == expected_patch
    # A primed word is a cache hit for the serial replay.
    _key, triple, hit = primed_cache.compile_word_record(trial, nq=nq, ref_state=ref, contract=contract, backend_obj=backend)
    assert hit and triple == expected_trial
    assert primed_cache.cache_hits == 1


@pytest.mark.parametrize("workers", [1, 2])
def test_compile_pool_timeout_is_bounded_and_kills_children(workers) -> None:
    state = _state((X0, ZZ, YY), np.array([0.3, -0.2, 0.5]))
    base = tuple(cpc.base_word(state))
    contract = CompileContract(compile_timeout_seconds=1.0e-6)
    before = {child.pid for child in multiprocessing.active_children()}
    started = time.monotonic()
    pool = cpc.CompileWarmPool(
        contract=contract,
        base_terms=base,
        atom_terms={},
        nq=cpc.state_num_qubits(state),
        ref_state=np.asarray(state.psi_ref, dtype=complex),
        workers=workers,
    )
    try:
        with pytest.raises(cpc.CompileTimeoutError, match="sealed timeout"):
            pool.compile_word(
                base,
                nq=cpc.state_num_qubits(state),
                ref_state=np.asarray(state.psi_ref, dtype=complex),
                contract=contract,
                backend_obj=None,
            )
    finally:
        pool.close()
    assert time.monotonic() - started < 5.0
    assert {child.pid for child in multiprocessing.active_children()} <= before


def test_compile_pool_forces_and_restores_blas_environment(monkeypatch) -> None:
    state = _state((X0, ZZ, YY), np.array([0.3, -0.2, 0.5]))
    base = tuple(cpc.base_word(state))
    prior = {name: str(index + 2) for index, name in enumerate(cpc.BLAS_THREAD_ENV_VARS)}
    for name, value in prior.items():
        monkeypatch.setenv(name, value)
    # rayon is pinned alongside BLAS so each worker's qiskit transpiler stays
    # single-threaded; an unset variable is removed again on close.
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    prior["RAYON_NUM_THREADS"] = None
    assert tuple(prior) == cpc.WORKER_THREAD_ENV_VARS
    pool = cpc.CompileWarmPool(
        contract=CompileContract(),
        base_terms=base,
        atom_terms={},
        nq=cpc.state_num_qubits(state),
        ref_state=np.asarray(state.psi_ref, dtype=complex),
        workers=1,
    )
    assert pool.prior_env == prior
    assert pool.pinned_env == {name: "1" for name in cpc.WORKER_THREAD_ENV_VARS}
    assert {name: os.environ[name] for name in cpc.WORKER_THREAD_ENV_VARS} == pool.pinned_env
    pool.close()
    assert {name: os.environ.get(name) for name in cpc.WORKER_THREAD_ENV_VARS} == prior


def test_compile_pool_fails_fast_from_python_c_entrypoint() -> None:
    import subprocess
    import sys

    code = """
from pipelines.time_dynamics.ap_mclachlan.compiled_patch_cost import CompileWarmPool, CompiledPatchCostError
from pipelines.time_dynamics.ap_mclachlan.exchange_config import CompileContract
try:
    CompileWarmPool(contract=CompileContract(), base_terms=(), atom_terms={}, nq=1, ref_state=[1, 0], workers=1)
except CompiledPatchCostError as exc:
    print(str(exc))
else:
    raise SystemExit(9)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    assert "require an importable file/module entrypoint" in completed.stdout

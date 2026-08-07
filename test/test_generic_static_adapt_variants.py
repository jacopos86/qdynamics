#!/usr/bin/env python3
"""Tests for benchmark-local generic static ADAPT variants."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.exact_bench import generic_static_adapt_variants as variants
from pipelines.exact_bench.static_prefix_runtime_seed_export import (
    _candidate_batches_with_execution_mode,
    _selected_batches_from_history,
)
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
from pipelines.exact_bench.table_i_canonical_cases import (
    TABLE_I_CANONICAL_CASE_IDS_BY_FAMILY,
    TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS,
    TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE,
    table_i_canonical_spec_by_case_id,
)
from pipelines.exact_bench.table_i_qiskit_resource_compile import (
    TABLE_I_GROUPED_EXACT_SYNTHESIS_ID,
    TableIQiskitCompileConfig,
    _transpile_table_i_circuit,
    build_table_i_execution_aware_circuit,
)
from pipelines.static_adapt.builders.shared_pauli_pool_contract import (
    SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
    SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1,
    SharedPauliPoolParent,
    build_shared_pauli_child_pool,
)
from src.quantum.ansatz_parameterization import expand_legacy_logical_theta
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


class _TinyLayout:
    total_qubits = 2
    fermion_qubits = 2

    def block(self, name: str):  # noqa: ANN201 - tiny fake layout
        if name == "fermion":
            return SimpleNamespace(start_qubit=0, stop_qubit=2)
        return None


def _fake_spec() -> SimpleNamespace:
    return SimpleNamespace(
        benchmark_id="hubbard_L2",
        family="hubbard",
        base_pipeline_args=("--problem", "hubbard", "--L", "2"),
        split="train",
        tags=(),
        features=None,
    )


def _fake_context(events: list[str]) -> SimpleNamespace:
    # Off-diagonal two-qubit Hamiltonian gives the pairwise qubit-excitation
    # pool a nonzero benchmark-local direction without involving exact targets.
    hamiltonian = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="xx", pc=0.5),
            PauliTerm(2, ps="yy", pc=0.5),
        ],
    )

    def _resolve_energy(ai_log=None):  # noqa: ANN001
        assert events == ["optimizer"]
        events.append("exact")
        return -1.0

    return SimpleNamespace(
        request=SimpleNamespace(num_sites=2, ordering="blocked"),
        layout=_TinyLayout(),
        hamiltonian=hamiltonian,
        reference_state=SimpleNamespace(build_state=lambda: np.eye(4, dtype=complex)[1]),
        exact_target=SimpleNamespace(resolve_energy=_resolve_energy),
        sector=SimpleNamespace(constraints=()),
    )


def _record_fake_powell_h_calls(kwargs: dict, theta: np.ndarray) -> None:
    recorder = kwargs.get("estimator_h_call_recorder")
    if recorder is None:
        return
    prepared = variants._prepare_selected_state(
        selected=kwargs["selected"],
        theta=np.asarray(theta, dtype=float),
        psi_ref=np.asarray(kwargs["psi_ref"], dtype=complex),
        pauli_action_cache=kwargs["pauli_action_cache"],
        parameterization_mode=str(kwargs.get("parameterization_mode", "logical_shared")),
        parameterization_layout=kwargs.get("parameterization_layout"),
    )
    recorder(prepared, "adapt_refit_powell_objective:objective")


def test_hh_static_variant_runtime_seed_payload_is_replay_shaped(tmp_path: Path) -> None:
    config = variants._get_config("static_geo_adapt_vqe")
    context = SimpleNamespace(
        request=SimpleNamespace(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.25,
            dv=0.0,
            omega0=1.0,
            g_ep=0.25,
            n_ph_max=2,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            n_fermions=2,
        ),
        layout=SimpleNamespace(total_qubits=1),
    )
    candidate = SimpleNamespace(
        label="paop_disp(0)",
        support=(0,),
        pauli_labels_exyz=("x", "y"),
        polynomial=PauliPolynomial(
            "JW",
            [PauliTerm(1, ps="x", pc=0.5), PauliTerm(1, ps="y", pc=-0.5)],
        ),
        execution_mode="termwise_product",
    )
    row = {
        "schema": variants.SCHEMA_VERSION,
        "method_label": "Geo-ADAPT-VQE",
        "pool_name": "full_meta",
        "required_pool_key": "full_meta",
        "energy": -1.0,
        "exact_energy": -1.0,
        "abs_delta_e": 0.0,
        "adapt_stop_reason": "unit",
        "position_policy": "append",
        "position_optimized_geo_adapt": False,
    }

    runtime_seed = variants._build_runtime_seed_payload(
        context=context,
        family="hh",
        case_id="hh_L2_nph2_three_model_sym_weak_weak",
        config=config,
        selected=(candidate,),
        selected_batches=((candidate,),),
        theta=np.asarray([0.125], dtype=float),
        psi_ref=np.asarray([1.0, 0.0], dtype=complex),
        psi_final=np.asarray([0.992197667, 0.124674733], dtype=complex),
        row=row,
        spec=_fake_spec(),
        generated_utc="2026-05-28T00:00:00+00:00",
    )

    assert runtime_seed["schema"] == "paper_ii_static_seed_runtime_payload_v1"
    assert runtime_seed["settings"]["problem"] == "hh"
    assert runtime_seed["settings"]["u"] == 0.25
    assert runtime_seed["settings"]["g_ep"] == 0.25
    assert runtime_seed["settings"]["n_ph_max"] == 2
    assert runtime_seed["adapt_vqe"]["algorithm_id"] == "static_geo_adapt_vqe"
    assert runtime_seed["adapt_vqe"]["operators"] == ["paop_disp(0)"]
    assert runtime_seed["adapt_vqe"]["optimal_point"] == [0.125]
    assert runtime_seed["initial_state"]["handoff_state_kind"] == "prepared_state"
    assert runtime_seed["ansatz_input_state"]["handoff_state_kind"] == "reference_state"
    assert runtime_seed["paper_ii_static_seed_export"]["runtime_loadability_status"] == (
        "runtime_seed_sidecar_written_not_dry_loaded"
    )

    payload = variants._write_artifacts(
        tmp_path,
        {"schema": variants.SCHEMA_VERSION, "algorithm_id": "static_geo_adapt_vqe"},
        [row],
        runtime_seed_payload=runtime_seed,
    )
    assert (tmp_path / "runtime_seed.json").exists()
    assert payload["runtime_seed_json"] == str(tmp_path / "runtime_seed.json")
    written = json.loads((tmp_path / "runtime_seed.json").read_text(encoding="utf-8"))
    assert written["adapt_vqe"]["method_label"] == "Geo-ADAPT-VQE"


def test_runtime_seed_logical_shared_does_not_emit_per_pauli_layout() -> None:
    config = variants._get_config("static_full_meta_append_adapt_vqe")
    context = SimpleNamespace(
        request=SimpleNamespace(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.25,
            dv=0.0,
            omega0=1.0,
            g_ep=0.25,
            n_ph_max=2,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            n_fermions=2,
        ),
        layout=SimpleNamespace(total_qubits=2),
    )
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="ex", pc=0.5),
            PauliTerm(2, ps="ey", pc=-0.25),
        ],
    )
    candidate = AnsatzTerm(label="macro_two_paulis", polynomial=poly)
    row = {
        "schema": variants.SCHEMA_VERSION,
        "method_label": "Append-ADAPT",
        "pool_name": "full_meta",
        "required_pool_key": "full_meta",
        "energy": -1.0,
        "exact_energy": -1.0,
        "abs_delta_e": 0.0,
        "adapt_stop_reason": "unit",
        "position_policy": "append",
        "position_optimized_geo_adapt": False,
    }

    runtime_seed = variants._build_runtime_seed_payload(
        context=context,
        family="hh",
        case_id="hh_L2_nph2_three_model_sym_weak_weak",
        config=config,
        selected=(candidate,),
        selected_batches=((candidate,),),
        theta=np.asarray([0.125], dtype=float),
        psi_ref=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=complex),
        psi_final=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=complex),
        row=row,
        spec=_fake_spec(),
        generated_utc="2026-05-28T00:00:00+00:00",
    )

    assert runtime_seed["adapt_vqe"]["parameterization_mode"] == "logical_shared"
    assert runtime_seed["adapt_vqe"]["parameterization"] is None
    assert runtime_seed["adapt_vqe"]["optimal_point"] == [0.125]
    assert runtime_seed["adapt_vqe"]["logical_optimal_point"] == [0.125]
    assert runtime_seed["adapt_vqe"]["selected_operator_execution_modes"] == ["termwise_product"]
    assert runtime_seed["adapt_vqe"]["selected_operator_pauli_terms"][0]
    assert {"pauli_exyz", "coeff_re", "coeff_im"} <= set(
        runtime_seed["adapt_vqe"]["selected_operator_pauli_terms"][0][0]
    )
    assert len(runtime_seed["adapt_vqe"]["selected_generator_semantics_sha256"]) == 64


def test_static_prefix_runtime_seed_export_extracts_exact_prefix_batches() -> None:
    payload = {
        "result": {
            "adapt_history": [
                {"selected_batch_labels": ["a"], "energy_after": -0.1},
                {"selected_batch_labels": ["b", "c"], "energy_after": -0.2},
                {"selected_batch_labels": ["d"], "energy_after": -0.3},
            ],
        },
    }

    batches, row = _selected_batches_from_history(payload, prefix_depth=3)

    assert batches == [["a"], ["b", "c"]]
    assert row["energy_after"] == -0.2
    with pytest.raises(ValueError, match="cuts through an adaptive batch"):
        _selected_batches_from_history(payload, prefix_depth=2)


def test_static_prefix_runtime_seed_export_can_preserve_source_execution_mode() -> None:
    candidate = variants._PoolCandidate(
        label="a",
        polynomial=object(),
        support=(),
        pauli_labels_exyz=(),
        construction="test",
        execution_mode="grouped_exact",
    )

    current = _candidate_batches_with_execution_mode([[candidate]], "current")
    termwise = _candidate_batches_with_execution_mode([[candidate]], "termwise_product")

    assert current[0][0] is candidate
    assert termwise[0][0].label == "a"
    assert termwise[0][0].execution_mode == "termwise_product"
    assert candidate.execution_mode == "grouped_exact"


def test_default_static_adapt_variant_case_ids_cover_table_i_canonical_suite() -> None:
    for algorithm_id in variants.GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS:
        for family, case_ids in TABLE_I_CANONICAL_CASE_IDS_BY_FAMILY.items():
            assert variants.default_static_adapt_variant_case_ids(family, algorithm_id) == tuple(case_ids)


def test_reduced_full_meta_pool_uses_selected_logical_filter(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    fake_filter = object()

    def fake_resolve_requested_pool_filters(**kwargs):  # noqa: ANN003, ANN202
        captured["filter_kwargs"] = kwargs
        return fake_filter

    def fake_resolve_pool_plan(**kwargs):  # noqa: ANN003, ANN202
        captured["plan_kwargs"] = kwargs
        return SimpleNamespace(
            pool=[SimpleNamespace(label="full_meta::x_0", polynomial=object())],
            pool_key="full_meta",
            selected_logical_filter_meta={
                "applied": True,
                "pool_size_before": 5,
                "pool_size_after": 1,
                "matched_count": 1,
            },
        )

    monkeypatch.setattr(variants, "resolve_requested_pool_filters", fake_resolve_requested_pool_filters)
    monkeypatch.setattr(variants, "resolve_pool_plan", fake_resolve_pool_plan)
    monkeypatch.setattr(variants, "_polynomial_labels_and_support", lambda polynomial: (("x",), (0,)))

    context = SimpleNamespace(
        family_key="harmonic_kerr_chain",
        request=SimpleNamespace(num_sites=2, n_ph_max=2),
    )
    source = tmp_path / "selected.json"
    result = variants._build_reduced_full_meta_candidate_pool_with_meta(
        context,
        max_terms=256,
        selected_logical_source_json=source,
        selected_logical_mode="family_closure_with_full_fallback",
        selected_logical_transfer_mode="boundary_v1",
    )

    assert captured["filter_kwargs"]["adapt_selected_logical_source_json"] == source
    assert captured["filter_kwargs"]["adapt_selected_logical_mode"] == "family_closure_with_full_fallback"
    assert captured["filter_kwargs"]["adapt_selected_logical_transfer_mode"] == "boundary_v1"
    assert captured["plan_kwargs"]["filter_resolution"] is fake_filter
    assert result.selected_logical_filter_meta["applied"] is True
    assert [candidate.label for candidate in result.candidates] == ["full_meta::x_0"]


def test_hh_full_meta_minus_hva_policy_only_applies_to_active_geo_and_append() -> None:
    hh_context = SimpleNamespace(
        family_key="hh",
        request=SimpleNamespace(problem_key="hh", num_sites=2, n_ph_max=2),
    )
    non_hh_context = SimpleNamespace(
        family_key="hubbard",
        request=SimpleNamespace(problem_key="hubbard", num_sites=2, n_ph_max=0),
    )

    for algorithm_id in ("static_geo_adapt_vqe", "static_full_meta_append_adapt_vqe"):
        config = variants._get_config(algorithm_id)
        assert variants._active_hh_full_meta_minus_hva_class_filter_json(
            config=config,
            context=hh_context,
        ) == variants._HH_FULL_META_MINUS_HVA_CLASS_FILTER_JSON
        assert variants._active_hh_full_meta_minus_hva_class_filter_json(
            config=config,
            context=non_hh_context,
        ) is None

    legacy_or_non_active = (
        "static_qubit_qeb_adapt_vqe",
        "static_geo_qubit_adapt_vqe",
        "static_geo_qeb_adapt_vqe",
        "static_pos_geo_adapt_vqe",
    )
    for algorithm_id in legacy_or_non_active:
        assert variants._active_hh_full_meta_minus_hva_class_filter_json(
            config=variants._get_config(algorithm_id),
            context=hh_context,
        ) is None


def test_hh_full_meta_pool_profile_can_force_unfiltered_for_active_geo_append() -> None:
    hh_context = SimpleNamespace(
        family_key="hh",
        request=SimpleNamespace(problem_key="hh", num_sites=2, n_ph_max=2),
    )
    for algorithm_id in ("static_geo_adapt_vqe", "static_full_meta_append_adapt_vqe"):
        profile, class_filter = variants._resolve_hh_full_meta_pool_profile(
            config=variants._get_config(algorithm_id),
            context=hh_context,
            hh_adaptive_pool_profile="full_meta_unfiltered",
        )
        assert profile == "full_meta_unfiltered"
        assert class_filter is None

        profile, class_filter = variants._resolve_hh_full_meta_pool_profile(
            config=variants._get_config(algorithm_id),
            context=hh_context,
            hh_full_meta_class_filter_json="off",
        )
        assert profile == "full_meta_unfiltered"
        assert class_filter is None


def test_hh_append_geo_runtime_split_expands_parent_pool_with_pauli_child_sets() -> None:
    parent = variants._PoolCandidate(
        label="full_meta::pair_hop",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(4, ps="zeee", pc=0.25),
                PauliTerm(4, ps="eexy", pc=0.5),
                PauliTerm(4, ps="eeyx", pc=-0.5),
            ],
        ),
        support=(0, 1),
        pauli_labels_exyz=("zeee", "eexy", "eeyx"),
        construction="full_meta::full_meta",
    )
    context = SimpleNamespace(
        family_key="hh",
        request=SimpleNamespace(problem_key="hh", num_sites=2, ordering="blocked"),
        layout=SimpleNamespace(total_qubits=4, fermion_qubits=4),
    )

    expanded, meta = variants._expand_pool_with_runtime_split_children(
        pool=(parent,),
        context=context,
        config=variants._get_config("static_full_meta_append_adapt_vqe"),
        split_mode="shortlist_pauli_children_v1",
        symmetry_policy="hard_guard",
        max_subset_size=3,
        max_terms=8,
    )

    child_candidates = [candidate for candidate in expanded if candidate.runtime_split_mode != "off"]
    assert [candidate.label for candidate in expanded][:1] == ["full_meta::pair_hop"]
    assert len(child_candidates) == 2
    assert meta["supports_only"] == "paper_i_hh_or_hubbard_full_meta_append_and_geo"
    assert meta["base_pool_term_count"] == 1
    assert meta["expanded_pool_term_count"] == 3
    assert meta["added_child_set_count"] == 2
    assert {candidate.parent_label for candidate in child_candidates} == {"full_meta::pair_hop"}
    assert {candidate.runtime_split_representation for candidate in child_candidates} == {"child_set"}
    assert {tuple(candidate.runtime_split_child_indices) for candidate in child_candidates} == {(2,), (0, 1)}
    assert all(candidate.runtime_split_symmetry_gate["checked"] is True for candidate in child_candidates)
    assert all(candidate.runtime_split_symmetry_gate["passed"] is True for candidate in child_candidates)
    by_indices = {tuple(candidate.runtime_split_child_indices): candidate for candidate in child_candidates}
    assert by_indices[(2,)].execution_mode == "termwise_product"
    assert by_indices[(0, 1)].execution_mode == "grouped_exact"
    ansatz_terms = {term.label: term for term in variants._ansatz_terms_from_candidates(child_candidates)}
    assert ansatz_terms[by_indices[(0, 1)].label].execution_mode == "grouped_exact"


def test_hh_append_geo_runtime_split_off_does_not_reduce_subset_cardinality() -> None:
    parent = variants._PoolCandidate(
        label="full_meta::pair_hop",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(4, ps="zeee", pc=0.25),
                PauliTerm(4, ps="eexy", pc=0.5),
                PauliTerm(4, ps="eeyx", pc=-0.5),
            ],
        ),
        support=(0, 1),
        pauli_labels_exyz=("zeee", "eexy", "eeyx"),
        construction="full_meta::full_meta",
    )
    context = SimpleNamespace(
        family_key="hh",
        request=SimpleNamespace(problem_key="hh", num_sites=2, ordering="blocked"),
        layout=SimpleNamespace(total_qubits=4, fermion_qubits=4),
    )

    expanded, meta = variants._expand_pool_with_runtime_split_children(
        pool=(parent,),
        context=context,
        config=variants._get_config("static_full_meta_append_adapt_vqe"),
        split_mode="shortlist_pauli_children_v1",
        symmetry_policy="off",
        max_subset_size=3,
        max_terms=8,
    )

    child_candidates = [candidate for candidate in expanded if candidate.runtime_split_mode != "off"]
    assert meta["symmetry_policy"] == "off"
    assert meta["symmetry_gate_enforced"] is False
    assert meta["symmetry_checked_child_atom_count"] == 0
    assert meta["symmetry_rejected_child_atom_count"] == 0
    assert meta["symmetry_checked_child_set_count"] == 0
    assert meta["added_child_set_count"] == 6
    assert {tuple(candidate.runtime_split_child_indices) for candidate in child_candidates} == {
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
    }
    assert all(candidate.runtime_split_symmetry_gate["checked"] is False for candidate in child_candidates)
    assert all(candidate.runtime_split_symmetry_gate["passed"] is True for candidate in child_candidates)


def test_hh_append_geo_shared_pauli_pool_matches_direct_contract_hash() -> None:
    parent = variants._PoolCandidate(
        label="full_meta::pair_hop",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(4, ps="zeee", pc=0.25),
                PauliTerm(4, ps="eexy", pc=0.5),
                PauliTerm(4, ps="eeyx", pc=-0.5),
            ],
        ),
        support=(0, 1),
        pauli_labels_exyz=("zeee", "eexy", "eeyx"),
        construction="full_meta::full_meta",
    )
    context = SimpleNamespace(
        family_key="hh",
        request=SimpleNamespace(problem_key="hh", num_sites=2, ordering="blocked"),
        layout=SimpleNamespace(total_qubits=4, fermion_qubits=4),
    )

    expanded, meta = variants._expand_pool_with_shared_pauli_children(
        pool=(parent,),
        context=context,
        config=variants._get_config("static_full_meta_append_adapt_vqe"),
        mode=SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
        symmetry_policy="hard_guard",
        max_subset_size=3,
        max_terms=8,
    )
    direct = build_shared_pauli_child_pool(
        parents=[
            SharedPauliPoolParent(
                label=str(parent.label),
                polynomial=parent.polynomial,
                family_id="hh",
                stage_family=str(parent.construction),
                construction=str(parent.construction),
                execution_mode=str(parent.execution_mode),
                generator_metadata=variants._candidate_parent_generator_metadata(parent),
            )
        ],
        mode=SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
        symmetry_policy="hard_guard",
        max_subset_size=3,
        problem_key="hh",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        max_terms=8,
    )

    assert [candidate.label for candidate in expanded] == [candidate.label for candidate in direct.candidates]
    assert meta["ordered_pool_hash"] == direct.meta["ordered_pool_hash"]
    assert meta["ordered_label_hash"] == direct.meta["ordered_label_hash"]
    assert meta["symmetry_policy"] == "hard_guard"
    assert meta["symmetry_gate_enforced"] is True
    assert meta["explicit_no_guard"] is False
    assert meta["contract_identity"]["symmetry_policy"] == "hard_guard"
    assert meta["pool_policy"] == "same_ordered_parent_plus_child_set_pool_for_snake_geo_append"
    assert [candidate.runtime_split_representation for candidate in expanded] == ["parent", "child_set", "child_set"]


def test_hh_append_geo_shared_pauli_pool_accepts_explicit_no_guard() -> None:
    parent = variants._PoolCandidate(
        label="full_meta::pair_hop",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(4, ps="zeee", pc=0.25),
                PauliTerm(4, ps="eexy", pc=0.5),
                PauliTerm(4, ps="eeyx", pc=-0.5),
            ],
        ),
        support=(0, 1),
        pauli_labels_exyz=("zeee", "eexy", "eeyx"),
        construction="full_meta::full_meta",
    )
    context = SimpleNamespace(
        family_key="hh",
        request=SimpleNamespace(problem_key="hh", num_sites=2, ordering="blocked"),
        layout=SimpleNamespace(total_qubits=4, fermion_qubits=4),
    )

    hard, hard_meta = variants._expand_pool_with_shared_pauli_children(
        pool=(parent,),
        context=context,
        config=variants._get_config("static_geo_adapt_vqe"),
        mode=SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
        symmetry_policy="hard_guard",
        max_subset_size=3,
        max_terms=8,
    )
    off, off_meta = variants._expand_pool_with_shared_pauli_children(
        pool=(parent,),
        context=context,
        config=variants._get_config("static_geo_adapt_vqe"),
        mode=SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
        symmetry_policy="off",
        max_subset_size=3,
        max_terms=8,
    )

    hard_labels = [candidate.label for candidate in hard]
    off_labels = [candidate.label for candidate in off]
    assert off_meta["symmetry_policy"] == "off"
    assert off_meta["symmetry_gate_enforced"] is False
    assert off_meta["explicit_no_guard"] is True
    assert off_meta["contract_identity"]["symmetry_policy"] == "off"
    assert off_meta["contract_identity"]["max_subset_size"] == 3
    assert off_meta["ordered_pool_hash"] != hard_meta["ordered_pool_hash"]
    assert set(off_labels) - set(hard_labels)
    assert "full_meta::pair_hop::child_set[0]" in off_labels
    child = next(candidate for candidate in off if candidate.runtime_split_representation == "child_set")
    assert child.runtime_split_symmetry_gate["checked"] is False


def test_hh_runtime_split_rejects_non_append_geo_comparators() -> None:
    parent = variants._PoolCandidate(
        label="full_meta::pair_hop",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(4, ps="zeee", pc=0.25),
                PauliTerm(4, ps="eexy", pc=0.5),
                PauliTerm(4, ps="eeyx", pc=-0.5),
            ],
        ),
        support=(0, 1),
        pauli_labels_exyz=("zeee", "eexy", "eeyx"),
        construction="full_meta::full_meta",
    )
    context = SimpleNamespace(
        family_key="hh",
        request=SimpleNamespace(problem_key="hh", num_sites=2, ordering="blocked"),
        layout=SimpleNamespace(total_qubits=4, fermion_qubits=4),
    )

    with pytest.raises(ValueError, match="Paper-I HH/Hubbard full_meta append and Geo"):
        variants._expand_pool_with_runtime_split_children(
            pool=(parent,),
            context=context,
            config=variants._get_config("static_qubit_qeb_adapt_vqe"),
            split_mode="shortlist_pauli_children_v1",
            symmetry_policy="off",
            max_subset_size=3,
            max_terms=8,
        )


def test_hubbard_runtime_split_expands_full_meta_append_geo_children() -> None:
    parent = variants._PoolCandidate(
        label="full_meta::pair_hop",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(4, ps="zeee", pc=0.25),
                PauliTerm(4, ps="eexy", pc=0.5),
                PauliTerm(4, ps="eeyx", pc=-0.5),
            ],
        ),
        support=(0, 1),
        pauli_labels_exyz=("zeee", "eexy", "eeyx"),
        construction="full_meta::full_meta",
    )
    context = SimpleNamespace(
        family_key="hubbard",
        request=SimpleNamespace(problem_key="hubbard", num_sites=2, ordering="blocked"),
        layout=SimpleNamespace(total_qubits=4, fermion_qubits=4),
    )

    expanded, meta = variants._expand_pool_with_runtime_split_children(
        pool=(parent,),
        context=context,
        config=variants._get_config("static_geo_adapt_vqe"),
        split_mode="shortlist_pauli_children_v1",
        symmetry_policy="hard_guard",
        max_subset_size=3,
        max_terms=8,
    )

    child_candidates = [candidate for candidate in expanded if candidate.runtime_split_mode != "off"]
    assert len(child_candidates) == 2
    assert meta["base_pool_term_count"] == 1
    assert meta["expanded_pool_term_count"] == 3
    assert {
        candidate.generator_metadata["family_id"]
        for candidate in child_candidates
        if isinstance(candidate.generator_metadata, dict)
    } == {"hubbard"}
    assert any(candidate.execution_mode == "grouped_exact" for candidate in child_candidates)


def test_hh_full_meta_comparator_pool_passes_minus_hva_class_filter(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_filter = object()

    def fake_resolve_requested_pool_filters(**kwargs):  # noqa: ANN003, ANN202
        captured["filter_kwargs"] = kwargs
        return fake_filter

    def fake_resolve_pool_plan(**kwargs):  # noqa: ANN003, ANN202
        captured["plan_kwargs"] = kwargs
        kwargs["ai_log"](
            "hardcoded_adapt_pool_cache_hit",
            cache_mode="disk",
            cache_scope="paper_i_holstein_sector",
        )
        return SimpleNamespace(
            pool=[
                SimpleNamespace(
                    label="hh_fermionic_reusable::hop",
                    polynomial=object(),
                    execution_mode="grouped_exact",
                )
            ],
            pool_key="full_meta",
            selected_logical_filter_meta=None,
            full_meta_class_filter_meta={
                "classifier_version": "hh_full_meta_v4",
                "keep_classes": ["hh_fermionic_reusable"],
                "class_counts_before": {"hva_layer": 3, "hh_fermionic_reusable": 1},
                "class_counts_after": {"hh_fermionic_reusable": 1},
                "dropped_classes": ["hva_layer"],
                "prebuild_skipped_classes": ["hva_layer"],
            },
            full_meta_label_filter_meta=None,
            pool_legal_subspace_filter_meta={"status": "not_requested"},
        )

    monkeypatch.setattr(variants, "resolve_requested_pool_filters", fake_resolve_requested_pool_filters)
    monkeypatch.setattr(variants, "resolve_pool_plan", fake_resolve_pool_plan)
    monkeypatch.setattr(variants, "_polynomial_labels_and_support", lambda polynomial: (("x",), (0,)))

    context = SimpleNamespace(
        family_key="hh",
        request=SimpleNamespace(problem_key="hh", num_sites=2, n_ph_max=2),
    )
    config = variants._get_config("static_geo_adapt_vqe")
    class_filter_json = variants._active_hh_full_meta_minus_hva_class_filter_json(
        config=config,
        context=context,
    )
    result = variants._build_full_meta_candidate_pool_with_meta(
        context,
        max_terms=256,
        hh_full_meta_class_filter_json=class_filter_json,
    )

    assert (
        Path(captured["filter_kwargs"]["adapt_pool_class_filter_json"])
        == variants._HH_FULL_META_MINUS_HVA_CLASS_FILTER_JSON
    )
    assert captured["filter_kwargs"]["adapt_pool"] == "full_meta"
    assert captured["plan_kwargs"]["filter_resolution"] is fake_filter
    assert [candidate.label for candidate in result.candidates] == ["hh_fermionic_reusable::hop"]
    assert result.candidates[0].execution_mode == "grouped_exact"
    assert result.full_meta_class_filter_meta["classifier_version"] == "hh_full_meta_v4"
    assert "hva_layer" in result.full_meta_class_filter_meta["dropped_classes"]
    assert "hva_layer" not in result.full_meta_class_filter_meta["class_counts_after"]
    assert result.pool_cache_events == (
        {
            "event": "hardcoded_adapt_pool_cache_hit",
            "cache_mode": "disk",
            "cache_scope": "paper_i_holstein_sector",
        },
    )


def test_pairwise_qubit_excitation_pool_uses_exyz_qubit0_lsb_convention() -> None:
    pool = variants.build_pairwise_qubit_excitation_pool(3)

    assert len(pool) == 3
    first = pool[0]
    assert first.label == "qeb_pair(0,1)"
    assert first.support == (0, 1)
    # q0 is rightmost: X on q0 and Y on q1 is encoded as e-y-x.
    assert first.pauli_labels_exyz == ("eyx", "exy")
    coeffs = [(term.pw2strng(), term.p_coeff) for term in first.polynomial.return_polynomial()]
    assert coeffs == [("exy", -0.5), ("eyx", 0.5)] or coeffs == [("eyx", 0.5), ("exy", -0.5)]


def test_full_meta_candidate_pool_uses_static_pool_builders_for_geo() -> None:
    spec = variants._spec_by_case_id("hubbard", "hubbard_L2", "static_geo_qubit_adapt_vqe")
    context = variants._resolve_context_from_spec(spec)

    pool = variants.build_full_meta_candidate_pool(context)

    assert pool
    assert any(candidate.label == "ham_full" for candidate in pool)
    assert any(str(candidate.label).startswith("uccsd_") for candidate in pool)
    assert all(candidate.pauli_labels_exyz for candidate in pool)
    assert all(candidate.support for candidate in pool)


def test_qiskit_cost_compile_synthesizes_commuting_grouped_exact_generator() -> None:
    candidate = variants._PoolCandidate(
        label="grouped_macro",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(2, ps="xy", pc=0.5),
                PauliTerm(2, ps="yx", pc=-0.5),
            ],
        ),
        support=(0, 1),
        pauli_labels_exyz=("xy", "yx"),
        construction="test",
        execution_mode="grouped_exact",
    )

    stats = variants._qiskit_compiled_stats_for_selected(
        selected=(candidate,),
        selected_batches=((candidate,),),
        config=variants._get_config("static_full_meta_append_adapt_vqe"),
        num_qubits=2,
        reference_state=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=complex),
        source_kind="qiskit_compiled_final_ansatz_circuit",
    )

    assert stats["compiled_circuit_stats_status"] == "ok"
    assert stats["compiled_resource_qiskit_validated"] is True
    assert stats["grouped_exact_synthesis_id"] == TABLE_I_GROUPED_EXACT_SYNTHESIS_ID
    assert stats["operator_synthesis"][0]["synthesis"] == "commuting_pauli_rotations_exact"
    assert stats["compiled_grouped_exact_operator_labels"] == [candidate.label]


def test_qiskit_grouped_exact_active_support_unitary_matches_executor() -> None:
    qiskit = pytest.importorskip("qiskit")
    candidate = variants._PoolCandidate(
        label="noncommuting_grouped_macro",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(1, ps="x", pc=0.5),
                PauliTerm(1, ps="z", pc=-0.25),
            ],
        ),
        support=(0,),
        pauli_labels_exyz=("x", "z"),
        construction="test",
        execution_mode="grouped_exact",
    )
    terms = variants._ansatz_terms_from_candidates((candidate,))
    reference = np.asarray([1.0, 0.0], dtype=complex)

    circuit, synthesis = build_table_i_execution_aware_circuit(
        ops=terms,
        num_qubits=1,
        reference_state=reference,
    )
    actual = np.asarray(qiskit.quantum_info.Statevector.from_instruction(circuit).data, dtype=complex)
    expected = CompiledAnsatzExecutor(terms).prepare_state(np.asarray([1.0]), reference)

    assert synthesis["operator_synthesis"][0]["synthesis"] == "active_support_unitary_exact"
    assert abs(np.vdot(expected, actual)) == pytest.approx(1.0, abs=1.0e-12)


def test_qiskit_grouped_exact_noncontiguous_support_matches_executor_after_transpile() -> None:
    qiskit = pytest.importorskip("qiskit")
    theta = 0.371
    candidate = variants._PoolCandidate(
        label="noncontiguous_noncommuting_grouped_macro",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(3, ps="xex", pc=0.375),
                PauliTerm(3, ps="zex", pc=-0.625),
            ],
        ),
        support=(0, 2),
        pauli_labels_exyz=("xex", "zex"),
        construction="test",
        execution_mode="grouped_exact",
    )
    terms = variants._ansatz_terms_from_candidates((candidate,))
    reference = np.asarray(
        [1.0, 0.25j, -0.5, 0.75j, 0.125, -0.375j, 0.625, 0.5j],
        dtype=complex,
    )
    reference = reference / np.linalg.norm(reference)
    compile_config = TableIQiskitCompileConfig(structure_theta_value=theta)

    circuit, synthesis = build_table_i_execution_aware_circuit(
        ops=terms,
        num_qubits=3,
        reference_state=reference,
        config=compile_config,
    )
    compiled = _transpile_table_i_circuit(circuit, config=compile_config)
    actual = np.asarray(qiskit.quantum_info.Statevector.from_instruction(compiled).data, dtype=complex)
    expected = CompiledAnsatzExecutor(terms).prepare_state(np.asarray([theta]), reference)
    overlap = complex(np.vdot(expected, actual))

    operator_synthesis = synthesis["operator_synthesis"][0]
    assert operator_synthesis["synthesis"] == "active_support_unitary_exact"
    assert operator_synthesis["commuting"] is False
    assert operator_synthesis["active_qubits"] == [0, 2]
    assert operator_synthesis["active_support_width"] == 2
    assert abs(overlap) == pytest.approx(1.0, abs=1.0e-10)
    np.testing.assert_allclose(actual * np.exp(-1.0j * np.angle(overlap)), expected, atol=1.0e-10)


def test_qiskit_grouped_exact_default_limit_covers_nph4_five_qubit_parent_support() -> None:
    candidate = variants._PoolCandidate(
        label="nph4_width_five_noncommuting_parent",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(5, ps="xxxxx", pc=0.5),
                PauliTerm(5, ps="zxxxx", pc=-0.25),
            ],
        ),
        support=(0, 1, 2, 3, 4),
        pauli_labels_exyz=("xxxxx", "zxxxx"),
        construction="test",
        execution_mode="grouped_exact",
    )
    terms = variants._ansatz_terms_from_candidates((candidate,))
    config = TableIQiskitCompileConfig()

    _circuit, synthesis = build_table_i_execution_aware_circuit(
        ops=terms,
        num_qubits=5,
        reference_state=None,
        config=config,
    )

    operator_synthesis = synthesis["operator_synthesis"][0]
    assert config.grouped_exact_max_active_qubits == 5
    assert operator_synthesis["synthesis"] == "active_support_unitary_exact"
    assert operator_synthesis["active_support_width"] == 5


def test_hh_pos_geo_position_policy_env_can_force_append(monkeypatch) -> None:
    config = variants._get_config("static_pos_geo_adapt_vqe")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_POS_GEO_POSITION_POLICY", "append")

    overridden = variants._apply_environment_position_policy_override(config, family="hh")

    assert overridden.position_policy == "append"
    assert variants._apply_environment_position_policy_override(config, family="hubbard").position_policy == "best_insert_refit"


def test_first_hit_record_emits_native_component_ledger() -> None:
    candidate = variants.build_pairwise_qubit_excitation_pool(2)[0]
    config = variants._get_config("static_pos_geo_adapt_vqe")

    record = variants._first_hit_record(
        threshold=1e-6,
        iteration=2,
        energy=-0.9999995,
        reference={
            "same_cutoff_exact_gs_energy": -1.0,
            "exact_reference_energy": -1.0,
            "exact_reference_n_ph_max": None,
            "primary_energy_metric": "same_cutoff_abs_delta_e",
            "primary_reference_energy": -1.0,
            "primary_reference_source": "same_cutoff_exact_gs_energy",
            "same_cutoff_error_role": "primary",
        },
        selected=[candidate],
        selected_batches=[[candidate]],
        config=config,
        hamiltonian_pauli_term_count=5,
        pool_term_count=7,
        nfev_total=11,
        gradient_scan_count=3,
        gradient_probe_count=13,
        metric_probe_count=29,
        metric_selector_probe_count=10,
        metric_qngd_refit_probe_count=4,
        metric_position_trial_probe_count=15,
        gradient_selector_probe_count=11,
        gradient_qngd_refit_probe_count=2,
        shots_per_pauli_term_proxy=1024,
    )

    assert record["source"] == "native_adaptive_iteration"
    assert record["first_hit_semantics"] == "native_first_crossing_after_adapt_iteration"
    assert abs(record["abs_delta_e"] - 5e-7) < 1e-12
    assert abs(record["abs_delta_e_same_cutoff"] - 5e-7) < 1e-12
    assert record["primary_energy_metric"] == "same_cutoff_abs_delta_e"
    assert record["S_alg"] == 53.0
    assert record["S_alg_N_H_outer_eval"] == 0.0
    assert record["S_alg_N_grad_probe"] == 13.0
    assert record["S_alg_N_metric_probe"] == 29.0
    assert record["S_alg_N_H_refit_eval"] == 11.0
    assert record["N_metric_selector_probe"] == 10.0
    assert record["N_metric_qngd_refit_probe"] == 4.0
    assert record["N_metric_position_trial_probe"] == 15.0
    assert record["N_metric_residual_probe"] == 0.0
    assert record["N_grad_selector_probe"] == 11.0
    assert record["N_grad_qngd_refit_probe"] == 2.0
    assert record["N_grad_residual_probe"] == 0.0
    ledger = record["table_i_measurement_event_ledger"]
    assert ledger["schema"] == variants.TABLE_I_EVENT_LEDGER_SCHEMA
    assert ledger["component_totals"]["N_metric_probe"] == 29.0
    assert ledger["metric_split_totals"]["status"] == "explicit_disjoint"
    assert ledger["gradient_split_totals"]["status"] == "explicit_disjoint"
    assert record["compiled_circuit_stats_status"] == "qiskit_first_hit_compile_unavailable"
    assert record["compiled_circuit_stats_error"] == "num_qubits_missing"
    assert record["compiled_resource_qiskit_validated"] is False
    assert "compiled_count_2q_total" not in record
    proxy = record["diagnostic_pauli_rotation_proxy_stats"]
    assert proxy["compiled_circuit_stats_status"] == "deterministic_pauli_rotation_proxy"
    assert proxy["compiled_count_2q_total"] == 4


def test_first_hit_threshold_normalization_accepts_list_inputs() -> None:
    assert variants._normalize_first_hit_thresholds([1e-8, 1e-6, 1e-6]) == (1e-6, 1e-8)


def test_full_meta_append_adapt_config_is_local_append_only_powell_comparator() -> None:
    config = variants._get_config("static_full_meta_append_adapt_vqe")
    guardrails = variants._guardrails(config, exact_reference_usage="reporting_only_after_optimization")

    assert "static_full_meta_append_adapt_vqe" in variants.GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS
    assert config.variant == "full_meta_append_only"
    assert config.display_name == "Append-only ADAPT-VQE (local full_meta)"
    assert config.method_kind == "full_meta_append_only_adapt"
    assert config.ansatz_name == "benchmark_local_full_meta_append_only_adapt"
    assert config.pool_kind == "full_meta"
    assert config.optimizer_kind == "powell"
    assert config.stop_rule == "raw_gradient"
    assert config.repeat_policy == "with_replacement"
    assert config.position_policy == "append"
    assert variants._pool_name_for_config(config) == "full_meta"
    assert variants._required_pool_key_for_config(config) == "full_meta"
    assert variants._taxonomy_role_for_config(config) == "same_pool_controller_comparator"
    assert guardrails["phase3_controller_called"] is False
    assert guardrails["static_adapt_controller_boundary"] == "not_called"
    assert guardrails["pool_source"] == "problem_local_full_meta_pool"
    assert guardrails["geo_reference_algorithm"] is None


def test_spsa_profile_override_derives_effective_config_without_mutating_defaults() -> None:
    for algorithm_id in (
        "static_full_meta_append_adapt_vqe",
        "static_qubit_qeb_adapt_vqe",
    ):
        config = variants._get_config(algorithm_id)
        effective, settings = variants._effective_optimizer_settings_for_config(
            config,
            optimizer_profile=PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
            optimizer_profile_source="env",
            adapt_spsa_maxiter=3,
            adapt_spsa_seed=99,
            adapt_spsa_a=0.07,
            adapt_spsa_c=0.03,
            adapt_spsa_alpha=0.601,
            adapt_spsa_gamma=0.11,
            adapt_spsa_big_a=9.0,
            optimizer_maxiter=5000,
            seed=42,
            optimizer_overlay_source="test",
        )

        expected_default = "powell" if algorithm_id == "static_full_meta_append_adapt_vqe" else "bfgs"
        assert config.optimizer_kind == expected_default
        assert variants._get_config(algorithm_id).optimizer_kind == expected_default
        assert effective.optimizer_kind == "spsa"
        assert settings["optimizer_profile"] == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
        assert settings["adapt_spsa_maxiter"] == 3
        assert settings["adapt_spsa_seed"] == 99
        assert settings["adapt_spsa_a"] == pytest.approx(0.07)
        assert settings["adapt_spsa_c"] == pytest.approx(0.03)
        assert settings["adapt_spsa_alpha"] == pytest.approx(0.601)
        assert settings["adapt_spsa_gamma"] == pytest.approx(0.11)
        assert settings["adapt_spsa_big_a"] == pytest.approx(9.0)
        assert settings["spsa_schedule"].a == pytest.approx(0.07)

    with pytest.raises(ValueError, match="visible Paper-I comparator methods"):
        variants._effective_optimizer_settings_for_config(
            variants._get_config("static_pos_geo_adapt_vqe"),
            optimizer_profile=PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
            optimizer_maxiter=5000,
            seed=42,
        )

    with pytest.raises(ValueError, match="require adapt_optimizer_kind=spsa"):
        variants._effective_optimizer_settings_for_config(
            variants._get_config("static_full_meta_append_adapt_vqe"),
            adapt_optimizer_kind="bfgs",
            adapt_spsa_a=0.07,
            optimizer_maxiter=5000,
            seed=42,
        )


def test_powell_optimizer_override_uses_scipy_minimize_without_spsa_schedule() -> None:
    config = variants._get_config("static_full_meta_append_adapt_vqe")
    effective, settings = variants._effective_optimizer_settings_for_config(
        config,
        adapt_optimizer_kind="powell",
        optimizer_maxiter=200,
        seed=42,
        optimizer_overlay_source="test",
    )

    assert config.optimizer_kind == "powell"
    assert effective.optimizer_kind == "powell"
    assert variants._requires_scipy_minimize(effective) is True
    assert settings["optimizer_kind"] == "powell"
    assert settings["optimizer_maxiter"] == 200
    assert settings["adapt_spsa_maxiter"] is None
    assert settings["adapt_spsa_seed"] is None
    assert settings["spsa_schedule"] is None


def test_rotosolve_optimizer_override_does_not_require_scipy_or_spsa_schedule() -> None:
    config = variants._get_config("static_full_meta_append_adapt_vqe")
    effective, settings = variants._effective_optimizer_settings_for_config(
        config,
        adapt_optimizer_kind="rotosolve",
        optimizer_maxiter=40,
        seed=42,
        optimizer_overlay_source="test",
    )

    assert config.optimizer_kind == "powell"
    assert effective.optimizer_kind == "rotosolve"
    assert variants._requires_scipy_minimize(effective) is False
    assert settings["optimizer_kind"] == "rotosolve"
    assert settings["optimizer_maxiter"] == 40
    assert settings["adapt_spsa_maxiter"] is None
    assert settings["adapt_spsa_seed"] is None
    assert settings["spsa_schedule"] is None


def test_powell_optimizer_override_reaches_scipy_minimize_method(monkeypatch, tmp_path: Path) -> None:
    methods: list[str | None] = []

    def _fake_minimize(objective, x0, method=None, options=None):  # noqa: ANN001, ANN003, ANN201
        methods.append(method)
        x = np.asarray(x0, dtype=float).reshape(-1) + 0.1
        return SimpleNamespace(x=x, fun=float(objective(x)), nfev=2, nit=1, success=True, message="ok")

    def _fake_sector_probability(ctx, psi):  # noqa: ANN001
        return {
            "sector_probability": 1.0,
            "sector_leak_probability": 0.0,
            "sector_leak_flag": False,
            "sector_leak_threshold": 1e-8,
            "boson_legal_probability_min": None,
            "boson_illegal_probability_max": None,
            "boson_truncation_leak_flag": False,
            "boson_subspace_diagnostics": None,
            "truncation_constraints_evaluated": [],
        }

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(variants, "_import_scipy_minimize", lambda: _fake_minimize)
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: _fake_context([]))
    monkeypatch.setattr(
        variants,
        "build_full_meta_candidate_pool",
        lambda context, *, max_terms=variants._POOL_TERM_CAP: variants.build_pairwise_qubit_excitation_pool(
            context.layout.total_qubits,
            max_terms=max_terms,
        ),
    )
    monkeypatch.setattr(variants, "sector_probability", _fake_sector_probability)

    for algorithm_id in ("static_full_meta_append_adapt_vqe", "static_geo_adapt_vqe"):
        methods.clear()
        payload = variants.run_generic_static_adapt_variant_single(
            family="hubbard",
            case_id="hubbard_L2",
            algorithm_id=algorithm_id,
            output_dir=tmp_path / algorithm_id,
            max_adapt_iterations=1,
            optimizer_maxiter=5,
            gradient_threshold=0.0,
            adapt_optimizer_kind="powell",
            optimizer_overlay_source="test",
        )

        row = payload["rows"][0]
        assert methods == ["Powell"]
        assert row["optimizer_kind"] == "powell"
        assert row["optimizer"] == "scipy.optimize.minimize:Powell"
        assert row["adapt_history"][0]["optimizer"] == "scipy.optimize.minimize:Powell"


@pytest.mark.parametrize("optimizer_kind", ["bfgs", "powell"])
def test_scipy_optimizer_failure_stops_and_marks_quality_nonpassing(
    monkeypatch,
    tmp_path: Path,
    optimizer_kind: str,
) -> None:
    candidate = variants.build_pairwise_qubit_excitation_pool(2)[0]

    def _fake_score(**kwargs):  # noqa: ANN003, ANN201
        return [
            {
                "label": candidate.label,
                "support": list(candidate.support),
                "pauli_labels_exyz": list(candidate.pauli_labels_exyz),
                "gradient": 1.0,
                "abs_gradient": 1.0,
                "selector_score": 1.0,
            }
        ]

    def _failed_optimize(**kwargs):  # noqa: ANN003, ANN201
        return np.asarray(kwargs["x0"], dtype=float), 0.0, {
            "nfev": 1,
            "nit": 1,
            "success": False,
            "message": "forced_failure",
            "optimizer": f"scipy.optimize.minimize:{optimizer_kind.title()}",
        }

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(variants, "_import_scipy_minimize", lambda: object())
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: _fake_context([]))
    monkeypatch.setattr(
        variants,
        "build_full_meta_candidate_pool",
        lambda context, *, max_terms=variants._POOL_TERM_CAP: (candidate,),
    )
    monkeypatch.setattr(variants, "_score_candidates", _fake_score)
    monkeypatch.setattr(variants, "_optimize_selected", _failed_optimize)
    monkeypatch.setattr(variants, "sector_probability", lambda context, psi: {"sector_probability": 1.0})

    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / optimizer_kind,
        max_adapt_iterations=3,
        gradient_threshold=0.0,
        adapt_optimizer_kind=optimizer_kind,
        optimizer_overlay_source="test",
    )

    row = payload["result"]
    assert payload["status"] == "completed_quality_nonpassing"
    assert row["quality_gate_reason"] == f"{optimizer_kind}_optimizer_failed"
    assert row["adapt_stop_reason"] == f"{optimizer_kind}_optimizer_failed"
    assert row["adapt_num_iterations"] == 1
    assert row["adapt_history"][0]["eligible_for_first_hit"] is False


def test_fixed_horizon_append_accepts_only_finite_nonincreasing_powell_maxiter_caps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = variants.build_pairwise_qubit_excitation_pool(2)[0]

    def _fake_score(**kwargs):  # noqa: ANN003, ANN201
        return [
            {
                "label": candidate.label,
                "support": list(candidate.support),
                "pauli_labels_exyz": list(candidate.pauli_labels_exyz),
                "gradient": 1.0,
                "abs_gradient": 1.0,
                "selector_score": 1.0,
            }
        ]

    def _capped_powell(**kwargs):  # noqa: ANN003, ANN201
        theta = np.asarray(kwargs["x0"], dtype=float).reshape(-1)
        return theta, -0.1, {
            "nfev": 7,
            "nit": int(kwargs["optimizer_maxiter"]),
            "status": 2,
            "success": False,
            "message": "Maximum number of iterations has been exceeded.",
            "optimizer": "scipy.optimize.minimize:Powell",
            "optimizer_decision_energy": -0.1,
            "optimizer_exact_energy": -0.1,
        }

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(variants, "_import_scipy_minimize", lambda: object())
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: _fake_context([]))
    monkeypatch.setattr(
        variants,
        "build_full_meta_candidate_pool",
        lambda context, *, max_terms=variants._POOL_TERM_CAP: (candidate,),
    )
    monkeypatch.setattr(variants, "_score_candidates", _fake_score)
    monkeypatch.setattr(variants, "_optimize_selected", _capped_powell)
    monkeypatch.setattr(variants, "sector_probability", lambda context, psi: {"sector_probability": 1.0})

    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "accepted_cap",
        max_adapt_iterations=3,
        optimizer_maxiter=2,
        gradient_threshold=0.0,
        generic_adapt_stop_policy="fixed_horizon_no_target_v1",
        powell_maxiter_cap_policy="accept_finite_nonincreasing_v1",
    )

    row = payload["result"]
    assert payload["status"] == "completed"
    assert row["adapt_stop_reason"] == "max_adapt_iterations"
    assert row["adapt_num_iterations"] == 3
    assert row["optimizer_success_all"] is True
    assert row["optimizer_raw_success_all"] is False
    assert row["optimizer_capped"] is True
    assert row["optimizer_capped_count"] == 3
    assert row["optimizer_capped_iterations"] == [0, 1, 2]
    assert row["optimizer_capped_accepted_count"] == 3
    assert row["optimizer_capped_accepted_iterations"] == [0, 1, 2]
    assert all(item["optimizer_capped"] for item in row["adapt_history"])
    assert all(item["optimizer_capped_accepted"] for item in row["adapt_history"])
    assert all(item["eligible_for_first_hit"] for item in row["adapt_history"])
    assert all(
        item["optimizer_cap_acceptance_reason"]
        == "finite_nonincreasing_powell_maxiter_accepted"
        for item in row["adapt_history"]
    )


@pytest.mark.parametrize(
    ("theta", "energy_after", "decision_energy", "status", "message", "reason"),
    [
        ([float("nan")], -1.1, -1.1, 2, "Maximum number of iterations has been exceeded.", "nonfinite_parameters"),
        ([0.0], -1.1, float("nan"), 2, "Maximum number of iterations has been exceeded.", "nonfinite_objective"),
        ([0.0], -0.9, -0.9, 2, "Maximum number of iterations has been exceeded.", "energy_increase_exceeds_tolerance"),
        ([0.0], -1.1, -1.1, 1, "Maximum number of iterations has been exceeded.", "not_powell_maxiter_only"),
        ([0.0], -1.1, -1.1, 2, "forced_failure", "not_powell_maxiter_only"),
    ],
)
def test_powell_cap_rejects_nonfinite_increasing_or_nonmaxiter_failures(
    theta: list[float],
    energy_after: float,
    decision_energy: float,
    status: int,
    message: str,
    reason: str,
) -> None:
    info = variants._classify_powell_maxiter_cap(
        config=variants._get_config("static_full_meta_append_adapt_vqe"),
        policy="accept_finite_nonincreasing_v1",
        theta=np.asarray(theta, dtype=float),
        energy_before=-1.0,
        energy_after=float(energy_after),
        optimizer_maxiter=200,
        opt_info={
            "success": False,
            "status": int(status),
            "nit": 200,
            "message": str(message),
            "optimizer": "scipy.optimize.minimize:Powell",
            "optimizer_decision_energy": float(decision_energy),
        },
    )

    assert info["success"] is False
    assert info["optimizer_capped_accepted"] is False
    assert info["optimizer_cap_acceptance_reason"] == reason


def test_powell_cap_accepts_only_numerical_tolerance_scale_energy_regression() -> None:
    info = variants._classify_powell_maxiter_cap(
        config=variants._get_config("static_full_meta_append_adapt_vqe"),
        policy="accept_finite_nonincreasing_v1",
        theta=np.asarray([0.0], dtype=float),
        energy_before=-1.0,
        energy_after=-1.0 + 5.0e-11,
        optimizer_maxiter=200,
        opt_info={
            "success": False,
            "status": 2,
            "nit": 200,
            "message": "Maximum number of iterations has been exceeded.",
            "optimizer": "scipy.optimize.minimize:Powell",
            "optimizer_decision_energy": -1.0 + 5.0e-11,
        },
    )

    assert info["optimizer_capped_accepted"] is True
    assert info["optimizer_cap_energy_nonincreasing"] is True


def test_powell_cap_policy_is_restricted_to_fixed_horizon_append_powell(
    tmp_path: Path,
) -> None:
    missing_horizon = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "missing_fixed_horizon",
        max_adapt_iterations=1,
        powell_maxiter_cap_policy="accept_finite_nonincreasing_v1",
    )
    assert missing_horizon["status"] == "failed"
    assert "requires fixed_horizon_no_target_v1" in missing_horizon["reason"]

    target_stop = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "target_stop",
        max_adapt_iterations=1,
        gradient_threshold=0.0,
        energy_stop_target=1.0e-4,
        generic_adapt_stop_policy="fixed_horizon_no_target_v1",
        powell_maxiter_cap_policy="accept_finite_nonincreasing_v1",
    )
    assert target_stop["status"] == "failed"
    assert "requires energy_stop_target to be absent" in target_stop["reason"]

    gradient_stop = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "gradient_stop",
        max_adapt_iterations=1,
        generic_adapt_stop_policy="fixed_horizon_no_target_v1",
        powell_maxiter_cap_policy="accept_finite_nonincreasing_v1",
    )
    assert gradient_stop["status"] == "failed"
    assert "requires gradient_threshold=0" in gradient_stop["reason"]

    geo = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_geo_adapt_vqe",
        output_dir=tmp_path / "geo",
        max_adapt_iterations=1,
        generic_adapt_stop_policy="fixed_horizon_no_target_v1",
        powell_maxiter_cap_policy="accept_finite_nonincreasing_v1",
    )
    assert geo["status"] == "failed"
    assert "restricted to append-only" in geo["reason"]


def test_rotosolve_optimizer_override_runs_without_scipy_minimize(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(variants.GENERIC_STATIC_ALLOW_UNSUPPORTED_ROTOSOLVE_FALLBACK_ENV, raising=False)

    def _unexpected_import():  # noqa: ANN202
        raise AssertionError("rotosolve should not import scipy.optimize.minimize")

    def _fake_sector_probability(ctx, psi):  # noqa: ANN001
        return {
            "sector_probability": 1.0,
            "sector_leak_probability": 0.0,
            "sector_leak_flag": False,
            "sector_leak_threshold": 1e-8,
            "boson_legal_probability_min": None,
            "boson_illegal_probability_max": None,
            "boson_truncation_leak_flag": False,
            "boson_subspace_diagnostics": None,
            "truncation_constraints_evaluated": [],
        }

    one_pauli_candidate = variants._PoolCandidate(
        label="test_single_xy",
        polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xy", pc=1.0)]),
        support=(0, 1),
        pauli_labels_exyz=("xy",),
        construction="test_single_pauli_coeff_1",
    )
    captured_rotosolve_kwargs: list[dict[str, object]] = []

    def _fake_rotosolve(objective, x0, *, maxiter, tol=1e-10, period=None, shift=None, callback=None):  # noqa: ANN001, ANN202
        del tol, callback
        captured_rotosolve_kwargs.append(
            {
                "maxiter": int(maxiter),
                "period": np.asarray(period, dtype=float).reshape(-1).tolist(),
                "shift": np.asarray(shift, dtype=float).reshape(-1).tolist(),
            }
        )
        x = np.asarray(x0, dtype=float).reshape(-1)
        return SimpleNamespace(
            x=x,
            fun=float(objective(x)),
            nfev=1,
            nit=1,
            success=True,
            message="ok",
            accepted_steps=0,
        )

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: False)
    monkeypatch.setattr(variants, "_import_scipy_minimize", _unexpected_import)
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: _fake_context([]))
    monkeypatch.setattr(
        variants,
        "build_full_meta_candidate_pool",
        lambda context, *, max_terms=variants._POOL_TERM_CAP: (one_pauli_candidate,),
    )
    monkeypatch.setattr(variants, "sector_probability", _fake_sector_probability)
    monkeypatch.setattr(variants, "rotosolve_coordinate_descent", _fake_rotosolve)

    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "rotosolve",
        max_adapt_iterations=1,
        optimizer_maxiter=2,
        gradient_threshold=0.0,
        adapt_optimizer_kind="rotosolve",
        optimizer_overlay_source="test",
    )

    row = payload["rows"][0]
    assert row["optimizer_kind"] == "rotosolve"
    assert row["optimizer"] == "repo_coordinate_descent:rotosolve_coordinate_descent"
    assert row["adapt_history"][0]["optimizer"] == "repo_coordinate_descent:rotosolve_coordinate_descent"
    assert captured_rotosolve_kwargs
    assert captured_rotosolve_kwargs[0]["period"] == pytest.approx([math.pi])
    assert captured_rotosolve_kwargs[0]["shift"] == pytest.approx([0.25 * math.pi])
    assert row["rotosolve_stencil_source"] == "compiled_executor_single_pauli_coefficients"
    assert row["rotosolve_period"] == pytest.approx([math.pi])
    assert row["rotosolve_shift"] == pytest.approx([0.25 * math.pi])
    assert row["adapt_history"][0]["rotosolve_stencil_source"] == "compiled_executor_single_pauli_coefficients"


def test_rotosolve_optimizer_failure_is_quality_nonpassing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(variants.GENERIC_STATIC_ALLOW_UNSUPPORTED_ROTOSOLVE_FALLBACK_ENV, raising=False)

    one_pauli_candidate = variants._PoolCandidate(
        label="test_single_xy",
        polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xy", pc=1.0)]),
        support=(0, 1),
        pauli_labels_exyz=("xy",),
        construction="test_single_pauli_coeff_1",
    )

    def _fake_sector_probability(ctx, psi):  # noqa: ANN001
        return {
            "sector_probability": 1.0,
            "sector_leak_probability": 0.0,
            "sector_leak_flag": False,
            "sector_leak_threshold": 1e-8,
            "boson_legal_probability_min": None,
            "boson_illegal_probability_max": None,
            "boson_truncation_leak_flag": False,
            "boson_subspace_diagnostics": None,
            "truncation_constraints_evaluated": [],
        }

    def _failed_rotosolve(objective, x0, *, maxiter, tol=1e-10, period=None, shift=None, callback=None):  # noqa: ANN001, ANN202
        del tol, period, shift, callback
        x = np.asarray(x0, dtype=float).reshape(-1)
        return SimpleNamespace(
            x=x,
            fun=float(objective(x)),
            nfev=1,
            nit=int(maxiter),
            success=False,
            message="maxiter_reached",
            accepted_steps=0,
        )

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: False)
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: _fake_context([]))
    monkeypatch.setattr(
        variants,
        "build_full_meta_candidate_pool",
        lambda context, *, max_terms=variants._POOL_TERM_CAP: (one_pauli_candidate,),
    )
    monkeypatch.setattr(variants, "sector_probability", _fake_sector_probability)
    monkeypatch.setattr(variants, "rotosolve_coordinate_descent", _failed_rotosolve)

    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "rotosolve_failure",
        max_adapt_iterations=1,
        optimizer_maxiter=2,
        gradient_threshold=0.0,
        adapt_optimizer_kind="rotosolve",
        optimizer_overlay_source="test",
    )

    row = payload["rows"][0]
    assert payload["status"] == "completed_quality_nonpassing"
    assert row["status"] == "quality_nonpassing"
    assert row["quality_gate_reason"] == "rotosolve_optimizer_failed"
    assert row["optimizer_success_all"] is False
    assert row["optimizer_messages"] == ["maxiter_reached"]


@pytest.mark.parametrize(
    "algorithm_id",
    ["static_full_meta_append_adapt_vqe", "static_geo_adapt_vqe"],
)
def test_rotosolve_multi_pauli_macro_uses_runtime_pauli_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    algorithm_id: str,
) -> None:
    monkeypatch.delenv(variants.GENERIC_STATIC_ALLOW_UNSUPPORTED_ROTOSOLVE_FALLBACK_ENV, raising=False)

    macro_candidate = variants._PoolCandidate(
        label="test_macro_xy_yx",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(2, ps="xy", pc=1.0),
                PauliTerm(2, ps="yx", pc=0.5),
            ],
        ),
        support=(0, 1),
        pauli_labels_exyz=("xy", "yx"),
        construction="test_multi_pauli_macro",
    )
    captured_rotosolve_kwargs: list[dict[str, object]] = []

    def _fake_sector_probability(ctx, psi):  # noqa: ANN001
        return {
            "sector_probability": 1.0,
            "sector_leak_probability": 0.0,
            "sector_leak_flag": False,
            "sector_leak_threshold": 1e-8,
            "boson_legal_probability_min": None,
            "boson_illegal_probability_max": None,
            "boson_truncation_leak_flag": False,
            "boson_subspace_diagnostics": None,
            "truncation_constraints_evaluated": [],
        }

    def _fake_rotosolve(objective, x0, *, maxiter, tol=1e-10, period=None, shift=None, callback=None):  # noqa: ANN001, ANN202
        del tol, callback
        x = np.asarray(x0, dtype=float).reshape(-1)
        captured_rotosolve_kwargs.append(
            {
                "maxiter": int(maxiter),
                "x0_size": int(x.size),
                "period": np.asarray(period, dtype=float).reshape(-1).tolist(),
                "shift": np.asarray(shift, dtype=float).reshape(-1).tolist(),
            }
        )
        return SimpleNamespace(
            x=x,
            fun=float(objective(x)),
            nfev=1,
            nit=1,
            success=True,
            message="ok",
            accepted_steps=0,
        )

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: False)
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: _fake_context([]))
    monkeypatch.setattr(
        variants,
        "build_full_meta_candidate_pool",
        lambda context, *, max_terms=variants._POOL_TERM_CAP: (macro_candidate,),
    )
    monkeypatch.setattr(variants, "sector_probability", _fake_sector_probability)
    monkeypatch.setattr(variants, "rotosolve_coordinate_descent", _fake_rotosolve)

    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id=algorithm_id,
        output_dir=tmp_path / f"rotosolve_macro_runtime_{algorithm_id}",
        max_adapt_iterations=1,
        optimizer_maxiter=2,
        gradient_threshold=0.0,
        adapt_optimizer_kind="rotosolve",
        optimizer_overlay_source="test",
    )

    row = payload["rows"][0]
    assert payload["status"] == "completed"
    assert captured_rotosolve_kwargs
    assert captured_rotosolve_kwargs[0]["x0_size"] == 2
    assert captured_rotosolve_kwargs[0]["period"] == pytest.approx([math.pi, 2.0 * math.pi])
    assert captured_rotosolve_kwargs[0]["shift"] == pytest.approx([0.25 * math.pi, 0.5 * math.pi])
    assert row["rotosolve_stencil_source"] == "compiled_executor_single_pauli_coefficients"
    assert row["rotosolve_period"] == pytest.approx([math.pi, 2.0 * math.pi])
    assert row["rotosolve_shift"] == pytest.approx([0.25 * math.pi, 0.5 * math.pi])
    assert row["theta_coordinate_mode"] == "per_pauli_term"
    assert row["parameterization_mode"] == "per_pauli_term"
    assert row["num_parameters"] == 2
    assert row["runtime_parameter_count"] == 2
    assert row["logical_parameter_count"] == 1
    assert len(row["optimal_point"]) == 2
    assert len(row["logical_optimal_point"]) == 1
    assert row["adapt_history"][0]["rotosolve_stencil_source"] == "compiled_executor_single_pauli_coefficients"


def test_projected_geo_metric_diagnostics_track_offdiagonal_geometry() -> None:
    tangents = [
        np.asarray([1.0, 0.0, 0.0], dtype=complex),
        np.asarray([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2.0),
    ]

    diag = variants._geo_metric_diagnostics(tangents, metric_floor=1e-8)

    assert diag["rank"] == 2
    assert diag["regularization"] >= 1e-8
    assert diag["condition"] is not None
    assert diag["offdiag_norm"] > 0.0


def test_geo_tangent_span_pseudoinverse_matches_dense_rank_deficient_metric() -> None:
    first = np.asarray(
        [1.0 + 0.5j, -0.25j, 0.75 - 0.5j], dtype=complex
    )
    second = np.asarray(
        [0.25 - 0.75j, 1.5 + 0.25j, -0.5j], dtype=complex
    )
    tangents = (first, 2.0 * first, second, first - second)
    force = np.asarray([0.3, -0.7, 1.1, 0.2], dtype=float)
    metric_floor = 1.0e-8

    dense_diag = variants._geo_metric_diagnostics(
        tangents, metric_floor=metric_floor
    )
    dense_metric = np.asarray(dense_diag["metric"], dtype=float)
    pinv_rcond = max(1.0e-10, metric_floor)
    dense_step = np.linalg.pinv(dense_metric, rcond=pinv_rcond) @ force
    span = variants._geo_selector_tangent_span_solve(
        tangents,
        force,
        metric_floor=metric_floor,
    )

    np.testing.assert_allclose(
        span["step"], dense_step, rtol=2.0e-11, atol=2.0e-12
    )
    np.testing.assert_allclose(
        span["metric_diagonal"],
        np.diag(dense_metric),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert span["rank"] == dense_diag["rank"]
    assert span["condition"] == pytest.approx(
        dense_diag["condition"], rel=2.0e-8, abs=2.0e-10
    )
    assert span["offdiag_norm"] == pytest.approx(
        dense_diag["offdiag_norm"], rel=2.0e-13, abs=2.0e-13
    )
    assert span["fs_norm"] == pytest.approx(
        math.sqrt(max(0.0, float(dense_step @ dense_metric @ dense_step))),
        rel=2.0e-11,
        abs=2.0e-12,
    )
    assert span["pinv_supported_rank"] < len(tangents)


def test_geo_tangent_span_cutoff_matches_dense_metric_pinv_support() -> None:
    # The metric eigenvalues are the squared tangent singular values.  With
    # pinv rcond=1e-10, 4e-10 is retained while 2.5e-11 is removed.
    tangents = (
        np.asarray([1.0, 0.0, 0.0], dtype=complex),
        np.asarray([0.0, 2.0e-5, 0.0], dtype=complex),
        np.asarray([0.0, 0.0, 5.0e-6], dtype=complex),
    )
    force = np.ones(3, dtype=float)
    metric_floor = 1.0e-12
    dense_metric = np.asarray(
        variants._geo_metric_diagnostics(
            tangents, metric_floor=metric_floor
        )["metric"],
        dtype=float,
    )
    dense_step = np.linalg.pinv(dense_metric, rcond=1.0e-10) @ force
    span = variants._geo_selector_tangent_span_solve(
        tangents,
        force,
        metric_floor=metric_floor,
    )

    np.testing.assert_allclose(span["step"], dense_step, rtol=1.0e-13, atol=0.0)
    assert span["pinv_rcond"] == 1.0e-10
    assert span["pinv_supported_rank"] == 2
    assert span["rank"] == 3
    assert span["step"][2] == 0.0


def test_geo_tangent_span_solve_preserves_full_metric_pair_accounting() -> None:
    tangents = (
        np.asarray([1.0, 0.0], dtype=complex),
        np.asarray([0.0, 1.0], dtype=complex),
        np.asarray([1.0, 1.0], dtype=complex),
        np.asarray([1.0, -1.0j], dtype=complex),
    )
    force = np.asarray([1.0, -0.5, 0.25, 0.75], dtype=float)
    span = variants._geo_selector_tangent_span_solve(
        tangents,
        force,
        metric_floor=1.0e-8,
    )
    assert len(span["step"]) == len(tangents)

    coordinates = tuple(f"candidate:{index}" for index in range(len(tangents)))
    ledger = variants._PaperIComparatorEstimatorLedger()
    receipt = ledger.record_symmetric_pair_family(
        _compact_metric_template(),
        coordinates=coordinates,
        component="N_metric",
        consumer_scope="round:0:selector_metric",
    )
    expected_pair_count = len(tangents) * (len(tangents) + 1) // 2
    assert receipt["member_count"] == expected_pair_count
    assert ledger.summary()["N_metric"] == expected_pair_count
    assert ledger.summary()["S_unique"] == expected_pair_count
    assert ledger.occurrence_summary()["S_alg"] == expected_pair_count


def test_hh_nph3_projected_singleton_geo_first_selector_matches_route_golden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Small real-pool parity gate for the scalable selector factorization."""

    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE_SCOPE", "exact")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE_DIR", str(tmp_path / "pool_cache"))
    case_id = TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS["weak-weak"]
    spec = table_i_canonical_spec_by_case_id(
        "hh",
        case_id,
        profile=TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE,
    )
    context = variants._resolve_context_from_spec(spec)
    parent_result = variants._build_full_meta_candidate_pool_with_meta(
        context,
        max_terms=None,
        hh_full_meta_class_filter_json=None,
    )
    pool, pool_meta = variants._expand_pool_with_shared_pauli_children(
        pool=parent_result.candidates,
        context=context,
        config=variants._get_config("static_geo_adapt_vqe"),
        mode=SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1,
        symmetry_policy="hard_guard",
        max_subset_size=1,
        max_terms=None,
    )
    assert len(parent_result.candidates) == 123
    assert len(pool) == 1622
    assert pool_meta["projected_singleton_candidate_count"] == 1622
    pauli_action_cache: dict[str, object] = {}
    compiled_pool = variants._compile_pool(
        pool,
        pauli_action_cache=pauli_action_cache,
    )
    psi = np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1)
    h_compiled = variants.compile_polynomial_action(
        context.hamiltonian,
        tol=1.0e-12,
        pauli_action_cache=pauli_action_cache,
    )
    _energy, hpsi = variants.energy_via_one_apply(psi, h_compiled)
    scored = variants._score_candidates(
        config=variants._get_config("static_geo_adapt_vqe"),
        psi=psi,
        hpsi=hpsi,
        compiled_pool=compiled_pool,
        selected_labels=set(),
        previous_selected_label=None,
        metric_floor=1.0e-8,
        adapt_iteration=0,
    )

    assert len(scored) == 1622
    assert len(scored) * (len(scored) + 1) // 2 == 1_316_253
    top = scored[0]
    assert top["label"] == (
        "hh_fermionic_reusable::bond_charge_current_nn_up(0,1)::split[0]::"
        "eeeeeexy::legal_projected"
    )
    assert top["selector_score"] == pytest.approx(
        0.0833333333333334, rel=0.0, abs=2.0e-13
    )
    assert top["geo_metric_rank"] == 82
    assert top["geo_metric_pinv_supported_rank"] == 82
    assert top["geo_natural_step_fs_norm"] == pytest.approx(
        2.999999999999742, rel=0.0, abs=2.0e-11
    )
    assert top["geo_metric_factorization"] == "exact_real_tangent_span_svd_v1"
    assert top[
        "geo_tangent_complex_row_count_after_exact_zero_compression"
    ] <= 64
    assert top["geo_tangent_real_row_count_after_exact_zero_compression"] <= 126


def test_hh_completion_case_is_routable_by_generic_append_comparator() -> None:
    case_id = TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS["weak-weak"]

    spec = variants._spec_by_case_id(
        "hh",
        case_id,
        variants.STATIC_FULL_META_APPEND_ADAPT_VQE,
    )

    assert spec.benchmark_id == case_id
    assert "--n-ph-max 3" in " ".join(spec.base_pipeline_args)


def test_geo_selector_scores_full_live_candidate_set(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = tuple(
        variants._PoolCandidate(
            label=label,
            polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
            support=(idx,),
            pauli_labels_exyz=("x",),
            construction="test",
        )
        for idx, label in enumerate(("low", "middle", "high"))
    )
    compiled_pool = tuple(variants._CompiledCandidate(candidate, candidate.label) for candidate in candidates)

    def _fake_apply(_psi, compiled):  # noqa: ANN001, ANN202
        scale = {"low": 1.0, "middle": 2.0, "high": 3.0}[str(compiled)]
        return np.asarray([0.0, 1j * scale, 0.0], dtype=complex)

    def _fake_grad(_hpsi, gpsi):  # noqa: ANN001, ANN202
        return float(np.imag(np.asarray(gpsi).reshape(-1)[1]))

    monkeypatch.setattr(variants, "apply_compiled_polynomial", _fake_apply)
    monkeypatch.setattr(variants, "adapt_commutator_grad_from_hpsi", _fake_grad)

    def _dense_pinv_must_not_run(*_args, **_kwargs):  # noqa: ANN202
        raise AssertionError("Geo selector allocated the dense candidate Gram pseudoinverse")

    monkeypatch.setattr(variants.np.linalg, "pinv", _dense_pinv_must_not_run)

    scored = variants._score_candidates(
        config=variants._get_config("static_geo_adapt_vqe"),
        psi=np.asarray([1.0, 0.0, 0.0], dtype=complex),
        hpsi=np.asarray([0.0, 1.0, 0.0], dtype=complex),
        compiled_pool=compiled_pool,
        selected_labels=set(),
        previous_selected_label=None,
        metric_floor=1e-8,
    )

    assert {row["label"] for row in scored} == {"low", "middle", "high"}
    assert all("geo_metric_candidate_cap" not in row for row in scored)
    assert all(row["geo_metric_candidate_count_before_screen"] == 3 for row in scored)
    assert all(row["geo_metric_candidate_count_after_screen"] == 3 for row in scored)
    assert all(row["geo_metric_prescreen_mode"] == "full_candidate_set" for row in scored)
    assert all(
        row["geo_metric_factorization"] == "exact_real_tangent_span_svd_v1"
        for row in scored
    )


def test_geo_selector_machine_precision_alias_ties_use_label_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = tuple(
        variants._PoolCandidate(
            label=label,
            polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
            support=(index,),
            pauli_labels_exyz=("x",),
            construction="test",
        )
        for index, label in enumerate(("alias_z", "alias_a", "lower"))
    )
    compiled_pool = tuple(
        variants._CompiledCandidate(candidate, candidate.label)
        for candidate in candidates
    )

    monkeypatch.setattr(
        variants,
        "apply_compiled_polynomial",
        lambda _psi, _compiled: np.asarray([0.0, 1.0j], dtype=complex),
    )
    monkeypatch.setattr(
        variants,
        "adapt_commutator_grad_from_hpsi",
        lambda _hpsi, _gpsi: 1.0,
    )

    def _span_solve(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        return {
            "step": np.asarray([1.0 + 2.0e-16, 1.0, 0.5], dtype=float),
            "metric_diagonal": np.ones(3, dtype=float),
            "regularization": 1.0e-8,
            "rank": 2,
            "pinv_supported_rank": 2,
            "condition": 1.0,
            "offdiag_norm": 0.0,
            "scale": 1.0,
            "fs_norm": 1.0,
            "l2_norm": 1.5,
            "max_abs_step": 1.0 + 2.0e-16,
            "pinv_rcond": 1.0e-8,
            "complex_row_count_before_exact_zero_compression": 2,
            "complex_row_count_after_exact_zero_compression": 1,
            "real_row_count_after_exact_zero_compression": 1,
        }

    monkeypatch.setattr(variants, "_geo_selector_tangent_span_solve", _span_solve)

    scored = variants._score_candidates(
        config=variants._get_config("static_geo_adapt_vqe"),
        psi=np.asarray([1.0, 0.0], dtype=complex),
        hpsi=np.asarray([0.0, 1.0], dtype=complex),
        compiled_pool=compiled_pool,
        selected_labels=set(),
        previous_selected_label=None,
        metric_floor=1.0e-8,
    )

    assert [row["label"] for row in scored] == ["alias_a", "alias_z", "lower"]
    assert scored[1]["selector_score"] > scored[0]["selector_score"]
    assert (
        scored[0]["geo_selector_numerical_tie_policy"]
        == "machine_precision_collinear_alias_then_label_v1"
    )


def test_rank_deficient_pos_geo_uses_pseudoinverse_without_diagonal_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = tuple(
        variants._PoolCandidate(
            label=label,
            polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
            support=(idx,),
            pauli_labels_exyz=("x",),
            construction="test",
        )
        for idx, label in enumerate(("duplicate_a", "duplicate_b"))
    )
    compiled_pool = tuple(variants._CompiledCandidate(candidate, candidate.label) for candidate in candidates)

    def _fake_apply(_psi, compiled):  # noqa: ANN001, ANN202
        scale = {"duplicate_a": 1.0, "duplicate_b": 2.0}[str(compiled)]
        return np.asarray([0.0, 1j * scale, 0.0], dtype=complex)

    def _fake_grad(_hpsi, gpsi):  # noqa: ANN001, ANN202
        return float(np.imag(np.asarray(gpsi).reshape(-1)[1]))

    monkeypatch.setattr(variants, "apply_compiled_polynomial", _fake_apply)
    monkeypatch.setattr(variants, "adapt_commutator_grad_from_hpsi", _fake_grad)

    scored = variants._score_candidates(
        config=variants._get_config("static_pos_geo_adapt_vqe"),
        psi=np.asarray([1.0, 0.0, 0.0], dtype=complex),
        hpsi=np.asarray([0.0, 1.0, 0.0], dtype=complex),
        compiled_pool=compiled_pool,
        selected_labels=set(),
        previous_selected_label=None,
        metric_floor=1e-8,
    )

    assert {row["label"] for row in scored} == {"duplicate_a", "duplicate_b"}
    assert all(row["geo_metric_rank_deficient"] is True for row in scored)
    assert all(row["geo_metric_diagonal_fallback"] is False for row in scored)
    assert all("rank_deficient_diagonal_fs_fallback" not in row["geo_selector_mode"] for row in scored)
    assert all(row["geo_metric_prescreen_mode"] == "full_candidate_set" for row in scored)


def test_selector_decision_noise_can_change_qeb_selection_without_overwriting_exact_fields(monkeypatch) -> None:
    low_exact = variants._PoolCandidate(
        label="low_exact_high_decision",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
        support=(0,),
        pauli_labels_exyz=("x",),
        construction="test",
    )
    high_exact = variants._PoolCandidate(
        label="high_exact_low_decision",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="y", pc=1.0)]),
        support=(1,),
        pauli_labels_exyz=("y",),
        construction="test",
    )
    compiled_pool = (
        variants._CompiledCandidate(low_exact, "low"),
        variants._CompiledCandidate(high_exact, "high"),
    )

    def _fake_apply(_psi, compiled):  # noqa: ANN001, ANN202
        return np.asarray([0.1 if compiled == "low" else 2.0], dtype=complex)

    def _fake_grad(_hpsi, gpsi):  # noqa: ANN001, ANN202
        return float(np.real(np.asarray(gpsi).reshape(-1)[0]))

    class _FakeRecorder:
        config = SimpleNamespace(enabled=True)

        def apply(self, value, *, surface, value_kind, phase, extra_scope=None):  # noqa: ANN001, ANN201
            assert surface == "adapt_selector_gradient"
            label = str((extra_scope or {}).get("candidate_label"))
            return 10.0 if label == low_exact.label else 0.01

    monkeypatch.setattr(variants, "apply_compiled_polynomial", _fake_apply)
    monkeypatch.setattr(variants, "adapt_commutator_grad_from_hpsi", _fake_grad)

    scored = variants._score_candidates(
        config=variants._get_config("static_qubit_qeb_adapt_vqe"),
        psi=np.asarray([1.0], dtype=complex),
        hpsi=np.asarray([0.0], dtype=complex),
        compiled_pool=compiled_pool,
        selected_labels=set(),
        previous_selected_label=None,
        metric_floor=1e-8,
        decision_noise_recorder=_FakeRecorder(),
        adapt_iteration=0,
    )

    assert [row["label"] for row in scored] == [low_exact.label, high_exact.label]
    assert scored[0]["abs_gradient"] == 0.1
    assert scored[0]["abs_gradient_decision"] == 10.0
    assert scored[1]["abs_gradient"] == 2.0
    assert scored[1]["abs_gradient_decision"] == 0.01
    batch = variants._select_batch(
        config=variants._get_config("static_qubit_qeb_adapt_vqe"),
        scored=scored,
        gradient_threshold=1.0,
    )
    assert [row["label"] for row in batch] == [low_exact.label]


def test_powell_decision_noise_metadata_preserves_exact_reporting_and_work_counts(monkeypatch, tmp_path: Path) -> None:
    ctx = _fake_context([])
    ctx.exact_target = SimpleNamespace(resolve_energy=lambda ai_log=None: -1.0)

    def _fake_minimize(objective, x0, method=None, options=None):  # noqa: ANN001, ANN003, ANN201
        x = np.asarray(x0, dtype=float).reshape(-1) + 0.1
        return SimpleNamespace(x=x, fun=float(objective(x)), nfev=1, nit=1, success=True, message="ok")

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(variants, "_import_scipy_minimize", lambda: _fake_minimize)
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: ctx)
    monkeypatch.setattr(
        variants,
        "build_full_meta_candidate_pool",
        lambda context, *, max_terms=variants._POOL_TERM_CAP: variants.build_pairwise_qubit_excitation_pool(
            context.layout.total_qubits,
            max_terms=max_terms,
        ),
    )
    monkeypatch.setattr(variants, "sector_probability", lambda context, psi: {"sector_probability": 1.0})

    exact_payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "exact",
        max_adapt_iterations=1,
        optimizer_maxiter=3,
        gradient_threshold=0.0,
    )
    noisy_payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "noisy",
        max_adapt_iterations=1,
        optimizer_maxiter=3,
        gradient_threshold=0.0,
        benchmark_decision_noise_config={
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "1e-3",
            "benchmark_decision_noise_seed": "20260515",
        },
    )

    exact_row = exact_payload["rows"][0]
    noisy_row = noisy_payload["rows"][0]
    metadata = noisy_payload["benchmark_decision_noise"]
    assert noisy_payload["benchmark_decision_noise_status"] == "ok"
    assert noisy_row["benchmark_decision_noise_status"] == "ok"
    assert metadata["semantic"] == "benchmark_decision_value_noise_not_physical_shots_v1"
    assert metadata["physical_shots_unchanged"] is True
    assert metadata["algorithmic_measurement_work_unchanged"] is True
    assert metadata["draw_count_total"] > 0
    assert set(metadata["surfaces_affected"]) == {"adapt_refit_powell_objective", "adapt_selector_gradient"}
    assert noisy_row["energy"] == noisy_row["optimizer_exact_energy"]
    assert noisy_row["delta_E_abs"] == abs(noisy_row["energy"] - noisy_row["exact_energy"])
    assert noisy_row["optimizer_decision_energy"] != noisy_row["optimizer_exact_energy"]
    assert noisy_row["benchmark_first_hits"] == {}
    for field in (
        "S_alg",
        "shots_total",
        "energy_eval_count_proxy",
        "gradient_operator_probe_count_proxy",
        "metric_operator_probe_count_proxy",
        "S_alg_N_other_quantum",
    ):
        assert noisy_row[field] == exact_row[field]
    rows_payload = json.loads((tmp_path / "noisy" / "rows.json").read_text(encoding="utf-8"))
    assert rows_payload["benchmark_decision_noise_status"] == "ok"
    assert rows_payload["benchmark_decision_noise"]["draw_count_total"] == metadata["draw_count_total"]


def test_pos_geo_decision_noise_records_geo_qngd_and_position_surfaces(monkeypatch, tmp_path: Path) -> None:
    ctx = _fake_context([])
    ctx.exact_target = SimpleNamespace(resolve_energy=lambda ai_log=None: -1.0)

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: False)
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: ctx)
    monkeypatch.setattr(
        variants,
        "build_full_meta_candidate_pool",
        lambda context, *, max_terms=variants._POOL_TERM_CAP: variants.build_pairwise_qubit_excitation_pool(
            context.layout.total_qubits,
            max_terms=max_terms,
        ),
    )
    monkeypatch.setattr(variants, "sector_probability", lambda context, psi: {"sector_probability": 1.0})

    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_pos_geo_adapt_vqe",
        output_dir=tmp_path,
        max_adapt_iterations=1,
        optimizer_maxiter=1,
        gradient_threshold=0.0,
        benchmark_decision_noise_config={
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "1e-3",
            "benchmark_decision_noise_seed": "20260515",
        },
    )

    assert payload["status"] in {"completed", "completed_quality_nonpassing"}
    row = payload["rows"][0]
    metadata = payload["benchmark_decision_noise"]
    surfaces = set(metadata["surfaces_affected"])
    assert "adapt_selector_geo_natural_step_norm" in surfaces
    assert "adapt_selector_geo_natural_step" in surfaces
    assert "adapt_refit_qngd_objective" in surfaces
    assert "adapt_pos_geo_position_trial_energy" in surfaces
    assert row["energy"] is not None
    assert row["delta_E_abs"] == abs(row["energy"] - row["exact_energy"])
    history = row["adapt_history"][0]
    assert history["geo_natural_step_fs_norm"] is not None
    assert history["geo_natural_step_fs_norm_decision"] is not None
    assert history["position_trials"][0]["energy_exact"] == history["position_trials"][0]["energy"]
    assert "energy_decision" in history["position_trials"][0]


def test_spsa_polish_is_seeded_energy_only_descent() -> None:
    def objective(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        return float(np.sum((theta - np.asarray([0.25, -0.5])) ** 2))

    theta0 = np.asarray([1.0, 1.0], dtype=float)
    energy0 = objective(theta0)
    first = variants._spsa_polish(
        theta0=theta0,
        energy0=energy0,
        objective=objective,
        rng_seed=17,
        maxiter=80,
        max_abs_step=0.25,
        accept_tol=1e-12,
    )
    second = variants._spsa_polish(
        theta0=theta0,
        energy0=energy0,
        objective=objective,
        rng_seed=17,
        maxiter=80,
        max_abs_step=0.25,
        accept_tol=1e-12,
    )

    assert first[1] < energy0
    assert first[2]["success"] is True
    assert first[2]["accepted_step_count"] > 0
    assert first[2]["nfev"] > 0
    assert np.allclose(first[0], second[0])
    assert first[1] == second[1]


def test_powell_refit_reuses_optimizer_endpoint_objective() -> None:
    context = _fake_context([])
    psi_ref = np.asarray(
        context.reference_state.build_state(),
        dtype=complex,
    ).reshape(-1)
    pauli_action_cache: dict[str, object] = {}
    h_compiled = variants.compile_polynomial_action(
        context.hamiltonian,
        tol=1e-12,
        pauli_action_cache=pauli_action_cache,
    )
    candidate = variants.build_pairwise_qubit_excitation_pool(2)[0]
    recorded: list[tuple[np.ndarray, str]] = []

    def _fake_minimize(fun, x0, *, method, options):  # noqa: ANN001, ANN201
        assert method == "Powell"
        assert options["maxiter"] == 5
        x_initial = np.asarray(x0, dtype=float).reshape(-1)
        fun(x_initial)
        x_final = x_initial + 0.125
        final_value = fun(x_final)
        return SimpleNamespace(
            x=x_final,
            fun=final_value,
            nfev=2,
            nit=1,
            status=0,
            success=True,
            message="ok",
        )

    theta, _energy, info = variants._optimize_selected(
        minimize_fn=_fake_minimize,
        selected=(candidate,),
        x0=np.zeros(1, dtype=float),
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        pauli_action_cache=pauli_action_cache,
        optimizer_maxiter=5,
        optimizer_method="Powell",
        estimator_h_call_recorder=lambda state, scope: recorded.append(
            (np.asarray(state, dtype=complex), str(scope))
        ),
    )

    assert theta.tolist() == pytest.approx([0.125])
    assert [scope for _state, scope in recorded] == [
        "adapt_refit_powell_objective:objective",
        "adapt_refit_powell_objective:objective",
    ]
    assert info["optimizer_objective_hamiltonian_occurrence_count"] == 2
    assert info["post_optimizer_exact_verification_hamiltonian_occurrence_count"] == 0
    assert info["optimizer_total_hamiltonian_occurrence_count"] == 2
    assert info["optimizer_energy_source"] == "optimizer_objective_exact_value"


def test_powell_refit_verifies_genuinely_unevaluated_endpoint() -> None:
    context = _fake_context([])
    psi_ref = np.asarray(
        context.reference_state.build_state(),
        dtype=complex,
    ).reshape(-1)
    pauli_action_cache: dict[str, object] = {}
    h_compiled = variants.compile_polynomial_action(
        context.hamiltonian,
        tol=1e-12,
        pauli_action_cache=pauli_action_cache,
    )
    candidate = variants.build_pairwise_qubit_excitation_pool(2)[0]
    recorded: list[tuple[np.ndarray, str]] = []

    def _fake_minimize(fun, x0, *, method, options):  # noqa: ANN001, ANN201
        x_initial = np.asarray(x0, dtype=float).reshape(-1)
        fun(x_initial)
        return SimpleNamespace(
            x=x_initial + 0.25,
            fun=-999.0,
            nfev=1,
            nit=1,
            status=0,
            success=True,
            message="stale fun at unevaluated endpoint",
        )

    theta, energy, info = variants._optimize_selected(
        minimize_fn=_fake_minimize,
        selected=(candidate,),
        x0=np.zeros(1, dtype=float),
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        pauli_action_cache=pauli_action_cache,
        optimizer_maxiter=5,
        optimizer_method="Powell",
        estimator_h_call_recorder=lambda state, scope: recorded.append(
            (np.asarray(state, dtype=complex), str(scope))
        ),
    )

    assert theta.tolist() == pytest.approx([0.25])
    assert energy != pytest.approx(-999.0)
    assert [scope for _state, scope in recorded] == [
        "adapt_refit_powell_objective:objective",
        (
            "adapt_refit_powell_objective:"
            "required_endpoint_exact_verification"
        ),
    ]
    assert info["optimizer_objective_hamiltonian_occurrence_count"] == 1
    assert (
        info[
            "post_optimizer_exact_verification_hamiltonian_occurrence_count"
        ]
        == 1
    )
    assert info["optimizer_total_hamiltonian_occurrence_count"] == 2
    assert info["optimizer_energy_source"] == (
        "required_endpoint_exact_verification"
    )


def test_spsa_polish_accepts_explicit_schedule() -> None:
    def objective(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        return float(np.sum((theta - np.asarray([0.25, -0.5])) ** 2))

    theta0 = np.asarray([1.0, 1.0], dtype=float)
    energy0 = objective(theta0)
    schedule = variants._SpsaPolishSchedule(a=0.07, c=0.03, alpha=0.601, gamma=0.11, big_a=9.0)

    _theta, _energy, info = variants._spsa_polish(
        theta0=theta0,
        energy0=energy0,
        objective=objective,
        rng_seed=17,
        maxiter=5,
        max_abs_step=0.25,
        accept_tol=1e-12,
        schedule=schedule,
    )

    assert info["spsa_a"] == pytest.approx(0.07)
    assert info["spsa_c"] == pytest.approx(0.03)
    assert info["spsa_alpha"] == pytest.approx(0.601)
    assert info["spsa_gamma"] == pytest.approx(0.11)
    assert info["spsa_A"] == pytest.approx(9.0)
    assert info["spsa_big_a"] == pytest.approx(9.0)


def test_primary_spsa_refit_uses_repo_native_minimize(monkeypatch) -> None:
    ctx = _fake_context([])
    psi_ref = np.asarray(ctx.reference_state.build_state(), dtype=complex).reshape(-1)
    h_compiled = variants.compile_polynomial_action(ctx.hamiltonian, tol=1e-12, pauli_action_cache={})
    candidate = variants.build_pairwise_qubit_excitation_pool(2)[0]
    calls: list[dict[str, object]] = []

    def _stationary_native_minimize(fun, x0, **kwargs):  # noqa: ANN001, ANN003, ANN201
        x = np.asarray(x0, dtype=float).reshape(-1)
        calls.append(dict(kwargs))
        return SimpleNamespace(
            x=x,
            fun=float(fun(x)),
            nfev=1,
            nit=1,
            success=True,
            message="spsa_completed(maxiter=1)",
            history=[{"iter": 1, "best_fun": float(fun(x))}],
            optimizer_memory={"optimizer": "SPSA", "available": False},
        )

    monkeypatch.setattr(variants, "spsa_minimize", _stationary_native_minimize)

    _theta, _energy, info = variants._optimize_selected_spsa(
        selected=(candidate,),
        x0=np.zeros(1, dtype=float),
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        pauli_action_cache={},
        optimizer_maxiter=1,
        spsa_seed=5,
    )

    assert calls
    assert calls[0]["seed"] == 5
    assert calls[0]["bounds"] is None
    assert calls[0]["project"] == "none"
    assert calls[0]["eval_repeats"] == 1
    assert calls[0]["avg_last"] == 0
    assert info["success"] is True
    assert info["optimizer"] == variants._NATIVE_SPSA_OPTIMIZER_LABEL
    assert info["spsa_refit_engine"] == variants._NATIVE_SPSA_OPTIMIZER_LABEL
    assert info["spsa_return_policy"] == "best_observed_with_x0_seed_avg_last_0"
    assert info["spsa_accepted_step_count"] is None
    assert info["spsa_optimizer_memory"]["optimizer"] == "SPSA"


def test_qngd_logical_tangent_matches_runtime_chain_rule_for_qeb_blocks() -> None:
    spec = variants._spec_by_case_id("hubbard", "hubbard_L2", "static_geo_qeb_adapt_vqe")
    context = variants._resolve_context_from_spec(spec)
    psi_ref = np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1)
    psi_ref = psi_ref / np.linalg.norm(psi_ref)
    pool = {candidate.label: candidate for candidate in variants.build_pairwise_qubit_excitation_pool(context.layout.total_qubits)}
    selected = [pool["qeb_pair(0,1)"], pool["qeb_double(0,3->1,2)"]]
    executor = CompiledAnsatzExecutor(
        selected,
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode="per_pauli_term",
    )
    theta = np.asarray([0.03, 0.11], dtype=float)
    runtime_theta = expand_legacy_logical_theta(theta, executor.layout)

    _psi, runtime_tangents = executor.prepare_state_with_runtime_tangents(runtime_theta, psi_ref)

    eps = 1e-6
    for logical_idx, block in enumerate(executor.layout.blocks):
        logical_tangent = np.zeros_like(psi_ref, dtype=complex)
        for runtime_idx in range(int(block.runtime_start), int(block.runtime_stop)):
            logical_tangent += np.asarray(runtime_tangents[int(runtime_idx)], dtype=complex).reshape(-1)
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[logical_idx] += eps
        theta_minus[logical_idx] -= eps
        psi_plus = executor.prepare_state(expand_legacy_logical_theta(theta_plus, executor.layout), psi_ref)
        psi_minus = executor.prepare_state(expand_legacy_logical_theta(theta_minus, executor.layout), psi_ref)
        finite_difference_tangent = (psi_plus - psi_minus) / (2.0 * eps)

        assert np.linalg.norm(logical_tangent - finite_difference_tangent) < 1e-7


def test_qngd_counts_force_and_metric_probes_per_inner_step() -> None:
    selected = (
        variants._PoolCandidate(
            label="x_q0",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="ex", pc=1.0)]),
            support=(0,),
            pauli_labels_exyz=("ex",),
            construction="test",
        ),
        variants._PoolCandidate(
            label="x_q1",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xe", pc=1.0)]),
            support=(1,),
            pauli_labels_exyz=("xe",),
            construction="test",
        ),
    )
    cache: dict[str, object] = {}
    h_compiled = variants.compile_polynomial_action(
        PauliPolynomial("JW", [PauliTerm(2, ps="zz", pc=1.0)]),
        tol=1e-12,
        pauli_action_cache=cache,
    )

    _theta, _energy, info = variants._optimize_selected_qngd(
        selected=selected,
        x0=np.zeros(2, dtype=float),
        psi_ref=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=complex),
        h_compiled=h_compiled,
        pauli_action_cache=cache,
        optimizer_maxiter=1,
        metric_floor=1e-8,
        spsa_seed=7,
    )

    assert info["qngd_metric_eval_count"] == 1
    assert info["qngd_metric_operator_probe_count_total"] == 3
    assert info["qngd_gradient_operator_probe_count_total"] == 2


def test_geo_qeb_replacement_policy_scores_full_pool_before_immediate_repeat_skip() -> None:
    geo_qeb = variants._get_config("static_geo_qeb_adapt_vqe")
    old_geo = variants._get_config("static_geo_qubit_adapt_vqe")

    assert variants._blocked_labels_for_config(
        geo_qeb,
        selected_labels={"qeb_pair(0,1)", "qeb_pair(1,2)"},
        previous_selected_label="qeb_pair(1,2)",
    ) == set()
    assert variants._guardrails(
        geo_qeb,
        exact_reference_usage="reporting_only_after_optimization",
    )["geo_replacement_policy"] == (
        "score_full_pool_with_replacement; skip_append_after_immediate_repeat_wins"
    )
    assert variants._blocked_labels_for_config(
        old_geo,
        selected_labels={"qeb_pair(0,1)", "qeb_pair(1,2)"},
        previous_selected_label="qeb_pair(1,2)",
    ) == {"qeb_pair(0,1)", "qeb_pair(1,2)"}


def test_geo_immediate_repeat_is_scored_then_skipped_without_depth_growth(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first = variants.build_pairwise_qubit_excitation_pool(2)[0]
    second = variants._PoolCandidate(
        label=f"{first.label}_alt",
        polynomial=first.polynomial,
        support=first.support,
        pauli_labels_exyz=first.pauli_labels_exyz,
        construction=first.construction,
    )
    # The fourth score call is the terminal selector verification.  Keep its
    # candidate set nonempty so the test proves that final gradients and the
    # Geo metric scan enter the canonical estimator ledger.
    winners = [first.label, first.label, second.label, first.label]
    score_calls = 0

    def _fake_score(**kwargs):  # noqa: ANN003, ANN201
        nonlocal score_calls
        if score_calls >= len(winners):
            return []
        winner = winners[score_calls]
        score_calls += 1
        ordered = [first, second] if winner == first.label else [second, first]
        return [
            {
                "label": candidate.label,
                "support": list(candidate.support),
                "pauli_labels_exyz": list(candidate.pauli_labels_exyz),
                "gradient": 1.0 if index == 0 else 0.5,
                "abs_gradient": 1.0 if index == 0 else 0.5,
                "selector_score": 1.0 if index == 0 else 0.5,
                "geo_metric_rank": 2,
                "geo_metric_condition": 1.0,
                "geo_metric_regularization": 1e-8,
                "geo_metric_offdiag_norm": 0.0,
                "geo_natural_step": 1.0 if index == 0 else 0.5,
                "geo_projected_residual_force": 1.0 if index == 0 else 0.5,
                "geo_projected_residual_score": 1.0 if index == 0 else 0.5,
                "geo_natural_step_fs_norm": 1.0,
                "geo_natural_step_l2_norm": 1.0,
                "geo_max_abs_natural_step": 1.0,
                "geo_metric_candidate_count_before_screen": 2,
                "geo_metric_candidate_count_after_screen": 2,
                "geo_metric_prescreen_mode": "full_candidate_set",
                "geo_selector_mode": "qeb_pool_projected_natural_gradient",
            }
            for index, candidate in enumerate(ordered)
        ]

    def _fake_powell(**kwargs):  # noqa: ANN003, ANN201
        theta = np.asarray(kwargs["x0"], dtype=float).reshape(-1) + 0.01
        _record_fake_powell_h_calls(kwargs, theta)
        p = len(kwargs["selected"])
        return theta, float(-p), {
            "nfev": 1,
            "nit": 1,
            "success": True,
            "message": "ok",
            "optimizer": "scipy.optimize.minimize:Powell",
        }

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(variants, "_import_scipy_minimize", lambda: object())
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: _fake_context([]))
    monkeypatch.setattr(
        variants,
        "build_full_meta_candidate_pool",
        lambda context, *, max_terms=variants._POOL_TERM_CAP: (first, second),
    )
    monkeypatch.setattr(variants, "_score_candidates", _fake_score)
    monkeypatch.setattr(variants, "_optimize_selected", _fake_powell)
    monkeypatch.setattr(variants, "sector_probability", lambda context, psi: {"sector_probability": 1.0})

    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_geo_adapt_vqe",
        output_dir=tmp_path,
        max_adapt_iterations=3,
        gradient_threshold=0.0,
    )

    assert payload["status"] == "completed", payload
    row = payload["result"]
    history = row["adapt_history"]
    assert row["selected_operators"] == [first.label, second.label]
    assert row["selected_operator_batches"] == [[first.label], [second.label]]
    assert [item["depth_after"] for item in history] == [1, 1, 2]
    assert [item["appended_operator_count"] for item in history] == [1, 0, 1]
    assert [item["geo_immediate_repeat_skipped"] for item in history] == [False, True, False]
    assert [item["candidate_count_scored"] for item in history] == [2, 2, 2]
    assert row["N_grad_selector_probe"] == 6.0
    assert row["N_grad_qngd_refit_probe"] == 0.0
    assert row["N_metric_selector_probe"] == 9.0
    assert row["N_metric_qngd_refit_probe"] == 0.0
    assert row["N_H_outer_eval"] == 4.0
    assert row["N_H_refit_eval"] == 3.0
    assert row["N_grad"] == 8.0
    assert row["N_metric"] == 12.0
    assert row["S_alg"] == 27.0
    assert row["legacy_aggregate_measurement_work_proxy"]["S_alg"] == 21.0
    assert row["estimator_call_accounting"]["complete"] is True
    assert row["estimator_call_accounting"]["per_round_receipts_close_to_terminal"] is True
    assert row["estimator_call_accounting"]["terminal_diagnostic_queries_included"] is True
    assert row["estimator_call_accounting"]["terminal_diagnostic_queries_excluded"] is False
    assert row["estimator_call_accounting"]["executed_occurrence_accounting"][
        "all_execution"
    ]["total_call_occurrences"] == 27
    assert [
        receipt["iteration"] for receipt in row["estimator_call_round_receipts"]
    ] == [0, 1, 2, 3]
    assert [
        receipt["receipt_kind"]
        for receipt in row["estimator_call_round_receipts"]
    ] == ["adaptive_round", "adaptive_round", "adaptive_round", "terminal_diagnostic"]
    terminal_receipt = row["adapt_terminal_estimator_call_receipt"]
    assert terminal_receipt == row["estimator_call_round_receipts"][-1]
    assert terminal_receipt["raw_occurrence_components"] == {
        "N_H_outer": 1,
        "N_H_refit": 0,
        "N_grad": 2,
        "N_metric": 3,
    }
    assert terminal_receipt["unique_component_delta"] == {
        "N_H_outer": 0,
        "N_H_refit": 0,
        "N_grad": 2,
        "N_metric": 3,
    }
    assert row["adapt_terminal_diagnostic_queries_in_S_alg"] is True
    occurrence_scopes = {
        occurrence["consumer_scope"]
        for occurrence in row["estimator_call_accounting"]["full_ledger"][
            "occurrences"
        ]
    }
    assert "terminal:hamiltonian_verification" in occurrence_scopes
    assert "terminal:selector_gradient" in occurrence_scopes
    assert "terminal:selector_metric" in occurrence_scopes
    assert not any(
        scope.endswith("post_optimizer_exact_verification")
        for scope in occurrence_scopes
    )
    assert row["optimizer_kind"] == "powell"


def test_append_adapt_allows_consecutive_reuse_and_charges_full_scan(monkeypatch, tmp_path: Path) -> None:
    candidate = variants.build_pairwise_qubit_excitation_pool(2)[0]
    score_calls = 0

    def _fake_score(**kwargs):  # noqa: ANN003, ANN201
        nonlocal score_calls
        if score_calls >= 2:
            return []
        score_calls += 1
        return [
            {
                "label": candidate.label,
                "support": list(candidate.support),
                "pauli_labels_exyz": list(candidate.pauli_labels_exyz),
                "gradient": 1.0,
                "abs_gradient": 1.0,
                "selector_score": 1.0,
            }
        ]

    def _fake_optimize(**kwargs):  # noqa: ANN003, ANN201
        theta = np.asarray(kwargs["x0"], dtype=float).reshape(-1) + 0.01
        _record_fake_powell_h_calls(kwargs, theta)
        return theta, -float(len(kwargs["selected"])), {
            "nfev": 1,
            "nit": 1,
            "success": True,
            "message": "ok",
            "optimizer": "scipy.optimize.minimize:Powell",
        }

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(variants, "_import_scipy_minimize", lambda: object())
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: _fake_context([]))
    monkeypatch.setattr(
        variants,
        "build_full_meta_candidate_pool",
        lambda context, *, max_terms=variants._POOL_TERM_CAP: (candidate,),
    )
    monkeypatch.setattr(variants, "_score_candidates", _fake_score)
    monkeypatch.setattr(variants, "_optimize_selected", _fake_optimize)
    monkeypatch.setattr(variants, "sector_probability", lambda context, psi: {"sector_probability": 1.0})

    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path,
        max_adapt_iterations=2,
        gradient_threshold=0.0,
        generic_adapt_stop_policy="fixed_horizon_no_target_v1",
        same_cutoff_exact_gs_energy=-2.0,
        first_hit_thresholds=(1.0e-8,),
    )

    assert payload["status"] == "completed", payload
    row = payload["result"]
    assert row["selected_operators"] == [candidate.label, candidate.label]
    assert row["selected_label_counts"] == {candidate.label: 2}
    assert [item["candidate_count_scored"] for item in row["adapt_history"]] == [1, 1]
    assert row["N_grad_selector_probe"] == 2.0
    assert row["N_H_outer_eval"] == 2.0
    assert row["N_H_refit_eval"] == 2.0
    assert row["S_alg"] == 6.0
    assert row["S_unique"] == 5.0
    assert row["paper_i_estimator_call_accounting"]["S_alg"] == 6.0
    assert row["estimator_call_accounting"]["all_branch_search_work"][
        "components"
    ] == {
        "N_H_outer": 2,
        "N_H_refit": 2,
        "N_grad": 2,
        "N_metric": 0,
    }
    assert row["legacy_aggregate_measurement_work_proxy"]["S_alg"] == 6.0
    assert row["estimator_call_accounting"]["winning_lineage"] == row[
        "estimator_call_accounting"
    ]["all_branch_search_work"]
    assert row["estimator_call_accounting"][
        "discarded_branch_only_by_unique_set_difference"
    ]["S_unique"] == 0
    assert row["adapt_terminal_diagnostic_queries_in_S_alg"] is False
    assert row["adapt_terminal_estimator_call_receipt"] is None
    assert row["adapt_terminal_diagnostic_hamiltonian_eval_count"] == 0
    assert row["adapt_terminal_diagnostic_gradient_probe_count"] == 0
    assert row["estimator_call_accounting"][
        "terminal_diagnostic_queries_included"
    ] is False
    assert row["estimator_call_accounting"][
        "terminal_diagnostic_queries_excluded"
    ] is True
    assert [
        receipt["receipt_kind"]
        for receipt in row["estimator_call_round_receipts"]
    ] == ["adaptive_round", "adaptive_round"]
    assert not any(
        occurrence["consumer_scope"].startswith("terminal:")
        for occurrence in row["estimator_call_accounting"]["full_ledger"][
            "occurrences"
        ]
    )
    assert all(
        history_row["estimator_call_round_receipt"]["prefix_closed"]
        for history_row in row["adapt_history"]
    )
    checkpoints = [
        history_row["active_prefix_checkpoint"]
        for history_row in row["adapt_history"]
    ]
    assert [checkpoint["active_operator_order"] for checkpoint in checkpoints] == [
        [candidate.label],
        [candidate.label, candidate.label],
    ]
    assert all(checkpoint["statevector_serialized"] is False for checkpoint in checkpoints)
    assert all("statevector" not in checkpoint for checkpoint in checkpoints)
    assert all("statevector_data" not in checkpoint for checkpoint in checkpoints)
    assert all(
        checkpoint["schema"]
        == variants.PAPER_I_COMPARATOR_ACTIVE_PREFIX_CHECKPOINT_SCHEMA
        for checkpoint in checkpoints
    )
    assert all(len(checkpoint["checkpoint_sha256"]) == 64 for checkpoint in checkpoints)
    assert checkpoints[0]["logical_parameter_count"] == 1
    assert checkpoints[0]["logical_theta"] == pytest.approx([0.01])
    assert checkpoints[0]["runtime_theta"] == pytest.approx(
        expand_legacy_logical_theta(
            np.asarray([0.01], dtype=float),
            variants.build_parameter_layout(
                [candidate],
                ignore_identity=True,
                coefficient_tolerance=1e-12,
                sort_terms=True,
            ),
        ).tolist()
    )
    assert checkpoints[0]["active_operators"][0]["execution_mode"] == "termwise_product"
    assert checkpoints[0]["active_operators"][0]["pauli_terms"]
    assert checkpoints[0]["sector_padding_audit"]["sector_probability"] == 1.0
    assert checkpoints[0]["ansatz_input_state_projective_fingerprint"]
    assert checkpoints[0]["prepared_state_projective_fingerprint"]
    first_hit = row["benchmark_first_hits"]["1e-08"]
    assert first_hit["S_alg"] == 6.0
    assert first_hit["legacy_aggregate_measurement_work_proxy"]["S_alg"] == 6.0
    assert first_hit["estimator_call_accounting"][
        "per_round_receipts_close_to_terminal"
    ] is True


def test_missing_scipy_writes_controlled_skip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: False)

    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_qubit_qeb_adapt_vqe",
        output_dir=tmp_path,
    )

    assert payload["status"] == "skipped_optional_dependency"
    row = payload["rows"][0]
    assert row["phase3_controller_called"] is False
    assert row["phase3_emulation"] is False
    assert row["uses_exact_for_decision"] is False
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "rows.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "generic_static_single.json").exists()
    assert (tmp_path / "metrics_proxy_summary.json").exists()


def _compact_metric_template(*, state: str = "state:test") -> variants.EstimatorCallKey:
    return variants.EstimatorCallKey(
        projective_state_fingerprint=state,
        hamiltonian_fingerprint="ham:test",
        backend_fingerprint="backend:test",
        precision_contract="precision:test",
        primitive_kind="metric_element",
        observable_or_formula_identity="fubini_study_metric_v1",
    )


def test_comparator_small_symmetric_family_keeps_legacy_full_payload() -> None:
    ledger = variants._PaperIComparatorEstimatorLedger()
    receipt = ledger.record_symmetric_pair_family(
        _compact_metric_template(),
        coordinates=("candidate:a", "candidate:b", "candidate:c"),
        component="N_metric",
        consumer_scope="round:0:selector_metric",
    )

    payload = ledger.to_payload()
    assert receipt == {
        "member_count": 6,
        "new_unique_primitive_count": 6,
        "serialization_mode": "materialized_v1",
    }
    assert payload["schema"] == "estimator_call_ledger_v1"
    assert len(payload["entries"]) == 6
    assert len(payload["occurrences"]) == 6
    assert payload["summary"]["S_unique"] == 6
    assert "S_alg" not in payload["summary"]
    assert payload["occurrence_summary"]["S_alg"] == 6
    assert payload["occurrence_summary"]["occurrence_sequences"] == list(
        range(1, 7)
    )
    assert variants._PaperIComparatorEstimatorLedger.validate_payload(payload)


def test_comparator_s_alg_counts_repeated_powell_estimator_callbacks() -> None:
    ledger = variants._PaperIComparatorEstimatorLedger()
    before = ledger.accounting_cursor()
    identity = variants.EstimatorCallKey(
        projective_state_fingerprint="state:powell-repeat",
        hamiltonian_fingerprint="ham:test",
        backend_fingerprint="backend:test",
        precision_contract="precision:test",
        primitive_kind="hamiltonian_expectation",
        observable_or_formula_identity="hamiltonian_expectation_v1",
    )
    for callback_index in range(2):
        ledger.record_call(
            identity,
            component="N_H_refit",
            consumer_scope=(
                f"round:0:adapt_refit_powell_objective:objective:{callback_index}"
            ),
        )
    round_receipt = variants._paper_i_comparator_round_receipt(
        ledger=ledger,
        before_payload=before,
        iteration=0,
        outcome="iteration_complete",
    )

    payload = ledger.to_payload()
    canonical, accounting = variants._paper_i_comparator_estimator_accounting(
        ledger=ledger,
        round_receipts=(round_receipt,),
        legacy_component_fields={},
    )

    assert payload["summary"]["S_unique"] == 1
    assert "S_alg" not in payload["summary"]
    assert payload["occurrence_summary"]["S_alg"] == 2
    assert round_receipt["S_alg_delta"] == 2
    assert round_receipt["S_unique_delta"] == 1
    assert canonical["S_alg"] == 2.0
    assert canonical["S_alg_N_H_refit_eval"] == 2.0
    assert canonical["S_unique"] == 1.0
    assert accounting["all_branch_search_work"]["S_alg"] == 2
    assert accounting["winning_lineage"]["S_alg"] == 2
    assert accounting["all_branch_unique_primitive_diagnostic"]["S_unique"] == 1
    event_ledger = canonical["table_i_measurement_event_ledger"]
    assert "canonical_unique_summary" not in event_ledger
    assert event_ledger["persistent_or_prior_run_cache_reductions_allowed"] is False
    assert event_ledger["already_acquired_optimizer_endpoint_reuse_allowed"] is True


def test_comparator_large_symmetric_family_is_exact_compact_and_replay_validated() -> None:
    coordinates = tuple(f"candidate:{index:04d}" for index in range(142))
    member_count = len(coordinates) * (len(coordinates) + 1) // 2
    ledger = variants._PaperIComparatorEstimatorLedger()
    before = ledger.accounting_cursor()
    block_receipt = ledger.record_symmetric_pair_family(
        _compact_metric_template(),
        coordinates=coordinates,
        component="N_metric",
        consumer_scope="round:0:selector_metric",
    )
    round_zero = variants._paper_i_comparator_round_receipt(
        ledger=ledger,
        before_payload=before,
        iteration=0,
        outcome="iteration_complete",
    )
    before_repeat = ledger.accounting_cursor()
    repeat_receipt = ledger.record_symmetric_pair_family(
        _compact_metric_template(),
        coordinates=coordinates,
        component="N_metric",
        consumer_scope="round:1:selector_metric",
    )
    round_one = variants._paper_i_comparator_round_receipt(
        ledger=ledger,
        before_payload=before_repeat,
        iteration=1,
        outcome="terminal_diagnostic_complete",
        receipt_kind="terminal_diagnostic",
    )

    payload = ledger.to_payload()
    assert block_receipt["serialization_mode"] == (
        "compact_exact_symmetric_family_v2"
    )
    assert block_receipt["new_unique_primitive_count"] == member_count
    assert repeat_receipt["new_unique_primitive_count"] == 0
    assert payload["schema"] == variants.PAPER_I_COMPARATOR_COMPACT_LEDGER_SCHEMA
    assert payload["serialization_mode"] == "bounded_exact_symmetric_family_v2"
    assert payload["summary"]["S_unique"] == member_count
    assert "S_alg" not in payload["summary"]
    assert payload["summary"]["N_metric"] == member_count
    assert payload["summary"]["primitive_ids"] == []
    assert payload["summary"]["primitive_ids_complete"] is False
    assert payload["occurrence_summary"]["total_call_occurrences"] == 2 * member_count
    assert payload["occurrence_summary"]["S_alg"] == 2 * member_count
    assert payload["occurrence_summary"]["occurrence_sequences"] == []
    assert payload["occurrence_summary"]["occurrence_sequences_complete"] is False
    assert payload["base_materialized_ledger"]["entries"] == []
    assert len(json.dumps(payload, sort_keys=True)) < 20_000
    assert round_zero["serialization_mode"] == "compact_exact_receipt_v2"
    assert round_zero["raw_occurrence_count"] == member_count
    assert round_zero["S_alg_delta"] == member_count
    assert round_zero["S_unique_delta"] == member_count
    assert round_zero["new_unique_primitive_ids"] == []
    assert round_zero["new_unique_primitive_ids_complete"] is False
    assert round_zero["round_occurrences"] == []
    assert round_one["raw_occurrence_count"] == member_count
    assert round_one["S_alg_delta"] == member_count
    assert round_one["S_unique_delta"] == 0
    assert round_one["new_unique_primitive_set_accumulator"]["count"] == 0
    assert variants._PaperIComparatorEstimatorLedger.validate_payload(payload)

    canonical, accounting = variants._paper_i_comparator_estimator_accounting(
        ledger=ledger,
        round_receipts=(round_zero, round_one),
        legacy_component_fields={},
        require_terminal_receipt=True,
    )
    assert canonical["S_alg"] == float(2 * member_count)
    assert canonical["S_alg_N_metric_probe"] == float(2 * member_count)
    assert accounting["S_alg"] == 2 * member_count
    assert accounting["S_unique"] == member_count
    assert accounting["all_branch_search_work"]["S_alg"] == 2 * member_count
    assert accounting["winning_lineage"]["S_alg"] == 2 * member_count
    assert accounting["all_branch_unique_primitive_diagnostic"]["S_unique"] == (
        member_count
    )
    assert accounting["per_round_receipts_close_to_terminal"] is True
    assert accounting["per_round_compact_exact_receipts_v2"] is True
    assert accounting["per_round_unique_primitive_count_sum"] == member_count
    assert accounting["per_round_raw_occurrence_count_sum"] == 2 * member_count
    assert accounting["terminal_diagnostic_queries_included"] is True
    assert accounting["terminal_diagnostic_queries_excluded"] is False

    tampered = json.loads(json.dumps(payload))
    tampered["compact_symmetric_family_blocks"][0]["member_count"] += 1
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        variants._PaperIComparatorEstimatorLedger.validate_payload(tampered)


def test_compact_comparator_dedup_is_state_keyed_and_hash_deterministic(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        variants, "_COMPARATOR_COMPACT_FAMILY_MEMBER_THRESHOLD", 3
    )
    coordinates = ("candidate:a", "candidate:b", "candidate:c", "candidate:d")

    def _build() -> dict[str, object]:
        ledger = variants._PaperIComparatorEstimatorLedger()
        for state, round_index in (
            ("state:a", 0),
            ("state:a", 1),
            ("state:b", 2),
        ):
            ledger.record_symmetric_pair_family(
                _compact_metric_template(state=state),
                coordinates=coordinates,
                component="N_metric",
                consumer_scope=f"round:{round_index}:selector_metric",
            )
        return ledger.to_payload()

    first = _build()
    second = _build()
    assert first["summary"]["S_unique"] == 20
    assert first["occurrence_summary"]["S_alg"] == 30
    assert first["occurrence_summary"]["total_call_occurrences"] == 30
    assert first["ledger_fingerprint"] == second["ledger_fingerprint"]
    assert [
        block["block_chain_sha256"]
        for block in first["compact_symmetric_family_blocks"]
    ] == [
        block["block_chain_sha256"]
        for block in second["compact_symmetric_family_blocks"]
    ]


def test_compact_comparator_threshold_crossing_closes_scope_branch_and_sequence(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        variants, "_COMPARATOR_COMPACT_FAMILY_MEMBER_THRESHOLD", 3
    )
    ledger = variants._PaperIComparatorEstimatorLedger()
    template = _compact_metric_template(state="state:shared")
    scope = "round:shared:selector_metric"
    branch = "winner"

    materialized = ledger.record_symmetric_pair_family(
        template,
        coordinates=("candidate:a", "candidate:b"),
        component="N_metric",
        consumer_scope=scope,
        branch_id=branch,
    )
    compact = ledger.record_symmetric_pair_family(
        template,
        coordinates=(
            "candidate:a",
            "candidate:b",
            "candidate:c",
            "candidate:d",
        ),
        component="N_metric",
        consumer_scope=scope,
        branch_id=branch,
    )

    assert materialized["serialization_mode"] == "materialized_v1"
    assert materialized["new_unique_primitive_count"] == 3
    assert compact["serialization_mode"] == "compact_exact_symmetric_family_v2"
    assert compact["new_unique_primitive_count"] == 7
    summary = ledger.summary()
    assert summary["S_unique"] == 10
    assert summary["unique_primitive_count_by_consumer_scope"][scope] == 10
    assert summary["unique_primitive_count_by_consumer_branch"][branch] == 10
    assert ledger.occurrence_summary()["S_alg"] == 13
    assert variants._PaperIComparatorEstimatorLedger.validate_payload(
        ledger.to_payload()
    )

    post_compact = ledger.record_call(
        variants.EstimatorCallKey(
            projective_state_fingerprint="state:after",
            hamiltonian_fingerprint="ham:test",
            backend_fingerprint="backend:test",
            precision_contract="precision:test",
            primitive_kind="hamiltonian_expectation",
            observable_or_formula_identity="hamiltonian_expectation_v1",
        ),
        component="N_H_outer",
        consumer_scope="round:after:outer_hamiltonian",
    )
    assert post_compact.occurrence_sequence == 14
    assert ledger.accounting_cursor()["occurrence_sequence_stop"] == 14


def test_generic_static_json_writer_is_atomic_and_streaming(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    variants._write_json(path, {"schema": "test", "values": list(range(5000))})

    assert json.loads(path.read_text(encoding="utf-8"))["values"][-1] == 4999
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_geo_qeb_does_not_require_scipy(monkeypatch, tmp_path: Path) -> None:
    ctx = _fake_context([])
    ctx.exact_target = SimpleNamespace(resolve_energy=lambda ai_log=None: -1.0)

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: False)
    monkeypatch.setattr(variants, "_import_scipy_minimize", lambda: (_ for _ in ()).throw(AssertionError("no scipy")))
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: ctx)
    monkeypatch.setattr(variants, "sector_probability", lambda context, psi: {"sector_probability": 1.0})

    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_geo_qeb_adapt_vqe",
        output_dir=tmp_path,
        max_adapt_iterations=0,
    )

    assert payload["status"] == "completed"
    row = payload["rows"][0]
    assert row["method_label"] == "Geo-ADAPT-VQE (QEB reference)"
    assert row["pool_name"] == "qubit_excitation_singles_doubles_pool"
    assert row["optimizer"] == "exact_bench_qngd:logical_shared_metric_backtracking"
    assert row["faithful_geo_adapt_vqe_implementation"] is False
    assert row["geo_outer_selector_source_faithful"] is True
    assert row["geo_inner_optimizer"] == "qngd"


def test_runner_resolves_exact_after_optimizer_and_emits_required_fields(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []

    def _fake_minimize(objective, x0, method=None, options=None):  # noqa: ANN001, ANN003, ANN201
        events.append("optimizer")
        x = np.asarray(x0, dtype=float).reshape(-1) + 0.1
        return SimpleNamespace(x=x, fun=float(objective(x)), nfev=2, nit=1, success=True, message="ok")

    def _fake_qngd(**kwargs):  # noqa: ANN003, ANN201
        events.append("optimizer")
        theta = np.asarray(kwargs["x0"], dtype=float).reshape(-1) + 0.1
        psi = variants._prepare_selected_state(
            selected=kwargs["selected"],
            theta=theta,
            psi_ref=kwargs["psi_ref"],
            pauli_action_cache=kwargs["pauli_action_cache"],
        )
        energy, _ = variants.energy_via_one_apply(psi, kwargs["h_compiled"])
        return theta, float(energy), {
            "nfev": 2,
            "nit": 1,
            "success": True,
            "message": "ok",
            "optimizer": "exact_bench_qngd:logical_shared_metric_backtracking",
            "qngd_metric_rank_last": 1,
            "qngd_metric_condition_last": 1.0,
            "qngd_step_fs_norm_last": 0.01,
            "qngd_step_l2_norm_last": 0.02,
            "qngd_max_abs_step_last": 0.02,
            "qngd_line_search_backtracks_total": 0,
        }

    def _fake_spsa(**kwargs):  # noqa: ANN003, ANN201
        events.append("optimizer")
        theta = np.asarray(kwargs["x0"], dtype=float).reshape(-1) + 0.1
        psi = variants._prepare_selected_state(
            selected=kwargs["selected"],
            theta=theta,
            psi_ref=kwargs["psi_ref"],
            pauli_action_cache=kwargs["pauli_action_cache"],
        )
        energy, _ = variants.energy_via_one_apply(psi, kwargs["h_compiled"])
        return theta, float(energy), {
            "nfev": 2,
            "nit": 1,
            "success": True,
            "message": "ok",
            "optimizer": variants._NATIVE_SPSA_OPTIMIZER_LABEL,
            "optimizer_decision_energy": float(energy),
            "optimizer_reported_energy": float(energy),
            "optimizer_exact_energy": float(energy),
            "spsa_refit_engine": variants._NATIVE_SPSA_OPTIMIZER_LABEL,
            "spsa_return_policy": "best_observed_with_x0_seed_avg_last_0",
            "spsa_seed": 7,
            "spsa_accepted_step_count": 1,
            "spsa_energy_decrease_total": 0.01,
        }

    def _fake_sector_probability(ctx, psi):  # noqa: ANN001
        assert events == ["optimizer", "exact"]
        events.append("sector")
        return {
            "sector_probability": 1.0,
            "sector_leak_probability": 0.0,
            "sector_leak_flag": False,
            "sector_leak_threshold": 1e-8,
            "boson_legal_probability_min": None,
            "boson_illegal_probability_max": None,
            "boson_truncation_leak_flag": False,
            "boson_subspace_diagnostics": None,
            "truncation_constraints_evaluated": [],
        }

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(variants, "_import_scipy_minimize", lambda: _fake_minimize)
    monkeypatch.setattr(variants, "_optimize_selected_qngd", _fake_qngd)
    monkeypatch.setattr(variants, "_optimize_selected_spsa", _fake_spsa)
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: _fake_context(events))
    monkeypatch.setattr(
        variants,
        "build_full_meta_candidate_pool",
        lambda context, *, max_terms=variants._POOL_TERM_CAP: variants.build_pairwise_qubit_excitation_pool(
            context.layout.total_qubits,
            max_terms=max_terms,
        ),
    )
    monkeypatch.setattr(variants, "sector_probability", _fake_sector_probability)

    for algorithm_id in variants.GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS:
        events.clear()
        out = tmp_path / algorithm_id
        payload = variants.run_generic_static_adapt_variant_single(
            family="hubbard",
            case_id="hubbard_L2",
            algorithm_id=algorithm_id,
            output_dir=out,
            max_adapt_iterations=1,
            optimizer_maxiter=5,
            gradient_threshold=0.0,
        )

        assert events == ["optimizer", "exact", "sector"]
        assert payload["status"] == "completed"
        assert payload["guardrails"]["phase3_controller_called"] is False
        assert payload["guardrails"]["phase3_emulation"] is False
        row = payload["rows"][0]
        assert row["method_id"] == algorithm_id
        assert row["phase3_controller_called"] is False
        assert row["phase3_emulation"] is False
        assert row["uses_exact_for_decision"] is False
        assert row["exact_reference_usage"] == "reporting_only_after_optimization"
        if algorithm_id == "static_qubit_qeb_adapt_vqe":
            assert row["method_label"] == "Qubit/QEB-ADAPT-VQE"
            assert row["pool_name"] == "qubit_excitation_singles_doubles_pool"
            assert row["taxonomy_role"] == "operator_class_comparator"
        elif algorithm_id == "static_geo_qeb_adapt_vqe":
            assert row["method_label"] == "Geo-ADAPT-VQE (QEB reference)"
            assert row["pool_name"] == "qubit_excitation_singles_doubles_pool"
            assert row["taxonomy_role"] == "operator_class_geo_comparator"
            assert row["faithful_geo_adapt_vqe_implementation"] is False
            assert row["geo_outer_selector_source_faithful"] is True
            assert row["geo_stop_rule"] == "fubini_study_natural_gradient_norm"
            assert row["raw_gradient_used_for_stop"] is False
            assert row["geo_inner_optimizer"] == "qngd"
            assert row["optimizer"] == "exact_bench_qngd:logical_shared_metric_backtracking"
            assert row["qngd_metric_rank_last"] == 1
        elif algorithm_id == "static_geo_adapt_vqe":
            assert row["method_label"] == "Geo-ADAPT-VQE"
            assert row["pool_name"] == "full_meta"
            assert row["taxonomy_role"] == "same_pool_controller_comparator"
            assert row["faithful_geo_adapt_vqe_implementation"] is False
            assert row["geo_outer_selector_source_faithful"] is True
            assert "problem_local_full_meta_pool_instead_of_excitation_pool" in row[
                "geo_source_algorithm_deviations"
            ]
            assert row["position_optimized_geo_adapt"] is False
            assert row["position_policy"] == "append"
            assert row["geo_inner_optimizer"] == "powell"
            assert row["optimizer"] == "scipy.optimize.minimize:Powell"
            assert row["optimizer_kind"] == "powell"
        else:
            assert row["pool_name"] == "full_meta"
            if algorithm_id == "static_full_meta_append_adapt_vqe":
                assert row["method_label"] == "Append-only ADAPT-VQE (local full_meta)"
                assert row["method_kind"] == "full_meta_append_only_adapt"
                assert row["taxonomy_role"] == "same_pool_controller_comparator"
                assert row["adapt_append_only"] is True
                assert row["raw_gradient_used_for_stop"] is True
                assert row["optimizer"] == "scipy.optimize.minimize:Powell"
                assert row["optimizer_kind"] == "powell"
                assert row["position_policy"] == "append"
                assert row["required_pool_key"] == "full_meta"
            elif algorithm_id == "static_pos_geo_adapt_vqe":
                assert row["method_label"] == "Pos-Geo-ADAPT-VQE"
                assert row["taxonomy_role"] == "same_pool_pos_geo_comparator"
                assert row["position_optimized_geo_adapt"] is True
                assert row["position_policy"] == "best_insert_refit"
            else:
                assert row["taxonomy_role"] == "same_pool_controller_comparator"
        assert row["energy"] is not None
        assert row["exact_energy"] == -1.0
        assert row["delta_E_abs"] is not None
        assert row["num_qubits"] == 2
        assert row["selected_operator_count"] == 1
        assert row["num_parameters"] == 1
        assert row["nfev"] == 2
        assert row["nit"] == 1
        assert row["first_hit_cost_source_kind"] == "qiskit_compiled_final_ansatz_circuit"
        assert row["compiled_resource_source_kind"] == "qiskit_compiled_final_ansatz_circuit"
        assert row["qiskit_first_hit_cost_validated"] is False
        if row["compiled_circuit_stats_status"] == "ok":
            assert row["compiled_resource_qiskit_validated"] is True
            assert row["compiled_count_2q_total"] is not None
            assert row["compiled_depth_total"] is not None
            assert row["compiled_op_counts"] is not None
        else:
            assert str(row["compiled_circuit_stats_status"]).startswith("qiskit_final_ansatz_compile_")
            assert row["compiled_resource_qiskit_validated"] is False
        proxy = row["diagnostic_pauli_rotation_proxy_stats"]
        assert proxy["depth_proxy"] == proxy["circuit_depth"] == proxy["compiled_depth_total"]
        assert proxy["compiled_count_2q_total"] == 4
        assert proxy["compiled_op_counts"]["cx"] == 4
        assert proxy["compiled_circuit_stats_status"] == "deterministic_pauli_rotation_proxy"
        assert row["shots_total"] > 0
        assert row["static_shot_estimate_status"] == "deterministic_proxy_not_physical_shots"
        assert row["shot_proxy_formula"].startswith("shots_total = shots_per_pauli_term_proxy")
        assert row["hamiltonian_pauli_term_count"] == 2
        assert row["phase3_controller_called"] is False
        assert (out / "result.json").exists()
        assert (out / "rows.json").exists()
        assert (out / "manifest.json").exists()
        assert (out / "generic_static_single.json").exists()
        assert (out / "metrics_proxy_runs.jsonl").exists()
        if algorithm_id == "static_geo_qubit_adapt_vqe":
            assert row["method_label"] == "legacy geometry diagnostic (removed from Table I)"
            assert row["faithful_geo_adapt_vqe_implementation"] is False
            assert row["geo_metric_floor"] == 1e-8
            assert "pinv(S" in row["geo_score_formula"]
            assert row["geo_selector_mode"] == "full_pool_projected_natural_gradient"
            assert row["geo_metric_rank_last"] is not None
            assert row["metric_operator_probe_count_proxy"] == 1
            hist = row["adapt_history"][0]
            assert len(hist["selector_metric_candidate_labels"]) == hist["candidate_count_scored"]
            assert hist["selector_metric_probe_count"] == 1
        elif algorithm_id == "static_geo_qeb_adapt_vqe":
            assert row["geo_metric_floor"] == 1e-8
            assert row["geo_selector_mode"] == "qeb_pool_projected_natural_gradient"
            assert row["geo_metric_rank_last"] is not None
            assert row["metric_operator_probe_count_proxy"] == 1
            assert row["geo_selection_with_replacement"] is True
            assert row["geo_immediate_repeat_blocked"] is True
            hist = row["adapt_history"][0]
            assert len(hist["selector_metric_candidate_labels"]) == hist["candidate_count_scored"]
            assert hist["selector_metric_probe_count"] == 1
            assert hist["qngd_metric_event_blocks"]
            block = hist["qngd_metric_event_blocks"][0]
            assert block["metric_operator_probe_count"] == (
                block["metric_eval_count"] * block["metric_pair_count_per_eval"]
            )
        elif algorithm_id == "static_geo_adapt_vqe":
            assert row["geo_metric_floor"] == 1e-8
            assert row["geo_selector_mode"] == "full_pool_projected_natural_gradient"
            assert row["geo_metric_rank_last"] is not None
            assert row["metric_operator_probe_count_proxy"] == 1
            assert row["geo_selection_with_replacement"] is True
            assert row["geo_immediate_repeat_blocked"] is True
            hist = row["adapt_history"][0]
            assert len(hist["selector_metric_candidate_labels"]) == hist["candidate_count_scored"]
            assert hist["selector_metric_probe_count"] == 1
        elif algorithm_id == "static_pos_geo_adapt_vqe":
            assert row["geo_metric_floor"] == 1e-8
            assert row["geo_selector_mode"].startswith("full_meta_pool_projected_natural_gradient_position_optimized")
            assert row["geo_metric_rank_last"] is not None
            assert row["metric_operator_probe_count_proxy"] >= 1
            assert row["geo_selection_with_replacement"] is True
            assert row["geo_immediate_repeat_blocked"] is True
            hist = row["adapt_history"][0]
            assert len(hist["selector_metric_candidate_labels"]) == hist["candidate_count_scored"]
            assert hist["selector_metric_probe_count"] == 1
            assert hist["qngd_metric_event_blocks"]
            block = hist["qngd_metric_event_blocks"][0]
            assert block["block_kind"] == "pos_geo_position_trial_qngd_metric"
            assert block["metric_operator_probe_count"] == (
                block["metric_eval_count"] * block["metric_pair_count_per_eval"]
            )
        else:
            assert row["metric_operator_probe_count_proxy"] == 0


def test_geo_qeb_run_loop_allows_non_immediate_duplicate_occurrences(monkeypatch, tmp_path: Path) -> None:
    base_candidate = variants.build_pairwise_qubit_excitation_pool(2)[0]
    alt_candidate = variants._PoolCandidate(
        label="qeb_pair_alt",
        polynomial=base_candidate.polynomial,
        support=base_candidate.support,
        pauli_labels_exyz=base_candidate.pauli_labels_exyz,
        construction=base_candidate.construction,
    )
    labels = [base_candidate.label, alt_candidate.label, base_candidate.label]
    calls: list[str | None] = []
    ctx = _fake_context([])
    ctx.exact_target = SimpleNamespace(resolve_energy=lambda ai_log=None: -1.0)

    def _fake_score(**kwargs):  # noqa: ANN003, ANN201
        previous = kwargs.get("previous_selected_label")
        if len(calls) >= len(labels):
            return []
        label = labels[len(calls)]
        calls.append(previous)
        assert label != previous
        support = [0, 1]
        return [
            {
                "label": label,
                "support": support,
                "pauli_labels_exyz": [],
                "gradient": 1.0,
                "abs_gradient": 1.0,
                "metric_variance": 1.0,
                "selector_score": 1.0,
                "geo_metric_rank": 1,
                "geo_metric_condition": 1.0,
                "geo_metric_regularization": 1e-8,
                "geo_metric_offdiag_norm": 0.0,
                "geo_natural_step": 1.0,
                "geo_projected_residual_force": 1.0,
                "geo_projected_residual_score": 1.0,
                "geo_natural_step_fs_norm": 1.0,
                "geo_natural_step_l2_norm": 1.0,
                "geo_max_abs_natural_step": 1.0,
                "geo_selector_mode": "qeb_pool_projected_natural_gradient",
            }
        ]

    def _fake_qngd(**kwargs):  # noqa: ANN003, ANN201
        theta = np.asarray(kwargs["x0"], dtype=float).reshape(-1) + 0.01
        return theta, float(3 - len(theta)), {
            "nfev": 1,
            "nit": 1,
            "success": True,
            "message": "ok",
            "optimizer": "exact_bench_qngd:logical_shared_metric_backtracking",
        }

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: False)
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: ctx)
    monkeypatch.setattr(
        variants,
        "build_pairwise_qubit_excitation_pool",
        lambda num_qubits, *, max_terms=variants._POOL_TERM_CAP: (base_candidate, alt_candidate),
    )
    monkeypatch.setattr(variants, "_score_candidates", _fake_score)
    monkeypatch.setattr(variants, "_optimize_selected_qngd", _fake_qngd)
    monkeypatch.setattr(variants, "sector_probability", lambda context, psi: {"sector_probability": 1.0})

    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_geo_qeb_adapt_vqe",
        output_dir=tmp_path,
        max_adapt_iterations=3,
        gradient_threshold=0.0,
    )

    row = payload["rows"][0]
    assert row["selected_operators"] == labels
    assert row["selected_operator_count"] == 3
    assert row["num_parameters"] == 3
    assert row["selected_label_counts"] == {base_candidate.label: 2, alt_candidate.label: 1}
    assert calls == [None, base_candidate.label, alt_candidate.label]


def test_geo_qeb_qngd_failure_is_quality_nonpassing_and_stops(monkeypatch, tmp_path: Path) -> None:
    candidate = variants.build_pairwise_qubit_excitation_pool(2)[0]
    ctx = _fake_context([])
    ctx.exact_target = SimpleNamespace(resolve_energy=lambda ai_log=None: -1.0)

    def _fake_score(**kwargs):  # noqa: ANN003, ANN201
        return [
            {
                "label": candidate.label,
                "support": [0, 1],
                "pauli_labels_exyz": [],
                "gradient": 1.0,
                "abs_gradient": 1.0,
                "metric_variance": 1.0,
                "selector_score": 1.0,
                "geo_metric_rank": 1,
                "geo_metric_condition": 1.0,
                "geo_metric_regularization": 1e-8,
                "geo_metric_offdiag_norm": 0.0,
                "geo_natural_step": 1.0,
                "geo_projected_residual_force": 1.0,
                "geo_projected_residual_score": 1.0,
                "geo_natural_step_fs_norm": 1.0,
                "geo_natural_step_l2_norm": 1.0,
                "geo_max_abs_natural_step": 1.0,
                "geo_selector_mode": "qeb_pool_projected_natural_gradient",
            }
        ]

    def _failed_qngd(**kwargs):  # noqa: ANN003, ANN201
        return np.asarray(kwargs["x0"], dtype=float).reshape(-1), 0.0, {
            "nfev": 1,
            "nit": 0,
            "success": False,
            "message": "qngd_line_search_failed",
            "optimizer": "exact_bench_qngd:logical_shared_metric_backtracking",
            "qngd_accepted_step_count": 0,
            "qngd_energy_decrease_total": 0.0,
            "qngd_metric_operator_probe_count_total": 1,
        }

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: False)
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _fake_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: ctx)
    monkeypatch.setattr(
        variants,
        "build_pairwise_qubit_excitation_pool",
        lambda num_qubits, *, max_terms=variants._POOL_TERM_CAP: (candidate,),
    )
    monkeypatch.setattr(variants, "_score_candidates", _fake_score)
    monkeypatch.setattr(variants, "_optimize_selected_qngd", _failed_qngd)
    monkeypatch.setattr(variants, "sector_probability", lambda context, psi: {"sector_probability": 1.0})

    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_geo_qeb_adapt_vqe",
        output_dir=tmp_path,
        max_adapt_iterations=3,
        gradient_threshold=0.0,
    )

    row = payload["rows"][0]
    assert payload["status"] == "completed_quality_nonpassing"
    assert row["status"] == "quality_nonpassing"
    assert row["quality_gate_reason"] == "qngd_optimizer_failed"
    assert row["adapt_stop_reason"] == "qngd_optimizer_failed"
    assert row["adapt_num_iterations"] == 1


def test_runner_failure_path_emits_normalized_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(
        variants,
        "_spec_by_case_id",
        lambda family, case_id, algorithm_id: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_geo_qubit_adapt_vqe",
        output_dir=tmp_path,
    )

    assert payload["status"] == "failed"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["guardrails"]["phase3_controller_called"] is False
    assert payload["guardrails"]["phase3_emulation"] is False
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "rows.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "metrics_proxy_summary.json").exists()


def _fake_canonical_append_result() -> SimpleNamespace:
    parent = SimpleNamespace(
        count=123,
        ordered_pool_sha256="1" * 64,
        to_dict=lambda: {
            "count": 123,
            "ordered_pool_sha256": "1" * 64,
        },
    )
    executable = SimpleNamespace(
        count=102,
        ordered_pool_sha256="2" * 64,
        to_dict=lambda: {
            "count": 102,
            "ordered_pool_sha256": "2" * 64,
        },
    )
    protocol = SimpleNamespace(
        sha256="3" * 64,
        candidate_representation="macro_generator_v1",
        optimizer_maxiter=200,
        to_dict=lambda: {
            "schema": "paper_i_append_adapt_resolved_protocol_v1",
            "sha256": "3" * 64,
            "candidate_representation": "macro_generator_v1",
            "optimizer_maxiter": 200,
        },
    )
    execution = {
        "algorithm_id": "paper_i_append_adapt_v1",
        "selector_identity": (
            "append_adapt_largest_absolute_commutator_gradient_v1"
        ),
        "selector_source_id": (
            "generic_static_full_meta_largest_absolute_commutator_gradient_v1"
        ),
        "accepted_operator_labels": ["macro-a", "macro-b"],
        "logical_theta": [0.1, 0.2],
        "runtime_theta": [0.1, 0.2],
        "final_energy": -1.25,
        "controller_rounds_completed": 2,
        "stop_reason": "maximum_controller_rounds",
        "history": [
            {
                "controller_round": 1,
                "candidate_count_scored": 102,
                "energy_after": -1.0,
                "optimizer": {"success": True},
            },
            {
                "controller_round": 2,
                "candidate_count_scored": 102,
                "energy_after": -1.25,
                "optimizer": {"success": True},
            },
        ],
        "estimator_accounting": {
            "components": {
                "N_H_outer": 2,
                "N_H_refit": 4,
                "N_grad": 207,
                "N_metric": 4,
            },
            "S_alg": 217,
            "closed_occurrence_reconciliation": True,
        },
        "compiled_resources": {
            "compiled_circuit_stats_status": "ok",
            "compiled_resource_source_kind": (
                "paper_i_append_adapt_terminal_ansatz_v1"
            ),
            "compiled_resource_qiskit_validated": True,
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "qiskit_version": "test-qiskit",
            "compiled_depth_total": 12,
            "compiled_count_2q_total": 8,
            "compiled_op_counts": {"cx": 8, "rz": 2},
        },
        "candidate_geometry_chart": (
            "exact_ordered_insertion_zero_angle_v1"
        ),
        "accepted_refit_scope": "full_ansatz_v1",
        "accepted_refit_coordinate_chart": (
            "supported_fs_whitened_fixed_v1"
        ),
    }
    result = SimpleNamespace(
        protocol=protocol,
        selector_identity=(
            "append_adapt_largest_absolute_commutator_gradient_v1"
        ),
        parent_inventory=parent,
        executable_pool=executable,
        result_payload=execution,
    )
    result.to_dict = lambda: {
        "schema": "paper_i_append_adapt_result_v1",
        "protocol": protocol.to_dict(),
        "selector_identity": result.selector_identity,
        "parent_inventory": parent.to_dict(),
        "executable_pool": executable.to_dict(),
        "result_payload": execution,
    }
    return result


def test_paper_i_append_comparison_dispatches_to_canonical_facade(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case_id = TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS[
        "weak-weak"
    ]
    spec = SimpleNamespace(
        benchmark_id=case_id,
        family="hh",
        base_pipeline_args=("--problem", "hh", "--L", "2"),
        split="paper_i_completion",
        tags=(),
        features=None,
        exact_reference_n_ph_max=3,
    )
    context = SimpleNamespace(
        family_key="hh",
        request=SimpleNamespace(num_sites=2, n_ph_max=3),
        layout=SimpleNamespace(total_qubits=8),
        exact_target=SimpleNamespace(resolve_energy=lambda ai_log=None: -1.5),
    )
    captured: dict[str, object] = {}

    def _fake_facade(problem, request):  # noqa: ANN001, ANN202
        captured["problem"] = problem
        captured["request"] = request
        return _fake_canonical_append_result()

    monkeypatch.setattr(
        variants,
        "_spec_by_case_id",
        lambda family, requested_case, algorithm_id: spec,
    )
    monkeypatch.setattr(
        variants, "_resolve_context_from_spec", lambda resolved_spec: context
    )
    monkeypatch.setattr(variants, "run_append_adapt", _fake_facade)
    monkeypatch.setattr(
        variants,
        "_run_impl",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("canonical Append must not use the generic executor")
        ),
    )

    payload = variants.run_generic_static_adapt_variant_single(
        family="hh",
        case_id=case_id,
        algorithm_id=variants.STATIC_FULL_META_APPEND_ADAPT_VQE,
        output_dir=tmp_path,
        max_adapt_iterations=2,
        optimizer_maxiter=200,
        gradient_threshold=0.0,
        seed=7,
        generic_adapt_stop_policy="fixed_horizon_no_target_v1",
        same_cutoff_exact_gs_energy=-1.5,
        exact_reference_n_ph_max=3,
    )

    request = captured["request"]
    assert captured["problem"] is context
    assert isinstance(request, variants.AppendAdaptRequest)
    assert isinstance(request.adapter, variants.MacroCandidateAdapter)
    assert request.execution.stop.maximum_controller_rounds == 2
    assert request.observation.checkpoint.path == (
        tmp_path / "canonical_append_checkpoint.json"
    )
    assert payload["status"] == "completed", payload.get("reason")
    assert payload["algorithm_id"] == (
        variants.STATIC_FULL_META_APPEND_ADAPT_VQE
    )
    assert payload["execution_owner"] == (
        variants.PAPER_I_APPEND_COMPARISON_EXECUTION_OWNER
    )
    assert payload["guardrails"]["canonical_append_facade_called"] is True
    assert payload["guardrails"]["generic_append_executor_called"] is False
    assert payload["guardrails"]["pool_source"] == (
        "canonical_ra_adapt_macro_executable_pool"
    )
    row = payload["result"]
    assert row["parent_inventory_count"] == 123
    assert row["pool_term_count"] == 102
    assert row["selected_operators"] == ["macro-a", "macro-b"]
    assert row["S_alg"] == 217.0
    assert row["compiled_circuit_stats_status"] == "ok"
    assert row["compiled_resource_qiskit_validated"] is True
    assert row["compile_convention"] == (
        "table_i_basis_gate_transpile_v1"
    )
    assert row["qiskit_version"] == "test-qiskit"
    assert row["compiled_resources"] == (
        _fake_canonical_append_result().result_payload[
            "compiled_resources"
        ]
    )
    assert row["execution_owner"] == (
        variants.PAPER_I_APPEND_COMPARISON_EXECUTION_OWNER
    )
    assert (tmp_path / "result.json").exists()


def test_canonical_append_compile_receipt_preserves_unavailable_provenance() -> None:
    receipt = {
        "compiled_circuit_stats_status": "qiskit_transpile_failed",
        "compiled_resource_source_kind": (
            "paper_i_append_adapt_terminal_ansatz_v1"
        ),
        "compiled_resource_qiskit_validated": False,
        "compile_convention": "table_i_basis_gate_transpile_v1",
        "qiskit_version": "2.1.0",
        "reason": "fixture failure",
        "compiled_basis_gates": ["rx", "rz", "cx"],
    }
    assert (
        variants._validated_canonical_append_compile_receipt(receipt)
        == receipt
    )
    with pytest.raises(ValueError, match="omitted its reason"):
        variants._validated_canonical_append_compile_receipt(
            {**receipt, "reason": ""}
        )
    with pytest.raises(ValueError, match="omitted fields"):
        variants._validated_canonical_append_compile_receipt(
            {
                key: value
                for key, value in receipt.items()
                if key != "qiskit_version"
            }
        )


def test_geo_and_frozen_append_replay_identities_bypass_canonical_facade(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def _legacy(**kwargs):  # noqa: ANN003, ANN202
        calls.append(str(kwargs["algorithm_id"]))
        return {"status": "legacy", "algorithm_id": kwargs["algorithm_id"]}

    monkeypatch.setattr(variants, "_run_impl", _legacy)
    monkeypatch.setattr(
        variants,
        "run_append_adapt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("frozen identities must not call canonical Append")
        ),
    )
    geo = variants.run_generic_static_adapt_variant_single(
        family="hh",
        case_id=TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS[
            "weak-weak"
        ],
        algorithm_id=variants.STATIC_GEO_ADAPT_VQE,
        output_dir=tmp_path / "geo",
    )
    replay = variants.run_generic_static_adapt_variant_single(
        family="hh",
        case_id=TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS[
            "weak-weak"
        ],
        algorithm_id=variants.STATIC_FULL_META_APPEND_ADAPT_VQE,
        output_dir=tmp_path / "replay",
        max_adapt_iterations=2,
        optimizer_maxiter=200,
        gradient_threshold=0.0,
        seed=7,
        generic_adapt_stop_policy="fixed_horizon_no_target_v1",
        initial_selected_operator_labels=("historical-macro",),
    )

    assert geo == {
        "status": "legacy",
        "algorithm_id": variants.STATIC_GEO_ADAPT_VQE,
    }
    assert replay == {
        "status": "legacy",
        "algorithm_id": variants.STATIC_FULL_META_APPEND_ADAPT_VQE,
    }
    assert calls == [
        variants.STATIC_GEO_ADAPT_VQE,
        variants.STATIC_FULL_META_APPEND_ADAPT_VQE,
    ]
    blockers = variants._paper_i_append_facade_migration_blockers(
        family="hh",
        case_id=TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS[
            "weak-weak"
        ],
        algorithm_id=variants.STATIC_FULL_META_APPEND_ADAPT_VQE,
        max_adapt_iterations=2,
        optimizer_maxiter=200,
        gradient_threshold=0.0,
        seed=7,
        energy_stop_target=None,
        benchmark_decision_noise_config=None,
        selected_logical_route=None,
        selected_logical_source_json=None,
        allow_repeats=None,
        progress_jsonl_path=None,
        generic_adapt_stop_policy="fixed_horizon_no_target_v1",
        powell_maxiter_cap_policy=None,
        generic_adapt_runtime_split_mode="off",
        shared_pauli_pool_mode="off",
        initial_selected_operator_labels=("historical-macro",),
        initial_selected_operator_batches=None,
        initial_theta=None,
        initial_adapt_history=None,
        optimizer_profile=None,
        adapt_optimizer_kind=None,
        adapt_spsa_maxiter=None,
        adapt_spsa_seed=None,
        adapt_spsa_a=None,
        adapt_spsa_c=None,
        adapt_spsa_alpha=None,
        adapt_spsa_gamma=None,
        adapt_spsa_big_a=None,
        optimizer_overlay_source=None,
        hh_seed_preopt_enabled=False,
        hh_adaptive_pool_profile=None,
        hh_full_meta_class_filter_json=None,
    )
    assert blockers == ("initial_operator_replay",)

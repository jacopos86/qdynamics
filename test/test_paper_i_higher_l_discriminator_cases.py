"""Contracts for the Paper-I higher-L discriminator diagnostic profile."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipelines.exact_bench import generic_static_adapt_variants as variants
from pipelines.exact_bench.static_reference_metrics import exact_energy_for_spec
from pipelines.exact_bench.table_i_canonical_cases import (
    TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE,
    table_i_canonical_spec_by_case_id,
    table_i_executable_case_ids_by_family,
    table_i_executable_specs,
    table_i_suite_profile,
)


EXPECTED = {
    ("hh", "hh_L3_nph3_higher_l_weak_strong"): {
        "energy": -1.4689835578168593,
        "reference_hash": "23b65b343cc68dcf27671356",
        "particles": (2, 1),
        "u": 0.25,
    },
    ("hh", "hh_L3_nph3_higher_l_intermediate_strong"): {
        "energy": -0.8367272230014656,
        "reference_hash": "3f96027e3d36ac18dabb937c",
        "particles": (2, 1),
        "u": 1.25,
    },
    ("hh", "hh_L3_nph3_higher_l_strong_strong"): {
        "energy": 0.7813398458338501,
        "reference_hash": "e157d07b035c6ed09f426577",
        "particles": (2, 1),
        "u": 8.0,
    },
    ("hubbard", "hubbard_L6_higher_l_weak"): {
        "energy": -6.621747713931651,
        "reference_hash": "62c1e7829d994a2cabeb48b7",
        "particles": (3, 3),
        "u": 0.25,
    },
    ("hubbard", "hubbard_L6_higher_l_strong"): {
        "energy": -1.7680987552612777,
        "reference_hash": "bfba503ceb147c8a94634adb",
        "particles": (3, 3),
        "u": 8.0,
    },
}


def test_higher_l_profile_is_exactly_the_five_ordered_cases() -> None:
    assert table_i_suite_profile("higher_l_discriminator") == (
        TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE
    )
    by_family = table_i_executable_case_ids_by_family(
        TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE
    )
    assert by_family == {
        "hh": (
            "hh_L3_nph3_higher_l_weak_strong",
            "hh_L3_nph3_higher_l_intermediate_strong",
            "hh_L3_nph3_higher_l_strong_strong",
        ),
        "hubbard": (
            "hubbard_L6_higher_l_weak",
            "hubbard_L6_higher_l_strong",
        ),
    }
    assert tuple((spec.family, spec.benchmark_id) for spec in table_i_executable_specs(
        TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE
    )) == tuple(EXPECTED)


@pytest.mark.parametrize("family,case_id", tuple(EXPECTED))
def test_higher_l_runtime_reference_and_sector_locks(family: str, case_id: str) -> None:
    expected = EXPECTED[(family, case_id)]
    spec = table_i_canonical_spec_by_case_id(
        family,
        case_id,
        TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE,
    )
    context = variants._resolve_context_from_spec(spec)

    assert context.layout.total_qubits == 12
    assert context.request.boundary == "open"
    assert context.request.ordering == "blocked"
    assert context.request.u == expected["u"]
    assert tuple(context.sector.num_particles) == expected["particles"]
    if family == "hh":
        assert context.request.num_sites == 3
        assert context.request.n_ph_max == 3
        assert context.request.g_ep == 0.7905694150420949
        assert spec.exact_reference_n_ph_max == 3
        assert context.sector.label == "half_filled_fermion_sector"
    else:
        assert context.request.num_sites == 6
        assert context.layout.boson_qubits == 0
        assert spec.exact_reference_n_ph_max is None
        assert context.sector.label == "half_filled_spin_sector"

    exact_energy, reference_hash, _key = exact_energy_for_spec(
        spec,
        n_ph_max=int(context.request.n_ph_max),
    )
    runtime_exact = float(context.exact_target.resolve_energy(ai_log=None))
    assert exact_energy == pytest.approx(expected["energy"], rel=0.0, abs=1.0e-12)
    assert runtime_exact == pytest.approx(exact_energy, rel=0.0, abs=1.0e-12)
    assert reference_hash == expected["reference_hash"]

    reference_state = np.asarray(context.reference_state.build_state(), dtype=complex)
    assert math.isclose(float(np.vdot(reference_state, reference_state).real), 1.0, abs_tol=1e-12)
    sector = variants.sector_probability(context, reference_state)
    assert sector["sector_probability"] == pytest.approx(1.0, abs=1e-12)
    assert sector["sector_leak_flag"] is False
    assert sector["boson_truncation_leak_flag"] is False


@pytest.mark.parametrize(
    "family,case_id,parent_count,raw_count,child_count,null_count,label_hash,pool_hash",
    (
        (
            "hh",
            "hh_L3_nph3_higher_l_weak_strong",
            251,
            5554,
            725,
            13,
            "ab695760046ea49d2a4d185a97cd4016fcc7c614fdf64d572b1f4739b1c68288",
            "636b834c5c0ef389966f9d9aaa23e1567af489409ebc088ed8bf52cbc11ee259",
        ),
        (
            "hubbard",
            "hubbard_L6_higher_l_weak",
            172,
            948,
            54,
            6,
            "51806ecb80b6f04163ca4496d6b39962366c558eb8beed2377a8ffbee6a93da6",
            "fd367d46c8974aa2ed8f36be74ea6210772a13deba91adc17378b1681197eb97",
        ),
    ),
)
def test_higher_l_projected_child_pool_and_commutator_guards(
    family: str,
    case_id: str,
    parent_count: int,
    raw_count: int,
    child_count: int,
    null_count: int,
    label_hash: str,
    pool_hash: str,
) -> None:
    spec = table_i_canonical_spec_by_case_id(
        family,
        case_id,
        TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE,
    )
    context = variants._resolve_context_from_spec(spec)
    parents = variants.build_full_meta_candidate_pool(context, max_terms=None)
    children, meta = variants._expand_pool_with_shared_pauli_children(
        pool=parents,
        context=context,
        config=variants._get_config("static_full_meta_append_adapt_vqe"),
        mode="projected_singleton_children_only_v1",
        symmetry_policy="hard_guard",
        max_subset_size=1,
        max_terms=9000,
    )

    assert len(parents) == parent_count
    assert meta["projected_singleton_source_term_count"] == raw_count
    assert len(children) == child_count
    assert meta["projected_singleton_null_count"] == null_count
    assert meta["ordered_label_hash"] == label_hash
    assert meta["ordered_pool_hash"] == pool_hash
    assert all(child.runtime_split_representation == "projected_singleton_child" for child in children)
    assert all(child.parent_label for child in children)
    for child in children:
        gate = child.runtime_split_symmetry_gate
        assert gate is not None
        assert gate["checked"] is True
        assert gate["passed"] is True
        assert gate["commutator_l1_total"] == 0.0
        assert gate["commutator_l1_up"] == 0.0
        assert gate["commutator_l1_dn"] == 0.0
        assert gate["fixed_count_sector"]["sector_leakage_max_abs"] == 0.0

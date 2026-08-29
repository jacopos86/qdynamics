"""Tests for versioned Paper-III Hubbard--Holstein pool profiles."""

from pipelines.qse_spectra.paper_iii_problem import load_problem
from pipelines.qse_spectra.pool_profiles import (
    HH_POOL_BASE_MACRO_V1,
    HH_POOL_ENCODED_BOSON_COMPLETION_V1,
    apply_hh_pool_policy,
    encoded_boson_completion_elements,
)


def test_encoded_completion_has_declared_sector_preserving_cardinality() -> None:
    nph1 = encoded_boson_completion_elements(num_qubits=6)
    nph3 = encoded_boson_completion_elements(num_qubits=8)

    assert len(nph1) == (4**2 - 1) * 5 == 75
    assert len(nph3) == (4**4 - 1) * 5 == 1275
    for element in nph3:
        label = str(element.pauli_label_exyz)
        assert len(label) == 8
        assert set(label[4:]) <= {"e", "z"}
        assert label[4:].count("z") <= 1


def test_completion_profile_preserves_base_prefix_and_has_new_digest() -> None:
    base = load_problem(
        regime="weak_weak",
        u=0.25,
        g_ep=0.353553390593,
        n_ph_max=1,
        target_roots=6,
    )
    assert apply_hh_pool_policy(base, HH_POOL_BASE_MACRO_V1) is base

    completed = apply_hh_pool_policy(
        base,
        HH_POOL_ENCODED_BOSON_COMPLETION_V1,
    )
    assert completed.basis[: len(base.basis)] == base.basis
    assert completed.costs[: len(base.costs)] == base.costs
    assert completed.resources[: len(base.resources)] == base.resources
    assert len(completed.basis) == len(base.basis) + 75
    assert len(completed.costs) == len(completed.basis)
    assert len(completed.resources) == len(completed.basis)
    assert completed.pool_digest != base.pool_digest
    assert (
        apply_hh_pool_policy(base, HH_POOL_ENCODED_BOSON_COMPLETION_V1)
        is completed
    )

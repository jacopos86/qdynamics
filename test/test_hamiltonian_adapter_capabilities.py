from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipelines.static_adapt.builders.problem_registry import (
    HamiltonianFamilyCapabilities,
    ProblemRequest,
    available_problem_keys,
    get_problem_family_spec,
    resolve_problem_context,
)
from pipelines.time_dynamics.adapters.hamiltonian import (
    BOSON_CHAIN_FAMILIES,
    DRIVEN_HAMILTONIAN_FLOW_FAMILIES,
    HAMILTONIAN_FLOW_FAMILIES,
    SPINFUL_LATTICE_FAMILIES,
    SPINLESS_LATTICE_FAMILIES,
    adapter_for_resolved_problem,
)
from pipelines.time_dynamics.adapters.observables import (
    observable_measurement_bundle_for_problem,
)


def _request(problem_key: str) -> ProblemRequest:
    return ProblemRequest(
        problem_key=str(problem_key),
        num_sites=1 if str(problem_key) == "spin_boson" else 2,
        t=1.0,
        u=2.0,
        dv=0.1,
        omega0=1.0,
        g_ep=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
        v_nn=0.3,
        t_prime=0.1,
    )


def test_registered_problem_families_expose_typed_capabilities() -> None:
    for problem_key in available_problem_keys():
        capabilities = get_problem_family_spec(problem_key).capabilities
        assert isinstance(capabilities, HamiltonianFamilyCapabilities)
        assert capabilities.primary_density_modes


def test_resolved_context_carries_registry_capabilities() -> None:
    for problem_key in available_problem_keys():
        if problem_key == "molecular_restricted_closed_shell":
            continue
        context = resolve_problem_context(_request(problem_key), hamiltonian=object())
        assert context.capabilities == get_problem_family_spec(problem_key).capabilities


def test_realtime_adapter_family_groups_are_capability_backed() -> None:
    assert HAMILTONIAN_FLOW_FAMILIES == (
        SPINFUL_LATTICE_FAMILIES | SPINLESS_LATTICE_FAMILIES | BOSON_CHAIN_FAMILIES
    )
    assert DRIVEN_HAMILTONIAN_FLOW_FAMILIES == HAMILTONIAN_FLOW_FAMILIES
    for family_key in HAMILTONIAN_FLOW_FAMILIES:
        adapter = adapter_for_resolved_problem(SimpleNamespace(family_key=family_key))
        assert adapter.supports_hamiltonian_flow_projective is True
        assert adapter.supports_driven_realtime is True
        assert adapter.drive_operator_kind is not None


def test_realtime_adapter_preserves_hh_default_and_molecular_unsupported_runtime() -> None:
    assert adapter_for_resolved_problem(None).family_key == "hh"
    assert adapter_for_resolved_problem(None).observable_kind == "hh_spinful_boson"

    molecular = adapter_for_resolved_problem(
        SimpleNamespace(family_key="molecular_restricted_closed_shell")
    )
    assert molecular.observable_kind == "unsupported"
    assert molecular.supports_measurement_observables is False
    assert molecular.drive_operator_kind is None


def test_adapter_capabilities_match_registry_for_every_family() -> None:
    for problem_key in available_problem_keys():
        assert (
            adapter_for_resolved_problem(SimpleNamespace(family_key=problem_key)).capabilities
            == get_problem_family_spec(problem_key).capabilities
        )


def test_molecular_realtime_observables_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="Unsupported observable measurement family"):
        observable_measurement_bundle_for_problem(
            resolved_problem=SimpleNamespace(family_key="molecular_restricted_closed_shell"),
            num_sites=2,
        )

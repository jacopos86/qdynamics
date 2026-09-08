from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.problem import ProblemRequest, RegisterBlockSpec, RegisterLayoutSpec
from pipelines.static_adapt.builders.problem_registry import resolve_problem_context
from pipelines.time_dynamics.ap_mclachlan.observables import (
    build_site_doublon_observable_plan,
    observable_row_fields,
)


def _hubbard_problem(ordering: str = "blocked"):
    return resolve_problem_context(
        ProblemRequest(
            problem_key="hubbard", num_sites=2, t=1.0, u=4.0, dv=0.0,
            omega0=1.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
            ordering=ordering, boundary="periodic",
        )
    )


@pytest.mark.parametrize("ordering,left_doublon,right_doublon", [
    ("blocked", 5, 10),
    ("interleaved", 3, 12),
])
@pytest.mark.parametrize("state_kind", ["left_doublon", "separated", "doublon_superposition"])
def test_hubbard_hand_states(ordering, left_doublon, right_doublon, state_kind):
    problem = _hubbard_problem(ordering)
    assert problem.capabilities.observable_kind == "spinful_lattice"
    assert problem.layout.fermion_qubits == problem.layout.total_qubits == 4
    plan = build_site_doublon_observable_plan(problem, dimension=16)
    assert plan is not None
    psi = np.zeros(16, dtype=complex)
    if state_kind == "left_doublon":
        psi[left_doublon] = 2.0  # The reporting kernel normalizes the supplied ray.
        up, dn, occupation, doublon, staggered, primary = [1, 0], [1, 0], [2, 0], 1, 1, 2
    elif state_kind == "separated":
        psi[9] = 1.0  # up at site 0, down at site 1 in either ordering.
        up, dn, occupation, doublon, staggered, primary = [1, 0], [0, 1], [1, 1], 0, 0, 0
    else:
        psi[left_doublon] = 1.0
        psi[right_doublon] = 1j
        # One doublon is certain, although the product of mean spin occupations
        # summed over sites would incorrectly give one half.
        up, dn, occupation, doublon, staggered, primary = [.5, .5], [.5, .5], [1, 1], 1, 0, 0
    row = observable_row_fields(psi, plan=plan)
    assert row["n_up_site"] == pytest.approx(up)
    assert row["n_dn_site"] == pytest.approx(dn)
    assert row["site_occupations"] == pytest.approx(occupation)
    assert row["doublon"] == pytest.approx(doublon)
    assert row["staggered"] == pytest.approx(staggered)
    assert row["primary_density"] == pytest.approx(primary)
    assert row["observable_family"] == "spinful_lattice_site_doublon"
    assert row["observable_schema"] == "spinful_lattice_site_doublon_observables_v1"


def test_hh_keeps_labels_and_respects_fermion_block_offset():
    problem = SimpleNamespace(
        family_key="hh",
        capabilities=SimpleNamespace(observable_kind="hh_spinful_boson"),
        request=SimpleNamespace(num_sites=2, ordering="blocked"),
        layout=RegisterLayoutSpec(
            total_qubits=6, fermion_qubits=4, boson_qubits=2,
            ordering="blocked", boson_encoding="binary",
            blocks=(RegisterBlockSpec("boson", "boson", 0, 2),
                    RegisterBlockSpec("fermion", "fermion", 2, 6)),
        ),
    )
    psi = np.zeros(64, dtype=complex)
    psi[(5 << 2) | 3] = 1.0  # Both boson bits occupied, plus the site-0 doublon.
    row = observable_row_fields(
        psi, plan=build_site_doublon_observable_plan(problem, dimension=64),
    )
    assert row["site_occupations"] == pytest.approx([2.0, 0.0])
    assert row["doublon"] == pytest.approx(1.0)
    assert row["observable_family"] == "hh_spinful_site_doublon"
    assert row["observable_schema"] == "hh_site_doublon_observables_v1"


@pytest.mark.parametrize("family,kind", [
    ("spinless_tv", "spinless_lattice"),
    ("bose_hubbard", "boson_chain"),
    ("spin_boson", "spin_boson"),
    ("toy", "unsupported"),
])
def test_unsupported_families_still_have_no_site_doublon_plan(family, kind):
    problem = SimpleNamespace(family_key=family, capabilities=SimpleNamespace(observable_kind=kind))
    assert build_site_doublon_observable_plan(problem, dimension=16) is None
    assert observable_row_fields(np.zeros(16), plan=None) == {}


def test_hubbard_dimension_and_state_guards():
    problem = _hubbard_problem()
    with pytest.raises(ValueError, match="dimension does not match resolved register layout"):
        build_site_doublon_observable_plan(problem, dimension=8)
    plan = build_site_doublon_observable_plan(problem, dimension=16)
    assert plan is not None
    with pytest.raises(ValueError, match="state dimension mismatch"):
        plan.evaluate(np.ones(8))
    with pytest.raises(ValueError, match="positive finite norm"):
        plan.evaluate(np.zeros(16))

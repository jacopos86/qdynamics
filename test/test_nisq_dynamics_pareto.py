#!/usr/bin/env python3
"""Tests for the NISQ dynamics Pareto explorer.

Validates:
  1. Hamiltonian construction + term counting
  2. Logical circuit cost calculations
  3. Term pruning correctness
  4. Suzuki-2 propagation accuracy vs exact
  5. Pareto front extraction
  6. End-to-end sweep (small, fast)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.nisq_dynamics_pareto import (
    build_hh_dynamics_hamiltonian,
    evolve_exact,
    evolve_suzuki2,
    logical_circuit_cost,
    prune_hamiltonian_terms,
    reorder_terms_by_weight,
    reorder_terms_by_qubit_locality,
    run_pareto_sweep,
    _cnot_count_for_label,
    _pauli_weight,
    _compute_pareto_front,
)
from src.quantum.pauli_actions import compile_pauli_action_exyz
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hh_L2():
    """Build L=2 open HH Hamiltonian (6 qubits)."""
    return build_hh_dynamics_hamiltonian(
        L=2, t_hop=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=1, boundary="open",
    )


@pytest.fixture(scope="module")
def psi0_L2():
    """HF reference state for L=2."""
    return hubbard_holstein_reference_state(
        dims=2, n_ph_max=1, boson_encoding="binary", indexing="blocked",
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestPauliWeight:
    def test_identity(self):
        assert _pauli_weight("eeeeee") == 0

    def test_single_z(self):
        assert _pauli_weight("eezee") == 1

    def test_weight_2(self):
        assert _pauli_weight("xyeeee") == 2

    def test_full_weight(self):
        assert _pauli_weight("xyzxyz") == 6


class TestCNOTCount:
    def test_identity(self):
        assert _cnot_count_for_label("eeeeee") == 0

    def test_single_qubit(self):
        assert _cnot_count_for_label("eezee") == 0

    def test_weight_2(self):
        # 2*(2-1) = 2
        assert _cnot_count_for_label("xyeeee") == 2

    def test_weight_4(self):
        # 2*(4-1) = 6
        assert _cnot_count_for_label("xyzxee") == 6


class TestHamiltonianConstruction:
    def test_L2_open_nq(self, hh_L2):
        assert hh_L2["nq"] == 6

    def test_L2_open_term_count(self, hh_L2):
        # L=2 open HH: 17 terms (1 identity + 6 weight-1 + 10 weight-2)
        assert len(hh_L2["ordered_labels"]) == 17

    def test_hermitian(self, hh_L2):
        hmat = hh_L2["hmat"]
        assert np.allclose(hmat, hmat.conj().T, atol=1e-12)

    def test_ground_state_energy(self, hh_L2):
        # Ground state energy should be a reasonable negative or small positive value
        gs_energy = hh_L2["evals"][0]
        assert math.isfinite(gs_energy)


class TestLogicalCircuitCost:
    def test_full_L2(self, hh_L2):
        cost = logical_circuit_cost(hh_L2["ordered_labels"], trotter_steps=1)
        # 10 weight-2 terms × 2 CNOTs × 2 (fwd+rev) = 40 per step
        assert cost["cnots_per_step"] == 40
        assert cost["total_cnots"] == 40
        assert cost["trotter_steps"] == 1
        assert cost["num_terms"] == 17

    def test_scaling_with_steps(self, hh_L2):
        cost1 = logical_circuit_cost(hh_L2["ordered_labels"], trotter_steps=1)
        cost5 = logical_circuit_cost(hh_L2["ordered_labels"], trotter_steps=5)
        assert cost5["total_cnots"] == 5 * cost1["total_cnots"]


class TestTermPruning:
    def test_no_pruning(self, hh_L2):
        labels, coeffs = prune_hamiltonian_terms(
            hh_L2["ordered_labels"], hh_L2["coeff_map"], threshold=0.0,
        )
        assert len(labels) == len(hh_L2["ordered_labels"])

    def test_aggressive_pruning_keeps_identity(self, hh_L2):
        labels, coeffs = prune_hamiltonian_terms(
            hh_L2["ordered_labels"], hh_L2["coeff_map"], threshold=100.0,
        )
        # Only identity (weight=0) should survive
        for label in labels:
            assert _pauli_weight(label) == 0

    def test_moderate_pruning_reduces_terms(self, hh_L2):
        labels_full = hh_L2["ordered_labels"]
        labels_pruned, _ = prune_hamiltonian_terms(
            labels_full, hh_L2["coeff_map"], threshold=0.3,
        )
        assert len(labels_pruned) < len(labels_full)
        assert len(labels_pruned) > 0


class TestReordering:
    def test_weight_sorted_preserves_terms(self, hh_L2):
        reordered = reorder_terms_by_weight(hh_L2["ordered_labels"])
        assert sorted(reordered) == sorted(hh_L2["ordered_labels"])

    def test_weight_sorted_ordering(self, hh_L2):
        reordered = reorder_terms_by_weight(hh_L2["ordered_labels"])
        weights = [_pauli_weight(l) for l in reordered]
        assert weights == sorted(weights)

    def test_qubit_local_preserves_terms(self, hh_L2):
        reordered = reorder_terms_by_qubit_locality(hh_L2["ordered_labels"])
        assert sorted(reordered) == sorted(hh_L2["ordered_labels"])


# ---------------------------------------------------------------------------
# Propagation tests
# ---------------------------------------------------------------------------

class TestSuzuki2Propagation:
    def test_identity_at_t0(self, hh_L2, psi0_L2):
        """At t=0, Trotter state should equal initial state."""
        compiled = {lbl: compile_pauli_action_exyz(lbl, hh_L2["nq"])
                    for lbl in hh_L2["ordered_labels"]}
        psi = evolve_suzuki2(
            psi0_L2, hh_L2["ordered_labels"], hh_L2["coeff_map"],
            compiled, t_final=0.0, trotter_steps=1,
        )
        assert abs(np.vdot(psi0_L2, psi)) > 1.0 - 1e-10

    def test_short_time_high_fidelity(self, hh_L2, psi0_L2):
        """At short time with many Trotter steps, fidelity should be high."""
        compiled = {lbl: compile_pauli_action_exyz(lbl, hh_L2["nq"])
                    for lbl in hh_L2["ordered_labels"]}
        t_final = 0.5
        psi_trot = evolve_suzuki2(
            psi0_L2, hh_L2["ordered_labels"], hh_L2["coeff_map"],
            compiled, t_final=t_final, trotter_steps=10,
        )
        psi_exact = evolve_exact(
            psi0_L2, hh_L2["hmat"], t_final, hh_L2["evals"], hh_L2["evecs"],
        )
        fidelity = abs(np.vdot(psi_exact, psi_trot)) ** 2
        assert fidelity > 0.99, f"Fidelity {fidelity} too low for t=0.5, 10 steps"

    def test_fewer_steps_lower_fidelity(self, hh_L2, psi0_L2):
        """Fewer Trotter steps should give lower fidelity than more."""
        compiled = {lbl: compile_pauli_action_exyz(lbl, hh_L2["nq"])
                    for lbl in hh_L2["ordered_labels"]}
        t_final = 1.0

        psi_exact = evolve_exact(
            psi0_L2, hh_L2["hmat"], t_final, hh_L2["evals"], hh_L2["evecs"],
        )

        fid_1 = abs(np.vdot(psi_exact, evolve_suzuki2(
            psi0_L2, hh_L2["ordered_labels"], hh_L2["coeff_map"],
            compiled, t_final=t_final, trotter_steps=1,
        ))) ** 2

        fid_5 = abs(np.vdot(psi_exact, evolve_suzuki2(
            psi0_L2, hh_L2["ordered_labels"], hh_L2["coeff_map"],
            compiled, t_final=t_final, trotter_steps=5,
        ))) ** 2

        assert fid_5 > fid_1, f"5 steps ({fid_5}) should beat 1 step ({fid_1})"

    def test_normalization_preserved(self, hh_L2, psi0_L2):
        """Trotter evolution should preserve norm."""
        compiled = {lbl: compile_pauli_action_exyz(lbl, hh_L2["nq"])
                    for lbl in hh_L2["ordered_labels"]}
        psi = evolve_suzuki2(
            psi0_L2, hh_L2["ordered_labels"], hh_L2["coeff_map"],
            compiled, t_final=2.0, trotter_steps=3,
        )
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10


class TestParetoFront:
    def test_empty_input(self):
        assert _compute_pareto_front([]) == []

    def test_single_point(self):
        rows = [{"total_cnots": 40, "fidelity_min": 0.5, "total_sq": 10,
                 "n_terms": 17, "trotter_steps": 1, "fidelity_final": 0.5,
                 "energy_error_final": 0.1, "variant_label": "test"}]
        pareto = _compute_pareto_front(rows)
        assert len(pareto) == 1

    def test_dominated_point_excluded(self):
        rows = [
            {"total_cnots": 40, "fidelity_min": 0.9, "total_sq": 10,
             "n_terms": 17, "trotter_steps": 1, "fidelity_final": 0.9,
             "energy_error_final": 0.1, "variant_label": "cheap_good"},
            {"total_cnots": 80, "fidelity_min": 0.5, "total_sq": 20,
             "n_terms": 17, "trotter_steps": 2, "fidelity_final": 0.5,
             "energy_error_final": 0.5, "variant_label": "expensive_bad"},
        ]
        pareto = _compute_pareto_front(rows)
        # expensive_bad is dominated: more CNOTs AND worse fidelity
        assert len(pareto) == 1
        assert pareto[0]["variant_label"] == "cheap_good"

    def test_pareto_preserves_tradeoffs(self):
        rows = [
            {"total_cnots": 40, "fidelity_min": 0.5, "total_sq": 10,
             "n_terms": 17, "trotter_steps": 1, "fidelity_final": 0.5,
             "energy_error_final": 0.3, "variant_label": "cheap"},
            {"total_cnots": 80, "fidelity_min": 0.9, "total_sq": 20,
             "n_terms": 17, "trotter_steps": 2, "fidelity_final": 0.9,
             "energy_error_final": 0.1, "variant_label": "expensive"},
        ]
        pareto = _compute_pareto_front(rows)
        assert len(pareto) == 2


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestEndToEndSweep:
    def test_small_sweep(self):
        """End-to-end test with minimal parameters."""
        result = run_pareto_sweep(
            L=2, t_hop=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=1,
            boundary="open", t_final=0.5, num_times=6,
            trotter_steps_list=[1, 2],
            prune_thresholds=[0.0, 0.3],
            orderings=["native"],
            drive_enabled=False,
            transpile_backends=[],
        )

        assert "results" in result
        assert "pareto_front" in result
        assert "summary" in result
        assert len(result["results"]) == 4  # 2 prune × 1 ordering × 2 steps

        # All results should have fidelity info
        for r in result["results"]:
            if not r.get("skipped"):
                assert "fidelity_min" in r
                assert "energy_error_final" in r
                assert 0.0 <= r["fidelity_min"] <= 1.0

        # Pareto front should be non-empty
        assert len(result["pareto_front"]) > 0

        # Summary should have expected keys
        assert "nisq_friendly_count" in result["summary"]
        assert "pareto_size" in result["summary"]

    def test_driven_sweep(self):
        """End-to-end test with drive enabled."""
        result = run_pareto_sweep(
            L=2, t_hop=1.0, U=4.0, omega0=1.0, g=0.5, n_ph_max=1,
            boundary="open", t_final=0.5, num_times=6,
            trotter_steps_list=[1, 3],
            prune_thresholds=[0.0],
            orderings=["native"],
            drive_enabled=True,
            drive_amplitude=0.3,
            drive_omega=2.0,
            drive_tbar=3.0,
            transpile_backends=[],
        )

        assert len(result["results"]) == 2
        for r in result["results"]:
            if not r.get("skipped"):
                assert 0.0 <= r["fidelity_min"] <= 1.0


# ---------------------------------------------------------------------------
# NISQ feasibility assertions
# ---------------------------------------------------------------------------

class TestNISQFeasibility:
    """Verify key NISQ-feasibility claims from the Pareto sweep."""

    def test_2_trotter_steps_under_100_cnots(self, hh_L2):
        """2 Suzuki-2 steps on full L=2 HH should be under 100 logical CNOTs."""
        cost = logical_circuit_cost(hh_L2["ordered_labels"], trotter_steps=2)
        assert cost["total_cnots"] <= 100, (
            f"2-step Trotter has {cost['total_cnots']} CNOTs, expected ≤ 100"
        )

    def test_pruned_3_steps_under_100_cnots(self, hh_L2):
        """3 steps with pruning to ~13 terms should stay under 100 CNOTs."""
        labels_p, coeffs_p = prune_hamiltonian_terms(
            hh_L2["ordered_labels"], hh_L2["coeff_map"], threshold=0.3,
        )
        cost = logical_circuit_cost(labels_p, trotter_steps=3)
        assert cost["total_cnots"] <= 100, (
            f"3-step pruned Trotter has {cost['total_cnots']} CNOTs"
        )

    def test_2_step_static_t1_fidelity(self, hh_L2, psi0_L2):
        """2 Trotter steps at t=1 with weight_sorted should exceed 0.95 fidelity."""
        labels = reorder_terms_by_weight(hh_L2["ordered_labels"])
        compiled = {lbl: compile_pauli_action_exyz(lbl, hh_L2["nq"]) for lbl in labels}

        psi_trot = evolve_suzuki2(
            psi0_L2, labels, hh_L2["coeff_map"], compiled,
            t_final=1.0, trotter_steps=2,
        )
        psi_exact = evolve_exact(
            psi0_L2, hh_L2["hmat"], 1.0, hh_L2["evals"], hh_L2["evecs"],
        )
        fidelity = abs(np.vdot(psi_exact, psi_trot)) ** 2
        assert fidelity > 0.95, (
            f"2-step weight_sorted fidelity {fidelity:.4f} at t=1, expected > 0.95"
        )

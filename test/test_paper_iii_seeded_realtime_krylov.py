from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.exact_bench import paper_iii_seeded_realtime_krylov_campaign as campaign
from pipelines.qse_spectra.paper_iii_problem import load_problem
from pipelines.qse_spectra.seeded_realtime_krylov import (
    SEEDED_REALTIME_KRYLOV_POLICY_ID,
    SeededRealtimeKrylovConfig,
    audit_krylov_energies,
    run_seeded_realtime_krylov,
)


class _ReferenceGuard:
    """Expose only online inputs; fail if an exact excited root is requested."""

    def __init__(self, problem):
        self.num_qubits = problem.num_qubits
        self.n_ph_max = problem.n_ph_max
        self.hamiltonian = problem.hamiltonian
        self.ground = problem.ground

    @property
    def references(self):  # pragma: no cover - accessed only on a regression
        raise AssertionError("online Krylov route requested exact excited roots")


def _weak_weak_nph1():
    return load_problem(
        regime="weak_weak",
        u=0.25,
        g_ep=0.353553390593,
        n_ph_max=1,
        target_roots=7,
    )


def test_config_rejects_invalid_online_policy() -> None:
    with pytest.raises(ValueError, match="k_max"):
        SeededRealtimeKrylovConfig(k_max=1)
    with pytest.raises(ValueError, match="time_step"):
        SeededRealtimeKrylovConfig(k_max=2, time_step=0.0)
    with pytest.raises(ValueError, match="overlap_relative_cutoff"):
        SeededRealtimeKrylovConfig(k_max=2, overlap_relative_cutoff=0.0)


def test_online_route_is_reference_blind_and_deterministic() -> None:
    problem = _weak_weak_nph1()
    config = SeededRealtimeKrylovConfig(k_max=5, include_qiskit=False)
    first = run_seeded_realtime_krylov(_ReferenceGuard(problem), config=config)
    second = run_seeded_realtime_krylov(_ReferenceGuard(problem), config=config)

    assert first.policy_receipt["policy_id"] == SEEDED_REALTIME_KRYLOV_POLICY_ID
    assert first.policy_receipt["time_step"] == 0.5
    assert first.policy_receipt["time_step_reference_tuning"] is False
    assert first.source_receipt == second.source_receipt
    assert first.trace == second.trace
    assert all("max_root_abs_error" not in row for row in first.trace)
    assert all(row["resources"]["qiskit_compiled"]["status"] == "NOT_REQUESTED" for row in first.trace)


def test_fixed_dt_reproduces_established_hh_six_root_crossing() -> None:
    problem = _weak_weak_nph1()
    run = run_seeded_realtime_krylov(
        _ReferenceGuard(problem),
        config=SeededRealtimeKrylovConfig(k_max=19, include_qiskit=False),
    )
    audited = audit_krylov_energies(run, problem.references)
    crossing = next(
        row
        for row in audited
        if row["max_root_abs_error"] is not None
        and row["max_root_abs_error"] <= 1.0e-6
    )

    assert crossing["k"] == 19
    assert crossing["retained_rank"] == 14
    assert crossing["K"] == 3420.0
    assert crossing["max_root_abs_error"] < 5.0e-7
    assert len(set(crossing["matched_pencil_root_indices"])) == 6
    assert crossing["root_match_convention"].endswith("hungarian_one_to_one")


def test_one_to_one_audit_does_not_reuse_a_ritz_root_for_a_degeneracy() -> None:
    problem = load_problem(
        regime="intermediate_weak",
        u=1.25,
        g_ep=0.353553390593,
        n_ph_max=1,
        target_roots=7,
    )
    run = run_seeded_realtime_krylov(
        _ReferenceGuard(problem),
        config=SeededRealtimeKrylovConfig(k_max=30, include_qiskit=False),
    )
    audited = audit_krylov_energies(run, problem.references)
    supported = [row for row in audited if row["max_root_abs_error"] is not None]

    assert supported
    assert min(row["max_root_abs_error"] for row in supported) > 0.2
    assert all(
        len(set(row["matched_pencil_root_indices"])) == 6 for row in supported
    )


def test_single_case_campaign_keeps_qksd_pending(tmp_path: Path) -> None:
    output = tmp_path / "matrix.json"
    assert (
        campaign.main(
            [
                "--cases",
                "peierls_u8_weak",
                "--no-include-qiskit",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    case = payload["cases"]["peierls_u8_weak"]

    assert case["threshold_crossings"]["1e-06"]["status"] == "REACHED"
    assert case["reference_boundary"]["exact_excited_roots_available_to_online_route"] is False
    assert case["terminal"]["qiskit_compiled"]["status"] == "NOT_REQUESTED"
    assert case["static_spectral_audit"]["status"] == "EVALUATED"
    assert payload["comparators"]["hamiltonian_power_qksd"]["status"] == "IMPLEMENTATION_PENDING"

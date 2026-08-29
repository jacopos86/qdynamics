from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.exact_bench import (
    paper_iii_block_seeded_realtime_krylov_campaign as campaign,
)
from pipelines.qse_spectra.block_seeded_realtime_krylov import (
    BLOCK_SEEDED_REALTIME_KRYLOV_POLICY_ID,
    BlockSeededRealtimeKrylovConfig,
    audit_block_krylov_energies,
    run_block_seeded_realtime_krylov,
)
from pipelines.qse_spectra.paper_iii_problem import load_problem


class _ReferenceGuard:
    """Expose the non-oracular block inputs and reject exact-root access."""

    def __init__(self, problem):
        self.num_qubits = problem.num_qubits
        self.n_ph_max = problem.n_ph_max
        self.hamiltonian = problem.hamiltonian
        self.ground = problem.ground
        self.basis = problem.basis
        self.resources = problem.resources
        self.key = getattr(problem, "key", None)

    @property
    def references(self):  # pragma: no cover - accessed only on a regression
        raise AssertionError("online block-Krylov route requested exact roots")


def _intermediate_weak_nph1():
    return load_problem(
        regime="intermediate_weak",
        u=1.25,
        g_ep=0.353553390593,
        n_ph_max=1,
        target_roots=7,
    )


def test_config_locks_one_seed_per_target_root() -> None:
    with pytest.raises(ValueError, match="block_size"):
        BlockSeededRealtimeKrylovConfig(k_max=20, block_size=5)
    with pytest.raises(ValueError, match="k_max"):
        BlockSeededRealtimeKrylovConfig(k_max=6)
    with pytest.raises(ValueError, match="time_step"):
        BlockSeededRealtimeKrylovConfig(k_max=7, time_step=0.0)


def test_online_block_is_reference_blind_deterministic_and_charges_sources() -> None:
    problem = _intermediate_weak_nph1()
    config = BlockSeededRealtimeKrylovConfig(k_max=13, include_qiskit=False)
    first = run_block_seeded_realtime_krylov(_ReferenceGuard(problem), config=config)
    second = run_block_seeded_realtime_krylov(_ReferenceGuard(problem), config=config)

    assert first.policy_receipt["policy_id"] == BLOCK_SEEDED_REALTIME_KRYLOV_POLICY_ID
    assert first.policy_receipt["block_size"] == 6
    assert first.source_receipt["uses_exact_reference"] is False
    assert len(first.source_receipt["admitted_seed_labels"]) == 6
    assert first.source_receipt == second.source_receipt
    assert first.trace == second.trace
    assert all("max_root_abs_error" not in row for row in first.trace)
    assert first.trace[0]["resources"]["source_preparations_charged"] == 6
    assert first.trace[1]["resources"]["source_preparations_charged"] == 12
    assert first.trace[1]["resources"]["hamiltonian_steps_charged"] == 6
    assert first.trace[1]["resources"]["qiskit_compiled"]["status"] == "NOT_REQUESTED"


def test_block_route_resolves_single_seed_degeneracy_failure() -> None:
    problem = _intermediate_weak_nph1()
    run = run_block_seeded_realtime_krylov(
        _ReferenceGuard(problem),
        config=BlockSeededRealtimeKrylovConfig(k_max=97, include_qiskit=False),
    )
    audited = audit_block_krylov_energies(run, problem.references)
    crossing = next(
        row
        for row in audited
        if row["max_root_abs_error"] is not None
        and row["max_root_abs_error"] <= 1.0e-6
    )

    assert crossing["block_size"] == 6
    assert len(set(crossing["matched_pencil_root_indices"])) == 6
    assert crossing["root_match_convention"].endswith("hungarian_one_to_one")


def test_single_case_campaign_records_static_audit_and_source_costs(
    tmp_path: Path,
) -> None:
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

    assert case["policy_id"] == BLOCK_SEEDED_REALTIME_KRYLOV_POLICY_ID
    assert case["reference_boundary"]["exact_excited_roots_available_to_online_route"] is False
    assert case["resource_receipt"]["source_preparation_cost_included"] is True
    assert case["static_spectral_audit"]["status"] == "EVALUATED"
    coverage = case["post_run_source_coverage_audit"]
    assert coverage["post_run_only"] is True
    assert coverage["used_by_online_route"] is False
    assert len(coverage["source_block_overlap_weight_by_root"]) == 7

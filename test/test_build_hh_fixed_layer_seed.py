"""Regime seed runner: frozen three-layer VQE policy, hard gate, sealed outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.time_dynamics import prospective_hh_seed_worker as worker
from pipelines.time_dynamics.prospective_execution_bindings import (
    ProspectiveExecutionError,
    SEED_ENERGY_GATE,
)
from pipelines.time_dynamics.runners import build_hh_fixed_layer_seed as runner


def _anchor_identity(tmp_path: Path):
    return runner.build_seed_identity(
        regime_id="anchor_u4_g0p5_nph1",
        u=4.0,
        g_ep=0.5,
        n_ph_max=1,
        optimizer_seed=97,
        artifact_path=str(tmp_path / "seed.json"),
    )


def _computation(order, *, signed_error: float) -> worker.SeedComputation:
    exact = -1.5
    return worker.SeedComputation(
        variational_energy=exact + signed_error,
        exact_energy=exact,
        exact_state_infidelity=1.0e-9,
        runtime_payload={
            "settings": {"L": 2, "u": 4.0, "g_ep": 0.5, "n_ph_max": 1},
            "adapt_vqe": {
                "parameterization_mode": "per_pauli_term",
                "parameterization_execution_mode": "per_pauli_term",
                "num_parameters": 216,
                "optimal_point": [0.0] * 3,
            },
            "exact_energy": exact,
        },
        optimizer_receipt={"restarts": 4, "optimizer_seed": order.seed.optimizer_seed},
        architecture_binding={"policy": dict(order.architecture)},
    )


def test_identity_speaks_the_worker_model_vocabulary(tmp_path: Path) -> None:
    seed = _anchor_identity(tmp_path)
    model = worker._model_from_seed(seed)
    assert (model.u, model.g_ep, model.n_ph_max) == (4.0, 0.5, 1)
    assert (model.t, model.omega0, model.dv) == (1.0, 1.0, 0.0)
    assert (model.sector_n_up, model.sector_n_dn) == (1, 1)
    assert seed.restarts == 4 and seed.energy_gate == SEED_ENERGY_GATE
    assert seed.seed_id == "paper_ii_regime_seeds_v1::seed::anchor_u4_g0p5_nph1"


def test_non_binary_aligned_cutoff_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="binary phonon register"):
        runner.build_seed_identity(
            regime_id="bad", u=1.0, g_ep=0.5, n_ph_max=2, optimizer_seed=1,
            artifact_path=str(tmp_path / "x.json"),
        )


def test_work_order_binds_the_frozen_architecture_policy(tmp_path: Path) -> None:
    order = runner.build_work_order(_anchor_identity(tmp_path))
    assert dict(order.architecture) == worker.architecture_policy()
    assert order.architecture["policy_id"] == "three_complete_ordered_hh_hamiltonian_layers_v2"
    assert order.architecture["optimizer"]["restarts"] == 4


def test_qualified_seed_writes_sealed_artifact_and_receipt(tmp_path: Path) -> None:
    seed = _anchor_identity(tmp_path)
    out = tmp_path / "raw" / "anchor" / "seed.json"
    code = runner.build_regime_seed(
        regime_id="anchor_u4_g0p5_nph1",
        seed=seed,
        output_json=out,
        repository_root=tmp_path,
        compute=lambda order: _computation(order, signed_error=4.0e-7),
    )
    assert code == 0
    artifact = json.loads(out.read_text())
    receipt = json.loads((out.parent / "seed_receipt.json").read_text())
    assert artifact["schema"] == runner.SEED_ARTIFACT_SCHEMA_V1
    assert artifact["adapt_vqe"]["parameterization_execution_mode"] == "per_pauli_term"
    # response_scales._physical_identity requires these to be explicit.
    assert artifact["settings"]["t_prime"] == 0.0
    assert artifact["settings"]["v_nn"] == 0.0
    assert artifact["settings"]["initial_state_source"] == runner.SEED_INITIALIZER
    assert artifact["regime_id"] == "anchor_u4_g0p5_nph1"
    assert artifact["conditioning_search_used"] is False
    core = {k: v for k, v in artifact.items() if k != "seed_artifact_sha256"}
    assert artifact["seed_artifact_sha256"] == runner._canonical_sha(core)
    assert receipt["status"] == "qualified"
    assert receipt["energy_error"] == pytest.approx(4.0e-7)
    assert receipt["artifact"]["sha256"] == runner._file_sha(out)
    assert receipt["num_parameters"] == 216
    assert receipt["architecture_policy_sha256"] == worker.architecture_policy()[
        "architecture_policy_sha256"
    ]
    with pytest.raises(FileExistsError):
        runner.build_regime_seed(
            regime_id="anchor_u4_g0p5_nph1", seed=seed, output_json=out,
            repository_root=tmp_path,
            compute=lambda order: _computation(order, signed_error=4.0e-7),
        )


def test_gate_failure_exits_three_and_leaves_no_artifact(tmp_path: Path) -> None:
    seed = _anchor_identity(tmp_path)
    out = tmp_path / "raw" / "anchor" / "seed.json"

    def _failing(order):
        raise ProspectiveExecutionError(
            "seed fails the hard Delta E <= 1e-6 gate: 3.2e-05"
        )

    code = runner.build_regime_seed(
        regime_id="anchor_u4_g0p5_nph1", seed=seed, output_json=out,
        repository_root=tmp_path, compute=_failing,
    )
    assert code == runner.EXIT_UNQUALIFIED == 3
    assert not out.exists()
    receipt = json.loads((out.parent / "seed_receipt.json").read_text())
    assert receipt["status"] == "unqualified"
    assert "3.2e-05" in receipt["failure"]
    assert receipt["artifact"] is None


def test_cli_parses_the_bundle_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    def _fake_build(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runner, "build_regime_seed", _fake_build)
    code = runner.main([
        "--regime-id", "weak_strong", "--u", "0.25", "--g-ep", "0.790569415042",
        "--n-ph-max", "7", "--optimizer-seed", "53",
        "--num-sites", "2", "--t", "1.0", "--omega0", "1.0", "--dv", "0.0",
        "--output-json", str(tmp_path / "seed.json"),
    ])
    assert code == 0
    assert captured["regime_id"] == "weak_strong"
    assert captured["seed"].physical["n_ph_max"] == 7
    assert captured["seed"].optimizer_seed == 53

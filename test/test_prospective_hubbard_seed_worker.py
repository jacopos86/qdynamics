"""Focused contracts for the prospective Paper-II Hubbard-L4 seed worker."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pipelines.time_dynamics.prospective_hubbard_seed_worker as worker
import pipelines.time_dynamics.runners.build_prospective_hubbard_seed as cli
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    LANE_HUBBARD,
    SourceEpochIdentity,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    SOURCE_EPOCH_RECEIPT_V1,
    SOURCE_TREE_MANIFEST_V1,
    ProspectiveExecutionError,
    SourceEpochReceipt,
    executor_bindings,
    resolve_bootstrap_action,
)
from pipelines.time_dynamics.prospective_hubbard_seed_worker import (
    EXPECTED_LOGICAL_PARAMETER_COUNT,
    EXPECTED_SECTOR_DIMENSION,
    EXPECTED_UNIQUE_GENERATOR_COUNT,
    SEED_WORKER_SCHEMA_V1,
    SeedComputation,
    architecture_policy,
    build_work_order,
    load_work_order,
    materialize_seed,
    validate_canonical_seed_request,
    write_work_order,
)
from src.quantum.vqe_latex_python_pairs import VQEResult


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _seal(payload: dict[str, object], key: str) -> dict[str, object]:
    result = dict(payload)
    result[key] = _canonical_sha(result)
    return result


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_receipt(root: Path) -> SourceEpochReceipt:
    epoch = SourceEpochIdentity(
        lane_id=LANE_HUBBARD,
        source_epoch_id="paper-ii-hubbard-l4-seed-worker-test",
        source_revision="a" * 40,
    )
    tree_sha = "d" * 64
    manifest = root / "source/tree_manifest.json"
    _write(
        manifest,
        _seal(
            {
                "schema": SOURCE_TREE_MANIFEST_V1,
                "lane_id": epoch.lane_id,
                "source_epoch_identity_sha256": epoch.identity_sha256,
                "source_revision": epoch.source_revision,
                "source_archive_sha256": epoch.source_archive_sha256,
                "source_tree_sha256": tree_sha,
                "provenance_kind": "git_archive",
            },
            "source_tree_manifest_sha256",
        ),
    )
    receipt_path = root / "source/source_epoch_receipt.json"
    _write(
        receipt_path,
        _seal(
            {
                "schema": SOURCE_EPOCH_RECEIPT_V1,
                "source_epoch": epoch.to_json_dict(),
                "source_tree_manifest": {
                    "path": manifest.relative_to(root).as_posix(),
                    "sha256": _file_sha(manifest),
                },
                "source_tree_sha256": tree_sha,
            },
            "source_epoch_receipt_sha256",
        ),
    )
    return SourceEpochReceipt.load(
        receipt_path, repository_root=root, expected_epoch=epoch
    )


def _request_and_order(root: Path):
    source = _source_receipt(root)
    plan = plan_bootstrap_actions(
        build_identity_registry(),
        evidence=BootstrapEvidence(source_epochs={LANE_HUBBARD: source.source_epoch}),
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    action = next(
        row
        for row in plan.ready_actions
        if row.lane_id == LANE_HUBBARD and row.kind == "build_seed"
    )
    request = resolve_bootstrap_action(action)
    return request, source, build_work_order(request, source_epoch_receipt=source)


def _computation(order, *, signed_error: float = 5.0e-7) -> SeedComputation:
    exact = -2.0
    return SeedComputation(
        variational_energy=exact + signed_error,
        exact_energy=exact,
        exact_state_infidelity=2.0e-9,
        runtime_payload={
            "schema_version": SEED_WORKER_SCHEMA_V1,
            "pipeline": "prospective_hubbard_seed_worker_test_double",
            "settings": {"problem": "hubbard", "adapt_pool": "uccsd_qeb_hva_blocks"},
            "adapt_vqe": {
                "pool_type": "uccsd_qeb_hva_blocks",
                "structure_locked": True,
                "parameterization_execution_mode": "logical_shared",
            },
        },
        optimizer_receipt={
            "method": "SLSQP",
            "restarts": 4,
            "optimizer_seed": order.seed.optimizer_seed,
            "restart_summaries": [
                {"restart_index": index, "success": True} for index in range(1, 5)
            ],
        },
        architecture_binding={
            "policy": architecture_policy(),
            "generator_pool_sha256": "b" * 64,
            "unique_generator_count": EXPECTED_UNIQUE_GENERATOR_COUNT,
            "logical_parameter_count": EXPECTED_LOGICAL_PARAMETER_COUNT,
            "sector_dimension": EXPECTED_SECTOR_DIMENSION,
            "conditioning_search_used": False,
            "snapshot_conditioning_used": False,
            "qiskit_compilation_used": False,
        },
    )


def test_architecture_and_only_hubbard_seed_identity_are_frozen() -> None:
    policy = architecture_policy()
    unsigned = dict(policy)
    digest = unsigned.pop("architecture_policy_sha256")

    assert digest == _canonical_sha(unsigned)
    assert policy["generator_pool"] == "uccsd_qeb_hva_blocks"
    assert policy["generator_filter"] == "fixed_spin_sector_logical_generators_v1"
    assert policy["generator_execution"] == "grouped_exact_logical_shared_v1"
    assert policy["layer_repeats"] == 2
    assert policy["expected_unique_generator_count"] == 113
    assert policy["expected_logical_parameter_count"] == 226
    assert policy["expected_sector_dimension"] == 36
    assert policy["optimizer"] == {
        "method": "SLSQP",
        "maxiter": 1800,
        "restarts": 4,
        "zero_first_restart": True,
        "initial_point_stddev": 0.3,
        "bounds": [-np.pi, np.pi],
        "energy_backend": "legacy",
    }
    assert policy["conditioning_search_used"] is False
    assert policy["snapshot_conditioning_used"] is False
    assert policy["fs_radius_tuning_used"] is False

    seeds = tuple(
        row for row in build_identity_registry().seeds if row.lane_id == LANE_HUBBARD
    )
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.seed_id.endswith("::l4__u4__half_filled")
    assert seed.identity_sha256 == "fbb6d7b539177031970bae02769583ae168fe2900c7887578cd606d0aa86e2a6"
    assert seed.optimizer_seed == 1_293_143_593
    assert seed.restarts == 4
    assert seed.energy_gate == 1.0e-6


def test_work_order_round_trip_and_canonical_request_rejection(tmp_path: Path) -> None:
    request, _source, order = _request_and_order(tmp_path)
    path = tmp_path / "work/seed_work_order.json"
    write_work_order(path, order)

    loaded, loaded_source = load_work_order(path, repository_root=tmp_path)
    assert loaded == order
    assert loaded_source.receipt_sha256 == order.source_epoch_receipt_sha256
    assert validate_canonical_seed_request(request) == order.seed

    tampered = replace(request, inputs={**request.inputs, "initializer": "changed"})
    with pytest.raises(ProspectiveExecutionError, match="diverges"):
        validate_canonical_seed_request(tampered)

    document = json.loads(path.read_text(encoding="utf-8"))
    document["architecture_policy"]["layer_repeats"] = 1
    _write(path, document)
    with pytest.raises(ProspectiveExecutionError, match="self digest mismatch"):
        load_work_order(path, repository_root=tmp_path)


def test_materialization_writes_receipt_load_compatible_immutable_evidence(
    tmp_path: Path,
) -> None:
    _request, _source, order = _request_and_order(tmp_path)

    receipt = materialize_seed(
        order,
        repository_root=tmp_path,
        compute=lambda current: _computation(current),
    )
    repeated = materialize_seed(
        order,
        repository_root=tmp_path,
        compute=lambda current: _computation(current),
    )
    artifact_path = tmp_path / receipt.artifact.path
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert repeated == receipt
    assert receipt.energy_error == pytest.approx(5.0e-7)
    assert receipt.exact_state_infidelity == pytest.approx(2.0e-9)
    assert artifact["architecture_binding"]["policy"]["layer_repeats"] == 2
    assert artifact["architecture_binding"]["unique_generator_count"] == 113
    assert artifact["optimizer_seed"] == order.seed.optimizer_seed
    assert artifact["restarts"] == 4
    assert artifact["conditioning_search_used"] is False
    assert artifact["snapshot_conditioning_used"] is False
    assert artifact["qiskit_compilation_used"] is False

    with pytest.raises(ProspectiveExecutionError, match="overwrite immutable"):
        materialize_seed(
            order,
            repository_root=tmp_path,
            compute=lambda current: _computation(current, signed_error=4.0e-7),
        )


def test_compute_uses_core_fixed_structure_and_four_seeded_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _request, _source, order = _request_and_order(tmp_path)
    captured: dict[str, object] = {}
    config = worker._config_from_seed(order.seed)

    class FakeAnsatz:
        num_parameters = EXPECTED_LOGICAL_PARAMETER_COUNT
        layout = SimpleNamespace(runtime_parameter_count=452)

        @staticmethod
        def prepare_state(theta, psi_ref):
            captured["prepared_theta"] = np.asarray(theta, dtype=float).tolist()
            return np.asarray(psi_ref, dtype=complex)

    generators = tuple(
        SimpleNamespace(label=f"layered-generator-{index}", execution_mode="grouped_exact")
        for index in range(EXPECTED_LOGICAL_PARAMETER_COUNT)
    )
    problem = SimpleNamespace(
        config=config,
        hamiltonian=object(),
        ansatz=FakeAnsatz(),
        generator_terms=generators,
        unique_generator_count=EXPECTED_UNIQUE_GENERATOR_COUNT,
        generator_pool_sha256="b" * 64,
        sector_dimension=EXPECTED_SECTOR_DIMENSION,
        reference_state=np.asarray([1.0, 0.0], dtype=complex),
        exact_energy=-2.0,
    )
    monkeypatch.setattr(
        worker, "build_pure_hubbard_fixed_vqe_problem", lambda actual: problem
    )
    monkeypatch.setattr(
        worker,
        "_exact_sector_ground_state",
        lambda _problem: (-2.0, np.asarray([1.0, 0.0], dtype=complex)),
    )

    def _vqe_minimize(*args, **kwargs):
        captured["vqe_kwargs"] = kwargs
        return VQEResult(
            energy=-2.0 + 5.0e-7,
            theta=np.linspace(0.0, 0.1, EXPECTED_LOGICAL_PARAMETER_COUNT),
            success=True,
            message="test",
            nfev=12,
            nit=8,
            best_restart=0,
            restart_summaries=[
                {"restart_index": index, "success": True} for index in range(1, 5)
            ],
        )

    monkeypatch.setattr(worker, "vqe_minimize", _vqe_minimize)
    monkeypatch.setattr(
        worker,
        "serialize_layout",
        lambda _layout: {
            "runtime_parameter_count": 452,
            "logical_operator_count": EXPECTED_LOGICAL_PARAMETER_COUNT,
            "blocks": [],
        },
    )
    monkeypatch.setattr(
        worker,
        "expand_legacy_logical_theta",
        lambda theta, _layout: np.repeat(np.asarray(theta, dtype=float), 2),
    )

    result = worker.compute_seed(order)

    assert captured["vqe_kwargs"]["restarts"] == 4
    assert captured["vqe_kwargs"]["seed"] == order.seed.optimizer_seed
    assert captured["vqe_kwargs"]["use_initial_point_first_restart"] is True
    assert np.array_equal(
        captured["vqe_kwargs"]["initial_point"],
        np.zeros(EXPECTED_LOGICAL_PARAMETER_COUNT),
    )
    assert result.architecture_binding["unique_generator_count"] == 113
    assert result.architecture_binding["logical_parameter_count"] == 226
    assert result.architecture_binding["runtime_parameter_count"] == 452
    assert result.optimizer_receipt["best_restart"] == 1
    assert result.runtime_payload["adapt_vqe"]["parameterization_execution_mode"] == (
        "logical_shared"
    )
    assert len(result.runtime_payload["adapt_vqe"]["optimal_point"]) == 452
    assert len(result.runtime_payload["adapt_vqe"]["logical_optimal_point"]) == 226


def test_materialization_rejects_failed_energy_gate_before_writing(tmp_path: Path) -> None:
    _request, _source, order = _request_and_order(tmp_path)

    with pytest.raises(ProspectiveExecutionError, match="hard Delta E"):
        materialize_seed(
            order,
            repository_root=tmp_path,
            compute=lambda current: _computation(current, signed_error=1.1e-6),
        )
    assert not (tmp_path / order.seed.artifact_path).exists()


def test_prepare_and_mocked_run_cli_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, source, expected = _request_and_order(tmp_path)
    request_path = tmp_path / "input/request.json"
    _write(request_path, request.to_json_dict())

    assert (
        cli.main(
            [
                "prepare",
                "--repo-root",
                str(tmp_path),
                "--request-json",
                request_path.relative_to(tmp_path).as_posix(),
                "--source-epoch-receipt",
                source.receipt_path,
                "--output",
                "work/seed_work_order.json",
            ]
        )
        == 0
    )
    prepared = json.loads(capsys.readouterr().out)
    loaded, _ = load_work_order(
        tmp_path / prepared["work_order"], repository_root=tmp_path
    )
    assert prepared["status"] == "prepared_not_executed"
    assert prepared["production_submission_enabled"] is False
    assert loaded == expected
    assert not (tmp_path / expected.seed.artifact_path).exists()

    fake_receipt = SimpleNamespace(
        seed_id=expected.seed.seed_id,
        receipt_path="output/fake_receipt.json",
        receipt_sha256="c" * 64,
        energy_error=5.0e-7,
        exact_state_infidelity=2.0e-9,
    )
    monkeypatch.setattr(cli, "materialize_seed", lambda *args, **kwargs: fake_receipt)
    assert (
        cli.main(
            [
                "run",
                "--repo-root",
                str(tmp_path),
                "--work-order",
                prepared["work_order"],
            ]
        )
        == 0
    )
    completed = json.loads(capsys.readouterr().out)
    assert completed["status"] == "qualified"
    assert completed["seed_id"] == expected.seed.seed_id

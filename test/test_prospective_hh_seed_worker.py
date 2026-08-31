"""Focused contracts for the prospective Paper-II HH seed worker."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pipelines.time_dynamics.prospective_hh_seed_worker as worker
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    LANE_HH,
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
from pipelines.time_dynamics.prospective_hh_seed_worker import (
    SEED_WORKER_SCHEMA_V1,
    SeedComputation,
    architecture_policy,
    build_work_order,
    load_work_order,
    materialize_seed,
    validate_canonical_seed_request,
    write_work_order,
)
from pipelines.time_dynamics.runners.build_prospective_hh_seed import main
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_receipt(root: Path) -> SourceEpochReceipt:
    epoch = SourceEpochIdentity(
        lane_id=LANE_HH,
        source_epoch_id="paper-ii-hh-product-seed-worker-test",
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
        evidence=BootstrapEvidence(source_epochs={LANE_HH: source.source_epoch}),
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    action = next(
        row for row in plan.ready_actions if row.lane_id == LANE_HH and row.kind == "build_seed"
    )
    request = resolve_bootstrap_action(action)
    return request, source, build_work_order(request, source_epoch_receipt=source)


def _computation(order, *, signed_error: float = 5.0e-7) -> SeedComputation:
    exact = -2.0
    policy = architecture_policy()
    return SeedComputation(
        variational_energy=exact + signed_error,
        exact_energy=exact,
        exact_state_infidelity=2.0e-9,
        runtime_payload={
            "schema_version": SEED_WORKER_SCHEMA_V1,
            "pipeline": "prospective_hh_seed_worker_test_double",
            "settings": {"adapt_pool": "fixed_scaffold_locked"},
            "adapt_vqe": {"structure_locked": True},
        },
        optimizer_receipt={
            "method": "L-BFGS-B",
            "restarts": 4,
            "optimizer_seed": order.seed.optimizer_seed,
            "restart_summaries": [
                {"restart_index": index, "success": True} for index in range(1, 5)
            ],
        },
        architecture_binding={
            "policy": policy,
            "architecture_id": "test-double-three-complete-hamiltonian-layers",
            "conditioning_search_used": False,
            "snapshot_conditioning_used": False,
        },
    )


def test_architecture_and_all_six_seed_identities_are_frozen() -> None:
    policy = architecture_policy()
    unsigned = dict(policy)
    digest = unsigned.pop("architecture_policy_sha256")

    assert digest == _canonical_sha(unsigned)
    assert policy["parent_labels"] == [
        "hop_layer",
        "onsite_layer",
        "phonon_layer",
        "eph_layer",
    ]
    assert policy["layer_repeats"] == 3
    assert (
        policy["layer_relation"]
        == "three_identical_complete_ordered_hamiltonian_parent_layers"
    )
    assert policy["optimizer"]["restarts"] == 4
    assert policy["optimizer"]["gradient_backend"] == "compiled_reverse_mode_exact_v1"
    assert policy["optimizer"]["optimizer_exact_reference_used"] is False
    assert policy["conditioning_search_used"] is False
    assert policy["snapshot_conditioning_used"] is False
    assert policy["fs_radius_tuning_used"] is False

    seeds = tuple(row for row in build_identity_registry().seeds if row.lane_id == LANE_HH)
    assert len(seeds) == 6
    assert len({row.identity_sha256 for row in seeds}) == 6
    assert len({row.optimizer_seed for row in seeds}) == 6
    assert all(row.restarts == 4 for row in seeds)
    assert all(row.energy_gate == 1.0e-6 for row in seeds)


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


def test_materialization_writes_loader_compatible_immutable_evidence(
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
    assert artifact["architecture_binding"]["policy"]["layer_repeats"] == 3
    assert artifact["architecture_binding"]["policy"]["parent_labels"] == [
        "hop_layer",
        "onsite_layer",
        "phonon_layer",
        "eph_layer",
    ]
    assert artifact["optimizer_seed"] == order.seed.optimizer_seed
    assert artifact["restarts"] == 4
    assert artifact["conditioning_search_used"] is False
    assert artifact["snapshot_conditioning_used"] is False
    assert artifact["qiskit_compilation_used"] is False
    assert not list(artifact_path.parent.glob("*conditioning*"))

    with pytest.raises(ProspectiveExecutionError, match="overwrite immutable"):
        materialize_seed(
            order,
            repository_root=tmp_path,
            compute=lambda current: _computation(current, signed_error=4.0e-7),
        )


def test_compute_uses_three_complete_layers_and_four_seeded_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _request, _source, order = _request_and_order(tmp_path)
    captured: dict[str, object] = {}
    qualification_order: list[str] = []
    parent_by_atom = {
        "hop": "hop_layer",
        "onsite": "onsite_layer",
        "phonon": "phonon_layer",
        "eph": "eph_layer",
    }

    class FakePool:
        pool_key = "full_meta"
        ordered_atom_contract_sha256 = "b" * 64

        def by_id(self, atom_id: str):
            return SimpleNamespace(parent_label=parent_by_atom[atom_id])

        def to_json_dict(self):
            return {"pool_key": self.pool_key, "ordered_atom_contract_sha256": self.ordered_atom_contract_sha256}

    class FakeExecutor:
        @staticmethod
        def prepare_state(theta, psi_ref):
            captured["prepared_theta"] = np.asarray(theta, dtype=float).tolist()
            return np.asarray(psi_ref, dtype=complex)

    architecture = SimpleNamespace(
        architecture_id="three-layer-test",
        to_json_dict=lambda: {
            "architecture_id": "three-layer-test",
            "layer_count": 3,
        },
    )
    layout = SimpleNamespace(
        logical_parameter_count=2,
        blocks=(object(), object(), object()),
    )
    runtime = SimpleNamespace(
        executor=FakeExecutor(), runtime_parameter_count=2, layout=layout
    )
    resolved = SimpleNamespace(
        reference_state=SimpleNamespace(
            build_state=lambda: np.asarray([1.0, 0.0], dtype=complex)
        )
    )

    monkeypatch.setattr(worker, "_make_replay_run_cfg", lambda *args, **kwargs: object())
    monkeypatch.setattr(worker, "_build_hh_hamiltonian", lambda _cfg: object())
    monkeypatch.setattr(worker, "_problem_request_from_model", lambda _model: object())
    monkeypatch.setattr(worker, "resolve_problem_context", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(worker, "hamiltonian_matrix", lambda _poly: np.eye(2))
    def _exact_ground_reference(**kwargs):
        qualification_order.append("exact_reference")
        return -2.0, np.asarray([1.0, 0.0], dtype=complex), {"source": "test"}

    monkeypatch.setattr(worker, "_exact_ground_reference", _exact_ground_reference)
    monkeypatch.setattr(worker, "build_fixed_vqe_generator_pool", lambda **kwargs: FakePool())
    monkeypatch.setattr(
        worker,
        "parent_complete_layer",
        lambda _pool, _parents: ("hop", "onsite", "phonon", "eph"),
    )

    def _architecture_from_layers(layers, **kwargs):
        captured["layers"] = tuple(tuple(layer) for layer in layers)
        captured["architecture_kwargs"] = kwargs
        return architecture

    monkeypatch.setattr(worker, "architecture_from_layers", _architecture_from_layers)
    monkeypatch.setattr(worker, "build_architecture_runtime", lambda *args, **kwargs: runtime)

    def _minimize_seed_vqe(*args, **kwargs):
        qualification_order.append("optimizer")
        captured["vqe_kwargs"] = kwargs
        return VQEResult(
            energy=-2.0 + 5.0e-7,
            theta=np.asarray([0.1, 0.2], dtype=float),
            success=True,
            message="test",
            nfev=12,
            nit=8,
            best_restart=0,
            restart_summaries=[
                {"restart_index": index, "success": True} for index in range(1, 5)
            ],
        )

    monkeypatch.setattr(worker, "_minimize_seed_vqe", _minimize_seed_vqe)
    monkeypatch.setattr(
        worker,
        "serialize_layout",
        lambda _layout: {
            "blocks": [
                {"runtime_terms_exyz": [{"pauli_exyz": "x"}]},
                {"runtime_terms_exyz": [{"pauli_exyz": "y"}]},
            ]
        },
    )
    monkeypatch.setattr(
        worker,
        "project_runtime_theta_block_mean",
        lambda theta, _layout: np.asarray(theta, dtype=float),
    )

    result = worker.compute_seed(order)

    assert captured["layers"] == (
        ("hop", "onsite", "phonon", "eph"),
        ("hop", "onsite", "phonon", "eph"),
        ("hop", "onsite", "phonon", "eph"),
    )
    assert captured["architecture_kwargs"]["construction_notes"]["layer_repeats"] == 3
    assert captured["vqe_kwargs"]["restarts"] == 4
    assert captured["vqe_kwargs"]["optimizer_seed"] == order.seed.optimizer_seed
    assert captured["vqe_kwargs"]["optimizer"]["zero_first_restart"] is True
    assert qualification_order == ["optimizer", "exact_reference"]
    assert result.architecture_binding["policy"]["layer_repeats"] == 3
    assert result.optimizer_receipt["optimizer_exact_reference_used"] is False
    assert result.optimizer_receipt["best_restart"] == 1


def test_materialization_rejects_failed_energy_gate_before_writing(tmp_path: Path) -> None:
    _request, _source, order = _request_and_order(tmp_path)

    with pytest.raises(ProspectiveExecutionError, match="hard Delta E"):
        materialize_seed(
            order,
            repository_root=tmp_path,
            compute=lambda current: _computation(current, signed_error=1.1e-6),
        )

    assert not (tmp_path / order.seed.artifact_path).exists()


def test_prepare_cli_writes_only_a_validated_work_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request, source, expected = _request_and_order(tmp_path)
    request_path = tmp_path / "input/request.json"
    _write(request_path, request.to_json_dict())

    assert (
        main(
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
    status = json.loads(capsys.readouterr().out)
    loaded, _ = load_work_order(
        tmp_path / status["work_order"], repository_root=tmp_path
    )

    assert status["status"] == "prepared_not_executed"
    assert status["production_submission_enabled"] is False
    assert loaded == expected
    assert not (tmp_path / expected.seed.artifact_path).exists()

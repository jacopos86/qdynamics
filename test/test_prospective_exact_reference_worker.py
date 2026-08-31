"""Digest and replay contracts for prospective exact-reference workers."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pytest

import pipelines.time_dynamics.prospective_hh_seed_worker as hh_seed
import pipelines.time_dynamics.prospective_hubbard_seed_worker as hubbard_seed
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    LANE_HH,
    LANE_HUBBARD,
    SourceEpochIdentity,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_exact_reference_worker import (
    ExactReferenceComputation,
    build_work_order,
    load_work_order,
    materialize_exact_reference,
    propagation_policy,
    write_work_order,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    SOURCE_EPOCH_RECEIPT_V1,
    SOURCE_TREE_MANIFEST_V1,
    ProspectiveExecutionError,
    SourceEpochReceipt,
    executor_bindings,
    resolve_bootstrap_action,
)


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


def _source_receipt(root: Path, lane_id: str) -> SourceEpochReceipt:
    epoch = SourceEpochIdentity(
        lane_id=lane_id,
        source_epoch_id=f"{lane_id}-exact-reference-worker-test",
        source_revision="a" * 40,
    )
    tree_sha = "d" * 64
    manifest = root / f"source/{lane_id}/tree_manifest.json"
    _write(
        manifest,
        _seal(
            {
                "schema": SOURCE_TREE_MANIFEST_V1,
                "lane_id": lane_id,
                "source_epoch_identity_sha256": epoch.identity_sha256,
                "source_revision": epoch.source_revision,
                "source_archive_sha256": epoch.source_archive_sha256,
                "source_tree_sha256": tree_sha,
                "provenance_kind": "git_archive",
            },
            "source_tree_manifest_sha256",
        ),
    )
    receipt_path = root / f"source/{lane_id}/source_epoch_receipt.json"
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


def _seed_computation(order, lane_id: str):
    module = hh_seed if lane_id == LANE_HH else hubbard_seed
    nq = 6 if lane_id == LANE_HH else 8
    amplitudes = (
        {
            "000101": {"re": 2.0**-0.5, "im": 0.0},
            "001010": {"re": 2.0**-0.5, "im": 0.0},
        }
        if lane_id == LANE_HH
        # Blocked L=4 ordering: two occupied up modes and two occupied down
        # modes.  A vacuum test double would violate the canonical sector and
        # can make all response scales spuriously zero.
        else {"00110011": {"re": 1.0, "im": 0.0}}
    )
    return module.SeedComputation(
        variational_energy=-1.9999995,
        exact_energy=-2.0,
        exact_state_infidelity=2.0e-9,
        runtime_payload={
            "schema_version": module.SEED_WORKER_SCHEMA_V1,
            "pipeline": "prospective_seed_worker_exact_reference_test_double",
            "settings": {"adapt_pool": "fixed_scaffold_locked"},
            "initial_state": {
                "source": "test_qualified_seed",
                "handoff_state_kind": "prepared_state",
                "nq_total": nq,
                "amplitudes_qn_to_q0": amplitudes,
            },
            "adapt_vqe": {"structure_locked": True},
        },
        optimizer_receipt={
            "method": "test",
            "restarts": 4,
            "optimizer_seed": order.seed.optimizer_seed,
            "restart_summaries": [
                {"restart_index": index, "success": True}
                for index in range(1, 5)
            ],
        },
        architecture_binding={
            "policy": module.architecture_policy(),
            "conditioning_search_used": False,
            "snapshot_conditioning_used": False,
        },
    )


def _prepared(root: Path, lane_id: str):
    source = _source_receipt(root, lane_id)
    registry = build_identity_registry()
    executors = tuple(row.binding_id for row in executor_bindings())
    seed_plan = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(source_epochs={lane_id: source.source_epoch}),
        available_executors=executors,
    )
    seed_action = next(
        row
        for row in seed_plan.ready_actions
        if row.lane_id == lane_id and row.kind == "build_seed"
    )
    seed_request = resolve_bootstrap_action(seed_action)
    module = hh_seed if lane_id == LANE_HH else hubbard_seed
    seed_order = module.build_work_order(
        seed_request, source_epoch_receipt=source
    )
    seed_receipt = module.materialize_seed(
        seed_order,
        repository_root=root,
        compute=lambda current: _seed_computation(current, lane_id),
    )
    reference_plan = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(
            seed_artifact_sha256={seed_receipt.seed_id: seed_receipt.artifact.sha256},
            source_epochs={lane_id: source.source_epoch},
        ),
        available_executors=executors,
    )
    reference_action = next(
        row
        for row in reference_plan.ready_actions
        if row.lane_id == lane_id
        and row.kind == "build_exact_reference"
        and row.inputs["seed_id"] == seed_receipt.seed_id
    )
    request = resolve_bootstrap_action(reference_action)
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed_receipt,
    )
    return source, seed_receipt, request, order


def _exact_computation(order, *, phase: float = 0.0) -> ExactReferenceComputation:
    grid = order.reference.grid
    times = tuple(
        float(grid["start"]) + index * float(grid["step"])
        for index in range(int(grid["points"]))
    )
    energy = tuple(1.0 + 0.2 * math.sin(time + phase) for time in times)
    doublon = tuple(0.25 + 0.04 * math.sin(0.5 * time + phase) for time in times)
    occupations = tuple(
        tuple(
            1.0 + ((-1.0) ** site) * 0.1 * math.sin(time + phase)
            for site in range(int(order.reference.physical["lattice_sites"]))
        )
        for time in times
    )
    sites = int(order.reference.physical["lattice_sites"])
    if order.reference.lane_id == LANE_HUBBARD:
        state_width = 1 << (2 * sites)
    else:
        levels = int(order.reference.physical["n_ph_max"]) + 1
        phonon_qubits = max(1, (levels - 1).bit_length())
        state_width = 1 << (2 * sites + sites * phonon_qubits)
    states = tuple(
        (
            complex(math.cos(0.1 * time), 0.0),
            complex(math.sin(0.1 * time), 0.0),
            *(0.0j for _ in range(state_width - 2)),
        )
        for time in times
    )
    return ExactReferenceComputation(times, energy, doublon, occupations, states)


@pytest.mark.parametrize("lane_id", [LANE_HH, LANE_HUBBARD])
def test_work_order_round_trip_binds_canonical_seed_and_source(
    tmp_path: Path, lane_id: str
) -> None:
    source, seed, request, order = _prepared(tmp_path, lane_id)
    path = tmp_path / f"work/{lane_id}/exact_reference_work_order.json"
    write_work_order(path, order)

    loaded, loaded_source, loaded_seed = load_work_order(
        path, repository_root=tmp_path
    )
    assert loaded == order
    assert loaded_source == source
    assert loaded_seed == seed
    assert loaded.seed_artifact_sha256 == request.inputs["seed_artifact_sha256"]
    assert loaded.propagation["implementation"] == (
        "exponential_midpoint_magnus2_order2"
    )
    assert loaded.propagation["qiskit_compilation"] is False

    changed = replace(
        request,
        inputs={**request.inputs, "seed_artifact_sha256": "f" * 64},
    )
    with pytest.raises(ProspectiveExecutionError, match="canonical planner"):
        build_work_order(
            changed, source_epoch_receipt=source, seed_receipt=seed
        )


def test_materialization_is_typed_atomic_and_idempotent(tmp_path: Path) -> None:
    _source, _seed, _request, order = _prepared(tmp_path, LANE_HH)
    receipt = materialize_exact_reference(
        order,
        repository_root=tmp_path,
        compute=lambda current: _exact_computation(current),
    )
    repeated = materialize_exact_reference(
        order,
        repository_root=tmp_path,
        compute=lambda current: _exact_computation(current),
    )

    assert repeated == receipt
    assert receipt.reference_id == order.reference.reference_id
    assert receipt.seed_artifact_sha256 == order.seed_artifact_sha256
    assert receipt.materialized_reference_key == order.request.inputs[
        "materialized_reference_key"
    ]
    assert set(receipt.response_scales) == {"t5", "t20"}
    assert all(
        value > 0.0
        for row in receipt.response_scales.values()
        for value in row.values()
    )
    data = json.loads((tmp_path / receipt.data.path).read_text(encoding="utf-8"))
    assert set(data) == {
        "schema",
        "reference_identity_sha256",
        "materialized_reference_key",
        "observable_schema",
        "ordered_times_sha256",
        "time",
        "energy",
        "doublon",
        "site_occupations",
        "statevector_real",
        "statevector_imag",
    }
    assert len(data["time"]) == 501
    assert len(data["statevector_real"]) == 501

    with pytest.raises(ProspectiveExecutionError, match="overwrite immutable"):
        materialize_exact_reference(
            order,
            repository_root=tmp_path,
            compute=lambda current: _exact_computation(current, phase=0.1),
        )


@pytest.mark.parametrize("lane_id", [LANE_HH, LANE_HUBBARD])
def test_materialization_rejects_nonphysical_statevector_width(
    tmp_path: Path, lane_id: str
) -> None:
    _source, _seed, _request, order = _prepared(tmp_path, lane_id)
    valid = _exact_computation(order)
    malformed = ExactReferenceComputation(
        valid.time,
        valid.energy,
        valid.doublon,
        valid.site_occupations,
        tuple((1.0 + 0.0j, 0.0j) for _ in valid.time),
    )

    with pytest.raises(ProspectiveExecutionError, match="Hilbert-space dimension"):
        materialize_exact_reference(
            order,
            repository_root=tmp_path,
            compute=lambda _current: malformed,
        )


def test_rejects_seed_receipt_digest_drift_before_compute(tmp_path: Path) -> None:
    _source, seed, _request, order = _prepared(tmp_path, LANE_HUBBARD)
    seed_path = tmp_path / seed.receipt_path
    seed_path.write_bytes(seed_path.read_bytes() + b" ")
    called = False

    def forbidden(_order):
        nonlocal called
        called = True
        return _exact_computation(_order)

    with pytest.raises(ProspectiveExecutionError, match="file digest mismatch"):
        materialize_exact_reference(
            order, repository_root=tmp_path, compute=forbidden
        )
    assert called is False


def test_default_hh_propagator_materializes_loader_verified_reference(
    tmp_path: Path,
) -> None:
    _source, _seed, _request, order = _prepared(tmp_path, LANE_HH)
    receipt = materialize_exact_reference(order, repository_root=tmp_path)

    assert receipt.reference_id == order.reference.reference_id
    assert receipt.reference_contract["propagator"] == (
        "paper_ii_exact_driven_reference_v1"
    )
    assert all(
        value > 0.0
        for row in receipt.response_scales.values()
        for value in row.values()
    )


def test_default_hubbard_propagator_materializes_loader_verified_reference(
    tmp_path: Path,
) -> None:
    _source, _seed, _request, order = _prepared(tmp_path, LANE_HUBBARD)
    receipt = materialize_exact_reference(order, repository_root=tmp_path)

    assert receipt.reference_id == order.reference.reference_id
    assert receipt.reference_contract["propagator"] == (
        "paper_ii_exact_driven_reference_v1"
    )
    assert receipt.reference_contract["model"] == "hubbard"
    assert all(
        value > 0.0
        for row in receipt.response_scales.values()
        for value in row.values()
    )


def test_propagation_policy_rejects_nonfrozen_drive() -> None:
    reference = next(
        row for row in build_identity_registry().references if row.lane_id == LANE_HH
    )
    changed = replace(reference, drive={**reference.drive, "time_sampling": "left"})
    with pytest.raises(ProspectiveExecutionError, match="frozen waveform"):
        propagation_policy(changed)

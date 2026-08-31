"""Identity and dependency contract for prospective Paper-II campaigns."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    EXEC_HH_AVQDS,
    EXEC_HH_REFERENCE,
    EXEC_HH_SCORE,
    EXEC_HH_SEED,
    EXEC_HUBBARD_AVQDS,
    EXEC_HUBBARD_REFERENCE,
    EXEC_HUBBARD_SCORE,
    EXEC_HUBBARD_SEED,
    LANE_HH,
    LANE_HUBBARD,
    LANE_SIAM,
    PROSPECTIVE_EXACT_REFERENCE_DATA_V1,
    BootstrapEvidence,
    SourceEpochIdentity,
    VerifiedActionEvidence,
    activation_cut_ladder,
    build_identity_registry,
    plan_bootstrap_actions,
    verify_external_bindings,
)
from pipelines.time_dynamics.parallel_benchmark_matrix import ParallelBenchmarkMatrix


SOURCE_EPOCHS = {
    LANE_HH: SourceEpochIdentity(
        lane_id=LANE_HH,
        source_epoch_id="paper-ii-hh-product-source-test",
        source_revision="a" * 40,
    ),
    LANE_HUBBARD: SourceEpochIdentity(
        lane_id=LANE_HUBBARD,
        source_epoch_id="paper-ii-hubbard-l4-source-test",
        source_archive_sha256="b" * 64,
    ),
}
EXECUTORS = {
    EXEC_HH_SEED,
    EXEC_HH_REFERENCE,
    EXEC_HH_AVQDS,
    EXEC_HH_SCORE,
    EXEC_HUBBARD_SEED,
    EXEC_HUBBARD_REFERENCE,
    EXEC_HUBBARD_AVQDS,
    EXEC_HUBBARD_SCORE,
}


def _sha(character: str) -> str:
    return character * 64


def test_registry_freezes_all_seed_and_reference_identities() -> None:
    registry = build_identity_registry()
    assert len(registry.seeds) == 8
    assert len(registry.references) == 39
    assert registry.delegated_planners[LANE_SIAM].endswith("plan_siam_nb4_ladder")

    hh_seeds = [row for row in registry.seeds if row.lane_id == LANE_HH]
    hubbard_seeds = [row for row in registry.seeds if row.lane_id == LANE_HUBBARD]
    siam_seeds = [row for row in registry.seeds if row.lane_id == LANE_SIAM]
    assert len(hh_seeds) == 6
    assert len(hubbard_seeds) == 1
    assert len(siam_seeds) == 1
    assert all(row.energy_gate == 1.0e-6 for row in (*hh_seeds, *hubbard_seeds))
    assert siam_seeds[0].external_artifact_sha256 == (
        "5752bc1e4bf20a52ecdd3ea262cd7265f01d48dbdd7151679ebb8a742bdb24b9"
    )
    payload = registry.to_json_dict()
    assert payload["matrix_payload_sha256"] == registry.matrix_payload_sha256
    assert payload["matrix_raw_sha256"] == registry.matrix_raw_sha256
    assert payload["matrix_expanded_semantic_sha256"] == (
        registry.matrix_expanded_semantic_sha256
    )
    assert "matrix_sha256" not in payload
    assert "matrix_file_sha256" not in payload
    assert "matrix_semantic_sha256" not in payload
    assert payload["accuracy_policy_sha256"] == hashlib.sha256(
        json.dumps(
            registry.accuracy_policy,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def test_hh_seed_is_shared_across_drives_but_references_are_per_cell() -> None:
    registry = build_identity_registry()
    cells = [
        row
        for row in registry.references
        if row.lane_id == LANE_HH
        and row.physical["n_ph_max"] == 3
        and row.physical["regime"] == "crossover"
    ]
    assert len(cells) == 6
    assert len({row.seed_id for row in cells}) == 1
    assert len({row.reference_id for row in cells}) == 6
    assert len({row.identity_sha256 for row in cells}) == 6
    assert {row.drive["drive_id"] for row in cells} == {
        "fastweak",
        "slowstrong",
        "weakslow",
        "strongfast",
        "midres",
        "fastfast",
    }


def test_identity_binds_full_physics_drive_grid_and_no_qiskit() -> None:
    registry = build_identity_registry()
    hh = next(row for row in registry.references if row.lane_id == LANE_HH)
    assert hh.physical["boundary"] == "open"
    assert hh.physical["ordering"] == "blocked"
    assert hh.physical["boson_encoding"] == "binary"
    assert hh.physical["n_up"] == hh.physical["n_down"] == 1
    assert hh.grid["points"] == 501
    assert hh.grid["prefix_points"] == 126
    assert hh.drive["pattern"] == "staggered"
    assert hh.drive["time_sampling"] == "midpoint"

    siam = next(row for row in registry.references if row.lane_id == LANE_SIAM)
    assert siam.grid["stop"] == 5.0
    assert siam.grid["prefix_stop"] == 1.0
    assert siam.grid["prefix_points"] == 101
    assert siam.drive["kind"] == "sudden_hybridization_quench"
    assert siam.external_manifest_sha256 == (
        "5586dbae4c0575711a9bca0e311f095e7814d3ce68a4d431ca9484bf46f61386"
    )

    payload = registry.to_json_dict()
    assert "qiskit" not in str(payload).lower()


def test_reference_content_key_changes_with_seed_artifact() -> None:
    reference = build_identity_registry().references[0]
    assert reference.materialized_key(_sha("a")) != reference.materialized_key(
        _sha("b")
    )


def test_registered_ladders_reach_each_frozen_floor_without_skipping() -> None:
    matrix = ParallelBenchmarkMatrix.read()
    hh = activation_cut_ladder(
        matrix.payload["activation_ladders"]["hh_and_hubbard_l4"]
    )
    siam = activation_cut_ladder(matrix.payload["activation_ladders"]["siam_nb4"])
    assert hh[:6] == (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5)
    assert hh[-1] == 1.0e-12
    assert siam[-1] == 1.0e-7
    assert len(hh) == 21
    assert len(siam) == 11


def test_siam_external_seed_and_reference_bindings_verify_from_disk() -> None:
    registry = build_identity_registry()
    evidence = verify_external_bindings(registry, workspace_root=".")
    assert len(evidence.seed_artifact_sha256) == 1
    assert len(evidence.reference_artifact_sha256) == 1
    assert next(iter(evidence.seed_artifact_sha256)).startswith(LANE_SIAM)
    assert evidence.source_epochs[LANE_SIAM].source_epoch_id == (
        "siam-nb4-source-ba50339c15ffe1aa6507"
    )


def test_without_positive_source_or_executor_bindings_nothing_is_ready() -> None:
    plan = plan_bootstrap_actions(build_identity_registry())
    assert plan.ready_actions == ()
    assert len(plan.waiting_actions) == 7
    assert all(
        "source_epoch_binding" in row["blockers"] for row in plan.waiting_actions
    )
    assert all(
        any(str(value).startswith("executor:") for value in row["blockers"])
        for row in plan.waiting_actions
    )


def test_seed_builds_are_independently_ready_once_bindings_exist() -> None:
    plan = plan_bootstrap_actions(
        build_identity_registry(),
        evidence=BootstrapEvidence(source_epochs=SOURCE_EPOCHS),
        available_executors=EXECUTORS,
    )
    assert len(plan.ready_actions) == 7
    assert {row.kind for row in plan.ready_actions} == {"build_seed"}
    assert {row.lane_id for row in plan.ready_actions} == {LANE_HH, LANE_HUBBARD}
    assert all(not row.qiskit_compilation_requested for row in plan.ready_actions)


def test_reference_and_reference_blind_scout_open_in_parallel_after_seeds() -> None:
    registry = build_identity_registry()
    seed_bindings = {
        row.seed_id: _sha("a")
        for row in registry.seeds
        if row.lane_id in {LANE_HH, LANE_HUBBARD}
    }
    plan = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(
            seed_artifact_sha256=seed_bindings,
            source_epochs=SOURCE_EPOCHS,
        ),
        available_executors=EXECUTORS,
    )
    assert len(plan.ready_actions) == 76
    assert sum(row.kind == "build_exact_reference" for row in plan.ready_actions) == 38
    assert sum(row.kind == "screen_avqds" for row in plan.ready_actions) == 38
    references = [
        row for row in plan.ready_actions if row.kind == "build_exact_reference"
    ]
    assert all(
        row.inputs["exact_data_schema"] == PROSPECTIVE_EXACT_REFERENCE_DATA_V1
        for row in references
    )
    scouts = [row for row in plan.ready_actions if row.kind == "screen_avqds"]
    assert all(row.exact_reference_blind for row in scouts)
    assert all(row.inputs["controller"] == "state_motion_5e-5" for row in scouts)
    assert all(
        row.inputs["segment"] == "calibration_prefix_t0_t5_126"
        for row in scouts
    )
    assert all(row.inputs["activation_cut"] == 1.0e-2 for row in scouts)
    assert all(row.inputs["reference_identity_sha256"] for row in scouts)
    assert all(row.inputs["physical"] for row in scouts)
    assert all(row.inputs["drive"]["drive_id"] for row in scouts)
    assert all(row.inputs["ordered_grid"]["ordered_times_sha256"] for row in scouts)
    assert all(row.inputs["ordered_grid"]["ordered_prefix_sha256"] for row in scouts)
    assert all(row.inputs["activation_ladder_sha256"] for row in scouts)
    assert all(row.inputs["accuracy_policy_sha256"] for row in scouts)
    assert all(row.inputs["source_epoch"]["lane_id"] == row.lane_id for row in scouts)


def test_scoring_waits_for_both_reference_and_scout_completion() -> None:
    registry = build_identity_registry()
    prospective_seeds = [
        row for row in registry.seeds if row.lane_id in {LANE_HH, LANE_HUBBARD}
    ]
    seed_bindings = {row.seed_id: _sha("a") for row in prospective_seeds}
    first = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(
            seed_artifact_sha256=seed_bindings,
            source_epochs=SOURCE_EPOCHS,
        ),
        available_executors=EXECUTORS,
    )
    scouts = [row for row in first.ready_actions if row.kind == "screen_avqds"]
    scout_evidence = {
        row.action_id: VerifiedActionEvidence(
            action_id=row.action_id,
            run_path=f"output/prospective/{row.subject_id}/run.json",
            trajectory_sha256=_sha("c"),
            completion_receipt_sha256=_sha("d"),
        )
        for row in scouts
    }
    reference_bindings = {
        row.reference_id: _sha("b")
        for row in registry.references
        if row.lane_id in {LANE_HH, LANE_HUBBARD}
    }
    second = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(
            seed_artifact_sha256=seed_bindings,
            reference_artifact_sha256=reference_bindings,
            verified_action_evidence=scout_evidence,
            source_epochs=SOURCE_EPOCHS,
        ),
        available_executors=EXECUTORS,
    )
    scores = [row for row in second.ready_actions if row.kind == "score_avqds"]
    assert len(scores) == 38
    assert all(
        row.inputs["accuracy_policy_id"] == "response_normalized_matched_pair_v2"
        for row in scores
    )
    assert all(row.inputs["screen_action_evidence"]["run_path"] for row in scores)
    assert all(
        row.inputs["screen_action_evidence"]["trajectory_sha256"] == _sha("c")
        for row in scores
    )
    assert all(
        row.inputs["screen_action_evidence"]["completion_receipt_sha256"]
        == _sha("d")
        for row in scores
    )


def test_source_epoch_identity_requires_lane_and_immutable_source_binding() -> None:
    with pytest.raises(ValueError, match="source revision or source archive"):
        SourceEpochIdentity(lane_id=LANE_HH, source_epoch_id="epoch")
    with pytest.raises(ValueError, match="mapping key/lane_id"):
        BootstrapEvidence(
            source_epochs={
                LANE_HH: SourceEpochIdentity(
                    lane_id=LANE_HUBBARD,
                    source_epoch_id="epoch",
                    source_revision="a" * 40,
                )
            }
        )


def test_scout_and_score_ids_bind_all_execution_and_evidence_identity() -> None:
    registry = build_identity_registry()
    prospective_seeds = [
        row for row in registry.seeds if row.lane_id in {LANE_HH, LANE_HUBBARD}
    ]
    seed_bindings = {row.seed_id: _sha("a") for row in prospective_seeds}
    first = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(
            seed_artifact_sha256=seed_bindings,
            source_epochs=SOURCE_EPOCHS,
        ),
        available_executors=EXECUTORS,
    )
    scout = next(row for row in first.ready_actions if row.kind == "screen_avqds")
    reference_bindings = {
        row.reference_id: _sha("b")
        for row in registry.references
        if row.lane_id in {LANE_HH, LANE_HUBBARD}
    }

    def score_for(receipt_character: str):
        evidence = VerifiedActionEvidence(
            action_id=scout.action_id,
            run_path="output/frozen/run.json",
            trajectory_sha256=_sha("c"),
            completion_receipt_sha256=_sha(receipt_character),
        )
        plan = plan_bootstrap_actions(
            registry,
            evidence=BootstrapEvidence(
                seed_artifact_sha256=seed_bindings,
                reference_artifact_sha256=reference_bindings,
                verified_action_evidence={scout.action_id: evidence},
                source_epochs=SOURCE_EPOCHS,
            ),
            available_executors=EXECUTORS,
        )
        return next(
            row
            for row in plan.ready_actions
            if row.kind == "score_avqds" and row.subject_id == scout.subject_id
        )

    assert score_for("d").action_id != score_for("e").action_id

    target_reference = next(
        row for row in registry.references if row.cell_id == scout.subject_id
    )
    changed_reference = replace(
        target_reference,
        drive={**target_reference.drive, "amplitude": 999.0},
    )
    changed_registry = replace(
        registry,
        references=tuple(
            changed_reference if row is target_reference else row
            for row in registry.references
        ),
    )
    changed = plan_bootstrap_actions(
        changed_registry,
        evidence=BootstrapEvidence(
            seed_artifact_sha256=seed_bindings,
            source_epochs=SOURCE_EPOCHS,
        ),
        available_executors=EXECUTORS,
    )
    changed_scout = next(
        row
        for row in changed.ready_actions
        if row.kind == "screen_avqds" and row.subject_id == scout.subject_id
    )
    assert changed_scout.action_id != scout.action_id


def _external_siam_registry(tmp_path: Path, mutation: str):
    registry = build_identity_registry()
    seed = next(row for row in registry.seeds if row.lane_id == LANE_SIAM)
    reference = next(
        row for row in registry.references if row.lane_id == LANE_SIAM
    )
    source_seed_path = Path(seed.artifact_path)
    source_package_path = source_seed_path.with_name("manifest.json")
    source_reference_path = Path(str(reference.external_manifest_path))
    source_data_path = Path(str(reference.external_data_path))

    seed_document = json.loads(source_seed_path.read_text(encoding="utf-8"))
    package_document = json.loads(source_package_path.read_text(encoding="utf-8"))
    reference_document = json.loads(source_reference_path.read_text(encoding="utf-8"))
    if mutation == "physical":
        seed_document["physical"]["u"] = 9.0
    elif mutation == "pool":
        seed_document["pool"]["ordered_labels"] = seed_document["pool"][
            "ordered_labels"
        ][:-1]
        seed_document["pool"]["block_count"] -= 1
    elif mutation == "source":
        package_document["source_epoch"]["source_epoch_id"] = "wrong-epoch"
    elif mutation == "propagator":
        reference_document["propagator_schema"] = "wrong_propagator"
    else:
        raise AssertionError(mutation)

    seed_path = tmp_path / "input" / "seed.json"
    package_path = seed_path.with_name("manifest.json")
    reference_path = tmp_path / "exact" / "manifest.json"
    data_path = tmp_path / "exact" / "reference.npz"
    seed_path.parent.mkdir(parents=True)
    reference_path.parent.mkdir(parents=True)
    seed_path.write_text(json.dumps(seed_document), encoding="utf-8")
    seed_sha = hashlib.sha256(seed_path.read_bytes()).hexdigest()
    package_document["seed_sha256"] = seed_sha
    package_document["files"]["seed"]["sha256"] = seed_sha
    package_path.write_text(json.dumps(package_document), encoding="utf-8")
    reference_document["seed_sha256"] = seed_sha
    reference_path.write_text(json.dumps(reference_document), encoding="utf-8")
    data_path.write_bytes(source_data_path.read_bytes())

    changed_seed = replace(
        seed,
        artifact_path="input/seed.json",
        external_artifact_sha256=seed_sha,
    )
    changed_reference = replace(
        reference,
        external_manifest_path="exact/manifest.json",
        external_manifest_sha256=hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest(),
        external_data_path="exact/reference.npz",
        external_data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
    )
    return replace(
        registry,
        seeds=tuple(changed_seed if row is seed else row for row in registry.seeds),
        references=tuple(
            changed_reference if row is reference else row
            for row in registry.references
        ),
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("physical", "seed U diverged"),
        ("pool", "pool families/order diverged"),
        ("source", "source epoch diverged"),
        ("propagator", "reference propagator diverged"),
    ),
)
def test_siam_external_verification_fails_closed_on_identity_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        verify_external_bindings(
            _external_siam_registry(tmp_path, mutation), workspace_root=tmp_path
        )

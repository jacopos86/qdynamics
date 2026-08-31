"""Deterministic bindings for the disabled Paper-II acquisition registry."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.acquisition_registry_contract import (
    AcquisitionRegistryContractError,
    CANONICAL_LANE_IDS,
    DEFAULT_REGISTRY_PATH,
    FUTURE_REQUIRED_BLOCKERS,
    PRODUCTION_REQUIRED_BLOCKERS,
    build_registry_payload,
    validate_registry_for_activation,
    validate_registry_payload,
    write_or_check_registry,
)
from pipelines.time_dynamics.acquisition_supervisor import CampaignRegistry
from pipelines.time_dynamics.parallel_benchmark_matrix import (
    ParallelBenchmarkMatrix,
)


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_registry_check_binds_exact_matrix_and_identity_counts() -> None:
    result = write_or_check_registry(
        repository_root=ROOT,
        registry_path=DEFAULT_REGISTRY_PATH,
        check_only=True,
    )
    assert result["physical_cell_count"] == 39
    assert result["seed_count"] == 8
    assert result["reference_count"] == 39
    assert result["production_enabled"] is False

    registry = CampaignRegistry.read(
        ROOT / DEFAULT_REGISTRY_PATH, repository_root=ROOT
    )
    contracts = registry.scientific_contracts
    assert contracts["matrix"]["physical_cell_count"] == 39
    assert contracts["identity_registry"]["seed_count"] == 8
    assert contracts["identity_registry"]["reference_count"] == 39
    planned = contracts["planned_report_contracts"]
    assert sum(row["planned_exact_references"] for row in planned.values()) == 39
    assert sum(row["planned_method_slots"] for row in planned.values()) == 78
    assert len(
        {
            contracts["matrix"]["raw_sha256"],
            contracts["matrix"]["payload_sha256"],
            contracts["matrix"]["expanded_semantic_sha256"],
        }
    ) == 3
    assert contracts["matrix"]["accuracy_policy_spec_sha256"] == (
        ParallelBenchmarkMatrix.read().accuracy_policy_spec_sha256
    )
    for lane_id in (
        "paper_ii_hh_product_v1",
        "paper_ii_hubbard_l4_scale_v1",
    ):
        lane = next(row for row in registry.lanes if row.lane_id == lane_id)
        assert lane.metadata["matrix_accuracy_policy_sha256"] == (
            contracts["matrix"]["accuracy_policy_spec_sha256"]
        )


def test_registry_generation_is_deterministic_and_excludes_cancelled_diagnostic() -> None:
    first = build_registry_payload(ROOT)
    second = build_registry_payload(ROOT)
    assert first == second
    text = json.dumps(first, sort_keys=True)
    assert "fixed_fs_ball_v1" not in text
    assert "rapidity_fs_ball_v1" not in text
    assert "n_ph_max=7" not in text
    assert tuple(row["lane_id"] for row in first["lanes"]) == CANONICAL_LANE_IDS
    assert first["scientific_contracts"]["matrix"]["path"] == (
        "pipelines/time_dynamics/campaigns/"
        "paper_ii_parallel_benchmark_matrix_v1.json"
    )


def test_registry_validation_rejects_stale_contract_and_source_digests() -> None:
    payload = build_registry_payload(ROOT)
    stale = deepcopy(payload)
    stale["scientific_contracts"]["matrix"]["raw_sha256"] = "0" * 64
    with pytest.raises(AcquisitionRegistryContractError, match="stale"):
        validate_registry_payload(stale, ROOT)

    stale = deepcopy(payload)
    stale["production_operations_blocked"]["source"]["sha256"] = "0" * 64
    with pytest.raises(AcquisitionRegistryContractError, match="source is stale"):
        validate_registry_payload(stale, ROOT)

    for section in ("identity_registry", "executor_registry", "scientific_planner"):
        stale = deepcopy(payload)
        stale["scientific_contracts"][section]["source"]["sha256"] = "0" * 64
        with pytest.raises(
            AcquisitionRegistryContractError, match="scientific contract bindings are stale"
        ):
            validate_registry_payload(stale, ROOT)

    stale = deepcopy(payload)
    stale["scientific_contracts"]["planned_report_contracts"][
        "paper_ii_hh_product_v1"
    ]["report_contract_sha256"] = "0" * 64
    with pytest.raises(
        AcquisitionRegistryContractError, match="scientific contract bindings are stale"
    ):
        validate_registry_payload(stale, ROOT)


def test_registry_validation_rejects_activation_or_removed_blockers() -> None:
    payload = build_registry_payload(ROOT)
    activated = deepcopy(payload)
    activated["lanes"][0]["submission_enabled"] = True
    with pytest.raises(AcquisitionRegistryContractError, match="disabled"):
        validate_registry_payload(activated, ROOT)

    missing = deepcopy(payload)
    blocked = missing["production_operations_blocked"]["blocked_by"]
    blocked.remove(PRODUCTION_REQUIRED_BLOCKERS[-1])
    with pytest.raises(AcquisitionRegistryContractError, match="blockers"):
        validate_registry_payload(missing, ROOT)

    missing = deepcopy(payload)
    future = next(
        lane
        for lane in missing["lanes"]
        if lane["lane_id"] == "paper_ii_hh_product_v1"
    )
    future["blocked_by"].remove(FUTURE_REQUIRED_BLOCKERS[-1])
    with pytest.raises(AcquisitionRegistryContractError, match="future lane blockers"):
        validate_registry_payload(missing, ROOT)


def test_registry_pre_submit_audits_remain_pending_and_receiptless() -> None:
    payload = build_registry_payload(ROOT)
    audits = payload["scientific_contracts"]["scientific_pre_submit_audits"]
    assert set(audits) == {
        "paper_ii_hh_product_v1",
        "paper_ii_hubbard_l4_scale_v1",
    }
    assert all(
        row == {"required": True, "status": "pending", "receipt": None}
        for row in audits.values()
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["lanes"].pop(2),
            "exact canonical lane set",
        ),
        (
            lambda payload: next(
                row
                for row in payload["lanes"]
                if row["lane_id"] == "paper_ii_hh_product_v1"
            ).update({"report_contract": "output/unrelated/report.json"}),
            "paths/planner/roots/digests",
        ),
        (
            lambda payload: next(
                row
                for row in payload["lanes"]
                if row["lane_id"] == "paper_ii_hubbard_l4_scale_v1"
            ).update({"planner_authority": "unrelated.module:planner"}),
            "planner binding",
        ),
        (
            lambda payload: next(
                row
                for row in payload["lanes"]
                if row["lane_id"] == "siam_nb4_pilot_v1"
            ).update({"state_root": "output/siam_screening/wrong"}),
            "paths/planner/roots/digests",
        ),
        (
            lambda payload: payload["scientific_contracts"]["matrix"].update(
                {"path": "/absolute/nonportable/matrix.json"}
            ),
            "stale",
        ),
    ],
)
def test_registry_rejects_lane_path_planner_root_and_absolute_path_drift(
    mutation, message: str
) -> None:
    payload = build_registry_payload(ROOT)
    mutation(payload)
    with pytest.raises(AcquisitionRegistryContractError, match=message):
        validate_registry_payload(payload, ROOT)


def test_registry_rejects_wrong_lane_identity() -> None:
    payload = build_registry_payload(ROOT)
    payload["lanes"][1]["lane_id"] = "paper_ii_hh_wrong_lane"
    with pytest.raises(AcquisitionRegistryContractError, match="canonical lane set"):
        validate_registry_payload(payload, ROOT)


def test_check_uses_independent_canonical_derivation_not_target_as_template(
    tmp_path: Path,
) -> None:
    target = tmp_path / "drifted-registry.json"
    drifted = build_registry_payload(ROOT)
    drifted["lanes"].pop()
    target.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        AcquisitionRegistryContractError,
        match="not the canonical generated payload",
    ):
        write_or_check_registry(
            repository_root=ROOT,
            registry_path=target,
            check_only=True,
        )


def test_activation_revalidates_source_artifact_after_permissive_parse(
    tmp_path: Path,
) -> None:
    target = tmp_path / "registry.json"
    payload = build_registry_payload(ROOT)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    registry = CampaignRegistry.read(target, repository_root=ROOT)
    assert registry.registry_path == target.resolve()
    payload["lanes"][0]["state_root"] = "output/wrong-state-root"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(AcquisitionRegistryContractError, match="paths/planner/roots"):
        validate_registry_for_activation(registry)


def test_activation_rejects_declassified_canonical_shape(tmp_path: Path) -> None:
    target = tmp_path / "registry.json"
    payload = build_registry_payload(ROOT)
    payload["registry_id"] = "declassified_registry"
    payload["scientific_contracts"] = {}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    registry = CampaignRegistry.read(target, repository_root=ROOT)
    with pytest.raises(AcquisitionRegistryContractError, match="declassified"):
        validate_registry_for_activation(registry)


def test_registered_callable_must_come_from_digest_bound_source(monkeypatch) -> None:
    import pipelines.time_dynamics.prospective_campaign_bootstrap as bootstrap

    original = bootstrap.build_identity_registry

    def impostor(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(bootstrap, "build_identity_registry", impostor)
    with pytest.raises(AcquisitionRegistryContractError, match="bound to its source"):
        build_registry_payload(ROOT)

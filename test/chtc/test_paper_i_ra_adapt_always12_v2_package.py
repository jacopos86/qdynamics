from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
V12_ROOT = (
    REPAIR_ROOT
    / "bundles/materializations/ra_adapt_stationary_late_core_v12"
)
V13_ROOT = (
    REPAIR_ROOT
    / "bundles/materializations/ra_adapt_stationary_late_core_v13"
)
PACKAGE_DIR = (
    REPAIR_ROOT / "stationary_ra_always12_r50_20260729_v2_chtc"
)
V12_SOURCE_LOCKS = (
    V12_ROOT / "source_materialization/source_locks_input.json"
)
V13_SOURCE_LOCKS = (
    V13_ROOT / "source_materialization/source_locks_input.json"
)
EXPECTED_PACKAGE_MANIFEST_SHA256 = (
    "467be866ac8abd01b109aefea69112aacb1b658da37c66eac92a4976d387fe9f"
)
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "1407947832291ab15ad91b0455058a6de689dac42cd1cb5282a76eeafbbc409d"
)
EXPECTED_SMOKE_RECEIPT_SHA256 = (
    "74a1e587599b174331ca2ea4d152370968f082efd065918896c53c38d81328fa"
)
EXPECTED_V13_FINAL_SHA256 = (
    "60f7c5cd29fe0c7c9f62c6dc8a8de2581eaad9a322ebfe0fab5d8c6220576274"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_contract() -> ModuleType:
    path = PACKAGE_DIR / "package_contract.py"
    spec = importlib.util.spec_from_file_location(
        "paper_i_ra_adapt_always12_v2_contract", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    return module


def _scalar_differences(
    before: Any,
    after: Any,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], Any, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        assert set(before) == set(after)
        result = []
        for key in sorted(before):
            result.extend(
                _scalar_differences(
                    before[key], after[key], (*path, key)
                )
            )
        return result
    if isinstance(before, list) and isinstance(after, list):
        assert len(before) == len(after)
        result = []
        for index, (left, right) in enumerate(zip(before, after)):
            result.extend(
                _scalar_differences(
                    left, right, (*path, index)
                )
            )
        return result
    return [] if before == after else [(path, before, after)]


def test_v13_changes_only_twelve_always_route_deltas() -> None:
    v12 = _load(V12_SOURCE_LOCKS)
    v13 = _load(V13_SOURCE_LOCKS)
    assert (
        V12_ROOT / "source_materialization/problem_baselines.json"
    ).read_bytes() == (
        V13_ROOT / "source_materialization/problem_baselines.json"
    ).read_bytes()
    assert v12.keys() == v13.keys()
    for key in v12:
        if key != "cell_locks":
            assert v12[key] == v13[key]

    changed = []
    for lock_id, before in v12["cell_locks"].items():
        after = v13["cell_locks"][lock_id]
        assert before["archive"] == after["archive"]
        assert before["member"] == after["member"]
        differences = _scalar_differences(before, after)
        if not lock_id.endswith("_always"):
            assert differences == []
            continue
        changed.append((lock_id, differences))
        assert len(differences) == (
            2 if lock_id.endswith("_ra_singleton_always") else 1
        )
        for path, _old, new in differences:
            assert new == "always_commutation_reduced"
            assert (
                path[-1] == "to"
                or path
                == (
                    "resolver_trace",
                    "core_source_anchor",
                    "route_derivation",
                    "target_insertion_policy",
                )
            )
    assert len(changed) == 12
    assert sum(len(differences) for _, differences in changed) == 18


def test_v13_always_protocols_are_typed_and_route_bound() -> None:
    contract = _load_contract()
    authority = contract.validate_core_authority(REPO_ROOT)
    final = authority["final"]
    assert final["sha256"] == EXPECTED_V13_FINAL_SHA256
    assert final["authorization"] == {
        "execution_authorized": False,
        "submission_authorized": False,
        "submission_state": "not_submitted",
        "submitted": False,
        "explicit_future_user_authorization_required": True,
    }
    for row in contract.direct_execution_rows():
        protocol = _load(
            Path(authority["bundle_root"])
            / "protocols"
            / f"{row['cell_id']}.json"
        )
        assert (
            protocol["request"]["method"]["insertion"]["kind"]
            == "always_commutation_reduced"
        )
        route = protocol["route_contract"]
        assert (
            route["execution_settings"]["adapt_insertion_mode"]
            == "full_commutation_reduced"
        )
        assert route["semantic_invariants"][
            "insertion_position_scope"
        ] == "full_logical_ansatz_commutation_classes_every_depth_v2"
        assert route["semantic_invariants"][
            "insertion_equivalence_policy"
        ] == (
            "termwise_cross_component_commutation_"
            "earliest_representative_v1"
        )


def test_v2_package_is_distinct_inert_and_reduction_closed() -> None:
    contract = _load_contract()
    manifest = _load(PACKAGE_DIR / "package_manifest.json")
    smoke = _load(PACKAGE_DIR / "two_round_smoke_receipt.json")
    contract.verify_self_digest(manifest, label="package manifest")
    contract.validate_smoke_receipt(smoke)
    assert manifest["sha256"] == EXPECTED_PACKAGE_MANIFEST_SHA256
    assert manifest["source_archive"]["sha256"] == (
        EXPECTED_SOURCE_ARCHIVE_SHA256
    )
    assert smoke["sha256"] == EXPECTED_SMOKE_RECEIPT_SHA256
    assert manifest["package_id"].endswith("_v2_chtc")
    assert manifest["authority_overlay_present"] is False
    assert manifest["execution_authorized"] is False
    assert manifest["submission_authorized"] is False
    assert manifest["remote_stage"] is False
    assert manifest["condor_submit"] is False
    assert not (PACKAGE_DIR / "authority").exists()
    assert len(list((PACKAGE_DIR / "jobs").glob("*.json"))) == 12

    for route in smoke["route_observations"]:
        second = route["accepted_round_reduction_receipts"][1]
        assert second["requested_positions"] == [0, 1]
        assert second["domain_open"] is True
        assert second["domain_state"] == "open"
        assert (
            second["effective_insertion_mode"]
            == "full_commutation_reduced"
        )
        assert second["requested_position_count"] == 2
        assert second["collapsed_position_count"] > 0
        assert second["retained_representative_count"] + second[
            "collapsed_position_count"
        ] == second["candidate_count"] * 2
        assert any(
            len(plan["representative_positions"]) == 1
            and list(plan["members_by_representative"].values())
            == [[0, 1]]
            for plan in second["candidate_position_plans"]
        )
        assert any(
            plan["representative_positions"] == [0, 1]
            and list(plan["members_by_representative"].values())
            == [[0], [1]]
            for plan in second["candidate_position_plans"]
        )


def test_v2_smoke_contract_rejects_crossing_class_tamper() -> None:
    contract = _load_contract()
    smoke = _load(PACKAGE_DIR / "two_round_smoke_receipt.json")
    tampered = copy.deepcopy(smoke)
    plans = tampered["route_observations"][0][
        "accepted_round_reduction_receipts"
    ][1]["candidate_position_plans"]
    collapsed = next(
        plan
        for plan in plans
        if len(plan["representative_positions"]) == 1
    )
    assert collapsed["commuting_crossings"] == [True]
    collapsed["commuting_crossings"] = [False]
    tampered.pop("sha256")
    tampered = contract.digested(tampered)
    with pytest.raises(
        contract.PackageContractError,
        match="commuting-crossing certificate",
    ):
        contract.validate_smoke_receipt(tampered)


def test_v2_package_read_only_validator_passes_without_authority() -> None:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATIC_ADAPT_HH_POOL_CACHE": "off",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
    }
    completed = subprocess.run(
        [sys.executable, str(PACKAGE_DIR / "validate_package.py")],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "passed"
    assert payload["direct_execution_count"] == 12
    assert payload["execution_authorized"] is False
    assert payload["submission_authorized"] is False
    assert payload["remote_stage"] is False
    assert payload["condor_submit"] is False
    assert not list(PACKAGE_DIR.rglob("__pycache__"))

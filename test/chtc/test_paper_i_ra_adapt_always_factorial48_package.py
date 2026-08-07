from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
MATERIALIZATION_ROOT = (
    REPAIR_ROOT
    / "bundles/materializations/ra_adapt_always_factorial48_v1"
)
PACKAGE_DIR = (
    REPAIR_ROOT / "ra_always_factorial48_r50_20260730_v1_chtc"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_contract() -> ModuleType:
    path = PACKAGE_DIR / "package_contract.py"
    spec = importlib.util.spec_from_file_location(
        "paper_i_ra_adapt_always_factorial48_contract", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def test_factorial_materialization_is_exact_inert_and_source_locked() -> None:
    contract = _load_contract()
    authority = contract.validate_factorial_authority(REPO_ROOT)
    final = authority["final"]
    assert final["status"] == "passed"
    assert final["run_class"] == "diagnostic"
    assert final["arm_count"] == 4
    assert final["cell_count_per_arm"] == 12
    assert final["total_cell_count"] == 48
    assert final["execution_authorized"] is False
    assert final["submission_authorized"] is False
    assert final["submitted"] is False
    assert authority["equality_audit"]["status"] == "passed"
    assert authority["equality_audit"]["arm_count"] == 4
    assert authority["equality_audit"]["base_cell_count"] == 12
    assert authority["equality_audit"]["variant_count"] == 48
    for paths in authority["artifact_destinations"].values():
        assert len(paths) == 48
        assert len(set(paths)) == 48

    delta = _load(
        MATERIALIZATION_ROOT
        / "source_materialization/factor_delta_receipt.json"
    )
    contract.verify_self_digest(delta, label="factor delta")
    assert delta["status"] == "passed"
    assert delta["allowed_changed_fields"] == [
        "active_gradient_policy",
        "resource_weighting_scope",
    ]
    assert delta["row_count"] == 48
    assert delta["all_non_axis_source_lock_fields_equal"] is True
    assert delta["all_archive_bindings_preserved"] is True
    assert delta["all_member_bindings_preserved"] is True
    assert delta["all_global_source_bindings_preserved"] is True


def test_factorial_rows_are_complete_and_use_only_corrected_always() -> None:
    contract = _load_contract()
    authority = contract.validate_factorial_authority(REPO_ROOT)
    rows = list(contract.direct_execution_rows())
    assert len(rows) == 48
    assert len({row["execution_id"] for row in rows}) == 48
    assert {
        (
            row["regime_id"],
            row["nph"],
            row["candidate_representation"],
            row["active_gradient_policy"],
            row["resource_weighting_scope"],
        )
        for row in rows
    } == {
        (regime, nph, representation, gradient, scope)
        for regime, nph in contract.REGIME_CUTOFF_PAIRS
        for representation in (
            "macro_generator_v1",
            "single_pauli_word_v1",
        )
        for _bundle, gradient, scope, _suffix
        in contract.BUNDLE_POLICIES
    }
    assert all(
        row["route_id"]
        in {"ra_macro_always", "ra_singleton_always"}
        for row in rows
    )
    assert not any(
        row["execution_id"] == row["base_cell_id"] for row in rows
    )

    for row in rows:
        binding = authority["protocol_bindings"][row["execution_id"]]
        protocol = _load(REPO_ROOT / binding["path"])
        assert (
            protocol["request"]["method"]["insertion"]["kind"]
            == "always_commutation_reduced"
        )
        assert (
            protocol["route_contract"]["execution_settings"][
                "adapt_insertion_mode"
            ]
            == "full_commutation_reduced"
        )
        assert protocol["active_gradient_policy"] == (
            row["active_gradient_policy"]
        )
        assert protocol["resource_weighting_scope"] == (
            row["resource_weighting_scope"]
        )
        assert protocol["bundle_materialization"]["cell_id"] == (
            row["cell_id"]
        )


def test_factorial_smoke_covers_four_arms_by_two_representations() -> None:
    contract = _load_contract()
    smoke = _load(PACKAGE_DIR / "two_round_smoke_receipt.json")
    contract.validate_smoke_receipt(smoke)
    assert len(smoke["observations"]) == 8
    assert {
        (
            row["active_gradient_policy"],
            row["resource_weighting_scope"],
            row["candidate_representation"],
        )
        for row in smoke["observations"]
    } == {
        (gradient, scope, representation)
        for _bundle, gradient, scope, _suffix
        in contract.BUNDLE_POLICIES
        for representation in (
            "macro_generator_v1",
            "single_pauli_word_v1",
        )
    }
    for row in smoke["observations"]:
        second = row["accepted_round_reduction_receipts"][1]
        assert second["requested_positions"] == [0, 1]
        assert second["effective_insertion_mode"] == (
            "full_commutation_reduced"
        )
        assert second["collapsed_position_count"] > 0


def test_package_has_48_bound_inert_jobs_and_neutral_queue_macros() -> None:
    contract = _load_contract()
    authority = contract.validate_factorial_authority(REPO_ROOT)
    manifest = _load(PACKAGE_DIR / "package_manifest.json")
    contract.verify_self_digest(manifest, label="package manifest")
    assert manifest["status"] == "passed"
    assert manifest["direct_execution_count"] == 48
    assert manifest["factorial_arm_count"] == 4
    assert manifest["authority_overlay_present"] is False
    assert manifest["execution_authorized"] is False
    assert manifest["submission_authorized"] is False
    assert manifest["submitted"] is False
    assert not (PACKAGE_DIR / "authority").exists()
    assert len(list((PACKAGE_DIR / "jobs").glob("*.json"))) == 48
    for row in contract.direct_execution_rows():
        job = _load(PACKAGE_DIR / "jobs" / f"{row['execution_id']}.json")
        assert job["protocol"] == authority["protocol_bindings"][
            row["execution_id"]
        ]
        assert job["execution_authorized"] is False
        assert job["submission_authorized"] is False
        assert job["submitted"] is False

    queue = (PACKAGE_DIR / "queue.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(queue) == 48
    assert all(len(line.split("\t")) == 5 for line in queue)
    submit = (PACKAGE_DIR / "submit.sub").read_text(encoding="utf-8")
    assert (
        "queue execution_id,cpus,memory_mb,disk_mb,"
        "max_runtime_seconds from queue.tsv"
    ) in submit
    assert "request_memory = $(memory_mb)MB" in submit
    assert "request_disk = $(disk_mb)MB" in submit
    assert "queue execution_id,request_" not in submit


def test_package_read_only_validator_passes() -> None:
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
    assert payload["direct_execution_count"] == 48
    assert payload["factorial_arm_count"] == 4
    assert payload["execution_authorized"] is False
    assert payload["submission_authorized"] is False
    assert payload["remote_stage"] is False
    assert payload["condor_submit"] is False
    assert not list(PACKAGE_DIR.rglob("__pycache__"))

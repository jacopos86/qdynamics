from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
PACKAGES = {
    "page12": REPAIR_ROOT
    / "paper_i_ra_adapt_page12_insertion_comparators_r50_20260812_v1_chtc",
    "page16": REPAIR_ROOT
    / (
        "paper_i_ra_adapt_page16_insertion_comparators_weak50_strong30_"
        "20260812_v1_chtc"
    ),
}
SOURCE_PACKAGES = {
    "page12": REPAIR_ROOT
    / (
        "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_"
        "phase23_no_lanes_cap24_tau1em4_r50_20260807_v1_chtc"
    ),
    "page16": REPAIR_ROOT
    / (
        "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_"
        "qiskit_phase23_no_lanes_cap24_tau1em4_weak50_strong30_"
        "20260811_v1_chtc"
    ),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _contract(package: Path, page_id: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"paper_i_{page_id}_insertion_comparator_contract",
        package / "package_contract.py",
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


def test_packages_have_only_the_twenty_four_new_comparator_jobs() -> None:
    all_ids: set[str] = set()
    for page_id, package in PACKAGES.items():
        contract = _contract(package, page_id)
        manifest = _load(package / "package_manifest.json")
        contract.verify_self_digest(manifest, label="package manifest")
        assert manifest["status"] == "passed_inert_twelve_cells"
        assert manifest["run_class"] == "diagnostic"
        assert manifest["row_count"] == 12
        assert manifest["comparator_policies"] == [
            "always_commutation_reduced",
            "append_only",
        ]
        assert manifest["plateau_reference_reused_not_rerun"] is True
        assert manifest["fresh_source_value_anchor"] is False
        assert manifest["strict_fresh_replay_sensitivity_claimed"] is False
        assert manifest["execution_authorized"] is False
        assert manifest["submission_authorized"] is False
        assert manifest["submitted"] is False
        assert len(manifest["execution_ids"]) == 12
        assert not any(
            execution_id.endswith("_plateau")
            for execution_id in manifest["execution_ids"]
        )
        all_ids.update(manifest["execution_ids"])
    assert len(all_ids) == 24


def test_source_archives_are_byte_identical_to_authenticated_sources() -> None:
    for page_id, package in PACKAGES.items():
        source = SOURCE_PACKAGES[page_id]
        observed = package / "source/source_locked.tar.gz"
        expected = source / "source/source_locked.tar.gz"
        assert observed.read_bytes() == expected.read_bytes()
        assert hashlib.sha256(observed.read_bytes()).hexdigest() == _load(
            package / "package_manifest.json"
        )["source_archive"]["sha256"]


def test_protocols_bind_typed_policy_and_matching_runtime_mode() -> None:
    expected_modes = {
        "always_commutation_reduced": "full_commutation_reduced",
        "append_only": "append_only",
    }
    for package in PACKAGES.values():
        manifest = _load(package / "package_manifest.json")
        jobs = {
            row["execution_id"]: _load(package / row["path"])
            for row in manifest["jobs"]
        }
        for row in manifest["protocols"]:
            protocol = _load(package / row["path"])
            job = jobs[row["execution_id"]]
            policy = job["comparator_policy"]
            route = protocol["route_contract"]
            invariants = route["semantic_invariants"]
            assert protocol["request"]["method"]["insertion"] == {
                "kind": policy
            }
            assert route["execution_settings"]["adapt_insertion_mode"] == (
                expected_modes[policy]
            )
            assert job["dispatch_template_contains_legacy_plateau_token"] is True
            assert job["fresh_source_value_anchor"] is False
            if policy == "always_commutation_reduced":
                assert invariants["insertion_position_scope"] == (
                    "full_logical_ansatz_commutation_classes_every_depth_v2"
                )
                assert invariants["insertion_equivalence_policy"] == (
                    "termwise_cross_component_commutation_"
                    "earliest_representative_v1"
                )
            else:
                assert not any(
                    (
                        key.startswith("plateau_")
                        and key
                        != "plateau_prior_mean_decrease_ratio_threshold"
                    )
                    or "insertion" in key
                    for key in invariants
                )
                assert invariants[
                    "plateau_prior_mean_decrease_ratio_threshold"
                ] == 1.0e-4


def test_non_insertion_equality_audits_close_all_rows() -> None:
    for package in PACKAGES.values():
        audit = _load(package / "non_insertion_equality_audit.json")
        unsigned = dict(audit)
        supplied = unsigned.pop("sha256")
        assert supplied == hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        assert audit["status"] == "passed"
        assert audit["planned_run_count"] == 12
        assert audit["all_non_insertion_executable_projections_equal"] is True
        assert all(
            row["non_insertion_executable_projection_equal"] is True
            for row in audit["rows"]
        )


def test_inert_package_validators_run_all_twelve_worker_preflights() -> None:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATIC_ADAPT_HH_POOL_CACHE": "off",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
    }
    for package in PACKAGES.values():
        completed = subprocess.run(
            [sys.executable, str(package / "validate_package.py")],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        receipt = json.loads(completed.stdout)
        assert receipt["status"] == "passed_inert_package"
        assert receipt["shallow_worker_preflight_count"] == 12
        assert receipt["execution_authorized"] is False
        assert receipt["submission_authorized"] is False

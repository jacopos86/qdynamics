from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chtc.paper_i_ra_adapt_repair_20260727 import (
    materialize_ra_qiskit_cost_macro_always13_local_v1 as materializer,
)
from pipelines.static_adapt.ra_adapt.bundles import (
    QISKIT_COST_ALWAYS13_ALGORITHM_ID,
    QISKIT_COST_ALWAYS13_BUNDLE_ID,
    QISKIT_COST_ALWAYS13_HORIZON,
    QISKIT_COST_ALWAYS13_SOURCE_PROTOCOL_SHA256,
    load_validated_bundle_protocol,
)
from pipelines.static_adapt.ra_adapt.contracts import canonical_sha256
from pipelines.static_adapt.sr_snake.contracts import (
    AlwaysCommutationReducedInsertion,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MATERIALIZATION_ROOT = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    "ra_adapt_qiskit_cost_macro_always13_local_v2"
)
TARGET_PROTOCOL = (
    MATERIALIZATION_ROOT
    / QISKIT_COST_ALWAYS13_BUNDLE_ID
    / "protocols/"
    "qiskit_cost_always13__strong_weak_u8__nph3__ra_macro_always.json"
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _verify_self_digest(value: dict[str, object]) -> None:
    unsigned = dict(value)
    supplied = unsigned.pop("sha256")
    assert supplied == canonical_sha256(unsigned)


def test_v2_audit_binds_exact_source_target_and_only_named_deltas() -> None:
    audit_path = (
        MATERIALIZATION_ROOT / "source_locked_sensitivity_audit.json"
    )
    final_path = MATERIALIZATION_ROOT / "final_materialization_receipt.json"
    audit = _load(audit_path)
    final = _load(final_path)
    _verify_self_digest(audit)
    _verify_self_digest(final)

    assert audit["schema"] == "source_locked_sensitivity_audit_v1"
    assert audit["status"] == "pass"
    source = audit["source"]
    assert isinstance(source, dict)
    assert source["source_protocol"]["canonical_sha256"] == (
        QISKIT_COST_ALWAYS13_SOURCE_PROTOCOL_SHA256
    )
    assert source["source_run_manifest"]["canonical_sha256"] == (
        materializer.EXPECTED_SOURCE_RUN_MANIFEST_CANONICAL_SHA256
    )
    sweep = audit["sweep"]
    assert isinstance(sweep, dict)
    assert sweep["wrapper_used"] is True
    assert sweep["wrapper"]["sha256"] == (
        materializer.EXPECTED_TARGET_RUNNER_FILE_SHA256
    )
    assert sweep["unresolved_source_fields"] == []
    assert sweep["fields_added_by_current_defaults"] == []

    approved = [
        "request.method.insertion",
        "request.execution.stop.maximum_controller_rounds",
    ]
    assert audit["approved_logical_deltas"] == approved
    assert audit["non_swept_settings_diff"] == []
    row = audit["planned_rows"][0]
    assert row["changed_fields_vs_source"] == approved
    assert row["non_swept_settings_diff"] == []
    assert audit["non_swept_executable_projection"]["equal"] is True

    drift = audit["implementation_inventory_drift"]
    assert drift["source_sha256"] == (
        materializer.SOURCE_IMPLEMENTATION_INVENTORY_SHA256
    )
    assert drift["target_sha256"] == (
        materializer.TARGET_IMPLEMENTATION_INVENTORY_SHA256
    )
    assert drift["inventory_equal"] is False
    assert [item["path"] for item in drift["changed_files"]] == [
        "pipelines/reporting/paper_i_run_summary.py",
        "pipelines/static_adapt/ra_adapt/bundles.py",
    ]

    binding = final["source_locked_sensitivity_audit"]
    assert isinstance(binding, dict)
    assert binding["canonical_sha256"] == audit["sha256"]
    assert binding["sha256"] == hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()


def test_v2_target_protocol_officially_loads_and_projection_is_equal() -> None:
    source = load_validated_bundle_protocol(materializer.SOURCE_PROTOCOL)
    target = load_validated_bundle_protocol(TARGET_PROTOCOL)
    assert target.algorithm_id == QISKIT_COST_ALWAYS13_ALGORITHM_ID
    assert target.horizon == QISKIT_COST_ALWAYS13_HORIZON
    assert isinstance(
        target.request.method.insertion,
        AlwaysCommutationReducedInsertion,
    )
    assert (
        materializer._non_swept_executable_projection(source)
        == materializer._non_swept_executable_projection(target)
    )

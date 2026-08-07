from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_core_full48_r50_20260728_v6_chtc"
)
sys.dont_write_bytecode = True
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(1, str(REPO_ROOT))

import package_contract as contract  # noqa: E402
import run_cell as cell_runner  # noqa: E402

from pipelines.static_adapt.ra_adapt import run_append_adapt  # noqa: E402
from pipelines.static_adapt.ra_adapt.bundles import (  # noqa: E402
    load_validated_bundle_protocol,
)
from pipelines.static_adapt.ra_adapt.replay_evidence import (  # noqa: E402
    bounded_prefix_replay_identity,
    build_append_controller_replay_evidence,
    build_signed_append_prefix_checkpoint,
)


APPEND_TABLE_I_RESOURCES = {
    "compile_convention": "table_i_basis_gate_transpile_v1",
    "compiled_count_2q_total": 250,
    "compiled_depth_2q_total": 210,
    "compiled_depth_total": 1112,
}


def test_append_table_i_projection_reads_serialized_canonical_fields() -> None:
    resources = {
        **APPEND_TABLE_I_RESOURCES,
        # Conflicting legacy names prove that the serialized canonical
        # Table-I fields own the Append projection.
        "compiled_two_qubit_count": 1,
        "compiled_two_qubit_depth": 2,
        "compiled_total_depth": 3,
    }

    assert cell_runner._compiled_resource_projection(
        resources,
        label="Append terminal compiled resources",
    ) == {"N2q": 250, "D2q": 210, "Dc": 1112}


def test_append_table_i_projection_fails_closed_on_bad_canonical_field() -> None:
    resources = {
        **APPEND_TABLE_I_RESOURCES,
        "compiled_count_2q_total": 250.5,
        "compiled_two_qubit_count": 250,
    }

    with pytest.raises(
        contract.PackageContractError,
        match="compiled two-qubit count must be an integer",
    ):
        cell_runner._compiled_resource_projection(
            resources,
            label="Append terminal compiled resources",
        )


def _full_protocol_bound_primary_fixture(
    *,
    protocol: Any,
    bounded_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rebind one real bounded prefix to the full protocol without new science."""

    primary = copy.deepcopy(bounded_payload)
    body = primary["result_payload"]
    bounded_rows = copy.deepcopy(body["history"])
    full_protocol_rows = copy.deepcopy(bounded_rows)
    for controller_round, row in enumerate(full_protocol_rows, start=1):
        checkpoint = row["active_prefix_checkpoint"]
        row["active_prefix_checkpoint"] = (
            build_signed_append_prefix_checkpoint(
                protocol=protocol,
                controller_round=controller_round,
                accepted_operator_labels=checkpoint[
                    "accepted_operator_labels"
                ],
                accepted_generator_identities=checkpoint[
                    "accepted_generator_identities"
                ],
                logical_parameters=checkpoint["logical_parameters"],
                runtime_parameters=checkpoint["runtime_parameters"],
                projective_state_fingerprint=checkpoint[
                    "projective_state_fingerprint"
                ],
                accepted_energy=checkpoint["accepted_energy"],
                accepted_refit=checkpoint["accepted_refit"],
                estimator_prefix=checkpoint["estimator_prefix"],
            )
        )
    full_protocol_replay = build_append_controller_replay_evidence(
        protocol=protocol,
        history=full_protocol_rows,
        estimator_ledger=body["estimator_call_ledger"],
        estimator_accounting=body["estimator_accounting"],
    )
    primary["scientific_receipts"][
        "controller_replay_evidence"
    ] = full_protocol_replay
    primary["scientific_receipts"][
        "controller_replay_evidence_sha256"
    ] = full_protocol_replay["sha256"]

    # G11 separately enforces the full-run horizon and consumes only its
    # first two rows. Keep the synthetic tail inert: this fixture tests the
    # bounded compatibility seam, not a local 50-round science execution.
    body["history"] = full_protocol_rows + [
        {"synthetic_terminal_tail_round": controller_round}
        for controller_round in range(3, 51)
    ]
    return primary, bounded_rows


def test_g11_append_accepts_protocol_bound_raw_history_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.chdir(tmp_path)

    job = next(
        row
        for row in contract.direct_execution_rows()
        if row["cell_id"]
        == "core__strong_weak_u8__nph3__append_macro"
    )
    authority = contract.validate_core_authority(REPO_ROOT)
    protocol_path = (
        REPO_ROOT
        / authority["protocol_bindings"][job["cell_id"]]["path"]
    )
    protocol = load_validated_bundle_protocol(protocol_path)
    problem = cell_runner._problem_from_protocol(protocol)
    bounded = cell_runner._bounded_append_protocol(
        job=job,
        final_protocol=protocol,
        problem=problem,
        authority=authority,
        rounds=2,
    )

    bounded_payload = run_append_adapt(problem, bounded).to_dict()
    bounded_replay = bounded_payload["scientific_receipts"][
        "controller_replay_evidence"
    ]
    primary_payload, bounded_rows = _full_protocol_bound_primary_fixture(
        protocol=protocol,
        bounded_payload=bounded_payload,
    )
    primary_rows = primary_payload["result_payload"]["history"]
    primary_replay = primary_payload["scientific_receipts"][
        "controller_replay_evidence"
    ]

    assert primary_rows[:2] != bounded_rows
    assert (
        primary_rows[0]["active_prefix_checkpoint"]["protocol_sha256"]
        == protocol.sha256
    )
    assert (
        bounded_rows[0]["active_prefix_checkpoint"]["protocol_sha256"]
        == bounded.sha256
    )
    assert bounded_prefix_replay_identity(
        primary_replay,
        controller_round=2,
    ) == bounded_prefix_replay_identity(
        bounded_replay,
        controller_round=2,
    )

    output_root = tmp_path / "worker_outputs"
    output_root.mkdir()
    diagnostic = cell_runner._run_g11_bounded_diagnostic(
        job=job,
        primary_payload=primary_payload,
        protocol=protocol,
        problem=problem,
        authority=authority,
        output_root=output_root,
    )

    assert diagnostic["status"] == "passed"
    assert diagnostic["evidence"]["method_family"] == "append_adapt"
    assert (
        diagnostic["evidence"]["primary_prefix_trajectory_sha256"]
        != diagnostic["evidence"]["trajectory_sha256"]
    )

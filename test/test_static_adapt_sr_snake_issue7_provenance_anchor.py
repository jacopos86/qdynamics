from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path
from typing import Any

import pytest

from pipelines.reporting.build_paper_i_hh_comparator_tracking_summary import (
    _iter_named_json_array,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
PROVENANCE_ANCHOR = (
    REPO_ROOT
    / "test"
    / "fixtures"
    / "paper_i_sr_snake_issue7_provenance_anchor.json"
)
LIVE_AUDIT_ENV = "RUN_PAPER_I_SR_SNAKE_PROVENANCE_AUDIT"
ROUTE_PROFILE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_v1"
)
ROUTE_DIGEST = (
    "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2"
)
EXPECTED_ROUTE = {
    "family": "singleton_response_snake",
    "profile": ROUTE_PROFILE,
    "profile_request": (
        "sr_snake_no_prune_symmetric_cost_projected_phase3_"
        "no_overlap_trust_v1"
    ),
    "profile_sha256": ROUTE_DIGEST,
    "visible_support_route_id": "no_overlap_trust_projected_phase3_nph3_7",
}
EXPECTED_PROBLEM = {
    "L": 2,
    "boson_encoding": "binary",
    "boundary": "open",
    "dv": 0.0,
    "fixed_particle_sector": {"n_dn": 1, "n_up": 1},
    "g_ep": 0.353553390593,
    "holstein_sector": "weak",
    "lambda": 0.25,
    "n_ph_reference": 3,
    "n_ph_work": 3,
    "omega0": 1.0,
    "ordering": "blocked",
    "problem": "hh",
    "same_cutoff_reference": True,
    "t": 1.0,
    "target_controller_round": 50,
    "u_over_t": 0.25,
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tar_member_sha256(path: Path, member_name: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    with tarfile.open(path, "r|gz") as archive:
        for info in archive:
            if info.name != member_name:
                archive.members.clear()
                continue
            handle = archive.extractfile(info)
            assert handle is not None, f"cannot extract {member_name} from {path}"
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest(), info.size
    raise AssertionError(f"missing tar member {member_name} in {path}")


def _sequence_sha256(values: list[Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            values,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _production_round_anchor(row: dict[str, Any]) -> dict[str, Any]:
    refit = row["accepted_refit"]
    invocation = refit["accepted_refit_invocation"]
    checkpoint = row["active_prefix_checkpoint"]
    ledger_receipt = checkpoint["estimator_ledger_receipt"]
    trust_receipt = row["route_a_trust_region_update"]
    transaction = trust_receipt["source_metric_trust_transaction"]
    fixed_sector = {
        str(constraint["quantity"]): int(constraint["target"])
        for constraint in checkpoint["state_sector_contract"][
            "fixed_count_constraints"
        ]
    }
    trust = {
        "endpoint_overlap_query_charge": trust_receipt[
            "endpoint_overlap_query_charge"
        ],
        "policy": trust_receipt["policy"],
        "update_reason": trust_receipt["update_reason"],
    }
    if transaction is None:
        trust["source_metric_transaction_failure"] = trust_receipt[
            "source_metric_trust_transaction_failure"
        ]
    else:
        trust.update(
            {
                "predicted_source_metric_displacement": transaction[
                    "predicted_source_metric_displacement"
                ],
                "realized_source_metric_displacement": transaction[
                    "realized_source_metric_displacement"
                ],
                "source_metric_transaction_complete": transaction[
                    "transaction_complete"
                ],
                "supported_metric_inverse_sqrt_constructed": transaction[
                    "supported_metric_inverse_sqrt_constructed"
                ],
                "supported_metric_whitening_active": transaction[
                    "supported_metric_whitening_active"
                ],
                "supported_rank": transaction["supported_rank"],
            }
        )
    return {
        "accepted_refit": {
            "coordinate_chart": invocation["config"]["coordinate_chart"],
            "final_energy": refit["final_energy"],
            "full_ansatz": invocation["config"]["full_ansatz"],
            "policy": refit["policy"],
            "supported_rank": refit["supported_rank"],
            "symmetric_metric_element_occurrences": invocation[
                "metric_query_accounting"
            ]["symmetric_metric_element_occurrences"],
        },
        "checkpoint": {
            "active_ansatz_depth": checkpoint["active_ansatz_depth"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "ledger_S_alg": ledger_receipt["cumulative_unique_primitives"]["S_alg"],
            "ledger_status": ledger_receipt["status"],
            "parameterization_mode": checkpoint["parameterization"]["mode"],
            "projective_state_fingerprint": checkpoint[
                "projective_state_fingerprint"
            ],
            "strict_replay_passed": checkpoint["strict_replay"]["passed"],
        },
        "depth": row["depth"],
        "energy_after_opt": row["energy_after_opt"],
        "fixed_particle_sector": fixed_sector,
        "generator_id": row["generator_id"],
        "phase3": {
            "coordinate_indices": row["phase3_response_coordinate_indices"],
            "pre_support_count": row["phase3_response_pre_support_count"],
            "supported_rank": row["phase3_response_supported_rank"],
        },
        "selected_op": row["selected_op"],
        "selected_position": row["selected_position"],
        "trust": trust,
    }


def _completed_result_anchor(
    archive_path: Path,
    *,
    member_name: str,
    first_hit_round: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r|gz") as archive:
        for info in archive:
            if info.name != member_name:
                archive.members.clear()
                continue
            handle = archive.extractfile(info)
            assert handle is not None, (
                f"cannot extract {member_name} from {archive_path}"
            )
            for row in _iter_named_json_array(handle, "history"):
                rows.append(row)
                if len(rows) == first_hit_round:
                    break
            break
    assert len(rows) == first_hit_round, (
        f"{member_name} has only {len(rows)} history rows before the first hit"
    )
    operators = [row["selected_op"] for row in rows]
    generator_ids = [row["generator_id"] for row in rows]
    energies = [row["energy_after_opt"] for row in rows]
    return {
        "accepted_energies_sha256_through_first_hit": _sequence_sha256(energies),
        "accepted_generator_ids_sha256_through_first_hit": _sequence_sha256(
            generator_ids
        ),
        "accepted_operators_sha256_through_first_hit": _sequence_sha256(operators),
        "accepted_positions_through_first_hit": [
            row["selected_position"] for row in rows
        ],
        "accepted_round_count_through_first_hit": len(rows),
        "round_1": _production_round_anchor(rows[0]),
        f"round_{first_hit_round}": _production_round_anchor(rows[-1]),
    }


def _argv_value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def test_repository_provenance_anchor_is_self_contained() -> None:
    anchor = _read_json(PROVENANCE_ANCHOR)
    assert anchor["schema"] == "paper_i_sr_snake_issue7_provenance_anchor_v1"
    assert anchor["route"] == EXPECTED_ROUTE
    assert anchor["paper_i_problem"] == EXPECTED_PROBLEM
    assert anchor["candidate_or_ablation_tracker"] == {
        "path": None,
        "status": "missing_candidate_or_ablation_specific_pointer",
    }

    weak_weak = anchor["known_values"]["weak_weak"]
    assert weak_weak["status"] == "hit"
    assert weak_weak["first_hit_outer_iteration"] == 21
    assert weak_weak["error"] == pytest.approx(2.650989383079505e-05, abs=1.0e-15)
    assert weak_weak["S_alg"] == 44216
    assert weak_weak["S_alg_components"] == {
        "N_H_outer": 1,
        "N_H_refit": 11371,
        "N_grad": 3094,
        "N_metric": 29750,
    }
    completed = weak_weak["completed_result_anchor"]
    assert completed["accepted_round_count_through_first_hit"] == 21
    assert completed["accepted_positions_through_first_hit"] == list(range(21))
    assert completed["round_1"]["fixed_particle_sector"] == {"n_dn": 1, "n_up": 1}
    assert completed["round_21"]["fixed_particle_sector"] == {"n_dn": 1, "n_up": 1}

    assert set(anchor["sources"]) == {
        "frozen_weak_weak_job",
        "normalized_weak_weak_manifest",
        "qiskit_target_prefix_costs",
        "runtime_postrun_s_alg_audit",
        "visible_results_support",
        "weak_weak_result_archive",
    }
    for receipt in anchor["sources"].values():
        assert Path(receipt["path"]).is_absolute() is False
        assert len(receipt["sha256"]) == 64
    assert (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        == ROUTE_PROFILE
    )


@pytest.mark.skipif(
    os.environ.get(LIVE_AUDIT_ENV) != "1",
    reason=(
        "set RUN_PAPER_I_SR_SNAKE_PROVENANCE_AUDIT=1 to validate ignored and "
        "fetched Paper-I source artifacts"
    ),
)
def test_opt_in_local_paper_i_sources_match_provenance_anchor() -> None:
    anchor = _read_json(PROVENANCE_ANCHOR)
    required_sources = {
        key: REPO_ROOT / value["path"]
        for key, value in anchor["sources"].items()
    }
    missing = [str(path) for path in required_sources.values() if not path.is_file()]
    assert not missing, f"missing immutable local Paper-I sources: {missing}"
    for key, path in required_sources.items():
        assert _sha256(path) == anchor["sources"][key]["sha256"]

    archive_receipt = anchor["sources"]["weak_weak_result_archive"]
    result_member_sha256, result_member_size = _tar_member_sha256(
        required_sources["weak_weak_result_archive"],
        archive_receipt["result_member"],
    )
    assert result_member_sha256 == archive_receipt["result_member_sha256"]
    completed_anchor = anchor["known_values"]["weak_weak"][
        "completed_result_anchor"
    ]
    assert result_member_size == completed_anchor["result_member_size_bytes"]
    observed_completed_anchor = _completed_result_anchor(
        required_sources["weak_weak_result_archive"],
        member_name=archive_receipt["result_member"],
        first_hit_round=21,
    )
    observed_completed_anchor["result_member_size_bytes"] = result_member_size
    assert observed_completed_anchor == completed_anchor

    support = _read_json(required_sources["visible_results_support"])
    snake_method = next(
        method for method in support["methods"] if method["short"] == "SNAKE"
    )
    assert snake_method["route_id"] == anchor["route"]["visible_support_route_id"]
    weak_weak = support["rows"]["SNAKE"]["weak_weak"]
    expected = anchor["known_values"]["weak_weak"]
    assert weak_weak["status"] == expected["status"]
    assert weak_weak["k"] == expected["first_hit_outer_iteration"]
    assert weak_weak["S_alg"] == expected["S_alg"]
    assert weak_weak["error"] == pytest.approx(expected["error"], abs=1.0e-15)
    assert weak_weak["S_alg_runtime_postrun_audit"]["status"] == (
        expected["runtime_postrun_closure_status"]
    )

    qiskit_costs = _read_json(required_sources["qiskit_target_prefix_costs"])
    qiskit_row = next(
        row
        for row in qiskit_costs["rows"]
        if row["route_id"] == anchor["route"]["visible_support_route_id"]
        and row["regime"] == "weak_weak"
    )
    assert qiskit_row["source"] == weak_weak["source"]
    assert qiskit_row["outer_iteration"] == expected["first_hit_outer_iteration"]
    assert qiskit_row["qiskit"] == {
        "D2q": expected["qiskit"]["D2q"],
        "Dc": expected["qiskit"]["Dc"],
        "N2q": expected["qiskit"]["N2q"],
    }
    assert qiskit_row["qiskit_compile"] == {
        "backend": None,
        "identity": expected["qiskit"]["compile_identity"],
        "optimization_level": expected["qiskit"]["optimization_level"],
        "reference_state_included": expected["qiskit"][
            "reference_state_included"
        ],
        "seed_transpiler": expected["qiskit"]["seed_transpiler"],
        "source_kind": "SNAKE structural active prefix",
    }
    assert qiskit_row["prefix_receipt"]["checkpoint_sha256"] == completed_anchor[
        "round_21"
    ]["checkpoint"]["checkpoint_sha256"]
    assert qiskit_row["prefix_receipt"]["strict_replay"]["passed"] is True
    assert qiskit_row["prefix_receipt"]["estimator_ledger_receipt"][
        "cumulative_unique_primitives"
    ] == {
        "S_alg": expected["S_alg"],
        "components": expected["S_alg_components"],
    }

    manifest = _read_json(required_sources["normalized_weak_weak_manifest"])
    assert manifest["route_identity"]["family"] == anchor["route"]["family"]
    assert manifest["route_identity"]["profile_request"] == anchor["route"][
        "profile_request"
    ]
    assert manifest["route_identity"]["profile_resolved"] == anchor["route"][
        "profile"
    ]
    assert manifest["route_identity"]["profile_contract_sha256"] == anchor["route"][
        "profile_sha256"
    ]
    physics = anchor["paper_i_problem"]
    for key in (
        "L",
        "boundary",
        "dv",
        "g_ep",
        "lambda",
        "n_ph_reference",
        "n_ph_work",
        "omega0",
        "ordering",
        "problem",
        "same_cutoff_reference",
        "t",
        "u_over_t",
    ):
        assert manifest["physics"][key] == physics[key]
    assert manifest["segment"]["target_controller_round"] == physics[
        "target_controller_round"
    ]

    job = _read_json(required_sources["frozen_weak_weak_job"])
    argv = job["command"]["argv"]
    assert _argv_value(argv, "--sr-route-profile") == anchor["route"][
        "profile_request"
    ]
    assert int(_argv_value(argv, "--adapt-max-depth")) == physics[
        "target_controller_round"
    ]
    assert _argv_value(argv, "--boson-encoding") == physics["boson_encoding"]
    assert _argv_value(argv, "--ordering") == physics["ordering"]
    assert float(_argv_value(argv, "--t")) == physics["t"]
    assert float(_argv_value(argv, "--u")) == physics["u_over_t"]
    assert float(_argv_value(argv, "--g-ep")) == physics["g_ep"]
    assert float(_argv_value(argv, "--omega0")) == physics["omega0"]
    assert float(_argv_value(argv, "--dv")) == physics["dv"]
    assert int(_argv_value(argv, "--n-ph-max")) == physics["n_ph_work"]

    audit = _read_json(required_sources["runtime_postrun_s_alg_audit"])
    audit_row = next(row for row in audit["rows"] if row["regime"] == "weak_weak")
    assert audit["status"] == "pass"
    assert audit_row["status"] == expected["runtime_postrun_closure_status"]
    assert audit_row["outer_iteration"] == expected["first_hit_outer_iteration"]
    assert audit_row["S_alg"] == expected["S_alg"]
    assert audit_row["components"] == expected["S_alg_components"]
    assert all(audit_row["closure"].values())

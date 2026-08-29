#!/usr/bin/env python3
"""Build the Paper-III static-QSE evidence report through LaTeX.

The report is presentation-only. It combines the locked resource-enriched
Hubbard--Holstein evidence, the independently generated one-to-one-matched
single- and block-seeded real-time-Krylov matrices after verifying identical
exact excitation references, the controlled Hubbard--Holstein pool ablation,
and the fresh Peierls--Hubbard/SIAM generality evidence. The PDF is always compiled
from the retained ``.tex`` source. Requested but incomplete model, comparator,
and resource cells remain visible as typed placeholders rather than
disappearing from the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_INPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_hh_resource_accounting_20260829_v1"
    / "paper_iii_static_qse_production_evidence.json"
)
DEFAULT_KRYLOV_MATRIX = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_seeded_realtime_krylov_matrix_20260829_v1"
    / "matrix.json"
)
DEFAULT_BLOCK_KRYLOV_MATRIX = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_block_seeded_realtime_krylov_matrix_20260829_v1"
    / "matrix.json"
)
DEFAULT_POOL_ABLATION_INPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_hh_pool_profile_ablation_20260828_v2"
    / "pool_profile_ablation.json"
)
DEFAULT_GENERALITY_INPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_qse_generality_20260829_v1"
    / "generality_evidence.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "output/pdf/paper_iii_static_qse_source_anchor_20260829_v2"
)
EXPECTED_INPUT_SCHEMA = "paper_iii_static_qse_production_evidence_v2"
EXPECTED_GENERALITY_SCHEMA = "paper_iii_qse_generality_evidence_v1"
EXPECTED_GENERALITY_RECEIPT_SCHEMA = "paper_iii_qse_generality_run_receipt_v1"
EXPECTED_KRYLOV_SCHEMA = "paper_iii_seeded_realtime_krylov_matrix_v1"
EXPECTED_KRYLOV_RECEIPT_SCHEMA = "paper_iii_seeded_realtime_krylov_matrix_v1_receipt"
EXPECTED_BLOCK_KRYLOV_SCHEMA = "paper_iii_block_seeded_realtime_krylov_matrix_v1"
EXPECTED_BLOCK_KRYLOV_RECEIPT_SCHEMA = (
    "paper_iii_block_seeded_realtime_krylov_matrix_v1_receipt"
)
EXPECTED_POOL_ABLATION_SCHEMA = "paper_iii_hh_pool_profile_ablation_v1"
MANIFEST_SCHEMA = "paper_iii_static_qse_production_manifest_v6_latex"

METHOD_SELECTED = "record_first_adaptive_qse"
METHOD_DAVIDSON = "direct_davidson_qse"
METHOD_KRYLOV = "seeded_realtime_krylov"
METHOD_BLOCK_KRYLOV = "block_seeded_realtime_krylov"
METHOD_FIXED = "fixed_linear_response_qse"
METHOD_FIXED_GENERALITY = "fixed_class_qse"


class ReportBuildError(RuntimeError):
    """Raised when the report cannot be built without hiding evidence gaps."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ReportBuildError(f"JSON input must be an object: {path}")
    return dict(payload)


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    )
    for raw, escaped in replacements:
        text = text.replace(raw, escaped)
    return text


def _sci(value: Any, digits: int = 2) -> str:
    number = float(value)
    if number == 0.0:
        return "$0$"
    mantissa, exponent = f"{number:.{digits}e}".split("e")
    return rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def _integer(value: Any) -> str:
    return f"{int(round(float(value))):,}"


def _case_label(case: Mapping[str, Any]) -> str:
    regime = str(case["regime"]).replace("_u8", "").replace("_", "--")
    return rf"{_tex_escape(regime)}, $n_{{\rm ph}}={int(case['n_ph_max'])}$"


def _status(value: str) -> str:
    return rf"\status{{{_tex_escape(value)}}}"


def _validate_resource_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != EXPECTED_INPUT_SCHEMA:
        raise ReportBuildError(
            f"expected {EXPECTED_INPUT_SCHEMA}, got {payload.get('schema_version')!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ReportBuildError("the locked Hubbard--Holstein matrix must have 12 rows")
    identity = payload.get("method_identity") or {}
    if identity.get("selected_method_id") != METHOD_SELECTED:
        raise ReportBuildError("unexpected canonical selected-QSE method id")
    if identity.get("direct_comparator_method_id") != METHOD_DAVIDSON:
        raise ReportBuildError("unexpected direct-Davidson method id")
    if bool(identity.get("implicit_davidson_fallback")):
        raise ReportBuildError("the locked selected-QSE evidence must not use fallback")
    for row in rows:
        for target in ("1e-4", "1e-6"):
            crossing = row.get("crossings", {}).get(target, {})
            required = (
                "selected_n2q",
                "direct_n2q",
                "selected_qiskit_compiled",
                "direct_qiskit_compiled",
                "selected_estimator_queries",
                "direct_estimator_queries",
            )
            missing = [field for field in required if field not in crossing]
            if missing:
                raise ReportBuildError(
                    f"{row.get('case', {}).get('case_key')} {target} lacks {missing}"
                )


def _validate_generality_payload(
    payload: Mapping[str, Any], *, input_path: Path
) -> Path:
    if payload.get("schema_version") != EXPECTED_GENERALITY_SCHEMA:
        raise ReportBuildError(
            f"expected {EXPECTED_GENERALITY_SCHEMA}, "
            f"got {payload.get('schema_version')!r}"
        )
    if payload.get("source_revision") != "7245d26eabb0435fe42bd2742e9e33c91a70f3f3":
        raise ReportBuildError("unexpected generality campaign source revision")
    method = payload.get("method_contract") or {}
    if method.get("selector_policy_id") != (
        "full_pencil_ky_fan_cost_ratio_novelty_floor_no_reward_v1"
    ):
        raise ReportBuildError("unexpected generality selector policy")
    boundary = payload.get("controller_boundary") or {}
    if boundary.get("implicit_fallback_enabled") is not False:
        raise ReportBuildError("generality evidence must exclude implicit fallback")
    if boundary.get("exact_excited_states_used_online") is not False:
        raise ReportBuildError("generality evidence used exact excited states online")

    cases_by_id = payload.get("cases")
    expected_cases = {
        "peierls_weak_weak",
        "peierls_weak_strong",
        "peierls_u8_weak",
        "siam_nb4_pilot",
    }
    if not isinstance(cases_by_id, Mapping) or set(cases_by_id) != expected_cases:
        raise ReportBuildError("generality evidence must contain the four locked cases")
    cases = list(cases_by_id.values())
    if {case.get("case_id") for case in cases} != expected_cases:
        raise ReportBuildError("generality case keys and embedded ids differ")
    for case in cases:
        if int(case.get("target_root_count", 0)) != 6:
            raise ReportBuildError(f"{case.get('case_id')} is not a six-root case")
        replay = case.get("selector_replay_lock") or {}
        if not all(
            replay.get(key) is True
            for key in (
                "selected_support_identical",
                "stop_reason_identical",
                "final_residual_identical",
            )
        ):
            raise ReportBuildError(f"{case.get('case_id')} failed selector replay lock")
        routes = case.get("routes") or {}
        if set(routes) != {"selected_qse", "direct_davidson", "fixed_qse"}:
            raise ReportBuildError(f"{case.get('case_id')} has an unexpected route set")
        selected = routes["selected_qse"]
        direct = routes["direct_davidson"]
        fixed = routes["fixed_qse"]
        if selected.get("method_id") != METHOD_SELECTED:
            raise ReportBuildError(f"{case.get('case_id')} selected route id changed")
        if direct.get("method_id") != METHOD_DAVIDSON:
            raise ReportBuildError(f"{case.get('case_id')} Davidson route id changed")
        if fixed.get("method_id") != METHOD_FIXED_GENERALITY:
            raise ReportBuildError(f"{case.get('case_id')} fixed route id changed")
        if selected.get("completion", {}).get("target_window_certified") is not True:
            raise ReportBuildError(f"{case.get('case_id')} selected window is uncertified")
        if selected.get("completion", {}).get("davidson_invoked") is not False:
            raise ReportBuildError(f"{case.get('case_id')} selected route invoked Davidson")
        if direct.get("completion", {}).get("target_window_complete") is not True:
            raise ReportBuildError(f"{case.get('case_id')} Davidson window is incomplete")
        for route_name, route in (("selected", selected), ("Davidson", direct)):
            crossing = route.get("threshold_crossings", {}).get("1e-06", {})
            if crossing.get("status") != "REACHED":
                raise ReportBuildError(
                    f"{case.get('case_id')} {route_name} did not reach 1e-6"
                )
            for resource_key in ("graph_span", "qiskit_compiled", "estimator"):
                if not isinstance(crossing.get(resource_key), Mapping):
                    raise ReportBuildError(
                        f"{case.get('case_id')} {route_name} lacks {resource_key}"
                    )
        if fixed.get("threshold_crossings", {}).get("1e-06", {}).get("status") != (
            "NOT_REACHED_WITHIN_CAP"
        ):
            raise ReportBuildError(f"{case.get('case_id')} fixed-route status changed")

    receipt_path = input_path.with_name("run_receipt.json")
    receipt = _load_json(receipt_path)
    if receipt.get("schema_version") != EXPECTED_GENERALITY_RECEIPT_SCHEMA:
        raise ReportBuildError("unexpected generality run-receipt schema")
    if receipt.get("output_sha256") != _sha256(input_path):
        raise ReportBuildError("generality run receipt does not match its evidence JSON")
    if receipt.get("source_revision") != payload.get("source_revision"):
        raise ReportBuildError("generality receipt and evidence revisions differ")
    return receipt_path


def _validate_krylov_matrix(
    payload: Mapping[str, Any], *, input_path: Path
) -> Path:
    if payload.get("schema_version") != EXPECTED_KRYLOV_SCHEMA:
        raise ReportBuildError(
            f"expected {EXPECTED_KRYLOV_SCHEMA}, got {payload.get('schema_version')!r}"
        )
    contract = payload.get("route_contract") or {}
    required_contract = {
        "method_id": METHOD_KRYLOV,
        "policy_id": "seeded_realtime_krylov_fixed_dt_0p5_v1",
        "target_root_count": 6,
        "root_matching": "minimum_total_abs_energy_hungarian_one_to_one",
        "online_exact_reference_use": False,
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise ReportBuildError(f"unexpected Krylov contract field {key!r}")
    cases = payload.get("cases")
    if not isinstance(cases, Mapping) or len(cases) != 16:
        raise ReportBuildError("Krylov matrix must contain all 16 declared cases")
    reached = 0
    for case_id, case in cases.items():
        if case.get("method_id") != METHOD_KRYLOV:
            raise ReportBuildError(f"{case_id} has an unexpected Krylov method id")
        if int(case.get("target_root_count", 0)) != 6:
            raise ReportBuildError(f"{case_id} is not a six-root Krylov case")
        if case.get("policy_id") != required_contract["policy_id"]:
            raise ReportBuildError(f"{case_id} changed the fixed Krylov policy")
        crossing = case.get("threshold_crossings", {}).get("1e-06", {})
        if crossing.get("status") == "REACHED":
            reached += 1
            numerical = crossing
        elif crossing.get("status") == "NOT_REACHED_WITHIN_CAP":
            numerical = case.get("terminal") or {}
        else:
            raise ReportBuildError(f"{case_id} has an unexpected 1e-6 status")
        required = ("k", "retained_rank", "K", "qiskit_compiled", "estimator")
        missing = [key for key in required if key not in numerical]
        if missing:
            raise ReportBuildError(f"{case_id} Krylov numerical record lacks {missing}")
        if numerical.get("max_root_abs_error") is None:
            raise ReportBuildError(f"{case_id} lacks a post-run six-root error")
    if reached != 10:
        raise ReportBuildError(f"expected 10/16 Krylov successes, found {reached}/16")
    qksd = payload.get("comparators", {}).get("hamiltonian_power_qksd", {})
    if qksd.get("status") != "IMPLEMENTATION_PENDING":
        raise ReportBuildError("QKSD placeholder status changed")

    receipt_path = input_path.with_name("receipt.json")
    receipt = _load_json(receipt_path)
    if receipt.get("schema_version") != EXPECTED_KRYLOV_RECEIPT_SCHEMA:
        raise ReportBuildError("unexpected Krylov matrix receipt schema")
    if receipt.get("sha256") != _sha256(input_path):
        raise ReportBuildError("Krylov matrix receipt does not match its JSON")
    if int(receipt.get("case_count", 0)) != 16:
        raise ReportBuildError("Krylov receipt case count changed")
    return receipt_path


def _validate_block_krylov_matrix(
    payload: Mapping[str, Any], *, input_path: Path
) -> Path:
    if payload.get("schema_version") != EXPECTED_BLOCK_KRYLOV_SCHEMA:
        raise ReportBuildError(
            f"expected {EXPECTED_BLOCK_KRYLOV_SCHEMA}, "
            f"got {payload.get('schema_version')!r}"
        )
    contract = payload.get("route_contract") or {}
    required_contract = {
        "method_id": METHOD_BLOCK_KRYLOV,
        "policy_id": "block_seeded_realtime_krylov_lowest_diagonal_b6_dt0p5_v1",
        "target_root_count": 6,
        "block_size": 6,
        "root_matching": "minimum_total_abs_energy_hungarian_one_to_one",
        "online_exact_reference_use": False,
        "same_policy_across_model_families": True,
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise ReportBuildError(f"unexpected block-RTK contract field {key!r}")
    cases = payload.get("cases")
    if not isinstance(cases, Mapping) or len(cases) != 16:
        raise ReportBuildError("block-RTK matrix must contain all 16 declared cases")
    expected_case_ids = {
        f"hh::{regime}::nph{nph}"
        for regime in (
            "weak_weak",
            "intermediate_weak",
            "strong_weak_u8",
            "weak_strong",
            "intermediate_strong",
            "strong_strong_u8",
        )
        for nph in (1, 3)
    } | {
        "peierls_weak_weak",
        "peierls_weak_strong",
        "peierls_u8_weak",
        "siam_nb4_pilot",
    }
    if set(cases) != expected_case_ids:
        raise ReportBuildError("block-RTK case identity set changed")
    reached = 0
    failed = 0
    for case_id, case in cases.items():
        if case.get("method_id") != METHOD_BLOCK_KRYLOV:
            raise ReportBuildError(f"{case_id} has an unexpected block-RTK method id")
        if int(case.get("target_root_count", 0)) != 6:
            raise ReportBuildError(f"{case_id} is not a six-root block-RTK case")
        if case.get("policy_id") != required_contract["policy_id"]:
            raise ReportBuildError(f"{case_id} changed the fixed block-RTK policy")
        crossing = case.get("threshold_crossings", {}).get("1e-06", {})
        if crossing.get("status") == "REACHED":
            reached += 1
            numerical = crossing
        elif crossing.get("status") == "NOT_REACHED_WITHIN_CAP":
            failed += 1
            numerical = case.get("terminal") or {}
        else:
            raise ReportBuildError(f"{case_id} has an unexpected block-RTK status")
        required = ("k", "retained_rank", "K", "qiskit_compiled", "estimator")
        missing = [key for key in required if key not in numerical]
        if missing:
            raise ReportBuildError(
                f"{case_id} block-RTK numerical record lacks {missing}"
            )
        if numerical.get("max_root_abs_error") is None:
            raise ReportBuildError(f"{case_id} lacks a post-run six-root error")
        if not isinstance(numerical.get("qiskit_compiled"), Mapping) or not all(
            key in numerical["qiskit_compiled"] for key in ("n2q", "d2q")
        ):
            raise ReportBuildError(f"{case_id} lacks block-RTK Qiskit annotations")
        if not isinstance(numerical.get("estimator"), Mapping) or (
            "queries" not in numerical["estimator"]
        ):
            raise ReportBuildError(f"{case_id} lacks block-RTK query accounting")
        coverage = case.get("post_run_source_coverage_audit") or {}
        if coverage.get("post_run_only") is not True:
            raise ReportBuildError(f"{case_id} source-coverage audit is not post-run")
        if coverage.get("used_by_online_route") is not False:
            raise ReportBuildError(f"{case_id} source audit leaked into block-RTK")
        missing_targets = coverage.get("target_root_indices_below_zero_weight_floor")
        if not isinstance(missing_targets, list):
            raise ReportBuildError(f"{case_id} lacks a typed source-coverage result")
        weights = coverage.get("source_block_overlap_weight_by_root")
        floor = coverage.get("zero_weight_floor")
        if not isinstance(weights, list) or len(weights) < 6 or floor is None:
            raise ReportBuildError(f"{case_id} lacks source weights or their floor")
        for root_index in missing_targets:
            if not isinstance(root_index, int) or not 1 <= root_index <= 6:
                raise ReportBuildError(
                    f"{case_id} source audit names an invalid target root"
                )
            if float(weights[root_index - 1]) >= float(floor):
                raise ReportBuildError(
                    f"{case_id} source audit contradicts its zero-weight floor"
                )
        if crossing.get("status") == "REACHED" and missing_targets:
            raise ReportBuildError(
                f"{case_id} reports success despite a missing target direction"
            )
        if crossing.get("status") != "REACHED" and not missing_targets:
            raise ReportBuildError(
                f"{case_id} failed without the locked zero-source-overlap diagnosis"
            )
    if reached != 11 or failed != 5:
        raise ReportBuildError(
            f"expected 11/16 block-RTK successes and five failures, "
            f"found {reached}/16 and {failed}"
        )

    receipt_path = input_path.with_name("receipt.json")
    receipt = _load_json(receipt_path)
    if receipt.get("schema_version") != EXPECTED_BLOCK_KRYLOV_RECEIPT_SCHEMA:
        raise ReportBuildError("unexpected block-RTK matrix receipt schema")
    if receipt.get("sha256") != _sha256(input_path):
        raise ReportBuildError("block-RTK matrix receipt does not match its JSON")
    if int(receipt.get("case_count", 0)) != 16:
        raise ReportBuildError("block-RTK receipt case count changed")
    return receipt_path


def _validate_pool_ablation(
    payload: Mapping[str, Any], *, input_path: Path
) -> Path:
    if payload.get("schema_version") != EXPECTED_POOL_ABLATION_SCHEMA:
        raise ReportBuildError(
            f"expected {EXPECTED_POOL_ABLATION_SCHEMA}, "
            f"got {payload.get('schema_version')!r}"
        )
    if payload.get("selector_policy_id") != (
        "full_pencil_ky_fan_cost_ratio_novelty_floor_no_reward_v1"
    ):
        raise ReportBuildError("pool ablation changed the production selector")
    expected_pools = {
        "hh_full_meta_macro_v1",
        "hh_full_meta_plus_encoded_boson_pauli_completion_v1",
    }
    if set(payload.get("pool_policy_ids") or ()) != expected_pools:
        raise ReportBuildError("pool ablation changed the two declared pool policies")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ReportBuildError("pool ablation must contain the 12 Hubbard--Holstein cells")
    expected_cases = {
        f"{regime}_nph{nph}"
        for regime in (
            "weak_weak",
            "intermediate_weak",
            "strong_weak_u8",
            "weak_strong",
            "intermediate_strong",
            "strong_strong_u8",
        )
        for nph in (1, 3)
    }
    observed_cases: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("case"), Mapping):
            raise ReportBuildError("pool ablation contains an untyped case row")
        case_key = str(row["case"].get("case_key", ""))
        observed_cases.add(case_key)
        selected = row.get("selected_qse_by_pool_policy")
        if not isinstance(selected, Mapping) or set(selected) != expected_pools:
            raise ReportBuildError(f"{case_key} lacks both pool-route summaries")
        for pool_id, route in selected.items():
            if not isinstance(route, Mapping):
                raise ReportBuildError(f"{case_key} {pool_id} route is untyped")
            if route.get("method_id") != METHOD_SELECTED:
                raise ReportBuildError(f"{case_key} {pool_id} method id changed")
            if route.get("pool_policy_id") != pool_id:
                raise ReportBuildError(f"{case_key} {pool_id} policy id changed")
            if not isinstance(route.get("crossings"), list):
                raise ReportBuildError(f"{case_key} {pool_id} lacks crossings")
        direct = row.get("direct_davidson")
        if not isinstance(direct, Mapping) or direct.get("method_id") != METHOD_DAVIDSON:
            raise ReportBuildError(f"{case_key} lacks the direct comparator summary")
    if observed_cases != expected_cases:
        raise ReportBuildError("pool ablation case set changed")
    boundary = payload.get("controller_boundary") or {}
    if boundary.get("implicit_fallback_enabled") is not False:
        raise ReportBuildError("pool ablation must exclude implicit fallback")
    if boundary.get("exact_audit_feeds_online_decisions") is not False:
        raise ReportBuildError("pool ablation used exact audit data online")
    lock_profiles = (payload.get("lock_gate") or {}).get("profiles") or {}
    base = lock_profiles.get("hh_full_meta_macro_v1") or {}
    completion = lock_profiles.get(
        "hh_full_meta_plus_encoded_boson_pauli_completion_v1"
    ) or {}
    if int(base.get("target_reached_case_count", -1)) != 10:
        raise ReportBuildError("base-pool ablation coverage changed from 10/12")
    if int(completion.get("target_reached_case_count", -1)) != 12:
        raise ReportBuildError("completion-pool ablation coverage changed from 12/12")
    if completion.get("coverage_qualified") is not True:
        raise ReportBuildError("completion pool lost its coverage qualification")

    status_path = input_path.with_name("status.json")
    status = _load_json(status_path)
    if status.get("schema_version") != (
        "paper_iii_hh_pool_profile_ablation_status_v1"
    ):
        raise ReportBuildError("unexpected pool-ablation status schema")
    if status.get("execution_status") != "COMPLETE":
        raise ReportBuildError("pool ablation is not complete")
    if int(status.get("case_count", 0)) != 12:
        raise ReportBuildError("pool-ablation status case count changed")
    return status_path


def _krylov_hh_index(
    payload: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    index: dict[tuple[str, int], Mapping[str, Any]] = {}
    for case_id, case in payload["cases"].items():
        if not str(case_id).startswith("hh::"):
            continue
        _, regime, nph = str(case_id).split("::")
        index[(regime, int(nph.removeprefix("nph")))] = case
    if len(index) != 12:
        raise ReportBuildError("Krylov matrix lacks the 12 Hubbard--Holstein cells")
    return index


def _verify_krylov_reference_identity(
    rows: Sequence[Mapping[str, Any]],
    generality: Mapping[str, Any],
    krylov: Mapping[str, Any],
    block_krylov: Mapping[str, Any],
) -> list[dict[str, str]]:
    receipts: list[dict[str, str]] = []
    hh_index = _krylov_hh_index(krylov)
    for row in rows:
        case = row["case"]
        key = (str(case["regime"]), int(case["n_ph_max"]))
        if key not in hh_index:
            raise ReportBuildError(f"Krylov trace is missing {key}")
        route_path = _resolve_artifact_path(row["route_artifact"]["path"])
        route = _load_json(route_path)
        current = route.get("post_run_exact_audit", {}).get("reference_excitations")
        single_prior = hh_index[key].get("reference_roots")
        block_prior = block_krylov["cases"][
            f"hh::{key[0]}::nph{key[1]}"
        ].get("reference_roots")
        if not all(
            isinstance(candidate, list)
            for candidate in (current, single_prior, block_prior)
        ):
            raise ReportBuildError(f"reference excitations are absent for {key}")
        for route_name, prior in (
            ("single-seed RTK", single_prior),
            ("block-seeded RTK", block_prior),
        ):
            if len(current) != len(prior) or any(
                abs(float(left) - float(right)) > 1.0e-12
                for left, right in zip(current, prior, strict=True)
            ):
                raise ReportBuildError(
                    f"current selected-QSE and {route_name} references differ for {key}"
                )
            receipts.append(
                {
                    "case_key": str(case["case_key"]),
                    "comparison": f"HH selected-QSE versus {route_name}",
                    "status": "identical_to_1e-12",
                }
            )
    for case_id, case in generality["cases"].items():
        current = case["reference_roots"]
        for route_name, matrix in (
            ("single-seed RTK", krylov),
            ("block-seeded RTK", block_krylov),
        ):
            prior = matrix["cases"][case_id]["reference_roots"]
            if len(current) != len(prior) or any(
                abs(float(left) - float(right)) > 1.0e-12
                for left, right in zip(current, prior, strict=True)
            ):
                raise ReportBuildError(
                    f"generality selected-QSE and {route_name} references "
                    f"differ for {case_id}"
                )
            receipts.append(
                {
                    "case_key": str(case_id),
                    "comparison": f"generality selected-QSE versus {route_name}",
                    "status": "identical_to_1e-12",
                }
            )
    return receipts


def _resolve_artifact_path(raw_path: Any) -> Path:
    path = Path(str(raw_path))
    if path.is_file():
        return path
    parts = path.parts
    if "output" in parts:
        candidate = REPO_ROOT.joinpath(*parts[parts.index("output") :])
        if candidate.is_file():
            return candidate
    raise ReportBuildError(f"referenced result file is missing: {path}")


def _load_verified_route(artifact: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve_artifact_path(artifact["path"])
    expected = artifact.get("sha256")
    if expected is not None and _sha256(path) != expected:
        raise ReportBuildError(f"route artifact digest changed: {path}")
    return _load_json(path)


def _first_reached_online_rung(
    route: Mapping[str, Any], *, target: float
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    online_rows = route.get("online_receipt", {}).get("rows")
    audit_rows = route.get("post_run_exact_audit", {}).get("rungs")
    if not isinstance(online_rows, list) or not isinstance(audit_rows, list):
        raise ReportBuildError("route lacks aligned online and post-run rungs")
    audit_by_k: dict[int, Mapping[str, Any]] = {}
    for audit in audit_rows:
        k = int(audit["k"])
        if k in audit_by_k:
            raise ReportBuildError(f"route contains duplicate post-run rung k={k}")
        audit_by_k[k] = audit
    online_keys = [int(row["k"]) for row in online_rows]
    if len(set(online_keys)) != len(online_keys) or set(online_keys) != set(audit_by_k):
        raise ReportBuildError("online and post-run route rungs are not one-to-one")
    for position, online in enumerate(online_rows):
        audit = audit_by_k[int(online["k"])]
        error = audit.get("max_root_abs_error")
        if error is None or float(error) > float(target):
            continue
        if int(online.get("retained_rank", 0)) < 6:
            raise ReportBuildError("a six-root crossing has retained rank below six")
        derived = dict(online)
        derived["max_root_abs_error"] = float(error)
        derived["per_root_abs_error"] = audit.get("per_root_abs_error")
        return derived, {"sequence_index": position, "audit": dict(audit)}
    return None


def _derive_hh_first_crossings(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Replace summary crossings in memory by earliest successful route prefixes.

    The sealed campaign post-run crossing records minimize graph-span cost
    among successful rungs. Publication tables instead use the earliest
    successful rung in each route's declared growth order. The complete sealed
    trace supports both readings, so this postprocessor changes no scientific
    run and records every difference in the output manifest.
    """

    from pipelines.exact_bench.paper_iii_qse_generality_campaign import (
        _direct_qiskit_cost,
        _record_qiskit_cost,
    )
    from pipelines.qse_spectra.paper_iii_problem import load_problem
    from pipelines.qse_spectra.pool_profiles import (
        HH_POOL_ENCODED_BOSON_COMPLETION_V1,
        apply_hh_pool_policy,
    )

    problem_cache: dict[tuple[str, str], Any] = {}
    qiskit_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for row in payload["rows"]:
        case = row["case"]
        case_key = str(case["case_key"])
        base_key = (case_key, "base")
        if base_key not in problem_cache:
            problem_cache[base_key] = load_problem(
                regime=str(case["regime"]),
                u=float(case["u"]),
                g_ep=float(case["g_ep"]),
                n_ph_max=int(case["n_ph_max"]),
                target_roots=6,
            )
        base_problem = problem_cache[base_key]
        completion_key = (case_key, "completion")
        if completion_key not in problem_cache:
            problem_cache[completion_key] = apply_hh_pool_policy(
                base_problem, HH_POOL_ENCODED_BOSON_COMPLETION_V1
            )

        route_specs = (
            (
                "selected",
                row["route_artifact"],
                problem_cache[completion_key],
                _record_qiskit_cost,
            ),
            (
                "direct",
                row["direct_route_artifact"],
                base_problem,
                _direct_qiskit_cost,
            ),
        )
        for route_name, artifact, problem, qiskit_builder in route_specs:
            route = _load_verified_route(artifact)
            stored_crossings = {
                float(item["error_target"]): item
                for item in route["post_run_exact_audit"]["crossings"]
            }
            for target_key, target in (("1e-4", 1.0e-4), ("1e-6", 1.0e-6)):
                result = _first_reached_online_rung(route, target=target)
                if result is None:
                    raise ReportBuildError(
                        f"{case_key} {route_name} lacks its declared {target_key} crossing"
                    )
                online, ordering = result
                if route_name == "selected":
                    support = tuple(int(index) for index in online["support_indices"])
                    cache_key = (case_key, route_name, support)
                    if cache_key not in qiskit_cache:
                        qiskit_cache[cache_key] = qiskit_builder(problem, support)
                else:
                    cache_key = (case_key, route_name, int(online["k"]))
                    if cache_key not in qiskit_cache:
                        qiskit_cache[cache_key] = qiskit_builder(problem, online)
                compiled = qiskit_cache[cache_key]
                crossing = row["crossings"][target_key]
                crossing[f"{route_name}_n2q"] = float(online["n2q"])
                crossing[f"{route_name}_qiskit_compiled"] = {
                    "n2q": float(compiled["n2q"]),
                    "d2q": float(compiled["d2q"]),
                    "n1q": float(compiled["n1q"]),
                    "status": str(compiled.get("status", "COMPILED")),
                }
                crossing[f"{route_name}_estimator_queries"] = int(
                    online["estimator"]["queries"]
                )
                crossing[f"{route_name}_k"] = int(online["k"])
                crossing[f"{route_name}_retained_rank"] = int(
                    online["retained_rank"]
                )
                crossing[f"{route_name}_max_root_abs_error"] = float(
                    online["max_root_abs_error"]
                )
                stored = stored_crossings[target]
                receipts.append(
                    {
                        "case_key": case_key,
                        "route": route_name,
                        "target": target_key,
                        "publication_policy": "earliest_successful_prefix_in_declared_growth_order",
                        "derived_sequence_index": int(ordering["sequence_index"]),
                        "derived_k": int(online["k"]),
                        "derived_K": float(online["n2q"]),
                        "derived_error": float(online["max_root_abs_error"]),
                        "sealed_summary_policy": "minimum_graph_span_cost_among_successful_rungs",
                        "sealed_summary_k": int(stored["k"]),
                        "sealed_summary_K": float(stored["n2q"]),
                        "differs_from_sealed_summary": (
                            int(stored["k"]) != int(online["k"])
                            or abs(float(stored["n2q"]) - float(online["n2q"]))
                            > 1.0e-12
                        ),
                    }
                )
    return receipts


def _krylov_record(
    case: Mapping[str, Any], target: str
) -> tuple[str, Mapping[str, Any]]:
    crossing_key = "1e-04" if target == "1e-4" else "1e-06"
    crossing = case["threshold_crossings"][crossing_key]
    status = str(crossing["status"])
    if status == "REACHED":
        return status, crossing
    if status == "NOT_REACHED_WITHIN_CAP":
        return status, case["terminal"]
    raise ReportBuildError(f"Krylov {target} has unexpected status {status!r}")


def _accuracy_cell(cost: Any) -> str:
    return rf"${_integer(cost)}$"


def _krylov_accuracy_cell(status: str, record: Mapping[str, Any]) -> str:
    cost = _integer(record["K"])
    if status == "REACHED":
        return rf"${cost}$"
    return rf"\mbox{{NR: {_sci(record['max_root_abs_error'], 1)} @ ${cost}$}}"


def _fixed_accuracy_cell(record: Mapping[str, Any]) -> str:
    return (
        rf"\mbox{{U: {_sci(record['max_root_energy_error'], 1)} @ "
        rf"${_integer(record['n2q'])}$}}"
    )


def _qiskit_cell(compiled: Mapping[str, Any], queries: Any) -> str:
    depth = compiled.get("depth", compiled.get("d2q"))
    if depth is None:
        raise ReportBuildError("Qiskit compilation record lacks two-qubit depth")
    return (
        rf"${_integer(compiled['n2q'])}/{_integer(depth)}$ "
        rf"[{_integer(queries)}]"
    )


def _krylov_resource_cell(status: str, record: Mapping[str, Any]) -> str:
    estimator = record.get("estimator") or {}
    prefix = "R" if status == "REACHED" else "NR"
    return (
        rf"{prefix} $k/r={int(record['k'])}/{int(record['retained_rank'])}$: "
        rf"$K={_integer(record['K'])}$; "
        rf"${_integer(record['qiskit_compiled']['n2q'])}/"
        rf"{_integer(record['qiskit_compiled']['d2q'])}$ "
        rf"[{_integer(estimator['queries'])}]"
    )


def _spectral_mean(row: Mapping[str, Any], method: str) -> float:
    return sum(
        float(row["transition_observables"][observable][method]["normalized_spectral_l1"])
        for observable in ("site0_density", "site0_displacement")
    ) / 2.0


def _generality_case_label(case_id: str) -> str:
    labels = {
        "peierls_weak_weak": r"Peierls weak--weak",
        "peierls_weak_strong": r"Peierls weak--strong",
        "peierls_u8_weak": r"Peierls $U=8$, weak bond",
        "siam_nb4_pilot": r"SIAM, $N_b=4$",
    }
    try:
        return labels[case_id]
    except KeyError as exc:
        raise ReportBuildError(f"unknown generality case {case_id!r}") from exc


def _generality_cases(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    order = {
        "peierls_weak_weak": 0,
        "peierls_weak_strong": 1,
        "peierls_u8_weak": 2,
        "siam_nb4_pilot": 3,
    }
    return sorted(
        payload["cases"].values(), key=lambda case: order[str(case["case_id"])]
    )


def _generality_crossing(route: Mapping[str, Any]) -> Mapping[str, Any]:
    crossing = route["threshold_crossings"]["1e-06"]
    if crossing.get("status") != "REACHED":
        raise ReportBuildError("expected a reached generality 1e-6 crossing")
    return crossing


def _trace_crossing_record(
    route: Mapping[str, Any], crossing: Mapping[str, Any]
) -> Mapping[str, Any]:
    crossing_k = int(crossing["k"])
    matching = [
        record
        for record in route["trace"]
        if int(record.get("k", -1)) == crossing_k
        and record.get("max_root_abs_error") is not None
        and abs(
            float(record["max_root_abs_error"])
            - float(crossing["max_root_abs_error"])
        )
        <= 1.0e-14
    ]
    if len(matching) != 1:
        raise ReportBuildError("could not identify unique generality crossing trace")
    return matching[0]


def _trace_crossing_rank(
    route: Mapping[str, Any], crossing: Mapping[str, Any]
) -> int:
    return int(_trace_crossing_record(route, crossing)["retained_rank"])


def _error_with_target_fraction(value: Any, target: float = 1.0e-6) -> str:
    error = float(value)
    return rf"{_sci(error)} ({error / target:.2g}$\times$)"


def _spectral_pair(route: Mapping[str, Any]) -> str:
    observables = route["static_spectral_audit"]["observables"]
    rendered: list[str] = []
    for observable in sorted(observables):
        spectrum = observables[observable]["broadened_spectrum"]
        if spectrum.get("comparison_status") == "ZERO_REFERENCE_WEIGHT":
            rendered.append(r"NA$_0$")
        else:
            rendered.append(_sci(spectrum["normalized_l1_error"], 1))
    return " / ".join(rendered)


def _plot_case_label(case_key: str) -> str:
    label = str(case_key)
    label = label.replace("intermediate", "int.")
    label = label.replace("strong_weak_u8", "strong-weak")
    label = label.replace("strong_strong_u8", "strong-strong")
    label = label.replace("weak_weak", "weak-weak")
    label = label.replace("weak_strong", "weak-strong")
    return label.replace("_nph", " / nph")


def _render_figures(payload: Mapping[str, Any], output_dir: Path) -> None:
    """Refresh the included figures with the report's reader-facing names."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(
        payload["rows"],
        key=lambda row: (int(row["case"]["n_ph_max"]), str(row["case"]["regime"])),
    )
    labels = [_plot_case_label(row["case"]["case_key"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.38
    figure, axes = plt.subplots(2, 1, figsize=(10.2, 7.2), sharex=True)
    for axis, observable, title in zip(
        axes,
        ("site0_density", "site0_displacement"),
        ("Site-0 density", "Site-0 displacement"),
        strict=True,
    ):
        selected = [
            float(row["transition_observables"][observable]["selected"]["normalized_spectral_l1"])
            for row in rows
        ]
        fixed = [
            float(
                row["transition_observables"][observable]["fixed_linear_response"][
                    "normalized_spectral_l1"
                ]
            )
            for row in rows
        ]
        axis.bar(
            [value - width / 2 for value in x],
            selected,
            width,
            color="#356cc8",
            label="operator-selected QSE (OS-QSE)",
        )
        axis.bar(
            [value + width / 2 for value in x],
            fixed,
            width,
            color="#c96d36",
            label="complete fixed LR-QSE",
        )
        axis.set_yscale("log")
        axis.set_ylabel(r"normalized spectral $L^1$")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2, which="both")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "static_spectral_errors.png", dpi=220)
    plt.close(figure)

    by_key = {row["case"]["case_key"]: row for row in rows}
    stress_keys = ("weak_strong_nph3", "intermediate_strong_nph3")
    observables = ("site0_density", "site0_displacement")
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=False)
    for row_index, case_key in enumerate(stress_keys):
        row = by_key[case_key]
        omega = row["spectral_grid"]["values"]
        for column_index, observable in enumerate(observables):
            axis = axes[row_index][column_index]
            block = row["transition_observables"][observable]
            exact = block["selected"]["spectrum"]["exact"]
            selected = block["selected"]["spectrum"]["predicted"]
            fixed = block["fixed_linear_response"]["spectrum"]["predicted"]
            axis.plot(
                omega,
                fixed,
                color="#c96d36",
                linewidth=1.1,
                linestyle="--",
                label="complete fixed LR-QSE",
                zorder=1,
            )
            axis.plot(
                omega,
                selected,
                color="#356cc8",
                linewidth=1.35,
                label="operator-selected QSE (OS-QSE)",
                zorder=2,
            )
            axis.plot(
                omega,
                exact,
                color="#111111",
                linewidth=1.05,
                linestyle=":",
                marker="o",
                markersize=2.0,
                markevery=40,
                label="exact reference",
                zorder=4,
            )
            observable_label = "site-0 density" if observable == "site0_density" else "site-0 displacement"
            axis.set_title(f"{_plot_case_label(case_key)} — {observable_label}")
            axis.set_xlabel(r"$\omega$")
            axis.set_ylabel(r"$A_O(\omega)$")
            axis.grid(alpha=0.18)
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "static_hard_case_spectra.png", dpi=220)
    plt.close(figure)


def _render_generality_band_figure(
    payload: Mapping[str, Any], krylov: Mapping[str, Any], output_dir: Path
) -> None:
    """Render separated six-root level diagrams at matched route crossings."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {
        "exact": "#111111",
        "selected": "#356cc8",
        "davidson": "#2c8b57",
        "krylov": "#7c4fb3",
        "fixed": "#c96d36",
    }
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    for axis, case in zip(axes.flat, _generality_cases(payload), strict=True):
        routes = case["routes"]
        selected_crossing = _generality_crossing(routes["selected_qse"])
        direct_crossing = _generality_crossing(routes["direct_davidson"])
        selected_trace = _trace_crossing_record(
            routes["selected_qse"], selected_crossing
        )
        direct_trace = _trace_crossing_record(
            routes["direct_davidson"], direct_crossing
        )
        fixed_trace = routes["fixed_qse"]["trace"][-1]
        fixed_resources = routes["fixed_qse"]["resources"]
        krylov_crossing = krylov["cases"][str(case["case_id"])][
            "threshold_crossings"
        ]["1e-06"]
        if krylov_crossing.get("status") != "REACHED":
            raise ReportBuildError("fresh generality Krylov case did not reach 1e-6")
        exact_roots = [float(value) for value in case["reference_roots"]]
        origin = exact_roots[0]
        level_sets = (
            ("exact", exact_roots),
            ("selected", selected_trace["root_energies"]),
            ("davidson", direct_trace["root_energies"]),
            ("krylov", krylov_crossing["matched_root_energies"]),
            ("fixed", fixed_trace["root_energies"]),
        )
        for x_position, (method, levels) in enumerate(level_sets):
            for level in levels:
                axis.hlines(
                    float(level) - origin,
                    x_position - 0.34,
                    x_position + 0.34,
                    color=colors[method],
                    linewidth=1.45 if method == "exact" else 1.2,
                    zorder=3,
                )

        selected_label = (
            "OS-QSE\n"
            f"K={_integer(selected_crossing['graph_span']['n2q'])}\n"
            f"Q={_integer(selected_crossing['qiskit_compiled']['n2q'])}; "
            f"M={_integer(selected_crossing['estimator']['queries'])}"
        )
        direct_label = (
            "Davidson\n"
            f"K={_integer(direct_crossing['graph_span']['n2q'])}\n"
            f"Q={_integer(direct_crossing['qiskit_compiled']['n2q'])}; "
            f"M={_integer(direct_crossing['estimator']['queries'])}"
        )
        krylov_label = (
            "single-seed RTK\n"
            f"K={_integer(krylov_crossing['K'])}\n"
            f"Q={_integer(krylov_crossing['qiskit_compiled']['n2q'])}; "
            f"M={_integer(krylov_crossing['estimator']['queries'])}"
        )
        fixed_label = (
            "fixed terminal\n"
            f"K={_integer(fixed_resources['graph_span']['n2q'])}\n"
            f"Q={_integer(fixed_resources['qiskit_compiled']['n2q'])}; "
            f"M={_integer(fixed_resources['estimator']['queries'])}"
        )
        axis.set_xticks(range(5))
        axis.set_xticklabels(
            ("exact\nreference", selected_label, direct_label, krylov_label, fixed_label)
        )
        axis.set_xlim(-0.55, 4.55)
        axis.set_ylabel(r"$E_r-E_1^{\rm exact}$")
        axis.set_title(_generality_case_label(str(case["case_id"])).replace("$", ""))
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", labelsize=7.5, pad=4)
    legend = (
        Line2D([0], [0], color=colors["exact"], lw=1.5, label="exact reference"),
        Line2D([0], [0], color=colors["selected"], lw=1.5, label="OS-QSE at first crossing"),
        Line2D([0], [0], color=colors["davidson"], lw=1.5, label="Davidson at first crossing"),
        Line2D([0], [0], color=colors["krylov"], lw=1.5, label="single-seed RTK at first crossing"),
        Line2D([0], [0], color=colors["fixed"], lw=1.5, label="fixed-class terminal"),
    )
    figure.legend(handles=legend, loc="upper center", ncol=5, frameon=False, fontsize=7.7)
    figure.text(
        0.5,
        0.018,
        r"Costs under each group: $K$ = graph-span $N_{2q}$; "
        r"$Q$ = Qiskit $N_{2q}$; $M$ = logical estimator queries.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 0.94), h_pad=2.0, w_pad=1.0)
    figure.savefig(output_dir / "generality_six_root_bands.png", dpi=240)
    plt.close(figure)


def _accuracy_table(
    rows: Sequence[Mapping[str, Any]],
    growth: Mapping[tuple[str, int], Mapping[str, Any]],
    target: str,
) -> str:
    body: list[str] = []
    for row in rows:
        case = row["case"]
        crossing = row["crossings"][target]
        status, krylov = _krylov_record(
            growth[(str(case["regime"]), int(case["n_ph_max"]))], target
        )
        body.append(
            " & ".join(
                (
                    _case_label(case),
                    _accuracy_cell(crossing["selected_n2q"]),
                    _accuracy_cell(crossing["direct_n2q"]),
                    _krylov_accuracy_cell(status, krylov),
                    _fixed_accuracy_cell(row["fixed_linear_response"]),
                )
            )
            + r" \\"
        )
    exponent = "-4" if target == "1e-4" else "-6"
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:accuracy_{target.replace('-', 'm')}}}{{Graph-span cost
$K$ at the first post-run crossing of
$\max_{{r\in\mathcal R}}|\Delta\varepsilon_r|\leq 10^{{{exponent}}}$.
OS-QSE and DQ use the locked current evidence; RTK uses its independently
generated trace after exact-reference equality was verified. R means reached,
NR means not reached within the declared RTK dimension cap (terminal error and
cost shown), and U means unattainable in the complete fixed LR-QSE manifold.}}
\begin{{center}}
\scriptsize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lcccc@{{}}}}
\toprule
Case & OS-QSE & direct quantum Davidson & seeded real-time Krylov & complete LR-QSE \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _route_audit_crossing_error(
    artifact: Mapping[str, Any], target: str
) -> float:
    path = Path(str(artifact["path"]))
    if _sha256(path) != artifact.get("sha256"):
        raise ReportBuildError(f"route artifact digest changed: {path}")
    route = _load_json(path)
    audit = route.get("post_run_exact_audit") or {}
    target_value = float(target)
    crossings = [
        crossing
        for crossing in audit.get("crossings") or []
        if abs(float(crossing.get("error_target", -1.0)) - target_value) <= 1.0e-15
    ]
    if len(crossings) != 1 or crossings[0].get("status") != "REACHED":
        raise ReportBuildError(f"route artifact lacks reached {target} crossing: {path}")
    crossing_k = int(crossings[0]["k"])
    rungs = [
        rung
        for rung in audit.get("rungs") or []
        if int(rung.get("k", -1)) == crossing_k
        and rung.get("max_root_abs_error") is not None
    ]
    if len(rungs) != 1:
        raise ReportBuildError(f"route artifact lacks unique crossing rung: {path}")
    return float(rungs[0]["max_root_abs_error"])


def _hh_achieved_error_table(
    rows: Sequence[Mapping[str, Any]],
    growth: Mapping[tuple[str, int], Mapping[str, Any]],
    target: str = "1e-6",
) -> str:
    target_value = float(target)
    body: list[str] = []
    for row in rows:
        case = row["case"]
        crossing = row["crossings"][target]
        selected_error = float(crossing["selected_max_root_abs_error"])
        direct_error = float(crossing["direct_max_root_abs_error"])
        krylov_status, krylov = _krylov_record(
            growth[(str(case["regime"]), int(case["n_ph_max"]))], target
        )
        krylov_error = _error_with_target_fraction(
            krylov["max_root_abs_error"], target_value
        )
        if krylov_status != "REACHED":
            krylov_error = "NR: " + krylov_error
        fixed_error = float(row["fixed_linear_response"]["max_root_energy_error"])
        body.append(
            " & ".join(
                (
                    _case_label(case),
                    _error_with_target_fraction(selected_error, target_value),
                    _error_with_target_fraction(direct_error, target_value),
                    krylov_error,
                    _error_with_target_fraction(fixed_error, target_value),
                )
            )
            + r" \\"
        )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:hh_achieved_errors}}{{Achieved maximum six-root error
at the $10^{{-6}}$ cost points in Table~\ref{{tab:accuracy_1em6}}. Each cell
also gives the error as a fraction of the target. NR marks a route that did not
cross; its terminal error is shown instead. This table prevents a cost
difference caused by accuracy overshoot from being mistaken for a selector
advantage.}}
\begin{{center}}
\scriptsize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lcccc@{{}}}}
\toprule
Case & OS-QSE & direct quantum Davidson & seeded real-time Krylov & complete LR-QSE \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _compiled_resource_table(
    rows: Sequence[Mapping[str, Any]],
    krylov: Mapping[tuple[str, int], Mapping[str, Any]],
    target: str,
) -> str:
    body: list[str] = []
    for row in rows:
        crossing = row["crossings"][target]
        fixed = row["fixed_linear_response"]
        case = row["case"]
        krylov_status, krylov_record = _krylov_record(
            krylov[(str(case["regime"]), int(case["n_ph_max"]))], target
        )
        krylov_cell = _qiskit_cell(
            krylov_record["qiskit_compiled"],
            krylov_record["estimator"]["queries"],
        )
        if krylov_status != "REACHED":
            krylov_cell = "NR: " + krylov_cell
        body.append(
            " & ".join(
                (
                    _case_label(row["case"]),
                    _qiskit_cell(
                        crossing["selected_qiskit_compiled"],
                        crossing["selected_estimator_queries"],
                    ),
                    _qiskit_cell(
                        crossing["direct_qiskit_compiled"],
                        crossing["direct_estimator_queries"],
                    ),
                    krylov_cell,
                    _qiskit_cell(fixed["qiskit_compiled"], fixed["estimator_queries"]),
                )
            )
            + r" \\"
        )
    exponent = "-4" if target == "1e-4" else "-6"
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:qiskit_{target.replace('-', 'm')}}}{{Post-run Qiskit
resource annotations at the $10^{{{exponent}}}$
crossing. Numeric cells are $N_{{2q}}/D$ [logical estimator queries].
Compilation uses FakeMarrakesh, optimization level 1, and transpiler seed 7.
For RTK, NR marks a terminal-cap resource rather than a crossing. Fixed LR-QSE
is shown at its complete terminal manifold because it does not reach either
energy target.}}
\begin{{center}}
\scriptsize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lcccc@{{}}}}
\toprule
Case & OS-QSE & direct quantum Davidson & seeded real-time Krylov & complete LR-QSE \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _krylov_resource_table(
    rows: Sequence[Mapping[str, Any]],
    krylov: Mapping[tuple[str, int], Mapping[str, Any]],
) -> str:
    body: list[str] = []
    for row in rows:
        case = row["case"]
        krylov_case = krylov[(str(case["regime"]), int(case["n_ph_max"]))]
        cells = []
        for target in ("1e-4", "1e-6"):
            status, record = _krylov_record(krylov_case, target)
            cells.append(_krylov_resource_cell(status, record))
        body.append(" & ".join((_case_label(case), *cells)) + r" \\")
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:krylov_resources}}{{Complete seeded real-time-Krylov
resource matrix after one-to-one Hungarian root assignment. Each cell is
status and $k$/retained rank: graph-span $K$; Qiskit
$N_{{2q}}/D_{{2q}}$ [logical estimator queries] at the first crossing or, for
NR, at the terminal cap. Every Krylov-basis preparation is charged.}}
\begin{{center}}
\fontsize{{6.5}}{{7.2}}\selectfont
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lcc@{{}}}}
\toprule
Case & $10^{{-4}}$ target & $10^{{-6}}$ target \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _terminal_table(rows: Sequence[Mapping[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        selected = row["selected_terminal"]
        body.append(
            " & ".join(
                (
                    _case_label(row["case"]),
                    rf"${selected['support_size']}/{selected['retained_rank']}$",
                    _integer(selected["exchange_rung_count"]),
                    _qiskit_cell(
                        selected["qiskit_compiled"], selected["estimator_queries"]
                    ),
                    _sci(selected["max_root_energy_error"]),
                    _sci(_spectral_mean(row, "selected")),
                    _sci(_spectral_mean(row, "fixed_linear_response")),
                )
            )
            + r" \\"
        )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:terminal_selected}}{{Terminal OS-QSE diagnostics.
Support/rank is the admitted support
size and supported-overlap retained rank. Exchange rungs count evaluated
exchange locations, not committed patches. The resource cell is Qiskit
$N_{{2q}}/D$ [logical estimator queries]. The spectral columns are the mean
normalized Lorentzian $L^1$ error over site-0 density and displacement;
$\eta=0.05$.}}
\begin{{center}}
\scriptsize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lcccccc@{{}}}}
\toprule
Case & support/rank & exchange rungs & Qiskit/resource & max root error & OS-QSE $L^1$ & LR-QSE $L^1$ \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _generality_resource_table(
    payload: Mapping[str, Any], krylov: Mapping[str, Any]
) -> str:
    body: list[str] = []
    for case in _generality_cases(payload):
        routes = case["routes"]
        selected = _generality_crossing(routes["selected_qse"])
        direct = _generality_crossing(routes["direct_davidson"])
        krylov_crossing = krylov["cases"][str(case["case_id"])][
            "threshold_crossings"
        ]["1e-06"]
        fixed = routes["fixed_qse"]["resources"]
        body.append(
            " & ".join(
                (
                    _generality_case_label(str(case["case_id"])),
                    _integer(selected["graph_span"]["n2q"]),
                    _qiskit_cell(
                        selected["qiskit_compiled"], selected["estimator"]["queries"]
                    ),
                    _integer(direct["graph_span"]["n2q"]),
                    _qiskit_cell(
                        direct["qiskit_compiled"], direct["estimator"]["queries"]
                    ),
                    _integer(krylov_crossing["K"]),
                    _qiskit_cell(
                        krylov_crossing["qiskit_compiled"],
                        krylov_crossing["estimator"]["queries"],
                    ),
                    _qiskit_cell(
                        fixed["qiskit_compiled"], fixed["estimator"]["queries"]
                    ),
                )
            )
            + r" \\"
        )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:generality_resources}}{{Fresh cross-family resource
evidence at the first crossing of
$\max_{{r\in\mathcal R}}|\Delta\varepsilon_r|\leq10^{{-6}}$.
Numeric compiled cells are Qiskit $N_{{2q}}/D_{{2q}}$ [logical estimator
queries]. $K$ is the separate online graph-span two-qubit coordinate. The
complete fixed-class QSE cell is terminal because that route reaches none of
the three energy targets. RTK charges all Krylov-basis preparations but excludes
the common source-state preparation under its inherited comparator convention.}}
\begin{{center}}
\scriptsize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lccccccc@{{}}}}
\toprule
& \multicolumn{{2}}{{c}}{{cost-weighted operator-selected QSE}} &
\multicolumn{{2}}{{c}}{{direct quantum Davidson}} &
\multicolumn{{2}}{{c}}{{single-seed RTK}} & fixed-class QSE \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}\cmidrule(l){{8-8}}
Case & $K$ & Qiskit [queries] & $K$ & Qiskit [queries] & $K$ & Qiskit [queries] & Qiskit [queries] \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _generality_certification_table(payload: Mapping[str, Any]) -> str:
    body: list[str] = []
    for case in _generality_cases(payload):
        routes = case["routes"]
        selected_route = routes["selected_qse"]
        direct_route = routes["direct_davidson"]
        fixed_route = routes["fixed_qse"]
        selected = _generality_crossing(selected_route)
        direct = _generality_crossing(direct_route)
        fixed_terminal = fixed_route["trace"][-1]
        selected_shape = rf"${int(selected['support_size'])}/{_trace_crossing_rank(selected_route, selected)}$"
        direct_shape = rf"${int(direct['support_size'])}/{_trace_crossing_rank(direct_route, direct)}$"
        if fixed_terminal.get("max_root_abs_error") is None:
            fixed_result = rf"rank ${int(fixed_terminal['retained_rank'])}<6$"
        else:
            fixed_result = (
                rf"rank ${int(fixed_terminal['retained_rank'])}$; "
                + _sci(fixed_terminal["max_root_abs_error"])
            )
        body.append(
            " & ".join(
                (
                    _generality_case_label(str(case["case_id"])),
                    selected_shape,
                    _error_with_target_fraction(selected["max_root_abs_error"]),
                    direct_shape,
                    _error_with_target_fraction(direct["max_root_abs_error"]),
                    fixed_result,
                )
            )
            + r" \\"
        )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:generality_certification}}{{Six-root certification
at the matched $10^{{-6}}$ threshold. ``First crossing'' is route-local: for
each route separately, growth prefixes are scanned in increasing admitted
dimension and the earliest prefix satisfying the tolerance is reported. It
does not mean the first route or table column. Shape is admitted
dimension/retained supported-overlap rank. Each error cell gives the achieved
maximum error and, in parentheses, its fraction of the $10^{{-6}}$ threshold.
The final column reports why the cheaper fixed class is not a solution to the
declared task.}}
\begin{{center}}
\scriptsize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lccccc@{{}}}}
\toprule
Case & OS-QSE shape & OS-QSE max error & Davidson shape & Davidson max error & fixed-class outcome \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _generality_krylov_outcome_table(
    payload: Mapping[str, Any], krylov: Mapping[str, Any]
) -> str:
    body: list[str] = []
    for case in _generality_cases(payload):
        crossing = krylov["cases"][str(case["case_id"])]["threshold_crossings"][
            "1e-06"
        ]
        body.append(
            " & ".join(
                (
                    _generality_case_label(str(case["case_id"])),
                    rf"${int(crossing['k'])}/{int(crossing['retained_rank'])}$",
                    _error_with_target_fraction(crossing["max_root_abs_error"]),
                    _integer(crossing["K"]),
                    _qiskit_cell(
                        crossing["qiskit_compiled"],
                        crossing["estimator"]["queries"],
                    ),
                )
            )
            + r" \\"
        )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:generality_krylov}}{{Single-seed RTK outcomes for the
four fresh cross-family cases under the fixed $\Delta t=0.5$ policy. Shape is
$k$/retained supported-overlap rank. Errors use one-to-one Hungarian root
assignment and include their fraction of the $10^{{-6}}$ target. The Qiskit
cell is $N_{{2q}}/D_{{2q}}$ [logical estimator queries].}}
\begin{{center}}
\scriptsize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lcccc@{{}}}}
\toprule
Case & RTK shape & RTK max error & $K$ & Qiskit [queries] \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _generality_spectral_table(
    payload: Mapping[str, Any], krylov: Mapping[str, Any]
) -> str:
    body: list[str] = []
    for case in _generality_cases(payload):
        routes = case["routes"]
        observable_names = sorted(
            routes["selected_qse"]["static_spectral_audit"]["observables"]
        )
        observable_labels = {
            "bond_displacement": "bond displacement",
            "site0_density": "site-0 density",
            "impurity_double_occupancy": "double occupancy",
            "impurity_magnetization": "magnetization",
        }
        body.append(
            " & ".join(
                (
                    _generality_case_label(str(case["case_id"])),
                    " / ".join(observable_labels[name] for name in observable_names),
                    _spectral_pair(routes["selected_qse"]),
                    _spectral_pair(routes["direct_davidson"]),
                    _spectral_pair(krylov["cases"][str(case["case_id"])]),
                    _spectral_pair(routes["fixed_qse"]),
                )
            )
            + r" \\"
        )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:generality_spectra}}{{Post-run static spectral audit.
Each route cell gives the normalized broadened-$L^1$ errors in the observable
order shown; the Lorentzian width is $\eta=0.05$. NA$_0$ means the exact
integrated reference weight is numerically zero, so no normalized ratio is
reported. Exact excited states enter only this audit, never selection or
stopping.}}
\begin{{center}}
\tiny
\begin{{tabular}}{{@{{}}p{{0.14\textwidth}}p{{0.20\textwidth}}p{{0.14\textwidth}}p{{0.14\textwidth}}p{{0.14\textwidth}}p{{0.14\textwidth}}@{{}}}}
\toprule
Case & observables (reported order) & OS-QSE & direct quantum Davidson & single-seed RTK & fixed-class QSE \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular}}
\end{{center}}
\end{{widetext}}
"""


def _matrix_case_label(case_id: str) -> str:
    if case_id.startswith("hh::"):
        _, regime, nph = case_id.split("::")
        return _case_label(
            {"regime": regime, "n_ph_max": int(nph.removeprefix("nph"))}
        )
    return _generality_case_label(case_id)


def _matrix_outcome(
    matrix: Mapping[str, Any], case_id: str
) -> tuple[str, Mapping[str, Any]]:
    return _krylov_record(matrix["cases"][case_id], "1e-6")


def _coverage_map_table(
    single: Mapping[str, Any], block: Mapping[str, Any]
) -> str:
    body: list[str] = []
    for case_id in sorted(
        single["cases"],
        key=lambda value: (
            0 if str(value).startswith("hh::") else 1,
            _matrix_case_label(str(value)),
        ),
    ):
        single_status, single_record = _matrix_outcome(single, str(case_id))
        block_status, block_record = _matrix_outcome(block, str(case_id))
        single_cell = (
            rf"R: {_sci(single_record['max_root_abs_error'])}"
            if single_status == "REACHED"
            else rf"NR: {_sci(single_record['max_root_abs_error'])}"
        )
        block_cell = (
            rf"R: {_sci(block_record['max_root_abs_error'])}"
            if block_status == "REACHED"
            else rf"NR: {_sci(block_record['max_root_abs_error'])}"
        )
        union = (
            "both"
            if single_status == block_status == "REACHED"
            else "single only"
            if single_status == "REACHED"
            else "block only"
            if block_status == "REACHED"
            else "neither"
        )
        body.append(
            " & ".join(
                (
                    _matrix_case_label(str(case_id)),
                    single_cell,
                    block_cell,
                    union,
                )
            )
            + r" \\"
        )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:krylov_coverage_map}}{{Six-root $10^{{-6}}$
coverage under two separately declared fixed real-time-Krylov initializations.
R means reached; NR reports the charged terminal-cap error. The final column is
a post-run comparison of two methods, not a case-wise oracle route and not a
hybrid algorithm.}}
\begin{{center}}
\scriptsize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lccc@{{}}}}
\toprule
Case & single-seed RTK & six-probe block RTK & success-set relation \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _block_krylov_resource_table(block: Mapping[str, Any]) -> str:
    body: list[str] = []
    for case_id in sorted(
        block["cases"],
        key=lambda value: (
            0 if str(value).startswith("hh::") else 1,
            _matrix_case_label(str(value)),
        ),
    ):
        status, record = _matrix_outcome(block, str(case_id))
        body.append(
            " & ".join(
                (
                    _matrix_case_label(str(case_id)),
                    "R" if status == "REACHED" else "NR",
                    rf"\({int(record['k'])}/{int(record['retained_rank'])}\)",
                    _sci(record["max_root_abs_error"]),
                    _integer(record["K"]),
                    rf"\({_integer(record['qiskit_compiled']['n2q'])}/"
                    rf"{_integer(record['qiskit_compiled']['d2q'])}\)",
                    _integer(record["estimator"]["queries"]),
                )
            )
            + r" \\"
        )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:block_krylov_resources}}{{Complete six-probe block-RTK
matrix at the first $10^{{-6}}$ crossing or, for NR, the charged terminal cap.
Shape is $k$/retained rank. Qiskit is aggregate
$N_{{2q}}/D_{{2q}}$ over independently prepared block-Krylov states. The
resource ledger charges every repeated source preparation and Hamiltonian
action; computational-basis source preparations contribute one-qubit but zero
two-qubit cost.}}
\begin{{center}}
\scriptsize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lcccccc@{{}}}}
\toprule
Case & status & shape & max error & $K$ & Qiskit $N_{{2q}}/D_{{2q}}$ & queries \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _block_source_failure_table(block: Mapping[str, Any]) -> str:
    body: list[str] = []
    for case_id, case in sorted(
        block["cases"].items(), key=lambda item: _matrix_case_label(str(item[0]))
    ):
        status, terminal = _matrix_outcome(block, str(case_id))
        if status == "REACHED":
            continue
        audit = case["post_run_source_coverage_audit"]
        indices = [
            int(value)
            for value in audit["target_root_indices_below_zero_weight_floor"]
        ]
        weights = audit["source_block_overlap_weight_by_root"]
        body.append(
            " & ".join(
                (
                    _matrix_case_label(str(case_id)),
                    ", ".join(rf"$r={index}$" for index in indices),
                    ", ".join(_sci(weights[index - 1], 1) for index in indices),
                    _sci(terminal["max_root_abs_error"]),
                    _integer(terminal["K"]),
                )
            )
            + r" \\"
        )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:block_source_failures}}{{Post-run source-coverage
diagnosis for all five block-RTK failures. The weight is the squared norm of
the exact target root projected onto the declared six-probe source block;
the zero-weight floor is $10^{{-20}}$. Exact roots were unavailable to the
online route. For an exact eigenstate absent from the fixed source block's
spectral support, extending the same time grid cannot create that component.
Individual weights inside an exactly degenerate eigenspace remain
basis-dependent, so the conclusion is about the represented target subspace.}}
\begin{{center}}
\scriptsize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lcccc@{{}}}}
\toprule
Case & absent target & source weight & terminal max error & terminal $K$ \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _pool_outcome_cell(
    route: Mapping[str, Any], target: float, problem: Any
) -> str:
    from pipelines.exact_bench.paper_iii_qse_generality_campaign import (
        _record_qiskit_cost,
    )

    result = _first_reached_online_rung(route, target=target)
    if result is not None:
        online, _ = result
        compiled = _record_qiskit_cost(
            problem, tuple(int(index) for index in online["support_indices"])
        )
        return (
            rf"R $k={int(online['k'])}$, $K={_integer(online['n2q'])}$, "
            rf"error {_sci(online['max_root_abs_error'], 1)}; "
            rf"Q ${_integer(compiled['n2q'])}/{_integer(compiled['d2q'])}$, "
            rf"M {_integer(online['estimator']['queries'])}"
        )
    online_rows = route["online_receipt"]["rows"]
    audit_by_k = {
        int(item["k"]): item for item in route["post_run_exact_audit"]["rungs"]
    }
    terminal = online_rows[-1]
    compiled = _record_qiskit_cost(
        problem, tuple(int(index) for index in terminal["support_indices"])
    )
    error = audit_by_k[int(terminal["k"])]["max_root_abs_error"]
    error_cell = "rank $<6$" if error is None else _sci(error, 1)
    return (
        rf"NR: {error_cell} at $K={_integer(terminal['n2q'])}$; "
        rf"Q ${_integer(compiled['n2q'])}/{_integer(compiled['d2q'])}$, "
        rf"M {_integer(terminal['estimator']['queries'])}"
    )


def _pool_ablation_table(payload: Mapping[str, Any]) -> str:
    from pipelines.qse_spectra.paper_iii_problem import load_problem
    from pipelines.qse_spectra.pool_profiles import apply_hh_pool_policy

    policies = (
        "hh_full_meta_macro_v1",
        "hh_full_meta_plus_encoded_boson_pauli_completion_v1",
    )
    body: list[str] = []
    for row in payload["rows"]:
        case = row["case"]
        route_by_policy: dict[str, Mapping[str, Any]] = {}
        for policy in policies:
            matched_path = _resolve_artifact_path(
                payload["profiles"][policy]["matched_evidence_path"]
            )
            route_path = (
                matched_path.parent
                / "routes"
                / str(case["case_key"])
                / "record_first_adaptive_qse.json"
            )
            route_by_policy[policy] = _load_json(route_path)
        base_problem = load_problem(
            regime=str(case["regime"]),
            u=float(case["u"]),
            g_ep=float(case["g_ep"]),
            n_ph_max=int(case["n_ph_max"]),
            target_roots=6,
        )
        problem_by_policy = {
            policies[0]: base_problem,
            policies[1]: apply_hh_pool_policy(base_problem, policies[1]),
        }
        completion_route = route_by_policy[policies[1]]
        completion_support = completion_route["online_receipt"]["rows"][-1][
            "support_indices"
        ]
        problem = problem_by_policy[policies[1]]
        completion_count = sum(
            str((problem.basis[int(index)].metadata or {}).get("record_family", ""))
            == "encoded_boson_completion"
            for index in completion_support
        )
        body.append(
            " & ".join(
                (
                    _case_label(case),
                    _pool_outcome_cell(
                        route_by_policy[policies[0]],
                        1.0e-4,
                        problem_by_policy[policies[0]],
                    ),
                    _pool_outcome_cell(
                        route_by_policy[policies[1]],
                        1.0e-4,
                        problem_by_policy[policies[1]],
                    ),
                    str(completion_count),
                )
            )
            + r" \\"
        )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:pool_ablation}}{{Current-selector pool-profile
ablation at the route-local first $10^{{-4}}$ crossing. Each route cell gives
status, $k$, graph-span $K$, error, Qiskit aggregate $Q=N_{{2q}}/D_{{2q}}$,
and logical queries $M$ at the same prefix (or charged terminal cap). The selector, seed,
supported-overlap cutoff, dimension cap, stopping rule, and no-fallback
boundary are held fixed. The last column counts encoded-boson completion
records in the terminal selected support; it establishes use of the added
family, not the marginal causal value of each record.}}
\begin{{center}}
\fontsize{{5.4}}{{6.1}}\selectfont
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lccc@{{}}}}
\toprule
Case & base physical-macro pool & completion pool & completion records at terminal \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _crossing_reconciliation_table(
    receipts: Sequence[Mapping[str, Any]],
) -> str:
    changed = [row for row in receipts if row["differs_from_sealed_summary"]]
    body = os.linesep.join(
        " & ".join(
            (
                _tex_escape(row["case_key"]),
                _tex_escape(row["route"]),
                rf"$10^{{{'-4' if row['target'] == '1e-4' else '-6'}}}$",
                rf"\({int(row['sealed_summary_k'])}/{_integer(row['sealed_summary_K'])}\)",
                rf"\({int(row['derived_k'])}/{_integer(row['derived_K'])}\)",
                _sci(row["derived_error"]),
            )
        )
        + r" \\"
        for row in changed
    )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:crossing_reconciliation}}{{Publication-crossing
reconciliation. The sealed Hubbard--Holstein campaign summary minimizes
graph-span $K$ over successful rungs, whereas this report and the manuscript
use the earliest successful prefix in declared growth order. Only the rows
shown differ. Entries are $k/K$; all resource coordinates in the main tables
are rebuilt at the publication prefix.}}
\begin{{center}}
\scriptsize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lccccc@{{}}}}
\toprule
Case & route & target & sealed minimum-$K$ & publication first & first-prefix error \\
\midrule
{body}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _screening_work_table(rows: Sequence[Mapping[str, Any]]) -> str:
    body: list[str] = []
    total_evaluations = 0
    total_row_elements = 0
    for row in rows:
        route = _load_verified_route(row["route_artifact"])
        trace = route["online_receipt"]["completion"]["record_stop"][
            "round_trace"
        ]
        evaluations = sum(
            int(round_row["screening"]["full_proposed_pencil_evaluations"])
            for round_row in trace
        )
        row_elements = sum(
            int(round_row["screening"]["full_proposed_hamiltonian_row_elements"])
            for round_row in trace
        )
        total_evaluations += evaluations
        total_row_elements += row_elements
        terminal = row["selected_terminal"]
        body.append(
            " & ".join(
                (
                    _case_label(row["case"]),
                    _integer(len(trace)),
                    _integer(evaluations),
                    _integer(row_elements),
                    _integer(terminal["support_size"]),
                    _integer(terminal["n2q"]),
                    _integer(terminal["exchange_rung_count"]),
                )
            )
            + r" \\"
        )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:screening_work}}{{Recorded acquisition activity for
the canonical Hubbard--Holstein OS-QSE routes. Proposed-pencil evaluations and
Hamiltonian row elements count online screening work. Terminal support and
$K$ describe the surviving selected manifold and do \emph{{not}} charge that
screening work. Exchange rungs are evaluated maintenance locations, not
accepted-patch counts. Across the matrix the traces record
{_integer(total_evaluations)} proposed-pencil evaluations and
{_integer(total_row_elements)} proposed Hamiltonian row elements.}}
\begin{{center}}
\fontsize{{6.5}}{{7.2}}\selectfont
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lrrrrrr@{{}}}}
\toprule
Case & rounds & pencil evaluations & row elements & terminal support & terminal $K$ & exchange rungs \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _benchmark_inventory_table(
    rows: Sequence[Mapping[str, Any]], generality: Mapping[str, Any]
) -> str:
    from pipelines.qse_spectra.paper_iii_problem import load_problem
    from pipelines.qse_spectra.pool_profiles import (
        HH_POOL_ENCODED_BOSON_COMPLETION_V1,
        apply_hh_pool_policy,
    )

    hh_pools = sorted(
        {
            len(
                apply_hh_pool_policy(
                    load_problem(
                        regime=str(row["case"]["regime"]),
                        u=float(row["case"]["u"]),
                        g_ep=float(row["case"]["g_ep"]),
                        n_ph_max=int(row["case"]["n_ph_max"]),
                        target_roots=6,
                    ),
                    HH_POOL_ENCODED_BOSON_COMPLETION_V1,
                ).basis
            )
            for row in rows
        }
    )
    body = [
        " & ".join(
            (
                r"Hubbard--Holstein dimer",
                r"12: six couplings $\times\ n_{\rm ph}=1,3$",
                r"encoded fermion--boson sector",
                ", ".join(str(value) for value in hh_pools),
                r"exact same-sector ground state",
            )
        )
        + r" \\"
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in _generality_cases(generality):
        grouped.setdefault(str(case["model_family"]), []).append(case)
    labels = {
        "peierls_hubbard": "Peierls--Hubbard dimer",
        "siam_finite_box": r"finite-box SIAM, $N_b=4$",
    }
    for family, cases in sorted(grouped.items()):
        receipts = [case["problem_receipt"] for case in cases]
        body.append(
            " & ".join(
                (
                    labels.get(family, family.replace("_", "--")),
                    str(len(cases)),
                    ", ".join(
                        sorted({f"{int(receipt['num_qubits'])} qubits" for receipt in receipts})
                    ),
                    ", ".join(
                        sorted({str(int(receipt["pool_size"])) for receipt in receipts})
                    ),
                    "exact same-sector ground state",
                )
            )
            + r" \\"
        )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:benchmark_inventory}}{{Benchmark inventory. Every
cell targets the six lowest excited roots in the seed sector. The exact
same-cutoff ground state is supplied as a benchmark-isolation seed; this
report therefore does not demonstrate approximate ground-state preparation or
bosonic-cutoff convergence. Exact excited roots enter only retrospective
audits.}}
\begin{{center}}
\small
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}p{{0.19\textwidth}}p{{0.23\textwidth}}p{{0.18\textwidth}}p{{0.12\textwidth}}p{{0.21\textwidth}}@{{}}}}
\toprule
Family & cases & encoded size & pool size(s) & seed used here \\
\midrule
{os.linesep.join(body)}
\bottomrule
\end{{tabular*}}
\end{{center}}
\end{{widetext}}
"""


def _method_definition() -> str:
    return r"""
\section{Production method identity}

Let $\mathcal B_\ell$ be the admitted record support and let
$\mathcal R=\{1,\ldots,R\}$, with $R=6$ here.  The ideal task is the
minimum-cost support that resolves the declared window,
\[
 \min_{\mathcal B\subseteq\mathcal P}K(\mathcal B)
 \quad\text{subject to}\quad
 \max_{r\in\mathcal R}|\varepsilon_r(\mathcal B)-\varepsilon_r^\star|
 \leq\epsilon_E .
\]
The exact roots $\varepsilon_r^\star$ define the retrospective benchmark and
are unavailable to online selection and stopping.  The implemented greedy
surrogate recomputes the supported generalized pencil for every admissible
proposal $A$ and uses
\[
 \mathcal F_{\mathcal R}(\mathcal B)=
 \sum_{r\in\mathcal R}\varepsilon_r(\mathcal B),\qquad
 \Delta_\ell(A)=\widetilde{\mathcal F}_{\mathcal R}(\mathcal B_\ell)
 -\widetilde{\mathcal F}_{\mathcal R}(\mathcal B_\ell\cup\{A\}).
\]
Before $R$ supported roots exist, missing terms in the proposed objective are
padded by one common round-local ceiling, $\max_A h_{11}(A)$.  This startup
rule rewards rank creation without injecting exact spectral information.

Within each round, positive gains and incremental graph-span costs are
normalized by their respective round maxima.  With
$\overline\Delta_\ell(A)=\Delta_\ell(A)/\max_{A'}\Delta_\ell(A')$ and
$\overline K_\ell(A)=K_\ell(A)/\max_{A'}K_\ell(A')$, the production score is
\[
 \mathcal S_\ell(A)=
 \frac{\overline\Delta_\ell(A)}
 {\max(0.05,\overline K_\ell(A))}.
\]
Away from the declared cost floor this orders candidates by gain per cost.
Novelty is not an additive reward: it is an admissibility floor of $10^{-12}$
against the retained supported-overlap frame.  Candidate residualization uses
two Gram--Schmidt passes against that frame.  Deterministic index order breaks
numerical ties.

Online stopping requires both a maximum target residual below $10^{-3}$ and
absence of an unused novel candidate whose diagonal $h_{11}$ lies within
$10^{-3}$ above the current $R$th Ritz value.  The second condition is
pool-relative; it is not a proof that an unrepresented lower state does not
exist outside the declared pool.  Pool exhaustion or failure to certify the
target window is returned as a structured failure status.  Davidson is never
invoked inside OS-QSE.

Maintenance proposes singleton delete--insert patches.  A $1\leftrightarrow1$
patch is applied atomically, the complete supported pencil is re-solved, and
the whole transaction is accepted only if the Ky--Fan objective/cost
dominance, retained-rank, and conditioning guards pass.  Rejection leaves the
support unchanged.  This is true joint exchange at singleton cardinality; it
does not imply the multi-record deletion ladders used by broader exchange
formulations.
"""


def _planned_placeholders() -> str:
    rows = (
        (
            r"Hubbard--Holstein, $n_{\mathrm{ph}}=1,3$",
            "OS-QSE; direct quantum Davidson; complete LR-QSE",
            "COMPLETE",
            "Locked 12-cell energy, static-spectrum, Qiskit, and query evidence",
        ),
        (
            r"Hubbard--Holstein, $n_{\mathrm{ph}}=1,3$",
            "single-seed real-time Krylov",
            "COMPLETE",
            "Fixed-policy matrix and audits complete; 6/12 reach the 1e-6 window",
        ),
        (
            "All three model families",
            "six-probe block-seeded real-time Krylov",
            "COMPLETE",
            "Fixed policy, 16-case matrix, charged resources, spectra, and source-coverage audit; 11/16 reach",
        ),
        (
            r"Hubbard--Holstein, $n_{\mathrm{ph}}=1,3$",
            "Hamiltonian-power quantum Krylov/QKSD",
            "IMPLEMENTATION_PENDING",
            "Desired Krylov comparator; no faithful implementation artifact",
        ),
        (
            "Peierls--Hubbard, three regimes",
            "OS-QSE; direct quantum Davidson; complete fixed-class QSE",
            "COMPLETE",
            "Matched 1e-6 energy, static-spectrum, Qiskit, query, and replay evidence",
        ),
        (
            "Peierls--Hubbard, three regimes",
            "single-seed real-time Krylov",
            "COMPLETE",
            "Fixed-policy matrix, Qiskit, query, and spectral audits; 3/3 reach",
        ),
        (
            "SIAM, $N_b=4$",
            "OS-QSE; direct quantum Davidson; complete fixed-class QSE",
            "COMPLETE",
            "Matched 1e-6 energy, static-spectrum, Qiskit, query, and replay evidence",
        ),
        (
            "SIAM, $N_b=4$",
            "single-seed real-time Krylov",
            "COMPLETE",
            "Fixed-policy matrix, Qiskit, query, and spectral audits; 1/1 reaches",
        ),
        (
            "Current OS-QSE mechanism",
            "cost-discount-off ablation",
            "PENDING",
            "Hold pool, seed, overlap cutoff, stopping, and exchange fixed",
        ),
        (
            "Current OS-QSE mechanism",
            "exchange-off ablation and accepted-patch ledger",
            "PENDING",
            "Current report counts evaluated exchange locations, not accepted patches",
        ),
        (
            "Construction accounting",
            "cumulative acquisition-work ledger",
            "PENDING",
            "Screening, rejected proposals, and exchange evaluations are not in final-support costs",
        ),
        (
            "Publication comparators",
            "qEOM/q-sc-EOM, VQD, SSVQE",
            "SCOPE_DECISION_REQUIRED",
            "Numerical comparison or explicit scoped literature discussion remains to be chosen",
        ),
        (
            "All model/method cells",
            "finite-shot resource evidence",
            "NOT_PLANNED",
            "No shot allocation or sampling model is declared in the current route",
        ),
    )
    body = os.linesep.join(
        " & ".join((model, method, _status(status), _tex_escape(detail))) + r" \\"
        for model, method, status, detail in rows
    )
    return rf"""
\begin{{widetext}}
\inlinetablecaption{{tab:planned_evidence}}{{Declared remaining evidence.
These placeholders are part of the
report contract: incomplete or desired results retain their final table
position and cannot disappear from later status reports.}}
\begin{{center}}
\scriptsize
\begin{{tabular}}{{@{{}}p{{0.20\textwidth}}p{{0.24\textwidth}}p{{0.17\textwidth}}p{{0.34\textwidth}}@{{}}}}
\toprule
Model/scope & method or evidence & status & completion boundary \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{center}}
\end{{widetext}}
"""


def _method_key() -> str:
    return r"""
\begin{widetext}
\inlinetablecaption{tab:method_key}{Reader-facing method architecture. There is one proposed method,
three external iterative comparators, and one conventional static baseline.
``Selected'' and ``direct'' are not algorithm names, and there is no method
called ``direct selected.''}
\begin{center}
\small
\begin{tabular}{@{}p{0.11\textwidth}p{0.23\textwidth}p{0.16\textwidth}p{0.44\textwidth}@{}}
\toprule
Role & full name & acronym / route id & construction and boundary \\
\midrule
Proposed & cost-weighted operator-selected QSE with certified atomic exchange
& OS-QSE / \nolinkurl{record_first_adaptive_qse}
& Full proposed-pencil Ky--Fan gain $\Delta$ divided by incremental cost $K$,
novelty floor, completion operator pool, and atomic $1\leftrightarrow1$
exchange. No Davidson fallback. \\
Comparator & independently initialized direct quantum Davidson
& DQ / \nolinkurl{direct_davidson_qse}
& Residual-synthesized directions grown independently of the operator pool. \\
Comparator & seeded real-time Krylov
& RTK / \nolinkurl{seeded_realtime_krylov}
& One deterministic complex sector kick, fixed $\Delta t=0.5$, supported
overlap, and one-to-one Hungarian post-run root assignment. Caps are 100 for
Hubbard--Holstein/SIAM and 40 for Peierls; every Krylov-state preparation is
charged, while the common source-state preparation is excluded. \\
Comparator & six-probe block-seeded real-time Krylov
& block RTK / \nolinkurl{block_seeded_realtime_krylov}
& Six lowest-diagonal computational probes that survive $Q_0$ projection and
double Gram--Schmidt, fixed block size six, fixed $\Delta t=0.5$, and
one-to-one post-run assignment. Every repeated source preparation and
Hamiltonian action is charged; the exact ground-state preparation is excluded. \\
Baseline & complete declared fixed-class QSE
& fixed QSE / \nolinkurl{fixed_linear_response_qse}; \nolinkurl{fixed_class_qse}
& The complete declared non-adaptive class for each model family. In the
Hubbard--Holstein matrix this is the conventional linear-response (LR-QSE)
manifold. \\
Ablation & record-only growth and selector-score variants
& not public methods
& Mechanism tests only; they never occupy comparator columns. \\
\bottomrule
\end{tabular}
\end{center}
\end{widetext}
"""


def _build_tex(
    payload: Mapping[str, Any],
    krylov_hh: Mapping[tuple[str, int], Mapping[str, Any]],
    krylov: Mapping[str, Any],
    block_krylov: Mapping[str, Any],
    pool_ablation: Mapping[str, Any],
    generality: Mapping[str, Any],
    crossing_receipts: Sequence[Mapping[str, Any]],
    *,
    input_path: Path,
    krylov_path: Path,
    krylov_receipt_path: Path,
    block_krylov_path: Path,
    block_krylov_receipt_path: Path,
    pool_ablation_path: Path,
    pool_ablation_status_path: Path,
    generality_path: Path,
    generality_receipt_path: Path,
    output_dir: Path,
) -> str:
    rows = sorted(
        payload["rows"],
        key=lambda row: (int(row["case"]["n_ph_max"]), str(row["case"]["regime"])),
    )
    summary = payload["summary"]
    identity = payload["method_identity"]
    spectral_plot = output_dir / "static_spectral_errors.png"
    hard_plot = output_dir / "static_hard_case_spectra.png"
    generality_band_plot = output_dir / "generality_six_root_bands.png"
    for plot in (spectral_plot, hard_plot, generality_band_plot):
        if not plot.is_file():
            raise ReportBuildError(f"required existing report figure is missing: {plot}")

    krylov_reached_1e4 = sum(
        _krylov_record(
            krylov_hh[(str(row["case"]["regime"]), int(row["case"]["n_ph_max"]))],
            "1e-4",
        )[0]
        == "REACHED"
        for row in rows
    )
    krylov_reached_1e6 = sum(
        _krylov_record(
            krylov_hh[(str(row["case"]["regime"]), int(row["case"]["n_ph_max"]))],
            "1e-6",
        )[0]
        == "REACHED"
        for row in rows
    )
    krylov_nph1_reached_1e6 = sum(
        _krylov_record(krylov_hh[(str(row["case"]["regime"]), 1)], "1e-6")[0]
        == "REACHED"
        for row in rows
        if int(row["case"]["n_ph_max"]) == 1
    )
    krylov_nph3_reached_1e6 = sum(
        _krylov_record(krylov_hh[(str(row["case"]["regime"]), 3)], "1e-6")[0]
        == "REACHED"
        for row in rows
        if int(row["case"]["n_ph_max"]) == 3
    )
    block_reached_1e6 = sum(
        _matrix_outcome(block_krylov, str(case_id))[0] == "REACHED"
        for case_id in block_krylov["cases"]
    )
    selected_k_counts = {
        target: sum(
            float(row["crossings"][target]["selected_n2q"])
            <= float(row["crossings"][target]["direct_n2q"])
            for row in rows
        )
        for target in ("1e-4", "1e-6")
    }
    selected_qiskit_counts = {
        target: sum(
            float(row["crossings"][target]["selected_qiskit_compiled"]["n2q"])
            <= float(row["crossings"][target]["direct_qiskit_compiled"]["n2q"])
            for row in rows
        )
        for target in ("1e-4", "1e-6")
    }
    selected_query_counts = {
        target: sum(
            int(row["crossings"][target]["selected_estimator_queries"])
            <= int(row["crossings"][target]["direct_estimator_queries"])
            for row in rows
        )
        for target in ("1e-4", "1e-6")
    }
    generality_cases = _generality_cases(generality)
    selected_cheaper_qiskit = sum(
        float(_generality_crossing(case["routes"]["selected_qse"])["qiskit_compiled"]["n2q"])
        < float(_generality_crossing(case["routes"]["direct_davidson"])["qiskit_compiled"]["n2q"])
        for case in generality_cases
    )
    selected_fewer_queries = sum(
        int(_generality_crossing(case["routes"]["selected_qse"])["estimator"]["queries"])
        < int(_generality_crossing(case["routes"]["direct_davidson"])["estimator"]["queries"])
        for case in generality_cases
    )

    return rf"""\documentclass[
  reprint,
  amsmath,
  amssymb,
  aps,
  pra,
  nofootinbib,
  floatfix
]{{revtex4-2}}

\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\hypersetup{{hidelinks}}
\raggedbottom
\newcommand{{\status}}[1]{{\textcolor{{blue!55!black}}{{\texttt{{#1}}}}}}
\newcommand{{\mathcalR}}{{\mathcal{{R}}}}
\newcommand{{\score}}{{\mathcal{{S}}}}
\newcommand{{\inlinetablecaption}}[2]{{%
  \refstepcounter{{table}}\label{{#1}}%
  \noindent\textsc{{Table~\thetable.}} #2\par\vspace{{0.6ex}}%
}}
\newcommand{{\inlinefigurecaption}}[2]{{%
  \refstepcounter{{figure}}\label{{#1}}%
  \noindent\textsc{{Figure~\thefigure.}} #2\par\vspace{{0.6ex}}%
}}
\pdfinfoomitdate=1
\pdfsuppressptexinfo=15
\ifdefined\pdftrailerid\pdftrailerid{{}}\fi

\begin{{document}}

\title{{Paper III static-QSE numerical source anchor: methods, crossings, resources, mechanisms, and audits}}
\author{{Paper-III reporting pipeline}}
\affiliation{{Evidence artifact for the active PRA/REV\TeX\ variant}}
\date{{Evidence snapshot 2026-08-29}}

\begin{{abstract}}
This document is the numerical source anchor for the static Paper III. It
collects locked Hubbard--Holstein, Peierls--Hubbard, and ten-qubit $N_b=4$
SIAM evidence; reconstructs every publication crossing as the earliest
successful prefix in declared route order; and charges graph-span $K$, Qiskit
circuit-structure annotations, logical estimator queries, and achieved error at
that same prefix. Cost-weighted operator-selected QSE (OS-QSE) reaches the
six-root $10^{{-6}}$ target in all 16 cases without Davidson continuation. Its
summed Qiskit two-qubit annotation is below that of independently initialized
direct quantum Davidson in all 16 cells. Complete single-seed and six-probe
block real-time-Krylov routes reach 10/16 and {block_reached_1e6}/16,
respectively; their complementary failures diagnose initialization-dependent
spectral support and are never combined into a post-hoc oracle route. A
controlled pool ablation reaches 10/12 Hubbard--Holstein cells with the base
physical-macro pool and 12/12 with encoded-boson completion. Static transition
spectra provide an independent state-quality audit. The reported costs describe
surviving support or route-local structural work, not finite-shot or complete
end-to-end hardware cost.
\end{{abstract}}

\maketitle

\section{{Evidence manifest}}

\begin{{table}}[h]
\caption{{Run-defining report manifest.}}
\centering
\small
\begin{{tabular}}{{@{{}}lp{{0.63\columnwidth}}@{{}}}}
\toprule
Field & declared value \\
\midrule
Models & two-site Hubbard--Holstein; Peierls--Hubbard dimer; SIAM $N_b=4$ \\
Cases & 12 Hubbard--Holstein cells plus three Peierls regimes and one SIAM case \\
Target window & $\mathcalR=\{{1,\ldots,6\}}$ excited roots \\
Seed & exact same-sector ground state (benchmark isolation) \\
OS-QSE selector & \nolinkurl{{{identity['selector_policy_id']}}} \\
Operator pool & \nolinkurl{{{identity['selected_pool_policy_id']}}} \\
Spectral kernel & Lorentzian, $\eta={float(payload['spectral_policy']['eta']):.2f}$ \\
Dynamics & excluded \\
Input SHA-256 & \texttt{{{_sha256(input_path)[:20]}}}$\ldots$ \\
RTK matrix SHA-256 & \texttt{{{_sha256(krylov_path)[:20]}}}$\ldots$ \\
RTK receipt & \texttt{{{_sha256(krylov_receipt_path)[:20]}}}$\ldots$ \\
Block RTK matrix SHA-256 & \texttt{{{_sha256(block_krylov_path)[:20]}}}$\ldots$ \\
Block RTK receipt & \texttt{{{_sha256(block_krylov_receipt_path)[:20]}}}$\ldots$ \\
Pool ablation SHA-256 & \texttt{{{_sha256(pool_ablation_path)[:20]}}}$\ldots$ \\
Pool status & \texttt{{{_sha256(pool_ablation_status_path)[:20]}}}$\ldots$ \\
Generality SHA-256 & \texttt{{{_sha256(generality_path)[:20]}}}$\ldots$ \\
Generality receipt & \texttt{{{_sha256(generality_receipt_path)[:20]}}}$\ldots$ \\
\bottomrule
\end{{tabular}}
\end{{table}}

The single- and block-seeded RTK artifacts were originally sealed with a
candidate evidence label. Their inclusion here is an explicit author-directed
promotion for comparator reporting; their JSONs and receipts remain immutable.
The selected-QSE and both RTK inputs have identical exact six-root references
in all 16 cells (tolerance $10^{{-12}}$).

{_method_key()}

{_benchmark_inventory_table(rows, generality)}

{_method_definition()}

\section{{Six-root energy accuracy and graph-span cost}}

Throughout this report, a first crossing is route-local: the recorded prefixes
of each route are scanned in increasing admitted dimension, and the earliest
prefix satisfying the stated six-root tolerance supplies both the reported
accuracy and its cost. It never means the first route or first table column.
With $n$ denoting that recorded order,
\[
 n^\star(\epsilon_E)=\min\left\{{n:\max_{{r\in\mathcal R}}
 |\Delta\varepsilon_r(n)|\leq\epsilon_E\right\}},\qquad
 K_x^\star(\epsilon_E)=K_x\!\left(n^\star(\epsilon_E)\right).
\]
Every resource coordinate $x$ and the achieved error are taken from this same
prefix. These crossings are retrospective exact-reference diagnostics; the
online route terminates using the residual/window rule above.

{_crossing_reconciliation_table(crossing_receipts)}

{_accuracy_table(rows, krylov_hh, '1e-4')}

{_accuracy_table(rows, krylov_hh, '1e-6')}

{_hh_achieved_error_table(rows, krylov_hh, '1e-6')}

OS-QSE is no more expensive than direct quantum Davidson in
{selected_k_counts['1e-4']}/12 cells at
$10^{{-4}}$ and
{selected_k_counts['1e-6']}/12 cells at
$10^{{-6}}$ under $K$. Seeded real-time Krylov reaches those targets in
{krylov_reached_1e4}/12 and {krylov_reached_1e6}/12 cells, respectively. Its
one-to-one $10^{{-6}}$ successes split as {krylov_nph1_reached_1e6}/6 for
$n_{{\rm ph}}=1$ and {krylov_nph3_reached_1e6}/6 for $n_{{\rm ph}}=3$.
The prior independent-nearest audit is superseded because it could reuse one
Ritz root for multiple degenerate targets.

\section{{Qiskit compilation and estimator work}}

{_compiled_resource_table(rows, krylov_hh, '1e-4')}

{_compiled_resource_table(rows, krylov_hh, '1e-6')}

The Qiskit cells are sums of independently transpiled direction/action costs,
not compiled full measurement circuits. Logical estimator queries are not
shots. OS-QSE and fixed LR-QSE use a qubit-wise-commuting basis-cover
convention, whereas direct quantum Davidson uses an ungrouped synthesized-basis
pencil convention. The counts therefore report each declared route's logical
workload; they are not interchangeable universal shot estimates. At the
current crossings OS-QSE has no larger Qiskit $N_{{2q}}$ in
{selected_qiskit_counts['1e-4']}/12
cells at $10^{{-4}}$ and
{selected_qiskit_counts['1e-6']}/12 at
$10^{{-6}}$. It has no larger declared estimator-query count in 12/12 cells at
both targets ({selected_query_counts['1e-4']}/12 and
{selected_query_counts['1e-6']}/12 by direct reconstruction).

{_krylov_resource_table(rows, krylov_hh)}

\subsection{{Accounting boundary: structure versus cumulative acquisition work}}

The online coordinate $K$ is the sum of graph-span two-qubit costs attached to
the surviving/admitted directions at the reported prefix. Qiskit values are
sums of independently transpiled direction/action annotations after preserving
each nonzero Pauli word and replacing complex coefficients by their
magnitudes. Summed depth is therefore an aggregate annotation, not the depth
of one executable linear-combination or measurement circuit. Logical estimator
queries form a third route-local coordinate: OS-QSE uses a qubit-wise-commuting
record-pencil cover, whereas Davidson and RTK use ungrouped synthesized-basis
accounting. No reported query is a shot allocation.

Candidate screening, rejected proposals, exchange-patch evaluation,
measurement reuse, source-state preparation, and transition-observable
measurement are not folded into the final-support $K$ or Qiskit columns unless
a route definition explicitly states otherwise. Consequently the comparisons
establish compact construction under the declared structural coordinates, not
a total-work or end-to-end hardware-runtime theorem.

{_screening_work_table(rows)}

\clearpage
\section{{Terminal selected manifolds and static spectra}}

{_terminal_table(rows)}

All spectra in the current evidence are terminal-manifold audits, not
first-crossing spectra. For Hubbard--Holstein, the 801-point uniform grid is
$[0,\max(1,\omega_6^\star+6\eta)]$ with $\eta=0.05$ and
\[
 \delta_{{L^1}}(A)=
 \frac{{\int |A(\omega)-A^\star(\omega)|\,d\omega}}
 {{\max\!\left(\int |A^\star(\omega)|\,d\omega,10^{{-15}}\right)}},
\]
evaluated by the trapezoidal rule; transition strengths below $10^{{-8}}$ are
treated as dark. The Peierls/SIAM and RTK postprocessors also use 801 points,
$\eta=0.05$, and trapezoidal integration, but choose the upper limit as the
largest exact or approximate pole plus $0.5$ and report normalized error as
undefined when exact integrated weight is at most $10^{{-12}}$. These two
frequency-window/zero-weight policies are stated separately and their errors
are not pooled into one cross-family statistic.

\begin{{widetext}}
\begin{{center}}
\includegraphics[width=0.94\textwidth]{{static_spectral_errors.png}}
\end{{center}}
\inlinefigurecaption{{fig:spectral_errors}}{{Normalized broadened-spectrum errors for site-0 density and phonon
displacement. OS-QSE has median normalized spectral $L^1$ error
{_sci(summary['median_selected_spectral_l1'])} and maximum
{_sci(summary['max_selected_spectral_l1'])}; complete fixed LR-QSE has median
{_sci(summary['median_fixed_spectral_l1'])} and maximum
{_sci(summary['max_fixed_spectral_l1'])}.}}
\end{{widetext}}

\clearpage
\begin{{widetext}}
\begin{{center}}
\includegraphics[width=0.94\textwidth]{{static_hard_case_spectra.png}}
\end{{center}}
\inlinefigurecaption{{fig:stress_spectra}}{{Exact, OS-QSE, and complete fixed LR-QSE spectra for the two
completion-pool stress cells. The exact reference is drawn last as a black
dotted curve with markers so it remains visible when OS-QSE overlaps it.
Davidson is not invoked inside OS-QSE.}}
\end{{widetext}}

\clearpage
\section{{Fresh Peierls--Hubbard and SIAM generality evidence}}

{_generality_resource_table(generality, krylov)}

{_generality_certification_table(generality)}

{_generality_krylov_outcome_table(generality, krylov)}

OS-QSE reaches the common $10^{{-6}}$ target with fewer Qiskit-compiled
two-qubit gates in {selected_cheaper_qiskit}/4 cases and fewer logical
estimator queries in {selected_fewer_queries}/4 cases. The fixed Peierls class
retains only rank five and therefore cannot represent a six-root window. The
fixed SIAM class retains rank 16 but its maximum six-root error is
$3.64\times10^{{-2}}$. Its lower cost is consequently not a matched-task win.
All four selector replays reproduce identical selected supports, stop reasons,
and final selector residuals.

Single-seed RTK reaches all four fresh Peierls/SIAM windows. Together with its
6/12 Hubbard--Holstein success count, the fixed-policy comparator reaches
10/16 declared cases. The six failures remain charged terminal-cap outcomes;
they are not dropped from cost summaries.

The achieved-error column is necessary for mechanism attribution. In the
three Peierls cases Davidson's first admissible crossing overshoots to
numerical precision, whereas OS-QSE stops between $1.32\times10^{{-9}}$ and
$5.99\times10^{{-8}}$. Those rows establish threshold-matched cost but do not
isolate how much of Davidson's extra cost comes from its coarse growth step and
accuracy overshoot. The SIAM crossing is cleaner: OS-QSE has error
$5.07\times10^{{-7}}$ versus Davidson's $9.60\times10^{{-7}}$ while using
$622$ versus $2{{,}}142$ compiled two-qubit gates and $4{{,}}666$ versus $46{{,}}008$
queries. In that case the cost advantage cannot be explained by lower achieved
accuracy and is consistent with the intended cost-weighted selection effect.

\begin{{widetext}}
\begin{{center}}
\includegraphics[width=0.94\textwidth]{{generality_six_root_bands.png}}
\end{{center}}
\inlinefigurecaption{{fig:generality_bands}}{{Separated six-root energy bands.
OS-QSE and Davidson levels are taken from each route's own first
$10^{{-6}}$ crossing; single-seed RTK is shown at its one-to-one crossing and
fixed-class levels and costs are terminal. All energies
are shifted by the lowest exact target root in that case. The exact reference
has its own horizontal group and therefore cannot be hidden by an overlapping
approximation. Costs printed below each group are charged at the same prefix
whose levels are drawn.}}
\end{{widetext}}

{_generality_spectral_table(generality, krylov)}

\clearpage
\section{{Krylov initialization and target-window coverage}}

{_coverage_map_table(krylov, block_krylov)}

{_block_krylov_resource_table(block_krylov)}

Single-seed RTK reaches 10/16 cases and the fixed six-probe block route reaches
{block_reached_1e6}/16. Their success sets are complementary: their union is a
post-run diagnostic, not one algorithm, and no case-wise choice is used in any
headline comparison. The block route charges all six source preparations at
each time slice and all Hamiltonian actions. Its larger initialization improves
coverage in some cells while losing target directions in others.

{_block_source_failure_table(block_krylov)}

The five block failures each contain a target direction with post-run source
weight below $10^{{-20}}$. This explains why increasing the time-series length
under the same fixed block cannot repair those cells. It does not imply a
universal failure of block Krylov: a different, declared non-exact block could
change the represented spectral support and must be benchmarked as a separate
route.

\section{{Pool expressivity and selector economy}}

{_pool_ablation_table(pool_ablation)}

With the current selector and all other route settings held fixed, the base
physical-macro pool reaches the $10^{{-4}}$ six-root target in 10/12
Hubbard--Holstein cells, whereas the encoded-boson completion pool reaches
12/12. Completion-family records appear in every completion-pool terminal
support. This establishes that the added family is exercised and repairs two
coverage failures. It does not, by itself, decompose the final savings into a
pure pool effect and a pure selector effect; a complete same-pool control would
be needed for that stronger attribution.

\clearpage
\section{{Visible completion plan}}

{_planned_placeholders()}

The implemented single- and block-seeded RTK matrices are complete comparator
evidence: each includes all 16 cases, compiled costs, query counts, retained
ranks, gaps, residuals, overlaps, and static spectral audits. Their failures
remain charged results, not missing cells. Hamiltonian-power QKSD remains
unimplemented. Current-policy cost-discount-off, exchange-off, accepted-patch,
and cumulative-work evidence remains explicitly unresolved. A later rebuild
must replace those statuses in place rather than silently changing method
identity.

\section{{Interpretation boundaries}}

\begin{{itemize}}
\item The sealed numerical payload contains 12 Hubbard--Holstein cells, three
Peierls--Hubbard regimes, and one non-smoke ten-qubit SIAM case. It is not a
claim about untested model sizes or couplings.
\item OS-QSE is the proposed algorithm. Direct quantum Davidson and
single- and block-seeded RTK are independent comparators; complete fixed-class
QSE is the conventional static baseline. The two RTK routes are never merged.
Record-only growth is an ablation.
\item Supported-overlap truncation, retained-rank checks, deterministic
selection, and atomic exchange certification remain part of OS-QSE.
\item The report contains no finite-shot sampling, uncertainty bars, hardware
runtime, or time-dynamics claim. RTK source-state preparation cost is excluded
under the inherited single-seed comparator convention; every Krylov-basis
preparation is charged. Block RTK separately charges all computational-basis
probe preparations and Hamiltonian actions but excludes the exact ground-state
preparation common to the benchmark.
\item Qiskit costs preserve each nonzero Pauli word and replace complex
coefficients by their magnitudes before transpilation. They are deterministic
circuit-structure annotations, not an LCU state-preparation claim.
\item Exact same-cutoff ground states are benchmark inputs. The evidence does
not establish approximate-seed robustness, bosonic-cutoff convergence,
finite-shot performance, universal superiority over Krylov methods, or
systematic SIAM generality beyond the one $N_b=4$ transfer case.
\end{{itemize}}

\end{{document}}
"""


def _compile_latex(tex_path: Path, pdf_path: Path) -> dict[str, Any]:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise ReportBuildError("latexmk is required for Paper-III report PDFs")
    with tempfile.TemporaryDirectory(prefix="paper_iii_static_qse_latex_") as tmp:
        build_dir = Path(tmp)
        command = (
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={build_dir}",
            str(tex_path.resolve()),
        )
        env = dict(os.environ)
        env.setdefault("SOURCE_DATE_EPOCH", "1788008400")
        completed = subprocess.run(
            command,
            cwd=tex_path.parent,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path = tex_path.with_suffix(".latexmk.log")
        log_path.write_text(completed.stdout, encoding="utf-8")
        built_pdf = build_dir / f"{tex_path.stem}.pdf"
        if completed.returncode != 0 or not built_pdf.is_file():
            tail = "\n".join(completed.stdout.splitlines()[-80:])
            raise ReportBuildError(f"LaTeX compilation failed:\n{tail}")
        shutil.copy2(built_pdf, pdf_path)
    return {
        "engine": "latexmk/pdflatex",
        "command": [str(part) for part in command],
        "log_path": str(log_path),
        "status": "compiled",
    }


def _git_revision() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def build_report(
    *,
    input_path: Path,
    krylov_path: Path,
    block_krylov_path: Path,
    pool_ablation_path: Path,
    generality_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    payload = _load_json(input_path)
    krylov = _load_json(krylov_path)
    block_krylov = _load_json(block_krylov_path)
    pool_ablation = _load_json(pool_ablation_path)
    generality = _load_json(generality_path)
    _validate_resource_payload(payload)
    crossing_receipts = _derive_hh_first_crossings(payload)
    krylov_receipt_path = _validate_krylov_matrix(
        krylov, input_path=krylov_path
    )
    block_krylov_receipt_path = _validate_block_krylov_matrix(
        block_krylov, input_path=block_krylov_path
    )
    pool_ablation_status_path = _validate_pool_ablation(
        pool_ablation, input_path=pool_ablation_path
    )
    generality_receipt_path = _validate_generality_payload(
        generality, input_path=generality_path
    )
    krylov_hh = _krylov_hh_index(krylov)
    receipts = _verify_krylov_reference_identity(
        payload["rows"], generality, krylov, block_krylov
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _render_figures(payload, output_dir)
    _render_generality_band_figure(generality, krylov, output_dir)
    tex_path = output_dir / "paper_iii_static_qse_source_anchor.tex"
    pdf_path = output_dir / "paper_iii_static_qse_source_anchor.pdf"
    tex = _build_tex(
        payload,
        krylov_hh,
        krylov,
        block_krylov,
        pool_ablation,
        generality,
        crossing_receipts,
        input_path=input_path,
        krylov_path=krylov_path,
        krylov_receipt_path=krylov_receipt_path,
        block_krylov_path=block_krylov_path,
        block_krylov_receipt_path=block_krylov_receipt_path,
        pool_ablation_path=pool_ablation_path,
        pool_ablation_status_path=pool_ablation_status_path,
        generality_path=generality_path,
        generality_receipt_path=generality_receipt_path,
        output_dir=output_dir,
    )
    tex_path.write_text(tex, encoding="utf-8")
    compile_receipt = _compile_latex(tex_path, pdf_path)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "report_role": "paper_iii_static_qse_numerical_source_anchor",
        "active_variant": "revtex4-2_aps_pra_reprint",
        "renderer": {
            "type": "latex",
            "direct_pdf_renderer_allowed": False,
            **compile_receipt,
        },
        "method_set": [
            METHOD_SELECTED,
            METHOD_DAVIDSON,
            METHOD_KRYLOV,
            METHOD_BLOCK_KRYLOV,
            METHOD_FIXED,
        ],
        "source_revision": _git_revision(),
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "inputs": [
            {"path": str(input_path), "sha256": _sha256(input_path)},
            {"path": str(krylov_path), "sha256": _sha256(krylov_path)},
            {
                "path": str(krylov_receipt_path),
                "sha256": _sha256(krylov_receipt_path),
            },
            {
                "path": str(block_krylov_path),
                "sha256": _sha256(block_krylov_path),
            },
            {
                "path": str(block_krylov_receipt_path),
                "sha256": _sha256(block_krylov_receipt_path),
            },
            {
                "path": str(pool_ablation_path),
                "sha256": _sha256(pool_ablation_path),
            },
            {
                "path": str(pool_ablation_status_path),
                "sha256": _sha256(pool_ablation_status_path),
            },
            {"path": str(generality_path), "sha256": _sha256(generality_path)},
            {
                "path": str(generality_receipt_path),
                "sha256": _sha256(generality_receipt_path),
            },
        ],
        "reference_identity_receipts": receipts,
        "publication_crossing_policy": {
            "definition": "earliest_successful_prefix_in_declared_growth_order",
            "same_prefix_for_all_resource_coordinates": True,
            "sealed_hh_summary_policy": "minimum_graph_span_cost_among_successful_rungs",
            "reconciliation_receipts": crossing_receipts,
        },
        "evidence_adoption": {
            "single_seed_rtk": "included_as_comparator_by_author_direction",
            "block_seeded_rtk": "included_as_comparator_by_author_direction",
            "sealed_input_files_mutated": False,
        },
        "outputs": [
            {"path": str(tex_path), "sha256": _sha256(tex_path)},
            {"path": str(pdf_path), "sha256": _sha256(pdf_path)},
            {
                "path": str(output_dir / "static_spectral_errors.png"),
                "sha256": _sha256(output_dir / "static_spectral_errors.png"),
            },
            {
                "path": str(output_dir / "static_hard_case_spectra.png"),
                "sha256": _sha256(output_dir / "static_hard_case_spectra.png"),
            },
            {
                "path": str(output_dir / "generality_six_root_bands.png"),
                "sha256": _sha256(output_dir / "generality_six_root_bands.png"),
            },
        ],
        "visible_placeholder_statuses": [
            "PENDING",
            "IMPLEMENTATION_PENDING",
            "DEFINITION_REQUIRED",
            "SCOPE_DECISION_REQUIRED",
            "NOT_PLANNED",
        ],
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--krylov-matrix", type=Path, default=DEFAULT_KRYLOV_MATRIX)
    parser.add_argument(
        "--block-krylov-matrix", type=Path, default=DEFAULT_BLOCK_KRYLOV_MATRIX
    )
    parser.add_argument(
        "--pool-ablation-input", type=Path, default=DEFAULT_POOL_ABLATION_INPUT
    )
    parser.add_argument("--generality-input", type=Path, default=DEFAULT_GENERALITY_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_report(
        input_path=args.input.resolve(),
        krylov_path=args.krylov_matrix.resolve(),
        block_krylov_path=args.block_krylov_matrix.resolve(),
        pool_ablation_path=args.pool_ablation_input.resolve(),
        generality_path=args.generality_input.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    for output in manifest["outputs"]:
        print(f"wrote {output['path']} ({output['sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

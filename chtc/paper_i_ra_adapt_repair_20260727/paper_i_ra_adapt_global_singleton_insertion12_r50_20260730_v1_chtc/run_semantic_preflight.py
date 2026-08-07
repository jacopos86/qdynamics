#!/usr/bin/env python3
"""Run bounded semantic checks for the global-singleton insertion package."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from package_contract import (  # noqa: E402
    ACTIVE_GRADIENT_POLICY,
    APPEND_ROUTE_ID,
    CALIBRATION_RECEIPT_NAME,
    CALIBRATION_SCHEMA,
    GLOBAL_POOL_BY_NPH,
    ORDERED_POOL_SHA256_BY_REGIME,
    PACKAGE_ID,
    PLATEAU_ROUTE_ID,
    RESOURCE_WEIGHTING_SCOPE,
    SMOKE_EXECUTION_IDS,
    SMOKE_RECEIPT_NAME,
    SMOKE_ROUNDS,
    SMOKE_SCHEMA,
    PackageContractError,
    canonical_json_bytes,
    canonical_sha256,
    digested,
    direct_execution_rows,
    repo_root_from_script,
    sha256_file,
    validate_calibration_receipt,
    validate_materialization_authority,
    validate_smoke_receipt,
)


def _problem_from_protocol(protocol: Any) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import (
        resolve_problem_context,
    )
    from pipelines.static_adapt.sr_snake.contracts import (
        ResolvedProblemReceipt,
    )

    receipt = protocol.problem
    problem = resolve_problem_context(
        ProblemRequest(
            problem_key=str(receipt.problem_key),
            num_sites=int(receipt.num_sites),
            t=float(receipt.t),
            u=float(receipt.u),
            dv=float(receipt.dv),
            omega0=float(receipt.omega0),
            g_ep=float(receipt.g_ep),
            n_ph_max=int(receipt.n_ph_max),
            boson_encoding=str(receipt.boson_encoding),
            ordering=str(receipt.ordering),
            boundary=str(receipt.boundary),
            include_zero_point=bool(receipt.include_zero_point),
            v_nn=float(receipt.v_nn),
            t_prime=float(receipt.t_prime),
            n_fermions=(
                None
                if receipt.n_fermions is None
                else int(receipt.n_fermions)
            ),
        )
    )
    if (
        ResolvedProblemReceipt.from_problem(problem).to_dict()
        != receipt.to_dict()
    ):
        raise PackageContractError(
            "Preflight problem reconstruction drifted."
        )
    return problem


def _receipt_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    plans = receipt.get("candidate_position_plans")
    retained = receipt.get("retained_representatives")
    if not isinstance(plans, list) or not isinstance(retained, list):
        raise PackageContractError(
            "Insertion receipt has no candidate-position population."
        )
    keys = (
        "schema",
        "policy",
        "domain_state",
        "domain_open",
        "effective_insertion_mode",
        "trigger_energy_before",
        "trigger_energy_after",
        "trigger_energy_decrease",
        "trigger_source",
        "energy_decrease_threshold",
        "threshold_comparison",
        "calibration_status",
        "patience",
        "hysteresis_active",
        "exact_reference_used",
        "append_position",
        "requested_positions",
        "candidate_count",
        "requested_position_count",
        "retained_representative_count",
        "collapsed_position_count",
    )
    return {
        **{
            key: receipt[key]
            for key in keys
            if key in receipt
        },
        "candidate_position_plans_sha256": canonical_sha256(plans),
        "retained_representatives_sha256": canonical_sha256(retained),
        "full_receipt_sha256": canonical_sha256(receipt),
    }


def _run_case(
    *,
    row: Mapping[str, Any],
    protocol_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    from pipelines.static_adapt.ra_adapt import (
        RAAdaptOperationalControls,
        run_ra_adapt,
    )
    from pipelines.static_adapt.ra_adapt.bundles import (
        load_validated_bundle_protocol,
    )
    from pipelines.static_adapt.ra_adapt.insertion_geometry import (
        validate_commutation_reduced_insertion_receipt,
    )

    protocol = load_validated_bundle_protocol(protocol_path)
    problem = _problem_from_protocol(protocol)
    insertion = protocol.request.method.insertion
    if (
        getattr(insertion, "kind", None) != row["insertion_policy"]
        or protocol.active_gradient_policy
        != ACTIVE_GRADIENT_POLICY
        or protocol.resource_weighting_scope
        != RESOURCE_WEIGHTING_SCOPE
        or int(protocol.executable_pool.count) != 6508
    ):
        raise PackageContractError(
            f"Preflight protocol drifted: {row['execution_id']}."
        )
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        observed = run_ra_adapt(
            problem,
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=SMOKE_ROUNDS
            ),
        )
    elapsed = time.perf_counter() - started
    payload = observed.to_dict()
    scientific = payload.get("scientific_receipts")
    accepted = (
        scientific.get("accepted_round_receipts")
        if isinstance(scientific, Mapping)
        else None
    )
    if not isinstance(accepted, list) or len(accepted) != SMOKE_ROUNDS:
        raise PackageContractError(
            f"Preflight did not complete two rounds: "
            f"{row['execution_id']}."
        )
    key = (
        "insertion_commutation_reduced"
        if row["route_id"] == APPEND_ROUTE_ID
        else "insertion_commutation_plateau"
    )
    reductions: list[dict[str, Any]] = []
    for round_index, accepted_row in enumerate(accepted, start=1):
        reduction = (
            accepted_row.get(key)
            if isinstance(accepted_row, Mapping)
            else None
        )
        if not isinstance(reduction, Mapping):
            raise PackageContractError(
                f"Preflight round {round_index} omitted {key}."
            )
        validate_commutation_reduced_insertion_receipt(
            reduction,
            expected_policy=str(reduction["policy"]),
            expected_requested_positions=tuple(
                int(value)
                for value in reduction["requested_positions"]
            ),
        )
        reductions.append(_receipt_projection(reduction))
    first_lineage = accepted[0].get("accepted_candidate_lineage")
    if (
        not isinstance(first_lineage, list)
        or len(first_lineage) != 1
        or not isinstance(first_lineage[0], Mapping)
    ):
        raise PackageContractError(
            "Preflight first round has no singleton lineage."
        )
    selected = {
        "candidate_label": str(
            first_lineage[0]["candidate_label"]
        ),
        "generator_identity": str(
            first_lineage[0]["generator_identity"]
        ),
    }
    return (
        {
            "execution_id": row["execution_id"],
            "route_id": row["route_id"],
            "regime_id": row["regime_id"],
            "nph": row["nph"],
            "candidate_adapter_id": row["candidate_adapter_id"],
            "active_gradient_policy": protocol.active_gradient_policy,
            "resource_weighting_scope": (
                protocol.resource_weighting_scope
            ),
            "global_pool_count": int(protocol.executable_pool.count),
            "global_pool_ordered_labels_sha256": (
                protocol.executable_pool.ordered_labels_sha256
            ),
            "global_pool_ordered_pool_sha256": (
                protocol.executable_pool.ordered_pool_sha256
            ),
            "protocol_path": protocol_path.relative_to(
                repo_root
            ).as_posix(),
            "protocol_file_sha256": sha256_file(protocol_path),
            "protocol_canonical_sha256": protocol.sha256,
            "typed_insertion_policy": getattr(
                insertion, "kind", None
            ),
            "controller_round_count": len(accepted),
            "elapsed_seconds": float(elapsed),
            "result_sha256": canonical_sha256(payload),
            "first_accepted_candidate": selected,
            "accepted_round_insertion_receipts": reductions,
            "scientific_result": False,
            "execution_evidence": False,
            "paper_evidence_allowed": False,
            "status": "passed",
        },
        selected,
    )


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _calibration_worker(
    *,
    protocol_path: Path,
    candidate_label: str,
    generator_identity: str,
) -> dict[str, Any]:
    from pipelines.static_adapt.adapt_pipeline import (
        _candidate_insertion_position_plans,
        _insertion_commutation_plateau_domain_receipt,
        _insertion_commutation_plateau_round_policy,
    )
    from pipelines.static_adapt.ra_adapt.bundles import (
        load_validated_bundle_protocol,
    )
    from pipelines.static_adapt.ra_adapt.insertion_geometry import (
        validate_commutation_reduced_insertion_receipt,
    )

    baseline_rss = _rss_bytes()
    started = time.perf_counter()
    protocol = load_validated_bundle_protocol(protocol_path)
    problem = _problem_from_protocol(protocol)
    inventory = protocol.request.adapter.executable_pool(problem)
    if (
        int(inventory.receipt.count) != 6508
        or inventory.receipt.ordered_labels_sha256
        != GLOBAL_POOL_BY_NPH[7]["ordered_labels_sha256"]
    ):
        raise PackageContractError(
            "Calibration did not reconstruct the exact nph7 pool."
        )
    matches = [
        candidate
        for candidate in inventory.candidates
        if str(candidate.label) == candidate_label
        and str(candidate.generator_identity) == generator_identity
    ]
    if len(matches) != 1:
        raise PackageContractError(
            "Calibration could not authenticate the accepted singleton."
        )
    pool = [candidate.term for candidate in inventory.candidates]
    selected_ops = [matches[0].term]
    synthetic_history = [
        {
            "schema": "plateau_calibration_synthetic_prior_v1",
            "energy_before_opt": 0.0,
            "energy_after_opt": 0.0,
            "scientific_result": False,
            "execution_evidence": False,
        }
    ]
    round_policy = _insertion_commutation_plateau_round_policy(
        history=synthetic_history
    )
    if (
        round_policy.get("domain_open") is not True
        or round_policy.get("trigger_energy_decrease") != 0.0
        or round_policy.get("energy_decrease_threshold") != 1.0e-8
    ):
        raise PackageContractError(
            "Synthetic calibration did not open the production plateau."
        )
    requested_positions = [0, 1]
    candidate_indices = list(range(len(pool)))
    plans = _candidate_insertion_position_plans(
        pool=pool,
        candidate_indices=candidate_indices,
        selected_ops=selected_ops,
        positions=requested_positions,
    )
    domain = _insertion_commutation_plateau_domain_receipt(
        round_policy=round_policy,
        candidate_position_plans=plans,
        pool=pool,
    )
    validate_commutation_reduced_insertion_receipt(
        domain,
        expected_policy="insertion_commutation_plateau_v1",
        expected_requested_positions=(0, 1),
    )
    projection = _receipt_projection(domain)
    serialized_bytes = len(canonical_json_bytes(domain))
    elapsed = time.perf_counter() - started
    peak_rss = _rss_bytes()
    memory_headroom_mb = int(
        math.ceil((peak_rss / (1024**2)) * 2.0)
    )
    disk_headroom_mb = int(
        math.ceil((serialized_bytes / (1024**2)) * 4.0)
    )
    return {
        "calibration_scope": (
            "production_plateau_decision_and_exact_full_domain_"
            "commutation_reducer_at_depth1_v1"
        ),
        "protocol_path": protocol_path.relative_to(
            REPO_ROOT
        ).as_posix(),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_canonical_sha256": protocol.sha256,
        "regime_id": "weak_strong",
        "nph": 7,
        "candidate_adapter_id": str(protocol.adapter_id),
        "candidate_count": len(pool),
        "global_pool_ordered_labels_sha256": (
            inventory.receipt.ordered_labels_sha256
        ),
        "global_pool_ordered_pool_sha256": (
            inventory.receipt.ordered_pool_sha256
        ),
        "selected_prefix_source": (
            "first_accepted_candidate_from_exact_two_round_"
            "plateau_smoke_v1"
        ),
        "selected_prefix_candidate_label": candidate_label,
        "selected_prefix_generator_identity": generator_identity,
        "synthetic_trigger_only": True,
        "synthetic_trigger_energy_decrease": 0.0,
        "requested_positions": requested_positions,
        "precollapse_candidate_position_pair_count": (
            len(pool) * len(requested_positions)
        ),
        "collapsed_representative_count": int(
            domain["collapsed_position_count"]
        ),
        "retained_representative_count": int(
            domain["retained_representative_count"]
        ),
        "open_domain_receipt": projection,
        "resource_observation": {
            "baseline_peak_rss_bytes": baseline_rss,
            "peak_rss_bytes": peak_rss,
            "incremental_peak_upper_bound_bytes": max(
                0, peak_rss - baseline_rss
            ),
            "elapsed_seconds": float(elapsed),
            "serialized_receipt_bytes": serialized_bytes,
            "bounded_calibration_derived_memory_with_2x_headroom_mb": (
                memory_headroom_mb
            ),
            "bounded_calibration_derived_disk_with_4x_"
            "receipt_headroom_mb": disk_headroom_mb,
            "derivation_scope": (
                "two_positions_depth1_lower_bound_not_depth50_v1"
            ),
        },
        "declared_nph7_package_resources": {
            "request_cpus": 4,
            "request_memory_mb": 90_112,
            "request_disk_mb": 98_304,
            "max_runtime_seconds": 259_200,
            "status": "provisional_not_demonstrated",
        },
        "package_resources_demonstrated": False,
        "package_resource_status": "provisional_not_demonstrated",
        "scientific_result": False,
        "execution_evidence": False,
        "checkpoint_emitted": False,
        "result_promotable": False,
    }


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
    os.link(temporary, path)
    temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-output",
        type=Path,
        default=PACKAGE_DIR / SMOKE_RECEIPT_NAME,
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=PACKAGE_DIR / CALIBRATION_RECEIPT_NAME,
    )
    parser.add_argument(
        "--calibration-worker", action="store_true"
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--candidate-label")
    parser.add_argument("--generator-identity")
    args = parser.parse_args(argv)

    if args.calibration_worker:
        if (
            args.protocol is None
            or args.candidate_label is None
            or args.generator_identity is None
        ):
            parser.error(
                "calibration worker requires protocol and candidate identity"
            )
        payload = _calibration_worker(
            protocol_path=args.protocol.resolve(),
            candidate_label=args.candidate_label,
            generator_identity=args.generator_identity,
        )
        print(canonical_json_bytes(payload).decode("ascii"))
        return 0

    smoke_output = args.smoke_output.resolve()
    calibration_output = args.calibration_output.resolve()
    if (
        smoke_output.exists()
        or smoke_output.is_symlink()
        or calibration_output.exists()
        or calibration_output.is_symlink()
    ):
        raise FileExistsError(
            "Refusing to overwrite semantic preflight receipts."
        )

    repo_root = repo_root_from_script(__file__)
    authority = validate_materialization_authority(repo_root)
    rows_by_id = {
        row["execution_id"]: row for row in direct_execution_rows()
    }
    observations: list[dict[str, Any]] = []
    selected_by_route: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(
        prefix="paper-i-global-singleton-smoke."
    ) as temporary_name:
        original = Path.cwd()
        os.chdir(temporary_name)
        try:
            for execution_id in SMOKE_EXECUTION_IDS:
                row = rows_by_id[execution_id]
                binding = authority["protocol_bindings"][
                    execution_id
                ]
                observation, selected = _run_case(
                    row=row,
                    protocol_path=repo_root / binding["path"],
                    repo_root=repo_root,
                )
                observations.append(observation)
                selected_by_route[str(row["route_id"])] = selected
        finally:
            os.chdir(original)

    smoke = digested(
        {
            "schema": SMOKE_SCHEMA,
            "package_id": PACKAGE_ID,
            "status": "passed",
            "captured_utc": datetime.now(
                timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "smoke_scope": (
                "one_canonical_nph7_regime_by_two_insertion_"
                "policies_bounded_to_two_rounds_v1"
            ),
            "maximum_controller_rounds": SMOKE_ROUNDS,
            "observations": observations,
            "scientific_result": False,
            "execution_evidence": False,
            "paper_evidence_allowed": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    validate_smoke_receipt(smoke)

    plateau_row = rows_by_id[SMOKE_EXECUTION_IDS[1]]
    plateau_binding = authority["protocol_bindings"][
        plateau_row["execution_id"]
    ]
    selected = selected_by_route[PLATEAU_ROUTE_ID]
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATIC_ADAPT_HH_POOL_CACHE": "off",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
    }
    worker = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--calibration-worker",
            "--protocol",
            str(repo_root / plateau_binding["path"]),
            "--candidate-label",
            selected["candidate_label"],
            "--generator-identity",
            selected["generator_identity"],
        ],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    calibration_body = json.loads(worker.stdout)
    if not isinstance(calibration_body, dict):
        raise PackageContractError(
            "Calibration worker did not return a mapping."
        )
    calibration = digested(
        {
            "schema": CALIBRATION_SCHEMA,
            "package_id": PACKAGE_ID,
            "status": "passed",
            "captured_utc": datetime.now(
                timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            **calibration_body,
        }
    )
    validate_calibration_receipt(calibration)
    _exclusive_json(smoke_output, smoke)
    _exclusive_json(calibration_output, calibration)
    print(
        canonical_json_bytes(
            {
                "status": "passed",
                "smoke_output": smoke_output.relative_to(
                    repo_root
                ).as_posix(),
                "smoke_sha256": smoke["sha256"],
                "calibration_output": (
                    calibration_output.relative_to(
                        repo_root
                    ).as_posix()
                ),
                "calibration_sha256": calibration["sha256"],
                "observation_count": len(observations),
                "maximum_controller_rounds": SMOKE_ROUNDS,
                "scientific_result": False,
                "execution_evidence": False,
                "remote_stage": False,
                "condor_submit": False,
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

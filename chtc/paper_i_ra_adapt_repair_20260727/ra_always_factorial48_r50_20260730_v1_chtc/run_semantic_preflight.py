#!/usr/bin/env python3
"""Run eight bounded two-round semantic smokes for the factorial."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping


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
    ALWAYS_INSERTION_KIND,
    PACKAGE_ID,
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
    validate_factorial_authority,
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
            "Smoke problem reconstruction drifted from its protocol."
        )
    return problem


def _mappings(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _mappings(child)


def _policy_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [
        row
        for row in _mappings(payload)
        if {
            "active_gradient_policy",
            "resource_weighting_scope",
            "active_gradient_indices_acquired",
            "active_gradient_charge",
        }.issubset(row)
    ]
    if not candidates:
        raise PackageContractError(
            "Smoke result has no active-gradient policy receipt."
        )
    row = candidates[-1]
    return {
        "active_gradient_policy": row["active_gradient_policy"],
        "resource_weighting_scope": row[
            "resource_weighting_scope"
        ],
        "active_gradient_indices_acquired": list(
            row["active_gradient_indices_acquired"]
        ),
        "active_gradient_charge": int(row["active_gradient_charge"]),
    }


def _cost_observation(
    resource_weighting_scope: str,
) -> dict[str, Any]:
    """Exercise the production Phase-I score branch with nonunit cost.

    ``PaperIRunSummary`` intentionally does not copy every candidate score
    payload.  This bounded probe therefore calls the same production scoring
    function used by the controller, with an explicit nonunit raw burden,
    and binds its scope to the already-loaded source-locked protocol.
    """

    from pipelines.scaffold.hh_continuation_scoring import (
        SimpleScoreConfig,
        phase1_score_payload,
    )
    from pipelines.scaffold.hh_continuation_types import (
        CandidateFeatures,
    )

    feature = CandidateFeatures(
        stage_name="phase1",
        candidate_label="factorial_semantic_probe",
        candidate_family="factorial_semantic_probe",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        g_signed=0.8,
        g_abs=0.8,
        g_lcb=0.8,
        sigma_hat=0.0,
        F_metric=1.0,
        metric_proxy=1.0,
        novelty=1.0,
        curvature_mode="factorial_semantic_probe",
        novelty_mode="factorial_semantic_probe",
        refit_window_indices=[],
        compiled_position_cost_proxy={},
        measurement_cache_stats={},
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        simple_score=None,
        score_version="factorial_semantic_probe_v1",
        c_bar_2q=3.0,
    )
    row = phase1_score_payload(
        feature,
        SimpleScoreConfig(
            lambda_2q=1.0,
            lambda_d=0.0,
            lambda_1q=0.0,
            lambda_theta=0.0,
            lambda_shot=0.0,
            resource_weighting_scope=resource_weighting_scope,
        ),
    )
    return {
        "probe_kind": (
            "production_phase1_score_payload_nonunit_raw_burden_v1"
        ),
        "resource_weighting_scope": row[
            "resource_weighting_scope"
        ],
        "phase1_resource_weighting_active": row[
            "phase1_resource_weighting_active"
        ],
        "phase1_effective_cost_factor": float(
            row.get("phase1_effective_cost_factor", 0.0)
        ),
        "phase1_effective_burden": float(
            row["phase1_effective_burden"]
        ),
        "phase1_raw_burden": float(row["phase1_raw_burden"]),
    }


def _active_gradient_observation(
    active_gradient_policy: str,
) -> dict[str, Any]:
    """Exercise live Phase-III promotion at active depth one."""

    import numpy as np
    from pipelines.scaffold import hh_continuation_scoring as scoring

    acquired = {
        "schema": "phase2_joint_geometry_reuse_v1",
        "append_position": 1,
        "G_AA": [[1.0]],
        "G_AB": [0.25],
        "G_BB": 1.0,
        "H_AA": [[1.0]],
        "H_AB": [0.5],
        "H_BB": 1.0,
        "descent_gradient": 0.5,
    }
    scaffold = SimpleNamespace(
        old_old_geometry_measured=True,
        old_old_metric_measured=True,
        old_old_hessian_measured=True,
        old_old_hessian_status="measured",
        old_old_hessian_fingerprint="factorial_probe_hessian",
        old_old_hessian_provenance={
            "source": "factorial_semantic_probe",
            "measured": True,
        },
        refit_window_indices=(0,),
        state_reconstruction_delta_norm=0.0,
        dpsi_window=(np.asarray([1.0 + 0.0j]),),
        hpsi_state=np.asarray([2.0 + 0.0j]),
        state_fingerprint="factorial_probe_state",
        ordered_scaffold_fingerprint="factorial_probe_scaffold",
        theta_fingerprint="factorial_probe_theta",
    )
    original_vdot = scoring.np.vdot
    original_compiled_fingerprint = (
        scoring._compiled_polynomial_fingerprint
    )
    original_candidate_fingerprint = (
        scoring._candidate_coordinate_fingerprint
    )
    query_count = 0

    def _counted_vdot(left: Any, right: Any) -> complex:
        nonlocal query_count
        query_count += 1
        return complex(original_vdot(left, right))

    try:
        scoring.np.vdot = _counted_vdot
        scoring._compiled_polynomial_fingerprint = (
            lambda _compiled: "factorial_probe_hamiltonian"
        )
        scoring._candidate_coordinate_fingerprint = (
            lambda _term, *, position_id: (
                f"factorial_probe_candidate:{position_id}"
            )
        )
        promoted = (
            scoring._promote_fresh_phase3_joint_geometry_receipt(
                acquired_payload=acquired,
                scaffold_context=scaffold,
                candidate_term=object(),
                position_id=1,
                h_compiled=object(),
                state_consistency_tolerance=1.0e-12,
                active_gradient_policy=active_gradient_policy,
            )
        )
    finally:
        scoring.np.vdot = original_vdot
        scoring._compiled_polynomial_fingerprint = (
            original_compiled_fingerprint
        )
        scoring._candidate_coordinate_fingerprint = (
            original_candidate_fingerprint
        )
    return {
        "probe_kind": (
            "production_phase3_active_gradient_promotion_depth1_v1"
        ),
        "active_depth": 1,
        "active_gradient_policy": active_gradient_policy,
        "active_gradient_indices_acquired": list(
            promoted["active_gradient_indices_acquired"]
        ),
        "active_gradient_charge": query_count,
        "active_gradient_source": promoted["active_gradient_source"],
        "g_A": list(promoted["g_A"]),
    }


def _run_case(
    *,
    row: Mapping[str, Any],
    protocol_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt import (
        RAAdaptOperationalControls,
        run_ra_adapt,
    )
    from pipelines.static_adapt.ra_adapt.bundles import (
        load_validated_bundle_protocol,
    )

    protocol = load_validated_bundle_protocol(protocol_path)
    insertion = protocol.request.method.insertion
    if (
        getattr(insertion, "kind", None) != ALWAYS_INSERTION_KIND
        or protocol.active_gradient_policy
        != row["active_gradient_policy"]
        or protocol.resource_weighting_scope
        != row["resource_weighting_scope"]
    ):
        raise PackageContractError(
            f"Smoke protocol axes drifted: {row['execution_id']}."
        )
    with contextlib.redirect_stdout(io.StringIO()):
        observed = run_ra_adapt(
            _problem_from_protocol(protocol),
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=SMOKE_ROUNDS
            ),
        )
    payload = observed.to_dict()
    scientific = payload.get("scientific_receipts")
    accepted = (
        scientific.get("accepted_round_receipts")
        if isinstance(scientific, Mapping)
        else None
    )
    if not isinstance(accepted, list) or len(accepted) != SMOKE_ROUNDS:
        raise PackageContractError(
            f"Smoke did not complete two rounds: {row['execution_id']}."
        )
    reductions: list[dict[str, Any]] = []
    for round_index, accepted_row in enumerate(accepted, start=1):
        reduction = (
            accepted_row.get("insertion_commutation_reduced")
            if isinstance(accepted_row, Mapping)
            else None
        )
        if not isinstance(reduction, Mapping):
            raise PackageContractError(
                f"Smoke round {round_index} omitted reduction."
            )
        reductions.append(dict(reduction))
    return {
        "execution_id": row["execution_id"],
        "base_cell_id": row["base_cell_id"],
        "bundle_id": row["bundle_id"],
        "route_id": row["route_id"],
        "candidate_representation": (
            protocol.candidate_representation
        ),
        "active_gradient_policy": protocol.active_gradient_policy,
        "resource_weighting_scope": (
            protocol.resource_weighting_scope
        ),
        "protocol_path": protocol_path.relative_to(
            repo_root
        ).as_posix(),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_canonical_sha256": protocol.sha256,
        "typed_insertion_policy": getattr(insertion, "kind", None),
        "controller_round_count": len(accepted),
        "result_sha256": canonical_sha256(payload),
        "trajectory_policy_observation": _policy_observation(payload),
        "active_gradient_observation": _active_gradient_observation(
            protocol.active_gradient_policy
        ),
        "phase1_cost_observation": _cost_observation(
            protocol.resource_weighting_scope
        ),
        "accepted_round_reduction_receipts": reductions,
        "paper_evidence_allowed": False,
        "status": "passed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_DIR / SMOKE_RECEIPT_NAME,
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite smoke: {output}")

    repo_root = repo_root_from_script(__file__)
    authority = validate_factorial_authority(repo_root)
    rows_by_id = {
        row["execution_id"]: row for row in direct_execution_rows()
    }
    observations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="paper-i-ra-always-factorial-smoke."
    ) as temporary_name:
        original = Path.cwd()
        os.chdir(temporary_name)
        try:
            for execution_id in SMOKE_EXECUTION_IDS:
                row = rows_by_id[execution_id]
                binding = authority["protocol_bindings"][
                    execution_id
                ]
                observations.append(
                    _run_case(
                        row=row,
                        protocol_path=repo_root / binding["path"],
                        repo_root=repo_root,
                    )
                )
        finally:
            os.chdir(original)

    receipt = digested(
        {
            "schema": SMOKE_SCHEMA,
            "package_id": PACKAGE_ID,
            "status": "passed",
            "captured_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "smoke_scope": (
                "four_factorial_arms_by_two_representations_"
                "bounded_to_two_rounds_v1"
            ),
            "maximum_controller_rounds": SMOKE_ROUNDS,
            "observations": observations,
            "requested_domain_policy": (
                "all_logical_positions_range_append_position_plus_one_v1"
            ),
            "representative_policy": (
                "earliest_member_per_exact_termwise_commutation_class_v1"
            ),
            "paper_evidence_allowed": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    validate_smoke_receipt(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(receipt) + b"\n")
    os.link(temporary, output)
    temporary.unlink()
    print(
        canonical_json_bytes(
            {
                "status": "passed",
                "output": output.relative_to(repo_root).as_posix(),
                "sha256": receipt["sha256"],
                "observation_count": len(observations),
                "maximum_controller_rounds": SMOKE_ROUNDS,
                "remote_stage": False,
                "condor_submit": False,
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

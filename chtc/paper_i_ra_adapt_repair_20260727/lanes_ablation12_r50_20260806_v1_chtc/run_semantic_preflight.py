#!/usr/bin/env python3
"""Run the bounded two-round commutation-reduction smoke.

The smoke uses the exact v13 nph=3 strong-weak protocols for both repaired
always routes.  It is local validation only: it never executes the 50-round
horizon, creates authority, contacts CHTC, or invokes HTCondor.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    ALWAYS_INSERTION_KIND,
    PACKAGE_ID,
    ROUTE_IDS,
    SMOKE_EXECUTION_IDS,
    SMOKE_RECEIPT_NAME,
    SMOKE_ROUNDS,
    SMOKE_SCHEMA,
    PackageContractError,
    canonical_json_bytes,
    canonical_sha256,
    digested,
    repo_root_from_script,
    sha256_file,
    validate_core_authority,
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
            "Smoke problem reconstruction drifted from v13 protocol."
        )
    return problem


def _run_route(
    *,
    route_id: str,
    execution_id: str,
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
    if getattr(insertion, "kind", None) != ALWAYS_INSERTION_KIND:
        raise PackageContractError(
            f"{route_id} smoke protocol lost the repaired typed policy."
        )
    problem = _problem_from_protocol(protocol)
    observed = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=SMOKE_ROUNDS
        ),
    )
    payload = observed.to_dict()
    scientific = payload.get("scientific_receipts")
    if not isinstance(scientific, Mapping):
        raise PackageContractError(
            f"{route_id} smoke has no scientific receipts."
        )
    accepted = scientific.get("accepted_round_receipts")
    if not isinstance(accepted, list) or len(accepted) != SMOKE_ROUNDS:
        raise PackageContractError(
            f"{route_id} smoke did not complete exactly two rounds."
        )
    reductions: list[dict[str, Any]] = []
    for round_index, row in enumerate(accepted, start=1):
        if not isinstance(row, Mapping):
            raise PackageContractError(
                f"{route_id} accepted-round receipt is malformed."
            )
        reduction = row.get("insertion_commutation_reduced")
        if not isinstance(reduction, Mapping):
            raise PackageContractError(
                f"{route_id} round {round_index} omitted reduction."
            )
        reductions.append(dict(reduction))
    second = reductions[1]
    return {
        "route_id": route_id,
        "source_execution_id": execution_id,
        "candidate_representation": protocol.candidate_representation,
        "protocol_path": protocol_path.relative_to(repo_root).as_posix(),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_canonical_sha256": protocol.sha256,
        "typed_insertion_policy": getattr(insertion, "kind", None),
        "controller_round_count": len(accepted),
        "result_sha256": canonical_sha256(payload),
        "accepted_round_reduction_receipts": reductions,
        "second_round_requested_positions": second.get(
            "requested_positions"
        ),
        "second_round_retained_representative_count": second.get(
            "retained_representative_count"
        ),
        "second_round_collapsed_position_count": second.get(
            "collapsed_position_count"
        ),
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
        raise FileExistsError(f"Refusing to overwrite smoke receipt: {output}")

    repo_root = repo_root_from_script(__file__)
    authority = validate_core_authority(repo_root)
    bundle_root = Path(authority["bundle_root"])
    observations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="paper-i-ra-always-reduced-smoke."
    ) as temporary_name:
        original = Path.cwd()
        os.chdir(temporary_name)
        try:
            for route_id, execution_id in zip(
                ROUTE_IDS, SMOKE_EXECUTION_IDS
            ):
                protocol_path = (
                    bundle_root
                    / "protocols"
                    / f"{execution_id}.json"
                )
                observations.append(
                    _run_route(
                        route_id=route_id,
                        execution_id=execution_id,
                        protocol_path=protocol_path,
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
                "exact_v13_nph3_protocols_bounded_to_two_rounds_v1"
            ),
            "maximum_controller_rounds": SMOKE_ROUNDS,
            "route_observations": observations,
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
                "maximum_controller_rounds": SMOKE_ROUNDS,
                "remote_stage": False,
                "condor_submit": False,
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

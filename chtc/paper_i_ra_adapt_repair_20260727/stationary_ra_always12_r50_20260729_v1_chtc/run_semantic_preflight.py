#!/usr/bin/env python3
"""Run bounded semantic P3 checks for the 12-cell RA-always successor.

This is a local, non-paper preflight. It exercises both selected RA-always
routes through the public facade, authenticates continuation and replay, and
requires every Phase-I generator to be scored at every logical insertion
position. It never executes the 50-round paper horizon or authorizes
submission.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    APPEND_ROUTES,
    CAMPAIGN_ID,
    INSERTION_CAPABLE_ROUTES,
    P3_ALWAYS_G5_ROUNDS,
    P3_FIXTURE_ID,
    P3_NPH,
    P3_PLATEAU_G5_ROUNDS,
    P3_REGIME_ID,
    P3_RECEIPT_SCHEMA,
    P3_SHORT_ROUNDS,
    PACKAGE_ID,
    POOL_AUTHORITY_BY_NPH,
    RA_ROUTES,
    REGIME_CUTOFF_PAIRS,
    ROUTE_IDS,
    PackageContractError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    control_plane_receipt,
    core_cell_id,
    digested,
    representation_for_route,
    repo_root_from_script,
    sha256_file,
    validate_core_authority,
    validate_p3_receipt,
)


SMOKE_REGIME = P3_REGIME_ID
FINAL_PROTOCOL_NPH = P3_NPH
FIXTURE_NPH = P3_NPH
FIXTURE_ID = P3_FIXTURE_ID
FRESH_ROUNDS = P3_SHORT_ROUNDS
RESUME_PREFIX_ROUNDS = 1
G5_WITNESS_ROUNDS = P3_PLATEAU_G5_ROUNDS


class SemanticPreflightError(RuntimeError):
    """Raised when a real bounded semantic observation does not close."""


def _module_is_repo_local(*, module: Any, repo_root: Path) -> bool:
    root = repo_root.resolve()
    origin = getattr(module, "__file__", None)
    if origin is not None:
        try:
            Path(origin).resolve().relative_to(root)
        except ValueError:
            return False
        return True
    locations = getattr(module, "__path__", None)
    if locations is None:
        return False
    resolved = tuple(Path(item).resolve() for item in locations)
    if not resolved:
        return False
    for location in resolved:
        try:
            location.relative_to(root)
        except ValueError:
            return False
    return True


def _activate_repo_source(repo_root: Path) -> None:
    root = repo_root.resolve()
    for name in list(sys.modules):
        if (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
            del sys.modules[name]
    sys.path[:] = [
        item
        for item in sys.path
        if Path(item or ".").resolve() != root
    ]
    sys.path.insert(0, root.as_posix())
    importlib.invalidate_caches()
    namespace = importlib.import_module("pipelines")
    concrete = importlib.import_module("pipelines.static_adapt.ra_adapt")
    if not _module_is_repo_local(
        module=namespace, repo_root=root
    ) or not _module_is_repo_local(module=concrete, repo_root=root):
        raise SemanticPreflightError(
            "Ambient pipelines package masked or merged with the active "
            "repository."
        )


def _assert_repo_imports(repo_root: Path) -> None:
    drifted: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if not (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
            continue
        if not _module_is_repo_local(module=module, repo_root=repo_root):
            drifted.append(name)
    if drifted:
        raise SemanticPreflightError(
            "P3 imported non-repository implementation modules: "
            + ", ".join(sorted(drifted))
        )


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticPreflightError(f"{label} is not a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise SemanticPreflightError(f"{label} is not a sequence.")
    return value


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _problem_from_protocol(protocol: Any) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import (
        resolve_problem_context,
    )
    from pipelines.static_adapt.sr_snake.contracts import (
        ResolvedProblemReceipt,
    )

    receipt = protocol.problem
    request = ProblemRequest(
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
    problem = resolve_problem_context(request)
    if ResolvedProblemReceipt.from_problem(problem).to_dict() != receipt.to_dict():
        raise SemanticPreflightError(
            "Reconstructed problem does not match the selected protocol."
        )
    return problem


def _fixture_authority_inputs(
    *,
    route_id: str,
    final_cell_id: str,
    authority: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from pipelines.static_adapt.ra_adapt import bundles as bundle_module
    from pipelines.static_adapt.ra_adapt.append import (
        APPEND_ADAPT_ALGORITHM_ID,
    )
    from pipelines.static_adapt.ra_adapt.engine import (
        RA_ADAPT_ALGORITHM_ID,
    )

    source_lock_id = next(
        str(row["source_lock_id"])
        for row in authority["cell_rows"]
        if row["cell_id"] == final_cell_id
    )
    lock = _mapping(
        authority["source_lock_cells"].get(source_lock_id),
        label=f"{route_id} final source lock",
    )
    resolver = _mapping(
        lock.get("resolver_trace"), label=f"{route_id} resolver trace"
    )
    member = _mapping(lock.get("member"), label=f"{route_id} member")
    archive = _mapping(lock.get("archive"), label=f"{route_id} archive")
    reference = _mapping(
        resolver.get("same_cutoff_ed_reference"),
        label=f"{route_id} same-cutoff reference",
    )
    resolver_source = _mapping(
        authority["global_source_locks"].get(
            "visible_settings_resolver"
        ),
        label="global visible-settings resolver source lock",
    )
    source_refs = {
        "source_locks_manifest_sha256": authority["document_bindings"][
            "source_locks.json"
        ]["canonical_sha256"],
        "implementation_source_inventory_sha256": authority[
            "implementation_inventory_sha256"
        ],
        "cell_source_lock_id": source_lock_id,
        "cell_source_lock_sha256": str(lock["sha256"]),
        "visible_provenance_sha256": str(member["sha256"]),
        "provenance_tracker_sha256": str(archive["sha256"]),
        "ed_cutoff_reference_sha256": str(reference["sha256"]),
        "resolver_script_sha256": str(resolver_source["sha256"]),
    }
    selector = "append_adapt" if route_id in APPEND_ROUTES else "ra_adapt"
    cell = bundle_module.BundleCellSpec(
        cell_id=f"semantic_fixture__{route_id}",
        stage="validation",
        regime_id=SMOKE_REGIME,
        nph=FIXTURE_NPH,
        route_id=route_id,
        algorithm_id=(
            APPEND_ADAPT_ALGORITHM_ID
            if selector == "append_adapt"
            else RA_ADAPT_ALGORITHM_ID
        ),
        selector_family=selector,
        candidate_representation=representation_for_route(route_id),
        horizon=(
            G5_WITNESS_ROUNDS
            if route_id.endswith("plateau")
            else P3_ALWAYS_G5_ROUNDS
            if route_id.endswith("always")
            else FRESH_ROUNDS
        ),
        source_lock_id=source_lock_id,
    )
    construction = {
        "fixture_identity": FIXTURE_ID,
        "fixture_regime_id": SMOKE_REGIME,
        "fixture_nph": FIXTURE_NPH,
        "route_id": route_id,
        "final_cell_id": final_cell_id,
        "final_protocol_sha256": authority["protocol_bindings"][
            final_cell_id
        ]["canonical_sha256"],
        "source_lock_refs": source_refs,
        "active_gradient_policy": "stationary_source_response_v1",
        "resource_weighting_scope": "late_resource_weighting_v1",
        "horizon": cell.horizon,
    }
    return cell, construction


def _fixture_protocol(
    *,
    route_id: str,
    final_cell_id: str,
    authority: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], Any]:
    from pipelines.static_adapt.ra_adapt import (
        AppendAdaptRequest,
        MacroCandidateAdapter,
        SinglePauliWordCandidateAdapter,
    )
    from pipelines.static_adapt.ra_adapt import bundles as bundle_module
    from pipelines.static_adapt.ra_adapt.append import (
        build_resolved_append_protocol,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        ACTIVE_GRADIENT_STATIONARY,
        RESOURCE_WEIGHTING_LATE,
        _attach_validated_bundle_protocol_authority,
    )
    from pipelines.static_adapt.ra_adapt.bundles import (
        load_validated_bundle_protocol,
    )
    from pipelines.static_adapt.sr_snake import (
        SRExecutionPolicy,
        SRStopPolicy,
    )

    cell, construction = _fixture_authority_inputs(
        route_id=route_id,
        final_cell_id=final_cell_id,
        authority=authority,
    )
    final_protocol_path = (
        Path(str(authority["bundle_root"]))
        / "protocols"
        / f"{final_cell_id}.json"
    )
    final_protocol = load_validated_bundle_protocol(final_protocol_path)
    final_binding = authority["protocol_bindings"][final_cell_id]
    final_materialization = final_protocol.bundle_materialization
    if (
        final_protocol.sha256 != final_binding["canonical_sha256"]
        or sha256_file(final_protocol_path) != final_binding["sha256"]
        or final_protocol_path.stat().st_size
        != int(final_binding["size_bytes"])
        or final_protocol.candidate_representation
        != representation_for_route(route_id)
        or int(final_protocol.problem.n_ph_max) != FIXTURE_NPH
        or final_protocol.active_gradient_policy
        != "stationary_source_response_v1"
        or final_protocol.resource_weighting_scope
        != "late_resource_weighting_v1"
        or final_materialization.cell_id != final_cell_id
    ):
        raise SemanticPreflightError(
            f"{route_id} exact final protocol authority drifted."
        )
    problem = _problem_from_protocol(final_protocol)
    construction = {
        **construction,
        "fixture_problem_receipt": final_protocol.problem.to_dict(),
        "bounded_protocol_mode": (
            "final_bundle_problem_and_source_authority_bounded_protocol_v1"
            if route_id in APPEND_ROUTES
            else "exact_final_bundle_protocol_with_operational_round_cap_v1"
        ),
    }
    if route_id not in APPEND_ROUTES:
        return final_protocol, {
            **construction,
            "fixture_protocol_sha256": final_protocol.sha256,
            "fixture_construction_sha256": canonical_sha256(construction),
        }, problem

    adapter = (
        MacroCandidateAdapter()
        if route_id == "append_macro"
        else SinglePauliWordCandidateAdapter()
    )
    authority_kwargs = {
        "cell": cell,
        "bundle_id": bundle_module.CORE_BUNDLE_ID,
        "bundle_manifest_sha256": authority["document_bindings"][
            "bundle_manifest.json"
        ]["canonical_sha256"],
        "source_locks_sha256": authority["document_bindings"][
            "source_locks.json"
        ]["canonical_sha256"],
        "source_lock_refs": construction["source_lock_refs"],
        "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
        "resource_weighting_scope": RESOURCE_WEIGHTING_LATE,
    }
    first_authority = (
        bundle_module._bundle_protocol_materialization_authority(
            **authority_kwargs
        )
    )
    request = AppendAdaptRequest(
        adapter=adapter,
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(
                maximum_controller_rounds=cell.horizon
            )
        ),
    )
    protocol = build_resolved_append_protocol(
        problem,
        request,
        materialization_authority=first_authority,
    )
    attached = _attach_validated_bundle_protocol_authority(
        protocol,
        bundle_module._bundle_protocol_materialization_authority(
            **authority_kwargs,
            protocol_sha256=protocol.sha256,
        ),
    )
    return attached, {
        **construction,
        "fixture_protocol_sha256": attached.sha256,
        "fixture_construction_sha256": canonical_sha256(construction),
    }, problem


def _observation(root: Path, *, stem: str) -> Any:
    from pipelines.static_adapt.sr_snake import (
        CheckpointObservation,
        EstimatorLedgerObservation,
        SRObservationPolicy,
    )

    return SRObservationPolicy(
        checkpoint=CheckpointObservation(
            path=root / f"{stem}.checkpoint.json",
            every_controller_rounds=1,
            keep_history_tail=100,
        ),
        estimator_ledger=EstimatorLedgerObservation(
            path=root / f"{stem}.ledger.json"
        ),
    )


def _trajectory(payload: Mapping[str, Any], *, append: bool) -> list[Any]:
    if append:
        source = _mapping(
            payload.get("result_payload"), label="Append result payload"
        ).get("history")
    else:
        source = _mapping(payload.get("run"), label="RA run").get(
            "accepted_trajectory"
        )
    return list(_sequence(source, label="accepted trajectory"))


def _validated_replay(
    payload: Mapping[str, Any],
    *,
    append: bool,
) -> Mapping[str, Any]:
    from pipelines.static_adapt.ra_adapt.replay_evidence import (
        validate_controller_replay_evidence,
    )

    scientific = _mapping(
        payload.get("scientific_receipts"), label="scientific receipts"
    )
    replay = validate_controller_replay_evidence(
        _mapping(
            scientific.get("controller_replay_evidence"),
            label="controller replay evidence",
        )
    )
    trajectory = _trajectory(payload, append=append)
    if (
        replay.get("sha256")
        != scientific.get("controller_replay_evidence_sha256")
        or len(replay.get("signed_controller_round_prefixes", ()))
        != len(trajectory)
    ):
        raise SemanticPreflightError(
            "Controller replay evidence does not close its trajectory."
        )
    return replay


def _result_observation(
    payload: Mapping[str, Any],
    *,
    append: bool,
) -> dict[str, Any]:
    trajectory = _trajectory(payload, append=append)
    if not trajectory:
        raise SemanticPreflightError("Bounded facade produced no trajectory.")
    return {
        "status": "passed",
        "result_sha256": canonical_sha256(payload),
        "trajectory_sha256": canonical_sha256(trajectory),
        "controller_rounds": len(trajectory),
    }


def _g5_witness(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    scientific = _mapping(
        payload.get("scientific_receipts"), label="G5 scientific receipts"
    )
    rounds = _sequence(
        scientific.get("accepted_round_receipts"),
        label="G5 accepted-round receipts",
    )
    scored = 0
    interior = 0
    first_interior_round: int | None = None
    population_rows: list[Mapping[str, Any]] = []
    for controller_round, raw in enumerate(rounds, start=1):
        row = _mapping(raw, label="accepted-round receipt")
        population = _mapping(
            row.get("scored_insertion_position_population"),
            label="scored-position population",
        )
        scored += int(population.get("scored_record_count", 0))
        round_interior = int(population.get("interior_scored_count", 0))
        interior += round_interior
        if round_interior > 0 and first_interior_round is None:
            first_interior_round = controller_round
        append_position = int(population.get("append_position", -1))
        phases = _sequence(
            population.get("phases"),
            label="scored-position phases",
        )
        if len(phases) != 3:
            raise SemanticPreflightError(
                "Scored-position receipt does not retain three phases."
            )
        phase_i = _mapping(phases[0], label="Phase-I scored population")
        phase_i_records = _sequence(
            phase_i.get("records"),
            label="Phase-I scored records",
        )
        positions_by_generator: dict[tuple[int, str], set[int]] = {}
        for raw_record in phase_i_records:
            record = _mapping(
                raw_record,
                label="Phase-I scored-position record",
            )
            generator_key = (
                int(record.get("pool_index", -1)),
                str(record.get("generator_id", "")),
            )
            positions_by_generator.setdefault(generator_key, set()).add(
                int(record.get("insertion_position", -1))
            )
        expected_positions = set(range(append_position + 1))
        if (
            append_position < 0
            or not positions_by_generator
            or any(
                positions != expected_positions
                for positions in positions_by_generator.values()
            )
        ):
            raise SemanticPreflightError(
                "Full insertion did not score every Phase-I generator at "
                f"every logical position in round {controller_round}."
            )
        population_rows.append(population)
    if scored <= 0 or interior <= 0 or first_interior_round is None:
        raise SemanticPreflightError(
            "Insertion smoke did not produce a nonvacuous interior "
            "scored-position population."
        )
    witness = {
        "status": "passed",
        "aggregate_g5_passed": True,
        "witness_controller_rounds": len(rounds),
        "first_interior_controller_round": first_interior_round,
        "scored_position_count": scored,
        "interior_scored_count": interior,
        "population_receipt_sha256": canonical_sha256(population_rows),
        "phase_i_full_logical_positions_verified": True,
        "phase_i_full_logical_position_round_count": len(rounds),
        "interior_witness_status": "observed",
    }
    return witness


def _run_ra_route(
    *,
    route_id: str,
    final_cell_id: str,
    protocol: Any,
    construction: Mapping[str, Any],
    problem: Any,
    work_root: Path,
) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt import (
        RAAdaptOperationalControls,
        run_ra_adapt,
    )
    from pipelines.static_adapt.ra_adapt.replay_evidence import (
        compare_bounded_controller_replays,
    )
    from pipelines.static_adapt.sr_snake import AcceptedStateResume

    route_root = work_root / route_id
    route_root.mkdir()
    invocations: list[dict[str, Any]] = []

    def invoke(stem: str, rounds: int, *, resume: Any = None) -> Any:
        invocations.append(
            {
                "entrypoint": "run_ra_adapt",
                "purpose": stem,
                "maximum_controller_rounds": rounds,
            }
        )
        control_kwargs = {
            "maximum_controller_rounds": rounds,
            "observation": _observation(route_root, stem=stem),
        }
        if resume is not None:
            control_kwargs["resume"] = resume
        controls = RAAdaptOperationalControls(**control_kwargs)
        return run_ra_adapt(
            problem,
            protocol,
            operational_controls=controls,
        )

    with _working_directory(route_root):
        primary = invoke("independent_primary", FRESH_ROUNDS)
        prefix = invoke("fresh_resume_prefix", RESUME_PREFIX_ROUNDS)
        prefix_checkpoint = route_root / "fresh_resume_prefix.checkpoint.json"
        prefix_checkpoint_sha = sha256_file(prefix_checkpoint)
        resumed = invoke(
            "authenticated_resume",
            FRESH_ROUNDS,
            resume=AcceptedStateResume(
                checkpoint_path=prefix_checkpoint,
                checkpoint_sha256=prefix_checkpoint_sha,
            ),
        )
        witness = None
        if route_id in INSERTION_CAPABLE_ROUTES:
            witness_rounds = (
                G5_WITNESS_ROUNDS
                if route_id.endswith("plateau")
                else P3_ALWAYS_G5_ROUNDS
            )
            witness = invoke(
                "g5_scored_position_witness",
                witness_rounds,
            )

    primary_payload = primary.to_dict()
    prefix_payload = prefix.to_dict()
    resumed_payload = resumed.to_dict()
    primary_trajectory = _trajectory(primary_payload, append=False)
    prefix_trajectory = _trajectory(prefix_payload, append=False)
    resumed_trajectory = _trajectory(resumed_payload, append=False)
    if (
        primary_trajectory[:RESUME_PREFIX_ROUNDS] != prefix_trajectory
        or resumed_trajectory[:RESUME_PREFIX_ROUNDS] != prefix_trajectory
        or primary_trajectory != resumed_trajectory
    ):
        raise SemanticPreflightError(
            f"{route_id} fresh/replay/resume trajectory identity failed."
        )
    primary_replay = _validated_replay(primary_payload, append=False)
    prefix_replay = _validated_replay(prefix_payload, append=False)
    resumed_replay = _validated_replay(resumed_payload, append=False)
    independent_comparison = compare_bounded_controller_replays(
        primary_replay,
        resumed_replay,
        controller_round=FRESH_ROUNDS,
    )
    resume_comparison = compare_bounded_controller_replays(
        prefix_replay,
        resumed_replay,
        controller_round=RESUME_PREFIX_ROUNDS,
    )
    if (
        independent_comparison.get("matched") is not True
        or resume_comparison.get("matched") is not True
    ):
        raise SemanticPreflightError(
            f"{route_id} signed replay comparison failed."
        )
    row: dict[str, Any] = {
        "route_id": route_id,
        "candidate_representation": str(
            protocol.candidate_representation
        ),
        "fixture_identity": FIXTURE_ID,
        "fixture_regime_id": construction["fixture_regime_id"],
        "fixture_nph": FIXTURE_NPH,
        "fixture_problem_receipt": construction[
            "fixture_problem_receipt"
        ],
        "bounded_protocol_mode": construction[
            "bounded_protocol_mode"
        ],
        "ordinary_smoke_controller_rounds": FRESH_ROUNDS,
        "final_protocol_nph": FINAL_PROTOCOL_NPH,
        "final_protocol_cell_id": final_cell_id,
        "run_class": "smoke",
        "paper_facing_result_allowed": False,
        "protocol_sha256": str(
            construction["final_protocol_sha256"]
        ),
        "fixture_protocol_sha256": str(protocol.sha256),
        "fixture_construction_sha256": str(
            construction["fixture_construction_sha256"]
        ),
        "facade_invocations": invocations,
        "maximum_controller_rounds_executed": max(
            call["maximum_controller_rounds"] for call in invocations
        ),
        "fresh_execution": _result_observation(
            prefix_payload, append=False
        ),
        "independent_replay": {
            **_result_observation(primary_payload, append=False),
            "matched": True,
            "comparison_sha256": independent_comparison["sha256"],
        },
        "authenticated_resume": {
            "status": "passed",
            "authenticated": True,
            "trajectory_prefix_matched": True,
            "checkpoint_file_sha256": prefix_checkpoint_sha,
            "resumed_result_sha256": canonical_sha256(resumed_payload),
            "comparison_sha256": resume_comparison["sha256"],
        },
        "status": "passed",
    }
    if witness is not None:
        witness_payload = witness.to_dict()
        if _trajectory(witness_payload, append=False)[:FRESH_ROUNDS] != (
            primary_trajectory
        ):
            raise SemanticPreflightError(
                f"{route_id} G5 witness changed the authenticated prefix."
            )
        always_route = route_id.endswith("always")
        full_insertion_verified = (
            getattr(
                getattr(protocol.request.method, "insertion", None),
                "kind",
                None,
            )
            == "full_commutation"
        )
        if always_route and not full_insertion_verified:
            raise SemanticPreflightError(
                f"{route_id} lost its typed full-insertion policy."
            )
        row["g5_scored_position_witness"] = {
            **_g5_witness(witness_payload),
            "execution_mode": (
                "independent_fresh_exact_final_nph3_protocol_v1"
            ),
            "trajectory_prefix_matched": True,
            "authenticated_prefix_controller_rounds": FRESH_ROUNDS,
            "full_insertion_policy_verified": (
                full_insertion_verified if always_route else None
            ),
        }
    return row


def _run_append_route(
    *,
    route_id: str,
    final_cell_id: str,
    protocol: Any,
    construction: Mapping[str, Any],
    problem: Any,
    work_root: Path,
) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt import run_append_adapt
    from pipelines.static_adapt.ra_adapt.replay_evidence import (
        compare_bounded_controller_replays,
    )
    route_root = work_root / route_id
    route_root.mkdir()
    payloads: list[Mapping[str, Any]] = []
    invocations: list[dict[str, Any]] = []
    with _working_directory(route_root):
        for stem in ("fresh_execution", "independent_reconstruction"):
            invocations.append(
                {
                    "entrypoint": "run_append_adapt",
                    "purpose": stem,
                    "maximum_controller_rounds": FRESH_ROUNDS,
                }
            )
            result = run_append_adapt(
                problem,
                protocol,
            )
            payloads.append(result.to_dict())
    first, second = payloads
    first_trajectory = _trajectory(first, append=True)
    second_trajectory = _trajectory(second, append=True)
    comparison = compare_bounded_controller_replays(
        _validated_replay(first, append=True),
        _validated_replay(second, append=True),
        controller_round=FRESH_ROUNDS,
    )
    if (
        first_trajectory != second_trajectory
        or comparison.get("matched") is not True
    ):
        raise SemanticPreflightError(
            f"{route_id} reconstruction replay identity failed."
        )
    replay = _validated_replay(first, append=True)
    boundary = _mapping(
        replay.get("resume_sidecar_closure"),
        label=f"{route_id} reconstruction boundary",
    )
    if (
        boundary.get("public_resume_execution_supported") is not False
        or boundary.get("reconstruction_fields_complete") is not True
    ):
        raise SemanticPreflightError(
            f"{route_id} did not expose its typed reconstruction boundary."
        )
    return {
        "route_id": route_id,
        "candidate_representation": str(
            protocol.candidate_representation
        ),
        "fixture_identity": FIXTURE_ID,
        "fixture_regime_id": construction["fixture_regime_id"],
        "fixture_nph": FIXTURE_NPH,
        "fixture_problem_receipt": construction[
            "fixture_problem_receipt"
        ],
        "bounded_protocol_mode": construction[
            "bounded_protocol_mode"
        ],
        "ordinary_smoke_controller_rounds": FRESH_ROUNDS,
        "final_protocol_nph": FINAL_PROTOCOL_NPH,
        "final_protocol_cell_id": final_cell_id,
        "run_class": "smoke",
        "paper_facing_result_allowed": False,
        "protocol_sha256": str(
            construction["final_protocol_sha256"]
        ),
        "fixture_protocol_sha256": str(protocol.sha256),
        "fixture_construction_sha256": str(
            construction["fixture_construction_sha256"]
        ),
        "facade_invocations": invocations,
        "maximum_controller_rounds_executed": FRESH_ROUNDS,
        "fresh_execution": _result_observation(first, append=True),
        "independent_replay": {
            **_result_observation(second, append=True),
            "matched": True,
            "comparison_sha256": comparison["sha256"],
        },
        "reconstruction_boundary": {
            "status": "authenticated_reconstruction_only_verified",
            "public_resume_execution_supported": False,
            "reconstruction_fields_complete": True,
            "boundary_sha256": str(boundary["sha256"]),
        },
        "status": "passed",
    }


def _pool_projection(receipt: Any) -> dict[str, Any]:
    return {
        "schema": str(receipt.schema),
        "candidate_representation": str(
            receipt.candidate_representation
        ),
        "count": int(receipt.count),
        "ordered_labels_sha256": str(
            receipt.ordered_labels_sha256
        ),
        "ordered_pool_sha256": str(receipt.ordered_pool_sha256),
        "source_parent_ordered_labels_sha256": (
            None
            if receipt.source_parent_ordered_labels_sha256 is None
            else str(
                receipt.source_parent_ordered_labels_sha256
            )
        ),
        "receipt_sha256": str(receipt.sha256),
    }


def _six_regime_pool_construction_proof(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove all six macro/singleton RA/Append construction contracts."""

    from pipelines.static_adapt.ra_adapt import (
        MacroCandidateAdapter,
        SinglePauliWordCandidateAdapter,
    )
    from pipelines.static_adapt.ra_adapt.bundles import (
        load_validated_bundle_protocol,
    )

    rows: list[dict[str, Any]] = []
    bundle_root = Path(str(authority["bundle_root"]))
    for regime_id, nph in REGIME_CUTOFF_PAIRS:
        ids = {
            "ra_macro": core_cell_id(
                regime_id, nph, "ra_macro_append_only"
            ),
            "append_macro": core_cell_id(
                regime_id, nph, "append_macro"
            ),
            "ra_singleton": core_cell_id(
                regime_id, nph, "ra_singleton_append_only"
            ),
            "append_singleton": core_cell_id(
                regime_id, nph, "append_singleton"
            ),
        }
        protocols = {
            name: load_validated_bundle_protocol(
                bundle_root / "protocols" / f"{cell_id}.json"
            )
            for name, cell_id in ids.items()
        }
        problem_receipts = {
            canonical_sha256(protocol.problem.to_dict())
            for protocol in protocols.values()
        }
        if len(problem_receipts) != 1:
            raise SemanticPreflightError(
                f"{regime_id} RA/Append problem receipts drifted."
            )
        problem = _problem_from_protocol(protocols["ra_macro"])
        macro_adapter = MacroCandidateAdapter()
        singleton_adapter = SinglePauliWordCandidateAdapter()
        macro_parent = macro_adapter.parent_inventory(problem)
        macro_pool = macro_adapter.executable_pool(problem)
        singleton_parent = singleton_adapter.parent_inventory(problem)
        ra_factory = singleton_adapter.executable_pool(problem)
        append_global = singleton_adapter.global_executable_pool(
            problem
        )
        ra_full_parent_staged = singleton_adapter.expose_children(
            ra_factory.candidates,
            problem=problem,
        )
        expected = POOL_AUTHORITY_BY_NPH[int(nph)]
        parent_projection = _pool_projection(
            macro_parent.receipt
        )
        macro_projection = _pool_projection(macro_pool.receipt)
        global_projection = _pool_projection(
            append_global.receipt
        )
        staged_projection = _pool_projection(
            ra_full_parent_staged.receipt
        )
        child_manifest_sha256 = canonical_sha256(
            [
                candidate.manifest_row()
                for candidate in append_global.candidates
            ]
        )
        staged_child_manifest_sha256 = canonical_sha256(
            [
                candidate.manifest_row()
                for candidate in ra_full_parent_staged.candidates
            ]
        )
        if (
            singleton_parent.receipt.ordered_labels
            != macro_parent.receipt.ordered_labels
            or singleton_parent.receipt.ordered_labels_sha256
            != macro_parent.receipt.ordered_labels_sha256
            or singleton_parent.receipt.ordered_pool_sha256
            != macro_parent.receipt.ordered_pool_sha256
            or ra_factory.receipt.to_dict()
            != singleton_parent.receipt.to_dict()
            or ra_factory.metadata.get("children_constructed")
            is not False
            or ra_factory.metadata.get("executable_pool_kind")
            != "guarded_singleton_children_factory_v1"
            or append_global.metadata.get("exposure_scope")
            != "global_parent_inventory_v1"
            or ra_full_parent_staged.metadata.get("exposure_scope")
            != "ra_retained_parent_shortlist_v1"
            or append_global.receipt.ordered_labels
            != ra_full_parent_staged.receipt.ordered_labels
            or append_global.receipt.ordered_labels_sha256
            != ra_full_parent_staged.receipt.ordered_labels_sha256
            or append_global.receipt.ordered_pool_sha256
            != ra_full_parent_staged.receipt.ordered_pool_sha256
            or child_manifest_sha256
            != staged_child_manifest_sha256
            or parent_projection["count"]
            != expected["parent_count"]
            or parent_projection["ordered_labels_sha256"]
            != expected["parent_ordered_labels_sha256"]
            or macro_projection["count"] != expected["macro_count"]
            or macro_projection["ordered_labels_sha256"]
            != expected["macro_ordered_labels_sha256"]
            or global_projection["count"]
            != expected["guarded_singleton_count"]
            or global_projection["ordered_labels_sha256"]
            != expected[
                "guarded_singleton_ordered_labels_sha256"
            ]
        ):
            raise SemanticPreflightError(
                f"{regime_id} pool/construction authority drifted."
            )
        macro_parent_dict = macro_parent.receipt.to_dict()
        macro_pool_dict = macro_pool.receipt.to_dict()
        singleton_parent_dict = singleton_parent.receipt.to_dict()
        append_global_dict = append_global.receipt.to_dict()
        if (
            protocols["ra_macro"].parent_inventory.to_dict()
            != macro_parent_dict
            or protocols["append_macro"].parent_inventory.to_dict()
            != macro_parent_dict
            or protocols["ra_macro"].executable_pool.to_dict()
            != macro_pool_dict
            or protocols["append_macro"].executable_pool.to_dict()
            != macro_pool_dict
            or protocols["ra_singleton"].parent_inventory.to_dict()
            != singleton_parent_dict
            or protocols["append_singleton"].parent_inventory.to_dict()
            != singleton_parent_dict
            or protocols["ra_singleton"].executable_pool.to_dict()
            != singleton_parent_dict
            or protocols["append_singleton"].executable_pool.to_dict()
            != append_global_dict
        ):
            raise SemanticPreflightError(
                f"{regime_id} protocol pool projection drifted."
            )
        construction = digested(
            {
                "schema": (
                    "paper_i_stationary_core_singleton_"
                    "construction_equivalence_v1"
                ),
                "regime_id": regime_id,
                "nph": int(nph),
                "ra_exposure": (
                    "staged_from_retained_parent_shortlist_v1"
                ),
                "append_exposure": (
                    "global_guarded_child_pool_v1"
                ),
                "comparison_parent_scope": (
                    "all_shared_parents_for_construction_equivalence_v1"
                ),
                "source_parent_count": int(
                    singleton_parent.receipt.count
                ),
                "source_parent_ordered_labels_sha256": str(
                    singleton_parent.receipt.ordered_labels_sha256
                ),
                "ra_staged_child_pool": staged_projection,
                "append_global_child_pool": global_projection,
                "ordered_child_manifest_sha256": (
                    child_manifest_sha256
                ),
                "canonical_unit_pauli_representatives": True,
                "hard_guarded": True,
                "construction_equivalent_for_identical_parent_supply": (
                    True
                ),
                "status": "passed",
            }
        )
        rows.append(
            digested(
                {
                    "schema": (
                        "paper_i_stationary_core_regime_"
                        "pool_construction_proof_v1"
                    ),
                    "regime_id": regime_id,
                    "nph": int(nph),
                    "problem_receipt_sha256": next(
                        iter(problem_receipts)
                    ),
                    "protocol_sha256s": {
                        name: protocol.sha256
                        for name, protocol in sorted(
                            protocols.items()
                        )
                    },
                    "parent_inventory": parent_projection,
                    "macro_coefficient_pool": macro_projection,
                    "singleton_parent_inventory": _pool_projection(
                        singleton_parent.receipt
                    ),
                    "singleton_append_global_pool": (
                        global_projection
                    ),
                    "singleton_construction_equivalence": construction,
                    "ra_append_macro_pool_equal": True,
                    "ra_append_singleton_parent_equal": True,
                    "status": "passed",
                }
            )
        )
    proof = digested(
        {
            "schema": (
                "paper_i_stationary_core_six_regime_"
                "pool_construction_proof_v1"
            ),
            "regime_count": 6,
            "regime_cutoff_pairs": [
                [regime_id, int(nph)]
                for regime_id, nph in REGIME_CUTOFF_PAIRS
            ],
            "rows": rows,
            "macro_ra_append_equality_all_regimes": True,
            "singleton_construction_equivalence_all_regimes": True,
            "status": "passed",
        }
    )
    return proof


def run_preflight(
    *,
    repo_root: Path,
    core_root: Path | None,
    output: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output = output.resolve()
    output_temporary = output.with_name(f".{output.name}.tmp")
    if (
        output.exists()
        or output.is_symlink()
        or output_temporary.exists()
        or output_temporary.is_symlink()
    ):
        raise SemanticPreflightError(
            "P3 output or its fixed atomic temporary already exists."
        )
    authority = validate_core_authority(
        repo_root, materialization_root=core_root
    )
    _activate_repo_source(repo_root)
    control_plane = control_plane_receipt(PACKAGE_DIR)
    generator_row = next(
        row
        for row in control_plane["files"]
        if row["path"] == "run_semantic_preflight.py"
    )
    pool_construction_proof = _six_regime_pool_construction_proof(
        authority
    )
    scratch_parent = Path(tempfile.gettempdir()).resolve()
    try:
        scratch_parent.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise SemanticPreflightError(
            "P3 temporary scratch must be external to the repository."
        )
    with tempfile.TemporaryDirectory(
        prefix="paper_i_stationary_core_p3_",
        dir=scratch_parent,
    ) as raw_work:
        work_root = Path(raw_work)
        route_rows: list[dict[str, Any]] = []
        for route_id in ROUTE_IDS:
            with tempfile.TemporaryDirectory(
                prefix=f"{route_id}_",
                dir=work_root,
            ) as raw_route_work:
                route_work_root = Path(raw_route_work)
                cell_id = core_cell_id(
                    SMOKE_REGIME, FINAL_PROTOCOL_NPH, route_id
                )
                protocol, construction, problem = _fixture_protocol(
                    route_id=route_id,
                    final_cell_id=cell_id,
                    authority=authority,
                )
                if route_id in APPEND_ROUTES:
                    row = _run_append_route(
                        route_id=route_id,
                        final_cell_id=cell_id,
                        protocol=protocol,
                        construction=construction,
                        problem=problem,
                        work_root=route_work_root,
                    )
                else:
                    row = _run_ra_route(
                        route_id=route_id,
                        final_cell_id=cell_id,
                        protocol=protocol,
                        construction=construction,
                        problem=problem,
                        work_root=route_work_root,
                    )
            if row["candidate_representation"] != representation_for_route(
                route_id
            ):
                raise SemanticPreflightError(
                    f"{route_id} representation drifted."
                )
            route_rows.append(row)
    _assert_repo_imports(repo_root)
    receipt = digested(
        {
            "schema": P3_RECEIPT_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "generator": {
                "path": "run_semantic_preflight.py",
                "sha256": generator_row["sha256"],
                "size_bytes": generator_row["size_bytes"],
            },
            "core_final_receipt_canonical_sha256": authority[
                "final_receipt_binding"
            ]["canonical_sha256"],
            "core_final_receipt_file_sha256": authority[
                "final_receipt_binding"
            ]["sha256"],
            "implementation_source_inventory_sha256": authority[
                "implementation_inventory_sha256"
            ],
            "active_gradient_policy": "stationary_source_response_v1",
            "resource_weighting_scope": "late_resource_weighting_v1",
            "execution_mode": "bounded_non_paper_semantic_preflight_v1",
            "governing_plan_p3_alignment": {
                "regime_id": SMOKE_REGIME,
                "nph": FIXTURE_NPH,
                "ordinary_smoke_controller_rounds": FRESH_ROUNDS,
                "route_coverage": "both_selected_ra_always_routes_v1",
                "ra_protocol_authority": (
                    "exact_final_stationary_core_protocol_v1"
                ),
                "append_protocol_authority": (
                    "exact_final_problem_and_source_authority_bounded_v1"
                ),
                "g5_execution_boundary": (
                    "separate_independent_fresh_witness_v1"
                ),
                "plateau_g5_round_cap": G5_WITNESS_ROUNDS,
                "always_g5_round_cap": P3_ALWAYS_G5_ROUNDS,
            },
            "full_horizon_executed": False,
            "maximum_controller_rounds_executed": max(
                int(row["maximum_controller_rounds_executed"])
                for row in route_rows
            ),
            "paper_facing_result_allowed": False,
            "p2_pool_construction_proof": pool_construction_proof,
            "p2_pool_construction_proof_sha256": (
                pool_construction_proof["sha256"]
            ),
            "route_observations": route_rows,
            "semantic_coverage": {
                "route_families": list(ROUTE_IDS),
                "candidate_representations": [
                    "macro_generator_v1",
                    "single_pauli_word_v1",
                ],
                "pool_construction_regime_count": 6,
                "cutoff_pool_coverage": [3, 7],
                "ra_fresh_resume_replay_routes": sorted(RA_ROUTES),
                "append_fresh_reconstruction_routes": sorted(APPEND_ROUTES),
                "nonvacuous_g5_routes": sorted(
                    INSERTION_CAPABLE_ROUTES
                ),
            },
            "status": "passed",
            "p3_passed": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
        }
    )
    atomic_write_json(output, receipt)
    validate_p3_receipt(
        receipt,
        receipt_file_sha256=sha256_file(output),
        authority=authority,
        control_plane=control_plane,
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root_from_script(__file__),
    )
    parser.add_argument("--core-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        receipt = run_preflight(
            repo_root=args.repo_root.resolve(),
            core_root=(
                None if args.core_root is None else args.core_root.resolve()
            ),
            output=args.output.resolve(),
        )
        print(canonical_json_bytes(receipt).decode("utf-8"))
        return 0
    except (
        OSError,
        PackageContractError,
        SemanticPreflightError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
